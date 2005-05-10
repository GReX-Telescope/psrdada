#include "dada_pwc.h"
#include "dada.h"

#include "ascii_header.h"
#include "utc.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

int dada_pwc_command_set_byte_count (dada_pwc_t* primary, FILE* output,
				   dada_pwc_command_t* command)
{
  if (!primary->utc_start) {
    fprintf (output, "UTC of first time sample unknown\n");
    return -1;
  }
  if (!primary->bytes_per_second) {
    fprintf (output, "bytes per second not known\n");
    return -1;
  }

  if (command->utc < primary->utc_start) {
    fprintf (output, "requested UTC precedes UTC of first time sample\n");
    command->utc = 0;
    command->byte_count = 0;
    return 0;
  }

  /* the number of second between start and requested UTC of command */
  command->byte_count = command->utc - primary->utc_start;
  /* the byte count at which to execute the command */
  command->byte_count *= primary->bytes_per_second;

  return 0;
}
  
/*! Set the command state */
int dada_pwc_command_set (dada_pwc_t* primary, FILE* output,
			  dada_pwc_command_t command)
{
  int ret = 0;

  if (!primary)
    return -1;
  
  pthread_mutex_lock (&(primary->mutex));

  while (primary->command.code != dada_pwc_no_command)
    pthread_cond_wait(&(primary->cond), &(primary->mutex));

  switch (command.code) {

  case dada_pwc_header:
    if (primary->state != dada_pwc_idle) {
      fprintf (output, "Cannot set header when not IDLE\n");
      ret = -1;
    }
    break;

  case dada_pwc_clock:
    if (primary->state != dada_pwc_prepared) {
      fprintf (output, "Cannot start clocking when not PREPARED\n");
      ret = -1;
    }
    break;

  case dada_pwc_record_start:
    if (primary->state != dada_pwc_clocking) {
      fprintf (output, "Cannot record start when not CLOCKING\n");
      ret = -1;
    }
    break;

  case dada_pwc_record_stop:
    if (primary->state != dada_pwc_recording) {
      fprintf (output, "Cannot record stop when not RECORDING\n");
      ret = -1;
    }
    break;

  case dada_pwc_start:
    if (primary->state != dada_pwc_prepared) {
      fprintf (output, "Cannot start when not PREPARED\n");
      ret = -1;
    }
    break;

  case dada_pwc_stop:
    if (primary->state != dada_pwc_clocking &&
	primary->state != dada_pwc_recording) {
      fprintf (output, "Cannot stop when not CLOCKING or RECORDING\n");
      ret = -1;
    }
    break;

  }

  if (ret == 0) {

    primary->command = command;
    pthread_cond_signal (&(primary->cond));

  }

  pthread_mutex_unlock (&(primary->mutex));

  return ret;
}

int dada_pwc_parse_bytes_per_second (dada_pwc_t* primary,
				     FILE* fptr, const char* header)
{
  unsigned npol;  /* number of polarizations */
  unsigned nbit;  /* nubmer of bits per sample */
  unsigned ndim;  /* number of dimensions */

  uint64_t bits_per_second = 0;

  double sampling_interval;

  if (ascii_header_get (header, "NPOL", "%d", &npol) < 0) {
    fprintf (fptr, "failed to parse NPOL - assuming 2\n");
    npol = 2;
  }

  if (ascii_header_get (header, "NBIT", "%d", &nbit) < 0) {
    fprintf (fptr, "failed to parse NBIT - assuming 8\n");
    nbit = 8;
  }

  if (ascii_header_get (header, "NDIM", "%d", &ndim) < 0) {
    fprintf (fptr, "failed to parse NDIM - assuming 1\n");
    ndim = 1;
  }

  primary->bits_per_sample = nbit * npol * ndim;

  if (ascii_header_get (header, "TSAMP", "%lf", &sampling_interval) < 0) {
    fprintf (fptr, "failed to parse TSAMP\n");
    primary->bytes_per_second = 0;
    return -1;
  }

  /* IMPORTANT: TSAMP is the sampling period in microseconds */
  bits_per_second = nbit * npol * ndim * ((uint64_t)(1e6/sampling_interval));

  primary->bytes_per_second = bits_per_second / 8;

  return 0;
}

int dada_pwc_cmd_header (void* context, FILE* fptr, char* args)
{
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  dada_pwc_t* primary = (dada_pwc_t*) context;
  char* hdr = args;

  if (!args) {
    fprintf (fptr, "no header specified\n");
    return -1;
  }

  if (strlen (args) > primary->header_size) {
    fprintf (fptr, "header too large (max %d bytes)\n", primary->header_size);
    return -1;
  }

  /* replace \ with new line */
  while ( (hdr = strchr(hdr, '\\')) != 0 )
    *hdr = '\n';

  if (dada_pwc_parse_bytes_per_second (primary, fptr, args) < 0)
    return -1;

  strcpy (primary->header, args);

  command.code = dada_pwc_header;
  command.header = primary->header;

  if (dada_pwc_command_set (primary, fptr, command) < 0)
    return -1;

  return 0;
}

uint64_t dada_pwc_parse_duration (dada_pwc_t* primary,
				  FILE* fptr, const char* args)
{
  unsigned hh, mm, ss;
  uint64_t byte_count;

  if (sscanf (args, "%u:%u:%u", &hh, &mm, &ss) == 3) {

    if (!primary->bytes_per_second) {
      fprintf (fptr, "bytes per second not known\n");
      return 0;
    }

    if (mm > 59) {
      fprintf (fptr, "invalid minutes = %u\n", mm);
      return 0;
    }

    if (ss > 59) {
      fprintf (fptr, "invalid seconds = %u\n", ss);
      return 0;
    }

    byte_count = hh;
    byte_count *= 3600;
    byte_count += 60 * mm;
    byte_count += ss;
    byte_count *= primary->bytes_per_second;

    return byte_count;

  }

  if (sscanf (args, "%llu", &byte_count) == 1) {

    if (!primary->bits_per_sample) {
      fprintf (fptr, "bits per sample not known\n");
      return 0;
    }

    byte_count *= primary->bits_per_sample;
    byte_count /= 8;

    return byte_count;

  }

  fprintf (fptr, "could not parse duration from '%s'\n", args);

  return 0;
}

int dada_pwc_cmd_duration (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;
  uint64_t byte_count = 0;

  int status = 0;

  if (!args) {
    fprintf (fptr, "please specify duration\n");
    return -1;
  }

  byte_count = dada_pwc_parse_duration (primary, fptr, args);

  if (!byte_count)
    return -1;

  pthread_mutex_lock (&(primary->mutex));

  while (primary->command.code != dada_pwc_no_command)
    pthread_cond_wait (&(primary->cond), &(primary->mutex));

  if (primary->state != dada_pwc_record_requested &&
      primary->state != dada_pwc_recording)
    primary->command.byte_count = byte_count;
  else {
    fprintf (stderr, "Cannot set DURATION while recording\n");
    status = -1;
  }

  pthread_mutex_unlock (&(primary->mutex));

  return status;
}


time_t dada_pwc_parse_time (FILE* fptr, char* args)  
{
  time_t utc = 0;

  if (!args)
    return utc;

  utc = str2time (args);
  if (utc == (time_t)-1) {
    fprintf (fptr, "Could not parse start time from '%s'\n", args);
    return -1;
  }

  return utc;
}

int dada_pwc_cmd_clock (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_clock;
  command.utc = 0;

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_record_start (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_record_start;
  command.utc = dada_pwc_parse_time (fptr, args);

  if (!command.utc) {
    fprintf (fptr, "rec_start requires a valid utc argument\n");
    return -1;
  }

  if (dada_pwc_command_set_byte_count (primary, fptr, &command) < 0)
    return -1;

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_record_stop (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_record_stop;
  command.utc = dada_pwc_parse_time (fptr, args);

  if (!command.utc) {
    fprintf (fptr, "rec_stop requires a valid utc argument\n");
    return -1;
  }

  if (dada_pwc_command_set_byte_count (primary, fptr, &command) < 0)
    return -1;

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_start (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_start;
  command.utc = dada_pwc_parse_time (fptr, args);

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_stop (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_stop;
  command.utc = dada_pwc_parse_time (fptr, args);

  if (command.utc && dada_pwc_command_set_byte_count (primary, fptr, &command)<0)
    return -1;
  return dada_pwc_command_set (primary, fptr, command);
}

/*! Create a new DADA primary write client connection */
dada_pwc_t* dada_pwc_create ()
{
  dada_pwc_t* primary = (dada_pwc_t*) malloc (sizeof(dada_pwc_t));
  assert (primary != 0);

  primary -> state = dada_pwc_idle;
  primary -> command.code = dada_pwc_no_command;
  primary -> bytes_per_second = 0;
  primary -> bits_per_sample = 0;
  primary -> utc_start = 0;

  /* default header size */
  primary -> header_size = DADA_DEFAULT_HDR_SIZE;
  primary -> header = (char *) malloc (primary->header_size);
  assert (primary->header != 0);

  /* default command port */
  primary -> port = DADA_DEFAULT_PWC_PORT;

  fprintf (stderr, "dada_pwc on port %d\n", primary->port);

  /* for multi-threaded use of primary */
  pthread_mutex_init(&(primary->mutex), NULL);
  pthread_cond_init (&(primary->cond), NULL);

  /* command parser */
  primary -> parser = command_parse_create ();

  command_parse_add (primary->parser, dada_pwc_cmd_header, primary,
		     "header", "set the primary header", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_duration, primary,
		     "duration", "set the duration of next recording", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_start, primary,
		     "start", "enter the recording state", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_stop, primary,
		     "stop", "enter the idle state", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_clock, primary,
		     "clock", "enter the clocking state", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_record_start, primary,
		     "rec_start", "enter the recording state", NULL);

  command_parse_add (primary->parser, dada_pwc_cmd_record_stop, primary,
		     "rec_stop", "enter the clocking state", NULL);

  primary -> server = 0;

  return primary;
}

int dada_pwc_set_header_size (dada_pwc_t* primary, unsigned header_size)
{
  if (!primary)
    return -1;

  pthread_mutex_lock (&(primary->mutex));

  primary -> header_size = header_size;
  primary -> header = (char *) realloc (primary->header, header_size);
  assert (primary->header != 0);

  pthread_mutex_unlock (&(primary->mutex));

  return 0;
}


/*! Destroy a DADA primary write client connection */
int dada_pwc_serve (dada_pwc_t* primary)
{
  if (!primary)
    return -1;

  if (primary->server) {
    fprintf (stderr, "dada_pwc_serve: server already launched");
    return -1;
  }

  primary -> server = command_parse_server_create (primary -> parser);

  command_parse_server_set_welcome (primary -> server,
				    "DADA primary write client command");

  /* open the command/control port */
  return command_parse_serve (primary->server, primary->port);
}

/*! Destroy a DADA primary write client connection */
int dada_pwc_destroy (dada_pwc_t* primary)
{
  return 0;
}

/*! Primary write client should exit when this is true */
int dada_pwc_quit (dada_pwc_t* primary)
{
  return 0;
}

/*! Check to see if a command has arrived */
int dada_pwc_command_check (dada_pwc_t* primary)
{
  if (!primary)
    return -1;

  if (primary->command.code == dada_pwc_no_command)
    return 0;

  return 1;
}

/*! Get the next command from the connection; wait until command received */
dada_pwc_command_t dada_pwc_command_get (dada_pwc_t* primary)
{
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;

  if (!primary) {
    command.code = -1;
    return command;
  }

  pthread_mutex_lock (&(primary->mutex));

  while (primary->command.code == dada_pwc_no_command)
    pthread_cond_wait(&(primary->cond), &(primary->mutex));

  command = primary->command;

  pthread_mutex_unlock (&(primary->mutex));

  return command;
}

/*! \param new_state the state that the primary write client has entered
    \param utc the UTC time at which this state was entered */
int dada_pwc_command_ack (dada_pwc_t* primary, int new_state, time_t utc)
{
  if (!primary)
    return -1;

  switch (primary->command.code) {

  case dada_pwc_no_command:
    fprintf (stderr, "Cannot acknowledge no command\n");
    return -1;

  case dada_pwc_header:
    if (new_state != dada_pwc_prepared) {
      fprintf (stderr, "HEADER acknowledgement state must be PREPARED\n");
      return -1;
    }
    break;

  case dada_pwc_clock:
    if (new_state != dada_pwc_clocking) {
      fprintf (stderr, "CLOCK acknowledgement state must be CLOCKING\n");
      return -1;
    }
    primary->utc_start = utc;
    break;

  case dada_pwc_record_start:
    if (new_state != dada_pwc_recording) {
      fprintf (stderr, "REC_START acknowledgement state must be RECORDING\n");
      return -1;
    }
    break;

  case dada_pwc_record_stop:
    if (new_state != dada_pwc_clocking) {
      fprintf (stderr, "REC_STOP acknowledgement state must be CLOCKING\n");
      return -1;
    }
    break;

  case dada_pwc_start:
    if (new_state != dada_pwc_recording) {
      fprintf (stderr, "START acknowledgement state must be RECORDING\n");
      return -1;
    }
    primary->utc_start = utc;
    break;

  case dada_pwc_stop:
    if (new_state != dada_pwc_idle) {
      fprintf (stderr, "STOP acknowledgement state must be IDLE\n");
      return -1;
    }
    break;

  }

  pthread_mutex_lock (&(primary->mutex));

  primary->command.code = dada_pwc_no_command;
  primary->command.byte_count = 0;
  primary->state = new_state;

  pthread_cond_signal (&(primary->cond));
  pthread_mutex_unlock (&(primary->mutex));

  return 0;
}



