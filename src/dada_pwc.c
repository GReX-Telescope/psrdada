#include "dada_pwc.h"
#include "dada.h"
#include "utc.h"

#include <stdlib.h>
#include <string.h>

/*! Set the command state */
int dada_pwc_command_set (dada_pwc_t* primary, FILE* output,
			  dada_pwc_command_t command)
{
  int ret = 0;

  if (!primary)
    return -1;
  
  pthread_mutex_lock(&(primary->mutex));

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

  pthread_mutex_unlock(&(primary->mutex));

  return ret;
}

int dada_pwc_cmd_header (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;
  char* hdr = args;

  if (strlen (args) > primary->header_size) {
    fprintf (fptr, "header too large (max %d bytes)\n", primary->header_size);
    return -1;
  }

  /* replace \ with new line */
  while ( (hdr = strchr(hdr, '\\')) != 0 )
    *hdr = '\n';

  if (args)
    strcpy (primary->header, args);
    
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_header;
  command.header = primary->header;

  if (dada_pwc_command_set (primary, fptr, command) < 0)
    return -1;

  return 0;
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
  command.utc = dada_pwc_parse_time (fptr, args);

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_record_start (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_record_start;
  command.utc = dada_pwc_parse_time (fptr, args);

  return dada_pwc_command_set (primary, fptr, command);
}

int dada_pwc_cmd_record_stop (void* context, FILE* fptr, char* args)
{
  dada_pwc_t* primary = (dada_pwc_t*) context;

  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;
  command.code = dada_pwc_record_stop;
  command.utc = dada_pwc_parse_time (fptr, args);

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

  return dada_pwc_command_set (primary, fptr, command);
}

/*! Create a new DADA primary write client connection */
dada_pwc_t* dada_pwc_create ()
{
  dada_pwc_t* primary = (dada_pwc_t*) malloc (sizeof(dada_pwc_t));

  /* default header size */
  primary -> header_size = DADA_DEFAULT_HDR_SIZE;
  primary -> header = (char *) malloc (primary->header_size);

  /* default command port */
  primary -> port = DADA_DEFAULT_PWC_PORT;

  fprintf (stderr, "dada_pwc on port %d\n", primary->port);

  primary -> state = dada_pwc_idle;
  primary -> command.code = dada_pwc_no_command;

  /* for multi-threaded use of primary */
  pthread_mutex_init(&(primary->mutex), NULL);
  pthread_cond_init (&(primary->cond), NULL);

  /* command parser */
  primary -> parser = command_parse_create ();

  command_parse_add (primary->parser, dada_pwc_cmd_header, primary,
		     "header", "set the primary header", NULL);

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

  pthread_mutex_lock(&(primary->mutex));

  primary -> header_size = header_size;
  primary -> header = (char *) realloc (primary->header_size, header_size);

  pthread_mutex_unlock(&(primary->mutex));

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

  pthread_mutex_lock(&(primary->mutex));

  while (primary->command.code == dada_pwc_no_command)
    pthread_cond_wait(&(primary->cond), &(primary->mutex));

  command = primary->command;

  pthread_mutex_unlock(&(primary->mutex));

  return command;
}

/*! Reply to the last command received */
int dada_pwc_command_ack (dada_pwc_t* primary, int new_state)
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
    break;

  case dada_pwc_stop:
    if (new_state != dada_pwc_idle) {
      fprintf (stderr, "STOP acknowledgement state must be IDLE\n");
      return -1;
    }
    break;

  }

  pthread_mutex_lock(&(primary->mutex));

  primary->command.code = dada_pwc_no_command;
  primary->state = new_state;

  pthread_cond_signal (&(primary->cond));
  pthread_mutex_unlock(&(primary->mutex));

  return 0;
}



