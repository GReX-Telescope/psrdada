#include "dada_pwc.h"
#include "utc.h"

#include <stdlib.h>

int dada_primary_get_header (void* context, FILE* fptr, char* args)
{
  dada_primary_t* primary = (dada_primary_t*) context;

  if (strlen (args) > primary->header_size)
    fprintf (fptr, "header too large (max %d bytes)\n", primary->header_size);

  pthread_mutex_lock(&(primary->mutex));
  strcpy (primary->header, args);
  pthread_mutex_unlock(&(primary->mutex));

  fprintf (fptr, "HEADER=%s\n", primary->header);
  return 0;
}

int dada_primary_get_start (void* context, FILE* fptr, char* args)
{
  dada_primary_t* primary = (dada_primary_t*) context;
  struct tm date;

  if (args) {
    primary->utc = str2tm (&date, args);
    if (primary->utc == (time_t)-1) {
      fprintf (fptr, "Could not parse start time from '%s'\n", args);
      return -1;
    }
  }

  primary -> command = dada_primary_start;

  return 0;
}

/*! Create a new DADA primary write client connection */
dada_primary_t* dada_primary_create ()
{
  dada_primary_t* primary = (dada_primary_t*) malloc (sizeof(dada_primary_t));

  /* default header size */
  primary -> header_size = 4096;

  /* default command port */
  primary -> port = 0xdada;

  primary -> state = dada_primary_idle;
  primary -> command = dada_primary_no_command;

  /* for multi-threaded use of primary */
  pthread_mutex_init(&(primary->mutex), NULL);
  pthread_cond_init (&(primary->cond), NULL);

  /* command parser */
  primary -> parser = command_parse_create ();

  command_parse_add (primary->parser, dada_primary_get_header, primary,
		     "header", "set the primary header", NULL);

  command_parse_add (primary->parser, dada_primary_get_start, primary,
		     "start", "enter the recording valid state", NULL);

  primary -> server = command_parse_server_create (primary -> parser);

  command_parse_server_set_welcome (primary -> server,
				    "DADA primary write client command");

  /* open the command/control port */
  command_parse_serve (primary->server, primary->port);

  return primary;
}

/*! Destroy a DADA primary write client connection */
int dada_primary_destroy (dada_primary_t* primary)
{
  return 0;
}

/*! Check to see if a command has arrived */
int dada_primary_command_check (dada_primary_t* primary)
{
  if (!primary)
    return -1;

  if (primary->command != dada_primary_no_command)
    return 1;

  return 0;
}

/*! Get the next command from the connection; wait until command received */
int dada_primary_command_get (dada_primary_t* primary)
{
  int command = dada_primary_no_command;
  
  if (!primary)
    return -1;

  pthread_mutex_lock(&(primary->mutex));

  while (primary->command == dada_primary_no_command)
    pthread_cond_wait(&(primary->cond), &(primary->mutex));

  command = primary->command;

  pthread_mutex_unlock(&(primary->mutex));

  return command;
}

/*! Reply to the last command received */
int dada_primary_command_reply (dada_primary_t* primary)
{
  if (!primary)
    return -1;

  if (primary->command == dada_primary_no_command)
    return -1;

  pthread_mutex_lock(&(primary->mutex));
  primary->command = dada_primary_no_command;
  pthread_cond_signal (&(primary->cond));
  pthread_mutex_unlock(&(primary->mutex));

  return 0;
}



