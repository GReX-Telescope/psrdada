#include "dada_pwc_nexus.h"
#include "ascii_header.h"
#include "dada_def.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

// #define _DEBUG 1

/*! Initialize a new dada_node_t struct with default empty values */
void dada_pwc_node_init (dada_pwc_node_t* node)
{
  node_t* node_base = (node_t*) node;
  node_init (node_base);

  node -> header = 0;
  node -> header_size = 0;
  node -> state = dada_pwc_undefined;
}

/*! Return pointer to a newly allocated and initialized dada_node_t struct */
node_t* dada_pwc_node_create ()
{
  dada_pwc_node_t* node = (dada_pwc_node_t*) malloc (sizeof(dada_pwc_node_t));
  assert (node != 0);
  dada_pwc_node_init (node);
  return (node_t*) node;
}

/*! Pointer to function that initializes a new connection with a node */
int dada_pwc_nexus_node_init (node_t* node)
{
  dada_pwc_node_t* dada_pwc_node = (dada_pwc_node_t*) node;

  unsigned buffer_size = 1024;
  static char* buffer = 0;

  char* key = 0;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_node_init: receiving the welcome message\n");
#endif

  if (node_recv (node, buffer, buffer_size) < 0) {
    dada_pwc_node -> state = dada_pwc_undefined;
    return -1;
  }

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_node_init: requesting state\n");
#endif

  if (node_send (node, "state") < 0) {
    dada_pwc_node -> state = dada_pwc_undefined;
    return -1;
  }

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_node_init: receiving state\n");
#endif

  if (node_recv (node, buffer, buffer_size) < 0) {
    dada_pwc_node -> state = dada_pwc_undefined;
    return -1;
  }

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_node_init: received '%s'\n", buffer);
#endif

  key = strtok (buffer, " \t\n\r");

  dada_pwc_node->state = dada_pwc_string_to_state (key);
  return 0;
}

/*! load lines from param_file, then take only first word from each line */
int dada_pwc_nexus_parse_params (string_array_t* params, const char* filename)
{
  const char* whitespace = " \r\t\n";
  unsigned iparam = 0;

  if (string_array_load (params, filename) < 0)
    return -1;

  for (iparam=0; iparam < string_array_size(params); iparam++)
    strtok (string_array_get(params, iparam), whitespace);

  return 0;
}

/*! Parse DADA PWC nexus configuration parameters from the config buffer */
int dada_pwc_nexus_parse (nexus_t* n, const char* config)
{
  /* the nexus is actually a DADA PWC nexus */
  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) n;

  /* the name of the parameter file parsed from the config */
  char* param_file = 0;

  /* the size of the header parsed from the config */
  unsigned hdr_size = 0;

  /* the status returned by sub-routines */
  int status = 0;

  /* call the base class configuration parser */
  if (nexus_parse (n, config) < 0)
    return -1;

  /* get the heaer size */
  if (ascii_header_get (config, "HDR_SIZE", "%u", &hdr_size) < 0)
    fprintf (stderr, "dada_pwc_nexus_parse: using default HDR_SIZE\n");
  else
    dada_pwc_set_header_size (nexus->pwc, hdr_size);

  /* get the specification parameter file name */
  param_file = malloc (FILENAME_MAX);
  assert (param_file != 0);

  if (ascii_header_get (config, "SPEC_PARAM_FILE", "%s", param_file) < 0)
    fprintf (stderr, "dada_pwc_nexus_parse: no SPEC_PARAM_FILE in config\n");

  else {
    /* load specification parameters from the specified file */
    fprintf (stderr, "dada_pwc_nexus_parse: loading specification parameters\n"
	     "from %s\n", param_file);
    status = dada_pwc_nexus_parse_params (nexus->config_params, param_file);
  }

  free (param_file);
  return status;
}

int dada_pwc_nexus_update_state (dada_pwc_nexus_t* nexus)
{
  nexus_t* base = (nexus_t*) nexus;
  dada_pwc_node_t* node = 0;
  dada_pwc_state_t state = 0;

  unsigned inode = 0;
  unsigned nnode = 0;

  pthread_mutex_lock (&(base->mutex));

  nnode = base->nnode;

  for (inode = 0; inode < nnode; inode++) {
    node = base->nodes[inode];
    if (inode == 0)
      state = node->state;
    else if (state != node->state) {
      state = dada_pwc_undefined;
      break;
    }
  }

  pthread_mutex_unlock (&(base->mutex));

  nexus->pwc->state = state;

  return 0;
}

int dada_pwc_nexus_handle_message (void* me, unsigned inode, const char* msg)
{
  char state_string [16];
  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) me;
  dada_pwc_node_t* node = nexus->nexus.nodes[inode];

  char* state_change = strstr (msg, "STATE = ");

  if (state_change) {
    sscanf (state_change, "STATE = %s", state_string);
    node->state = dada_pwc_string_to_state (state_string);
    dada_pwc_nexus_update_state (nexus);
  }

  return 0;
}

/*! Send a unique header to each of the nodes */
int dada_pwc_nexus_cmd_config (void* context, FILE* fptr, char* args);

/*! Report on the state of each of the nodes */
int dada_pwc_nexus_cmd_state (void* context, FILE* fptr, char* args)
{
  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) context;
  nexus_t* base = (nexus_t*) context;
  dada_pwc_node_t* node = 0;

  unsigned inode = 0;
  unsigned nnode = 0;

  fprintf (fptr, "overall: %s\n", 
	   dada_pwc_state_to_string( nexus->pwc->state ));

  pthread_mutex_lock (&(base->mutex));

  nnode = base->nnode;

  for (inode = 0; inode < nnode; inode++) {
    node = base->nodes[inode];
    fprintf (fptr, "PWC_%d: %s\n", inode,
	     dada_pwc_state_to_string( node->state ));
  }

  pthread_mutex_unlock (&(base->mutex));

  return 0;
}

int dada_pwc_nexus_cmd_duration (void* context, FILE* fptr, char* args)
{
  unsigned buffer_size = 128;
  static char* buffer = 0;

  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) context;

  if (dada_pwc_cmd_duration (nexus->pwc, fptr, args) < 0)
    return -1;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

  snprintf (buffer, buffer_size, "duration %s", args);
  return nexus_send ((nexus_t*)nexus, buffer);
}

void dada_pwc_nexus_init (dada_pwc_nexus_t* nexus)
{
  nexus_t* nexus_base = (nexus_t*) nexus;
  nexus_init (nexus_base);

  if (nexus_base->node_prefix)
    free (nexus_base->node_prefix);
  nexus_base->node_prefix = strdup ("PWC");
  assert (nexus_base->node_prefix != 0);

  nexus_base->node_port = DADA_DEFAULT_PWC_PORT;
  nexus_base->node_create = &dada_pwc_node_create;
  nexus_base->node_init   = &dada_pwc_nexus_node_init;
  nexus_base->nexus_parse = &dada_pwc_nexus_parse;
  nexus_base->mirror = nexus_create ();

  nexus_base->mirror->node_port = DADA_DEFAULT_PWC_LOG;

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_init dada_pwc_create\n");
#endif

  
  nexus->pwc = dada_pwc_create ();

  /* do not convert times and sample counts into bytes */
  nexus->pwc->convert_to_bytes = 0;

  /* set up the monitor of the mirror */
  nexus->monitor = monitor_create ();
  nexus->monitor->nexus = nexus_base->mirror;
  nexus->monitor->handle_message = &dada_pwc_nexus_handle_message;
  nexus->monitor->context = nexus;

  /* convert time_t to local time strings */
  nexus->convert_to_tm = localtime;

  nexus->config_params = string_array_create ();

  nexus->header_template = 0;

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_init command_parse_add\n");
#endif

  /* replace the header command with the config command */
  command_parse_remove (nexus->pwc->parser, "header");
  command_parse_add (nexus->pwc->parser, dada_pwc_nexus_cmd_config, nexus,
		     "config", "configure all nodes", NULL);

  /* replace the state command */
  command_parse_remove (nexus->pwc->parser, "state");
  command_parse_add (nexus->pwc->parser, dada_pwc_nexus_cmd_state, nexus,
		     "state", "get the current state of all nodes", NULL);

  /* replace the duration command */
  command_parse_remove (nexus->pwc->parser, "duration");
  command_parse_add (nexus->pwc->parser, dada_pwc_nexus_cmd_duration, nexus,
		     "duration", "set the duration of next recording", NULL);
}

/*! Create a new DADA nexus */
dada_pwc_nexus_t* dada_pwc_nexus_create ()
{
  dada_pwc_nexus_t* nexus = 0;
  nexus = (dada_pwc_nexus_t*) malloc (sizeof(dada_pwc_nexus_t));
  assert (nexus != 0);
  dada_pwc_nexus_init (nexus);
  return nexus;
}

/*! Destroy a DADA nexus */
int dada_pwc_nexus_destroy (dada_pwc_nexus_t* nexus)
{
  return nexus_destroy ((nexus_t*) nexus);
}

int dada_pwc_nexus_configure (dada_pwc_nexus_t* nexus, const char* filename)
{
  return nexus_configure ((nexus_t*) nexus, filename);
}

int dada_pwc_nexus_send (dada_pwc_nexus_t* nexus, dada_pwc_command_t command)
{
  unsigned buffer_size = 128;
  static char* buffer = 0;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

  switch (command.code) {

  case dada_pwc_clock:

    if (dada_pwc_set_state (nexus->pwc, dada_pwc_clocking, time(0)) < 0)
      return -1;

    return nexus_send ((nexus_t*)nexus, "clock");
    
  case dada_pwc_record_start:

    if (dada_pwc_set_state (nexus->pwc, dada_pwc_recording, time(0)) < 0)
      return -1;

    strftime (buffer, buffer_size, "rec_start " DADA_TIMESTR,
	      nexus->convert_to_tm (&command.utc));
    return nexus_send ((nexus_t*)nexus, buffer);
    
  case dada_pwc_record_stop:

    if (dada_pwc_set_state (nexus->pwc, dada_pwc_clocking, time(0)) < 0)
      return -1;

    strftime (buffer, buffer_size, "rec_stop " DADA_TIMESTR,
	      nexus->convert_to_tm (&command.utc));
    return nexus_send ((nexus_t*)nexus, buffer);
    
  case dada_pwc_start:
    
    if (dada_pwc_set_state (nexus->pwc, dada_pwc_recording, time(0)) < 0)
      return -1;

    if (!command.utc)
      return nexus_send ((nexus_t*)nexus, "start");

    strftime (buffer, buffer_size, "start " DADA_TIMESTR,
	      nexus->convert_to_tm (&command.utc));
    return nexus_send ((nexus_t*)nexus, buffer);
    
  case dada_pwc_stop:

    if (dada_pwc_set_state (nexus->pwc, dada_pwc_idle, time(0)) < 0)
      return -1;

    if (!command.utc)
      return nexus_send ((nexus_t*)nexus, "stop");

    strftime (buffer, buffer_size, "stop " DADA_TIMESTR,
	      nexus->convert_to_tm (&command.utc));
    return nexus_send ((nexus_t*)nexus, buffer);
    
  }

  return -1;
}

int dada_pwc_nexus_serve (dada_pwc_nexus_t* nexus)
{
  /* the DADA PWC command */
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;

  if (dada_pwc_serve (nexus->pwc) < 0) {
    fprintf (stderr, "dada_pwc_nexus_serve: could not start PWC server\n");
    return -1;
  }

  monitor_launch (nexus->monitor);

  while (!dada_pwc_quit (nexus->pwc)) {

    command = dada_pwc_command_get (nexus->pwc);

    if (dada_pwc_nexus_send (nexus, command) < 0) {

      fprintf (stderr, "error issuing command = %d\n", command.code);

    }

  }

  return 0;
}
