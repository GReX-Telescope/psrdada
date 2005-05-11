#include "dada_pwc_nexus.h"
#include "ascii_header.h"
#include "dada.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*! Initialize a new dada_node_t struct with default empty values */
void dada_node_init (dada_node_t* node)
{
  node_t* node_base = (node_t*) node;
  node_init (node_base);

  node -> header = 0;
  node -> header_size = 0;
}

/*! Return pointer to a newly allocated and initialized dada_node_t struct */
node_t* dada_node_create ()
{
  dada_node_t* node = (dada_node_t*) malloc (sizeof(dada_node_t));
  assert (node != 0);
  dada_node_init (node);
  return (node_t*) node;
}

/*! load lines from param_file, then take only first word from each line */
int dada_pwc_nexus_parse_params (string_array* params, const char* param_file)
{
  const char* whitespace = " \r\t\n";
  unsigned iparam = 0;

  if (string_array_load (params, param_file) < 0)
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

/*! Send a unique header to each of the nodes */
int dada_pwc_nexus_cmd_config (void* context, FILE* fptr, char* args);

void dada_pwc_nexus_init (dada_pwc_nexus_t* nexus)
{
  nexus_t* nexus_base = (nexus_t*) nexus;
  nexus_init (nexus_base);

  if (nexus_base->node_prefix)
    free (nexus_base->node_prefix);

  nexus_base->node_port = DADA_DEFAULT_PWC_PORT;
  nexus_base->node_prefix = strdup ("PWC");
  assert (nexus_base->node_prefix != 0);

  nexus_base->node_create = &dada_node_create;
  nexus_base->nexus_parse = &dada_pwc_nexus_parse;

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_init dada_pwc_create\n");
#endif

  nexus->pwc = dada_pwc_create ();

  /* do not convert times and sample counts into bytes */
  nexus->pwc->convert_to_bytes = 0;

  nexus->config_params = string_array_create ();

  nexus->header_template = 0;

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_init command_parse_add\n");
#endif

  command_parse_add (nexus->pwc->parser, dada_pwc_nexus_cmd_config, nexus,
		     "config", "configure all nodes", NULL);

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
  int status = 0;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

  switch (command.code) {

  case dada_pwc_clock:
    fprintf (stderr, "start clocking\n");
    nexus_send ((nexus_t*)nexus, "clock");
    status = dada_pwc_set_state (nexus->pwc, dada_pwc_clocking, time(0));
    break;
    
  case dada_pwc_record_start:
    fprintf (stderr, "clocking->recording\n");
    strftime (buffer, buffer_size, "rec_start %F-%T", localtime(&command.utc));
    nexus_send ((nexus_t*)nexus, buffer);
    status = dada_pwc_set_state (nexus->pwc, dada_pwc_recording, time(0));
    break;
    
  case dada_pwc_record_stop:
    fprintf (stderr, "recording->clocking\n");
    strftime (buffer, buffer_size, "rec_stop %F-%T", localtime(&command.utc));
    nexus_send ((nexus_t*)nexus, buffer);
    status = dada_pwc_set_state (nexus->pwc, dada_pwc_clocking, time(0));
    break;
    
  case dada_pwc_start:
    fprintf (stderr, "start recording\n");
    if (!command.utc)
      nexus_send ((nexus_t*)nexus, "start");
    else {
      strftime (buffer, buffer_size, "start %F-%T", localtime(&command.utc));
      nexus_send ((nexus_t*)nexus, buffer);
    }
    status = dada_pwc_set_state (nexus->pwc, dada_pwc_recording, time(0));
    break;
    
  case dada_pwc_stop:
    fprintf (stderr, "stopping\n");
    if (!command.utc)
      nexus_send ((nexus_t*)nexus, "stop");
    else {
      strftime (buffer, buffer_size, "stop %F-%T", localtime(&command.utc));
      nexus_send ((nexus_t*)nexus, buffer);
    }
    status = dada_pwc_set_state (nexus->pwc, dada_pwc_idle, time(0));
    break;
    
  }

  return status;
}

int dada_pwc_nexus_serve (dada_pwc_nexus_t* nexus)
{
  /* the DADA PWC command */
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;

  if (dada_pwc_serve (nexus->pwc) < 0) {
    fprintf (stderr, "dada_pwc_nexus_serve: could not start PWC server\n");
    return -1;
  }

  while (!dada_pwc_quit (nexus->pwc)) {

    command = dada_pwc_command_get (nexus->pwc);

    if (dada_pwc_nexus_send (nexus, command) < 0) {

      fprintf (stderr, "error issuing command = %d\n", command.code);

    }

  }

  return 0;
}
