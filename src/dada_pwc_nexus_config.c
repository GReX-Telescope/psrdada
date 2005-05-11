#include "dada_pwc_nexus.h"
#include "futils.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*! parse a configuration into unique headers for each primary write client */
int dada_pwc_nexus_header_parse (dada_pwc_nexus_t* n, const char* buffer);

int dada_pwc_nexus_cmd_config (void* context, FILE* output, char* args)
{
  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) context;
  dada_node_t* node = 0;

  unsigned inode, nnode = nexus_get_nnode ((nexus_t*) nexus);

  char* filename = args;
  char* buffer = 0;
  char* hdr = 0;

  long fsize = 0;
  int error = 0;

  if (nexus->pwc->state != dada_pwc_idle) {
    fprintf (output, "Cannot config when not IDLE\n");
    return -1;
  }

  if (!args) {
    fprintf (output, "Please specify config filename\n");
    return -1;
  }

  filename = strsep (&args, " \t\n\r");

  if (!filename || filename[0] == '\0') {
    fprintf (output, "Please specify config filename\n");
    return -1;
  } 

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_config fopen (%s, 'r')\n", filename);
#endif

  fsize = filesize (filename);

  if (fsize < 1) {
    fprintf (output, "Cannot open '%s'\n", filename);
    return -1;
  }

  buffer = (char *) malloc (fsize + 1);
  assert (buffer != 0);

  if (fileread (filename, buffer, fsize+1) < 0) {
    fprintf (output, "Cannot read '%s'\n", filename);
    return -1;
  }

  error = dada_pwc_nexus_header_parse (nexus, buffer);
  free (buffer);

  if (error)
    return error;

  for (inode=0; inode < nnode; inode++) {

    node = (dada_node_t*) nexus->nexus.nodes[inode];
    strcpy (nexus->pwc->header, "header ");
    strcat (nexus->pwc->header, node->header);

    hdr = nexus->pwc->header;
    /* replace new line with \\ */
    while ( (hdr = strchr(hdr, '\n')) != 0 )
      *hdr = '\\';

    nexus_send_node ((nexus_t*) nexus, inode, nexus->pwc->header);

  }

  nexus->pwc->state = dada_pwc_prepared;
  return 0;
}

