#include "dada_pwc_nexus.h"
#include "futils.h"

/*! parse a configuration into unique headers for each primary write client */
int dada_pwc_nexus_header_parse (dada_pwc_nexus_t* n, const char* buffer);

int dada_pwc_nexus_config (void* context, FILE* output, char* args)
{
  dada_pwc_nexus_t* nexus = (dada_pwc_nexus_t*) context;
  char* filename = args;
  char* buffer = 0;
  long fsize = 0;
  int error = 0;

  if (nexus->pwc->state != dada_pwc_idle) {
    fprintf (output, "Cannot config when not IDLE\n");
    return -1;
  }

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_nexus_config fopen (%s, 'r')\n", args);
#endif

  fsize = filesize (filename);

  if (fsize < 1) {
    fprintf (output, "Cannot open '%s'\n", filename);
    return -1;
  }

  buffer = (char *) malloc (fsize + 1);

  if (fileread (filename, buffer, fsize+1) < 0) {
    fprintf (output, "Cannot read '%s'\n", filename);
    return -1;
  }

  error = dada_pwc_nexus_header_parse (nexus, buffer);
  free (buffer);

  return error;

}
