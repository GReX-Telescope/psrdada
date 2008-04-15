
#include "dada_def.h"
#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/* #define _DEBUG 1 */

void usage()
{
  fprintf (stdout,
	   "dada_edit - query or change DADA header parameters in file[s]\n"
	   "\n"
	   "USAGE: dada_edit [options] filename[s]\n"
	   "\n"
	   "WHERE options are:\n"
	   " -c KEY=VAL     set KEY to VAL \n"
	   " -i header.txt  over-write header with text in specified file\n");
}


int main (int argc, char **argv)
{
  /* Flag set in verbose mode */
  char verbose = 0;

  /* command string to be parsed into key=val pair */
  char* command = 0;
  char* key = 0;
  char* val = 0;

  /* name of file to be installed in header */
  char* install = 0;
  FILE* install_file = 0;

  /* descriptor of currently open file */
  FILE* current_file = 0;

  /* header read from current_file */
  char* current_header = 0;
  unsigned current_header_size = 0;

  /* header size read from current file */
  unsigned hdr_size = DADA_DEFAULT_HEADER_SIZE;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:i:hv")) != -1)

    switch (arg)
    {      
    case 'c':
      command = optarg;
      key = strdup (command);
      val = strchr (key, '=');
      if (!val)
      {
	fprintf (stderr, "dada_edit could not parse key=value from '%s'\n",
		 optarg);
	return -1;
      }

      /* terminate the key and advance to the value */
      *val = '\0';
      val ++;

      break;

    case 'i':
      install = optarg;
      break;

    case 'v':
      verbose=1;
      break;
   
    case 'h':
    default:
      usage ();
      return 0;
      
    }

  for (arg = optind; arg < argc; arg++)
  {
    current_file = fopen (argv[arg], "r+");
    if (!current_file)
    {
      fprintf (stderr, "dada_edit: could not open '%s': %s\n",
	       argv[arg], strerror(errno));
      continue;
    }

    do
    {
      rewind (current_file);

      current_header = realloc (current_header, hdr_size);
      current_header_size = hdr_size;

      if (fread (current_header, 1, current_header_size, current_file)
	  != current_header_size)
      {
	fprintf (stderr, "dada_edit: could not read %u bytes: %s\n",
		 current_header_size, strerror(errno));
	return -1;
      }

      /* Get the header size */
      if (ascii_header_get (current_header, "HDR_SIZE", "%u", &hdr_size) != 1)
      {
	fprintf (stderr, "dada_edit: could not parse HDR_SIZE\n");
	return -1;
      }

      /* Ensure that the incoming header fits in the client header buffer */
    }
    while (hdr_size > current_header_size);

    if (command && ascii_header_set (current_header, key, val) != 0)
    {
      fprintf (stderr, "dada_edit: could not set %s = %s\n", key, val);
      return -1;
    }

    if (install)
    {
    }

    if (command || install)
    {
      rewind (current_file);

      if (fwrite (current_header, 1, current_header_size, current_file)
	  != current_header_size)
      {
	fprintf (stderr, "dada_edit: could not write %u bytes: %s\n",
		 current_header_size, strerror(errno));
	return -1;
      }
    }

    else
    {
      fprintf (stdout, current_header);
    }

    fclose (current_file);
  }

  return EXIT_SUCCESS;
}

