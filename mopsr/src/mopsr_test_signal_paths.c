/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "mopsr_delays.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>


void usage ()
{
	fprintf(stdout, "mopsr_test_signal_paths file\n"
    " -a nant     number of antennae\n" 
    " -c nchan    number of channels\n" 
    " -t nsamp    number of samples\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nchan = 40;
  unsigned nant = 4;
  unsigned ndim = 2;
  unsigned ntap = 5;
  unsigned nsamp = 512;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:c:d:ht:v")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'c':
        nchan = atoi(optarg);
        break;

      case 'd':
        device = atoi (optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

      case 't':
        nsamp = atoi (optarg);
        break;
      
      case 'v':
        verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 argument is required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  unsigned isamp, ichan, iant, imod;;

  char * signal_paths_file = strdup (argv[optind]);

  int npfbs;
  mopsr_pfb_t * pfbs = read_signal_paths_file (signal_paths_file, &npfbs);
  if (!pfbs)
  {
    fprintf (stderr, "failed to read signal parths file [%s]\n", signal_paths_file);
    return -1;
  }

  free (pfbs);

  return EXIT_SUCCESS;
}
