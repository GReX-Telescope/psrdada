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
#include <cpgplot.h>


void usage ()
{
	fprintf(stdout, "mopsr_test_delays bays_file modules_file\n"
    " -c nchan    number of channels\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  int nchan = 128;

  char verbose = 0;

  while ((arg = getopt(argc, argv, "c:hv")) != -1) 
  {
    switch (arg)  
    {
      case 'c':
        nchan = atoi(optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

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

  unsigned ichan, imod;

  char * bays_file = strdup (argv[optind+1]);
  int nbay;
  mopsr_bay_t * all_bays = read_bays_file (bays_file, &nbay);
  if (!all_bays)
  {
    fprintf (stderr, "ERROR: failed to read bays file [%s]\n", bays_file);
    return EXIT_FAILURE;
  }

  char * modules_file = strdup (argv[optind+1]);
  int nmod;
  mopsr_module_t * modules = read_modules_file (modules_file, &nmod);
  if (!modules)
  {
    fprintf (stderr, "ERROR: failed to read modules file [%s]\n", modules_file);
    return EXIT_FAILURE;
  }

  // preconfigure a source
  mopsr_source_t source;
  char position[32];
  sprintf (source.name, "3C273");
  fprintf (stderr, "source=%s\n", source.name);
  sprintf (position, "12:29:06.7");
  mopsr_delays_hhmmss_to_rad (position, &(source.raj));
  sprintf (position, "02:03:09.0");
  mopsr_delays_ddmmss_to_rad (position, &(source.decj));
  fprintf (stderr, "ra=%lf dec=%lf\n", source.raj, source.decj);

  time_t utc_start = str2utctime ("2014-03-05-15:34:30");
  if (utc_start == (time_t)-1)
  {
    fprintf(stderr, "open: could not parse start time\n");
    return -1;
  }

  mopsr_chan_t * channels = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t) * nchan);
  for (ichan=0; ichan<nchan; ichan++)
  {
    channels[ichan].number = ichan;
    channels[ichan].bw     = 0.78125;
    channels[ichan].cfreq  = (800 + (0.78125/2) + (ichan * 0.78125));
  }

  struct timeval timestamp;

  mopsr_delay_t ** delays = (mopsr_delay_t **) malloc(sizeof(mopsr_delay_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays[imod] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);

  cpgopen ("/xs");
  cpgask (0);

  // advance time by 1 millisecond each plot
  const double delta_time = 2;
  double obs_offset_seconds = 0;

  char apply_instrumental = 0;
  char apply_geometric = 1;
  char is_tracking = 0;
  double tsamp = 1.28;
  unsigned nant = 1;

  while ( 1 )
  {
    struct timeval timestamp;
    timestamp.tv_sec = floor(obs_offset_seconds);
    timestamp.tv_usec = (obs_offset_seconds - (double) timestamp.tv_sec) * 1000000;
    timestamp.tv_sec += utc_start;

    if (calculate_delays (nbay, all_bays, nant, modules, nchan, channels,
                          source, timestamp, delays, apply_instrumental,
                          apply_geometric, is_tracking, tsamp) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    timestamp.tv_sec -= utc_start;
    mopsr_delays_plot (nmod, nchan, delays, timestamp);

    obs_offset_seconds += delta_time;
    usleep (100000);
  }

  cpgclos();

  free (channels);
  free (modules);

  for (imod=0; imod<nmod; imod++)
    free (delays[imod]);
  free (delays);


  return 0;
}
