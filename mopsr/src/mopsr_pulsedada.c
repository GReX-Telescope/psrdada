#include "dada_def.h"
#include "dada_generator.h"
#include "ascii_header.h"
#include "multilog.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

/* #define _DEBUG 1 */
void fill_int8_array (int8_t * data, unsigned npoints, double stddev);

void usage()
{
  fprintf (stdout,
	   "mopsr_pulsedada [options] header_file\n"
     " -g       write gaussian distributed data\n"
     " -d len   duration of event in milliseconds [default 20]\n"
     " -e time  epoch of event [default 10]\n"
     " -t secs  total length of data to generate [default 20]\n"
     " -h       print help\n");
}

int main (int argc, char **argv)
{
  /* DADA Logger */
  multilog_t* log = 0;

  /* header to use for the data block */
  char * header_file = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* ascii char to fill data with */
  char fill_char = 0;

  // in milliseconds
  float event_duration = 20;

  // in seconds
  float event_epoch = 10;

  // in seconds
  float data_length = 20;

  int arg = 0;

  while ((arg=getopt(argc,argv,"d:e:ht:v")) != -1)
    switch (arg) {

    case 'd':
      if (sscanf(optarg, "%f", &event_duration) != 1)
      {
        fprintf (stderr, "ERROR: could not parse event duration from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'e':
      if (sscanf(optarg, "%f", &event_epoch) != 1)
      {
        fprintf (stderr, "ERROR: could not parse event epoch from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'h':
      usage();
      return EXIT_SUCCESS;

    case 't':
      if (sscanf (optarg, "%f", &data_length) != 1) {
        fprintf (stderr,"ERROR: could not parse data length from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return EXIT_FAILURE;
    }

  if ((argc - optind) != 1) 
  {
    fprintf (stderr, "Error: a header file must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  header_file = strdup(argv[optind]);
  if (verbose)
    fprintf (stderr, "Header file= %s\n", header_file);

  size_t header_size = 4096;
  char header[header_size+1];

  if (fileread (header_file, header, header_size) < 0) {
    fprintf (stderr, "Could not read header from %s\n", header_file);
    exit (EXIT_FAILURE);
  }

  log = multilog_open ("mopsr_pulsedada", 0);

  multilog_add (log, stderr);

  unsigned nchan, ndim, nbit, nant;
  float tsamp;
  uint64_t resolution;
  char utc_start[20];

  ascii_header_get (header, "NCHAN", "%u", &nchan);
  ascii_header_get (header, "NDIM", "%u", &ndim);
  ascii_header_get (header, "NBIT", "%u", &nbit);
  ascii_header_get (header, "NANT", "%u", &nant);
  ascii_header_get (header, "TSAMP", "%f", &tsamp);
  ascii_header_get (header, "RESOLUTION", "%"PRIu64, &resolution);
  ascii_header_get (header, "UTC_START", "%s", utc_start);

  double nsamps = (double) data_length / (double) (tsamp / 1000000);

  uint64_t nsamp = (uint64_t) floor (nsamps);

  uint64_t nsamp_resolution = resolution / (nchan * ndim * (nbit/8) * nant);

  fprintf (stderr, "nsamps=%lf nsamp=%"PRIu64" nsamp_resolution=%"PRIu64"\n", nsamps, nsamp, nsamp_resolution);

  uint64_t nblocks = nsamp / nsamp_resolution;

  // seed the RNG
  srand ( time(NULL) );

  char * lo_data = (char *) malloc (sizeof(char) * 2 * nsamp_resolution);
  char * hi_data = (char *) malloc (sizeof(char) * 2 * nsamp_resolution);

  uint64_t isamp_tot = 0;
  uint64_t event_start_samp = (uint64_t) floor ((1000000 * event_epoch) / tsamp);
  uint64_t event_end_samp = event_start_samp + ((uint64_t) ((1000 * event_duration) / tsamp));

  int perms = S_IRUSR | S_IRGRP;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;

  char filename[1024];
  sprintf (filename, "%s.dada", utc_start);
  int fd = open (filename, flags, perms);
  uint64_t iblock;
  unsigned ichan, iant, isamp;

  write (fd, header, 4096);
  fprintf (stderr, "nblocks=%"PRIu64" event samples [%"PRIu64" - %"PRIu64"]\n", nblocks, event_start_samp, event_end_samp);

  for (iblock=0; iblock<nblocks; iblock++)
  {
    if (verbose)
      fprintf (stderr, "writing block %"PRIu64" of %"PRIu64"\n", iblock, nblocks);
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        fill_int8_array (lo_data, 2 * nsamp_resolution, 20);

        if ((isamp_tot + nsamp_resolution >= event_start_samp) &&
            (isamp_tot + nsamp_resolution < event_end_samp))
        {
          if (verbose)
            fprintf (stderr, "[%d][%d][%d] fill hi\n", iblock, ichan, iant);
          fill_int8_array (hi_data, 2 * nsamp_resolution, 60);
        }

        for (isamp=0; isamp<nsamp_resolution; isamp++)
        {
          if (isamp_tot+isamp >= event_start_samp && isamp_tot+isamp < event_end_samp)
            write (fd, hi_data + 2*isamp, 2);
          else
            write (fd, lo_data + 2*isamp, 2);

        }
      }
      isamp_tot += nsamp_resolution;
    }
  }

  close (fd);

  return EXIT_SUCCESS;
}

void fill_int8_array (int8_t * data, unsigned nvals, double stddev)
{
  int i;
  for (i=0; i < nvals; i++)
  {
    data[i] = (int8_t) rand_normal (0, (double) stddev);
  }
}

