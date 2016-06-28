/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "multilog.h"
#include "mopsr_def.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>
#include <complex.h>
#include <float.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <inttypes.h>

#include "stopwatch.h"

void usage()
{
  fprintf (stdout,
           "mopsr_test_cpu_bf\n"
           " -a ant     number of antenna\n"
           " -t nsamp   number of time samples\n"
           " -v         verbose mode\n");
}


int main (int argc, char **argv)
{
  int device = 0;         // cuda device to use

  void * h_in;            // host memory for input
  void * h_out;           // host memory for output

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  int nant = 352;
  int nsamp = 32768;
  int ndim = 2;

  while ((arg=getopt(argc,argv,"a:t:v")) != -1)
  {
    switch (arg) 
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 't':
        nsamp = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  multilog_t * log = multilog_open ("mopsr_test_sum_ant", 0);
  multilog_add (log, stderr);

  unsigned nblocks = 1;
  size_t nbytes_in = nant * nsamp * ndim;
  size_t nbytes_ou = nsamp * ndim * sizeof(float);

  if (verbose)
    multilog (log, LOG_INFO, "allocating %ld bytes for input\n", nbytes_in);
  h_in = malloc(nbytes_in);
  if (!h_in)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of memory\n", nbytes_in);
    return -1;
  }

  if (verbose)
    multilog (log, LOG_INFO, "allocating %ld bytes for output\n", nbytes_ou);
  h_out = malloc(nbytes_ou);
  if (!h_out)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of memory\n", nbytes_ou);
    return -1;
  }

  memset (h_in, 0, nbytes_in);
  memset (h_out, 0, nbytes_ou);

  const unsigned max_bytes = 4096;
  const unsigned chunk_nsamps = max_bytes / (ndim * sizeof(float)); // read 512 samples at a time (1024 bytes)
  const unsigned chunk_nvals = chunk_nsamps * ndim;
  const unsigned nchunk = nsamp / chunk_nsamps;

  int8_t in_buf[chunk_nvals];
  complex float ou_buf[chunk_nsamps];
  complex float val;

  unsigned ichunk, iant, ival;

  // initialise data rate timing library
  stopwatch_t wait_sw;

  if (verbose)
    multilog (log, LOG_INFO, "Starting\n");
  StartTimer(&wait_sw);
  {
    for (ichunk=0; ichunk<nchunk; ichunk++)
    {
      int8_t * indat = (int8_t *) h_in + (ichunk * chunk_nsamps * ndim);

      for (iant=0; iant<nant; iant++)
      {
        const complex float phase = iant + 2 * I;

        memcpy (in_buf, indat, chunk_nvals);

        for (ival=0; ival<chunk_nsamps; ival++)
        {
          val = ((float) in_buf[2*ival]) + ((float) in_buf[2*ival+1]) * I;
          ou_buf[ival] += val * phase;
        }
        indat += nsamp;
      }
      memcpy ((void *) (h_out + (ichunk * chunk_nsamps)), (void *) ou_buf, chunk_nsamps * ndim * sizeof(float));
    }
  }
  StopTimer(&wait_sw);

  if (verbose)
    multilog (log, LOG_INFO, "Ended\n");

  unsigned long elapsed_ms = ReadTimer(&wait_sw);
  double bytes_per_second = 550000000;
  double time_sec = (double) elapsed_ms / 1e3;
  double data_length = nbytes_in / bytes_per_second;
  double performance = time_sec / data_length;

  multilog (log, LOG_INFO, "time=%lf data_length=%lf  performance=%lf %% realtime\n", time_sec, data_length, performance * 100);

  free (h_in);
  free (h_out);
}

