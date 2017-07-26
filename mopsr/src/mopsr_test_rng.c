/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "dada_cuda.h"
#include "mopsr_cuda.h"
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
#include <cuda_runtime.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <inttypes.h>

void usage()
{
  fprintf (stdout,
           "mopsr_testrng\n"
           " -a nant    number of antenna\n"
           " -c nchan   number of channels\n"
           " -d device  cuda device to use\n"
           " -i ipfb    pfb index\n"
           " -p npfb    number of pfbs\n"
           " -v         verbose mode\n");
}


int main (int argc, char **argv)
{

  int device = 0;         // cuda device to use

  cudaStream_t stream;    // cuda stream for engine

  void * d_rstates;       // device memory for input
  void * h_rstates;       // host memory for input

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  int nchan = 320;
  int nant  = 8;
  unsigned npfb  = 44;
  unsigned ipfb  = 0;

  while ((arg=getopt(argc,argv,"a:c:d:i:p:v")) != -1)
  {
    switch (arg) 
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'd':
        device = atoi(optarg);
        break;

      case 'i':
        ipfb = atoi(optarg);
        break;

      case 'c':
        nchan = atoi(optarg);
        break;

      case 'p':
        npfb = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;
        
      default:
        usage ();
        return 0;
    }
  }

  multilog_t * log = multilog_open ("mopsr_testrng", 0);
  multilog_add (log, stderr);

  // select the gpu device
  int n_devices = dada_cuda_get_device_count();
  multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

  if ((device < 0) && (device >= n_devices))
  {
    multilog (log, LOG_ERR, "no CUDA devices available [%d]\n",
              n_devices);
    return -1;
  }

  if (dada_cuda_select_device (device) < 0)
  {
    multilog (log, LOG_ERR, "could not select requested device [%d]\n", device);
    return -1;
  }

  char * device_name = dada_cuda_get_device_name (device);
  if (!device_name)
  {
    multilog (log, LOG_ERR, "could not get CUDA device name\n");
    return -1;
  }
  multilog (log, LOG_INFO, "Using device %d : %s\n", device, device_name);
  free(device_name);

  // setup the cuda stream for operations
  cudaError_t error = cudaStreamCreate(&stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create CUDA stream\n");
    return -1;
  }

  const size_t maxthreads = 1024;
  const unsigned nrngs = nchan * nant * maxthreads;

  size_t curand_size = (size_t) (nrngs * hires_curandState_size());

  multilog (log, LOG_INFO, "allocating %ld bytes of device memory for d_rstates\n", curand_size);
  error = cudaMalloc (&d_rstates, curand_size);
  if (verbose)
    multilog (log, LOG_INFO, "d_rstates=%p\n", d_rstates);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not allocate %ld bytes of device memory\n", curand_size);
    return -1;
  }

  error = cudaMallocHost (&h_rstates, curand_size);
  if (verbose)
    multilog (log, LOG_INFO, "d_rstates=%p\n", d_rstates);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not allocate %ld bytes of host memory\n", curand_size);
    return -1;
  }

  unsigned long long seed = (unsigned long long) 0;

  hires_init_rng_sparse (stream, seed, nrngs, ipfb, npfb, d_rstates);

  error = cudaMemcpyAsync (h_rstates, d_rstates, curand_size, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  cudaStreamSynchronize(stream);
/*
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP;
  int fd = open ("RNG_states.raw", flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "ERROR: failed to open output file for writing: %s\n", strerror(errno));
    return (EXIT_FAILURE);
  }

  write (fd, h_rstates, curand_size);
  close (fd);
*/
  cudaFree (d_rstates);
  cudaFreeHost (h_rstates);

  return EXIT_SUCCESS;
}

