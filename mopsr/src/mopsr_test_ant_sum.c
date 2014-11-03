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
           "mopsr_test_sum_ant\n"
           " -c nchan   number of channels\n"
           " -d device  cuda device to use\n"
           " -t nsamp   number of time samples\n"
           " -v         verbose mode\n");
}




int main (int argc, char **argv)
{
  int device = 0;         // cuda device to use

  cudaStream_t stream;    // cuda stream for engine

  void * d_in;            // device memory for input
  void * d_out;           // device memory for output
  void * h_in;            // host memory for input
  void * h_out;           // host memory for output

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  int nant = 4;
  int nchan = 40;
  int nsamp = 128;
  int ndim = 2;

  while ((arg=getopt(argc,argv,"a:c:d:t:v")) != -1)
  {
    switch (arg) 
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'd':
        device = atoi(optarg);
        break;

      case 'c':
        nchan = atoi(optarg);
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

  unsigned nblocks = 1024;
  size_t nbytes = nchan * nant * nsamp * ndim;
  size_t block_size = nbytes * nblocks;

  error = cudaMalloc (&d_in, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }

  error = cudaMalloc (&d_out, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_in, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_out, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }

  // initialize input data
  unsigned npart = block_size / (nchan * nant * ndim);
  unsigned ipart, ichan, iant;
  int8_t * h_in_ptr = (int8_t *) h_in;

  // test samp offsets are correct
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (ipart=0; ipart<npart; ipart++)
      {
        h_in_ptr[0] = (int8_t) (ipart % 32);
        h_in_ptr[1] = (int8_t) (ipart % 32);
        h_in_ptr += 2;
      }
    }
  }

  {
    multilog (log, LOG_INFO, "cudaMemcpyHostToDevice (%d)\n", block_size);
    error = cudaMemcpyAsync (d_in, h_in, block_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    multilog (log, LOG_INFO, "mopsr_input_sum_ant (%d, %d, %d)\n", block_size, nchan, nant);
    mopsr_input_sum_ant (stream, d_in, d_out, block_size, nchan, nant);
    check_error_stream( "mopsr_input_sum_ant", stream);

    multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", block_size);
    error = cudaMemcpyAsync (h_out, d_out, block_size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }

  cudaStreamSynchronize(stream);

#if 1
  // TF test
  uint64_t nerr = 0;
  int8_t * h_out_ptr = (int8_t *) h_out;
  int8_t val;
  multilog (log, LOG_INFO, "[F][T]\n");
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (ipart=0; ipart<npart; ipart++)
    {
      val = (int8_t) (ipart % 32);
      if ((h_out_ptr[0] != val) || (h_out_ptr[1] != val))
      {
        if (nerr < 50)
          multilog (log, LOG_INFO, "[%d][%d] WRONG (%d, %d) != %d\n", ichan, ipart, h_out_ptr[0], h_out_ptr[1], val);
        nerr ++;
      }
      h_out_ptr += 2;
    }
  }
#endif

  cudaFree (d_in);
  cudaFree (d_out);
  cudaFreeHost (h_in);
  cudaFreeHost (h_out);

  return EXIT_SUCCESS;
}

