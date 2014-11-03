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
           "mopsr_test_transpose_TS_to_ST\n"
           " -c nchan   number of antennas\n"
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
  int nant = 352;
  int nsamp = 1024;
  int ndim = 2;

  while ((arg=getopt(argc,argv,"c:d:t:v")) != -1)
  {
    switch (arg) 
    {
      case 'd':
        device = atoi(optarg);
        break;

      case 'c':
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

  multilog_t * log = multilog_open ("mopsr_test_transpose_FT_to_TF", 0);
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

  unsigned nblocks = 64;
  size_t nbytes = nant * nsamp * ndim;
  size_t in_block_size = nbytes * nblocks;
  size_t out_block_size = in_block_size * sizeof(float);

  error = cudaMalloc (&d_in, in_block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", in_block_size);
    return -1;
  }

  // d_out includes unpack from int8_t to float 
  error = cudaMalloc (&d_out, out_block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", out_block_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_in, in_block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", in_block_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_out, out_block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", out_block_size);
    return -1;
  }

  // initialize input data
  unsigned npart = in_block_size / (nant * ndim);
  unsigned ipart, iant;
  int8_t * h_in_ptr = (int8_t *) h_in;

  // encode values as TS
  for (ipart=0; ipart<npart; ipart++)
  {
    for (iant=0; iant<nant; iant++)
    {
      h_in_ptr[0] = (int8_t) (ipart % 128);
      h_in_ptr[1] = (int8_t) (ipart % 128);
      h_in_ptr += 2;
    }
  }

  {
    multilog (log, LOG_INFO, "cudaMemcpyHostToDevice (%ld)\n", in_block_size);
    error = cudaMemcpyAsync (d_in, h_in, in_block_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    multilog (log, LOG_INFO, "mopsr_transpose_TS_to_ST (%ld, %d)\n", in_block_size, nant);
    mopsr_transpose_TS_to_ST (d_in, d_out, in_block_size, nant, stream);
    check_error_stream( "mopsr_transpose_TS_to_ST", stream);

    multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%ld)\n", out_block_size);
    error = cudaMemcpyAsync (h_out, d_out, out_block_size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }

  cudaStreamSynchronize(stream);

  // ST test
  uint64_t nerr = 0;
  float * h_out_ptr = (float *) h_out;
  float val;
  multilog (log, LOG_INFO, "[S][T]\n");
  for (iant=0; iant<nant; iant++)
  {
    for (ipart=0; ipart<npart; ipart++)
    {
      val = (float) (ipart % 128);
      if ((h_out_ptr[0] != val) || (h_out_ptr[1] != val))
      {
        if (nerr < 50)
          multilog (log, LOG_INFO, "[%d][%d] WRONG (%f, %f) != %f\n", iant, ipart, h_out_ptr[0], h_out_ptr[1], val);
        nerr ++;
      }
      else
      {
        //multilog (log, LOG_INFO, "[%d][%d] RIGHT (%f, %f) != %f\n", iant, ipart, h_out_ptr[0], h_out_ptr[1], val);
      }
      h_out_ptr += 2;
    }
  }

#if 1
  h_in_ptr = (int8_t *) h_in;

  // test samp offsets are correct
  for (ipart=0; ipart<npart; ipart++)
  {
    for (iant=0; iant<nant; iant++)
    {
      h_in_ptr[0] = (int8_t) (iant % 128);
      h_in_ptr[1] = (int8_t) (iant % 128);
      h_in_ptr += 2;
    }
  }

  {
    multilog (log, LOG_INFO, "cudaMemcpyHostToDevice (%d)\n", in_block_size);
    error = cudaMemcpyAsync (d_in, h_in, in_block_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    multilog (log, LOG_INFO, "mopsr_transpose_TS_to_ST (%d, %d)\n", in_block_size, nant);
    mopsr_transpose_TS_to_ST (d_in, d_out, in_block_size, nant, stream);
    check_error_stream( "mopsr_transpose_TS_to_ST", stream);

    multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", out_block_size);
    error = cudaMemcpyAsync (h_out, d_out, out_block_size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }

  cudaStreamSynchronize(stream);

  // TF test
  nerr = 0;
  h_out_ptr = (float *) h_out;
  multilog (log, LOG_INFO, "[S][T]\n");
  for (iant=0; iant<nant; iant++)
  {
    for (ipart=0; ipart<npart; ipart++)
    {
      val = (float) (iant % 128);
      if ((h_out_ptr[0] != val) || (h_out_ptr[1] != val))
      {
        if (nerr < 50)
          multilog (log, LOG_INFO, "[%d][%d] WRONG (%f, %f) != %f\n", ipart, iant, h_out_ptr[0], h_out_ptr[1], val);
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

  return (EXIT_SUCCESS);
}

