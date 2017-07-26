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
           "mopsr_testkernel\n"
           " -a nant    number of antenna\n"
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
  void * d_corr;          // device memory for corrections
  void * h_in;            // host memory for input
  void * h_out;           // host memory for output
  void * h_corr;          // host memory for corrections

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  int nchan = 40;
  int nant  = 16;
  int nsamp = 32;
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

  multilog_t * log = multilog_open ("mopsr_testkernel", 0);
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

  unsigned unique_corrections = 32;
  size_t corr_size = sizeof(float complex) * unique_corrections * nchan;
  error = cudaMallocHost((void **) &(h_corr), corr_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of pinned memory\n", corr_size);
    return -1;
  }

  unsigned ichan, ipt, icorr;
  float theta;
  float ratio = 2 * M_PI * (5.0 / 32.0);
  float complex * h_corr_ptr = (float complex *) h_corr;
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (ipt=0; ipt < unique_corrections; ipt++)
    {
      icorr = (ichan * unique_corrections) + ipt;
      theta = (ichan + 20) * ratio * ipt;
      h_corr_ptr[icorr] = sin (theta) - cos(theta) * I;
    }
  }

  error = cudaMalloc((void **) &d_corr, corr_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "failed to malloc %d bytes on device: %s\n", corr_size, cudaGetErrorString(error));
    return -1;
  }

  error = cudaMemcpyAsync (d_corr, h_corr, corr_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // initialize input data
  unsigned npart = block_size / (nchan * nant * ndim);
  unsigned ipart, iant, chanant, ival;

  //int8_t * h_in_ptr = (int8_t *) h_in;
  uint16_t * h_in_ptr = (uint16_t *) h_in;
  uint16_t val;

  srand(4);
  for (ipart=0; ipart<npart; ipart++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        val = rand() % 65535;
        h_in_ptr[0] = val;
        h_in_ptr++;
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

    multilog (log, LOG_INFO, "mopsr_input_transpose_TFS_to_FST (%d, %d, %d)\n", block_size, nchan, nant);
    mopsr_input_transpose_TFS_to_FST_hires (stream, d_in, d_out, block_size, nchan, nant);
    check_error_stream( "mopsr_input_transpose_TFS_to_FST", stream);

    //multilog (log, LOG_INFO, "mopsr_input_rephase (%d, %d, %d)\n", block_size, nchan, nant);
    //mopsr_input_rephase (stream, d_out, d_corr, block_size, nchan, nant);
    //check_error_stream( "mopsr_input_rephase", stream);

    //multilog (log, LOG_INFO, "mopsr_input_transpose_FST_to_STF (%d, %d, %d)\n", block_size, nchan, nant);
    //mopsr_input_transpose_FST_to_STF (stream, d_out, d_in, block_size, nchan, nant);
    //check_error_stream( "mopsr_input_transpose_FST_to_STF", stream);

    multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", block_size);
    error = cudaMemcpyAsync (h_out, d_out, block_size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }

  cudaStreamSynchronize(stream);

  unsigned nchanant = nchan * nant;
  unsigned in_idx;

#if 1
  // FST test
  uint64_t nerr = 0;
  uint16_t * h_out_ptr = (uint16_t *) h_out;
  h_in_ptr = (uint16_t *) h_in;
  multilog (log, LOG_INFO, "[F][S][T]\n");
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (ipart=0; ipart<npart; ipart++)
      {
        in_idx = (ipart * nchanant) + (ichan * nant) + iant;
        val = h_in_ptr[in_idx];
        if (h_out_ptr[0] != val)
        {
          if (nerr < 32)
            multilog (log, LOG_INFO, "[%d][%d][%d] WRONG (%d) != h_in[%d]=%d\n", ichan, iant, ipart, h_out_ptr[0], in_idx, val);
          nerr ++;
        }
        //else
        //  multilog (log, LOG_INFO, "[%d][%d][%d] RIGHT (%d) == h_in[%d]=%d\n", ichan, iant, ipart, h_out_ptr[0], in_idx, val);

        h_out_ptr ++;
      }
    }
  }
#endif

#if 0
  // STF test
  uint64_t nerr = 0;
  int16_t * h_out_ptr = (int16_t *) h_out;
  multilog (log, LOG_INFO, "[S][T][F]\n");
  for (iant=0; iant<nant; iant++)
  {
    for (ipart=0; ipart<npart; ipart++)
    {
      for (ichan=0; ichan<nchan; ichan++)
      {
        in_idx = (ipart * nchanant) + (ichan * nant) + iant;
        val = h_in_ptr[in_idx];

        if (h_out_ptr[0] != val)
        {
          if (nerr < 32)
            multilog (log, LOG_INFO, "[%d][%d][%d] WRONG (%d) != h_in[%d]=%d\n", ichan, iant, ipart, h_out_ptr[0], in_idx, val);
          nerr ++;
        }
        h_out_ptr ++;
/*
        //chanant = (ichan * nant + iant) % 127;
        val = (int8_t) (ipart % 128);
        if ((h_out_ptr[0] != val) || (h_out_ptr[1] != val))
        {
          if (nerr < 50)
            multilog (log, LOG_INFO, "[%d][%d][%d] WRONG (%d, %d) != %d [expected]\n", iant, ipart, ichan, h_out_ptr[0], h_out_ptr[1], val);

          nerr ++;
        }
        h_out_ptr += 2;
*/
      }
    }
  }
#endif

  cudaFree (d_in);
  cudaFree (d_out);
  cudaFree (d_corr);
  cudaFreeHost (h_in);
  cudaFreeHost (h_out);
  cudaFreeHost (h_corr);

  return EXIT_SUCCESS;
}

