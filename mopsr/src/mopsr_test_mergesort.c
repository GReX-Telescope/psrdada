/***************************************************************************
 *  
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "multilog.h"
#include "dada_cuda.h"
#include "mopsr_delays_cuda_hires.h"
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
           "mopsr_test_mergesort\n"
           " -v         verbose mode\n");
}

int compare (const void * a, const void * b);
int compare (const void * a, const void * b)
{
  return ( *(float*)a - *(float*)b );
}


int main (int argc, char **argv)
{
  int device = 0;         // cuda device to use

  cudaStream_t stream;    // cuda stream for engine

  void * d_keys_in;
  void * d_vals_in;
  void * d_keys_out;
  void * d_vals_out;
  void * d_keys_out2;
  void * d_vals_out2;

  void * h_keys_in;
  void * h_vals_in;
  void * h_keys_out;
  void * h_vals_out;
  void * h_keys_out2;
  void * h_vals_out2;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"v")) != -1)
  {
    switch (arg) 
    {
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

  unsigned npt = 1024;

  size_t keys_size = npt * sizeof(float);

  cudaMallocHost ((void **) &h_keys_in, keys_size);
  cudaMallocHost ((void **) &h_keys_out, keys_size);
  cudaMallocHost ((void **) &h_keys_out2, keys_size);

  cudaMalloc ((void **) &d_keys_in, keys_size);
  cudaMalloc ((void **) &d_keys_out, keys_size);
  cudaMalloc ((void **) &d_keys_out2, keys_size);

  unsigned i;
  float * in = (float *) h_keys_in;
  for (i=0; i<npt; i++)
  {
    in[i] = npt - i;
  }

  cudaMemcpyAsync (d_keys_in, h_keys_in, keys_size, cudaMemcpyHostToDevice, stream);

  cudaStreamSynchronize(stream);

  //test_merge_sort  (stream, (float *) d_keys_out, (float*) d_keys_in, npt, 1);
  //cudaMemcpyAsync (h_keys_out, d_keys_out, keys_size, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  test_merge_sort2 (stream, (float *) d_keys_out2, (float *) d_keys_in, npt, 1);
  cudaMemcpyAsync (h_keys_out2, d_keys_out2, keys_size, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  float * tmp = (float *) malloc(sizeof(float) * npt);

  for (i=0; i<npt; i++)
  {
    tmp[i] = in[i];
  }
    
  // now quicksort the array
  qsort ((void *) tmp, npt, sizeof(float), compare);

  float * gout = (float *) h_keys_out;
  float * gout2 = (float *) h_keys_out2;

  unsigned nerrs = 0;

  for (i=0; i<npt; i++)
  {
    if (tmp[i] != gout2[i])
      nerrs++;
  }
  fprintf (stderr, "nerrors=%u\n", nerrs);
  cudaFree (d_keys_in);
  cudaFree (d_keys_out);
  cudaFree (d_keys_out2);
  cudaFreeHost (h_keys_in);
  cudaFreeHost (h_keys_out);
  cudaFreeHost (h_keys_out2);
  free (tmp);

  return EXIT_SUCCESS;
}

