/***************************************************************************
 *  
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_generator.h"
#include "dada_def.h"
#include "multilog.h"
#include "dada_cuda.h"
#include "mopsr_delays_cuda.h"
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
           "mopsr_test_sk_kernel\n"
           " -a nant    number of antenna\n"
           " -c nchan   number of channels\n"
           " -d device  cuda device to use\n"
           " -t nsamp   number of time samples\n"
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

  void * d_in;
  void * d_out;
  void * d_s1;
  void * d_s2;
  void * d_mask;
  void * d_sigmas;
  void * d_rstates;
  void * d_thresh;
  void * h_in;
  void * h_out;
  void * h_s1;
  void * h_s2;
  void * h_mask;
  void * h_thresh;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  unsigned nchan = 20;
  unsigned nant  = 2;
  uint64_t nsamp = 16*1024*6;
  unsigned ndim = 2;
  char replace_noise = 1;

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
        nsamp = (uint64_t) atoi(optarg);
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
  
  size_t ant_scales_size = nant * sizeof(float);
  multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for ant scales\n", ant_scales_size);
  void * h_ant_scales;
  error = cudaMallocHost ((void **) &h_ant_scales, ant_scales_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ant_scales_size);
    return -1;
  }

  unsigned iant;
  float * scale = (float *) h_ant_scales;
  for (iant=0; iant<nant; iant++)
    scale[iant] = 1;
  mopsr_delay_copy_scales (stream, h_ant_scales, ant_scales_size);

  unsigned nblocks = 1;
  size_t nbytes = nchan * nant * nsamp * ndim * sizeof(float);
  size_t block_size = nbytes * nblocks;

  multilog (log, LOG_INFO, "allocating %ld bytes for d_in\n", block_size);
  error = cudaMalloc (&d_in, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }
  multilog (log, LOG_INFO, "allocating %ld bytes for d_out\n", block_size);
  error = cudaMalloc (&d_out, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size);
    return -1;
  }

  size_t nbytes_s2 = nchan * nant * (nsamp / 1024) * sizeof(float);
  multilog (log, LOG_INFO, "allocating %ld bytes for d_s2\n", nbytes_s2);
  error = cudaMalloc (&d_s2, nbytes_s2);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", nbytes_s2);
    return -1;
  }

  size_t n_memory = MOPSR_MEMORY_BLOCKS;
  size_t nbytes_s1 = nbytes_s2 * n_memory;
  multilog (log, LOG_INFO, "allocating %ld bytes for d_s1\n", nbytes_s1);
  error = cudaMalloc (&d_s1, nbytes_s1);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", nbytes_s1);
    return -1;
  }
  cudaMemsetAsync (d_s1, 0, nbytes_s1, stream);

  size_t nbytes_mask = nchan * nant * (nsamp / 1024) * sizeof(int8_t);
  size_t block_size_mask = nbytes_mask * nblocks;
  multilog (log, LOG_INFO, "allocating %ld bytes for input\n", block_size_mask);
  error = cudaMalloc (&d_mask, block_size_mask);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size_mask);
    return -1;
  }

  size_t nbytes_sigmas = nchan * nant * sizeof(float);
  size_t block_size_sigmas = nbytes_sigmas * nblocks;
  multilog (log, LOG_INFO, "allocating %ld bytes for input\n", block_size_sigmas);
  error = cudaMalloc (&d_sigmas, block_size_sigmas);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", block_size_sigmas);
    return -1;
  }

  size_t thresh_size = nchan * nant * sizeof(float) * 2;
  multilog (log, LOG_INFO, "allocating %ld bytes for input\n", thresh_size);
  error = cudaMalloc (&d_thresh, thresh_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of device memory\n", thresh_size);
    return -1;
  }

  const size_t maxthreads = 1024;
  const unsigned nrngs = nchan * nant * maxthreads;
  size_t d_curand_size = (size_t) (nrngs * mopsr_curandState_size());
  multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_rstates\n", d_curand_size);
  error = cudaMalloc (&d_rstates, d_curand_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", d_curand_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_in, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", block_size);
    return -1;
  }
  error = cudaMallocHost((void **) &h_out, block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", block_size);
    return -1;
  }

  error = cudaMallocHost((void **) &h_s1, nbytes_s1);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", nbytes_s1);
    return -1;
  }
  error = cudaMallocHost((void **) &h_s2, nbytes_s2);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", nbytes_s2);
    return -1;
  }

  error = cudaMallocHost((void **) &h_mask, block_size_mask);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", block_size_mask);
    return -1;
  }

  error = cudaMallocHost((void **) &h_thresh, thresh_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "could not create allocated %ld bytes of host memory\n", thresh_size);
    return -1;
  }

  // initialise the RNGs
  unsigned long long seed = (unsigned long long) time(0) * 1000000;
  mopsr_init_rng (stream, seed, nrngs, d_rstates);

  srand ( time(NULL) );
  fill_gaussian_float ((float *) h_in, (nchan * nant * nsamp * ndim), 10.0f);

  // mask some bad!
  float * in = (float * ) h_in;
  float * out = (float * ) h_out;
  unsigned ichan, isamp, i, isum;
  unsigned nsums = nsamp / 1024;
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (isum=0; isum<nsums; isum++)
      {
        for (i=0; i<1024; i++)
        {
          //if (ichan == 10)
          //if ((isum % 2 == 0) && ichan == 10 && 0)
          //{
          //  in[0] += 50;
          //  in[1] += 50;
          //}
          in += 2;
          out += 2;
        }
      }
    }
  }

  // copy input to GPU
  multilog (log, LOG_INFO, "cudaMemcpyHostToDevice(%d)\n", block_size);
  error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }
  error = cudaMemcpyAsync (d_out, h_out, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  size_t memory_stride = nant * nchan * (nsamp / 1024) * sizeof(float);
  size_t memory_offset = 0;
  for (i=0; i<MOPSR_MEMORY_BLOCKS;i++)
  {
    mopsr_test_skcompute (stream, d_in, d_s1 + memory_offset, d_s2, nchan, nant, nbytes);
    memory_offset += memory_stride;
  }

  multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d) - S1\n", nbytes_s1);
  error = cudaMemcpyAsync (h_s1, d_s1, nbytes_s1, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
   multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
   return -1;
  }
  multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d) - S2\n", nbytes_s2);
  error = cudaMemcpyAsync (h_s2, d_s2, nbytes_s2, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
   multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
   return -1;
  }

  cudaStreamSynchronize(stream);

  // now check output!
  float tol = 1.0001;

  in = (float * ) h_in;
  float * outs1 = (float * ) h_s1;
  float * outs2 = (float * ) h_s2;

  float s1, s2, power, re, im, gpu_s1, gpu_s2;

  multilog (log, LOG_INFO, "Testing the SK component calculation\n");

  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (isamp=0; isamp<nsamp; isamp+=1024)
      {
        s1 = 0;
        s2 = 0;
        for (i=0; i<1024; i++)
        {
          re = in[2*i];
          im = in[2*i+1];
          power = (re * re) + (im * im);
          s1 += power;
          s2 += (power * power);
        }

        gpu_s1 = outs1[0];
        gpu_s2 = outs2[0];

        if (((s1 > gpu_s1) && (s1 / gpu_s1 > tol)) || ((gpu_s1 > s1) && (gpu_s1 / s1 > tol)))
          printf ("[%d][%d][%d] S1 %f != %f\n", ichan, iant, isamp/1024, s1, gpu_s1);
        if (((s2 > gpu_s2) && (s2 / gpu_s2 > tol)) || ((gpu_s2 > s2) && (gpu_s2 / s2 > tol)))
          printf ("[%d][%d][%d] S2 %f != %f\n", ichan, iant, isamp/1024, s2, gpu_s2);

        in  += 2048;
        outs1 ++;
        outs2 ++;
      }
    }
  }
  multilog (log, LOG_INFO, "Testing the SK component complete\n");

  unsigned s1_count = n_memory;
  unsigned s1_memory = n_memory;
  unsigned ndat_cpl = nsamp / 1024;

  mopsr_test_compute_power_limits (stream, d_s1, d_thresh, nsums, nant, nchan, ndat_cpl, s1_count, s1_memory, d_rstates);

  multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", thresh_size);
  error = cudaMemcpyAsync (h_thresh, d_thresh, thresh_size, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
   multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
   return -1;
  }
  cudaStreamSynchronize(stream);

  fprintf (stderr, "nsums=%d\n", nsums);

  outs1 = (float * ) h_s1;
  float * h_thr = (float *) h_thresh;
  float * tmp = (float *) malloc(sizeof(float) * nsums);

  float upper, lower, gpu_median, gpu_sigma, gpu_upper, gpu_lower;

  multilog (log, LOG_INFO, "Testing the SK power limits\n");
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (i=0; i<nsums; i++)
      {
        tmp[i] = outs1[i];
      }
      outs1 += nsums;
    
      // now quicksort the array
      qsort ((void *) tmp, nsums, sizeof(float), compare);
      float median = tmp[nsums/2];
      for (i=0; i<nsums; i++)
        tmp[i] = fabsf(tmp[i] - median);

      qsort ((void *) tmp, nsums, sizeof(float), compare);
      float stddev = tmp[nsums/2] * 1.4826;

      upper = median + 4 * stddev;
      lower = median - 4 * stddev;

      gpu_median = h_thr[0];
      gpu_sigma  = h_thr[1];

      gpu_upper = gpu_median + 4 * gpu_sigma;
      gpu_lower = gpu_median - 4 * gpu_sigma;

      if (((upper > gpu_upper) && (upper / gpu_upper > tol)) || ((gpu_upper > upper) && (gpu_upper / upper > tol)))
        fprintf (stderr, "[%d][%d] [%f - %f] median=%f sigma=%f || [%f - %f] median=%f\n",
                ichan, iant, upper, lower, 
                median, stddev,
                gpu_upper, gpu_lower, gpu_lower + ((gpu_upper - gpu_lower)/2));
      if (((lower > gpu_lower) && (lower / gpu_lower > tol)) || ((gpu_lower > lower) && (gpu_lower / lower > tol)))
        fprintf (stderr, "[%d][%d] [%f - %f] median=%f sigma=%f || [%f - %f] median=%f\n",
                ichan, iant, upper, lower, 
                median, stddev,
                gpu_upper, gpu_lower, gpu_lower + ((gpu_upper - gpu_lower)/2));

      h_thr += 2;
    }
  }
  multilog (log, LOG_INFO, "Testing the power limits complete\n");

  // now compute the zap mask based on the s1s2 data
  mopsr_test_skdetect (stream, d_s1, d_s2, d_thresh, d_mask, d_sigmas, nsums, nant, nchan, nsamp);

  multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", block_size_mask);
  error = cudaMemcpyAsync (h_mask, d_mask, nbytes_mask, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
   multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
   return -1;
  }

  cudaStreamSynchronize(stream);

  const float m = 1024;
  const float m_fac = (m + 1) / (m - 1);

  int8_t * h_ptr = (int8_t *) h_mask;
  in = (float * ) h_in;
  outs1 = (float * ) h_s1;
  outs2 = (float * ) h_s2;

  float sk_l = 0.834186;
  float sk_h = 1.21695;
  float sk;

  multilog (log, LOG_INFO, "Testing the zap mask\n");

  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    { 
      gpu_upper = h_thr[0];
      gpu_lower = h_thr[1];

      for (isum=0; isum<nsums; isum++)
      {
        s1 = outs1[0];
        s2 = outs2[0];
        sk = m_fac * (m * (s2 / (s1 * s1)) - 1);

        if (sk < sk_l || sk > sk_h || s1 < gpu_lower || s1 > gpu_upper)
        {
          if (*h_ptr == 0)
            if (verbose)
              fprintf (stderr, "[%d][%d][%d] s1=%f s2=%f sk=%f [%f - %f] thresh[%f - %f] mask=%d\n", ichan, iant, isum, s1, s2, sk, sk_l, sk_h, gpu_upper, gpu_lower, *h_ptr);
        }
        else
        {
          if (*h_ptr != 0)
            if (verbose)
              fprintf (stderr, "[%d][%d][%d] s1=%f s2=%f sk=%f [%f - %f] thresh[%f - %f] mask=%d\n", ichan, iant, isum, s1, s2, sk, sk_l, sk_h, gpu_upper, gpu_lower, *h_ptr);
        }
        h_ptr ++;
        outs1 ++;
        outs2 ++;
      }
      h_thr += 2;
    }
  }
  multilog (log, LOG_INFO, "Testing the zap mask complete\n");

#ifdef _DEBUG
  h_ptr = (int8_t *) h_mask;
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      fprintf(stderr, "[%d][%d]", ichan, iant);
      for (isum=0; isum<nsums; isum++)
      {
        if (*h_ptr == 0)
          fprintf (stderr, " ");
        else
          fprintf (stderr, "%d", *h_ptr);
        h_ptr ++;
      }
      fprintf (stderr, "\n");
    }
  }
#endif

  // now mask the input data
  mopsr_delay_copy_scales (stream, h_ant_scales, ant_scales_size);
  mopsr_test_skmask (stream, d_in, d_out, d_mask, d_rstates, d_sigmas, nsums, nchan, nant, nsamp, replace_noise);

  multilog (log, LOG_INFO, "cudaMemcpyDeviceToHost (%d)\n", block_size);
  error = cudaMemcpyAsync (h_out, d_out, nbytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
   multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
   return -1;
  }

  cudaStreamSynchronize(stream);

#ifdef _DEBUG
  // now check what has been zapped!
  in = (float * ) h_in;
  int8_t * out8 = (int8_t  *) h_out;
  h_ptr = (int8_t *) h_mask;
  unsigned err = 0;
  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (isamp=0; isamp<nsamp; isamp += 1024)
      {
        int8_t mask = *h_ptr;
        for (i=0; i<1024; i++)
        {
          if ((fabsf(in[0] - (float) out8[0]) > 1.1) && err < 128 && mask == 0)
          {
            fprintf (stderr, "A [%d][%d][%d] %f != %d mask=%d\n", ichan, iant, (isamp + i), in[0], out8[0], mask);
            err++;
          }

          if ((fabsf(in[1] - (float) out8[1]) > 1.1) && err < 128 && mask == 0)
          {
            fprintf (stderr, "B [%d][%d][%d] %f != %d mask=%d\n", ichan, iant, (isamp + i), in[1], out8[1], mask);
            err++;
          }

          in += 2;
          out8 += 2;
        }
        h_ptr++;
      }
    }
  }
#endif

  cudaFree (d_in);
  cudaFree (d_out);
  cudaFree (d_s1);
  cudaFree (d_s2);
  cudaFree (d_mask);
  cudaFree (d_sigmas);
  cudaFree (d_rstates);
  cudaFree (d_thresh);
  cudaFreeHost (h_in);
  cudaFreeHost (h_out);
  cudaFreeHost (h_s1);
  cudaFreeHost (h_s2);
  cudaFreeHost (h_mask);
  cudaFreeHost (h_thresh);

  return EXIT_SUCCESS;
}

