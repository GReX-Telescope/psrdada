/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_cuda.h"
#include "mopsr_dbcalib.h"
#include "mopsr_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <time.h>

void usage ()
{
	fprintf(stdout, "mopsr_test_accumulate_bandpass\n"
    " -a nant     number of antennae\n" 
    " -t nsamp    number of samples\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nant = 4;
  unsigned ndim = 2;
  unsigned nsamp = 1024*10;
  unsigned batch_size = 1024;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:ht:v")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        nant = atoi(optarg);
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


  if (nsamp%batch_size!=0)
    {
      fprintf (stderr, "nsamp must be a multiple of the batch_size [%d]\n", batch_size);      
      return -1;
    }

  // cuda setup                                                                                                          
  // select the gpu device                                                                                               
  int n_devices = dada_cuda_get_device_count();
  fprintf (stderr, "Detected %d CUDA devices\n", n_devices);

  if ((device < 0) && (device >= n_devices))
    {
      fprintf (stderr, "no CUDA devices available [%d]\n",
	       n_devices);
      return -1;
    }

  fprintf (stderr, "main: dada_cuda_select_device(%d)\n", device);
  if (dada_cuda_select_device (device) < 0)
    {
      fprintf (stderr, "could not select requested device [%d]\n", device);
      return -1;
    }

  char * device_name = dada_cuda_get_device_name (device);
  if (!device_name)
    {
      fprintf (stderr, "could not get CUDA device name\n");
      return -1;
    }
  fprintf (stderr, "Using device %d : %s\n", device, device_name);
  free(device_name);

  cudaStream_t stream;
  fprintf (stderr, "main: cudaStreamCreate()\n");
  cudaError_t error = cudaStreamCreate(&stream);
  fprintf (stderr, "main: stream=%p\n", stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not create CUDA stream\n");
      return -1;
    }
  
  //Determine antenna pairs
  // static delay code starts here                                                                                       
  int npairs = (nant*nant - nant)/2; //number of pairs                                                  
  mopsr_baseline_t* d_pairs; //device pair array                                                                         
  mopsr_baseline_t* h_pairs;
  
  fprintf (stderr, "Found %d baseline pairs\n",npairs);
  
  // allocate memory for generating baseline combinations                                                           
  h_pairs = (mopsr_baseline_t*) malloc(sizeof(mopsr_baseline_t) * npairs);
  if (h_pairs == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }

  error = cudaMalloc ((void**)&d_pairs, sizeof(mopsr_baseline_t) * npairs);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }

  fprintf (stderr, "Determining baseline combinations...");
  
  int pair_idx,idx,ii;
  pair_idx = idx = 0;
  while (idx < nant)
    {
      for (ii=idx+1; ii < nant; ii++)
	{
	  h_pairs[pair_idx].a = idx;
	  h_pairs[pair_idx].b = ii;
	  pair_idx++;
	}
      idx++;
    }

  fprintf (stderr, " Complete\n");
  fprintf (stderr, "Allocating all host and device arrays...");
  error = cudaMemcpyAsync (d_pairs, h_pairs, sizeof(mopsr_baseline_t) * npairs, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "Async host to device failed\n");
      return -1;
    }
  
  // Allocate remaining all memory buffers
  
  cuComplex* h_input = (cuComplex*) malloc(sizeof(cuComplex)*nant*nsamp); //input data
  if (h_input == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }
  
  cufftComplex * d_input;
  error = cudaMalloc ((void**)&d_input, sizeof(cufftComplex) * nant*nsamp);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }

  cufftComplex * d_summed;
  error = cudaMalloc ((void**)&d_summed, sizeof(cufftComplex) * batch_size * nant);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }
  error = cudaMemset (d_summed, 0, sizeof(cufftComplex) * batch_size * nant);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not memset device memory\n");
      return -1;
    }


  cuComplex * h_summed;
  h_summed = malloc(sizeof(cufftComplex) * batch_size * nant);
  if (h_summed == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }
    
  cufftComplex * d_cross_power;
  error = cudaMalloc ((void**)&d_cross_power, sizeof(cufftComplex) * npairs * batch_size);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }

  error = cudaMemset (d_cross_power, 0, sizeof(cufftComplex) * batch_size * npairs);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not memset device memory\n");
      return -1;
    }
  

  cuComplex * h_cross_power;
  h_cross_power = malloc(sizeof(cufftComplex) * batch_size * npairs);
  if (h_cross_power == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }
  
  float * h_delays;
  h_delays = malloc(sizeof(float) * npairs);
  if (h_delays == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }

  float * h_delay_errors;
  h_delay_errors = malloc(sizeof(float) * npairs);
  if (h_delays == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }

  float * d_delays;
  error = cudaMalloc ((void**)&d_delays, sizeof(float) * npairs);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }

  float * d_delay_errors;
  error = cudaMalloc ((void**)&d_delay_errors, sizeof(float) * npairs);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "could not allocate device memory\n");
      return -1;
    }
  fprintf (stderr, " Complete.\n");

  fprintf (stderr, "Populating input data array...");
  // array of zeros with a diagonal of 1 
  // data is assumed to be already transposed into ST order

  srand(time(NULL));
  int * randvals = malloc(sizeof(int) * nant);
  if (randvals == NULL)
    {
      fprintf (stderr, "could not allocate host memory\n");
      return -1;
    }

  int jj;
  srand(time(0));
  for (ii=0;ii<nant;ii++)
    {
      for (jj=0;jj<nsamp;jj++)
	{
	  h_input[ii*nsamp + jj] = make_cuComplex((float) rand()/RAND_MAX,(float)rand()/RAND_MAX);
	}
      randvals[ii] = rand() % batch_size/2;
      h_input[ii*nsamp + randvals[ii] ] = make_cuComplex(1,0);
    }
  
  fprintf (stderr, " Complete.\n");

  fprintf (stderr, "Copying input to host...");
  error = cudaMemcpyAsync (d_input, h_input, sizeof(cuComplex) * nant * nsamp, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "Async host to device failed\n");
      return -1;
    }
  fprintf (stderr, " Complete.\n");

  fprintf (stderr, "Planning and executing 1D FFT...");
  cufftHandle fft_plan;
  cufftResult cufft_error;
  cufft_error = cufftPlan1d(&fft_plan, batch_size, CUFFT_C2C, nsamp*nant/batch_size);
  cufft_error = cufftExecC2C(fft_plan,d_input, d_input, CUFFT_FORWARD);
  //Should error check
  fprintf (stderr, " Complete.\n");

  unsigned nbatch = nsamp/batch_size;
  mopsr_accumulate_cp_spectra (d_input, d_cross_power, d_pairs, batch_size,
			       nbatch, npairs, nsamp, stream);
  
  
  error = cudaMemcpyAsync (h_cross_power, d_cross_power, sizeof(cuComplex) * batch_size * npairs, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "device to host failed\n");
      return -1;
    }

  error = cudaMemcpyAsync (h_input, d_input, sizeof(cuComplex) * nant * nsamp, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "device to host failed\n");
      return -1;
    }
  cudaStreamSynchronize(stream);

  fprintf (stderr, "Dumping to file...");
  FILE * file;
  file = fopen("dump.input","w");
  fwrite(h_input,sizeof(cuComplex), nant * nsamp, file);
  fclose(file);
  file = fopen("dump.cp","w");
  fwrite(h_cross_power,sizeof(cuComplex), npairs*batch_size, file);
  fclose(file);
  

  cufftHandle inverse_fft_plan;
  cufft_error = cufftPlan1d(&inverse_fft_plan, batch_size, CUFFT_C2C, npairs);
  cufft_error = cufftExecC2C(inverse_fft_plan, d_cross_power, d_cross_power, CUFFT_INVERSE);
  //Should error check
  
  fprintf (stderr, "Determining static delay...");
  mopsr_static_delay (d_cross_power, d_delays, d_delay_errors, npairs, batch_size, stream);
  fprintf (stderr, " Complete.\n");

  error = cudaMemcpyAsync (h_delays, d_delays, sizeof(float) * npairs, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "Async device to host failed\n");
      return -1;
    }

  error = cudaMemcpyAsync (h_delay_errors, d_delay_errors, sizeof(float) * npairs, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    {
      fprintf (stderr, "Async device to host failed\n");
      return -1;
    }

  cudaStreamSynchronize(stream);
  
  for (ii=0;ii<nant;ii++){
    fprintf (stderr, "Antenna %d: %d offset (%d)\n",ii,randvals[ii],randvals[ii]%batch_size);
  }
  
  for (ii=0;ii<npairs;ii++){
    fprintf (stderr, "Baseline %d -> %d: %f %f offset ",h_pairs[ii].a,h_pairs[ii].b,h_delays[ii], h_delay_errors[ii]);
    if (randvals[h_pairs[ii].a]-randvals[h_pairs[ii].b] == round(h_delays[ii]))
      fprintf (stderr, "[CORRECT]\n");
    else 
      fprintf (stderr, "[WRONG]\n");
  }

  
  free(h_input);
  cudaFree(d_input);
  free(h_pairs);
  cudaFree(d_pairs);
  cudaFree(d_cross_power);
  free(h_delays);
  free(h_delay_errors);
  cudaFree(d_delays);
  cudaFree(d_delay_errors);
  cudaFree(d_summed);
  free(h_summed);
  free(randvals);


  return 0;
}
