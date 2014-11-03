/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_cuda.h"
#include "mopsr_delays.h"
#include "mopsr_cuda.h"
#include "mopsr_delays_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#define SKZAP

void usage ()
{
	fprintf(stdout, "mopsr_test_delays bays_file modules_file\n"
    " -a nant     number of antennae\n" 
    " -c nchan    number of channels\n" 
    " -t nsamp    number of samples\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nchan = 40;
  unsigned nant = 4;
  unsigned ndim = 2;
  unsigned ntap = 25;
  unsigned nsamp = 1024;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:c:d:ht:v")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'c':
        nchan = atoi(optarg);
        break;

      case 'd':
        device = atoi (optarg);
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

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 argument are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  unsigned isamp, ichan, iant, imod;;

  unsigned nbay;
  char * bays_file = strdup (argv[optind]);
  mopsr_bay_t * all_bays = read_bays_file (bays_file, &nbay);
  if (!all_bays)
  {
    fprintf (stderr, "ERROR: failed to read bays file [%s]\n", bays_file);
    return EXIT_FAILURE;
  }

  int nmod;
  char * modules_file = strdup (argv[optind+1]);
  mopsr_module_t * modules = read_modules_file (modules_file, &nmod);
  if (!modules)
  {
    fprintf (stderr, "ERROR: failed to read modules file [%s]\n", modules_file);
    return EXIT_FAILURE;
  }

  // preconfigure a source
  mopsr_source_t source;
  sprintf (source.name, "J0437-4715");
  source.raj  = 1.17635;
  source.decj = 0.65013;

  source.raj  = 4.82147205863433;

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

  // determine the timestamp corresponding to the current byte of data
  timestamp.tv_sec = (long int) time(NULL);
  timestamp.tv_usec = 0;

  timestamp.tv_sec += 60;

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

  // setup the cuda stream for operations
  cudaStream_t stream;
  fprintf (stderr, "main: cudaStreamCreate()\n");
  cudaError_t error = cudaStreamCreate(&stream);
  fprintf (stderr, "main: stream=%p\n", stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "could not create CUDA stream\n");
    return -1;
  }

  unsigned nbytes_per_samp = nchan * nant * ndim;
  const uint64_t nbytes = nsamp * nbytes_per_samp;

  fprintf (stderr, "nchan=%u nant=%u nsamp=%"PRIu64" nbytes=%"PRIu64"\n", nchan, nant, nsamp, nbytes);

  void * d_sample_delayed;
  void * d_fractional_delayed;

  fprintf (stderr, "main: cudaMalloc(%"PRIu64") for d_fractional_delayed\n", nbytes);
  error = cudaMalloc(&d_fractional_delayed, nbytes);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMalloc failed for %ld bytes\n", nbytes);
    return -1;
  }

  void * h_in;
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_in\n", nbytes * 2);
  error = cudaMallocHost (&h_in, nbytes * 2);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", nbytes * 2);
    return -1;
  }

  void * d_in;
  fprintf (stderr, "main: cudaMalloc(%"PRIu64") for d_in\n", nbytes);
  error = cudaMalloc (&d_in, nbytes);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMalloc failed for %ld bytes\n", nbytes);
    return -1;
  }

  void * d_fbuf;
  size_t d_fbuf_size = nbytes * sizeof(float);
  fprintf (stderr, "main: allocating %ld bytes of device memory for d_fbuf\n", d_fbuf_size);
  error = cudaMalloc (&d_fbuf, d_fbuf_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: could not allocate %ld bytes of device memory\n", d_fbuf_size);
    return -1;
  }

  size_t maxthreads = 1024;
  const unsigned nrngs = nchan * nant * maxthreads;
  size_t d_curand_size = (size_t) (nrngs * mopsr_curandState_size());
  void * d_rstates = 0;
  if (verbose)
   fprintf (stderr, "alloc: allocating %ld bytes of device memory for d_rstates\n", d_curand_size);
  error = cudaMalloc (&d_rstates, d_curand_size);
  if (verbose)
    fprintf (stderr, "alloc: d_rstates=%p\n", d_rstates);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "alloc: could not allocate %ld bytes of device memory\n", d_curand_size);
    return -1;
  }

  unsigned long long seed = (unsigned long long) time(0);
  mopsr_init_rng (stream, seed, nrngs, d_rstates);

  size_t d_sigmas_size = (size_t) (nchan * nant * sizeof(float));
  void * d_sigmas = 0;
  if (verbose)
    fprintf (stderr, "alloc: allocating %ld bytes of device memory for d_sigmas\n", d_sigmas_size);
  error = cudaMalloc (&d_sigmas, d_sigmas_size);
  if (verbose)
    fprintf (stderr, "alloc: d_rstates=%p\n", d_rstates);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "alloc: could not allocate %ld bytes of device memory\n", d_sigmas_size);
    return -1;
  }
  cudaMemsetAsync(d_sigmas, 0, d_sigmas_size, stream);

  size_t ant_scales_size = nant * sizeof(float);
  float * h_ant_scales;
  error = cudaMallocHost ((void **) &h_ant_scales, ant_scales_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: could not allocate %ld bytes of host memory\n", ant_scales_size);
    return -1;
  }

  float * h_fractional_delays;
  size_t delays_size = sizeof(float) * nchan * nant;
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_fractional_delays\n", delays_size);
  error = cudaMallocHost ((void **) &h_fractional_delays, delays_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", delays_size);
    return -1;
  }

  float * h_fringe_coeffs;
  size_t fringes_size = sizeof(float) * nchan * nant;
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_fringe_coeffs\n", fringes_size);
  error = cudaMallocHost ((void **) &h_fringe_coeffs, fringes_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", fringes_size);
    return -1;
  }

  float * h_fringe_coeffs_ds;
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_fringe_coeffs_ds\n", fringes_size);
  error = cudaMallocHost ((void **) &h_fringe_coeffs_ds, fringes_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", fringes_size);
    return -1;
  }

  float * h_delays_ds;
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_delays_ds\n", fringes_size);
  error = cudaMallocHost ((void **) &h_delays_ds, fringes_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", fringes_size);
    return -1;
  }

  void * d_fractional_delays;
  fprintf (stderr, "main: cudaMalloc(%"PRIu64") for d_fractional_delays\n", delays_size);
  error = cudaMalloc(&d_fractional_delays, delays_size);
  if (error != cudaSuccess)
  {         
    fprintf (stderr, "main: cudaMallocfailed for %ld bytes\n", delays_size);
    return -1;        
  }                     

  void * h_out;
  size_t nbytes_out = nbytes + (ndim * nchan * nant * (ntap-1));
  fprintf (stderr, "main: cudaMallocHost(%"PRIu64") for h_out\n", nbytes * 3);
  error = cudaMallocHost (&h_out, nbytes * 3);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "main: cudaMallocHost failed for %ld bytes\n", nbytes * 3);
    return -1;
  }

  // struct for all the goodies
  transpose_delay_t ctx;

  fprintf (stderr, "main: mopsr_transpose_delay_alloc(%ld, %ld, %ld, %ld)\n", nbytes, nchan, nant, ntap);
  if (mopsr_transpose_delay_alloc (&ctx, nbytes, nchan, nant, ntap) < 0)
  {
    fprintf (stderr, "mopsr_transpose_delay_alloc failed\n");
    return EXIT_FAILURE;
  }

  // copy the scales to the GPU 
  for (iant=0; iant<nant; iant++)
    h_ant_scales[iant] = 1;
  mopsr_delay_copy_scales (stream, h_ant_scales, ant_scales_size);

  const unsigned fixed_delay = (ntap / 2) + 1;

  // initialize data in h_in to constants for ant number
  int8_t * ptr = (int8_t *) h_in;
  //fprintf (stderr, "[T][F][S]\n");
  
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        ptr[0] = (int8_t) ((((128 - fixed_delay) - iant) + (isamp)) % 128);
        ptr[1] = ptr[0];
        ptr += 2;
      }
    }
  }

  const unsigned slip = 3;

  // then for the second half have a jump of 1 number
  for (isamp=nsamp; isamp<2*nsamp; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        if (isamp < nsamp + (fixed_delay+iant))
          ptr[0] = (int8_t) ((((128 - fixed_delay) - iant) + isamp) % 128);
        else
          ptr[0] = (int8_t) ((((128 - fixed_delay) - iant) + (isamp - slip)) % 128);
        ptr[1] = ptr[0];
        ptr += 2;
      }
    }
  }

  char apply_instrumental = 1;
  char apply_geometric = 1;
  char is_tracking = 1;
  double tsamp = 1.28;

  // update the delays
  fprintf (stderr, "main: calculate_delays()\n");
  if (calculate_delays (nbay, all_bays, nmod, modules, nchan, channels, source, 
                        timestamp, delays, apply_instrumental, 
                        apply_geometric, is_tracking, tsamp) < 0)
  {
    fprintf (stderr, "main: calculate_delays() failed\n");
    return -1;
  }

  // manually set the delays
  for (iant=0; iant<nant; iant++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      // override the sample delays for testing...
      delays[iant][ichan].samples = fixed_delay + (iant);
      // set the host fractional delays 
      //h_fractional_delays[ichan*nant + iant] = delays[iant][ichan].fractional;
      h_fractional_delays[ichan*nant + iant] = 0.0;

      //h_fringe_coeffs[ichan*nant + iant] = delays[iant][ichan].fringe_coeff;
      h_fringe_coeffs[ichan*nant + iant] = 0;

      //h_delays_ds[ichan*nant + iant] = delays[iant][ichan].fractional_ds;
      h_delays_ds[ichan*nant + iant] = 0;

      //h_fringe_coeffs_ds[ichan*nant + iant] = delays[iant][ichan].fringe_coeff_ds;
      h_fringe_coeffs_ds[ichan*nant + iant] = 0;
    }
  }

  unsigned i;
/*
  // copy input from host to device
  fprintf (stderr, "main: cudaMemcpyAsync to=%p from=%p size=%ld\n", d_in, h_in, nbytes);
  error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
  d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
  fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

  // copy the delays to the GPU
  fprintf (stderr, "main: copy fractional delays to GPU %d bytes\n", delays_size);
  error = cudaMemcpyAsync (d_fractional_delays, h_fractional_delays, delays_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

    
  // now do 2 iterations
  unsigned i;
  for (i=0; i<3; i++)
  {
    size_t in_offset = nbytes;
    size_t ou_offset = i * nbytes;

    fprintf (stderr, "main: cudaMemcpyAsync to=%p from=%p size=%ld\n", d_in, h_in + in_offset, nbytes);
    error = cudaMemcpyAsync (d_in, h_in + in_offset, nbytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    // manually set the delays for secondary jump
    for (iant=0; iant<nant; iant++)
    {
      for (ichan=0; ichan<nchan; ichan++)
      {
        delays[iant][ichan].samples = fixed_delay + (iant) + slip;
      }
    }

    ptr = (int8_t *) (h_in + in_offset);
    for (isamp=nsamp; isamp<2*nsamp; isamp++)
    {
      for (ichan=0; ichan<nchan; ichan++)
      {
        for (iant=0; iant<nant; iant++)
        {
          ptr[0] = (int8_t) ((((128 - fixed_delay) - iant) + (isamp - slip)) % 128);
          ptr[1] = ptr[0];
          ptr += 2;
        }
      }
    }

    fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
    d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
    fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

    fprintf (stderr, "main: mopsr_delay_fractional(%ld)\n", nbytes_out);
#ifdef SKZAP
    mopsr_delay_fractional_sk_scale (stream, d_sample_delayed, d_fractional_delayed, d_fbuf,
                                     d_rstates, d_sigmas, 
                                     d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                                     h_fringe_coeffs_ds, fringes_size,
                                     nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#else
    mopsr_delay_fractional (stream, d_sample_delayed, d_fractional_delayed,
                            d_fractional_delays, h_fringe_coeffs, h_delays_ds, 
                            h_fringe_coeffs_ds, fringes_size,
                            nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#endif
    fprintf (stderr, "main: mopsr_delay_fractional returned\n");

    // copy output from device to host
    fprintf (stderr, "main: copy delayed data back %d bytes\n", nbytes);
    error = cudaMemcpyAsync (h_out + ou_offset, d_fractional_delayed, nbytes, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    cudaStreamSynchronize (stream);
  }

  fprintf (stderr, "[B][F][S][T] (re, im) - Testing Sample Integrity\n");
  ptr = (int8_t *) h_out;
  int val;
  uint64_t nerr = 0;

  uint64_t bad_count = 0;
  uint64_t good_count = 0;
  for (i=0; i<3; i++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          val = isamp % 128;
          if (ptr[0] != (int8_t) val)
          {
            if (nerr < 100)
              fprintf (stderr, "[%d][%d][%d][%d] (%d,%d) != %d\n", i, ichan, iant, isamp, ptr[0], ptr[1], val);
            nerr++;
            bad_count++;
          }
          else
            good_count++;
          ptr += 2;
        }
      }
    }
  }

  fprintf (stderr, "bad=%"PRIu64" good=%"PRIu64"\n", bad_count, good_count);

  //////////////////////////////////////////////////////////////////////////////
  //
  // Test Antenna integrity
  //
  ptr = (int8_t *) h_in;
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        ptr[0] = (int8_t) iant;
        ptr[1] = ptr[0];
        ptr += 2;
      }
    }
  }

  // copy input from host to device
  error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
  d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
  fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

  for (i=0; i<1; i++)
  {
    fprintf (stderr, "main: copy datablock delays to GPU %d bytes\n", nbytes);
    error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
    d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
    fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

    // copy the delays to the GPU
    fprintf (stderr, "main: copy fractional delays to GPU %d bytes\n", delays_size);
    error = cudaMemcpyAsync (d_fractional_delays, h_fractional_delays, delays_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }


    fprintf (stderr, "main: mopsr_delay_fractional(%ld)\n", nbytes_out);
#ifdef SKZAP
    mopsr_delay_fractional_sk_scale (stream, d_sample_delayed, d_fractional_delayed, d_fbuf,
                                     d_rstates, d_sigmas, 
                                     d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                                     h_fringe_coeffs_ds, fringes_size,
                                     nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#else
    mopsr_delay_fractional (stream, d_sample_delayed, d_fractional_delayed,
                            d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                            h_fringe_coeffs_ds, fringes_size,
                            nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#endif
    fprintf (stderr, "main: mopsr_delay_fractional returned\n");

    // copy output from device to host
    fprintf (stderr, "main: copy delayed data back %d bytes\n", nbytes);
    error = cudaMemcpyAsync (h_out, d_fractional_delayed, nbytes, cudaMemcpyDeviceToHost, stream);
    //error = cudaMemcpyAsync (h_out, d_sample_delayed, nbytes_out, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    cudaStreamSynchronize (stream);

#if 1
    fprintf (stderr, "[F][S][T] (re, im) - Testing Antenna Integrity\n");
    ptr = (int8_t *) h_out;
    int val;
    uint64_t nerr = 0;
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          val = iant;
          if (ptr[0] != (int8_t) val)
          {
            if (nerr < 10)
              fprintf (stderr, "[%d][%d][%d] (%d,%d) != %d\n", ichan, iant, isamp, ptr[0], ptr[1], val);
            nerr++;
          }
          ptr += 2;
        }
      }
    }
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  //
  // Test Channel integrity
  //
  ptr = (int8_t *) h_in;
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        ptr[0] = (int8_t) ichan;
        ptr[1] = ptr[0];
        ptr += 2;
      }
    }
  }

  // copy input from host to device
  error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
  d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
  fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

  for (i=0; i<1; i++)
  {
    fprintf (stderr, "main: copy datablock delays to GPU %d bytes\n", nbytes);
    error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
    d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
    fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

    // copy the delays to the GPU
    fprintf (stderr, "main: copy fractional delays to GPU %d bytes\n", delays_size);
    error = cudaMemcpyAsync (d_fractional_delays, h_fractional_delays, delays_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    fprintf (stderr, "main: mopsr_delay_fractional(%ld)\n", nbytes_out);
#ifdef SKZAP
    mopsr_delay_fractional_sk_scale (stream, d_sample_delayed, d_fractional_delayed, d_fbuf,
                                     d_rstates, d_sigmas, 
                                     d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                                     h_fringe_coeffs_ds, fringes_size,
                                     nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#else
    mopsr_delay_fractional (stream, d_sample_delayed, d_fractional_delayed,
                            d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                            h_fringe_coeffs_ds, fringes_size,
                            nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#endif
    fprintf (stderr, "main: mopsr_delay_fractional returned\n");

    // copy output from device to host
    fprintf (stderr, "main: copy delayed data back %d bytes\n", nbytes);
    error = cudaMemcpyAsync (h_out, d_fractional_delayed, nbytes, cudaMemcpyDeviceToHost, stream);
    //error = cudaMemcpyAsync (h_out, d_sample_delayed, nbytes_out, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    cudaStreamSynchronize (stream);

#if 1
    fprintf (stderr, "[F][S][T] (re, im) - Testing Channel Integrity\n");
    ptr = (int8_t *) h_out;
    int val;
    uint64_t nerr = 0;
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          val = ichan;
          if (ptr[0] != (int8_t) val)
          {
            if (nerr < 10)
              fprintf (stderr, "[%d][%d][%d] (%d,%d) != %d\n", ichan, iant, isamp, ptr[0], ptr[1], val);
            nerr++;
          }
          ptr += 2;
        }
      }
    }
#endif
  }
*/

  ////////////////////////////////////////////////////////////////
  // Test random number generator
  //
  uint8_t * u8_in = (uint8_t *) h_in;
  srand(4);

  fprintf (stderr, "[T][F][S] - Input for RNG integridy\n");
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        u8_in[0] = (uint8_t) (rand() % 255);
        u8_in[1] = (uint8_t) (rand() % 255);

        //if (isamp < 20 && ichan == 0 && iant ==0)
        //  fprintf (stderr, "[%d][%d][%d] = %"PRIu8", %"PRIu8"\n", isamp, ichan, iant, u8_in[0], u8_in[1]);

        u8_in += 2;
      }
    }
  }

  fprintf (stderr, "fixed_delay=%u\n", fixed_delay);

  // manually set the delays for secondary jump
  for (iant=0; iant<nant; iant++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      delays[iant][ichan].samples = fixed_delay;
    }
  }

  // copy input from host to device
  error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // reset things
  ctx.first_kernel = 1;

  fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
  d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
  fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

  for (i=0; i<1; i++)
  {
    fprintf (stderr, "main: copy datablock delays to GPU %d bytes\n", nbytes);
    error = cudaMemcpyAsync (d_in, h_in, nbytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    fprintf (stderr, "main: mopsr_transpose_delay(%ld)\n", nbytes);
    d_sample_delayed = mopsr_transpose_delay (stream, &ctx, d_in, nbytes, delays);
    fprintf (stderr, "main: mopsr_transpose_delay returned %p\n", d_sample_delayed);

    // copy the delays to the GPU
    fprintf (stderr, "main: copy fractional delays to GPU %d bytes\n", delays_size);
    error = cudaMemcpyAsync (d_fractional_delays, h_fractional_delays, delays_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }


    fprintf (stderr, "main: mopsr_delay_fractional(%ld)\n", nbytes_out);
#ifdef SKZAP
    mopsr_delay_fractional_sk_scale (stream, d_sample_delayed, d_fractional_delayed, d_fbuf,
                                     d_rstates, d_sigmas,
                                     d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                                     h_fringe_coeffs_ds, fringes_size,
                                     nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#else
    mopsr_delay_fractional (stream, d_sample_delayed, d_fractional_delayed,
                            d_fractional_delays, h_fringe_coeffs, h_delays_ds,
                            h_fringe_coeffs_ds, fringes_size,
                            nbytes_out, ctx.nchan, ctx.nant, ctx.ntap);
#endif
    fprintf (stderr, "main: mopsr_delay_fractional returned\n");

    // copy output from device to host
    fprintf (stderr, "main: copy delayed data back %d bytes\n", nbytes);
    error = cudaMemcpyAsync (h_out, d_fractional_delayed, nbytes, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    cudaStreamSynchronize (stream);

#if 1
    fprintf (stderr, "[F][S][T] out, in - Testing Random Integrity\n");
    uint8_t * u8_out = (uint8_t *) h_out;
    u8_in = (uint8_t *) h_in;
    int val;
    uint8_t nerr = 0;
    unsigned hout_idx = 0;
    unsigned hin_idx = 0;
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          uint8_t o1 = u8_out[hout_idx];
          uint8_t o2 = u8_out[hout_idx+1];

          if (isamp + fixed_delay < nsamp)
          {
            hin_idx = (((isamp + fixed_delay) * nchan * nant) + (ichan * nant) + iant) * 2;
            uint8_t i1 = u8_in[hin_idx];
            uint8_t i2 = u8_in[hin_idx+1];

            if (o1 != i1 || o2 != i2)
            {
              if (nerr < 30)
              {
                fprintf (stderr, "[%d][%d][%d] out(%"PRIu8",%"PRIi8") != in(%"PRIu8",%"PRIi8") \n", ichan, iant, isamp, o1, o2, i1, i2);
                nerr++;
              }
            }
          }
          hout_idx += 2;
        }
      }
    }
#endif
  }

  if (mopsr_transpose_delay_dealloc (&ctx) < 0)
  {
    fprintf (stderr, "main: mopsr_transpose_delay_dealloc failed\n");
    return -1;
  } 

  if (h_in)
    cudaFreeHost(h_in);
  h_in = 0;

  if (d_in)
    cudaFree(d_in);
  d_in = 0;

  if (d_fbuf)
    cudaFree (d_fbuf);
  d_fbuf = 0;

  if (d_rstates)
    cudaFree (d_rstates);
  d_rstates = 0;

  if (d_sigmas)
    cudaFree (d_sigmas);
  d_sigmas = 0;

  if (h_ant_scales)
    cudaFreeHost(h_ant_scales);
  h_ant_scales = 0;

  if (h_fringe_coeffs)
    cudaFreeHost(h_fringe_coeffs);
  h_fringe_coeffs = 0;

  if (h_delays_ds)
    cudaFreeHost(h_delays_ds);
  h_delays_ds = 0;

  if (h_fringe_coeffs_ds)
    cudaFreeHost(h_fringe_coeffs_ds);
  h_fringe_coeffs_ds = 0;

  if (h_fractional_delays)
    cudaFreeHost(h_fractional_delays);
  h_fractional_delays = 0;

  if (d_fractional_delayed)
    cudaFree (d_fractional_delayed);
  d_fractional_delayed = 0;

  free (channels);
  free (modules);

  for (imod=0; imod<nmod; imod++)
    free (delays[imod]);
  free (delays);


  return 0;
}
