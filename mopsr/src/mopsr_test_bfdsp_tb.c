
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
#include "dada_generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

int mopsr_tie_beams_cpu (void * h_in, void * h_out, void * h_tb_phasors, uint64_t nbytes, unsigned nant, unsigned nchan);

#define SKZAP

void usage ()
{
	fprintf(stdout, "mopsr_test_bfdsp_tb bays_file modules_file\n"
    " -a nant     number of antennae [default 352]\n" 
    " -c nchan    number of channels [default 10]\n" 
    " -t nsamp    number of samples [default 16384]\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nant = 352;
  uint64_t nsamp = 16384;
  unsigned nchan = 10;

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
        nsamp = (uint64_t) atoi (optarg);
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

  unsigned isamp, iant, imod;

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

  mopsr_chan_t * channels = (mopsr_chan_t *) malloc (sizeof(mopsr_chan_t) * nchan);
  unsigned ichan;
  unsigned start_channel = 204;
  double chan_bw = 31.25 / 320.0;
  for (ichan=0; ichan<nchan; ichan++)
  {
    channels[ichan].number = ichan + start_channel;
    channels[ichan].bw     = chan_bw;
    channels[ichan].cfreq  = (800 + (0.78125/2) + (ichan * chan_bw));
  }

  struct timeval timestamp;

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

  /*
  fprintf (stderr, "main: dada_get_device_name(%d)\n", device);
  char * device_name = dada_cuda_get_device_name (device);
  if (!device_name)
  {
    fprintf (stderr, "could not get CUDA device name\n");
    return -1;
  }
  //fprintf (stderr, "Using device %d : %s\n", device, device_name);
  //free(device_name);
  */

  // setup the cuda stream for operations
  cudaStream_t stream;
  cudaError_t error;

  fprintf (stderr, "main: cudaStreamCreate()\n");
  error = cudaStreamCreate(&stream);
  fprintf (stderr, "main: stream=%p\n", stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "could not create CUDA stream\n");
    return -1;
  }

  unsigned ndim = 2;

  unsigned nbyte_in = sizeof(int8_t);
  unsigned nbyte_ou = sizeof(float);

  uint64_t in_block_size = nsamp * nant  * ndim * nbyte_in * nchan;
  uint64_t ou_block_size = nsamp * 1     * ndim * nbyte_ou * nchan;

  fprintf (stderr, "IN:  nant=%u  nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", nant,  nchan, ndim, nbyte_in*8, nsamp, in_block_size);
  fprintf (stderr, "OUT: nbeam=%u nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", 1,     nchan, ndim, nbyte_ou*8, nsamp, ou_block_size);

  void * d_in;
  void * d_tb;
  void * d_tb_phasors;
  void * h_in;
  void * h_out;
  void * h_out_cpu;
  float * h_tb_phasors;

  error = cudaMallocHost( &h_in, in_block_size); 
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", in_block_size);
    return -1;
  }
  int8_t * ptr = (int8_t *) h_in;

  srand ( time(NULL) );
  double stddev = 32;

  int8_t v1 = (int8_t) rand_normal (0, stddev);
  int8_t v2 = (int8_t) rand_normal (0, stddev);

  for (ichan=0; ichan<nchan; ichan++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (isamp=0; isamp<nsamp; isamp++)
      {
        ptr[0] = (int8_t) rand_normal (0, stddev);
        ptr[1] = (int8_t) rand_normal (0, stddev);
        ptr += 2;
      }
    }
  }

  error = cudaMallocHost( &h_out, ou_block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", ou_block_size);
    return -1;
  }

  h_out_cpu = malloc(ou_block_size);

  double C = 2.99792458e8;
  double cfreq = 843e6;

  size_t phasors_size = nchan * nant * sizeof(float) * 2;
  error = cudaMallocHost( (void **) &(h_tb_phasors), phasors_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  complex float * phasors_ptr = (complex float *) h_tb_phasors;
  unsigned re, im;
  double md_angle = 10.0;

  // Phasors are stored in FS ordering
  for (iant=0; iant<nant; iant++)
  {
    double geometric_delay = (sin(md_angle) * modules[iant].dist) / C;
    for (ichan=0; ichan<nchan; ichan++)
    {
      double theta = -2 * M_PI * channels[ichan].cfreq * 1000000 * geometric_delay;
      phasors_ptr[ichan*nant+iant] = (float) cos(theta) + (float) sin(theta) * I;
    }
  }

  if (verbose)
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_in\n", in_block_size);
  error = cudaMalloc( &(d_in), in_block_size);
  if (verbose)
    fprintf(stderr, "alloc: d_in=%p\n", d_in);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", in_block_size);
    return -1;
  }

  if (verbose)
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_tb\n", ou_block_size);
  error = cudaMalloc( &(d_tb), ou_block_size);
  if (verbose)
    fprintf(stderr, "alloc: d_tb=%p\n", d_tb);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", ou_block_size);
    return -1;
  }

  if (verbose)
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_tb_phasors\n", phasors_size);
  error = cudaMalloc( &(d_tb_phasors), phasors_size);
  if (verbose)
    fprintf(stderr, "alloc: d_tb=%p\n", d_tb_phasors);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  cudaStreamSynchronize(stream);

  if (verbose > 1)
    fprintf(stderr, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", phasors_size, d_tb_phasors, h_tb_phasors);
  error = cudaMemcpyAsync (d_tb_phasors, h_tb_phasors, phasors_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              d_tb_phasors, h_tb_phasors);
    return -1;
  }

  char check_cpu_gpu = 1;
  unsigned itrial;
  for (itrial=0; itrial<1; itrial++)
  {
    // copy the whole block to the GPU
    if (verbose > 1)
      fprintf(stderr, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", in_block_size, d_in, h_in);
    error = cudaMemcpyAsync (d_in, h_in, in_block_size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                d_in, h_in);
      return -1;
    }

    if (verbose > 1)
      fprintf(stderr, "io_block: mopsr_tie_beam()\n");
    mopsr_tie_beam (stream, d_in, d_tb, d_tb_phasors, in_block_size, nant, nchan);
    if (check_cpu_gpu)
    {
      if (verbose > 1)
        fprintf(stderr, "io_block: mopsr_tie_beams_cpu()\n");
      mopsr_tie_beams_cpu (h_in, h_out_cpu, h_tb_phasors, in_block_size, nant, nchan);
    }

    if (verbose > 1)
      fprintf(stderr, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                h_out, d_tb, ou_block_size);
    error = cudaMemcpyAsync ( h_out, d_tb, ou_block_size,
                              cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

    cudaStreamSynchronize(stream);
  }

  if (check_cpu_gpu)
  {
    complex float * cpu_ptr = (complex float *) h_out_cpu;
    complex float * gpu_ptr = (complex float *) h_out;
    ptr = (int8_t *) h_in;
    int8_t re8, im8;

    uint64_t ival = 0;
    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<nchan; ichan++)
      {
        float percent_diff = (cpu_ptr[ival] - gpu_ptr[ival]) / cpu_ptr[ival];
        if (percent_diff > 1e-4)
          fprintf (stderr, "[%d][%d] mismatch percent_diff=%f (cpu==%f != gpu==%f)\n", ichan, isamp, percent_diff * 100, cpu_ptr[ival], gpu_ptr[ival]);
        ival++;
      }
    } 
  }

  if (d_in)
    cudaFree(d_in);
  d_in = 0;

  if (d_tb)
    cudaFree (d_tb);
  d_tb = 0;

  if (d_tb_phasors)
    cudaFree (d_tb_phasors);
  d_tb_phasors = 0;

  if (h_in)
    cudaFreeHost(h_in);
  h_in = 0;

  if (h_out)
    cudaFreeHost(h_out);
  h_out = 0;

  if (h_tb_phasors)
    cudaFreeHost(h_tb_phasors);
  h_tb_phasors = 0;

  free (modules);

  return 0;
}

// simpler CPU version
int mopsr_tie_beams_cpu (void * h_in, void * h_out, void * h_tb_phasors, uint64_t nbytes, unsigned nant, unsigned nchan)
{
  unsigned ndim = 2;

  // data is ordered in ST order
  uint64_t nsamp = nbytes / (nchan * nant * ndim);
  uint64_t nsampant = nbytes / (nchan * ndim);

  complex float * phasors = (complex float *) h_tb_phasors;

  int16_t * in16 = (int16_t *) h_in;
  complex float * ou = (complex float *) h_out;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;

  complex float val, beam_sum, phasor;

  unsigned ichan, iant;
  uint64_t isamp;

  for (ichan=0; ichan<nchan; ichan++)
  {
    for (isamp=0; isamp<nsamp; isamp++)
    {
      beam_sum = 0 + 0 * I;
      for (iant=0; iant<nant; iant++)
      {
        // unpack this sample and antenna
        val16 = in16[(ichan*nsampant) + (iant*nsamp) + isamp];
        val = ((float) val8[0]) + ((float) val8[1]) * I;

        beam_sum = (val * phasors[ichan*nant+iant]) + beam_sum;
      }
      ou[isamp*nchan+ichan] = beam_sum;
    }
  }
  return 0;
}
