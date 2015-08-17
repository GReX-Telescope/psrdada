
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

int mopsr_tie_beams_cpu (void * h_in, void * h_out, void * h_tb_phasors, uint64_t nbytes, unsigned nant);

#define SKZAP

void usage ()
{
	fprintf(stdout, "mopsr_test_bfdsp_tb bays_file modules_file\n"
    " -a nant     number of antennae\n" 
    " -t nsamp    number of samples\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nant = 352;
  uint64_t nsamp = 16384 * 6;
  const unsigned nchan_in = 1;
  const unsigned nchan_ou = 1;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:d:ht:v")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        nant = atoi(optarg);
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

  mopsr_chan_t channels;
  unsigned ichan = 42;
  channels.number = ichan;
  channels.bw     = 0.78125;
  channels.cfreq  = (800 + (0.78125/2) + (ichan * 0.78125));

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

  uint64_t in_block_size = nsamp * nant  * ndim * nbyte_in * nchan_in;
  uint64_t ou_block_size = nsamp * 1     * ndim * nbyte_ou * nchan_ou;

  fprintf (stderr, "IN:  nant=%u  nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", nant,  nchan_in, ndim, nbyte_in*8, nsamp, in_block_size);
  fprintf (stderr, "OUT: nbeam=%u nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", 1,     nchan_ou, ndim, nbyte_ou*8, nsamp, ou_block_size);

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

  for (iant=0; iant<nant; iant++)
  {
    for (isamp=0; isamp<nsamp; isamp++)
    {
      //ptr[0] = (int8_t) iant + 1;
      //ptr[1] = (int8_t) isamp + 1;
      ptr[0] = (int8_t) rand_normal (0, stddev);
      ptr[1] = (int8_t) rand_normal (0, stddev);
      ptr += 2;
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

  size_t phasors_size = nant * sizeof(float) * 2;
  error = cudaMallocHost( (void **) &(h_tb_phasors), phasors_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  complex float * phasors_ptr = (complex float *) h_tb_phasors;
  unsigned re, im;
  float md_angle = 0.0;

  for (iant=0; iant<nant; iant++)
  {
    phasors_ptr[iant] = cosf(md_angle) + sinf(md_angle) * I;
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
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_tb\n", phasors_size);
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

  char check_cpu_gpu = 0;
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

    mopsr_tie_beam (stream, d_in, d_tb, d_tb_phasors, in_block_size, nant);
    if (check_cpu_gpu)
      mopsr_tie_beams_cpu (h_in, h_out_cpu, h_tb_phasors, in_block_size, nant);

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

    for (isamp=0; isamp<nsamp; isamp++)
    {
      ptr = ((int8_t *) h_in) + (2 * isamp);
      for (iant=0; iant<nant; iant++)
      {
        re8 = ptr[0];
        im8 = ptr[1];

        //if (isamp < 3)
        //  fprintf (stderr, "%d=(%d,%d) ", iant, re8, im8);
        ptr += 2 * nsamp;
      }

      if (cpu_ptr[isamp] != gpu_ptr[isamp])
        fprintf (stderr, "mismatch on isamp=%d\n", isamp);
      //if (isamp < 3)
      //  fprintf (stderr, "cpu=(%f, %f) gpu=(%f,%f)\n", creal(cpu_ptr[isamp]), cimag(cpu_ptr[isamp]), creal(gpu_ptr[isamp]), cimag(gpu_ptr[isamp]));
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
int mopsr_tie_beams_cpu (void * h_in, void * h_out, void * h_tb_phasors, uint64_t nbytes, unsigned nant)
{
  unsigned ndim = 2;

  // data is ordered in ST order
  uint64_t nsamp = nbytes / (nant * ndim);

  complex float * phasors = (complex float *) h_tb_phasors;

  int16_t * in16 = (int16_t *) h_in;
  complex float * ou = (complex float *) h_out;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;

  complex float val, beam_sum, phasor;

  unsigned iant;
  uint64_t isamp;

  for (isamp=0; isamp<nsamp; isamp++)
  {
    beam_sum = 0 + 0 * I;
    for (iant=0; iant<nant; iant++)
    {
      // unpack this sample and antenna
      val16 = in16[iant*nsamp + isamp];
      val = ((float) val8[0]) + ((float) val8[1]) * I;

      beam_sum = (val * phasors[iant]) + beam_sum;
      //beam_sum = val + beam_sum;
    }
    ou[isamp] = beam_sum;
  }
}
