
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
	fprintf(stdout, "mopsr_test_bfdsp bays_file modules_file\n"
    " -a nant     number of antennae\n" 
    " -b nbeam    number of beams\n" 
    " -t nsamp    number of samples\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned nant = 352;
  int nbeam = 512;
  const unsigned ndim = 2;
  uint64_t nsamp = 64;
  const unsigned tdec = 512;
  const unsigned nchan_out = 4;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:b:d:ht:v")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'b':
        nbeam = atoi(optarg);
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

  if (nbeam == -1)
    nbeam = nant;

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 argument are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  unsigned isamp, iant, imod, ibeam;

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
  cudaError_t error;

  fprintf (stderr, "main: cudaStreamCreate()\n");
  error = cudaStreamCreate(&stream);
  fprintf (stderr, "main: stream=%p\n", stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "could not create CUDA stream\n");
    return -1;
  }

  unsigned nbyte_in = sizeof(int8_t);
  unsigned nbyte_ou = sizeof(float);

  uint64_t in_block_size = nsamp * nant * ndim * nbyte_in;
  uint64_t ou_block_size = (nsamp * nbeam * nbyte_ou * nchan_out) / tdec;

  fprintf (stderr, "nant=%u nbeam=%u nchan_out=%u tdec=%u nsamp=%"PRIu64" in_block_size=%"PRIu64" ou_block_size=%"PRIu64"\n", nant, nbeam, nchan_out, tdec, nsamp, in_block_size, ou_block_size);

  void * d_in;
  void * d_fbs;
  void * d_phasors;
  void * h_in;
  void * h_out;
  void * h_out_cpu;
  float * h_ant_factors;
  float * h_sin_thetas;
  float * h_phasors;

  error = cudaMallocHost( &h_in, in_block_size); 
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", in_block_size);
    return -1;
  }
  int8_t * ptr = (int8_t *) h_in;
  for (iant=0; iant<nant; iant++)
  {
    for (isamp=0; isamp<nsamp; isamp++)
    {
      ptr[0] = (int8_t) iant + 1;
      ptr[1] = (int8_t) isamp + 1;
      ptr += 2;
      //fprintf (stderr, "[%d][%d] = %d + i%d\n", iant, isamp, iant+1, isamp+1);
    }
  }

  error = cudaMallocHost( &h_out, ou_block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", ou_block_size);
    return -1;
  }

  h_out_cpu = malloc(ou_block_size);

  // allocte CPU memory for each re-phasor
  size_t ant_factors_size = sizeof(float) * nant;
  error = cudaMallocHost( (void **) &(h_ant_factors), ant_factors_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", ant_factors_size);
    return -1;
  }

  double C = 2.99792458e8;
  double cfreq = 843e6;
  double dist;
  // the h_dist will be -2 * PI * FREQ * dist / C
  for (iant=0; iant<nant; iant++)
  {
    dist = 4.42*iant;
    h_ant_factors[iant] = (float) ((-2 * M_PI * cfreq * dist) / C);
  }

  size_t beams_size = sizeof(float) * nbeam;
  error = cudaMallocHost( (void **) &(h_sin_thetas), beams_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", beams_size);
    return -1;
  }

  // assume that the beams tile from -2 degrees to + 2 degrees in even steps 
  for (ibeam=0; ibeam<nbeam; ibeam++)
  {  
    float fraction = (float) ibeam / (float) (nbeam-1);
    h_sin_thetas[ibeam] = sinf((fraction * 4) - 2);
  }

  size_t phasors_size = nant * nbeam * sizeof(float) * 2;
  error = cudaMallocHost( (void **) &(h_phasors), phasors_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  unsigned nbeamant = nant * nbeam;
  float * phasors_ptr = (float *) h_phasors;
  unsigned re, im;
  for (ibeam=0; ibeam<nbeam; ibeam++)
  {
    for (iant=0; iant<nant; iant++)
    {
      re = (ibeam + iant + 2);
      im = (ibeam + iant + 3);

      h_phasors[ibeam * nant + iant] = (float) re;
      h_phasors[(ibeam * nant + iant) + nbeamant] = (float) im;
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
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_fbs\n", ou_block_size);
  error = cudaMalloc( &(d_fbs), ou_block_size);
  if (verbose)
    fprintf(stderr, "alloc: d_fbs=%p\n", d_fbs);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", ou_block_size);
    return -1;
  }

  if (verbose)
    fprintf(stderr, "alloc: allocating %ld bytes of device memory for d_fbs\n", phasors_size);
  error = cudaMalloc( &(d_phasors), phasors_size);
  if (verbose)
    fprintf(stderr, "alloc: d_fbs=%p\n", d_phasors);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  cudaStreamSynchronize(stream);

  if (verbose > 1)
    fprintf(stderr, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", phasors_size, d_phasors, h_phasors);
  error = cudaMemcpyAsync (d_phasors, h_phasors, phasors_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              d_phasors, h_phasors);
    return -1;
  }

  unsigned itrial;
  for (itrial=0; itrial<1; itrial++)
  {
//#if USE_GPU
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

    // form the tiled, detected and integrated fan beamsa
    //mopsr_tile_beams (stream, d_in, d_fbs, h_sin_thetas, h_ant_factors, in_block_size, nbeam, nant, tdec);
    mopsr_tile_beams_precomp (stream, d_in, d_fbs, d_phasors, in_block_size, nbeam, nant, tdec);

    if (verbose > 1)
      fprintf(stderr, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                h_out, d_fbs, ou_block_size);
    error = cudaMemcpyAsync ( h_out, d_fbs, ou_block_size,
                              cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
//#else

    cudaStreamSynchronize(stream);

    /*
    mopsr_tile_beams_cpu (h_in, h_out_cpu, h_phasors, in_block_size, nbeam, nant, tdec);

    unsigned nchunk = nsamp / 64;
    float * gpu = (float *) h_out;
    float * cpu = (float *) h_out_cpu;

    unsigned ichunk;
    for (ibeam=0; ibeam<nbeam; ibeam++)
    {
      for (ichunk=0; ichunk<nchunk; ichunk++)
      {
        fprintf (stderr, "[%d][%d] cpu=%f gpu=%f\n",ibeam, ichunk, cpu[ibeam*nchunk+ichunk], gpu[ichunk*nbeam+ibeam]);
      }
    }
    */

//#endif
  }

  cudaStreamSynchronize(stream);

  if (d_in)
    cudaFree(d_in);
  d_in = 0;

  if (d_fbs)
    cudaFree (d_fbs);
  d_fbs = 0;

  if (d_phasors)
    cudaFree (d_phasors);
  d_phasors = 0;

  if (h_ant_factors)
    cudaFreeHost(h_ant_factors);
  h_ant_factors= 0;

  if (h_in)
    cudaFreeHost(h_in);
  h_in = 0;

  if (h_out)
    cudaFreeHost(h_out);
  h_out = 0;

  if (h_phasors)
    cudaFreeHost(h_phasors);
  h_phasors = 0;

  if (h_sin_thetas)
    cudaFreeHost(h_sin_thetas);
  h_sin_thetas = 0;

  free (modules);

  return 0;
}

// simpler CPU version
int mopsr_tile_beams_cpu (void * h_in, void * h_out, void * h_phasors, uint64_t nbytes, unsigned nbeam, unsigned nant, unsigned tdec)
{
  unsigned ndim = 2;

  // data is ordered in ST order
  uint64_t nsamp = nbytes / (nant * ndim);

  float * phasors = (float *) h_phasors;

  int16_t * in16 = (int16_t *) h_in;
  float * ou     = (float *) h_out;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;

  const unsigned nbeamant = nbeam * nant;
  const unsigned ant_stride = nsamp;
  complex float val, beam_sum, phasor, steered;
  float beam_power;

  // intergrate 32 samples together
  const unsigned ndat = tdec;
  unsigned nchunk = nsamp / tdec;

  unsigned ibeam, ichunk, idat, isamp, iant;
  for (ibeam=0; ibeam<nbeam; ibeam++)
  {
    isamp = 0;
    for (ichunk=0; ichunk<nchunk; ichunk++)
    {
      beam_power = 0;
      for (idat=0; idat<ndat; idat++)
      {
        beam_sum = 0 + 0 * I;
        for (iant=0; iant<nant; iant++)
        {
          // unpack this sample and antenna
          val16 = in16[iant*nsamp + isamp];
          val = ((float) val8[0]) + ((float) val8[1]) * I;

          // the required phase rotation for this beam and antenna
          phasor = phasors[ibeam * nant + iant] + phasors[(ibeam * nant) + iant + nbeamant] * I;
          steered = val * phasor;
          // add the steered tied array beam to the total
          beam_sum += steered;
        }
        beam_power += (creal(beam_sum) * creal(beam_sum)) + (cimag(beam_sum) * cimag(beam_sum));
        isamp++;
      }
      ou[ibeam*nchunk+ichunk] = beam_power;
    }
  }
}
