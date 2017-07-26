/***************************************************************************
 *
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "dada_generator.h"
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

#define JUST_GPU

int mopsr_tile_beams_cpu (void * h_in, void * h_out, void * h_phasors, uint64_t nbytes, unsigned nbeam, unsigned nchan, unsigned nant, unsigned tdec);

void usage ()
{
  fprintf(stdout, "mopsr_test_bfdsp bays_file modules_file\n"
    " -a nant     number of antennae\n" 
    " -b nbeam    number of beams\n" 
    " -c nchan    number of channels\n" 
    " -d id       cuda device id\n"
    " -l nloops   number of blocks to execute\n" 
    " -t nsamp    number of samples [in a block]\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n"
  );
}

int main(int argc, char** argv)
{
  int arg = 0;

  unsigned nant = 352;
  int nbeam = 352;
  const unsigned ndim_in = 2;
  const unsigned ndim_ou = 1;
#ifdef HIRES
  uint64_t nsamp = 16384;
  unsigned tdec  = 32;
  unsigned nchan = 10;
#else
  uint64_t nsamp = 16384 * 6;
  //uint64_t nsamp = 4096;
  unsigned tdec  = 512;
  unsigned nchan = 1;
#endif
  unsigned nloop = 1;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "a:c:b:d:hl:t:v")) != -1) 
  {
    switch (arg)
    {
      case 'a':
        nant = atoi(optarg);
        break;

      case 'c':
        nchan = atoi(optarg);
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

      case 'l':
        nloop = atoi (optarg);
        break;

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

  mopsr_chan_t * channels = (mopsr_chan_t *) malloc (sizeof(mopsr_chan_t) * nchan);
  unsigned schan = 204;
  double chan_bw = 100.0 / 1024;
  double base_freq = 800;
  unsigned ichan;
  for (ichan=0; ichan<nchan; ichan++)
  {
    channels[ichan].number = schan + ichan;
    channels[ichan].bw     = chan_bw;
    channels[ichan].cfreq  = base_freq + (chan_bw/2) + (ichan * chan_bw);
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

  uint64_t in_block_size = nsamp        * nant  * ndim_in * nbyte_in * nchan;
  uint64_t ou_block_size = (nsamp/tdec) * nbeam * ndim_ou * nbyte_ou * nchan;

  fprintf (stderr, "IN:  nant=%u  nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", nant,  nchan, ndim_in, nbyte_in*8, nsamp, in_block_size);
  fprintf (stderr, "OUT: nbeam=%u nchan=%u ndim=%u nbit=%u nsamp=%"PRIu64" block_size=%"PRIu64"\n", nbeam, nchan, ndim_ou, nbyte_ou*8, nsamp/tdec, ou_block_size);

  void * d_in;
  void * d_fbs;
  void * d_phasors;
  void * h_in;
  void * h_out;
  void * h_out_cpu;
  float * h_ant_factors;
  float * h_phasors;

  error = cudaMallocHost( &h_in, in_block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", in_block_size);
    return -1;
  }
  int8_t * ptr = (int8_t *) h_in;

  srand ( time(NULL) );
  double stddev = 5;

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

  size_t phasors_size = nchan * nant * nbeam * sizeof(float) * 2;
  error = cudaMallocHost( (void **) &(h_phasors), phasors_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", phasors_size);
    return -1;
  }

  unsigned nbeamant = nant * nbeam;
  float * phasors_ptr = (float *) h_phasors;
  unsigned re, im;
  unsigned chan_stride = nbeam * nant * 2;

  for (ibeam=0; ibeam<nbeam; ibeam++)
  {
    double md_angle = (double) ibeam / (double) nbeam;
    for (iant=0; iant<nant; iant++)
    {
      double geometric_delay = (sin(md_angle) * modules[iant].dist) / C;
      for (ichan=0; ichan<nchan; ichan++)
      {
        double theta = -2 * M_PI * channels[ichan].cfreq * 1000000 * geometric_delay;

        // packed in FS order with the cos, and sin terms separate
        unsigned idx = (ichan * chan_stride) + (ibeam * nant) + iant;
        h_phasors[idx] = (float) cos(theta);
        h_phasors[idx + nbeamant] = (float) sin(theta);
      }
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

  unsigned iloop;
  for (iloop=0; iloop<nloop; iloop++)
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

    // form the tiled, detected and integrated fan beamsa
    mopsr_tile_beams_precomp (stream, d_in, d_fbs, d_phasors, in_block_size, nbeam, nant, nchan);

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
  }
  cudaStreamSynchronize(stream);

#ifndef JUST_GPU

  unsigned ndim = 2;
  unsigned nsamp_out = in_block_size / (nchan * nant * ndim * tdec);

  fprintf (stderr, "mopsr_tile_beams_cpu()\n");
  mopsr_tile_beams_cpu (h_in, h_out_cpu, h_phasors, in_block_size, nbeam, nchan, nant, tdec);

  float * gpu_ptr = (float *) h_out;
  float * cpu_ptr = (float *) h_out_cpu;

  uint64_t ival=0;
  for (ibeam=0; ibeam<nbeam; ibeam++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (isamp=0; isamp<nsamp_out; isamp++)
      {
        const float cpu = cpu_ptr[ival];
        const float gpu = gpu_ptr[ival];
        float ratio = cpu > gpu ? cpu / gpu : gpu / cpu;
        if (fabs(ratio) > 1.00001)
          fprintf (stderr, "[%d][%d][%d] mismatch ratio=%f (cpu==%f != gpu==%f)\n", ibeam, ichan, isamp, ratio, cpu, gpu);
        ival++;
      }
    }
  }

#endif

  if (d_in)
  {
    error = cudaFree(d_in);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFree(d_in) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  d_in = 0;

  if (d_fbs)
  {
    error = cudaFree(d_fbs);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFree(d_fbs) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  d_fbs = 0;

  if (d_phasors)
  {
    error = cudaFree(d_phasors);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFree(d_phasors) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  d_phasors = 0;

  if (h_ant_factors)
  {
    error = cudaFreeHost(h_ant_factors);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFreeHost(h_ant_factors) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  h_ant_factors= 0;

  if (h_in)
  {
    error = cudaFreeHost(h_in);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFreeHost(h_in) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  h_in = 0;

  if (h_out)
  {
    error = cudaFreeHost(h_out);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFreeHost(h_out) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  h_out = 0;

  if (h_phasors)
  {
    error = cudaFreeHost(h_phasors);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "cudaFreeHost(h_phasors) failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }
  h_phasors = 0;

  free (modules);
  free (h_out_cpu);

  return 0;
}

// simpler CPU version
int mopsr_tile_beams_cpu (void * h_in, void * h_out, void * h_phasors, uint64_t nbytes, unsigned nbeam, unsigned nchan, unsigned nant, unsigned tdec)
{
  unsigned ndim = 2;

  // data is ordered in ST order
  unsigned nsamp = (unsigned) (nbytes / (nant * nchan * ndim));

  float * phasors = (float *) h_phasors;

  int16_t * in16 = (int16_t *) h_in;
  float * ou     = (float *) h_out;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;

  const unsigned nbeamant = nbeam * nant;
  const unsigned nantsamp = nant * nsamp;
  const unsigned ant_stride = nsamp;
  complex float val, beam_sum, phasor, steered;
  float beam_power;
  //complex double beam_sum_d;
  //double beam_power_d;
  const float scale = 127.5;

  // intergrate samples together
  const unsigned ndat = tdec;
  unsigned nchunk = nsamp / tdec; // ndat_out

  float re, im;
  complex float beam_phasors[nant];

  const uint64_t beam_stride = nchan * nchunk;
  const uint64_t chan_stride = nchunk;

  fprintf (stderr, "nbytes=%u nant=%u nsamp=%u, nchunk=%u\n", nbytes, nant, nsamp, nchunk);

  unsigned ichan, ibeam, ichunk, idat, isamp, iant, idx;
  unsigned ou_offset = 0;
  for (ichan=0; ichan<nchan; ichan++)
  {
    int16_t * in16 = ((int16_t *) h_in) + ichan * nantsamp;
    ou_offset = ichan * chan_stride;
    for (ibeam=0; ibeam<nbeam; ibeam++)
    {
      // compute phasors for all ant for this beam
      for (iant=0; iant<nant; iant++)
      {
        idx = (ichan * nbeamant * 2) + (ibeam * nant) + iant;
        beam_phasors[iant] = phasors[idx] + phasors[idx + nbeamant] * I;
      }

      isamp = 0;
      for (ichunk=0; ichunk<nchunk; ichunk++)
      {
        beam_power = 0;
        for (idat=0; idat<ndat; idat++)
        {
          if (ibeam == 0)
          {
            float incoherent_power = 0;
            for (iant=0; iant<nant; iant++)
            {
              val16 = in16[iant*nsamp + isamp];
              re = ((float) val8[0]) / scale;
              im = ((float) val8[1]) / scale;
              incoherent_power += (re * re) + (im * im);
            }
            beam_power   += incoherent_power;
          }
          else
          {
            // compute the "beam_sum" of all antena for each time sample
            beam_sum   = 0 + 0 * I;
            for (iant=0; iant<nant; iant++)
            {
              // unpack this sample and antenna
              val16 = in16[iant*nsamp + isamp];
              re = ((float) val8[0]) / scale;
              im = ((float) val8[1]) / scale;
              val = re + im * I;

              steered = val * beam_phasors[iant];
              beam_sum += steered;
              //beam_sum += val;

              /*
              if (ibeam == 1 && ichan == 0 && ichunk < 1)
              {
                fprintf (stderr, "[%u][%u] steered=(%f,%f), beam_sum=(%f,%f)\n",
                        ichan, ibeam, 
                        crealf(steered), cimagf(steered), 
                        crealf(beam_sum), cimagf(beam_sum));
              }
              */
            }
            beam_power   += (crealf(beam_sum)  * crealf(beam_sum))  + (cimagf(beam_sum)  * cimagf(beam_sum));
          }
          /*
          if (ibeam == 1 && ichan == 0 && ichunk < 1)
          {
            fprintf (stdout, "CPU %u %f %f\n", idat, crealf(beam_sum), cimagf(beam_sum));
          }
          */
          isamp++;
        }
        /*
        if (ibeam == 1 && ichan == 0 && ichunk < 1)
        {
          fprintf (stdout, "CPU %u %f\n", ichunk, beam_power);
        }
        */


        //if (fabsf (beam_power / (float) beam_power_d) > 1.001)
        //  fprintf (stderr, "[%u][%u] beam_power=%e beam_power_d=%le\n", ichan, ibeam, beam_power, beam_power_d);

        // output in SFT order
        ou[ou_offset + ichunk] = beam_power;
      }
      ou_offset += beam_stride;
    }
  }
}

