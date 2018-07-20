/***************************************************************************
 *
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *
 ****************************************************************************/

#include "dada_generator.h"
#include "dada_cuda.h"
#include "mopsr_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

void usage ()
{
  fprintf(stdout, "mopsr_test_bfst\n"
    " -a nant     number of antennae [default 352]\n" 
    " -c nchan    number of channels [default 10]\n" 
    " -d id       cuda device id\n"
    " -t nsamp    number of samples [default 16384]\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n"
  );
}

int main(int argc, char** argv)
{
  int arg = 0;

  unsigned ndim = 2;
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
  if (argc-optind != 0)
  {
    fprintf(stderr, "ERROR: 0 argument are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

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

  size_t block_size = nchan * nant * nsamp * ndim;
  void * h_in;
  void * d_in;
  void * d_out;
  void * h_out;

  error = cudaMallocHost( &h_in, block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", block_size);
    return -1;
  } 

  error = cudaMallocHost( &h_out, block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of pinned host memory\n", block_size);
    return -1;
  } 

  error = cudaMalloc( &d_in, block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", block_size);
    return -1;
  }

  error = cudaMalloc( &d_out, block_size);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "alloc: could not allocate %ld bytes of device memory\n", block_size);
    return -1;
  }

  int8_t * ptr = (int8_t *) h_in;

  srand ( time(NULL) );
  double stddev = 5;
 
  unsigned nant_per_block = 8;
  unsigned nblock = nant / nant_per_block;
  unsigned iblock, ichan, iant, isamp;

  for (iblock=0; iblock < nblock; iblock++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant_per_block; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          ptr[0] = (int8_t) rand_normal (0, stddev);
          ptr[1] = (int8_t) rand_normal (0, stddev);
          ptr += 2;
        }
      }
    }
  }

  cudaStreamSynchronize(stream);

  if (verbose > 1)
    fprintf(stderr, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", block_size, d_in, h_in);
  error = cudaMemcpyAsync (d_in, h_in, block_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              d_in, h_in);
    return -1;
  }

  // form the tiled, detected and integrated fan beamsa
  mopsr_transpose_BFST_FST (stream, d_in, d_out, block_size, nant, nchan);

  if (verbose > 1)
    fprintf(stderr, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
              h_out, d_out, block_size);
  error = cudaMemcpyAsync ( h_out, d_out, block_size,
                            cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  cudaStreamSynchronize(stream);

  int8_t * in_ptr = (int8_t *) h_in;
  int8_t * out_ptr = (int8_t *) h_out;

  uint64_t out_ant_stride = nsamp;
  uint64_t out_chan_stride = nant * nsamp;
  uint64_t nerrors = 0;

  for (iblock=0; iblock < nblock; iblock++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant_per_block; iant++)
      {
        for (isamp=0; isamp<nsamp; isamp++)
        {
          unsigned oant = iblock*nant_per_block + iant;
          uint64_t out_offset = (ichan * out_chan_stride + oant * out_ant_stride + isamp) * 2;

          if (((in_ptr[0] != out_ptr[out_offset]) || (in_ptr[1] != out_ptr[out_offset+1])) && ( nerrors < 10))
          {
            fprintf (stderr, "[%d][%d][%d][%d] (%d,%d) != (%d,%d) out_offset=%lu\n", iblock, ichan, iant, isamp, in_ptr[0], in_ptr[1], out_ptr[out_offset], out_ptr[out_offset+1], out_offset);
            nerrors++;
          }
          in_ptr += 2;
        }
      }
    }
  }

  error = cudaFree(d_in);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaFree(d_in) failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  error = cudaFree(d_out);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaFree(d_out) failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  error = cudaFreeHost(h_in);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaFreeHost(h_in) failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  error = cudaFreeHost(h_out);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaFreeHost(h_out) failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

