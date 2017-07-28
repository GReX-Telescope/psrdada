/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_cuda.h"
#include "mopsr_def.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "ascii_header.h"

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

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage()
{
  fprintf (stdout,
           "mopsr_dbphasedb [options] in_key out_key\n"
           " -b core   bind process to CPU core\n"
           " -d id     run on GPU device\n"
           " -s        1 transfer, then exit\n"
           " -v        verbose mode\n");
}

typedef struct {

  // output DADA key
  key_t key;

  // output HDU
  dada_hdu_t * hdu;

  multilog_t * log;

  char order[4];

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  uint8_t * block;

  uint64_t block_size;

  unsigned block_open;

  uint64_t bytes_written;

  unsigned nchan;

  unsigned npol;

  unsigned ndim;

  unsigned nant;

  unsigned int first;

  unsigned unique_corrections;

  float complex * corrections;

  unsigned chan_offset;

  int device;             // cuda device to use

  cudaStream_t stream;    // cuda stream for engine

  void * d_in;            // device memory for input

  void * d_out;           // device memory for output

  void * d_corr;          // device memory for co-efficients

} mopsr_dbrephasedb_t;

int dbrephasedb_init (mopsr_dbrephasedb_t * ctx, dada_hdu_t * in_hdu);
int dbrephasedb_destroy (mopsr_dbrephasedb_t * ctx, dada_hdu_t * in_hdu);

#define MOPSR_DBREPHASEDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbrephasedb_open (dada_client_t* client)
{
  mopsr_dbrephasedb_t * ctx = (mopsr_dbrephasedb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  // header parameters that will be adjusted
  unsigned old_nbit = 0;
  unsigned new_nbit = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "dbrephasedb_open()\n");

  // lock writer status on the out HDU
  if (dada_hdu_lock_write (ctx->hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->key);
    return -1;
  }

  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    ctx->nchan = 128;
    multilog (log, LOG_WARNING, "header had no NCHAN, assuming %d\n", ctx->nchan);
  }

  if (ascii_header_get (client->header, "NPOL", "%d", &(ctx->npol)) != 1)
  {
    ctx->npol = 1;
    multilog (log, LOG_WARNING, "header had no NPOL assuming %d\n", ctx->npol);
  }

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    ctx->nant = 1;
    multilog (log, LOG_WARNING, "header had no NANT assuming %d\n", ctx->nant);
  }

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    ctx->ndim = 2;
    multilog (log, LOG_WARNING, "header had no NDIM assuming %d\n", ctx->ndim);
  }

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "header had no ORDER\n");
    return -1;
  }

  if (ascii_header_get (client->header, "CHAN_OFFSET", "%u", &(ctx->chan_offset)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read CHAN_OFFSET from header\n");
    return -1;
  }

  if (strcmp(ctx->order, "TFS") != 0)
  {
    multilog (log, LOG_ERR, "ORDER [%s] was not TFS\n", ctx->order);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NCHAN=%d NANT=%d, NDIM=%d NPOL=%d\n", ctx->nchan, ctx->nant, ctx->ndim, ctx->npol);

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  // ensure its the same for the output data block
  assert( header_size == ipcbuf_get_bufsz (ctx->hdu->header_block) );

  // get the next free header block on the out HDU
  header = ipcbuf_get_next_write (ctx->hdu->header_block);
  if (!header)  {
    multilog (log, LOG_ERR, "could not get next header block\n");
    return -1;
  }

  // copy the header from the in to the out
  memcpy (header, client->header, header_size);

  // the GPU version must convert from TFS to FST
  if (ctx->device >= 0)
  {
    if (ascii_header_set (header, "ORDER", "%s", "FST") < 0)
    {
      multilog (log, LOG_ERR, "could not write ORDER=FST to out header\n");
      return -1;
    }
  }

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", ctx->key);

  client->transfer_bytes = 0;
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  ctx->bytes_written = 0; 
  client->header_transfer = 0;

  ctx->unique_corrections = 32;
  size_t corr_size = sizeof(float complex) * ctx->unique_corrections * ctx->nchan;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: generating %d bytes of corrections\n", corr_size);

  cudaError_t error;
  if (ctx->device >= 0)
    error = cudaMallocHost((void **) &(ctx->corrections), corr_size);
  else
    ctx->corrections = (float complex *) malloc(corr_size);

  unsigned ichan, ipt, icorr;
  float theta;
  float ratio = 2 * M_PI * (5.0 / 32.0);
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (ipt=0; ipt < ctx->unique_corrections; ipt++)
    {
      icorr = (ichan * ctx->unique_corrections) + ipt;
      theta = (ichan + ctx->chan_offset) * ratio * ipt;
      ctx->corrections[icorr] = sin (theta) - cos(theta) * I;
      //fprintf (stderr, "ctx->corrections[%d] = (%f, %f)\n", icorr, creal(ctx->corrections[icorr]), cimag(ctx->corrections[icorr]));
    }
  }

  if (ctx->device >= 0)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: cudaMalloc(%d) for d_corr\n", corr_size);
    error = cudaMalloc((void **) &(ctx->d_corr), corr_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "failed to malloc %d bytes on device: %s\n", corr_size, cudaGetErrorString(error));
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: cudaMemcpy(%d) for d_corr\n", corr_size);
    error = cudaMemcpyAsync (ctx->d_corr, ctx->corrections, corr_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
  }

  return 0;
}


/*! Function that closes the data transfer */
int dbrephasedb_close (dada_client_t* client, uint64_t bytes_written)
{
  // the mopsr_dbrephasedb specific data
  mopsr_dbrephasedb_t* ctx = 0;

  // status and error logging facility
  multilog_t* log;

  assert (client != 0);

  ctx = (mopsr_dbrephasedb_t*) client->context;

  assert (ctx != 0);
  assert (ctx->hdu != 0);

  log = client->log;
  assert (log != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%llu, bytes_out=%llu\n", ctx->bytes_in, ctx->bytes_out );

  if (ctx->block_open)
  {
    if (ipcio_close_block_write (ctx->hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "dbrephasedb_close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->block_open = 0;
    ctx->bytes_written = 0;
  }

  if (dada_hdu_unlock_write (ctx->hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbrephasedb_close: cannot unlock DADA HDU (key=%x)\n", ctx->key);
    return -1;
  }

  if (ctx->device >= 0)
  {
    cudaThreadSynchronize();
    if (ctx->d_corr)
      cudaFree (ctx->d_corr);
    ctx->d_corr = 0;
  }

  if (ctx->corrections)
    if (ctx->device >= 0)
      cudaFreeHost(&(ctx->corrections));
    else
      free (ctx->corrections);
  ctx->corrections = 0;

  return 0;
}

/*! Pointer to the function that transfers data to/from the target via direct block IO*/
int64_t dbrephasedb_write_block (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t in_block_id)
{
  assert (client != 0);
  mopsr_dbrephasedb_t* ctx = (mopsr_dbrephasedb_t*) client->context;
  multilog_t * log = client->log;

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "write_block: processing %llu bytes\n", in_data_size);

  // current DADA buffer block ID (unused)
  uint64_t out_block_id = 0;

  int64_t bytes_read = in_data_size;

  // number of bytes to be written to out DB
  uint64_t bytes_to_write = in_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "block_write: opening block\n");

  // open output DB for writing
  if (!ctx->block_open)
  {
    ctx->block = (int8_t *) ipcio_open_block_write(ctx->hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  const unsigned frame_size = (ctx->nchan * ctx->ndim * ctx->nant);

  const uint64_t nframe = in_data_size / frame_size;

  fprintf (stderr, "nframe=%d\n", nframe);

  if (in_data_size % frame_size)
    multilog (log, LOG_ERR, "input block size [%"PRIu64"] mod %d = %d, mismatch!!\n", in_data_size, frame_size, in_data_size % frame_size);
    
  unsigned int iframe, iant, ichan, icorr;
  int8_t * in;
  int8_t * out;
  unsigned int offset;
  float complex val;

  ctx->first = 1;

  in  = (int8_t *) in_data;
  out = (int8_t *) ctx->block;

  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      icorr = (ichan * ctx->unique_corrections) + (iframe % ctx->unique_corrections);

      for (iant=0; iant < ctx->nant; iant++)
      {
        val = (((float) in[0]) + 0.5) + (((float) in[1]) + 0.5) * I;
        val *= ctx->corrections[icorr];

        out[0] = (int8_t) floor(crealf(val) - 0.5);
        out[1] = (int8_t) floor(cimagf(val) - 0.5);

        in  += 2;
        out += 2;
      }
    }
  }

  // close output DB for writing
  if (ctx->block_open)
  {
    ipcio_close_block_write(ctx->hdu->data_block, bytes_to_write);
    ctx->block_open = 0;
    ctx->block = 0;
  }

  ctx->bytes_in += bytes_read;
  ctx->bytes_out += bytes_to_write;

  return bytes_read;
}

int64_t dbrephasedb_block_gpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_dbrephasedb_t* ctx = (mopsr_dbrephasedb_t*) client->context;

  multilog_t * log = client->log;

  cudaError_t error;
  uint64_t out_block_id;

  error = cudaMemcpyAsync (ctx->d_in, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }
  
  // execute transpose kernel
  multilog (log, LOG_INFO, "mopsr_input_transpose_TFS_to_FST (%d, %d, %d)\n", bytes, ctx->nchan, ctx->nant);
  mopsr_input_transpose_TFS_to_FST (ctx->stream, ctx->d_in, ctx->d_out, bytes, ctx->nchan, ctx->nant);
  check_error_stream( "mopsr_input_transpose_TFS_to_FST", ctx->stream);

  // execute rephasing kernel
  multilog (log, LOG_INFO, "mopsr_input_rephase (%d, %d, %d)\n", bytes, ctx->nchan, ctx->nant);
  mopsr_input_rephase (ctx->stream, ctx->d_out, ctx->d_corr, bytes, ctx->nchan, ctx->nant);
  check_error_stream( "mopsr_input_rephase", ctx->stream);

  // open output block
  if (!ctx->block_open)
  {
    ctx->block = (int8_t *) ipcio_open_block_write(ctx->hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  error = cudaMemcpyAsync (ctx->block, ctx->d_out, bytes, cudaMemcpyDeviceToHost, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // this ensures the all operations have completed before closing the output block
  cudaStreamSynchronize(ctx->stream);

  ipcio_close_block_write (ctx->hdu->data_block, bytes);
  ctx->block_open = 0;

  ctx->bytes_in += bytes;
  ctx->bytes_out += bytes;

  return (int64_t) bytes;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbrephasedb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  fprintf(stderr, "dbrephasedb_write should be disabled!!!!!\n");

  return data_size;
}

int dbrephasedb_init (mopsr_dbrephasedb_t * ctx, dada_hdu_t * in_hdu)
{
  multilog_t * log = ctx->log;

  // output DB block size must be bit_p times the input DB block size
  ctx->block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  int64_t out_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) ctx->hdu->data_block);

  if (ctx->block_size != out_block_size)
  {
    multilog (log, LOG_ERR, "init: output block size must be the same as input block size\n");
    return -1;
  }

  if (ctx->device >= 0)
  {
    // select the gpu device
    int n_devices = dada_cuda_get_device_count();
    multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

    if ((ctx->device < 0) && (ctx->device >= n_devices))
    {
      multilog (log, LOG_ERR, "dbdelaydb_init: no CUDA devices available [%d]\n",
                n_devices);
      return -1;
    }

    if (dada_cuda_select_device (ctx->device) < 0)
    {
      multilog (log, LOG_ERR, "dbdelaydb_init: could not select requested device [%d]\n",
                ctx->device);
      return -1;
    }

    char * device_name = dada_cuda_get_device_name (ctx->device);
    if (!device_name)
    {
      multilog (log, LOG_ERR, "dbdelaydb_init: could not get CUDA device name\n");
      return -1;
    }
    multilog (log, LOG_INFO, "Using device %d : %s\n", ctx->device, device_name);
    free(device_name);

    // setup the cuda stream for operations
    cudaError_t error = cudaStreamCreate(&(ctx->stream));
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "init: could not create CUDA stream\n");
      return -1;
    }

    if (!ctx->d_in)
    {
      error = cudaMalloc (&(ctx->d_in), ctx->block_size);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "dbdelaydb_init: could not create allocated %ld bytes of device memory\n", ctx->block_size);
        return -1;
      }
    }

    if (!ctx->d_out)
    {
      error = cudaMalloc (&(ctx->d_out), ctx->block_size);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "init: could not create allocated %ld bytes of device memory\n", ctx->block_size);
        return -1;
      }
    }

    // ensure that we register the DADA DB buffers as Cuda Host memory
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: registering input HDU buffers\n");
    if (dada_cuda_dbregister(in_hdu) < 0)
    {
      fprintf (stderr, "failed to register in_hdu DADA buffers as pinned memory\n");
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: registering output HDU buffers\n");
    if (dada_cuda_dbregister(ctx->hdu) < 0)
    {
      fprintf (stderr, "failed to register out_hdu DADA buffers as pinned memory\n");
      return -1;
    }
  }

  return 0;
}

int dbrephasedb_destroy (mopsr_dbrephasedb_t * ctx, dada_hdu_t * in_hdu)
{
  if (ctx->device >= 0)
  {
    if (ctx->d_in)
      cudaFree (ctx->d_in);
    ctx->d_in = 0;

    if (ctx->d_out)
      cudaFree (ctx->d_out);
    ctx->d_out = 0;

    if (dada_cuda_dbunregister (in_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister input DADA buffers\n");
      return -1;
    }

    if (dada_cuda_dbunregister (ctx->hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister output DADA buffers\n");
      return -1;
    }
  }
  return 0;
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_dbrephasedb_t dbrephasedb = MOPSR_DBREPHASEDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // core to run on
  int core = -1;

  // input data block HDU key
  key_t in_key = 0;

  // cuda device to use, default CPU
  int device = -1;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:d:sv")) != -1)
  {
    switch (arg) 
    {
      
      case 'c':
        core = atoi(optarg);
        break;

      case 'd':
        device = atoi(optarg);
        break;

      case 's':
        single_transfer = 1;
        break;

      case 'v':
        verbose++;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  dbrephasedb.verbose = verbose;
  dbrephasedb.device = device;

  int num_args = argc-optind;
  unsigned i = 0;
   
  if (num_args != 2)
  {
    fprintf(stderr, "mopsr_dbrephasedb: must specify 2 datablocks\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose > 1)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbrephasedb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (verbose > 1)
    fprintf (stderr, "parsing output key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbrephasedb.key)) != 1) {
    fprintf (stderr, "mopsr_dbrephasedb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbrephasedb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // open connection to the in/read DB
  hdu = dada_hdu_create (log);

  dada_hdu_set_key (hdu, in_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;


  // open connection to the out/write DB
  dbrephasedb.hdu = dada_hdu_create (log);
  
  // set the DADA HDU key
  dada_hdu_set_key (dbrephasedb.hdu, dbrephasedb.key);
  
  // connect to the out HDU
  if (dada_hdu_connect (dbrephasedb.hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    multilog (log, LOG_ERR, "cannot connected to DADA HDU (key=%x)\n", dbrephasedb.key);
    return EXIT_FAILURE;
  } 

  dbrephasedb.log = log;

  if (dbrephasedb_init (&dbrephasedb, hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to initialized reuqired resources\n");
    dada_hdu_disconnect (hdu);
    dada_hdu_disconnect (dbrephasedb.hdu);
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block   = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = dbrephasedb_open;
  client->io_function    = dbrephasedb_write;
  if (device >= 0)
    client->io_block_function = dbrephasedb_block_gpu;
  else
    client->io_block_function = dbrephasedb_write_block;

  client->close_function = dbrephasedb_close;
  client->direction      = dada_client_reader;

  client->context = &dbrephasedb;
  client->quiet = (verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

