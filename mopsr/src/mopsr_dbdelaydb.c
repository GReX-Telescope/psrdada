/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_cuda.h"
#include "mopsr_delays.h"
#include "mopsr_dbdelaydb.h"

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

void usage ()
{
	fprintf(stdout, "mopsr_dbdelaydb [options] inkey outkey\n"
    "\n"
    " -d id    use GPU device with id, default 0\n"
    " -n taps  number of FIR taps to use [default %d]\n"
    " -p file  read static phase information from file\n"
    " -s       process single DADA transfer then exit\n"
    " -v       verbose output\n", MOPSR_DBDELAYDB_DEFAULT_NTAPS
  );
}

int main(int argc, char** argv) 
{
  // dbdelaydb contextual struct
  mopsr_dbdelaydb_t ctx;

  // DADA Header plus Data Units
  dada_hdu_t* in_hdu = 0;
  dada_hdu_t* out_hdu = 0;

  // DADA Primary Read Client main loop
  dada_client_t* client = 0;

  // DADA Logger
  multilog_t* log = 0;

  int arg = 0;

  unsigned quit = 0;

  key_t in_key;

  key_t out_key;

  ctx.d_delays = 0;
  ctx.d_in = 0;
  ctx.d_out = 0;

  // default values
  ctx.verbose = 0;
  ctx.device = 0;
  ctx.ntaps = MOPSR_DBDELAYDB_DEFAULT_NTAPS;
  ctx.start_md_angle = 0;
  
  while ((arg = getopt(argc, argv, "d:hn:sv")) != -1) 
  {
    switch (arg)  
    {
      case 'd':
        ctx.device = atoi(optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

      case 'n':
        ctx.ntaps = atoi(optarg);
        break;

      case 's':
        quit = 1;
        break;

      case 'v':
        ctx.verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  if (sscanf (argv[optind], "%x", &in_key) != 1)
  {
    fprintf (stderr, "ERROR: could not parse inkey from %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }

  if (sscanf (argv[optind+1], "%x", &out_key) != 1)
  {
    fprintf (stderr, "ERROR: could not parse inkey from %s\n", argv[optind+1]);
    exit(EXIT_FAILURE);
  }

  log = multilog_open ("mopsr_dbdelaydb", 0); 

  multilog_add (log, stderr);

  // this client is primarily a reader on the input HDU
  in_hdu = dada_hdu_create (log);
  dada_hdu_set_key(in_hdu, in_key);
  if (dada_hdu_connect (in_hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not connect to input HDU\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_lock_read (in_hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not lock read on input HDU\n");
    return EXIT_FAILURE;
  }

  // now create the output HDU
  out_hdu = dada_hdu_create (log);
  dada_hdu_set_key(out_hdu, out_key);
  if (dada_hdu_connect (out_hdu) < 0)
  { 
    fprintf (stderr, "ERROR: could not connect to output HDU\n");
    return EXIT_FAILURE;
  }

  ctx.log   = log;

  if (dbdelaydb_init (&ctx, in_hdu, out_hdu) < 0)
  {
    fprintf (stderr, "ERROR: failed to initalise data structures\n");
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block        = in_hdu->data_block;
  client->header_block      = in_hdu->header_block;

  client->open_function     = dbdelaydb_open;
  client->io_function       = dbdelaydb_delay;
  client->io_block_function = dbdelaydb_delay_block_cpu;
  client->close_function    = dbdelaydb_close;
  client->direction         = dada_client_reader;

  client->context = &ctx;

  while (!client->quit)
  {
    if (ctx.verbose)
      multilog(client->log, LOG_INFO, "main: dada_client_read()\n");
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (ctx.verbose)
      multilog(client->log, LOG_INFO, "main: dada_unlock_read()\n");
    if (dada_hdu_unlock_read (in_hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      quit = 1;
    }

    if (quit)
      client->quit = 1;
    else
    {
      if (ctx.verbose)
        multilog(client->log, LOG_INFO, "main: dada_lock_read()\n");
      if (dada_hdu_lock_read (in_hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (ctx.verbose)
    multilog(client->log, LOG_INFO, "main: dada_hdu_disconnect()\n");

  if (dbdelaydb_destroy (&ctx, in_hdu, out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to release resources\n");
  }

  if (dada_hdu_disconnect (in_hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

/*! Perform initialization */
int dbdelaydb_init ( mopsr_dbdelaydb_t* ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu)
{
  multilog_t * log = ctx->log;

  // prepare the filter

  // for saftey and to get the correct response for a
  // delay of zero the number of filter taps must be odd
  if (ctx->ntaps % 2 == 0)
  {
    multilog (log, LOG_ERR, "init: number of taps must be odd\n");
    return -1;
  }

  // allocate memory for sinc filter
  ctx->filter = (float*) malloc (ctx->ntaps*sizeof(float));
  if (ctx->filter == NULL)
  {
    multilog (log, LOG_ERR, "init: could not allocate filter memory\n");
    return -1;
  }

  // if we are using a GPU engine
  if (ctx->device >= 0)
  {
    // select the gpu device
    int n_devices = dada_cuda_get_device_count();
    multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

    if (ctx->device >= n_devices)
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
      multilog (log, LOG_ERR, "dbdelaydb_init: could not create CUDA stream\n");
      return -1;
    }
  }

  // input and output datablocks must be the same size
  ctx->block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  if (ctx->block_size != ipcbuf_get_bufsz ((ipcbuf_t *) out_hdu->data_block))
  {
    multilog (log, LOG_ERR, "init: HDU block size mismatch input [%"PRIu64"] != ouput [%"PRIu64"]\n",
              ctx->block_size, ipcbuf_get_bufsz ((ipcbuf_t *) out_hdu->data_block));
    return -1;
  }

  ctx->work_buffer_size = ctx->block_size;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "init: allocating work buffer of %ld bytes\n", ctx->work_buffer_size);
  ctx->work_buffer = malloc (ctx->work_buffer_size);
  if (!ctx->work_buffer)
  {
    multilog (log, LOG_ERR, "init: could not allocated %ld bytes for work buffer\n", ctx->work_buffer_size);
    return -1;
  }

  ctx->out_hdu = out_hdu;

  if (ctx->device >= 0)
  {
    // ensure that we register the DADA DB buffers as Cuda Host memory
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: registering input HDU buffers\n");
    if (dada_cuda_dbregister(in_hdu) < 0)
    {
      fprintf (stderr, "failed to register in_hdu DADA buffers as pinned memory\n");
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: registering outputHDU buffers\n");
    if (dada_cuda_dbregister(out_hdu) < 0)
    {
      fprintf (stderr, "failed to register out_hdu DADA buffers as pinned memory\n");
      return -1;
    }

    ctx->h_delays = ctx->d_delays = 0;
    ctx->d_in = ctx->d_out = 0;
    ctx->d_overlap = 0;
  }

  return 0;
}

// allocate observation specific RAM buffers (mostly things that depend on NCHAN, NANT)
int dbdelaydb_alloc (mopsr_dbdelaydb_t * ctx)
{
  unsigned nchanant = ctx->nchan * ctx->nant;
  
  multilog_t * log = ctx->log;

  // Buffers for the fractions sample delays (host and device)
  ctx->delays_size = sizeof(float) * nchanant;
  cudaError_t error = cudaMalloc( &(ctx->d_delays), ctx->delays_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->delays_size);
    return -1;
  }

  error = cudaMallocHost ((void **) &(ctx->h_delays), ctx->delays_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->delays_size);
    return -1;
  }

  // since data will be FST order, we want our GPU RAM to allow space for the overlap (half_ntap + 1 * 2)
  // half_ntap is for the 
  unsigned half_ntap = ctx->ntaps / 2;
  ctx->d_buffer_size   = (size_t) ctx->block_size + (nchanant * (half_ntap + 1) * 2);

  // device memory for undelayed data
  error = cudaMalloc( &(ctx->d_in), ctx->block_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->d_buffer_size);
    return -1;
  }

  // device memory for fractionally delayed data
  error = cudaMalloc( &(ctx->d_out), ctx->d_buffer_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->d_buffer_size);
    return -1;
  }

  ctx->d_overlap_size = nchanant * ctx->ntaps * sizeof (int16_t);
  error = cudaMalloc( &(ctx->d_overlap), ctx->d_overlap_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->d_overlap_size);
    return -1;
  }
  return 0;
}

// de-allocate the observation specific memory
int dbdelaydb_dealloc (mopsr_dbdelaydb_t * ctx)
{
  if (ctx->device >= 0)
  {
    if (ctx->d_delays)
      cudaFree (ctx->d_delays);
    ctx->d_delays = 0;

    if (ctx->d_in)
      cudaFree (ctx->d_in);
    ctx->d_in = 0;

    if (ctx->d_out)
      cudaFree (ctx->d_out);
    ctx->d_out = 0;
  }
  return 0;
}

// determine application memory 
int dbdelaydb_destroy (mopsr_dbdelaydb_t * ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu)
{
  if (ctx->filter);
    free (ctx->filter);
  ctx->filter = 0;

  if (ctx->work_buffer)
    free (ctx->work_buffer);
  ctx->work_buffer = 0;

  if (ctx->device >= 0)
  {
    if (dada_cuda_dbunregister (in_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister input DADA buffers\n");
      return -1;
    }

    if (dada_cuda_dbunregister (out_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister input DADA buffers\n");
      return -1;
    }
  }
  return 0;
}
 

int dbdelaydb_open (dada_client_t* client)
{
  assert (client != 0);
  mopsr_dbdelaydb_t* ctx = (mopsr_dbdelaydb_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "dbdelaydb_open()\n");

  // lock the output datablock for writing
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: locking write on output HDU\n");
  if (dada_hdu_lock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "open: could not lock write on output HDU\n");
    return -1;
  }

  // initialize
  ctx->block_open = 0;
  ctx->bytes_written = 0;
  ctx->curr_block = 0;
  ctx->first_time = 1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: extracting params from header\n");

  // extract required metadata from header
  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NCHAN from header\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }

  if (ascii_header_get (client->header, "TSAMP", "%lf", &(ctx->tsamp)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read TSAMP from header\n");
    return -1;
  }

  if (ascii_header_get (client->header, "START_MD_ANGLE", "%lf", &(ctx->start_md_angle)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read START_MD_ANGLE from header\n");
    return -1;
  }

  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BYTES_PER_SECOND from header\n");
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  char tmp[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", tmp) == 1)
  {
    multilog (log, LOG_INFO, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    multilog (log, LOG_INFO, "open: UTC_START=UNKNOWN\n");
  }
  // convert UTC_START to a unix UTC
  ctx->utc_start = str2utctime (tmp);
  if (ctx->utc_start == (time_t)-1) 
  {
    multilog (log, LOG_ERR, "open: could not parse start time from '%s'\n", tmp);
    return -1;
  }

  // allocate the observation specific memory
  multilog (log, LOG_INFO, "open: allocating observation specific memory\n");
  if (dbdelaydb_alloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "open: could not alloc obs specific memory\n");
    return -1;
  }

  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: getting next free output header buffer\n");
  char * header = ipcbuf_get_next_write (ctx->out_hdu->header_block);
  if (!header)
  {
    multilog (log, LOG_ERR, "open: could not get next header block\n");
    return -1;
  }

  // copy the header from the in to the out
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: copying header from input to output\n");
  memcpy (header, client->header, header_size);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: marking output header filled\n");

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->out_hdu->header_block, header_size) < 0) 
  {
    multilog (log, LOG_ERR, "open: could not mark header_block filled\n");
    return -1;
  }

  client->transfer_bytes = transfer_size;
  client->optimal_bytes = 64*1024*1024;

  // we do not want to explicitly transfer the DADA header
  client->header_transfer = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: completed\n");

  return 0;

}

int dbdelaydb_close (dada_client_t* client, uint64_t bytes_written)
{

  assert (client != 0);
  mopsr_dbdelaydb_t* ctx = (mopsr_dbdelaydb_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbdelaydb_close()\n");

  if (ctx->block_open)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: ipcio_close_block_write bytes_written=%"PRIu64"\n", ctx->bytes_written);
    if (ipcio_close_block_write (ctx->out_hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->block_open = 0;
    ctx->bytes_written = 0;
  }

  if (dbdelaydb_dealloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "close: dbdelaydb_dealloc failed\n");
  } 

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "close: cannot unlock output HDU\n");
    return -1;
  }

  return 0;
}

/*! used for transferring the header, and uses pageable memory */
int64_t dbdelaydb_delay (dada_client_t* client, void * buffer, uint64_t bytes)
{
  assert (client != 0);
  mopsr_dbdelaydb_t* ctx = (mopsr_dbdelaydb_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;
  multilog (log, LOG_ERR, "delay: should not be called\n");

  return (int64_t) bytes;
}

int64_t dbdelaydb_delay_block_cpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_dbdelaydb_t* ctx = (mopsr_dbdelaydb_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  const uint64_t nsamp = bytes / (ctx->nchan * ctx->nant * ctx->ndim);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "delay_block_cpu: bytes=%"PRIu64", nchan=%d, nant=%d, ndim=%d, nsamp=%"PRIu64"\n",
              bytes, ctx->nchan, ctx->nant, ctx->ndim, nsamp);

  int16_t * in = (int16_t *) buffer;
  int16_t * out = (int16_t *) ctx->work_buffer;
  const uint64_t out_chan_stride = nsamp * ctx->nant;
  const uint64_t out_ant_stride  = ctx->nant;
  uint64_t out_block_id;

  // perform host re-ordering of data from TFS to FST [assume 8 bits/sample]
  uint64_t isamp;
  unsigned ichan, iant;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "delay_block_cpu: performing reordering\n");
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0;ichan<ctx->nchan; ichan++)
    {
      for (iant=0; iant<ctx->nant; iant++)
      {
        // copy 1 complex value [16 bits] at a time
        out[ichan*out_chan_stride + iant*out_ant_stride + isamp] = *in;
        in++;
      }
    }
  }
  
  // now open the output data block for direct I/O
  if (!ctx->block_open)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "delay_block_cpu: opening output block\n");
    ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  in = (int16_t *) ctx->work_buffer;
  out = (int16_t *) ctx->curr_block;
  size_t stride = nsamp * ctx->ndim;

  float delay = 0.1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "delay_block_cpu: applying filter\n");

  // now perform the subsample correction for each channel / antenna
  for (ichan=0;ichan<ctx->nchan; ichan++)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      sinc_filter((complex16*) in, (complex16*) out, ctx->filter, nsamp, ctx->ntaps, delay);
      in += stride;
      out += stride;
    }
  }

  ctx->bytes_written = bytes;
  if (ctx->block_open)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "delay_block_cpu: closing output block [%"PRIu64"]\n", ctx->bytes_written);
    ipcio_close_block_write (ctx->out_hdu->data_block, ctx->bytes_written);
    ctx->block_open = 0;
  }
  return (int64_t) ctx->bytes_written;
}

/*
 * GPU engine for delaying a block of data 
 */
int64_t dbdelaydb_delay_block_gpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_dbdelaydb_t* ctx = (mopsr_dbdelaydb_t*) client->context;

  multilog_t * log = ctx->log;

  cudaError_t error;
  uint64_t out_block_id;
  unsigned ichan, iant;

  // copy the whole block to the GPU
  error = cudaMemcpyAsync (ctx->d_in, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  if (ctx->first_time)
  {
    error = cudaMemsetAsync(ctx->d_overlap, 0, ctx->d_overlap_size, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemsetAsyc failed: %s\n", cudaGetErrorString(error));
      return -1;
    }
    ctx->first_time = 0;
  }

  // determine the timestamp corresponding to the current byte of data
  double obs_offset_seconds = (double) ctx->bytes_read / (double) ctx->bytes_per_second;
  struct timeval timestamp;
  timestamp.tv_sec = floor(obs_offset_seconds);
  timestamp.tv_usec = (obs_offset_seconds - (double) timestamp.tv_sec) * 1000000;

  char apply_instrumental = 1;
  char apply_geometric = 1;
  char is_tracking = 0;

  // update the delays
  if (calculate_delays (ctx->nbay, ctx->all_bays, ctx->nant, ctx->mods, ctx->nchan, ctx->chans, 
                        ctx->source, timestamp, ctx->delays, ctx->start_md_angle,
                        apply_instrumental, apply_geometric,
                        is_tracking, ctx->tsamp) < 0)
  {
    multilog (log, LOG_ERR, "delay_block_gpu: failed to update delays\n");
    return -1;
  }

  // layout the delays in host memory
  for (ichan=0; ichan < ctx->nchan; ichan++)
    for (iant=0; iant < ctx->nant; iant++)
      ctx->h_delays[ichan*ctx->nant + iant] = ctx->delays[iant][ichan].fractional;

  // copy the delays to the device
  error = cudaMemcpyAsync (ctx->d_delays, ctx->h_delays, ctx->delays_size, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // assume we are working in FST order (Freq, Space, Time)
  mopsr_input_delay_fractional (ctx->stream, ctx->d_in, ctx->d_out, ctx->d_overlap, ctx->d_delays, 
                     bytes, ctx->nchan, ctx->nant, ctx->ntaps);

  // copy back to output buffer
  if (!ctx->block_open)
  {
    ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  void * h_ptr = ctx->curr_block;
  void * d_ptr = ctx->d_out;
  char  block_full = 1;
  unsigned nchanant = ctx->nchan * ctx->nant;
  size_t chanant_stride = bytes / nchanant;
  size_t to_copy, d_offset, h_offset, shift;

  for (ichan=0; ichan < ctx->nchan; ichan++)
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      // sample offset on the device [initially 0]
      d_offset = ctx->d_byte_offsets[ichan][iant];

      // bytes to copy back for this chanant
      to_copy = chanant_stride - d_offset;

      // shift is new offset - old offset [samples]
      shift = ctx->sample_offsets[ichan][iant] - ctx->delays[iant][ichan].samples;
      if (shift != 0)
      {
        to_copy  -= shift * 2;
        d_offset += shift * 2;
      }
  
      // h_byte_offsets are [initially 0]
      h_offset = ctx->h_byte_offsets[ichan][iant];

      // dont overfill the buffer
      if (h_offset + to_copy > chanant_stride)
        to_copy = chanant_stride - h_offset;

      // bytes delayed can differ for *every* channel and antenna
      error = cudaMemcpyAsync (h_ptr + h_offset, d_ptr + d_offset, to_copy, cudaMemcpyDeviceToHost, ctx->stream);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
        return -1;
      }

      // record how many byters we have copied
      ctx->h_byte_offsets[ichan][iant] += to_copy;
      ctx->d_byte_offsets[ichan][iant] = d_offset + to_copy;

      if (ctx->h_byte_offsets[ichan][iant] < chanant_stride)
        block_full = 0;

      h_ptr += chanant_stride;
      d_ptr += chanant_stride;
    }
  }

  cudaStreamSynchronize(ctx->stream);

  if (block_full)
  {
    ipcio_close_block_write (ctx->out_hdu->data_block, bytes);
    ctx->block_open = 0;

    if (!ctx->block_open)
    {
      ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
      ctx->block_open = 1;
    }

    h_ptr = ctx->curr_block;
    d_ptr = ctx->d_out;
    
    // copy anything remaining in the d_ptr
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      for (iant=0; iant < ctx->nant; iant++)
      {
        d_offset = ctx->d_byte_offsets[ichan][iant];
        to_copy = chanant_stride - d_offset;

        // no shifting this time! [TODO check this]

        // new block so h_offset = 0
        h_offset = ctx->h_byte_offsets[ichan][iant] = 0;

        error = cudaMemcpyAsync (h_ptr + h_offset, d_ptr + d_offset, to_copy, cudaMemcpyDeviceToHost, ctx->stream);
        if (error != cudaSuccess)
        {
          multilog (log, LOG_ERR, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
          return -1;
        }

        // record how many byters we have copied
        ctx->h_byte_offsets[ichan][iant] += to_copy;
        ctx->d_byte_offsets[ichan][iant] = d_offset + to_copy;
      }
    }
  }

  for (ichan=0; ichan < ctx->nchan; ichan++)
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      if (ctx->d_byte_offsets[ichan][iant] != chanant_stride)
      {
        multilog (log, LOG_ERR, "d_byte_offsets[%d][%d] != %d\n", ichan, iant, chanant_stride);
        return -1;
      }
      ctx->d_byte_offsets[ichan][iant] = 0;
      ctx->sample_offsets[ichan][iant] = ctx->delays[iant][ichan].samples;
    }
  }

  ctx->bytes_read += bytes;
  return (int64_t) bytes;
}
