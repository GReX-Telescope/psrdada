/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "mopsr_cuda.h"
#include "mopsr_delays.h"
#include "mopsr_bfdsp.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <float.h>

void usage ()
{
	fprintf(stdout, "mopsr_bfdsp [options] inkey outkey\n"
    "\n"
    "  Apply DSP operations to MOPSR BF pipeine:\n"
    "    Tied-Array beam (nyquist)\n"
    "    Fan-Beams, detected, integrated\n"
    "\n"
    "  inkey             PSRDADA hexideciaml key for input ST data\n"
    "  outkey            PSRDADA hexideciaml key for input output Fan beams\n"
    "\n"
    "  -d <id>           use GPU device with id [default 0]\n"
    "  -b nbeam          create nbeam fan beams [default 352]\n"
    "  -t key,ra,dec     create a tied array beam on PSRDADA key at RA and DEC\n"
    "  -v                verbose output\n"
    "  *                 optional\n");
}

int main(int argc, char** argv) 
{
  // bfdsp contextual struct
  mopsr_bfdsp_t ctx;

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

  ctx.d_in = 0;
  ctx.d_fbs = 0;

  // default values
  ctx.internal_error = 0;
  ctx.verbose = 0;
  ctx.device = 0;
  ctx.nbeam = -1;

  while ((arg = getopt(argc, argv, "b:d:hstv")) != -1) 
  {
    switch (arg)  
    {
      case 'b':
        ctx.nbeam = atoi(optarg);
        break;

      case 'd':
        ctx.device = atoi(optarg);
        break;

      case 'h':
        usage ();
        return 0;

      case 's':
        quit = 1;
        break;

      case 't':
        ctx.obs_tracking = 1;
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

  // read the modules file that describes the array
  char * bays_file = argv[optind+2];

  // read the modules file that describes the array
  char * modules_file = argv[optind+3];

  // read the signal path file that describes what is connected to what
  char * signal_path_file = argv[optind+4];

  log = multilog_open ("mopsr_bfdsp", 0); 

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

  ctx.log = log;

  if (bfdsp_init (&ctx, in_hdu, out_hdu, bays_file, modules_file, signal_path_file) < 0)
  {
    fprintf (stderr, "ERROR: failed to initalise data structures\n");
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block        = in_hdu->data_block;
  client->header_block      = in_hdu->header_block;

  client->open_function     = bfdsp_open;
  client->io_function       = bfdsp_io;
  client->io_block_function = bfdsp_io_block;
  client->close_function    = bfdsp_close;
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
    
    if (ctx.internal_error)
    {
      multilog (log, LOG_ERR, "internal error ocurred during read\n");
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

  if (bfdsp_destroy (&ctx, in_hdu, out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to release resources\n");
  }

  if (dada_hdu_disconnect (in_hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

/*! Perform initialization */
int bfdsp_init ( mopsr_bfdsp_t* ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu, char * bays_file, char * modules_file, char * signal_paths_file)
{
  multilog_t * log = ctx->log;

  ctx->all_bays = read_bays_file (bays_file, &(ctx->nbays));
  if (!ctx->all_bays)
  {
    multilog (log, LOG_ERR, "init: failed to read bays file [%s]\n", bays_file);
    return -1;
  }

  ctx->all_modules = read_modules_file (modules_file, &(ctx->nmodules));
  if (!ctx->all_modules)
  {
    multilog (log, LOG_ERR, "init: failed to read modules file [%s]\n", modules_file);
    return -1;
  }

  // select the gpu device
  int n_devices = dada_cuda_get_device_count();
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

  if ((ctx->device < 0) || (ctx->device >= n_devices))
  {
    multilog (log, LOG_ERR, "bfdsp_init: no CUDA devices available [%d]\n",
              n_devices);
    return -1;
  }

  if (dada_cuda_select_device (ctx->device) < 0)
  {
    multilog (log, LOG_ERR, "bfdsp_init: could not select requested device [%d]\n",
              ctx->device);
    return -1;
  }

  char * device_name = dada_cuda_get_device_name (ctx->device);
  if (!device_name)
  {
    multilog (log, LOG_ERR, "bfdsp_init: could not get CUDA device name\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Using device %d : %s\n", ctx->device, device_name);

  free(device_name);

  // setup the cuda stream for operations
  cudaError_t error = cudaStreamCreate(&(ctx->stream));
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "bfdsp_init: could not create CUDA stream\n");
    return -1;
  }

  // input block must be a multiple of the output block in bytes
  ctx->in_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  ctx->ou_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) out_hdu->data_block);
  if (ctx->in_block_size / ctx->tdec != ctx->ou_block_size)
  {
    multilog (log, LOG_ERR, "bfdsp_init: input block size / dec != outout block size: input [%"PRIu64"], ouput [%"PRIu64"]\n",
              ctx->in_block_size, ctx->ou_block_size);
    return -1;
  }

  ctx->out_hdu = out_hdu;

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
  if (dada_cuda_dbregister(out_hdu) < 0)
  {
    fprintf (stderr, "failed to register out_hdu DADA buffers as pinned memory\n");
    return -1;
  }

  ctx->d_in = ctx->d_fbs = 0;

  if (ctx->verbose)
     multilog (log, LOG_INFO, "init: completed\n");

  return 0;
}

// allocate observation specific RAM buffers (mostly things that depend on NANT)
int bfdsp_alloc (mopsr_bfdsp_t * ctx)
{
  multilog_t * log = ctx->log;

  // allocte CPU memory for each re-phasor
  size_t dist_size = sizeof(float) * ctx->nant;
  cudaError_t error = cudaMallocHost( (void **) &(ctx->h_ant_factors), dist_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", dist_size);
    return -1;
  }

  unsigned iant;
  for (iant=0; iant<ctx->nant; iant++)
    ctx->h_ant_factors[iant] = iant * 4;

  size_t beams_size = sizeof(float) * ctx->nbeam;
  error = cudaMallocHost( (void **) &(ctx->h_sin_thetas), beams_size);
  if (error != cudaSuccess)
  {   
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", beams_size);
    return -1;
  }

  unsigned ibeam;
  for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    ctx->h_sin_thetas[ibeam] = ibeam * 0.1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_in\n", ctx->in_block_size);
  error = cudaMalloc( &(ctx->d_in), ctx->in_block_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_in=%p\n", ctx->d_in);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->in_block_size);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_fbs\n", ctx->ou_block_size);
  error = cudaMalloc( &(ctx->d_fbs), ctx->ou_block_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_fbs=%p\n", ctx->d_fbs);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->ou_block_size);
    return -1;
  }

  cudaStreamSynchronize(ctx->stream);

  return 0;
}

// de-allocate the observation specific memory
int bfdsp_dealloc (mopsr_bfdsp_t * ctx)
{
  // ensure no operations are pending
  cudaStreamSynchronize(ctx->stream);

  if (ctx->h_ant_factors)
    cudaFreeHost(ctx->h_ant_factors);
  ctx->h_ant_factors = 0;

  if (ctx->h_sin_thetas)
    cudaFreeHost (ctx->h_sin_thetas);
  ctx->h_sin_thetas = 0;

  if (ctx->d_in)
    cudaFree (ctx->d_in);
  ctx->d_in = 0;

  if (ctx->d_fbs)
    cudaFree (ctx->d_fbs);
   ctx->d_fbs = 0;

  return 0;
}

// determine application memory 
int bfdsp_destroy (mopsr_bfdsp_t * ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu)
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
 

int bfdsp_open (dada_client_t* client)
{
  assert (client != 0);
  mopsr_bfdsp_t* ctx = (mopsr_bfdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "bfdsp_open()\n");

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
  ctx->bytes_read = 0;
  ctx->bytes_written = 0;
  ctx->curr_block = 0;
  ctx->first_time = 1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: extracting params from header\n");

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NANT=%d\n", ctx->nant);

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }
  if (ctx->nbeam == -1)
    ctx->nbeam = ctx->nant;

  if (ascii_header_get (client->header, "TSAMP", "%lf", &(ctx->tsamp)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read TSAMP from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: TSAMP=%lf\n", ctx->tsamp);

  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BYTES_PER_SECOND from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: BYTES_PER_SECOND=%"PRIu64"\n", ctx->bytes_per_second);

  if (ascii_header_get (client->header, "CHAN_OFFSET", "%u", &(ctx->channel.number)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read CHAN_OFFSET from header\n");
    return -1;
  }

  // for the boresite of the TB
  if (ascii_header_get (client->header, "SOURCE", "%s", ctx->source.name) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read SOURCE from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: SOURCE=%s\n", ctx->source.name);

  char position[32];
  if (ascii_header_get (client->header, "RA", "%s", position) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read RA from header\n");
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: RA (HMS) = %s\n", position);
  if (mopsr_delays_hhmmss_to_rad (position, &(ctx->source.raj)) < 0)
  {
    multilog (log, LOG_ERR, "open: could not parse RA from %s\n", position);
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: RA (rad) = %lf\n", ctx->source.raj);

  if (ascii_header_get (client->header, "DEC", "%s", position) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read RA from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: DEC (DMS) = %s\n", position);
  if (mopsr_delays_ddmmss_to_rad (position, &(ctx->source.decj)) < 0)
  {
    multilog (log, LOG_ERR, "open: could not parse DEC from %s\n", position);
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: DEC (rad) = %lf\n", ctx->source.decj);

  if (ascii_header_get (client->header, "UT1_OFFSET", "%f", &(ctx->ut1_offset)) == 1)
  {
    multilog (log, LOG_INFO, "open: UT1_OFFSET=%f\n", ctx->ut1_offset);
  } 
  else
  {
    ctx->ut1_offset = -0.272;
    multilog (log, LOG_INFO, "open: hard coding UT1_OFFSET=%f\n", ctx->ut1_offset);
  }

  char tmp[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", tmp) == 1)
  {
    if (ctx->verbose)
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

  // now calculate the apparent RA and DEC for the current timestamp
  struct timeval timestamp;
  timestamp.tv_sec = ctx->utc_start;
  timestamp.tv_usec = (long) (ctx->ut1_offset * 1000000);

  calc_app_position (ctx->source.raj, ctx->source.decj, timestamp, 
                    &(ctx->source.ra_curr), &(ctx->source.dec_curr));

  // multilog (log, LOG_INFO, "open: coords J2000=(%lf, %lf) CURR=(%lf, %lf)\n", 
  //           ctx->source.raj, ctx->source.decj, ctx->source.ra_curr, ctx->source.dec_curr);

  if (ascii_header_get (client->header, "BW", "%lf", &(ctx->channel.bw)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BW from header\n");
    return -1;
  }

  float freq;
  if (ascii_header_get (client->header, "FREQ", "%lf", &(ctx->channel.cfreq)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read FREQ from header\n");
    return -1;
  }

  char order[4];
  if (ascii_header_get (client->header, "ORDER", "%s", order) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read ORDER from header\n");
    return -1;
  }
  if (strcmp (order, "ST") != 0)
  {
    multilog (log, LOG_ERR, "open: input ORDER was not ST\n");
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  // allocate the observation specific memory
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: allocating observation specific memory\n");
  if (bfdsp_alloc (ctx) < 0)
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

  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, ctx->ou_block_size) < 0)
  {
    multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", ctx->ou_block_size);
    return -1;
  }

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read OBS_OFFSET from header\n");
   return -1;
  }
  ctx->bytes_read += obs_offset;

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

int bfdsp_close (dada_client_t* client, uint64_t bytes_written)
{

  assert (client != 0);
  mopsr_bfdsp_t* ctx = (mopsr_bfdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "bfdsp_close()\n");

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

  if (bfdsp_dealloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "close: bfdsp_dealloc failed\n");
  } 

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "close: cannot unlock output HDU\n");
    return -1;
  }

  return 0;
}

/*! used for transferring the header, and uses pageable memory */
int64_t bfdsp_io (dada_client_t* client, void * buffer, uint64_t bytes)
{
  assert (client != 0);
  mopsr_bfdsp_t* ctx = (mopsr_bfdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;
  multilog (log, LOG_ERR, "io: should not be called\n");

  return (int64_t) bytes;
}

/*
 * GPU engine for delaying a block of data 
 */
int64_t bfdsp_io_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_bfdsp_t* ctx = (mopsr_bfdsp_t*) client->context;
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: buffer=%p, bytes=%"PRIu64" block_id=%"PRIu64"\n", buffer, bytes, block_id);

  cudaError_t error;
  uint64_t out_block_id;
  unsigned ichan, iant;
  uint64_t bytes_out = bytes / ctx->tdec;;

  // copy the whole block to the GPU
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", bytes, ctx->d_in, buffer);
  error = cudaMemcpyAsync (ctx->d_in, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              ctx->d_in, buffer);
    return -1;
  }

  // form the tiled, detected and integrated fan beamsa
  mopsr_tile_beams (ctx->stream, ctx->d_in, ctx->d_fbs, ctx->h_sin_thetas, ctx->h_ant_factors,
                    bytes, ctx->nbeam, ctx->nant, ctx->tdec);

  // copy back to output buffer
  if (!ctx->block_open)
  {
    ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
    ctx->block_open = 1;
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
              (void *) ctx->curr_block, ctx->d_fbs, bytes_out);
  error = cudaMemcpyAsync ( (void *) ctx->curr_block, ctx->d_fbs, ctx->ou_block_size, 
                            cudaMemcpyDeviceToHost, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
    ctx->internal_error = 1;
    return -1;
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: cudaStreamSynchronize()\n");
  cudaStreamSynchronize(ctx->stream);

  ipcio_close_block_write (ctx->out_hdu->data_block, bytes_out);

  ctx->bytes_read += bytes;
  return (int64_t) bytes;
}
