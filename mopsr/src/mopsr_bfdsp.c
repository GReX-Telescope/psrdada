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

#define USE_GPU

void usage ()
{
	fprintf(stdout, "mopsr_bfdsp [options] inkey bays.txt modules.txt\n"
    "\n"
    "  Apply DSP operations to MOPSR BF pipeine:\n"
    "    Tied-Array beam (nyquist)\n"
    "    Fan-Beams, detected, integrated\n"
    "\n"
    "  inkey             PSRDADA hexideciaml key for input ST data\n"
    "\n"
    "  -b nbeam          generate nbeam fan beams [default nant]\n"
    "  -d <id>           use GPU device with id [default 0]\n"
    "  -f fb_key         output integrated, tiled beams on fb_key [ST ordering]\n"
    "  -t tb_key         output baseband tied array beam on tb_key [T ordering]\n"
    "  -v                verbose output\n"
    "  -0                zero output data stream\n"
    "  *                 optional\n");
}

int main(int argc, char** argv) 
{
  // bfdsp contextual struct
  mopsr_bfdsp_t ctx;

  // DADA Header plus Data Units
  dada_hdu_t* in_hdu = 0;
  dada_hdu_t* fb_hdu = 0;
  dada_hdu_t* tb_hdu = 0;

  // DADA Primary Read Client main loop
  dada_client_t* client = 0;

  // DADA Logger
  multilog_t* log = 0;

  int arg = 0;

  unsigned quit = 0;

  key_t in_key;
  key_t fb_key = 0;
  key_t tb_key = 0;

  ctx.d_in = 0;
  ctx.d_fbs = 0;

  // default values
  ctx.internal_error = 0;
  ctx.verbose = 0;
  ctx.device = 0;
  ctx.nbeam = 0;
  ctx.zero_data = 0;

  ctx.fan_beams = 0;
  ctx.tied_beam = 0;

  while ((arg = getopt(argc, argv, "b:d:f:hst:v0")) != -1) 
  {
    switch (arg)  
    {
      case 'b':
        ctx.nbeam = atoi(optarg);
        break;

      case 'd':
        ctx.device = atoi(optarg);
        break;

      case 'f':
        ctx.fan_beams = 1;
        if (sscanf (optarg, "%x", &fb_key) != 1)
        {
          fprintf (stderr, "ERROR: could not parse fb_key from %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;

      case 'h':
        usage ();
        return 0;

      case 's':
        quit = 1;
        break;

      case 't':
        ctx.tied_beam = 1;
        if (sscanf (optarg, "%x", &tb_key) != 1)
        {
          fprintf (stderr, "ERROR: could not parse tb_key from %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;

      case 'v':
        ctx.verbose ++;
        break;

      case '0':
        ctx.zero_data = 1;
        break;

      default:
        usage ();
        return 0;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  if (sscanf (argv[optind], "%x", &in_key) != 1)
  {
    fprintf (stderr, "ERROR: could not parse inkey from %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }

  // read the modules file that describes the array
  char * bays_file = argv[optind+1];

  // read the modules file that describes the array
  char * modules_file = argv[optind+2];

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

  // now create the output HDUs
  if (ctx.fan_beams)
  {
    multilog (log, LOG_INFO, "preparing Fan Beam HDU\n");
    fb_hdu = dada_hdu_create (log);
    dada_hdu_set_key(fb_hdu, fb_key);
    if (dada_hdu_connect (fb_hdu) < 0)
    { 
      fprintf (stderr, "ERROR: could not connect to output HDU\n");
      return EXIT_FAILURE;
    }
  }

  if (ctx.tied_beam)
  {
    multilog (log, LOG_INFO, "preparing Tied Beam HDU\n");
    tb_hdu = dada_hdu_create (log);
    dada_hdu_set_key(tb_hdu, tb_key);
    if (dada_hdu_connect (tb_hdu) < 0)
    {
      fprintf (stderr, "ERROR: could not connect to output HDU\n");
      return EXIT_FAILURE;
    }
  }

  ctx.log = log;

  if (bfdsp_init (&ctx, in_hdu, fb_hdu, tb_hdu, bays_file, modules_file) < 0)
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
    multilog(client->log, LOG_INFO, "main: bfdsp_destroy()\n");
  if (bfdsp_destroy (&ctx, in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to release resources\n");
  }

  if (ctx.verbose)
    multilog(client->log, LOG_INFO, "main: dada_hdu_disconnect()\n");
  if (dada_hdu_disconnect (in_hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

/*! Perform initialization */
int bfdsp_init ( mopsr_bfdsp_t* ctx, dada_hdu_t * in_hdu, dada_hdu_t * fb_hdu, dada_hdu_t * tb_hdu, char * bays_file, char * modules_file)
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

#ifdef USE_GPU
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
#endif

  ctx->fb_block_size = 0;
  ctx->fb_hdu = 0;

  // input block must be a multiple of the output block in bytes
  ctx->in_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  if (fb_hdu)
  {
    ctx->fb_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) fb_hdu->data_block);
    if (ctx->in_block_size % ctx->fb_block_size != 0)
    {
      multilog (log, LOG_ERR, "bfdsp_init: input block size must be a multiple of the output block size\n");
      return -1;
    }
    ctx->fb_hdu = fb_hdu;
  }

  ctx->tb_block_size = 0;
  ctx->tb_hdu = 0;
  if (tb_hdu)
  {
    ctx->tb_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) tb_hdu->data_block);
    ctx->tb_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) tb_hdu->data_block);
    if (ctx->in_block_size % ctx->tb_block_size != 0)
    {
      multilog (log, LOG_ERR, "bfdsp_init: input block size must be a multiple of the output block size\n");
      return -1;
    }
    ctx->tb_hdu = tb_hdu;
  }

#ifdef USE_GPU
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
  if (fb_hdu)
  {
    if (dada_cuda_dbregister(fb_hdu) < 0)
    {
      fprintf (stderr, "failed to register fb_hdu DADA buffers as pinned memory\n");
      return -1;
    }
  }
  if (tb_hdu)
  {
    if (dada_cuda_dbregister(tb_hdu) < 0)
    {
      fprintf (stderr, "failed to register tb_hdu DADA buffers as pinned memory\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "init: initating %ld bytes of device memory for d_in\n", ctx->in_block_size);
  error = cudaMalloc( &(ctx->d_in), ctx->in_block_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "init: d_in=%p\n", ctx->d_in);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "init: could not allocate %ld bytes of device memory\n", ctx->in_block_size);
    return -1;
  }

  if (fb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: allocating %ld bytes of device memory for d_fbs\n", ctx->fb_block_size);
    error = cudaMalloc( &(ctx->d_fbs), ctx->fb_block_size);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: d_fbs=%p\n", ctx->d_fbs);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "init: could not allocate %ld bytes of device memory\n", ctx->fb_block_size);
      return -1;
    }
  }

  if (tb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: allocating %ld bytes of device memory for d_tbs\n", ctx->tb_block_size);
    error = cudaMalloc( &(ctx->d_tb), ctx->tb_block_size);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: d_tbs=%p\n", ctx->d_tb);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "init: could not allocate %ld bytes of device memory\n", ctx->tb_block_size);
      return -1;
    }
  }

#endif
  if (ctx->verbose)
     multilog (log, LOG_INFO, "init: completed\n");

  return 0;
}

// allocate observation specific RAM buffers (mostly things that depend on NANT)
int bfdsp_alloc (mopsr_bfdsp_t * ctx)
{
  multilog_t * log = ctx->log;

  unsigned iant, ibeam;

  if (ctx->fan_beams)
  {
    ctx->phasors_size = ctx->nant * ctx->nbeam * sizeof(complex float);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of pinned host memory for phasors\n", ctx->phasors_size);

#ifdef USE_GPU
    cudaError_t error = cudaMallocHost( (void **) &(ctx->h_phasors), ctx->phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of pinned host memory\n", ctx->phasors_size);
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for phasors\n", ctx->phasors_size);
    error = cudaMalloc( (void **) &(ctx->d_phasors), ctx->phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->phasors_size);
      return -1;    
    } 
  }

  if (ctx->tied_beam)
  {
    ctx->tb_phasors_size = ctx->nant * sizeof(complex float);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of pinned host memory for phasors\n", ctx->tb_phasors_size);
    cudaError_t error = cudaMallocHost( (void **) &(ctx->h_tb_phasors), ctx->tb_phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of pinned host memory\n", ctx->tb_phasors_size);
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for phasors\n", ctx->tb_phasors_size);
    error = cudaMalloc( (void **) &(ctx->d_tb_phasors), ctx->tb_phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->tb_phasors_size);
      return -1;
    }
  }

#else
  ctx->h_phasors = (float *) malloc(ctx->phasors_size);
#endif

  if (ctx->fan_beams)
  {
    // compute the phasors for each beam and antenna
    double C = 2.99792458e8;
    double dist, fraction, sin_md, geometric_delay, theta, angle_rads;
    const unsigned nbeamant = ctx->nbeam * ctx->nant;
    unsigned idx;

    //double range = (4.0 / 352) * ctx->nbeam;
    double range = 4.0;

    ctx->h_beam_offsets = (float *) malloc (ctx->nbeam * sizeof(float));

    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: computing beam offsets nbeam=%d\n", ctx->nbeam); 

    for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    {
      fraction = (double) ibeam / (double) (ctx->nbeam-1);
      // centred around delay midpoint
      angle_rads = ((fraction * range) - (range/2)) * DD2R;
      // leading edge
      //angle_rads = (fraction * range) * DD2R;
     
      ctx->h_beam_offsets[ibeam] = (float) angle_rads;
      sin_md = sin(angle_rads);

      //fprintf (stderr, "[%d] beam angle=%f\n", ibeam, ctx->h_beam_offsets[ibeam]);

      // the h_dist will be -2 * PI * FREQ * dist / C
      for (iant=0; iant<ctx->nant; iant++)
      {
        geometric_delay = (sin_md * ctx->modules[iant]->dist) / C;
        theta = -2 * M_PI * ctx->channel.cfreq * 1000000 * geometric_delay;

        idx = ibeam * ctx->nant + iant;
        ctx->h_phasors[idx]          = (float) cos(theta);
        ctx->h_phasors[idx+nbeamant] = (float) sin(theta);

        //fprintf (stderr, "[%d][%d] freq=%lf delay=%le angle=%lf degrees, theta=%lf (%e, i%e)\n", ibeam, iant, ctx->channel.cfreq, geometric_delay, angle_rads / DD2R, theta, ctx->h_phasors[idx], ctx->h_phasors[idx+nbeamant]);
      }
    }
  }

  if (ctx->tied_beam)
  {
    float angle = 0.0;
    for (iant=0; iant<ctx->nant; iant++)
    {
      ctx->h_tb_phasors[iant] = cosf(angle) + sinf(angle) * I;
    }
  }

#ifdef USE_GPU
 
  if (ctx->fan_beams)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "alloc: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", ctx->phasors_size, ctx->d_phasors, ctx->h_phasors);
    cudaError_t error = cudaMemcpyAsync ((void *) ctx->d_phasors, (void *) ctx->h_phasors, ctx->phasors_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                ctx->d_phasors, ctx->h_phasors);
      return -1;
    }
  }

  if (ctx->tied_beam)
  {

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "alloc: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", ctx->tb_phasors_size, ctx->d_tb_phasors, ctx->h_tb_phasors);
    cudaError_t error = cudaMemcpyAsync ((void *) ctx->d_tb_phasors, (void *) ctx->h_tb_phasors, ctx->tb_phasors_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                ctx->d_tb_phasors, ctx->h_tb_phasors);
      return -1;
    }
  }
 
#endif

#ifdef USE_GPU
  cudaStreamSynchronize(ctx->stream);
#endif

  return 0;
}

// de-allocate the observation specific memory
int bfdsp_dealloc (mopsr_bfdsp_t * ctx)
{
#ifdef USE_GPU
  // ensure no operations are pending
  cudaStreamSynchronize(ctx->stream);
#endif
/*
  if (ctx->h_ant_factors)
    cudaFreeHost(ctx->h_ant_factors);
  ctx->h_ant_factors = 0;

  if (ctx->h_sin_thetas)
    cudaFreeHost (ctx->h_sin_thetas);
  ctx->h_sin_thetas = 0;
*/


#ifdef USE_GPU
  if (ctx->fan_beams && ctx->h_phasors)
    cudaFreeHost (ctx->h_phasors);
  ctx->h_phasors = 0;

  if (ctx->tied_beam && ctx->h_tb_phasors)
    cudaFreeHost (ctx->h_tb_phasors);
  ctx->h_tb_phasors = 0;

  if (ctx->d_in)
    cudaFree (ctx->d_in);
  ctx->d_in = 0;

  if (ctx->fan_beams && ctx->d_fbs)
    cudaFree (ctx->d_fbs);
   ctx->d_fbs = 0;

  if (ctx->fan_beams && ctx->d_tb)
    cudaFree (ctx->d_tb);
   ctx->d_tb = 0;

  if (ctx->fan_beams && ctx->d_phasors)
    cudaFree (ctx->d_phasors);
  ctx->d_phasors = 0;

  if (ctx->tied_beam && ctx->d_tb_phasors)
    cudaFree (ctx->d_tb_phasors);
  ctx->d_tb_phasors = 0;

#else
  if (ctx->h_phasors)
    free (ctx->h_phasors);
  ctx->h_phasors = 0;

#endif

  return 0;
}

// determine application memory 
int bfdsp_destroy (mopsr_bfdsp_t * ctx, dada_hdu_t * in_hdu)
{
#ifdef USE_GPU
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "dada_cuda_dbunregister (in_hdu)\n");
  if (dada_cuda_dbunregister (in_hdu) < 0)
  {
    multilog (ctx->log, LOG_ERR, "failed to unregister input DADA buffers\n");
    return -1;
  }

  if (ctx->fan_beams)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "dada_cuda_dbunregister (fb_hdu)\n");
    if (dada_cuda_dbunregister (ctx->fb_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister Fan Beam DADA buffers\n");
      return -1;
    }
  }

  if (ctx->tied_beam)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "dada_cuda_dbunregister (tb_hdu)\n");
    if (dada_cuda_dbunregister (ctx->tb_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister Tied Beam DADA buffers\n");
      return -1;
    }
  } 
#endif
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
    multilog (log, LOG_INFO, "open: locking write on output HDUs\n");
  if (ctx->fan_beams)
  {
    if (dada_hdu_lock_write (ctx->fb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "open: could not lock write on FB HDU\n");
      return -1;
    }
  }
  if (ctx->tied_beam)
  {
    if (dada_hdu_lock_write (ctx->tb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "open: could not lock write on TB HDU\n");
      return -1;
    }
  }


  // initialize
  ctx->tb_block_open = 0;
  ctx->fb_block_open = 0;
  ctx->bytes_read = 0;
  ctx->bytes_written = 0;
  ctx->tb_block = 0;
  ctx->fb_block = 0;
  ctx->first_time = 1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: extracting params from header\n");

  // read NANT from incoming header
  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NANT=%d\n", ctx->nant);

  // read NDIM from incoming header (should be 2)
  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }

  if (ctx->nbeam == -1)
    ctx->nbeam = ctx->nant;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NBEAM=%d\n", ctx->nbeam);

  unsigned fb_divisor = 1;
  unsigned tb_divisor = 1;

  // I/O Rate change for this operation is as follows:
  //   8bits -> 32bits,        * 4
  //   1 chan -> 4 chan        * 4
  //   1 beam -> ? beams       * ?
  //   ? ants -> 1 ant         / ?
  //   2dim -> 1dim,           / 2
  //   1.28us -> 327.68 us     / 256
  //   =============================
  //   Factor                  / 64 [ 17301504 / 64 = 270336]

  // using the x1024 kernel
  unsigned nbit_n = 4;
  unsigned nchan_n = 4;
  unsigned ndim_d  = 2;
  unsigned nsamp_d = 256;

  // using the 64_blk kernel
  //nbit_n = 4;
  //nchan_n = 1;
  //ndim_d  = 2;
  //nsamp_d = 64;

  // using the 2048_blk kernel
  nbit_n = 4;
  nchan_n = 1;
  ndim_d  = 2;
  nsamp_d = 512;

  if (ctx->fan_beams)
  {
    unsigned numerator   = (nbit_n * nchan_n * ctx->nbeam);
    unsigned denominator = (ndim_d * nsamp_d * ctx->nant);

    if (denominator % numerator != 0)
    {
      multilog (log, LOG_ERR, "open: requested nbeam did not result in integer denominator change in data rate\n");
      return -1;
    } 
    fb_divisor = denominator / numerator;
  }

  if (ctx->tied_beam)
    tb_divisor = ctx->nant / nbit_n;

  // calculate new TSAMP
  double in_tsamp, fb_tsamp; 
  if (ascii_header_get (client->header, "TSAMP", "%lf", &in_tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read TSAMP from header\n");
    return -1;
  }
  fb_tsamp = in_tsamp * nsamp_d;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: TSAMP in=%lf out=%lf\n", in_tsamp, fb_tsamp);

  // calculate new BYTES_PER_SECOND
  uint64_t fb_bytes_per_second, tb_bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BYTES_PER_SECOND from header\n");
    return -1;
  }
  fb_bytes_per_second = ctx->bytes_per_second / fb_divisor;
  tb_bytes_per_second = ctx->bytes_per_second / tb_divisor;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: BYTES_PER_SECOND in=%"PRIu64" fb=%"PRIu64" tb=%"PRIu64"\n", 
              ctx->bytes_per_second, fb_bytes_per_second, tb_bytes_per_second);

  // calculate new OBS_OFFSET
  uint64_t in_obs_offset, fb_obs_offset, tb_obs_offset;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &in_obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read OBS_OFFSET from header\n");
    return -1;
  }
  fb_obs_offset = in_obs_offset / fb_divisor;
  fb_obs_offset = in_obs_offset / tb_divisor;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: OBS_OFFSET in=%"PRIu64" fb=%"PRIu64" tb=%"PRIu64"\n", 
              in_obs_offset, fb_obs_offset, tb_obs_offset);

  int64_t in_file_size = -1;
  int64_t fb_file_size = -1;
  int64_t tb_file_size = -1;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIi64"", &in_file_size) == 1)
  {
    fb_file_size = in_file_size / fb_divisor; 
    tb_file_size = in_file_size / tb_divisor; 
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: FILE_SIZE in=%"PRIi64" fb=%"PRIi64" tb=%"PRIi64"\n", 
              in_file_size, fb_file_size, tb_file_size);

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
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: UT1_OFFSET=%f\n", ctx->ut1_offset);
  } 
  else
  {
    ctx->ut1_offset = -0.272;
    multilog (log, LOG_INFO, "open: hard coding UT1_OFFSET=%f\n", ctx->ut1_offset);
  }

  char tmp[32];

  if (ascii_header_get (client->header, "OBSERVING_TYPE", "%s", tmp) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read OBSERVING_TYPE in header\n");
    return -1;
  }

  // assume that the delay engine will be steering the boresight beam to
  // the specified RA/DEC
  ctx->steer_tb = 0;

  // if the delay engine has been asked to apply no delays, then we must
  // re-steer the primary beam boresight to the specified RA/DEC
  if (strcmp(tmp, "STATIONARY") == 0)
    ctx->steer_tb = 1;

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

  struct tm * utc = gmtime (&(ctx->utc_start));

  cal_app_pos_iau (ctx->source.raj, ctx->source.decj, utc, &(ctx->source.ra_curr), &(ctx->source.dec_curr));

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

  // extract the module identifiers
  ctx->modules = (mopsr_module_t **) malloc (sizeof(mopsr_module_t *) * ctx->nant);
  char module_list[3072];
  if (ascii_header_get (client->header, "ANTENNAE", "%s", module_list) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read ANTENNAE from header\n");
    return -1;
  }

  unsigned iant, imod;
  for (iant=0; iant<ctx->nant; iant++)
    ctx->modules[iant] = 0;

  const char *sep = ",";
  char * saveptr;
  char * str = strtok_r(module_list, sep, &saveptr);
  iant=0;
  while (str && iant<ctx->nant)
  {
    for (imod=0; imod<ctx->nmodules; imod++)
    {
      if (strcmp(str, ctx->all_modules[imod].name) == 0)
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "open: matched iant=%d to imod=%d [%s]\n", iant, imod, str);
        ctx->modules[iant] = &(ctx->all_modules[imod]);
      }
    }
    if (ctx->modules[iant] == 0)
    {
      multilog (log, LOG_ERR, "open: could not find a module that matched %s\n", str);
      return -1;
    }
    iant++;
    str = strtok_r(NULL, sep, &saveptr);
  }

  // allocate the observation specific memory
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: allocating observation specific memory\n");
  if (bfdsp_alloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "open: could not alloc obs specific memory\n");
    return -1;
  }

  // remove the antenna from the outgoing header as it is no longer relevant
  if (ascii_header_del (client->header, "ANTENNAE") < 0)
  {
    multilog (log, LOG_ERR, "open: could not delete ANTENNAE in outgoing header\n");
    return -1;
  }

  if (ascii_header_set (client->header, "NANT", "%d", 1) < 0)
  {
    multilog (log, LOG_ERR, "open: could not set NANT=1 in outgoing header\n");
    return -1;
  }

  if (ascii_header_set (client->header, "NBIT", "%d", 32) < 0)
  {     
    multilog (log, LOG_ERR, "open: could not set NBIT=32 in outgoing header\n");
    return -1;    
  }                 

  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  ctx->bytes_read += in_obs_offset;

  if (ctx->fb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: getting next free FB header buffer\n");
    char * header = ipcbuf_get_next_write (ctx->fb_hdu->header_block);
    if (!header)
    {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    // copy the header from the in to the out
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: copying header from input to output\n");
    memcpy (header, client->header, header_size);

    // double check that the fb_block_size is fb_divisor times smaller than the input block size
    if (ctx->in_block_size / ctx->fb_block_size != fb_divisor)
    {
      multilog (log, LOG_ERR, "open: in/out block size factor did not match fb_divisor\n");
      return -1;
    }

    if (ascii_header_set (header, "STATE", "%s", "Intensity") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set STATE=Intensity in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, ctx->fb_block_size) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", ctx->fb_block_size);
      return -1;
    }


    if (ascii_header_set (header, "ORDER", "%s", "ST") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=ST in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "NDIM", "%d", 1) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NDIM=1 in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "NBEAM", "%d", ctx->nbeam) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NBEAM=%d in outgoing header\n", ctx->nbeam);
      return -1;    
    }

    if (ascii_header_set (header, "NCHAN", "%d", nchan_n) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NCHAN=%d in outgoing header\n", nchan_n);
      return -1;
    }

    if (ascii_header_set (header, "TSAMP", "%lf", fb_tsamp) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set TSAMP=%lf in outgoing header\n", fb_tsamp);
      return -1;
    }
  
    if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, fb_obs_offset) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set OBS_OFFSET=%"PRIu64" in outgoing header\n", fb_obs_offset);
      return -1;
    }

    if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, fb_bytes_per_second) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BYTES_PER_SECOND=%"PRIu64" in outgoing header\n", fb_bytes_per_second);
      return -1;
    }

    if ((fb_file_size > 0) && (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, fb_file_size) < 0))
    {
      multilog (log, LOG_ERR, "open: could not set FILE_SIZE=%"PRIu64" in outgoing header\n", fb_file_size);
      return -1;
    }

    if (ascii_header_set (header, "NBEAM", "%d", ctx->nbeam) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NBEAM=%d in outgoing header\n", ctx->nbeam);
      return -1;
    }

    int ibeam;

    // md offets will be [-0.0000,] 7 or 8 chars long
    char strval[9];
    char * mdlist = (char *) malloc(sizeof(char) * 7 * ctx->nbeam);
    mdlist[0] = '\0';
    for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    {
      //sprintf(strval, "%4.3f", ctx->h_beam_offsets[ibeam]);
      sprintf(strval, "0");
      strcat (mdlist, strval);
      if (ibeam < ctx->nbeam-1)
        strcat (mdlist,",");
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: mdlist=%d , len=%d header_size=%d\n", strlen(mdlist), strlen(header), header_size);

    if (ascii_header_set (header, "BEAM_MD_OFFSETS", "%s", mdlist) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BEAM_MD_OFFSETS=%s in outgoing header\n", ibeam, mdlist);
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: marking FB header filled, len=%d header_size=%d\n", strlen(header), header_size);


    // mark the outgoing header as filled
    if (ipcbuf_mark_filled (ctx->fb_hdu->header_block, header_size) < 0) 
    {
      multilog (log, LOG_ERR, "open: could not mark header_block filled\n");
      return -1;
    }
  }

  if (ctx->tb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: getting next free TB header buffer\n");
    char * header = ipcbuf_get_next_write (ctx->tb_hdu->header_block);
    if (!header)
    {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    // copy the header from the in to the out
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: copying header from input to output\n");
    memcpy (header, client->header, header_size);

    // double check that the tb_block_size is divisor times smaller than the input block size
    if (ctx->in_block_size / ctx->tb_block_size != tb_divisor)
    {
      multilog (log, LOG_ERR, "open: in/out block size factor did not match divisor\n");
      return -1;
    }

    if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, ctx->tb_block_size) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", ctx->tb_block_size);
      return -1;
    }

    if (ascii_header_set (header, "ORDER", "%s", "T") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=T in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "NBEAM", "%d", 1) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NBEAM=%d in outgoing header\n", 1);
      return -1;    
    }

    if (ascii_header_set (header, "NCHAN", "%d", nchan_n) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NCHAN=%d in outgoing header\n", nchan_n);
      return -1;
    }

    if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, tb_obs_offset) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set OBS_OFFSET=%"PRIu64" in outgoing header\n", tb_obs_offset);
      return -1;
    }

    if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, tb_bytes_per_second) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BYTES_PER_SECOND=%"PRIu64" in outgoing header\n", tb_bytes_per_second);
      return -1;
    }

    if ((tb_file_size > 0) && (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, tb_file_size) < 0))
    {
      multilog (log, LOG_ERR, "open: could not set FILE_SIZE=%"PRIu64" in outgoing header\n", tb_file_size);
      return -1;
    }


    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: marking output header filled\n");

    // mark the outgoing header as filled
    if (ipcbuf_mark_filled (ctx->tb_hdu->header_block, header_size) < 0) 
    {
      multilog (log, LOG_ERR, "open: could not mark header_block filled\n");
      return -1;
    }
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

  if (ctx->fb_block_open)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: ipcio_close_block_write bytes_written=%"PRIu64"\n", ctx->bytes_written);
    if (ipcio_close_block_write (ctx->fb_hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->fb_block_open = 0;
    ctx->bytes_written = 0;
  }

  if (bfdsp_dealloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "close: bfdsp_dealloc failed\n");
  } 

  if (ctx->fan_beams)
  {
    if (dada_hdu_unlock_write (ctx->fb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "close: cannot unlock FB HDU\n");
      return -1;
    }
  }
  if (ctx->tied_beam)
  {
    if (dada_hdu_unlock_write (ctx->tb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "close: cannot unlock TB HDU\n");
      return -1;
    }

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

#ifdef USE_GPU 
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

  // form a tied array beam if requested
  if (ctx->tied_beam)
  {
    // update phasors for the tied array beam
    bfdsp_update_tb (ctx);

    mopsr_tie_beam (ctx->stream, ctx->d_in, ctx->d_tb, ctx->d_tb_phasors, ctx->in_block_size, ctx->nant);

    if (!ctx->tb_block_open)
    {
      ctx->tb_block = ipcio_open_block_write (ctx->tb_hdu->data_block, &out_block_id);
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: opened TB output block %"PRIu64" %p\n", out_block_id, (void *) ctx->tb_block);
      ctx->tb_block_open = 1;
    }

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                (void *) ctx->tb_block, ctx->d_tb, ctx->tb_block_size);
    error = cudaMemcpyAsync ( (void *) ctx->tb_block, ctx->d_tb, ctx->tb_block_size,
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      ctx->internal_error = 1;
      return -1;
    }
  }

  if (ctx->fan_beams)
  {
    // form the tiled, detected and integrated tied array beams
    mopsr_tile_beams_precomp (ctx->stream, ctx->d_in, ctx->d_fbs, ctx->d_phasors, ctx->in_block_size, ctx->nbeam, ctx->nant, ctx->tdec);
#endif

    // copy back to output buffer
    if (!ctx->fb_block_open)
    {
      ctx->fb_block = ipcio_open_block_write (ctx->fb_hdu->data_block, &out_block_id);
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: opened output block %"PRIu64" %p\n", out_block_id, (void *) ctx->fb_block);
      ctx->fb_block_open = 1;
    }

#ifndef USE_GPU
    mopsr_tile_beams_cpu (buffer, ctx->fb_block, ctx->h_phasors, ctx->in_block_size, ctx->nbeam, ctx->nant, 512);
#else

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                (void *) ctx->fb_block, ctx->d_fbs, ctx->fb_block_size);
    error = cudaMemcpyAsync ( (void *) ctx->fb_block, ctx->d_fbs, ctx->fb_block_size, 
                              cudaMemcpyDeviceToHost, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      ctx->internal_error = 1;
      return -1;
    }
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: cudaStreamSynchronize()\n");
  cudaStreamSynchronize(ctx->stream);
#endif

  if (ctx->fb_block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: closing FB data block for %"PRIu64" bytes\n", ctx->fb_block_size);
    ipcio_close_block_write (ctx->fb_hdu->data_block, ctx->fb_block_size);
    ctx->fb_block_open = 0;
  }

  if (ctx->tb_block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: closing TB data block for %"PRIu64" bytes\n", ctx->tb_block_size);
    ipcio_close_block_write (ctx->tb_hdu->data_block, ctx->tb_block_size);
    ctx->tb_block_open = 0;
  }

  ctx->bytes_read += bytes;
  return (int64_t) bytes;
}

// simpler CPU version
int mopsr_tile_beams_cpu (void * h_in, void * h_out, void * h_phasors, uint64_t nbytes, unsigned nbeam, unsigned nant, unsigned tdec)
{
  unsigned ndim = 2;

  // data is ordered in ST order
  unsigned nsamp = (unsigned) (nbytes / (nant * ndim));

  float * phasors = (float *) h_phasors;

  int16_t * in16 = (int16_t *) h_in;
  float * ou     = (float *) h_out;

  int16_t val16;
  int8_t * val8 = (int8_t *) &val16;

  const unsigned nbeamant = nbeam * nant;
  const unsigned ant_stride = nsamp;
  complex float val, beam_sum, phasor, steered;
  float beam_power;

  // intergrate samples together
  const unsigned ndat = tdec;
  unsigned nchunk = nsamp / tdec;

  complex float beam_phasors[nant];

  //fprintf (stderr, "nsamp=%u, nchunk=%u\n", nsamp, nchunk);

  unsigned ibeam, ichunk, idat, isamp, iant, idx;
  for (ibeam=0; ibeam<nbeam; ibeam++)
  {
    fprintf (stderr, "ibeam=%i\n", ibeam);

    // compute phasors for all ant for this beam
    for (iant=0; iant<nant; iant++)
    {
      idx = (ibeam * nant) + iant;
      beam_phasors[iant] = phasors[idx] + phasors[idx + nbeamant] * I;
    }

    isamp = 0;
    for (ichunk=0; ichunk<nchunk; ichunk++)
    {
      beam_power = 0;
      for (idat=0; idat<ndat; idat++)
      {
        // compute the "beam_sum" of all antena for each time sample
        beam_sum = 0 + 0 * I;
        for (iant=0; iant<nant; iant++)
        {
          // unpack this sample and antenna
          val16 = in16[iant*nsamp + isamp];
          val = ((float) val8[0]) + ((float) val8[1]) * I;

          // fused multiply and add
          beam_sum = val * beam_phasors[iant] + beam_sum;

          // the required phase rotation for this beam and antenna
          //phasor = phasors[idx] + phasors[idx + nbeamant] * I;
          //steered = val * neam_phasors[iant] + steer;
          // add the steered tied array beam to the total
          //beam_sum += steered;
          //beam_sum += val;
        }

        beam_power += (creal(beam_sum) * creal(beam_sum)) + (cimag(beam_sum) * cimag(beam_sum));
        isamp++;
      }

      // output in ST order
      ou[ibeam * nchunk + ichunk] = beam_power;
    }
  }
}

// determine if there is a pulsar to time within the primary beam
void bfdsp_update_tb (mopsr_bfdsp_t * ctx)
{
  // if the tied array beam must be steered to the source RA/DEC from MD=0
  if (ctx->steer_tb)
  {
    // determine current offset between PSR and MD=0 in radians
    // note that UT1 offset was already accounted for in the AQDSP engine
    const double mid_byte = (double) ctx->bytes_read + (ctx->in_block_size / 2);
    const double mid_time = (double) ctx->utc_start + (mid_byte / (double) ctx->bytes_per_second);

    struct timeval timestamp;
    timestamp.tv_sec = (long) floor (mid_time);
    timestamp.tv_usec = (long) floor ((mid_time - (double) timestamp.tv_sec) * 1e6);

    double jer_delay = calc_jer_delay (ctx->source.ra_curr, ctx->source.dec_curr, timestamp);

    unsigned iant;
    const double C = 2.99792458e8;
    for (iant=0; iant<ctx->nant; iant++)
    {
      double geometric_delay = (jer_delay * ctx->modules[iant]->dist) / C;
      double theta = -2 * M_PI * ctx->channel.cfreq * 1000000 * geometric_delay;
      ctx->h_tb_phasors[iant] = (float) cos(theta) + (float) sin(theta) * I;
    }

    // copy these to device memory
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "update_tb: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", 
                ctx->tb_phasors_size, ctx->d_tb_phasors, ctx->h_tb_phasors);
    cudaError_t error = cudaMemcpyAsync ((void *) ctx->d_tb_phasors, (void *) ctx->h_tb_phasors, 
                             ctx->tb_phasors_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (ctx->log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                ctx->d_tb_phasors, ctx->h_tb_phasors);
      return;
    }
  }
}
