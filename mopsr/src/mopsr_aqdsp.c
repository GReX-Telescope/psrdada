/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "mopsr_cuda.h"
#include "mopsr_delays.h"
#include "mopsr_aqdsp.h"

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
	fprintf(stdout, "mopsr_aqdsp [options] inkey outkey bays_file modules_file signal_path_file\n"
    "\n"
    "  Apply DSP operations to MOPSR AQ pipeine:\n"
    "    TFS -> FST transpose\n"
    "    integer sample delay correction *\n"
    "    fractional sample delay correction *\n"
    "    oversampling filterbank rephasing correction + rescaling\n"
    "    FST to STF transpose *\n"
    "    FST to TF coherrent ant sum + transpose *\n"
    "\n"
    "  inkey             PSRDADA hexideciaml key for input data stream\n"
    "  outkey            PSRDADA hexideciaml key for output data stream\n"
    "  bays_file         file containing all bay definitions\n"
    "  modules_file      file containing all module definitions\n"
    "  signal_path_file  file containing all signal path connections\n"
    "\n"
    "  -a                coherrently add antenna, output order TF\n"
    "  -d <id>           use GPU device with id [default 0]\n"
    "  -g                do not apply geometric delays [default apply them]\n"
    "  -l utc            lock the geometric delay to the specific utc, no geometric correction\n"
    "  -n <taps>         number of FIR taps to use [default %d]\n"
    "  -p <id>           override the PFB_ID from the header (e.g. EG03)\n"
    "  -r                do not apply delay corrections\n"
    "  -o                transpose output to STF [default FST]\n"
    "  -s                process single observation only\n"
    "  -t                observing is tracking [default delay steering]\n"
    "  -v                verbose output\n"
    "  *                 optional\n"
    , MOPSR_AQDSP_DEFAULT_NTAPS
  );
}

int main(int argc, char** argv) 
{
  // aqdsp contextual struct
  mopsr_aqdsp_t ctx;

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
  ctx.internal_error = 0;
  ctx.verbose = 0;
  ctx.device = 0;
  ctx.ntaps = MOPSR_AQDSP_DEFAULT_NTAPS;
  ctx.correct_delays = 1;
  ctx.output_stf = 0;
  ctx.sum_ant = 0;
  sprintf (ctx.pfb_id, "XXXX");

  ctx.lock_utc_flag = 0;
  ctx.obs_tracking = 0;
  ctx.geometric_delays = 1;

  while ((arg = getopt(argc, argv, "ad:ghl:n:op:rstv")) != -1) 
  {
    switch (arg)  
    {
      case 'a':
        ctx.sum_ant = 1;
        break;

      case 'd':
        ctx.device = atoi(optarg);
        break;

      case 'g':
        ctx.geometric_delays = 0;
        break;
      
      case 'h':
        usage ();
        return 0;

      case 'l':
        ctx.lock_utc_flag = 1;
        ctx.lock_utc_time = str2utctime (optarg);
        if (ctx.lock_utc_time == (time_t)-1)
        {
          fprintf(stderr, "main: could not parse lock utc time from '%s'\n", optarg);
          return (EXIT_FAILURE);
        }
        break;

      case 'n':
        ctx.ntaps = atoi(optarg);
        break;

      case 'o':
        ctx.output_stf = 1;
        break;

      case 'p':
        strncpy (ctx.pfb_id, optarg, 4);
        break;

      case 'r':
        ctx.correct_delays = 0;
        break;

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
  if (argc-optind != 5)
  {
    fprintf(stderr, "ERROR: 5 arguments are required\n");
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

  log = multilog_open ("mopsr_aqdsp", 0); 

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

  if (aqdsp_init (&ctx, in_hdu, out_hdu, bays_file, modules_file, signal_path_file) < 0)
  {
    fprintf (stderr, "ERROR: failed to initalise data structures\n");
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block        = in_hdu->data_block;
  client->header_block      = in_hdu->header_block;

  client->open_function     = aqdsp_open;
  client->io_function       = aqdsp_io;
  client->io_block_function = aqdsp_io_block;
  client->close_function    = aqdsp_close;
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

  if (aqdsp_destroy (&ctx, in_hdu, out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to release resources\n");
  }

  if (dada_hdu_disconnect (in_hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

/*! Perform initialization */
int aqdsp_init ( mopsr_aqdsp_t* ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu, char * bays_file, char * modules_file, char * signal_paths_file)
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

  ctx->pfbs = read_signal_paths_file (signal_paths_file, &(ctx->npfbs));
  if (!ctx->pfbs)
  {
    multilog (log, LOG_ERR, "init: failed to read signal parths file [%s]\n", signal_paths_file);
    return -1;
  }

  // prepare the filter. for saftey and to get the correct response for a
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

  // select the gpu device
  int n_devices = dada_cuda_get_device_count();
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

  if ((ctx->device < 0) || (ctx->device >= n_devices))
  {
    multilog (log, LOG_ERR, "aqdsp_init: no CUDA devices available [%d]\n",
              n_devices);
    return -1;
  }

  if (dada_cuda_select_device (ctx->device) < 0)
  {
    multilog (log, LOG_ERR, "aqdsp_init: could not select requested device [%d]\n",
              ctx->device);
    return -1;
  }

  char * device_name = dada_cuda_get_device_name (ctx->device);
  if (!device_name)
  {
    multilog (log, LOG_ERR, "aqdsp_init: could not get CUDA device name\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Using device %d : %s\n", ctx->device, device_name);

  free(device_name);

  // setup the cuda stream for operations
  cudaError_t error = cudaStreamCreate(&(ctx->stream));
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "aqdsp_init: could not create CUDA stream\n");
    return -1;
  }

  // input block must be a multiple of the output block in bytes
  ctx->block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  if (ctx->block_size % ipcbuf_get_bufsz ((ipcbuf_t *) out_hdu->data_block) != 0)
  {
    multilog (log, LOG_ERR, "aqdsp_init: input must be a multiple of outout HDU block size: input [%"PRIu64"], ouput [%"PRIu64"]\n",
              ctx->block_size, ipcbuf_get_bufsz ((ipcbuf_t *) out_hdu->data_block));
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

  ctx->h_delays = ctx->d_delays = 0;
  ctx->h_fringes = 0;
  ctx->h_ant_scales = 0;
  ctx->d_in = ctx->d_out = 0;

  if (ctx->verbose)
     multilog (log, LOG_INFO, "init: completed\n");

  return 0;
}

// allocate observation specific RAM buffers (mostly things that depend on NCHAN, NANT)
int aqdsp_alloc (mopsr_aqdsp_t * ctx)
{
  multilog_t * log = ctx->log;

  unsigned nchanant = ctx->nchan * ctx->nant;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: nchan=%u nant=%u nchanant=%u\n", ctx->nchan, ctx->nant, nchanant);

  // allocate CPU memory for each delay 
  size_t delays_size = sizeof(mopsr_delay_t *) * ctx->nant;
  ctx->delays = (mopsr_delay_t **) malloc (delays_size);
  if (!ctx->delays)
  {
    multilog (log, LOG_ERR, "alloc: failed to malloc %ld bytes\n", delays_size);
    return EXIT_FAILURE;
  }

  unsigned iant;
  for (iant=0; iant<ctx->nant; iant++)
    ctx->delays[iant] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * ctx->nchan);

  // N.B. dont need to allocate RAM for each module as they will simply point to 
  // ctx->all_modules
  size_t modules_ptr_size = sizeof(mopsr_module_t) * ctx->nant;
  ctx->modules = (mopsr_module_t *) malloc(modules_ptr_size);
  if (!ctx->modules)
  {
    multilog (log, LOG_ERR, "alloc: failed to malloc %ld bytes\n", modules_ptr_size);
    return EXIT_FAILURE;
  }

  // use the PFB from header (or command line) to configure the modules we will be using
  unsigned ipfb;
  unsigned pfb_id = -1;
  for (ipfb=0; ipfb<MOPSR_MAX_PFBS; ipfb++)
  {
    if (strcmp(ctx->pfbs[ipfb].id, ctx->pfb_id) == 0)
      pfb_id = ipfb; 
  }
  if (pfb_id == -1)
  {
    multilog (log, LOG_ERR, "alloc: failed to find pfb_id=%s in signal path configuration file\n",  ctx->pfb_id);
    return -1;
  }

  iant = 0;
  unsigned ipfb_mod, imod;

  if (ctx->verbose > 1)
  {
    multilog (log, LOG_INFO, "alloc: ctx->pfbs[%d].id=%s\n", pfb_id, ctx->pfbs[pfb_id].id);
    for (ipfb_mod=0; ipfb_mod < MOPSR_MAX_MODULES_PER_PFB; ipfb_mod++)
      multilog (log, LOG_INFO, "alloc: ctx->pfbs[%d].modules[%d]=%s\n", pfb_id, ipfb_mod, ctx->pfbs[pfb_id].modules[ipfb_mod]);
  }

  unsigned ants_found = 0;
  for (iant=0; iant<ctx->nant; iant++)
  {
    for (imod=0; imod<ctx->nmodules; imod++)
    {
      if (strcmp(ctx->pfbs[pfb_id].modules[ctx->ant_ids[iant]], ctx->all_modules[imod].name) == 0)
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "alloc: ctx->modules[%d] == ctx->all_modules[%d] = %s\n", ctx->ant_ids[iant], imod, ctx->all_modules[imod].name);
        ctx->modules[iant] = ctx->all_modules[imod];
        ants_found++;
      }
    }
  }
        
/*
  for (ipfb_mod=0; ipfb_mod < MOPSR_MAX_MODULES_PER_PFB; ipfb_mod++)
  {
    if (strcmp(ctx->pfbs[pfb_id].modules[ipfb_mod], "-") == 0)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "alloc: skipping ctx->pfbs[%d].modules[%d] == %s\n", ipfb, ipfb_mod, ctx->pfbs[pfb_id].modules[ipfb_mod]);
      // skip as it is not connected
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "alloc: searching for ctx->pfbs[%d].modules[%d]=%s\n", pfb_id, ipfb_mod, ctx->pfbs[pfb_id].modules[ipfb_mod]);
      for (imod=0; imod<ctx->nmodules; imod++)
      {
        if (strcmp(ctx->pfbs[pfb_id].modules[ipfb_mod],ctx->all_modules[imod].name) == 0)
        {
          if (iant < ctx->nant)
          {
            if (ctx->verbose)
              multilog (log, LOG_INFO, "alloc: ctx->modules[%d] = ctx->all_modules[%d] = %s\n", iant, imod, ctx->all_modules[imod].name);
            ctx->modules[iant] = ctx->all_modules[imod];
            iant++;
          }
        }
      }
    }
  }
  */

  if (ants_found != ctx->nant)
  {
    multilog (log, LOG_ERR, "alloc: failed to identify the specific modules for this PFB [only %d of %d]\n", ants_found, ctx->nant);
    return -1;
  }

  if (ctx->verbose)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      //multilog (log, LOG_INFO, "alloc: [%d] module.name=%s dist=%lf fixed_delay=%le scale=%lf bay_idx=%u\n", iant, ctx->modules[iant].name, ctx->modules[iant].dist, ctx->modules[iant].fixed_delay, ctx->modules[iant].scale, ctx->modules[iant].bay_idx);
    }
  }

  // allocate CPU memory for each channel definition
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: setting up channels struct\n");
  unsigned ichan;
  ctx->channels = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t) * ctx->nchan);
  for (ichan=0; ichan < ctx->nchan; ichan++)
  {
    ctx->channels[ichan].number = ctx->chan_offset + ichan;
    ctx->channels[ichan].bw     = ctx->channel_bw;
    ctx->channels[ichan].cfreq  = ctx->base_freq + (ctx->channel_bw/2) + (ichan * ctx->channel_bw);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: channels[%d] number=%d bw=%f cfreq=%f\n", ichan, ctx->channels[ichan].number, 
                ctx->channels[ichan].bw, ctx->channels[ichan].cfreq);
  }

#ifdef _DEBUG
  multilog (log, LOG_INFO, "alloc: channels[0].number = %d\n", ctx->channels[0].number);
  multilog (log, LOG_INFO, "alloc: channels[0].bw = %f\n", ctx->channels[0].bw);
  multilog (log, LOG_INFO, "alloc: channels[0].cfreq = %f\n", ctx->channels[0].cfreq);
#endif

  // buffers for the fractions sample delays (host and device)
  ctx->delays_size = sizeof(float) * nchanant;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on GPU for delays\n", ctx->delays_size);
  cudaError_t error = cudaMalloc( &(ctx->d_delays), ctx->delays_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->delays_size);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for delays\n", ctx->delays_size);
  error = cudaMallocHost ((void **) &(ctx->h_delays), ctx->delays_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->delays_size);
    return -1;
  }

  // buffers for the fringe coefficients (host only)
  ctx->fringes_size = sizeof(float) * nchanant;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for fringes\n", ctx->fringes_size);
  error = cudaMallocHost ((void **) &(ctx->h_fringes), ctx->fringes_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->fringes_size);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for fringe_coeffs_ds\n", ctx->fringes_size);
  error = cudaMallocHost ((void **) &(ctx->h_fringe_coeffs_ds), ctx->fringes_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->fringes_size);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for delays_ds\n", ctx->fringes_size);
  error = cudaMallocHost ((void **) &(ctx->h_delays_ds), ctx->fringes_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->fringes_size);
    return -1;
  }

  // device memory for input data
  ctx->d_buffer_size   = (size_t) ctx->block_size;

#ifdef SKZAP
  size_t d_fbuf_size = (size_t) ctx->block_size * sizeof(float);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_fbuf\n", d_fbuf_size);
  error = cudaMalloc (&(ctx->d_fbuf), d_fbuf_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_fbuf=%p\n", ctx->d_fbuf);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", d_fbuf_size);
    return -1;
  }

  // setup the RNG generation states
  const size_t maxthreads = 1024;
  const unsigned nrngs = ctx->nchan * ctx->nant * maxthreads; 
  size_t d_curand_size = (size_t) (nrngs * mopsr_curandState_size());
  if (ctx->verbose)
   multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_rstates\n", d_curand_size);
  error = cudaMalloc (&(ctx->d_rstates), d_curand_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_rstates=%p\n", ctx->d_rstates);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", d_curand_size);
    return -1;
  }

  // initialise the RNGs
  //unsigned long long seed = (unsigned long long) time(0);
  //mopsr_init_rng (ctx->stream, seed, nrngs, ctx->d_rstates);

  size_t d_sigmas_size = (size_t) (ctx->nchan * ctx->nant * sizeof(float));
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_sigmas\n", d_sigmas_size);
  error = cudaMalloc (&(ctx->d_sigmas), d_sigmas_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_sigmas=%p\n", ctx->d_sigmas);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", d_sigmas_size);
    return -1;
  }
  cudaMemsetAsync(ctx->d_sigmas, 0, d_sigmas_size, ctx->stream);

#endif

  // use helper function to initialize memory for sample delay
  if (ctx->correct_delays)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: mopsr_transpose_delay_alloc()\n");
    ctx->gtx = (transpose_delay_t *) malloc (sizeof(transpose_delay_t));
    if (!ctx->gtx)
    {
      multilog (log, LOG_ERR, "alloc: could not malloc %ld bytes\n", sizeof(transpose_delay_t));
      return -1;
    }
    if (mopsr_transpose_delay_alloc (ctx->gtx, ctx->d_buffer_size, ctx->nchan, ctx->nant, ctx->ntaps) < 0)
    {
      multilog (log, LOG_ERR, "alloc: mopsr_transpose_delay_alloc failed\n");
      return -1;
    }
    cudaMemsetAsync (ctx->gtx->curr->d_buffer, 0, ctx->gtx->buffer_size, ctx->stream);
    cudaMemsetAsync (ctx->gtx->next->d_buffer, 0, ctx->gtx->buffer_size, ctx->stream);
  }

  // device memory for delayed and rephased data
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_out\n", ctx->d_buffer_size);
  error = cudaMalloc( &(ctx->d_out), ctx->d_buffer_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_out=%p\n", ctx->d_out);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->d_buffer_size);
    return -1;
  }

/*
  // cpu memory for rephasing corrections
  size_t corr_size = sizeof (float complex) * MOPSR_UNIQUE_CORRECTIONS * ctx->nchan;
  float complex * h_corr = (float complex *) malloc(corr_size);
  if (!h_corr)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of CPU memory\n", corr_size);
    return -1;
  }

  // device memory for rephasing corrections
  error = cudaMalloc((void **) &(ctx->d_corr), corr_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", corr_size);
    return -1;
  }

  // now compute the corrections based on the channel offset from header
  unsigned ipt, icorr;
  float theta;
  float ratio = 2 * M_PI * (5.0 / 32.0);
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (ipt=0; ipt < MOPSR_UNIQUE_CORRECTIONS; ipt++)
    {
      icorr = (ichan * MOPSR_UNIQUE_CORRECTIONS) + ipt;
      theta = (ctx->chan_offset + ichan) * ratio * ipt;
      h_corr[icorr] = sin (theta) - cos(theta) * I;
    }
  }

  error = cudaMemcpyAsync (ctx->d_corr, h_corr, corr_size, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }
*/

  // scaling factors for antenna
  ctx->ant_scales_size = ctx->nant * sizeof(float);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes on CPU for ant scales\n", ctx->ant_scales_size);
  error = cudaMallocHost ((void **) &(ctx->h_ant_scales), ctx->ant_scales_size);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of host memory\n", ctx->ant_scales_size);
    return -1;
  }

  // now copy scaling factors from modules
  for (iant=0; iant<ctx->nant; iant++)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: ctx->modules[%d].scale=%f\n", iant, ctx->modules[iant].scale);
    ctx->h_ant_scales[iant] = ctx->modules[iant].scale;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: ctx->h_ant_scales[%d]=%f\n", iant, ctx->h_ant_scales[iant]);
  }
 
  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "alloc: copying delay ant_scales to GPU [%ld]\n", ctx->ant_scales_size);
    mopsr_delay_copy_scales (ctx->stream, ctx->h_ant_scales, ctx->ant_scales_size);
  }

  // now copy to the symbol
  /*if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: copying rephase ant_scales to GPU [%ld]\n", ctx->ant_scales_size);
  mopsr_input_rephase_scales (ctx->stream, ctx->h_ant_scales, ctx->ant_scales_size);
  */

  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for d_in\n", ctx->d_buffer_size);
  error = cudaMalloc( &(ctx->d_in), ctx->block_size);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "alloc: d_in=%p\n", ctx->d_in);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->d_buffer_size);
    return -1;
  }

  cudaStreamSynchronize(ctx->stream);

/*
  free (h_corr);
*/
  return 0;
}

// de-allocate the observation specific memory
int aqdsp_dealloc (mopsr_aqdsp_t * ctx)
{
  // ensure no operations are pending
  cudaStreamSynchronize(ctx->stream);

  if (ctx->h_delays)
    cudaFreeHost(ctx->h_delays);
  ctx->h_delays = 0;

  if (ctx->d_delays)
    cudaFree (ctx->d_delays);
  ctx->d_delays = 0;

  if (ctx->h_fringes)
    cudaFreeHost(ctx->h_fringes);
  ctx->h_fringes = 0;

  if (ctx->h_delays_ds)
    cudaFreeHost(ctx->h_delays_ds);
  ctx->h_delays_ds = 0;

  if (ctx->h_fringe_coeffs_ds)
    cudaFreeHost(ctx->h_fringe_coeffs_ds);
  ctx->h_fringe_coeffs_ds = 0;

  if (ctx->correct_delays)
    mopsr_transpose_delay_dealloc (ctx->gtx);

  if (ctx->h_ant_scales)
    cudaFreeHost(ctx->h_ant_scales);
  ctx->h_ant_scales = 0;

  if (ctx->d_in)
    cudaFree (ctx->d_in);
  ctx->d_in = 0;

#ifdef SKZAP
  if (ctx->d_fbuf)
    cudaFree (ctx->d_fbuf);
  ctx->d_fbuf = 0;

  if (ctx->d_rstates)
    cudaFree (ctx->d_rstates);
  ctx->d_rstates = 0;

  if (ctx->d_sigmas)
    cudaFree (ctx->d_sigmas);
  ctx->d_sigmas = 0;
#endif

  if (ctx->d_out)
    cudaFree (ctx->d_out);
   ctx->d_out = 0;

  return 0;
}

// determine application memory 
int aqdsp_destroy (mopsr_aqdsp_t * ctx, dada_hdu_t * in_hdu, dada_hdu_t * out_hdu)
{
  if (ctx->filter);
    free (ctx->filter);
  ctx->filter = 0;

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
 

int aqdsp_open (dada_client_t* client)
{
  assert (client != 0);
  mopsr_aqdsp_t* ctx = (mopsr_aqdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "aqdsp_open()\n");

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

  // extract required metadata from header
  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NCHAN from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NCHAN=%d\n", ctx->nchan);

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NANT from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NANT=%d\n", ctx->nant);
  unsigned iant;
  char ant_title[16];
  for (iant=0; iant<ctx->nant; iant++)
  {
    sprintf (ant_title, "ANT_ID_%u", iant);
    if (ascii_header_get (client->header, ant_title, "%u", &(ctx->ant_ids[iant])) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", ctx->ant_ids[iant]);
      return -1;
    }
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
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: TSAMP=%lf\n", ctx->tsamp);

  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BYTES_PER_SECOND from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: BYTES_PER_SECOND=%"PRIu64"\n", ctx->bytes_per_second);

  if (ascii_header_get (client->header, "CHAN_OFFSET", "%u", &(ctx->chan_offset)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read CHAN_OFFSET from header\n");
    return -1;
  }

  if (strcmp(ctx->pfb_id, "XXXX") == 0)
  {
    if (ascii_header_get (client->header, "PFB_ID", "%s", &(ctx->pfb_id)) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read PFB_ID from header\n");
      return -1;
    }
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: PFB_ID=%s\n", ctx->pfb_id);

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

  float bw;
  if (ascii_header_get (client->header, "BW", "%f", &bw) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BW from header\n");
    return -1;
  }
  ctx->channel_bw = bw / ctx->nchan;

  float freq;
  if (ascii_header_get (client->header, "FREQ", "%f", &freq) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BW from header\n");
    return -1;
  }
  ctx->base_freq = freq - (bw / 2.0);

  char order[4];
  if (ascii_header_get (client->header, "ORDER", "%s", order) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read ORDER from header\n");
    return -1;
  }
  if (strcmp (order, "TFS") != 0)
  {
    multilog (log, LOG_ERR, "open: input ORDER was not TFS\n");
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  // allocate the observation specific memory
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: allocating observation specific memory\n");
  if (aqdsp_alloc (ctx) < 0)
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

  if (ctx->output_stf)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting ORDER=STF\n");
    if (ascii_header_set (header, "ORDER", "%s", "STF") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=SFT in outgoing header\n");
      return -1;
    }
  }
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting ORDER=FST\n");
    if (ascii_header_set (header, "ORDER", "%s", "FST") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=FST in outgoing header\n");
      return -1;
    }
  }

  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, ctx->block_size) < 0)
  {
    multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", ctx->block_size);
    return -1;
  }

  if (ctx->correct_delays)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting DELAY_CORRECTED=TRUE\n");
    if (ascii_header_set (header, "DELAY_CORRECTED", "%s", "TRUE") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set DELAY_CORRECTED=TRUE in outgoing header\n");
      return -1;
    }
  }

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read OBS_OFFSET from header\n");
   return -1;
  }
  ctx->bytes_read += obs_offset;

  if (ctx->sum_ant)
  {
    int new_nant = 1;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting NANT=%d\n", new_nant);
    if (ascii_header_set (header, "NANT", "%d", new_nant) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NANT=1 in outgoing header\n");
      return -1;
    }

    uint64_t new_bytes_per_second = ctx->bytes_per_second / ctx->nant;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting BYTES_PER_SECOND=%"PRIu64"\n", new_bytes_per_second);
    if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BYTES_PER_SECOND=%"PRIu64" in outgoing header\n", new_bytes_per_second);
      return -1;
    }

    uint64_t filesize;
    if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &filesize) == 1)
    {
      filesize /= ctx->nant;
      if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, filesize) < 0)
      {
        multilog (log, LOG_ERR, "open: could not set FILE_SIZE=%"PRIu64" in outgoing header\n", filesize);
        return -1;
      }
    }

    obs_offset /= ctx->nant;
    if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, obs_offset) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set OBS_OFFSET=%"PRIu64" in outgoing header\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: setting PHASE_CORRECTED=TRUE\n");
  if (ascii_header_set (header, "PHASE_CORRECTED", "%s", "TRUE") < 0)
  {
    multilog (log, LOG_ERR, "open: could not set PHASE_CORRECTED=TRUE in outgoing header\n");
    return -1;
  }

  // write the antenna to the output header
  char ant_list[4096];
  char * ant_ptr = ant_list;

  strncpy(ant_ptr, ctx->modules[0].name,5);
  ant_ptr += 5;
  for (iant=1; iant<ctx->nant; iant++)
  {
    sprintf (ant_ptr, ",%s", ctx->modules[iant].name);
    ant_ptr += 6;
  }
  if (ascii_header_set (header, "ANTENNAE", "%s", ant_list) < 0)
  {
    multilog (log, LOG_ERR, "open: could not set ANTENNAE in outgoing header\n");
    return -1;
  }

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

int aqdsp_close (dada_client_t* client, uint64_t bytes_written)
{

  assert (client != 0);
  mopsr_aqdsp_t* ctx = (mopsr_aqdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "aqdsp_close()\n");

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

  if (aqdsp_dealloc (ctx) < 0)
  {
    multilog (log, LOG_ERR, "close: aqdsp_dealloc failed\n");
  } 

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "close: cannot unlock output HDU\n");
    return -1;
  }

  return 0;
}

/*! used for transferring the header, and uses pageable memory */
int64_t aqdsp_io (dada_client_t* client, void * buffer, uint64_t bytes)
{
  assert (client != 0);
  mopsr_aqdsp_t* ctx = (mopsr_aqdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;
  multilog (log, LOG_ERR, "io: should not be called\n");

  return (int64_t) bytes;
}

/*
 * GPU engine for delaying a block of data 
 */
int64_t aqdsp_io_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_aqdsp_t* ctx = (mopsr_aqdsp_t*) client->context;
  multilog_t * log = ctx->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: buffer=%p, bytes=%"PRIu64" block_id=%"PRIu64"\n", buffer, bytes, block_id);

  cudaError_t error;
  uint64_t out_block_id;
  unsigned ichan, iant;
  uint64_t bytes_out = bytes;

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

  // rephase each coarse channel 
/*
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "io_block: mopsr_input_rephase_TFS(%ld)\n", bytes);
   mopsr_input_rephase_TFS (ctx->stream, ctx->d_in, bytes, ctx->nchan, ctx->nant, ctx->chan_offset);
*/

  char write_output = 1;
  char apply_instrumental = 1;
  char apply_geometric = ctx->geometric_delays;

  {
    // middle byte of block
    const double mid_byte = (double) ctx->bytes_read + (bytes / 2);

    // determine the timestamp corresponding to the middle byte of this block
    double obs_offset_seconds = mid_byte / (double) ctx->bytes_per_second;

    // add UT1 offset to UTC
    obs_offset_seconds += ctx->ut1_offset;

    // conert to timestamp
    struct timeval timestamp;
    timestamp.tv_sec = floor(obs_offset_seconds);
    timestamp.tv_usec = (obs_offset_seconds - (double) timestamp.tv_sec) * 1000000;
    timestamp.tv_sec += ctx->utc_start;

    if (ctx->lock_utc_flag)
    {
      timestamp.tv_sec = ctx->lock_utc_time;
      timestamp.tv_usec = 0;
    }

    // update the delays
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: utc_start=%d + obs_offset=%lf timestamp=%ld.%ld\n",
                ctx->utc_start, obs_offset_seconds, timestamp.tv_sec, timestamp.tv_usec); 
    if (calculate_delays (ctx->nbays, ctx->all_bays, ctx->nant, ctx->modules, 
                          ctx->nchan, ctx->channels, ctx->source, timestamp, 
                          ctx->delays, apply_instrumental,
                          apply_geometric, ctx->obs_tracking, ctx->tsamp) < 0)
    {
      multilog (log, LOG_ERR, "delay_block_gpu: failed to update delays\n");
      return -1;
    }

    //fprintf (stderr, "io_block: obs_offset_seconds=%lf [3][10].samples=%u fractional=%7.6lf\n", obs_offset_seconds, 
    //         ctx->delays[3][10].samples, ctx->delays[3][10].fractional);

    // layout the fractional delays in host memory
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      for (iant=0; iant < ctx->nant; iant++)
      {
        if (ctx->verbose > 1) 
          multilog (log, LOG_INFO, "io_block: iant=%d ichan=%d tot_samps=%7.6lf, "
                    "samples=%u fractional=%7.6lf fringe=%7.6lf\n", 
                    iant, ichan, ctx->delays[iant][ichan].tot_samps, 
                    ctx->delays[iant][ichan].samples, 
                    ctx->delays[iant][ichan].fractional,
                    ctx->delays[iant][ichan].fringe_coeff);

        ctx->h_delays[ichan*ctx->nant + iant]  = (float) ctx->delays[iant][ichan].fractional;
        ctx->h_fringes[ichan*ctx->nant + iant] = (float) ctx->delays[iant][ichan].fringe_coeff;
        ctx->h_delays_ds[ichan*ctx->nant + iant] = (float) ctx->delays[iant][ichan].fractional_ds;
        ctx->h_fringe_coeffs_ds[ichan*ctx->nant + iant] = (float) ctx->delays[iant][ichan].fringe_coeff_ds;
#if 0
        if ((ichan == 0) && (iant == 3))
        {
          fprintf(stderr, "toff=%lf iant=%u fringe=%f, (%f %fi)\n", obs_offset_seconds, iant, ctx->h_fringes[ichan*ctx->nant + iant], cosf(ctx->h_fringes[ichan*ctx->nant + iant]), sinf(ctx->h_fringes[ichan*ctx->nant + iant]));
        }
#endif
      }
    }
  }

  // if we are performing the delay correction
  if (ctx->correct_delays)
  {
    ///////////////////////////////////////////////////////////////////////////
    // apply the integer sample delay
    //
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: mopsr_transpose_delay()\n");
    void * d_sample_delayed = mopsr_transpose_delay (ctx->stream, ctx->gtx, ctx->d_in, bytes, ctx->delays);

    // the iteration the datablock will not be filled, so check for this
    if (d_sample_delayed)
    {
      mopsr_delay_copy_scales (ctx->stream, ctx->h_ant_scales, ctx->ant_scales_size);

      /////////////////////////////////////////////////////////////////////////
      // apply the fractional sample delay - note this includes the fringe rotation
      // operation
      //
      size_t bytes_tapped = bytes + (ctx->ndim * ctx->nchan * ctx->nant * (ctx->ntaps-1));

#ifdef SKZAP
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_delay_fractional_sk_scale(%ld)\n", bytes_tapped);
        mopsr_delay_fractional_sk_scale (ctx->stream, d_sample_delayed, ctx->d_out, ctx->d_fbuf,
                                ctx->d_rstates, ctx->d_sigmas,
                                ctx->d_delays, ctx->h_fringes, ctx->h_delays_ds,
                                ctx->h_fringe_coeffs_ds, ctx->fringes_size,
                                bytes_tapped, ctx->nchan,
                                ctx->nant, ctx->ntaps);
#else
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_delay_fractional(%ld)\n", bytes_tapped);
        mopsr_delay_fractional (ctx->stream, d_sample_delayed, ctx->d_out, 
                                ctx->d_delays, ctx->h_fringes, ctx->h_delays_ds, 
                                ctx->h_fringe_coeffs_ds, ctx->fringes_size,
                                bytes_tapped, ctx->nchan, 
                                ctx->nant, ctx->ntaps);
#endif
      write_output = 1;
    }
    else
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "io_block: skipping output due to non full block\n");
      write_output = 0;
    }

    //////////////////////////////////////////////////////////////////////////
    // copy the fractional delays to device memory
    // IMPORTANT - since we always lag behind a block with integer delays, the
    // fractional delays are copied at the end of the loop so that the fractional
    // delays from the previous iteration are applied with the corresponding 
    // integer delays from the previous block
    //
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync delays H2D %ld\n", ctx->delays_size);
    error = cudaMemcpyAsync (ctx->d_delays, ctx->h_delays, ctx->delays_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }


  }
  else
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: mopsr_input_transpose_TFS_to_FST(%ld)\n", bytes);
    mopsr_input_transpose_TFS_to_FST (ctx->stream, ctx->d_in, ctx->d_out, bytes, ctx->nchan, ctx->nant);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: mopsr_fringe_rotate (%ld)\n", bytes);
    mopsr_fringe_rotate (ctx->stream, ctx->d_out, ctx->h_fringes, ctx->fringes_size, 
                         bytes, ctx->nchan, ctx->nant);
  }


  // if we have some output in d_out
  if (write_output)
  {
#if 0
    if (first_time)
    {
      ///////////////////////////////////////////////////////////////////////////
      //
      // calculate the RMS of each antenna
      //
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_input_calc_rms(%ld)\n", bytes);
      mopsr_input_calc_rms (ctx->stream, ctx->d_out, ctx->d_rms, bytes, ctx->nchan, ctx->nant);

      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_input_combine_corr_rms()\n");
      mopsr_input_combine_corr_rms (ctx->stream, ctx->d_corr, ctx->d_rms, ctx->nchan, ctx->nant);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // rephase the time-domain data to fix the over sampling [ disabled see TFS version above]
    //
    //if (ctx->verbose > 1)
    //  multilog (log, LOG_INFO, "io_block: mopsr_input_rephase(%ld)\n", bytes);
    //mopsr_input_rephase (ctx->stream, ctx->d_out, ctx->d_corr, bytes, ctx->nchan, ctx->nant);

    void * device_output;

    ///////////////////////////////////////////////////////////////////////////
    // coherrent addition of all ant for this AQ stream
    //
    if (ctx->sum_ant)
    {
      // sum antenna from FST to FT [out of place]
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_input_sum_ant(%ld)\n", bytes);
      mopsr_input_sum_ant (ctx->stream, ctx->d_out, ctx->d_in, bytes, ctx->nchan, ctx->nant);
      bytes_out /= ctx->nant;

      // transpose from FT to TF [out of place
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: mopsr_input_transpose_FT_to_TF(%ld)\n", bytes_out);
      mopsr_input_transpose_FT_to_TF (ctx->stream, ctx->d_in, ctx->d_out, bytes_out, ctx->nchan);
      device_output = ctx->d_out;
    }
    else
    {
      if (ctx->output_stf)
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "io_block: mopsr_input_transpose_FST_to_STF(%ld)\n", bytes);
        mopsr_input_transpose_FST_to_STF (ctx->stream, ctx->d_out, ctx->d_in, bytes, ctx->nchan, ctx->nant);
        device_output = ctx->d_in;
      }
      else
      {
        device_output = ctx->d_out;
      }
    }

    // copy back to output buffer
    if (!ctx->block_open)
    {
      ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
      ctx->block_open = 1;
    }

    // bytes delayed can differ for *every* channel and antenna
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                (void *) ctx->curr_block, device_output, bytes_out);
    error = cudaMemcpyAsync ( (void *) ctx->curr_block, device_output, bytes_out, 
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
    ctx->block_open = 0;
  }

  ctx->bytes_read += bytes;
  return (int64_t) bytes;
}
