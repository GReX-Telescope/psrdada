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

//#define BAY_DIST
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
    "  -c core           bind processing to CPU core\n"
    "  -d <id>           use GPU device with id [default 0]\n"
    "  -f fb_key         output integrated, tiled beams on fb_key [SFT ordering]\n"
    "  -m mb_key         output integrated, module beams on mb_key [SFT ordering]\n"
    "  -t tb_key         output baseband tied array beams on tb_key [FT ordering]\n"
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
  dada_hdu_t* tb_hdus[MAX_TBS];
  dada_hdu_t* mb_hdu = 0;

  // DADA Primary Read Client main loop
  dada_client_t* client = 0;

  // DADA Logger
  multilog_t* log = 0;

  int arg = 0;

  unsigned quit = 0;

  key_t in_key;
  key_t fb_key = 0;
  key_t mb_key = 0;
  key_t tb_keys[MAX_TBS];

  ctx.d_in = 0;
  ctx.d_fbs = 0;
  ctx.d_tbs[MAX_TBS];
  ctx.d_mbs = 0;

  // default values
  ctx.internal_error = 0;
  ctx.verbose = 0;
  ctx.device = 0;
  ctx.nbeam = 0;
  ctx.zero_data = 0;

  ctx.fan_beams = 0;
  ctx.mod_beams = 0;
  ctx.n_tbs = 0;
  
  int core = -1;

  unsigned i;
  for (i=0; i<MAX_TBS; i++)
  {
    tb_hdus[i] = 0;
    tb_keys[i] = 0;
    ctx.d_tbs[i] = 0;
  }

  while ((arg = getopt(argc, argv, "b:c:d:f:hm:st:v0")) != -1) 
  {
    switch (arg)  
    {
      case 'b':
        ctx.nbeam = atoi(optarg);
        break;

      case 'c':
        core = atoi (optarg);
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

      case 'm':
        ctx.mod_beams = 1;
        if (sscanf (optarg, "%x", &mb_key) != 1)
        {
          fprintf (stderr, "ERROR: could not parse mb_key from %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;

      case 's':
        quit = 1;
        break;

      case 't':
        if (ctx.n_tbs >= MAX_TBS)
        {
          fprintf (stderr, "ERROR: maximum of %d Tied Beams\n", MAX_TBS);
          exit(EXIT_FAILURE);
        }
        if (sscanf (optarg, "%x", &tb_keys[ctx.n_tbs]) != 1)
        {
          fprintf (stderr, "ERROR: could not parse tb_keys from %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        ctx.n_tbs++;
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

  // now create the output HDUs
  if (ctx.mod_beams)
  {
    multilog (log, LOG_INFO, "preparing Module Beam HDU\n");
    mb_hdu = dada_hdu_create (log);
    dada_hdu_set_key(mb_hdu, mb_key);
    if (dada_hdu_connect (mb_hdu) < 0)
    {
      fprintf (stderr, "ERROR: could not connect to output HDU\n");
      return EXIT_FAILURE;
    }
  }

  for (i=0; i<ctx.n_tbs; i++)
  {
    multilog (log, LOG_INFO, "preparing Tied Beam HDU %d\n", i);
    tb_hdus[i] = dada_hdu_create (log);
    dada_hdu_set_key(tb_hdus[i], tb_keys[i]);
    if (dada_hdu_connect (tb_hdus[i]) < 0)
    {
      fprintf (stderr, "ERROR: could not connect to output HDU\n");
      return EXIT_FAILURE;
    }
  }

  ctx.log = log;

  if (bfdsp_init (&ctx, in_hdu, fb_hdu, tb_hdus, mb_hdu, bays_file, modules_file) < 0)
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
int bfdsp_init ( mopsr_bfdsp_t* ctx, dada_hdu_t * in_hdu, dada_hdu_t * fb_hdu, 
                 dada_hdu_t ** tb_hdus, dada_hdu_t * mb_hdu, 
                 char * bays_file, char * modules_file)
{
  multilog_t * log = ctx->log;
  unsigned i;

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
    //if (ctx->in_block_size % ctx->fb_block_size != 0)
    //{
    //  multilog (log, LOG_ERR, "bfdsp_init: input block size must be a multiple of the output block size\n");
    //  return -1;
    //}
    ctx->fb_hdu = fb_hdu;
  }

  for (i=0; i<ctx->n_tbs; i++)
  {
    ctx->tb_block_sizes[i] = 0;
    ctx->tb_hdus[i] = 0;
    if (tb_hdus[i])
    {
      ctx->tb_block_sizes[i] = ipcbuf_get_bufsz ((ipcbuf_t *) tb_hdus[i]->data_block);
      if (ctx->in_block_size % ctx->tb_block_sizes[i] != 0)
      {
        multilog (log, LOG_ERR, "bfdsp_init: input block size must be a multiple of the output block size\n");
        return -1;
      }
      ctx->tb_hdus[i] = tb_hdus[i];
    }
  }

  ctx->mb_block_size = 0;
  ctx->mb_hdu = 0;

  // input block must be a multiple of the output block in bytes
  ctx->in_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);
  if (mb_hdu)
  {
    ctx->mb_block_size = ipcbuf_get_bufsz ((ipcbuf_t *) mb_hdu->data_block);
    if (ctx->in_block_size % ctx->mb_block_size != 0)
    {
      multilog (log, LOG_ERR, "bfdsp_init: input block size must be a multiple of the output block size\n");
      return -1;
    }
    ctx->mb_hdu = mb_hdu;
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
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (tb_hdus[i])
    {
      if (dada_cuda_dbregister(tb_hdus[i]) < 0)
      {
        fprintf (stderr, "failed to register tb_hdus[%d] DADA buffers as pinned memory\n", i);
        return -1;
      }
    }
  }

  if (mb_hdu)
  {
    if (dada_cuda_dbregister(mb_hdu) < 0)
    {
      fprintf (stderr, "failed to register mb_hdu DADA buffers as pinned memory\n");
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

  for (i=0; i<ctx->n_tbs; i++)
  {
    if (tb_hdus[i])
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: allocating %ld bytes of device memory for d_tbs[%d]\n", i, ctx->tb_block_sizes[i]);
      error = cudaMalloc( &(ctx->d_tbs[i]), ctx->tb_block_sizes[i]);
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: d_tbs[%d]=%p\n", i, ctx->d_tbs[i]);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "init: could not allocate %ld bytes of device memory\n", ctx->tb_block_sizes[i]);
        return -1;
      }
    }
  }

  if (mb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: allocating %ld bytes of device memory for d_mbs\n", ctx->mb_block_size);
    error = cudaMalloc( &(ctx->d_mbs), ctx->mb_block_size);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: d_mbs=%p\n", ctx->d_mbs);
    if (error != cudaSuccess)
    { 
      multilog (log, LOG_ERR, "init: could not allocate %ld bytes of device memory\n", ctx->mb_block_size);
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

  unsigned iant, ibeam, ichan;

  if (ctx->fan_beams)
  {
#ifdef EIGHT_BIT_PHASORS
    ctx->phasors_size = ctx->nchan * ctx->nant * ctx->nbeam * sizeof(int8_t) * 2;
#else
    ctx->phasors_size = ctx->nchan * ctx->nant * ctx->nbeam * sizeof(complex float);
#endif
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

  ctx->tb_phasors_size = ctx->nchan * ctx->nant * sizeof(complex float);
  unsigned i;
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of pinned host memory for phasors\n", ctx->tb_phasors_size);
    cudaError_t error = cudaMallocHost( (void **) &(ctx->h_tb_phasors[i]), ctx->tb_phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of pinned host memory\n", ctx->tb_phasors_size);
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: allocating %ld bytes of device memory for phasors\n", ctx->tb_phasors_size);
    error = cudaMalloc( (void **) &(ctx->d_tb_phasors[i]),  ctx->tb_phasors_size);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "alloc: could not allocate %ld bytes of device memory\n", ctx->tb_phasors_size);
      return -1;
    }
  }

#else
#ifdef EIGHT_BIT_PHASORS
  ctx->h_phasors = (float *) malloc(ctx->phasors_size);
#else
  ctx->h_phasors = (int8_t *) malloc(ctx->phasors_size);
#endif
#endif

  if (ctx->fan_beams)
  {
    // compute the phasors for each beam and antenna
    double C = 2.99792458e8;
    double dist, fraction, sin_md, geometric_delay, theta, angle_rads;
    const unsigned nbeamant = ctx->nbeam * ctx->nant;
    unsigned idx;

    // *2 for cos and sin terms
    unsigned chan_stride = ctx->nbeam * ctx->nant * 2;
    unsigned beam_stride = ctx->nant;

    //double range = (4.0 / 352) * ctx->nbeam;
    double range = 4.0;

#ifdef BAY_DIST
    char bay[4];
#endif

    int beam_offset_idx;
    ctx->h_beam_offsets = (float *) malloc (ctx->nbeam * sizeof(float));

    if (ctx->verbose)
      multilog (log, LOG_INFO, "alloc: computing beam offsets nbeam=%d\n", ctx->nbeam); 

    // the 0'th beam is irrelevant since it is the incoherrent beam
    for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    {
      beam_offset_idx = (int) ibeam - (ctx->nbeam / 2);
      angle_rads = ctx->beam_spacing * beam_offset_idx;
      ctx->h_beam_offsets[ibeam] = (float) (angle_rads / DD2R);
      sin_md = sin(angle_rads);

      for (iant=0; iant<ctx->nant; iant++)
      {
        dist = ctx->modules[iant]->dist;

#ifdef BAY_DIST
        // extract the bay name for this module
        strncpy (bay, ctx->modules[iant]->name, 3);
        bay[3] = '\0';
        unsigned ibay;

        // if we are tracking then the ring antenna phase each of the 4
        // modules to the bay centre
        for (ibay=0; ibay<ctx->nbays; ibay++)
        {
          if (strcmp(ctx->all_bays[ibay].name, bay) == 0)
          {
            dist = ctx->all_bays[ibay].dist;
          }
        }
#endif

        geometric_delay = (sin_md * dist) / C;

        for (ichan=0; ichan<ctx->nchan; ichan++)
        {
          theta = -2 * M_PI * ctx->channels[ichan].cfreq * 1000000 * geometric_delay;

          // packed in FS order with the cos, and sin terms separate
          idx = (ichan * chan_stride) + (ibeam * ctx->nant) + iant;

#ifdef EIGHT_BIT_PHASORS
          // TODO check distribution
          ctx->h_phasors[idx]          = (int8_t) (cos(theta) * 128);
          ctx->h_phasors[idx+nbeamant] = (int8_t) (sin(theta) * 128);
#else
          ctx->h_phasors[idx]          = (float) cos(theta);
          ctx->h_phasors[idx+nbeamant] = (float) sin(theta);
#endif

          //fprintf (stderr, "[%d][%d][%d] freq=%lf delay=%le angle=%lf degrees, theta=%lf (%e, i%e)\n", ichan, ibeam, iant, ctx->channels[ichan].cfreq, geometric_delay, angle_rads / DD2R, theta, ctx->h_phasors[idx], ctx->h_phasors[idx+nbeamant]);
        }
      }
    }
  }

  // initialize TBs
  for (i=0; i<ctx->n_tbs; i++)
  {
    float angle = 0.0;
    complex float phasor = cosf(angle) + sinf(angle) * I;
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (iant=0; iant<ctx->nant; iant++)
      {
        ctx->h_tb_phasors[i][ichan*ctx->nant+iant] = phasor;
      }
    }
  }

#ifdef USE_GPU
  if (ctx->fan_beams)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "alloc: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", 
                ctx->phasors_size, ctx->d_phasors, ctx->h_phasors);
    cudaError_t error = cudaMemcpyAsync ((void *) ctx->d_phasors, (void *) ctx->h_phasors,
                                         ctx->phasors_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                ctx->d_phasors, ctx->h_phasors);
      return -1;
    }
  }

  for (i=0; i<ctx->n_tbs; i++)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "alloc: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", ctx->tb_phasors_size, ctx->d_tb_phasors, ctx->h_tb_phasors);
    cudaError_t error = cudaMemcpyAsync ((void *) (ctx->d_tb_phasors[i]), 
                                         (void *) (ctx->h_tb_phasors[i]), 
                                         ctx->tb_phasors_size, cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
                ctx->d_tb_phasors[i], ctx->h_tb_phasors[i]);
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

  unsigned i;
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (ctx->h_tb_phasors[i])
      cudaFreeHost (ctx->h_tb_phasors[i]);
    ctx->h_tb_phasors[i] = 0;

    if (ctx->d_tb_phasors[i])
      cudaFree (ctx->d_tb_phasors[i]);
    ctx->d_tb_phasors[i] = 0;


    if (ctx->d_tbs[i])
      cudaFree (ctx->d_tbs[i]);
    ctx->d_tbs[i] = 0;
  }

  if (ctx->d_in)
    cudaFree (ctx->d_in);
  ctx->d_in = 0;

  if (ctx->fan_beams && ctx->d_fbs)
    cudaFree (ctx->d_fbs);
   ctx->d_fbs = 0;

  if (ctx->fan_beams && ctx->d_phasors)
    cudaFree (ctx->d_phasors);
  ctx->d_phasors = 0;

  if (ctx->mod_beams && ctx->d_mbs)
    cudaFree (ctx->d_mbs);
  ctx->d_mbs = 0;

  if (ctx->channels)
    free (ctx->channels);
  ctx->channels = 0;

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

  unsigned i;
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "dada_cuda_dbunregister (tb_hdus[%d])\n", i);
    if (dada_cuda_dbunregister (ctx->tb_hdus[i]) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister Tied Beam DADA buffers\n");
      return -1;
    }
  } 

  if (ctx->mod_beams)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "dada_cuda_dbunregister (mb_hdu)\n");
    if (dada_cuda_dbunregister (ctx->mb_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister Fan Beam DADA buffers\n");
      return -1;
    }
  }
#endif
  return 0;
}
 

int bfdsp_open (dada_client_t* client)
{
  assert (client != 0);
  mopsr_bfdsp_t* ctx = (mopsr_bfdsp_t*) client->context;

  multilog_t * log = (multilog_t *) client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "bfdsp_open()\n");

  mopsr_bfdsp_hdr_t tbs[MAX_TBS];
  mopsr_bfdsp_hdr_t fb;
  mopsr_bfdsp_hdr_t mb;

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
  
  unsigned i;
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (dada_hdu_lock_write (ctx->tb_hdus[i]) < 0)
    {
      multilog (log, LOG_ERR, "open: could not lock write on TB HDU[%d]\n",i);
      return -1;
    }
  }
  if (ctx->mod_beams)
  {
    if (dada_hdu_lock_write (ctx->mb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "open: could not lock write on MB HDU\n");
      return -1;
    }
  }

  ascii_header_del (client->header, "RANKED_MODULES");
  ascii_header_del (client->header, "AQ_PROC_FILE");
  ascii_header_del (client->header, "DELAY_CORRECTED");
  ascii_header_del (client->header, "RFI_MITIGATION");
  ascii_header_del (client->header, "CONFIG");
  ascii_header_del (client->header, "PFB_ID");
  ascii_header_del (client->header, "PHASE_CORRECTED");

  // initialize
  ctx->tb_blocks_open = 0;
  ctx->fb_block_open = 0;
  ctx->mb_block_open = 0;
  ctx->bytes_read = 0;
  ctx->bytes_written = 0;
  ctx->fb_block = 0;
  ctx->mb_block = 0;
  for (i=0; i<ctx->n_tbs; i++)
  {
    ctx->tb_blocks[i] = 0;
  }
  ctx->first_time = 1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: extracting params from header\n");

  // read NCHAN from the incoming header
  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read NCHAN from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NCHAN=%d\n", ctx->nchan);
#ifndef HIRES
  if (ctx->nchan != 1)
  {
    multilog (log, LOG_ERR, "open: NCHAN==%d in lowres mode\n", ctx->nchan);
    return -1;
  }
#endif

  // allocate some memory for channel definitions
  ctx->channels = (mopsr_chan_t *) malloc (sizeof(mopsr_chan_t) * ctx->nchan);
  if (!ctx->channels)
  {
    multilog (log, LOG_ERR, "open: failed to allocate memory for channels\n");
    return -1;
  }

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

  // calculate new TSAMP
  double in_tsamp, fb_tsamp, mb_tsamp; 
  if (ascii_header_get (client->header, "TSAMP", "%lf", &in_tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read TSAMP from header\n");
    return -1;
  }

  if (ctx->nbeam == -1)
    ctx->nbeam = ctx->nant;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NBEAM=%d\n", ctx->nbeam);

  double fb_divisor = 1;
  unsigned tb_divisor = 1;
  unsigned mb_divisor = 1;

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
  nbit_n = 4;     // 8-bit to 32-bit
  nchan_n = 1;    // no change in channelisation
  ndim_d  = 2;    // 2dim to 1 dim
  nsamp_d = 512;  // reduction in tsamp by 512

#ifdef HIRES
  nsamp_d = 32;
  // 10.24 us -> 327.68 us
#endif

  if (ctx->fan_beams)
  {
    unsigned numerator   = (nbit_n * nchan_n * ctx->nbeam);
    unsigned denominator = (ndim_d * nsamp_d * ctx->nant);

    //if (denominator % numerator != 0)
    if (((ctx->in_block_size * numerator) / denominator) != ctx->fb_block_size)
    {
      multilog (log, LOG_ERR, "open: FB block size did not match input * configuration\n");
      multilog (log, LOG_ERR, "open: numerator=%u denominator=%u\n", numerator, denominator);
      return -1;
    } 
    fb_divisor = (double) denominator / (double) numerator;
    fb_tsamp = in_tsamp * nsamp_d;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TSAMP in=%lf FB=%lf\n", in_tsamp, fb_tsamp);
  }

  if (ctx->n_tbs > 0)
    tb_divisor = ctx->nant / nbit_n;

  if (ctx->mod_beams)
  {
#ifdef HIRES
    nbit_n  = 4;    // 8 - 32 bit
    nsamp_d = 32;   // integrate 32 samples
    ndim_d  = 2;    // detect
#else
    nbit_n  = 4;    // 8 - 32 bit
    nsamp_d = 512;  // integrate 512 samples
    ndim_d  = 2;    // detect
#endif

    unsigned numerator = nbit_n;
    unsigned denominator = nsamp_d * ndim_d;
    mb_divisor = denominator / numerator;
    mb_tsamp = in_tsamp * nsamp_d;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TSAMP in=%lf MB=%lf\n", in_tsamp, mb_tsamp);
  }

  // calculate new BYTES_PER_SECOND
  uint64_t fb_bytes_per_second, tb_bytes_per_second, mb_bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BYTES_PER_SECOND from header\n");
    return -1;
  }
  fb_bytes_per_second = ctx->bytes_per_second / fb_divisor;
  tb_bytes_per_second = ctx->bytes_per_second / tb_divisor;
  mb_bytes_per_second = ctx->bytes_per_second / mb_divisor;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: BYTES_PER_SECOND in=%"PRIu64" fb=%"PRIu64" tb=%"PRIu64" mb=%"PRIu64"\n", 
              ctx->bytes_per_second, fb_bytes_per_second, tb_bytes_per_second, mb_bytes_per_second);

  // calculate new OBS_OFFSET
  uint64_t in_obs_offset, fb_obs_offset, tb_obs_offset, mb_obs_offset;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &in_obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read OBS_OFFSET from header\n");
    return -1;
  }
  fb_obs_offset = in_obs_offset / fb_divisor;
  tb_obs_offset = in_obs_offset / tb_divisor;
  mb_obs_offset = in_obs_offset / mb_divisor;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: OBS_OFFSET in=%"PRIu64" fb=%"PRIu64" tb=%"PRIu64" mb=%"PRIu64"\n", 
              in_obs_offset, fb_obs_offset, tb_obs_offset, mb_obs_offset);

  int64_t in_file_size = -1;
  int64_t fb_file_size = -1;
  int64_t tb_file_size = -1;
  int64_t mb_file_size = -1;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIi64"", &in_file_size) == 1)
  {
    fb_file_size = in_file_size / fb_divisor; 
    tb_file_size = in_file_size / tb_divisor; 
    mb_file_size = in_file_size / mb_divisor; 
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: FILE_SIZE in=%"PRIi64" fb=%"PRIi64" tb=%"PRIi64" mb=%"PRIu64"\n", 
              in_file_size, fb_file_size, tb_file_size, mb_file_size);

  unsigned chan_offset;
  if (ascii_header_get (client->header, "CHAN_OFFSET", "%u", &chan_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read CHAN_OFFSET from header\n");
    return -1;
  }

  // for the boresite
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

  if (ascii_header_get (client->header, "MD_ANGLE", "%f", &(ctx->md_angle)) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read MD_ANGLE from header\n");
    return -1;
  }

  // in degrees
  double primary_beam_width = 4.0;
  ctx->beam_spacing = primary_beam_width / ctx->nbeam;
  if (ascii_header_get (client->header, "FB_BEAM_SPACING", "%lf", &(ctx->beam_spacing)) != 1)
  {
    multilog (log, LOG_INFO, "open: no FB_BEAM_SPACING specified, using %lf degrees\n", ctx->beam_spacing);
  }
  // convert to radians
  ctx->beam_spacing *= (M_PI / 180.0);

  // true if AQDSP has steered antenna to boresight
  ctx->delay_tracking = 0;
  char tmp[32];
  if (ascii_header_get (client->header, "DELAY_TRACKING", "%s", tmp) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read DELAY_TRACKING in header\n");
    return -1;
  }
  if (strcmp(tmp, "true") == 0)
    ctx->delay_tracking = 1;

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

  //multilog (log, LOG_INFO, "open: RA (rad) = %le\n", ctx->source.raj);
  //multilog (log, LOG_INFO, "open: DEC (rad) = %le\n", ctx->source.decj);
  cal_app_pos_iau (ctx->source.raj, ctx->source.decj, utc, &(ctx->source.ra_curr), &(ctx->source.dec_curr));
  //multilog (log, LOG_INFO, "open: RA curr = %le\n", ctx->source.ra_curr);
  //multilog (log, LOG_INFO, "open: DE curr = %le\n", ctx->source.dec_curr);

  // multilog (log, LOG_INFO, "open: coords J2000=(%lf, %lf) CURR=(%lf, %lf)\n", 
  //           ctx->source.raj, ctx->source.decj, ctx->source.ra_curr, ctx->source.dec_curr);

  double bw;
  if (ascii_header_get (client->header, "BW", "%lf", &bw) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read BW from header\n");
    return -1;
  }

  double freq;
  if (ascii_header_get (client->header, "FREQ", "%lf", &freq) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read FREQ from header\n");
    return -1;
  }

  double chan_bw = bw / ctx->nchan;
  double base_freq = freq - (bw/2);
  unsigned ichan;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: base_freq=%lf chan_bw=%lf chan_offset=%u\n", base_freq, chan_bw, chan_offset);
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    ctx->channels[ichan].number = chan_offset + ichan;
    ctx->channels[ichan].bw = chan_bw;
    ctx->channels[ichan].cfreq = base_freq + (chan_bw/2) + ichan * chan_bw;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "ctx->channels[%d] number=%u bw=%lf cfreq=%lf\n", ichan, ctx->channels[ichan].number, ctx->channels[ichan].bw, ctx->channels[ichan].cfreq);
  }

  char order[4];
  if (ascii_header_get (client->header, "ORDER", "%s", order) != 1)
  {
    multilog (log, LOG_ERR, "open: could not read ORDER from header\n");
    return -1;
  }
  if ((strcmp (order, "ST") != 0) && (strcmp(order, "FST") != 0))
  {
    multilog (log, LOG_ERR, "open: input ORDER was not ST or FST\n");
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

  if (ascii_header_set (client->header, "NBIT", "%d", 32) < 0)
  {     
    multilog (log, LOG_ERR, "open: could not set NBIT=32 in outgoing header\n");
    return -1;    
  }                 

  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  // extract relevant header params for each output and clear them from
  // the header if extracted
  if (ctx->mb_hdu)
  {
    if (ascii_header_get (client->header, "MB_PID", "%s", mb.pid) != 1)
    {
       multilog (log, LOG_ERR, "open: could not read MB_PID from header\n");
       return -1;
    }
    ascii_header_del (client->header, "MB_PID");
  }

  if (ctx->fb_hdu)
  {
    if (ascii_header_get (client->header, "FB_PID", "%s", fb.pid) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read FB_PID from header\n");
      return -1;
    }
    if (ascii_header_get (client->header, "FB_MODE", "%s", fb.mode) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read FB_MODE from header\n");
      return -1;
    }

    ascii_header_del (client->header, "FB_PID");
    ascii_header_del (client->header, "FB_MODE");
  }

  char key[32];
  for (i=0; i<ctx->n_tbs; i++)
  {
    sprintf(key, "TB%d_PID", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].pid) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    sprintf(key, "TB%d_MODE", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].mode) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    sprintf(key, "TB%d_PROC_FILE", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].proc_file) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    sprintf(key, "TB%d_SOURCE", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].source) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    sprintf(key, "TB%d_RA", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].ra) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    sprintf(key, "TB%d_DEC", i);
    if (ascii_header_get (client->header, key, "%s", tbs[i].dec) != 1)
    {
      multilog (log, LOG_ERR, "open: could not read %s from header\n", key);
      return -1;
    }
    ascii_header_del (client->header, key);

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TB[%d] RA (HMS) = %s\n", i, tbs[i].ra);
    if (mopsr_delays_hhmmss_to_rad (tbs[i].ra, &(ctx->tb_sources[i].raj)) < 0)
    {
      multilog (log, LOG_ERR, "open: TB[%d] could not parse RA from %s\n", i, tbs[i].ra);
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TB[%d] RA (rad) = %lf\n", i, ctx->tb_sources[i].raj);

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TB[%d] DEC (DMS) = %s\n", i,  tbs[i].dec);
    if (mopsr_delays_ddmmss_to_rad (tbs[i].dec, &(ctx->tb_sources[i].decj)) < 0)
    {
      multilog (log, LOG_ERR, "open: TB[%d] could not parse DEC from %s\n", i, tbs[i].dec);
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: TB[%d] DEC (rad) = %lf\n", i, ctx->tb_sources[i].decj);

    //multilog (log, LOG_INFO, "open: TB[%d] RA (rad) = %le\n", i, ctx->tb_sources[i].raj);
    //multilog (log, LOG_INFO, "open: TB[%d] DEC (rad) = %le\n", i, ctx->tb_sources[i].decj);
    cal_app_pos_iau (ctx->tb_sources[i].raj, ctx->tb_sources[i].decj, utc, &(ctx->tb_sources[i].ra_curr), &(ctx->tb_sources[i].dec_curr));
    //multilog (log, LOG_INFO, "open: TB[%d] RA curr = %le\n", i, ctx->tb_sources[i].ra_curr);
    //multilog (log, LOG_INFO, "open: TB[%d] DE curr = %le\n", i, ctx->tb_sources[i].dec_curr);
  }

  // if we have Module Beams
  if (ctx->mb_hdu)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: getting next free MB header buffer\n");
    char * header = ipcbuf_get_next_write (ctx->mb_hdu->header_block);
    if (!header)
    {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    // copy the header from the in to the out
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: copying header from input to output\n");
    memcpy (header, client->header, header_size);

    // double check that the mb_block_size is mb_divisor times smaller than the input block size
    if (ctx->in_block_size / ctx->mb_block_size != mb_divisor)
    {
      multilog (log, LOG_ERR, "open: in/out block size factor did not match mb_divisor\n");
      return -1;
    }

    if (ascii_header_set (header, "STATE", "%s", "Intensity") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set STATE=Intensity in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, ctx->mb_block_size) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", ctx->mb_block_size);
      return -1;
    }

    if (ctx->nchan == 1)
      sprintf (order, "%s", "ST");
    else
      sprintf (order, "%s", "SFT");
    if (ascii_header_set (header, "ORDER", "%s", order) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=%s in outgoing header\n", order);
      return -1;
    }

    if (ascii_header_set (header, "NDIM", "%d", 1) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NDIM=1 in outgoing header\n");
      return -1;
    }

    if (ascii_header_set (header, "TSAMP", "%lf", mb_tsamp) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set TSAMP=%lf in outgoing header\n", mb_tsamp);
      return -1;
    }

    if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, mb_obs_offset) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set OBS_OFFSET=%"PRIu64" in outgoing header\n", mb_obs_offset);
      return -1;
    }

    if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, mb_bytes_per_second) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BYTES_PER_SECOND=%"PRIu64" in outgoing header\n", mb_bytes_per_second);
      return -1;
    }

    // hack for now TODO fix!!
    if (ascii_header_set (header, "NBEAM", "%d", ctx->nant) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NBEAM=%d in outgoing header\n", ctx->nant);
      return -1;
    }
    if (ascii_header_set (header, "NANT", "%d", 1) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NANT=%d in outgoing header\n", 1);
      return -1;
    }

    if ((mb_file_size > 0) && (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, mb_file_size) < 0))
    {
      multilog (log, LOG_ERR, "open: could not set FILE_SIZE=%"PRIu64" in outgoing header\n", mb_file_size);
      return -1;
    }

    if (ascii_header_set (header, "PID", "%s", mb.pid) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set PID in outgoing header\n", 1);
      return -1;
    }

    int ibeam;
    // md offets will be [0,] 2 chars long
    char strval[9];
    size_t mdlist_len = (sizeof(char) * 2 * ctx->nant) + 1;
    char * mdlist = (char *) malloc(mdlist_len);
    mdlist[0] = '\0';
    for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    {
      sprintf(strval, "0");
      strcat (mdlist, strval);
      if (ibeam < ctx->nbeam-1)
        strcat (mdlist,",");
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: strlen(mdlist)=%d, size=%dlen=%d header_size=%d\n", strlen(mdlist), mdlist_len, strlen(header), header_size);
    if (ascii_header_set (header, "BEAM_MD_OFFSETS", "%s", mdlist) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BEAM_MD_OFFSETS=%s in outgoing header\n", ibeam, mdlist);
      return -1;
    }
    free (mdlist);

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: marking MB header filled, len=%d header_size=%d\n", strlen(header), header_size);

    // mark the outgoing header as filled
    if (ipcbuf_mark_filled (ctx->mb_hdu->header_block, header_size) < 0)
    {
      multilog (log, LOG_ERR, "open: could not mark header_block filled\n");
      return -1;
    }
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
    if (ctx->in_block_size / fb_divisor != ctx->fb_block_size)
    {
      multilog (log, LOG_ERR, "open: in block size did not match fb block size / fb_divisor\n");
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

    if (ctx->nchan == 1)
      sprintf (order, "%s", "ST");
    else
      sprintf (order, "%s", "SFT");
    if (ascii_header_set (header, "ORDER", "%s", order) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=%s in outgoing header\n", order);
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

    if (ascii_header_set (header, "NCHAN", "%d", ctx->nchan) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NCHAN=%d in outgoing header\n", ctx->nchan);
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

    // md offets will be [-0.000000,] 7 or 8 chars long
    int ibeam;
    char strval[10];
    size_t mdlist_len = (sizeof(char) * 11 * ctx->nbeam);
    char * mdlist = (char *) malloc(mdlist_len);
    mdlist[0] = '\0';
    for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
    {
      sprintf(strval, "%7.6f", ctx->h_beam_offsets[ibeam]);
      strcat (mdlist, strval);
      if (ibeam < ctx->nbeam-1)
        strcat (mdlist,",");
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: strlen(mdlist)=%d, mdlist_len=%ld strlen(header)=%d header_size=%d\n", strlen(mdlist), mdlist_len, strlen(header), header_size);
    if (ascii_header_set (header, "BEAM_MD_OFFSETS", "%s", mdlist) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set BEAM_MD_OFFSETS=%s in outgoing header\n", ibeam, mdlist);
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: free (mdlist)\n");

    free (mdlist);

    if (ascii_header_set (header, "PID", "%s", fb.pid) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set PID in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "MODE", "%s", fb.mode) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set MODE in outgoing header\n", 1);
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

  for (i=0; i<ctx->n_tbs; i++)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: getting next free TB header buffer\n");
    char * header = ipcbuf_get_next_write (ctx->tb_hdus[i]->header_block);
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
    if (ctx->in_block_size / ctx->tb_block_sizes[i] != tb_divisor)
    {
      multilog (log, LOG_ERR, "open: in[%lu]/out[%lu] block size factor did not match divisor [%d]\n", ctx->in_block_size, ctx->tb_block_sizes[i], tb_divisor);
      return -1;
    }

    //uint64_t resolution = ctx->tb_block_sizes[i];
    uint64_t resolution = ctx->nchan * nchan_n;
    if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, resolution) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set RESOLUTION=%"PRIu64" in outgoing header\n", resolution);
      return -1;
    }

    if (ctx->nchan == 1)
      sprintf (order, "%s", "T");
    else
    {
      //sprintf (order, "%s", "FT");
      sprintf (order, "%s", "TF");
    }
    if (ascii_header_set (header, "ORDER", "%s", order) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=%s in outgoing header\n", order);
      return -1;
    }

    if (ascii_header_set (header, "NBEAM", "%d", 1) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NBEAM=%d in outgoing header\n", 1);
      return -1;    
    }

    if (ascii_header_set (header, "NCHAN", "%d", ctx->nchan * nchan_n) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set NCHAN=%d in outgoing header\n", ctx->nchan * nchan_n);
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

    if (ascii_header_set (header, "PID", "%s", tbs[i].pid) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set PID in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "MODE", "%s", tbs[i].mode) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set MODE in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "PROC_FILE", "%s", tbs[i].proc_file) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set PROC_FILE in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "SOURCE", "%s", tbs[i].source) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set SOURCE in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "RA", "%s", tbs[i].ra) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set RA in outgoing header\n", 1);
      return -1;
    }
    if (ascii_header_set (header, "DEC", "%s", tbs[i].dec) < 0)
    {
      multilog (log, LOG_ERR, "open: could not set DEC in outgoing header\n", 1);
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: marking output header filled\n");

    // mark the outgoing header as filled
    if (ipcbuf_mark_filled (ctx->tb_hdus[i]->header_block, header_size) < 0) 
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

  if (ctx->mb_block_open)
  { 
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: ipcio_close_block_write bytes_written=%"PRIu64"\n", ctx->bytes_written);
    if (ipcio_close_block_write (ctx->mb_hdu->data_block, ctx->bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->mb_block_open = 0;
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
  if (ctx->mod_beams)
  {
    if (dada_hdu_unlock_write (ctx->mb_hdu) < 0)
    {
      multilog (log, LOG_ERR, "close: cannot unlock MB HDU\n");
      return -1;
    }
  }

  unsigned i;
  for (i=0; i<ctx->n_tbs; i++)
  {
    if (dada_hdu_unlock_write (ctx->tb_hdus[i]) < 0)
    {
      multilog (log, LOG_ERR, "close: cannot unlock TB HDU[%d]\n",i);
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

  if (ctx->verbose)
    multilog (log, LOG_INFO, "io_block: buffer=%p, bytes=%"PRIu64" block_id=%"PRIu64"\n", buffer, bytes, block_id);

  cudaError_t error;
  uint64_t out_block_id;
  unsigned ichan, iant, i;

#ifdef USE_GPU 
  // copy the whole block to the GPU
  if (ctx->verbose)
    multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", bytes, ctx->d_in, buffer);
  error = cudaMemcpyAsync (ctx->d_in, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              ctx->d_in, buffer);
    ctx->internal_error = 1;
    return -1;
  }

  if (ctx->n_tbs > 0)
  {
    if (!ctx->tb_blocks_open)
    {
      for (i=0; i<ctx->n_tbs; i++)
      {
        ctx->tb_blocks[i] = ipcio_open_block_write (ctx->tb_hdus[i]->data_block, &out_block_id);
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "io_block: opened TB output block %"PRIu64" %p\n", out_block_id, (void *) ctx->tb_blocks[i]);
      }
    }
    ctx->tb_blocks_open = 1;
  }

  // form a tied array beam if requested
  for (i=0; i<ctx->n_tbs; i++)
  {
    // update phasors for the tied array beam
    bfdsp_update_tb (ctx, i);

    mopsr_tie_beam (ctx->stream, ctx->d_in, ctx->d_tbs[i], ctx->d_tb_phasors[i], ctx->in_block_size, ctx->nant, ctx->nchan);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                (void *) ctx->tb_blocks[i], ctx->d_tbs[i], ctx->tb_block_sizes[i]);
    error = cudaMemcpyAsync ( (void *) ctx->tb_blocks[i], ctx->d_tbs[i], ctx->tb_block_sizes[i],
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
    mopsr_tile_beams_precomp (ctx->stream, ctx->d_in, ctx->d_fbs, ctx->d_phasors, ctx->in_block_size, ctx->nbeam, ctx->nant, ctx->nchan);
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
 
  if (ctx->mod_beams)
  {
#ifdef HIRES
    unsigned tdec = 32;
#else
    unsigned tdec = 512;
#endif
    mopsr_mod_beams (ctx->stream, ctx->d_in, ctx->d_mbs, ctx->in_block_size, ctx->nant, ctx->nchan, tdec);

    // copy back to output buffer
    if (!ctx->mb_block_open)
    {
      ctx->mb_block = ipcio_open_block_write (ctx->mb_hdu->data_block, &out_block_id);
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: opened output block %"PRIu64" %p\n", out_block_id, (void *) ctx->mb_block);
      ctx->mb_block_open = 1;
    }

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: cudaMemcpyAsync(%p, %p, %"PRIu64", D2H)\n",
                (void *) ctx->mb_block, ctx->d_mbs, ctx->mb_block_size);
    error = cudaMemcpyAsync ( (void *) ctx->mb_block, ctx->d_mbs, ctx->mb_block_size,
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

  if (ctx->tb_blocks_open)
  {
    for (i=0; i<ctx->n_tbs; i++)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "io_block: closing TB data block for %"PRIu64" bytes\n", ctx->tb_block_sizes[i]);
      ipcio_close_block_write (ctx->tb_hdus[i]->data_block, ctx->tb_block_sizes[i]);
    }
    ctx->tb_blocks_open = 0;
  }

  if (ctx->mb_block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "io_block: closing MB data block for %"PRIu64" bytes\n", ctx->mb_block_size);
    ipcio_close_block_write (ctx->mb_hdu->data_block, ctx->mb_block_size);
    ctx->mb_block_open = 0;
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
  return 0;
}

// determine if there is a pulsar to time within the primary beam
void bfdsp_update_tb (mopsr_bfdsp_t * ctx, unsigned i)
{
  // determine current offset between PSR and MD=0 in radians
  // note that UT1 offset was already accounted for in the AQDSP engine
  const double mid_byte = (double) ctx->bytes_read + (ctx->in_block_size / 2);
  const double mid_time = (double) ctx->utc_start + (mid_byte / (double) ctx->bytes_per_second);

  struct timeval timestamp;
  timestamp.tv_sec = (long) floor (mid_time);
  timestamp.tv_usec = (long) floor ((mid_time - (double) timestamp.tv_sec) * 1e6);

  double boresight_delay = sin(ctx->md_angle);
  // if AQDSP is tracking the source, then use the boresight delay
  if (ctx->delay_tracking)
    boresight_delay = calc_jer_delay (ctx->source.ra_curr, ctx->source.dec_curr, timestamp);
  double tb_delay = calc_jer_delay (ctx->tb_sources[i].ra_curr, ctx->tb_sources[i].dec_curr, timestamp);
  double jer_delay = tb_delay - boresight_delay;

  //fprintf (stderr, "boresight ra=%le tb_ra=%le diff=%le\n", ctx->source.ra_curr, ctx->tb_sources[i].ra_curr, ctx->source.ra_curr - ctx->tb_sources[i].ra_curr);

  if (ctx->verbose > 1)
    fprintf (stderr, "tb_idx=%u, boresight_delay=%le, tb_delay=%le jer_delay=%le\n", i, boresight_delay, tb_delay, jer_delay);

  unsigned iant, ichan;
  const double C = 2.99792458e8;

  // Phasors are stored in FS ordering
  for (iant=0; iant<ctx->nant; iant++)
  {
    double geometric_delay = (jer_delay * ctx->modules[iant]->dist) / C;
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      double theta = -2 * M_PI * ctx->channels[ichan].cfreq * 1000000 * geometric_delay;
      //fprintf (stderr, "iant=%u geometric_delay=%le theta=%lf \n", iant, geometric_delay, theta);
      ctx->h_tb_phasors[i][ichan*ctx->nant+iant] = (float) cos(theta) + (float) sin(theta) * I;
    }
  }

  // copy these to device memory
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "update_tb: cudaMemcpyAsync block H2D %ld: (%p <- %p)\n", 
              ctx->tb_phasors_size, ctx->d_tb_phasors[i], ctx->h_tb_phasors[i]);
  cudaError_t error = cudaMemcpyAsync ((void *) ctx->d_tb_phasors[i], 
                                       (void *) ctx->h_tb_phasors[i], 
                                       ctx->tb_phasors_size, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (ctx->log, LOG_ERR, "cudaMemcpyAsyc H2D failed: %s (%p <- %p)\n", cudaGetErrorString(error),
              ctx->d_tb_phasors[i], ctx->h_tb_phasors[i]);
    return;
  }
}

void bfdsp_calculate_statistics (mopsr_bfdsp_t * ctx)
{
  float * data = (float *) ctx->fb_block;

}
