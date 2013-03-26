#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <cpgplot.h>

typedef struct {

  multilog_t * log;

  char * utc_start;

  char * device;

  // number of time frames currently integrated into the antennae arrays
  unsigned int i_integrate;

  // number of time frames to integrate into each plot
  unsigned int t_integrate;

  // number of time epocs to skip between each plot
  unsigned int t_skip;

  // signal for plotting thread to begin plot [0==accumulate data, 1 means plot]
  unsigned plot_pending;

  // signal from the integration thread 
  unsigned sum_pending;

  unsigned int header_written;

  int64_t ** antennae;

  float ymin;

  float ymax;

  // number of antennae
  unsigned int nant;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  int verbose;

} dbplotbp_t;

int quit_threads = 0;

void plot_thread (void * arg);

#define MOPSR_DBPLOTBP_INIT { 0, "", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

void usage()
{
  fprintf (stdout,
     "mopsr_dbplot [options]\n"
     " -k key     connect to key data block\n"
     " -s         single transfer only\n"
     " -t num     integrate num time frames into spectra\n"
     " -D device  pgplot device name [default ?]\n"
     " -v         be verbose\n");
}


/*! Function that opens the data transfer target */
int dbplot_open_function (dada_client_t* client)
{

  dbplotbp_t * ctx = (dbplotbp_t *) client->context;

  multilog_t* log;

  // the header
  char* header = 0;

  // size of each transfer in bytes, as defined by TRANSFER_SIZE attribute 
  uint64_t xfer_size = 64*1024*1024;

  // the optimal buffer size for writing to file 
  uint64_t optimal_bytes = 8*1024*1024;

  assert (client != 0);

  log = client->log;
  assert (log != 0);

  header = client->header;
  assert (header != 0);

  ctx->header_written = 0;

  header = client->header;
  assert (header != 0);

  // get the NDIM
  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no NDIM\n");
  }
  if (ctx->verbose)
     multilog (log, LOG_INFO, "open: parsed NDIM=%d\n", ctx->ndim);

  // get the NCHAN
  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no NCHAN\n");
  } 
  if (ctx->verbose)
     multilog (log, LOG_INFO, "open: parsed NCHAN=%d\n", ctx->nchan);

  // get the NANT
  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no NANT\n");
  } 
  if (ctx->verbose)
     multilog (log, LOG_INFO, "open: parsed NANT=%d\n", ctx->nant);

  // get the RESOLUTION
  if (ascii_header_get (client->header, "RESOLUTION", "%d", &(ctx->resolution)) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no RESOLUTION\n");
  }
  if (ctx->verbose)
     multilog (log, LOG_INFO, "open: parsed RESOLUTION=%d\n", ctx->resolution);

  multilog (log, LOG_INFO, "open: ndim=%d, nchan=%d, nant=%d, resolution=%d\n", ctx->ndim, ctx->nchan, ctx->nant, ctx->resolution);
  ctx->antennae = (int64_t **) malloc (sizeof (int64_t *) * ctx->nant);
  unsigned int iant, ichan;
  for (iant=0; iant < ctx->nant; iant++)
    ctx->antennae[iant] = (int64_t *) malloc (sizeof (int64_t) * ctx->nchan);
  for (iant=0; iant < ctx->nant; iant++)
    for (ichan=0; ichan < ctx->nchan; ichan++)
      ctx->antennae[iant][ichan] = 0;

  ctx->utc_start = (char *) malloc (sizeof(char) * 32);
  if (ascii_header_get (client->header, "UTC_START", "%s", ctx->utc_start) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no UTC_START\n");
    strcpy (ctx->utc_start, "UNKNOWN");
  }

  ctx->i_integrate = 0;

  client->optimal_bytes = optimal_bytes;
  client->fd = 1;

  return 0;
}

/*! Function that closes the data file */
int dbplot_close_function (dada_client_t* client, uint64_t bytes_written)
{
  dbplotbp_t * ctx  = (dbplotbp_t *) client->context;

  if (bytes_written < client->transfer_bytes)
  {
    multilog (client->log, LOG_INFO, "Transfer stopped early at %"PRIu64" bytes\n",
	      bytes_written);
  }
  return 0;
}


int dbplotdb_destroy (dbplotbp_t * ctx)
{
  if (ctx->utc_start)
    free (ctx->utc_start);
  ctx->utc_start = 0;

  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if (ctx->antennae[iant])
      free (ctx->antennae[iant]);
    ctx->antennae[iant] = 0;
  }
  if (ctx->antennae)
    free(ctx->antennae);
  ctx->antennae = 0;

  return 0;
}


// process 1 block of data, integrating all samples into the buffer
int64_t dbplotbp_block_function (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  dbplotbp_t * ctx  = (dbplotbp_t *) client->context;

  unsigned int ipkt = 0;
  unsigned int ichan = 0;
  unsigned int iframe = 0;
  unsigned int iant = 0;

  // ensure we get a whole packet
  unsigned int npkt = bytes / ctx->resolution;
  const unsigned int nframes = ctx->resolution / (ctx->nchan * ctx->nant * ctx->ndim);

  multilog (client->log, LOG_INFO, "block: bytes=%"PRIu64", n_frames=%d remainder=%d\n", bytes, nframes, bytes % ctx->resolution);

  int8_t * in = (int8_t *) buffer;

  while (ctx->plot_pending)
    usleep(10000);
  ctx->sum_pending = 1;

  // foreach UDP packet
  for (ipkt=0; ipkt < npkt; ipkt += (1 + ctx->t_skip))
  {
    // foreach time frame
    for (iframe=0; iframe < nframes; iframe++)
    {
      // foreach channel
      for (ichan=0; ichan < ctx->nchan; ichan++)
      {
        // foreach channel
        for (iant=0; iant < ctx->nant; iant++)
        {
          int64_t a = (int64_t) in[1];
          int64_t b = (int64_t) in[0];
          ctx->antennae[iant][ichan] += (a*a) + (b*b);
          in += 2;
        }
      }
    }
    ctx->i_integrate += nframes;

    //multilog (client->log, LOG_INFO, "block: skip = %d\n", ctx->t_skip * ctx->nchan * nframes * ctx->nant * 2);

    in += ctx->t_skip * ctx->nchan * nframes * ctx->nant * 2;
  }

  if (ctx->plot_pending)
    multilog (client->log, LOG_INFO, "block: plot_pending whilst sum_pending !!!!\n");

  ctx->sum_pending = 0;

  return (int64_t) bytes;
}


/*! Pointer to the function that transfers data to/from the target */
int64_t dbplot_buffer_function (dada_client_t* client, 
			    void* data, uint64_t data_size)
{
  dbplotbp_t * ctx = (dbplotbp_t *) client->context;

  /* Dont plot the header silly */
  if (!ctx->header_written)
  {
    ctx->header_written = 1;
    return (int64_t) data_size;
  } 
  else 
  {
    multilog (client->log, LOG_ERR, "buffer: Should not be called!\n");
    return 0;
  }
}


void plot_thread (void * arg)
{
  dbplotbp_t * ctx = (dbplotbp_t *) arg;

  float * x_points = 0;
  float ** y_points = 0;

  unsigned ichan = 0;
  unsigned iant = 0;
  float n_integrate = 0;

  int i=0;
  int j=0;
  int k=0;

  ctx->ymin = 1000000;
  ctx->ymax = -1000000;

  time_t curr_time;
  time_t prev_time;

  while (!quit_threads)
  {
    prev_time = curr_time;
    curr_time = time(0);

    // every second...
    while (prev_time == curr_time)
    {
      curr_time = time(0);
      usleep(10000);
    }
    multilog (ctx->log, LOG_INFO, "plot_thread: i_integrate=%d\n", ctx->i_integrate);

    // if something has been integrated, then the nchan must be set
    if (ctx->i_integrate > 0)
    {
      //multilog (ctx->log, LOG_INFO, "plot_thread: plotting 1\n");

      if (!x_points)
      {
        //multilog (ctx->log, LOG_INFO, "plot_thread: allocating xpoints\n");
        x_points = (float *) malloc (sizeof(float) * ctx->nchan);
        for (ichan=0; ichan < ctx->nchan; ichan++)
          x_points[ichan] = (float) ichan;
      }

      if (!y_points)
      {
        //multilog (ctx->log, LOG_INFO, "plot_thread: allocating ypoints\n");
        y_points = (float **) malloc(sizeof(float *) * ctx->nant);
        for (iant=0; iant < ctx->nant; iant++)
          y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan);
      }

      //multilog (ctx->log, LOG_INFO, "plot_thread: assigning floats, checking min/max\n");

      // ensure we dont try to read
      while (ctx->sum_pending)
        usleep(10000);

      // tell integrator (block_function) not to start any new calculations
      ctx->plot_pending = 1;

      n_integrate = (float) (ctx->i_integrate * ctx->i_integrate) * 2;

      for (iant=0; iant < ctx->nant; iant++)
      {
        for (ichan=0; ichan < ctx->nchan; ichan++)
        {
          y_points[iant][ichan] = (float) (ctx->antennae[iant][ichan]); // n_integrate;
          if (y_points[iant][ichan] < ctx->ymin) ctx->ymin = y_points[iant][ichan];
          if (y_points[iant][ichan] > ctx->ymax) ctx->ymax = y_points[iant][ichan];

          // now reset the antennae
          ctx->antennae[iant][ichan] = 0;
        }
      }

      // reset our count
      ctx->i_integrate = 0;

      if (ctx->sum_pending)
        multilog (ctx->log, LOG_INFO, "plot_thread: sum_pending whilst plot_pending !!!!\n");

      // release integrator from any waiting
      ctx->plot_pending = 0;

      cpgbbuf();

      cpgsci(1);
      cpgenv(0, ctx->nchan, ctx->ymin, (1.1*ctx->ymax), 0, 0);
      cpglab("Channel", "Power", "Bandpass"); 

      float x = 0;
      float y = 0;
      char ant_label[32];

      sprintf(ant_label, "Int %d", (int) n_integrate);
      cpgmtxt("T", 0.5, 0.0, 0.0, ant_label);

      for (iant=0; iant < ctx->nant; iant++)
      {
        sprintf(ant_label, "Ant %d", iant);
        cpgsci(iant + 2);
        cpgmtxt("T", 1.5 + (1.0 * iant), 0.0, 0.0, ant_label);
        cpgline(ctx->nchan, x_points, y_points[iant]);
      }
      cpgebuf();

      //multilog (ctx->log, LOG_INFO, "plot_thread: plotting done\n");
    }
  }
  multilog (ctx->log, LOG_INFO, "plot_thread: exiting\n");
}


int main (int argc, char **argv)
{

  /* context struct */
  dbplotbp_t dbplot = MOPSR_DBPLOTBP_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* PGPLOT device name */
  char * device = "/xs";

  /* dada key for SHM */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int single_xfer_only = 0;

  unsigned int t_integrate = 61440;

  int arg = 0;

  /* TODO the amount to conduct a busy sleep inbetween clearing each sub
   * block */
  int busy_sleep = 0;

  while ((arg=getopt(argc,argv,"dD:st:vk:")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 's':
      single_xfer_only = 1;
      break;

    case 't':
      t_integrate = atoi(optarg);
      break;

    case 'v':
      verbose=1;
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
        return -1;
      }
      break;
      
    default:
      usage ();
      return 0;
      
  }

  dbplot.device = strdup(device);
  dbplot.header_written = 0;
  dbplot.t_integrate = t_integrate;
  dbplot.t_skip = 3;
  dbplot.verbose = verbose;

  /* Open pgplot device window */
  if (cpgopen(dbplot.device) != 1) {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  log = multilog_open ("mopsr_dbplot", daemon);
  dbplot.log = log;

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  // plotting thread
  pthread_t plot_thread_id;

  if (verbose)
    multilog(log, LOG_INFO, "starting plot_thread()\n");
  int rval = pthread_create (&plot_thread_id, 0, (void *) plot_thread, (void *) &dbplot);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating plot_thread: %s\n", strerror(rval));
    return -1;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = dbplot_open_function;
  client->io_function       = dbplot_buffer_function;
  client->io_block_function = dbplotbp_block_function;
  client->close_function    = dbplot_close_function;
  client->direction         = dada_client_reader;

  client->context = &dbplot;

  while (!quit) {
    
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (single_xfer_only)
      quit = 1;

    if (client->quit)
      quit = 1;

  }

  quit_threads = 1;

  if (verbose)
    multilog(log, LOG_INFO, "joining plot_thread\n");
  void * result = 0;
  pthread_join (plot_thread_id, &result);

  if (verbose)
    multilog(log, LOG_INFO, "plot_thread joined\n");

  // close PGPLOT device
  cpgclos();

  // release memory
  dbplotdb_destroy (&dbplot);

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
