/***************************************************************************
 *  
 *    Copyright (C) 2017 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/**
 * Performs cornerturn send via TCP/IP sockets
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>
#include <limits.h>

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "bpsr_ct.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

// Globals
int quit_signal = 0;

void * bpsr_dbnic_init_thread (void * arg);

void usage()
{
  fprintf (stdout,
           "bpsr_dbnic [options] send_id cornerturn.cfg\n"
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -s                single transfer only\n"
     " -v                verbose output\n"
     " -h                print this usage\n\n"
     " id                sender ID in cornerturn configuration\n"
     " cornerturn.cfg    ascii configuration file defining cornerturn\n",
     DADA_DEFAULT_BLOCK_KEY);
}


/*! Function that opens the data transfer target */
int bpsr_dbnic_open (dada_client_t* client)
{
  // the bpsr_dbnic specific data
  assert (client != 0);
  bpsr_ct_send_t* ctx = (bpsr_ct_send_t*) client->context;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "bpsr_dbnic_open()\n");
 
  // the header
  assert(client->header != 0);
  char * header = client->header;

  ctx->fds = (int *) malloc (sizeof(int) * ctx->ct.nconn);

  unsigned i = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "open: connecting to %s:%d\n",
               ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);

    ctx->fds[i] = sock_open (ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);

    if (ctx->fds[i] < 0)
    {
      multilog(ctx->log, LOG_ERR, "open: failed to connect to %s:%d\n",
               ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);
      return -1;
    }
  }

  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = 0;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int bpsr_dbnic_close (dada_client_t* client, uint64_t bytes_written)
{
  // the bpsr_dbnic specific data
  bpsr_ct_send_t* ctx = (bpsr_ct_send_t*) client->context;

  // status and error logging facility
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "bpsr_dbnic_close()\n");

  unsigned i = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "open_connections: connecting to %s:%d\n",
                ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);

    sock_close (ctx->fds[i]);
    ctx->fds[i] = 0;
  }
  free (ctx->fds);

  return 0;
}

/*! transfer data to ibdb used for sending header only */
int64_t bpsr_dbnic_send (dada_client_t* client, void * buffer, uint64_t bytes)
{
  bpsr_ct_send_t * ctx = (bpsr_ct_send_t *) client->context;

  multilog_t * log = client->log;

  unsigned int i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "bpsr_dbnic_send()\n");

  unsigned int old_nchan, new_nchan;
  if (ascii_header_get (buffer, "NCHAN", "%d", &old_nchan) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NCHAN from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old NCHAN=%d\n", old_nchan);

  unsigned int old_nbeam, new_nbeam;
  if (ascii_header_get (buffer, "NBEAM", "%d", &old_nbeam) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NBEAM from header\n");
  }

  float old_freq, new_freq;
  if (ascii_header_get (buffer, "FREQ", "%f", &old_freq) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read FREQ from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old FREQ=%f\n", old_freq);

  float old_bw, new_bw;
  if (ascii_header_get (buffer, "BW", "%f", &old_bw) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read BW from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old BW=%f\n", old_bw);

  uint64_t old_bytes_per_second, new_bytes_per_second;
  if (ascii_header_get (buffer, "BYTES_PER_SECOND", "%"PRIu64"", &old_bytes_per_second) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read BYTES_PER_SECOND from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old BYTES_PER_SECOND=%"PRIu64"\n", old_bytes_per_second);

  uint64_t old_obs_offset, new_obs_offset;
  if (ascii_header_get (buffer, "OBS_OFFSET", "%"PRIu64"", &old_obs_offset) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read OBS_OFFSET from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old OBS_OFFSET=%"PRIu64"\n", old_obs_offset);

  uint64_t old_resolution, new_resolution;
  if (ascii_header_get (buffer, "RESOLUTION", "%"PRIu64"", &old_resolution) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read RESOLUTION from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old RESOLUTION=%"PRIu64"\n", old_resolution);

  int64_t old_file_size, new_file_size;
  if (ascii_header_get (buffer, "FILE_SIZE", "%"PRIi64"", &old_file_size) != 1)
  {
    int old_file_size = -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old FILE_SIZE=%"PRIi64"\n", old_file_size);

  float freq_low = old_freq - (old_bw / 2);
  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "send: freq_low=%f\n", freq_low);
  }

  // these parameters are divided by the number of connections
  new_bytes_per_second = old_bytes_per_second / ctx->ct.nconn;
  new_resolution = old_resolution / ctx->ct.nconn;
  new_obs_offset = old_obs_offset / ctx->ct.nconn;
  new_file_size = old_file_size / ctx->ct.nconn;

  if (ctx->forward_cornerturn)
  {
    new_bw = old_bw / ctx->ct.nconn;
    new_nchan = old_nchan / ctx->ct.nconn;
    new_nbeam = old_nbeam * ctx->ct.nsend;
  }
  else
  {
    new_bw = old_bw * ctx->ct.nconn;
    new_nchan = old_nchan * ctx->ct.nconn;
    new_nbeam = old_nbeam / ctx->ct.nrecv;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting BW=%f in header\n", new_bw);
  if (ascii_header_set (buffer, "BW", "%f", new_bw) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set BW=%f in header\n", new_bw);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  if (ascii_header_set (buffer, "BYTES_PER_SECOND", "%"PRIu64"", new_bytes_per_second) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting RESOLUTION=%"PRIu64" in header\n", new_resolution);
  if (ascii_header_set (buffer, "RESOLUTION", "%"PRIu64"", new_resolution) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set RESOLUTION=%"PRIu64" in header\n", new_resolution);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
  if (ascii_header_set (buffer, "OBS_OFFSET", "%"PRIu64"", new_obs_offset) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
  }

  if (old_file_size > 0)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    if (ascii_header_set (buffer, "FILE_SIZE", "%"PRIi64"", new_file_size) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting BW=%d in header\n",new_bw);
  if (ascii_header_set (buffer, "BW", "%d", new_bw) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set BW=%d in header\n", new_bw);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting NCHAN=%d in header\n",new_nchan);
  if (ascii_header_set (buffer, "NCHAN", "%d", new_nchan) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set NCHAN=%d in header\n", new_nchan);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting NBEAM=%d in header\n",new_nbeam);
  if (ascii_header_set (buffer, "NBEAM", "%d", new_nbeam) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set NBEAM=%d in header\n", new_nbeam);
  }


  // copy the header to the header memory buffer
  for (i=0; i<ctx->ct.nconn; i++)
  {
    // each conn / channel will have a unique FREQ
    if (ctx->forward_cornerturn)
    {
      new_freq = freq_low + (new_bw / 2) + (new_bw * i);
      if (ctx->verbose)
        multilog (log, LOG_INFO, "send: setting FREQ=%f in header\n", new_freq);
      if (ascii_header_set (buffer, "FREQ", "%f", new_freq) < 0)
      { 
        multilog (log, LOG_WARNING, "send: failed to set FREQ=%f in header\n", new_freq);
      }
    }

    int flags = 0;
    size_t sent = send (ctx->fds[i], buffer, bytes, flags);
  }

  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", bytes);

  return bytes;
}


/*! Transfers 1 datablock at a time to bpsr_nicdb */
int64_t bpsr_dbnic_send_block (dada_client_t* client, void * buffer, 
                               uint64_t bytes, uint64_t block_id)
{
  bpsr_ct_send_t * ctx = (bpsr_ct_send_t*) client->context;
  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: bytes=%"PRIu64", block_id=%"PRIu64"\n", bytes, block_id);

  char * buf = (char *) buffer;

  unsigned bytes_to_send = (unsigned) (bytes / ctx->ct.nconn);
  unsigned nblocks = bytes_to_send / ctx->ct.atomic_size;

  unsigned i, j, k;
  int flags = 0;
  size_t sent;

  for (i=0; i<nblocks; i++)
  {
    for (j=0; j<ctx->ct.nconn; j++)
    {
      k = (j + ctx->ct.conn_info[j].isend) % ctx->ct.nrecv;
      sent = send (ctx->fds[k], buf + (k*ctx->ct.atomic_size), ctx->ct.atomic_size, flags);
    }
    buf += ctx->ct.send_resolution;
  }
  return (int64_t) bytes;
}

int bpsr_dbnic_destroy (bpsr_ct_send_t * ctx) 
{
  unsigned i=0;
  int rval = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
  }
  return rval;
}

/** 
 * open TCP sockets to the listen CT recipients
 */
int bpsr_dbnic_open_connections (bpsr_ct_send_t * ctx, multilog_t * log)
{
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "bpsr_dbnic_open_connections()\n");

  ctx->fds = (int *) malloc (sizeof(int) * ctx->ct.nconn);

  unsigned i = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "open_connections: connecting to %s:%d\n", 
               ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);

    ctx->fds[i] = sock_open (ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);

    if (ctx->fds[i] < 0)
    {
      multilog(ctx->log, LOG_ERR, "open_connections: failed to connect to %s:%d\n",
               ctx->ct.conn_info[i].host, ctx->ct.conn_info[i].port);
      return -1;
    }
  }

  return 0;
}

/*! Simple signal handler for SIGINT */
void signal_handler(int signalValue)
{
  quit_signal = 1;
  exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Node configuration */
  bpsr_ct_send_t ctx = BPSR_CT_SEND_INIT;

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
  int quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  char * cornerturn_cfg = 0;

  unsigned int send_id = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hk:sv")) != -1)
  {
    switch (arg) 
    {
      case 'h':
        usage ();
        return 0;
      
      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr,"bpsr_dbnic: could not parse key from %s\n",optarg);
          return EXIT_FAILURE;
        }
        break;

      case 's':
        quit = 1;
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 1;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 2) {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  send_id = atoi(argv[optind]);
  cornerturn_cfg = strdup(argv[optind+1]);

  // do not use the syslog facility
  log = multilog_open ("bpsr_dbnic", 0);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_DBIB_LOG);
  }
  else
    multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = bpsr_dbnic_open;
  client->io_function       = bpsr_dbnic_send;
  client->io_block_function = bpsr_dbnic_send_block;
  client->close_function    = bpsr_dbnic_close;
  client->direction         = dada_client_reader;

  client->quiet = 1;
  client->context = &ctx;

  ctx.log = log;
  ctx.verbose = verbose;
  ctx.quit = quit;

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  // parse the cornerturn configuration
  if (verbose)
    multilog (log, LOG_INFO, "main: bpsr_setup_recv ()\n");
  ctx.ct.conn_info = bpsr_setup_recv (cornerturn_cfg, &(ctx.ct), send_id);
  if (!ctx.ct.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn configuration file\n");
    dada_hdu_unlock_read(hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: dada_client_read()\n");

  while (!client->quit)
  {
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");
    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (quit || ctx.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (verbose)
        multilog (log, LOG_INFO, "main: dada_hdu_lock_read()\n");
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: dada_hdu_disconnect()\n");
  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "main: bpsr_dbnic_destroy()\n");
  if (bpsr_dbnic_destroy (&ctx) < 0)
  {
    multilog(log, LOG_ERR, "bpsr_dbnic_destroy failed\n");
  }

  return EXIT_SUCCESS;
}

