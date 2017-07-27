/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>
#include <limits.h>
#include <pthread.h>

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "bpsr_ct.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"

// Globals
int quit_threads = 0;

void * bpsr_nicdb_thread (void * arg);

void usage()
{
  fprintf (stdout,
    "bpsr_nicdb_FST [options] recv_id cornerturn.cfg\n"
    " -k <key>          hexadecimal shared memory key  [default: %x]\n"
    " -s                single transfer only\n"
    " -v                verbose output\n"
    " -h                print this usage\n\n"
    " id                receiver ID in cornerturn configuration\n"
    " cornerturn.cfg    ascii configuration file defining cornerturn\n",
    DADA_DEFAULT_BLOCK_KEY);
}

/*! Function that opens the data transfer target */
int bpsr_nicdb_open (dada_client_t* client)
{
  // the bpsr_nicdb specific data
  assert (client != 0);
  bpsr_ct_recv_t* ctx = (bpsr_ct_recv_t*) client->context;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "bpsr_nicdb_open()\n");
  
  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = 0;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  ctx->obs_ending = 0;
}

/*! Function that closes the data file */
int bpsr_nicdb_close (dada_client_t* client, uint64_t bytes_written)
{
  // the bpsr_nicdb specific data
  bpsr_ct_recv_t* ctx = (bpsr_ct_recv_t*) client->context;

  // status and error logging facility
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "bpsr_nicdb_close()\n");

  if (bytes_written < client->transfer_bytes) 
  {
    multilog (log, LOG_INFO, "transfer stopped early at %"PRIu64" bytes, expecting %"PRIu64"\n",
              bytes_written, client->transfer_bytes);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: transferred %"PRIu64" bytes\n", bytes_written);

  return 0;
}

/*! data transfer function, for just the header */
int64_t bpsr_nicdb_recv (dada_client_t* client, void * buffer, uint64_t bytes)
{
  bpsr_ct_recv_t * ctx = (bpsr_ct_recv_t *) client->context;

  multilog_t * log = client->log;

  unsigned int i = 0;

  // lock the control mutex
  pthread_mutex_lock (&(ctx->mutex));

  // receive the ASCII header from all the transmitters
  char all_prepared = 0;
  while (!all_prepared)
  {
    all_prepared = 1;
    for (i=0; i<ctx->ct.nconn; i++)
    {
      if (ctx->states[ctx->ct.conn_info[i].irecv] != PREPARED)
      {
        all_prepared = 0;
      }
    }
    if (!all_prepared)
    {
      pthread_cond_wait (&(ctx->cond), &(ctx->mutex));
    }
  }

  // copy the header from buffer 0 for now
  memcpy (buffer, ctx->headers[0], DADA_DEFAULT_HEADER_SIZE);

  // get the bytes per second
  uint64_t old_bytes_per_second, new_bytes_per_second;
  if (ascii_header_get (buffer, "BYTES_PER_SECOND", "%"PRIu64"", &old_bytes_per_second) != 1)
  {
    multilog (log, LOG_WARNING, "recv: failed to read BYTES_PER_SECOND from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old BYTES_PER_SECOND=%"PRIu64"\n", old_bytes_per_second);

  uint64_t old_obs_offset, new_obs_offset;
  if (ascii_header_get (buffer, "OBS_OFFSET", "%"PRIu64"", &old_obs_offset) != 1)
  {
    multilog (log, LOG_WARNING, "recv: failed to read OBS_OFFSET from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old OBS_OFFSET=%"PRIu64"\n", old_obs_offset);

  uint64_t old_resolution, new_resolution;
  if (ascii_header_get (buffer, "RESOLUTION", "%"PRIu64"", &old_resolution) != 1)
  {
    multilog (log, LOG_WARNING, "recv: failed to read RESOLUTION from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old RESOLUTION=%"PRIu64"\n", old_resolution);

  int64_t old_file_size, new_file_size;
  if (ascii_header_get (buffer, "FILE_SIZE", "%"PRIi64"", &old_file_size) != 1)
  {
    old_file_size = -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old FILE_SIZE=%"PRIi64"\n", old_file_size);

  // number of beams
  unsigned old_nbeam, new_nbeam;
  if (ascii_header_get (buffer, "NBEAM", "%u", &old_nbeam) != 1)
  {
    multilog (log, LOG_ERR, "recv: failed to read NBEAM from header\n");
    return 0;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old NBEAM=%u\n", old_nbeam);

  new_resolution = old_resolution * ctx->ct.nconn;
  new_obs_offset = old_obs_offset * ctx->ct.nconn;
  new_bytes_per_second = old_bytes_per_second * ctx->ct.nconn;
  new_file_size = old_file_size * ctx->ct.nconn;

  if (ctx->forward_cornerturn)
  {
    new_nbeam = ctx->ct.nconn;
  }
  else
  {
    new_nbeam = 1;
  }
    
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: setting NBEAM=%u in header\n",new_nbeam);
  if (ascii_header_set (buffer, "NBEAM", "%u", new_nbeam) < 0)
  {
    multilog (log, LOG_ERR, "recv: failed to set NBEAM=%u in header\n", new_nbeam);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: setting BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  if (ascii_header_set (buffer, "BYTES_PER_SECOND", "%"PRIu64"", new_bytes_per_second) < 0)
  {
    multilog (log, LOG_WARNING, "recv: failed to set BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: setting OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
  if (ascii_header_set (buffer, "OBS_OFFSET", "%"PRIu64"", new_obs_offset) < 0)
  {
    multilog (log, LOG_WARNING, "recv: failed to set OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: setting RESOLUTION=%"PRIu64" in header\n", new_resolution);
  if (ascii_header_set (buffer, "RESOLUTION", "%"PRIu64"", new_resolution) < 0)
  {
    multilog (log, LOG_WARNING, "recv: failed to set RESOLUTION=%"PRIu64" in header\n", new_resolution);
  }

  if (old_file_size > 0)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "recv: setting FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    if (ascii_header_set (buffer, "FILE_SIZE", "%"PRIi64"", new_file_size) < 0)
    {
      multilog (log, LOG_WARNING, "recv: failed to set FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    }
  }

  // use the size of the first cms header
  return (int64_t) bytes;
}


/*! Transfers 1 datablock at a time to bpsr_nicdb */
int64_t bpsr_nicdb_recv_block (dada_client_t* client, void * buffer, 
                               uint64_t data_size, uint64_t block_id)
{
  multilog_t * log = client->log;

  // the bpsr_nicdb specific data
  bpsr_ct_recv_t* ctx = (bpsr_ct_recv_t*) client->context;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "recv_block: block_size=%"PRIu64", block_id=%"PRIu64"\n", data_size, block_id);

  // the pointer for the threads to receive into 
  ctx->recv_block = (char *) buffer;
  ctx->recv_bufsz = data_size;

  // signal the receiving threads to receive data
  unsigned i;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    ctx->states[i] = EMPTY;
  }
  pthread_cond_signal (&ctx->cond);
  pthread_mutex_unlock (&ctx->mutex);

  // wait for completion
  char received = 1;
  while (!received)
  {
    // lock the control mutex
    pthread_mutex_lock (&(ctx->mutex));

    // check the state of all threads
    for (i=0; i<ctx->ct.nconn; i++)
    {
      if (ctx->states[i] != FULL)
      {
        received = 0;
      }
    }

    // if not all locked, release the mutex and wait on COND
    if (!received)
    {
      pthread_cond_wait (&(ctx->cond), &(ctx->mutex));
    }
  }

  uint64_t bytes_received = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    bytes_received += ctx->bytes[i];
  }

  // all data has been received, and the mutex is locked by main thread
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "recv_block: bytes transferred=%"PRIi64"\n", bytes_received);

  return bytes_received;
}

int bpsr_nicdb_destroy (bpsr_ct_recv_t * ctx) 
{
  unsigned i=0;
  int rval = 0;
  for (i=0; i<ctx->ct.nconn; i++)
  {
  }
  return rval;
}

/**
 * start the receiving threads, 1 per connection
 */
int bpsr_nicdb_start_threads (bpsr_ct_recv_t * ctx, multilog_t * log)
{
  bpsr_conn_t * conns = ctx->ct.conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "bpsr_nicdb_start_threads: nconn=%u()\n", ctx->ct.nconn);

  ctx->states = (int *) malloc (sizeof(int) * ctx->ct.nconn);
  ctx->bytes = (uint64_t *) malloc (sizeof(uint64_t) * ctx->ct.nconn);
  ctx->threads = (pthread_t *) malloc (sizeof(pthread_t) * ctx->ct.nconn);
  ctx->ctxs = (bpsr_conn_recv_t *) malloc (sizeof(bpsr_conn_recv_t) * ctx->ct.nconn);
  ctx->results = (void **) malloc (sizeof(void *) * ctx->ct.nconn);

  unsigned i = 0;
  int rval = 0;
  // start all the connection threads
  for (i=0; i<ctx->ct.nconn; i++)
  {
    ctx->ctxs->conn_info = &(conns[i]);
    ctx->ctxs->ctx = ctx;

    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "start_threads: conns[%d]=%p\n", i, &(conns[i]));
    rval = pthread_create(&(ctx->threads[i]), 0, (void *) bpsr_nicdb_thread,
                        (void *) &(ctx->ctxs[i]));
    if (rval != 0)
    {
      multilog (ctx->log, LOG_INFO, "start_threads: error creating thread\n");
      return -1;
    }
  }
  return 0;
}

/**
 * stop the receiving threads
 */
int bpsr_nicdb_stop_threads (bpsr_ct_recv_t * ctx, multilog_t * log)
{
  bpsr_conn_t * conns = ctx->ct.conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "bpsr_nicdb_start_threads: nconn=%u()\n", ctx->ct.nconn);

  unsigned i;
  void * result;
  for (i=0; i<ctx->ct.nconn; i++)
  {
    pthread_join (ctx->threads[i], &result);
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "start_threads: connection thread %d joined\n", i);
  }

  free (ctx->states);
  free (ctx->bytes);

  return 0;
}



/**
 * receiving thread that creates a listen socket, and transfer data as instructed
 */
void * bpsr_nicdb_thread (void * arg)
{
  bpsr_conn_recv_t * thread_ctx = (bpsr_conn_recv_t *) arg;

  bpsr_conn_t * conn = thread_ctx->conn_info;
  bpsr_ct_recv_t * ctx = thread_ctx->ctx;

  multilog_t * log    = ctx->log;

  // fixed receiver id for this connection
  const unsigned recv_id = conn->irecv;

  // create a listening socket on the specified port
  int listen_fd = sock_create (conn->port);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "thread: listening for connections on %d\n", conn->port);

  size_t bytes_to_recv;
  ssize_t recvd;
  int flags = 0;

  // main control loop for thread
  while (!quit_threads)
  {
    // wait for a connection on the listening socket
    int fd = sock_accept (listen_fd); 

    // wait for the IDLE state
    pthread_mutex_lock (&(ctx->mutex));
    while (ctx->states[recv_id] != IDLE)
    {
      pthread_cond_wait (&(ctx->cond), &(ctx->mutex));
    }
    pthread_mutex_unlock (&(ctx->mutex));

    // receive a header on the socket, blocking
    bytes_to_recv = DADA_DEFAULT_HEADER_SIZE;
    recvd = recv (fd, ctx->headers[recv_id], bytes_to_recv, flags);

    if (recvd == bytes_to_recv)
    {
      multilog(log, LOG_INFO, "thread: received %ld bytes\n", recvd);
    }
    else
    {
      multilog(log, LOG_ERR, "recv returned %ld bytes, expecting, %ld\n", recvd, bytes_to_recv); 
      quit_threads = 1;
    }

    // incidate that the header for this thread has been received
    pthread_mutex_lock (&(ctx->mutex));
    ctx->states[recv_id] = PREPARED;
    pthread_cond_signal (&(ctx->cond));
    pthread_mutex_unlock (&(ctx->mutex));
    
    // starting offset for all connections on this socket
    size_t buffer_offset = conn->recv_offset;
    size_t bytes_to_recv = (size_t) conn->atomic_size;

    char keep_receiving = 1;
    while (keep_receiving)
    {
      // wait for the mutex 
      pthread_mutex_lock (&(ctx->mutex));

      // wait for the receiving buffer to be EMPTY, 
      while (ctx->states[recv_id] != EMPTY)
      {
        pthread_cond_wait (&(ctx->cond), &(ctx->mutex));
      }

      // the buffer to be filled is now empty, receive data on TCP socket
      uint64_t total_bytes_to_receive = ctx->recv_bufsz / ctx->ct.nconn;
      uint64_t bytes_received = 0;
      ssize_t got;

      char * buf = ctx->recv_block + buffer_offset;
      int flags = 0;
      while (bytes_received < total_bytes_to_receive && keep_receiving)
      {
        got = recv (fd, (void *) buf, bytes_to_recv, flags);
        if (got == bytes_to_recv)
        {
          // increase input buffer by stride
          buf += ctx->ct.recv_resolution;
          // increase total bytes received
          bytes_received += got;
        }
        else if (got == 0)
        {
          if (ctx->verbose)
            multilog(log, LOG_INFO, "thread[%d]: received 0 bytes\n", recv_id);
          keep_receiving = 0;
        }
        else
        {
          multilog(log, LOG_INFO, "thread[%d]: received %ld bytes\n", recv_id, got);
          keep_receiving = 0;
          quit_threads = 1;
        }
      }

      // signal this buffer element as full (for this thread)
      pthread_mutex_lock (&(ctx->mutex));
      if (bytes_received == total_bytes_to_receive)
      {
        ctx->states[recv_id] = FULL;
      }
      else
      {
        if (got == 0)
        {
          ctx->states[recv_id] = FINISHED;
        }
        else
        {
          ctx->states[recv_id] = ERROR;
        }
      }

      // signal all threads waiting on the cond to wake
      pthread_cond_broadcast (&ctx->cond);
      // release the mutex
      pthread_mutex_unlock (&ctx->mutex);

    } // while keep_receiving
  } // the observation has finished

  ctx->results[recv_id] = 0;
  pthread_exit((void *) &(ctx->results[recv_id]));
}


/*! Simple signal handler for SIGINT */
void signal_handler(int signalValue)
{
  quit_threads = 1;
  exit(EXIT_FAILURE);
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Node configuration */
  bpsr_ct_recv_t ctx;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  int verbose = 0;

  /* Quit flag */
  int quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  char * cornerturn_cfg = 0;

  int recv_id = -1;

  int arg = 0;

  unsigned i = 0;

  while ((arg=getopt(argc,argv,"hk:sv")) != -1)
  {
    switch (arg) 
    {
      case 'h':
        usage ();
        return 0;
      
      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr,"bpsr_nicdb: could not parse key from %s\n",optarg);
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
    fprintf(stderr, "ERROR: 2 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  recv_id = atoi(argv[optind]);

  cornerturn_cfg = strdup(argv[optind+1]);

  // do not use the syslog facility
  log = multilog_open ("bpsr_nicdb", 0);
  multilog_add (log, stderr);
  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = bpsr_nicdb_open;
  client->io_function       = bpsr_nicdb_recv;
  client->io_block_function = bpsr_nicdb_recv_block;
  client->close_function    = bpsr_nicdb_close;
  client->direction         = dada_client_writer;

  client->quiet = 1;
  client->context = &ctx;

  ctx.log = log;
  ctx.verbose = verbose;

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  // parse the cornerturn configuration
  if (verbose)
    multilog (log, LOG_INFO, "main: bpsr_setup_recv (%d)\n", recv_id);
  ctx.conn_info = bpsr_setup_recv (cornerturn_cfg, &(ctx.ct), recv_id);
  if (!ctx.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn config\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  // start the receiving threads
  if (verbose)
    multilog (log, LOG_INFO, "main: bpsr_nicdb_start_threads()\n");
  if (bpsr_nicdb_start_threads (&ctx, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to open IB connections\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "dada_client_write()\n");
    if (dada_client_write (client) < 0)
    {
      multilog (log, LOG_ERR, "dada_client_write() failed\n");
      client->quit = 1;
    }

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_write()\n");
    if (dada_hdu_unlock_write (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      client->quit = 1;
    }

    if (quit || ctx.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (verbose)
        multilog (log, LOG_INFO, "main: dada_hdu_lock_write()\n");
      if (dada_hdu_lock_write (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        client->quit = 1;
        break;
      }
    }
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: dada_hdu_disconnect()\n");
  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "main: bpsr_nicdb_destroy()\n");
  if (bpsr_nicdb_destroy (&ctx) < 0)
  {
    multilog(log, LOG_ERR, "bpsr_nicdb_destroy failed\n");
  }

  return EXIT_SUCCESS;
}

