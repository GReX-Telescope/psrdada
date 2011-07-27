/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "dada_ib.h"

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

// globals
int quit_signal = 0;

void usage()
{
  fprintf (stdout,
	   "dada_dbib [options] hostname\n"
     " -c <bytes>        default chunk size for IB transport [default %d]\n"
	   " -d                run as daemon\n" 
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -p <port>         port on which to connect [default %d]\n"
     " -s                single transfer only\n"
     " -v                verbose output\n"
     " -h                print help text\n", 
     DADA_IB_DEFAULT_CHUNK_SIZE, 
     DADA_DEFAULT_BLOCK_KEY, 
     DADA_DEFAULT_IBDB_PORT);
}

typedef struct {

  // machine to which data will be written
  const char * host;

  // port to connect to machine 
  unsigned port;

  // chunk size for IB transport
  unsigned chunk_size; 

  // number of chunks in a data block buffer
  unsigned chunks_per_block; 

  unsigned connected;

  unsigned verbose;

  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_cm_t * ib_cm;

} dada_dbib_t;

#define DADA_DBIB_INIT { "", 0, 0, 0, 0, 0, "",  0}

/*! Function that opens the data transfer target */
int dbib_open (dada_client_t* client)
{

  // the dada_dbib specific data
  assert (client != 0);
  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  // the ib communcation manager
  assert(dbib->ib_cm != 0);
  dada_ib_cm_t * ib_cm = dbib->ib_cm;
  
  if (dbib->verbose)
    multilog (client->log, LOG_INFO, "dbib_open()\n");

  // the header
  assert(client->header != 0);
  char* header = client->header;

  // observation id, as defined by OBS_ID attribute
  char obs_id [DADA_OBS_ID_MAXLEN] = "";

  uint64_t transfer_size;
  if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIu64, &transfer_size) != 1)
  {
    multilog (client->log, LOG_INFO, "open: TRANSFER_SIZE not set in header\n");
    transfer_size = 0;
  }

  uint64_t obs_offset = 0;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (client->log, LOG_WARNING, "open: OBS_OFFSET not set in header\n");
    transfer_size = 0;
  }

  int64_t obs_xfer= 0;
  if (ascii_header_get (client->header, "OBS_XFER", "%"PRIi64, &obs_xfer) != 1)
  {
    multilog (client->log, LOG_WARNING, "open: OBS_XFER not set in header\n");
    obs_xfer = -2;
  }

  char utc_start[64];
  if (ascii_header_get (client->header, "UTC_START", "%s", utc_start) != 1) 
  {
    multilog (client->log, LOG_WARNING, "Header with no UTC_START\n");
    strcpy (utc_start, "UNKNOWN");
  }

  if (dbib->verbose)
    multilog (client->log, LOG_INFO, "open: UTC_START=%s OBS_OFFSET=%"PRIu64" OBS_XFER=%"PRIi64" TRANSFER_SIZE=%"PRIu64"\n",
              utc_start, obs_offset, obs_xfer, transfer_size);

  // assume we do not know how many bytes to transfer
  client->transfer_bytes = transfer_size;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int dbib_close (dada_client_t* client, uint64_t bytes_written)
{

  /* the dada_dbib specific data */
  dada_dbib_t* dbib = 0;

  /* status and error logging facility */
  multilog_t* log;

  dbib = (dada_dbib_t*) client->context;
  assert (dbib != 0);

  log = client->log;
  assert (log != 0);

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  if (dbib->verbose)
    multilog(log, LOG_INFO, "dbib_close() bytes_written=%"PRIu64"\n", bytes_written);

  // if we have transferred less than we anticipated, we will need to tell ibdb that the xfer is ending
  if (client->transfer_bytes && (bytes_written != client->transfer_bytes)) 
  {
    // since we posted a recv in the send_block function, we need to wait for that recv in the cleanup
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: recv_message on sync_from [READY]\n");
    if (dada_ib_recv_message (ib_cm, DADA_IB_READY_KEY) < 0)
    {
      multilog(log, LOG_ERR, "close: recv_message on sync_from [READY] failed\n");
      return -1;
    }

    // ibdb will be waiting for the next block, send 0 bytes to xfer as an EOD message 
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: sending EOD via sync_to\n");
    if (dada_ib_send_message (ib_cm, DADA_IB_BYTES_TO_XFER_KEY, 0) < 0)
    {
      multilog(log, LOG_ERR, "close: send_message on sync_to [EOD] failed\n");
      return -1;
    }
  }
  else
  {
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: no need to preform closing steps\n");
  }
  
  if (bytes_written < client->transfer_bytes) {
    multilog (log, LOG_INFO, "Transfer stopped early at %"PRIu64" bytes\n",
              bytes_written);
  }

  return 0;

}

/*! transfer data to ibdb. used for sending header only */
int64_t dbib_send (dada_client_t* client, void * buffer, uint64_t bytes)
{

  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (ib_cm->header_mb->size != bytes)
  {
    multilog (log, LOG_ERR, "header was %"PRIu64" bytes, expected %"PRIu64"\n", bytes, ib_cm->header_mb->size);
    return -1;
  }

  // wait on sync from the the ready message, to ensure that ibdb is ready
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: recv_message on sync_from [READY]\n");
  if (dada_ib_recv_message (ib_cm, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "send: recv_message on sync_from [READY] failed\n");
    return -1;
  }

  // post recv on sync_from for the ready message
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: post_recv on sync_from [READY]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send: dada_ib_post_recv on sync_from [READY] failed\n");
    return -1;
  }

  // copy the header to the header memory buffer
  memcpy (ib_cm->header_mb->buffer, buffer, bytes);

  // send the header memory buffer to dada_ibdb
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: post_send on header_mb\n");
  if (dada_ib_post_send(ib_cm, ib_cm->header_mb) < 0)
  {
    multilog(log, LOG_ERR, "send: dada_ib_post_send on header_mb failed\n");
    return -1;
  }

  // wait for send confirmation
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: wait_send on header_mb\n");
  if (dada_ib_wait_send(ib_cm, ib_cm->header_mb) < 0)
  {
    multilog(log, LOG_ERR, "send: dada_ib_wait_send on header_mb failed\n");
    return -1;
  }

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", bytes);

  return bytes;

}


/*! Pointer to the function that transfers data to the target, 1 data block at a time */
int64_t dbib_send_block(dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  
  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: buffer=%p, bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, bytes, block_id);

  // wait for ibdb's ready handshake
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: recv_message on sync_from [READY]\n");
  if (dada_ib_recv_message (ib_cm, DADA_IB_READY_KEY) < 0)
  { 
    multilog (log, LOG_ERR, "send_block: recv_message on sync_from [READY] failed\n");
    return -1;
  }
  if (ib_cm->sync_from_val[1] != 0)
  {
    multilog (log, LOG_ERR, "send_block: recv_message on sync_from [READY] val wrong: %"PRIu64"\n",
              ib_cm->sync_from_val[1]);
    return -1;
  }

  // post a recv for the remote db buffer
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv on sync_from [BLOCK ID]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: post_recv sync_from failed [BLOCK ID]\n");
    return -1;
  }

  if (dbib->verbose && bytes && bytes != ib_cm->bufs_size)
    multilog(log, LOG_INFO, "send_block: bytes=%"PRIu64", ib_cm->bufs_size=%"PRIu64"\n", bytes, ib_cm->bufs_size);

  // tell ibdb how many bytes we are sending
  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "send_block: send_message on sync_to [BYTES TO XFER]=%"PRIu64"\n", bytes);
  if (dada_ib_send_message (ib_cm, DADA_IB_BYTES_TO_XFER_KEY, bytes) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_message on sync_to [BYTES TO XFER] failed\n");
    return -1;
  }

  // wait for the remote buffer information to be sent from ibdb
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: recv_message [BLOCK ID]\n");
  if (dada_ib_recv_message (ib_cm, 0) < 0)
  {
    multilog(log, LOG_ERR, "send_block: recv_message [BLOCK ID] failed\n");
    return -1;
  }

  uintptr_t remote_buf_va = (uintptr_t) ib_cm->sync_from_val[0];
  uint32_t remote_buf_rkey = (uint32_t) ib_cm->sync_from_val[1];

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: local_block_id=%"PRIu64", remote_buf_va=%p, "
             "remote_buf_rkey=%p\n", block_id, remote_buf_va, remote_buf_rkey);

  // transmit the local data block to the remote end via RDMA
  if (dada_ib_post_sends (ib_cm, buffer, bytes, dbib->chunk_size,
                          ib_cm->local_blocks[block_id].buf_lkey,
                          remote_buf_rkey, remote_buf_va) < 0)
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_post_sends_gap failed\n");
    return -1;
  }

  // prepost a recv for the next READY message
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv on sync_from [READY]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: post_recv on sync_from [READU] failed\n");
    return -1;
  }

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: sent %"PRIu64" bytes\n", bytes);

  // send the number of valid bytes transferred to dada_ibdb
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: send_message on sync_to [BYTES XFERED]\n");
  if (dada_ib_send_message (ib_cm, DADA_IB_BYTES_XFERRED_KEY, bytes) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_message on sync_to [BYTES XFERED] failed\n");
    return -1;
  }

  if (dbib->verbose)
   multilog(log, LOG_INFO, "send_block: notified ibdb that %"PRIu64" bytes sent\n", bytes);

  return bytes;

}

/*
 * required initialization of IB device and associate verb structs
 */
dada_ib_cm_t * dbib_ib_init(dada_dbib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{

  uint64_t db_nbufs = 0;
  uint64_t db_bufsz = 0;
  uint64_t hb_nbufs = 0;
  uint64_t hb_bufsz = 0;
  char ** db_buffers = 0;
  char ** hb_buffers = 0;

  assert (ctx != 0);
  assert (hdu != 0);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "dbib_ib_init()\n");

  // get the datablock addresses for memory registration
  db_buffers = dada_hdu_db_addresses(hdu, &db_nbufs, &db_bufsz);
  hb_buffers = dada_hdu_hb_addresses(hdu, &hb_nbufs, &hb_bufsz);

  // check that the chunk size is a factor of DB size
  if (db_bufsz % ctx->chunk_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: chunk size [%d] was not a factor "
             "of data block size [%"PRIu64"]\n", ctx->chunk_size, db_bufsz);
    return 0;
  }

  ctx->chunks_per_block = db_bufsz / ctx->chunk_size;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "ib_init: chunks_per_block=%"PRIu64"\n", ctx->chunks_per_block);

  // create the CM and CM channel
  dada_ib_cm_t * ib_cm = dada_ib_create_cm(db_nbufs, log);
  if (!ib_cm)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm failed\n");
    return 0;
  }

  ib_cm->verbose = ctx->verbose;
  ib_cm->send_depth = ctx->chunks_per_block;
  ib_cm->recv_depth = 1;
  ib_cm->port = ctx->port;
  ib_cm->bufs_size = db_bufsz;
  ib_cm->header_size = hb_bufsz;
  ib_cm->db_buffers = db_buffers;

  // resolve the route to the server
  if (ctx->verbose)
    multilog(log, LOG_INFO, "init: dada_ib_connect_cm(%s, %d)\n", ctx->host, ctx->port);
  if (dada_ib_connect_cm(ib_cm, ctx->host, ctx->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_connect_cm failed\n");
    return 0;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_verbs failed\n");
    return 0;
  }

  // register each data block buffer for use with IB transport
  int flags = IBV_ACCESS_LOCAL_WRITE;
  if (dada_ib_reg_buffers(ib_cm, db_buffers, db_bufsz, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_register_memory_buffers failed\n");
    return 0;
  }

  // register a local buffer for the header
  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "ib_init: could not allocate memory for header\n");
    return 0;
  }

  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_ERR, "ib_init: could not register header mb\n");
    return 0;
  }
  ib_cm->header_mb->wr_id = 100000;

  // create the Queue Pair
  if (ctx->verbose)
    multilog(log, LOG_INFO, "ib_init: dada_ib_create_qp()\n");
  if (dada_ib_create_qp (ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp (%p)\n",
             ib_cm);
    return 0;
  }

  // post recv on sync from for the ready message, prior to sending header
  if (ctx->verbose)
    multilog(log, LOG_INFO, "ib_init: post_recv on sync_from [ready new xfer]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_post_recv on sync_from [ready new xfer] failed\n");
    return 0;
  }

  // open the connection to ibdb
  if (!ctx->connected)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "ib_init: dada_ib_connect\n");

    // connect to the server
    if (dada_ib_connect(ib_cm) < 0)
    {
      multilog(log, LOG_ERR, "ib_init: dada_ib_connect failed\n");
      return 0;
    }
    ctx->connected = 1;

    if (ctx->verbose)
      multilog (log, LOG_INFO, "ib_init: connection established\n");

  }

  return ib_cm;
}


/*! Simple signal handler for SIGINT */
void signal_handler(int signalValue) {

  if (quit_signal) {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_signal = 1;

}


int main (int argc, char **argv)
{
  /* DADA Data Block to Node configuration */
  dada_dbib_t dbib = DADA_DBIB_INIT;

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

  /* Destination port */
  dbib.port = DADA_DEFAULT_IBDB_PORT;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* chunk size for IB transport */
  unsigned chunk_size = DADA_IB_DEFAULT_CHUNK_SIZE;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:dk:p:svh")) != -1)
  {
    switch (arg) {
      
    case 'c':
      if (optarg)
      {
        chunk_size = atoi(optarg);
        break;
      }
      else
      {
        fprintf(stderr, "ERROR: no chunk size specified\n");
        usage();
        return EXIT_FAILURE;
      } 

    case 'd':
      daemon=1;
      break;
      
    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_dbib: could not parse key from %s\n",optarg);
        return EXIT_FAILURE;
      }
      break;

    case 'p':
      if (optarg)
      {
        dbib.port = atoi(optarg);
        break;
      }
      else
      {
        fprintf(stderr, "ERROR: no port specified with -p flag\n");
        usage();
        return EXIT_FAILURE;
      }

    case 's':
      quit = 1;
      break;

    case 'v':
      verbose++;
      break;

    case 'h':
      usage ();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 1) {
    fprintf(stderr, "ERROR: 1 command line argument is required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  dbib.host = strdup(argv[optind]);

  log = multilog_open ("dada_dbib", daemon);

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

  client->open_function     = dbib_open;
  client->io_function       = dbib_send;
  client->io_block_function = dbib_send_block;
  client->close_function    = dbib_close;
  client->direction         = dada_client_reader;

  client->context = &dbib;

  dbib.chunk_size = chunk_size;
  dbib.verbose = verbose;

  if (verbose)
    multilog(client->log, LOG_INFO, "main: dbib_ib_init()\n");

  // Init IB network
  dbib.ib_cm = dbib_ib_init (&dbib, hdu, log);
  if (!dbib.ib_cm)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  while (!client->quit)
  {

    if (verbose)
      multilog(client->log, LOG_INFO, "main: dada_client_read()\n");
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog(client->log, LOG_INFO, "main: dada_unlock_read()\n");
    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      quit = 1;
    }

    if (quit) 
      client->quit = 1;
    else
    {
      if (verbose)
        multilog(client->log, LOG_INFO, "main: dada_lock_read()\n");
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (verbose)
    multilog(client->log, LOG_INFO, "main: dada_hdu_disconnect()\n");

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog(client->log, LOG_INFO, "main: dada_ib_client_destroy()\n");
  if (dada_ib_client_destroy(dbib.ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_client_destroy failed\n");
  }

  return EXIT_SUCCESS;
}
