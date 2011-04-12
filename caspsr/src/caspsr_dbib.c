/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "dada_ib.h"
#include "caspsr_def.h"

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
#include <signal.h>

// Globals
int quit_signal = 0;

void usage()
{
  fprintf (stdout,
	   "caspsr_dbib [options] hostname idistrib ndistrib\n"
     " -c <bytes>        default chunk size for IB transport [default %d]\n"
	   " -d                run as daemon\n" 
     " -h                print this usage\n"
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -p <port>         port on which to connect [default %d]\n"
     " -s                single transfer only\n"
     " -S                single transfer with multiple xfers\n"
     " -v                verbose output\n",
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

  // which distributor we are 
  unsigned i_distrib;

  // total number of distributors
  unsigned n_distrib;

  unsigned connected;

  unsigned verbose;

  char * header;

  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  /* signal quit based when OBS_XFER == -1 */
  int quit;

  // Infiniband Connection Manager
  dada_ib_cm_t * ib_cm;

} caspsr_dbib_t;

#define CASPSR_DBIB_INIT { "", 0, 0, 0, 0, 0, 0, 0, "", "", 0, 0 }

/*! Function that opens the data transfer target */
int caspsr_dbib_open (dada_client_t* client)
{

  // the caspsr_dbib specific data
  assert (client != 0);
  caspsr_dbib_t* dbib = (caspsr_dbib_t*) client->context;

  // the ib communcation manager
  assert(dbib->ib_cm != 0);
  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  if (dbib->verbose)
    multilog (client->log, LOG_INFO, "caspsr_dbib_open()\n");
  
  // the header
  assert(client->header != 0);
  char* header = client->header;

  // observation id, as defined by OBS_ID attribute
  char obs_id [DADA_OBS_ID_MAXLEN] = "";

  int64_t obs_xfer = 0;
  uint64_t obs_offset = 0;
 
  // Get the observation ID 
  if (ascii_header_get (header, "OBS_XFER", "%"PRIi64, &obs_xfer) != 1) {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_XFER\n");
  }

  // signal that this is the final xfer
  if (obs_xfer == -1)
    dbib->quit = 1;

  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1) {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_OFFSET\n");
  }

  multilog (client->log, LOG_INFO, "open: OBS_XFER=%"PRIi64", OBS_OFFSET=%"PRIu64"\n", obs_xfer, obs_offset);

  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = 0;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int caspsr_dbib_close (dada_client_t* client, uint64_t bytes_written)
{

  /* the caspsr_dbib specific data */
  caspsr_dbib_t* dbib = 0;

  /* status and error logging facility */
  multilog_t* log;

  dbib = (caspsr_dbib_t*) client->context;
  assert (dbib != 0);

  log = client->log;
  assert (log != 0);

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  if (dbib->verbose)
    multilog(log, LOG_INFO, "caspsr_dbib_close()\n");

  if (dbib->quit) 
  {
    if (dbib->verbose)
      multilog(log, LOG_INFO, "close: skipping comms on "
               "close as dbib->quit true\n");
  }
  else
  {

    // since we posted a recv in the send_block function, we need
    // to wait recv on that post to clean up
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: dada_ib_wait_recv on sync_from [ready]\n");
    if (dada_ib_wait_recv (ib_cm, ib_cm->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "close: dada_ib_wait_recv on sync_from [ready] failed\n");
      return -1;
    } 

    // pre-post a recv for the ready message for the next xfer
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: dada_ib_post_recv on sync_from [ready new xfer]\n");
    if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "close: dada_ib_wait_recv on sync_from [ready new xfer] failed\n");
      return -1;
    }
 
    // if the transfer happened to finish correctly, then ibdb will be
    // waiting for the next block, instruct it via sync_to vals of 0
    // that the xfer is over
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: sending 0'd sync_to\n");
    ib_cm->sync_to_val[0] = 2;
    ib_cm->sync_to_val[1] = 0;
    if (dada_ib_post_send(ib_cm, ib_cm->sync_to) < 0)
    {
      multilog(log, LOG_ERR, "close: dada_ib_post_send on sync_to failed\n");
      return -1;
    }

    // wait for confirmation of the send
    if (dbib->verbose)
      multilog (log, LOG_INFO, "close: dada_ib_wait_recv on sync_to\n");
    if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
    {
      multilog(log, LOG_ERR, "close: dada_ib_wait_recv on sync_to failed\n");
      return -1;
    }
  }

  if (bytes_written < client->transfer_bytes) {
    multilog (log, LOG_INFO, "Transfer stopped early at %"PRIu64" bytes\n",
              bytes_written);

    if (ascii_header_set (client->header, "TRANSFER_SIZE", "%"PRIu64,
        bytes_written) < 0)  {
      multilog (client->log, LOG_ERR, "close: Could not set TRANSFER_SIZE\n");
      return -1;
    }
  }

  return 0;
}

/*! transfer data to ibdb. used for sending header only */
int64_t caspsr_dbib_send (dada_client_t* client, void * buffer, uint64_t bytes)
{

  caspsr_dbib_t* dbib = (caspsr_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (dbib->verbose)
    multilog (log, LOG_INFO, "caspsr_dbib_send()\n");

  if (ib_cm->header_mb->size != bytes)
  {
    multilog (log, LOG_ERR, "send: header was %"PRIu64" bytes, expected %"PRIu64"\n", bytes, ib_cm->header_mb->size);
    return -1;
  }

  // post recv on sync from for the ready message, prior to sending header
/*
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: post_recv on sync_from [ready]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send: dada_ib_post_recv on sync_from [ready] failed\n");
    return -1;
  }
*/

  // wait recv on sync from the the ready message, to ensure that ibdb is ready
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: wait_recv on sync_from [ready]\n");
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send: wait_recv on sync_from [ready] failed\n");
    return -1;
  }
  
  // if this is a "fake" transfer, dont post_recv for the sync_from [ready], just send header  
  if (dbib->quit)
  {
    if (dbib->verbose)
      multilog (client->log, LOG_INFO, "send: skipping post_recv on sync_from for [ready] since OBS_XFER==-1\n");
  }
  else 
  {
    // post recv on sync_from for the ready message
    if (dbib->verbose)
      multilog(log, LOG_INFO, "send: post_recv on sync_from [ready 2]\n");
    if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send: dada_ib_post_recv on sync_from [ready 2] failed\n");
      return -1;
    }
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
    multilog(log, LOG_INFO, "send: wait_recv on header_mb\n");
  if (dada_ib_wait_recv(ib_cm, ib_cm->header_mb) < 0)
  {
    multilog(log, LOG_ERR, "send: dada_ib_wait_recv on header_mb failed\n");
    return -1;
  }

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", bytes);

  return bytes;

}


/*! Transfers 1 datablock at a time to caspsr_ibdb */
int64_t caspsr_dbib_send_block(dada_client_t* client, void * buffer, 
                               uint64_t bytes, uint64_t block_id)
{
  
  caspsr_dbib_t* dbib = (caspsr_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: buffer=%p, bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, bytes, block_id);

  if (dbib->quit)
  {
    if (dbib->verbose)
      multilog(log, LOG_INFO, "send_block: dbib->quit true, OBS_XFER==-1, "
               "skippping send_block\n");
    return bytes;
  }

  // wait for ibdb's ready handshake
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: wait_recv on sync_from [ready]\n");
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: wait_recv on sync_from [ready] failed\n");
    return -1;
  }

  if (ib_cm->sync_from_val[0] != 1)
  {
    multilog(log, LOG_ERR, "send_block: sync_from key[%"PRIu64"] != 1 [ready]\n",
             ib_cm->sync_from_val[0]);
    return -1;
  }

  if (ib_cm->sync_from_val[1] != 0)
  {
    multilog(log, LOG_ERR, "send_block:sync_from val[%"PRIu64"] != 0 [ready]\n",
             ib_cm->sync_from_val[1]);
    return -1;
  }

#ifdef _DEBUG
  multilog(log, LOG_INFO, "send_block: ibdb reports ready=%"PRIu64"\n", ib_cm->sync_from_val[1]);
#endif

  // post a recv for the remote db buffer
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv on sync_from [block id]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: post_recv sync_from failed [block id]\n");
    return -1;
  }

  // tell ibdb how many bytes we are sending
  ib_cm->sync_to_val[0] = 2;
  ib_cm->sync_to_val[1] = bytes;
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_send on sync_to for %"PRIu64" bytes\n", bytes);
  if (dada_ib_post_send(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "send_block: post_send on sync_to [bytes to send] failed\n");
    return -1;
  }

  // wait for confirmation of the send
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: wait_recv on sync_to\n", bytes);
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "send_block: wait_recv on sync_to [bytes to send] failed\n");
    return -1;
  }

  int64_t start_byte;

  // wait for the remote buffer information to be sent from ibdb
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: wait_recv on sync_from [block id]\n");
  if (dada_ib_wait_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_wait_recv failed\n");
    return -1;
  }

  uintptr_t remote_buf_va = (uintptr_t) ib_cm->sync_from_val[0];
  uint32_t remote_buf_rkey = (uint32_t) ib_cm->sync_from_val[1];

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: local_block_id=%"PRIu64", remote_buf_va=%p, "
             "remote_buf_rkey=%p\n", block_id, remote_buf_va, remote_buf_rkey);

  // transmit the local data block to the remote end via RDMA
  if (dada_ib_post_sends_gap (ib_cm, buffer, bytes, dbib->chunk_size,
                              ib_cm->local_blocks[block_id].buf_lkey, 
                              remote_buf_rkey, remote_buf_va,
                              dbib->i_distrib * UDP_DATA, (dbib->n_distrib - 1) * UDP_DATA) < 0) 
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_post_sends_gap failed\n");
    return -1;
  }

  // prepost a recv for the next READY message
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv on sync_from [ready]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_post_recv failed\n");
    return -1;
  }

  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: sent %"PRIu64" bytes\n", bytes);

  // send the number of valid bytes transferred to dada_ibdb
  ib_cm->sync_to_val[0] = 3;
  ib_cm->sync_to_val[1] = bytes;
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: post_send on sync_to [bytes sent]\n");
  if (dada_ib_post_send(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_post_send failed\n");
    return -1;
  }
    
  // wait for confirmation of the send
  if (dbib->verbose)
    multilog(log, LOG_INFO, "send_block: wait_recv on sync_to [bytes sent]\n");
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "send_block: dada_ib_wait_recv failed\n");
    return -1;
  }
  
  if (dbib->verbose)
   multilog(log, LOG_INFO, "send_block: notified ibdb that %"PRIu64" bytes sent\n", bytes);

  return bytes;
}

/*
 * required initialization of IB device and associate verb structs
 */
dada_ib_cm_t * caspsr_dbib_ib_init (caspsr_dbib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
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
    multilog(log, LOG_INFO, "caspsr_dbib_ib_init()\n");

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
  ib_cm->depth = ctx->chunks_per_block+1;
  ib_cm->port = ctx->port;
  ib_cm->bufs_size = db_bufsz;
  ib_cm->header_size = hb_bufsz;
  ib_cm->db_buffers = db_buffers;

  // resolve the route to the server
  if (ctx->verbose)
    multilog(log, LOG_INFO, "ib_init: connecting to cm at %s:%d\n", ctx->host, ctx->port);

  if (dada_ib_connect_cm(ib_cm, ctx->host, ctx->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_connect_cm failed\n");
    return 0;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm, ib_cm->depth) < 0)
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
  ib_cm->header_mb->wr_id = 10000;

  // create the Queue Pair
  if (dada_ib_create_qp (ib_cm, (ctx->chunks_per_block+1), 1) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp (%p, %"PRIu64", 1) failed\n", 
             ib_cm, (ctx->chunks_per_block+1));
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
  caspsr_dbib_t dbib = CASPSR_DBIB_INIT;

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

  /* Flag for quitting after multiple xfers */
  int quit_xfer = 0;

  /* Destination port */
  dbib.port = DADA_DEFAULT_IBDB_PORT;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* chunk size for IB transport */
  unsigned chunk_size = DADA_IB_DEFAULT_CHUNK_SIZE;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:dhk:p:sSv")) != -1)
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
        fprintf (stderr,"caspsr_dbib: could not parse key from %s\n",optarg);
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

    case 'S':
      quit_xfer = 1;
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
  if (argc-optind != 3) {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  dbib.host = strdup(argv[optind]);
  dbib.i_distrib = atoi(argv[optind+1]);
  dbib.n_distrib = atoi(argv[optind+2]);

  // do not use the syslog facility
  log = multilog_open ("caspsr_dbib", 0);

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

  client->open_function     = caspsr_dbib_open;
  client->io_function       = caspsr_dbib_send;
  client->io_block_function = caspsr_dbib_send_block;
  client->close_function    = caspsr_dbib_close;
  client->direction         = dada_client_reader;

  client->quiet = 1;
  client->context = &dbib;

  dbib.chunk_size = chunk_size;
  dbib.verbose = verbose;

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  // Init IB network
  dbib.ib_cm = caspsr_dbib_ib_init (&dbib, hdu, log);
  if (!dbib.ib_cm)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

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

    if (quit || (quit_xfer && dbib.quit))
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

  if (dada_ib_client_destroy(dbib.ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_client_destroy failed\n");
  }

  return EXIT_SUCCESS;
}
