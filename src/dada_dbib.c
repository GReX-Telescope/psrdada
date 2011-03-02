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

  char * header;

  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_cm_t * ib_cm;

  dada_ib_mb_t * header_mb;

} dada_dbib_t;

#define DADA_DBIB_INIT { "", 0, 0, 0, 0, 0, "", "", 0, 0}

/*! Function that opens the data transfer target */
int dbib_open (dada_client_t* client)
{

  /* the dada_dbib specific data */
  assert (client != 0);
  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  /* the ib communcation manager */
  assert(dbib->ib_cm != 0);
  dada_ib_cm_t * ib_cm = dbib->ib_cm;
  
  /* the header */
  assert(client->header != 0);
  char* header = client->header;

  /* observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN] = "";

  /* size of each transfer in bytes, as defined by TRANSFER_SIZE attribute */
  uint64_t xfer_size = 0;
 
  if (!dbib->connected)
  {

    // connect to the server and retrieve the remote datablock addresses
    /*dbib->remote_blocks = dada_ib_connect(ib_cm, client->log);
    if (!dbib->remote_blocks)
    {
      multilog(client->log, LOG_ERR, "dada_ib_connect failed\n");
      return -1;
    }
    */
    if (dada_ib_connect(ib_cm) < 0) 
    {
      multilog(client->log, LOG_ERR, "open: dada_ib_connect failed\n");
      return -1;
    }

    dbib->connected = 1;
  }

  /* Get the observation ID */
  if (ascii_header_get (header, "OBS_ID", "%s", obs_id) != 1) {
    multilog (client->log, LOG_WARNING, "Header with no OBS_ID\n");
    strcpy (obs_id, "UNKNOWN");
  }

  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = 0;

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
    multilog(log, LOG_INFO, "dbib_close()\n");

  if (bytes_written < client->transfer_bytes) {
    multilog (log, LOG_INFO, "Transfer stopped early at %"PRIu64" bytes\n",
	      bytes_written);

    if (ascii_header_set (client->header, "TRANSFER_SIZE", "%"PRIu64,
			  bytes_written) < 0)  {
      multilog (client->log, LOG_ERR, "Could not set TRANSFER_SIZE\n");
      return -1;
    }
  }

  if (dbib->verbose)
    multilog (log, LOG_INFO, "close: dada_ib_wait_recv sync_from\n");
  if (dada_ib_wait_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
    return -1;
  } 

  /*
  ib_cm->sync_to_val[0] = 0;
  ib_cm->sync_to_val[1] = 0;
  if (dada_ib_post_send(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_post_send failed\n");
    return -1;
  }

  // wait for confirmation of the send
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_wait_recv failed\n");
    return -1;
  }
  */
  return 0;
}

/*! transfer data to server. used for sending header only */
int64_t dbib_send(dada_client_t* client, void * buffer, uint64_t bytes)
{

  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (dbib->header_mb->size != bytes)
  {
    multilog (log, LOG_ERR, "header was %"PRIu64" bytes, expected %"PRIu64"\n", bytes, dbib->header_mb->size);
    return -1;
  }

  // prepost a recieve on the sync_from memory buffer
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_post_recv failed\n");
    return -1;
  }

  // copy the header to the header memory buffer
  memcpy (dbib->header_mb->buffer, buffer, bytes);

  // send the header memory buffer to dada_ibdb
  if (dada_ib_post_send(ib_cm, dbib->header_mb) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_post_send failed\n");
    return -1;
  }

  // wait for send confirmation
  if (dada_ib_wait_recv(ib_cm, dbib->header_mb) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
    return -1;
  }

  if (dbib->verbose > 1)
    multilog(log, LOG_INFO, "dbib_send: returning %"PRIu64" bytes\n", bytes);

  return bytes;

}


/*! Pointer to the function that transfers data to the target, 1 data block at a time */
int64_t dbib_send_block(dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  
  dada_dbib_t* dbib = (dada_dbib_t*) client->context;

  dada_ib_cm_t * ib_cm = dbib->ib_cm;

  multilog_t* log = client->log;

  if (dbib->verbose > 1)
    multilog(log, LOG_INFO, "dbib_send_block: buffer=%p, bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, bytes, block_id);

  int64_t start_byte;
  uint64_t remote_block_id;

  // wait for the remote block ID to be sent from dada_ibdb
  if (dada_ib_wait_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_wait_recv failed\n");
    return -1;
  }

  //uint64_t remote_buf_va = ib_cm->sync_from_val[0];
  uintptr_t remote_buf_va = (uintptr_t) ib_cm->sync_from_val[0];
  uint32_t remote_buf_rkey = (uint32_t) ib_cm->sync_from_val[1];

  if (dbib->verbose)
    multilog(log, LOG_INFO, "dbib_send_block: local_block_id=%"PRIu64", remote_buf_va=%p remote_buf_rkey=%p\n", block_id, remote_buf_va, remote_buf_rkey);

  // transmit the local data block to the remote end via RDMA
  if (dada_ib_post_sends(ib_cm, buffer, bytes, dbib->chunk_size,
                         ib_cm->local_blocks[block_id].buf_lkey,
                         remote_buf_rkey, remote_buf_va) < 0) 
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_post_sends failed\n");
    return -1;
  }

  // prepost a recv for the remote datablock id for the next transfer
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_post_recv failed\n");
    return -1;
  }

  if (dbib->verbose)
    multilog(log, LOG_INFO, "dbib_send_block: sent %"PRIu64" bytes\n", bytes);

  // send the number of valid bytes transferred to dada_ibdb
  ib_cm->sync_to_val[0] = bytes;
  if (dada_ib_post_send(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_post_send failed\n");
    return -1;
  }
  
  // wait for confirmation of the send
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(log, LOG_ERR, "dbib_send_block: dada_ib_wait_recv failed\n");
    return -1;
  }

  return bytes;
}

/*
 * required initialization of IB device and associate verb structs
 */
dada_ib_cm_t * dbib_ib_init(dada_dbib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{

  uint64_t nbufs = 0;
  uint64_t bufsz = 0;
  char ** db_buffers = 0;
  char ** hb_buffers = 0;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "dbib_ib_init()\n");

  assert (ctx != 0);
  assert (hdu != 0);

  // get the datablock addresses for memory registration
  db_buffers = dada_hdu_db_addresses(hdu, &nbufs, &bufsz);

  // check that the chunk size is a factor of DB size
  if (bufsz % ctx->chunk_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: chunk size [%d] was not a factor "
             "of data block size [%"PRIu64"]\n", ctx->chunk_size, bufsz);
    return 0;
  }

  ctx->chunks_per_block = bufsz / ctx->chunk_size;

  // create the CM and CM channel
  if (ctx->verbose)
    multilog(log, LOG_INFO, "init: dada_ib_create_cm()\n");

  dada_ib_cm_t * ib_cm = dada_ib_create_cm(nbufs, log);
  if (!ib_cm)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm failed\n");
    return 0;
  }

  ib_cm->verbose = ctx->verbose;

  // resolve the route to the server
  if (ctx->verbose)
    multilog(log, LOG_INFO, "init: dada_ib_connect_cm(%s, %d)\n", ctx->host, ctx->port);
  if (dada_ib_connect_cm(ib_cm, ctx->host, ctx->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_connect_cm failed\n");
    return 0;
  }

  // create the IB verb structures necessary
  if (ctx->verbose)
    multilog(log, LOG_INFO, "init: dada_ib_create_verbs(%d)\n", (ctx->chunks_per_block+1));
  if (dada_ib_create_verbs(ib_cm, (ctx->chunks_per_block+1)) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_verbs failed\n");
    return 0;
  }

  // register each data block buffer for use with IB transport
  int flags = IBV_ACCESS_LOCAL_WRITE;
  if (dada_ib_reg_buffers(ib_cm, db_buffers, bufsz, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_register_memory_buffers failed\n");
    return 0;
  }

  // register a local buffer for the header
  hb_buffers = dada_hdu_hb_addresses(hdu, &nbufs, &bufsz);
  ctx->header = (char *) malloc(sizeof(char) * bufsz);
  ctx->header_mb = dada_ib_reg_buffer(ib_cm, ctx->header, bufsz, flags);
  if (!ctx->header_mb)
  {
    multilog(log, LOG_ERR, "ib_init: could not register header mb\n");
    return 0;
  }
  ctx->header_mb->wr_id = 100000;

  // create the Queue Pair
  if (dada_ib_create_qp (ib_cm, (ctx->chunks_per_block+1), 1) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp (%p, %"PRIu64", 1) failed\n", 
             ib_cm, (ctx->chunks_per_block+1));
    return 0;
  }

  return ib_cm;
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

  fprintf(stderr, "dbib_ib_init()\n");
  // Init IB network
  dbib.ib_cm = dbib_ib_init (&dbib, hdu, log);
  if (!dbib.ib_cm)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  fprintf(stderr, "dbib_ib_init done\n");
  while (!client->quit)
  {
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (quit) 
      client->quit = 1;
  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_ib_dereg_buffer(dbib.header_mb) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_dereg_buffer failed\n"); 
  }

  if (dada_ib_client_destroy(dbib.ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_client_destory failed\n");
  }


  free(dbib.header);

  return EXIT_SUCCESS;
}
