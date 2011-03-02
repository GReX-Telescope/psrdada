#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "dada_ib.h"

#include "ascii_header.h"
#include "daemon.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>


void usage()
{
  fprintf (stdout,
     "dada_ibdb [options]\n"
     " -c <bytes>    default chunk size for IB transport [default: %d]\n"
     " -d            run as daemon\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -p <port>     port on which to listen [default: %d]\n"
     " -s            single transfer only\n"
     " -d            run as daemon\n",
     DADA_IB_DEFAULT_CHUNK_SIZE,
     DADA_DEFAULT_BLOCK_KEY,
     DADA_DEFAULT_IBDB_PORT);
}


typedef struct dada_ibdb {
  
  // port to listen on for connections 
  unsigned port;

  // chunk size for IB transport
  unsigned chunk_size; 

  // number of chunks in a data block buffer
  unsigned chunks_per_block;

  // verbose messages
  char verbose;

  // flag for active RDMA connection
  unsigned connected;
    
  char * header;
  
  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_cm_t * ib_cm;

  // memory block for transfer of the header 
  dada_ib_mb_t * header_mb;

} dada_ibdb_t;

#define DADA_IBDB_INIT { 0, 0, 0, 0, 0, "", "", 0, 0 }

/*! transfer function for transfer of just header */
int64_t dada_ibdb_recv (dada_client_t* client, void* data, uint64_t data_size)
{

  dada_ibdb_t * ibdb = (dada_ibdb_t *) client->context;

  dada_ib_cm_t * ib_cm = ibdb->ib_cm;

  multilog_t* log = client->log;

  if (ibdb->verbose)
    multilog (log, LOG_INFO, "dada_ibdb_recv()\n");

  // check that the header size matches the requested data_size
  if (ibdb->header_mb->size != data_size)
  {
    multilog (client->log, LOG_ERR, "recv: header was %"PRIu64" bytes, "
              "expected %"PRIu64"\n", data_size, ibdb->header_mb->size);
    return -1;
  }

  // n.b. dont have to post recv for the header, this was done in open fn

  if (ibdb->verbose > 1)
    multilog(client->log, LOG_INFO, "recv: dada_ib_wait_recv\n");

  // wait for the transfer of the header
  if (dada_ib_wait_recv(ib_cm, ibdb->header_mb) < 0)
  {
    multilog(client->log, LOG_ERR, "recv: dada_ib_wait_recv failed\n");
    return -1;
  }

  // copy the header to specified buffer
  if (ibdb->verbose > 1)
    multilog(client->log, LOG_INFO, "recv: memcpy %"PRIu64" bytes\n", data_size);

  memcpy (data, ibdb->header_mb->buffer, data_size);

  if (ibdb->verbose > 1)
    multilog(log, LOG_INFO, "recv: returning %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t dada_ibdb_recv_block (dada_client_t* client, void* data, 
                         uint64_t data_size, uint64_t block_id)
{

  dada_ibdb_t * ibdb = (dada_ibdb_t *) client->context;

  dada_ib_cm_t * ib_cm = ibdb->ib_cm;

  // instruct the client to fill the specified block_id remotely via RDMA
  ib_cm->sync_to_val[0] = (uint64_t) ib_cm->local_blocks[block_id].buf_va;
  ib_cm->sync_to_val[1] = (uint64_t) ib_cm->local_blocks[block_id].buf_rkey;

  multilog(client->log, LOG_INFO, "recv_block: block_id=%"PRIu64", buf_va=%p, buf_rkey=%p\n",
           block_id, ib_cm->sync_to_val, (ib_cm->sync_to_val +8));

  if (ibdb->verbose > 1)
    multilog(client->log, LOG_INFO, "recv_block: data=%p, bytes=%"PRIu64", "
             "block_id=%"PRIu64"\n", data, data_size, block_id);

  // pre post recv for the client's completion message 
  if (dada_ib_post_recv(ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(client->log, LOG_ERR, "recv_block: dada_ib_wait_recv failed\n");
    return -1;
  }

  // send the client the sync value (buffer ID to be filled)
  if (dada_ib_post_send (ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(client->log, LOG_ERR, "recv_block: dada_ib_post_send failed\n");
    return -1;
  }

  // confirm that the sync value has been sent 
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_to) < 0)
  {
    multilog(client->log, LOG_ERR, "recv_block: dada_ib_wait_recv failed\n");
    return -1;
  }

  // remote RDMA transfer is ocurring now...

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv_block: waiting for completion "
             "on block %"PRIu64"\n", block_id);

  // accept the completion message for the "current" transfer
  if (dada_ib_wait_recv(ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(client->log, LOG_ERR, "recv_block: dada_ib_wait_recv failed\n");
    return -1;
  }

  uint64_t bytes_received = ib_cm->sync_from_val[0];

  if (ibdb->verbose > 1)
    multilog(client->log, LOG_INFO, "recv_block: bytes transferred=%"PRIu64"\n", 
             bytes_received);

  return (int64_t) bytes_received;

}


/*! Function that closes the data file */
int dada_ibdb_close (dada_client_t* client, uint64_t bytes_written)
{

  dada_ibdb_t * ibdb = (dada_ibdb_t *) client->context;

  if (ibdb->verbose)
    multilog (client->log, LOG_INFO, "dada_ibdb_close()\n");

  // need to post_send on the sync_to buffer as dbib will be waiting on this
  ibdb->ib_cm->sync_to_val[0] = 0;
  ibdb->ib_cm->sync_to_val[1] = 0;
  if (dada_ib_post_send (ibdb->ib_cm, ibdb->ib_cm->sync_to) < 0)
  {
    multilog(client->log, LOG_ERR, "close: dada_ib_post_send on sync_to failed\n");
    return -1;
  }

  if (ascii_header_set (client->header, "TRANSFER_SIZE", "%"PRIu64, bytes_written) < 0)  
  {
    multilog (client->log, LOG_ERR, "close: could not set TRANSFER_SIZE\n");
    return -1;
  }

  return 0;
}

/*! Function that opens the data transfer target */
int dada_ibdb_open (dada_client_t* client)
{

  assert (client != 0);

  dada_ibdb_t * ibdb = (dada_ibdb_t *) client->context;

  dada_ib_cm_t * ib_cm = ibdb->ib_cm;

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "dada_ibdb_open()\n");

  // get the next data block buffer to be written for first post_recv
  uint64_t block_id = ipcbuf_get_write_index ((ipcbuf_t *) client->data_block);
  ib_cm->sync_to_val[0] = (uint64_t) ib_cm->local_blocks[block_id].buf_va;
  ib_cm->sync_to_val[1] = (uint64_t) ib_cm->local_blocks[block_id].buf_rkey;
  
  multilog(client->log, LOG_INFO, "open: block_id=%"PRIu64", buf_va=%p, buf_rkey=%p\n",
           block_id, ib_cm->sync_to_val[0], ib_cm->sync_to_val[1]);


  // pre-post receive for the header transfer 
  if (dada_ib_post_recv(ib_cm, ibdb->header_mb) < 0)
  {
    multilog(client->log, LOG_ERR, "open: dada_ib_wait_recv failed\n");
    return -1;
  }

  if (!ibdb->connected)
  {
    // accept the proper IB connection
    if (dada_ib_accept (ib_cm) < 0)
    {
      multilog(client->log, LOG_ERR, "open: dada_ib_accept failed\n");
      return -1;
    }
    ibdb->connected = 1;
  }

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "dada_ibdb_open() returns\n");

  return 0;
}

/*
 * required initialization of IB device and associate verb structs
 */
dada_ib_cm_t * dada_ibdb_ib_init(dada_ibdb_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{

  uint64_t nbufs = 0;
  uint64_t bufsz = 0;
  char ** db_buffers = 0;

  assert (ctx != 0);
  assert (hdu != 0);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "dada_ibdb_ib_init()\n");

  // get the information about the data block
  db_buffers = dada_hdu_db_addresses(hdu, &nbufs, &bufsz);

  // this a strict requirement at this stage
  if (bufsz % ctx->chunk_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: chunk size [%d] was not a factor "
             "of data block size[%"PRIu64"]\n", ctx->chunk_size, bufsz);
    return 0;
  }

  ctx->chunks_per_block = bufsz / ctx->chunk_size;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "ib_init: dada_ib_create_cm\n");
  dada_ib_cm_t * ib_cm = dada_ib_create_cm(nbufs, log);
  if (!ib_cm)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm failed\n");
    return 0; 
  }

  ib_cm->verbose = ctx->verbose;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "ib_init: dada_ib_listen_cm\n");
  // listen for a connection request on the specified port
  if (dada_ib_listen_cm(ib_cm, ctx->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_listen_cm failed\n");
    return 0;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm,  (ctx->chunks_per_block+1)) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_verbs failed\n");
    return 0;
  }

  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

  if (dada_ib_reg_buffers(ib_cm, db_buffers, bufsz, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_register_memory_buffers failed\n");
    return 0;
  }

  // create and register a buffer for the header
  char ** header_block_buffers = dada_hdu_hb_addresses(hdu, &nbufs, &bufsz);
  ctx->header = (char *) malloc(sizeof(char) * bufsz);
  ctx->header_mb = dada_ib_reg_buffer(ib_cm, ctx->header, bufsz, flags);
  if (!ctx->header_mb)
  {
    multilog(log, LOG_ERR, "ib_init: could not register header mb\n");
    return 0;
  }
  ctx->header_mb->wr_id = 100000;

  if (dada_ib_create_qp (ib_cm, 1, 1) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp failed\n");
    return 0;
  }

  return ib_cm;

}


int main (int argc, char **argv)
{

  /* IB DB configuration */
  dada_ibdb_t ibdb = DADA_IBDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* port on which to listen for incoming connections */
  int port = DADA_DEFAULT_IBDB_PORT;

  /* chunk size for IB transport */
  unsigned chunk_size = DADA_IB_DEFAULT_CHUNK_SIZE;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:dk:p:sv")) != -1)
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
        fprintf (stderr,"dada_ibdb: could not parse key from %s\n",optarg);
        return EXIT_FAILURE;
      }
      break;
      
    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      quit = 1;
      break;

    case 'v':
      verbose++;
      break;
      
    default:
      usage ();
      return 0;
      
    }
  }

  log = multilog_open ("dada_ibdb", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_IBDB_LOG);
  }
  else
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

  client->open_function     = dada_ibdb_open;
  client->io_function       = dada_ibdb_recv;
  client->io_block_function = dada_ibdb_recv_block;
  client->close_function    = dada_ibdb_close;
  client->direction         = dada_client_writer;

  client->context = &ibdb;

  // initialize IB resources
  ibdb.chunk_size = chunk_size;
  ibdb.verbose = verbose;
  ibdb.port = port;

  ibdb.ib_cm = dada_ibdb_ib_init (&ibdb, hdu, log);
  if (!ibdb.ib_cm)
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");

  while (!client->quit) {

    if (dada_client_write (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (quit) {
      client->quit = 1;
    }

  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_ib_dereg_buffer(ibdb.header_mb) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_dereg_buffer failed\n");
  }

  if (dada_ib_destroy(ibdb.ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "dada_ib_client_destory failed\n");
  }


  return EXIT_SUCCESS;
}
