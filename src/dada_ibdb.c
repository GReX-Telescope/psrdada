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

  unsigned xfer_ending;

  // Infiniband Connection Manager
  dada_ib_cm_t * ib_cm;

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
  if (ib_cm->header_mb->size != data_size)
  {
    multilog (client->log, LOG_ERR, "recv: header was %"PRIu64" bytes, "
              "expected %"PRIu64"\n", data_size, ib_cm->header_mb->size);
    return -1;
  }

  // send ready message to ibdb to inform we are ready to receive header
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv: send_message on sync_to [READY]\n");
  if (dada_ib_send_message (ib_cm, DADA_IB_READY_KEY, 0) < 0)
  {
    multilog(client->log, LOG_ERR, "recv: send_message on sync_to [READY] failed\n");
    return -1;
  }

  // wait for the transfer of the header
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv: wait_recv on header_mb\n");
  if (dada_ib_wait_recv(ib_cm, ib_cm->header_mb) < 0)
  {
    multilog(client->log, LOG_ERR, "recv: wait_recv on header_mb failed\n");
    return -1;
  }

  // post recv for the number of bytes in the first xfer
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv: post_recv on sync_from [BYTES TO XFER]\n");
  if (dada_ib_post_recv(ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(client->log, LOG_ERR, "recv: post_recv on sync_from [BYTES TO XFER] failed\n");
    return -1;
  }

  // copy the header to specified buffer
  if (ibdb->verbose > 1)
    multilog(client->log, LOG_INFO, "recv: memcpy %"PRIu64" bytes\n", data_size);

  memcpy (data, ib_cm->header_mb->buffer, data_size);

  if (ibdb->verbose)
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

  // send the ready message to dbib
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv_block: send_message on sync_to [READY]\n");
  if (dada_ib_send_message (ib_cm, DADA_IB_READY_KEY, 0) < 0) 
  {
    multilog(client->log, LOG_ERR, "recv_block: send_message on sync_to [READY] failed\n");
    return -1;
  }

  // get the number of bytes to be xferred
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv_block: recv_message on sync_from [BYTES TO XFER]\n");
  if (dada_ib_recv_message (ib_cm, DADA_IB_BYTES_TO_XFER_KEY) < 0)
  {
    multilog(client->log, LOG_ERR, "recv_block: recv_message on sync_from [BYTES TO XFER] failed\n");
    return -1;
  }
  uint64_t bytes_to_be_received = ib_cm->sync_from_val[1];
  uint64_t bytes_received = 0;

  // this is a signal for the end of data
  if (bytes_to_be_received == 0) 
  {
    ibdb->xfer_ending = 1;
    bytes_received = 0;
  }
  else
  {

    if (ibdb->verbose  && (bytes_to_be_received != ib_cm->bufs_size))
      multilog(client->log, LOG_INFO, "recv_block: bytes to be recvd=%"PRIu64"\n", bytes_to_be_received);

    // post recv for [bytes received]
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: post_recv on sync_from for [BYTES XFERRED]\n");
    if (dada_ib_post_recv(ib_cm, ib_cm->sync_from) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: post_recv on sync_from for [BYTES XFERRED] failed\n");
      return -1;
    }

    // instruct the client to fill the specified block_id remotely via RDMA
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: send_message on sync_to [BLOCK ID]\n");
    if (dada_ib_send_message (ib_cm, (uint64_t) ib_cm->local_blocks[block_id].buf_va,
                              (uint64_t) ib_cm->local_blocks[block_id].buf_rkey) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: send_message on sync_to [BLOCK ID] failed\n");
      return -1;
    }

    // remote RDMA transfer is ocurring now...
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: waiting for completion "
               "on block %"PRIu64"...\n", block_id);

    // ask the client how many bytes were received
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: recv_message on sync_from [BYTES XFERRED]\n");
    if (dada_ib_recv_message (ib_cm, DADA_IB_BYTES_XFERRED_KEY) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: recv_message on sync_from [BYTES XFERRED] failed\n");
      return -1;
    }
    bytes_received = ib_cm->sync_from_val[1];

    // post recv for the number of bytes in the next send_block transfer
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: post_recv on sync_from [BYTES TO XFER]\n");
    if (dada_ib_post_recv(ib_cm, ib_cm->sync_from) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: post_recv on sync_from [BYTES TO XFER] failed\n");
      return -1;
    }
  }

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv_block: bytes transferred=%"PRIu64"\n", 
             bytes_received);

  return (int64_t) bytes_received;

}


/*! Function that closes the data file */
int dada_ibdb_close (dada_client_t* client, uint64_t bytes_written)
{

  dada_ibdb_t * ibdb = (dada_ibdb_t *) client->context;

  dada_ib_cm_t * ib_cm = ibdb->ib_cm;

  if (ibdb->verbose)
    multilog (client->log, LOG_INFO, "dada_ibdb_close()\n");


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

  // post receive for the header transfer 
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "open: post_recv on header_mb\n");
  if (dada_ib_post_recv(ib_cm, ib_cm->header_mb) < 0)
  {
    multilog(client->log, LOG_ERR, "open: post_recv on header_mb failed\n");
    return -1;
  }

  if (!ib_cm->ib_connected)
  {
    // accept the proper IB connection
    if (dada_ib_accept (ib_cm) < 0)
    {
      multilog(client->log, LOG_ERR, "open: dada_ib_accept failed\n");
      return -1;
    }
    ib_cm->ib_connected = 1;
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

  uint64_t db_nbufs = 0;
  uint64_t db_bufsz = 0;
  uint64_t hb_nbufs = 0;
  uint64_t hb_bufsz = 0;
  char ** db_buffers = 0;
  char ** hb_buffers = 0;

  assert (ctx != 0);
  assert (hdu != 0);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "dada_ibdb_ib_init()\n");

  // get the information about the data block
  db_buffers = dada_hdu_db_addresses(hdu, &db_nbufs, &db_bufsz);

  // get the header block buffer size
  hb_buffers = dada_hdu_hb_addresses(hdu, &hb_nbufs, &hb_bufsz);

  // this a strict requirement at this stage
  if (db_bufsz % ctx->chunk_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: chunk size [%d] was not a factor "
             "of data block size[%"PRIu64"]\n", ctx->chunk_size, db_bufsz);
    return 0;
  }

  ctx->chunks_per_block = db_bufsz / ctx->chunk_size;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "ib_init: dada_ib_create_cm\n");
  dada_ib_cm_t * ib_cm = dada_ib_create_cm(db_nbufs, log);
  if (!ib_cm)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm failed\n");
    return 0; 
  }

  ib_cm->verbose = ctx->verbose;
  ib_cm->send_depth = 1;
  ib_cm->recv_depth = 1;
  ib_cm->port = ctx->port;
  ib_cm->bufs_size = db_bufsz;
  ib_cm->header_size = hb_bufsz;
  ib_cm->db_buffers = db_buffers;

  ib_cm->cm_connected = 0;
  ib_cm->ib_connected = 0;

  // listen for a connection request on the specified port
  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "ib_init: dada_ib_listen_cm\n");
  if (dada_ib_listen_cm(ib_cm, ctx->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_listen_cm failed\n");
    return 0;
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs(ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_verbs failed\n");
    return 0;
  }

  // register each data block buffer with as a MR within the PD
  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (dada_ib_reg_buffers(ib_cm, db_buffers, db_bufsz, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_register_memory_buffers failed\n");
    return 0;
  }

  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "ib_init_thread: could not allocate memory for header\n");
    return 0;
  }

  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_ERR, "ib_init: could not register header mb\n");
    return 0;
  }
  ib_cm->header_mb->wr_id = 100000;

  if (dada_ib_create_qp (ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp failed\n");
    return 0;
  }

  ib_cm->cm_connected = 1;
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

  while (!client->quit) 
  {
    if (dada_client_write (client) < 0)
    {
      multilog (log, LOG_ERR, "Error during transfer\n");
      quit = 1;
    }

    if (verbose)
      multilog (client->log, LOG_INFO, "main: dada_ib_disconnect()\n");
    if (dada_ib_disconnect(ibdb.ib_cm) < 0)
    {
      multilog(client->log, LOG_ERR, "dada_ib_disconnect failed\n");
    }

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_write()\n");
    if (dada_hdu_unlock_write (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      quit = 1;
    }

    if (quit)
      client->quit = 1;
    else
    {
      if (dada_hdu_lock_write (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }

      // reallocate IB resources
      ibdb.ib_cm = dada_ibdb_ib_init (&ibdb, hdu, log);
      if (!ibdb.ib_cm)
        multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;


  return EXIT_SUCCESS;
}
