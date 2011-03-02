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

void * caspsr_ibdb_ib_init_thread (void * arg);

void usage()
{
  fprintf (stdout,
     "caspsr_ibdb [options] n_distrib\n"
     " -c <bytes>    default chunk size for IB transport [default: %d]\n"
     " -d            run as daemon\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -p <port>     port on which to listen [default: %d]\n"
     " -s            single transfer only\n"
     " -v            verbose output\n",
     DADA_IB_DEFAULT_CHUNK_SIZE,
     DADA_DEFAULT_BLOCK_KEY,
     DADA_DEFAULT_IBDB_PORT);
}


typedef struct caspsr_ibdb {
  
  // port to listen on for connections 
  unsigned port;

  // chunk size for IB transport
  unsigned chunk_size; 

  // number of chunks in a data block buffer
  unsigned chunks_per_block;

  // verbose messages
  char verbose;

  // flag for active RDMA connection
  //unsigned * connected;

  // number of distributors
  unsigned n_distrib;
    
  // header memory buffers
  char ** headers;
  
  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  // Infiniband Connection Manager
  dada_ib_cm_t ** ib_cms;

  // memory blocks for transfer of the headers
  //dada_ib_mb_t ** header_mbs;

  multilog_t * log;

} caspsr_ibdb_t;

#define CASPSR_IBDB_INIT { 0, 0, 0, 0, 0, 0, "", 0, 0}

/*! transfer function for transfer of just header */
int64_t caspsr_ibdb_recv (dada_client_t* client, void* data, uint64_t data_size)
{

  caspsr_ibdb_t * ibdb = (caspsr_ibdb_t *) client->context;

  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  multilog_t* log = client->log;

  if (ibdb->verbose)
    multilog (log, LOG_INFO, "caspsr_ibdb_recv()\n");

  unsigned i = 0;

  for (i=0; i<ibdb->n_distrib; i++)
  {
    // check that the header size matches the requested data_size
    if (ibdb->ib_cms[i]->header_mb->size != data_size)
    {
      multilog (client->log, LOG_ERR, "recv: [%d] header was %"PRIu64" bytes, "
                "expected %"PRIu64"\n", i, data_size, ibdb->ib_cms[i]->header_mb);
      return -1;
    }

    // n.b. dont have to post recv for the header, this was done in open fn

    if (ibdb->verbose > 1)
      multilog(client->log, LOG_INFO, "recv: [%d] dada_ib_wait_recv\n", i);

    // wait for the transfer of the header
    if (dada_ib_wait_recv(ib_cms[i], ibdb->ib_cms[i]->header_mb) < 0)
    {
      multilog(client->log, LOG_ERR, "recv: [%d] dada_ib_wait_recv failed\n", i);
      return -1;
    }
  }


  // pre post recv for the number of bytes in the next send_block transfer
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv: post_recv[%d] on sync_from for bytes in next xfer\n", i);
    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(client->log, LOG_ERR, "recv: [%d] dada_ib_post_recv failed\n", i);
      return -1;
    }
  }

  // copy the header to specified buffer
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog(client->log, LOG_INFO, "recv: [%d] memcpy %"PRIu64" bytes\n", i, data_size);

    memcpy (data, ibdb->ib_cms[i]->header_mb->buffer, data_size);
  }

  if (ibdb->verbose > 1)
    multilog(log, LOG_INFO, "recv: returning %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t caspsr_ibdb_recv_block (dada_client_t* client, void* data, 
                         uint64_t data_size, uint64_t block_id)
{

  caspsr_ibdb_t * ibdb = (caspsr_ibdb_t *) client->context;

  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "caspsr_ibdb_recv_block()\n");

  unsigned i = 0;

  // send the ready message to the dbib's
  //multilog(client->log, LOG_INFO, "recv_block: SEND READY\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    ib_cms[i]->sync_to_val[0] = 1;
    ib_cms[i]->sync_to_val[1] = 0;
    //multilog(client->log, LOG_INFO, "recv_block: post_send [%d] on sync_to for READY\n", i);
    if (dada_ib_post_send (ib_cms[i], ib_cms[i]->sync_to) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: [%d] post_send on sync_to for READY failed\n", i);
      return -1;
    }
  }
  for (i=0; i<ibdb->n_distrib; i++)
  {
    //multilog(client->log, LOG_INFO, "recv_block: wait_recv[%d] on sync_to for READY\n", i);
    if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->sync_to) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: [%d] wait_recv on sync_from [bytes to be sent] failed\n", i);
      return -1;
    }
  }

  
  // wait for the number of bytes to be received
  //multilog(client->log, LOG_INFO, "recv_block: RECV BYTES TO BE RECVD\n");
  uint64_t total_bytes = 0;
  for (i=0; i<ibdb->n_distrib; i++)
  {
    //multilog(client->log, LOG_INFO, "recv_block: [%d] wait_recv on sync from [bytes to be recvd]\n", i);
    if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(client->log, LOG_ERR, "recv_block: [%d] wait_recv on sync_from [bytes to be recvd] failed\n", i);
      return -1;
    }

    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: [%d] bytes to be recvd=%"PRIu64"\n", i, ib_cms[i]->sync_from_val[1]);

    assert(ib_cms[i]->sync_from_val[0] == 2);
    total_bytes += ib_cms[i]->sync_from_val[1];
  }

  if (ibdb->verbose) 
    multilog(client->log, LOG_INFO, "recv_block: total bytes to be recvd=%"PRIu64"\n", total_bytes);

  if (total_bytes) 
  {
    // pre post recv for the client's completion message for each connection
    for (i=0; i<ibdb->n_distrib; i++)
    {
      if (ibdb->verbose)
        multilog(client->log, LOG_INFO, "recv_block: post_recv [%d] on sync_from for bytes recvd\n", i);
      if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
      {
        multilog(client->log, LOG_ERR, "recv_block: [%d] post_recv on sync_from for bytes recvd failed\n", i);
        return -1;
      }
    }

    // instruct the client to fill the specified block_id remotely via RDMA
    for (i=0; i<ibdb->n_distrib; i++)
    {
      ib_cms[i]->sync_to_val[0] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_va;
      ib_cms[i]->sync_to_val[1] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_rkey;
    }

    if (ibdb->verbose > 1)
      multilog(client->log, LOG_INFO, "recv_block: data=%p, bytes=%"PRIu64", "
               "block_id=%"PRIu64"\n", data, data_size, block_id);

    // send the client the sync value (buffer ID to be filled)
    for (i=0; i<ibdb->n_distrib; i++)
    {
      //multilog(client->log, LOG_INFO, "recv_block: post_send [%d] on sync_to for block id\n", i);
      if (dada_ib_post_send (ib_cms[i], ib_cms[i]->sync_to) < 0)
      {
        multilog(client->log, LOG_ERR, "recv_block: [%d] post_send on sync_to for block id failed\n", i);
        return -1;
      }
    }

    // confirm that the sync value has been sent 
    for (i=0; i<ibdb->n_distrib; i++)
    {
      //multilog(client->log, LOG_INFO, "recv_block: wait_recv [%d] on sync_to for block id\n", i);
      if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->sync_to) < 0)
      {
        multilog(client->log, LOG_ERR, "recv_block: [%d] wait_recv on sync to for block id failed\n", i);
        return -1;
      }
    }

    // remote RDMA transfer is ocurring now...

    if (ibdb->verbose)
      multilog(client->log, LOG_INFO, "recv_block: waiting for completion "
               "on block %"PRIu64"\n", block_id);

    // accept the completion message for the "current" transfer
    total_bytes = 0;
    for (i=0; i<ibdb->n_distrib; i++)
    {
      //multilog(client->log, LOG_INFO, "recv_block: wait_recv [%d] on sync_from for bytes sent \n", i);
      if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
      {
        multilog(client->log, LOG_ERR, "recv_block: [%d] wait_recv on sync_from for bytes sent failed\n", i);
        return -1;
      }
      assert(ib_cms[i]->sync_from_val[0] == 3);
      total_bytes += ib_cms[i]->sync_from_val[1];
    }

    // pre post recv for the number of bytes in the next send_block transfer
    for (i=0; i<ibdb->n_distrib; i++)
    {
      if (ibdb->verbose)
        multilog(client->log, LOG_INFO, "recv_block: post_recv[%d] on sync_from for bytes to be sent \n", i);
      if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
      {
        multilog(client->log, LOG_ERR, "recv_block: [%d] post_recv on sync_from for bytes to be sent failed\n", i);
        return -1;
      }
    }
  }

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "recv_block: bytes transferred=%"PRIu64"\n", total_bytes);

  return (int64_t) total_bytes;

}


/*! Function that closes the data file */
int caspsr_ibdb_close (dada_client_t* client, uint64_t bytes_written)
{

  caspsr_ibdb_t * ibdb = (caspsr_ibdb_t *) client->context;

  if (ibdb->verbose)
    multilog (client->log, LOG_INFO, "caspsr_ibdb_close()\n");

  unsigned i = 0;
  
  /*

  // need to post_send on the sync_to buffer as dbib will be waiting on this
  for (i=0;  i<ibdb->n_distrib; i++)
  {
    ibdb->ib_cms[i]->sync_to_val[0] = 0;
    ibdb->ib_cms[i]->sync_to_val[1] = 0;
    if (ibdb->verbose)
      multilog (client->log, LOG_INFO, "close: post_send [%d] on sync_to\n", i);
    if (dada_ib_post_send (ibdb->ib_cms[i], ibdb->ib_cms[i]->sync_to) < 0)
    {
      multilog(client->log, LOG_ERR, "close: dada_ib_post_send [%d] on sync_to failed\n", i);
      return -1;
    }
  }

  // confirm that the sync value has been sent 
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (dada_ib_wait_recv(ibdb->ib_cms[i], ibdb->ib_cms[i]->sync_to) < 0)
    {
      multilog(client->log, LOG_ERR, "close: [%d] dada_ib_wait_recv on sync_to failed\n", i);
      return -1;
    }
  }

  */

  if (ascii_header_set (client->header, "TRANSFER_SIZE", "%"PRIu64, bytes_written) < 0)  {
    multilog (client->log, LOG_ERR, "close: could not set TRANSFER_SIZE\n");
    return -1;
  }

  return 0;
}

/*! Function that opens the data transfer target */
int caspsr_ibdb_open (dada_client_t* client)
{

  assert (client != 0);

  caspsr_ibdb_t * ibdb = (caspsr_ibdb_t *) client->context;

  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  unsigned i = 0;

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "caspsr_ibdb_open()\n");

  // get the next data block buffer to be written for first post_recv
  uint64_t block_id = ipcbuf_get_write_index ((ipcbuf_t *) client->data_block);

  for (i=0; i<ibdb->n_distrib; i++)
  {
    ib_cms[i]->sync_to_val[0] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_va;
    ib_cms[i]->sync_to_val[1] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_rkey;
  }

  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "open: block_id=%"PRIu64"\n", block_id);

  // pre-post receive for the header transfer 
  if (ibdb->verbose)
    multilog(client->log, LOG_INFO, "open: post_recv for headers\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog(client->log, LOG_INFO, "open: post_recv on header_mb %d\n", i);

    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(client->log, LOG_ERR, "open: dada_ib_post_recv failed\n");
      return -1;
    }
  }

  // accept each connection
  int accept_result = 0;

  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (!ib_cms[i]->ib_connected)
    {
      if (ibdb->verbose)
        multilog(client->log, LOG_INFO, "open: ib_cms[%d] accept\n", i);
      if (dada_ib_accept (ib_cms[i]) < 0)
      {
        multilog(client->log, LOG_ERR, "open: dada_ib_accept failed\n");
        accept_result = -1;
      }
      ib_cms[i]->ib_connected = 1;
    }
  }

  return accept_result;

}

/*
 * required initialization of IB device and associate verb structs
 */
int caspsr_ibdb_ib_init(caspsr_ibdb_t * ctx, dada_hdu_t * hdu, multilog_t * log)
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
    multilog(log, LOG_INFO, "caspsr_ibdb_ib_init()\n");

  // get the information about the data block
  db_buffers = dada_hdu_db_addresses(hdu, &db_nbufs, &db_bufsz);

  // get the header block buffer size
  hb_buffers = dada_hdu_hb_addresses(hdu, &hb_nbufs, &hb_bufsz);

  // this a strict requirement at this stage
  if (db_bufsz % ctx->chunk_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: chunk size [%d] was not a factor "
             "of data block size[%"PRIu64"]\n", ctx->chunk_size, db_bufsz);
    return -1;
  }

  ctx->chunks_per_block = db_bufsz / ctx->chunk_size;

  // create some pointers for the cms
  ctx->ib_cms = (dada_ib_cm_t **) malloc(sizeof(dada_ib_cm_t *) * ctx->n_distrib);
  assert(ctx->ib_cms != 0);

  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  unsigned i = 0;

  for (i=0; i<ctx->n_distrib; i++)
  {

    if (ctx->verbose > 1)
      multilog(log, LOG_INFO, "ib_init: dada_ib_create_cm\n");

    ctx->ib_cms[i] = dada_ib_create_cm(db_nbufs, log);
    if (!ctx->ib_cms[i])
    {
      multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm failed\n");
      return -1; 
    }

    ctx->ib_cms[i]->verbose = ctx->verbose;
    ctx->ib_cms[i]->depth = (ctx->chunks_per_block+1);
    ctx->ib_cms[i]->port = ctx->port + i;
    ctx->ib_cms[i]->bufs_size = db_bufsz;
    ctx->ib_cms[i]->header_size = hb_bufsz;
    ctx->ib_cms[i]->db_buffers = db_buffers;

  }

  // accept 
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ctx->n_distrib);

  for (i=0; i<ctx->n_distrib; i++)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "ib_init: caspsr_ibdb_ib_init_thread ib_cms[%d]=%p\n",
                i, ctx->ib_cms[i]); 

    rval = pthread_create(&(connections[i]), 0, (void *) caspsr_ibdb_ib_init_thread, 
                          (void *) ctx->ib_cms[i]);
    if (rval != 0)
    {
      multilog (ctx->log, LOG_INFO, "ib_init: error creating ib_init_thread\n");
      return -1;
    }
  }

  void * result;
  int init_result = 0;
  for (i=0; i<ctx->n_distrib; i++) {
    pthread_join (connections[i], &result);
    if (!ctx->ib_cms[i]->cm_connected) 
      init_result = -1;
  }
  free(connections);
}

void * caspsr_ibdb_ib_init_thread (void * arg)
{

  dada_ib_cm_t * ib_cm = (dada_ib_cm_t *) arg;

  multilog_t * log = ib_cm->log;

  if (ib_cm->verbose)
    multilog (log, LOG_INFO, "ib_init_thread: dada_ib_accept ib_cm=%p\n", ib_cm);
    
  ib_cm->cm_connected = 0;
  ib_cm->ib_connected = 0;

  // listen for a connection request on the specified port
  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "ib_init_thread: dada_ib_listen_cm\n");

  if (dada_ib_listen_cm(ib_cm, ib_cm->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_listen_cm failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // create the IB verb structures necessary
  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: depth=%"PRIu64"\n", (ib_cm->depth));

  if (dada_ib_create_verbs(ib_cm, (ib_cm->depth)) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_create_verbs failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

  // register each data block buffer with as a MR within the PD
  if (dada_ib_reg_buffers(ib_cm, ib_cm->db_buffers, ib_cm->bufs_size, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_register_memory_buffers failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  assert(ib_cm->header);

  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: reg header_mb\n");
  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_INFO, "ib_init_thread: reg header_mb failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->header_mb->wr_id = 10000;

  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: dada_ib_create_qp\n");
  if (dada_ib_create_qp (ib_cm, 1, 1) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->cm_connected = 1;
  pthread_exit((void *) &(ib_cm->cm_connected));

}

/*
 *  Main. 
 */
int main (int argc, char **argv)
{

  /* IB DB configuration */
  caspsr_ibdb_t ibdb = CASPSR_IBDB_INIT;

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

  /* number of distributors sending data */
  unsigned n_distrib = 0;

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

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"caspsr_ibdb: could not parse key from %s\n",optarg);
        return EXIT_FAILURE;
      }
      break;
      
    case 'd':
      daemon=1;
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

  if ((argc - optind) != 1) {
    fprintf (stderr, "Error: number of distributors must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  n_distrib = atoi(argv[optind]);
  if ((n_distrib < 1) || (n_distrib > 4))
  {
    fprintf (stderr, "Error: number of distributors must be [1-4]\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  // do not use the syslog facility
  log = multilog_open ("caspsr_ibdb", 0);

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

  client->open_function     = caspsr_ibdb_open;
  client->io_function       = caspsr_ibdb_recv;
  client->io_block_function = caspsr_ibdb_recv_block;
  client->close_function    = caspsr_ibdb_close;
  client->direction         = dada_client_writer;

  client->context = &ibdb;

  ibdb.chunk_size = chunk_size;
  ibdb.verbose = verbose;
  ibdb.port = port;
  ibdb.n_distrib = n_distrib;
  ibdb.log = log;

  // initialize IB resources
  if (caspsr_ibdb_ib_init (&ibdb, hdu, log) < 0) 
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    client->quit = 1;
  }

  while (!client->quit) 
  {
    if (dada_client_write (client) < 0)
    {
      multilog (log, LOG_ERR, "Error during transfer\n");
      quit = 1;
    }
    
    if (dada_hdu_unlock_write (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock write on hdu\n");
      return EXIT_FAILURE;
    }

    if (quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_write (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock write on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  unsigned i=0;
  for (i=0; i<ibdb.n_distrib; i++)
  {
    if (dada_ib_destroy(ibdb.ib_cms[i]) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_destory failed\n");
    }
  }

  //free (ibdb.connected);

  return EXIT_SUCCESS;
}
