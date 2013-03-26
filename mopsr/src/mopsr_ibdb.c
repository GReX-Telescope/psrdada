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

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_msg.h"
#include "mopsr_def.h"
#include "mopsr_ib.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

// Globals
int quit_signal = 0;

void * mopsr_ibdb_init_thread (void * arg);

void usage()
{
  fprintf (stdout,
	   "mopsr_ibdb [options] cornerturn.cfg\n"
     " cornerturn.cfg    ascii configuration file defining cornerturn\n\n"
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -s                single transfer only\n"
     " -v                verbose output\n"
     " -h                print this usage\n",
     DADA_IB_DEFAULT_CHUNK_SIZE, 
     DADA_DEFAULT_BLOCK_KEY);
}

mopsr_conn_t * mopsr_parse_cornerturn_cfg (const char * config_file, mopsr_ibdb_t * ctx)
{
  char config[DADA_DEFAULT_HEADER_SIZE];

  if (fileread (config_file, config, DADA_DEFAULT_HEADER_SIZE) < 0)
  {
    fprintf (stderr, "ERROR: could not read ASCII configuration from %s\n", config_file);
    return 0;
  }

  if (ascii_header_get (config , "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no NCHAN\n");
    return 0;
  }

  if (ascii_header_get (config , "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no NANT\n");
    return 0;
  }

  unsigned int nhost = 0;
  if (ascii_header_get (config , "NHOST", "%d", &nhost) != 1)
  {
    multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no NHOST\n");
    return 0;             
  }                         

  // for now require that NHOST == NCHAN
  if (nhost != ctx->nchan)
  {
    multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: NHOST != NCHAN\n");
    return 0;
  }

  mopsr_conn_t * conns = (mopsr_conn_t *) malloc (sizeof(mopsr_conn_t) * ctx->nchan);

  unsigned int i;
  char key[16];
  char host[64];

  int ant_first;
  int ant_last;

  unsigned int ichan;
  int chan_baseport;
  int chan_first;
  int chan_last;
  int prev_chan = -1;

  char hostname[_POSIX_HOST_NAME_MAX] = "unknown";;
  gethostname(hostname, _POSIX_HOST_NAME_MAX);

  if (ascii_header_get (config , "CHAN_BASEPORT", "%d", chan_baseport) != 1)
  {
    multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no CHAN_BASEPORT\n", key);
    return 0;
  }

  for (i=0; i<nhost; i++)
  {
    sprintf (key, "HOST_%d\n", i);
    if (ascii_header_get (config , key, "%s", host) != 1)
    {
      multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "ANT_FIRST_HOST_%d\n", i);
    if (ascii_header_get (config , key, "%d", &ant_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "ANT_LAST_HOST_%d\n", i);
    if (ascii_header_get (config , key, "%d", &ant_last) != 1)
    {       
      multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no %s\n", key);
      return 0;         
    }                     

    sprintf (key, "CHAN_FIRST_HOST_%d\n", i);
    if (ascii_header_get (config , key, "%d", &chan_first) != 1)
    {
      multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no %s\n", key);
      return 0;
    }

    sprintf (key, "CHAN_LAST_HOST_%d\n", i);
    if (ascii_header_get (config , key, "%d", &chan_last) != 1)
    {
      multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: config with no %s\n", key);
      return 0;
    }

    // for the first host, check ports / channels
    for (ichan = chan_first; ichan < chan_last; ichan++)
    {
      if (ichan != prev_chan + 1)
      {
        multilog (ctx->log, LOG_ERR, "parse_cornerturn_cfg: non-continuous channels for %s\n", host);
        return 0;
      } 

      // set destination host for opening IB connection
      strcpy (conns[ichan].host, host);

      // set destination port for opening IB connection
      conns[ichan].port = chan_baseport + ichan;
      conns[ichan].chan = ichan;
      conns[ichan].ib_cm = ctx->ib_cms[ichan];
      conns[ichan].ant_first = ant_first;
      conns[ichan].ant_last  = ant_last;

      prev_chan = ichan;
    }
  }

  return conns;
}

/*! Function that opens the data transfer target */
int mopsr_ibdb_open (dada_client_t* client)
{

  // the mopsr_ibdb specific data
  assert (client != 0);
  mopsr_ibdb_t* ctx = (mopsr_ibdb_t*) client->context;

  // the ib communcation managers
  assert(ctx->ib_cms != 0);
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "mopsr_ibdb_open()\n");
  
  // the header
  assert(client->header != 0);
  char * header = client->header;

  uint64_t obs_offset = 0;
  uint64_t transfer_size = 0;
 
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1) {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_OFFSET\n");
  }

  if (ascii_header_get (header, "TRANSFER_SIZE", "%"PRIu64, &transfer_size) != 1) {
    multilog (client->log, LOG_WARNING, "open: header with no TRANSFER_SIZE\n");
  }

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: OBS_OFFSET=%"PRIu64"\n", obs_offset);

  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = transfer_size;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int mopsr_ibdb_close (dada_client_t* client, uint64_t bytes_written)
{
  // the mopsr_ibdb specific data
  mopsr_ibdb_t* ctx = (mopsr_ibdb_t*) client->context;

  // status and error logging facility
  multilog_t* log = client->log;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "mopsr_ibdb_close()\n");

  unsigned i = 0;
  for (i=0; i<ctx->n_src; i++)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: post_recv [READY]\n");
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "close: post_recv [READY] failed\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: send_messages [EOD]\n");
  if (dada_ib_send_messages (ib_cms, ctx->n_src, DADA_IB_BYTES_TO_XFER_KEY, 0) < 0)
  {
    multilog(log, LOG_ERR, "close: send_messages [EOD] failed\n");
    return -1;
  }

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
int64_t mopsr_ibdb_recv (dada_client_t* client, void * buffer, uint64_t bytes)
{
  mopsr_ibdb_t * ctx = (mopsr_ibdb_t *) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  unsigned int i = 0;

  // send READY message to inform sender we are ready for the header
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv: send_messages [READY HDR]\n");
  if (dada_ib_send_messages(ib_cms, ctx->n_src, DADA_IB_READY_KEY, 0) < 0)
  {
    multilog(ctx->log, LOG_ERR, "recv: send_messages [READY HDR] failed\n");
    return 0;
  }

  // wait for transfer of the headers
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv: wait_recv [HEADER]\n");
  for (i=0; i<ctx->n_src; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "recv: [%d] wait_recv [HEADER]\n", i);
    if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(ctx->log, LOG_ERR, "recv: [%d] wait_recv [HEADER] failed\n", i);
      return 0;
    }
    if (ib_cms[i]->verbose > ctx->verbose)
      ctx->verbose = ib_cms[i]->verbose;
  }

  // copy header from just first src, up to a maximum of bytes
  memcpy (buffer, ctx->ib_cms[0]->header_mb->buffer, bytes);

  // use the size of the first cms header
  return (int64_t) bytes;
}


/*! Transfers 1 datablock at a time to mopsr_ibdb */
int64_t mopsr_ibdb_recv_block (dada_client_t* client, void * buffer, 
                               uint64_t data_size, uint64_t block_id)
{
  
  mopsr_ibdb_t * ctx = (mopsr_ibdb_t*) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "recv_block: bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, block_id);

  unsigned int i;

  // send READY message to inform sender we are ready for DATA 
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv: send_messages [READY DATA]\n");
  if (dada_ib_send_messages(ib_cms, ctx->n_src, DADA_IB_READY_KEY, 0) < 0)
  {
    multilog(ctx->log, LOG_ERR, "recv: send_messages [READY DATA] failed\n");
    return 0;
  }

  // wait for the number of bytes to be received
  uint64_t bytes_to_xfer = 0;
  if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "recv_block: recv_messages [BYTES TO XFER]\n");
  if (dada_ib_recv_messages(ib_cms, ctx->n_src, DADA_IB_BYTES_TO_XFER_KEY) < 0)
  {
    multilog(ctx->log, LOG_ERR, "recv_block: recv_messages [BYTES TO XFER] failed\n");
    return -1;
  }

  // count the number of bytes to be received
  for (i=0; i<ctx->n_src; i++)
  {
    bytes_to_xfer += ib_cms[i]->sync_from_val[1];
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "recv_block: [%d] bytes to be recvd=%"PRIu64"\n",
                i, ib_cms[i]->sync_from_val[1]);
  }

  if ((ctx->verbose) || (bytes_to_xfer != ib_cms[0]->bufs_size && bytes_to_xfer != 0))
      multilog(ctx->log, LOG_INFO, "recv_block: total bytes to be recvd=%"PRIu64"\n", bytes_to_xfer);

  // if the number of bytes to be received is less than the block size, this is the end of the observation
  if (bytes_to_xfer < data_size)
  {
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "recv_block: bytes_to_xfer=%"PRIu64" < %"PRIu64", end of obs\n", bytes_to_xfer, data_size);
  }

  // post recv for the number of bytes that were transferred
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv_block: post_recv [BYTES XFERRED]\n");
  for (i=0; i<ctx->n_src; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "recv_block: [%d] post_recv on sync_from [BYTES XFERRED]\n", i);
    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(ctx->log, LOG_ERR, "recv_block: [%d] post_recv on sync_from [BYTES XFERRED] failed\n", i);
      return -1;
    }
  }

  // send the memory address of the block_id to be filled remotely via RDMA
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv_block: send_messages [BLOCK ID]\n");
  for (i=0; i<ctx->n_src; i++)
  {
    ib_cms[i]->sync_to_val[0] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_va;
    ib_cms[i]->sync_to_val[1] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_rkey;

    uintptr_t buf_va = (uintptr_t) ib_cms[i]->sync_to_val[0];
    uint32_t buf_rkey = (uint32_t) ib_cms[i]->sync_to_val[1];
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "recv_block: [%d] local_block_id=%"PRIu64", local_buf_va=%p, "
                         "local_buf_rkey=%p\n", i, block_id, buf_va, buf_rkey);
  }
  if (dada_ib_send_messages(ib_cms, ctx->n_src, UINT64_MAX, UINT64_MAX) < 0)
  {
    multilog(ctx->log, LOG_ERR, "recv_block: send_messages [BLOCK ID] failed\n");
    return -1;
  }

  // remote RDMA transfer is ocurring now...
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv_block: waiting for completion "
             "on block %"PRIu64"...\n", block_id);

  // wait for the number of bytes transferred
  uint64_t bytes_xferred = 0;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv_block: recv_messages [BYTES XFERRED]\n");
  if (dada_ib_recv_messages(ib_cms, ctx->n_src, DADA_IB_BYTES_XFERRED_KEY) < 0)
  {
    multilog(ctx->log, LOG_ERR, "recv_block: recv_messages [BYTES XFERRED] failed\n");
    return -1;
  }
  for (i=0; i<ctx->n_src; i++)
  {
    bytes_xferred += ib_cms[i]->sync_from_val[1];
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "recv_block: [%d] bytes recvd=%"PRIu64"\n",
                i, ib_cms[i]->sync_from_val[1]);
  }

  // post receive for the BYTES TO XFER in next block function call
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "recv_block: post_recv [BYTES TO XFER]\n");
  for (i=0; i<ctx->n_src; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "recv_block: [%d] post_recv [BYTES TO XFER]\n", i);
    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(ctx->log, LOG_ERR, "recv_block: [%d] post_recv [BYTES TO XFER] failed\n", i);
      return -1;
    }
  }

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "recv_block: bytes transferred=%"PRIu64"\n", bytes_xferred);

  return (int64_t) bytes_xferred;
}


/*
 * required initialization of IB device and associate verb structs
 */
int mopsr_ibdb_ib_init (mopsr_ibdb_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "mopsr_ibdb_ib_init()\n");

  // infiniband post_send size will be 
  unsigned int chunk_size = ctx->nant * 2;
  unsigned int modulo_size = chunk_size * ctx->nchan;

  uint64_t db_nbufs = 0;
  uint64_t db_bufsz = 0;
  uint64_t hb_nbufs = 0;
  uint64_t hb_bufsz = 0;
  char ** db_buffers = 0;
  char ** hb_buffers = 0;

  // get the datablock addresses for memory registration
  db_buffers = dada_hdu_db_addresses(hdu, &db_nbufs, &db_bufsz);
  hb_buffers = dada_hdu_hb_addresses(hdu, &hb_nbufs, &hb_bufsz);

  // check that the chunk size is a factor of DB size
  if (db_bufsz % modulo_size != 0)
  {
    multilog(log, LOG_ERR, "ib_init: modulo size [%d] was not a factor "
             "of data block size [%"PRIu64"]\n", modulo_size, db_bufsz);
    return -1;
  }
  uint64_t recv_depth = db_bufsz / modulo_size;

  // alloc memory for ib_cm ptrs
  ctx->ib_cms = (dada_ib_cm_t **) malloc(sizeof(dada_ib_cm_t *) * ctx->n_src);
  if (!ctx->ib_cms)
  {
    multilog(log, LOG_ERR, "ib_init: could not allocate memory\n");
    return -1;
  }

  // create the CM and CM channel
  unsigned isrc;
  for (isrc=0; isrc < ctx->n_src; isrc++)
  {
    ctx->ib_cms[isrc] = dada_ib_create_cm(db_nbufs, log);
    if (!ctx->ib_cms[isrc])
    {
      multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm on src %d of %d  failed\n", isrc, ctx->n_src);
      return -1;
    }

    ctx->ib_cms[isrc]->verbose = ctx->verbose;
    ctx->ib_cms[isrc]->send_depth = 1;
    ctx->ib_cms[isrc]->recv_depth = 1;
    ctx->ib_cms[isrc]->port = ctx->conn_info[isrc].port;
    ctx->ib_cms[isrc]->bufs_size = db_bufsz;
    ctx->ib_cms[isrc]->header_size = hb_bufsz;
    ctx->ib_cms[isrc]->db_buffers = db_buffers;
  }

  return 0;
}

int mopsr_ibdb_destroy (mopsr_ibdb_t * ctx) 
{

  unsigned i=0;
  for (i=0; i<ctx->n_src; i++)
  {
    if (dada_ib_destroy(ctx->ib_cms[i]) < 0)
    {
      multilog(ctx->log, LOG_ERR, "dada_ib_destroy for %d failed\n", i);
    }
  }


}

int mopsr_ibdb_open_connections (mopsr_ibdb_t * ctx, multilog_t * log)
{
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  mopsr_conn_t * conns = ctx->conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_ibdb_open_connections()\n");

  unsigned i = 0;
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ctx->n_src);

  // start all the connection threads
  for (i=0; i<ctx->n_src; i++)
  {
    if (!ib_cms[i]->cm_connected)
    {
      if (ctx->verbose > 1)
        multilog (ctx->log, LOG_INFO, "open_connections: mopsr_ibdb_init_thread ib_cms[%d]=%p\n",
                  i, ctx->ib_cms[i]);

      rval = pthread_create(&(connections[i]), 0, (void *) mopsr_ibdb_init_thread,
                          (void *) &conns[i]);
      if (rval != 0)
      {
        multilog (ctx->log, LOG_INFO, "open_connections: error creating init_thread\n");
        return -1;
      }
    }
    else
    {
      multilog (ctx->log, LOG_INFO, "open_connections: ib_cms[%d] already connected\n", i);
    }
  }

  // join all the connection threads
  void * result;
  int init_cm_ok = 1;
  for (i=0; i<ctx->n_src; i++)
  {
    if (ib_cms[i]->cm_connected <= 1)
    {
      pthread_join (connections[i], &result);
      if (!ib_cms[i]->cm_connected)
        init_cm_ok = 0;
    }

    // set to 2 to indicate connection is established
    ib_cms[i]->cm_connected = 2;
  }
  free(connections);
  if (!init_cm_ok)
  {
    multilog(ctx->log, LOG_ERR, "open_connections: failed to init CM connections\n");
    return -1;
  }

  // pre-post receive for the header transfer 
      if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "open_connections: post_recv for headers\n");
  for (i=0; i<ctx->n_src; i++)
  {
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "open_connections: [%d] post_recv on header_mb [HEADER]\n", i);

    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(ctx->log, LOG_ERR, "open_connections: [%d] post_recv on header_mb [HEADER] failed\n", i);
      return -1;
    }
  }

  // accept each connection
  int accept_result = 0;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "open_connections: accepting connections\n");

  for (i=0; i<ctx->n_src; i++)
  {
    if (!ib_cms[i]->ib_connected)
    {
      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "open_connections: ib_cms[%d] accept\n", i);
      if (dada_ib_accept (ib_cms[i]) < 0)
      {
        multilog(ctx->log, LOG_ERR, "open_connections: dada_ib_accept failed\n");
        accept_result = -1;
      }
      ib_cms[i]->ib_connected = 1;
    }
  }

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "open_connections: connections accepted\n");

  return accept_result;
}


/*
 * Thread to open connection between sender and receivers
 */
void * mopsr_ibdb_init_thread (void * arg)
{

  mopsr_conn_t * conn = (mopsr_conn_t *) arg;

  dada_ib_cm_t * ib_cm = conn->ib_cm;

  multilog_t * log = ib_cm->log;

  if (ib_cm->verbose > 1)
    multilog (log, LOG_INFO, "init_thread: ib_cm=%p\n", ib_cm);

  ib_cm->cm_connected = 0;
  ib_cm->ib_connected = 0;

  // listen for a connection request on the specified port
  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "init_thread: dada_ib_listen_cm\n");

  if (dada_ib_listen_cm(ib_cm, ib_cm->port) < 0)
  {
    multilog(log, LOG_ERR, "init_thread: dada_ib_listen_cm failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // create the IB verb structures necessary
  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "init_thread: depth=%"PRIu64"\n", ib_cm->send_depth + ib_cm->recv_depth);

  if (dada_ib_create_verbs(ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "init_thread: dada_ib_create_verbs failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

  // register each data block buffer with as a MR within the PD
  if (dada_ib_reg_buffers(ib_cm, ib_cm->db_buffers, ib_cm->bufs_size, flags) < 0)
  {
    multilog(log, LOG_ERR, "init_thread: dada_ib_register_memory_buffers failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "init_thread: could not allocate memory for header\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "init_thread: reg header_mb\n");
  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_INFO, "init_thread: reg header_mb failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->header_mb->wr_id = 10000;

  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "init_thread: dada_ib_create_qp\n");
  if (dada_ib_create_qp (ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init: dada_ib_create_qp failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->cm_connected = 1;
  pthread_exit((void *) &(ib_cm->cm_connected));
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
  mopsr_ibdb_t ibdb = MOPSR_IBDB_INIT;

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

  /* chunk size for IB transport */
  unsigned chunk_size = DADA_IB_DEFAULT_CHUNK_SIZE;

  char * cornerturn_cfg = 0;

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
          fprintf (stderr,"mopsr_ibdb: could not parse key from %s\n",optarg);
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
  if (argc-optind != 1) {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  cornerturn_cfg = strdup(argv[optind]);

  // do not use the syslog facility
  log = multilog_open ("mopsr_ibdb", 0);

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

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = mopsr_ibdb_open;
  client->io_function       = mopsr_ibdb_recv;
  client->io_block_function = mopsr_ibdb_recv_block;
  client->close_function    = mopsr_ibdb_close;
  client->direction         = dada_client_writer;

  client->quiet = 1;
  client->context = &ibdb;
  ibdb.verbose = verbose;

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  // Initiase IB resources
  if (mopsr_ibdb_ib_init (&ibdb, hdu, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  // parse the cornerturn configuration
  ibdb.conn_info = mopsr_parse_cornerturn_cfg (cornerturn_cfg, &ibdb);
  if (!ibdb.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn configuration file\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  // open IB connections with the receivers
  if (mopsr_ibdb_open_connections (&ibdb, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to open IB connections\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  while (!client->quit)
  {

    if (dada_client_write (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_write()\n");
    if (dada_hdu_unlock_write (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (quit || ibdb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_write (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (mopsr_ibdb_destroy (&ibdb) < 0)
  {
    multilog(log, LOG_ERR, "mopsr_ibdb_destroy failed\n");
  }

  return EXIT_SUCCESS;
}
