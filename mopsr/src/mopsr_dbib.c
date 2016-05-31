/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 *  Use multiple infiniband connections for perform the cornerturn for channels
 *  and antenna in the MOPSR backend. This application is coupled with 
 *  mopsr_ibdb and the cornerturn.cfg file in the ../config directory 
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
#include "mopsr_def.h"
#include "mopsr_ib.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

// Globals
int quit_signal = 0;

void * mopsr_dbib_init_thread (void * arg);

void usage()
{
  fprintf (stdout,
	   "mopsr_dbib [options] cornerturn.cfg\n"
     " cornerturn.cfg    ascii configuration file defining cornerturn\n\n"
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -s                single transfer only\n"
     " -v                verbose output\n"
     " -h                print this usage\n",
     DADA_IB_DEFAULT_CHUNK_SIZE, 
     DADA_DEFAULT_BLOCK_KEY);
}

mopsr_bf_conn_t * mopsr_parse_cornerturn_cfg (const char * config_file, mopsr_dbib_t * ctx)
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

  mopsr_bf_conn_t * conns = (mopsr_bf_conn_t *) malloc (sizeof(mopsr_bf_conn_t) * ctx->nchan);

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

#if 0
    if (strcmp(host, hostname) == 0)
    {
      ctx->ant_first = ant_first;
      ctx->ant_last  = ant_last;
    }
#endif

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
      conns[ichan].ant_last = ant_last;

      ctx->ib_cms[ichan].port = conns[ichan].port;

      prev_chan = ichan;
    }
  }

  return conns;
}

/*! Function that opens the data transfer target */
int mopsr_dbib_open (dada_client_t* client)
{

  // the mopsr_dbib specific data
  assert (client != 0);
  mopsr_dbib_t* ctx = (mopsr_dbib_t*) client->context;

  // the ib communcation managers
  assert(ctx->ib_cms != 0);
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "mopsr_dbib_open()\n");
  
  // the header
  assert(client->header != 0);
  char * header = client->header;

  uint64_t obs_offset = 0;
 
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1) {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_OFFSET\n");
  }

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: OBS_OFFSET=%"PRIu64"\n", obs_offset);

  // assumed that we do not know how much data will be transferred
  client->transfer_bytes = 0;

  // this is not used in block by block transfers
  client->optimal_bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int mopsr_dbib_close (dada_client_t* client, uint64_t bytes_written)
{
  // the mopsr_dbib specific data
  mopsr_dbib_t* ctx = (mopsr_dbib_t*) client->context;

  // status and error logging facility
  multilog_t* log = client->log;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "mopsr_dbib_close()\n");

  unsigned i = 0;
  for (i=0; i<ctx->n_dst; i++)
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
  if (dada_ib_send_messages (ib_cms, ctx->n_dst, DADA_IB_BYTES_TO_XFER_KEY, 0) < 0)
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

/*! transfer data to ibdb used for sending header only */
int64_t mopsr_dbib_recv (dada_client_t* client, void * buffer, uint64_t bytes)
{
  mopsr_dbib_t * ctx = (mopsr_dbib_t *) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  unsigned int i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "mopsr_dbib_recv()\n");

  for (i=0; i<ctx->n_dst; i++)
  {
    if (ib_cms[i]->header_mb->size != bytes)
    {
      multilog (log, LOG_ERR, "send: header was %"PRIu64" bytes, expected %"PRIu64"\n", bytes, ib_cms[i]->header_mb->size);
      return -1;
    }
  }

  // wait for READY message so we know ibdb is ready for the HEADER
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: recv_messages [READY HDR]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->n_dst, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "send: recv_messages [READY HDR] failed\n");
    return -1;
  }
  
  // post recv on sync_from for the READY message
  for (i=0; i<ctx->n_dst; i++)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "send: post_recv [READY DATA]\n");
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send: post_recv [READY DATA] failed\n");
      return -1;
    }
  }

  // copy the header to the header memory buffer
  for (i=0; i<ctx->n_dst; i++)
  {
    memcpy (ib_cms[i]->header_mb->buffer, buffer, bytes);
  }

  uint64_t transfer_size = 0;
  for (i=0; i<ctx->n_dst; i++)
  {
    if (ascii_header_get (ib_cms[i]->header_mb->buffer, "TRANSFER_SIZE", "%"PRIu64, &transfer_size) != 1) 
      if (ctx->verbose)
        multilog (client->log, LOG_INFO, "send: header with no TRANSFER_SIZE\n");
    else 
      if (ctx->verbose)
        multilog (client->log, LOG_INFO, "send: TRANSFER_SIZE=%"PRIu64"\n", transfer_size);
  }
  
  // send the header memory buffer to dada_ibdb
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: post_recv on header_mb [HEADER]\n");
  for (i=0; i<ctx->n_dst; i++)
  {
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(log, LOG_ERR, "send: post_recv on header_mb [HEADER] failed\n");
      return -1;
    }
  }

  // wait for send confirmation
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: wait_recv on header_mb [HEADER]\n");
  for (i=0; i<ctx->n_dst; i++)
  {
    if (dada_ib_wait_recv (ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(log, LOG_ERR, "send: wait_recv on header_mb [HEADER] failed\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", bytes);

  return bytes;

}


/*! Transfers 1 datablock at a time to mopsr_ibdb */
int64_t mopsr_dbib_recv_block (dada_client_t* client, void * buffer, 
                               uint64_t bytes, uint64_t block_id)
{
  
  mopsr_dbib_t * ctx = (mopsr_dbib_t*) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: bytes=%"PRIu64", block_id=%"PRIu64"\n", bytes, block_id);

  unsigned int i;

  // wait for READY message so we know ibdb is ready for main transfer loop
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: recv_messages [READY DATA]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->n_dst, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "send: recv_messages [READY DATA] failed\n");
    return -1;
  } 

  // pre post a recv for the remote db buffer to be filled 
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv [BLOCK ID]\n");
  for (i=0; i < ctx->n_dst; i++)
  {
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send_block: post_recv [BLOCK ID] failed\n");
      return -1;
    }
  }

  // if we are not sending a full block, then this will be the end of the observation
  if (bytes != ib_cms[0]->bufs_size)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "send_block: bytes=%"PRIu64", ib_cm->bufs_size=%"PRIu64"\n", bytes, ib_cms[0]->bufs_size);
  }

  // tell ibdb how many bytes we are sending
  uint64_t bytes_to_xfer = bytes / ctx->n_dst;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES TO XFER]=%"PRIu64"\n", bytes_to_xfer);
  if (dada_ib_send_messages (ib_cms, ctx->n_dst, DADA_IB_BYTES_TO_XFER_KEY, bytes_to_xfer) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES TO XFER] failed\n");
    return -1;
  }

  // wait for the remote buffer information to be sent from ibdb
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: recv_messages [BLOCK ID]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->n_dst, 0) < 0)
  {
    multilog(log, LOG_ERR, "send_block: recv_messages [BLOCK ID] failed\n");
    return -1;
  }

  // 2 stack arrays for remote keys and addresses
  uintptr_t remote_buf_va[ctx->n_dst];
  uint32_t remote_buf_rkey[ctx->n_dst];
  for (i=0; i < ctx->n_dst; i++)
  {
    remote_buf_va[i] = (uintptr_t) ib_cms[i]->sync_from_val[0];
    remote_buf_rkey[i] = (uint32_t) ib_cms[i]->sync_from_val[1];
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "send_block: local_block_id=%"PRIu64", remote_buf_va=%p, "
                "remote_buf_rkey=%p\n", block_id, remote_buf_va[i], remote_buf_rkey[i]);
  }

  const unsigned int src_ant = ctx->ant_last - ctx->ant_first;
  // distance between samples from same antenna in src (but different channels)
  const unsigned int src_ant_stride = src_ant * 2;
  // distance between samples from same antenna in dst (only 1 channel, but all antennae)
  const unsigned int dst_ant_stride = ctx->nant * 2;
  // distance between samples from different time epochs
  const unsigned int frame_stride = ctx->nchan * src_ant_stride;
  const uint64_t n_frame = bytes / frame_stride;

  // setup fixed antenna offset for subsequenet operations
  for (i=0; i < ctx->n_dst; i++)
  {
    remote_buf_va[i] += (src_ant_stride * ctx->ant_first);
  }

  // RDMA TRANSFER RUNS NOW!
  uint64_t iframe;
  unsigned int ichan;

  struct ibv_sge       sge;
  struct ibv_send_wr   send_wr = { };
  struct ibv_send_wr * bad_send_wr;

  sge.addr   = (uintptr_t) buffer;
  sge.length = frame_stride;
  sge.lkey   = ib_cms[0]->local_blocks[block_id].buf_lkey,

  send_wr.sg_list  = &sge;
  send_wr.num_sge  = 1;
  send_wr.next     = NULL;

  for (iframe = 0; iframe < bytes; iframe += frame_stride)
  {
    // here dst / aka channel
    for (ichan = 0; ichan < ctx->n_dst; ichan++)
    {
      // increase remote channel memory address by relevant amount 
      send_wr.wr.rdma.remote_addr = remote_buf_va[ichan] + (iframe * dst_ant_stride);
      send_wr.wr.rdma.rkey        = remote_buf_rkey[ichan];
      send_wr.opcode              = IBV_WR_RDMA_WRITE;
      send_wr.send_flags          = 0;
      send_wr.wr_id               = iframe;

      if (ibv_post_send (ib_cms[ichan]->cm_id->qp, &send_wr, &bad_send_wr))
      {
        multilog(log, LOG_ERR, "send_block: ibv_post_recv [iframe=%"PRIu64", ichan=%d failed\n", iframe, ichan);
        return -1;
      }

      // incremement address pointers by 1 channel's stride
      sge.addr += src_ant_stride;
    }
  }

  // post a recv for the next call to send_block
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv [READY DATA]\n");
  for (i=0; i < ctx->n_dst; i++)
  {
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send_block: post_recv [READY DATA] failed\n");
      return -1;
    }
  }

  // tell each ibdb how many bytes we actually sent
  uint64_t bytes_xferred = bytes / ctx->n_dst;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED]=%"PRIu64"\n", bytes_xferred);
  if (dada_ib_send_messages (ib_cms, ctx->n_dst, DADA_IB_BYTES_XFERRED_KEY, bytes_xferred) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED] failed\n");
    return -1;
  }

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: sent %"PRIu64" bytes\n", bytes);

  return bytes;
}

/*
 * required initialization of IB device and associate verb structs
 */
int mopsr_dbib_ib_init (mopsr_dbib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "mopsr_dbib_ib_init()\n");

  // infiniband post_recv size will be 
  unsigned int chunk_size = (ctx->ant_last - ctx->ant_first) * 2;
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
  uint64_t send_depth = db_bufsz / modulo_size;

  // alloc memory for ib_cm ptrs
  ctx->ib_cms = (dada_ib_cm_t **) malloc(sizeof(dada_ib_cm_t *) * ctx->n_dst);
  if (!ctx->ib_cms)
  {
    multilog(log, LOG_ERR, "ib_init: could not allocate memory\n");
    return -1;
  }

  // create the CM and CM channel
  unsigned idst;
  for (idst=0; idst < ctx->n_dst; idst++)
  {
    ctx->ib_cms[idst] = dada_ib_create_cm(db_nbufs, log);
    if (!ctx->ib_cms[idst])
    {
      multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm on dst %d of %d  failed\n", idst, ctx->n_dst);
      return -1;
    }

    ctx->ib_cms[idst]->verbose = ctx->verbose;
    ctx->ib_cms[idst]->send_depth = send_depth;
    ctx->ib_cms[idst]->recv_depth = 1;
    ctx->ib_cms[idst]->bufs_size = db_bufsz;
    ctx->ib_cms[idst]->header_size = hb_bufsz;
    ctx->ib_cms[idst]->db_buffers = db_buffers;
    ctx->ib_cms[idst]->port = 

  }

  return 0;
}

int mopsr_dbib_destroy (mopsr_dbib_t * ctx) 
{

  unsigned i=0;
  for (i=0; i<ctx->n_dst; i++)
  {
    if (dada_ib_destroy(ctx->ib_cms[i]) < 0)
    {
      multilog(ctx->log, LOG_ERR, "dada_ib_destroy for %d failed\n", i);
    }
  }


}

int mopsr_dbib_open_connections (mopsr_dbib_t * ctx, multilog_t * log)
{
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  mopsr_bf_conn_t * conns = ctx->conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_dbib_open_connections()\n");

  unsigned i = 0;
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ctx->n_dst);

  // start all the connection threads
  for (i=0; i<ctx->n_dst; i++)
  {
    if (!ib_cms[i]->cm_connected)
    {
      if (ctx->verbose > 1)
        multilog (ctx->log, LOG_INFO, "open_connections: mopsr_dbib_init_thread ib_cms[%d]=%p\n",
                  i, ctx->ib_cms[i]);

      rval = pthread_create(&(connections[i]), 0, (void *) mopsr_dbib_init_thread,
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
  for (i=0; i<ctx->n_dst; i++)
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

  return 0;
}


/*
 * Thread to open connection between sender and receivers
 */
void * mopsr_dbib_init_thread (void * arg)
{

  mopsr_bf_conn_t * conn = (mopsr_bf_conn_t *) arg;

  dada_ib_cm_t * ib_cm = conn->ib_cm;

  multilog_t * log = ib_cm->log;

  if (ib_cm->verbose > 1)
    multilog (log, LOG_INFO, "ib_init_thread: ib_cm=%p\n", ib_cm);

  ib_cm->cm_connected = 0;
  ib_cm->ib_connected = 0;

  // resolve the route to the server
  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: connecting to cm at %s:%d\n", conn->host, conn->port);
  if (dada_ib_connect_cm (ib_cm, conn->host, conn->port) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_connect_cm failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // create the IB verb structures necessary
  if (dada_ib_create_verbs (ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_create_verbs failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // register each data block buffer for use with IB transport
  int flags = IBV_ACCESS_LOCAL_WRITE;
  if (dada_ib_reg_buffers(ib_cm, ib_cm->db_buffers, ib_cm->bufs_size, flags) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_register_memory_buffers failed for %s:%d\n", conn->host, conn->port);
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // register a local buffer for the header
  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "ib_init_thread: could not allocate memory for header\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: dada_ib_reg_buffer(header)\n");
  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_ERR, "ib_init_thread: could not register header mb\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }
  ib_cm->header_mb->wr_id = 10000;

  // create the Queue Pair
  if (dada_ib_create_qp (ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_create_qp failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // post recv on sync from for the ready message, prior to sending header
  if (ib_cm->verbose)
    multilog(log, LOG_INFO, "ib_init_thread: post_recv [READY NEW XFER]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: post_recv [READY NEW XFER] failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose)
    multilog (log, LOG_INFO, "ib_init_thread: dada_ib_connect()\n");

  // connect to the receiver 
  if (dada_ib_connect(ib_cm) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_connect failed to %s:%d\n"conn->host, conn->port);
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose)
    multilog (log, LOG_INFO, "ib_init_thread: connection established\n");

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
  mopsr_dbib_t dbib = MOPSR_DBIB_INIT;

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
          fprintf (stderr,"mopsr_dbib: could not parse key from %s\n",optarg);
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
  log = multilog_open ("mopsr_dbib", 0);

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

  client->open_function     = mopsr_dbib_open;
  client->io_function       = mopsr_dbib_recv;
  client->io_block_function = mopsr_dbib_recv_block;
  client->close_function    = mopsr_dbib_close;
  client->direction         = dada_client_reader;

  client->quiet = 1;
  client->context = &dbib;
  dbib.verbose = verbose;

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  // Initiase IB resources
  if (mopsr_dbib_ib_init (&dbib, hdu, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  // parse the cornerturn configuration
  dbib.conn_info = mopsr_parse_cornerturn_cfg (cornerturn_cfg, &dbib);
  if (!dbib.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn configuration file\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  // open IB connections with the receivers
  if (mopsr_dbib_open_connections (&dbib, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to open IB connections\n");
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

    if (quit || dbib.quit)
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

  if (mopsr_dbib_destroy (&dbib) < 0)
  {
    multilog(log, LOG_ERR, "mopsr_dbib_destroy failed\n");
  }

  return EXIT_SUCCESS;
}
