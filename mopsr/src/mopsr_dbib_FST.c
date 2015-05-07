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
           "mopsr_dbib [options] send_id cornerturn.cfg\n"
     " -k <key>          hexadecimal shared memory key  [default: %x]\n"
     " -s                single transfer only\n"
     " -v                verbose output\n"
     " -h                print this usage\n\n"
     " send_id           sender ID as defined in config file\n"
     " cornerturn.cfg    ascii configuration file defining cornerturn\n",
     DADA_IB_DEFAULT_CHUNK_SIZE, 
     DADA_DEFAULT_BLOCK_KEY);
}


/*! Function that opens the data transfer target */
int mopsr_dbib_open (dada_client_t* client)
{

  // the mopsr_dbib specific data
  assert (client != 0);
  mopsr_bf_ib_t* ctx = (mopsr_bf_ib_t*) client->context;

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

  ctx->obs_ending = 0;

  return 0;
}

/*! Function that closes the data file */
int mopsr_dbib_close (dada_client_t* client, uint64_t bytes_written)
{
  // the mopsr_dbib specific data
  mopsr_bf_ib_t* ctx = (mopsr_bf_ib_t*) client->context;

  // status and error logging facility
  multilog_t* log = client->log;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "mopsr_dbib_close()\n");

  // wait for READY message so we know ibdb is ready for main transfer loop
  if (ctx->verbose)
    multilog(log, LOG_INFO, "close: recv_messages [READY DATA]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->nconn, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "close: recv_messages [READY DATA] failed\n");
    return -1;
  }

  // if we expect to send another observation
  if (!ctx->quit)
  {
    // post recv on sync_from for the READY HDR message
    if (ctx->verbose)
      multilog(log, LOG_INFO, "close: post_recv [READY HDR]\n");
    unsigned i;
    for (i=0; i<ctx->nconn; i++)
    {
      if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
      {
        multilog(log, LOG_ERR, "close: post_recv [READY HDR] failed\n");
        return -1;
      }
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: send_messages [EOD]\n");
  if (dada_ib_send_messages (ib_cms, ctx->nconn, DADA_IB_BYTES_TO_XFER_KEY, 0) < 0)
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
int64_t mopsr_dbib_send (dada_client_t* client, void * buffer, uint64_t bytes)
{
  mopsr_bf_ib_t * ctx = (mopsr_bf_ib_t *) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  unsigned int i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "mopsr_dbib_send()\n");

  for (i=0; i<ctx->nconn; i++)
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
  if (dada_ib_recv_messages (ib_cms, ctx->nconn, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "send: recv_messages [READY HDR] failed\n");
    return -1;
  }
  
  // post recv on sync_from for the READY message
  for (i=0; i<ctx->nconn; i++)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "send: post_recv [READY DATA]\n");
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send: post_recv [READY DATA] failed\n");
      return -1;
    }
  }

  unsigned int old_nchan, new_nchan;
  new_nchan = 1;
  if (ascii_header_get (buffer, "NCHAN", "%d", &old_nchan) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NCHAN from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old NCHAN=%d\n", old_nchan);

  unsigned int nant;
  if (ascii_header_get (buffer, "NANT", "%d", &nant) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NANT from header\n");
  }
  unsigned iant;
  char ant_id[16];
  for (iant=0; iant<nant; iant++)
  {
    sprintf (ant_id, "ANT_ID_%u", iant);
    if (ascii_header_del (buffer, ant_id) < 0)
      multilog (log, LOG_WARNING, "send: failed to delete %s from header\n", ant_id);
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
  float chan_bw  = old_bw / (float) old_nchan;

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "send: freq_low=%f\n", freq_low);
    multilog (log, LOG_INFO, "send: chan_bw=%f\n", chan_bw);
  }

  new_bw = chan_bw;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting BW=%f in header\n", new_bw);
  if (ascii_header_set (buffer, "BW", "%f", new_bw) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set BW=%f in header\n", new_bw);
  }

  new_bytes_per_second = old_bytes_per_second / old_nchan;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  if (ascii_header_set (buffer, "BYTES_PER_SECOND", "%"PRIu64"", new_bytes_per_second) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  }

  new_resolution = old_resolution / old_nchan;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting RESOLUTION=%"PRIu64" in header\n", new_resolution);
  if (ascii_header_set (buffer, "RESOLUTION", "%"PRIu64"", new_resolution) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set RESOLUTION=%"PRIu64" in header\n", new_resolution);
  }

  if (old_file_size > 0)
  {
    new_file_size = old_file_size / old_nchan;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    if (ascii_header_set (buffer, "FILE_SIZE", "%"PRIi64"", new_file_size) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set FILE_SIZE=%"PRIi64" in header\n", new_file_size);
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: setting NCHAN=%d in header\n",new_nchan);
  if (ascii_header_set (buffer, "NCHAN", "%d", new_nchan) < 0)
  {
    multilog (log, LOG_WARNING, "send: failed to set NCHAN=%d in header\n", new_nchan);
  }

  if (ctx->verbose)
   multilog (log, LOG_INFO, "open: setting ORDER=ST\n");
  if (ascii_header_set (buffer, "ORDER", "%s", "ST") < 0)
  {
    multilog (log, LOG_ERR, "open: could not set ORDER=ST in outgoing header\n");
    return -1;
  }



  // copy the header to the header memory buffer
  for (i=0; i<ctx->nconn; i++)
  {
    // each conn / channel will have a unique FREQ
    new_freq = freq_low + (chan_bw / 2) + (chan_bw * i);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting FREQ=%f in header\n", new_freq);
    if (ascii_header_set (buffer, "FREQ", "%f", new_freq) < 0)
    { 
      multilog (log, LOG_WARNING, "send: failed to set FREQ=%f in header\n", new_freq);
    }

    memcpy (ib_cms[i]->header_mb->buffer, buffer, bytes);
  }

  uint64_t transfer_size = 0;
  for (i=0; i<ctx->nconn; i++)
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
    multilog(log, LOG_INFO, "send: post_send on header_mb [HEADER]\n");
  for (i=0; i<ctx->nconn; i++)
  {
    if (dada_ib_post_send (ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(log, LOG_ERR, "send: post_send on header_mb [HEADER] failed\n");
      return -1;
    }
  }

  // wait for send confirmation
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: wait_send on header_mb [HEADER]\n");
  for (i=0; i<ctx->nconn; i++)
  {
    if (dada_ib_wait_send (ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(log, LOG_ERR, "send: wait_send on header_mb [HEADER] failed\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog(log, LOG_INFO, "send: returning %"PRIu64" bytes\n", bytes);

  return bytes;

}


/*! Transfers 1 datablock at a time to mopsr_ibdb */
int64_t mopsr_dbib_send_block (dada_client_t* client, void * buffer, 
                               uint64_t bytes, uint64_t block_id)
{
  
  mopsr_bf_ib_t * ctx = (mopsr_bf_ib_t*) client->context;

  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: bytes=%"PRIu64", block_id=%"PRIu64"\n", bytes, block_id);

  unsigned int i;

  // wait for READY message so we know ibdb is ready for main transfer loop
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: recv_messages [READY DATA]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->nconn, DADA_IB_READY_KEY) < 0)
  {
    multilog(log, LOG_ERR, "send_block: recv_messages [READY DATA] failed\n");
    return -1;
  } 

  // pre post a recv for the remote db buffer to be filled 
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv [BLOCK ID]\n");
  for (i=0; i < ctx->nconn; i++)
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
    ctx->obs_ending = 1;
  }

  // tell ibdb how many bytes we are sending
  uint64_t bytes_to_xfer = bytes / ctx->nconn;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES TO XFER]=%"PRIu64"\n", bytes_to_xfer);
  if (dada_ib_send_messages (ib_cms, ctx->nconn, DADA_IB_BYTES_TO_XFER_KEY, bytes_to_xfer) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES TO XFER] failed\n");
    return -1;
  }

  // wait for the remote buffer information to be sent from ibdb
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: recv_messages [BLOCK ID]\n");
  if (dada_ib_recv_messages (ib_cms, ctx->nconn, 0) < 0)
  {
    multilog(log, LOG_ERR, "send_block: recv_messages [BLOCK ID] failed\n");
    return -1;
  }

  // 2 stack arrays for remote keys and addresses
  uintptr_t remote_buf_va[ctx->nconn];
  uint32_t remote_buf_rkey[ctx->nconn];
  for (i=0; i < ctx->nconn; i++)
  {
    remote_buf_va[i] = (uintptr_t) ib_cms[i]->sync_from_val[0];
    remote_buf_rkey[i] = (uint32_t) ib_cms[i]->sync_from_val[1];
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send_block: [%d] local_block_id=%"PRIu64", remote_buf_va=%p, "
                "remote_buf_rkey=%p\n", i, block_id, remote_buf_va[i], remote_buf_rkey[i]);
  }

  const unsigned int ndim = 2;

  // number of antenna's in source data
  const unsigned int src_nant = (ctx->conn_info[0].ant_last - ctx->conn_info[0].ant_first) + 1;

  // number of time samples in the block
  const uint64_t nsamp = bytes / (ctx->nchan * src_nant * ndim);

  // number of bytes to be copied for each channel
  const uint64_t chan_bytes = bytes / ctx->nchan;

  // destination remote address offset for these antenna
  const uint64_t dst_ant_offset = nsamp * ndim * ctx->conn_info[0].ant_first;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send_block: ctx->conn_info[0].ant_first=%u, nsamp=%"PRIu64", dst_ant_offset=%"PRIu64"\n", ctx->conn_info[0].ant_first, nsamp, dst_ant_offset);

  unsigned int ichan;

  // RDMA WRITES now!

  struct ibv_sge       sge;
  struct ibv_send_wr   send_wr = { };
  struct ibv_send_wr * bad_send_wr;

  sge.addr   = (uintptr_t) buffer;
  sge.length = chan_bytes;

  send_wr.sg_list  = &sge;
  send_wr.num_sge  = 1;
  send_wr.next     = NULL;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send_block: bytes=%"PRIu64" nsamp=%"PRIu64" chan_bytes=%"PRIu64"\n", bytes, nsamp, chan_bytes );

  // we have 1 channel per connection !
  for (ichan=0; ichan<ctx->nconn; ichan++)
  {
    sge.lkey = ib_cms[ichan]->local_blocks[block_id].buf_lkey,

    // increase remote channel memory address by relevant amount 
    send_wr.wr.rdma.remote_addr = remote_buf_va[ichan] + dst_ant_offset;
    send_wr.wr.rdma.rkey        = remote_buf_rkey[ichan];
    send_wr.opcode              = IBV_WR_RDMA_WRITE;
    send_wr.send_flags          = 0;
    send_wr.wr_id               = ichan;

    if (ibv_post_send (ib_cms[ichan]->cm_id->qp, &send_wr, &bad_send_wr))
    {
      multilog(log, LOG_ERR, "send_block: ibv_post_send [ichan=%d] failed\n", ichan);
      return -1;
    }

    // incremement local address pointers by 1 channel's stride
    sge.addr += chan_bytes;
  }

  // post a recv for the next call to send_block
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: post_recv [READY DATA]\n");
  for (i=0; i < ctx->nconn; i++)
  {
    if (dada_ib_post_recv (ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(log, LOG_ERR, "send_block: post_recv [READY DATA] failed\n");
      return -1;
    }
  }

  // tell each ibdb how many bytes we actually sent
  uint64_t bytes_xferred = bytes / ctx->nconn;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED]=%"PRIu64"\n", bytes_xferred);
  if (dada_ib_send_messages (ib_cms, ctx->nconn, DADA_IB_BYTES_XFERRED_KEY, bytes_xferred) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED]=%"PRIu64" failed\n", bytes_xferred);
    return -1;
  }

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: sent %"PRIu64" bytes\n", bytes);

  return (int64_t) bytes;
}

/*
 * required initialization of IB device and associate verb structs
 */
int mopsr_dbib_ib_init (mopsr_bf_ib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "mopsr_dbib_ib_init()\n");

  const unsigned ndim = 2;
  const unsigned src_nant = (ctx->conn_info[0].ant_last - ctx->conn_info[0].ant_first) + 1;
  const unsigned int modulo_size = ctx->nchan * src_nant * ndim;

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

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "ib_init: modulo_size=%"PRIu64"\n", modulo_size);
  // alloc memory for ib_cm ptrs
  ctx->ib_cms = (dada_ib_cm_t **) malloc(sizeof(dada_ib_cm_t *) * ctx->nconn);
  if (!ctx->ib_cms)
  {
    multilog(log, LOG_ERR, "ib_init: could not allocate memory\n");
    return -1;
  }

  // create the CM and CM channel
  unsigned i;
  for (i=0; i < ctx->nconn; i++)
  {
    ctx->ib_cms[i] = dada_ib_create_cm(db_nbufs, log);
    if (!ctx->ib_cms[i])
    {
      multilog(log, LOG_ERR, "ib_init: dada_ib_create_cm on dst %d of %d  failed\n", i, ctx->nconn);
      return -1;
    }
    ctx->conn_info[i].ib_cm = ctx->ib_cms[i];
    ctx->ib_cms[i]->verbose = ctx->verbose;
    ctx->ib_cms[i]->send_depth = ctx->nchan + 1; // only ever have this many outstanding post sends
    ctx->ib_cms[i]->recv_depth = 1;
    ctx->ib_cms[i]->bufs_size = db_bufsz;
    ctx->ib_cms[i]->header_size = hb_bufsz;
    ctx->ib_cms[i]->db_buffers = db_buffers;
  }

  return 0;
}

int mopsr_dbib_destroy (mopsr_bf_ib_t * ctx) 
{
  unsigned i=0;
  int rval = 0;
  for (i=0; i<ctx->nconn; i++)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "destroy: dada_ib_client_destroy()\n");
    if (dada_ib_client_destroy(ctx->ib_cms[i]) < 0)
    {
      multilog(ctx->log, LOG_ERR, "dada_ib_client_destroy for %d failed\n", i);
      rval = -1;
    }
  }
  return rval;
}

int mopsr_dbib_open_connections (mopsr_bf_ib_t * ctx, multilog_t * log)
{
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  mopsr_bf_conn_t * conns = ctx->conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_dbib_open_connections()\n");

  unsigned i = 0;
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ctx->nconn);

  // start all the connection threads
  for (i=0; i<ctx->nconn; i++)
  {
    if (!ib_cms[i]->cm_connected)
    {
      if (ctx->verbose > 1)
        multilog (ctx->log, LOG_INFO, "open_connections: mopsr_dbib_init_thread ib_cms[%d]=%p\n",
                  i, ctx->ib_cms[i]);

      rval = pthread_create(&(connections[i]), 0, (void *) mopsr_dbib_init_thread,
                          (void *) &(conns[i]));
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
  for (i=0; i<ctx->nconn; i++)
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

  multilog (ctx->log, LOG_INFO, "open_connections: connections opened\n");

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
    multilog(log, LOG_INFO, "ib_init_thread: connecting to cm at %s:%d\n", conn->ib_host, conn->port);
  if (dada_ib_connect_cm (ib_cm, conn->ib_host, conn->port) < 0)
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
    multilog(log, LOG_ERR, "ib_init_thread: dada_ib_register_memory_buffers failed to %s:%d\n", conn->host, conn->port);
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // allocate memory required
  ib_cm->header = (char *) malloc(sizeof(char) * ib_cm->header_size);
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "ib_init_thread: could not allocate memory for header\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  // register a local buffer for the header
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
    multilog(log, LOG_INFO, "ib_init_thread: post_recv [READY HDR]\n");
  if (dada_ib_post_recv (ib_cm, ib_cm->sync_from) < 0)
  {
    multilog(log, LOG_ERR, "ib_init_thread: post_recv [READY HDR] failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose)
    multilog (log, LOG_INFO, "ib_init_thread: dada_ib_connect\n");

  // connect to the receiver 
  if (dada_ib_connect(ib_cm) < 0)
  {
    multilog (log, LOG_ERR, "ib_init_thread: dada_ib_connect failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose)
    multilog (log, LOG_INFO, "ib_init_thread: CM connection established\n");
   ib_cm->cm_connected = 1;

  pthread_exit((void *) &(ib_cm->cm_connected));
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
  mopsr_bf_ib_t ctx = MOPSR_IB_INIT;

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
  if (argc-optind != 2) {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  send_id = atoi(argv[optind]);
  cornerturn_cfg = strdup(argv[optind+1]);

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
  client->io_function       = mopsr_dbib_send;
  client->io_block_function = mopsr_dbib_send_block;
  client->close_function    = mopsr_dbib_close;
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
    multilog (log, LOG_INFO, "main: mopsr_parse_cornerturn_cfg()\n");
  ctx.conn_info = mopsr_setup_cornerturn_send (cornerturn_cfg, &ctx, send_id);
  if (!ctx.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn configuration file\n");
    dada_hdu_unlock_read(hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
  {
    unsigned iconn;
    for (iconn=0; iconn < ctx.nconn; iconn++)
    {
      multilog (log, LOG_INFO, "conn_info[%d] %d -> %d\n", iconn, ctx.conn_info[iconn].ant_first, ctx.conn_info[iconn].ant_last);
    }
  }

  // deal breaker!
  assert (ctx.nchan == ctx.nconn);

  if (ctx.verbose)
    multilog (log, LOG_INFO, "Initializing IB\n");

  // Initiase IB resources
  if (verbose)
    multilog (log, LOG_INFO, "main: mopsr_dbib_ib_init()\n");
  if (mopsr_dbib_ib_init (&ctx, hdu, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    dada_hdu_unlock_read (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
    multilog (log, LOG_INFO, "Opening Connections\n");

  // open IB connections with the receivers
  if (verbose)
    multilog (log, LOG_INFO, "main: mopsr_dbib_open_connections()\n");
  if (mopsr_dbib_open_connections (&ctx, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to open IB connections\n");

    multilog (log, LOG_INFO, "main: mopsr_dbib_destroy()\n");
    if (mopsr_dbib_destroy (&ctx) < 0)
    {
      multilog(log, LOG_ERR, "mopsr_dbib_destroy failed\n");
    }

    dada_hdu_unlock_read (hdu);
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
    multilog (log, LOG_INFO, "main: mopsr_dbib_destroy()\n");
  if (mopsr_dbib_destroy (&ctx) < 0)
  {
    multilog(log, LOG_ERR, "mopsr_dbib_destroy failed\n");
  }

  return EXIT_SUCCESS;
}

