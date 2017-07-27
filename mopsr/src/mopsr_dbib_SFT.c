/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 *  Use multiple infiniband connections for perform the cornerturn for 
 *  detected, beam-formed data to the beam-search streams.
 *  This application is coupled with mopsr_ibdb_SFT and the 
 *  mopsr_beams_cornerturn.cfg file in the ../config directory 
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
           "mopsr_dbib_SFT [options] send_id cornerturn.cfg\n"
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
  mopsr_bp_ib_t* ctx = (mopsr_bp_ib_t*) client->context;

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
  mopsr_bp_ib_t* ctx = (mopsr_bp_ib_t*) client->context;

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
  mopsr_bp_ib_t * ctx = (mopsr_bp_ib_t *) client->context;

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

  if (ascii_header_get (buffer, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NDIM from header, assuming 1\n");
    ctx->ndim = 1;
  }

  if (ascii_header_get (buffer, "NBIT", "%d", &(ctx->nbit)) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NBIT from header, assuming 32\n");
    ctx->nbit = 32;
  }
  ctx->nbyte = ctx->nbit / 8;

  if (ascii_header_get (buffer, "NPOL", "%d", &(ctx->npol)) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NPOL from header, assuming 1\n");
    ctx->npol = 1;
  }

  // ensure the number of channels in the data-block match the number of 
  // channels in the BP cornerturn configuration
  unsigned nchan;
  if (ascii_header_get (buffer, "NCHAN", "%u", &nchan) != 1)
  {
    multilog (log, LOG_ERR, "send: failed to read NCHAN from header\n");
    return -1;
  }

  unsigned nchan_send = (ctx->conn_info[0].chan_last - ctx->conn_info[0].chan_first) + 1;
  if (nchan != nchan_send)
  {
    multilog (log, LOG_ERR, "send: NCHAN in datablock [%u] did not match "
              "cornerturn configuration [%u]\n",nchan, nchan_send);
    return -1;
  }

  // transformations required for the detected data streams are
  // NBEAM : done via sender (e.g. 352 -> 32)
  // NCHAN : done on receiver (e.g. 1 -> 40)
  // FREQ  : done on receiver
  // BW    : done on receiver

  unsigned int old_nbeam;
  if (ascii_header_get (buffer, "NBEAM", "%d", &old_nbeam) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read NBEAM from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old NBEAM=%d\n", old_nbeam);

  uint64_t old_bytes_per_second;
  if (ascii_header_get (buffer, "BYTES_PER_SECOND", "%"PRIu64"", &old_bytes_per_second) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read BYTES_PER_SECOND from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old BYTES_PER_SECOND=%"PRIu64"\n", old_bytes_per_second);

  uint64_t old_resolution;
  if (ascii_header_get (buffer, "RESOLUTION", "%"PRIu64"", &old_resolution) != 1)
  {
    multilog (log, LOG_WARNING, "send: failed to read RESOLUTION from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old RESOLUTION=%"PRIu64"\n", old_resolution);

  uint64_t old_obs_offset;
  if (ascii_header_get (buffer, "OBS_OFFSET", "%"PRIu64"", &old_obs_offset) != 1)
  {
    multilog (log, LOG_WARNING, "recv: failed to read OBS_OFFSET from header\n");
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "recv: old OBS_OFFSET=%"PRIu64"\n", old_obs_offset);

  int64_t old_file_size;
  if (ascii_header_get (buffer, "FILE_SIZE", "%"PRIi64"", &old_file_size) != 1)
  {
    int old_file_size = -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "send: old FILE_SIZE=%"PRIi64"\n", old_file_size);

  // read the md_offets into memory
  float * md_offsets = (float *) malloc (sizeof(float) * old_nbeam);
  memset (md_offsets, 0, sizeof(float) * old_nbeam);
  char * md_list = (char *) malloc (sizeof(char) * 12 * old_nbeam);
  char have_beam_md_offsets = 0;
   
  // extract the module identifiers
  if (ascii_header_get (client->header, "BEAM_MD_OFFSETS", "%s", md_list) == 1)
  {
    have_beam_md_offsets = 1;
    const char *sep = ",";
    char * saveptr;
    char * str = strtok_r(md_list, sep, &saveptr);
    unsigned ibeam=0;
    while (str && ibeam<old_nbeam)
    {
      sscanf(str,"%f", &(md_offsets[ibeam]));
      str = strtok_r(NULL, sep, &saveptr);
      ibeam++;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: ascii_header_del BEAM_MD_OFFSETS");
    if (ascii_header_del (buffer, "BEAM_MD_OFFSETS") < 0)
    {
      multilog (log, LOG_ERR, "open: failed to delete BEAM_MD_OFFSETS from the header\n");
      return -1;
    }
  }
  else
    multilog (log, LOG_WARNING, "send: failed to read BEAM_MD_OFFSETS from header\n");

  if (old_nbeam > 1)
  {
    if (ctx->verbose)
    multilog (log, LOG_INFO, "open: setting ORDER=SFT\n");
    if (ascii_header_set (buffer, "ORDER", "%s", "SFT") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=SFT in outgoing header\n");
      return -1;
    }
  } 
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: setting ORDER=FT\n");
    if (ascii_header_set (buffer, "ORDER", "%s", "FT") < 0)
    {
      multilog (log, LOG_ERR, "open: could not set ORDER=FT in outgoing header\n");
      return -1;
    }
  }

  // post recv on sync_from for the READY message
  for (i=0; i<ctx->nconn; i++)
  {
    // Need to manipulate the connection specific header buffer
    memcpy (ib_cms[i]->header_mb->buffer, buffer, bytes);

    // determine the new nbeam for this connection
    unsigned new_nbeam = (ctx->conn_info[i].beam_last - ctx->conn_info[i].beam_first) + 1;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting NBEAM=%d in header for connection id=%d\n", new_nbeam, i);
    if (ascii_header_set (ib_cms[i]->header_mb->buffer, "NBEAM", "%d", new_nbeam) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set NBEAM=%d in header for connection id=%d\n", new_nbeam, i);
    }

    // is the factor by which the data rate is changed on the sending side only
    uint64_t new_bytes_per_second = old_bytes_per_second * new_nbeam / old_nbeam;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
    if (ascii_header_set (ib_cms[i]->header_mb->buffer, "BYTES_PER_SECOND", "%"PRIu64"", new_bytes_per_second) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
    }

    uint64_t new_resolution = old_resolution * new_nbeam / old_nbeam;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting RESOLUTION=%"PRIu64" in header\n", new_resolution);
    if (ascii_header_set (ib_cms[i]->header_mb->buffer, "RESOLUTION", "%"PRIu64"", new_resolution) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set RESOLUTION=%"PRIu64" in header\n", new_resolution);
    }

    uint64_t new_obs_offset = old_obs_offset * new_nbeam / old_nbeam;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
    if (ascii_header_set (ib_cms[i]->header_mb->buffer, "OBS_OFFSET", "%"PRIu64"", new_obs_offset) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set OBS_OFFSET=%"PRIu64" in header\n", new_obs_offset);
    }

    if (old_file_size > 0)
    {
      int64_t new_file_size = old_file_size * new_nbeam / old_nbeam;
      if (ctx->verbose)
        multilog (log, LOG_INFO, "send: setting FILE_SIZE=%"PRIi64" in header\n", new_file_size);
      if (ascii_header_set (ib_cms[i]->header_mb->buffer, "FILE_SIZE", "%"PRIi64"", new_file_size) < 0)
      {
        multilog (log, LOG_WARNING, "send: failed to set FILE_SIZE=%"PRIi64" in header\n", new_file_size);
      }
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "send: setting NBEAM=%d in header\n",new_nbeam);
    if (ascii_header_set (ib_cms[i]->header_mb->buffer, "NBEAM", "%d", new_nbeam) < 0)
    {
      multilog (log, LOG_WARNING, "send: failed to set NBEAM=%d in header\n", new_nbeam);
    }

    if (have_beam_md_offsets)
    {
      // copy the header to the header memory buffer
      char keyword[64];
      md_list[0] = '\0';
      // each beam will have a unique MD_OFFSET angle
      unsigned ibeam;
      for (ibeam=ctx->conn_info[i].beam_first; ibeam<ctx->conn_info[i].beam_last; ibeam++)
      {
        sprintf (keyword, "%4.3f", md_offsets[ibeam]);
        strcat (md_list, keyword);
        if (ibeam < ctx->conn_info[i].beam_last)
          strcat (md_list, ",");
        else
        {
          if (ascii_header_set (ib_cms[i]->header_mb->buffer, "BEAM_MD_OFFSETS", "%s", md_list) < 0)
          {
            multilog (log, LOG_WARNING, "open: failed to write BEAM_MD_OFFSETS=%s "
                     "to the outgoing header\n", md_list);
          }
          md_list[0] = '\0';
        }
      }
    }
  }

  free (md_offsets);
  free (md_list);

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
  
  mopsr_bp_ib_t * ctx = (mopsr_bp_ib_t*) client->context;

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
  const uint64_t bytes_to_send_per_beam = bytes / ctx->nbeam;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES TO XFER]=%"PRIu64" (per beam)\n", bytes_to_send_per_beam);
  if (dada_ib_send_messages (ib_cms, ctx->nconn, DADA_IB_BYTES_TO_XFER_KEY, bytes_to_send_per_beam) < 0)
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
  for (i=0; i<ctx->nconn; i++)
  {
    remote_buf_va[i] = (uintptr_t) ib_cms[i]->sync_from_val[0];
    remote_buf_rkey[i] = (uint32_t) ib_cms[i]->sync_from_val[1];
    if (ctx->verbose)
      multilog (log, LOG_INFO, "send_block: [%d] local_block_id=%"PRIu64", remote_buf_va=%p, "
                "remote_buf_rkey=%p\n", i, block_id, remote_buf_va[i], remote_buf_rkey[i]);
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "send_block: bytes=%"PRIu64" nbeam_send=%u "
              "nchan_send=%u ndim=%u nbyte=%u\n", bytes, ctx->nbeam, ctx->conn_info[0].nchan,
              ctx->ndim, ctx->nbyte);

  char * buf_ptr = (char *) buffer;

  struct ibv_sge       sge;
  struct ibv_send_wr   send_wr = { };
  struct ibv_send_wr * bad_send_wr;

  sge.length = bytes_to_send_per_beam;

  send_wr.sg_list  = &sge;
  send_wr.num_sge  = 1;
  send_wr.next     = NULL;

  unsigned ibeam, obeam;
  for (i=0; i<ctx->nconn; i++)
  {
    // local access key 
    sge.lkey = ib_cms[i]->local_blocks[block_id].buf_lkey;

    unsigned nbeam = ctx->conn_info[i].nbeam;
    for (obeam=0; obeam<nbeam; obeam++)
    {
      ibeam = obeam + ctx->conn_info[i].beam_first;

      sge.addr = (uintptr_t) (buf_ptr + ctx->in_offsets[i][obeam]);
      send_wr.wr.rdma.remote_addr = remote_buf_va[i] + ctx->out_offsets[i][obeam];
      send_wr.wr.rdma.rkey        = remote_buf_rkey[i];
      send_wr.opcode              = IBV_WR_RDMA_WRITE;
      send_wr.send_flags          = 0;
      send_wr.wr_id               = ibeam;

      if (ibv_post_send (ib_cms[i]->cm_id->qp, &send_wr, &bad_send_wr))
      {
        multilog(log, LOG_ERR, "send_block: ibv_post_send [iconn=%u, ibeam=%u] failed\n", i, ibeam);
        return -1;
      }
    }
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
  if (ctx->verbose)
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED]=%"PRIu64"\n", bytes_to_send_per_beam);
  if (dada_ib_send_messages (ib_cms, ctx->nconn, DADA_IB_BYTES_XFERRED_KEY, bytes_to_send_per_beam) < 0)
  {
    multilog(log, LOG_INFO, "send_block: send_messages [BYTES XFERRED]=%"PRIu64" failed\n", bytes_to_send_per_beam);
    return -1;
  }

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "send_block: sent %"PRIu64" bytes\n", bytes);

  return (int64_t) bytes;
}

/*
 * required initialization of IB device and associate verb structs
 */
int mopsr_dbib_ib_init (mopsr_bp_ib_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "mopsr_dbib_ib_init()\n");

  // size of a "sample", DB size must be a multiple of this
  const unsigned modulo_size = ctx->conn_info[0].nchan * ctx->nbeam * ctx->ndim * ctx->nbyte;

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
    
    unsigned nbeam = (ctx->conn_info[i].beam_last - ctx->conn_info[i].beam_first) + 1;

    // number of outstanding post_sends on each connection to a receiver (+1)
    ctx->ib_cms[i]->send_depth = nbeam + 1;
    ctx->ib_cms[i]->recv_depth = 1;
    ctx->ib_cms[i]->bufs_size = db_bufsz;
    ctx->ib_cms[i]->header_size = hb_bufsz;
    ctx->ib_cms[i]->db_buffers = db_buffers;
  }

  return 0;
}

int mopsr_dbib_destroy (mopsr_bp_ib_t * ctx) 
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

int mopsr_dbib_open_connections (mopsr_bp_ib_t * ctx, multilog_t * log)
{
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  mopsr_bp_conn_t * conns = ctx->conn_info;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_dbib_open_connections()\n");

  unsigned i = 0;
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ctx->nconn);

  char all_connected = 0;
  int attempts = 0;

  while (!all_connected && attempts < 1)
  {
    // start all the connection threads
    for (i=0; i<ctx->nconn; i++)
    {
      if (ib_cms[i]->cm_connected == 0)
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
        if (ib_cms[i]->cm_connected == 0)
        {
          multilog (ctx->log, LOG_WARNING, "ib_cms[%d] releasing resources\n", i);
          if (dada_ib_disconnect (ib_cms[i]))
          {
            multilog(ctx->log, LOG_ERR, "dada_ib_disconnect for %d failed\n", i);
            return -1;
          }
          init_cm_ok = 0;
        }
      }
      else
      {
        // set to 2 to indicate connection is established
        ib_cms[i]->cm_connected = 2;
      }
    }

    if (init_cm_ok)
    {
      all_connected = 1;
    }
    else
    {
      multilog (ctx->log, LOG_INFO, "open_connections: sleep(1)\n");
      sleep(1);
    }
    attempts++;
  }

  free(connections);
  if (!all_connected)
  {
    multilog(ctx->log, LOG_ERR, "open_connections: failed to init CM connections\n");
    return -1;
  }

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "open_connections: connections opened\n");

  return 0;
}


/*
 * Thread to open connection between sender and receivers
 */
void * mopsr_dbib_init_thread (void * arg)
{
  mopsr_bp_conn_t * conn = (mopsr_bp_conn_t *) arg;

  dada_ib_cm_t * ib_cm = conn->ib_cm;

  multilog_t * log = ib_cm->log;

  if (ib_cm->verbose > 1)
    multilog (log, LOG_INFO, "ib_init_thread: ib_cm=%p\n", ib_cm);

  ib_cm->cm_connected = 0;
  ib_cm->ib_connected = 0;
  ib_cm->port = conn->port;

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

  // dont overwhelm the receiver
  unsigned irecv = conn->irecv;
  if (irecv < conn->isend)
    irecv += conn->nsend;

  //unsigned wait_us = 2e6 + (10000 * (irecv - conn->isend));
  //multilog (log, LOG_INFO, "ib_init_thread: usleep(%d)\n", wait_us);
  //usleep(wait_us);

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

void setup_conn_offsets (mopsr_bp_ib_t * ctx, uint64_t bufsz)
{
  multilog_t * log = ctx->log;

  // sender beam stride in bytes
  uint64_t src_beam_stride = bufsz / ctx->nbeam;

  const uint64_t bytes_per_sample = (ctx->conn_info[0].nchan * ctx->nbit * ctx->ndim) / 8;

  // number of time samples in the block
  const uint64_t nsamp = src_beam_stride / bytes_per_sample;

  // channel stride in bytes
  const unsigned chan_stride = (nsamp * ctx->ndim * ctx->npol * ctx->nbit) / 8;

  // stride for a beam on cornerturn destination
  const uint64_t dst_beam_stride = (nsamp * ctx->nchan * ctx->ndim * ctx->nbit * ctx->npol) / 8;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "setup_conn_offsets: ctx->conn_info[0].chan_first=%u, "
              "nsamp=%"PRIu64", dst_beam_stride=%"PRIu64"\n", ctx->conn_info[0].chan_first,
              nsamp, dst_beam_stride);

  ctx->in_offsets  = (uint64_t **) malloc(sizeof(uint64_t *) * ctx->nconn);
  ctx->out_offsets = (uint64_t **) malloc(sizeof(uint64_t *) * ctx->nconn);

  unsigned iconn;
  for (iconn=0; iconn < ctx->nconn; iconn++)
  {
    if (ctx->verbose)
    { 
      multilog (log, LOG_INFO, "conn_info[%d] channels: %d -> %d\n", iconn, ctx->conn_info[iconn].chan_first, ctx->conn_info[iconn].chan_last);
      multilog (log, LOG_INFO, "conn_info[%d] beams: %d -> %d\n", iconn, ctx->conn_info[iconn].beam_first, ctx->conn_info[iconn].beam_last);
    }

    const unsigned nbeam = ctx->conn_info[iconn].nbeam;
    ctx->in_offsets[iconn]  = (uint64_t *) malloc (sizeof(uint64_t) * nbeam);
    ctx->out_offsets[iconn] = (uint64_t *) malloc (sizeof(uint64_t) * nbeam);

    unsigned obeam;
    const unsigned ichan = ctx->conn_info[iconn].chan_first;

    for (obeam=0; obeam<nbeam; obeam++)
    {
      const unsigned ibeam = obeam + ctx->conn_info[iconn].beam_first;
      ctx->in_offsets[iconn][obeam]  = (ibeam * src_beam_stride);
      ctx->out_offsets[iconn][obeam] = (obeam * dst_beam_stride) + (ichan * chan_stride); 
    }
  }
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
  mopsr_bp_ib_t ctx = MOPSR_BP_IB_INIT;

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
  ctx.conn_info = mopsr_setup_bp_cornerturn_send (cornerturn_cfg, &ctx, send_id);
  if (!ctx.conn_info)
  {
    multilog (log, LOG_ERR, "Failed to parse cornerturn configuration file\n");
    dada_hdu_unlock_read(hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  uint64_t bufsz = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // configure connection input and output offsets
  setup_conn_offsets (&ctx, bufsz);

  if (ctx.verbose)
  {
    unsigned iconn;
    for (iconn=0; iconn < ctx.nconn; iconn++)
    {
      multilog (log, LOG_INFO, "conn_info[%d] beams: %d -> %d\n", iconn, 
                ctx.conn_info[iconn].beam_first, ctx.conn_info[iconn].beam_last);
    }
  }

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
