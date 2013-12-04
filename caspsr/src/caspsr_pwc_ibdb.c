/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <math.h>

#include "caspsr_pwc_ibdb.h"
#include "caspsr_def.h"

void usage()
{
  fprintf (stdout,
     "caspsr_pwc_ibdb [options] n_distrib\n"
     " -c <port      port to open for PWCC commands [default: %d]\n"
     " -C <bytes>    default chunk size for IB transport [default: %d]\n"
     " -d            run as daemon\n"
     " -H <file>     header file for manual mode\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -l <port      multilog output port [default: %x]\n"
     " -n <secs>     seconds to acquire data for\n"
     " -p <port>     port on which to listen [default: %d]\n"
     " -s            single transfer only\n"
     " -v            verbose output\n",
     CASPSR_DEFAULT_PWC_PORT,
     DADA_IB_DEFAULT_CHUNK_SIZE,
     DADA_DEFAULT_BLOCK_KEY,
     CASPSR_DEFAULT_PWC_LOGPORT,
     DADA_DEFAULT_IBDB_PORT);
}


/*! PWCM header valid function. Returns 1 if valid, 0 otherwise */
int caspsr_pwc_ibdb_header_valid (dada_pwc_main_t* pwcm) 
{
  unsigned utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  // Check if the UTC_START is set in the header
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) 
    valid = 0;

  // Check whether the UTC_START is set to UNKNOWN
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

  return valid;
}


/*! PWCM error function. Called when buffer function returns 0 bytes
 * Returns 0=ok, 1=soft error, 2 hard error */
int caspsr_pwc_ibdb_error (dada_pwc_main_t* pwcm) 
{
  int error = 0;
  
  // check if the header is valid
  if (caspsr_pwc_ibdb_header_valid(pwcm)) 
    error = 0;
  else  
    error = 2;
  
  return error;
}

/*! PWCM xfer pending function.
 * called when the buffer_function returns 0 bytes. Returns:
 *   0  waiting for SOD on current xfer, continue waiting for data
 *   1  at the end of an xfer, but obs continuing
 *   2  at the end of an obs 
 */ 
int caspsr_pwc_ibdb_xfer_pending (dada_pwc_main_t * pwcm) {
  
  caspsr_pwc_ibdb_t * ctx = (caspsr_pwc_ibdb_t *) pwcm->context;
  
  int xfer_state = 0;
  
  // check if the header is valid 
  if (caspsr_pwc_ibdb_header_valid (pwcm)) 
  {
    // if the current xfer is ending
    if (ctx->xfer_ending)
    {
      xfer_state = DADA_XFER_ENDING;
    }
    // we are at the end of the obs, and have data in the current buffer
    //else if ((ctx->sod) || (ctx->xfer_count == -1))
    else if (ctx->obs_ending)
    {
      xfer_state = DADA_OBS_ENDING;
    }
    // at the start of first xfer and dont have SOD yet
    else 
    {
      xfer_state = DADA_XFER_NORMAL;
    }
  }
  
  return xfer_state;

}


/*! PWCM new xfer function.
 *  Start a new xfer and return next byte that the buffer function will receive
 */
uint64_t caspsr_pwc_ibdb_new_xfer (dada_pwc_main_t * pwcm) 
{

  assert (pwcm->context != 0);
  caspsr_pwc_ibdb_t * ctx = (caspsr_pwc_ibdb_t *) pwcm->context;

  if (pwcm->verbose > 1)
    multilog (pwcm->log, LOG_INFO, "caspsr_pwc_ibdb_new_xfer()\n");

  assert (ctx->ib_cms != 0);
  dada_ib_cm_t ** ib_cms = ctx->ib_cms;

  uint64_t first_byte = 0;
  uint64_t xfer_header_size = 0;
  char * xfer_header = 0;

  // receive the xfer header from ibdb
  if (pwcm->verbose)
    multilog (pwcm->log, LOG_INFO, "new_xfer: recv()\n");
  xfer_header = caspsr_pwc_ibdb_recv (pwcm, &xfer_header_size);
  if (!xfer_header)
  {
    multilog (pwcm->log, LOG_ERR, "new_xfer: caspsr_pwc_ibdb_recv failed\n"); 
    return 0;
  }

  // get the xfer_count from the xfer header
  if (ascii_header_get (xfer_header, "OBS_XFER", "%"PRIi64, &(ctx->xfer_count)) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "Could not read OBS_XFER from xfer header\n");
    multilog (pwcm->log, LOG_INFO, "%s", xfer_header);
    return 0;
  }

  // get the obs_offset from the xfer header
  if (ascii_header_get (xfer_header, "OBS_OFFSET", "%"PRIu64, &first_byte) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "Could not read OBS_OFFSET from xfer header\n");
    return 0;
  }

  if (ascii_header_get (xfer_header, "TRANSFER_SIZE", "%"PRIu64, &(ctx->xfer_size)) < 0)
    multilog (pwcm->log, LOG_WARNING, "Could not read TRANSFER_SIZE from xfer header\n");

  ctx->xfer_size *= ctx->n_distrib;

  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "new_xfer: OBS_XFER=%"PRIi64" OBS_OFFSET=%"PRIu64
                                  " TRANSFER_SIZE=%"PRIu64"\n", ctx->xfer_count, 
                                  first_byte, ctx->xfer_size);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "new_xfer: prev_xfer_bytes=%"PRIu64"\n", ctx->xfer_bytes);

  ctx->xfer_bytes = 0;
  ctx->xfer_ending = 0;
  ctx->obs_ending = 0;

  // write these into the pwcm's header
  if (ascii_header_set (pwcm->header, "OBS_OFFSET", "%"PRIu64, first_byte) < 0)
    multilog (pwcm->log, LOG_WARNING, "Could not write OBS_OFFSET to header\n");
  if (ascii_header_set (pwcm->header, "OBS_XFER", "%"PRIi64, ctx->xfer_count) < 0)
    multilog (pwcm->log, LOG_WARNING, "Could not write OBS_XFER to header\n");

  // If this header is the EOBS header
  if (ctx->xfer_count == -1)
  {
    ctx->obs_ending = 1;
    ctx->sod = 1;
    ctx->xfer_size = 1;
    if (ascii_header_set (pwcm->header, "TRANSFER_SIZE", "%"PRIi64, ctx->xfer_size) < 0)
      multilog (pwcm->log, LOG_WARNING, "Could not write TRANSFER_SIZE to header\n");
    multilog (pwcm->log, LOG_INFO, "new_xfer: OBS_XFER=%"PRIi64", OBS ENDING\n", ctx->xfer_count);
    if (ctx->verbose)
      multilog(pwcm->log, LOG_INFO, "new_xfer: observation is ending\n");
    return first_byte;
  }

  if (ascii_header_set (pwcm->header, "TRANSFER_SIZE", "%"PRIi64, ctx->xfer_size) < 0)
    multilog (pwcm->log, LOG_WARNING, "Could not write TRANSFER_SIZE to header\n");

  // pre post recv on BYTES_TO_XFER
  unsigned i = 0;
  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "new_xfer: post_recv on sync_from [BYTES TO XFER]\n");
  for (i=0; i<ctx->n_distrib; i++)
  {
    if (ctx->verbose > 1)
      multilog(pwcm->log, LOG_INFO, "new_xfer: [%d] post_recv on sync_from [BYTES TO XFER]\n", i);
    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(pwcm->log, LOG_ERR, "new_xfer: [%d] dada_ib_post_recv failed [BYTES TO XFER]\n", i);
      return 0;
    }
  }

  multilog (pwcm->log, LOG_INFO, "new_xfer: OBS_XFER=%"PRIi64", "
            "OBS_OFFSET=%"PRIu64"\n", ctx->xfer_count, first_byte);

  return first_byte;
}


/*! PWCM start function, called before start of observation */
time_t caspsr_pwc_ibdb_start (dada_pwc_main_t * pwcm, time_t start_utc)
{
  assert (pwcm != 0);
  caspsr_pwc_ibdb_t * ibdb = (caspsr_pwc_ibdb_t *) pwcm->context;

  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  unsigned i = 0;

  if (ibdb->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "caspsr_pwc_ibdb_start()\n");

  // accept the CM connection
  int rval = 0;
  pthread_t * connections = (pthread_t *) malloc(sizeof(pthread_t) * ibdb->n_distrib);

  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (!ibdb->ib_cms[i]->cm_connected) 
    {
      if (ibdb->verbose > 1)
        multilog (ibdb->log, LOG_INFO, "open: caspsr_pwc_ibdb_ib_init_thread ib_cms[%d]=%p\n",
                  i, ibdb->ib_cms[i]);

      rval = pthread_create(&(connections[i]), 0, (void *) caspsr_pwc_ibdb_ib_init_thread,
                          (void *) ibdb->ib_cms[i]);
      if (rval != 0)
      {
        multilog (ibdb->log, LOG_INFO, "open: error creating ib_init_thread\n");
        return -1;
      }
    }
    else 
    {
      multilog (ibdb->log, LOG_INFO, "open: ib_cms[%d] already connected\n", i);
    }
  }

  void * result;
  int init_cm_ok = 1;
  for (i=0; i<ibdb->n_distrib; i++) 
  {
    if (ibdb->ib_cms[i]->cm_connected <= 1)
    {
      pthread_join (connections[i], &result);
      if (!ibdb->ib_cms[i]->cm_connected)
        init_cm_ok = 0;
    }

    // set to 2 to indicate connection is established
    ibdb->ib_cms[i]->cm_connected = 2;
  }
  free(connections);
  if (!init_cm_ok)
  {
    multilog(pwcm->log, LOG_ERR, "open: failed to init CM connections\n");
    return -1;
  }

  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "open: connections initialized\n");

  // pre-post receive for the header transfer 
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "open: post_recv for headers\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog(pwcm->log, LOG_INFO, "open: [%d] post_recv on header_mb [HEADER]\n", i);

    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(pwcm->log, LOG_ERR, "open: [%d] post_recv on header_mb [HEADER] failed\n", i);
      return -1;
    }
  }

  // accept each connection
  int accept_result = 0;

  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "open: accepting connections\n");

  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (!ib_cms[i]->ib_connected)
    {
      if (ibdb->verbose)
        multilog(pwcm->log, LOG_INFO, "open: ib_cms[%d] accept\n", i);
      if (dada_ib_accept (ib_cms[i]) < 0)
      {
        multilog(pwcm->log, LOG_ERR, "open: dada_ib_accept failed\n");
        accept_result = -1;
      }
      ib_cms[i]->ib_connected = 1;
    }
  }

  ibdb->xfer_bytes = 0;
  ibdb->observation_bytes = 0;

  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "open: connections accepted\n");

  return accept_result;
}


/*! pwcm buffer function [for header only] */
void * caspsr_pwc_ibdb_recv (dada_pwc_main_t * pwcm, int64_t * size)
{

  assert (pwcm != 0);
  assert (pwcm->context != 0);
  caspsr_pwc_ibdb_t * ibdb = (caspsr_pwc_ibdb_t *) pwcm->context;

  assert (ibdb->ib_cms != 0);
  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  assert (pwcm->log != 0);
  multilog_t* log = pwcm->log;

  if (ibdb->verbose > 1)
    multilog (log, LOG_INFO, "caspsr_pwc_ibdb_recv()\n");

  unsigned i = 0;

  // send READY message to inform sender we are ready for the header
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv: send_messages [READY HDR]\n");
  if (dada_ib_send_messages(ib_cms, ibdb->n_distrib, DADA_IB_READY_KEY, 0) < 0)
  {
    multilog(pwcm->log, LOG_ERR, "recv: send_messages [READY HDR] failed\n");
    return 0;
  }

  // wait for transfer of the headers
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv: wait_recv [HEADER]\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog(pwcm->log, LOG_INFO, "recv: [%d] wait_recv [HEADER]\n", i);
    if (dada_ib_wait_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
    {
      multilog(pwcm->log, LOG_ERR, "recv: [%d] wait_recv [HEADER] failed\n", i);
      return 0;
    }
    if (ib_cms[i]->verbose > ibdb->verbose)
      ibdb->verbose = ib_cms[i]->verbose;
  }

  // use the size of the first cms header
  *size = (int64_t) ibdb->ib_cms[0]->header_mb->size;

  return ibdb->ib_cms[0]->header_mb->buffer;

}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t caspsr_pwc_ibdb_recv_block (dada_pwc_main_t* pwcm, void* data, 
                                    uint64_t data_size, uint64_t block_id)
{

  caspsr_pwc_ibdb_t * ibdb = (caspsr_pwc_ibdb_t *) pwcm->context;

  dada_ib_cm_t ** ib_cms = ibdb->ib_cms;

  if (ibdb->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "caspsr_pwc_ibdb_recv_block (%"PRIu64")\n", data_size);

  if (ibdb->xfer_ending)
    return 0;

  unsigned i = 0;

  // send READY message to inform sender we are ready for DATA 
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv: send_messages [READY DATA]\n");
  if (dada_ib_send_messages(ib_cms, ibdb->n_distrib, DADA_IB_READY_KEY, 0) < 0)
  {
    multilog(pwcm->log, LOG_ERR, "recv: send_messages [READY DATA] failed\n");
    return 0;
  }

  // wait for the number of bytes to be received
  uint64_t bytes_to_xfer = 0;
  if (ibdb->verbose)
        multilog(pwcm->log, LOG_INFO, "recv_block: recv_messages [BYTES TO XFER]\n");
  if (dada_ib_recv_messages(ib_cms, ibdb->n_distrib, DADA_IB_BYTES_TO_XFER_KEY) < 0)
  { 
    multilog(pwcm->log, LOG_ERR, "recv_block: recv_messages [BYTES TO XFER] failed\n");
    return -1;
  }

  // count the number of bytes to be received
  for (i=0; i<ibdb->n_distrib; i++)
  {
    bytes_to_xfer += ib_cms[i]->sync_from_val[1];
    if (ibdb->verbose > 1)
      multilog (pwcm->log, LOG_INFO, "recv_block: [%d] bytes to be recvd=%"PRIu64"\n",
                i, ib_cms[i]->sync_from_val[1]);
  }

  if ((ibdb->verbose) || (bytes_to_xfer != 256000000 && bytes_to_xfer != 0))
      multilog(pwcm->log, LOG_INFO, "recv_block: total bytes to be recvd=%"PRIu64"\n", bytes_to_xfer);

  // if the number of bytes to be received is less than the block size, this is the final transfer and
  // the end of the observation
  if (bytes_to_xfer < data_size)
  {
    if (ibdb->verbose)
      multilog(pwcm->log, LOG_INFO, "recv_block: bytes_to_xfer=%"PRIu64" < %"PRIu64", end of obs\n", bytes_to_xfer, data_size);
    ibdb->xfer_ending = 1;
  }

  // post recv for the number of bytes that were transferred
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv_block: post_recv [BYTES XFERRED]\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog(pwcm->log, LOG_INFO, "recv_block: [%d] post_recv on sync_from [BYTES XFERRED]\n", i);
    if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
    {
      multilog(pwcm->log, LOG_ERR, "recv_block: [%d] post_recv on sync_from [BYTES XFERRED] failed\n", i);
      return -1;
    }
  }

  // send the memory address of the block_id to be filled remotely via RDMA
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv_block: send_messages [BLOCK ID]\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    ib_cms[i]->sync_to_val[0] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_va;
    ib_cms[i]->sync_to_val[1] = (uint64_t) ib_cms[i]->local_blocks[block_id].buf_rkey;

    uintptr_t buf_va = (uintptr_t) ib_cms[i]->sync_to_val[0];
    uint32_t buf_rkey = (uint32_t) ib_cms[i]->sync_to_val[1];
    if (ibdb->verbose > 1)
      multilog(pwcm->log, LOG_INFO, "recv_block: [%d] local_block_id=%"PRIu64", local_buf_va=%p, "
                         "local_buf_rkey=%p\n", i, block_id, buf_va, buf_rkey);
  }
  if (dada_ib_send_messages(ib_cms, ibdb->n_distrib, UINT64_MAX, UINT64_MAX) < 0)
  {
    multilog(pwcm->log, LOG_ERR, "recv_block: send_messages [BLOCK ID] failed\n");
    return -1;
  }

  // remote RDMA transfer is ocurring now...
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv_block: waiting for completion "
             "on block %"PRIu64"...\n", block_id);

  // wait for the number of bytes transferred
  uint64_t bytes_xferred = 0;
  if (ibdb->verbose)
    multilog(pwcm->log, LOG_INFO, "recv_block: recv_messages [BYTES XFERRED]\n");
  if (dada_ib_recv_messages(ib_cms, ibdb->n_distrib, DADA_IB_BYTES_XFERRED_KEY) < 0)
  {
    multilog(pwcm->log, LOG_ERR, "recv_block: recv_messages [BYTES XFERRED] failed\n");
    return -1;
  }
  for (i=0; i<ibdb->n_distrib; i++)
  {
    bytes_xferred += ib_cms[i]->sync_from_val[1];
    if (ibdb->verbose > 1)
      multilog (pwcm->log, LOG_INFO, "recv_block: [%d] bytes recvd=%"PRIu64"\n",
                i, ib_cms[i]->sync_from_val[1]);
  }

  ibdb->xfer_bytes += bytes_xferred;
  ibdb->observation_bytes += bytes_xferred;

  // if we have reached the TRANSFER_SIZE, then this will be the end of the XFER
  // of if bytes xferred < full size, then this is final block of OBS
  if ((ibdb->xfer_bytes == ibdb->xfer_size) || (bytes_xferred < data_size) || (data_size == 0))
  {
    ibdb->xfer_ending = 1;

    if (ibdb->verbose > 1)
      multilog (pwcm->log, LOG_INFO, "recv_block: xfer_bytes=%"PRIu64" xfer_size=%"PRIu64" "
                "bytes_xferred=%"PRIu64", data_size=%"PRIu64"\n", 
                ibdb->xfer_bytes, ibdb->xfer_size, bytes_xferred, data_size);

    if (ibdb->verbose)
    {
      if (ibdb->xfer_bytes == ibdb->xfer_size)
        multilog(pwcm->log, LOG_INFO, "recv_block: last block of xfer, obs continuing\n");
      if (bytes_xferred < data_size)
        multilog(pwcm->log, LOG_INFO, "recv_block: partially full block of xfer, obs ending\n");
      if ((bytes_xferred == 0) && (data_size == 0))
        multilog(pwcm->log, LOG_INFO, "recv_block: empty block of xfer, obs ending\n");
    }

    // now get ready for the next header
    if (ibdb->verbose)
      multilog(pwcm->log, LOG_INFO, "recv_block: post_recv [HEADER]\n");
    for (i=0; i<ibdb->n_distrib; i++)
    {
      if (ibdb->verbose > 1)
        multilog(pwcm->log, LOG_INFO, "recv_block: [%d] post_recv [HEADER]\n", i);
      if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->header_mb) < 0)
      {
        multilog(pwcm->log, LOG_ERR, "recv_block: [%d] post_recv [HEADER] failed\n");
        return -1;
      }
    }
  }
  else
  {
    // post receive for the BYTES TO XFER in next block function call
    if (ibdb->verbose)
      multilog(pwcm->log, LOG_INFO, "recv_block: post_recv [BYTES TO XFER]\n");
    for (i=0; i<ibdb->n_distrib; i++)
    {
      if (ibdb->verbose > 1)
        multilog(pwcm->log, LOG_INFO, "recv_block: [%d] post_recv [BYTES TO XFER]\n", i);
      if (dada_ib_post_recv(ib_cms[i], ib_cms[i]->sync_from) < 0)
      {
        multilog(pwcm->log, LOG_ERR, "recv_block: [%d] post_recv [BYTES TO XFER] failed\n", i);
        return -1;
      }
    }
  }

  if (ibdb->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "recv_block: bytes transferred=%"PRIu64"\n", bytes_xferred);

  return (int64_t) bytes_xferred;

}


/*! PWCM stop function, called at end of observation */
int caspsr_pwc_ibdb_stop (dada_pwc_main_t* pwcm)
{

  caspsr_pwc_ibdb_t * ibdb = (caspsr_pwc_ibdb_t *) pwcm->context;

  if (ibdb->verbose > 1)
  {
    multilog (pwcm->log, LOG_INFO, "caspsr_pwc_ibdb_stop()\n");
    multilog (pwcm->log, LOG_INFO, "stop: last_xfer_bytes=%"PRIu64"\n",
                                    ibdb->xfer_bytes);
  }

  if (ibdb->verbose)
    multilog (pwcm->log, LOG_INFO, "received %"PRIu64" bytes\n", 
                                   ibdb->observation_bytes);
  unsigned i=0;
  if (ibdb->verbose)
    multilog (pwcm->log, LOG_INFO, "stop: ib_disconnet()\n");
  for (i=0; i<ibdb->n_distrib; i++)
  {
    if (ibdb->verbose > 1)
      multilog (pwcm->log, LOG_INFO, "stop: ib_disconnect[%d]\n", i);
    if (dada_ib_disconnect(ibdb->ib_cms[i]) < 0)
    {
      multilog(ibdb->log, LOG_ERR, "dada_ib_disconnect failed\n");
    }
  }

  return 0;
}


/*
 * required initialization of IB device and associate verb structs
 */
int caspsr_pwc_ibdb_ib_init(caspsr_pwc_ibdb_t * ctx, dada_hdu_t * hdu, multilog_t * log)
{

  uint64_t db_nbufs = 0;
  uint64_t db_bufsz = 0;
  uint64_t hb_nbufs = 0;
  uint64_t hb_bufsz = 0;
  char ** db_buffers = 0;
  char ** hb_buffers = 0;

  assert (ctx != 0);
  assert (hdu != 0);

  if (ctx->verbose > 1)
    multilog(log, LOG_INFO, "caspsr_pwc_ibdb_ib_init()\n");

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
    ctx->ib_cms[i]->send_depth = 1;
    ctx->ib_cms[i]->recv_depth = 1;
    ctx->ib_cms[i]->port = ctx->port + i;
    ctx->ib_cms[i]->bufs_size = db_bufsz;
    ctx->ib_cms[i]->header_size = hb_bufsz;
    ctx->ib_cms[i]->db_buffers = db_buffers;
  }

  return 0;
}

void * caspsr_pwc_ibdb_ib_init_thread (void * arg)
{

  dada_ib_cm_t * ib_cm = (dada_ib_cm_t *) arg;

  multilog_t * log = ib_cm->log;

  if (ib_cm->verbose > 1)
    multilog (log, LOG_INFO, "ib_init_thread: ib_cm=%p\n", ib_cm);
    
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
  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "ib_init_thread: depth=%"PRIu64"\n", ib_cm->send_depth + ib_cm->recv_depth);

  if (dada_ib_create_verbs(ib_cm) < 0)
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
  if (!ib_cm->header)
  {
    multilog(log, LOG_ERR, "ib_init_thread: could not allocate memory for header\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "ib_init_thread: reg header_mb\n");
  ib_cm->header_mb = dada_ib_reg_buffer(ib_cm, ib_cm->header, ib_cm->header_size, flags);
  if (!ib_cm->header_mb)
  {
    multilog(log, LOG_INFO, "ib_init_thread: reg header_mb failed\n");
    pthread_exit((void *) &(ib_cm->cm_connected));
  }

  ib_cm->header_mb->wr_id = 10000;

  if (ib_cm->verbose > 1)
    multilog(log, LOG_INFO, "ib_init_thread: dada_ib_create_qp\n");
  if (dada_ib_create_qp (ib_cm) < 0)
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
  caspsr_pwc_ibdb_t ibdb = CASPSR_PWC_IBDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA PWCM */
  dada_pwc_main_t* pwcm = 0;

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

  /* PWC control port */
  int control_port = CASPSR_DEFAULT_PWC_PORT;

  /* Multilog LOG port */
  int log_port = CASPSR_DEFAULT_PWC_LOGPORT;

  /* Quit flag */
  char quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* header file for manual mode */
  char * header_file = 0;

  /* number of seconds to acquire data for */
  unsigned nsecs = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:C:dH:k:l:n:p:sv")) != -1)
  {
    switch (arg) {

    case 'c':
      if (optarg)
      {
        control_port = atoi(optarg);
        break;
      }
      else
      {
        fprintf(stderr, "ERROR: no control portspecified\n");
        usage();
        return EXIT_FAILURE;
      }

    case 'C':
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

    case 'H':
      if (optarg) {
        header_file = strdup(optarg);
      } else {
        fprintf(stderr,"ERROR: -H flag requires a file arguement\n");
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"caspsr_pwc_ibdb: could not parse key from %s\n",optarg);
        return EXIT_FAILURE;
      }
      break;

    case 'l':
      if (optarg) {
        log_port = atoi(optarg);
        break;
      } else {
        fprintf (stderr,"caspsr_pwc_ibdb: no log_port specified\n");
        usage();
        return EXIT_FAILURE;
      }


    case 'n':
      nsecs = atoi (optarg);
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

  if (header_file)
  {
    fprintf(stderr, "Header file = %s\n", header_file);
  }

  // do not use the syslog facility
  log = multilog_open ("caspsr_pwc_ibdb", 0);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, log_port);
  }
  else
  {
    multilog_add (log, stderr);
    multilog_serve (log, log_port);
  }

  pwcm = dada_pwc_main_create();

  pwcm->log                   = log;
  pwcm->start_function        = caspsr_pwc_ibdb_start;
  pwcm->buffer_function       = caspsr_pwc_ibdb_recv;
  pwcm->block_function        = caspsr_pwc_ibdb_recv_block;
  pwcm->stop_function         = caspsr_pwc_ibdb_stop;
  pwcm->xfer_pending_function = caspsr_pwc_ibdb_xfer_pending;
  pwcm->new_xfer_function     = caspsr_pwc_ibdb_new_xfer;
  pwcm->header_valid_function = caspsr_pwc_ibdb_header_valid;
  pwcm->context               = &ibdb;
  pwcm->verbose               = verbose;

  hdu = dada_hdu_create (log);

  dada_hdu_set_key (hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  pwcm->data_block            = hdu->data_block;
  pwcm->header_block          = hdu->header_block;

  ibdb.chunk_size = chunk_size;
  ibdb.verbose = verbose;
  ibdb.port = port;
  ibdb.n_distrib = n_distrib;
  ibdb.log = log;

  // initialize IB resources
  if (caspsr_pwc_ibdb_ib_init (&ibdb, hdu, log) < 0)
  {
    multilog (log, LOG_ERR, "Failed to initialise IB resources\n");
    quit = 1;
  }

  if (header_file && !quit) 
  {

    multilog (pwcm->log, LOG_INFO, "Manual Mode\n");

    char *   header_buf = 0;
    uint64_t header_size = 0;
    char *   buffer = 0;
    unsigned buffer_size = 64;

    // get the next header
    header_size = ipcbuf_get_bufsz (hdu->header_block);
    multilog (pwcm->log, LOG_INFO, "header block size = %llu\n", header_size);
    pwcm->header = ipcbuf_get_next_write (hdu->header_block);

    if (!pwcm->header) 
    {
      multilog (pwcm->log, LOG_ERR, "could not get next header block\n");
      return EXIT_FAILURE;
    }

    // allocate some memory for the header
    multilog (pwcm->log, LOG_INFO, "reading custom header %s\n", header_file);
    header_buf = (char *) malloc(sizeof(char) * header_size);
    if (!header_buf) 
    {
      multilog (pwcm->log, LOG_ERR, "could not allocate memory for header_buf\n");
      return EXIT_FAILURE;
    }

    buffer = (char *) malloc (sizeof(char) * buffer_size);
    if (!buffer) 
    {
      multilog (pwcm->log, LOG_ERR, "could not allocate memory for buffer\n");
      return EXIT_FAILURE;
    }

    // read the header from file
    if (fileread (header_file, header_buf, header_size) < 0) 
    {
      multilog (pwcm->log, LOG_ERR, "could not read header from %s\n",
                header_file);
      return EXIT_FAILURE;
    }

    time_t utc = time(0);
    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime(&utc));
    if (ascii_header_set (header_buf, "UTC_START", "%s", buffer) < 0) 
    {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return EXIT_FAILURE;
    }

    // HACK just set the UTC_START to roughly now
    multilog (pwcm->log, LOG_INFO, "UTC_START [now] = %s\n",buffer);

    // copy the header_buf to the header block
    memcpy(pwcm->header, header_buf, header_size);

    if (verbose)
      multilog (pwcm->log, LOG_INFO, "running start function\n");

    // note that the start function will set the OBS_OFFSET & RECV_HOST
    utc = pwcm->start_function(pwcm, 0);
    if (utc == -1 ) {
      multilog (pwcm->log, LOG_ERR, "could not run start function\n");
      return EXIT_FAILURE;
    }

    // marked the header as filled
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)
    {
      multilog (pwcm->log, LOG_ERR, "could not mark filled header block\n");
      return EXIT_FAILURE;
    }
    multilog (pwcm->log, LOG_INFO, "header marked filled\n");

    int64_t bytes_to_acquire = 1;
    uint64_t first_byte = 0;

    if (nsecs)
      bytes_to_acquire = 100000000 * (int64_t) nsecs;

    fprintf(stderr, "bytes_to_acquire = %"PRIu64", nsecs=%d\n", bytes_to_acquire, nsecs);

    // get the data buffer block size for block operations
    uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t*) pwcm->data_block);
    uint64_t bytes_written = 0;

    uint64_t bsize = 0;
    unsigned continue_xfer = 0;
    uint64_t xfer_bytes_written = 0;
    uint64_t block_id = 0;

    while (bytes_to_acquire > 0) 
    {

      continue_xfer = 1;
      xfer_bytes_written = 0;

      // continue to call the block function for the current xfer
      while (continue_xfer) 
      {

        // get the pointer to the next empty block and its ID
        buffer = ipcio_open_block_write (pwcm->data_block, &block_id);
        if (!buffer) 
        {
          multilog (log, LOG_ERR, "ipcio_open_block_write error %s\n", strerror(errno));
          continue_xfer = 0;
          bytes_to_acquire = -1;
        }

        // call the block function to fill the buffer with data
        bytes_written = pwcm->block_function (pwcm, buffer, block_size, block_id);

        // check for errors
        if (bytes_written < 0) 
        {
          multilog (log, LOG_ERR, "block_function failed\n");
          ipcio_close_block_write (pwcm->data_block, 0);
          continue_xfer = 0;
          bytes_to_acquire = -1;
          break;
        }

        // close the block if 0 or more bytes were written
        if (ipcio_close_block_write (pwcm->data_block, (uint64_t) bytes_written) < 0) {
          multilog (log, LOG_ERR, "ipcio_close_block_write error %s\n", strerror(errno));
          continue_xfer = 0;
          bytes_to_acquire = -1;
        }

        if (bytes_written < block_size)
        {
          multilog (log, LOG_INFO, "main: %"PRIu64" of %"PRIu64" bytes "
                    "written, end of xfer/input\n", bytes_written, block_size);
        }

        // if we wrote 0 bytes, its end of xfer or end of obs
        if (bytes_written == 0)
        {
          multilog(pwcm->log, LOG_INFO, "main: block_function returned 0 bytes\n");
          continue_xfer = 0;
        }
        else
        {
          xfer_bytes_written += bytes_written;
          if (nsecs)
            bytes_to_acquire -= bytes_written;
        }
      }

      if (bytes_to_acquire <= 0) {
        multilog(pwcm->log, LOG_INFO, "main: acquired enough bytes [bytes_to_acquire=%"PRId64"\n", bytes_to_acquire);
      }

      /* check whether a new xfer is pending 
       *  0 OBS still waiting for data on first XFER
       *  1 OBS continuing, end of XFER
       *  2 OBS end
       */
      int xfer_pending_status = pwcm->xfer_pending_function (pwcm);
      multilog(pwcm->log, LOG_INFO, "main: xfer_pending_status = %d, xfer_bytes_written=%"PRIu64"\n", xfer_pending_status, xfer_bytes_written);

      if (xfer_pending_status > 0) {

        // check whether any bytes have been written in this xfer
        uint64_t bytes_written_this_xfer = pwcm->data_block->bytes;
        multilog(pwcm->log, LOG_INFO, "main : current DB had %"PRIu64" bytes written\n",
                                       bytes_written_this_xfer);

        // if we have written 0 bytes, we must write at least 1 byte so that 
        // the SOD and EOD are seperated
        if (bytes_written_this_xfer == 0) {
          if (ipcio_write(pwcm->data_block, buffer, 1) != 1)
          {
            multilog(pwcm->log, LOG_ERR, "main: failed to write 1 byte to empty datablock at end of OBS\n");
            return EXIT_FAILURE;
          }
        }

        /* close the current data block */
        multilog(pwcm->log, LOG_INFO, "main: closing data block\n");
        if (dada_hdu_unlock_write (hdu) < 0)
          return EXIT_FAILURE;

        /* re-connect to the data block */
        multilog(pwcm->log, LOG_INFO, "main: re-opening data block\n");
        if (dada_hdu_lock_write (hdu) < 0)
           return EXIT_FAILURE;

        /* get the next header block */
        pwcm->header = ipcbuf_get_next_write (hdu->header_block);

        /* copy the header_buf to the header block */
        memcpy(pwcm->header, header_buf, header_size);

        /* get the OBS_OFFSET of the next xfer */
        if (xfer_pending_status == 1) {

          multilog(pwcm->log, LOG_INFO, "main: new_xfer_function()\n");
          first_byte = pwcm->new_xfer_function(pwcm);
          multilog(log, LOG_INFO, "main: XFER %"PRIi64" on byte=%"PRIu64"\n", ibdb.xfer_count, first_byte);

        }
        /* signify that this header is the end of the observation */
        if (xfer_pending_status == 2) {

          multilog(log, LOG_INFO, "main: OBS ended on XFER %"PRIi64", writing new header with OBS_XFER=-1\n", ibdb.xfer_count);
          int64_t end_of_xfer = -1;

          if (ascii_header_set (pwcm->header, "OBS_XFER", "%"PRIi64, end_of_xfer) < 0) {
              multilog (pwcm->log, LOG_ERR, "Could not write OBS_XFER to header\n");
            return EXIT_FAILURE;
          }

          uint64_t obs_offset = 0;
          if (ascii_header_set (pwcm->header, "OBS_OFFSET", "%"PRIu64, obs_offset) < 0) {
            multilog (pwcm->log, LOG_ERR, "Could not write OBS_OFFSET to header\n");
            return EXIT_FAILURE;
          }

          multilog(log, LOG_ERR, "main: unrecognized error status %d\n", xfer_pending_status);
        }

        /* marked the header as filled */
        if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
          multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
          return EXIT_FAILURE;
        }

        /* end of OBS */
        if (xfer_pending_status == 2) {

          /* write 1 byte so that EOD/SOD are separated */
          ipcio_write(hdu->data_block, buffer, 1);
          multilog(pwcm->log, LOG_WARNING, "main: OBS ended, wrote 1 byte to data block\n");
          bytes_to_acquire = 0;

        }

      /* we are still waiting for the first packet */
      } else {

        multilog(pwcm->log, LOG_INFO, "main: waiting for first packet\n");
      }
    }

    fprintf(stderr, "stop_function\n");
    if ( pwcm->stop_function(pwcm) != 0) 
      fprintf(stderr, "Error stopping acquisition");

  }
  else
  {
    pwcm->header = hdu->header;

    if (verbose)
      fprintf (stdout, "caspsr_pwc_ibdb: creating dada pwc control interface\n");

    pwcm->pwc = dada_pwc_create();

    pwcm->pwc->port = control_port;

    if (verbose)
      fprintf (stdout, "caspsr_pwc_ibdb: creating dada server\n");
    if (dada_pwc_serve (pwcm->pwc) < 0) {
      fprintf (stderr, "caspsr_pwc_ibdb: could not start server\n");
      return EXIT_FAILURE;
    }

    if (verbose)
      fprintf (stdout, "caspsr_pwc_ibdb: entering PWC main loop\n");

    if (dada_pwc_main (pwcm) < 0) {
      fprintf (stderr, "caspsr_pwc_ibdb: error in PWC main loop\n");
      return EXIT_FAILURE;
    }
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    fprintf (stdout, "caspsr_pwc_ibdb: destroying pwc\n");
  dada_pwc_destroy (pwcm->pwc);

  if (verbose)
    fprintf (stdout, "caspsr_pwc_ibdb: destroying pwc main\n");
  dada_pwc_main_destroy (pwcm);
 
  unsigned i=0;
  for (i=0; i<ibdb.n_distrib; i++)
  {
    if (dada_ib_destroy(ibdb.ib_cms[i]) < 0)
    {
      multilog(log, LOG_ERR, "dada_ib_destroy failed\n");
    }
  }

  //free (ibdb.connected);

  return EXIT_SUCCESS;
}
