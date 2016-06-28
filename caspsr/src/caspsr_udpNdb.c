/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 * caspsr_udpNdb.
 *
 * Reads UDP packets from an ibob and writes them to multiple
 * datablocks ready for time multiplexing to processing nodes
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>

#include "caspsr_udpNdb.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "sock.h"

/* debug mode */
#define _DEBUG 1

/* packets per xfer. means 8 seconds with 2 distribs */
#define CASPSR_UDPNDB_PACKS_PER_XFER 781250


/* global variables */
int quit_threads = 0;
int start_pending = 0;
int stop_pending = 0;
int recording = 0;
uint64_t stop_byte = 0;

void usage()
{
  fprintf (stdout,
	   "caspsr_udpNdb [options] i_dist n_dist key1 [keyi]\n"
	   " -i iface       interface for UDP packets [default all interfaces]\n"
	   " -o port        port for 'telnet' control commands\n"
	   " -p port        port for incoming UDP packets [default %d]\n"
	   " -q port        port for outgoing packets [default %d]\n"
     " -t secs        acquire for secs only [default continuous]\n"
     " -n packets     write num packets to each datablock [default %d]\n"
     " -b bytes       number of packets to append to each transfer\n"
     " -d             run as a daemon\n"
     " -h             print help text\n"
     " -v             verbose messages\n"
     "\n"
     "i_dist          index of distributor\n"
     "n_dist          number of distributors\n"
     "key1 - keyi     shared memory keys to write data to\n"
     "\n",
     CASPSR_DEFAULT_UDPNDB_PORT, CASPSR_DEFAULT_UDPNDB_PORT, 
     CASPSR_UDPNDB_PACKS_PER_XFER);
}


/* 
 *  intialize UDP receiver resources
 */
int caspsr_udpNdb_init_receiver (udpNdb_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_init_receiver()\n");

  // create a CASPSR socket which can hold 1 UDP packet
  ctx->sock = caspsr_init_sock();

  // create a "zereod packet" for quick padding
  ctx->zeroed_packet = (char *) malloc (UDP_DATA);
  memset(ctx->zeroed_packet, 0, UDP_DATA);

  ctx->ooo_packets = 0;
  ctx->recv_core = 0;
  ctx->n_sleeps = 0;
  ctx->mb_rcv_ps = 0;
  ctx->mb_drp_ps = 0;
  ctx->hdu_open = 0;
  ctx->block_open = 0;
  ctx->obs_header = 0;
  ctx->header = 0;
  ctx->header_size = 0;

  // allocate required memory strucutres
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();
  return 0;

}

/* 
 *  destory UDP receiver resources 
 */
int caspsr_udpNdb_destroy_receiver (udpNdb_t * ctx)
{
  caspsr_free_sock(ctx->sock);

  if (ctx->zeroed_packet)
    free (ctx->zeroed_packet);
  ctx->zeroed_packet = 0;

}

/*
 *  reset receiver before an observation commences
 */
void caspsr_udpNnb_reset_receiver (udpNdb_t * ctx) 
{

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_reset_receiver()\n");

  ctx->packet_reset = 0;
  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);

}

/* 
 *  open socket and prepare for start of data capture
 */
time_t caspsr_udpNdb_start_function (udpNdb_t * ctx)
{

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "caspsr_udpNdb_start_function()\n");

  // open socket
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: creating udp socket on %s:%d\n", ctx->interface, ctx->port);
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 256 MB
  int sock_buf_size = 256*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  // setup the next_seq to the initial value
  ctx->next_seq = 0;
  ctx->n_sleeps = 0;
  ctx->ooo_packets = 0;
  ctx->obs_offset = 0;
  ctx->obs_xfer = 0;

  // create the obs header for this observation
  ctx->header_size = DADA_DEFAULT_HEADER_SIZE;
  ctx->obs_header = (char *) malloc (DADA_DEFAULT_HEADER_SIZE);
  if (!ctx->obs_header)
  {
    multilog (ctx->log, LOG_ERR, "start_function: malloc failed for obs_header\n");
    return -1;
  }

  char localhost[HOST_NAME_MAX] = "unknown";
  gethostname(localhost,HOST_NAME_MAX);

  int tmp = 0;
  if (ascii_header_set (ctx->obs_header, "HDR_SIZE", "%d", ctx->header_size) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write HDR_SIZE to header\n");
  if (ascii_header_set (ctx->obs_header, "OBS_OFFSET", "%"PRIu64, ctx->obs_offset) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write OBS_OFFSET to header\n");
  if (ascii_header_set (ctx->obs_header, "OBS_XFER", "%"PRIi64, ctx->obs_xfer) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write OBS_XFER to header\n");
  if (ascii_header_set (ctx->obs_header, "RECV_HOST", "%s", localhost) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write RECV_HOST to header\n");
  if (ascii_header_set (ctx->obs_header, "I_DISTRIB", "%d", ctx->i_distrib) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write I_DISTRIB to header\n");
  if (ascii_header_set (ctx->obs_header, "N_DISTRIB", "%d", ctx->n_distrib) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write N_DISTRIB to header\n");

  return 0;
}

/*
 *  close a HDU "end a transfer"
 */
int caspsr_udpNdb_close_hdu (udpNdb_t * ctx, uint64_t bytes_written)
{

  if ((ctx->verbose > 1) || (ctx->verbose && stop_pending))
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_close_hdu(%"PRIu64")\n", bytes_written);

  if (!ctx->hdu_open)
  {
    multilog (ctx->log, LOG_ERR, "close_hdu: hdu was already closed\n");
    return -1;
  }

  if (ctx->block_open) 
  {
    if ((ctx->verbose && stop_pending) || (ctx->verbose > 1))
      multilog (ctx->log, LOG_INFO, "close_hdu: caspsr_udpNdb_close_buffer(%"PRIu64", 1)\n", bytes_written);
    if (caspsr_udpNdb_close_buffer (ctx, bytes_written, 1) < 0)
    {
      multilog (ctx->log, LOG_INFO, "close_hdu: caspsr_udpNdb_close_buffer failed\n");
      return -1;
    }
  }

  // close the current hdu, ending the transfer
  if ((ctx->verbose && stop_pending) || (ctx->verbose > 1))
    multilog (ctx->log, LOG_INFO, "close_hdu: dada_hdu_unlock_write on hdu %d\n", ctx->ihdu);
  if (dada_hdu_unlock_write (ctx->hdus[ctx->ihdu]) < 0)
  {
    multilog (ctx->log, LOG_ERR, "close_hdu: could not unlock write on hdu %d\n", ctx->ihdu);
    return -1;
  }
  ctx->hdu_open = 0;

  return 0;

}

/* 
 * open a HDU "start a transfer"
 */
int caspsr_udpNdb_open_hdu (udpNdb_t * ctx)
{

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_open_hdu()\n");

  ipcbuf_t * header_block = 0;
  ipcbuf_t * data_block = 0; 
  uint64_t header_size;
  char * header = 0;
  uint64_t transfer_size = ctx->packets_per_xfer * UDP_DATA;
    
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "open_hdu: dada_hdu_lock_write ihdu=%d\n", ctx->ihdu);

  if (dada_hdu_lock_write (ctx->hdus[ctx->ihdu]) < 0)
  {
    multilog (ctx->log, LOG_ERR, "open_hdu: could not lock write on hdu %d\n", ctx->ihdu);
    return -1;
  }
  ctx->hdu_open = 1;

  header_block = ctx->hdus[ctx->ihdu]->header_block;

  // write a new header block
  header_size = ipcbuf_get_bufsz (header_block);
  header = ipcbuf_get_next_write (header_block);

  // copy the current observations header
  memcpy (header, ctx->obs_header, header_size);

  // set the obs offset for this transfer, and transfer number
  if (ascii_header_set (header, "OBS_XFER", "%"PRIi64, ctx->obs_xfer) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write OBS_XFER to header\n");

  if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, ctx->obs_offset) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write OBS_OFFSET to header\n");

  if (ascii_header_set (header, "HDR_SIZE", "%"PRIu64, header_size) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write HDR_SIZE to header\n");

  if (ascii_header_set (header, "TRANSFER_SIZE", "%"PRIu64, transfer_size) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write TRANSFER_SIZE to header\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "open_hdu: OBS_XFER=%"PRIi64", OBS_OFFSET=%"PRIu64"\n", 
             ctx->obs_xfer, ctx->obs_offset);

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "header=%s\n", header);

  // mark the header as filled
  if (ipcbuf_mark_filled (header_block, header_size) < 0)  
  {
    multilog (ctx->log, LOG_ERR, "open_hdu: could not mark filled Header Block\n");
    return -1;
  }

  if (caspsr_udpNdb_open_buffer (ctx) < 0)
  {
    multilog (ctx->log, LOG_ERR, "open_hdu: caspsr_udpNdb_open_buffer failed\n");
    return -1;
  }

  return 0;
}


/* 
 *   incremements the hdu index when an xfer is complete 
 */
int caspsr_udpNdb_new_hdu (udpNdb_t * ctx)
{

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "caspsr_udpNdb_new_hdu()\n");

  if (caspsr_udpNdb_close_hdu (ctx, ctx->hdu_bufsz) < 0)
  {
    multilog(ctx->log, LOG_ERR, "new_hdu: caspsr_udpNdb_close_hdu failed\n");
    return -1;
  }

  unsigned old_hdu = ctx->ihdu;

  // incremement the hdu counter
  ctx->ihdu = (ctx->ihdu + 1) % ctx->nhdus;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "new_hdu: hdu %d -> %d\n", old_hdu, ctx->ihdu);

  // update the xfer number if we have reset the ihdu counter to 0
  if (ctx->ihdu < old_hdu)
    ctx->obs_xfer++;

  // increment the OBS offset by the required amount
  ctx->obs_offset += ctx->packets_per_xfer * UDP_DATA * ctx->n_distrib;

  if (caspsr_udpNdb_open_hdu (ctx) < 0)
  {
    multilog(ctx->log, LOG_ERR, "new_hdu: caspsr_udpNdb_open_hdu failed\n");
    return -1;
  }

  // increment buffer byte markers
  ctx->buffer_start_byte = ctx->buffer_end_byte + UDP_DATA;
  ctx->buffer_end_byte = ctx->buffer_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "new_hdu: buffer bytes [%"PRIu64" - %"PRIu64"]\n", 
             ctx->buffer_start_byte, ctx->buffer_end_byte);

  // increment xfer byte markers
  ctx->xfer_start_byte = ctx->xfer_end_byte + UDP_DATA;
  ctx->xfer_end_byte = ctx->xfer_start_byte + ( ctx->packets_per_xfer - 1) * UDP_DATA;
  if (ctx->verbose)
  {
    multilog(ctx->log, LOG_INFO, "new_hdu: xfer bytes [%"PRIu64" - %"PRIu64"]\n", 
             ctx->xfer_start_byte, ctx->xfer_end_byte);
  }

  return 0;
}

/* 
 *  open a data block buffer ready for direct access
 */
int caspsr_udpNdb_open_buffer (udpNdb_t * ctx)
{

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_open_buffer()\n");

  if (!ctx->hdu_open)
  {
    multilog (ctx->log, LOG_ERR, "open_buffer: hdu not open\n");
    return -1;
  }

  if (ctx->block_open)
  {
    multilog (ctx->log, LOG_ERR, "open_buffer: buffer already opened\n");
    return -1;
  }

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "open_buffer: ipcio_open_block_write hdu=%d\n", ctx->ihdu);

  uint64_t block_id = 0;

  ctx->current_buffer = ipcio_open_block_write (ctx->hdus[ctx->ihdu]->data_block, &block_id);
  if (!ctx->current_buffer)
  { 
    multilog (ctx->log, LOG_ERR, "open_buffer: ipcio_open_block_write failed\n");
    return -1;
  }

  ctx->block_open = 1;

  return 0;
}

/*
 *  close a data buffer, assuming a full block has been written
 */
int caspsr_udpNdb_close_buffer (udpNdb_t * ctx, uint64_t bytes_written, unsigned eod)
{

  if (ctx->verbose && stop_pending || ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_close_buffer(%"PRIu64", %d)\n", bytes_written, eod);

  if (!ctx->hdu_open)
  {
    multilog (ctx->log, LOG_ERR, "close_buffer: hdu not open\n");
    return -1;
  }

  if (!ctx->block_open)
  { 
    multilog (ctx->log, LOG_ERR, "close_buffer: buffer already closed\n");
    return -1;
  }

  // log any buffers that are not full, except for the 1 byte "EOD" buffer
  if ((bytes_written != 1) && (bytes_written != ctx->hdu_bufsz))
    multilog (ctx->log, (eod ? LOG_INFO : LOG_WARNING), "close_buffer: "
              "bytes_written[%"PRIu64"] != hdu_bufsz[%"PRIu64"]\n", 
              bytes_written, ctx->hdu_bufsz);

  if (eod)
  {
    if (ipcio_update_block_write (ctx->hdus[ctx->ihdu]->data_block, bytes_written) < 0)
    {
      multilog (ctx->log, LOG_ERR, "close_buffer: ipcio_update_block_write failed\n");
      return -1;
    }
  }
  else 
  {
    if (ipcio_close_block_write (ctx->hdus[ctx->ihdu]->data_block, bytes_written) < 0)
    {
      multilog (ctx->log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      return -1;
    }
  }

  ctx->current_buffer = 0;
  ctx->block_open = 0;

  return 0;
}


/* 
 *  move to the next ring buffer element. return pointer to base address of new buffer
 */
int caspsr_udpNdb_new_buffer (udpNdb_t * ctx)
{

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpNdb_new_buffer()\n");

  if (caspsr_udpNdb_close_buffer (ctx, ctx->hdu_bufsz, 0) < 0)
  {
    multilog (ctx->log, LOG_INFO, "new_buffer: caspsr_udpNdb_close_buffer failed\n");
    return -1;
  }

  if (caspsr_udpNdb_open_buffer (ctx) < 0) 
  {
    multilog (ctx->log, LOG_INFO, "new_buffer: caspsr_udpNdb_open_buffer failed\n");
    return -1;
  }

  // increment buffer byte markers
  ctx->buffer_start_byte = ctx->buffer_end_byte + UDP_DATA;
  ctx->buffer_end_byte = ctx->buffer_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "new_buffer: buffer_bytes [%"PRIu64" - %"PRIu64"]\n", 
             ctx->buffer_start_byte, ctx->buffer_end_byte);

  return 0;

}

/*
 * Receive UDP data for 1 observation, continually writing it to 
 * datablocks
 */
void * caspsr_udpNdb_receive_obs (void * arg)
{

  udpNdb_t * ctx = (udpNdb_t *) arg;

  /* multilogging facility */
  multilog_t * log = ctx->log;

  /* pointer for header decode function */
  unsigned char * arr;

  /* raw sequence number */
  uint64_t raw_seq_no = 0;

  /* "fixed" raw sequence number */
  uint64_t fixed_raw_seq_no = 0;

  /* offset raw sequence number */
  uint64_t offset_raw_seq_no = 0;

  /* decoded sequence number */
  uint64_t seq_no = 0;

  /* previously received sequence number */
  uint64_t prev_seq_no = 0;

  /* decoded channel id */
  uint64_t ch_id = 0;

  /* all packets seem to have an offset of -3 [sometimes -2] OMG */
  int64_t global_offset = -1;

  /* amount the seq number should increment by */
  int64_t seq_offset = 0;

  /* amount the seq number should increment by */
  uint64_t seq_inc = 1024 * ctx->n_distrib;

  /* used to decode/encode sequence number */
  uint64_t tmp = 0;
  unsigned char ch;

  /* flag for timeout */
  int timeout_ocurred = 0;

  /* flag for ignoring packets */
  unsigned ignore_packet = 0;

  /* small counter for problematic packets */
  int problem_packets = 0;

  /* flag for having a packet */
  unsigned have_packet = 0;

  /* data received from a recv_from call */
  size_t got = 0;

  /* offset for packet loss calculation */
  uint64_t offset = 0;

  /* determine the sequence number boundaries for curr and next buffers */
  int errsv;

  /* flag for start of data */
  unsigned waiting_for_start = 1;

  /* pointer to the current receiver */
  udpNdb_receiver_t * r = 0;

  /* total bytes received + dropped */
  uint64_t bytes_total = 0;

  /* remainder of the fixed raw seq number modulo seq_inc */
  uint64_t remainder = 0;

  // start byte of the current UDP packet
  uint64_t packet_byte = 0;

  // used to pad packets
  uint64_t pad_start = 0;
  uint64_t iseq = 0;

  int64_t byte_offset = 0;
  uint64_t seq_byte = 0;

  unsigned i = 0;
  unsigned j = 0;
  int thread_result = 0;

  struct timeval timeout;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "caspsr_udpNdb_receive_obs()\n");

  // set the CPU that this thread shall run on
  if (dada_bind_thread_to_core(ctx->recv_core) < 0)
    multilog(ctx->log, LOG_WARNING, "receive_obs: failed to bind to core %d\n", ctx->recv_core);

  // calculate the offset expected in the sequence number
  seq_offset = 1024 * (ctx->i_distrib);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "receive_obs: seq_inc=%"PRIu64", "
              "seq_offset=%"PRId64", gobal_offset=%"PRIi64"\n", 
              seq_inc, seq_offset, global_offset);

  // set recording state once we enter this main loop
  recording = 1;

  // open first hdu and first buffer
  ctx->ihdu = 0;
  ctx->buffer_start_byte = 0;
  ctx->buffer_end_byte = ctx->buffer_start_byte + ( ctx->packets_per_buffer - 1) * UDP_DATA;
  ctx->xfer_start_byte = 0;
  ctx->xfer_end_byte = ctx->xfer_start_byte + ( ctx->packets_per_xfer - 1) * UDP_DATA;

  if (ctx->verbose)
        multilog(log, LOG_INFO, "receive_obs: caspsr_udpNdb_open_hdu()\n");
  if (caspsr_udpNdb_open_hdu (ctx) < 0) 
  {
    multilog (log, LOG_ERR, "receive_obs: caspsr_udpNdb_open_hdu failed\n");
    thread_result = -1;
    pthread_exit((void *) &thread_result);
  }

  /* Continue to receive packets */
  while (!quit_threads && !stop_pending) 
  {

    have_packet = 0; 

    // incredibly tight loop to try and get a packet
    while (!have_packet && !quit_threads && !stop_pending)
    {

      // receive 1 packet into the socket buffer
      got = recvfrom (ctx->sock->fd, ctx->sock->buffer, UDP_PAYLOAD, 0, NULL, NULL);

      if (got == UDP_PAYLOAD) 
      {
        have_packet = 1;
        ignore_packet = 0;
      } 
      else if (got == -1) 
      {
        errsv = errno;
        if (errsv == EAGAIN) 
        {
          ctx->n_sleeps++;

        } 
        else 
        {
          multilog (log, LOG_ERR, "receive_obs: recvfrom failed %s\n", strerror(errsv));
          thread_result = -1;
          pthread_exit((void *) &thread_result);
        }
      } 
      else // we received a packet of the WRONG size, ignore it
      {
        multilog (log, LOG_ERR, "receive_obs: received %d bytes, expected %d\n", got, UDP_PAYLOAD);
        ignore_packet = 1;
      }
    }

    // we have a valid packet within the timeout
    if (have_packet) 
    {

      // decode the packet sequence number
      arr = ctx->sock->buffer;
      raw_seq_no = UINT64_C (0);
      for (i = 0; i < 8; i++ )
      {
        tmp = UINT64_C (0);
        tmp = arr[8 - i - 1];
        raw_seq_no |= (tmp << ((i & 7) << 3));
      }

      // handle the global offset in sequence numbers
      fixed_raw_seq_no = raw_seq_no + global_offset;

      // adjust for the offset of this distributor
      offset_raw_seq_no = fixed_raw_seq_no - seq_offset;

      // check the remainder of the sequence number, for errors in global offset
      remainder = offset_raw_seq_no % seq_inc;

      if (remainder == 0) 
      {
        // do nothing
      } 
      else if (remainder < (seq_inc / 2)) 
      {
        multilog(ctx->log, LOG_WARNING, "receive_obs: adjusting global offset from %"PRIi64" to "
                 "%"PRIi64"\n", global_offset, (global_offset - remainder));
        global_offset -= remainder;
        fixed_raw_seq_no = raw_seq_no + global_offset;
        offset_raw_seq_no = fixed_raw_seq_no - seq_offset;
        remainder = offset_raw_seq_no % seq_inc;
      }
      else if (remainder >= seq_inc / 2)
      {
        multilog(ctx->log, LOG_WARNING, "receive_obs: adjusting global offset from %"PRIi64" "
                 "to %"PRIi64"\n", global_offset, (global_offset + (seq_inc - remainder)));
        global_offset += (seq_inc - remainder);
        fixed_raw_seq_no = raw_seq_no + global_offset;
        offset_raw_seq_no = fixed_raw_seq_no - seq_offset;
        remainder = offset_raw_seq_no % seq_inc;
      } 
      else 
      {
        // the offset was too great to "fix"
      }

      seq_no = (offset_raw_seq_no) / seq_inc;

      pad_start = seq_no;

      if (prev_seq_no)
      {
        if (seq_no == prev_seq_no + 1)
        {
          // this is normal, do nothing
        }
        else if (seq_no < prev_seq_no)
        {
          // this is a normal packet reset
          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "receive_obs: RESET raw=%"PRIu64", fixed=%"PRIu64", "
                     "offset=%"PRIu64", seq_inc=%"PRIu64", remainder=%d, seq=%"PRIu64", "
                     "prev_seq=%"PRIu64", seq_diff=%"PRIi64"\n", raw_seq_no, fixed_raw_seq_no, 
                     offset_raw_seq_no, seq_inc, (int) (offset_raw_seq_no % seq_inc), 
                     seq_no, prev_seq_no, (int64_t) seq_no - (int64_t) prev_seq_no);
          else
            multilog(ctx->log, LOG_INFO, "packet reset detected seq=%"PRIu64" prev=%"PRIu64"\n",
                     seq_no, prev_seq_no);
        }
        else
        {
          // this is a problematic packet sequence number jump
          multilog(ctx->log, LOG_INFO, "receive_obs: JUMP raw=%"PRIu64", fixed=%"PRIu64", "
                   "offset=%"PRIu64", seq_inc=%"PRIu64", remainder=%d, seq=%"PRIu64", "
                   "prev_seq=%"PRIu64", seq_diff=%"PRIi64"\n", raw_seq_no, fixed_raw_seq_no, 
                   offset_raw_seq_no, seq_inc, (int) (offset_raw_seq_no % seq_inc), 
                   seq_no, prev_seq_no, (int64_t) seq_no - (int64_t) prev_seq_no);

          uint64_t total_dropped = ctx->packets->dropped + (int64_t) seq_no - (int64_t) prev_seq_no;
          double   dropped_data = (double) total_dropped * (double) UDP_DATA;
          if (dropped_data > 1e12)
            multilog(ctx->log, LOG_WARNING, "dropped %"PRIi64" pkts, %6.2f TB\n", total_dropped, dropped_data / 1e12);
          else if (dropped_data > 1e9)
            multilog(ctx->log, LOG_WARNING, "dropped %"PRIi64" pkts, %6.2f GB\n", total_dropped, dropped_data / 1e9);
          else if (dropped_data > 1e6)
            multilog(ctx->log, LOG_WARNING, "dropped %"PRIi64" pkts, %6.2f MB\n", total_dropped, dropped_data / 1e6);
          else
            multilog(ctx->log, LOG_WARNING, "dropped %"PRIi64" pkts, %6.0f B\n", total_dropped, dropped_data);

          // need to start padding with zero's until we catch up
         pad_start = prev_seq_no + 1;
        }
      }

      if ((!ignore_packet) && (remainder != 0)) 
      {
        multilog(ctx->log, LOG_INFO, "receive_obs: PROB  raw=%"PRIu64", fixed=%"PRIu64", "
                 "offset=%"PRIu64", seq_inc=%"PRIu64", remainder=%d, seq=%"PRIu64", "
                 "seq_diff=%"PRIi64"\n", raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no, 
                  seq_inc, (int) (offset_raw_seq_no % seq_inc), seq_no, 
                  (int64_t) seq_no - (int64_t) prev_seq_no);
        ignore_packet = 1;
        ctx->packet_reset = 1;
        problem_packets++;
      }

      prev_seq_no = seq_no;

      if ((seq_no < 3) && (!ignore_packet)) 
      {
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "receive_obs: START raw=%"PRIu64", fixed=%"PRIu64", "
                   "offset=%"PRIu64", seq=%"PRIu64"\n", raw_seq_no, fixed_raw_seq_no, 
                    offset_raw_seq_no, seq_no);

      }

      // If we are waiting for the sequence number to be reset
      if ((waiting_for_start) && (!ignore_packet)) 
      {
        if (seq_no < 10000) 
        {
          if (ctx->verbose) 
          {
            multilog(ctx->log, LOG_INFO, "receive_obs: ctx->packet_reset = 1\n");
            multilog(ctx->log, LOG_INFO, "receive_obs: buffer_start_byte=%"PRIu64"\n", ctx->buffer_start_byte);
            multilog(ctx->log, LOG_INFO, "receive_obs: buffer_end_byte=%"PRIu64"\n", ctx->buffer_end_byte);
            multilog(ctx->log, LOG_INFO, "receive_obs: xfer_start_byte=%"PRIu64"\n", ctx->xfer_start_byte);
            multilog(ctx->log, LOG_INFO, "receive_obs: xfer_end_byte=%"PRIu64"\n", ctx->xfer_end_byte);
          }

          ctx->packet_reset = 1;
          waiting_for_start = 0;
          ctx->next_seq = 0;
        }
        else 
          ignore_packet = 1;
      }
    }

    // If we will process the packet
    if (!ignore_packet) 
    {

      have_packet = 0;

      // this loop should only be iterated once unless there is a packet jump
      for (iseq=pad_start; iseq<=seq_no; iseq++)
      {

        if (iseq > ctx->next_seq)
          ctx->ooo_packets += iseq - ctx->next_seq;

        // determine the offset for this packet in the current ring buffer
        seq_byte = iseq * UDP_DATA;

        // check for the stopping condition
        if (stop_byte)
        {
          if (seq_byte >= stop_byte)
          {
            if (ctx->verbose)
            {
              multilog(ctx->log, LOG_INFO, "receive_obs: STOP seq_byte[%"PRIu64"]"
                       " >= stop_byte[%"PRIu64"], stopping\n", seq_byte, stop_byte);
              multilog(ctx->log, LOG_INFO, "receive_obs: STOP buffer[%"PRIu64" - "
                       "%"PRIu64"]\n", ctx->buffer_start_byte, ctx->buffer_end_byte);
              multilog(ctx->log, LOG_INFO, "receive_obs: STOP xfer[%"PRIu64" - "
                       "%"PRIu64"]\n", ctx->xfer_start_byte, ctx->xfer_end_byte);
            }
            stop_pending = 1;

            // try to determine how much data has been written
            uint64_t bytes_just_written = 0;
            if (seq_byte < ctx->buffer_end_byte)
              bytes_just_written = seq_byte - ctx->buffer_start_byte;
            else
              bytes_just_written = ctx->hdu_bufsz;

            if (ctx->verbose)
            {
              multilog(ctx->log, LOG_INFO, "receive_obs: STOP hdu=%d, bytes_just_"
                       "written=%"PRIu64" bytes_in_xfer=%"PRIu64" bytes\n", 
                       ctx->ihdu, bytes_just_written, (seq_byte - ctx->xfer_start_byte));
              multilog(ctx->log, LOG_INFO, "receive_obs: STOP caspsr_udpNdb_close_"
                       "hdu(%"PRIu64")\n", bytes_just_written);
            }

            if (caspsr_udpNdb_close_hdu (ctx, bytes_just_written) < 0)
            {
              multilog(ctx->log, LOG_ERR, "receive_obs: caspsr_udpNdb_close_hdu failed\n");
              thread_result = -1;
              pthread_exit((void *) &thread_result);
            }
            stop_byte = 0;
            break;
          }
        }

        if (stop_pending) {
          multilog(ctx->log, LOG_ERR, "receive_obs: stop_pending after break - SHOULD NOT HAPPEN!\n");
        }

        // packet arrived too late, ignore it
        if (seq_byte < ctx->buffer_start_byte)
        {
          multilog (log, LOG_WARNING, "PACKET DROP seq_byte=%"PRIu64", buffer_start_byte=%"PRIu64"\n", 
                        seq_byte, ctx->buffer_end_byte);
          ctx->packets->dropped++;
          ctx->bytes->dropped += UDP_DATA;
        }
        else
        {
          // the packet arrived on time, process it
          if (seq_byte <= ctx->buffer_end_byte)
          {
            // no need to adjust, belongs in current block

          } 
          // the packet belongs in a subsequent buffer of this xfer
          else if (seq_byte < ctx->xfer_end_byte)
          {
            if (ctx->verbose)
              multilog (log, LOG_INFO, "BLOCK COMPLETE hdu[%d] seq_no=%"PRIu64", "
                        "seq_byte=%"PRIu64", buffer_end_byte=%"PRIu64"\n", 
                        ctx->ihdu, iseq, seq_byte, ctx->buffer_end_byte);
            if (caspsr_udpNdb_new_buffer (ctx) < 0)
            {
              multilog(ctx->log, LOG_ERR, "receive_obs: caspsr_udpNdb_new_buffer failed\n");
              thread_result = -1;
              pthread_exit((void *) &thread_result);
            }
          } 
          // the packet belongs in the next xfer
          else 
          {
            if (ctx->verbose)
              multilog (log, LOG_INFO, "XFER COMPLETE hdu[%d] seq_no=%"PRIu64", "
                        "seq_byte=%"PRIu64", xfer_end_byte=%"PRIu64"\n", 
                        ctx->ihdu, iseq, seq_byte, ctx->xfer_end_byte);

            // close current buffer, hdu, open next hdu and open first block
            if (caspsr_udpNdb_new_hdu (ctx) < 0)
            {
              multilog(ctx->log, LOG_ERR, "receive_obs: caspsr_udpNdb_new_hdu failed\n");
              thread_result = -1;
              pthread_exit((void *) &thread_result);
            }
          }

          if (!ctx->current_buffer)
          {
            multilog (log, LOG_ERR, "receive_obs: failed to move to new buffer element\n");
          }

          byte_offset = seq_byte - ctx->buffer_start_byte;

          if ((byte_offset < 0) || (byte_offset > (ctx->hdu_bufsz-UDP_DATA)))
          {
            multilog (log, LOG_INFO, "receive_obs: ctx->hdu_bufsz=%"PRIu64"\n", ctx->hdu_bufsz);
            multilog (log, LOG_INFO, "receive_obs: byte_offset=%"PRIi64"\n", byte_offset);
            multilog (log, LOG_INFO, "receive_obs: seq_byte=%"PRIu64"\n", seq_byte);
            multilog (log, LOG_INFO, "receive_obs: buffer_start_byte=%"PRIu64", buffer_end_byte=%"PRIu64"\n",
                      ctx->buffer_start_byte, ctx->buffer_end_byte);

            multilog(ctx->log, LOG_ERR, "receive_obs: packet sequence jumped too far in future\n");
            thread_result = -1;
            pthread_exit((void *) &thread_result);
          }

          if (iseq == seq_no)
          {
            // write the packet data (minus seq_no and chid) into ring buffer
            memcpy (ctx->current_buffer + byte_offset, ctx->sock->buffer + UDP_HEADER, UDP_DATA);
            ctx->packets->received++;
          }
          else
          {
            memcpy (ctx->current_buffer + byte_offset, ctx->zeroed_packet, UDP_DATA);
            ctx->packets->dropped++;
          }

          // update statistics
          ctx->bytes->received += UDP_DATA;
        }
      } 
    }

    // we have ignored more than 10 packets
    if (problem_packets > 10) {
      multilog (ctx->log, LOG_WARNING, "Ignored more than 10 packets, exiting\n");
      quit_threads = 1;  
    }

    //if (ctx->bytes_to_acquire && (ctx->bytes_to_acquire > bytes_total))
    //  quit_threads = 1;

  }

  stop_pending = 0;

  if (quit_threads && ctx->verbose) 
    multilog (ctx->log, LOG_INFO, "main_function: quit_threads detected\n");
 
  if (ctx->verbose) 
    multilog(log, LOG_INFO, "receiving thread exiting\n");

  /* return 0 */
  pthread_exit((void *) &thread_result);
}

/*
 * Close the udp socket and file
 */

int udpNdb_stop_function (udpNdb_t* ctx)
{

  multilog_t *log = ctx->log;

  /* get our context, contains all required params */
  if (ctx->packets->dropped && ctx->next_seq > 0) {
    double percent = (double) ctx->packets->dropped / (double) ctx->next_seq;
    percent *= 100;

    multilog(log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64 " = %8.6f %\n"
             ,ctx->packets->dropped, ctx->next_seq, percent);
  }

  // re-open each datablock and write a 1 byte xfer with OBS_XFER==-1 to 
  // signifiy the end of the observation
  ctx->obs_xfer = -1;
  if (ctx->verbose)
    multilog(log, LOG_INFO, "stop: setting obs_xfer=%"PRIi64"\n",
              ctx->obs_xfer);

  sleep(3);

  unsigned i=0;
  unsigned j=0;
  unsigned last_hdu = ctx->ihdu;
  for (i=0; i < ctx->nhdus; i++)
  {
    sleep(3);

    j = (i + 1 + last_hdu) % ctx->nhdus;

    ctx->ihdu = j;

    ctx->obs_offset += ctx->packets_per_xfer * UDP_DATA * ctx->n_distrib;

    if (ctx->verbose)
      multilog(log, LOG_INFO, "stop: hdu=%d, setting obs_offset="
               "%"PRIi64"\n", ctx->ihdu, ctx->obs_offset);

    if (ctx->verbose)
      multilog(log, LOG_INFO, "stop: caspsr_udpNdb_open_hdu() [%d] \n", ctx->ihdu);
    if (caspsr_udpNdb_open_hdu(ctx) < 0)
      multilog(log, LOG_ERR, "stop: failed to open hdu[%d]\n", ctx->ihdu);

    // write just 1 byte of data

    if (ctx->verbose)
      multilog(log, LOG_INFO, "stop: caspsr_udpNdb_close_hdu(%d) [%d]\n", 1, ctx->ihdu);
    if (caspsr_udpNdb_close_hdu(ctx, 1) < 0)
      multilog(log, LOG_ERR, "stop: failed to close hdu[%d]\n", ctx->ihdu);

  }

  close(ctx->sock->fd);
  recording = 0;

  if (ctx->obs_header)
    free (ctx->obs_header);
  ctx->obs_header = 0;

  return 0;

}


/* 
 *  connect to the specified datablocks
 */
int caspsr_udpNdb_connect_dbs (udpNdb_t * ctx, char ** argv, int num_dbs)
{

  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "caspsr_udpNdb_connect_dbs()\n");

  unsigned i;
  key_t dada_key;

  ctx->nhdus = num_dbs;
  ctx->hdus = (dada_hdu_t **) malloc(sizeof(dada_hdu_t *) * num_dbs);
  assert (ctx->hdus);

  for (i=0; i<ctx->nhdus; i++)
  {
    uint64_t block_size = 0;

    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "connect_dbs: connect hdu[%d] %s\n", i, argv[i]);

    if (sscanf (argv[i], "%x", &dada_key) != 1) {
      multilog (ctx->log, LOG_ERR, "could not parse key %d from %s\n", i, argv[i]);
      return -1;
    }

    ctx->hdus[i] = dada_hdu_create (ctx->log);

    dada_hdu_set_key (ctx->hdus[i], dada_key);

    if (dada_hdu_connect (ctx->hdus[i]) < 0)
    {
      multilog (ctx->log, LOG_ERR, "could not connect to hdu %d\n", i);
      return -1; 
    }

    // if we dont yet know the packets per buffer
    block_size = ipcbuf_get_bufsz ((ipcbuf_t *) ctx->hdus[i]->data_block);
    if (!ctx->packets_per_buffer)
    {
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "connect_dbs: hdu%d block size=%"PRIu64"\n", i, block_size);

      if (block_size % UDP_DATA != 0)
      {
        multilog (ctx->log, LOG_ERR, "connect_dbs: data block size for hdu%d"
                  "[%"PRIu64"] was not a multiple of the UDP_DATA size [%d]\n",
                  i, block_size, UDP_DATA);
        return -1;
      }
      ctx->hdu_bufsz = block_size;
      ctx->packets_per_buffer = ctx->hdu_bufsz / UDP_DATA;

      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "connect_dbs: packets_per_buffer=%"PRIu64"\n",
                  ctx->packets_per_buffer);
    }
    else
    {
      if (block_size != ctx->hdu_bufsz)
      {
        multilog (ctx->log, LOG_ERR, "data block size for hdu%d [%"PRIu64"] "
                  "did not match hdu0 [%"PRIu64"]\n", i,  block_size, ctx->hdu_bufsz);
        return -1;
      }
    }
  }

  return 0;
}

/*  
 *  unlock write and disconnect from all hdus
 */
int caspsr_udpNdb_destroy_dbs (udpNdb_t * ctx)
{

  unsigned i=0;
  if (ctx->hdus)
  {
    for (i=0; i<ctx->nhdus; i++)
    {
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "destroy_dbs: disconnecting hdu%d\n", i);
      // only the current hdu should be open/locked
      //if (ctx->ihdu == i) 
      //{
      //  if (dada_hdu_unlock_write (ctx->hdus[i]) < 0)
      //    multilog (ctx->log, LOG_ERR, "could not unlock write on hdu %d\n", i);
      //}
      if (dada_hdu_disconnect (ctx->hdus[i]) < 0)
        multilog (ctx->log, LOG_ERR, "could not unlock write on hdu %d\n", i);
    }

    free (ctx->hdus);
    ctx->hdus = 0;
  }

}


int main (int argc, char **argv)
{

  /* DADA Logger */ 
  multilog_t* log = 0;

  /* number of this distributor */
  unsigned i_distrib = 0;

  /* total number of distributors being used */
  unsigned n_distrib = 0;

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port for control commands */
  int control_port = 0;

  /* port for incoming UDP packets */
  int inc_port = CASPSR_DEFAULT_UDPNDB_PORT;

  /* port for outgoing IB connections packets */
  int out_port = CASPSR_DEFAULT_UDPNDB_PORT;

  /* multilog output port */
  int l_port = CASPSR_DEFAULT_PWC_LOGPORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  int verbose = 0;

  /* number of seconds/bytes to acquire for */
  unsigned nsecs = 0;

  /* actual struct with info */
  udpNdb_t udpNdb;

  /* custom header from a file, implies no controlling pwcc */
  char * header_file = NULL;

  /* Pointer to array of "read" data */
  char *src;

  /* Ignore dropped packets */
  unsigned ignore_dropped = 0;

  /* number of packets for each tranfer */
  unsigned packets_per_xfer = CASPSR_UDPNDB_PACKS_PER_XFER;

  /* number of packets to append to each transfer */ 
  int packets_to_append = 0;

  int arg = 0;

  /* statistics thread */
  pthread_t stats_thread_id;

  /* control thread */
  pthread_t control_thread_id;

  /* receiving thread */
  pthread_t receiving_thread_id;

  while ((arg=getopt(argc,argv,"b:di:l:n:o:p:q:t:vh")) != -1) {
    switch (arg) {

    case 'b':
      packets_to_append = atoi(optarg);
      break;

    case 'd':
      daemon = 1;
      break; 

    case 'i':
      if (optarg)
        interface = optarg;
      break;
    
    case 'l':
      if (optarg) {
        l_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'n':
      packets_per_xfer = atoi(optarg);
      break;

    case 'o':
      control_port = atoi(optarg);
      break;

    case 'p':
      inc_port = atoi (optarg);
      break;

    case 'q':
      out_port = atoi (optarg);
      break;

    case 't':
      nsecs = atoi (optarg);
      break;

    case 'v':
      verbose++;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }
  
  if (packets_to_append) {
    fprintf(stderr, "packet appending not implemented yet!\n");
    exit(EXIT_FAILURE);
  }

  /* check the command line arguments */
  int num_dbs = (argc-optind) - 2;
  int i = 0;
  int rval = 0;
  void* result = 0;

  if (num_dbs < 1 || num_dbs > 16) {
    fprintf(stderr, "ERROR: must have at between 1 and 16 \n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  /* parse the number this distributor */
  if (sscanf(argv[optind], "%d", &i_distrib) != 1) {
    fprintf(stderr, "ERROR: failed to parse the number of this distributor\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  /* parse the number of distributors */
  if (sscanf(argv[optind+1], "%d", &n_distrib) != 1) {
    fprintf(stderr, "ERROR: failed to parse the number of distributors\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  if ((n_distrib < 2) || (n_distrib > 4)) {
    fprintf(stderr, "ERROR: number of distributors must be between 2 and 4 [parsed %d]\n", n_distrib);
    usage();
    exit(EXIT_FAILURE);
  }

  if ((i_distrib < 0) || (i_distrib >= n_distrib)) {
    fprintf(stderr, "ERROR: this distributor's number must be between 0 and %d [parsed %d]\n", (n_distrib-1), i_distrib);
    usage();
    exit(EXIT_FAILURE);
  }

  log = multilog_open ("caspsr_udpNdb", 0);

  if (daemon)
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  udpNdb.log = log;
  udpNdb.verbose = verbose;
  udpNdb.port = inc_port;
  udpNdb.interface = strdup(interface);
  udpNdb.control_port = control_port;
  udpNdb.packets_per_buffer = 0;
  udpNdb.packets_per_xfer = packets_per_xfer;
  udpNdb.n_distrib = n_distrib;
  udpNdb.i_distrib = i_distrib;
  udpNdb.bytes_to_acquire = -1;

  // connect to each data block
  if (verbose)
    multilog (log, LOG_INFO, "main: caspsr_udpNdb_connect_dbs(%d)\n", num_dbs);
  if (caspsr_udpNdb_connect_dbs(&udpNdb, &(argv[optind+2]), num_dbs) < 0) 
  {
    multilog (log, LOG_ERR, "could not connect to all datablocks\n");
    return EXIT_FAILURE;
  }

  // initialize the socket
  if (caspsr_udpNdb_init_receiver (&udpNdb) < 0)
  {
    multilog (log, LOG_ERR, "could not initialize socket\n");
    return EXIT_FAILURE;
  }

  signal(SIGINT, signal_handler);

  // start the control thread
  if (control_port) 
  {
    if (verbose)
      multilog(log, LOG_INFO, "starting control_thread()\n");
    rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpNdb);
    if (rval != 0) {
      multilog(log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
      return -1;
    }
  }

  if (verbose)
    multilog(log, LOG_INFO, "starting stats_thread()\n");
  rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpNdb);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }

  // Main control loop
  while (!quit_threads) 
  {

    caspsr_udpNnb_reset_receiver (&udpNdb);

    // wait for a START command before initialising receivers
    while (!start_pending && !quit_threads && control_port) 
      sleep(1);

    if (quit_threads)
      break;

    time_t utc = caspsr_udpNdb_start_function(&udpNdb);
    if (utc == -1 ) {
      multilog(log, LOG_ERR, "Could not run start function\n");
      return EXIT_FAILURE;
    }

    /* process on last 4 cores */
    udpNdb.recv_core = (i_distrib - (i_distrib % 2)) + 4; 

    /* make the receiving thread high priority */
    // Disabled atm
    /*
    int min_pri = sched_get_priority_min(SCHED_RR);
    int max_pri = sched_get_priority_max(SCHED_RR);
    if (verbose)
      multilog(log, LOG_INFO, "sched_priority: min=%d, max=%d\n", min_pri, max_pri);

    struct sched_param recv_parm;
    recv_parm.sched_priority=max_pri;

    pthread_attr_t recv_attr;

    if (pthread_attr_init(&recv_attr) != 0) {
      fprintf(stderr, "pthread_attr_init failed: %s\n", strerror(errno));
      return 1;
    }
  
    if (pthread_attr_setinheritsched(&recv_attr, PTHREAD_EXPLICIT_SCHED) != 0) {
      fprintf(stderr, "pthread_attr_setinheritsched failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedpolicy(&recv_attr, SCHED_RR) != 0) {
      fprintf(stderr, "pthread_attr_setschedpolicy failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedparam(&recv_attr,&recv_parm) != 0) {
      fprintf(stderr, "pthread_attr_setschedparam failed: %s\n", strerror(errno));
      return 1;
    }
    */


    /* set the total number of bytes to acquire */
    udpNdb.bytes_to_acquire = 1600 * 1000 * 1000 * (int64_t) nsecs;
    udpNdb.bytes_to_acquire /= udpNdb.n_distrib;

    if (verbose)
    { 
      if (udpNdb.bytes_to_acquire) 
        multilog(log, LOG_INFO, "bytes_to_acquire = %"PRIu64" Million Bytes, nsecs=%d\n", udpNdb.bytes_to_acquire/1000000, nsecs);
      else
        multilog(log, LOG_INFO, "Acquiring data indefinitely\n");
    }

    if (verbose)
      multilog(log, LOG_INFO, "starting caspsr_udpNdb_receive_obs thread\n");
    //rval = pthread_create (&receiving_thread_id, &recv_attr, (void *) caspsr_udpNdb_receive_obs , (void *) &udpNdb);
    rval = pthread_create (&receiving_thread_id, 0, (void *) caspsr_udpNdb_receive_obs , (void *) &udpNdb);
    if (rval != 0) {
      multilog(log, LOG_INFO, "Error creating caspsr_udpNdb_receive_obs thread: %s\n", strerror(rval));
      return -1;
    }

    if (verbose) 
      multilog(log, LOG_INFO, "joining caspsr_udpNdb_receive_obs thread\n");
    pthread_join (receiving_thread_id, &result);

    if (verbose) 
      multilog(log, LOG_INFO, "udpNdb_stop_function\n");
    if ( udpNdb_stop_function(&udpNdb) != 0)
      fprintf(stderr, "Error stopping acquisition");

    if (!control_port)
      quit_threads = 1;

  }

  if (control_port)
  {
    if (verbose)
      multilog(log, LOG_INFO, "joining control_thread\n");
    pthread_join (control_thread_id, &result);
  }

  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  pthread_join (stats_thread_id, &result);

  /* clean up memory */
  if ( caspsr_udpNdb_destroy_receiver (&udpNdb) < 0) 
    fprintf(stderr, "failed to clean up receivers\n");

  if (caspsr_udpNdb_destroy_dbs (&udpNdb) < 0)
  {
    fprintf(stderr, "failed to disconnect from dbs\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}


/*
 *  Thread to control the acquisition of data, allows only 1 connection at a time
 */
void control_thread (void * arg) 
{

  udpNdb_t * ctx = (udpNdb_t *) arg;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: starting\n");

  // port on which to listen for control commands
  int port = ctx->control_port;

  // buffer for incoming command strings
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  assert (buffer != 0);

  const char* whitespace = " \r\t\n";
  char * command = 0;
  char * args = 0;
  time_t utc_start = 0;

  FILE *sockin = 0;
  FILE *sockout = 0;
  int listen_fd = 0;
  int fd = 0;
  char *rgot = 0;
  int readsocks = 0;
  fd_set socks;
  struct timeval timeout;

  // create a socket on which to listen
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: creating socket on port %d\n", port);

  listen_fd = sock_create (&port);
  if (listen_fd < 0)  {
    multilog(ctx->log, LOG_ERR, "Failed to create socket for control commands: %s\n", strerror(errno));
    free (buffer);
    return;
  }

  while (!quit_threads) {

    // reset the FD set for selecting  
    FD_ZERO(&socks);
    FD_SET(listen_fd, &socks);
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    readsocks = select(listen_fd+1, &socks, (fd_set *) 0, (fd_set *) 0, &timeout);

    // error on select
    if (readsocks < 0) 
    {
      perror("select");
      exit(EXIT_FAILURE);
    }

    // no connections, just ignore
    else if (readsocks == 0) 
    {
    } 

    // accept the connection  
    else 
    {
   
      if (ctx->verbose) 
        multilog(ctx->log, LOG_INFO, "control_thread: accepting conection\n");

      fd =  sock_accept (listen_fd);
      if (fd < 0)  {
        multilog(ctx->log, LOG_WARNING, "control_thread: Error accepting "
                                        "connection %s\n", strerror(errno));
        break;
      }

      sockin = fdopen(fd,"r");
      if (!sockin)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating input "
                                        "stream %s\n", strerror(errno));


      sockout = fdopen(fd,"w");
      if (!sockout)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating output "
                                        "stream %s\n", strerror(errno));

      setbuf (sockin, 0);
      setbuf (sockout, 0);

      rgot = fgets (buffer, bufsize, sockin);

      if (rgot && !feof(sockin)) {

        buffer[strlen(buffer)-2] = '\0';

        args = buffer;

        // parse the command and arguements
        command = strsep (&args, whitespace);

        if (ctx->verbose)
        {
          multilog(ctx->log, LOG_INFO, "control_thread: command=%s\n", command);
          if (args != NULL)
            multilog(ctx->log, LOG_INFO, "control_thread: args=%s\n", args);
        }

        // REQUEST STATISTICS
        if (strcmp(command, "STATS") == 0) 
        {
          fprintf (sockout, "mb_rcv_ps=%4.1f,mb_drp_ps=%4.1f,"
                            "ooo_pkts=%"PRIu64",mb_free=%4.1f,mb_total=%4.1f\r\n", 
                             ctx->mb_rcv_ps, ctx->mb_drp_ps, 
                             ctx->ooo_packets, ctx->mb_free, ctx->mb_total);
          fprintf (sockout, "ok\r\n");
        }

        else if (strcmp(command, "SET_UTC_START") == 0)
        {
          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: SET_UTC_START command received\n");
          if (args == NULL)
          {
            multilog(ctx->log, LOG_ERR, "control_thread: no time specified for SET_UTC_START\n");
            fprintf(sockout, "fail\r\n");
          }
          else
          {
            time_t utc = str2utctime (args);
            if (utc == (time_t)-1)
            {
              multilog(ctx->log, LOG_WARNING, "control_thread: could not parse "
                       "UTC_START time from %s\n", args);
              fprintf(sockout, "fail\r\n");
            }
            else
            {
              if (ctx->verbose)
                multilog(ctx->log, LOG_INFO, "control_thread: parsed UTC_START as %d\n", utc);
              utc_start = utc;
              fprintf(sockout, "ok\r\n");
              multilog(ctx->log, LOG_INFO, "set_utc_start %s\n", args);
            }
          }
        }

        // START COMMAND
        else if (strcmp(command, "START") == 0) {

          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: START command received\n");

          start_pending = 1;
          while (recording != 1) 
          {
            sleep(1);
          }
          start_pending = 0;
          fprintf(sockout, "ok\r\n");

          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording started\n");
        }

        // FLUSH COMMAND - stop acquisition of data, but flush all packets already received
        else if (strcmp(command, "FLUSH") == 0)
        {
          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: FLUSH command received, stopping recording\n");

          stop_pending = 1;
          while (recording != 0)
          {
            sleep(1);
          }
          stop_pending = 0;
          fprintf(sockout, "ok\r\n");

          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording stopped\n");
        }

        // UTC_STOP command
        else if (strcmp(command, "UTC_STOP") == 0)
        {
          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: UTC_STOP command received\n");

          if (args == NULL) 
          {
            multilog(ctx->log, LOG_ERR, "control_thread: no UTC specified for UTC_STOP\n");
            fprintf(sockout, "fail\r\n");
          }
          else
          {
            time_t utc = str2utctime (args);
            if (utc == (time_t)-1) 
            {
              multilog(ctx->log, LOG_WARNING, "control_thread: could not parse "
                       "UTC_STOP time from %s\n", args);
              fprintf(sockout, "fail\r\n");
            }
            else
            {
              if (ctx->verbose)
                multilog(ctx->log, LOG_INFO, "control_thread: parsed UTC_STOP as %d\n", utc); 
              uint64_t byte_to_stop = (utc - utc_start);
              byte_to_stop *= 800 * 1000 * 1000;
              if (ctx->verbose)
                multilog(ctx->log, LOG_INFO, "control_thread: total_secs=%d, "
                         "stopping byte=%"PRIu64"\n", (utc - utc_start), byte_to_stop);
              stop_byte = byte_to_stop;
              stop_pending = 0;
              utc_start = 0;
              fprintf(sockout, "ok\r\n");
              multilog(ctx->log, LOG_INFO, "utc_stop %s\n", args);
            }
          }
        }

        // STOP command, stops immediately
        else if (strcmp(command, "STOP") == 0)
        {
          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: STOP command received, stopping immediately\n");

          stop_pending = 2;
          while (recording != 0)
          {
            sleep(1);
          }
          stop_pending = 0;
          fprintf(sockout, "ok\r\n");

          if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording stopped\n");
        }


        // QUIT COMMAND, immediately exit 
        else if (strcmp(command, "QUIT") == 0) 
        {
          multilog(ctx->log, LOG_INFO, "control_thread: QUIT command received, exiting\n");
          quit_threads = 1;
          fprintf(sockout, "ok\r\n");
        }

        // UNRECOGNISED COMMAND
        else 
        {
          multilog(ctx->log, LOG_WARNING, "control_thread: unrecognised command: %s\n", buffer);
          fprintf(sockout, "fail\r\n");
        }
      }
    }

    close(fd);
  }
  close(listen_fd);

  free (buffer);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: exiting\n");

}

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  udpNdb_t * ctx = (udpNdb_t *) arg;

  uint64_t b_rcv_total = 0;
  uint64_t b_rcv_1sec = 0;
  uint64_t b_rcv_curr = 0;

  uint64_t b_drp_total = 0;
  uint64_t b_drp_1sec = 0;
  uint64_t b_drp_curr = 0;

  uint64_t s_rcv_total = 0;
  uint64_t s_rcv_1sec = 0;
  uint64_t s_rcv_curr = 0;

  uint64_t ooo_pkts = 0;

  while (!quit_threads)
  {
            
    /* get a snapshot of the data as quickly as possible */
    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    s_rcv_curr = ctx->n_sleeps;
    ooo_pkts   = ctx->ooo_packets;

    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_drp_1sec = b_drp_curr - b_drp_total;
    s_rcv_1sec = s_rcv_curr - s_rcv_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_drp_total = b_drp_curr;
    s_rcv_total = s_rcv_curr;

    ctx->mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    ctx->mb_drp_ps = (double) b_drp_1sec / 1000000;

    /* determine how much memory is free in the receivers */

    if (ctx->verbose)
      fprintf (stderr,"R=%4.1f, D=%4.1f [MB/s], s_s=%"PRIu64", OoO pkts=%"PRIu64"\n", ctx->mb_rcv_ps, ctx->mb_drp_ps, s_rcv_1sec, ooo_pkts);

    sleep(1);
  }
}

/*
 *  Simple signal handler to exit more gracefully
 */
void signal_handler(int signalValue) {

  if (quit_threads) {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;

}
