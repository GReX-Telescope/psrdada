/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>

#include "caspsr_udp.h"

caspsr_sock_t * caspsr_init_sock()
{
  caspsr_sock_t * b = (caspsr_sock_t *) malloc(sizeof(caspsr_sock_t));

  assert(b != NULL);

  b->size = sizeof(char) * UDP_PAYLOAD;
  b->buffer = (char *) malloc(b->size);

  assert(b->buffer != NULL);

  b->fd = 0;
  b->have_packet = 0;

  return b;
}


/* 
 * opens the UDP socket, setting the kernel's buffer to 512MB, setting it to non-blocking and
 * clearing any packets at the buffer */
int caspsr_open_sock(caspsr_receiver_t * ctx)
{

  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);

  if (ctx->sock->fd < 0) {
    multilog (ctx->log, LOG_ERR, "caspsr_open_sock: failed to create udp socket %s:%d\n", ctx->interface, ctx->port);
    return -1;
  }
    
  /* set the socket size to 512 MB */
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, 536870912);
      
  /* set the socket to non-blocking */
  sock_nonblock(ctx->sock->fd);
        
  /* clear any packets buffered by the kernel */
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "caspsr_open_sock: cleared %d bytes [%d packets] just after the "
                                  "scoket was opened\n", cleared, (cleared/UDP_PAYLOAD));

  return 0;
}

/* 
 * close the caspsr_receiver's UDP socket 
 */
int caspsr_close_sock(caspsr_receiver_t * ctx)
{
  int rval = 0;
  if (ctx->sock->fd)
  {
    rval = close(ctx->sock->fd);
    if (rval == -1) 
      multilog (ctx->log, LOG_ERR, "caspsr_close_sock: close failed: %s\n", strerror(errno));
  }

  ctx->sock->fd = 0;
  ctx->sock->have_packet = 0;
  
  return rval;

}


void caspsr_free_sock(caspsr_sock_t* b)
{
  b->fd = 0;
  b->size = 0;
  b->have_packet = 0;
  free(b->buffer);
}


caspsr_data_t * caspsr_init_data()
{
  caspsr_data_t * b = (caspsr_data_t *) malloc(sizeof(caspsr_data_t));

  assert(b != NULL);

  b->size = sizeof(char) * UDP_DATA * UDP_NPACKS;

  if (posix_memalign ( (void **) &(b->buffer), 512, b->size) != 0) {
    fprintf(stderr, "caspsr_init_data: failed to allocate aligned memory: %s\n",
                    strerror(errno));
    return 0;
  }


  if (! b->buffer)
  {
    fprintf(stderr, "caspsr_init_data: failed to allocate %"PRIu64" bytes\n", b->size);
    return 0;
  }

  if (mlock((void *)  b->buffer, b->size) < 0)
    fprintf(stderr, "caspsr_init_data: failed to lock buffer memory: %s\n", strerror(errno));

  caspsr_reset_data_t(b);

  return b;
}

void caspsr_reset_data_t(caspsr_data_t * b)
{
  b->count = 0;
  b->min = 0;
  b->max = 0;
}

void caspsr_zero_data(caspsr_data_t * b)
{
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(b->buffer, zerodchar, b->size);
}

void caspsr_free_data(caspsr_data_t * b)
{
  if (b->buffer) 
  {
    if (munlock((void *) b->buffer, (size_t) b->size) < 0 )
      fprintf(stderr, "caspsr_free_data: failed to munlock buffer: %s\n", strerror(errno));
    free(b->buffer);
  }
  else 
    fprintf(stderr, "free_caspsr_buffer: b->buffer was unexpectedly freed\n");

  b->size = 0;
  b->count = 0;
  b->min = 0;
  b->max = 0;
}


/* copy b_length data from s to b, subject to the s_offset, wrapping if
 * necessary and return the new s_offset */
unsigned int caspsr_encode_data(char * b, char * s, unsigned int b_length, 
                              unsigned int s_length, unsigned int s_offset) {
#ifdef _DEBUG
  fprintf(stderr, "caspsr_encode_data: b_length=%d, s_length=%d, s_offset=%d\n",
                  b_length, s_length, s_offset);
#endif

  if (s_offset + b_length > s_length) {

    unsigned part = s_length - s_offset;
    memcpy(b, s + s_offset, part);
    memcpy(b + part, s, b_length-part);
    return (b_length-part);

  } else {

    memcpy(b, s + s_offset, b_length);
    return (s_offset + b_length);

  }

}

void caspsr_decode_header (unsigned char * b, uint64_t *seq_no, uint64_t * ch_id)
{
  /*
  // *seq_no = (uint64_t) (b[0]<<56 | b[1]<<48 | b[2]<<40 | b[3]<<32 | b[4]<<24 | b[5]<<16 | b[6] << 8 | b[7]);
  *seq_no  = UINT64_C (0);
  *seq_no |= (uint64_t) b[0]<<56;
  *seq_no |= (uint64_t) b[1]<<48;
  *seq_no |= (uint64_t) b[2]<<40;
  *seq_no |= (uint64_t) b[3]<<32;
  *seq_no |= (uint64_t) (b[4]<<24 | b[5]<<16 | b[6] << 8 | b[7]);

  // *ch_id = (uint64_t) (b[8]<<56 | b[9]<<48 | b[10]<<40 | b[11]<<32 | b[12]<<24 | b[13]<<16 | b[15] << 8 | b[15]);
  *ch_id  = UINT64_C (0);
  *ch_id |= (uint64_t) b[8]<<56;
  *ch_id |= (uint64_t) b[9]<<48;
  *ch_id |= (uint64_t) b[10]<<40;
  *ch_id |= (uint64_t) b[11]<<32;
  *ch_id |= (uint64_t) b[12]<<24;
  *ch_id |= (uint64_t) (b[13]<<16 | b[15] << 8 | b[15]);
  */
  
  uint64_t tmp = 0; 
  unsigned i = 0;
  *seq_no = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[8 - i - 1];
    *seq_no |= (tmp << ((i & 7) << 3));
  }

  *ch_id = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    *ch_id |= (tmp << ((i & 7) << 3));
  }
}

void caspsr_encode_header (char *b, uint64_t seq_no, uint64_t ch_id)
{

  b[0] = (uint8_t) (seq_no>>56);
  b[1] = (uint8_t) (seq_no>>48);
  b[2] = (uint8_t) (seq_no>>40); 
  b[3] = (uint8_t) (seq_no>>32);
  b[4] = (uint8_t) (seq_no>>24);
  b[5] = (uint8_t) (seq_no>>16);
  b[6] = (uint8_t) (seq_no>>8); 
  b[7] = (uint8_t) (seq_no);

  b[8]  = (uint8_t) (ch_id>>56);
  b[9]  = (uint8_t) (ch_id>>48);
  b[10] = (uint8_t) (ch_id>>40); 
  b[11] = (uint8_t) (ch_id>>32);
  b[12] = (uint8_t) (ch_id>>24);
  b[13] = (uint8_t) (ch_id>>16);
  b[14] = (uint8_t) (ch_id>>8); 
  b[15] = (uint8_t) (ch_id);

  /*
  size_t i;
  unsigned char ch;
  for (i = 0; i < bytes; i++ )
  {
    ch = (num >> ((i & 7) << 3)) & 0xFF;
    if (type == LITTLE)
      arr[i] = ch;
    else if (type == BIG)
      arr[bytes - i - 1] = ch;
  }

  uint64ToByteArray (h->seq_no, (size_t) 8, (unsigned char*) b, (int) BIG);
  uint64ToByteArray (h->ch_id,  (size_t) 8, (unsigned char*) b+8, (int) BIG);
  */

}

/*
 *  Initiliazes data structures and parameters for the receiver 
 */
int caspsr_receiver_init(caspsr_receiver_t * ctx, unsigned i_dest, unsigned n_dest, unsigned n_packets,
                        unsigned n_append )
{

  multilog_t* log = ctx->log;

  //uint64_t buffer_size = UDP_DATA * UDP_NPACKS;

  /* intialize our data structures */
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();
  ctx->sock    = caspsr_init_sock();
  ctx->curr    = caspsr_init_data();
  ctx->next    = caspsr_init_data();

  /* Intialize the stopwatch timer */
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  /* 10 usecs between packets */
  ctx->timer_sleep = 10;
  ctx->timer = (StopWatch *) malloc(sizeof(StopWatch));

  /* xfer spacing */
  ctx->xfer_bytes   = (n_packets + n_append) * UDP_DATA;
  ctx->xfer_offset  = i_dest * n_packets;
  ctx->xfer_gap     = n_packets * n_dest;
  ctx->xfer_packets = n_packets + n_append; 

  caspsr_reset_receiver_t(ctx);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "xfer_bytes=%"PRIu64", xfer_offset=%"PRIu64", xfer_gap=%"PRIu64", xfer_packets=%"PRIu64"\n", ctx->xfer_bytes, ctx->xfer_offset, ctx->xfer_gap, ctx->xfer_packets);

  return 0;
  
}

void caspsr_reset_receiver_t(caspsr_receiver_t * ctx)
{
  ctx->xfer_s_seq  = 0;
  ctx->xfer_e_seq  = 0;
  ctx->xfer_count  = 0;
  ctx->xfer_ending = 0;

  caspsr_reset_data_t(ctx->curr);
  caspsr_reset_data_t(ctx->next);

  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);

  caspsr_close_sock(ctx);

}

int caspsr_receiver_dealloc(caspsr_receiver_t * ctx)
{
  caspsr_close_sock(ctx);
  caspsr_free_data(ctx->curr);
  caspsr_free_data(ctx->next);
  caspsr_free_sock(ctx->sock);
}


/* 
 * start an xfer, setting the expected start and end sequence numbers, incremement xfer count
 * and return the expected starting byte
 */
uint64_t caspsr_start_xfer(caspsr_receiver_t * ctx) 
{

  caspsr_zero_data(ctx->curr);
  caspsr_zero_data(ctx->next);

  /* offset based on this dest index */
  ctx->xfer_s_seq = ctx->xfer_offset;

  /* offset based on the number of xfers */
  ctx->xfer_s_seq += ctx->xfer_count * ctx->xfer_gap;
  ctx->xfer_e_seq  = ctx->xfer_s_seq + ctx->xfer_packets;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "xfer: %"PRIu64", s_seq=%"PRIu64", e_seq=%"PRIu64", bytes=%"PRIu64"\n", 
                 ctx->xfer_count, ctx->xfer_s_seq, ctx->xfer_e_seq, ctx->xfer_bytes);

  ctx->sod = 0;
  ctx->xfer_count++;
  ctx->xfer_ending = 0;

  return (ctx->xfer_s_seq * UDP_DATA);

}

/* 
 * receive size packets and return a pointer to the captured data, updating size
 * with the number of packets captured
 */
void * caspsr_xfer(caspsr_receiver_t * ctx, int64_t * size)
{

  ctx->got_enough = 0;

  /* pointer to start of socket buffer */
  unsigned char * arr;

  /* Flag for timeout */
  int timeout_ocurred = 0;

  /* How much data has actaully been received */
  uint64_t data_received = 0;

  /* sequence number of the packet */
  uint64_t seq_no = 0;

  /* ch_id of the packet */
  uint64_t ch_id = 0;

  /* for decoding sequence number */
  uint64_t tmp = 0;
  unsigned i = 0;

  /* switch the data buffer pointers */
  ctx->temp = ctx->curr;
  ctx->curr = ctx->next;
  ctx->next = ctx->temp;
  ctx->temp = 0;

  /* update the counters for the next buffer */
  ctx->next->count = 0;
  ctx->next->min = ctx->curr->max + 1;
  ctx->next->max = ctx->next->min + UDP_NPACKS -1;

  /* TODO check the performance hit of doing this */
  caspsr_zero_data(ctx->next);
  
  /* Determine the sequence number boundaries for curr and next buffers */
  int errsv;

  /* Assume we will be able to return a full buffer */
  *size = (UDP_DATA * UDP_NPACKS);

  /* Continue to receive packets */
  while (!ctx->got_enough) {

    /* If there is a packet in the buffer from the previous call */
    if (ctx->sock->have_packet) {

      ctx->sock->have_packet = 0;

    /* Get a packet from the socket */
    } else {

      ctx->sock->have_packet = 0;
      ctx->timer_count = 0;

      while (!ctx->sock->have_packet && !ctx->got_enough) {

        /* to control how often we check for packets, and accurate timeout */
        StopWatch_Start(ctx->timer);

        /* get a packet from the (non-blocking) socket */
        ctx->sock->got = recvfrom (ctx->sock->fd, ctx->sock->buffer, UDP_PAYLOAD, 0, NULL, NULL);

        /* if we successfully got a packet */
        if (ctx->sock->got == UDP_PAYLOAD) {

          ctx->sock->have_packet = 1;
          StopWatch_Stop(ctx->timer);

        /* if no packet at the socket */
        } else if (ctx->sock->got == -1) {

          errsv = errno;

          /* if no packet at the socket, busy sleep for 10us */
          if (errsv == EAGAIN) {
            ctx->timer_count += (uint64_t) ctx->timer_sleep;
            StopWatch_Delay(ctx->timer, ctx->timer_sleep);
  
          /* otherwise recvfrom failed, this is an error */
          } else {
            multilog(ctx->log, LOG_ERR, "caspsr_xfer: recvfrom failed: %s\n", strerror(errsv));
            ctx->got_enough = 1;
            StopWatch_Stop(ctx->timer);
          }

       /* we got a packet of the wrong size */
        } else {

          multilog(ctx->log, LOG_ERR, "caspsr_xfer: got %d, expected %d\n", ctx->sock->got, UDP_PAYLOAD);
          StopWatch_Stop(ctx->timer);

        }

        /* if the xfer is ending soon and the we have slept for 10ms, this should be an end of xfer, return */
        if (ctx->xfer_ending && ctx->timer_count >= 10000) {
          if (ctx->verbose) 
            multilog(ctx->log, LOG_INFO, "caspsr_xfer: xfer ending and no packet received for 10ms\n");
          ctx->got_enough = 1;
        }

        /* if we have a 100ms timeout [more severe problem] */
        if (ctx->timer_count >= 100000) {

          /* if we have SOD for the current xfer then this is a problem, warn and return */
          if (ctx->sod) {
            multilog(ctx->log, LOG_WARNING, "caspsr_xfer: SOD and 100ms packet timeout\n");
            ctx->got_enough = 1;

          /* else we are waiting the first packet of an xfer, this is ok */
          } else {

            /* if we have waited over 1s, then return so that the dada_pwc_main loop can 
             * handle other actions, such as setting UTC_START, or stopping */
            if (ctx->timer_count > 1000000) {
              if (ctx->verbose)
                multilog(ctx->log, LOG_INFO, "caspsr_xfer: !SOD and 1s packet timeout\n");
              ctx->got_enough = 1;
            }
          }
        }
      }
    }

    /* check that the packet is of the correct size */
    if (ctx->sock->have_packet && ctx->sock->got != UDP_PAYLOAD) {
      multilog(ctx->log, LOG_WARNING, "caspsr_xfer: Received a packet of unexpected size\n");
      ctx->got_enough = 1;
    }

    /* If we did get a packet within the timeout */
    if (!ctx->got_enough && ctx->sock->have_packet) {

      /* Decode the packets apsr specific header */
      arr = ctx->sock->buffer;
      seq_no = UINT64_C (0);
      for (i = 0; i < 8; i++ )
      {
        tmp = UINT64_C (0);
        tmp = arr[8 - i - 1];
        seq_no |= (tmp << ((i & 7) << 3));
      }

      seq_no /= 1024;

      /* If we are waiting for the first packet, and the sequence number is within 1000
       * of the starting value */
      if ((! ctx->sod) && (seq_no < ctx->xfer_s_seq + 1000)) {
        ctx->sod = 1;        

        /* update the min/max sequence numbers for the receiving buffers */
        ctx->curr->min = ctx->xfer_s_seq;
        ctx->curr->max = ctx->curr->min + UDP_NPACKS - 1;
        ctx->next->min = ctx->curr->max + 1;
        ctx->next->max = ctx->next->min + UDP_NPACKS - 1;

        if (ctx->verbose) 
          multilog(ctx->log, LOG_INFO, "START xfer=%d, seq_no=%"PRIu64" window=[%"PRIu64" - %"PRIu64"]\n", (ctx->xfer_count-1), seq_no, ctx->curr->min, ctx->curr->max);
      }

      /* if we have start of data */
      if (ctx->sod) {

        /* check for nearing the end of an xfer */
        if (ctx->xfer_e_seq - seq_no < 100) {
          if ((!ctx->xfer_ending) && (ctx->verbose)) {
            multilog(ctx->log, LOG_INFO, "caspsr_xfer: xfer %"PRIu64" ending soon xfer_e_seq=%"PRIu64", seq_no=%"PRIu64"\n", (ctx->xfer_count-1), ctx->xfer_e_seq, seq_no);
          } 
          ctx->xfer_ending = 1;
        }

        /* Increment statistics */
        ctx->packets->received++;
        ctx->packets->received_per_sec++;
        ctx->bytes->received += UDP_DATA;
        ctx->bytes->received_per_sec += UDP_DATA;

        /* we are going to process the packet we have */
        ctx->sock->have_packet = 0;

        if (seq_no < ctx->curr->min) {
          //fprintf(stderr, "Packet underflow %"PRIu64" < min (%"PRIu64")\n",
          //          seq_no, ctx->curr->min);

        } else if (seq_no <= ctx->curr->max) {
          memcpy( ctx->curr->buffer + (seq_no - ctx->curr->min) * UDP_DATA,
                  ctx->sock->buffer + 16,
                  UDP_DATA);
          ctx->curr->count++;

        } else if (seq_no <= ctx->next->max) {

          memcpy( ctx->next->buffer + (seq_no - ctx->next->min) * UDP_DATA,
                  ctx->sock->buffer + 16,
                  UDP_DATA);
          ctx->next->count++;

        } else {

          multilog(ctx->log, LOG_WARNING, "caspsr_xfer: not keeping up: "
                   "curr[%"PRIu64":%"PRIu64"] %5.2f%, next [%"PRIu64":%"PRIu64"]%5.2f%, "
                   "seq_no=%"PRIu64"\n", 
                   ctx->curr->min, ctx->curr->max, ((float) ctx->curr->count / (float) UDP_NPACKS)*100,
                   ctx->next->min, ctx->next->max, ((float) ctx->next->count / (float) UDP_NPACKS)*100,
                   seq_no);

          ctx->sock->have_packet = 1;
          ctx->got_enough = 1;
        }

        /* If we have filled the current buffer, then we can stop */
        if (ctx->curr->count >= UDP_NPACKS)
          ctx->got_enough = 1;

        /* if the next buffer is 50% full, then we can stop */
        if (ctx->next->count >= (UDP_NPACKS * 0.50)) {
          ctx->got_enough = 1;
          if (ctx->verbose) 
            multilog(ctx->log, LOG_WARNING, "caspsr_xfer: not keeping up: "
                     "curr[%"PRIu64":%"PRIu64"] %5.2f%, next [%"PRIu64":%"PRIu64"]%5.2f%, "
                     "seq_no=%"PRIu64" timer=%"PRIu64" us\n",
                     ctx->curr->min, ctx->curr->max, ((float) ctx->curr->count / (float) UDP_NPACKS)*100,
                     ctx->next->min, ctx->next->max, ((float) ctx->next->count / (float) UDP_NPACKS)*100,
                    seq_no, ctx->timer_count);

        }
      }
    }
  }

  if (ctx->xfer_ending) {
    *size = ctx->curr->count * UDP_DATA;
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "caspsr_xfer: EoXFER curr_count=%"PRIu64", next_count=%"PRIu64"\n",
               ctx->curr->count, ctx->next->count);

  } else {

    /* Some checks before returning */
    if (ctx->curr->count) {

      /* If the timeout ocurred, this is most likely due to end of data */
      if (ctx->timer_count >= 100000) {
        *size = ctx->curr->count * UDP_DATA;
        multilog(ctx->log, LOG_WARNING, "caspsr_xfer: Suspected EOD, timeout=%"PRIu64" us, returning %"PRIu64" bytes\n",ctx->timer_count, *size);

      } else { 

         /* if we didn't get a full buffers worth */
        if (ctx->curr->count < UDP_NPACKS) {

          ctx->packets->dropped += (UDP_NPACKS - ctx->curr->count);
          ctx->packets->dropped_per_sec += (UDP_NPACKS - ctx->curr->count);
          ctx->bytes->dropped += (UDP_DATA * (UDP_NPACKS - ctx->curr->count));
          ctx->bytes->dropped_per_sec += (UDP_DATA * (UDP_NPACKS - ctx->curr->count));

          multilog (ctx->log, LOG_WARNING, "dropped %"PRIu64" packets [total %"PRIu64"]\n",
                   (UDP_NPACKS - ctx->curr->count), ctx->packets->dropped);
        }
      }

    } else {
      if (ctx->verbose) 
        multilog(ctx->log, LOG_INFO, "caspsr_xfer: timeout=%"PRIu64" us, returning 0 bytes\n", ctx->timer_count);
      *size = 0;
    }
  }
  assert(ctx->curr->buffer != 0);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "caspsr_xfer: buffer_function returning %"PRIi64" bytes\n", *size);

  return (void *) ctx->curr->buffer;

}

/* 
 * stop the current xfer
 */
int caspsr_stop_xfer(caspsr_receiver_t * ctx)
{
  caspsr_reset_receiver_t(ctx);
  return 0;
}
