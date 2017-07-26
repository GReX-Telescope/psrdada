/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mopsr_udpdb_dual.h"
#include "mopsr_def.h"
#include "mopsr_util.h"

#define PFB_START_CHANNEL 204
#define STATE_IDLE 0
#define STATE_STARTING 1
#define STATE_RECORDING 2

int quit_threads = 0;
void stats_thread(void * arg);

void usage()
{
  fprintf (stdout,
     "mopsr_udpdb_dual [options]\n"
     " -b <core>     bind compuation to CPU core\n"
     " -c <port>     port to open for PWCC commands [default: %d]\n"
     " -d            run as daemon\n"
     " -m <id>       set PFB_ID on monitoring files\n"
     " -M <dir>      write monitoring files to dir\n"
     " -i <ip:port>  ip address and port to acquire\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -l <port>     multilog output port [default: %d]\n"
     " -s            single transfer only\n"
     " -v            verbose output\n",
     DADA_DEFAULT_BLOCK_KEY,
     MOPSR_DEFAULT_PWC_LOGPORT,
     MOPSR_DEFAULT_UDPDB_PORT);
}

int mopsr_udpdb_dual_init (mopsr_udpdb_dual_t * ctx) 
{
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_dual_init()\n");

  ctx->socks = (mopsr_sock_t **) malloc (sizeof(mopsr_sock_t *) * ctx->ninputs);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: initalizing socket to %"PRIu64" bytes\n", UDP_PAYLOAD);
  unsigned i;
  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i] = mopsr_init_sock();
  }

  ctx->enough = (char *) malloc(sizeof(char) * ctx->ninputs);

  // allocate required memory strucutres
  ctx->packets  = init_stats_t();
  ctx->bytes    = init_stats_t();
  ctx->pkt_size = UDP_DATA;

  multilog(ctx->log, LOG_INFO, "init: pkt_size=%d bytes\n", ctx->pkt_size);

  // setup a zeroed packet for fast memcpys
  ctx->zeroed_packet = malloc (ctx->pkt_size);
  memset (ctx->zeroed_packet, 0, ctx->pkt_size);

  size_t sock_buffer_size = 64 * 1024 * 1024;
  // open the socket for receiving UDP data
  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->fd = dada_udp_sock_in(ctx->log, ctx->interfaces[i], ctx->ports[i], ctx->verbose);
    if (ctx->socks[i]->fd < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to create udp socket %s:%d\n", ctx->interfaces[i], ctx->ports[i]);
      return -1;
    }

    // set the socket size to 64 MB
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "init: setting socket buffer size to %ld MB\n", sock_buffer_size / (1024*1024));
    dada_udp_sock_set_buffer_size (ctx->log, ctx->socks[i]->fd, ctx->verbose, sock_buffer_size);

    // set the socket to non-blocking
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "init: setting socket to non-blocking\n");
    sock_nonblock(ctx->socks[i]->fd);
  }

  pthread_mutex_init(&(ctx->mutex), NULL);

  mopsr_udpdb_dual_reset (ctx);

  return 0;
}

void mopsr_udpdb_dual_reset (mopsr_udpdb_dual_t * ctx)
{
  pthread_mutex_lock (&(ctx->mutex));

  ctx->capture_started = 0;
  ctx->start_byte = 0;
  ctx->end_byte   = 0;
  ctx->obs_bytes  = 0;
  ctx->n_sleeps   = 0;
  ctx->timeouts   = 0;
  ctx->last_block = 0;
  ctx->state      = STATE_IDLE;
  ctx->pkt_start  = 1;

  unsigned i=0;
  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->seq_offset = i * ctx->pkt_size;
    ctx->socks[i]->block_count = 0;
    ctx->socks[i]->have_packet = 0;
    ctx->socks[i]->prev_seq = -1;
    ctx->enough[i] = 0;
  }

  reset_stats_t(ctx->packets); 
  reset_stats_t(ctx->bytes); 

  pthread_mutex_unlock (&(ctx->mutex));
}

int mopsr_udpdb_dual_free (mopsr_udpdb_dual_t * ctx)
{
  unsigned i;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_dual_free()\n");

  if (ctx->interfaces)
  {
    for (i=0; i<ctx->ninputs; i++)
    {
      if (ctx->interfaces[i])
        free (ctx->interfaces[i]);
      ctx->interfaces[i] = 0;
    }
    free (ctx->interfaces);
  }
  ctx->interfaces = 0;

  if (ctx->ports)
    free (ctx->ports);
  ctx->ports = 0;

  if (ctx->enough)
    free (ctx->enough);
  ctx->enough = 0;

  if (ctx->mdir)
    free (ctx->mdir);
  ctx->mdir = 0;

  if (ctx->pfb_id)
    free (ctx->pfb_id);
  ctx->pfb_id = 0;

  if (ctx->zeroed_packet)
    free (ctx->zeroed_packet);
  ctx->zeroed_packet = 0;

  // since we are opening the socket in open, close here
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "free: closing UDP socket\n");

  for (i=0; i<ctx->ninputs; i++)
  {
    if (ctx->socks[i]->fd)
      close(ctx->socks[i]->fd);
    ctx->socks[i]->fd = 0;
    mopsr_free_sock (ctx->socks[i]);
  }

  pthread_mutex_lock (&(ctx->mutex));
  pthread_mutex_unlock (&(ctx->mutex));
  pthread_mutex_destroy (&(ctx->mutex));

  return 0;
}


/*! PWCM header valid function. Returns 1 if valid, 0 otherwise */
int mopsr_udpdb_dual_header_valid (dada_pwc_main_t* pwcm) 
{
  if (pwcm->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_dual_header_valid()\n");
  unsigned utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  // Check if the UTC_START is set in the header
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) 
    valid = 0;

  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "header_valid: UTC_START=%s\n", utc_buffer);

  // Check whether the UTC_START is set to UNKNOWN
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "header_valid: valid=%d\n", valid);
  return valid;
}


/*! PWCM error function. Called when buffer function returns 0 bytes
 * Returns 0=ok, 1=soft error, 2 hard error */
int mopsr_udpdb_dual_error (dada_pwc_main_t* pwcm) 
{
  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_dual_error()\n");
  int error = 0;
  
  // check if the header is valid
  if (mopsr_udpdb_dual_header_valid(pwcm)) 
    error = 0;
  else  
    error = 2;
  
  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_dual_error: error=%d\n", error);
  return error;
}

/*! PWCM start function, called before start of observation */
time_t mopsr_udpdb_dual_start (dada_pwc_main_t * pwcm, time_t start_utc)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_dual_start()\n");

  // reset statistics and volatile variables
  mopsr_udpdb_dual_reset (ctx);

  if (ascii_header_get (pwcm->header, "PKT_START", "%"PRIu64, &(ctx->pkt_start)) != 1)
  {
    multilog (pwcm->log, LOG_ERR, "start: failed to read PKT_START from header\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: PKT_START=%"PRIu64"\n", ctx->pkt_start);

  char utc_buffer[20];
  time_t start_time = 0;
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) != 1)
  {
    multilog (pwcm->log, LOG_INFO, "start: no UTC_START found in header, "
              "expecting set_utc_start command\n");
  }
  else
  {
    if (ctx->verbose)
      multilog (pwcm->log, LOG_INFO, "start: UTC_START=%s\n", utc_buffer);
    start_time = str2utctime(utc_buffer);
  }

  if (ascii_header_get (pwcm->header, "BYTES_PER_SECOND", "%"PRIu64"", &(ctx->bytes_per_second)) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read BYTES_PER_SECOND from header\n");
    return -1;
  }
  if (!ctx->bytes_per_second)
  {
    multilog (pwcm->log, LOG_WARNING, "start: BYTES_PER_SECOND == 0\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: BYTES_PER_SECOND=%"PRIu64"\n", ctx->bytes_per_second);

  // set the channel offset in the header
  if (ascii_header_set (pwcm->header, "CHAN_OFFSET", "%d", PFB_START_CHANNEL) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set CHAN_OFFSET=%d in header\n",
              PFB_START_CHANNEL);
  }

  char ant_tag[16];
  int real_ant_id;
  int i;

  for (i=0; i<16; i++)
  {
    sprintf (ant_tag, "ANT_ID_%d", i);
    real_ant_id = mopsr_get_hires_ant_number (i);
    if (ctx->verbose)
      multilog (pwcm->log, LOG_INFO, "setting %s=%d\n", ant_tag, real_ant_id);

    if (ascii_header_set (pwcm->header, ant_tag, "%d", real_ant_id) < 0)
    {
        sprintf (ant_tag, "ANT_ID_%d", i);
        multilog (pwcm->log, LOG_WARNING, "start: failed to set %s=%d in header\n",
                ant_tag, real_ant_id);
    }
  }

  // set ordering if data in header
  if (ascii_header_set (pwcm->header, "ORDER", "TFS") < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set ORDER=TFS in header\n");
  }

  // lock the socket mutex
  pthread_mutex_lock (&(ctx->mutex));

  // instruct the mon thread to not touch the socket
  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "start: disabling idle state\n");
  ctx->state = STATE_RECORDING;

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: clearing packets at socket\n");
  size_t cleared;
  for (i=0; i<ctx->ninputs; i++)
  {
    cleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
  }

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: recv_from (decoding packet)\n");
  // decode so we have nframe, etc
  for (i=0; i<ctx->ninputs; i++)
    mopsr_decode_red (ctx->socks[i]->buf, &(ctx->hdr), ctx->ports[i]);

  pthread_mutex_unlock (&(ctx->mutex));

  // centralised control will inform us of UTC_START, so return 0
  return start_time;
}

/*! pwcm buffer function [for header only] */
void * mopsr_udpdb_dual_recv (dada_pwc_main_t * pwcm, int64_t * size)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) pwcm->context;

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "mopsr_udpdb_dual_recv()\n");

  return pwcm->header;
}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t mopsr_udpdb_dual_recv_block (dada_pwc_main_t * pwcm, void * block, 
                                uint64_t block_size, uint64_t block_id)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_dual_recv_block()\n");

  uint64_t seq_byte, byte_offset, expected_seq, iseq;
  uint64_t pkts_caught = 0;
  unsigned i, iput;

  // number of output bytes per sequence number
  const unsigned bytes_per_seq = ctx->pkt_size * ctx->ninputs;

  // if we have started capture, then increment the pkt ranges
  if (ctx->capture_started)
  {
    ctx->start_pkt  = ctx->end_pkt + 1;
    ctx->end_pkt    = ctx->start_pkt + (ctx->packets_per_buffer - 1);
    if (ctx->verbose)
      multilog(pwcm->log, LOG_INFO, "recv_block: [%"PRIu64" - %"PRIu64"]\n", ctx->start_pkt, ctx->end_pkt);
  }
  // otherwise clear any buffered packets at the socket
  else
  {
    size_t ncleared;
    for (i=0; i<ctx->ninputs; i++)
      ncleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
  }

  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->block_count = 0;
    ctx->enough[i] = 0;
  }
  char enough = 0;

  // now begin capture loop
  while (!enough)
  {
    for (i=0; i<ctx->ninputs; i++)
    {
      // try to get a packet from the socks[i]
      if (!ctx->socks[i]->have_packet && !enough)
      {
        ctx->socks[i]->got = recvfrom (ctx->socks[i]->fd, ctx->socks[i]->buf, UDP_PAYLOAD, 0, NULL, NULL);

        if (ctx->socks[i]->got == UDP_PAYLOAD)
        {
          ctx->socks[i]->have_packet = 1;
        }
        else if (ctx->socks[i]->got == -1)
        {
          if (errno == EAGAIN)
          {
            ctx->n_sleeps++;
          }
          else
          {
            multilog (pwcm->log, LOG_ERR, "recv_block: recvfrom failed %s\n", strerror(errno));
            enough = 1;
          }
        }
        else // we received a packet of the WRONG size, ignore it
        {
          multilog (pwcm->log, LOG_ERR, "recv_block: received %d bytes, expected %d\n", ctx->socks[i]->got, UDP_PAYLOAD);
        }
      }

      // we now have a packet
      if (!enough && ctx->socks[i]->have_packet)
      {
        // just decode the sequence number
        mopsr_decode_seq (ctx->socks[i]->buf, &(ctx->hdr));

        if (ctx->verbose > 2)
          multilog (pwcm->log, LOG_INFO, "recv_block: seq_no=%"PRIu64"\n", ctx->hdr.seq_no);

        // wait for start packet on all inputs
        if ((ctx->capture_started < ctx->ninputs) &&
            (ctx->hdr.seq_no < (ctx->pkt_start + 1e6)) &&
            (ctx->hdr.seq_no >= ctx->pkt_start))
        {
          ctx->socks[i]->prev_seq  = ctx->pkt_start - 1;
          if (ctx->capture_started == 0)
          {
            ctx->start_pkt = ctx->pkt_start;
            ctx->end_pkt   = ctx->start_pkt + (ctx->packets_per_buffer - 1);
          }

          if (ctx->verbose)
          {
            mopsr_decode_red (ctx->socks[i]->buf, &(ctx->hdr), ctx->ports[i]);
            multilog (pwcm->log, LOG_INFO, "recv_block: START [%"PRIu64" - "
                      "%"PRIu64"] seq_no=%"PRIu64"\n", ctx->start_pkt,
                      ctx->end_pkt, ctx->hdr.seq_no);
          }
          ctx->capture_started++;
        }
     
        if (ctx->socks[i]->prev_seq >= 0)
        {
          // expected first byte based on last packet
          expected_seq = ctx->socks[i]->prev_seq + 1;

          const uint64_t zeroed_seq = ctx->hdr.seq_no > ctx->end_pkt ? (ctx->end_pkt + 1) : ctx->hdr.seq_no;

          if (zeroed_seq != expected_seq)
            multilog (pwcm->log, LOG_INFO, "recv_block: [%d] zeroing from seq %"PRIu64" to %"PRIu64"\n", i, expected_seq, zeroed_seq);

          // handle any dropped packets up to either seq_no or end of block
          for (iseq=expected_seq; iseq < zeroed_seq; iseq++)
          {
            byte_offset = (iseq - ctx->start_pkt) * bytes_per_seq + ctx->socks[i]->seq_offset;
            if (byte_offset + ctx->pkt_size > block_size)
            {
              multilog (pwcm->log, LOG_INFO, "recv_block: trying to zero too much data seq_offset=%"PRIu64" iseq=%"PRIu64" start_pkt=%"PRIu64" byte_offset=%"PRIu64"\n", (iseq - ctx->start_pkt), iseq, ctx->start_pkt, byte_offset);
              byte_offset = 0;
            }
            // TODO revert
            //memcpy (block + byte_offset, ctx->zeroed_packet, ctx->pkt_size);
          }

          // packet belonged in a previous block [to late - oh well]
          if (ctx->hdr.seq_no < ctx->start_pkt)
          {
            multilog (pwcm->log, LOG_INFO, "recv_block: [%d] seq_no=%"PRIu64", seq_byte [%"
                      PRIu64"] < start_byte [%"PRIu64"]\n", i, ctx->hdr.seq_no, seq_byte, ctx->start_byte);

            // we are going to ignore (& consume) this packet
            ctx->socks[i]->have_packet = 0;
          }
          // packet resides in current block
          else if (ctx->hdr.seq_no <= ctx->end_pkt)
          {
            ctx->socks[i]->have_packet = 0;
            byte_offset = (ctx->hdr.seq_no - ctx->start_pkt) * bytes_per_seq + ctx->socks[i]->seq_offset;
            if (byte_offset + ctx->pkt_size > block_size)
            {
              multilog (pwcm->log, LOG_INFO, "recv_block: trying to copy out of bounds! %lu = (%lu - %ld) * %u + %lu\n", byte_offset, ctx->hdr.seq_no, ctx->start_pkt, bytes_per_seq, ctx->socks[i]->seq_offset);
              byte_offset = 0;
            }

            memcpy (block + byte_offset, ctx->socks[i]->buf + UDP_HEADER, ctx->pkt_size);
            ctx->socks[i]->prev_seq = ctx->hdr.seq_no;
            pkts_caught++;
          }
          // packet resides in a future block
          else
          {
            // we have enough data from this input
            ctx->enough[i] = 1;

            // test we have enough from all inputs
            enough = 1;
            for (iput=0; iput<ctx->ninputs; iput++)
            {
              if (ctx->verbose)
                multilog (pwcm->log, LOG_INFO, "recv_block: got enough on %d: ctx->socks[%d]->enough=%d\n", i, iput, ctx->enough[iput]);
              if (!ctx->enough[iput])
              {
                enough = 0;
              }
            }

            // mark the prev seq number for this input to the end of the buffer
            ctx->socks[i]->prev_seq = ctx->end_pkt;
          }
        }
        // while we are waiting to start just write the packet to memory soas to busy sleep
        else
        {
          byte_offset = 0;
          memcpy (block + ctx->socks[i]->seq_offset, ctx->socks[i]->buf + UDP_HEADER, ctx->pkt_size);
          ctx->socks[i]->have_packet = 0;
        }
      }
    }
  }

  // update the statistics
  if (pkts_caught)
  {
    ctx->packets->received += pkts_caught;
    ctx->bytes->received   += (pkts_caught * ctx->pkt_size);
      
    uint64_t dropped = (ctx->packets_per_buffer * ctx->ninputs) - pkts_caught;
    if (dropped)
    {
      multilog (pwcm->log, LOG_WARNING, "dropped %"PRIu64" packets %5.2f percent\n", dropped, 100.0 * (float) dropped / (float) (ctx->packets_per_buffer * ctx->ninputs));
      ctx->packets->dropped += dropped;
      ctx->bytes->dropped += (dropped * ctx->pkt_size);
    }
  }

  // save a pointer to the last full block of data for the monitor thread
  if (ctx->capture_started)
  {
    ctx->last_block = block;
    if (ctx->verbose)
      multilog (pwcm->log, LOG_INFO, "recv_block: setting last_block=%p\n", (void *) ctx->last_block);
  }

  return (int64_t) block_size;
}


/*! PWCM stop function, called at end of observation */
int mopsr_udpdb_dual_stop (dada_pwc_main_t* pwcm)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog (pwcm->log, LOG_INFO, "mopsr_udpdb_dual_stop()\n");

  ctx->capture_started = 0;

  pthread_mutex_lock (&(ctx->mutex));
  ctx->state = STATE_IDLE;
  pthread_mutex_unlock (&(ctx->mutex));

  ctx->last_block = 0;

  multilog (pwcm->log, LOG_INFO, "received %"PRIu64" bytes\n", ctx->bytes->received);
  return 0;
}

int mopsr_udpdb_dual_parse_input (mopsr_udpdb_dual_t * ctx, char * optarg)
{
  const char *sep = ":";
  char * saveptr;
  char * str;

  // incremenet the number of inputs being processed
  ctx->ninputs++;
  ctx->interfaces = (char **) realloc (ctx->interfaces, ctx->ninputs * sizeof(char *));
  ctx->ports      = (int *)   realloc (ctx->ports, ctx->ninputs * sizeof(int));

  // parse the IP address from the argument
  str = strtok_r(optarg, sep, &saveptr);
  if (str== NULL)
  {
    fprintf(stderr, "parse_input: misformatted input option for ip\n");
    return -1;
  }
  ctx->interfaces[ctx->ninputs-1] = (char *) malloc ((strlen(str)+1) * sizeof(char));
  strcpy(ctx->interfaces[ctx->ninputs-1], str);

  // parse the port address from the argument
  str = strtok_r(NULL, sep, &saveptr);
  if (str== NULL)
  {
    fprintf(stderr, "parse_input: misformatted input option for port\n");
    return -1;
  }
  if (sscanf(str, "%d", &(ctx->ports[ctx->ninputs-1])) != 1)
  {
    fprintf(stderr, "parse_input: misformatted input option for port\n");
    return -1;
  }

  return 0;
}


/*
 *  Main. 
 */
int main (int argc, char **argv)
{

  /* IB DB configuration */
  mopsr_udpdb_dual_t ctx;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA PWCM */
  dada_pwc_main_t* pwcm = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* PWC control port */
  int control_port = MOPSR_DEFAULT_PWC_PORT;

  /* Multilog LOG port */
  int log_port = MOPSR_DEFAULT_PWC_LOGPORT;

  /* Quit flag */
  char quit = 0;

  /* CPU core on which to operate */
  int recv_core = -1;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* Monitoring of UDP data */
  pthread_t mon_thread_id;

  int arg = 0;

  const char *sep = ":";
  char * saveptr;
  char * str;

  ctx.ninputs = 0;
  ctx.interfaces = (char **) malloc (1 * sizeof(char *));
  ctx.ports      = (int *) malloc (1 * sizeof(int));

  ctx.mdir = 0;
  ctx.verbose = 0;
  ctx.pfb_id = malloc (sizeof(char) * 16);
  sprintf (ctx.pfb_id, "%s", "XX");

  while ((arg=getopt(argc,argv,"b:c:dhi:k:l:m:M:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        if (optarg)
        {
          recv_core = atoi(optarg);
          break;
        }
        else
        {
          fprintf (stderr, "ERROR: not cpu ID specified\n");
          usage();
          return EXIT_FAILURE;
        }
            
      case 'c':
        if (optarg)
        {
          control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "ERROR: no control port specified\n");
          usage();
          return EXIT_FAILURE;
        }

      case 'd':
        daemon=1;
        break;

      case 'h':
        usage();
        return 0;

      case 'i':
        if (optarg)
        {
          if (mopsr_udpdb_dual_parse_input(&ctx, optarg ) < 0)
          {
            fprintf(stderr, "ERROR: failed to parse input from %s\n", optarg);
            usage();
            return (EXIT_FAILURE);
          }
        }
        else
        {
          fprintf (stderr, "ERROR: -i option requires an argument\n");
          usage();
          return EXIT_FAILURE;
        }
        break;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr,"ERROR: could not parse key from %s\n",optarg);
          return EXIT_FAILURE;
        }
        break;

      case 'l':
        if (optarg) {
          log_port = atoi(optarg);
          break;
        } else {
          fprintf (stderr, "ERROR: -l option requires an argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 'm':
        if (optarg) {
          sprintf(ctx.pfb_id, "%s", optarg);
          break;
        } else {
          fprintf (stderr,"mopsr_udpdb: -m requires argument\n");
          usage();
          mopsr_udpdb_dual_free (&ctx);
          return EXIT_FAILURE;
        }

      case 'M':
        if (optarg) {
          ctx.mdir = (char *) malloc (sizeof (char) * strlen(optarg)+1);
          strcpy (ctx.mdir , optarg);
          break;
        } else {
          fprintf (stderr, "ERROR: -m option requires an argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 'v':
        ctx.verbose++;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  if (ctx.ninputs == 0)
  {
    fprintf (stderr, "ERROR: at least one IP/PORT required\n");
    return EXIT_FAILURE;
  }

  // do not use the syslog facility
  log = multilog_open ("mopsr_udpdb_dual", 0);

  if (daemon) 
  {
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
  pwcm->start_function        = mopsr_udpdb_dual_start;
  pwcm->buffer_function       = mopsr_udpdb_dual_recv;
  pwcm->block_function        = mopsr_udpdb_dual_recv_block;
  pwcm->stop_function         = mopsr_udpdb_dual_stop;
  pwcm->header_valid_function = mopsr_udpdb_dual_header_valid;
  pwcm->context               = &ctx;
  pwcm->verbose               = ctx.verbose;

  ctx.log = log;
  ctx.core = recv_core;

  hdu = dada_hdu_create (log);

  dada_hdu_set_key (hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  pwcm->data_block            = hdu->data_block;
  pwcm->header_block          = hdu->header_block;

  // get the block size of the data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // ensure the block size is a multiple of the packet size
  if (block_size % UDP_DATA != 0)
  {
    fprintf (stderr, "mopsr_udpdb_dual: block size [%"PRIu64"] was not a "
                     "multiple of the UDP payload size [%d\n", block_size,
                     UDP_DATA);
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }


  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_dual: creating dada pwc control interface\n");

  pwcm->pwc = dada_pwc_create();
  pwcm->pwc->port = control_port;

  if (ctx.core >= 0)
    if (dada_bind_thread_to_core(ctx.core) < 0)
      multilog(log, LOG_WARNING, "mopsr_udpdb_dual: failed to bind to core %d\n", ctx.core);

  mopsr_udpdb_dual_init (&ctx);

  // packet size define in the above
  ctx.packets_per_buffer = block_size / (ctx.pkt_size * ctx.ninputs);

  if (ctx.verbose)
    multilog(log, LOG_INFO, "mopsr_udpdb_dual: starting mon_thread()\n");
  int rval = pthread_create (&mon_thread_id, 0, (void *) mon_thread, (void *) &ctx);
  if (rval != 0)
  {
    multilog(log, LOG_ERR, "mopsr_udpdb_dual: could not create monitor thread: %s\n", strerror(rval));
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_dual: creating dada server\n");
  if (dada_pwc_serve (pwcm->pwc) < 0)
  {
    fprintf (stderr, "mopsr_udpdb_dual: could not start server\n");
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_dual: entering PWC main loop\n");

  if (dada_pwc_main (pwcm) < 0)
  {
    fprintf (stderr, "mopsr_udpdb_dual: error in PWC main loop\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_dual: destroying pwc\n");
  dada_pwc_destroy (pwcm->pwc);

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_dual: destroying pwc main\n");
  dada_pwc_main_destroy (pwcm);

  if (ctx.verbose)
    multilog(log, LOG_INFO, "mopsr_udpdb_dual: joining mon_thread\n");
  quit_threads = 1;
  void * result;
  //pthread_join (mon_thread_id, &result);

  if (ctx.verbose)
    multilog(log, LOG_INFO, "mopsr_udpdb_dual: freeing resources\n");

  mopsr_udpdb_dual_free (&ctx);
 
  return EXIT_SUCCESS;
}


/*
 * Thread to provide raw access to a recent packet over the socket
 */
void mon_thread (void * arg)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) arg;

  if (ctx->core >= 0)
    if (dada_bind_thread_to_core(ctx->core+1) < 0)
      multilog(ctx->log, LOG_WARNING, "mon_thread: failed to bind to core %d\n", ctx->core+1);

  char local_time[32];
  char mon_file[512];

  int fd;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  size_t nwrote, ncleared;
  time_t now, now_plus;
  time_t sleep_secs = 5;

  // this is 10.24 * 64 == 655.36 us
  unsigned int npackets = 64;
  unsigned int ipacket = 0;
  void * in, * ou;

  // these are constant
  const unsigned nant = 8;
  const unsigned ndim = 2;
  const unsigned chan_span = nant * ndim;

  const size_t ou_stride = ctx->ninputs * chan_span;
  const size_t in_stride = chan_span;
  unsigned int iframe, ichan;

  char * packets = (char *) malloc ((UDP_DATA * npackets * ctx->ninputs) + UDP_HEADER);
  size_t out_size = UDP_HEADER + (npackets * ctx->ninputs * UDP_DATA);
  unsigned i=0;

  mopsr_hdr_t hdr;

  while (!quit_threads)
  {
    if (ctx->state == STATE_IDLE)
    {
      // if in the idle state, lock the mutex as the mon thread
      // will have control of the socket
      pthread_mutex_lock (&(ctx->mutex));

      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "mon_thread: clearing buffered packets\n");
      for (i=0; i<ctx->ninputs; i++)
      {
        ncleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
      }
      usleep (1000);

      ipacket = 0;
      while (!quit_threads && ipacket < npackets)
      {
        for (i=0; i<ctx->ninputs; i++)
        {
          ctx->socks[i]->got = 0;
          while  (!quit_threads && ctx->socks[i]->got != UDP_PAYLOAD)
          {
            //multilog (ctx->log, LOG_INFO, "mon_thread: recvfrom on [%d]\n", i);
            ctx->socks[i]->got = recvfrom (ctx->socks[i]->fd, ctx->socks[i]->buf, UDP_PAYLOAD, 0, NULL, NULL);
            if (ctx->socks[i]->got == UDP_PAYLOAD)
            {
              mopsr_decode_red (ctx->socks[i]->buf, &hdr, ctx->ports[i]);

              // since ordering is TFPS, we need to interleave data from inputs
              in = (void *) (ctx->socks[i]->buf + UDP_HEADER);
              ou = (void *) (packets + UDP_HEADER + (ipacket * ctx->ninputs * UDP_DATA) + (i * chan_span));
              for (iframe=0; iframe < hdr.nframe; iframe++)
              {
                for (ichan=0; ichan < hdr.nchan; ichan++)
                {
                  memcpy (ou, in, chan_span);
                  ou += ou_stride;
                  in += in_stride;
                }
              }
#ifdef _DEBUG
              if ((ctx->socks[i]->prev_seq != 0) && (hdr.seq_no != ctx->socks[i]->prev_seq + 1) && ctx->verbose)
              {
                multilog(ctx->log, LOG_INFO, "mon_thread: hdr.seq=%"PRIu64" prev_seq=%"PRIi64" [%"PRIu64"]\n", 
                         hdr.seq_no, ctx->socks[i]->prev_seq, hdr.seq_no - ctx->socks[i]->prev_seq);
              }
              ctx->socks[i]->prev_seq = hdr.seq_no;
#endif
            }
            else if ((ctx->socks[i]->got == -1) && (errno == EAGAIN))
            {
              // packet not at socket due to clear + nonblock
              usleep (10000);
            }
            else
            {
              // more serious!
              multilog(ctx->log, LOG_INFO, "mon_thread: init & got[%d] != UDP_PAYLOAD[%d]\n",
                       ctx->socks[i]->got, UDP_PAYLOAD);
              sleep (1);
            }
          }
        }
        ipacket++;
      }

      // release control of the socket
      pthread_mutex_unlock (&(ctx->mutex));

      // udpate header in packets array
      hdr.nframe = npackets;
    }
    else
    {
      // decode current header from sock[0]
      mopsr_decode_red (ctx->socks[0]->buf, &hdr, ctx->ports[0]);

      // since ordering is TFPS, we need to interleave data from inputs
      for (i=0; i<ctx->ninputs; i++)
      {
        // if reading blocked TFS data from the last block
        if (ctx->last_block)
        {
          in = (void *) (ctx->last_block + ctx->socks[i]->seq_offset);
          hdr.nframe = npackets;
        }
        else
        {
          in = (void *) (ctx->socks[i]->buf + UDP_HEADER);
          hdr.nframe = 1;
        }

        ou = (void *) (packets + UDP_HEADER + (i * chan_span));
      
        for (iframe=0; iframe < hdr.nframe; iframe++)
        {
          for (ichan=0; ichan < hdr.nchan; ichan++)
          {
            memcpy (ou, in, chan_span);
            ou += ou_stride;
            in += in_stride;
          }
        }
      }
    }

    hdr.nant *= ctx->ninputs;
    mopsr_encode (packets, &hdr);
    out_size = UDP_HEADER + (hdr.nframe * hdr.nant * hdr.nchan * hdr.nbit * 2) / 8;

    if (out_size)
    {
      now = time(0);
      strftime (local_time, 32, DADA_TIMESTR, localtime(&now));
      sprintf (mon_file, "%s/%s.%s.dump", ctx->mdir, local_time, ctx->pfb_id);
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "mon_thread: creating %s\n", mon_file);
      fd = open(mon_file, flags, perms);
      if (fd < 0)
      {
        multilog (ctx->log, LOG_ERR, "mon_thread: failed to open '%s' for writing: %s\n", mon_file, strerror(errno));
      }
      else
      {
        nwrote = write (fd, packets, out_size);
        close (fd);
      }
    }

    now_plus = time(0);
    while (!quit_threads && now + sleep_secs >= now_plus)
    {
      usleep (500000);
      now_plus = time(0);
    }
  }

  multilog (ctx->log, LOG_INFO, "mon_thread: free(packets)\n");
  free (packets);
}
