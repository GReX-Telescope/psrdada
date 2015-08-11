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

#include "mopsr_udpdb_selants.h"
#include "mopsr_def.h"
#include "mopsr_util.h"

#define MOPSR_PFB_NANT 16
#define MOPSR_PFB_START_CHAN 25
#define MOPSR_PFB_END_CHAN 64

#define MOPSR_UDPDB_BLOCKING

#define STATE_IDLE 0
#define STATE_STARTING 1
#define STATE_RECORDING 2

int quit_threads = 0;
void stats_thread(void * arg);

void usage()
{
  fprintf (stdout,
     "mopsr_udpdb_selants_sel_ants [options]\n"
     " -a <ant>      antenna to select [can be used multiple times]\n"
     " -b <core>     bind compuation to CPU core\n"
     " -c <port>     port to open for PWCC commands [default: %d]\n"
     " -C <schan>    start channel to select [%d-%d]\n"
     " -D <echan>    end channel to select [%d-%d]\n"
     " -d            run as daemon\n"
     " -m <id>       set PFB_ID on monitoring files\n"
     " -M <dir>      write monitoring files to dir\n"
     " -i <ipaddr>   listen for UDP only on this interface [default all]\n"
     " -k <key>      hexadecimal shared memory key  [default: %x]\n"
     " -l <port>     multilog output port [default: %d]\n"
     " -p <port>     port on which to receive UDP packets[default: %d]\n"
     " -s            single transfer only\n"
     " -v            verbose output\n"
     " -0            insert 0's into data stream instead of UDP data\n",
     MOPSR_DEFAULT_PWC_PORT,
     MOPSR_PFB_START_CHAN,
     MOPSR_PFB_END_CHAN-1,
     MOPSR_PFB_START_CHAN+1,
     MOPSR_PFB_END_CHAN,
     DADA_DEFAULT_BLOCK_KEY,
     MOPSR_DEFAULT_PWC_LOGPORT,
     MOPSR_DEFAULT_UDPDB_PORT);
}

int mopsr_udpdb_selants_init (mopsr_udpdb_selants_t * ctx) 
{
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_selants_init()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: initalizing socket to %"PRIu64" bytes\n", UDP_PAYLOAD);
  ctx->sock = mopsr_init_sock();

  // allocate required memory strucutres
  ctx->packets  = init_stats_t();
  ctx->bytes    = init_stats_t();
  ctx->pkt_size = UDP_DATA / ctx->decifactor;

  ctx->buf = (int16_t *) malloc (ctx->pkt_size);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: decimated pkt_size=%d bytes\n", ctx->pkt_size);

  // setup a zeroed packet for fast memcpys
  ctx->zeroed_packet = malloc (ctx->pkt_size);
  memset (ctx->zeroed_packet, 0, ctx->pkt_size);

  // open the socket for receiving UDP data
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0)
  {
    multilog (ctx->log, LOG_ERR, "failed to create udp socket %s:%d\n", ctx->interface, ctx->port);
    return -1;
  }

  // set the socket size to 64 MB
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting socket buffer size to 128 MB\n");
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, 134217728);

#ifndef MOPSR_UDPDB_BLOCKING
  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting socket to non-blocking\n");
  if (sock_nonblock(ctx->sock->fd) < 0)
    multilog(ctx->log, LOG_ERR, "init: sock_nonblock failed: %s\n", strerror(errno));
#endif

  pthread_mutex_init(&(ctx->mutex), NULL);

  mopsr_udpdb_selants_reset (ctx);

  return 0;
}

void mopsr_udpdb_selants_reset (mopsr_udpdb_selants_t * ctx)
{
  pthread_mutex_lock (&(ctx->mutex));

  ctx->capture_started = 0;
  ctx->got_enough = 0;
  ctx->obs_bytes  = 0;
  ctx->n_sleeps   = 0;
  ctx->timeouts   = 0;
  ctx->state      = STATE_IDLE;
  ctx->pkt_start  = 1;

  ctx->sock->have_packet = 0;

  reset_stats_t(ctx->packets); 
  reset_stats_t(ctx->bytes); 

  pthread_mutex_unlock (&(ctx->mutex));
}

int mopsr_udpdb_selants_free (mopsr_udpdb_selants_t * ctx)
{
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_selants_free()\n");

  multilog(ctx->log, LOG_INFO, "free: freeing interface\n");
  if (ctx->interface)
    free (ctx->interface);
  ctx->interface = 0;

  multilog(ctx->log, LOG_INFO, "free: freeing mdir\n");
  if (ctx->mdir)
    free (ctx->mdir);
  ctx->mdir = 0;

  multilog(ctx->log, LOG_INFO, "free: freeing pfb_id\n");
  if (ctx->pfb_id)
    free (ctx->pfb_id);
  ctx->pfb_id = 0;

  multilog(ctx->log, LOG_INFO, "free: freeing zeroed_packet\n");
  if (ctx->zeroed_packet)
    free (ctx->zeroed_packet);
  ctx->zeroed_packet = 0;

  // since we are opening the socket in open, close here
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "free: closing UDP socket\n");

  if (ctx->sock)
  {
    if (ctx->sock->fd)
      close(ctx->sock->fd);
    ctx->sock->fd = 0;
    mopsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;

  if (ctx->buf)
    free (ctx->buf);
  ctx->buf = 0;

  pthread_mutex_lock (&(ctx->mutex));
  pthread_mutex_unlock (&(ctx->mutex));
  pthread_mutex_destroy (&(ctx->mutex));

  return 0;
}


/*! PWCM header valid function. Returns 1 if valid, 0 otherwise */
int mopsr_udpdb_selants_header_valid (dada_pwc_main_t* pwcm) 
{
  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_header_valid()\n");
  unsigned utc_size = 64;
  char utc_buffer[utc_size];
  int valid = 1;

  // Check if the UTC_START is set in the header
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) 
    valid = 0;

  // Check whether the UTC_START is set to UNKNOWN
  if (strcmp(utc_buffer,"UNKNOWN") == 0)
    valid = 0;

  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_header_valid: valid=%d\n", valid);
  return valid;
}


/*! PWCM error function. Called when buffer function returns 0 bytes
 * Returns 0=ok, 1=soft error, 2 hard error */
int mopsr_udpdb_selants_error (dada_pwc_main_t* pwcm) 
{
  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_error()\n");
  int error = 0;
  
  // check if the header is valid
  if (mopsr_udpdb_selants_header_valid(pwcm)) 
    error = 0;
  else  
    error = 2;
  
  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_error: error=%d\n", error);
  return error;
}

/*! PWCM start function, called before start of observation */
time_t mopsr_udpdb_selants_start (dada_pwc_main_t * pwcm, time_t start_utc)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_start()\n");

  // reset statistics and volatile variables
  mopsr_udpdb_selants_reset (ctx);
    
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
    multilog (pwcm->log, LOG_INFO, "start: no UTC_START found in header, expecting set_utc_start command\n");
  }
  else
  {
    if (ctx->verbose)
      multilog (pwcm->log, LOG_INFO, "start: UTC_START=%s\n", utc_buffer);
    start_time = str2utctime(utc_buffer);
  }

  int header_nant;
  if (ascii_header_get (pwcm->header, "NANT", "%d", &header_nant) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read NANT from header\n");
  }
  if (header_nant != ctx->old_nant)
  {
    multilog (pwcm->log, LOG_ERR, "start: header NANT[%d] != PFB NANT[%d]\n",
              header_nant, MOPSR_PFB_NANT);
    return -1;
  }

  int header_nchan;
  if (ascii_header_get (pwcm->header, "NCHAN", "%d", &header_nchan) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read NCHAN from header\n");
  }
  if (header_nchan != ctx->old_nchan)
  {
    multilog (pwcm->log, LOG_ERR, "start: header NCHAN[%d] != PFB NCHAN[%d]\n",
              header_nchan, ctx->old_nchan);
    return -1;
  }

  // centre freq will change if we select channels
  float old_freq, new_freq;
  if (ascii_header_get (pwcm->header, "FREQ", "%f", &old_freq) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read FREQ from header\n");
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: old FREQ=%f\n", old_freq);

  // bandwidth will chan if we select channels
  float old_bw, new_bw;
  if (ascii_header_get (pwcm->header, "BW", "%f", &old_bw) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read BW from header\n");
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: old BW=%f\n", old_bw);

  uint64_t old_bytes_per_second, new_bytes_per_second;

  if (ascii_header_get (pwcm->header, "BYTES_PER_SECOND", "%"PRIu64"", &old_bytes_per_second) != 1)
  {     
    multilog (pwcm->log, LOG_WARNING, "start: failed to read BYTES_PER_SECOND from header\n");
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: old BYTES_PER_SECOND=%"PRIu64"\n", old_bytes_per_second);

  uint64_t old_resolution, new_resolution;
  if (ascii_header_get (pwcm->header, "RESOLUTION", "%"PRIu64"", &old_resolution) != 1)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to read RESOLUTION from header\n");
  }
  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: old RESOLUTION=%"PRIu64"\n", old_resolution);

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting NANT=%d in header\n", ctx->new_nant);
  if (ascii_header_set (pwcm->header, "NANT", "%d", ctx->new_nant) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set NANT=%d in header\n", ctx->new_nant);
  }

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting NCHAN=%d in header\n", ctx->new_nchan);
  if (ascii_header_set (pwcm->header, "NCHAN", "%d", ctx->new_nchan) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set NCHAN=%d in header\n", ctx->new_nchan);
  }

  new_bytes_per_second = old_bytes_per_second / ctx->decifactor;
  new_resolution = old_resolution / ctx->decifactor;

  // calculate new centre frequencies
  float freq_low = old_freq - (old_bw / 2);
  float chan_bw  = old_bw / ctx->old_nchan;

  if (ctx->verbose)
  {
    multilog (pwcm->log, LOG_INFO, "start: freq_low=%f\n", freq_low);
    multilog (pwcm->log, LOG_INFO, "start: chan_bw=%f\n", chan_bw);
  }

  int delta_schan = ctx->schan - MOPSR_PFB_START_CHAN;
  int delta_echan = ctx->echan - MOPSR_PFB_START_CHAN;

  float new_start_chan_cfreq = freq_low + (delta_schan * chan_bw) + (chan_bw / 2);
  float new_end_chan_cfreq   = freq_low + (delta_echan * chan_bw) + (chan_bw / 2);

  if (ctx->verbose)
  {
    multilog (pwcm->log, LOG_INFO, "start: new_start_chan_cfreq=%f\n", new_start_chan_cfreq);
    multilog (pwcm->log, LOG_INFO, "start: new_end_chan_cfreq=%f\n", new_end_chan_cfreq);
  }

  new_bw = (new_end_chan_cfreq - new_start_chan_cfreq)  + chan_bw;
  new_freq = (new_start_chan_cfreq - (chan_bw / 2)) + (new_bw / 2);

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting BW=%f in header\n", new_bw);
  if (ascii_header_set (pwcm->header, "BW", "%f", new_bw) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set BW=%f in header\n", new_bw);
  }

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting FREQ=%f in header\n", new_freq);
  if (ascii_header_set (pwcm->header, "FREQ", "%f", new_freq) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set FREQ=%f in header\n", new_freq);
  }

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  if (ascii_header_set (pwcm->header, "BYTES_PER_SECOND", "%"PRIu64"", new_bytes_per_second) < 0)
  {       
    multilog (pwcm->log, LOG_WARNING, "start: failed to set BYTES_PER_SECOND=%"PRIu64" in header\n", new_bytes_per_second);
  }               
  ctx->bytes_per_second = new_bytes_per_second;

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "start: setting RESOLUTION=%"PRIu64" in header\n", new_resolution);
  if (ascii_header_set (pwcm->header, "RESOLUTION", "%"PRIu64"", new_resolution) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set RESOLUTION=%"PRIu64" in header\n", new_resolution);
  }

  // set the channel offset in the header
  if (ascii_header_set (pwcm->header, "CHAN_OFFSET", "%d", ctx->schan) < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set CHAN_OFFSET=%d in header\n", ctx->schan);
  }

  char ant_tag[16]; 
  int real_ant_id;
  int i;
  for (i=0; i < ctx->new_nant; i++)
  {
    sprintf (ant_tag, "ANT_ID_%d", i);
    real_ant_id = mopsr_get_new_ant_number (ctx->output_ants[i]);
    if (ctx->verbose)
      multilog (pwcm->log, LOG_INFO, "setting %s=%d\n", ant_tag, real_ant_id);

    if (ascii_header_set (pwcm->header, ant_tag, "%d", real_ant_id) < 0)
    {
      multilog (pwcm->log, LOG_WARNING, "start: failed to set %s=%d in header\n", 
                ant_tag, real_ant_id);
    }
  }

  // update this as it is used for command calculations
  pwcm->pwc->bytes_per_second = new_bytes_per_second;

  // set ordering if data in header
  if (ascii_header_set (pwcm->header, "ORDER", "TFS") < 0)
  {
    multilog (pwcm->log, LOG_WARNING, "start: failed to set ORDER=TFS in header\n");
  }
  
  // lock the socket mutex
  pthread_mutex_lock (&(ctx->mutex));

  // if we are receiving real data, not inserting fake zeros
  if (ctx->zeros)
  {
    if (ctx->verbose)
      multilog(pwcm->log, LOG_INFO, "start: init stopwatch\n");
    RealTime_Initialise(1);
    StopWatch_Initialise(1);
    ctx->state = STATE_STARTING;
  }
  else
  {
    // instruct the mon thread to not touch the socket
    if (ctx->verbose)
      multilog(pwcm->log, LOG_INFO, "start: disabling idle state\n");
    ctx->state = STATE_RECORDING;
  }

#ifdef MOPSR_UDPDB_BLOCKING
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: setting socket to non-blocking\n");
  if (sock_nonblock(ctx->sock->fd) < 0)
    multilog(ctx->log, LOG_ERR, "start: sock_nonblock failed: %s\n", strerror(errno));

#endif

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "start: usleep(1000)\n");
  usleep(1000);

  // get a single packet
  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "start: recvfrom\n");
  ctx->sock->got = recvfrom (ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL);
  if (ctx->verbose)
    multilog(pwcm->log, LOG_INFO, "start: got packet size=%d\n", ctx->sock->got);

  // if we did not receive a packet this is an error condition
  if (ctx->sock->got == -1)
  {
    multilog(ctx->log, LOG_ERR, "start: no UDP packets received\n");
    pthread_mutex_unlock (&(ctx->mutex));
    return -1;
  }

#ifdef MOPSR_UDPDB_BLOCKING
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: setting socket to blocking\n");
  if (sock_block(ctx->sock->fd) < 0)
    multilog(ctx->log, LOG_ERR, "start: sock_block failed: %s\n", strerror(errno));
#endif

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: decoding packet\n");

  // decode so we have nframe, etc
  mopsr_decode (ctx->sock->buf, &(ctx->hdr));

  pthread_mutex_unlock (&(ctx->mutex));

  // return the start time, 0 if no UTC_START provided in header
  return start_time;
}

/*! pwcm buffer function [for header only] */
void * mopsr_udpdb_selants_recv (dada_pwc_main_t * pwcm, int64_t * size)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) pwcm->context;

  if (ctx->verbose)
    multilog (pwcm->log, LOG_INFO, "mopsr_udpdb_selants_recv()\n");

  return pwcm->header;
}

/*
 * transfer function write data directly to the specified memory
 * block buffer with the specified block_id and size
 */
int64_t mopsr_udpdb_selants_recv_block (dada_pwc_main_t * pwcm, void * block, 
                                uint64_t block_size, uint64_t block_id)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_recv_block()\n");

  const unsigned nchan = ctx->nchan;
  const unsigned new_nant  = ctx->new_nant;
  const unsigned old_nant  = ctx->old_nant;

  uint64_t seq_no, byte_offset, expected_seq, iseq;
  uint64_t timeout_max = 10000000;
  uint64_t block_count = 0;
  uint64_t pkts_caught = 0;
  unsigned i;
  int errsv;

  unsigned in_prefix_stride = (ctx->schan - MOPSR_PFB_START_CHAN) * ctx->old_nant;
  unsigned in_suffix_stride = (MOPSR_PFB_END_CHAN - ctx->echan) * ctx->old_nant; 

  int16_t * in, * out;
  unsigned int iframe, ichan, iant;

  // if we have started capture, then increment the block sizes
  if (ctx->capture_started)
  {
    ctx->start_pkt  = ctx->end_pkt + 1;
    ctx->end_pkt    = ctx->start_pkt + (ctx->packets_per_buffer - 1);
    if (ctx->verbose)
      multilog(pwcm->log, LOG_INFO, "recv_block: [%"PRIu64" - %"PRIu64"]\n", ctx->start_pkt, ctx->end_pkt);
  }

  // now begin capture loop
  ctx->got_enough = 0;
  while (!ctx->got_enough)
  {
    while (!ctx->sock->have_packet && !ctx->got_enough)
    {
      ctx->sock->got = recvfrom (ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL);

      if (ctx->sock->got == UDP_PAYLOAD)
      {
        ctx->sock->have_packet = 1;
      }
      else if (ctx->sock->got == -1)
      {
        errsv = errno;
#ifndef MOPSR_UDPDB_BLOCKING
        if (errsv == EAGAIN)
        {
          ctx->n_sleeps++;
          if (ctx->capture_started)
            ctx->timeouts++;
          if (ctx->timeouts > timeout_max)
          {
            multilog(pwcm->log, LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n", ctx->timeouts, timeout_max);
            ctx->got_enough = 1;
          }
        }
        else
#endif
        {
          multilog (pwcm->log, LOG_ERR, "recv_block: recvfrom failed %s\n", strerror(errsv));
          ctx->got_enough = 1;
        }
      }
      else // we received a packet of the WRONG size, ignore it
      {
        multilog (pwcm->log, LOG_WARNING, "recv_block: received %d bytes, expected %d\n", ctx->sock->got, UDP_PAYLOAD);
        ctx->got_enough = 0;
        sleep (1);
      }
    }
    ctx->timeouts = 0;

    // we now have a packet
    if (!ctx->got_enough && ctx->sock->have_packet)
    {
      // just deocde the sequence number, rely on mon thread 
      mopsr_decode_seq (ctx->sock->buf, &(ctx->hdr));

#ifdef _DEBUG
        multilog (pwcm->log, LOG_INFO, "recv_block: seq_no=%"PRIu64"\n", ctx->hdr.seq_no);
#endif

      // wait for packet reset
      if ((!ctx->capture_started) && (ctx->hdr.seq_no < ctx->pkt_start + 1000) && (ctx->hdr.seq_no >= ctx->pkt_start))
      {
        ctx->prev_seq  = ctx->pkt_start - 1;
        ctx->start_pkt = ctx->pkt_start;
        ctx->end_pkt   = ctx->start_pkt + (ctx->packets_per_buffer - 1);

        //if (ctx->verbose)
          multilog (pwcm->log, LOG_INFO, "recv_block: START [%"PRIu64" - "
                    "%"PRIu64"] seq_no=%"PRIu64"\n", ctx->start_pkt, 
                    ctx->end_pkt, ctx->hdr.seq_no);
        ctx->capture_started = 1;
      }
     
      if (ctx->capture_started)
      {
        // expected first byte based on last packet
        expected_seq = ctx->prev_seq + 1;

        const uint64_t zeroed_seq = ctx->hdr.seq_no > ctx->end_pkt ? (ctx->end_pkt + 1) : ctx->hdr.seq_no;

        // handle any dropped packets up to either seq_no or end of block
        for (iseq=expected_seq; iseq < zeroed_seq; iseq++)
        {
          byte_offset = (iseq - ctx->start_pkt) * ctx->pkt_size;
          memcpy (block + byte_offset, ctx->zeroed_packet, ctx->pkt_size);
          ctx->packets->dropped++;
          block_count++;
        }

        // if this is an out of order packet (unlikely! no switch)
        if (ctx->hdr.seq_no < ctx->start_pkt)
        {
          ctx->sock->have_packet = 0;
          ctx->prev_seq = ctx->hdr.seq_no;
        }
        // if the current resides in this block
        else if (ctx->hdr.seq_no <= ctx->end_pkt)
        {
          ctx->sock->have_packet = 0;
          byte_offset = (ctx->hdr.seq_no - ctx->start_pkt) * ctx->pkt_size;

          if (ctx->zeros)
            memcpy (block + byte_offset, ctx->zeroed_packet, ctx->pkt_size);
          else
          {
            in  = (int16_t *) (ctx->sock->buf + UDP_HEADER);
            out = (int16_t *) (ctx->buf);
            for (iframe=0; iframe<ctx->hdr.nframe; iframe++)
            {
              in += in_prefix_stride;
              for (ichan=ctx->schan; ichan<=ctx->echan; ichan++)
              {
                for (iant=0; iant<new_nant; iant++)
                {
                  out[iant] = in[ctx->output_ants[iant]];
                }
                out += new_nant;
                in  += old_nant;
              }
              in += in_suffix_stride;
            }

            memcpy (block + byte_offset, (void *) ctx->buf, ctx->pkt_size);
          }
          ctx->prev_seq = ctx->hdr.seq_no;
          pkts_caught++;
        }
        else
        {
          ctx->got_enough = 1;
          ctx->prev_seq = ctx->end_pkt;
        }
      }
      else
      {
        byte_offset = 0;
        memcpy (block, ctx->sock->buf + UDP_HEADER, ctx->pkt_size);
        ctx->sock->have_packet = 0;
      }
    }
  }


  // update the statistics
  if (pkts_caught)
  {
    ctx->packets->received += pkts_caught;
    ctx->bytes->received   += (pkts_caught * ctx->pkt_size);

    uint64_t dropped = ctx->packets_per_buffer - pkts_caught;
    if (dropped)
    {
      multilog (pwcm->log, LOG_WARNING, "dropped %"PRIu64" packets %5.2f percent\n", dropped, (float) dropped / (float) ctx->packets_per_buffer);
      ctx->packets->dropped += dropped;
      ctx->bytes->dropped += (dropped * ctx->pkt_size);
    }
  }

  return (int64_t) block_size;
}

/*
 * Does not actually receive UDP data, simply writes 0's into the data block at 
 * the rate specified by BYTES_PER_SECOND
 */
int64_t mopsr_udpdb_selants_fake_block (dada_pwc_main_t * pwcm, void * block,
                                    uint64_t block_size, uint64_t block_id)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog(pwcm->log, LOG_INFO, "mopsr_udpdb_selants_fake_block()\n");

  // on first block just start 
  if (ctx->state == STATE_STARTING)
  {
    StopWatch_Start(&(ctx->wait_sw));

    // for fake block mode, we never touch the socket, so the idle thread
    // can have full access
    pthread_mutex_lock (&(ctx->mutex));
    ctx->state = STATE_IDLE;
    pthread_mutex_unlock (&(ctx->mutex));
  }

  // get the delay in micro seconds
  double to_delay = ((double) block_size) / (double) ctx->bytes_per_second;
  to_delay *= 1000000;

  // now delay for the requisite time
  if (ctx->verbose > 1)
    multilog (pwcm->log, LOG_INFO, "fake_block: %"PRIu64" bytes -> %lf seconds\n", block_size, to_delay/1000000);
  StopWatch_Delay(&(ctx->wait_sw), to_delay);

  // now start the stopwatch ready for the next fake_block call
  StopWatch_Start(&(ctx->wait_sw));

  return (int64_t) block_size;
}


/*! PWCM stop function, called at end of observation */
int mopsr_udpdb_selants_stop (dada_pwc_main_t* pwcm)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog (pwcm->log, LOG_INFO, "mopsr_udpdb_selants_stop()\n");

  if (ctx->zeros)
    StopWatch_Stop (&(ctx->wait_sw));

  ctx->capture_started = 0;

  pthread_mutex_lock (&(ctx->mutex));
  ctx->state = STATE_IDLE;
  pthread_mutex_unlock (&(ctx->mutex));

  multilog (pwcm->log, LOG_INFO, "received %"PRIu64" bytes\n", ctx->bytes->received);
  return 0;
}

/*
 *  Main. 
 */
int main (int argc, char **argv)
{

  /* IB DB configuration */
  mopsr_udpdb_selants_t ctx;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA PWCM */
  dada_pwc_main_t* pwcm = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* port on which to listen for incoming connections */
  int port = MOPSR_DEFAULT_UDPDB_PORT;

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

  // whether to run stats thread
  char run_stats_thread = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* Monitoring of UDP data */
  pthread_t mon_thread_id;

  int arg = 0;

  ctx.interface = malloc(sizeof(char) * 17);
  sprintf(ctx.interface, "%s", "all");

  ctx.mdir = 0;

  ctx.port = MOPSR_DEFAULT_UDPDB_PORT;
  ctx.verbose = 0;
  ctx.mdir = 0;
  ctx.pfb_id = malloc (sizeof(char) * 16);
  ctx.zeros = 0;
  sprintf (ctx.pfb_id, "%s", "XX");

  // these are defined in the PFB firmware
  ctx.old_nant = MOPSR_PFB_NANT;
  ctx.new_nant = 0;

  int schan = MOPSR_PFB_START_CHAN;
  int echan = MOPSR_PFB_END_CHAN;
  ctx.old_nchan = (echan - schan) + 1;

  while ((arg=getopt(argc,argv,"a:b:c:C:dD:hi:k:l:m:M:p:sv0")) != -1)
  {
    switch (arg)
    {
      case 'a':
        if (ctx.new_nant < MOPSR_PFB_NANT)
        {
          ctx.output_ants[ctx.new_nant] = mopsr_get_new_ant_index(atoi(optarg));
          ctx.new_nant++;
        }
        else
        {
          fprintf (stderr, "ERROR: too many -a options specified\n");
          return (EXIT_FAILURE);
        }
        break;

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
          mopsr_udpdb_selants_free (&ctx);
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
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }

      case 'C':
        if (optarg)
        { 
          schan = atoi(optarg);
          break;
        }
        else
        {
          fprintf (stderr, "ERROR: -C requires an argument\n");
          usage();
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }

      case 'd':
        daemon=1;
        break;

      case 'D':
        if (optarg)
        {
          echan = atoi(optarg);
          break;
        }
        else
        {
          fprintf (stderr, "ERROR: -D requires an argument\n");
          usage();            
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;                    
        }

      case 'h':
        usage();
        return EXIT_SUCCESS;

      case 'i':
        if (optarg)
        {
          sprintf(ctx.interface, "%s", optarg);
        }
        else
        {
          fprintf (stderr,"mopsr_udpdb_selants: -i requires argument\n");
          usage();
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }
        break;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr,"mopsr_udpdb_selants: could not parse key from %s\n",optarg);
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }
        break;

      case 'l':
        if (optarg) {
          log_port = atoi(optarg);
          break;
        } else {
          fprintf (stderr,"mopsr_udpdb_selants: no log_port specified\n");
          usage();
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }

      case 'm':
        if (optarg) {
          sprintf(ctx.pfb_id, "%s", optarg);
          break;
        } else {
          fprintf (stderr,"mopsr_udpdb_selants: -m requires argument\n");
          usage();
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }

      case 'M':
        if (optarg) {
          if (ctx.mdir)
            free (ctx.mdir);
          ctx.mdir = (char *) malloc (sizeof (char) * strlen(optarg)+1);
          strcpy (ctx.mdir , optarg);
          break;
        } else {
          fprintf (stderr,"mopsr_udpdb_selants: no dir specified\n");
          usage();
          mopsr_udpdb_selants_free (&ctx);
          return EXIT_FAILURE;
        }

      case 's':
        run_stats_thread = 1;
        break;

      case 'p':
        ctx.port = atoi (optarg);
        break;

      case 'v':
        ctx.verbose++;
        break;
        
      case '0':
        ctx.zeros = 1;
        break;
        
      default:
        usage ();
        return 0;
    }
  }

  if (ctx.new_nant == 0)
  {
    fprintf (stderr, "ERROR: at least 1 antenna must be specified via -a argument\n");
    usage();
    return EXIT_FAILURE;
  }

  // ensure start/end chans are sensible
  if ((schan != MOPSR_PFB_START_CHAN) || (echan != MOPSR_PFB_END_CHAN))
  {
    if ((schan < MOPSR_PFB_START_CHAN) || (schan > MOPSR_PFB_END_CHAN) || (schan > echan))
    {
      fprintf (stderr, "ERROR: specified start channel [%d] was outside limits\n", schan);
      return (EXIT_FAILURE);
    }
    if ((echan > MOPSR_PFB_END_CHAN) || (echan < MOPSR_PFB_START_CHAN) || (echan < schan))
    {
      fprintf (stderr, "ERROR: specified end channel [%d] was outside limits\n", echan);
      return (EXIT_FAILURE);
    }
  }

  ctx.schan = schan;
  ctx.echan = echan;
  ctx.new_nchan = (echan - schan) + 1;

  ctx.decifactor = (ctx.old_nchan / ctx.new_nchan) * (ctx.old_nant / ctx.new_nant);

  if (ctx.verbose)
  {
    fprintf (stderr, "main: decifactor=%u = (%d/%d) * (%d/%d)\n", ctx.decifactor, ctx.old_nchan, ctx.new_nchan, ctx.old_nant, ctx.new_nant);
    fprintf (stderr, "main: decifactor=%u = (%d) * (%d)\n", ctx.decifactor, (ctx.old_nchan / ctx.new_nchan), (ctx.old_nant / ctx.new_nant));
  }

  // do not use the syslog facility
  log = multilog_open ("mopsr_udpdb", 0);

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
  pwcm->start_function        = mopsr_udpdb_selants_start;
  pwcm->buffer_function       = mopsr_udpdb_selants_recv;
  if (ctx.zeros)
    pwcm->block_function        = mopsr_udpdb_selants_fake_block;
  else
    pwcm->block_function        = mopsr_udpdb_selants_recv_block;
  pwcm->stop_function         = mopsr_udpdb_selants_stop;
  pwcm->header_valid_function = mopsr_udpdb_selants_header_valid;
  pwcm->context               = &ctx;
  pwcm->verbose               = ctx.verbose;

  ctx.log = log;

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
  if (block_size % (UDP_DATA / ctx.decifactor) != 0)
  {
    fprintf (stderr, "mopsr_udpdb_selants: block size [%"PRIu64"] was not a "
                     "multiple of the decimated UDP payload size [%d] decifactor=%d\n", block_size,
                     UDP_DATA / ctx.decifactor, ctx.decifactor);
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_selants: creating dada pwc control interface\n");

  pwcm->pwc = dada_pwc_create();
  pwcm->pwc->port = control_port;

  if (recv_core >= 0)
    if (dada_bind_thread_to_core(recv_core) < 0)
      multilog(log, LOG_WARNING, "mopsr_udpdb_selants: failed to bind to core %d\n", recv_core);

  mopsr_udpdb_selants_init (&ctx);

  // packet size define in the above
  ctx.packets_per_buffer = block_size / ctx.pkt_size;

  if (ctx.mdir)
  {
    if (ctx.verbose)
      multilog(log, LOG_INFO, "mopsr_udpdb_selants: starting mon_thread()\n");
    int rval = pthread_create (&mon_thread_id, 0, (void *) mon_thread, (void *) &ctx);
    if (rval != 0)
    {
      multilog(log, LOG_ERR, "mopsr_udpdb_selants: could not create monitor thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }
  }

  pthread_t stats_thread_id = 0;
  if (run_stats_thread)
  {
    multilog(log, LOG_INFO, "starting stats_thread()\n");
    if (pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &ctx) < 0) {
      perror ("Error creating new thread");
       return -1;
    }
  }

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_selants: creating dada server\n");
  if (dada_pwc_serve (pwcm->pwc) < 0)
  {
    fprintf (stderr, "mopsr_udpdb_selants: could not start server\n");
    return EXIT_FAILURE;
  }

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_selants: entering PWC main loop\n");

  if (dada_pwc_main (pwcm) < 0)
  {
    fprintf (stderr, "mopsr_udpdb_selants: error in PWC main loop\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_selants: destroying pwc\n");
  dada_pwc_destroy (pwcm->pwc);

  if (ctx.verbose)
    fprintf (stdout, "mopsr_udpdb_selants: destroying pwc main\n");
  dada_pwc_main_destroy (pwcm);

  if (ctx.mdir)
  { 
    if (ctx.verbose)
      multilog(log, LOG_INFO, "mopsr_udpdb_selants: joining mon_thread\n");
    quit_threads = 1;
    void * result;
    pthread_join (mon_thread_id, &result);
  }

  if (run_stats_thread)
  {
    quit_threads = 1;
    void* result = 0;
    fprintf (stderr, "joining stats_thread\n");
    pthread_join (stats_thread_id, &result);
  }

  if (ctx.verbose)
    multilog(log, LOG_INFO, "mopsr_udpdb_selants: freeing resources\n");

  mopsr_udpdb_selants_free (&ctx);
 
  return EXIT_SUCCESS;
}


/*
 * Thread to provide raw access to a recent packet over the socket
 */
void mon_thread (void * arg)
{
  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) arg;

  char local_time[32];
  char mon_file[512];

  int fd;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  size_t nwrote, got, ncleared;
  time_t now, now_plus;
  time_t sleep_secs = 5;
  unsigned int npackets = 64;
  unsigned int ipacket = 0;
  uint64_t prev_seq = 0;

  char * packets = (char *) malloc (UDP_DATA * npackets + UDP_HEADER);
  size_t out_size = UDP_HEADER + (npackets * UDP_DATA);
  mopsr_hdr_t hdr;

  while ( !quit_threads )
  {
    if (ctx->state == STATE_IDLE)
    {
      // if in the idle state, lock the mutex as the mon thread 
      // will have control of the socket
      pthread_mutex_lock (&(ctx->mutex));

#ifdef MOPSR_UDPDB_BLOCKING
      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "mon: setting socket to non-blocking\n");
      if (sock_nonblock(ctx->sock->fd) < 0)
        multilog(ctx->log, LOG_ERR, "mon: sock_nonblock failed: %s\n", strerror(errno));
#endif
      ncleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);
#ifdef MOPSR_UDPDB_BLOCKING
      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "mon: setting socket to non-blocking\n");
      if (sock_block(ctx->sock->fd) < 0)
        multilog(ctx->log, LOG_ERR, "mon: sock_block failed: %s\n", strerror(errno));
#endif

      usleep (1000);

      ipacket = 0;
      while (!quit_threads && ipacket < npackets)
      {
        got = 0;
        while  (!quit_threads && got != UDP_PAYLOAD)
        {
          got = recvfrom (ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL);
          if (got == UDP_PAYLOAD)
          {
            memcpy (packets, ctx->sock->buf, UDP_HEADER);
            memcpy (packets + UDP_HEADER + (ipacket * UDP_DATA), ctx->sock->buf + UDP_HEADER, UDP_DATA);
            mopsr_decode (packets, &hdr);
            if ((prev_seq != 0) && (hdr.seq_no != prev_seq + 1) && ctx->verbose)
            {
              multilog(ctx->log, LOG_INFO, "mon_thread: hdr.seq=%"PRIu64" "
                       "prev_seq=%"PRIu64" [%"PRIu64"]\n", hdr.seq_no, 
                       prev_seq, hdr.seq_no - prev_seq);
            }
            prev_seq = hdr.seq_no;
            ipacket++;
          }
          else if (got == -1)
          {
#ifdef MOPSR_UDPDB_BLOCKING
            multilog (ctx->log, LOG_ERR, "mon_thread: recvfrom failed %s\n", strerror(errno));
#else
            // packet not at socket due to clear + nonblock
            if (errno == EAGAIN)
              usleep (10000);
#endif
          }
          else
          {
            // more serious!
            multilog(ctx->log, LOG_INFO, "mon_thread: init & got[%d] != UDP_PAYLOAD[%d]\n",
                     got, UDP_PAYLOAD);
            sleep (1);
          }
        }
      }

      // release control of the socket
      pthread_mutex_unlock (&(ctx->mutex));

      mopsr_decode (packets, &hdr);
      hdr.nframe *= npackets;
      mopsr_encode (packets, &hdr);
      out_size = UDP_HEADER + (npackets * UDP_DATA);
    }
    else
    {
      memcpy (packets, ctx->sock->buf, UDP_PAYLOAD);
      out_size = UDP_PAYLOAD;
    }

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

    // wait the appointed amount of time
    now_plus = time(0);
    while (!quit_threads && now + sleep_secs >= now_plus)
    {
      usleep (500000);
      now_plus = time(0);
    }
  }

  free (packets);
}

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  mopsr_udpdb_selants_t * ctx = (mopsr_udpdb_selants_t *) arg;

  uint64_t bytes_received_total = 0;
  uint64_t bytes_received_this_sec = 0;
  uint64_t bytes_dropped_total = 0;
  uint64_t bytes_dropped_this_sec = 0;
  double   gb_received_ps = 0;
  double   gb_dropped_ps = 0;
  uint64_t n_sleeps_total = 0;
  uint64_t n_sleeps_this_sec = 0;

  struct timespec ts;

  while (!quit_threads)
  {
    bytes_received_this_sec = ctx->bytes->received - bytes_received_total;
    bytes_dropped_this_sec  = ctx->bytes->dropped - bytes_dropped_total;
    n_sleeps_this_sec = ctx->n_sleeps - n_sleeps_total;

    bytes_received_total = ctx->bytes->received;
    bytes_dropped_total = ctx->bytes->dropped;
    n_sleeps_total = ctx->n_sleeps;

    gb_received_ps = (double) 8 * bytes_received_this_sec / 1000000000;
    gb_dropped_ps = (double) 8 * bytes_dropped_this_sec / 1000000000;

    fprintf(stderr,"T=%7.3lf  R=%7.3lf  D=%6.2lf [Gib/s] | packets=%"PRIu64" dropped=%"PRIu64" sleeps=%"PRIu64"\n", (gb_received_ps+gb_dropped_ps), gb_received_ps, gb_dropped_ps, ctx->packets->received, ctx->packets->dropped, n_sleeps_this_sec);
    sleep(1);
  }
}

