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

#define UDP_TIMEOUT_MAX 1000000
#define FASTER 1

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpdb_dual [options]\n"
     " -b <core>     bind compuation to CPU core\n"
     " -c <port>     port to open for PWCC commands [default: %d]\n"
     " -d            run as daemon\n"
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
    ctx->socks[i] = mopsr_init_sock();

  ctx->buffer = (unsigned char *) malloc (sizeof(char) * UDP_PAYLOAD);

  // allocate required memory strucutres
  ctx->packets  = init_stats_t();
  ctx->bytes    = init_stats_t();
  ctx->pkt_size = UDP_DATA;

  multilog(ctx->log, LOG_INFO, "init: pkt_size=%d bytes\n", ctx->pkt_size);

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

  mopsr_udpdb_dual_reset (ctx);

  return 0;
}

void mopsr_udpdb_dual_reset (mopsr_udpdb_dual_t * ctx)
{
  ctx->capture_started = 0;
  ctx->got_enough = 0;
  ctx->start_byte = 0;
  ctx->end_byte   = 0;
  ctx->obs_bytes  = 0;
  ctx->n_sleeps   = 0;
  ctx->timeouts   = 0;
  ctx->idle_state = 1;
  ctx->last_block = 0;

  unsigned i=0;
  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->seq_offset = -1;
    ctx->socks[i]->block_count = 0;
    ctx->socks[i]->have_packet = 0;
  }

  reset_stats_t(ctx->packets); 
  reset_stats_t(ctx->bytes); 
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

  if (ctx->mdir)
    free (ctx->mdir);
  ctx->mdir = 0;

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

  // instruct the mon thread to not touch the socket
  ctx->idle_state = 0;

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start: clearing packets at socket\n");
  size_t cleared;
  unsigned i;
  for (i=0; i<ctx->ninputs; i++)
    cleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);

  // centralised control will inform us of UTC_START, so return 0
  time_t start_time = 0;
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

  uint64_t seq_byte, byte_offset;
  uint64_t timeout_max = 1000000;
  uint64_t block_count = 0;
  int64_t total_bytes = 0;
  unsigned i;
  int errsv;

  void *in, *ou;

  // constants used in this TEST mode
  const unsigned nframe = 8;
  const unsigned nchan = 32;
  const unsigned nant = 16;
  const uint64_t nsamps = nframe * nchan;
  uint64_t isamp;

  const unsigned chan_span = nant * 2;
  const size_t ou_stride = ctx->ninputs * chan_span;
  const size_t in_stride = chan_span;
  unsigned int iframe, ichan, iput;

  // if we have started capture, then increment the block sizes
  if (ctx->capture_started)
  {
    ctx->start_byte = ctx->end_byte + ctx->pkt_size;
    ctx->end_byte   = (ctx->start_byte + block_size) - ctx->pkt_size;
  }
  else
  {
    size_t ncleared;
    for (i=0; i<ctx->ninputs; i++)
      ncleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
  }

  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->block_count = 0;
  }

  // now begin capture loop
  ctx->got_enough = 0;
  while (!ctx->got_enough)
  {
    for (i=0; i<ctx->ninputs; i++)
    {
      while (!ctx->socks[i]->have_packet && !ctx->got_enough)
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
            ctx->got_enough = 1;
          }
        }
        else // we received a packet of the WRONG size, ignore it
        {
          multilog (pwcm->log, LOG_ERR, "recv_block: received %d bytes, expected %d\n", ctx->socks[i]->got, UDP_PAYLOAD);
        }
      }

      // we now have a packet
      if (!ctx->got_enough && ctx->socks[i]->have_packet)
      {
        mopsr_decode_seq (ctx->socks[i]->buf, &(ctx->hdr));

        if (ctx->verbose > 2)
          multilog (pwcm->log, LOG_INFO, "recv_block: seq_no=%"PRIu64"\n", ctx->hdr.seq_no);

        // wait for packet reset on all inputs [special case]
        if (ctx->capture_started < ctx->ninputs) 
        {
          if (i == 0)
          {
            ctx->start_byte = ctx->hdr.seq_no * ctx->pkt_size * ctx->ninputs;
            ctx->end_byte   = (ctx->start_byte + block_size) - ctx->pkt_size;
          }

          multilog (pwcm->log, LOG_INFO, "recv_block: START [%"PRIu64" - %"PRIu64"] seq_no=%"PRIu64" input=%d\n", ctx->start_byte, ctx->end_byte, ctx->hdr.seq_no, i);
          ctx->socks[i]->seq_offset = (int64_t) (ctx->hdr.seq_no * ctx->pkt_size * ctx->ninputs) - ctx->start_byte;
          multilog (pwcm->log, LOG_INFO, "recv_block: socks[%d]->seq_offset=%"PRIi64"\n",  i, ctx->socks[i]->seq_offset);

          ctx->capture_started++;
        }
     
        if (ctx->capture_started)
        {
          seq_byte = (ctx->hdr.seq_no * ctx->pkt_size * ctx->ninputs) - ctx->socks[i]->seq_offset;
          //multilog (pwcm->log, LOG_INFO, "seq_byte=%"PRIu64"\n", seq_byte);
  
          // packet belonged in a previous block [to late - oh well]
          if (seq_byte < ctx->start_byte)
          {
            multilog (pwcm->log, LOG_INFO, "recv_block: [%d] seq_no=%"PRIu64", seq_byte [%"PRIu64"] < start_byte [%"PRIu64"]\n", i, ctx->hdr.seq_no, seq_byte, ctx->start_byte);
            ctx->packets->dropped++;
            ctx->bytes->dropped += ctx->pkt_size;

            // we are going to ignore (& consume) this packet
            ctx->socks[i]->have_packet = 0;
          }
          else
          {
            // packet belongs in this block [yay]
            if (seq_byte <= ctx->end_byte)
            {
              byte_offset = seq_byte - ctx->start_byte;

              ctx->socks[i]->block_count++;

              // we are going to consume this packet
              ctx->socks[i]->have_packet = 0;

              // since ordering is TFPS, we need to interleave data from inputs [urgh!]
#ifdef FASTER
              in = (void *) (ctx->socks[i]->buf + UDP_HEADER);
              ou = (void *) (block + byte_offset + (i * chan_span));

              for (isamp=0; isamp<nsamps; isamp++)
              {
                memcpy (ou, in, chan_span);
                ou += ou_stride;
                in += in_stride;
              }
#else
              uint64_t in_offset = UDP_HEADER;
              uint64_t ou_offset = byte_offset + (i * chan_span);

              for (iframe=0; iframe < nframe; iframe++)
              {
                for (ichan=0; ichan < nchan; ichan++)
                {
                  in = ctx->socks[i]->buf + in_offset;
                  ou = block + ou_offset;

                  //multilog (pwcm->log, LOG_INFO, "recv_block: [%d]  memcpy in_offset= %"PRIu64" ou_offset=%"PRIu64"\n", i, in_offset, ou_offset);
                  memcpy (ou, in, chan_span);

                  in_offset += in_stride;
                  ou_offset += ou_stride;

                }
              }
#endif
              ctx->bytes->received += ctx->pkt_size;
              ctx->packets->received++;
              block_count++;
            }
            // packet belongs in a future block [dang]
            else
            {
              // but since we have 2 inputs, lets make sure that we have removed items from both queues
              //multilog (pwcm->log, LOG_INFO, "recv_block: [%d] seq_no=%"PRIu64" seq_byte [%"PRIu64"] > end_byte [%"PRIu64"]\n", i, ctx->hdr.seq_no, seq_byte, ctx->end_byte);

              // if not all inputs are buffering a packet, keep 
              ctx->got_enough = 1;
              for (iput=0; iput<ctx->ninputs; iput++)
              {
                //multilog (pwcm->log, LOG_INFO, "recv_block: iput=%d block_count=%"PRIu64" have_packet=%d\n", iput, ctx->socks[iput]->block_count, ctx->socks[iput]->have_packet);
                if (ctx->socks[iput]->have_packet == 0)
                  ctx->got_enough = 0;
              }
            }
          }
        }
      }

      // check for a full block [normal / yay]
      if (block_count >= ctx->packets_per_buffer)
        ctx->got_enough = 1;
    }
  }

  if (block_count != ctx->packets_per_buffer)
  {
    uint64_t dropped = ctx->packets_per_buffer - block_count;
    if (dropped)
    {
      if (ctx->verbose)
        multilog (pwcm->log, LOG_INFO, "recv_block: dropped=%"PRIu64"\n", dropped);
      ctx->packets->dropped += dropped;
      ctx->bytes->dropped += (dropped * ctx->pkt_size);
    }
  } 

  multilog (pwcm->log, LOG_INFO, "recv_block: block_count=%"PRIu64" [%"PRIu64" %"PRIu64"]\n", block_count, ctx->socks[0]->block_count, ctx->socks[1]->block_count);
  multilog (pwcm->log, LOG_INFO, "recv_block: n_sleeps=%"PRIu64"\n", ctx->n_sleeps);

  ctx->n_sleeps = 0;
  total_bytes += (int64_t) block_size;

  // save a pointer to the last full block of data for the monitor thread
  if (ctx->capture_started)
  {
    //multilog (pwcm->log, LOG_INFO, "recv_block: setting last_block=%p\n", (void *) ctx->last_block);
    ctx->last_block = block;
  }

  return total_bytes;
}


/*! PWCM stop function, called at end of observation */
int mopsr_udpdb_dual_stop (dada_pwc_main_t* pwcm)
{
  mopsr_udpdb_dual_t * ctx = (mopsr_udpdb_dual_t *) pwcm->context;

  if (ctx->verbose > 1)
    multilog (pwcm->log, LOG_INFO, "mopsr_udpdb_dual_stop()\n");

  ctx->capture_started = 0;
  ctx->idle_state = 1;
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
  ctx.mdir = ".";

  while ((arg=getopt(argc,argv,"b:c:di:k:l:M:sv")) != -1)
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
    if (mopsr_udpdb_dual_parse_input (&ctx, "192.168.3.100:4001") < 0)
    {
      fprintf (stderr, "ERROR: failed to parse default input\n");
      return EXIT_FAILURE;
    }
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
  ctx.packets_per_buffer = block_size / ctx.pkt_size;

  if (ctx.verbose)
    multilog(log, LOG_INFO, "mopsr_udpdb_dual: starting mon_thread()\n");
  //int rval = pthread_create (&mon_thread_id, 0, (void *) mon_thread, (void *) &ctx);
  int rval = 0;
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
  unsigned int npackets = 32;
  unsigned int ipacket = 0;
  void * in, * ou;

  const unsigned nframe = 8;
  const unsigned nchan = 32;
  const unsigned nant = 16;
  const unsigned chan_span = nant * 2;
  const size_t ou_stride = ctx->ninputs * chan_span;
  const size_t in_stride = chan_span;
  unsigned int iframe, ichan;

  char * packets = (char *) malloc ((UDP_DATA * npackets * ctx->ninputs) + UDP_HEADER);
  size_t out_size = UDP_HEADER + (npackets * ctx->ninputs * UDP_DATA);
  unsigned i=0;

  mopsr_hdr_t hdr;
  hdr.nchan  = 32;
  hdr.nant   = 16 * ctx->ninputs;
  hdr.nframe = 8;
  hdr.nbit   = 8;
  hdr.ant_id = 0;
  hdr.seq_no = 0;

  while ( !quit_threads )
  {
    if (ctx->idle_state)
    {
      //multilog (ctx->log, LOG_INFO, "mon_thread: clearing buffered packets\n");
      for (i=0; i<ctx->ninputs; i++)
      {
        ncleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
      }
      usleep (1000);

      ipacket = 0;
      while (!quit_threads && ipacket < npackets)
      {
        //multilog (ctx->log, LOG_INFO, "mon_thread: while %d < %d\n", ipacket, npackets);
        for (i=0; i<ctx->ninputs; i++)
        {
          //multilog (ctx->log, LOG_INFO, "mon_thread: foreach input [%d]\n", i);
          ctx->socks[i]->got = 0;
          while  (!quit_threads && ctx->socks[i]->got != UDP_PAYLOAD)
          {
            //multilog (ctx->log, LOG_INFO, "mon_thread: recvfrom on [%d]\n", i);
            ctx->socks[i]->got = recvfrom (ctx->socks[i]->fd, ctx->buffer, UDP_PAYLOAD, 0, NULL, NULL);
            if (ctx->socks[i]->got == UDP_PAYLOAD)
            {
              // since ordering is TFPS, we need to interleave data from inputs
              in = (void *) (ctx->buffer + UDP_HEADER);
              ou = (void *) (packets + UDP_HEADER + (ipacket * UDP_DATA) + (i * chan_span));
              //multilog (ctx->log, LOG_INFO, "mon_thread: decoding output offset=%d\n",  (ipacket * UDP_DATA) + (i * chan_span));
              for (iframe=0; iframe < nframe; iframe++)
              {
                for (ichan=0; ichan < nchan; ichan++)
                {
                  memcpy (ou, in, chan_span);
                  ou += ou_stride;
                  in += in_stride;
                }
              }

              //multilog (ctx->log, LOG_INFO, "mon_thread: decoding hdr seq\n");
              // deocde the header...
              mopsr_decode_seq (packets, &hdr);

              //multilog (ctx->log, LOG_INFO, "mon_thread: hdr seq=%"PRIu64"\n", hdr.seq_no);
              if ((ctx->socks[i]->prev_seq != 0) && (hdr.seq_no != ctx->socks[i]->prev_seq + 1) && ctx->verbose)
              {
                multilog(ctx->log, LOG_INFO, "mon_thread: hdr.seq=%"PRIu64" prev_seq=%"PRIu64" [%"PRIu64"]\n", 
                         hdr.seq_no, ctx->socks[i]->prev_seq, hdr.seq_no - ctx->socks[i]->prev_seq);
              }
              ctx->socks[i]->prev_seq = hdr.seq_no;
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

      //multilog (ctx->log, LOG_INFO, "mon_thread: encoding header\n", i);
      hdr.nframe = nframe * npackets;
      mopsr_encode (packets, &hdr);
      out_size = UDP_HEADER + (npackets * UDP_DATA * ctx->ninputs);
    }
    else
    {
#ifdef RUN_WHILE_RECORDING
      //multilog (ctx->log, LOG_INFO, "mon_thread: else case\n");
      // if we re-read data from that last written shared memory block
      if (ctx->last_block)
      {
        memcpy (packets + UDP_HEADER, ctx->last_block, (npackets * UDP_DATA * ctx->ninputs));
        out_size = UDP_HEADER + (npackets * UDP_DATA * ctx->ninputs);
        hdr.nframe = nframe * npackets;
        mopsr_encode (packets, &hdr);
      }
      // read 1 packets worth of data from the socket buffers for each input
      else
      {
        // since ordering is TFPS, we need to interleave data from inputs
        for (i=0; i<ctx->ninputs; i++)
        {
          in = (void *) (ctx->buffer + UDP_HEADER);
          ou = (void *) (packets + UDP_HEADER + (i * chan_span));
          for (iframe=0; iframe < nframe; iframe++)
          {
            for (ichan=0; ichan < nchan; ichan++)
            {
              memcpy (ou, in, chan_span);
              ou += ou_stride;
              in += in_stride;
            }
          }
        }
        hdr.nframe = nframe;
        mopsr_encode (packets, &hdr);
        out_size = UDP_HEADER + (UDP_DATA * ctx->ninputs);
      }
#else
      out_size = 0;
#endif
    }

    if (out_size)
    {
      now = time(0);
      strftime (local_time, 32, DADA_TIMESTR, localtime(&now));
      sprintf (mon_file, "%s/%s.dump", ctx->mdir, local_time);
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
