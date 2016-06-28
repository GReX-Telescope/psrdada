/*
 * caspsr_udpNdebug. Reads UDP packets and checks the header for correctness
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
//#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>

#include "caspsr_udpNdebug.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "sock.h"

/* debug mode */
#define _DEBUG 1

/* global variables */
int quit_threads = 0;

void usage()
{
  fprintf (stdout,
	   "caspsr_udpNdebug [options] i_dist n_dist\n"
	   " -i iface       interface for UDP packets [default all interfaces]\n"
	   " -p port        port for incoming UDP packets [default %d]\n"
     " -h             print help text\n"
     " -v             verbose messages\n"
     "\n"
     "i_dist          index of distributor\n"
     "n_dist          number of distributors\n"
     "\n",
     CASPSR_DEFAULT_UDPNDEBUG_PORT);
}


time_t udpNdebug_start_function (udpNdebug_t * ctx)
{

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "udpNdebug_start_function()\n");

  ctx->buffer = (char *) malloc(sizeof(char) * UDP_PAYLOAD);
  assert(ctx->buffer);

  /* open sockets */
  ctx->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->udp_port, ctx->verbose);
  if (ctx->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0;
  }

  /* set the socket size to */
  int sock_buf_size = 256*1024*1024;
  dada_udp_sock_set_buffer_size (ctx->log, ctx->fd, ctx->verbose, sock_buf_size);

  int on = 1;
  setsockopt(ctx->fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

  /* set the socket to non-blocking */
  sock_nonblock(ctx->fd);

  /* setup the next_seq to the initial value */
  ctx->packets_this_xfer = 0;
  ctx->next_seq = 0;
  ctx->n_sleeps = 0;
  ctx->ooo_packets = 0;
  ctx->ooo_ch_ids = 0;

  return 0;
}


/*
 * Read data from the UDP socket and write to the ring buffers
 */
void * receiving_thread (void * arg)
{

  udpNdebug_t * ctx = (udpNdebug_t *) arg;

  /* multilogging facility */
  multilog_t * log = ctx->log;

  /* pointer for header decode function */
  unsigned char * arr;

  /* raw sequence number */
  uint64_t raw_seq_no = 0;

  /* "fixed" raw sequence number */
  uint64_t fixed_raw_seq_no = 0;

  /* offset raw sequence number */
  int64_t offset_raw_seq_no = 0;

  /* decoded sequence number */
  int64_t seq_no = 0;

  /* previously received sequence number */
  int64_t prev_seq_no = 0;

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

  /* total bytes received + dropped */
  uint64_t bytes_total = 0;

  /* remainder of the fixed raw seq number modulo seq_inc */
  int64_t remainder = 0;

  unsigned i = 0;
  unsigned j = 0;
  int thread_result = 0;

  struct timeval timeout;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "receiving thread starting\n");

  /* set the CPU that this thread shall run on */
  if (dada_bind_thread_to_core(ctx->recv_core) < 0)
    multilog(ctx->log, LOG_WARNING, "failed to bind receiving_thread to core %d\n", ctx->send_core);

  /* the expected ch_id */
  ctx->ch_id = ctx->i_distrib + 1;

  /* calculate the offset expected in the sequence number */
  seq_offset = 1024 * (ctx->i_distrib);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "receiving_thread: seq_inc=%"PRIu64", seq_offset=%"PRId64", gobal_offset=%"PRIi64"\n", seq_inc, seq_offset, global_offset);

  /* Continue to receive packets, until asked to quit */
  while (!quit_threads) 
  {

    have_packet = 0; 

    /* incredibly tight loop to try and get a packet */
    while (!have_packet && !quit_threads)
    {

      /* receive 1 packet into the current receivers buffer + offset */
      got = recvfrom (ctx->fd, ctx->buffer, UDP_PAYLOAD, 0, NULL, NULL);

      /* if we received a packet as expected */
      if (got == UDP_PAYLOAD) {

        have_packet = 1;
        ignore_packet = 0;

      /* a problem ocurred, most likely no packet at the non-blocking socket */
      } else if (got == -1) {

        errsv = errno;

        if (errsv == EAGAIN) {
          ctx->n_sleeps++;
        } else {
          multilog (log, LOG_ERR, "recvfrom failed: %s\n", strerror(errsv));
          thread_result = -1;
          pthread_exit((void *) &thread_result);
        }

      /* we received a packet of the WRONG size, ignore it */
      } else {

        multilog (log, LOG_ERR, "Received %d bytes, expected %d\n",
                                got, UDP_PAYLOAD);
        have_packet = 1;
        ignore_packet = 0;
        //ignore_packet = 1;

      }
    }

    /* If we did get a packet within the timeout */
    if (have_packet) {

      /* set this pointer to the start of the buffer */
      arr = ctx->buffer;

      /* Decode the packets apsr specific header */
      raw_seq_no = UINT64_C (0);
      for (i = 0; i < 8; i++ )
      {
        tmp = UINT64_C (0);
        tmp = arr[8 - i - 1];
        raw_seq_no |= (tmp << ((i & 7) << 3));
      }

      /* handle the global offset in sequence numbers */
      fixed_raw_seq_no = raw_seq_no + global_offset;

      /* adjust for the offset of this distributor */
      offset_raw_seq_no = fixed_raw_seq_no - seq_offset;

      /* check the remainder of the sequence number, for errors in global offset */
      remainder = offset_raw_seq_no % seq_inc;

      if (ctx->next_seq == 0) {
      }

      if (remainder == 0) {
        // do nothing
      } 
      else if  (remainder < (seq_inc / 2))
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
        multilog(ctx->log, LOG_WARNING, "remainder too large to adjust %d\n", remainder);
      }

      seq_no = (offset_raw_seq_no) / seq_inc;

      ch_id = UINT64_C (0);
      for (i = 0; i < 8; i++ )
      {
        tmp = UINT64_C (0);
        tmp = arr[16 - i - 1];
        ch_id |= (tmp << ((i & 7) << 3));
      }

      if ((prev_seq_no) && ((seq_no - prev_seq_no) != 1)) {
        multilog(ctx->log, LOG_INFO, "****************************************************************\n");
        problem_packets = 0;
      }
      if (ctx->next_seq == 0) {
        multilog(ctx->log, LOG_INFO, "FIRST [%"PRIu64"] %"PRIu64" -> %"PRIu64" -> %"PRIi64", remainder=%d, seq=%"PRIi64", prev_seq=%"PRIi64"\n", ch_id, raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no,  remainder, seq_no, prev_seq_no);
        ctx->next_seq = seq_no;
      }

      if (remainder != 0) {
        multilog(ctx->log, LOG_INFO, "PROB  [%"PRIu64"]  %"PRIu64" -> %"PRIu64" -> %"PRIi64", remainder=%d, seq=%"PRIi64", prev_seq=%"PRIi64"\n", ch_id, raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no,  remainder, seq_no, prev_seq_no);
        problem_packets++;
      } 
      else if ((prev_seq_no) && ((seq_no - prev_seq_no) != 1))
      {
        multilog(ctx->log, LOG_INFO, "RESET [%"PRIu64"]  %"PRIu64" -> %"PRIu64" -> %"PRIi64", remainder=%d, seq=%"PRIi64", prev_seq=%"PRIi64"\n", ch_id, raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no,  remainder, seq_no, prev_seq_no);
      } 
      else if ((seq_no < 10) && (!ignore_packet)) {

        multilog(ctx->log, LOG_INFO, "START [%"PRIu64"]  %"PRIu64" -> %"PRIu64" -> %"PRIi64", remainder=%d, seq=%"PRIi64", prev_seq=%"PRIi64"\n", ch_id, raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no,  remainder, seq_no, prev_seq_no);

      } 
      else 
      {
        // do nothing
      }

      prev_seq_no = seq_no;
    }

    if (!ignore_packet) {

      have_packet = 0;
      bytes_total += UDP_PAYLOAD;

    } 

    // we have ignored more than 10 packets
    if (problem_packets > 10) {
      multilog (ctx->log, LOG_WARNING, "Ignored more than 10 packets, exiting\n");
      quit_threads = 1;  
    }

  }

  if (ctx->verbose) 
    multilog(log, LOG_INFO, "receiving thread exiting\n");

  /* return 0 */
  pthread_exit((void *) &thread_result);
}

/*
 * Close the udp socket and file
 */

int udpNdebug_stop_function (udpNdebug_t* ctx)
{

  multilog_t *log = ctx->log;

  free (ctx->buffer);
  ctx->buffer = 0;
  close(ctx->fd);
  return 0;

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

  /* port for incoming UDP packets */
  int inc_port = CASPSR_DEFAULT_UDPNDEBUG_PORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* actual struct with info */
  udpNdebug_t udpNdebug;

  /* Pointer to array of "read" data */
  char *src;

  int arg = 0;

  /* statistics thread */
  pthread_t stats_thread_id;

  pthread_t receiving_thread_id;

  while ((arg=getopt(argc,argv,"di:p:vh")) != -1) {
    switch (arg) {

    case 'd':
      daemon = 1;
      break; 

    case 'i':
      if (optarg)
        interface = optarg;
      break;
    
    case 'p':
      inc_port = atoi (optarg);
      break;

    case 'v':
      verbose = 1;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }
  
  /* check the command line arguments */
  int i = 0;
  int rval = 0;
  void* result = 0;

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

  log = multilog_open ("caspsr_udpNdebug", 0);
  multilog_add (log, stderr);

  /* Setup context information */
  udpNdebug.log = log;
  udpNdebug.verbose = verbose;
  udpNdebug.udp_port = inc_port;
  udpNdebug.interface = strdup(interface);
  udpNdebug.n_distrib = n_distrib;
  udpNdebug.i_distrib = i_distrib;
  udpNdebug.bytes_to_acquire = -1;
  udpNdebug.ooo_ch_ids = 0;
  udpNdebug.ooo_packets = 0;
  udpNdebug.send_core = 0;
  udpNdebug.recv_core = 0;
  udpNdebug.send_sleeps = 0;
  udpNdebug.n_sleeps = 0;
  udpNdebug.mb_rcv_ps = 0;
  udpNdebug.mb_drp_ps = 0;
  udpNdebug.mb_snd_ps = 0;

  signal(SIGINT, signal_handler);

  if (verbose)
    multilog(log, LOG_INFO, "starting stats_thread()\n");
  rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpNdebug);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }

  if (verbose)
    multilog(log, LOG_INFO, "udpNdebug_start_function()\n");
  time_t utc = udpNdebug_start_function(&udpNdebug);
  if (utc == -1 ) {
    multilog(log, LOG_ERR, "Could not run start function\n");
    return EXIT_FAILURE;
  }

  /* process on last 4 cores */
  udpNdebug.recv_core = (i_distrib - (i_distrib % 2)) + 4; 

  /* make the receiving thread high priority */
  int min_pri = sched_get_priority_min(SCHED_RR);
  int max_pri = sched_get_priority_max(SCHED_RR);
  if (verbose)
    multilog(log, LOG_INFO, "sched_priority: min=%d, max=%d\n", min_pri, max_pri);

  struct sched_param send_parm;
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

  multilog(log, LOG_INFO, "Acquiring data indefinitely\n");

  if (verbose)
    multilog(log, LOG_INFO, "starting receiver_thread()\n");

  rval = pthread_create (&receiving_thread_id, &recv_attr, (void *) receiving_thread, (void *) &udpNdebug);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating receiving_thread: %s\n", strerror(rval));
    return -1;
  }

  if (verbose) 
    multilog(log, LOG_INFO, "joining receiving_thread\n");
  
  pthread_join (receiving_thread_id, &result);

  if (verbose)
    multilog(log, LOG_INFO, "main: receiveing thread joined\n");


  if (verbose) 
    multilog(log, LOG_INFO, "udpNdebug_stop_function\n");

  if ( udpNdebug_stop_function(&udpNdebug) != 0)
    fprintf(stderr, "Error stopping acquisition");

  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  pthread_join (stats_thread_id, &result);

  return EXIT_SUCCESS;

}


/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  udpNdebug_t * ctx = (udpNdebug_t *) arg;

  uint64_t b_rcv_total = 0;
  uint64_t b_rcv_1sec = 0;
  uint64_t b_rcv_curr = 0;

  uint64_t b_drp_total = 0;
  uint64_t b_drp_1sec = 0;
  uint64_t b_drp_curr = 0;

  uint64_t b_snd_total = 0;
  uint64_t b_snd_1sec = 0;
  uint64_t b_snd_curr = 0;

  uint64_t s_snd_total = 0;
  uint64_t s_snd_1sec = 0;
  uint64_t s_snd_curr = 0;

  uint64_t s_rcv_total = 0;
  uint64_t s_rcv_1sec = 0;
  uint64_t s_rcv_curr = 0;

  uint64_t ooo_pkts = 0;
  uint64_t ooo_chids = 0;

  unsigned i=0;
  unsigned next_receiver = 0;
  uint64_t bytes_buffered = 0;
  double mbytes_buffered = 0;
  uint64_t bytes_free = 0;

  while (!quit_threads)
  {
            
    /* get a snapshot of the data as quickly as possible */
    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    b_snd_curr = ctx->r_bytes;
    s_snd_curr = ctx->send_sleeps;
    s_rcv_curr = ctx->n_sleeps;
    ooo_pkts   = ctx->ooo_packets;
    ooo_chids  = ctx->ooo_ch_ids;

    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_snd_1sec = b_snd_curr - b_snd_total;
    b_drp_1sec = b_drp_curr - b_drp_total;
    s_rcv_1sec = s_rcv_curr - s_rcv_total;
    s_snd_1sec = s_snd_curr - s_snd_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_snd_total = b_snd_curr;
    b_drp_total = b_drp_curr;
    s_rcv_total = s_rcv_curr;
    s_snd_total = s_snd_curr;

    ctx->mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    ctx->mb_drp_ps = (double) b_drp_1sec / 1000000;
    ctx->mb_snd_ps = (double) b_snd_1sec / 1000000;

    if (ctx->verbose)
      fprintf (stderr,"R=%4.1f, D=%4.1f, S=%4.1f [MB/s] Free=%4.1f MB r_s=%"PRIu64", s_s=%"PRIu64", OoO pkts=%"PRIu64", chids=%"PRIu64"\n", ctx->mb_rcv_ps, ctx->mb_drp_ps, ctx->mb_snd_ps, ctx->mb_free, s_rcv_1sec, s_snd_1sec, ooo_pkts, ooo_chids);

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
