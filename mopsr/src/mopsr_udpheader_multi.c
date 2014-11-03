/*
 * mopsr_udpheader
 *
 * Simply listens on a specified port for udp packets encoded
 * in the MOPSR format
 *
 */

#include <sys/socket.h>
#include <math.h>

#include "mopsr_udpheader.h"
#include "sock.h"

void stats_thread(void * arg);
int quit_threads = 0;

void usage()
{
  fprintf (stdout,
    "mopsr_udpheadea_multi [options]\n"
     " -h             print help text\n"
     " -i ip:port     ip address and port to acquire\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpheader_prepare (udpheader_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  // open socket
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: creating udp socket on %s:%d\n", ctx->interface, ctx->port);
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 16 MB
  int sock_buf_size = 16*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  udpheader_reset(ctx);
}

int udpheader_reset (udpheader_t * ctx)
{
  ctx->n_sleeps = 0;
  ctx->capture_started = 0;

  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);
}


time_t udpheader_start_function (udpheader_t * ctx, time_t start_utc)
{

  multilog_t* log = ctx->log;

  // open UDP socket ready for capture
  udpheader_prepare (ctx);

  ctx->prev_time = time(0);
  ctx->curr_time = ctx->prev_time;

  ctx->seq_incr = 1;
  ctx->seq_max = 256;
  ctx->prev_seq_no = 0;

  return 0;
}

void* udpheader_read_function (udpheader_t* ctx, uint64_t* size)
{
  multilog_t * log = ctx->log;

  // Flag to drop out of for loop
  unsigned int ant_id = 0;

  // Assume we will be able to return a full buffer
  *size = 0;

  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  unsigned int nchan = 8;
  unsigned int nant = 16;
  unsigned int ant_stride = nant * 2;
  unsigned int nframe = UDP_DATA / (nchan * ant_stride);
  unsigned int iframe;
  unsigned int ichan;
  char buf[UDP_DATA];

  mopsr_hdr_t hdr;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "read: entering main loop for %d byte packets from fd=%d\n", UDP_PAYLOAD, ctx->sock->fd);

  /* Continue to receive packets */
  while (!quit_threads)
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL );

      if (got == UDP_PAYLOAD)
      {
        ctx->sock->have_packet = 1;
        timeouts = 0;
      }
      else if (got == -1)
      {
        errsv = errno;
        if (errsv == EAGAIN)
        {
          ctx->n_sleeps++;
          if (ctx->capture_started)
            timeouts++;
          if (timeouts > timeout_max)
          {
            multilog(log, LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
            quit_threads = 1;
          }
        }
        else
        {
          multilog (log, LOG_ERR, "receive_obs: recvfrom failed %s\n", strerror(errsv));
          return 0;
        }
      }
      else // we received a packet of the WRONG size, ignore it
      {
        multilog (log, LOG_ERR, "receive_obs: received %d bytes, expected %d\n", got, UDP_PAYLOAD);
        quit_threads = 1;
      }
    }
    if (timeouts > timeout_max)
    {
      multilog(log, LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
    }

    timeouts = 0;

    if (ctx->sock->have_packet)
    {
      mopsr_decode (ctx->sock->buf, &hdr);

      if ((ctx->verbose > 1) && (ctx->packets->received < 10))
        multilog (ctx->log, LOG_INFO, "PKT: %"PRIu64"\n", hdr.seq_no);

      // if first packet
      if ((!ctx->capture_started) && (hdr.seq_no < 1000))
      {
        ctx->capture_started = 1;
        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "receive_obs: START seq_no=%"PRIu64" ant_id=%u\n", hdr.seq_no, hdr.ant_id);
        ctx->packets->received ++;
        ctx->bytes->received += UDP_DATA;
      }
      else
      {
        if (hdr.seq_no == ctx->prev_seq_no + 1)
        {
          // this is normal, do nothing
          ctx->packets->received ++;
          ctx->bytes->received += UDP_DATA;

          char * in  = ctx->sock->buf + UDP_HEADER;
          char * out = buf;
          uint64_t bytes_copied = 0;

          // reorder the data in 32 byte chunks
          for (iframe=0; iframe<nframe; iframe++)
          {
            for (ichan=0; ichan<nchan; ichan++)
            {
              out = buf + (ichan * nframe) + (iframe * ant_stride);
              memcpy (out, in, ant_stride);
              in += ant_stride;
              bytes_copied += ant_stride;
            }
          }
        }
        else if ((hdr.seq_no > ctx->prev_seq_no + 1) && (ctx->prev_seq_no))
        {
          uint64_t n_pkts_dropped = hdr.seq_no - (ctx->prev_seq_no + 1);
          ctx->packets->dropped += n_pkts_dropped;
          ctx->bytes->dropped += n_pkts_dropped * UDP_DATA;
        }
        else
        {
          multilog (ctx->log, LOG_INFO, "read: reset ? seq_no = %"PRIu64"\n", hdr.seq_no);
        }
      }

      ctx->prev_seq_no = hdr.seq_no;
    }
  }

  return 0;
}

/*
 * Close the udp socket and file
 */

int udpheader_stop_function (udpheader_t* ctx)
{

  /* get our context, contains all required params */
  float percent_dropped = 0;
  multilog (ctx->log, LOG_INFO, "stop: closing socket\n");
  close(ctx->sock->fd);

  return 0;

}

int udpheader_init (udpheader_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

  ctx->prev_time = time(0);
  ctx->curr_time = ctx->prev_time;

  //ctx->dropped_packets_to_fill = 0;
  //ctx->received = 0;
  //ctx->error_seconds = 10;
  //ctx->packet_length = 0;
  //ctx->state = NOTRECORDING;

  // allocate required memory strucutres
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  return 0;
}



int main (int argc, char **argv)
{

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = MOPSR_DEFAULT_UDPDB_PORT;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  /* actual struct with info */
  udpheader_t udpheader;

  /* Pointer to array of "read" data */
  char *src;

  while ((arg=getopt(argc,argv,"i:p:vh")) != -1) {
    switch (arg) {

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'p':
      port = atoi (optarg);
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

  /* statistics thread */
  pthread_t stats_thread_id;

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_udpheader", 0);
  multilog_add (log, stderr);
  multilog_serve (log, DADA_DEFAULT_PWC_LOG);

  /* Setup context information */
  udpheader.log = log;
  udpheader.verbose = verbose;
  udpheader.port = port;
  udpheader.interface = strdup(interface);
  udpheader.plot_log = 0;

  // allocate require resources
  udpheader_init (&udpheader);

  time_t utc = udpheader_start_function(&udpheader,0);

  if (utc == -1 ) {
    fprintf(stderr,"Error: udpheader_start_function failed\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog(log, LOG_INFO, "starting stats_thread()\n");
  int rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpheader);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }


  while (!quit_threads) {

    uint64_t bsize = 1024;

   /* TODO Add a quit control to the read function */
    src = (char *) udpheader_read_function(&udpheader, &bsize);

    /* Quit if we dont get a packet for at least 1 second whilst recording */
    //if ((bsize <= 0) && (udpheader.state == RECORDING)) 
    //  quit = 1;

    if (udpheader.verbose == 2)
      fprintf(stdout,"udpheader_read_function: read %"PRIu64" bytes\n", bsize);
  }    

  if ( udpheader_stop_function(&udpheader) != 0)
    fprintf(stderr, "Error stopping acquisition");

  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  void * result = 0;
  pthread_join (stats_thread_id, &result);


  return EXIT_SUCCESS;

}

void stats_thread(void * arg) {

  udpheader_t * ctx = (udpheader_t *) arg;
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
  float gb_rcv_ps = 0;
  float mb_rcv_ps = 0;
  float mb_drp_ps = 0;

  while (!quit_threads)
  {

    /* get a snapshot of the data as quickly as possible */
    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    s_rcv_curr = ctx->n_sleeps;

    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_drp_1sec = b_drp_curr - b_drp_total;
    s_rcv_1sec = s_rcv_curr - s_rcv_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_drp_total = b_drp_curr;
    s_rcv_total = s_rcv_curr;

    mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    mb_drp_ps = (double) b_drp_1sec / 1000000;
    gb_rcv_ps = b_rcv_1sec * 8;
    gb_rcv_ps /= 1000000000;

    /* determine how much memory is free in the receivers */
    fprintf (stderr,"R=%6.5f [Gib/s], D=%4.1f [MiB/s], D=%"PRIu64" pkts, s_s=%"PRIu64"\n", gb_rcv_ps, mb_drp_ps, ctx->packets->dropped, s_rcv_1sec);

    sleep(1);
  }

}


