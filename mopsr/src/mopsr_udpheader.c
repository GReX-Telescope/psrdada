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
     "mopsr_udpheader [options]\n"
     " -b core        bind to specified CPU core\n"
     " -h             print help text\n"
     " -i ip:port     ip address and port to acquire\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpheader_prepare (udpheader_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  unsigned int i;
  int sock_buf_size = 16*1024*1024;
  size_t cleared;

  for (i=0; i<ctx->ninputs; i++)
  {
    // open socket[s]
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "prepare: creating udp socket on %s:%d\n", ctx->interfaces[i], ctx->ports[i]);
    ctx->socks[i]->fd = dada_udp_sock_in(ctx->log, ctx->interfaces[i], ctx->ports[i], ctx->verbose);
    if (ctx->socks[i]->fd < 0) {
      multilog (ctx->log, LOG_ERR, "Error, failed to create udp socket at %s:%d\n", ctx->interfaces[i], ctx->ports[i]);
      return -1;
    }

    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "prepare: setting buffer size to %d\n", sock_buf_size);
    dada_udp_sock_set_buffer_size (ctx->log, ctx->socks[i]->fd, ctx->verbose, sock_buf_size);

    // set the socket to non-blocking
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "prepare: setting non_block\n");
    sock_nonblock(ctx->socks[i]->fd);

    // clear any packets buffered by the kernel
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket: UDP_PAYLOAD=%d\n", UDP_PAYLOAD);
    size_t cleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
  }
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
  unsigned i;
  for (i=0; i<ctx->ninputs; i++)
  {
    ctx->socks[i]->prev_seq = 0;
  }
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

  // setup 256 MB of memory to cycle through
  size_t main_memory_size = 256*1024*1024;
  void * main_memory = malloc(main_memory_size);
  size_t main_memory_offset = 0;

  unsigned int i;

  mopsr_hdr_t hdr;

  // For select polling
  struct timeval timeout;
  fd_set *rdsp = NULL;
  fd_set readset;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "read: entering main loop for %d byte packets\n", UDP_PAYLOAD);

  int normal = 0;

  // Continue to receive packets indefinitely
  while (!quit_threads)
  {
    // for each input get a packet
    for (i=0; i<ctx->ninputs; i++)
    {
      ctx->socks[i]->have_packet = 0;

      while (!ctx->socks[i]->have_packet && !quit_threads)
      {
        // receive 1 packet into the socket buffer
        ctx->socks[i]->got = recvfrom ( ctx->socks[i]->fd, ctx->socks[i]->buf, UDP_PAYLOAD, 0, NULL, NULL );

        if (ctx->socks[i]->got == UDP_PAYLOAD)
        {
          ctx->socks[i]->have_packet = 1;
          timeouts = 0;
        }
        else if (ctx->socks[i]->got == -1)
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
            sleep(1);
            return 0;
          }
        }
        else // we received a packet of the WRONG size, ignore it
        {
          multilog (log, LOG_ERR, "receive_obs: received %d bytes, expected %d\n", ctx->socks[i]->got, UDP_PAYLOAD);
          quit_threads = 1;
        }
      }

      if (timeouts > timeout_max)
      {
        multilog(log, LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
      }
      timeouts = 0;

      if (ctx->socks[i]->have_packet)
      {
        mopsr_decode (ctx->socks[i]->buf, &hdr);

        if ((ctx->verbose > 1) && (ctx->packets->received < 100))
          multilog (ctx->log, LOG_INFO, "PKT: %"PRIu64"\n", hdr.seq_no);

        if ((ctx->verbose) && (ctx->packets->received < 1))
          mopsr_print_header (&hdr);

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
          if (hdr.seq_no < 10)
            multilog (ctx->log, LOG_INFO, "receive_obs: START seq_no=%"PRIu64"\n", hdr.seq_no);

          ctx->packets->received ++;
          ctx->bytes->received += UDP_DATA;

#ifdef COUNT
          if (hdr.seq_no == ctx->socks[i]->prev_seq+ 1)
          {
            // this is normal, do nothing
            ctx->packets->received ++;
            ctx->bytes->received += UDP_DATA;
/*
            if (normal)
            {
              char * in  = ctx->socks[i]->buf + UDP_HEADER;
              char * out = (char *) main_memory;
              uint64_t bytes_copied = 0;

              // reorder the data in 32 byte chunks
              for (iframe=0; iframe<nframe; iframe++)
              {
                for (ichan=0; ichan<nchan; ichan++)
                {
                  out = main_memory + (ichan * nframe) + (iframe * ant_stride);
                  memcpy (out, in, ant_stride);
                  in += ant_stride;
                  bytes_copied += ant_stride;
                }
              }
            }
            else
            {
              const unsigned nframe = 8;
              const unsigned nchan = 32;
              const unsigned nant = 16;
              const unsigned chan_span = nant * 2;

              size_t ou_stride = ctx->ninputs * chan_span;
              size_t in_stride = chan_span;

              //unsigned int ival;
              //const unsigned int chan_nval = hdr.nant * 2;
              void * in = (void *) (ctx->socks[i]->buf + UDP_HEADER);
              void * ou = (void *) (main_memory + main_memory_offset + (i * chan_span));

              //fprintf (stderr, "chan_span=%d nframe=%d\n", chan_span, nframe);
              // since ordering is TFPS, we need to interleave data from inputs [urgh!]
              for (iframe=0; iframe < nframe; iframe++)
              {
                for (ichan=0; ichan < nchan; ichan++)
                {
                  memcpy (ou, in, chan_span);
                  ou += ou_stride;
                  in += in_stride;
                }
              }
              //normal = 1;
              if (i == ctx->ninputs -1)
              {
                main_memory_offset = (main_memory_offset + (ctx->ninputs * UDP_DATA));
                if (main_memory_offset + (ctx->ninputs * UDP_DATA) > main_memory_size)
                  main_memory_offset = 0;
              }
            }
            */
          }
          else if ((hdr.seq_no > ctx->socks[i]->prev_seq + 1) && (ctx->socks[i]->prev_seq))
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "hdr.seq_no=%lu prev_seq=%lu\n", hdr.seq_no, ctx->socks[i]->prev_seq);
            uint64_t n_pkts_dropped = hdr.seq_no - (ctx->socks[i]->prev_seq + 1);
            ctx->packets->dropped += n_pkts_dropped;
            ctx->bytes->dropped += n_pkts_dropped * UDP_DATA;
          }
          else if (hdr.seq_no == ctx->socks[i]->prev_seq && ctx->socks[i]->prev_seq)
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "read: duplicate seq_no=%"PRIu64" prev=%"PRIu64"\n", hdr.seq_no, ctx->socks[i]->prev_seq);
            ctx->packets->dropped++;
            ctx->bytes->dropped += UDP_DATA;
          }
          else
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "read: reset? seq_no=%"PRIu64" prev=%"PRIu64"\n", hdr.seq_no, ctx->socks[i]->prev_seq);
          }
#endif
        }

        ctx->socks[i]->prev_seq = hdr.seq_no;
      }
    }
  }

  free(main_memory);
  return 0;
}

/*
 * Close the udp socket and file
 */

int udpheader_stop_function (udpheader_t* ctx)
{
  // get our context, contains all required params
  float percent_dropped = 0;
  multilog (ctx->log, LOG_INFO, "stop: closing socket[s]\n");
  unsigned int i;
  for (i=0; i<ctx->ninputs; i++)
    close (ctx->socks[i]->fd);
  return 0;
}

int udpheader_init (udpheader_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket for each input
  ctx->socks = (mopsr_sock_t **) malloc (ctx->ninputs * sizeof(mopsr_sock_t *));
  unsigned isock;
  for (isock=0; isock<ctx->ninputs; isock++)
  {
    ctx->socks[isock] = mopsr_init_sock();
  }

  ctx->prev_time = time(0);
  ctx->curr_time = ctx->prev_time;

  // allocate required memory strucutres
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  return 0;
}


int main (int argc, char **argv)
{
  // Flag set in verbose mode
  int verbose = 0;

  int arg = 0;

  unsigned int ninputs = 0;

  int core = -1;

  /* actual struct with info */
  udpheader_t udpheader;

  /* Pointer to array of "read" data */
  char *src;

  const char *sep = ":";
  char * saveptr;
  char * str;

  udpheader.interfaces = (char **) malloc (1 * sizeof(char *));
  udpheader.ports      = (int *) malloc (1 * sizeof(int));

  while ((arg=getopt(argc,argv,"b:i:vh")) != -1) 
  {
    switch (arg) 
    {
      case 'b':
        if (optarg)
          core = atoi(optarg);
        break;

      case 'i':
        if (optarg)
        {
          ninputs++;
          udpheader.interfaces = (char **) realloc (udpheader.interfaces, ninputs * sizeof(char *));
          udpheader.ports      = (int *) realloc (udpheader.ports, ninputs * sizeof(int));

          // parse the IP address from the argument
          str = strtok_r(optarg, sep, &saveptr);
          if (str== NULL)
          {
            fprintf(stderr, "mopsr_udpheader: misformatted input option for ip\n");
            return (EXIT_FAILURE);
          }
          udpheader.interfaces[ninputs-1] = (char *) malloc ((strlen(str)+1) * sizeof(char));
          strcpy(udpheader.interfaces[ninputs-1], str);

          // parse the port address from the argument
          str = strtok_r(NULL, sep, &saveptr);
          if (str== NULL)
          {
            fprintf(stderr, "mopsr_udpheader: misformatted input option for port\n");
            return (EXIT_FAILURE);
          }
          if (sscanf(str, "%d", &(udpheader.ports[ninputs-1])) != 1)
          {
            fprintf(stderr, "mopsr_udpheader: misformatted input option for port\n");
            return (EXIT_FAILURE);
          }
        }
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

  // statistics thread
  pthread_t stats_thread_id;

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_udpheader", 0);
  multilog_add (log, stderr);
  multilog_serve (log, DADA_DEFAULT_PWC_LOG);

  if (core >= 0)
    if (dada_bind_thread_to_core(core) < 0)
      multilog(log, LOG_WARNING, "mopsr_udpheader: failed to bind to core %d\n", core);

  // setup context information
  udpheader.ninputs = ninputs;
  udpheader.log = log;
  udpheader.verbose = verbose;

  // handle no arguments case
  if (ninputs == 0)
  {
    ninputs = 1;
    udpheader.interfaces[0] = (char *) malloc (16 * sizeof(char *));
    sprintf (udpheader.interfaces[0], "%s", "192.168.3.100");
    udpheader.ports[0] = 4001;
    udpheader.ninputs = 1;
  }

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

  while (!quit_threads) 
  {
    uint64_t bsize = 1024;

    // TODO Add a quit control to the read function
    src = (char *) udpheader_read_function(&udpheader, &bsize);

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
  float gb_drp_ps = 0;

  while (!quit_threads)
  {

    // get a snapshot of the data as quickly as possible
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

    gb_rcv_ps = b_rcv_1sec * 8;
    gb_rcv_ps /= 1000000000;

    gb_drp_ps = b_drp_1sec * 8;
    gb_drp_ps /= 1000000000;

    gb_rcv_ps = b_rcv_1sec * 8;
    gb_rcv_ps /= 1000000000;

    /* determine how much memory is free in the receivers */
    fprintf (stderr,"R=%6.3f [Gib/s], D=%6.3f [Gib/s], D=%"PRIu64" pkts, s_s=%"PRIu64"\n", gb_rcv_ps, gb_drp_ps, ctx->packets->dropped, s_rcv_1sec);

    sleep(1);
  }

}


