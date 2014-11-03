/*
 * mopsr_udplongprbs
 *
 * Simply listens on a specified port for udp packets encoded
 * in the MOPSR format
 *
 */

#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "multilog.h"

#include "futils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>

#include <sys/socket.h>
#include <math.h>

#include "sock.h"

/* structures dmadb datatype  */
typedef struct {

  multilog_t *    log;

  unsigned        ninputs;
  char **         interfaces;
  int *           ports;
  mopsr_sock_t ** socks;

  uint64_t        n_sleeps;
  int             capture_started;
  int             verbose;

  stats_t *       packets;
  stats_t *       bytes;

  time_t          prev_time;
  time_t          curr_time;

  uint64_t        seq_incr;
  uint64_t        seq_max;

  unsigned        ant1;
  unsigned        ant2;

} udplongprbs_t;

/* Re-implemented functinos from dada_pwc */
time_t  udplongprbs_start_function (udplongprbs_t* udplongprbs, time_t start_utc);
void*   udplongprbs_read_function (udplongprbs_t* udplongprbs, uint64_t* size);
int     udplongprbs_stop_function (udplongprbs_t* udplongprbs);

/* Utility functions */
void quit (udplongprbs_t* udplongprbs);
void signal_handler (int signalValue);



void stats_thread(void * arg);
int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udplongprbs [options]\n"
     " -b core        bind to specified CPU core\n"
     " -f ant1        first antenna to comparei [default 0]\n"
     " -h             print help text\n"
     " -t ant2        second antenna to compare [default 1]\n"
     " -i ip:port     ip address and port to acquire\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udplongprbs_prepare (udplongprbs_t * ctx)
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
      multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
    size_t cleared = dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
  }
  udplongprbs_reset(ctx);
}

int udplongprbs_reset (udplongprbs_t * ctx)
{
  ctx->n_sleeps = 0;
  ctx->capture_started = 0;

  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);
}


time_t udplongprbs_start_function (udplongprbs_t * ctx, time_t start_utc)
{

  multilog_t* log = ctx->log;

  // open UDP socket ready for capture
  if (udplongprbs_prepare (ctx) < 0)
    return -1;

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

void* udplongprbs_read_function (udplongprbs_t* ctx, uint64_t* size)
{
  multilog_t * log = ctx->log;

  // Flag to drop out of for loop
  unsigned int ant_id = 0;

  // Assume we will be able to return a full buffer
  *size = 0;

  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 100000000;

  const unsigned int nchan = 40;
  const unsigned int nant = 16;
  const unsigned int ndim = 2;
  const unsigned int nframe = 6;
  const unsigned int npkts = 128;

  int16_t data [2][nframe*npkts];

  int iframe, jframe, iseq, ipkt ;
  unsigned int iant, ichan;

  for (iframe=0; iframe<nframe; iframe++)
  {
    data[0][ipkt*nframe + iframe] = 0;
    data[1][ipkt*nframe + iframe] = 0;
  }

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
  char have_lag = 0;
  int start_packets[ctx->ninputs];
  int a1 = mopsr_get_new_ant_index (ctx->ant1);
  int a2 = mopsr_get_new_ant_index (ctx->ant2);
  for (i=0; i<ctx->ninputs; i++)
    start_packets[i] = 0;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "read: ant1: %d -> %d  ant2: %d -> %d\n", ctx->ant1, a1, ctx->ant2, a2);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "read: ninputs=%d\n", ctx->ninputs);

  // Continue to receive packets indefinitely
  while (!quit_threads)
  {
    have_lag = 0;
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
            return 0;
          }
        }
        else // we received a packet of the WRONG size, ignore it
        {
          multilog (log, LOG_WARNING, "receive_obs: received %d bytes, expected %d\n", ctx->socks[i]->got, UDP_PAYLOAD);
        }
      }

      if (timeouts > timeout_max)
      {
        multilog(log, LOG_INFO, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
      }
      timeouts = 0;

      //multilog (ctx->log, LOG_INFO, "socks[%d]->have_packet=%d\n", i, ctx->socks[i]->have_packet);

      if (ctx->socks[i]->have_packet)
      {
        mopsr_decode (ctx->socks[i]->buf, &hdr);

        if ((ctx->verbose > 1) && (ctx->packets->received < 10))
          multilog (ctx->log, LOG_INFO, "PKT: %"PRIu64"\n", hdr.seq_no);

        if ((ctx->verbose > 1) && (ctx->packets->received < 1))
          mopsr_print_header (&hdr);

        // if first packet
        if ((!ctx->capture_started) && (hdr.seq_no < 1000))
        {
          ctx->capture_started = 1;
          //if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "receive_obs: START seq_no=%"PRIu64" ant_id=%u\n", hdr.seq_no, hdr.ant_id);
          ctx->packets->received ++;
          ctx->bytes->received += UDP_DATA;
        }
        else
        {
          if (hdr.seq_no < 5)
            fprintf (stderr, "recv seq=%"PRIu64" from EG%02d\n", hdr.seq_no, i+1);

          if (hdr.seq_no > 10000)
          {
            if (start_packets[i] == 1)
            {
              fprintf (stderr, "unregistering input %d\n", i);
              start_packets[i] = 0;
            }
          }

          if ((hdr.seq_no >= 5) && (hdr.seq_no < npkts+5))
          {
            ipkt = (unsigned) (hdr.seq_no - 5);
            start_packets[i]++;
            if (ctx->verbose)
              fprintf (stderr, "PFB=%d seq_no=%"PRIu64" ipkt=%d\n", i, hdr.seq_no, ipkt);
            int16_t * in  = (int16_t *) (ctx->socks[i]->buf + UDP_HEADER);
            for (iframe=0; iframe<nframe; iframe++)
            {
              for (ichan=0; ichan<nchan; ichan++)
              {
                if (ichan == 30)
                {
                  //for (iant=0; iant<nant; iant++)
                  //  fprintf (stderr, "in[%d]=%"PRIi16" [%d]\n", iant, in[iant], mopsr_get_new_ant_number (iant));
                  // if only 1 input, compare ants from same packet
                  if (ctx->ninputs == 1)
                  {
                    data[0][ipkt * nframe + iframe] = in[a1];
                    data[1][ipkt * nframe + iframe] = in[a2];
                  }
                  else
                  {
                    if (i == 0)
                      data[i][ipkt * nframe + iframe] = in[a1];
                    else
                      data[i][ipkt * nframe + iframe] = in[a2];
                  }
                }
                in += nant;
              }
            }
            //usleep (100000);
            //dada_sock_clear_buffered_packets(ctx->socks[i]->fd, UDP_PAYLOAD);
          }
          else
          {
            //multilog (ctx->log, LOG_INFO, "read: reset ? seq_no = %"PRIu64"\n", hdr.seq_no);
          }
        }

        ctx->socks[i]->have_packet = 0;
        ctx->socks[i]->prev_seq = hdr.seq_no;
      }
    }

    unsigned isamp, jsamp, nsamp;
    ichan = 0;
    iant = 0;
    // we we received 2 start packets, check lag
    char have_enough = 1;
    for (i=0; i<ctx->ninputs; i++)
      if (start_packets[i] != npkts)
        have_enough = 0;

    if (have_enough)
    {
      int lag = 0;
      char seq_valid = 0;
      // now search for iant1 in iant1 in chan0, ant0
      multilog(log, LOG_INFO, "searching for lag\n");
      nsamp = nframe * npkts;
      for (isamp=8; !have_lag && isamp<nsamp-8; isamp++)
      {
        //multilog(log, LOG_INFO, "isamp=%d\n", isamp);
        for (jsamp=8; !have_lag && jsamp<nsamp-8; jsamp++)
        {
          if (data[0][isamp] == data[1][jsamp])
          {
            seq_valid = 1;
            for (iseq=-8; iseq<8; iseq++)
            {
              if (ctx->verbose)
                multilog(log, LOG_INFO, "partial map %d, %d, testing[%d] %d == %d\n", 
                          isamp, jsamp, iseq, data[0][isamp+iseq], data[1][jsamp+iseq]);
              if (data[0][isamp+iseq] != data[1][jsamp+iseq])
                seq_valid = 0;
            }
            if (seq_valid)
            {
              lag = jsamp - isamp;
              have_lag = 1;
              multilog(log, LOG_INFO, "LAG %d  ip0=%d ip1=%d\n", lag, isamp, jsamp);
            }
          }
        }
      }

      multilog(log, LOG_INFO, "searching done\n");

      for (i=0; i<ctx->ninputs; i++)
        start_packets[i] = 0;
    }
  }

  free(main_memory);
  return 0;
}

/*
 * Close the udp socket and file
 */

int udplongprbs_stop_function (udplongprbs_t* ctx)
{
  // get our context, contains all required params
  float percent_dropped = 0;
  multilog (ctx->log, LOG_INFO, "stop: closing socket[s]\n");
  unsigned int i;
  for (i=0; i<ctx->ninputs; i++)
    close (ctx->socks[i]->fd);
  return 0;
}

int udplongprbs_init (udplongprbs_t * ctx)
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
  udplongprbs_t udplongprbs;

  /* Pointer to array of "read" data */
  char *src;

  const char *sep = ":";
  char * saveptr;
  char * str;

  udplongprbs.interfaces = (char **) malloc (1 * sizeof(char *));
  udplongprbs.ports      = (int *) malloc (1 * sizeof(int));

  udplongprbs.ant1 = 0;
  udplongprbs.ant2 = 1;

  while ((arg=getopt(argc,argv,"b:f:i:t:vh")) != -1) 
  {
    switch (arg) 
    {
      case 'b':
        if (optarg)
          core = atoi(optarg);
        break;

      case 'f':
        udplongprbs.ant1 = atoi(optarg);
        break;

      case 'i':
        if (optarg)
        {
          ninputs++;
          udplongprbs.interfaces = (char **) realloc (udplongprbs.interfaces, ninputs * sizeof(char *));
          udplongprbs.ports      = (int *) realloc (udplongprbs.ports, ninputs * sizeof(int));

          // parse the IP address from the argument
          str = strtok_r(optarg, sep, &saveptr);
          if (str== NULL)
          {
            fprintf(stderr, "mopsr_udplongprbs: misformatted input option for ip\n");
            return (EXIT_FAILURE);
          }
          udplongprbs.interfaces[ninputs-1] = (char *) malloc ((strlen(str)+1) * sizeof(char));
          strcpy(udplongprbs.interfaces[ninputs-1], str);

          // parse the port address from the argument
          str = strtok_r(NULL, sep, &saveptr);
          if (str== NULL)
          {
            fprintf(stderr, "mopsr_udplongprbs: misformatted input option for port\n");
            return (EXIT_FAILURE);
          }
          if (sscanf(str, "%d", &(udplongprbs.ports[ninputs-1])) != 1)
          {
            fprintf(stderr, "mopsr_udplongprbs: misformatted input option for port\n");
            return (EXIT_FAILURE);
          }
        }
        break;

      case 't':
        udplongprbs.ant2 = atoi(optarg);
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
  //pthread_t stats_thread_id;

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_udplongprbs", 0);
  multilog_add (log, stderr);
  multilog_serve (log, DADA_DEFAULT_PWC_LOG);

  if (core >= 0)
    if (dada_bind_thread_to_core(core) < 0)
      multilog(log, LOG_WARNING, "mopsr_udplongprbs: failed to bind to core %d\n", core);

  // setup context information
  udplongprbs.ninputs = ninputs;
  udplongprbs.log = log;
  udplongprbs.verbose = verbose;

  // handle no arguments case
  if (ninputs == 0)
  {
    ninputs = 1;
    udplongprbs.interfaces[0] = (char *) malloc (16 * sizeof(char *));
    sprintf (udplongprbs.interfaces[0], "%s", "192.168.3.100");
    udplongprbs.ports[0] = 4001;
    udplongprbs.ninputs = 1;
  }

  // allocate require resources
  udplongprbs_init (&udplongprbs);

  time_t utc = udplongprbs_start_function(&udplongprbs,0);

  if (utc == -1 ) {
    fprintf(stderr,"Error: udplongprbs_start_function failed\n");
    return EXIT_FAILURE;
  }

  /*
  if (verbose)
    multilog(log, LOG_INFO, "starting stats_thread()\n");
  int rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udplongprbs);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }
*/

  while (!quit_threads) 
  {
    uint64_t bsize = 1024;

    // TODO Add a quit control to the read function
    src = (char *) udplongprbs_read_function(&udplongprbs, &bsize);

    if (udplongprbs.verbose == 2)
      fprintf(stdout,"udplongprbs_read_function: read %"PRIu64" bytes\n", bsize);
  }    

  if ( udplongprbs_stop_function(&udplongprbs) != 0)
    fprintf(stderr, "Error stopping acquisition");
/*
  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  void * result = 0;
  pthread_join (stats_thread_id, &result);
*/

  return EXIT_SUCCESS;

}

void stats_thread(void * arg) {

  udplongprbs_t * ctx = (udplongprbs_t *) arg;
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


