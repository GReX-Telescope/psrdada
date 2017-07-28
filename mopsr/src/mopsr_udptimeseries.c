/*
 * mopsr_udptimeseries
 *
 * Simply listens on a specified port for udp packets encoded
 * in the MOPSR format
 *
 */


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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <math.h>
#include <cpgplot.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "mopsr_util.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#include "stopwatch.h"

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // the antenna to plot
  int antenna;

  // channel to plot
  unsigned int channel;

  // number of channels
  unsigned int nchan;

  // number of antenna in each packet
  unsigned int nant;

  // number of samples per plot
  unsigned int nsamps;

  unsigned int verbose;

  float * timeseries;

  uint64_t num_integrated;

  float ymax;

  size_t pkt_size;

  size_t data_size;

} udptimeseries_t;

int udptimeseries_init (udptimeseries_t * ctx);
int udptimeseries_prepare (udptimeseries_t * ctx);
int udptimeseries_destroy (udptimeseries_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udptimeseries [options]\n"
     " -h             print help text\n"
     " -a ant         antenna index to display [default 0]\n"
     " -c chan        channel to display [default 0]\n"
     " -D device      display on pgplot device [default /xs]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t samps       number of time samples to display [default 2048]\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udptimeseries_prepare (udptimeseries_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udptimeseries_reset(ctx);
  return 0;
}

int udptimeseries_reset (udptimeseries_t * ctx)
{
  unsigned i = 0;
  for (i=0; i<ctx->nsamps * MOPSR_NDIM; i++)
    ctx->timeseries[i] = 0;
  ctx->num_integrated = 0;
  return 0;
}

int udptimeseries_destroy (udptimeseries_t * ctx)
{
  unsigned ichan,iant,idim;
  if (ctx->timeseries)
    free (ctx->timeseries);
  ctx->timeseries = 0;

  if (ctx->sock)
  {
    close(ctx->sock->fd);
    mopsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
  return 0;
}

/*
 * Close the udp socket and file
 */

int udptimeseries_init (udptimeseries_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

  ctx->sock->bufsz = 9000;
  free (ctx->sock->buf);
  ctx->sock->buf = (char *) malloc (ctx->sock->bufsz);
  assert(ctx->sock->buf != NULL);

  // open socket
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: creating udp socket on %s:%d\n", ctx->interface, ctx->port);
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 4 MB
  int sock_buf_size = 4*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // get a packet to determine the packet size and type of data
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buf, 9000, 0, NULL, NULL);
  size_t data_size = ctx->pkt_size - UDP_HEADER;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, data_size);
  ctx->data_size = ctx->pkt_size - UDP_HEADER;

  // decode the header
  multilog (ctx->log, LOG_INFO, "init: decoding packet\n");
  mopsr_hdr_t hdr;
#ifdef HIRES
  mopsr_decode_red (ctx->sock->buf, &hdr, ctx->port);
#else
  mopsr_decode (ctx->sock->buf, &hdr);
#endif

  multilog (ctx->log, LOG_INFO, "init: nchan=%u nant=%u nframe=%u\n", hdr.nchan, hdr.nant, hdr.nframe);

  ctx->nchan = hdr.nchan;
  ctx->nant = hdr.nant;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: allocating %d bytes for ctx->timeseries\n", sizeof(float) * ctx->nsamps * MOPSR_NDIM);
  ctx->timeseries = (float *) malloc (sizeof(float) * ctx->nsamps * MOPSR_NDIM);
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

  char * device = "/xs";

  /* actual struct with info */
  udptimeseries_t udptimeseries;

  /* Pointer to array of "read" data */
  char *src;

  int nant = -1;

  unsigned int nsamps = 2048;

  int nchan = -1;

  double sleep_time = 1;

  unsigned int zap = 0;

  int antenna = 0;

  int channel = 0;

  while ((arg=getopt(argc,argv,"a:c:d:D:i:n:p:q:s:t:vh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
      break;

    case 'c':
      channel = atoi (optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      sleep_time = (double) atof (optarg);
      break;

    case 't':
      nsamps = atoi (optarg);
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

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_udptimeseries", 0);
  multilog_add (log, stderr);

  udptimeseries.log = log;
  udptimeseries.verbose = verbose;

  udptimeseries.interface = strdup(interface);
  udptimeseries.port = port;

  // allocate require resources, open socket
  if (udptimeseries_init (&udptimeseries) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  if (nchan > 0)
    udptimeseries.nchan = nchan;
  if (nant > 0)
    udptimeseries.nant = nant;
  udptimeseries.nsamps = nsamps;

  udptimeseries.antenna = antenna;
  udptimeseries.channel = channel;

  mopsr_util_t opts;
  opts.ant_code = 0;
  opts.ant = antenna;
  opts.nchan = udptimeseries.nchan;
  opts.nant = udptimeseries.nant;
  opts.chans[0] = channel;
  opts.chans[1] = channel;
  opts.ymin = 0;
  opts.ymax = 0;
  opts.plot_plain = 0;

  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_dbplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // cloear packets ready for capture
  udptimeseries_prepare (&udptimeseries);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  udptimeseries_t * ctx = &udptimeseries;
  unsigned int timeseries_offset;

  uint64_t frames_per_packet = ctx->data_size / (ctx->nant * ctx->nchan * 2);
  mopsr_hdr_t hdr;

  StartTimer(&wait_sw);

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, ctx->pkt_size, 0, NULL, NULL );

      if (got == ctx->pkt_size) 
      {
        ctx->sock->have_packet = 1;
        timeouts = 0;
      }
      else if (got == -1)
      {
        errsv = errno;
        if (errsv == EAGAIN)
        {
          timeouts++;
          if (timeouts > timeout_max)
          {
            multilog(log, LOG_INFO, "main: timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
            quit_threads = 1;
          }
        }
        else
        {
          multilog (log, LOG_ERR, "main: recvfrom failed %s\n", strerror(errsv));
          return 0;
        }
      }
      else // we received a packet of the WRONG size, ignore it
      {
        multilog (log, LOG_ERR, "main: received %d bytes, expected %d\n", got, ctx->pkt_size);
        quit_threads = 1;
      }
    }
    if (timeouts > timeout_max)
    {
      multilog(log, LOG_INFO, "main: timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
    }
    timeouts = 0;

    if (ctx->sock->have_packet)
    {
      StartTimer(&wait_sw);
#ifdef HIRES
      mopsr_decode_red (ctx->sock->buf, &hdr, ctx->port);
#else
      mopsr_decode (ctx->sock->buf, &hdr);
#endif

      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // integrate packet into totals
      timeseries_offset = ctx->num_integrated * MOPSR_NDIM;

      if (ctx->verbose > 1)
        multilog (ctx->log, LOG_INFO, "main: timeseries_offset=%u channel=%u antenna=%u nchan=%u nant=%u\n", 
                  timeseries_offset, ctx->channel, ctx->antenna, ctx->nchan, ctx->nant);

      mopsr_extract_channel (ctx->timeseries + timeseries_offset, 
                             ctx->sock->buf + UDP_HEADER, ctx->data_size, 
                             ctx->channel, ctx->antenna, 
                             ctx->nchan, ctx->nant);
      ctx->num_integrated += frames_per_packet;

      if (ctx->num_integrated >= ctx->nsamps)
      {
        multilog(ctx->log, LOG_INFO, "plotting %d of %d\n", ctx->num_integrated, ctx->nsamps);
        mopsr_plot_time_series (ctx->timeseries, ctx->channel, ctx->num_integrated, &opts);
        udptimeseries_reset (ctx);
        DelayTimer(&wait_sw, sleep_time);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
        StartTimer(&wait_sw);
      } 
    }
  }

  cpgclos();

  return EXIT_SUCCESS;

}
