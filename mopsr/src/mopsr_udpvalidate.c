/*
 * mopsr_udpvalidate
 *
 * Simply listens on a specified port for udp packets and checks that all the data
 * are 0 on the specified ports.
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

#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#define PACKET_SIZE 7696
#define PAYLOAD_SIZE 7680

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  uint64_t pkts_processed;

  uint64_t ant_zero_errors[16];

  //uint64_t ant_nonzero_errors[16];

  char ant_zeroed[16];

  // number of antenna
  unsigned int nant;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  size_t pkt_size;

  size_t data_size;

  int antenna;

} udpvalidate_t;

int udpvalidate_init (udpvalidate_t * ctx);
int udpvalidate_prepare (udpvalidate_t * ctx);
int udpvalidate_destroy (udpvalidate_t * ctx);

void check_packet (udpvalidate_t * ctx, char * buffer, unsigned int size);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpvalidate [options]\n"
     " -a nant        antenna that should not be zeroed\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -p port        port on which to listen [default %d]\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpvalidate_prepare (udpvalidate_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpvalidate_reset(ctx);
}

int udpvalidate_reset (udpvalidate_t * ctx)
{
  unsigned iant;
  for (iant=0; iant < 16; iant++)
  {
    ctx->ant_zero_errors[iant] = 0;
    //ctx->ant_nonzero_errors[iant] = 0;
  }
  ctx->pkts_processed = 0;
}

int udpvalidate_destroy (udpvalidate_t * ctx)
{
  if (ctx->sock)
  {
    close(ctx->sock->fd);
    mopsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
}

/*
 * Close the udp socket and file
 */

int udpvalidate_init (udpvalidate_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

  // now set it up to allow for all valid jumbo packets
  ctx->sock->bufsz = sizeof(char) * PACKET_SIZE;
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
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "init: recv_from (%d, %p, %d, 0, NULL,NULL)\n", ctx->sock->fd, (void *) ctx->sock->buf, PACKET_SIZE);
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buf, PACKET_SIZE, 0, NULL, NULL);
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

  char bin[9];
  unsigned i;
  for (i=0; i<8; i++)
  {
    char_to_bstring(bin, ctx->sock->buf[i]);
    fprintf (stderr, "buf[%d]=%s\n", i, bin);
  }
  multilog (ctx->log, LOG_INFO, "init: nchan=%u nant=%u nframe=%u\n", hdr.nchan, hdr.nant, hdr.nframe);

  ctx->nchan = hdr.nchan;
  ctx->nant = hdr.nant;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

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
  udpvalidate_t udpvalidate;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int plot_log = 0;

  double sleep_time = 1000000;

  char zap_dc = 0;

  int antenna_to_plot = -1;

  unsigned iant;
  for (iant=0; iant<16; iant++)
  {
    udpvalidate.ant_zeroed[iant] = 1;
  }

  while ((arg=getopt(argc,argv,"a:hi:p:v")) != -1) {
    switch (arg) {

    case 'a':
      iant = atoi(optarg);
#ifdef HIRES
      unsigned packed_ant = mopsr_get_hires_ant_number (iant);
#else
      unsigned packed_ant = mopsr_get_new_ant_number (iant);
#endif
      udpvalidate.ant_zeroed[packed_ant] = 0;
      break;

    case 'h':
      usage();
      return 0;
      
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

    default:
      usage ();
      return 0;
      
    }
  }

  multilog_t* log = multilog_open ("mopsr_udpvalidate", 0);
  multilog_add (log, stderr);

  udpvalidate.log = log;
  udpvalidate.verbose = verbose;

  udpvalidate.interface = strdup(interface);
  udpvalidate.port = port;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpvalidate: using device %s\n", device);

  // allocate require resources, open socket
  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpvalidate: init()\n");
  if (udpvalidate_init (&udpvalidate) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpvalidate: prepare()\n");
  udpvalidate_prepare (&udpvalidate);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  udpvalidate_t * ctx = &udpvalidate;

  mopsr_hdr_t hdr;
  unsigned i;

  time_t now = time(0);;
  time_t prev = now;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpvalidate: while(!quit)\n");
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
#ifdef HIRES
      mopsr_decode_red (ctx->sock->buf, &hdr, ctx->port);
#else
      mopsr_decode (ctx->sock->buf, &hdr);
#endif

      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", hdr.seq_no, (hdr.seq_no - prev_seq_no));
      prev_seq_no = hdr.seq_no;

      if (ctx->verbose > 2)
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64"\n", hdr.seq_no);

      // integrate packet into totals
      check_packet (ctx, ctx->sock->buf + UDP_HEADER, ctx->pkt_size - UDP_HEADER);
      ctx->pkts_processed++;
    }

    now = time(0);
    if (now > prev)
    {
      uint64_t zero_errors = 0;
      //uint64_t nonzero_errors = 0;
      for (i=0; i<16; i++)
      {
        zero_errors += ctx->ant_zero_errors[i];
        //nonzero_errors += ctx->ant_nonzero_errors[i];
      }
      multilog (ctx->log, LOG_INFO, "checked=%"PRIu64" zero_errors=%"PRIu64"\n", ctx->pkts_processed, zero_errors);
    }
    prev = now;
  }

  return EXIT_SUCCESS;
}

void check_packet (udpvalidate_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "integrate_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  unsigned nframe = size / (ctx->nant * ctx->nchan * 2);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "integrate_packet: assigning floats, checking min/max: nframe=%d\n", nframe);

  int8_t * in = (int8_t *) buffer;
  float re, im;
  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      for (iant=0; iant < ctx->nant; iant++)
      {
        if (ctx->ant_zeroed[iant])
        {
          if ((in[0] != 0) || (in[1] != 0))
            ctx->ant_zero_errors[iant]++;
        }
        /*
        else
        {
          if ((in[0] == 0) && (in[1] == 0))
            ctx->ant_nonzero_errors[iant]++;
        }
        */
        in += 2;
      }
    }
  }
}

