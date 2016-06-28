/*
 * mopsr_udpplot
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
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#include "stopwatch.h"

#define PACKET_SIZE 7696
#define PAYLOAD_SIZE 7680

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // identifying code for antenna number
  unsigned int ant_code;

  // number of antenna
  unsigned int nant;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

  size_t pkt_size;

  size_t data_size;

  int antenna;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void integrate_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void dump_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void print_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void print_detected_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void plot_packet (udpplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpplot [options]\n"
     " -a nant        antenna to plot\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpplot_reset(ctx);
}

int udpplot_reset (udpplot_t * ctx)
{
  unsigned ichan;
  for (ichan=0; ichan < ctx->nchan; ichan++)
    ctx->x_points[ichan] = (float) ichan;

  unsigned iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan);
    for (ichan=0; ichan < ctx->nchan; ichan++)
      ctx->y_points[iant][ichan] = 0;
  }
  ctx->num_integrated = 0;
}

int udpplot_destroy (udpplot_t * ctx)
{
  unsigned int iant;
  for (iant=0; iant<ctx->nant; iant++)
  {
    if (ctx->y_points[iant])
      free(ctx->y_points[iant]);
    ctx->y_points[iant] = 0;
  }

  if (ctx->y_points)
    free(ctx->y_points);
  ctx->y_points = 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;

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

int udpplot_init (udpplot_t * ctx)
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
  //if (ctx->data_size == PAYLOAD_SIZE)
  //  mopsr_decode_v2 (ctx->sock->buf, &hdr);
  //else
  mopsr_decode (ctx->sock->buf, &hdr);

  multilog (ctx->log, LOG_INFO, "init: nchan=%u nant=%u nframe=%u\n", hdr.nchan, hdr.nant, hdr.nframe);

  ctx->nchan = hdr.nchan;
  ctx->nant = hdr.nant;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan);
  }

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
  udpplot_t udpplot;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int plot_log = 0;

  double sleep_time = 1;

  char zap_dc = 0;

  int antenna_to_plot = -1;

  while ((arg=getopt(argc,argv,"a:D:hi:lp:s:vz")) != -1) {
    switch (arg) {

    case 'a':
      antenna_to_plot = atoi(optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'h':
      usage();
      return 0;
      
    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'l':
      plot_log = 1;
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      sleep_time = (double) atof (optarg);
      break;

    case 'v':
      verbose++;
      break;

    case 'z':
      zap_dc = 1;
      break;

    default:
      usage ();
      return 0;
      
    }
  }

  multilog_t* log = multilog_open ("mopsr_udpplot", 0);
  multilog_add (log, stderr);

  udpplot.log = log;
  udpplot.verbose = verbose;

  udpplot.interface = strdup(interface);
  udpplot.port = port;
  udpplot.ant_code = 0;
  udpplot.antenna = antenna_to_plot;

  udpplot.num_integrated = 0;
  udpplot.to_integrate = 128;

  udpplot.plot_log = plot_log;
  if (plot_log)
    udpplot.ymin = 1;
  else
    udpplot.ymin = 0;
  udpplot.ymax = -100000;

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_udpplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // allocate require resources, open socket
  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpplot: init()\n");
  if (udpplot_init (&udpplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpplot: prepare()\n");
  udpplot_prepare (&udpplot);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  udpplot_t * ctx = &udpplot;

  mopsr_hdr_t hdr;
  unsigned i;
  char mgt_locks[17];
  char mgt_locks_long[17];

  StartTimer(&wait_sw);

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpplot: while(!quit)\n");
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

      mopsr_decode (ctx->sock->buf, &hdr);
      //mopsr_decode_v2 (ctx->sock->buf, &hdr);

      for (i=0; i<16; i++)
        mgt_locks[i] = ((char) mopsr_get_bit_from_16 (hdr.mgt_locks, i)) + '0';
      mgt_locks[16] = '\0';

      for (i=0; i<16; i++)
        mgt_locks_long[i] = ((char) mopsr_get_bit_from_16 (hdr.mgt_locks_long, i)) + '0';
      mgt_locks_long[16] = '\0';

     if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "main: seq=%"PRIu64" nbit=%u nant=%u nchan=%u start_chan=%u nframe=%u locks=[%s] locks_long=[%s]\n", hdr.seq_no, hdr.nbit, hdr.nant, hdr.nchan, hdr.start_chan, hdr.nframe, mgt_locks, mgt_locks_long);

      //mopsr_encode (ctx->sock->buf, &hdr);
      //mopsr_decode (ctx->sock->buf, &hdr);

      for (i=0; i<16; i++)
        mgt_locks[i] = ((char) mopsr_get_bit_from_16 (hdr.mgt_locks, i)) + '0';
      mgt_locks[16] = '\0';
      for (i=0; i<16; i++)
        mgt_locks_long[i] = ((char) mopsr_get_bit_from_16 (hdr.mgt_locks_long, i)) + '0';
      mgt_locks_long[16] = '\0';

      //multilog (ctx->log, LOG_INFO, "main: seq=%"PRIu64" nbit=%u nant=%u nchan=%u start_chan=%u nframe=%u locks=[%s] locks_long=[%s]\n", hdr.seq_no, hdr.nbit, hdr.nant, hdr.nchan, hdr.start_chan, hdr.nframe, mgt_locks, mgt_locks_long);
      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", hdr.seq_no, (hdr.seq_no - prev_seq_no));
      prev_seq_no = hdr.seq_no;

      // integrate packet into totals
      integrate_packet (ctx, ctx->sock->buf + UDP_HEADER, ctx->pkt_size - UDP_HEADER);
      ctx->num_integrated++;

      if (ctx->verbose)
      {
        print_detected_packet (ctx, ctx->sock->buf + UDP_HEADER, (ctx->nchan * ctx->nant * 2));
        print_packet (ctx, ctx->sock->buf + UDP_HEADER, (10 * ctx->nant * 2));
        dump_packet (ctx, ctx->sock->buf + UDP_HEADER, ctx->pkt_size - UDP_HEADER);
      }

      if (ctx->num_integrated >= ctx->to_integrate)
      {
        plot_packet (ctx);
        udpplot_reset (ctx);
        DelayTimer(&wait_sw, sleep_time);
        StartTimer(&wait_sw);
      }

      DelayTimer(&wait_sw, sleep_time);

      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
      size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);

    }
  }

  cpgclos();

  return EXIT_SUCCESS;

}

void print_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  char ant0r[9];
  char ant0i[9];
  char ant1r[9];
  char ant1i[9];

  unsigned ibyte = 0;
  fprintf(stderr, "chan\tant0real ant0imag ant1real ant1imag\n");
  for (ibyte=0; ibyte<size; ibyte += 4)
  {
    char_to_bstring (ant0r, buffer[ibyte+0]);
    char_to_bstring (ant0i, buffer[ibyte+1]);
    char_to_bstring (ant1r, buffer[ibyte+2]);
    char_to_bstring (ant1i, buffer[ibyte+3]);
    fprintf (stderr, "%d\t%s %s %s %s\t%d\t%d\t%d\t%d\n", ibyte/4, ant0r, ant0i, ant1r, ant1i, (int8_t) buffer[ibyte+0], (int8_t) buffer[ibyte+1], (int8_t) buffer[ibyte+2], (int8_t) buffer[ibyte+3]);
  }
}

void print_detected_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  int8_t * ptr = (int8_t *) buffer;

  unsigned int ichan, iant;
  multilog (ctx->log, LOG_INFO, "[chan][ant] = [sqld]\n");
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      int re = (int) ptr[0];
      int im = (int) ptr[1];
      float po = (float) ((re*re) + (im*im));
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "[%d][%d] = %f\n", ichan, iant, po);
      ptr += 2;
    }
  }
}


void dump_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open ("packet.raw", flags, perms);
  ssize_t n_wrote = write (fd, buffer, size);
  close (fd);
}

void integrate_packet (udpplot_t * ctx, char * buffer, unsigned int size)
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
        re = ((float) in[0]) + 0.5;
        im = ((float) in[1]) + 0.5;
        ctx->y_points[iant][ichan] += ((re*re) + (im*im));
        in += 2;
      }
    }
  }
}

void plot_packet (udpplot_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float xmin = 0.0;
  float xmax = (float) ctx->nchan;

  // calculate limits
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna == -1) || (mopsr_get_new_ant_number (iant) == ctx->antenna))
    {
      for (ichan=0; ichan < ctx->nchan; ichan++)
      {
        if (ctx->plot_log)
          ctx->y_points[iant][ichan] = (ctx->y_points[iant][ichan] > 0) ? log10(ctx->y_points[iant][ichan]) : 0;
        if (ctx->y_points[iant][ichan] > ctx->ymax) ctx->ymax = ctx->y_points[iant][ichan];
        if (ctx->y_points[iant][ichan] < ctx->ymin) ctx->ymin = ctx->y_points[iant][ichan];
      }
    }
  }
  //if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->ymin=%f, ctx->ymax=%f\n", ctx->ymin, ctx->ymax);

  cpgbbuf();
  cpgeras();

  unsigned nx = 4;
  unsigned ny = 4;
  unsigned ix, iy;
  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      iant = iy * nx + ix;

      float xl = 0.1 + 0.2f * ix;
      float xr = xl + 0.2;

      float yb = 0.1 + 0.2 * iy;
      float yt = yb + 0.2;

      cpgsci(1);
      cpgswin(xmin, xmax, ctx->ymin, 1.1 * ctx->ymax);
      cpgsvp(xl, xr, yb, yt);
      if (iy == 0)
        if (ix == 0)
          if (ctx->plot_log)
            cpgbox("BCNT", 0.0, 0.0, "BCLNSTV", 0.0, 0.0);
          else
            cpgbox("BCNT", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
        else
          if (ctx->plot_log)
            cpgbox("BCNT", 0.0, 0.0, "BCLST", 0.0, 0.0);
          else
            cpgbox("BCNT", 0.0, 0.0, "BCST", 0.0, 0.0);
      else
        if (ix == 0)
          if (ctx->plot_log)
            cpgbox("BCT", 0.0, 0.0, "BCLNSTV", 0.0, 0.0);
          else
            cpgbox("BCT", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
        else
          if (ctx->plot_log)
            cpgbox("BCT", 0.0, 0.0, "BCLST", 0.0, 0.0);
          else
            cpgbox("BCT", 0.0, 0.0, "BCST", 0.0, 0.0);

      char ant_label[8];
      sprintf(ant_label, "Ant %u", mopsr_get_new_ant_number (iant));
      cpgmtxt("T", -1.0, 0.05, 0.0, ant_label);
      cpgsci(2);
      cpgline(ctx->nchan, ctx->x_points, ctx->y_points[iant]);
      cpgsci(1);
    }
  }
  cpgebuf();
}
