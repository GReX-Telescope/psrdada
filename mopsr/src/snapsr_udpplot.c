/*
 * snapsr_udpplot
 *
 * Simply listens on a specified port for udp packets encoded
 * in the SNAPSR format
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
#include "snapsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"
#include "stopwatch.h"

typedef struct {

  multilog_t * log;

  snapsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // number of subbands
  unsigned nsubband;

  // subband to acquire/plot [-1 means all]
  int subband_id;

  // antenna to acquire/plot [-1 means all]
  int antenna_id; 

  // number of snap boards transmitting
  unsigned nsnap;

  // snap board to receive packets from
  int snap_id;

  // total number of antenna across all snap boards
  unsigned int nant;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // number of bins in histogram
  unsigned int nbin;

  // size of the UDP packet
  unsigned int resolution;

  // time to sleep between each plot in us
  double sleep_time;

  unsigned int verbose;

  float * x_points;

  float ** y_points;
  
  // histograms [ant][dim][bin]
  float *** hist;

  // histogram bins[bin]
  float * bins;

  uint64_t num_integrated;

  uint64_t to_integrate;

  char plot_log;

  char zap_dc;

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
void plot_bandpass (udpplot_t * ctx);
void plot_histogram (udpplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "snapsr_udpplot [options]\n"
     " -a ant         plot only the specified antenna\n"
     " -b subband     plot only the specified subband\n"
     " -c snap        plot only the specified snap\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t npkt        number of packets to integrate [default 32]\n"
     " -v             verbose messages\n",
     SNAPSR_DEFAULT_UDP_PORT);
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "snapsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpplot_reset(ctx);
  return 0;
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

  unsigned idim, ibin;
  for (iant=0; iant < ctx->nant; iant++)
    for (idim=0; idim<ctx->ndim; idim++)
      for (ibin=0; ibin<ctx->nbin; ibin++)
        ctx->hist[iant][idim][ibin] = 0;
        
  ctx->num_integrated = 0;
  return 0;
}

int udpplot_destroy (udpplot_t * ctx)
{
  unsigned iant, ichan, idim;
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

  if (ctx->hist)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      for (idim=0; idim<ctx->ndim; idim++)
      {
        free (ctx->hist[iant][idim]);
        ctx->hist[iant][idim] = 0;
      }
      free (ctx->hist[iant]);
      ctx->hist[iant] = 0;
    }
    free(ctx->hist);
  }
  ctx->hist = 0;

  if (ctx->bins)
  {
    free (ctx->bins);
  }
  ctx->bins = 0;

  if (ctx->sock)
  {
    close(ctx->sock->fd);
    snapsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
  return 0;
}

/*
 * Close the udp socket and file
 */

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "snapsr_udpdb_init_receiver()\n");

  // create a SNAPSR socket
  ctx->sock = snapsr_init_sock();

  // open socket
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: creating udp socket on %s:%d\n", ctx->interface, ctx->port);
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) 
  {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 4 MB
  int sock_buf_size = 4*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  ctx->data_size = SNAPSR_UDP_DATA_BYTES;
  ctx->pkt_size  = SNAPSR_UDP_PAYLOAD_BYTES;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, ctx->data_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock (ctx->sock->fd);
  
  // total number of channels (across all subbands)
  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan);

  // total number of antenna (across all snaps)
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  unsigned iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan);
  }

  multilog (ctx->log, LOG_INFO, "init: nant=%d ndim=%d nbin=%d\n", ctx->nant, ctx->ndim, ctx->nbin);
  unsigned idim;
  ctx->hist = (float ***) malloc (sizeof (float **) * ctx->nant);
  for (iant=0; iant<ctx->nant; iant++)
  {
    ctx->hist[iant] = (float **) malloc (sizeof (float *) * ctx->ndim);
    for (idim=0; idim<ctx->ndim; idim++)
    {
      ctx->hist[iant][idim] = (float *) malloc (sizeof (float) * ctx->nbin);
    }
  }

  unsigned ibin;
  ctx->bins = (float *) malloc (sizeof(float) * ctx->nbin);
  for (ibin=0; ibin<ctx->nbin; ibin++)
    ctx->bins[ibin] = ((float) ibin) - 127;

  return 0;
}



int main (int argc, char **argv)
{
  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = SNAPSR_DEFAULT_UDP_PORT;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  char * device = "/xs";

  // actual struct with info
  udpplot_t udpplot;

  udpplot_t * ctx = &udpplot;

  // defaults
  ctx->plot_log = 0;
  ctx->zap_dc = 0;
  ctx->nsubband = SNAPSR_NSUBBAND;
  ctx->subband_id = -1;
  ctx->nchan = ctx->nsubband * SNAPSR_NCHAN_PER_SUBBAND;
  ctx->nant = SNAPSR_NANT_PER_PACKET;
  ctx->antenna_id = -1;
  ctx->snap_id = -1;
  ctx->nsnap = 1;
  ctx->sleep_time = 1;
  ctx->to_integrate = 32;
  ctx->verbose = 0;
  ctx->nbin = 256;
  ctx->ndim = 2;

  while ((arg=getopt(argc,argv,"a:b:c:D:hi:lp:s:t:vz")) != -1) 
  {
    switch (arg) {

    case 'a':
      ctx->antenna_id = atoi(optarg);
      ctx->nant = 12;
      break;

    case 'b':
      ctx->subband_id = atoi(optarg);
      //ctx->nsubband = 1;
      //ctx->nchan = SNAPSR_NCHAN_PER_SUBBAND;
      break;

    case 'c':
      ctx->snap_id = atoi(optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'h':
      usage();
      return 0;
      
    case 'i':
      ctx->interface = strdup(optarg);
      break;

    case 'l':
      ctx->plot_log = 1;
      break;

    case 'p':
      ctx->port = atoi (optarg);
      break;

    case 's':
      ctx->sleep_time = (double) atof (optarg);
      break;

    case 't':
      ctx->to_integrate = atoi(optarg);
      break;

    case 'v':
      ctx->verbose++;
      break;

    case 'z':
      ctx->zap_dc = 1;
      break;

    default:
      usage ();
      return 0;
      
    }
  }

  fprintf (stderr, "port=%d\n", ctx->port);

  multilog_t* log = multilog_open ("snapsr_udpplot", 0);
  multilog_add (log, stderr);

  ctx->log = log;

  ctx->num_integrated = 0;
  if (ctx->plot_log)
    ctx->ymin = 1;
  else
    ctx->ymin = 0;
  ctx->ymax = 1;

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "snapsr_udpplot: using device %s\n", device);

  // allocate require resources, open socket
  if (verbose)
    multilog(log, LOG_INFO, "snapsr_udpplot: init()\n");
  if (udpplot_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  if (verbose)
    multilog(log, LOG_INFO, "snapsr_udpplot: prepare()\n");
  udpplot_prepare (ctx);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  snapsr_hdr_t hdr;
  unsigned i;

  StartTimer(&wait_sw);

  if (verbose)
    multilog(log, LOG_INFO, "snapsr_udpplot: while(!quit)\n");
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
      snapsr_decode (ctx->sock->buf, &(ctx->sock->hdr));

      if (ctx->verbose)
        snapsr_print_header (&(ctx->sock->hdr));

      char * data = ctx->sock->buf + SNAPSR_UDP_HEADER_BYTES;

      // integrate packet into totals
      integrate_packet (ctx, data, SNAPSR_UDP_DATA_BYTES);

      ctx->num_integrated++;

      if (ctx->verbose > 1)
      {
        print_detected_packet (ctx, data, ctx->data_size);
        print_packet (ctx, data, ctx->data_size);
        //dump_packet (ctx, ctx->sock->buf + UDP_HEADER, ctx->pkt_size - UDP_HEADER);
      }

      if (ctx->num_integrated >= ctx->to_integrate)
      {
        plot_bandpass (ctx);
        plot_histogram (ctx);
        udpplot_reset (ctx);
        DelayTimer(&wait_sw, ctx->sleep_time);
        StartTimer(&wait_sw);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
      }

    }
  }

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
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "integrate_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "integrate_packet: assigning floats, checking min/max: nframe=%d\n", ctx->sock->hdr.nframe);

  int8_t * in = (int8_t *) buffer;
#ifdef INTERPRET_16BIT
  uint16_t in16;
#endif
  float re, im;
  int re_bin, im_bin;

  char bstring1[9];
  char bstring2[9];

  for (iframe=0; iframe<ctx->sock->hdr.nframe; iframe++)
  {
    for (ichan=0; ichan<ctx->sock->hdr.nchan; ichan++)
    {
      const unsigned ochan = ctx->sock->hdr.subband_id * SNAPSR_NCHAN_PER_SUBBAND + ichan;
      for (iant=0; iant<ctx->sock->hdr.nant; iant++)
      {
        const unsigned oant = ctx->sock->hdr.snap_id * SNAPSR_NANT_PER_PACKET + iant;
        if ( (ctx->antenna_id == -1 || ctx->antenna_id == oant) &&
             (ctx->snap_id == -1 || ctx->snap_id == ctx->sock->hdr.snap_id) && 
             (ctx->subband_id == -1 || ctx->subband_id == ctx->sock->hdr.subband_id) )
        {
#ifdef INTERPRET_16BIT
          in16 = 0;
          uint16_t tmp = in[0];
          in16 |= (tmp << 8);
          tmp = in[1];
          in16 |= (tmp & 0xff);
#else
          //re = ((float) in[0]) + 0.5;
          //im = ((float) in[1]) + 0.5;
          re = ((float) in[0]);
          im = ((float) in[1]);
          ctx->y_points[oant][ochan] += ((re*re) + (im*im));

          re_bin = ((int) re) + 127;
          im_bin = ((int) im) + 127;

          if (re_bin < 0) re_bin = 0;
          if (im_bin < 0) im_bin = 0;
          if (re_bin > 255) re_bin = 255;
          if (im_bin > 255) im_bin = 255;

          ctx->hist[oant][0][re_bin]++;
          ctx->hist[oant][1][im_bin]++;
#endif
        }
        in += 2;
      }
    }
  }
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "integrate_packet: done\n");
}

void plot_bandpass (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_bandpass()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float xmin = 0.0;
  float xmax = (float) ctx->nchan;

  if (ctx->verbose)
  {
    multilog (ctx->log, LOG_INFO, "plot_bandpass nant=%u nchan=%u\n", ctx->nant, ctx->nchan);
    multilog (ctx->log, LOG_INFO, "plot_bandpass antenna_id=%d snap_id=%d subband_id=%d\n", ctx->antenna_id, ctx->snap_id, ctx->subband_id);
  }

  // calculate limits
  for (iant=0; iant < ctx->nant; iant++)
  {
    if (ctx->antenna_id == -1  || ctx->antenna_id == iant)
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

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_bandpass: ctx->ymin=%f, ctx->ymax=%f\n", ctx->ymin, ctx->ymax);

  if (cpgbeg (0, "1/xs", 1, 1) != 1)
  {
    multilog(ctx->log, LOG_INFO, "snapsr_udpplot: error opening plot device 1/xs\n");
    exit(1);
  }

  cpgbbuf();

  unsigned nx = 4;
  unsigned ny = ctx->nant / 4;
  float ywidth = (1.0 - 0.2) / (float) ny;
  unsigned ix, iy;
  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      iant = iy * nx + ix;

      float xl = 0.1 + 0.2f * ix;
      float xr = xl + 0.2;

      float yb = 0.1 + ywidth * iy;
      float yt = yb + ywidth;

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
      sprintf(ant_label, "Ant %u", iant);
      cpgmtxt("T", -1.0, 0.05, 0.0, ant_label);
      cpgsci(2);
      cpgline(ctx->nchan, ctx->x_points, ctx->y_points[iant]);
      cpgsci(1);
    }
  }
  cpgebuf();
  cpgend();
}

void plot_histogram (udpplot_t * ctx)
{
  int iant, idim, ibin;

  float xmin = -127;
  float xmax = 128;
  float ymin = 0;
  float ymax = 0;
  char all_zero = 0;

  // determine min/max of the distribution
  for (iant=0; iant < ctx->nant; iant++)
  {
    for (idim=0; idim<ctx->ndim; idim++)
    {
      for (ibin=0; ibin<ctx->nbin; ibin++)
      {
        if (ibin != 127)
        {
          const uint64_t v = ctx->hist[iant][idim][ibin];
          if (v > ymax)
            ymax = v;
          if (v > 0)
            all_zero = 1;
        }
      }
    }
  }

  if (cpgbeg (0, "2/xs", 1, 1) != 1)
  {
    multilog(ctx->log, LOG_INFO, "snapsr_udpplot: error opening plot device 2/xs\n");
    exit(1);
  }

  char ant_label[8];
  unsigned nx = 4;
  unsigned ny = ctx->nant / 4;
  float ywidth = (1.0 - 0.2) / (float) ny;
  unsigned ix, iy;
  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      iant = iy * nx + ix;

      float xl = 0.1 + 0.2f * ix;
      float xr = xl + 0.2;

      float yb = 0.1 + ywidth * iy;
      float yt = yb + ywidth;

      cpgbbuf();
      cpgsci(1);
      cpgswin(xmin, xmax, ymin, 1.1 * ymax);
      cpgsvp(xl, xr, yb, yt);

      if (iy == 0)
        if (ix == 0)
          cpgbox("BCNT", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
        else
          cpgbox("BCNT", 0.0, 0.0, "BCST", 0.0, 0.0);
      else
        if (ix == 0)
          cpgbox("BCT", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
        else
          cpgbox("BCT", 0.0, 0.0, "BCST", 0.0, 0.0);

      sprintf(ant_label, "Ant %u", iant);
      cpgmtxt("T", -1.0, 0.05, 0.0, ant_label);
      cpgsci(2);

      // draw dotted line for the centre of the distribution
      cpgsls(2);
      cpgslw(2);
      cpgmove (0, 0);
      cpgdraw (0, ymax);
      cpgsls(1);

      cpgsci(2);
      cpgslw(3);
      cpgbin (ctx->nbin, ctx->bins, ctx->hist[iant][0], 0);
      cpgslw(1);
      cpgsci(1);

      cpgslw(3);
      cpgsci(3);
      cpgbin (ctx->nbin, ctx->bins, ctx->hist[iant][1], 0);
      cpgsci(1);
      cpgslw(1);
    }
  }

  cpgebuf();
  cpgend();
}

