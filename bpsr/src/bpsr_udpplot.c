/*
 * bpsr_udpplot
 *
 * Simply listens on a specified port for udp packets encoded
 * in the BPSRformat
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
#include <float.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "bpsr_def.h"
#include "bpsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#include "stopwatch.h"

typedef struct {

  multilog_t * log;

  bpsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // number of pols
  unsigned int npol;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  unsigned int nbin;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float ** hists;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void print_raw (udpplot_t * ctx, char * buffer);
void integrate_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void dump_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void print_y_points (udpplot_t * ctx);
void plot_packet (udpplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "bpsr_udpplot [options]\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -p port        port on which to listen [default %d]\n"
     " -t num         number of packets to integrate into each plot [default 1]\n"
     " -s secs        sleep this many seconds between plotting [default 2]\n"
     " -v             verbose messages\n",
     BPSR_DEFAULT_UDPDB_PORT);
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "bpsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, BPSR_UDP_4POL_PAYLOAD_BYTES);

  udpplot_reset(ctx);
}

int udpplot_reset (udpplot_t * ctx)
{
  unsigned ichan;
  for (ichan=0; ichan < ctx->nchan; ichan++)
    ctx->x_points[ichan] = (float) ichan;

  unsigned ipol, ibin;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      ctx->y_points[ipol][ichan] = 0;
    }
    for (ibin=0; ibin<ctx->nbin; ibin++)
      ctx->hists[ipol][ibin] = 0;
  }
  ctx->num_integrated = 0;
}

int udpplot_destroy (udpplot_t * ctx)
{
  unsigned int ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    if (ctx->y_points[ipol])
      free(ctx->y_points[ipol]);
    if (ctx->y_points[ipol])
      free(ctx->hists[ipol]);
    ctx->y_points[ipol] = 0;
    ctx->hists[ipol] = 0;
  }

  if (ctx->y_points)
    free(ctx->y_points);
  ctx->y_points = 0;

  if (ctx->hists)
    free(ctx->hists);
  ctx->hists= 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;

  if (ctx->sock)
  {
    close(ctx->sock->fd);
    bpsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
}

/*
 * Close the udp socket and file
 */

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "bpsr_udpdb_init_receiver()\n");

  // create a socket which can hold variable num of UDP packets
  ctx->sock = bpsr_init_sock();

  // open socket
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: creating udp socket on %s:%d\n", ctx->interface, ctx->port);
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 4 MB
  int sock_buf_size = 4*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  // y_points stores PP, QQ, PQ_re, PQ_Im
  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan);
  ctx->y_points = (float **) malloc(sizeof(float *) * 4);
  ctx->hists = (float **) malloc(sizeof(float *) * 4);
  unsigned int ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    ctx->y_points[ipol] = (float *) malloc (sizeof(float) * ctx->nchan);
    ctx->hists[ipol] = (float *) malloc (sizeof(float) * ctx->nbin);
  }

  return 0;
}



int main (int argc, char **argv)
{

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = BPSR_DEFAULT_UDPDB_PORT;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  char * device = "/xs";

  /* actual struct with info */
  udpplot_t udpplot;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nchan = 1024;

  unsigned int nbin = 256;

  unsigned int to_integrate = 1;

  unsigned int plot_log = 0;

  double sleep_time = 2000000;

  while ((arg=getopt(argc,argv,"b:D:i:ln:p:s:t:vh")) != -1) {
    switch (arg) {

    case 'b':
      nbin = atoi(optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'l':
      plot_log = 1;
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 's':
      sleep_time = (double) atof (optarg);
      sleep_time *= 1000000;
      break;

    case 't':
      to_integrate = atoi (optarg);
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

  multilog_t* log = multilog_open ("bpsr_udpplot", 0);
  multilog_add (log, stderr);

  udpplot.log = log;
  udpplot.verbose = verbose;

  udpplot.interface = strdup(interface);
  udpplot.port = port;
  udpplot.npol = 4;
  udpplot.nchan = nchan;
  udpplot.nbin = nbin;
  udpplot.to_integrate = to_integrate;

  udpplot.plot_log = plot_log;
  udpplot.ymin = 100000;
  udpplot.ymax = -100000;

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "bpsr_dbplot: using device %s\n", device);

  // allocate require resources, open socket
  if (udpplot_init (&udpplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  udpplot_prepare (&udpplot);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  uint64_t acc_len = 25;
  uint64_t sequence_incr = 512 * acc_len;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  udpplot_t * ctx = &udpplot;

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, BPSR_UDP_4POL_PAYLOAD_BYTES, 0, NULL, NULL );

      if (got == BPSR_UDP_4POL_PAYLOAD_BYTES)
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
        multilog (log, LOG_ERR, "main: received %d bytes, expected %d\n", got, BPSR_UDP_4POL_PAYLOAD_BYTES);
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

      seq_no = decode_header (ctx->sock->buf);
      seq_no /= sequence_incr;

      if (ctx->verbose) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // integrate packet into totals
      integrate_packet (ctx, ctx->sock->buf + BPSR_UDP_4POL_HEADER_BYTES, BPSR_UDP_4POL_DATASIZE_BYTES);
      ctx->num_integrated++;

      if (ctx->verbose)
      {
        print_y_points(ctx);
      }

      if (ctx->num_integrated >= ctx->to_integrate)
      {
        plot_packet (ctx);
        udpplot_reset (ctx);
        DelayTimer(&wait_sw, sleep_time);
        StopTimer(&wait_sw);
      }

      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
      size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, BPSR_UDP_4POL_PAYLOAD_BYTES);
      if (ctx->verbose)
        multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
    }
  }

  return EXIT_SUCCESS;

}

void print_y_points (udpplot_t * ctx)
{
  unsigned int ichan, ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    fprintf (stderr, "pol=%d [", ipol);
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      fprintf (stderr, " %f", ctx->y_points[ipol][ichan]);
    }
    fprintf (stderr, "]\n");
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

void print_raw (udpplot_t * ctx, char * buffer)
{
  fprintf (stderr, "buffer[0]=%d\n", buffer[0]);
  fprintf (stderr, "buffer[1]=%d\n", buffer[1]);
  fprintf (stderr, "buffer[2]=%d\n", buffer[2]);
  fprintf (stderr, "buffer[3]=%d\n", buffer[3]);
  fprintf (stderr, "buffer[4]=%d\n", buffer[4]);
  fprintf (stderr, "buffer[5]=%d\n", buffer[5]);
  fprintf (stderr, "buffer[6]=%d\n", buffer[6]);
  fprintf (stderr, "buffer[7]=%d\n", buffer[7]);
}

void integrate_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "integrate_packet()\n");

  unsigned ichan = 0;

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "integrate_packet: assigning floats, checking min/max\n");

  int pp[2];
  int qq[2];
  int pp_re[2];
  int qq_re[2];

  uint8_t * unsigned_ints = (uint8_t *) buffer;
  int8_t *  signed_ints   = (int8_t *)  buffer + 4;

  // form power spectra for each qty
  for (ichan=0; ichan<ctx->nchan; ichan+=2)
  {
    // pp 
    ctx->y_points[0][ichan]    += (float) unsigned_ints[0];
    ctx->y_points[0][ichan+1]  += (float) unsigned_ints[1];

    // qq
    ctx->y_points[1][ichan]    += (float) unsigned_ints[2];
    ctx->y_points[1][ichan+1]  += (float) unsigned_ints[3];

    // pq_re
    ctx->y_points[2][ichan]    += (float) signed_ints[0];
    ctx->y_points[2][ichan+1]  += (float) signed_ints[2];

    // pq_im
    ctx->y_points[3][ichan]    += (float) signed_ints[1];
    ctx->y_points[3][ichan+1]  += (float) signed_ints[3];

    if ((ichan > 160) && (ichan < 900))
    {
      ctx->hists[0][(uint8_t) unsigned_ints[0]]++;
      ctx->hists[0][(uint8_t) unsigned_ints[1]]++;
      ctx->hists[1][(uint8_t) unsigned_ints[2]]++;
      ctx->hists[1][(uint8_t) unsigned_ints[3]]++;
      ctx->hists[2][(uint8_t) (signed_ints[0] + 128)]++;
      ctx->hists[2][(uint8_t) (signed_ints[2] + 128)]++;
      ctx->hists[3][(uint8_t) (signed_ints[1] + 128)]++;
      ctx->hists[3][(uint8_t) (signed_ints[3] + 128)]++;
    }

    unsigned_ints += 8;
    signed_ints += 8;
  }

}

void plot_packet (udpplot_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  unsigned ichan = 0;
  unsigned ipol= 0;
  unsigned iframe = 0;
  float xmin = 0.0;
  float xmax = (float) ctx->nchan;

  ctx->ymin =  1000000;
  ctx->ymax = -1000000;

  // zap chan 0
  for (ipol=0; ipol < 4; ipol++)
  {
    ctx->y_points[ipol][0] = 0;
  }
  // calculate limits
  for (ipol=0; ipol < 2; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      if (ctx->plot_log)
        ctx->y_points[ipol][ichan] = (ctx->y_points[ipol][ichan] > 0) ? log10(ctx->y_points[ipol][ichan]) : 0;
      if (ctx->y_points[ipol][ichan] > ctx->ymax) ctx->ymax = ctx->y_points[ipol][ichan];
      if (ctx->y_points[ipol][ichan] < ctx->ymin) ctx->ymin = ctx->y_points[ipol][ichan];
    }
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->ymin=%f, ctx->ymax=%f\n", ctx->ymin, ctx->ymax);

  if (cpgopen("1/xs") != 1) 
  {
    multilog(ctx->log, LOG_INFO, "bpsr_dbplot: error opening plot device [1/xs]\n");
    exit(1);
  }
  cpgask(0);
  cpgbbuf();
  cpgsci(1);
  if (ctx->plot_log)
  {
    cpgenv(xmin, xmax, ctx->ymin, 1.1 * ctx->ymax, 0, 20);
    cpglab("Channel", "log\\d10\\u(Power)", "Bandpass"); 
  }
  else
  {
    cpgenv(xmin, xmax, ctx->ymin, 1.1*ctx->ymax, 0, 0);
    cpglab("Channel", "Power", "Bandpass"); 
  }

  char pol_label[8];
  for (ipol=0; ipol < 2; ipol++)
  {
    sprintf(pol_label, "Pol %u", ipol);
    cpgsci(ipol + 2);
    cpgmtxt("T", 1.5 + (1.0 * ipol), 0.0, 0.0, pol_label);
    cpgline(ctx->nchan, ctx->x_points, ctx->y_points[ipol]);
  }
  cpgebuf();
  cpgclos();

  if (cpgopen("2/xs") != 1) 
  {
    multilog(ctx->log, LOG_INFO, "bpsr_dbplot: error opening plot device [2/xs]\n");
    exit(1);
  }

  ctx->ymin = 1000000;
  ctx->ymax = -1000000;
  for (ipol=2; ipol<4; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      if (ctx->y_points[ipol][ichan] > ctx->ymax) ctx->ymax = ctx->y_points[ipol][ichan];
      if (ctx->y_points[ipol][ichan] < ctx->ymin) ctx->ymin = ctx->y_points[ipol][ichan];
    }
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->ymin=%f, ctx->ymax=%f\n", ctx->ymin, ctx->ymax);

  cpgask(0);
  cpgbbuf();
  cpgsci(1);
  
  cpgenv(xmin, xmax, (ctx->ymin*1.1) - 0.1, (ctx->ymax*1.1) + 0.1, 0, 0);
  cpglab("Channel", "Power", "Complex Power Spectrum"); 

  for (ipol=2; ipol<4; ipol++)
  {
    sprintf(pol_label, "Cross Pol %u", ipol - 2);
    cpgsci(ipol);
    cpgmtxt("T", 1.5 + (1.0 * (ipol-2)), 0.0, 0.0, pol_label);
    cpgline(ctx->nchan, ctx->x_points, ctx->y_points[ipol]);
  }
  cpgebuf();
  cpgclos();

  if (cpgopen("3/xs") != 1)
  {
    multilog(ctx->log, LOG_INFO, "bpsr_dbplot: error opening plot device [3/xs]\n");
    exit(1);
  }

  float xmin1 = -1;
  float xmax1 = 256;
  float xmin2 = -129;
  float xmax2 = 128;
  float ymax = -FLT_MAX;

  float * xbins1 = (float *) malloc(sizeof(float) * ctx->nbin);
  float * xbins2 = (float *) malloc(sizeof(float) * ctx->nbin);
  int ibin;
  for (ibin=0; ibin<ctx->nbin; ibin++)
  {
    xbins1[ibin] = (float) ibin;
    xbins2[ibin] = (float) ibin - 128;

    for (ipol=0; ipol<ctx->npol; ipol++)
    {
      if (ctx->hists[ipol][ibin] > ymax)
        ymax = ctx->hists[ipol][ibin];
    }
  }

  cpgbbuf();
  cpgsci(1);

  cpgswin(xmin1, xmax1, 0, ymax);
  cpgsvp(0.05, 0.95, 0.55, 0.95);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Count", "Histograms [all channels except DC]");

  cpgsci(2);
  cpgslw(3);
  cpgbin (ctx->nbin, xbins1, ctx->hists[0], 0);
  cpgslw(1);
  cpgsci(1);

  cpgsci(3);
  cpgslw(3);
  cpgbin (ctx->nbin, xbins1, ctx->hists[1], 0);
  cpgslw(1);
  cpgsci(1);

  cpgswin(xmin2, xmax2, 0, ymax);
  cpgsvp(0.05, 0.95, 0.05, 0.45);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Count", "");

  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax);
  cpgsls(1);

  cpgsci(2);
  cpgslw(3);
  cpgbin (ctx->nbin, xbins2, ctx->hists[2], 0);
  cpgslw(1);
  cpgsci(1);

  cpgsci(3);
  cpgslw(3);
  cpgbin (ctx->nbin, xbins2, ctx->hists[3], 0);
  cpgslw(1);
  cpgsci(1);


  


  cpgebuf();
  cpgclos();

  free (xbins1);
  free (xbins2);
}

