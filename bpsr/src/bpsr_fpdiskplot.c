/*
 * bpsr_fpdiskplot.c
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

#include "dada_hdu.h"
#include "dada_def.h"
#include "bpsr_def.h"
#include "bpsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"


typedef struct {

  multilog_t * log;

  // pgplot device
  char * device;

  // number of pols
  unsigned int npol;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** ypoints;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

} fpdiskplot_t;

int fpdiskplot_init (fpdiskplot_t * ctx);
int fpdiskplot_prepare (fpdiskplot_t * ctx);
int fpdiskplot_destroy (fpdiskplot_t * ctx);

void print_raw (fpdiskplot_t * ctx, char * buffer);
void integrate_frame (fpdiskplot_t * ctx, char * buffer, unsigned int size);
void dump_packet (fpdiskplot_t * ctx, char * buffer, unsigned int size);
void print_ypoints (fpdiskplot_t * ctx);
void plot_packet (fpdiskplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "bpsr_fpdiskplot [options]\n"
     " -h             print help text\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -p port        port on which to listen [default %d]\n"
     " -t num         number of packets to integrate into each plot [default 1]\n"
     " -s secs        sleep this many seconds between plotting [default 2]\n"
     " -v             verbose messages\n",
     BPSR_DEFAULT_UDPDB_PORT);
}

int fpdiskplot_prepare (fpdiskplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "bpsr_udpdb_prepare()\n");

  fpdiskplot_reset(ctx);
}

int fpdiskplot_reset (fpdiskplot_t * ctx)
{
  unsigned ichan;
  for (ichan=0; ichan < ctx->nchan; ichan++)
    ctx->x_points[ichan] = (float) ichan;

  unsigned ipol;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
      ctx->ypoints[ipol][ichan] = 0;
  }
  ctx->num_integrated = 0;
}

int fpdiskplot_destroy (fpdiskplot_t * ctx)
{
  unsigned int ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    if (ctx->ypoints[ipol])
      free(ctx->ypoints[ipol]);
    ctx->ypoints[ipol] = 0;
  }

  if (ctx->ypoints)
    free(ctx->ypoints);
  ctx->ypoints = 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;
}

/*
 * Close the udp socket and file
 */

int fpdiskplot_init (fpdiskplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "bpsr_udpdb_init_receiver()\n");

  // ypoints stores PP, QQ, PQ_re, PQ_Im
  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan);
  ctx->ypoints = (float **) malloc(sizeof(float *) * 4);
  unsigned int ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    ctx->ypoints[ipol] = (float *) malloc (sizeof(float) * ctx->nchan);
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
  fpdiskplot_t fpdiskplot;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nchan = 1024;

  unsigned int to_integrate = 1;

  unsigned int plot_log = 0;

  double sleep_time = 2000000;

  while ((arg=getopt(argc,argv,"D:ln:t:vh")) != -1) {
    switch (arg) {

    case 'D':
      device = strdup(optarg);
      break;

    case 'l':
      plot_log = 1;
      break;

    case 'n':
      nchan = atoi (optarg);
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

  multilog_t* log = multilog_open ("bpsr_fpdiskplot", 0);
  multilog_add (log, stderr);

  fpdiskplot.log = log;
  fpdiskplot.verbose = verbose;

  fpdiskplot.npol = 4;
  fpdiskplot.nchan = nchan;
  fpdiskplot.to_integrate = to_integrate;

  fpdiskplot.plot_log = plot_log;
  fpdiskplot.ymin = 100000;
  fpdiskplot.ymax = -100000;

  if (verbose)
    multilog(log, LOG_INFO, "bpsr_dbplot: using device %s\n", device);

  // allocate require resources, open socket
  if (fpdiskplot_init (&fpdiskplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not init memory\n");
    exit(1);
  }

  // cloear packets ready for capture
  fpdiskplot_prepare (&fpdiskplot);

  fpdiskplot_t * ctx = &fpdiskplot;

  int flags = O_RDONLY;
  char * fname = strdup (argv[optind]);
  int fd = open (fname, flags);

  if (fd < 0)
  {
    fprintf(stderr, "Error opening %s: %s\n", fname, strerror (errno));
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf(stderr, "opened file\n");

  int npol = 4;

  char header[4096];
  read(fd, header, 4096);

  size_t frame_size = 1024 * npol * sizeof(int8_t);
  char buffer[4096];

  // while
  {
    read(fd, buffer, frame_size);

    integrate_frame (ctx, buffer, frame_size);
    ctx->num_integrated++;

    if (ctx->verbose)
    {
      print_ypoints(ctx);
    }

    if (ctx->num_integrated >= ctx->to_integrate)
    {
      plot_packet (ctx);
      fpdiskplot_reset (ctx);
    }
  }

  return EXIT_SUCCESS;

}

void print_ypoints (fpdiskplot_t * ctx)
{
  unsigned int ichan, ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    fprintf (stderr, "pol=%d [", ipol);
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      fprintf (stderr, " %f", ctx->ypoints[ipol][ichan]);
    }
    fprintf (stderr, "]\n");
  }
}


void dump_packet (fpdiskplot_t * ctx, char * buffer, unsigned int size)
{
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open ("packet.raw", flags, perms);
  ssize_t n_wrote = write (fd, buffer, size);
  close (fd);
}

void print_raw (fpdiskplot_t * ctx, char * buffer)
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

void integrate_frame (fpdiskplot_t * ctx, char * buffer, unsigned int size)
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

  for (ichan=0; ichan<ctx->nchan; ichan+=2)
  {
    // pp 
    ctx->ypoints[0][ichan]    += (float) unsigned_ints[0];
    ctx->ypoints[0][ichan+1]  += (float) unsigned_ints[1];

    // qq
    ctx->ypoints[1][ichan]    += (float) unsigned_ints[2];
    ctx->ypoints[1][ichan+1]  += (float) unsigned_ints[3];

    // pq_re
    ctx->ypoints[2][ichan]    += (float) signed_ints[0];
    ctx->ypoints[2][ichan+1]  += (float) signed_ints[2];

    // pq_im
    ctx->ypoints[3][ichan]    += (float) signed_ints[1];
    ctx->ypoints[3][ichan+1]  += (float) signed_ints[3];

    unsigned_ints += 8;
    signed_ints += 8;
  }

}

void plot_packet (fpdiskplot_t * ctx)
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
    ctx->ypoints[ipol][0] = 0;
  }
  // calculate limits
  for (ipol=0; ipol < 2; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      if (ctx->plot_log)
        ctx->ypoints[ipol][ichan] = (ctx->ypoints[ipol][ichan] > 0) ? log10(ctx->ypoints[ipol][ichan]) : 0;
      if (ctx->ypoints[ipol][ichan] > ctx->ymax) ctx->ymax = ctx->ypoints[ipol][ichan];
      if (ctx->ypoints[ipol][ichan] < ctx->ymin) ctx->ymin = ctx->ypoints[ipol][ichan];
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
    cpgline(ctx->nchan, ctx->x_points, ctx->ypoints[ipol]);
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
      if (ctx->ypoints[ipol][ichan] > ctx->ymax) ctx->ymax = ctx->ypoints[ipol][ichan];
      if (ctx->ypoints[ipol][ichan] < ctx->ymin) ctx->ymin = ctx->ypoints[ipol][ichan];
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
    cpgline(ctx->nchan, ctx->x_points, ctx->ypoints[ipol]);
  }
  cpgebuf();
  cpgclos();

}

