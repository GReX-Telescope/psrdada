/*
 * mopsr_udpcorr
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
#include <float.h>
#include <complex.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <math.h>
#include <cpgplot.h>
#include <fftw3.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#include "stopwatch.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // identifying code for antenna's in packet
  unsigned int ant_code;

  // number of antennae
  unsigned int nant;

  // number of FFT points for correlation
  unsigned int npt;

  unsigned int nchan_in;

  // number of output channels
  unsigned int nchan_out;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  unsigned int nsamp;

  // number of FFTs to batch in
  unsigned nbatch;

  size_t buffer_size;
  void * buffer;

  float * fwd1_in;
  float * fwd2_in;

  fftwf_complex * fwd1_out;
  fftwf_complex * fwd2_out;

  complex float * xcorr;
  complex float * xcorr_sum;
  complex float xcorr_fscr;

  complex float * lag;

  float * channels;
  float * amps;
  float * phases;

  float * mean1;
  float * mean2;
  unsigned mean_npt;

  unsigned idump;
  unsigned ndump;
  float * amps_t;
  float * phases_t;
  float * ac1_t;
  float * ac2_t;
  float ac1;
  float ac2;
  float * power1;
  float * power2;

  fftwf_plan plan_fwd;
  fftwf_plan plan_bwd;

  float ymin;

  float ymax;

  size_t pkt_size;

  size_t hdr_size;

  size_t data_size;

  char dsb;

  char zap_dc;

} udpcorr_t;

void signal_handler(int signalValue);
int udpcorr_init (udpcorr_t * ctx);
int udpcorr_prepare (udpcorr_t * ctx);
int udpcorr_destroy (udpcorr_t * ctx);

int udpcorr_acquire (udpcorr_t * ctx);
int udpcorr_unpack (udpcorr_t * ctx);
int udpcorr_extract (udpcorr_t * ctx, unsigned batch, unsigned channel);
int udpcorr_correlate (udpcorr_t * ctx);

void plot_cc_freq (udpcorr_t * ctx, char * title, char * device);
void plot_cc_time (udpcorr_t * ctx, unsigned max_samples_to_plot);
void plot_ps (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "caspsr_udpcorr [options]\n"
     " -b nbatch      number of fft batches to integrate into each dump\n"
     " -d             dual side band data [default no]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -n npt         number of points in each fft [hardcoded 4096]\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between correlator dumps [default 0.5]\n"
     " -t ndump       total number of correlator dumps [default 1024]\n"
     " -z             zap DC channel [0]\n"
     " -v             verbose messages\n"
     " -h             print help text\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpcorr_prepare (udpcorr_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket: %ld bytes\n", ctx->pkt_size);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpcorr_reset(ctx);
  ctx->idump = 0;
  return 0;
}

int udpcorr_reset (udpcorr_t * ctx)
{
  unsigned int ipt;

  ctx->xcorr_fscr = 0;
  ctx->ac1 = 0;
  ctx->ac2 = 0;

  // reset the integrated sum
  for (ipt=0; ipt<ctx->nchan_out; ipt++)
  {
    ctx->xcorr[ipt] = 0;
    ctx->power1[ipt] = 0;
    ctx->power2[ipt] = 0;
  }
  return 0;
}

int udpcorr_destroy (udpcorr_t * ctx)
{
  fftwf_destroy_plan (ctx->plan_fwd);
  fftwf_destroy_plan (ctx->plan_bwd);

  fftwf_free (ctx->fwd1_in);
  fftwf_free (ctx->fwd2_in);
  fftwf_free (ctx->fwd1_out);
  fftwf_free (ctx->fwd2_out);

  free (ctx->amps);
  free (ctx->phases);
  free (ctx->channels);
  free (ctx->amps_t);
  free (ctx->phases_t);
  free (ctx->ac1_t);
  free (ctx->ac2_t);

  free (ctx->xcorr);
  free (ctx->xcorr_sum);

  free (ctx->lag);
  free (ctx->buffer);

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

int udpcorr_init (udpcorr_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init()\n");

  ctx->hdr_size = 16;

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
    multilog (ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // get a packet to determine the packet size and type of data
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buf, 9000, 0, NULL, NULL);
  ctx->data_size = ctx->pkt_size - ctx->hdr_size;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, ctx->data_size);

  // decode the header
  ctx->nant = 2;
  ctx->nchan_in  = 1;
  ctx->nchan_out = ctx->npt / 2;
  ctx->nsamp = ctx->npt * ctx->nbatch;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  if (ctx->verbose)
  {
    multilog(ctx->log, LOG_INFO, "init: nsamp=%d\n", ctx->nsamp);
    multilog(ctx->log, LOG_INFO, "init: nchan_in=%d\n", ctx->nchan_in);
    multilog(ctx->log, LOG_INFO, "init: npt=%d\n", ctx->npt);
    multilog(ctx->log, LOG_INFO, "init: nbatch=%d\n", ctx->nbatch);
    multilog(ctx->log, LOG_INFO, "init: nant=%d\n", ctx->nant);
    multilog(ctx->log, LOG_INFO, "init: ndim=%d\n", ctx->ndim);
    sleep(2);
  }

  // create an input buffer to store captured data in
  ctx->buffer_size = ctx->nchan_in * ctx->npt * ctx->nbatch * ctx->ndim * ctx->nant;
  //if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: raw buffer size=%ld\n", ctx->buffer_size);
  ctx->buffer = malloc (ctx->buffer_size);

  size_t nbytes = ctx->npt * ctx->ndim * sizeof(float);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: fft buffer size=%ld\n", nbytes);

  ctx->fwd1_in  = (float *) fftwf_malloc (nbytes);
  ctx->fwd2_in  = (float *) fftwf_malloc (nbytes);
  ctx->fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  ctx->fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  nbytes = ctx->nchan_out * sizeof(complex float);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: xcorr size=%ld\n", nbytes);
  ctx->xcorr = (complex float *) malloc (nbytes);
  ctx->xcorr_sum = (complex float *) malloc (nbytes);
  bzero (ctx->xcorr, nbytes);
  bzero (ctx->xcorr_sum, nbytes);

  ctx->lag = (complex float *) malloc (nbytes);

  nbytes = ctx->nchan_out * sizeof(float);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: cc vs freq size=%ld\n", nbytes);
  // as a function of frequnecy channel
  ctx->channels = (float *) malloc (nbytes);
  ctx->phases = (float *) malloc (nbytes);
  ctx->amps = (float *) malloc (nbytes);
  ctx->power1 = (float *) malloc (nbytes);
  ctx->power2 = (float *) malloc (nbytes);
  ctx->mean1 = (float *) malloc (nbytes);
  ctx->mean2 = (float *) malloc (nbytes);
  ctx->mean_npt = 0;

  bzero (ctx->phases, nbytes);
  bzero (ctx->amps, nbytes);
  bzero (ctx->power1, nbytes);
  bzero (ctx->power2, nbytes);

  unsigned ichan=0; 

  for (ichan=0; ichan<ctx->nchan_out; ichan++)
  {
    float freq = (((float) ichan / (float) ctx->nchan_out) * 100) + 800;
    ctx->channels[ichan] = freq;
  }

  nbytes = ctx->ndump * sizeof(float); 
  ctx->phases_t = (float *) malloc (nbytes);
  ctx->amps_t = (float *) malloc (nbytes);
  ctx->ac1_t = (float *) malloc (nbytes);
  ctx->ac2_t = (float *) malloc (nbytes);

  int direction_flags = FFTW_FORWARD;
  int flags = FFTW_ESTIMATE;
  ctx->plan_fwd = fftwf_plan_dft_r2c_1d (ctx->npt, ctx->fwd1_in, ctx->fwd1_out, FFTW_ESTIMATE);
  ctx->plan_bwd = fftwf_plan_dft_1d (ctx->nchan_out, ctx->fwd1_out, ctx->lag, FFTW_BACKWARD, FFTW_ESTIMATE);

  return 0;
}


int main (int argc, char **argv)
{
  int arg = 0;

  /* actual struct with info */
  udpcorr_t udpcorr;

  /* Pointer to array of "read" data */
  char *src;

  double sleep_time = 1;

  udpcorr.ndim = 1;
  udpcorr.npt = 2048;
  udpcorr.ndump = 1024;
  udpcorr.nbatch = 1024;

  udpcorr.dsb = 0;
  udpcorr.device = "/xs";
  udpcorr.interface = "any";
  udpcorr.port = MOPSR_DEFAULT_UDPDB_PORT;
  udpcorr.verbose = 0;
  udpcorr.zap_dc = 0;

  while ((arg=getopt(argc,argv,"b:dD:i:n:op:r:s:t:vw:zh")) != -1) 
  {
    switch (arg)
    {
      case 'b':
        udpcorr.nbatch = atoi(optarg);
        break;

      case 'd':
        udpcorr.dsb = 1;
        break;

      case 'D':
        udpcorr.device = strdup(optarg);
        break;

      case 'i':
        udpcorr.interface = strdup(optarg);
        break;

      case 'n':
        udpcorr.npt = atoi (optarg);
        break;

      case 'p':
        udpcorr.port = atoi (optarg);
        break;

      case 's':
        sleep_time = (double) atof (optarg);
        break;

      case 't':
        udpcorr.ndump = atoi (optarg);
        break;

      case 'v':
        udpcorr.verbose++;
        break;

      case 'z':
        udpcorr.zap_dc= 1;
        break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  udpcorr_t * ctx = &udpcorr;

  if (argc-optind != 0)
  {
    fprintf (stderr, "ERROR: no line arguments are expected\n");
    usage();
    exit(1);
  }

  signal(SIGINT, signal_handler);

  multilog_t* log = multilog_open ("mopsr_udpcorr", 0);
  multilog_add (log, stderr);

  udpcorr.log = log;

  // allocate require resources, open socket
  if (udpcorr_init (&udpcorr) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  // clear packets ready for capture
  udpcorr_prepare (&udpcorr);

  StartTimer(&wait_sw);

  while (!quit_threads && ctx->idump < ctx->ndump) 
  {
    // acquire input data and correlate each packet
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: acquire 1 packet()\n");
    udpcorr_acquire (ctx);

    udpcorr_endblock (ctx);

    DelayTimer(&wait_sw, sleep_time);

    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: clearing packets at socket [%ld]\n", ctx->pkt_size);
    size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
    StartTimer(&wait_sw);

    udpcorr_reset (ctx);

    if (kbhit())
    {
      getch();
      fprintf(stderr, "keypressed, clearing data\n");
      udpcorr_reset(ctx);
      ctx->idump = 0;
    }

  }

  udpcorr_destroy(ctx);

  return EXIT_SUCCESS;
}


void signal_handler(int signalValue)
{
  if (quit_threads) 
  {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;
}

// acquire enough UDP input data
int udpcorr_acquire (udpcorr_t * ctx)
{
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  char have_enough = 0;
  unsigned ibatch = 0;

  // keep acquiring UDP packets
  while (!have_enough && !quit_threads)
  {
    // try to get a UDP packet
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
            multilog (ctx->log, LOG_INFO, "main: timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",
                      timeouts, timeout_max);
            quit_threads = 1;
          }
        }
        else
        {
          multilog (ctx->log, LOG_ERR, "main: recvfrom failed %s\n", strerror(errsv));
          return 0;
        }
      }
      else // we received a packet of the WRONG size, ignore it
      {
        multilog (ctx->log, LOG_ERR, "main: received %d bytes, expected %d\n", got, ctx->pkt_size);
        quit_threads = 1;
      }
    }
    if (timeouts > timeout_max)
    {
      multilog(ctx->log, LOG_INFO, "main: timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
    }
    timeouts = 0;

    if (ctx->sock->have_packet)
    {
      udpcorr_unpack (ctx);
      udpcorr_correlate (ctx);

      ctx->sock->have_packet = 0;
      ibatch++;
      if (ibatch >= ctx->nbatch)
        have_enough = 1;
    }
  }
  return 0;
}

int udpcorr_unpack (udpcorr_t * ctx)
{
  const unsigned ndat = ctx->npt;
  //const float scale = 1;
  const float scale = 127.5;

  unsigned idat, ival, i;
  int8_t * raw = (int8_t *) (ctx->sock->buf + ctx->hdr_size);

  i = 0;
  for (idat=0; idat<ndat; idat+=4)
  {
    for (ival=0; ival<4; ival++)
      ctx->fwd1_in[idat+ival] = ((float) raw[i+ival] + 0.5) / scale;;
    i += 4;
    
    for (ival=0; ival<4; ival++)
      ctx->fwd2_in[idat+ival] = ((float) raw[i+ival] + 0.5) / scale;
    i += 4;
  }
  return 0;
}

int udpcorr_correlate (udpcorr_t * ctx)
{
  fftwf_execute_dft_r2c (ctx->plan_fwd, ctx->fwd1_in, ctx->fwd1_out);
  fftwf_execute_dft_r2c (ctx->plan_fwd, ctx->fwd2_in, ctx->fwd2_out);

  complex float val;

  unsigned ichan;
  for (ichan=0; ichan<ctx->nchan_out; ichan++)
  {
    if (ichan != 0)
    {
      const complex float p1 = ctx->fwd1_out[ichan] / ctx->nchan_out;
      const complex float p2 = ctx->fwd2_out[ichan] / ctx->nchan_out;
      const complex float xcorr = p1 * conj(p2);

      const float ac1 = cabsf(p1)*cabsf(p1);
      const float ac2 = cabsf(p2)*cabsf(p2);

      ctx->ac1 += ac1;
      ctx->ac2 += ac2;

      ctx->xcorr_fscr += xcorr;

      ctx->power1[ichan] += ac1;
      ctx->power2[ichan] += ac2;
      ctx->xcorr[ichan] += xcorr;
    }
  }
  return 0;
}
   
int udpcorr_endblock (udpcorr_t * ctx)
{
  unsigned ichan;
  float xcorr_scale;
  // compute a mean power for input 1 and 2
  float mean1 = 0;
  float mean2 = 0;

  for (ichan=0; ichan<ctx->nchan_out; ichan++)
  {
    if (ichan != 0)
    {
      mean1 += ctx->power1[ichan];
      mean2 += ctx->power2[ichan];
    }
  }

  mean1 /= (ctx->nchan_out);
  mean2 /= (ctx->nchan_out);
  xcorr_scale = sqrtf(mean1 * mean2);

  for (ichan=0; ichan<ctx->nchan_out; ichan++)
  {
    if (ichan != 0)
    {
      ctx->power1[ichan] /= mean1;
      ctx->power2[ichan] /= mean2;

      ctx->xcorr_sum[ichan] += ctx->xcorr[ichan];
      ctx->amps[ichan] = cabsf(ctx->xcorr[ichan]) / xcorr_scale;
      ctx->phases[ichan] = cargf(ctx->xcorr[ichan]);
    }
  }

  ctx->xcorr_fscr /= xcorr_scale;
  ctx->amps_t[ctx->idump] = cabsf(ctx->xcorr_fscr) / xcorr_scale;
  ctx->phases_t[ctx->idump] = cargf(ctx->xcorr_fscr) * (180/M_PI);
  ctx->ac1_t[ctx->idump] = ctx->ac1;
  ctx->ac2_t[ctx->idump] = ctx->ac2;
  ctx->idump++;

  if (ctx->idump > 1)
    plot_cc_time(ctx, 200);

  plot_ps (ctx->power1, ctx->nchan_out, "1/xs", "Bandpass A", 0, -1, 0);
  plot_ps (ctx->power2, ctx->nchan_out, "2/xs", "Bandpass A", 0, -1, 0);
  plot_cc_freq(ctx, "Cross Corr vs Freq - current dump", "3/xs");

  ctx->mean_npt = 0;

  return 0;
}

// plot the cross correlation as a function of frequnecy
void plot_cc_freq (udpcorr_t * ctx, char * title, char * device)
{
  float xmin = FLT_MAX;
  float xmax = FLT_MIN;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float yvals[ctx->nchan_out];

  unsigned i;
  for (i=0; i<ctx->nchan_out; i++)
  {
    yvals[i] = 0;
    if (ctx->amps[i] > 0)
      yvals[i] = ctx->amps[i];
    //yvals[i] = log(ctx->amps[i]);
    if (yvals[i] > ymax)
      ymax = yvals[i];
    if (yvals[i] < ymin)
      ymin = yvals[i];
    if (ctx->channels[i] > xmax)
      xmax = ctx->channels[i];
    if (ctx->channels[i] < xmin)
      xmin = ctx->channels[i];
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.10, 0.95, 0.5, 0.9);

  cpgsci(1);
  cpgbox("BCST", 0.0, 0.0, "BCNTV", 0.0, 0.0);
  cpglab("", "", title);

  cpgsci(2);
  cpgline(ctx->nchan_out, ctx->channels, yvals);
  cpgsci(1);

  cpgswin(xmin, xmax, (-1* M_PI), M_PI);
  cpgsvp(0.10, 0.95, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Freq", "Phase", "");

  cpgsci(2);
  cpgpt(ctx->nchan_out, ctx->channels, ctx->phases, -2);

  cpgebuf();
  cpgclos();
}

void plot_cc_time (udpcorr_t * ctx, unsigned max_samples_to_plot)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin_cc = FLT_MAX;
  float ymax_cc = -FLT_MAX;
  float ymin_ac1 = FLT_MAX;
  float ymax_ac1 = -FLT_MAX;
  float ymin_ac2 = FLT_MAX;
  float ymax_ac2 = -FLT_MAX;

  float xvals[max_samples_to_plot];
  float yac1[max_samples_to_plot];
  float yac2[max_samples_to_plot];
  float ycc[max_samples_to_plot];

  unsigned start = 0;
  unsigned end = ctx->idump;
  if (end - start > max_samples_to_plot)
    start = end - max_samples_to_plot;
  unsigned to_plot = end - start;

  unsigned i;
  float y;
  for (i=start; i<end; i++)
  {
    // cross correlation
    y = 0;
    if (ctx->amps_t[i] > 0)
      y = (ctx->amps_t[i]);
    if (y > ymax_cc)
      ymax_cc = y;
    if (y < ymin_cc)
      ymin_cc = y;
    ycc[i-start] = y;

    // auto correlations
    y = 0;
    if (ctx->ac1_t[i] > 0)
      y = (ctx->ac1_t[i]);
    if (y > ymax_ac1)
      ymax_ac1 = y;
    if (y < ymin_ac1)
      ymin_ac1 = y;
    yac1[i-start] = y;

    y = 0;
    if (ctx->ac2_t[i] > 0)
      y = (ctx->ac2_t[i]);
    if (y > ymax_ac2)
      ymax_ac2 = y;
    if (y < ymin_ac2)
      ymin_ac2 = y;
    yac2[i-start] = y;

    xvals[i-start] = (float) i;
    if (xvals[i-start] > xmax)
      xmax = xvals[i-start];
    if (xvals[i-start] < xmin)
      xmin = xvals[i-start];
  }

  if (fabs(ymax_cc - ymin_cc) < 1e-10)
  {
    float yrange = ymax_cc - ymin_cc;
    float mid = ymax_cc;
    ymin_cc -= yrange/2;
    ymax_cc += yrange/2;
  }

  if (fabs(ymax_ac1 - ymin_ac1) < 1e-10)
  {
    float yrange = ymax_ac1 - ymin_ac1;
    float mid = ymax_ac1;
    ymin_ac1 -= yrange/2;
    ymax_ac1 += yrange/2;
  }

  if (fabs(ymax_ac2 - ymin_ac2) < 1e-10)
  {
    float yrange = ymax_ac2 - ymin_ac2;
    float mid = ymax_ac2;
    ymin_ac2 -= yrange/2;
    ymax_ac2 += yrange/2;
  }

  if (cpgbeg(0, "4/xs", 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  float yrange, ydelta;

  yrange = ymax_ac1 - ymin_ac1;
  ydelta = yrange * 0.1;
  ymin_ac1 -= ydelta;
  ymax_ac1 += ydelta;

  cpgswin(xmin, xmax, ymin_ac1, ymax_ac1);
  cpgsvp(0.1, 0.85, 0.7, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCMSTV", 0.0, 0.0);
  cpglab("", "AC1", "Cross Correlation vs Time");

  cpgsci(2);
  cpgslw(4);
  cpgline(to_plot, xvals, yac1);
  cpgslw(1);
  cpgsci(1);

  yrange = ymax_ac2 - ymin_ac2;
  ydelta = yrange * 0.1;
  ymin_ac2 -= ydelta;
  ymax_ac2 += ydelta;

  cpgswin(xmin, xmax, ymin_ac2, ymax_ac2);
  cpgsvp(0.1, 0.85, 0.5, 0.7);
  cpgbox("BCST", 0.0, 0.0, "BCMSTV", 0.0, 0.0);
  cpglab("", "AC2", "");

  cpgsci(3);
  cpgslw(4);
  cpgline(to_plot, xvals, yac2);
  cpgslw(1);
  cpgsci(1);

  yrange = ymax_cc - ymin_cc;
  ydelta = yrange * 0.1;
  ymin_cc -= ydelta;
  ymax_cc += ydelta;

  cpgswin(xmin, xmax, ymin_cc, ymax_cc);
  cpgsvp(0.1, 0.85, 0.3, 0.5);
  cpgbox("BCST", 0.0, 0.0, "BCMSTV", 0.0, 0.0);
  cpglab("", "CC", "");

  cpgsci(1);
  cpgslw(4);
  cpgline(to_plot, xvals, ycc);
  cpgslw(1);
  cpgsci(1);

  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  for (i=0; i<ctx->idump; i++)
  {
    if (ctx->phases_t[i] > ymax)
      ymax = ctx->phases_t[i];
    if (ctx->phases_t[i] < ymin)
      ymin = ctx->phases_t[i];
  }

  cpgswin(xmin, xmax, (-1 *M_PI), M_PI);
  cpgsvp(0.1, 0.85, 0.1, 0.3);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Phase", "");

  cpgsci(3);
  cpgslw(5);
  cpgpt(to_plot, xvals, ctx->phases_t + start, -1);
  cpgslw(1);

  cpgebuf();
  cpgclos();
}

void plot_ps (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts)
{
  float xmin = 0;
  float xmax = npts;
  float ymin, ymax;
  float xvals[npts];
  unsigned xpeak = 0;
  unsigned i;

  if (ymin_in == -1)
    ymin = FLT_MAX;
  else
    ymin = ymin_in;

  if (ymax_in == -1)
    ymax = -FLT_MAX;
  else
    ymax = ymax_in;

  for (i=0; i<npts; i++)
  {
    if ((ymax_in == -1) && (yvals[i] > ymax))
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if ((ymin_in == -1) && (yvals[i] < ymin))
    {
      ymin = yvals[i];
    }

    xvals[i] = 800.0 + (((float) i / npts) * 100);
    xvals[i] = (float) i;
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Freq Channel", "Flux", title);

  cpgsci(2);

  if (use_pts)
    cpgpt(npts, xvals, yvals, -3);
  else
    cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();
}


int kbhit()
{
  struct timeval tv = { 0L, 0L };
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(0, &fds);
  return select(1, &fds, NULL, NULL, &tv);
}

int getch()
{
  int r;
  unsigned char c;
  if ((r = read(0, &c, sizeof(c))) < 0)
  {
    return r;
  } else {
    return c;
  }
}
