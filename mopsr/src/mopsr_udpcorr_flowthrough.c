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

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

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

  unsigned ant1;

  unsigned ant2;

  // number of FFT points for correlation
  unsigned int npt;

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

  int16_t * raw1;
  int16_t * raw2;

  fftwf_complex * fwd1_in;
  fftwf_complex * fwd2_in;

  fftwf_complex * fwd1_out;
  fftwf_complex * fwd2_out;

  complex float * sum;

  float * amps;
  float * phases;

  unsigned idump;
  unsigned ndump;
  float * amps_t;
  float * phases_t;

  fftwf_plan plan_fwd;

  float ymin;

  float ymax;

  size_t pkt_size;

  size_t data_size;

  char dsb;

  char zap_dc;

  mopsr_hdr_t hdr;

} udpcorr_t;

int udpcorr_init (udpcorr_t * ctx);
int udpcorr_prepare (udpcorr_t * ctx);
int udpcorr_destroy (udpcorr_t * ctx);


int udpcorr_acquire (udpcorr_t * ctx);
int udpcorr_unpack (udpcorr_t * ctx);
int udpcorr_extract (udpcorr_t * ctx, unsigned ibatch);
int udpcorr_correlate (udpcorr_t * ctx);
int udpcorr_plot (udpcorr_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpcorr [options] ant1 ant2\n"
     " ant1           first antenna in pair\n"
     " ant2           second antenna in pair\n"
     " -d             dual side band data [default no]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -n npt         number of points in each fft [default 128]\n"
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
}

int udpcorr_reset (udpcorr_t * ctx)
{
  unsigned int ipt;

  // reset integrated totals
  for (ipt=0; ipt<ctx->npt; ipt++)
  {
    ctx->phases[ipt] = 0; 
    ctx->amps[ipt] = 0; 
  }

  ctx->idump = 0;
}

int udpcorr_destroy (udpcorr_t * ctx)
{
  fftwf_destroy_plan (ctx->plan_fwd);

  fftwf_free (ctx->fwd1_in);
  fftwf_free (ctx->fwd2_in);
  fftwf_free (ctx->fwd1_out);
  fftwf_free (ctx->fwd2_out);

  free (ctx->amps);
  free (ctx->phases);
  free (ctx->amps_t);
  free (ctx->phases_t);

  free (ctx->sum);

  free (ctx->raw1);
  free (ctx->raw2);
  free (ctx->buffer);

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

int udpcorr_init (udpcorr_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init()\n");

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
  size_t data_size = ctx->pkt_size - UDP_HEADER;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, data_size);
  ctx->data_size = ctx->pkt_size - UDP_HEADER;

  // decode the header
  multilog (ctx->log, LOG_INFO, "init: decoding packet\n");
  mopsr_hdr_t hdr;
  if (ctx->data_size == 8192)
    mopsr_decode_v2 (ctx->sock->buf, &hdr);
  else
    mopsr_decode (ctx->sock->buf, &hdr);

  multilog (ctx->log, LOG_INFO, "init: nchan=%u nant=%u nframe=%u\n", hdr.nchan, hdr.nant, hdr.nframe);

  assert (hdr.nchan == 1);
  assert (hdr.nant = 4);
   
  ctx->nant = hdr.nant;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  ctx->nchan_out = ctx->npt;

  ctx->buffer_size = ctx->npt * ctx->nbatch * ctx->ndim * ctx->nant;
  ctx->buffer = malloc (ctx->buffer_size);
  ctx->nsamp = ctx->npt * ctx->nbatch;

  size_t nbytes = ctx->npt * ctx->nbatch * ctx->ndim;
  ctx->raw1 = (int16_t *) malloc (nbytes);
  ctx->raw2 = (int16_t *) malloc (nbytes);

  nbytes = ctx->npt * ctx->ndim * sizeof(fftwf_complex);
  ctx->fwd1_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  ctx->fwd2_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  ctx->fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  ctx->fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  nbytes = ctx->npt * ctx->ndim * sizeof(float);

  // as a function of frequnecy channel
  ctx->phases = (float *) malloc (nbytes);
  ctx->amps = (float *) malloc (nbytes);

  nbytes = ctx->ndump * sizeof(float); 
  ctx->phases_t = (float *) malloc (nbytes);
  ctx->amps_t = (float *) malloc (nbytes);

  int direction_flags = FFTW_FORWARD;
  int flags = FFTW_ESTIMATE;
  ctx->plan_fwd = fftwf_plan_dft_1d (ctx->npt, ctx->fwd1_in, ctx->fwd1_out, FFTW_FORWARD, FFTW_ESTIMATE);

  return 0;
}


int main (int argc, char **argv)
{
  int arg = 0;

  /* actual struct with info */
  udpcorr_t udpcorr;

  /* Pointer to array of "read" data */
  char *src;

  double sleep_time = 1000000;

  udpcorr.dsb = 0;
  udpcorr.device = "/xs";
  udpcorr.interface = "any";
  udpcorr.npt = 128;
  udpcorr.port = MOPSR_DEFAULT_UDPDB_PORT;
  udpcorr.ndump = 1024;
  udpcorr.nbatch = 1024;
  udpcorr.verbose = 0;
  udpcorr.zap_dc = 0;

  while ((arg=getopt(argc,argv,"dD:i:n:op:r:s:t:vw:zh")) != -1) 
  {
    switch (arg)
    {
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
        sleep_time *= 1000000;
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


  if (argc-optind != 2)
  {
    fprintf (stderr, "ERROR: 2 command line arguments are required\n");
    usage();
    exit(1);
  }

  if (sscanf(argv[optind], "%u", &(ctx->ant1)) != 1)
  {
    fprintf (stderr, "ERROR: could not parse ant1 from %s\n", argv[optind]);
    exit(1);
  }

  if (sscanf(argv[optind+1], "%u", &(ctx->ant2)) != 1)
  {
    fprintf (stderr, "ERROR: could not parse ant2 from %s\n", argv[optind+1]);
    exit(1);
  }

  multilog_t* log = multilog_open ("mopsr_udpcorr", 0);
  multilog_add (log, stderr);

  udpcorr.log = log;

  // allocate require resources, open socket
  if (udpcorr_init (&udpcorr) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  udpcorr.nchan_out = udpcorr.npt;

  // initialise data rate timing library 
  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  // clear packets ready for capture
  udpcorr_prepare (&udpcorr);

  StopWatch_Start(&wait_sw);

  while (!quit_threads) 
  {
    // acquire input data
    udpcorr_acquire (ctx);

    // correlate input data
    udpcorr_correlate (ctx);

    // plot correlations
    udpcorr_plot (ctx);

    StopWatch_Delay(&wait_sw, sleep_time);

    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: clearing packets at socket [%ld]\n", ctx->pkt_size);
    size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
    StopWatch_Start(&wait_sw);
  }

  cpgclos();

  return EXIT_SUCCESS;
}


// acquire enough UDP input data
int udpcorr_acquire (udpcorr_t * ctx)
{
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  char have_enough = 0;
  size_t offset = 0;

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
            multilog (ctx->log, LOG_INFO, "main: timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
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
      mopsr_decode_v2 (ctx->sock->buf, &(ctx->hdr));

      if (prev_seq_no == 0)
      {
        mopsr_print_header (&ctx->hdr);
      }

      // make a quick copy of the udp packet
      memcpy (ctx->buffer + offset, ctx->sock->buf + UDP_HEADER, ctx->data_size);
      offset += ctx->data_size;

      ctx->sock->have_packet = 0;
      if (offset >= ctx->buffer_size)
        have_enough = 1;
    }
  }
}

int udpcorr_extract (udpcorr_t * ctx, unsigned ibatch)
{
  // now extract the desired channels for this batch
  unsigned ipt, iant;
  int16_t * buf = (int16_t *) ctx->buffer + (ibatch * ctx->npt * ctx->nant * ctx->ndim);

  for (ipt=0; ipt<ctx->npt; ipt++)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      if (iant == ctx->ant1)
        ctx->raw1[ipt] = buf[0];
      if (iant == ctx->ant2)
        ctx->raw2[ipt] = buf[0];
      buf++;
    }
  }

  return 0;
}


int udpcorr_unpack (udpcorr_t * ctx)
{
  float re, im;
  unsigned isamp, iant;

  for (isamp=0; isamp<ctx->nsamp; isamp++)
  {
    for (iant=0; iant<ctx->nant; iant++)
    { 
      re = (((float) ctx->raw1[2*isamp]) + 0.5) / 127.5;
      im = (((float) ctx->raw1[2*isamp+1]) + 0.5) / 127.5;
      ctx->fwd1_in[isamp] = re + im * I;

      re = (((float) ctx->raw2[2*isamp]) + 0.5) / 127.5;
      im = (((float) ctx->raw2[2*isamp+1]) + 0.5) / 127.5;
      ctx->fwd2_in[isamp] = re + im * I;
    }
  }
}

int udpcorr_correlate (udpcorr_t * ctx)
{
  unsigned ibatch, ipt;
  complex float val;

  for (ibatch=0; ibatch<ctx->nbatch; ibatch++)
  {
    udpcorr_extract(ctx, ibatch);

    udpcorr_unpack (ctx);

    fftwf_execute_dft (ctx->plan_fwd, ctx->fwd1_in, ctx->fwd1_out);
    fftwf_execute_dft (ctx->plan_fwd, ctx->fwd2_in, ctx->fwd2_out);

    for (ipt=0; ipt<ctx->npt; ipt++)
    {
      // compute cross correlation
      val = conj(ctx->fwd1_out[ipt]) * ctx->fwd2_out[ipt];
      ctx->sum[ipt] += val;
    }
  }

  float re, im;
  // compute ampt and phase as a function of time
  for (ipt=0; ipt<ctx->npt; ipt++)
  {
    re = creal(ctx->sum[ipt]);
    im = cimag(ctx->sum[ipt]);
    ctx->amps_t[ctx->idump] = (re * re) + (im * im);
    ctx->phases_t[ctx->idump] = atan2f(im, re);
    ctx->idump++;
  }

  return 0;
}

int udpcorr_plot (udpcorr_t * ctx)
{

}
