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

  // number of input channels
  unsigned int nchan_in;

  // number of FFT points to perform on input channels
  unsigned int nfft;

  // number of output channels
  unsigned int nchan_out;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // which antenna to display [-1 for both]
  int antenna;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float *** fft_in;     // input timeseries

  float ** fft_out;     // output spectra (oversampled)

  float ** fft_in2;     // input spectra (critically sampled)

  float ** fft_out2;    // output timeseries (critically sampled)

  float ** fft_in3;     // input timeseries (critically sampled)

  float ** fft_out3;    // output spectra (critically sampled)

  unsigned int fft_count;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

  float base_freq;

  float bw;

  int zap_dc;

  float xmin;

  float xmax;

  fftwf_plan plan;

  unsigned unique_corrections;

  float complex * corrections;

  mopsr_hdr_t hdr;

  char dsb;

  char rotate_oversampled;

  size_t pkt_size;

  size_t data_size;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void append_packet (udpplot_t * ctx, char * buffer, unsigned int size);
void detect_data (udpplot_t * ctx);
void fft_data (udpplot_t * ctx);
void plot_data (udpplot_t * ctx);
void plot_data_new (udpplot_t * ctx, mopsr_hdr_t hdr);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpplot [options]\n"
     " -a ant         antenna to display [default all]\n"
     " -b bw          sky bandwidth of data [default 100]\n"
     " -c nchan       number of channels in each packet [default 128]\n"
     " -d             dual side band data [default no]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -F min,max     set the min,max x-value (e.g. frequency zoom) \n"
     " -f nfft        Nfft on each coarse channel [default 16, max 16]\n"
     " -n nant        number of antenna in packet [default 4]\n"
     " -p port        port on which to listen [default %d]\n"
     " -o             rotate oversampled channels\n"
     " -r min,max     set the min,max y-value (e.g. saturate birdies) \n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t num         Number of FFTs to integrate into each plot [default 8]\n"
     " -w freq        base frequnecy [default 799.609375MHz]\n"
     " -z             zap DC channel [0]\n"
     " -v             verbose messages\n"
     " -h             print help text\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket: %ld bytes\n", ctx->pkt_size);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpplot_reset(ctx);
}

int udpplot_reset (udpplot_t * ctx)
{
  unsigned ichan;
  float mhz_per_out_chan = ctx->bw / (float) ctx->nchan_out;
  float half_bw = mhz_per_out_chan / 2.0;
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    ctx->x_points[ichan] = (ctx->base_freq + half_bw) + (((float) ichan) * mhz_per_out_chan); 
    //fprintf (stderr, "x_points[%d]=%f\n", ichan, ctx->x_points[ichan]);
  }

  unsigned iant;
  unsigned ifft;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
    {
      ctx->y_points[iant][ichan] = 0;
      ctx->fft_out[iant][ichan] = 0;
    }
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      for (ifft=0; ifft < ctx->nfft; ifft++)
      {
        ctx->fft_in[iant][ichan][ifft] = 0;
      }
    }
  }
  ctx->num_integrated = 0;
  ctx->fft_count = 0;
}

int udpplot_destroy (udpplot_t * ctx)
{

  fftwf_destroy_plan (ctx->plan);
  unsigned int iant;
  unsigned int ichan;
  for (iant=0; iant<ctx->nant; iant++)
  {
    if (ctx->y_points[iant])
      free(ctx->y_points[iant]);
    ctx->y_points[iant] = 0;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      if (ctx->fft_in[iant][ichan])
        free (ctx->fft_in[iant][ichan]);
      ctx->fft_in[iant][ichan] = 0;
    }
    if (ctx->fft_in[iant])
      free (ctx->fft_in[iant]);
    ctx->fft_in[iant] = 0;
    if (ctx->fft_out[iant])
      free (ctx->fft_out[iant]);
    ctx->fft_out[iant] = 0;
  }

  if (ctx->fft_in)
    free (ctx->fft_in);
  ctx->fft_in = 0;

  if (ctx->fft_out)
    free (ctx->fft_out);
  ctx->fft_out = 0;

  if (ctx->y_points)
    free(ctx->y_points);
  ctx->y_points = 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;

  if (ctx->nchan_in > 1)
  {
    if (ctx->corrections)
      free (ctx->corrections);
    ctx->corrections = 0;
  }

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

  ctx->nchan_in = hdr.nchan;
  ctx->nant = hdr.nant;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  // only need to appy corrections for PFB packets
  if (ctx->nchan_in > 1)
  {
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "prepare: setting up oversampling corrections\n");
    ctx->unique_corrections = 32;
    ctx->corrections = (float complex *) malloc(sizeof(float complex) * ctx->unique_corrections * ctx->nchan_in);
    unsigned ichan, ipt, icorr;
    float theta;
    float ratio = 2 * M_PI * (5.0 / 32.0);
    for (ichan=0; ichan<ctx->nchan_in; ichan++)
    {
      for (ipt=0; ipt < 32; ipt++)
      {
        icorr = (ichan * ctx->unique_corrections) + ipt;
        theta = ichan * ratio * ipt;
        ctx->corrections[icorr] = sin (theta) - cos(theta) * I;
      }
    }
  }

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: nchan_in=%u\n", ctx->nchan_in);

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  ctx->fft_in = (float ***) malloc(sizeof(float **) * ctx->nant);
  ctx->fft_out = (float **) malloc(sizeof(float *) * ctx->nant);

  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[iant] = (float **) malloc (sizeof(float *) * ctx->nchan_in);
    ctx->fft_out[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out * 2);

    unsigned int ichan;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      ctx->fft_in[iant][ichan] = (float *) malloc (sizeof(float) * ctx->nfft * 2);
    }
  }

  float* input = ctx->fft_in[0][0];
  float* output = ctx->fft_out[0];

  CHECK_ALIGN(input);
  CHECK_ALIGN(output);

  int direction_flags = FFTW_FORWARD;
  int flags = FFTW_ESTIMATE;
  ctx->plan = fftwf_plan_dft_1d (ctx->nfft, (fftwf_complex*) input, (fftwf_complex*)output, direction_flags, flags);

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

  int nant = -1;

  int nchan = -1;

  unsigned int nfft = 64;

  unsigned int plot_log = 0;

  double sleep_time = 1;

  float base_freq = 799.609375;

  unsigned int to_integrate = 1024;

  float xmin = 0.0;
  float xmax = 0.0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  int antenna = -1;

  unsigned zap_dc = 0;
  char dsb = 0;
  char rotate_oversampled = 0;
  float bw = 100.0;

  while ((arg=getopt(argc,argv,"a:b:c:dD:f:F:i:ln:op:r:s:t:vw:zh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
      break;

    case 'b':
      bw = atof(optarg);
      break;

    case 'c':
      nchan = atoi (optarg);
      break;

    case 'd':
      dsb = 1;
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'f':
      nfft = atoi (optarg);
      break;

    case 'F':
    {
      if (sscanf (optarg, "%f,%f", &xmin, &xmax) != 2)
      {
        fprintf (stderr, "could not parse xrange from %s\n", optarg);
        return (EXIT_FAILURE);
      }
      break;
    }

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'l':
      plot_log = 1;
      break;

    case 'n':
      nant = atoi (optarg);
      break;

    case 'o':
      rotate_oversampled = 1;
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 'r':
    {
      sscanf (optarg, "%f,%f", &ymin, &ymax);
      break;
    }

    case 's':
      sleep_time = (double) atof (optarg);
      break;

    case 't':
      to_integrate = atoi (optarg);
      break;

    case 'v':
      verbose++;
      break;

    case 'w':
      base_freq = atof(optarg);
      break;

    case 'z':
      zap_dc = 1;
      break;

    case 'h':
      usage();
      return 0;
      
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

  udpplot.base_freq = base_freq;
  udpplot.bw = bw;
  udpplot.dsb = dsb;

  if ((xmin == 0) && (xmax == 0))
  {
    xmin = udpplot.base_freq;
    xmax = udpplot.base_freq + udpplot.bw;
  }

  // allocate require resources, open socket
  if (udpplot_init (&udpplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  if (nchan > 0)
    udpplot.nchan_in = nchan;
  else
  if (nant > 0)
    udpplot.nant = nant;

  udpplot.nfft = nfft;
  udpplot.nchan_out = udpplot.nchan_in * nfft;
  udpplot.ant_code = 0;

  udpplot.antenna = antenna;

  udpplot.zap_dc = zap_dc;
  udpplot.num_integrated = 0;
  udpplot.fft_count = 0;
  udpplot.to_integrate = to_integrate;
  udpplot.rotate_oversampled = rotate_oversampled;

  udpplot.plot_log = plot_log;
  udpplot.ymin = 100000;
  udpplot.ymax = -100000;
  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  multilog(log, LOG_INFO, "mopsr_udpfftplot: %f %f\n", udpplot.xmin, udpplot.xmax);

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udpfftplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_udpfftplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // cloear packets ready for capture
  udpplot_prepare (&udpplot);

  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  udpplot_t * ctx = &udpplot;

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
      if (ctx->data_size == 8192)
        mopsr_decode_v2 (ctx->sock->buf, &(ctx->hdr));
      else
        mopsr_decode (ctx->sock->buf, &(ctx->hdr));

      if (prev_seq_no == 0)
      {
        mopsr_print_header (&ctx->hdr);
      }

      if (ctx->verbose) 
        multilog (ctx->log, LOG_INFO, "main: seq_no=%"PRIu64" difference=%"PRIu64" packets\n", ctx->hdr.seq_no, (ctx->hdr.seq_no - prev_seq_no));
      prev_seq_no = ctx->hdr.seq_no;

      // append packet to fft input
      append_packet (ctx, ctx->sock->buf + UDP_HEADER, ctx->data_size);

      if (ctx->fft_count >= ctx->nfft)
      {
        fft_data (ctx);
        detect_data (ctx);

        ctx->num_integrated ++;
        ctx->fft_count = 0;
      }

      if (ctx->num_integrated >= ctx->to_integrate)
      {
        multilog (ctx->log, LOG_INFO, "plotting %d FFTs in %d channels\n", ctx->num_integrated, ctx->nchan_out);
        if (ctx->nant == 16)
          plot_data_new (ctx, ctx->hdr);
        else
          plot_data (ctx);
        udpplot_reset (ctx);
        DelayTimer(&wait_sw, sleep_time);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket [%ld]\n", ctx->pkt_size);
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

// copy data from packet in the fft input buffer
void append_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet()\n");

  unsigned ichan, iant, ifft;
  int8_t * in = (int8_t *) buffer;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: nchan_in=%u nant=%u\n", ctx->nchan_in, ctx->nant);

  unsigned int nframes   = size / (ctx->nchan_in * ctx->nant * MOPSR_NDIM);
  unsigned int start_fft = ctx->fft_count;
  unsigned int end_fft   = start_fft + nframes;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: size=%u nframes=%u\n", size, nframes);

  unsigned icorr;
  float complex val;

  if (end_fft > ctx->nfft)
    end_fft = ctx->nfft;

  for (ifft=start_fft; ifft < end_fft; ifft++)
  {
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      if (ctx->nchan_in > 1)
        icorr = ichan * (ctx->unique_corrections) + (ifft % ctx->unique_corrections);

      for (iant=0; iant < ctx->nant; iant++)
      {
        if (ctx->rotate_oversampled)
        {
          val = ((float) in[0]) + ((float) in[1]) * I;
          val *= ctx->corrections[icorr];
          ctx->fft_in[iant][ichan][(2*ifft) + 0] = creal(val);
          ctx->fft_in[iant][ichan][(2*ifft) + 1] = cimag(val);
        } 
        else
        {
          ctx->fft_in[iant][ichan][(2*ifft) + 0] = (float) in[0];
          ctx->fft_in[iant][ichan][(2*ifft) + 1] = (float) in[1];
        }
        in += 2;
      }
    }
    ctx->fft_count++;
  }
}

void fft_data (udpplot_t * ctx)
{
  unsigned int iant, ichan;
  float * src;
  float * dest;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        src = ctx->fft_in[iant][ichan]; 
        dest = ctx->fft_out[iant] + (ichan * ctx->nfft * 2);

        fftwf_execute_dft (ctx->plan, (fftwf_complex*) src, (fftwf_complex*) dest);
      }
    }
  }
}

void detect_data (udpplot_t * ctx)
{
  unsigned iant = 0;
  unsigned ichan = 0;
  unsigned ibit = 0;
  unsigned halfbit = ctx->nfft / 2;
  unsigned offset = 0;
  unsigned basechan = 0;
  unsigned newchan;
  unsigned shift;
  float a, b;

  fprintf (stderr, "detect: zap_dc=%d\n", ctx->zap_dc);

  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        offset = (ichan * ctx->nfft * 2);
        basechan = ichan * ctx->nfft;

        if (ctx->dsb)
        {
          // first half [flipped - A]
          for (ibit=halfbit; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];  
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];  
            newchan = (ibit-halfbit);
            ctx->y_points[iant][basechan + newchan] += ((a*a) + (b*b));
          }

          // second half [B]
          for (ibit=0; ibit<halfbit; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];
            newchan = (ibit+halfbit);
            ctx->y_points[iant][basechan + newchan] += ((a*a) + (b*b));
            if (ctx->zap_dc && ibit == 0)
              ctx->y_points[iant][basechan + newchan] = 0;
          }
        }
        else
        {
          for (ibit=0; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];
            ctx->y_points[iant][basechan + ibit] += ((a*a) + (b*b));

            if (ctx->zap_dc && ibit == 0)
            {
              fprintf (stderr, "zapping chan=%d\n", basechan + ibit);
              ctx->y_points[iant][basechan + ibit] = 0;
            }
          }
        }

      }
    }
  }
}

void plot_data (udpplot_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  int ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float ymin = ctx->ymin;
  float ymax = ctx->ymax;

  int xchan_min = -1;
  int xchan_max = -1;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f ctx->xmax=%f\n", ctx->xmin, ctx->xmax);

  // determined channel ranges for the x limits
  if (xchan_min == -1)
    for (ichan=ctx->nchan_out; ichan >= 0; ichan--)
      if (ctx->x_points[ichan] >= ctx->xmin)
        xchan_min = ichan;

  if (xchan_max == -1) 
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
      if (ctx->x_points[ichan] <= ctx->xmax)
        xchan_max = ichan;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_packet: chan_min=%d xchan_max=%d\n", xchan_min, xchan_max);

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      if ((ctx->antenna < 0) || (ctx->antenna == iant))
      {
        for (ichan=0; ichan < ctx->nchan_out; ichan++)
        { 
          if (ctx->plot_log)
            ctx->y_points[iant][ichan] = (ctx->y_points[iant][ichan] > 0) ? log10(ctx->y_points[iant][ichan]) : 0;
          if ((ichan > xchan_min) && (ichan < xchan_max))
          {
            if (ctx->y_points[iant][ichan] > ymax) ymax = ctx->y_points[iant][ichan];
            if (ctx->y_points[iant][ichan] < ymin) ymin = ctx->y_points[iant][ichan];
          }
        }
      }
    }
  }
  if (ctx->verbose)
  {
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f, ctx->xmax=%f\n", ctx->xmin, ctx->xmax);
    multilog (ctx->log, LOG_INFO, "plot_packet: ymin=%f, ymax=%f\n", ymin, ymax);
  }

  cpgbbuf();
  cpgsci(1);
  if (ctx->plot_log)
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 20);
    cpglab("Frequency [MHz]", "log\\d10\\u(Power)", "Bandpass"); 
  }
  else
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 0);
    cpglab("Frequency [MHz]", "Power", "Bandpass"); 
  }

  float line_x[2];
  float line_y[2];
  float percent_chan;

  float oversampling_difference = ((5.0 / 32.0) * ctx->nfft) / 2.0;
  cpgsls(4);
  for (ichan=0; ichan < ctx->nchan_in; ichan++)
  {
    line_y[0] = ymin;
    line_y[1] = ymin + ((ymax - ymin)/4);

    percent_chan = (float) ichan / (float) ctx->nchan_in;
    percent_chan *= ctx->bw;
    line_x[0] = line_x[1] = ctx->base_freq + percent_chan;
    //multilog (ctx->log, LOG_INFO, "plot_packet: ichan=%d, percent_chan=%f freq=%f\n", ichan, percent_chan, line_x[0]);
    //line_x[0] = line_x[1] = ichan * ctx->nfft; 
    cpgline(2, line_x, line_y);

    /*
    line_y[0] = ymin;
    line_y[1] = ymin + (ymax - ymin) / 2;
    
    line_x[0] = line_x[1] = (ichan * ctx->nfft) - oversampling_difference;
    cpgline(2, line_x, line_y);

    line_x[0] = line_x[1] = (ichan * ctx->nfft) + oversampling_difference;
    cpgline(2, line_x, line_y);
    */
  }
  cpgsls(1);

  int col, ls;
  char ant_label[8];
  int ant_id;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {

      //ant_id = mopsr_get_new_ant_number(iant);

      // TODO - revert! special case for 4 Input mode
      if ((iant == 0) || (iant == 1))
      {
        ant_id = mopsr_get_ant_number (ctx->hdr.ant_id, iant);
      }
      else
      {
        ant_id = mopsr_get_ant_number (ctx->hdr.ant_id2, iant-2);
      }

      sprintf(ant_label, "Ant %u", ant_id);
     
      col = (iant % 14) + 2;
      ls  = (iant / 14) + 1;

      cpgsci(col);
      cpgsls(ls);
      cpgmtxt("T", -1.5 - (0.9 * ant_id), 0.01, 0.0, ant_label);
      cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[iant]);
      cpgsls(1);
    }
  }
  cpgebuf();
}

void plot_data_new (udpplot_t * ctx, mopsr_hdr_t hdr)
{
  int ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float ymin = ctx->ymin;
  float ymax = ctx->ymax;

  int xchan_min = -1;
  int xchan_max = -1;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f ctx->xmax=%f\n", ctx->xmin, ctx->xmax);

  // determined channel ranges for the x limits
  if (xchan_min == -1)
    for (ichan=ctx->nchan_out; ichan >= 0; ichan--)
      if (ctx->x_points[ichan] >= ctx->xmin)
        xchan_min = ichan;

  if (xchan_max == -1)
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
      if (ctx->x_points[ichan] <= ctx->xmax)
        xchan_max = ichan;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_packet: chan_min=%d xchan_max=%d\n", xchan_min, xchan_max);

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      if ((ctx->antenna < 0) || (ctx->antenna == iant))
      {
        for (ichan=0; ichan < ctx->nchan_out; ichan++)
        {
          if (ctx->plot_log)
            ctx->y_points[iant][ichan] = (ctx->y_points[iant][ichan] > 0) ? log10(ctx->y_points[iant][ichan]) : 0;
          if ((ichan > xchan_min) && (ichan < xchan_max))
          {
            if (ctx->y_points[iant][ichan] > ymax) ymax = ctx->y_points[iant][ichan];
            if (ctx->y_points[iant][ichan] < ymin) ymin = ctx->y_points[iant][ichan];
          }
        }
      }
    }
  }
  if (ctx->verbose)
  {
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f, ctx->xmax=%f\n", ctx->xmin, ctx->xmax);
    multilog (ctx->log, LOG_INFO, "plot_packet: ymin=%f, ymax=%f\n", ymin, ymax);
  }

  char ant_label[32];
  char x_label[32];
  int ant_id, a, locked;

  if (ctx->plot_log)
    sprintf (x_label, "%s", "log\\d10\\u(Power)");
  else
    sprintf (x_label, "%s",  "Power");

  cpgbbuf();
  cpgsci(1);

  cpgpage();

  cpgswin(ctx->xmin, ctx->xmax, ymin, ymax);

  for (iant=0; iant < 16; iant++)
  {
    int index = mopsr_get_new_ant_index(iant);
  }

  // TL
  cpgsvp(0.1, 0.5, 0.5, 0.9);
  if (ctx->plot_log)
    cpgbox("BCST", 0.0, 0.0, "BCLNST", 0.0, 0.0);
  else  
    cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", x_label, "");

  for (iant=0; iant < 4; iant++)
  {
    a = iant - 0;
    ant_id = mopsr_get_new_ant_index(iant);
    locked = mopsr_get_bit_from_16 (hdr.mgt_locks, iant);
    sprintf(ant_label, "Ant %u", iant);
    if (!locked)
      sprintf(ant_label, "Ant %u NO MGT LOCK", iant);
    cpgsci(a + 2);
    cpgmtxt("T", -1.5 - (0.9 * a), 0.01, 0.0, ant_label);
    cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ant_id]);
  }
  cpgsci(1);

  // TR
  cpgsvp(0.5, 0.9, 0.5, 0.9);
  if (ctx->plot_log)
    cpgbox("BCST", 0.0, 0.0, "BCLST", 0.0, 0.0);
  else
    cpgbox("BCST", 0.0, 0.0, "BCST", 0.0, 0.0);
  cpglab("", "", "");

  for (iant=4; iant < 8; iant++)
  {
    a = iant - 4;
    ant_id = mopsr_get_new_ant_index(iant);
    locked = mopsr_get_bit_from_16 (hdr.mgt_locks, iant);
    sprintf(ant_label, "Ant %u", iant);
    if (!locked)
      sprintf(ant_label, "Ant %u NO MGT LOCK", iant);
    cpgsci(a + 2);
    cpgmtxt("T", -1.5 - (0.9 * a), 0.01, 0.0, ant_label);
    cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ant_id]);
  }
  cpgsci(1);

  // BL
  cpgsvp(0.1, 0.5, 0.1, 0.5);
  if (ctx->plot_log)
    cpgbox("BCNST", 0.0, 0.0, "BCNLST", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Freq [MHz]", x_label, "");

  for (iant=8; iant < 12; iant++)
  {
    a = iant - 8;
    ant_id = mopsr_get_new_ant_index(iant);
    locked = mopsr_get_bit_from_16 (hdr.mgt_locks, iant);
    sprintf(ant_label, "Ant %u", iant);
    if (!locked)
      sprintf(ant_label, "Ant %u NO MGT LOCK", iant);
    cpgsci(a + 2);
    cpgmtxt("T", -1.5 - (0.9 * a), 0.01, 0.0, ant_label);
    cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ant_id]);
  }
  cpgsci(1);

  // BR
  cpgsvp(0.5, 0.9, 0.1, 0.5);
  if (ctx->plot_log)
    cpgbox("BCNST", 0.0, 0.0, "BCLST", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCST", 0.0, 0.0);
  cpglab("Freq [MHz]", "", "");

  for (iant=12; iant < 16; iant++)
  {
    a = iant - 12;
    ant_id = mopsr_get_new_ant_index(iant);
    locked = mopsr_get_bit_from_16 (hdr.mgt_locks, iant);
    sprintf(ant_label, "Ant %u", iant);
    if (!locked)
      sprintf(ant_label, "Ant %u NO MGT LOCK", iant);
    cpgsci(a + 2);
    cpgmtxt("T", -1.5 - (0.9 * a), 0.01, 0.0, ant_label);
    cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ant_id]);
  }

  cpgebuf();

}
