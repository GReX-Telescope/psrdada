/*
 * snapsr_udpplot
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
#include "snapsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"
#include "stopwatch.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

typedef struct {

  multilog_t * log;

  snapsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // identifying code for antenna's in packet
  unsigned int ant_code;

  // number of antennae
  unsigned int nant;

  // channel to plot timeseries of
  unsigned int channel;

  // number of input channels
  unsigned int nchan_in;

  // number of input channels per sub-band
  unsigned int nchan_pkt;

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

  uint64_t fft_seq;

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

  snapsr_udp_hdr_t hdr;

  char dsb;

  size_t pkt_size;

  size_t data_size;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void append_packet (udpplot_t * ctx, char * buffer, unsigned int size, snapsr_udp_hdr_t * hdr);
void detect_data (udpplot_t * ctx);
void fft_data (udpplot_t * ctx);
void plot_data (udpplot_t * ctx);
void plot_timeseries (udpplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "snapsr_udpplot [options]\n"
     " -a ant         antenna to display [default all]\n"
     " -b bw          sky bandwidth of data [default 100]\n"
     " -c chan        channel to plot timeseries [default 0]\n"
     " -d             dual side band data [default no]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -F min,max     set the min,max x-value (e.g. frequency zoom) \n"
     " -f nfft        Nfft on each coarse channel [default 16, max 16]\n"
     " -n nant        number of antenna in packet [default 4]\n"
     " -p port        port on which to listen [default %d]\n"
     " -r min,max     set the min,max y-value (e.g. saturate birdies) \n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t num         Number of FFTs to integrate into each plot [default 8]\n"
     " -w freq        base frequnecy [default 799.609375MHz]\n"
     " -z             zap DC channel [0]\n"
     " -v             verbose messages\n"
     " -h             print help text\n",
     50000);
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "snapsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket: %ld bytes\n", ctx->pkt_size);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: cleared %ld packets\n", cleared);

  udpplot_reset(ctx);
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: reset completed\n");
}

int udpplot_reset (udpplot_t * ctx)
{
  unsigned ichan;
  float mhz_per_out_chan = ctx->bw / (float) ctx->nchan_out;
  float half_bw = mhz_per_out_chan / 2.0;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "reset: setting xpoints: nchan=%u mhz_per_out_chan=%f\n", ctx->nchan_out, mhz_per_out_chan);
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
      for (ifft=0; ifft < ctx->nfft*2; ifft++)
      {
        ctx->fft_in[iant][ichan][ifft] = 0;
      }
    }
  }
  ctx->num_integrated = 0;
  ctx->fft_count = 0;
  ctx->fft_seq = 0;
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

  if (ctx->sock)
  {
    close(ctx->sock->fd);
    snapsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
}

/*
 * Close the udp socket and file
 */

int udpplot_init (udpplot_t * ctx)
{
  //if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "snapsr_udpdb_init() nfft=%u\n", ctx->nfft);

  // create a SNAPSR socket which can hold variable num of UDP packet
  ctx->sock = snapsr_init_sock();

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

  // set the socket size to 64 MB
  int sock_buf_size = 64*1024*1024;
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // get a packet to determine the packet size and type of data
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buf, 9000, 0, NULL, NULL);
  ctx->data_size = ctx->pkt_size - SNAPSR_UDP_HEADER_BYTES;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, ctx->data_size);

  // decode the header
  multilog (ctx->log, LOG_INFO, "init: decoding packet: data_size=%d\n", ctx->data_size);
  snapsr_udp_hdr_t hdr;
  snapsr_decode (ctx->sock->buf, &hdr);
  multilog (ctx->log, LOG_INFO, "init: subband_id=%u nchan=%u nant=%u nframe=%u\n", hdr.subband_id, hdr.nchan, hdr.nant, hdr.nframe);
  snapsr_print_header (&hdr);

  ctx->nchan_in = hdr.nchan * SNAPSR_NSUBBAND;
  ctx->nant = hdr.nant;
  ctx->nchan_out = ctx->nchan_in * ctx->nfft;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock (ctx->sock->fd);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: nchan_in=%u nchan_out=%u\n", ctx->nchan_in, ctx->nchan_out);

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  ctx->fft_in = (float ***) malloc(sizeof(float **) * ctx->nant);
  ctx->fft_out = (float **) malloc(sizeof(float *) * ctx->nant);

  size_t fft_buf_size = sizeof(float) * ctx->nfft * 2;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: nfft=%u fft_buf_size=%ld\n", ctx->nfft, fft_buf_size);

  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[iant] = (float **) malloc (sizeof(float *) * ctx->nchan_in);
    ctx->fft_out[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out * 2);

    unsigned int ichan;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      ctx->fft_in[iant][ichan] = (float *) malloc (fft_buf_size);
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
  int port = 50000;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  char * device = "/xs";

  /* actual struct with info */
  udpplot_t udpplot;

  /* Pointer to array of "read" data */
  char *src;

  int nant = -1;

  int channel = 0;

  unsigned int nfft = 64;

  unsigned int plot_log = 0;

  double sleep_time = 1;

  float base_freq = 799.609375;

  unsigned int to_integrate = 32 ;

  float xmin = 0.0;
  float xmax = 0.0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  int antenna = -1;

  unsigned zap_dc = 0;
  char dsb = 0;
  float bw = 100.0;

  while ((arg=getopt(argc,argv,"a:b:c:dD:f:F:i:ln:p:r:s:t:vw:zh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
      break;

    case 'b':
      bw = atof(optarg);
      break;

    case 'c':
      channel = atoi (optarg);
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

  multilog_t* log = multilog_open ("snapsr_udpplot", 0);
  multilog_add (log, stderr);

  udpplot.log = log;
  udpplot.verbose = verbose;

  udpplot.interface = strdup(interface);
  udpplot.port = port;

  udpplot.channel = channel;
  udpplot.base_freq = base_freq;
  udpplot.bw = bw;
  udpplot.dsb = dsb;

  if ((xmin == 0) && (xmax == 0))
  {
    xmin = udpplot.base_freq;
    xmax = udpplot.base_freq + udpplot.bw;
  }

  udpplot.nant = SNAPSR_NANT_PER_PACKET;
  udpplot.nfft = nfft;
  udpplot.nchan_pkt = SNAPSR_NCHAN_PER_SUBBAND;
  udpplot.ant_code = 0;

  udpplot.antenna = antenna;

  udpplot.zap_dc = zap_dc;
  udpplot.num_integrated = 0;
  udpplot.fft_count = 0;
  udpplot.fft_seq = 0;
  udpplot.to_integrate = to_integrate;

  udpplot.plot_log = plot_log;
  udpplot.ymin = 100000;
  udpplot.ymax = -100000;
  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  // allocate require resources, open socket
  if (udpplot_init (&udpplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  multilog(log, LOG_INFO, "snapsr_udpfftplot: %f %f\n", udpplot.xmin, udpplot.xmax);

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "snapsr_udpfftplot: using device %s\n", device);

  // clear packets ready for capture
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
      snapsr_decode (ctx->sock->buf, &(ctx->sock->hdr));

      if ((prev_seq_no > 0) && (ctx->sock->hdr.seq_no != prev_seq_no + 1))
        multilog (ctx->log, LOG_INFO, "main: seq_no=%"PRIu64" difference=%"PRIu64" packets\n", ctx->hdr.seq_no, (ctx->sock->hdr.seq_no - prev_seq_no));
      prev_seq_no = ctx->sock->hdr.seq_no;

      // append packet to fft input
      append_packet (ctx, ctx->sock->buf + SNAPSR_UDP_HEADER_BYTES, ctx->data_size, &(ctx->sock->hdr));

      // now that we have enough consecutive packets, fft and detect
      if (ctx->fft_count >= ctx->nfft)
      {
        if (ctx->verbose)
          multilog (ctx->log, LOG_INFO, "fft_data()\n");
        fft_data (ctx);
        detect_data (ctx);

        ctx->num_integrated ++;
        ctx->fft_count = 0;
        ctx->fft_seq = 0;
        
        if (ctx->antenna > 0)
          plot_timeseries (ctx);

        // clear buffered packets
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket [%ld]\n", ctx->pkt_size);
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);

        // reset the prev_seq_no
        prev_seq_no = 0;
      }

      // we have now integrated enough packets
      if (ctx->num_integrated >= ctx->to_integrate)
      {
        multilog (ctx->log, LOG_INFO, "plotting %d FFTs in %d channels\n", ctx->num_integrated, ctx->nchan_out);

        plot_data (ctx);

        udpplot_reset (ctx);
        DelayTimer(&wait_sw, sleep_time);

        StartTimer(&wait_sw);
      }
    }
  }

  return EXIT_SUCCESS;
}

// copy data from packet in the fft input buffer
void append_packet (udpplot_t * ctx, char * buffer, unsigned int size, snapsr_udp_hdr_t * hdr)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet()\n");

  unsigned ichan, iant, ifft;
  int8_t * in = (int8_t *) buffer;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: nchan_in=%u nant=%u subband_id=%u\n", ctx->nchan_in, ctx->nant, hdr->subband_id);

  // tag the seq number for the start of this FFT
  if (ctx->fft_seq == 0)
  {
    // wait for subband_id == 0
    if (hdr->subband_id != 0)
    {
      return;
    }
    ctx->fft_seq = hdr->seq_no;
  }

  uint64_t ssamp = (((hdr->seq_no - ctx->fft_seq) - hdr->subband_id) / SNAPSR_NSUBBAND) * SNAPSR_NFRAME_PER_PACKET;
  unsigned nframes = size / (ctx->nchan_pkt * ctx->nant * 2);
  unsigned iframe;

  for (iframe=0; iframe<nframes; iframe++)
  {
    unsigned isamp = ssamp + iframe;
    for (ichan=0; ichan < ctx->nchan_pkt; ichan++)
    {
      const unsigned ochan = ctx->sock->hdr.subband_id * SNAPSR_NCHAN_PER_SUBBAND + ichan;
      for (iant=0; iant < ctx->nant; iant++)
      {
        const unsigned oant = ctx->sock->hdr.snap_id * SNAPSR_NANT_PER_PACKET + iant;
        ctx->fft_in[oant][ochan][(2*isamp) + 0] = (float) in[0];
        ctx->fft_in[oant][ochan][(2*isamp) + 1] = (float) in[1];
        in += 2;
      }
    }
  }

  if (hdr->subband_id == SNAPSR_NSUBBAND - 1)
  {
    ctx->fft_count += nframes;
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

  multilog (ctx->log, LOG_INFO, "plot_packet: antenna=%d\n", ctx->antenna);

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    multilog (ctx->log, LOG_INFO, "plot_packet: calculating limits\n");
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

  if (cpgbeg (0, "1/xs", 1, 1) != 1)
  {
    multilog(ctx->log, LOG_INFO, "plot_packet: error opening plot device 1/xs\n");
    exit(1);
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
    cpgline(2, line_x, line_y);
  }
  cpgsls(1);

  int col, ls;
  char ant_label[8];
  int ant_id;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      multilog (ctx->log, LOG_INFO, "plot_packet: plotting antenna=%d\n", iant);
      sprintf(ant_label, "Ant %u", iant);
     
      col = (iant % 14) + 2;
      ls  = (iant / 14) + 1;

      cpgsci(col);
      cpgsls(ls);
      cpgmtxt("T", -1.5 - (0.9 * iant), 0.01, 0.0, ant_label);
      cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[iant]);
      cpgsls(1);
    }
  }
  cpgebuf();
  cpgend();
}

void plot_timeseries (udpplot_t * ctx)
{
  unsigned ndat = ctx->nfft;

  float * re = (float *) malloc (ndat * sizeof(float));
  float * im = (float *) malloc (ndat * sizeof(float));
  float * x  = (float *) malloc (ndat * sizeof(float));

  unsigned chan = (unsigned) ctx->channel;
  unsigned ant  = (unsigned) ctx->antenna;

  unsigned i;
  for (i=0; i<ndat; i++)
  {
    x[i] = (float) i;
    re[i] = ctx->fft_in[ant][chan][(2*i) + 0];
    im[i] = ctx->fft_in[ant][chan][(2*i) + 1];
  }

  float xmin = 0;
  float xmax = ndat;
  float ymin = -128;
  float ymax = 129;

  if (cpgbeg (0, "2/xs", 1, 1) != 1)
  {
    multilog(ctx->log, LOG_INFO, "plot_timeseries: error opening plot device 1/xs\n");
    exit(1);
  }

  cpgbbuf();
  cpgsci(1);
  cpgslw(1);
  int symbol = -1;

  char label[128];

  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  sprintf (label, "Complex Timeseries for Channel %d, Ant %u", chan, ant);
  cpglab("Time", "Voltage", label);

  // Real line
  cpgsci(2);
  cpgslw(5);
  cpgpt (ndat, x, re, symbol);
  cpgslw(1);

  cpgsci(3);
  cpgslw(5);
  cpgpt (ndat, x, im, symbol);
  cpgslw(1);

  cpgebuf();
}
