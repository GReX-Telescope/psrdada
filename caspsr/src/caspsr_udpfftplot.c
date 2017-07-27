/*
 * caspsr_udpplot
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
#include "caspsr_def.h"
#include "caspsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"
#include "stopwatch.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

typedef struct {

  multilog_t * log;

  caspsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // number of FFT points to perform on input channels
  unsigned int nfft;

  // number of output channels
  unsigned int nchan_out;

  // number of polarisations [should always be 2]
  unsigned int npol;

  // number of bins in the histogram
  unsigned int nbin;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float ** fft_in;      // input timeseries

  float ** fft_out;     // output spectra (oversampled)

  unsigned ** hist;     // histograms

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
void hist_data (udpplot_t * ctx);
void plot_data (udpplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "caspsr_udpplot [options]\n"
     " -b bw          sky bandwidth of data [default 100]\n"
     " -d             dual side band data [default no]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -F min,max     set the min,max x-value (e.g. frequency zoom) \n"
     " -f nfft        Nfft to perform [default 2048, max 2048]\n"
     " -p port        port on which to listen [default %d]\n"
     " -o             rotate oversampled channels\n"
     " -r min,max     set the min,max y-value (e.g. saturate birdies) \n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t num         Number of FFTs to integrate into each plot [default 8]\n"
     " -w freq        base frequnecy [default 799.609375MHz]\n"
     " -z             zap DC channel [0]\n"
     " -v             verbose messages\n"
     " -h             print help text\n"
     );
}

int udpplot_prepare (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "udpplot_prepare()\n");

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
  }

  unsigned ipol;
  unsigned ifft;
  unsigned ibin;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
    {
      ctx->y_points[ipol][ichan] = 0;
      ctx->fft_out[ipol][ichan] = 0;
    }
    for (ifft=0; ifft < ctx->nfft; ifft++)
    {
      ctx->fft_in[ipol][ifft] = 0;
    }
    for (ibin=0; ibin<ctx->nbin; ibin++)
    {
      ctx->hist[ipol][ibin] = 0;
    }
  }
  ctx->num_integrated = 0;
  ctx->fft_count = 0;
}

int udpplot_destroy (udpplot_t * ctx)
{

  fftwf_destroy_plan (ctx->plan);

  unsigned ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    if (ctx->y_points[ipol])
      free(ctx->y_points[ipol]);
    ctx->y_points[ipol] = 0;
    if (ctx->fft_in[ipol])
      free (ctx->fft_in[ipol]);
    ctx->fft_in[ipol] = 0;
    if (ctx->fft_out[ipol])
      free (ctx->fft_out[ipol]);
    ctx->fft_out[ipol] = 0;
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
    caspsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
}

/*
 * Close the udp socket and file
 */

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpdb_init()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = caspsr_init_sock();

  ctx->sock->size = 9000;
  if (ctx->sock->buffer)
    free (ctx->sock->buffer);
  ctx->sock->buffer = (char *) malloc (ctx->sock->size);
  assert(ctx->sock->buffer != NULL);

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
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buffer, 9000, 0, NULL, NULL);
  size_t data_size = ctx->pkt_size - UDP_HEADER;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, data_size);
  ctx->data_size = ctx->pkt_size - UDP_HEADER;

  // decode the header
  multilog (ctx->log, LOG_INFO, "init: decoding packet\n");
  uint64_t seq_no, ch_id;
  caspsr_decode_header (ctx->sock->buffer, &seq_no, &ch_id);
  multilog (ctx->log, LOG_INFO, "init: seq=%lu ch_id=%lu\n", seq_no, ch_id);

  ctx->npol = 2;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->npol);
  ctx->fft_in   = (float **) malloc(sizeof(float *) * ctx->npol);
  ctx->fft_out  = (float **) malloc(sizeof(float *) * ctx->npol);
  ctx->hist = (unsigned **) malloc(sizeof(unsigned *) * ctx->npol);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: nfft=%u nchan_out=%u\n", ctx->nfft, ctx->nchan_out);

  unsigned int ipol;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    ctx->y_points[ipol] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[ipol] = (float *) malloc (sizeof(float) * ctx->nfft);
    ctx->fft_out[ipol] = (float *) malloc (sizeof(float) * 2 * ctx->nchan_out);
    ctx->hist[ipol] = (unsigned *) malloc(sizeof(unsigned) * ctx->nbin);
  }

  float* input = ctx->fft_in[0];
  float* output = ctx->fft_out[0];

  CHECK_ALIGN(input);
  CHECK_ALIGN(output);

  int flags = FFTW_ESTIMATE;
  ctx->plan = fftwf_plan_dft_r2c_1d (ctx->nfft, (float *) input, (fftwf_complex*)output, flags);
  return 0;
}



int main (int argc, char **argv)
{
  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port on which to listen for incoming connections */
  int port = 4002;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  char * device = "/xs";

  /* actual struct with info */
  udpplot_t udpplot;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nfft = 4096;

  unsigned int plot_log = 0;

  double sleep_time = 1.0f / 1e6;

  float base_freq = 799.609375;

  unsigned int to_integrate = 1024;

  float xmin = 0.0;
  float xmax = 0.0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  unsigned zap_dc = 0;
  char dsb = 0;
  float bw = 100.0;

  while ((arg=getopt(argc,argv,"b:dD:f:F:i:lp:r:s:t:vw:zh")) != -1) {
    switch (arg) {

    case 'b':
      bw = atof(optarg);
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

    case 'p':
      port = atoi (optarg);
      break;

    case 'r':
      sscanf (optarg, "%f,%f", &ymin, &ymax);
      break;

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

  multilog_t* log = multilog_open ("caspsr_udpplot", 0);
  multilog_add (log, stderr);

  udpplot.log = log;
  udpplot.verbose = verbose;

  udpplot.interface = strdup(interface);
  udpplot.port = port;

  udpplot.base_freq = base_freq;
  udpplot.bw = bw;
  udpplot.dsb = dsb;

  udpplot.nfft = nfft;
  udpplot.nchan_out = nfft / 2;
  udpplot.nbin = 256;

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

  udpplot.zap_dc = zap_dc;
  udpplot.num_integrated = 0;
  udpplot.fft_count = 0;
  udpplot.to_integrate = to_integrate;

  udpplot.plot_log = plot_log;
  udpplot.ymin = 100000;
  udpplot.ymax = -100000;
  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  multilog(log, LOG_INFO, "caspsr_udpfftplot: %f %f\n", udpplot.xmin, udpplot.xmax);

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "caspsr_udpfftplot: using device %s\n", device);

  // cloear packets ready for capture
  udpplot_prepare (&udpplot);

  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  uint64_t seq_no, ch_id;

  udpplot_t * ctx = &udpplot;

  StartTimer(&wait_sw);

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buffer, ctx->pkt_size, 0, NULL, NULL );

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
      caspsr_decode_header (ctx->sock->buffer, &seq_no, &ch_id);
      seq_no /= 2048;


      if (ctx->verbose) 
        multilog (ctx->log, LOG_INFO, "main: seq_no=%"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // append packet to fft input
      append_packet (ctx, ctx->sock->buffer + 16, ctx->data_size);

      if (ctx->fft_count >= ctx->nfft)
      {
        fft_data (ctx);
        detect_data (ctx);

        ctx->num_integrated ++;
        ctx->fft_count = 0;
      }

      if (ctx->num_integrated >= ctx->to_integrate)
      {
        multilog (ctx->log, LOG_INFO, "plotting %d packets in %d channels\n", ctx->num_integrated, ctx->nchan_out);
        plot_data (ctx);
        hist_data (ctx);
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

  return EXIT_SUCCESS;
}

// copy data from packet in the fft input buffer
void append_packet (udpplot_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet()\n");

  unsigned ipt, ipol, ifft;
  int8_t * in = (int8_t *) buffer;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: npol=%u\n", ctx->npol);

  unsigned int ndat = size / ctx->npol;
  unsigned int start_fft = ctx->fft_count;
  unsigned int end_fft   = start_fft + ndat;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: size=%u ndat=%u\n", size, ndat);

  float complex val;
  if (end_fft > ctx->nfft)
    end_fft = ctx->nfft;

  ifft = start_fft;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet: fft points: start=%u end=%u ndat=%u\n", start_fft, end_fft, ndat);

  int bin;
  while (ifft < end_fft)
  {
    for (ipol=0; ipol<ctx->npol; ipol++)
    {
      for (ipt=0; ipt<4; ipt++)
      {
        //fprintf (stderr, "[%d][%d] = in[%d]\n", ipol, ifft + ipt, ipt);
        ctx->fft_in[ipol][ifft + ipt] = (float) in[ipt] + 0.5;
        bin = ((int) in[ipt]) + 128;
        ctx->hist[ipol][bin]++;
      }
      in += 4;
    }
    ifft += 4;
  }

  ctx->fft_count += ndat;

  /*
  unsigned i;
  for (i=0; i<end_fft; i++)
    fprintf (stderr,"[%d] %f\n", i, ctx->fft_in[0][i]);
  */
}

void fft_data (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "fft_data()\n"); 
  unsigned int ipol;
  float * src;
  float * dest;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    src = ctx->fft_in[ipol]; 
    dest = ctx->fft_out[ipol];
    fftwf_execute_dft_r2c (ctx->plan, (float *) src, (fftwf_complex*) dest);
  }
}

void detect_data (udpplot_t * ctx)
{
  unsigned ipol = 0;
  unsigned ichan = 0;
  unsigned ibit = 0;
  unsigned halfbit = ctx->nchan_out / 2;
  unsigned offset = 0;
  unsigned basechan = 0;
  unsigned newchan;
  unsigned shift;
  float a, b;

  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    if (ctx->dsb)
    {
      // first half [flipped - A]
      for (ibit=halfbit; ibit<ctx->nchan_out; ibit++)
      {
        a = ctx->fft_out[ipol][(ibit*2) + 0];  
        b = ctx->fft_out[ipol][(ibit*2) + 1];  
        newchan = (ibit-halfbit);
        ctx->y_points[ipol][newchan] += ((a*a) + (b*b));
      }

      // second half [B]
      for (ibit=0; ibit<halfbit; ibit++)
      {
        a = ctx->fft_out[ipol][(ibit*2) + 0];
        b = ctx->fft_out[ipol][(ibit*2) + 1];
        newchan = (ibit+halfbit);
        ctx->y_points[ipol][newchan] += ((a*a) + (b*b));
        if (ctx->zap_dc && ibit == 0)
          ctx->y_points[ipol][newchan] = 0;
      }
    }
    else
    {
      for (ibit=0; ibit<ctx->nchan_out; ibit++)
      {
        a = ctx->fft_out[ipol][(ibit*2) + 0];
        b = ctx->fft_out[ipol][(ibit*2) + 1];
        ctx->y_points[ipol][ibit] += ((a*a) + (b*b));

        if (ctx->zap_dc && ibit == 0)
        {
          fprintf (stderr, "zapping chan=%d\n", ibit);
          ctx->y_points[ipol][ibit] = 0;
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
  unsigned ipol = 0;
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
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      for (ichan=0; ichan < ctx->nchan_out; ichan++)
      { 
        if (ctx->plot_log)
          ctx->y_points[ipol][ichan] = (ctx->y_points[ipol][ichan] > 0) ? log10(ctx->y_points[ipol][ichan]) : 0;
        if ((ichan > xchan_min) && (ichan < xchan_max))
        {
          if (ctx->y_points[ipol][ichan] > ymax) ymax = ctx->y_points[ipol][ichan];
          if (ctx->y_points[ipol][ichan] < ymin) ymin = ctx->y_points[ipol][ichan];
        }
      }
    }
  }
  if (ctx->verbose)
  {
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f, ctx->xmax=%f\n", ctx->xmin, ctx->xmax);
    multilog (ctx->log, LOG_INFO, "plot_packet: ymin=%f, ymax=%f\n", ymin, ymax);
  }

  if (cpgopen("1/xs") != 1) 
  {
    multilog(ctx->log, LOG_INFO, "caspsr_udpfftplot: error opening 1/xs\n");
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

  cpgsls(1);

  int col, ls;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    col = ipol + 2;
    cpgsci(col);
    cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ipol]);
  }
  cpgebuf();
  cpgclos();
}

void hist_data (udpplot_t * ctx)
{
  float x[ctx->nbin];
  float p0[ctx->nbin];
  float p1[ctx->nbin];
  int ibin, ipol;

  float ymin = 0;
  float ymax_p0= 0;
  float ymax_p1 = 0;
  char all_zero = 1;

  for (ibin=0; ibin<ctx->nbin; ibin++)
  {
    x[ibin] = ((float) ibin * (256 / ctx->nbin)) - 128;
    if (ctx->plot_log)
    {
      if (ctx->hist[0][ibin] == 0)
        p0[ibin] = 0;
      else
        p0[ibin] = log10((float) ctx->hist[0][ibin]);
      if (ctx->hist[1][ibin] == 0)
        p1[ibin] = 0;
      else
        p1[ibin] = log10((float) ctx->hist[1][ibin]);
    }
    else
    {
      p0[ibin] = (float) ctx->hist[0][ibin];
      p1[ibin] = (float) ctx->hist[1][ibin];
    }

    if (p0[ibin] > ymax_p0)
    {
      ymax_p0 = p0[ibin];
     }
    if (p1[ibin] > ymax_p1)
    {
      ymax_p1 = p1[ibin];
    }
  }

  if (cpgopen("2/xs") != 1) 
  {
    multilog(ctx->log, LOG_INFO, "caspsr_udpfftplot: error opening plot device\n");
    exit(1);
  }

  cpgbbuf();
  cpgsci(1);

  cpgbbuf();
  cpgsci(1);
  
  char title[128];
  char label[64];
    
  ymax_p0 *= 1.2;
  ymax_p1 *= 1.2;
    
  float ymax = ymax_p0 > ymax_p1 ? ymax_p0 : ymax_p1;
  
  sprintf (title, "Input histograms of each polarisation");

  // p0
  cpgswin(-128, 128, ymin, ymax_p0);
  cpgsvp(0.1, 0.9, 0.5, 0.9);
  if (ctx->plot_log)
    cpgbox("BCST", 0.0, 0.0, "BCLNST", 0.0, 0.0);
  else
    cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Count", title);

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_p0);
  cpgsls(1);

  cpgmtxt("T", -2.2, 0.05, 0.0, "P0");
  cpgsci(2);
  cpgslw(3);
  cpgbin (ctx->nbin, x, p0, 0);
  cpgslw(1);
  cpgsci(1);

  cpgebuf();
  cpgbbuf();

  cpgswin(-128, 128, ymin, ymax_p1);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  if (ctx->plot_log)
    cpgbox("BCNST", 0.0, 0.0, "BCLNST", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("State", "Count", "");

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_p1);
  cpgsls(1);
  
  // Im line
  cpgmtxt("T", -2.2, 0.05, 0.0, "P1");
  cpgslw(3);
  cpgsci(3);
  cpgbin (ctx->nbin, x, p1, 0);
  cpgsci(1);
  cpgslw(1);

  cpgebuf();
  cpgclos();
}

