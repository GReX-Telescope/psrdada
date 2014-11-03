/*
 * mopsr_dbundersampledb
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

  float *** fft_in;

  float ** fft_out;

  unsigned int fft_count;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

  float base_freq;

  int zap_dc;

  fftwf_plan plan;

} dbundersampledb_t;

int dbundersampledb_init (dbundersampledb_t * ctx);
int dbundersampledb_prepare (dbundersampledb_t * ctx);
int dbundersampledb_destroy (dbundersampledb_t * ctx);

void append_packet (dbundersampledb_t * ctx, char * buffer, unsigned int size);
void detect_data (dbundersampledb_t * ctx);
void fft_data (dbundersampledb_t * ctx);
void plot_data (dbundersampledb_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_dbundersampledb [options]\n"
     " -a ant         antenna to display [default all]\n"
     " -b freq        frequency of first channel [default 800 MHz]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -f nfft        Nfft on each coarse channel [default 16, max 16]\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t num         Number of FFTs to integrate into each plot [default 8]\n"
     " -z             zap DC channel [0]\n"
     " -v             verbose messages\n"
     " -h             print help text\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int dbundersampledb_prepare (dbundersampledb_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  dbundersampledb_reset(ctx);
}

int dbundersampledb_reset (dbundersampledb_t * ctx)
{
  unsigned ichan;
  float percent_chan;
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    percent_chan = (float) ichan / (float) ctx->nchan_out;
    percent_chan *= 100; // MHz
    ctx->x_points[ichan] = ctx->base_freq + percent_chan;
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

int dbundersampledb_destroy (dbundersampledb_t * ctx)
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
    mopsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
}

/*
 * Close the udp socket and file
 */

int dbundersampledb_init (dbundersampledb_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

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

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  ctx->fft_in = (float ***) malloc(sizeof(float **) * ctx->nant);
  ctx->fft_out = (float **) malloc(sizeof(float *) * ctx->nant);

  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[iant] = (float **) malloc (sizeof(float *) * ctx->nchan_in);
    ctx->fft_out[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);

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
  dbundersampledb_t dbundersampledb;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nant = 2;

  unsigned int nchan = 128;

  unsigned int nfft = 16;

  unsigned int plot_log = 0;

  double sleep_time = 1000000;

  float base_freq = 800.0;

  unsigned int to_integrate = 8;

  int antenna = -1;

  unsigned zap_dc = 0;

  while ((arg=getopt(argc,argv,"a:b:D:f:i:lp:s:t:vzh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
      break;

    case 'b':
      base_freq = atof(optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'f':
      nfft = atoi (optarg);
      break;

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
      sleep_time *= 1000000;
      break;

    case 't':
      to_integrate = atoi (optarg);
      break;

    case 'v':
      verbose++;
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

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_dbundersampledb", 0);
  multilog_add (log, stderr);

  dbundersampledb.log = log;
  dbundersampledb.verbose = verbose;

  dbundersampledb.interface = strdup(interface);
  dbundersampledb.port = port;

  dbundersampledb.nchan_in = nchan;
  dbundersampledb.nfft = nfft;
  dbundersampledb.nchan_out = nchan * nfft;
  dbundersampledb.nant = nant;
  dbundersampledb.ant_code = 0;

  dbundersampledb.antenna = antenna;

  dbundersampledb.zap_dc = zap_dc;
  dbundersampledb.num_integrated = 0;
  dbundersampledb.fft_count = 0;
  dbundersampledb.to_integrate = to_integrate;

  dbundersampledb.plot_log = plot_log;
  dbundersampledb.ymin = 100000;
  dbundersampledb.ymax = -100000;
  dbundersampledb.base_freq = base_freq;

  // initialise data rate timing library 
  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_dbplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // allocate require resources, open socket
  if (dbundersampledb_init (&dbundersampledb) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // cloear packets ready for capture
  dbundersampledb_prepare (&dbundersampledb);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  dbundersampledb_t * ctx = &dbundersampledb;

  StopWatch_Start(&wait_sw);

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL );

      if ((got == UDP_PAYLOAD) || (got + 8 == UDP_PAYLOAD))
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
        multilog (log, LOG_ERR, "main: received %d bytes, expected %d\n", got, UDP_PAYLOAD);
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

      mopsr_decode_header(ctx->sock->buf, &seq_no, &(ctx->ant_code));

      if (ctx->verbose) 
        multilog (ctx->log, LOG_INFO, "main: seq_no=%"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // append packet to fft input
      append_packet (ctx, ctx->sock->buf + UDP_HEADER, UDP_DATA);

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
        plot_data (ctx);
        dbundersampledb_reset (ctx);
        StopWatch_Delay(&wait_sw, sleep_time);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
          size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
        StopWatch_Start(&wait_sw);
      }
    }
  }

  cpgclos();

  return EXIT_SUCCESS;
}

// copy data from packet in the fft input buffer
void append_packet (dbundersampledb_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_packet()\n");

  unsigned ichan, iant, ifft;
  int8_t * in = (int8_t *) buffer;

  unsigned int nframes   = size / (ctx->nchan_in * ctx->nant * MOPSR_NDIM);
  unsigned int start_fft = ctx->fft_count;
  unsigned int end_fft   = start_fft + nframes;

  if (end_fft > ctx->nfft)
    end_fft = ctx->nfft;

  for (ifft=start_fft; ifft < end_fft; ifft++)
  {
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      for (iant=0; iant < ctx->nant; iant++)
      {
        ctx->fft_in[iant][ichan][(2*ifft) + 0] = (float) in[0];   // re
        ctx->fft_in[iant][ichan][(2*ifft) + 1] = (float) in[1];   // im
        in += 2;
      }
    }
    ctx->fft_count++;
  }
}

#if 0
void fft_packet (dbundersampledb_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "fft_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned ifft = 0;

  int8_t * in = (int8_t *) buffer;
  for (ifft=0; ifft < ctx->nfft; ifft++)
  {
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      for (iant=0; iant < ctx->nant; iant++)
      {
        ctx->fft_in[iant][ichan][(2*ifft) + 0] = (float) in[0];   // re
        ctx->fft_in[iant][ichan][(2*ifft) + 1] = (float) in[1];   // im
        in += 2;
      }
    }
  }

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
#endif

void fft_data (dbundersampledb_t * ctx)
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

void detect_data (dbundersampledb_t * ctx)
{
  unsigned iant = 0;
  unsigned ichan = 0;
  float a, b;

  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_out; ichan++)
      {
        a = ctx->fft_out[iant][(2*ichan)+0];
        b = ctx->fft_out[iant][(2*ichan)+1];
        ctx->y_points[iant][ichan] += ((a*a) + (b*b));
        if (ctx->zap_dc && ichan == 0)
          ctx->y_points[iant][ichan] = 0;
      }
    }
  }
}


void plot_data (dbundersampledb_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float xmin = ctx->x_points[0] - 5;
  float xmax = ctx->x_points[ctx->nchan_out-1] + 5;

  // calculate limits
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_out; ichan++)
      {
        if (ctx->plot_log)
          ctx->y_points[iant][ichan] = (ctx->y_points[iant][ichan] > 0) ? log10(ctx->y_points[iant][ichan]) : 0;
        if (ctx->y_points[iant][ichan] > ctx->ymax) ctx->ymax = ctx->y_points[iant][ichan];
        if (ctx->y_points[iant][ichan] < ctx->ymin) ctx->ymin = ctx->y_points[iant][ichan];
      }
    }
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->ymin=%f, ctx->ymax=%f\n", ctx->ymin, ctx->ymax);

  cpgbbuf();
  cpgsci(1);
  if (ctx->plot_log)
  {
    cpgenv(xmin, xmax, ctx->ymin, 1.1 * ctx->ymax, 0, 20);
    cpglab("Freq", "log\\d10\\u(Power)", "Bandpass"); 
  }
  else
  {
    cpgenv(xmin, xmax, ctx->ymin, 1.1*ctx->ymax, 0, 0);
    cpglab("Freq", "Power", "Bandpass"); 
  }

  char ant_label[8];
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      sprintf(ant_label, "Ant %u", mopsr_get_ant_number (ctx->ant_code, iant));
      cpgsci(iant + 2);
      cpgmtxt("T", 1.5 + (1.0 * iant), 0.0, 0.0, ant_label);
      cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[iant]);
    }
  }
  cpgebuf();
}
