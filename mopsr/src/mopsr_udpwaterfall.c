/*
 * mopsr_udpwaterfall
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

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  // identifying code for antenna numbres
  unsigned int ant_code;

  // the antenna to plot
  unsigned int antenna;

  // number of antenna in each packet
  unsigned int nant;

  // number of samples per plot
  unsigned int nsamps;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float *  data;

  uint64_t num_integrated;

  unsigned int plot_log;

  unsigned int zap;

} udpwaterfall_t;

int udpwaterfall_init (udpwaterfall_t * ctx);
int udpwaterfall_prepare (udpwaterfall_t * ctx);
int udpwaterfall_destroy (udpwaterfall_t * ctx);

void integrate_packet (udpwaterfall_t * ctx, char * buffer, unsigned int size);
void dump_packet (udpwaterfall_t * ctx, char * buffer, unsigned int size);
void print_packet (udpwaterfall_t * ctx, char * buffer, unsigned int size);
void plot_packet (udpwaterfall_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udpwaterfall [options]\n"
     " -h             print help text\n"
     " -a ant         antenna index to display [default 0]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             plot logarithmically\n"
     " -p port        port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t samps       number of time samples to display [default 128]\n"
     " -z             zap preconfigured channels\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpwaterfall_prepare (udpwaterfall_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  udpwaterfall_reset(ctx);
}

int udpwaterfall_reset (udpwaterfall_t * ctx)
{
  unsigned idat;
  unsigned ndat = ctx->nchan * ctx->nsamps;
  for (idat=0; idat < ndat; idat++)
    ctx->data[idat] = 0;
  ctx->num_integrated = 0;
}

int udpwaterfall_destroy (udpwaterfall_t * ctx)
{
  if (ctx->data)
    free(ctx->data);
  ctx->data = 0;

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

int udpwaterfall_init (udpwaterfall_t * ctx)
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

  unsigned ndat = ctx->nchan * ctx->nsamps;
  ctx->data = (float *) malloc (sizeof(float) * ndat);

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
  udpwaterfall_t udpwaterfall;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nant = 2;

  unsigned int antenna = 0;

  unsigned int nsamps = 128;

  unsigned int nchan = 128;

  unsigned int plot_log = 0;

  double sleep_time = 1000000;

  unsigned int zap = 0;

  while ((arg=getopt(argc,argv,"a:D:i:ln:p:s:t:vzh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
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
      nsamps = atoi (optarg);
      break;

    case 'v':
      verbose++;
      break;

    case 'z':
      zap = 1;
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

  multilog_t* log = multilog_open ("mopsr_udpwaterfall", 0);
  multilog_add (log, stderr);

  udpwaterfall.log = log;
  udpwaterfall.verbose = verbose;

  udpwaterfall.interface = strdup(interface);
  udpwaterfall.port = port;

  udpwaterfall.ant_code = 0;
  udpwaterfall.nant = nant;
  udpwaterfall.nchan = nchan;
  udpwaterfall.nsamps = nsamps;
  udpwaterfall.antenna = antenna;

  udpwaterfall.plot_log = plot_log;
  udpwaterfall.zap = zap;

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
  if (udpwaterfall_init (&udpwaterfall) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // cloear packets ready for capture
  udpwaterfall_prepare (&udpwaterfall);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  udpwaterfall_t * ctx = &udpwaterfall;

  uint64_t frames_per_packet = UDP_DATA / (ctx->nant * ctx->nchan * 2);

  multilog(log, LOG_INFO, "main: frames_per_packet=%"PRIu64"\n", frames_per_packet);

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL );

      //if ((got == UDP_PAYLOAD) || (got + 8 == UDP_PAYLOAD))
      if (got == UDP_PAYLOAD)
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
      StopWatch_Start(&wait_sw);

      mopsr_decode_header(ctx->sock->buf, &seq_no, &(ctx->ant_code));

      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // integrate packet into totals
      integrate_packet (ctx, ctx->sock->buf + UDP_HEADER, UDP_DATA);
      ctx->num_integrated += frames_per_packet;

      //if (ctx->verbose)
      //  print_packet (ctx, ctx->sock->buf + UDP_HEADER, (ctx->nchan * ctx->nsamps * 2));
      if (ctx->num_integrated >= ctx->nsamps)
      {
        //multilog (ctx->log, LOG_INFO, "main: plot_packet\n");
        plot_packet (ctx);
        udpwaterfall_reset (ctx);
        StopWatch_Delay(&wait_sw, sleep_time);

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
      } 
    }
  }

  cpgclos();

  return EXIT_SUCCESS;

}

void integrate_packet (udpwaterfall_t * ctx, char * buffer, unsigned int size)
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
  int re, im;
  float frame_sum = 0;

  for (iframe=0; iframe < nframe; iframe++)
  {
    frame_sum = 0;
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "integrating frame %d\n", iframe);
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      if (ctx->verbose > 1)
        multilog (ctx->log, LOG_INFO, "integrating integrating chan %d\n", ichan);
      for (iant=0; iant < ctx->nant; iant++)
      {
        if (ctx->verbose > 1)
          multilog (ctx->log, LOG_INFO, "integrating integrating ant%d\n", iant);
        if (iant == ctx->antenna )
        {
          re = (int) in[0];
          im = (int) in[1];
          int f = ctx->num_integrated + iframe;
          ctx->data[ichan * ctx->nsamps + f] = (float) ((re*re) + (im*im));
          frame_sum += ctx->data[ichan * ctx->nsamps + f];
        }
        in += 2;
      }
    }
  }
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "integrate_packet: done\n");
}

void plot_packet (udpwaterfall_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  int zap = ctx->zap;
  float xmin = 0.0;
  float xmax = (float) ctx->nchan;
  float v;

  cpgbbuf();
  cpgsci(1);

  char label[64];
  sprintf (label, "Waterfall for Ant %u", mopsr_get_ant_number (ctx->ant_code, ctx->antenna));

  cpgenv(0, ctx->nsamps, 0, ctx->nchan, 0, 0);
  cpglab("Integration", "Channel", label);

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float length = (1.08 * ctx->nsamps) / 1000000;

  float x_min = 0;
  float x_max = ctx->nsamps;

  float y_min = 0;
  float y_max = ctx->nchan;

  float x_res = (x_max-x_min)/ctx->nsamps;
  float y_res = (y_max-y_min)/ctx->nchan;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = ctx->nchan * ctx->nsamps;
  float z_min = 100000000000;
  float z_max = 1;
  float z_avg = 0;

  if (ctx->plot_log && z_min > 0)
    z_min = logf(z_min);
  if (ctx->plot_log && z_max > 0)
    z_max = logf(z_max);

  unsigned int ichan, isamp;
  unsigned int i;
  unsigned int ndat_avg = 0;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (isamp=0; isamp<ctx->nsamps; isamp++)
    {
      i = ichan * ctx->nsamps + isamp;
      if (ctx->plot_log && ctx->data[i] > 0)
        ctx->data[i] = logf(ctx->data[i]);
        if ((!zap) || (zap && ichan != 0))
        {
          if (ctx->data[i] > z_max) z_max = ctx->data[i];
          if (ctx->data[i] < z_min) z_min = ctx->data[i];
        }
        z_avg += ctx->data[i];
        ndat_avg++;
    }
  }

  z_avg /= (float) ndat_avg;

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot: z_min=%f z_max=%f z_avg=%f\n", z_min, z_max, z_avg);

  if (zap)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (isamp=0; isamp<ctx->nsamps; isamp++)
      {
        i = ichan * ctx->nsamps + isamp;
        if (ichan == 0)
          ctx->data[i] = z_avg;
        if (ichan == 0)
          ctx->data[i] = 0;
        //if ((ichan > 105 && ichan < 115) || ichan==127)
        //if (ichan > 105)
        //  ctx->data[i] = z_avg;
      }
    }
  }
  
  cpgimag(ctx->data, ctx->nsamps, ctx->nchan, 1, ctx->nsamps, 1, ctx->nchan, z_min, z_max, trf);
  cpgebuf();
}
