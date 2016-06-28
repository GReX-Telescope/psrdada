/*
 * mopsr_udphist
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
#include "mopsr_util.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

#include "stopwatch.h"

#define MY_UDP_DATA    8192
#define MY_UDP_PAYLOAD 8208

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  // pgplot device
  char * device;

  unsigned int ant_code;

  // the antenna to plot
  int antenna;

  // number of antenna in each packet
  unsigned int nant;

  // number of samples per plot
  unsigned int nsamps;

  // channel to plot
  int channel;

  // number of channels
  unsigned int nchan;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  unsigned int nbin;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  size_t histogram_size;
  unsigned int * histogram;

  uint64_t num_integrated;

  unsigned int chan_loop;

  unsigned int zap;

  int channel_loop;

  float ymax;

  char type;

} udphist_t;

int udphist_init (udphist_t * ctx);
int udphist_prepare (udphist_t * ctx);
int udphist_destroy (udphist_t * ctx);

void mopsr_form_histogram_16 (unsigned int * histogram, char * buffer, ssize_t bytes,  mopsr_util_t * opts);
void mopsr_plot_histogram_16 (unsigned int * histogram, mopsr_util_t * opts);

//void integrate_packet (udphist_t * ctx, char * buffer, unsigned int size);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "mopsr_udphist [options]\n"
     " -h             print help text\n"
     " -a ant         antenna index to display [default all]\n"
     " -c chan        channel to display [default all chans]\n"
     " -D device      display on pgplot device [default /xs]\n"
     " -i interface   ip/interface for inc. UDP packets [default all]\n"
     " -l             loop over channels\n"
     " -p port        port on which to listen [default %d]\n"
     " -q [r|c]       just display real or complex data [default both]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t samps       number of time samples to display [default 128]\n"
     " -z             zap preconfigured channels\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udphist_prepare (udphist_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  //if (ctx->verbose)
  //  multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  //size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, MY_UDP_PAYLOAD);

  udphist_reset(ctx);
}

int udphist_reset (udphist_t * ctx)
{
  memset(ctx->histogram, 0, ctx->histogram_size);
  ctx->num_integrated = 0;
}

int udphist_destroy (udphist_t * ctx)
{
  if (ctx->histogram)
    free (ctx->histogram);
  ctx->histogram = 0;

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

int udphist_init (udphist_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

  ctx->sock->bufsz = sizeof(char) * MY_UDP_PAYLOAD;
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
    multilog(ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  ctx->histogram_size = sizeof(unsigned int) * ctx->ndim * ctx->nbin;
  ctx->histogram = (unsigned int *) malloc (ctx->histogram_size);
  if (!ctx->histogram)
  {
    multilog(ctx->log, LOG_ERR, "init: could allocated %ld bytes for histogram\n", ctx->histogram_size);
    return -1;
  }

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
  udphist_t udphist;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int nant = 2;

  unsigned int nsamps = 4096;

  unsigned int nchan = 128;

  unsigned int chan_loop = 0;

  double sleep_time = 1;

  unsigned int zap = 0;

  int antenna = 0;

  int channel = -1;

  char type = 'b';

  while ((arg=getopt(argc,argv,"a:c:D:i:ln:p:q:s:t:vzh")) != -1) {
    switch (arg) {

    case 'a':
      antenna = atoi(optarg);
      break;

    case 'c':
      channel = atoi (optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'i':
      if (optarg)
        interface = optarg;
      break;

    case 'l':
      chan_loop = 1;
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 'q':
      type = optarg[0];
      break;

    case 's':
      sleep_time = (double) atof (optarg);
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

  multilog_t* log = multilog_open ("mopsr_udphist", 0);
  multilog_add (log, stderr);

  udphist.log = log;
  udphist.verbose = verbose;

  udphist.interface = strdup(interface);
  udphist.port = port;

  udphist.ant_code = 0;
  udphist.nchan = nchan;
  udphist.nant = nant;
  udphist.ndim = 2;
  udphist.nbin = 65536;
  udphist.nsamps = nsamps;

  udphist.antenna = antenna;
  udphist.channel_loop = chan_loop; 
  udphist.channel = channel;
  udphist.type = type;

  udphist.ymax = 0;
  udphist.zap = zap;

  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "mopsr_udphist: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_udphist: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  // allocate require resources, open socket
  if (udphist_init (&udphist) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  udphist_prepare (&udphist);

  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  udphist_t * ctx = &udphist;
  mopsr_util_t opts;
  mopsr_hdr_t hdr;

  opts.lock_flag  = -1;
  opts.plot_log   = 0;
  opts.zap        = 0;
  opts.ant        = -1;
  opts.chans[0]   = -1;
  opts.chans[1]   = 127;
  if (channel != -1)
  {
    opts.chans[0]   = channel;
    opts.chans[1]   = channel;
  }
  opts.nbin       = 65536;
  opts.ndim       = 2;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;
  opts.nchan      = nchan;
  opts.nant       = nant;
  opts.ant        = antenna;

  unsigned nbyte = 2;
  size_t   frame_size = ctx->nant * ctx->nchan * opts.ndim * nbyte; 
  uint64_t frames_per_packet = MY_UDP_DATA / frame_size;

  multilog(log, LOG_INFO, "mopsr_udphist: nsamps=%u frame_size=%ld frames_per_packet=%"PRIu64"\n", nsamps, frame_size, frames_per_packet);

  unsigned int npackets = nsamps / frames_per_packet;
  if (nsamps % frames_per_packet)
    npackets++;

  char * data = (char *) malloc (MY_UDP_DATA * npackets);
  multilog(log, LOG_INFO, "mopsr_udphist: npackets=%u datasize=%d\n", npackets, MY_UDP_DATA * npackets);

  while (!quit_threads) 
  {
    ctx->sock->have_packet = 0;

    //fprintf (stderr, "acquiring packet\n");

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      got = recvfrom ( ctx->sock->fd, ctx->sock->buf, MY_UDP_PAYLOAD, 0, NULL, NULL );

      if ((got == MY_UDP_PAYLOAD) || (got + 8 == MY_UDP_PAYLOAD))
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
        multilog (log, LOG_ERR, "main: received %d bytes, expected %d\n", got, MY_UDP_PAYLOAD);
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

      memcpy ( data + (ctx->num_integrated * frame_size), ctx->sock->buf + UDP_HEADER, MY_UDP_DATA);
      ctx->num_integrated += frames_per_packet;

      if (ctx->num_integrated >= ctx->nsamps)
      {
        mopsr_form_histogram_16 (ctx->histogram, data, ctx->num_integrated * frame_size, &opts);
        mopsr_plot_histogram_16 (ctx->histogram, &opts);
        udphist_reset (ctx);

        DelayTimer(&wait_sw, sleep_time);

        if (ctx->channel_loop)
          ctx->channel = (ctx->channel + 1) % ctx->nchan;

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, MY_UDP_PAYLOAD);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
      } 
    }
  }

  cpgclos();

  if (data)
    free(data);

  udphist_destroy (ctx);

  return EXIT_SUCCESS;

}

void mopsr_form_histogram_16 (unsigned int * histogram, char * buffer, ssize_t bytes,  mopsr_util_t * opts)
{
  unsigned ibin = 0;
  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  unsigned nframe = bytes / (opts->nant * opts->nchan * 2 * 2);

  char conv[2];
  char * tmp = buffer;

  int16_t * in = (int16_t *) conv;
  int16_t re, im;
  int re_bin, im_bin;

  unsigned int re_out_index;
  unsigned int im_out_index;

  // set re and im bins to 0 [idim][ibin]
  for (ibin=0; ibin < opts->nbin; ibin++)
  {
    histogram[ibin] = 0;                  // Re
    histogram[opts->nbin + ibin] = 0;     // Im
  }

  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < opts->nchan; ichan++)
    {
      for (iant=0; iant < opts->nant; iant++)
      {
        if (iant == opts->ant)
        {
          if ((opts->chans[0] == -1) || ((opts->chans[0] >= 0) && (ichan >= opts->chans[0]) && (ichan <= opts->chans[1])))
          {
            conv[0] = tmp[0];
            conv[1] = tmp[1];
            re = in[0];

            conv[0] = tmp[2];
            conv[1] = tmp[3];
            im = in[0];

            //fprintf (stderr, "re=%"PRIi16" im=%"PRIi16"\n", re, im);
            re = re >> 4;
            im = im >> 4;

            // determine the bin for each value [-32768 -> 32767]
            re_bin = ((int) re) + 32768;
            im_bin = ((int) im) + 32768;


            histogram[re_bin]++;
            histogram[opts->nbin + im_bin]++;
          }
        }
        tmp += 4;
        //in += 2;
      }
    }
  }
}

void mopsr_plot_histogram_16 (unsigned int * histogram, mopsr_util_t * opts)
{
  size_t nbytes = sizeof(float) * opts->nbin;
  float * x = (float *) malloc (nbytes);
  if (!x)
  {
    fprintf (stderr, "could not allocate %ld bytes for x\n", nbytes);
    return;
  }
  float * re = (float *) malloc (nbytes);
  if (!re)
  { 
    fprintf (stderr, "could not allocate %ld bytes for re\n", nbytes);
    return;
  }
  float * im = (float *) malloc (nbytes);
  if (!im)
  { 
    fprintf (stderr, "could not allocate %ld bytes for im\n", nbytes);
    return;
  }

  int ibin;
  int ichan, iant;

  float ymin = 0;
  float ymax_re= 0;
  float ymax_im = 0;
  char all_zero = 1;

  cpgeras();

  for (ibin=0; ibin<opts->nbin; ibin++)
  {
    x[ibin] = ((float) ibin) - 32768;
    re[ibin] = (float) histogram[ibin];
    im[ibin] = (float) histogram[opts->nbin + ibin];

    if (re[ibin] > ymax_re)
    {
      ymax_re = re[ibin];
     }
    if (im[ibin] > ymax_im)
    {
      ymax_im = im[ibin];
    }
    if ((x[ibin] != 0) && (re[ibin] > 0 || im[ibin] > 0))
      all_zero = 0;
  }

  float xmin = -32768;
  float xmax = 32768;

  for (ibin=0; ibin<opts->nbin; ibin++)
  {
    if (((re[ibin] > 0) || (im[ibin] > 0)) && (xmin == -32768))
    {
      //fprintf (stderr, "xmin ibin=%d\n", ibin);
      xmin = ibin - 32768;
    }
  }

  for (ibin=opts->nbin-1; ibin>0; ibin--)
  {     
    if (((re[ibin] > 0) || (im[ibin] > 0)) && (xmax == 32768))
    {
      //fprintf (stderr, "xmax ibin=%d\n", ibin);
      xmax = ibin - 32768;
    }
  }

  cpgbbuf();
  cpgsci(1);

  char title[128];
  char label[64];

  if (opts->chans[0] < 0)
    sprintf(title, "Module %d Histogram, all channels", opts->ant_id);
  else if (opts->chans[0] == opts->chans[1])
    sprintf(title, "Module %d Histogram, channel %d", opts->ant_id, opts->chans[0]);
  else
    sprintf(title, "Module %d Histogram, channels %d to %d", opts->ant_id, opts->chans[0], opts->chans[1]);

  ymax_re *= 1.1;
  ymax_im *= 1.1;

  xmin = -200;
  xmax = 200;

  // real
  fprintf (stderr, "xmin=%f xmax=%f\n", xmin, xmax);
  cpgswin(xmin, xmax, ymin, ymax_re);
  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Count", title);

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_re);
  cpgsls(1);

  // Real line
  if (!opts->plot_plain)
    cpgmtxt("T", -2.2, 0.05, 0.0, "Real");
  cpgsci(2);
  cpgslw(3);
  cpgbin (opts->nbin, x, re, 0);
  cpgslw(1);
  cpgsci(1);

  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("State", "Count", "");
  fprintf (stderr, "xmin=%f xmax=%f\n", xmin, xmax);
  cpgswin(xmin, xmax, ymin, ymax_im);

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_im);
  cpgsls(1);

  // Im line
  cpgmtxt("T", -2.2, 0.05, 0.0, "Imag");
  cpgslw(3);
  cpgsci(3);
  cpgbin (opts->nbin, x, im, 0);
  cpgsci(1);
  cpgslw(1);

  cpgebuf();

  free (x);
  free (re);
  free (im);
}

