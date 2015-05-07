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

  uint64_t **** data;

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

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);

  udphist_reset(ctx);
}

int udphist_reset (udphist_t * ctx)
{
  unsigned int ichan,iant,idim,ibin;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      for (idim=0; idim<ctx->ndim; idim++)
      {
        for (ibin=0; ibin<ctx->nbin; ibin++)
        {
          ctx->data[ichan][iant][idim][ibin] = 0;
        }
      }
    }
  }
  ctx->num_integrated = 0;
}

int udphist_destroy (udphist_t * ctx)
{
  unsigned ichan,iant,idim;
  if (ctx->data)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (iant=0; iant<ctx->nant; iant++)
      {
        for (idim=0; idim<ctx->ndim; idim++)
        {
          free (ctx->data[ichan][iant][idim]);
          ctx->data[ichan][iant][idim] = 0;
        }
        free (ctx->data[ichan][iant]);
        ctx->data[ichan][iant] = 0;
      }
      free (ctx->data[ichan]);
      ctx->data[ichan] = 0;
    }
    free(ctx->data);
  }
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

int udphist_init (udphist_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "mopsr_udpdb_init_receiver()\n");

  // create a MOPSR socket which can hold variable num of UDP packet
  ctx->sock = mopsr_init_sock();

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

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: mallocing %d x %d * %d * %d\n", ctx->nchan, ctx->nant, ctx->ndim, ctx->nbin);
  unsigned ichan,iant,idim;
  ctx->data = (uint64_t ****) malloc (sizeof (uint64_t ***) * ctx->nchan);
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    ctx->data[ichan] = (uint64_t ***) malloc (sizeof (uint64_t **) * ctx->nant);
    for (iant=0; iant<ctx->nant; iant++)
    {
      ctx->data[ichan][iant] = (uint64_t **) malloc (sizeof (uint64_t *) * ctx->ndim);
      for (idim=0; idim<ctx->ndim; idim++)
      {
        ctx->data[ichan][iant][idim] = (uint64_t *) malloc (sizeof (uint64_t) * ctx->nbin);
      }
    }
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

  unsigned int nant = 16;

  unsigned int nsamps = 6;

  unsigned int nchan = 40;

  unsigned int chan_loop = 0;

  double sleep_time = 1000000;

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
  udphist.nbin = 256;
  udphist.nsamps = nsamps;

  udphist.antenna = antenna;
  udphist.channel_loop = chan_loop; 
  udphist.channel = channel;
  udphist.type = type;

  udphist.ymax = 0;
  udphist.zap = zap;

  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

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
  opts.nbin       = 256;
  opts.ndim       = 2;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;
  opts.nchan      = nchan;
  opts.nant       = nant;
  opts.ant        = antenna;

  size_t   frame_size = ctx->nant * ctx->nchan * opts.ndim; 
  uint64_t frames_per_packet = UDP_DATA / frame_size;

  multilog(log, LOG_INFO, "mopsr_udphist: nsamps=%u frame_size=%ld frames_per_packet=%"PRIu64"\n", nsamps, frame_size, frames_per_packet);

  unsigned int npackets = nsamps / frames_per_packet;
  if (nsamps % frames_per_packet)
    npackets++;

  multilog(log, LOG_INFO, "mopsr_udphist: npackets=%u\n", npackets);

  char * data = (char *) malloc (UDP_DATA * npackets);
  unsigned int * histogram = (unsigned int *) malloc (sizeof(unsigned int) * opts.ndim * opts.nbin);

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
      StopWatch_Start(&wait_sw);

      mopsr_decode (ctx->sock->buf, &hdr);

      if ((antenna == 0) || (antenna == 1))
      {
        opts.ant_id = mopsr_get_ant_number (hdr.ant_id, antenna);
      }
      else
      {
        opts.ant_id = mopsr_get_ant_number (hdr.ant_id2, antenna % 2);
      }
      opts.ant_id = mopsr_get_new_ant_number (antenna);

      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", hdr.seq_no, (hdr.seq_no - prev_seq_no));
      prev_seq_no = hdr.seq_no;

      memcpy ( data + (ctx->num_integrated * frame_size), ctx->sock->buf + UDP_HEADER, UDP_DATA);
      ctx->num_integrated += frames_per_packet;

      if (ctx->num_integrated >= ctx->nsamps)
      {
        mopsr_form_histogram (histogram, data, ctx->num_integrated * frame_size, &opts);
        mopsr_plot_histogram (histogram, &opts);

        udphist_reset (ctx);
        StopWatch_Delay(&wait_sw, sleep_time);
        if (ctx->channel_loop)
          ctx->channel = (ctx->channel + 1) % ctx->nchan;

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: clearing packets at socket\n");
        size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, UDP_PAYLOAD);
        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "main: cleared %d packets\n", cleared);
      } 
    }
  }

  cpgclos();

  if (data)
    free(data);

  if (histogram)
    free (histogram);

  return EXIT_SUCCESS;

}

void hist_packet (udphist_t * ctx, char * buffer, unsigned int size)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "hist_packet()\n");

  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  unsigned nframe = size / (ctx->nant * ctx->nchan * 2);

  int8_t * in = (int8_t *) buffer;
  int8_t re, im;
  int re_bin, im_bin;

  unsigned int re_out_index;
  unsigned int im_out_index;

  unsigned int chan_stride = ctx->nsamps * ctx->ndim;

  for (iframe=0; iframe < nframe; iframe++)
  {
    if (ctx->verbose > 2)
      multilog (ctx->log, LOG_INFO, "histing frame %d\n", iframe);
    for (ichan=0; ichan < ctx->nchan; ichan++)
    {
      if (ctx->verbose > 2)
        multilog (ctx->log, LOG_INFO, "histing chan %d\n", ichan);
      for (iant=0; iant < ctx->nant; iant++)
      {
        if (ctx->verbose > 2)
          multilog (ctx->log, LOG_INFO, "histing ant %d\n", iant);
        re = in[0];
        im = in[1];

        // determine the bin for each value [-128 -> 127]
        re_bin = ((int) re) + 128;
        im_bin = ((int) im) + 128;

        ctx->data[ichan][iant][0][re_bin]++;
        ctx->data[ichan][iant][1][im_bin]++;

        in += 2;
      }
    }
  }
}

void plot_packet (udphist_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  float x[256];
  float re[2][256];
  float im[2][256];
  int ibin;
  int ichan, iant;

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: getting limits\n");

  for (ibin=0; ibin<ctx->nbin; ibin++)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "plot_packet: getting limits\n");
    for (iant=0; iant<ctx->nant; iant++)
    {
      re[iant][ibin] = 0;
      im[iant][ibin] = 0;
    }

    x[ibin] = ((float)ibin) - 128;
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (iant=0; iant<ctx->nant; iant++)
      {
        // only collect iant and ichan if specified, else all
        if ( ((ctx->antenna < 0) || (ctx->antenna == iant)) &&
             ((ctx->channel < 0) || (ctx->channel == ichan)) )
        {
          re[iant][ibin] += (float) ctx->data[ichan][iant][0][ibin];
          im[iant][ibin] += (float) ctx->data[ichan][iant][1][ibin];

          if (re[iant][ibin] > ctx->ymax)
          {
            ctx->ymax = re[iant][ibin];
          }
          if (im[iant][ibin] > ctx->ymax)
          {
            ctx->ymax = im[iant][ibin];
          }
        }
      }
    }
  }

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet: plotting\n");

  cpgbbuf();
  cpgsci(1);

  char label[64];
  if (ctx->channel < 0)
    sprintf(label, "Complex Histogram for all channels");
  else
    sprintf(label, "Complex Histogram for Channel %d", ctx->channel);

  cpgenv(-128, 128, 0, ctx->ymax * 1.1, 0, 0);
  cpglab("State", "Count", label);

  int colour = 2;
  float label_y_offset = 0.5;
  for (iant=0; iant<ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      if ((ctx->type == 'b') || (ctx->type == 'r'))
      {
        // Real line
        cpgsci(colour);
        sprintf (label, "Real Ant %u", mopsr_get_ant_number (ctx->ant_code, iant));
        cpgmtxt("T", label_y_offset, 0.0, 0.0, label);
        cpgbin (ctx->nbin, x, re[iant], 0);
        //cpgline (ctx->nbin, x, re[iant]);
        label_y_offset += 1;
        colour++;
      }

      if ((ctx->type == 'b') || (ctx->type == 'c'))
      {
        cpgsci(colour);
        sprintf (label, "Imag Ant %u", mopsr_get_ant_number (ctx->ant_code, iant));
        cpgmtxt("T", label_y_offset, 0.0, 0.0, label);
        cpgbin (ctx->nbin, x, im[iant], 0);
        label_y_offset += 1;
        colour++;
      }
    }
  }
  cpgebuf();
}
