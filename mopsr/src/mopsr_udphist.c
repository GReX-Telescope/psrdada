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
void hist_packet (udphist_t * ctx, char * buffer, unsigned int size);
void dump_packet (udphist_t * ctx, char * buffer, unsigned int size);
void print_packet (udphist_t * ctx, char * buffer, unsigned int size);
void plot_packet (udphist_t * ctx);

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

  unsigned int nant = 2;

  unsigned int nsamps = 128;

  unsigned int nchan = 128;

  unsigned int chan_loop = 0;

  double sleep_time = 1000000;

  unsigned int zap = 0;

  int antenna = -1;

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
    multilog(log, LOG_INFO, "mopsr_dbplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    multilog(log, LOG_INFO, "mopsr_dbplot: error opening plot device\n");
    exit(1);
  }
  cpgask(0);

  // allocate require resources, open socket
  if (udphist_init (&udphist) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // cloear packets ready for capture
  udphist_prepare (&udphist);

  uint64_t seq_no = 0;
  uint64_t prev_seq_no = 0;
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;

  udphist_t * ctx = &udphist;

  uint64_t frames_per_packet = UDP_DATA / (ctx->nant * ctx->nchan * 2);

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

      mopsr_decode_header(ctx->sock->buf, &seq_no, &(ctx->ant_code));

      if (ctx->verbose > 1) 
        multilog (ctx->log, LOG_INFO, "main: seq_no= %"PRIu64" difference=%"PRIu64" packets\n", seq_no, (seq_no - prev_seq_no));
      prev_seq_no = seq_no;

      // integrate packet into totals
      //integrate_packet (ctx, ctx->sock->buf + UDP_HEADER, UDP_DATA);
      hist_packet (ctx, ctx->sock->buf + UDP_HEADER, UDP_DATA);
      ctx->num_integrated += frames_per_packet;

      //if (ctx->verbose)
      //  print_packet (ctx, ctx->sock->buf + UDP_HEADER, (ctx->nchan * ctx->nsamps * 2));
      if (ctx->num_integrated >= ctx->nsamps)
      {
        //multilog (ctx->log, LOG_INFO, "main: plot_packet\n");
        plot_packet (ctx);
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

  return EXIT_SUCCESS;

}

void print_packet (udphist_t * ctx, char * buffer, unsigned int size)
{
  char ant0r[9];
  char ant0i[9];
  char ant1r[9];
  char ant1i[9];

  unsigned ibyte = 0;
  fprintf(stderr, "chan\tant0imag ant0real ant1imag ant1real\n");
  for (ibyte=0; ibyte<size; ibyte += 4)
  {
    char_to_bstring (ant0i, buffer[ibyte+0]);
    char_to_bstring (ant0r, buffer[ibyte+1]);
    char_to_bstring (ant1i, buffer[ibyte+2]);
    char_to_bstring (ant1r, buffer[ibyte+3]);
    fprintf (stderr, "%d\t%s %s %s %s\t%d\t%d\t%d\t%d\n", ibyte/4, ant0i, ant0r, ant1i, ant1r, (int8_t) buffer[ibyte+0], (int8_t) buffer[ibyte+1], (int8_t) buffer[ibyte+2], (int8_t) buffer[ibyte+3]);
  }
}

void dump_packet (udphist_t * ctx, char * buffer, unsigned int size)
{
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open ("packet.raw", flags, perms);
  ssize_t n_wrote = write (fd, buffer, size);
  close (fd);
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

  for (ibin=0; ibin<ctx->nbin; ibin++)
  {
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
  cpgbbuf();
  cpgsci(1);

  char label[64];
  if (ctx->channel < 0)
    sprintf(label, "Complex Histogram for all channels");
  else
    sprintf(label, "Complex Histogram for Channel %d", ctx->channel);

  cpgenv(-128, 127, 0, ctx->ymax * 1.1, 0, 0);
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
