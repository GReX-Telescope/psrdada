/*
 * caspsr_udptsplot
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
#include <pthread.h>
#include <float.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <math.h>
#include <cpgplot.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "caspsr_def.h"
#include "caspsr_udp.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"
#include "stopwatch.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

#define IDLE 0
#define ACTIVE 1

typedef struct {

  multilog_t * log;

  caspsr_sock_t * socks[2];

  char * interfaces[2];

  int ports[2];

  // pgplot device
  char * device;

  // number of data points in time series
  unsigned int ndat;
  unsigned int ndat_plot;
  unsigned int ndat_scrunch;

  // number of polarisations [should always be 2]
  unsigned int npol;

  float tsamp;

  unsigned int verbose;

  int8_t ** raw_data;
  size_t raw_data_size;

  int8_t * raw_combined;

  float * x_points;

  float ** y_points;

  uint64_t num_integrated;

  uint64_t to_integrate;

  float ymin;

  float ymax;

  float xmin;

  float xmax;

  size_t pkt_size;

  size_t data_size;

  pthread_cond_t cond;

  pthread_mutex_t mutex;

  int control_state;

  pthread_cond_t conds[2];

  pthread_mutex_t mutexes[2];

  int64_t starts[2];

  char full[2];

  unsigned thread_count;

} udptsplot_t;

int get_packet (caspsr_sock_t * sock, unsigned pkt_size);
int udptsplot_init (udptsplot_t * ctx);
int udptsplot_prepare (udptsplot_t * ctx);
int udptsplot_destroy (udptsplot_t * ctx);
int udptsplot_reset (udptsplot_t * ctx);
void udptsplot_thread (void * arg);

void append_data (udptsplot_t * ctx);
void plot_data (udptsplot_t * ctx);

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
     "caspsr_udptsplot [options]\n"
     " -i interface1  first UDP interface [default 192.168.3.100]\n"
     " -j interface2  second UDP interface [default 192.168.4.100]\n"
     " -n nsamp       total number of timesamples to capture per plot [default 4096]\n"
     " -p port1       port on which to listen [default %d]\n"
     " -q port2       port on which to listen [default %d]\n"
     " -s secs        sleep this many seconds between plotting [default 0.5]\n"
     " -t num         sampling time in micro seconds [default 0.00125]\n"
     " -v             verbose messages\n"
     " -h             print help text\n"
     );
}

int udptsplot_prepare (udptsplot_t * ctx)
{
  unsigned i;
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "udptsplot_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket: %ld bytes\n", ctx->pkt_size);
  for (i=0; i<2; i++)
    dada_sock_clear_buffered_packets(ctx->socks[i]->fd, ctx->pkt_size);

  udptsplot_reset(ctx);
}

int udptsplot_reset (udptsplot_t * ctx)
{
  unsigned ipol, idat;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (idat=0; idat < ctx->ndat_plot; idat++)
    {
      ctx->y_points[ipol][idat] = 0;
    }
  }
  ctx->num_integrated = 0;
}

int udptsplot_destroy (udptsplot_t * ctx)
{

  unsigned ipol;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    if (ctx->y_points[ipol])
      free(ctx->y_points[ipol]);
    ctx->y_points[ipol] = 0;
  }
  if (ctx->y_points)
    free(ctx->y_points);
  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;

  unsigned i;
  for (i=0; i<2; i++)
  {
    close(ctx->socks[i]->fd);
    caspsr_free_sock (ctx->socks[i]);
    ctx->socks[i] = 0;
  }
}

/*
 * Close the udp socket and file
 */

int udptsplot_init (udptsplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "caspsr_udpdb_init()\n");

  unsigned i;
  for (i=0; i<2; i++)
  {
    // create a MOPSR socket which can hold variable num of UDP packet
    ctx->socks[i] = caspsr_init_sock();

    ctx->socks[i]->size = 9000;
    if (ctx->socks[i]->buffer)
      free (ctx->socks[i]->buffer);
    ctx->socks[i]->buffer = (char *) malloc (ctx->socks[i]->size);
    assert(ctx->socks[i]->buffer != NULL);

    // open socket
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "init: creating udp socket on %s:%d\n", ctx->interfaces[i], ctx->ports[i]);
    ctx->socks[i]->fd = dada_udp_sock_in(ctx->log, ctx->interfaces[i], ctx->ports[i], ctx->verbose);
    if (ctx->socks[i]->fd < 0) {
      multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
      return -1;
    }

    // set the socket size to 128 MB
    int sock_buf_size = 128*1024*1024;
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "init: setting buffer size to %d\n", sock_buf_size);
    dada_udp_sock_set_buffer_size (ctx->log, ctx->socks[i]->fd, ctx->verbose, sock_buf_size);
  }

  // get a packet to determine the packet size and type of data
  ctx->pkt_size = recvfrom (ctx->socks[0]->fd, ctx->socks[0]->buffer, 9000, 0, NULL, NULL);
  size_t data_size = ctx->pkt_size - UDP_HEADER;
  multilog (ctx->log, LOG_INFO, "init: pkt_size=%ld data_size=%ld\n", ctx->pkt_size, data_size);
  ctx->data_size = ctx->pkt_size - UDP_HEADER;

  // decode the header
  multilog (ctx->log, LOG_INFO, "init: decoding packet\n");
  uint64_t seq_no, ch_id;
  caspsr_decode_header (ctx->socks[0]->buffer, &seq_no, &ch_id);
  multilog (ctx->log, LOG_INFO, "init: seq=%lu ch_id=%lu\n", seq_no, ch_id);

  ctx->npol = 2;

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: setting non_block\n");
  for (i=0; i<2; i++)
  {
    sock_nonblock(ctx->socks[i]->fd);
  }

  ctx->ndat_plot = ctx->ndat / ctx->ndat_scrunch;

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->ndat_plot);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->npol);

  ctx->raw_data_size = sizeof(int8_t) * ctx->ndat * ctx->npol / 2;
  if (ctx->raw_data_size < 32*1024*1024)
    ctx->raw_data_size = 32*1024*1024;

  ctx->raw_data = (int8_t **) malloc (sizeof(int8_t *) * 2);
  ctx->raw_data[0] = (int8_t *) malloc (ctx->raw_data_size);
  ctx->raw_data[1] = (int8_t *) malloc (ctx->raw_data_size);

  ctx->raw_combined = (int8_t *) malloc (sizeof(int8_t) * ctx->ndat * ctx->npol);


  unsigned int ipol;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    ctx->y_points[ipol] = (float *) malloc (sizeof(float) * ctx->ndat_plot);
  }
  uint64_t idat;
  for (idat=0; idat < ctx->ndat_plot; idat++)
    ctx->x_points[idat] = (float) (idat * ctx->ndat_scrunch  * ctx->tsamp);

  return 0;
}

int get_packet (caspsr_sock_t * sock, unsigned pkt_size)
{
  int got;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1e9;
  sock->have_packet = 0;
  while (!sock->have_packet && !quit_threads)
  {
    got = recvfrom ( sock->fd, sock->buffer, pkt_size, 0, NULL, NULL );
    if (got == pkt_size)
    {
      sock->have_packet = 1;
      timeouts = 0;
    }
    else if (got == -1)
    {
      int errsv = errno;
      if (errsv == EAGAIN)
      {
        timeouts++;
        if (timeouts > timeout_max)
        {
          fprintf (stderr, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
          quit_threads = 1;
        }
      }
      else
      {
        fprintf (stderr, "recvfrom failed %s\n", strerror(errsv));
        return 0;
      }
    }
    else // we received a packet of the WRONG size, ignore it
    {
      fprintf (stderr, "received %d bytes, expected %d\n", got, pkt_size);
      quit_threads = 1;
    }
  }
  if (timeouts > timeout_max)
  {
    fprintf (stderr, "timeouts[%"PRIu64"] > timeout_max[%"PRIu64"]\n",timeouts, timeout_max);
  }

  return got;
}

void udptsplot_thread (void * arg)
{
  udptsplot_t * ctx = (udptsplot_t *) arg;

  // lock the master mutex
  pthread_mutex_lock (&(ctx->mutex));
  
  unsigned tid = ctx->thread_count;
  ctx->thread_count++;

  pthread_cond_signal (&(ctx->cond));
  pthread_mutex_unlock (&(ctx->mutex));

  // thread specific control
  pthread_mutex_t * mutex_recv = &(ctx->mutexes[tid]);
  pthread_cond_t * cond_recv = &(ctx->conds[tid]);

  // socket for this thread
  caspsr_sock_t * sock = ctx->socks[tid];
  void * raw_data= ctx->raw_data[tid];

  uint64_t raw_seq, chid;
  char done;
  int got;
  int64_t offset;

  while (!quit_threads)
  {
    // wait for blocks to be marked as empty
    pthread_mutex_lock (mutex_recv);
    while (ctx->full[tid] && ctx->control_state == ACTIVE)
      pthread_cond_wait (cond_recv, mutex_recv);
    pthread_mutex_unlock (mutex_recv);

    bzero (raw_data, ctx->raw_data_size);

    // clear buffered packets
    size_t cleared = dada_sock_clear_buffered_packets(sock->fd, ctx->pkt_size);
    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "worker [%u]: cleared %d packets\n", tid, cleared);
    
    // simply dump twice as many packets as is necessary into local buffer
    done = 0;
    while (!done)
    {
      // get a fresh packet
      got = get_packet (sock, ctx->pkt_size);

      if (got == ctx->pkt_size)
      {
        // decode header and seq
        caspsr_decode_header (sock->buffer, &raw_seq, &chid);

        // update the common packet header
        if (ctx->starts[tid] == -1)
        {
          ctx->starts[tid] = (int64_t) raw_seq;
        }

        offset = ((int64_t) raw_seq - ctx->starts[tid]) * 4;

        if (offset + ctx->data_size < ctx->raw_data_size)
        {
          //multilog(ctx->log, LOG_INFO, "worker [%u]: memcpy offset=%ld\n", tid, offset);
          memcpy (raw_data + offset, sock->buffer + UDP_HEADER, ctx->data_size);

          //if (tid == 0)
          //  bzero(raw_data + offset, ctx->data_size);
        }
        else
          done = 1;
      }
    }

    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "worker [%u]: setting full = 1\n", tid);

    // indicate that we have filled the data buffer
    ctx->full[tid] = 1;

    if (ctx->verbose > 1)
      multilog(ctx->log, LOG_INFO, "worker [%u]: signalling master\n", tid);
    // signal master thread
    pthread_cond_signal (&(ctx->cond));
    pthread_mutex_unlock (&(ctx->mutex));
  }
}



int main (int argc, char **argv)
{
  /* Interface on which to listen for udp packets */
  char * interface1 = "any";
  char * interface2 = "any";

  /* port on which to listen for incoming connections */
  int port1 = 33108;
  int port2 = 33109;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  char * device = "/xs";

  /* actual struct with info */
  udptsplot_t udptsplot;

  /* Pointer to array of "read" data */
  char *src;

  unsigned int ndat = 4096;

  double sleep_time = 1.0f / 1e6;

  float tsamp = 0.00125;

  float xmin = 0.0;
  float xmax = 0.0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  int ndat_scrunch = 1;

  while ((arg=getopt(argc,argv,"a:D:i:j:n:p:q:s:t:vh")) != -1) {
    switch (arg) {

    case 'a':
      ndat_scrunch = atoi (optarg);
      break;

    case 'D':
      device = strdup(optarg);
      break;

    case 'n':
      ndat = atoi (optarg);
      break;

    case 'i':
      if (optarg)
        interface1 = optarg;
      break;

    case 'j':
      if (optarg)
        interface2 = optarg;
      break;

    case 'p':
      port1 = atoi (optarg);
      break;

    case 'q':
      port2 = atoi (optarg);
      break;

    case 's':
      sleep_time = (double) atof (optarg);
      break;

    case 't':
      tsamp = atof (optarg);
      break;

    case 'v':
      verbose++;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  multilog_t* log = multilog_open ("caspsr_udptsplot", 0);
  multilog_add (log, stderr);

  udptsplot.log = log;
  udptsplot.verbose = verbose;

  udptsplot.interfaces[0] = strdup(interface1);
  udptsplot.interfaces[1] = strdup(interface2);
  udptsplot.ports[0] = port1;
  udptsplot.ports[1] = port2;

  udptsplot.ndat = ndat;
  udptsplot.ndat_scrunch = ndat_scrunch;
  udptsplot.tsamp = tsamp;

  pthread_cond_init( &(udptsplot.cond), NULL);
  pthread_mutex_init( &(udptsplot.mutex), NULL);
  unsigned i;
  for (i=0; i<2; i++)
  {
    pthread_cond_init( &(udptsplot.conds[i]), NULL);
    pthread_mutex_init( &(udptsplot.mutexes[i]), NULL);
  }

  if ((xmin == 0) && (xmax == 0))
  {
    xmin = 0;
    xmax = (float) ndat * tsamp;
  }

  // allocate require resources, open socket
  if (udptsplot_init (&udptsplot) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  udptsplot.xmin = xmin;
  udptsplot.xmax = xmax;
  udptsplot.ymin = ymin;
  udptsplot.ymax = ymax;

  multilog(log, LOG_INFO, "caspsr_udptsplot: %f %f\n", udptsplot.xmin, udptsplot.xmax);

  // initialise data rate timing library 
  stopwatch_t wait_sw;

  if (verbose)
    multilog(log, LOG_INFO, "caspsr_udptsplot: using device %s\n", device);

  udptsplot.control_state = IDLE;
  udptsplot.thread_count = 0;

  // cloear packets ready for capture
  udptsplot_prepare (&udptsplot);

  int64_t prev_seq_nos[2] = {0, 0};
  size_t got = 0;
  int errsv = 0;
  uint64_t timeouts = 0;
  uint64_t timeout_max = 1000000;
  uint64_t raw, ch_id;
  int64_t seq_no, offset;

  udptsplot_t * ctx = &udptsplot;

  ctx->starts[0] = -1;
  ctx->starts[1] = -1;
  ctx->full[0] = 0;
  ctx->full[1] = 0;

  multilog (ctx->log, LOG_INFO, "main: lock master\n");

  // first wait for each of the threads to have started
  pthread_mutex_lock (&(ctx->mutex));
  multilog (ctx->log, LOG_INFO, "main: lock workers\n");
  pthread_mutex_lock (&(ctx->mutexes[0]));
  pthread_mutex_lock (&(ctx->mutexes[1]));

  multilog (ctx->log, LOG_INFO, "main: create threads\n");
  pthread_t thread1_id;
  pthread_t thread2_id;
  pthread_create (&thread1_id, NULL, (void *) udptsplot_thread, (void *) ctx);
  pthread_create (&thread2_id, NULL, (void *) udptsplot_thread, (void *) ctx);

  StartTimer(&wait_sw);

  multilog (ctx->log, LOG_INFO, "main: waiting for thread_count == 2\n");
  // start sequence number for each thread
  // wait for each thread to have started on the primary mutex 
  while (ctx->thread_count != 2)
    pthread_cond_wait (&(ctx->cond), &(ctx->mutex));

  multilog (ctx->log, LOG_INFO, "main: setting state to active\n");
  // tell them to start
  ctx->control_state = ACTIVE;

  // unlock primary
  pthread_mutex_unlock (&(ctx->mutex));

  // and unlock them
  pthread_cond_signal (&(ctx->conds[0]));
  pthread_cond_signal (&(ctx->conds[1]));
  pthread_mutex_unlock (&(ctx->mutexes[0]));
  pthread_mutex_unlock (&(ctx->mutexes[1]));

  // wait for them to finish
  while (!quit_threads) 
  {
    // try to lock the prinary mutex, waiting for full buffers
    pthread_mutex_lock (&(ctx->mutex));
    while (!ctx->full[0] || !ctx->full[1])
    {
      pthread_cond_wait (&(ctx->cond), &(ctx->mutex));
    }
    
    // align data
    append_data (ctx);

    // plot data
    plot_data (ctx);

    udptsplot_reset (ctx);

    // wait the approprirate time before signalling threads to start again
    DelayTimer(&wait_sw, sleep_time);

    StartTimer(&wait_sw);

    // acquire the 
    pthread_mutex_lock (&(ctx->mutexes[0]));
    pthread_mutex_lock (&(ctx->mutexes[1]));

    // start sequence number for each thread
    ctx->starts[0] = -1;
    ctx->starts[1] = -1;
    ctx->full[0] = 0;
    ctx->full[1] = 0;

    // unlock primary
    pthread_mutex_unlock (&(ctx->mutex));
    
    // and unlock them
    pthread_cond_signal (&(ctx->conds[0]));
    pthread_cond_signal (&(ctx->conds[1]));
    pthread_mutex_unlock (&(ctx->mutexes[0]));
    pthread_mutex_unlock (&(ctx->mutexes[1]));
  }

  void * result;
  pthread_join (thread1_id, &result);
  pthread_join (thread2_id, &result);

  return EXIT_SUCCESS;
}

// copy data from packet in the dat input buffer
void append_data (udptsplot_t * ctx)
{
  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "append_data()\n");

  int64_t packets[2] = { 0, 0 };
  int64_t offsets[2] = { 0, 0 };

  if (ctx->verbose)
    fprintf (stderr, "starts %ld %ld\n", ctx->starts[0], ctx->starts[1]);

  // determine the byte offsets for each thread
  if (ctx->starts[0] > ctx->starts[1])
    offsets[1] = (ctx->starts[0] - ctx->starts[1]) + 1024;
  else
    offsets[0] = (ctx->starts[1] - ctx->starts[0]) + 1024;

  packets[0] = (ctx->starts[0] - ctx->starts[1]) / 1024;
  packets[1] = (ctx->starts[1] - ctx->starts[0]) / 1024;

  if (ctx->verbose)
    fprintf (stderr, "packets differential: %ld, %ld\n", packets[0], packets[1]);

  offsets[0] /= 2;
  offsets[1] /= 2;

  offsets[0] *= 8;
  offsets[1] *= 8;

  // interleave the data streams
  if (ctx->verbose)
    fprintf (stderr, "byte offsets %ld %ld\n", offsets[0], offsets[1]);

  int8_t * ins[2];
  ins[0] = (int8_t *) ctx->raw_data[0];
  ins[1] = (int8_t *) ctx->raw_data[1];

  int8_t * out = (int8_t *) ctx->raw_combined;

  int64_t out_offset = 0;
  int64_t in_offset = 0;
  int64_t out_nbytes = ctx->ndat * 2;

  while (out_offset < out_nbytes)
  {
    if (packets[0] > packets[1])
    {
      //fprintf (stderr, "memcpy1 (%ld, %ld + %ld)\n", out_offset, offsets[0], in_offset);
      memcpy (out + out_offset, ins[0] + offsets[0] + in_offset, 8192);
      out_offset += 8192;
    }

    //fprintf (stderr, "memcpy1 (%ld, %ld + %ld)\n", out_offset, offsets[1], in_offset);
    memcpy (out + out_offset, ins[1] + offsets[1] + in_offset, 8192);
    out_offset += 8192;

    if (packets[0] < packets[1])
    {
      //fprintf (stderr, "memcpy1 (%ld, %ld + %ld)\n", out_offset, offsets[0], in_offset);
      memcpy (out + out_offset, ins[0] + offsets[0] + in_offset, 8192);
      out_offset += 8192;
    }

    in_offset += 8192;
  }

  uint64_t idat;
  unsigned ipol, ipt, opt;

  for (idat=0; idat<ctx->ndat; idat += 4)
  {
    for (ipol=0; ipol<2; ipol++)
    {
      for (ipt=0; ipt<4; ipt++)
      {
        opt = (idat + ipt) / ctx->ndat_scrunch;
        ctx->y_points[ipol][opt] += ((float) out[ipt]) + 0.5;
      }
      out += 4;
    }
  }

  for (ipol=0; ipol<2; ipol++)
    for (idat=0; idat<ctx->ndat_plot; idat++)
      ctx->y_points[ipol][idat] /= sqrt(ctx->ndat_scrunch);
}

void plot_data (udptsplot_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "plot_packet()\n");

  unsigned ipol = 0;
  uint64_t idat = 0;
  float ymin = ctx->ymin;
  float ymax = ctx->ymax;

  if (ctx->verbose > 1)
    multilog (ctx->log, LOG_INFO, "plot_packet: ctx->xmin=%f ctx->xmax=%f\n", ctx->xmin, ctx->xmax);

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      for (idat=0; idat < ctx->ndat_plot; idat++)
      { 
        if (ctx->y_points[ipol][idat] > ymax) ymax = ctx->y_points[ipol][idat];
        if (ctx->y_points[ipol][idat] < ymin) ymin = ctx->y_points[ipol][idat];
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
    multilog(ctx->log, LOG_INFO, "caspsr_udptsplot: error opening 1/xs\n");
    exit(1);
  }

  cpgbbuf();
  cpgsci(1);
  ymin = -127.5;
  ymax = 127.5;

  cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 0);
  cpglab("Time [us]", "Voltage", "Time Series"); 

  cpgsls(1);

  int col, ls;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    col = ipol + 2;
    cpgsci(col);
    cpgpt(ctx->ndat_plot, ctx->x_points, ctx->y_points[ipol], -2);
  }
  cpgebuf();
  cpgclos();
}
