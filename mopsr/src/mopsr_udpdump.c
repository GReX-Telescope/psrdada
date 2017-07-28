/*
 * mopsr_udpdump
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

#include "dada_pwc.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "mopsr_util.h"
#include "multilog.h"
#include "futils.h"
#include "sock.h"

static char quit_threads = 0;
void signal_handler (int signalValue);

typedef struct
{
  dada_pwc_t * pwc;

  dada_pwc_command_t command;
}
udpdump_pwcm_t;

typedef struct {

  multilog_t * log;

  mopsr_sock_t * sock;

  char * interface;

  int port;

  int core;

  // identifying string for source PFB
  char * pfb_id;

  // directory for monitoring files
  char * mdir;

  unsigned int verbose;

  unsigned npackets;

  char * data;

  size_t data_size;

  int control_port;

  int log_port;

  ssize_t pkt_size;

  udpdump_pwcm_t pwcm;

} udpdump_t;


int udpdump_init (udpdump_t * ctx);
int udpdump_prepare (udpdump_t * ctx);
int udpdump_destroy (udpdump_t * ctx);
void control_thread (void *);

//void integrate_packet (udpdump_t * ctx, char * buffer, unsigned int size);

void usage()
{
  fprintf (stdout,
     "mopsr_udpdump [options]\n"
     " -h             print help text\n"
     " -b <core>      bind computation to CPU core\n"
     " -c <port>      PWCC control port for nexus happiness\n"
     " -l <logport>   PWCC log port for nexus happiness\n"
     " -m <id>        set PFB_ID on monitoring files\n"
     " -M <dir>       write monitoring files to dir\n"
     " -i <ipaddr>    listen for UDP only on this interface [default all]\n"
     " -p <port>      port on which to receive UDP packets[default: %d]\n"
     " -n <packets>   integrate packets into dump[default: 64]\n"
     " -v             verbose messages\n",
     MOPSR_DEFAULT_UDPDB_PORT);
}

int udpdump_prepare (udpdump_t * ctx)
{
  if (ctx->verbose > 1)
    multilog(ctx->log, LOG_INFO, "mopsr_udpdb_prepare()\n");

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "prepare: clearing packets at socket [size=%ld]\n", ctx->pkt_size);
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);

  udpdump_reset(ctx);
  return 0;
}

int udpdump_reset (udpdump_t * ctx)
{
  memset (ctx->data, 0, ctx->data_size);
  return 0;
}

int udpdump_destroy (udpdump_t * ctx)
{
  if (ctx->data)
    free (ctx->data);
  ctx->data = 0;

  if (ctx->sock)
  {
    close(ctx->sock->fd);
    mopsr_free_sock (ctx->sock);
  }
  ctx->sock = 0;
  return 0;
}

/*
 * Close the udp socket and file
 */

int udpdump_init (udpdump_t * ctx)
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

  // try to get a packet to determine the packet size of this function
  usleep (1000);

  size_t data_size;
  ctx->pkt_size = recvfrom (ctx->sock->fd, ctx->sock->buf, UDP_PAYLOAD, 0, NULL, NULL);
  if (ctx->pkt_size <= 0)
  {
    ctx->pkt_size = UDP_DATA + UDP_HEADER;
    data_size = UDP_DATA;
  }
  else
  {
    data_size = ctx->pkt_size - UDP_HEADER;
  }

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "init: pkt_size=%ld udp_data_size=%ld\n", ctx->pkt_size, data_size);

  ctx->data_size = (data_size * ctx->npackets) + UDP_HEADER * sizeof(char);
  ctx->data = (char *) malloc (ctx->data_size);

  return 0;
}



int main (int argc, char **argv)
{
  /* actual struct with info */
  udpdump_t udpdump;

  udpdump_t * ctx = &udpdump;

  ctx->interface = malloc(sizeof(char) * 17);
  sprintf(ctx->interface, "%s", "all");

  ctx->pfb_id = malloc (sizeof(char) * 16);
  sprintf (ctx->pfb_id, "%s", "XX");

  ctx->npackets = 64;
  ctx->port = MOPSR_DEFAULT_UDPDB_PORT;
  ctx->verbose = 0;
  ctx->core = -1;
  ctx->mdir = 0;

  // control thread
  pthread_t control_thread_id;
  ctx->control_port = -1;
  ctx->log_port = -1;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:c:i:l:m:M:n:p:vh")) != -1) {
    switch (arg) {

    case 'b':
      ctx->core = atoi(optarg);
      break;

    case 'c':
      ctx->control_port = atoi(optarg);
      break;

    case 'i':
      if (optarg)
      {
        sprintf(ctx->interface, "%s", optarg);
        break;
      }
      else
      {
        fprintf (stderr, "mopsr_udpdump: -i requires argument\n");
        usage();
        return EXIT_FAILURE;
      }

    case 'l':
      ctx->log_port = atoi(optarg);
      break;

    case 'm':
      if (optarg) 
      {
        sprintf(ctx->pfb_id, "%s", optarg);
        break;
      } else {
        fprintf (stderr, "mopsr_udpdump: -m requires argument\n");
        usage();
        return EXIT_FAILURE;
      }

    case 'M':
      if (optarg) {
        if (ctx->mdir)
          free (ctx->mdir);
        ctx->mdir = (char *) malloc (sizeof (char) * strlen(optarg)+1);
        strcpy (ctx->mdir , optarg);
        break;
      } else {
        fprintf (stderr, "mopsr_udpdump: no dir specified\n");
        usage();
        return EXIT_FAILURE;
      }

    case 'n':
      ctx->npackets = atoi (optarg);
      break;

    case 'p':
      ctx->port = atoi (optarg);
      break;

    case 'v':
      ctx->verbose++;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  assert ((MOPSR_UDP_DATASIZE_BYTES + MOPSR_UDP_COUNTER_BYTES) == MOPSR_UDP_PAYLOAD_BYTES);

  multilog_t* log = multilog_open ("mopsr_udpdump", 0);
  multilog_add (log, stderr);
  if (ctx->log_port > 0)
    multilog_serve (log, ctx->log_port);

  ctx->log = log;

  // allocate require resources, open socket
  if (udpdump_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not create UDP socket\n");
    exit(1);
  }

  // clear packets ready for capture
  udpdump_prepare (ctx);

  // start the control thread
  if (ctx->control_port > 0)
  {
    ctx->pwcm.pwc = dada_pwc_create();
    ctx->pwcm.pwc->port = ctx->control_port;
    ctx->pwcm.pwc->log = log;
    if (dada_pwc_serve (ctx->pwcm.pwc) < 0)
    {
      fprintf (stderr, "mopsr_udpdb: could not start server\n");
      return EXIT_FAILURE;
    }

    if (ctx->verbose)
      multilog(log, LOG_INFO, "starting control_thread()\n");
    int rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) ctx);
    if (rval != 0) {
      multilog(log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
      return -1;
    }
  }

  int fd;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  size_t nwrote, got, ncleared;
  time_t now, now_plus;
  uint64_t prev_seq = 0;
  unsigned int ipacket = 0;
  time_t sleep_secs = 5;
  size_t pkt_data_size = ctx->pkt_size - UDP_HEADER;

  char local_time[32];
  char mon_file[512];

  mopsr_hdr_t hdr;

  while ( !quit_threads )
  {
    ncleared = dada_sock_clear_buffered_packets(ctx->sock->fd, ctx->pkt_size);
    usleep (1000);

    ipacket = 0;
    while (!quit_threads && ipacket < ctx->npackets)
    {
      got = 0;
      while  (!quit_threads && got != ctx->pkt_size)
      {
        got = recvfrom (ctx->sock->fd, ctx->sock->buf, ctx->pkt_size, 0, NULL, NULL);
        if (got == ctx->pkt_size)
        {
          memcpy (ctx->data, ctx->sock->buf, UDP_HEADER);
          memcpy (ctx->data + UDP_HEADER + (ipacket * pkt_data_size), ctx->sock->buf + UDP_HEADER, pkt_data_size);
          mopsr_decode (ctx->data, &hdr);
          if ((prev_seq != 0) && (hdr.seq_no != prev_seq + 1) && ctx->verbose)
          {
            multilog(ctx->log, LOG_INFO, "main: hdr.seq=%"PRIu64" prev_seq=%"PRIu64" [%"PRIu64"]\n", hdr.seq_no, prev_seq, hdr.seq_no - prev_seq);
          }
          prev_seq = hdr.seq_no;
          ipacket++;
        }
        else if ((got == -1) && (errno == EAGAIN))
        {
          // packet not at socket due to clear + nonblock
          usleep (10000);
        }
        else
        {
          // more serious!
          multilog(ctx->log, LOG_INFO, "main: init & got[%d] != UDP_PAYLOAD[%d]\n",
                   got, UDP_PAYLOAD);
          sleep (1);
        }
      }
    }

    mopsr_decode (ctx->data, &hdr);
    if (ctx->verbose)
      mopsr_print_header (&hdr);
    // special case for pass thru mode
    if (hdr.nchan == 1)
      hdr.nsamp *= ctx->npackets;
    else
      hdr.nframe *= ctx->npackets;
    mopsr_encode (ctx->data, &hdr);
    if (ctx->verbose)
      mopsr_print_header (&hdr);

    now = time(0);
    strftime (local_time, 32, DADA_TIMESTR, localtime(&now));
    sprintf (mon_file, "%s/%s.%s.dump", ctx->mdir, local_time, ctx->pfb_id);
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "main: creating %s\n", mon_file);
    fd = open(mon_file, flags, perms);
    if (fd < 0)
    {
      multilog (ctx->log, LOG_ERR, "main: failed to open '%s' for "
                "writing: %s\n", mon_file, strerror(errno));
    }
    else
    {
      nwrote = write (fd, ctx->data, ctx->data_size);
      close (fd);
    }
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "wrote %d bytes\n", ctx->data_size);

    // wait the appointed amount of time
    now_plus = time(0);
    while (!quit_threads && now + sleep_secs >= now_plus)
    {
      usleep (500000);
      now_plus = time(0);
    }
  }

  if (ctx->control_port > 0)
  {
    void * result;
    if (ctx->verbose)
      multilog(log, LOG_INFO, "joining control_thread\n");
    pthread_join (control_thread_id, &result);
  }


  udpdump_destroy (ctx);

  return EXIT_SUCCESS;
}

void control_thread (void * arg)
{
  udpdump_t * ctx = (udpdump_t*) arg;
  udpdump_pwcm_t * pwcm = &(ctx->pwcm);
  dada_pwc_t * pwc = pwcm->pwc;

  dada_pwc_state_t state = dada_pwc_idle;

  time_t utc = 0;
  size_t utc_size = 20;
  char utc_buffer[utc_size];

  char idle_loop, recording_loop;

  while (!quit_threads && !dada_pwc_quit (pwc))
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "control_thread: entering idle loop\n");

    // IDLE LOOP
    idle_loop = 1;
    while (idle_loop && !dada_pwc_quit (pwcm->pwc))
    {
      if (quit_threads)
        break;

      if (dada_pwc_command_check (pwc))
      {
        pwcm->command = dada_pwc_command_get (pwc);

        if (dada_pwc_quit (pwc))
        {
          multilog (ctx->log, LOG_INFO, "dada_pwc_quit is true!\n");
          break;
        }
     
        if (pwcm->command.code == dada_pwc_reset)
        {
          if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_reset\n");
          dada_pwc_set_state (pwc, dada_pwc_idle, 0);
        }
        else if (pwcm->command.code == dada_pwc_header)
        {
          if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "pwcm->command == dada_header\n");

          char utc_buffer[20];
          if (ascii_header_get (pwcm->command.header, "UTC_START", "%s", utc_buffer) != 1)
          {
            multilog (ctx->log, LOG_INFO, "no UTC_START found in header, expecting set_utc_start command\n");
          }
          else
          {
            multilog (ctx->log, LOG_INFO, "UTC_START = %s\n", utc_buffer);
            utc = str2utctime(utc_buffer);
          }
          dada_pwc_set_state (pwc, dada_pwc_prepared, 0);
        }
        else if (pwcm->command.code == dada_pwc_clock)
        {
          if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_clock\n");
          dada_pwc_set_state (pwc, dada_pwc_clocking, utc);
          recording_loop = 1;
          idle_loop = 0;
        }
        else if (pwcm->command.code == dada_pwc_start)
        {
          if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_start\n");
          dada_pwc_set_state (pwc, dada_pwc_recording, utc);
          recording_loop = 1;
          idle_loop = 0;
        }
        else if (pwcm->command.code == dada_pwc_stop)
        {
          if (ctx->verbose)
            multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_stop\n");
          dada_pwc_set_state (pwc, dada_pwc_idle, 0);
          idle_loop = 0;
        }
      }
      else
      {
        usleep (10000);
      }
    }


    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "control_thread: entering recording loop\n");

    // RECORDING LOOP
    while (recording_loop && !dada_pwc_quit (pwcm->pwc))
    {
      if (quit_threads)
        break;

      if (dada_pwc_command_check (pwc))
      {
        pwcm->command = dada_pwc_command_get (pwc);

        if (pwcm->command.code == dada_pwc_set_utc_start) 
        {
          strftime (utc_buffer, utc_size, DADA_TIMESTR, (struct tm*) gmtime(&(pwcm->command.utc)));
          multilog (ctx->log, LOG_INFO, "UTC_START = %s\n", utc_buffer);

          pthread_mutex_lock (&(pwc->mutex));
          pthread_cond_signal (&(pwc->cond));
          pthread_mutex_unlock (&(pwc->mutex));
        }
        else
        {
          if (pwcm->command.code == dada_pwc_record_stop)
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_record_stop\n");
            dada_pwc_set_state (pwc, dada_pwc_clocking, 0);
          }
          else if (pwcm->command.code == dada_pwc_record_start)
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_record_start\n");
            dada_pwc_set_state (pwc, dada_pwc_recording, 0);
          }
          else if (pwcm->command.code == dada_pwc_stop)
          {
            if (ctx->verbose)
              multilog (ctx->log, LOG_INFO, "pwcm->command == dada_pwc_stop\n");
            multilog (ctx->log, LOG_INFO, "stopping... entering idle state\n");
            recording_loop = 0;
          }
          else
          {
            multilog (ctx->log, LOG_INFO, "pwcm->command == unrecognized!! %d\n", pwcm->command.code);
          }
        }
      }
      else
      {
        usleep(10000);
      }
    }

    if (!quit_threads)
    {
      if (ctx->verbose)
        multilog (ctx->log, LOG_INFO, "dada_pwc_set_state (dada_pwc_idle)\n");
      dada_pwc_set_state (pwc, dada_pwc_idle, 0);
    }
  }

  quit_threads = 1;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: exiting\n");
}

/*! Simple signal handler for SIGINT */
void signal_handler(int signalValue)
{
  if (quit_threads)
  {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;
}

