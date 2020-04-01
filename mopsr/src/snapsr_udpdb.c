/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "snapsr_udp.h"
#include "snapsr_def.h"
#include "ascii_header.h"
#include "daemon.h"
#include "dada_affinity.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>

//#define _DEBUG
#define NO_1PPS

int quit_threads = 0;
void stats_thread(void * arg);
void signal_handler(int signalValue);

typedef struct {

  multilog_t * log;               // logging interface
  char * interface;               // ethX interface to use
  int port;                       // UDP capture port
  char * header_file;             // file containing DADA header
  unsigned verbose;               // verbosity level
  snapsr_sock_t * sock;             // custom socket struct
  stats_t * packets;              // packet stats
  stats_t * bytes;                // bytes stats
  uint64_t n_sleeps;              // busy sleep counter
  unsigned capture_started;       // flag for start of data
  uint64_t buffer_start_byte;     // buffer byte counters
  uint64_t buffer_end_byte;
  unsigned header_written;
  int recv_core;
  snapsr_udp_hdr_t   hdr;
  int control_port;

} snapsr_udpdb_t;

void usage()
{
  fprintf (stdout,
     "snapsr_udpdb [options] header\n"
#ifdef HAVE_AFFINITY
     " -c core      bind process to CPU core\n"
#endif
     " -C port      send control commands to port [default : 5000]\n"
     " -k key       hexadecimal shared memory key  [default: %x]\n"
     " -i ip        only listen on specified ip address [default: any]\n"
     " -p port      port for incoming UDP packets [default %d]\n"
     " -v           be verbose\n"
     " -s           1 transfer, then exit\n"
     "header        DADA header file contain obs metadata\n",
     DADA_DEFAULT_BLOCK_KEY, SNAPSR_DEFAULT_UDPDB_PORT);
}

int snapsr_init (snapsr_udpdb_t * ctx)
{
  // allocate memory for socket
  ctx->sock = snapsr_init_sock();

  // allocate memory for packet/byte stats
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  ctx->n_sleeps = 0;
  ctx->capture_started = 0;
  ctx->buffer_start_byte = 0;
  ctx->buffer_end_byte = 0;
  ctx->header_written = 0;
 
#ifdef HAVE_AFFINITY
  // set the CPU that this thread shall run on
  if (ctx->recv_core > 0)
    if (dada_bind_thread_to_core(ctx->recv_core) < 0)
      multilog(ctx->log, LOG_WARNING, "receive_obs: failed to bind to core %d\n", ctx->recv_core);
#endif

  return 0;
}


/*! Function that opens the data transfer target */
int snapsr_udpdb_open (dada_client_t* client)
{
  assert (client != 0);

  // contextual data for dada_udpdb
  snapsr_udpdb_t * ctx = (snapsr_udpdb_t *) client->context;
  assert(ctx != 0);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "snapsr_udpdb_open()\n");

  // DADA ascii header
  char * header = client->header;
  assert (header != 0);

  // read the ASCII DADA header from the file
  if (fileread (ctx->header_file, client->header, client->header_size) < 0)
    multilog (client->log, LOG_ERR, "Could not read header from %s\n", ctx->header_file);

  char utc_time[64];
  time_t start;
#ifdef NO_1PPS
  // write the current UTC time as the UTC_START. Note this will not be precisely 
  // correct until we add a scheme to synchronise the SNAP with the 1PPS and pass that
  // information in via a header

  time_t prev = time(0);
  start = prev + 1;
  while (start > prev)
    prev = time(0);

#else

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "listening on control port %d\n", ctx->control_port);

  // wait for a UTC_START from the SNAP control script
  int listen_fd = sock_create (&(ctx->control_port));
  if (listen_fd < 0)
  {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create control socket\n");
    return -1;
  }

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "listening on selecting on listen_fd=%d\n", listen_fd);

  struct timeval timeout;
  timeout.tv_sec = 60;
  timeout.tv_usec = 0;

  fd_set readset;
  FD_ZERO(&readset);
  FD_SET(listen_fd, &readset);
  int readsocks = select(listen_fd+1, &readset, NULL, NULL, &timeout);

  if (readsocks < 0)
  {
    multilog (ctx->log, LOG_ERR, "select failure\n");
    return -1;
  }

  if (readsocks == 0)
  {
    multilog (ctx->log, LOG_ERR, "No control connection within 60s\n");
    return -1;
  }

  int fd = sock_accept (listen_fd);
  if (fd < 0)
  {
    multilog(ctx->log, LOG_WARNING, "Error accepting connection %s\n", strerror(errno));
    return -1;
  }
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "accepted control connection\n");

  FILE *sockin = fdopen(fd,"r");
  if (!sockin)
    multilog(ctx->log, LOG_WARNING, "control_thread: error creating input "
                                     "stream %s\n", strerror(errno));
  FILE* sockout = fdopen(fd,"w");
  if (!sockout)
    multilog(ctx->log, LOG_WARNING, "control_thread: error creating output "
                                        "stream %s\n", strerror(errno));

  setbuf (sockin, 0);
  setbuf (sockout, 0);

  char * rgot = fgets (utc_time, 64, sockin);
  utc_time[strlen(utc_time)-2] = '\0';
  start = str2utctime (utc_time);
  if (start == (time_t)-1)
  {
    multilog(ctx->log, LOG_WARNING, "control_thread: could not parse "
             "UTC_START time from %s\n", utc_time);
    fprintf(sockout, "fail\r\n");
  }
  else
  {
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "control_thread: parsed UTC_START as %d\n", start);
    fprintf(sockout, "ok\r\n");
  }

  close (fd);
  close (listen_fd);
#endif
  strftime (utc_time, 64, DADA_TIMESTR, gmtime(&start));
  if (ascii_header_set (header, "UTC_START", "%s", utc_time) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write UTC_START to header\n");

  multilog (ctx->log, LOG_INFO, "UTC_START=%s\n", utc_time);

  // open socket
  ctx->sock->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->port, ctx->verbose);
  if (ctx->sock->fd < 0) 
  {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return -1;
  }

  // set the socket size to 64 MB
  int sock_buf_size = 64*1024*1024;
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: setting buffer size to %d\n", sock_buf_size);
  dada_udp_sock_set_buffer_size (ctx->log, ctx->sock->fd, ctx->verbose, sock_buf_size);

  // set the socket to non-blocking
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: setting non_block\n");
  sock_nonblock(ctx->sock->fd);

  // clear any packets buffered by the kernel
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "start_function: clearing packets at socket\n");
  size_t cleared = dada_sock_clear_buffered_packets(ctx->sock->fd, SNAPSR_UDP_PAYLOAD_BYTES);

  ctx->header_written = 0;

  return 0;
}

/*! Transfer header to data block */
int64_t snapsr_udpdb_recv (dada_client_t* client, void* data, uint64_t data_size)
{

  snapsr_udpdb_t* ctx = (snapsr_udpdb_t *) client->context;

  ctx->header_written = 1;
  multilog (client->log, LOG_INFO, "recv: read header\n");
  return data_size;
}

/*! Transfer UDP data to data block, receives 1 block worth of UDP packets */
int64_t snapsr_udpdb_recv_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  snapsr_udpdb_t * ctx = (snapsr_udpdb_t *) client->context;

  unsigned keep_receiving = 1;
  uint64_t bytes_received = 0;
  uint64_t seq_no = 0;
  uint16_t ant_id = 0;
  uint64_t seq_byte = 0;
  unsigned ignore_packet = 0;
  int errsv = 0;
  uint64_t byte_offset = 0;

  const unsigned block_sample_stride = SNAPSR_NSUBBAND * SNAPSR_NCHAN_PER_SUBBAND * SNAPSR_NANT_PER_PACKET * 2;
  const unsigned packet_sample_stride = SNAPSR_BYTES_PER_SAMPLE;
  const unsigned bytes_per_eight_samples = SNAPSR_NSUBBAND * SNAPSR_NCHAN_PER_SUBBAND * SNAPSR_NANT_PER_PACKET * SNAPSR_NFRAME_PER_PACKET * 2;
  const unsigned bytes_per_sample = SNAPSR_NSUBBAND * SNAPSR_NCHAN_PER_SUBBAND * SNAPSR_NANT_PER_PACKET * 2;

  uint64_t chunk_offset;

  // update start/end bytes for this block
  if (ctx->capture_started)
  {
    ctx->buffer_start_byte = ctx->buffer_end_byte + SNAPSR_UDP_DATA_BYTES;
    ctx->buffer_end_byte = (ctx->buffer_start_byte + data_size) - SNAPSR_UDP_DATA_BYTES;
    if (ctx->verbose)
      multilog (client->log, LOG_INFO, "recv_block: CONT  [%"PRIu64" - %"PRIu64"]\n", ctx->buffer_start_byte, ctx->buffer_end_byte);
  }

  while (keep_receiving && !quit_threads)
  {
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "recv_block: get packet\n");
#endif

    ctx->n_sleeps = 0;

    while (!ctx->sock->have_packet && !quit_threads)
    {
      // receive 1 packet into the socket buffer
      ctx->sock->got = recvfrom (ctx->sock->fd, ctx->sock->buf, SNAPSR_UDP_PAYLOAD_BYTES, 0, NULL, NULL);

      if (ctx->sock->got == SNAPSR_UDP_PAYLOAD_BYTES)
      {
        ctx->sock->have_packet = 1;
        ignore_packet = 0;
      }
      else if (ctx->sock->got == -1)
      {
        errsv = errno;
        if (errsv == EAGAIN)
        {
          ctx->n_sleeps++;
        }
        else
        {
          multilog (client->log, LOG_ERR, "recv_block: recvfrom failed %s\n", strerror(errsv));
          return -1;
        }
      }
      else // we received a packet of the WRONG size, ignore it
      {
        snapsr_decode (ctx->sock->buf, &(ctx->hdr));
        multilog (client->log, LOG_ERR, "recv_block: received %d bytes, expected %d from packet with subband_id=%u seq=%lu\n", ctx->sock->got, SNAPSR_UDP_PAYLOAD_BYTES, ctx->hdr.subband_id, ctx->hdr.seq_no);
        if (ctx->verbose)
        {
          unsigned j=0;
          for (j=0; j<ctx->sock->got; j+=8)
          {
            multilog (client->log, LOG_INFO, "%d : %02x %02x %02x %02x %02x %02x %02x %02x\n", 
                       j/8,
                       0xff & ctx->sock->buf[j],
                       0xff & ctx->sock->buf[j+1],
                       0xff & ctx->sock->buf[j+2],
                       0xff & ctx->sock->buf[j+3],
                       0xff & ctx->sock->buf[j+4],
                       0xff & ctx->sock->buf[j+5],
                       0xff & ctx->sock->buf[j+6],
                       0xff & ctx->sock->buf[j+7]);
          }
        }


        ignore_packet = 1;
      }

      // if packets stop flowing
      if (ctx->capture_started && ctx->n_sleeps > 100000)
        return 0;
    }

#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "recv_block: have packet\n");
#endif

    // if we received a packet
    if (ctx->sock->have_packet)
    {
      // reset this for next iteration
      ctx->sock->have_packet = 0;

      // decode sequence number
      snapsr_decode (ctx->sock->buf, &(ctx->hdr));

#ifdef _DEBUG
      multilog (client->log, LOG_INFO, "recv_block: seq_no=%"PRIu64"\n", ctx->hdr.seq_no);
#endif

      // if first packet
#ifdef NO_1PPS
      if (!ctx->capture_started && ctx->hdr.subband_id == 0)
      {
        ctx->buffer_start_byte = ctx->hdr.seq_no * SNAPSR_UDP_DATA_BYTES;
#else
      if (!ctx->capture_started && ctx->hdr.seq_no < 1000)
      {
        ctx->buffer_start_byte = 0;
#endif
        ctx->buffer_end_byte = (ctx->buffer_start_byte + data_size) - SNAPSR_UDP_DATA_BYTES;
        ctx->capture_started = 1;
        if (ctx->verbose)
          multilog (client->log, LOG_INFO, "recv_block: START [%"PRIu64" - %"PRIu64"] data_size=%lu\n", ctx->buffer_start_byte, ctx->buffer_end_byte, data_size);
      }

      if (ctx->capture_started)
      {
        seq_byte = ctx->hdr.seq_no * SNAPSR_UDP_DATA_BYTES;

        // if packet arrived too late, ignore
        if (seq_byte < ctx->buffer_start_byte)
        {
          ctx->packets->dropped++;
          ctx->bytes->dropped += SNAPSR_UDP_DATA_BYTES;
        }
        else
        {
          // packet belongs in this buffer
          if (seq_byte <= ctx->buffer_end_byte)
          {
            byte_offset = seq_byte - ctx->buffer_start_byte;
            chunk_offset = (byte_offset / bytes_per_eight_samples) * bytes_per_eight_samples;
            byte_offset = chunk_offset + ((byte_offset % bytes_per_eight_samples) / 8);
      
            char * from = ctx->sock->buf + SNAPSR_UDP_HEADER_BYTES;
            char * to   = ((char *) data) + byte_offset;

            unsigned i;
            for (i=0; i<SNAPSR_NFRAME_PER_PACKET; i++)
            {
              //fprintf(stderr, "%d subband_id=%u seq_byte=%lu end_byte=%lu offset=%ld\n", i, ctx->hdr.subband_id, seq_byte, ctx->buffer_start_byte, to - (char *) data);
              memcpy (to, from, packet_sample_stride);
              from += packet_sample_stride;
              to   += block_sample_stride;
            }
            ctx->packets->received++;

            if (seq_byte == ctx->buffer_end_byte)
              keep_receiving = 0;
          }
          // packet belongs in subsequent buffer
          else
          {
            ctx->packets->dropped++;
            keep_receiving = 0;
          }
          ctx->bytes->received += SNAPSR_UDP_DATA_BYTES;
        }
      }
    }
  }
  if (quit_threads)
    return 0;
  else
    return data_size;
}



/*! Function that closes socket */
int snapsr_udpdb_close (dada_client_t* client, uint64_t bytes_written)
{

  assert (client != 0);

  // status and error logging facility
  multilog_t* log = client->log;
  assert (log != 0);

  // contextual data for dada_udpdb
  snapsr_udpdb_t * ctx = (snapsr_udpdb_t *) client->context;
  assert(ctx != 0);

  close(ctx->sock->fd);


  return 0;
}

int main (int argc, char **argv)
{

  // udpdb contextual struct
  snapsr_udpdb_t udpdb;

  // DADA Header plus Data Unit 
  dada_hdu_t* hdu = 0;

  // DADA Primary Read Client main loop
  dada_client_t* client = 0;

  // DADA Logger
  multilog_t* log = 0;

  // Flag set in verbose mode
  char verbose = 0;

  char * header_file = 0;

  // Quit flag
  char single_transfer = 0;

  // dada key for SHM 
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  // ethernet interface to receive packets on
  char * interface = "any";

  // port to receive data on
  int port = SNAPSR_DEFAULT_UDPDB_PORT;

  int arg = 0;

  int core = -1;
  int cport = 5000;

  while ((arg=getopt(argc,argv,"C:c:k:i:p:sv")) != -1)
    switch (arg) {
  
    case 'C':
      if (optarg)
      {
        cport = atoi(optarg);
        break;
      }
      else
      {
        fprintf (stderr, "ERROR: -C flag requires argument\n");
        return EXIT_FAILURE;
      }


    case 'c':
      if (optarg)
      {
        core = atoi(optarg);
        break;
      }
      else
      {
        fprintf (stderr, "ERROR: -c flag requires argument\n");
        return EXIT_FAILURE;
      }

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n", optarg);
        return EXIT_FAILURE;
      }
      break;

    case 'i':
      if (optarg)
      {
        interface = strdup(optarg);
        break;
      }
      else
      {
        fprintf (stderr, "ERROR: -i flag requires argument\n");
        return EXIT_FAILURE;
      }

    case 'p':
      if (optarg)
      {
        port = atoi(optarg);
        break;
      }
      else
      {
        fprintf (stderr, "ERROR: -p flag requires argument\n");
        return EXIT_FAILURE;
      }

    case 's':
      single_transfer = 1;
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return 0;
      
    }

  // check the header file was supplied
  if ((argc - optind) != 1) 
  {
    fprintf (stderr, "ERROR: header must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  }

  udpdb.header_file = strdup(argv[optind]);

  // check the header can be read
  FILE* fptr = fopen (udpdb.header_file, "r");
  if (!fptr) 
  {
    fprintf (stderr, "ERROR: could not open '%s' for reading: %s\n", header_file, strerror(errno));
    return(EXIT_FAILURE);
  }
  fclose(fptr);

  // handle SIGINT gracefully
  signal(SIGINT, signal_handler);

  udpdb.interface = strdup(interface);
  udpdb.port = port;
  udpdb.control_port = cport;
  udpdb.verbose = verbose;
  udpdb.recv_core = core;

  // allocate memory for socket etc
  if (verbose)
    multilog (log, LOG_INFO, "main: initialising resources\n");
  if (snapsr_init (&udpdb) < 0)
  {
    multilog (log, LOG_ERR, "could not initialize resources\n");
    return EXIT_FAILURE;
  }

  log = multilog_open ("dada_udpdb", 0);
  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (verbose)
    multilog (log, LOG_INFO, "main: connecting to HDU %x\n", dada_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf(stderr, "ERROR: could not connect to HDU %x\n", dada_key);
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: locking write on HDU %x\n", dada_key);
  if (dada_hdu_lock_write (hdu) < 0)
  {
    fprintf(stderr, "ERROR: could not lock write on HDU %x\n", dada_key);
    return EXIT_FAILURE;
  }

  // check that the DADA buffer block size is a multiple of the SNAPSR_UDP_DATA_BYTES
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu->data_block);
  if (block_size % SNAPSR_UDP_DATA_BYTES != 0)
  {
    fprintf(stderr, "ERROR: DADA buffer size must be a multiple of SNAPSR_UDP_DATA_BYTES size\n");
    dada_hdu_unlock_write (hdu);
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }
  if (verbose)
    multilog (log, LOG_INFO, "main: DADA block_size=%"PRIu64", SNAPSR_UDP_DATA_BYTES size=%d\n", block_size, SNAPSR_UDP_DATA_BYTES);

  client = dada_client_create ();

  udpdb.verbose = verbose;
  udpdb.log = log;

  client->context = &udpdb;
  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function     = snapsr_udpdb_open;
  client->io_function       = snapsr_udpdb_recv;
  client->io_block_function = snapsr_udpdb_recv_block;
  client->close_function    = snapsr_udpdb_close;

  client->direction         = dada_client_writer;

  pthread_t stats_thread_id;
  if (verbose)
    multilog(log, LOG_INFO, "main: starting stats_thread()\n");
  int rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpdb);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: dada_client_write\n");
  if (dada_client_write (client) < 0)
    multilog (log, LOG_ERR, "Error during transfer\n");

  quit_threads = 1;
  void* result = 0;
  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  pthread_join (stats_thread_id, &result);

  if (dada_hdu_unlock_write (hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not unlock read on hdu\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_disconnect (hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not disconnect from HDU\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) 
{
  snapsr_udpdb_t * ctx = (snapsr_udpdb_t *) arg;

  uint64_t bytes_received_total = 0;
  uint64_t bytes_received_this_sec = 0;
  uint64_t bytes_dropped_total = 0;
  uint64_t bytes_dropped_this_sec = 0;
  double   mb_received_ps = 0;
  double   mb_dropped_ps = 0;

  struct timespec ts;

  sleep(2);

  fprintf(stderr,"Bytes\t\t\t\tPackets\n");
  fprintf(stderr,"Received\t Dropped\tReceived\tDropped\t\n");

  while (!quit_threads)
  {
    bytes_received_this_sec = ctx->bytes->received - bytes_received_total;
    bytes_dropped_this_sec  = ctx->bytes->dropped - bytes_dropped_total;

    if (bytes_received_total > 0 || bytes_dropped_total > 0)
    {
      mb_received_ps = (double) bytes_received_this_sec / (1024*1024);
      mb_dropped_ps = (double) bytes_dropped_this_sec / (1024*1024);

      fprintf(stderr,"%7.1f MB/s\t%7.1f MB/s\t%"PRIu64" pkts\t\t%"PRIu64" pkts\n", mb_received_ps, mb_dropped_ps, ctx->packets->received, ctx->packets->dropped);
    }

    bytes_received_total = ctx->bytes->received;
    bytes_dropped_total = ctx->bytes->dropped;


    sleep(1);
  }
}

/*! Simple signal handler for SIGINT */
void signal_handler(int signalValue)
{
  fprintf (stderr, "Received Signal\n");
  if (quit_threads)
  {
    fprintf (stderr, "Exiting\n");
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;
}


