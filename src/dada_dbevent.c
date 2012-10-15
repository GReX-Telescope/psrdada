/***************************************************************************
 *  
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 * Attaches to in input data block as a viewer, and opens a socket to listen
 * for requests to write temporal events to the output data block. Can seek
 * back in time over cleared data blocks
 */

#include "dada_hdu.h"
#include "dada_def.h"
#include "node_array.h"
#include "multilog.h"
#include "diff_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

#define IPCBUF_EODACK 3   /* acknowledgement of end of data */
#define DADA_DBEVENT_DEFAULT_PORT 30000
#define DADA_DBEVENT_DEFAULT_INPUT_BUFFER 80
#define DADA_DBEVENT_DEFAULT_INPUT_DELAY 60


int quit = 0;

typedef struct {

  // input HDU
  dada_hdu_t * in_hdu;  

  // output HDU
  dada_hdu_t * out_hdu;

  // multilog 
  multilog_t * log;

  // verbosity
  int verbose;

} dada_dbevent_t;

#define DADA_DBEVENT_INIT { 0, 0, 0, 0 }

int dump_event(dada_dbevent_t * dbevent, double event_start_utc, double event_end_utc, float event_snr, float event_dm);
int dbevent_view_eod (ipcio_t* ipcio);
void usage();

void usage()
{
  fprintf (stdout,
     "dada_dbevent [options] inkey outkey\n"
     " inkey      input hexadecimal shared memory key\n"
     " outkey     input hexadecimal shared memory key\n"
     " -b percent madelay procesing of the input buffer up to this amount [default %d %]\n"
     " -t delay   maximum delay (s) to retain data for [default %d %]\n"
     " -h         print this help text\n"
     " -p port    port to listen for event commands [default %d]\n"
     " -v         be verbose\n", 
     DADA_DBEVENT_DEFAULT_INPUT_BUFFER, 
     DADA_DBEVENT_DEFAULT_INPUT_DELAY, 
     DADA_DBEVENT_DEFAULT_PORT);
}

void signal_handler(int signalValue) 
{
  fprintf(stderr, "dada_dbevent: SIGINT/TERM\n");
  quit = 1;
}

int main (int argc, char **argv)
{
  // core dbevent data struct
  dada_dbevent_t dbevent = DADA_DBEVENT_INIT;

  // DADA Logger
  multilog_t* log = 0;

  // flag set in verbose mode
  char verbose = 0;

  // port to listen for event requests
  int port = DADA_DBEVENT_DEFAULT_PORT;

  // input hexadecimal shared memory key
  key_t in_dada_key;

  // output hexadecimal shared memory key
  key_t out_dada_key;

  char daemon = 0;

  float input_data_block_threshold = DADA_DBEVENT_DEFAULT_INPUT_BUFFER;

  time_t input_maximum_delay = DADA_DBEVENT_DEFAULT_INPUT_DELAY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hp:t:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        if (sscanf (optarg, "%f", &input_data_block_threshold) != 1)
        {
          fprintf (stderr, "dada_dbevent: could not parse input buffer level from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      case 'h':
        usage();
        return EXIT_SUCCESS;

      case 'v':
        verbose = 1;
        break;

      case 'p':
        if (sscanf (optarg, "%d", &port) != 1) 
        {
          fprintf (stderr, "dada_dbevent: could not parse port from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      case 't':
        if (sscanf (optarg, "%d", &input_maximum_delay) != 1)
        {
          fprintf (stderr, "dada_dbevent: could not parse maximum input delay from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

      default:
        usage ();
        return EXIT_SUCCESS;
    }
  }

  if (argc - optind != 2)
  { 
    fprintf (stderr, "dada_dbevent: expected 2 command line arguments\n");
    usage();
    return EXIT_FAILURE;
  }

  if (sscanf (argv[optind], "%x", &in_dada_key) != 1) 
  {
    fprintf (stderr,"dada_dbevent: could not parse in_key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (sscanf (argv[optind+1], "%x", &out_dada_key) != 1) 
  {
    fprintf (stderr,"dada_dbevent: could not parse out_key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  // install some signal handlers
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  log = multilog_open ("dada_dbevent", 0);
  multilog_add (log, stderr);

  dbevent.verbose = verbose;
  dbevent.log = log;

  if (verbose)
    multilog(log, LOG_INFO, "connecting to data blocks\n");

  dbevent.in_hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbevent.in_hdu, in_dada_key);
  if (dada_hdu_connect (dbevent.in_hdu) < 0)
  {
    multilog(log, LOG_ERR, "could not connect to input HDU\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read(dbevent.in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not open input HDU as viewer\n");
    return EXIT_FAILURE;
  }

  dbevent.out_hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbevent.out_hdu, out_dada_key);
  if (dada_hdu_connect (dbevent.out_hdu) < 0)
  {
    multilog(log, LOG_ERR, "could not connect to output HDU\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write (dbevent.out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not open output HDU as writer\n");
    return EXIT_FAILURE;
  }

  // open listening socket
  int listen_fd = sock_create (&port);
  if (listen_fd < 0)
  { 
    multilog (log, LOG_ERR, "could not open socket: %s\n", strerror(errno));
    quit = 2;
  }

  fd_set fds;
  struct timeval timeout;
  int fds_read;
  
  size_t sock_buffer_size = 1024;
  char * sock_buffer = (char*) malloc ( sock_buffer_size );
  if ( !sock_buffer )
  {
    multilog (log, LOG_ERR, "could not allocated memory for sock_buffer\n");
    quit = 2;
  }

  // now get the header from the input data block
  multilog(log, LOG_INFO, "waiting for input header\n");
  if (dada_hdu_open (dbevent.in_hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not get input header\n");
    quit = 1;
  }
  else
  {
    fprintf (stderr, "==========\n");
    fprintf (stderr, "%s", dbevent.in_hdu->header);
    fprintf (stderr, "==========\n");
  }

  ipcbuf_t * db = (ipcbuf_t *) dbevent.in_hdu->data_block;

  // get the number of buffers in the input data block
  uint64_t in_nbufs = ipcbuf_get_nbufs (db);
  time_t * in_buf_times = (time_t *) malloc (sizeof(time_t) * in_nbufs);
  uint64_t ibuf = 0;
  for (ibuf=0; ibuf < in_nbufs; ibuf++)
    in_buf_times[ibuf] = 0;

  uint64_t curr_buf = ipcbuf_get_write_count (db);
  uint64_t prev_buf = ipcbuf_get_read_count (db);
  uint64_t buf = 0;

  uint64_t read_buf = 0;
  time_t read_time = 0;
  time_t read_time_diff = 0;
  unsigned time_ok = 0;

  while ( ! quit )
  {
    // setup file descriptor set for listening
    FD_ZERO(&fds);
    FD_SET(listen_fd, &fds);
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    fds_read = select(listen_fd+1, &fds, (fd_set *) 0, (fd_set *) 0, &timeout);

    // problem with select call
    if (fds_read < 0)
    {
      multilog (log, LOG_ERR, "select failed: %s\n", strerror(errno));
      quit = 2;
      break;
    }
    // select timed out, check input HDU for end of data
    else if (fds_read == 0)
    {
      // check when the input was last written to
      curr_buf = ipcbuf_get_write_count (db);
      multilog (log, LOG_INFO, "TIME: curr_buf=%"PRIu64"\n", curr_buf);
      while (prev_buf < curr_buf)
      {
        buf = prev_buf % in_nbufs;
        in_buf_times[buf] = time(0);
        multilog (log, LOG_INFO, "TIME: in_buf_times[%"PRIu64"] = %d\n", buf, in_buf_times[buf]);
        prev_buf ++;
      }

      time_ok = 0;
      while (!time_ok)
      {
        // now determine the current read buffer
        read_buf = ipcbuf_get_read_count (db);
        buf = read_buf % in_nbufs;
        read_time = in_buf_times[buf];
        read_time_diff = time(0) - read_time;

        if (verbose > 1)
          multilog (log, LOG_INFO, "TIME: read_buf=%"PRIu64" buf=%"PRIu64" read_time=%d time(0)=%d diff=%d\n",
                    read_buf, buf, read_time, time(0), read_time_diff);

        if ((curr_buf >= read_buf) && (read_time_diff > input_maximum_delay))
        {
          uint64_t bytes;
          uint64_t block_id;
          ipcio_open_block_read (dbevent.in_hdu->data_block, &bytes, &block_id);
          if (verbose > 1)
            multilog (log, LOG_INFO, "TIME: ipcio reading block %"PRIu64" of size %"PRIu64"\n", block_id, bytes);
          ipcio_close_block_read (dbevent.in_hdu->data_block, bytes);
        }
        else
          time_ok = 1;

        if (ipcbuf_eod (db))
        {
          time_ok = 1;
        }
      }
    }
    // we received a new connection on our listening FD, process comand
    else
    {
      int fd = 0;
      FILE *sockin = 0;
      char * rgot = 0;

      // parse these event parameters
      double event_start_utc = 0;
      double event_end_utc = 0;
      float event_snr = 0;
      float event_dm = 0;

      fd = sock_accept (listen_fd);
      if (fd < 0)
      {
        multilog(log, LOG_WARNING, "error accepting connection %s\n", strerror(errno));
        break;
      }

      sockin = fdopen(fd,"r");
      if (!sockin)
        multilog(log, LOG_WARNING, "error creating input stream %s\n", strerror(errno));
      setbuf (sockin, 0);

      while (!feof(sockin))
      {
        rgot = fgets (sock_buffer, sock_buffer_size, sockin);
        multilog (log, LOG_INFO, " <- %s\n", sock_buffer);
        if (sscanf(sock_buffer, "%lf %lf %f %f", &event_start_utc, &event_end_utc, &event_snr, &event_dm) != 4)
          multilog(log, LOG_WARNING, "failed to parse input '%s'\n", sock_buffer);

        // try to dump the event
        int did_dump = dump_event(&dbevent, event_start_utc, event_end_utc, event_snr, event_dm);
      }
      fclose(sockin);
    }

    // check how full the input datablock is
    float percent_full = ipcio_percent_full (dbevent.in_hdu->data_block) * 100;
    multilog (log, LOG_INFO, "input datablock %5.2f percent full\n", percent_full);

    uint64_t bytes = 0;
    uint64_t block_id = 0;
    while (!quit && percent_full > input_data_block_threshold)
    {
      ipcio_open_block_read (dbevent.in_hdu->data_block, &bytes, &block_id);
      multilog (log, LOG_INFO, "ipcio reading block %"PRIu64" of size %"PRIu64"\n", block_id, bytes);
      ipcio_close_block_read (dbevent.in_hdu->data_block, bytes);
      percent_full = ipcio_percent_full (dbevent.in_hdu->data_block) * 100;
      multilog (log, LOG_INFO, "input datablock reduced to %5.2f percent full\n", percent_full);
    }

    if (ipcbuf_eod (db))
    {
      multilog (log, LOG_INFO, "EOD now true\n");
      quit = 1;
    }

  }

  free(in_buf_times);

  if (dada_hdu_disconnect (dbevent.in_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: disconnect from input data block failed\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_unlock_write (dbevent.out_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: unlock write on output data block failed\n");
    return EXIT_FAILURE;
  }
  if (dada_hdu_disconnect (dbevent.out_hdu) < 0)
  {
    fprintf (stderr, "dada_dbevent: disconnect from output data block failed\n");
    return EXIT_FAILURE;
  }

  fprintf (stderr, "dada_dbevent: DONE :)\n");
  return EXIT_SUCCESS;
}

int dump_event(dada_dbevent_t * dbevent, double event_start_utc, double event_end_utc, float event_snr, float event_dm)
{
  multilog (dbevent->log, LOG_INFO, "dump_event: \n");
}

int dbevent_view_eod (ipcio_t* ipcio)
{
  ipcbuf_t * buf = &(ipcio->buf);
  fprintf (stderr, "ipcio_view_eod()\n");

  fprintf (stderr, "ipcio_view_eod: viewbuf=%"PRIu64"\n", buf->viewbuf);
  fprintf (stderr, "ipcio_view_eod: write_buf=%"PRIu64"\n", ipcbuf_get_write_count( buf ) );

  if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
    buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;
  fprintf (stderr, "ipcio_view_eod: viewbuf=%"PRIu64"\n", buf->viewbuf);

  ipcio->bytes = 0;
  ipcio->curbuf = 0;

  uint64_t current = ipcio_tell (ipcio);
  fprintf (stderr, "ipcio_view_eod: current=%"PRIu64"\n", current);
  uint64_t byte_resolution=2048;

  uint64_t too_far = current % byte_resolution;
  if (too_far)
  {
    int64_t absolute_bytes = ipcio_seek (ipcio,
           current + byte_resolution - too_far,
           SEEK_SET);
    if (absolute_bytes < 0)
      return -1;
  }

  uint64_t write_buf = ipcbuf_get_write_count( buf );

  // set view buffer to previous buffer to the current write buffer
  if (write_buf > 0)
    buf->viewbuf = write_buf - 1;

  fprintf (stderr, "ipcio_view_eod: viewbuf=%"PRIu64"\n", buf->viewbuf);

  ipcio->bytes = 0;
  ipcio->curbuf = 0;
  return 0;
}

