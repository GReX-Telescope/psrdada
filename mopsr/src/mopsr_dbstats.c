/*
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

typedef struct {

  multilog_t * log;

  unsigned int verbose;

  unsigned int nant;

  unsigned int nchan;

  unsigned int ndim;

  unsigned int nsamp;

  int8_t ** ant_raw;

  size_t ant_raw_size;

  unsigned int control_port;

} mopsr_dbstats_t;

int quit_threads = 0;

void usage ();
void control_thread (void *);
int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution);

void usage()
{
  fprintf (stdout,
     "mopsr_dbstats [options]\n"
     " -c port    control port for binary data\n"
     " -k key     connect to key data block\n"
     " -t num     integrate num frames into monitoring output\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  /* DADA Header plus Data Unit */
  dada_hdu_t * hdu = 0;

  mopsr_dbstats_t dbstats;

  mopsr_dbstats_t * ctx;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Quit flag */
  char quit = 0;

  /* dada key for SHM */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  ctx = &dbstats;

  ctx->verbose = 0;
  ctx->control_port = 54321;
  ctx->nant = 2;
  ctx->nchan = 128;
  ctx->ndim = 2;
  ctx->nsamp = 1024;

  while ((arg=getopt(argc,argv,"c:t:vk:")) != -1)
  {
    switch (arg)
    {
      case 'c':
        ctx->control_port = atoi (optarg);
        break;

      case 't':
        ctx->nsamp = atoi(optarg);
        break;

      case 'v':
        ctx->verbose++;
        break;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) 
        {
          fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
          return -1;
        }
        break;
      
      default:
        usage ();
        return 0;
    } 
  }

  fprintf (stderr, "main: initialization\n");

  // setup multilogger
  ctx->log = multilog_open ("mopsr_dbstats", 0);
  multilog_add (ctx->log, stderr);

  // init HDU 
  hdu = dada_hdu_create (ctx->log);
  dada_hdu_set_key(hdu, dada_key);

  // connect to HDU
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: dada_hdu_connect()\n");
  if (dada_hdu_connect (hdu) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not connect to HDU on key %x\n", dada_key);
    return EXIT_FAILURE;
  }

  // open HDU as a passive viewer
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: dada_hdu_open_view()\n");
  if (dada_hdu_open_view(hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    multilog (ctx->log, LOG_ERR, "could not open HDU for viewing\n");
    return EXIT_FAILURE;
  }

  // open the HDU (check if this necessary)
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: dada_hdu_open()\n");
  if (dada_hdu_open (hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    multilog (ctx->log, LOG_ERR, "could not open HDU\n");
    return EXIT_FAILURE;
  }

  // extract key paramters from the HDU's headera
  unsigned int byte_resolution;
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: ascii_header_get (RESOULTION)\n");
  if (ascii_header_get (hdu->header, "RESOLUTION", "%u", &byte_resolution) < 0)
  {
    multilog (ctx->log, LOG_WARNING, "HEADER with no RESOLUTION\n");
    byte_resolution = 0;
  }
  multilog (ctx->log, LOG_INFO, "RESOULTION %u\n", byte_resolution);

  if (ascii_header_get (hdu->header, "NCHAN", "%u", &(ctx->nchan)) < 0)
  {
    multilog (ctx->log, LOG_ERR, "HEADER with no NCHAN\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }
  multilog (ctx->log, LOG_INFO, "NCHAN %u\n", ctx->nchan);

  if (ascii_header_get (hdu->header, "NANT", "%u", &(ctx->nant)) < 0)
  {
    multilog (ctx->log, LOG_ERR, "HEADER with no NANT\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }
  multilog (ctx->log, LOG_INFO, "NANT %u\n", ctx->nant);
  
  if (ascii_header_get (hdu->header, "NDIM", "%u", &(ctx->ndim)) < 0)
  {
    multilog (ctx->log, LOG_ERR, "HEADER with no NDIM\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }
  multilog (ctx->log, LOG_INFO, "NDIM %u\n", ctx->nant);

  if (!byte_resolution)
    byte_resolution = ctx->nchan * ctx->nant * ctx->ndim;

  if (byte_resolution != ctx->nchan * ctx->nant * ctx->ndim) 
  {
    multilog (ctx->log, LOG_WARNING, "RESOLUTION not correct\n");
    byte_resolution = ctx->nchan * ctx->nant * ctx->ndim;
  }

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: byte_resolution=%u\n", byte_resolution);
  
  // allocate memory for data buffer read
  size_t buffer_size = byte_resolution * ctx->nsamp;
  int8_t * buffer = (int8_t *) malloc (sizeof (int8_t) * buffer_size);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "main: buffer_szie=%d, nsamps=%d\n", buffer_size, ctx->nsamp);

  unsigned int in_offset, out_offset, isamp, ichan, iant;

  // allocated array for each antenna
  ctx->ant_raw = (int8_t **) malloc (ctx->nant * sizeof (int8_t *));
  for (iant=0; iant<ctx->nant; iant++)
  {
    ctx->ant_raw_size = sizeof (int8_t) * ctx->nchan * ctx->ndim * ctx->nsamp;
    ctx->ant_raw[iant] = (int8_t *) malloc (ctx->ant_raw_size);
  }

  uint64_t bytes_to_read = byte_resolution * ctx->nsamp;
  int64_t  bytes_read = 0;
  pthread_t control_thread_id;

  // start the control thread
  if (ctx->control_port)
  {
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "main: starting control_thread()\n");
    int rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) ctx);
    if (rval != 0) {
      multilog (ctx->log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
      return -1;
    }
  }

  // TODO improve this
  while ( ! quit_threads )
  {
    // move to the last block/byte based on resolution
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "main: ipcio_view_eod()\n");
    if (ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
    {
      multilog (ctx->log, LOG_ERR, "main: ipcio_view_eod failed\n");
      quit_threads = 1;
      break;
    }

    // read the required amount from the data block
    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "main: ipcio_read %"PRIu64" bytes\n", bytes_to_read);
    bytes_read = ipcio_read (hdu->data_block, (char *) buffer, bytes_to_read);
    if (bytes_read < 0)
    {
      multilog (ctx->log, LOG_ERR, "main: ipcio_read failed\n");
      quit_threads = 1;
      break;
      return EXIT_FAILURE;
    }

    if (bytes_read < bytes_to_read)
    {
      multilog (ctx->log, LOG_WARNING, "main: ipcio_read returned %"PRIi64" bytes, "
                "requested %"PRIu64"\n", bytes_read, bytes_to_read);
    }

    in_offset = 0;
    out_offset = 0;
    for (isamp=0; isamp<ctx->nsamp; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        for (iant=0; iant<ctx->nant; iant++)
        {
          out_offset = isamp * ctx->nchan * ctx->ndim + (ichan * ctx->ndim);
          ctx->ant_raw[iant][out_offset + 0] = buffer[in_offset + 0];
          ctx->ant_raw[iant][out_offset + 1] = buffer[in_offset + 1];
          in_offset += 2;
        }
      }
    }

    if (ctx->verbose > 1)
      multilog (ctx->log, LOG_INFO, "main: sleep (1)\n");
    sleep (1);
  }

  if (ctx->control_port)
  {
    void * result;
    if (ctx->verbose)
      multilog(ctx->log, LOG_INFO, "main: joining control_thread\n");
    pthread_join (control_thread_id, &result);
  }

  if (buffer)
    free (buffer);

  if (ctx->ant_raw)
  {
    for (iant=0; iant<ctx->nant; iant++)
    {
      if (ctx->ant_raw[iant])
        free (ctx->ant_raw[iant]);
      ctx->ant_raw[iant] = 0;
    }
    free (ctx->ant_raw);
  }
  ctx->ant_raw = 0;

  if (dada_hdu_disconnect (hdu) < 0)
  {
    fprintf (stderr, "ERROR: could not disconnect from HDU\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution)
{
  ipcbuf_t* buf = &(ipcio->buf);

#ifdef _DEBUG
  fprintf (stderr, "ipcio_view_eod: write_buf=%"PRIu64"\n",
     ipcbuf_get_write_count( buf ) );
#endif

  buf->viewbuf ++;

  if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
    buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

  ipcio->bytes = 0;
  ipcio->curbuf = 0;

  uint64_t current = ipcio_tell (ipcio);
  uint64_t too_far = current % byte_resolution;
  if (too_far)
  {
    int64_t absolute_bytes = ipcio_seek (ipcio,
           current + byte_resolution - too_far,
           SEEK_SET);
    if (absolute_bytes < 0)
      return -1;
  }

  return 0;
}


/*
 *
 */
void control_thread (void * arg) 
{
  mopsr_dbstats_t * ctx = (mopsr_dbstats_t *) arg;

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: starting\n");

  // port on which to listen for control commands
  int port = ctx->control_port;

  // buffer for incoming command strings
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  assert (buffer != 0);

  const char* whitespace = " \r\t\n";
  char * command = 0;
  char * args = 0;
  time_t utc_start = 0;

  FILE *sockin = 0;
  FILE *sockout = 0;
  int listen_fd = 0;
  int fd = 0;
  char *rgot = 0;
  int readsocks = 0;
  fd_set socks;
  struct timeval timeout;

  // create a socket on which to listen
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: creating socket on port %d\n", port);

  listen_fd = sock_create (&port);
  if (listen_fd < 0)  {
    multilog(ctx->log, LOG_ERR, "Failed to create socket for control commands: %s\n", strerror(errno));
    free (buffer);
    return;
  }

  while (!quit_threads) {

    // reset the FD set for selecting  
    FD_ZERO(&socks);
    FD_SET(listen_fd, &socks);
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    readsocks = select(listen_fd+1, &socks, (fd_set *) 0, (fd_set *) 0, &timeout);

    // error on select
    if (readsocks < 0) 
    {
      perror("select");
      exit(EXIT_FAILURE);
    }

    // no connections, just ignore
    else if (readsocks == 0) 
    {
    } 

    // accept the connection  
    else 
    {
   
      if (ctx->verbose) 
        multilog(ctx->log, LOG_INFO, "control_thread: accepting conection\n");

      fd =  sock_accept (listen_fd);
      if (fd < 0)  {
        multilog(ctx->log, LOG_WARNING, "control_thread: Error accepting "
                                        "connection %s\n", strerror(errno));
        break;
      }

      sockin = fdopen(fd,"r");
      if (!sockin)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating input "
                                        "stream %s\n", strerror(errno));


      sockout = fdopen(fd,"w");
      if (!sockout)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating output "
                                        "stream %s\n", strerror(errno));

      setbuf (sockin, 0);
      setbuf (sockout, 0);

      while (!feof(sockin)) 
      {
        rgot = fgets (buffer, bufsize, sockin);

        if (rgot && !feof(sockin))
        {

          buffer[strlen(buffer)-2] = '\0';

          args = buffer;

          // parse the command and arguements
          command = strsep (&args, whitespace);

          if (ctx->verbose)
          { 
            multilog(ctx->log, LOG_INFO, "control_thread: command=%s\n", command);
            if (args != NULL)
              multilog(ctx->log, LOG_INFO, "control_thread: args=%s\n", args);
          }

          // REQUEST STATISTICS
          if (strcmp(command, "help") == 0) 
          {
            fprintf (sockout, "Available commands:\r\n");
            fprintf (sockout, " help        print these help commands\r\n");
            fprintf (sockout, " nant        print number of antenna\r\n");
            fprintf (sockout, " nchan       print number of channels\r\n");
            fprintf (sockout, " nsamp       print number of samples\r\n");
            fprintf (sockout, " dump <ant>  dump raw binary data over socket\r\n");
            fprintf (sockout, " quit        request this program to exit\r\n");
            fprintf (sockout, "ok\r\n");
          }

          else if (strcmp(command, "nant") == 0)
          {
            fprintf (sockout, "%u\r\n", ctx->nant);
            fprintf (sockout, "ok\r\n");
          }

          else if (strcmp(command, "nchan") == 0)
          {
            fprintf (sockout, "%u\r\n", ctx->nchan);
            fprintf (sockout, "ok\r\n");
          }

          else if (strcmp(command, "nsamp") == 0)
          {
            fprintf (sockout, "%u\r\n", ctx->nsamp);
            fprintf (sockout, "ok\r\n");
          }

          else if (strcmp(command, "dump") == 0)
          {
            int ant;
            sscanf(args, "%d", &ant);
            //multilog(ctx->log, LOG_INFO, "control_thread: request for ant=%d, size=%d\n", ant, ctx->ant_raw_size);
            if ((ant >= 0) && (ant < ctx->nant))
              write (fd, ctx->ant_raw[ant], ctx->ant_raw_size);
          }

          else if (strcmp(command, "quit") == 0) 
          {
            multilog(ctx->log, LOG_INFO, "control_thread: QUIT command received, exiting\n");
            quit_threads = 1;
            fprintf(sockout, "ok\r\n");
          }

          // UNRECOGNISED COMMAND
          else 
          {
            multilog(ctx->log, LOG_WARNING, "control_thread: unrecognised command: %s\n", buffer);
            fprintf(sockout, "fail\r\n");
          }
        }
      }
      close(fd);
    }
  }
  close(listen_fd);

  free (buffer);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: exiting\n");

}
