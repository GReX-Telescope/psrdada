#include "dada_hdu.h"
#include "dada_def.h"

#include "node_array.h"
#include "multilog.h"

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

#define DBOVERFLOW_SIGINT 1
#define DBOVERFLOW_SIGTERM 2
#define DBOVERFLOW_DB_FULL 3
#define DBOVERFLOW_DB_DESTROYED 4

int quit = 0;

void usage()
{
  fprintf (stdout,
     "dada_dboverflow [options]\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -v         be verbose\n"
     " -d         run as daemon\n", DADA_DEFAULT_BLOCK_KEY);
}

void signal_handler(int signalValue) 
{
  if (quit) 
  {
    fprintf(stderr, "dada_dboverlfow: repeated SIGINT/TERM, exiting\n");
    exit(EXIT_FAILURE);
  }
  if (signalValue == SIGINT)
    quit = DBOVERFLOW_SIGINT;
  else
    quit = DBOVERFLOW_SIGTERM;
}

int main (int argc, char **argv)
{

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  /* TODO the amount to conduct a busy sleep inbetween clearing each sub
   * block */
  while ((arg=getopt(argc,argv,"vk:")) != -1)
    switch (arg) {

    case 'v':
      verbose=1;
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_dboverflow: could not parse key from %s\n",optarg);
        return -1;
      }
      break;
      
    default:
      usage ();
      return 0;

    }

  // gracefully handle SIGINT
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  log = multilog_open ("dada_dboverflow", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);
  
  if (verbose)
    multilog_fprintf (stderr, LOG_INFO, "connecting to hdu\n");

  if (dada_hdu_connect (hdu) < 0)
  {
    return EXIT_FAILURE;
  }

  if (dada_hdu_open_view(hdu) < 0)
  {
    fprintf (stderr, "dada_dboverflow: dada_hdu_open_view() failed\n");
    return EXIT_FAILURE;
  }

  // raw pointer to the datablock
  ipcbuf_t *db     = (ipcbuf_t *) hdu->data_block;

  uint64_t full_bufs = 0;
  uint64_t clear_bufs = 0;
  uint64_t bufs_read = 0;
  uint64_t bufs_read_last = 0;
  uint64_t bufs_written = 0;
  int64_t available_bufs = 0;

  uint64_t bufsz = ipcbuf_get_bufsz (db);
  uint64_t nbufs = ipcbuf_get_nbufs (db);
  uint64_t nbytes = nbufs * bufsz;
  float    nmbytes = (float) nbytes / (1024*1024);

  if (verbose)
  {
    multilog_fprintf(stderr, LOG_INFO, "nbufs= %"PRIu64", bufsz=%"PRIu64"\n", nbufs, bufsz);
    multilog_fprintf(stderr, LOG_INFO, "size=%"PRIu64" bytes, %4.2f MB\n", nbytes, nmbytes);
  }

  unsigned current_read_xfer = 0;
  unsigned wait_count = 5;

  while (!quit) 
  {

    // use a semctl GETVAL operation to test for DB existence
    if ( ipcbuf_get_nfull (db) == (uint64_t) - 1)
    {
      fprintf(stderr, "dada_dboverflow: datablock destroyed, exiting\n");
      quit = DBOVERFLOW_DB_DESTROYED;
    }
    else 
    {
      bufs_read_last = bufs_read;
      bufs_written = ipcbuf_get_write_count (db);
      bufs_read = ipcbuf_get_read_count (db);
      full_bufs = ipcbuf_get_nfull (db);
      clear_bufs = ipcbuf_get_nclear (db);
      available_bufs = (nbufs - full_bufs);

      if (verbose)
        multilog_fprintf (stderr, LOG_INFO, "free=%"PRIu64", full=%"PRIu64
                          " written=%"PRIu64", read=%"PRIu64"\n", available_bufs, 
                          full_bufs, bufs_written, bufs_read);

      // since we can often be stuck with 1 free buffer 
      if ((available_bufs <= 1) && (bufs_read_last == bufs_read))
      {
        fprintf (stderr, "dada_dboverflow: no free blocks, %d seconds remaining\n", wait_count);
        if (wait_count <= 0)
        {
          quit = DBOVERFLOW_DB_FULL;
        }
        wait_count--;
      }
      else
      {
        // reset the wait count if we recovered
        wait_count = 5;
      }

      if (!quit)
        sleep(1);
    }
  }

  fprintf (stderr, "dada_dboverflow: quit=%d\n", quit);

  if (dada_hdu_disconnect (hdu) < 0)
  {
    fprintf (stderr, "dada_dboverflow: dada_hdu_disconnect() failed\n");
    return quit;
  }

  return quit;
}

