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

#include "dada_hdu.h"
#include "dada_def.h"
#include "multilog.h"
#include "futils.h"
#include "ipcutil.h"

// constants taken from ipcbuf.c
#define IPCBUF_READ   1   /* semaphore locks reader (+clear) status */
#define IPCBUF_VIEWER  1  /* connected */

int quit = 0;

void usage()
{
  fprintf (stdout,
     "dada_dbrecover [options] input output\n"
     " input      input datablock key\n"
     " output     output datablock key\n"
     " -H file    write file to output datablock header\n"
     " -v         be verbose\n"
     " -d         run as daemon\n");
}

void signal_handler(int signalValue) 
{
  if (quit) 
  {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit = 1;
}

int main (int argc, char **argv)
{

  char * header_file = 0;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hH:v")) != -1)
    switch (arg) {

    case 'H':
      header_file = strdup(optarg);
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return 0;

    }

  int num_args = argc-optind;
  if (num_args != 2)
  {
    fprintf (stderr, "ERROR: 2 command line arguments required, encountered %d\n", num_args);
    usage();
    return (EXIT_FAILURE);
  }
  
  key_t in_key;
  key_t out_key;

  if (sscanf (argv[optind], "%x", &in_key) != 1)
  {
    fprintf (stderr, "dada_dbrecover: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (sscanf (argv[optind+1], "%x", &out_key) != 1)
  {
    fprintf (stderr, "dada_dbrecover: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  // gracefully handle SIGINT
  signal(SIGINT, signal_handler);

  log = multilog_open ("dada_dbscrubber", 0);

  multilog_add (log, stderr);

  // setup connection to input HDU
  hdu = dada_hdu_create (log);
  dada_hdu_set_key(hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "Could not connect to input HDU with key %x\n", in_key);
    return (EXIT_FAILURE);
  }

  // setup connection to output HDU
  dada_hdu_t * out_hdu = dada_hdu_create (log);
  dada_hdu_set_key (out_hdu, out_key);
  if (dada_hdu_connect (out_hdu) < 0)
  {
    fprintf (stderr, "Could not connect to output HDU with key %x\n", out_key);
    return (EXIT_FAILURE);
  }

  if (dada_hdu_lock_write (out_hdu) < 0)
  {
    fprintf (stderr, "cannot lock write DADA HDU (key=%x)\n", out_key);
    return -1;
  }

  uint64_t header_size = ipcbuf_get_bufsz (out_hdu->header_block);
  char * header = ipcbuf_get_next_write (out_hdu->header_block);
  if (!header) 
  {
    multilog (log, LOG_ERR, "could not get next header block\n");
    return -1;
  }

  if (header_file)
  {
    if (fileread (header_file, header, header_size) < 0)
    {
      multilog (log, LOG_ERR, "Could not read header from %s\n", header_file);
    }
  }

  if (ipcbuf_mark_filled (out_hdu->header_block, header_size) < 0)
  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }

  ipcbuf_t * id;

  // check the header block's READ semaphore to see if we will be able to connect
  id = hdu->header_block;
  int hb_read = semctl (id->semid_connect, IPCBUF_READ, GETVAL);
  if (verbose)
    multilog_fprintf (stderr, LOG_INFO, "main: HB IPCBUF_READ=%d\n", hb_read);
  if (hb_read == 0)
  {
    if (verbose)
      multilog_fprintf (stderr, LOG_INFO, "main: increment HB IPCBUF_READ\n");
    if (ipc_semop (id->semid_connect, IPCBUF_READ, 1, SEM_UNDO) < 0)
    {
      fprintf (stderr, "main: error increment HB IPCBUF_READ\n");
      return -1;
    }
  }

  // check the data block's READ semaphore
  id = (ipcbuf_t *) hdu->data_block;
  int db_read = semctl (id->semid_connect, IPCBUF_READ, GETVAL);
  if (verbose)
    multilog_fprintf (stderr, LOG_INFO, "main: DB IPCBUF_READ=%d\n", db_read);
  if (db_read == 0)
  {
    if (verbose)
      multilog_fprintf (stderr, LOG_INFO, "main: increment DB IPCBUF_READ\n");
    if (ipc_semop (id->semid_connect, IPCBUF_READ, 1, SEM_UNDO) < 0)
    {
      fprintf (stderr, "main: error increment DB IPCBUF_READ\n");
      return -1;
    }
  }

  // the HB and DB should now be lockable
  if (verbose)
    multilog_fprintf (stderr, LOG_INFO, "locking read on HDU\n");
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "Could not lock read on HDU\n");
    return EXIT_FAILURE;
  }

  ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;
  ipcio_t * ipc = hdu->data_block;

  if (ipc ->rdwrt != 'R') 
  {
    multilog_fprintf(stderr, LOG_ERR, "not a designated reader\n");
    quit = 1;
  }

  /*
  if (verbose > 1)
    multilog_fprintf(stderr, LOG_INFO, "HB: ipcbuf_mark_cleared()\n");
  ipcbuf_mark_cleared (hb);
*/

  uint64_t bytes_cleared = 0;
  uint64_t block_id = 0;

  ipc->curbufsz = ipcbuf_get_bufsz (db);
  ipc->bytes += ipc->curbufsz;
  bytes_cleared += ipc->curbufsz;

  if (verbose > 1)
    multilog_fprintf(stderr, LOG_INFO, "marking %"PRIu64" bytes as read\n", ipc->curbufsz);

  block_id = ipcbuf_get_read_index (db);

  // whilst we are not at the EOD 
  while (!quit && !ipcbuf_eod(db)) 
  {
    if (!ipc->curbuf)
    {
      block_id = ipcbuf_get_read_index (db);

      // get the next readable block, with bytes = curbufsz
      if (verbose > 1)
        multilog_fprintf(stderr, LOG_INFO, "DB: ipcbuf_get_next_read() read_index=%"PRIu64"\n", block_id);
      ipc->curbuf = ipcbuf_get_next_read ((ipcbuf_t*)ipc, &(ipc->curbufsz));

      if (!ipc->curbuf)
      {
        multilog_fprintf (stderr, LOG_ERR, "DB: ipcbuf_next_read failed\n");
        return -1;
      }

      // write a full block of input data to the output 
      ipcio_write (out_hdu->data_block, ipc->curbuf, ipc->curbufsz);

      // this is where the data in the buffer is "read"
      if (verbose > 1)
        multilog_fprintf(stderr, LOG_INFO, "marking %"PRIu64" bytes as read\n", ipc->curbufsz);
      ipc->bytes += ipc->curbufsz;
      bytes_cleared += ipc->curbufsz;

      // mark this buffer as cleared
      if (verbose > 1)
        multilog_fprintf(stderr, LOG_INFO, "DB: ipcbuf_mark_cleared()\n");
      if (ipcbuf_mark_cleared ((ipcbuf_t*)ipc) < 0)
      {
        multilog_fprintf (stderr, LOG_INFO, "DB: ipcbuf_mark_cleared failed\n");
        return -1;
      }

      ipc->curbuf = 0;
      ipc->bytes = 0;
    }
  }

  ipcio_close (out_hdu->data_block);

  if (ipcbuf_eod(db))
    multilog_fprintf(stderr, LOG_INFO, "reached EOD\n");

  multilog_fprintf(stderr, LOG_INFO, "cleared %"PRIu64" bytes\n", bytes_cleared);

  if (dada_hdu_unlock_read (hdu) < 0)
    multilog_fprintf (stderr, LOG_ERR, "Could not unlock read on input HDU\n");

  if (dada_hdu_unlock_write (out_hdu) < 0)
    multilog_fprintf (stderr, LOG_ERR, "Could not unlock read on output HDU\n");

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (out_hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

