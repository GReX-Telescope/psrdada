#include "dada_hdu.h"
#include "dada_def.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

void usage()
{
fprintf (stdout,
     "dada_dbmonitor [options]\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -v         be verbose\n"
     " -d         run as daemon\n", DADA_DEFAULT_BLOCK_KEY);
}


int main (int argc, char **argv)
{

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  uint64_t full_bufs = 0;
  uint64_t clear_bufs = 0;
  uint64_t bufs_read = 0;
  uint64_t bufs_written = 0;
  int64_t available_bufs = 0;
  int iread = 0;

  while ((arg=getopt(argc,argv,"dvk:")) != -1)
  {
    switch (arg)
    {
      case 'd':
        daemon=1;
        break;

      case 'v':
        verbose=1;
        break;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr,"dada_dbmonitor: could not parse key from %s\n",optarg);
          return -1;
        }
        break;
        
      default:
        usage ();
        return 0;

    }
  }

  log = multilog_open ("dada_dbmonitor", daemon);

  if (daemon)
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, DADA_DEFAULT_DBMONITOR_LOG);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);
 
  if (verbose)
    fprintf (stderr, "main: connecting to HDU\n");
  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  // pointers to header and data blocks
  ipcbuf_t *hb = hdu->header_block;
  ipcbuf_t *db = (ipcbuf_t *) hdu->data_block;
          
  uint64_t hdr_bufsz = ipcbuf_get_bufsz (hb);
  uint64_t hdr_nbufs = ipcbuf_get_nbufs (hb);
  uint64_t hdr_bytes = hdr_nbufs * hdr_bufsz;

  fprintf(stderr,"HEADER BLOCK:\n");
  fprintf(stderr,"Number of buffers: %"PRIu64"\n", hdr_nbufs);
  fprintf(stderr,"Buffer size: %"PRIu64"\n", hdr_bufsz);
  fprintf(stderr,"Total buffer memory: %"PRIu64" KB\n",(hdr_bytes/(1024)));

  uint64_t data_bufsz = ipcbuf_get_bufsz (db);
  uint64_t data_nbufs = ipcbuf_get_nbufs (db);
  uint64_t data_bytes = data_nbufs * data_bufsz;
  int n_readers = ipcbuf_get_nreaders (db);

  fprintf(stderr,"DATA BLOCK:\n");
  fprintf(stderr,"Number of readers: %d\n", n_readers);
  fprintf(stderr,"Number of buffers: %"PRIu64"\n", data_nbufs);
  fprintf(stderr,"Buffer size: %"PRIu64"\n", data_bufsz);
  fprintf(stderr,"Total buffer memory: %"PRIu64" MB\n",(data_bytes/(1024*1024)));

  fprintf(stderr,"\n");
  fprintf(stderr,"HEADER               ");
  if (n_readers == 1)
    fprintf(stderr,"DATA\n");
  else
  {
    for (iread=0; iread<n_readers; iread++)
      fprintf(stderr,"DATA%d                      ", iread);
    fprintf(stderr,"\n");
  }

  fprintf(stderr,"FRE FUL CLR  W  R");

  for (iread=0; iread<n_readers; iread++)
    fprintf(stderr,"    FRE FUL CLR     W     R");
  fprintf(stderr, "\n");

  
  while (!quit) 
  {
    bufs_written = ipcbuf_get_write_count (hb);
    bufs_read = ipcbuf_get_read_count (hb);
    full_bufs = ipcbuf_get_nfull (hb);
    clear_bufs = ipcbuf_get_nclear (hb);
    available_bufs = (hdr_nbufs - full_bufs);
    
    fprintf(stderr,"%3"PRIi64" %3"PRIu64" %3"PRIu64" %2"PRIu64" %2"PRIu64,
                    available_bufs, full_bufs, clear_bufs, bufs_written, bufs_read);

    for (iread=0; iread<n_readers; iread++)
    {
      bufs_written = ipcbuf_get_write_count (db);
      bufs_read = ipcbuf_get_read_count_iread (db, iread);
      full_bufs = ipcbuf_get_nfull_iread (db, iread);
      clear_bufs = ipcbuf_get_nclear_iread (db, iread);
      available_bufs = (data_nbufs - full_bufs);
    
      fprintf(stderr,"    %3"PRIi64" %3"PRIu64" %3"PRIu64" %5"PRIu64" %5"PRIu64,
                    available_bufs, full_bufs, clear_bufs, bufs_written, bufs_read);
    }
    fprintf (stderr, "\n");

    sleep(1);
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
