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
     "dada_dbmetric [options]\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -v         be verbose\n", DADA_DEFAULT_BLOCK_KEY);
}


int main (int argc, char **argv)
{

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* dada key for SHM */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"vk:")) != -1)
    switch (arg) {
                                                                                
    case 'v':
      verbose=1;
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "dada_dbmetric: could not parse key from %s\n",optarg);
        return -1;
      }
      break;

    default:
      usage ();
      return 0;
                                                                                
  }
  
  log = multilog_open ("dada_dbmetric", 0);
  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  uint64_t read, full, cleared; 
  uint64_t full_bufs;
  uint64_t clear_bufs;
  uint64_t bufs_read;
  uint64_t bufs_written;
  int n_readers = -1;
  int iread = 0;

  // pointers to header and data blocks
  ipcbuf_t *hb = hdu->header_block;
  ipcbuf_t *db = (ipcbuf_t *) hdu->data_block;

  n_readers = db->sync->n_readers;
          
  uint64_t nhbufs = ipcbuf_get_nbufs (hb);
  uint64_t ndbufs = ipcbuf_get_nbufs (db);

  if (verbose) 
    fprintf (stderr,"TOTAL,FULL,CLEAR,W_BUF,R_BUF,TOTAL,FULL,CLEAR,W_BUF,"
                    "R_BUF\n");

  bufs_written = ipcbuf_get_write_count (db);
  bufs_read = 0;
  full_bufs = 0;
  clear_bufs = ndbufs;

  for (iread=0; iread < n_readers; iread++)
  {
    read = ipcbuf_get_read_count_iread (db, iread);
    if (read > bufs_read)
      bufs_read = read; 

    full = ipcbuf_get_nfull_iread (db, iread);
    if (full > full_bufs)
      full_bufs = full;

    cleared = ipcbuf_get_nclear_iread (db, iread);
    if (cleared < clear_bufs)
      clear_bufs = cleared;
  }
    
  fprintf(stderr,"%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",",
                 ndbufs, full_bufs, clear_bufs, bufs_written, bufs_read);


  bufs_written = ipcbuf_get_write_count (hb);
  bufs_read = 0;
  full_bufs = 0;
  clear_bufs = nhbufs;

  for (iread=0; iread < n_readers; iread++)
  {
    read = ipcbuf_get_read_count_iread (hb, iread);
    if (read > bufs_read)
      bufs_read = read;

    full = ipcbuf_get_nfull_iread (hb, iread);
    if (full > full_bufs)
      full_bufs = full;

    cleared = ipcbuf_get_nclear_iread (hb, iread);
    if (cleared < clear_bufs)
      clear_bufs = cleared;

  }

  fprintf(stderr,"%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64"\n",
                 nhbufs, full_bufs, clear_bufs, bufs_written, bufs_read);

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
