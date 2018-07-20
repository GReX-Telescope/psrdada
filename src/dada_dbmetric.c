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

  // limits for data and header blocks across all readers
  uint64_t max_read_data, max_full_data, min_cleared_data;
  uint64_t max_read_header, max_full_header, min_cleared_header;

  uint64_t bufs_read_data, bufs_written_data;
  uint64_t bufs_read_header, bufs_written_header;

  int n_readers = -1;
  int iread = 0;

  // pointers to header and data blocks
  ipcbuf_t *hb = hdu->header_block;
  ipcbuf_t *db = (ipcbuf_t *) hdu->data_block;

  n_readers = db->sync->n_readers;
          
  uint64_t nhbufs = ipcbuf_get_nbufs (hb);
  uint64_t ndbufs = ipcbuf_get_nbufs (db);

  uint64_t read_data[n_readers];
  uint64_t full_data[n_readers];
  uint64_t cleared_data[n_readers];

  uint64_t read_header[n_readers];
  uint64_t full_header[n_readers];
  uint64_t cleared_header[n_readers];


  bufs_written_data = ipcbuf_get_write_count (db);

  max_read_data = 0;
  max_full_data = 0;
  min_cleared_data = ndbufs;

  for (iread=0; iread < n_readers; iread++)
  {
    // acquire the metrics for each reader
    read_data[iread]    = ipcbuf_get_read_count_iread (db, iread);
    full_data[iread]    = ipcbuf_get_nfull_iread (db, iread);
    cleared_data[iread] = ipcbuf_get_nclear_iread (db, iread);

    // determine the limits
    if (read_data[iread] > max_read_data)
      max_read_data = read_data[iread];
    if (full_data[iread] > max_full_data)
      max_full_data = full_data[iread];
    if (cleared_data[iread] < min_cleared_data)
      min_cleared_data = cleared_data[iread];
  }
    
  bufs_written_header = ipcbuf_get_write_count (hb);
  max_read_header = 0;
  max_full_header = 0;
  min_cleared_header = ndbufs;

  for (iread=0; iread < n_readers; iread++)
  {
    // acquire the metrics for each reader
    read_header[iread]    = ipcbuf_get_read_count_iread (hb, iread);
    full_header[iread]    = ipcbuf_get_nfull_iread (hb, iread);
    cleared_header[iread] = ipcbuf_get_nclear_iread (hb, iread);

    // determine the limits
    if (read_header[iread] > max_read_header)
      max_read_header = read_header[iread];
    if (full_header[iread] > max_full_header)
      max_full_header = full_header[iread];
    if (cleared_header[iread] < min_cleared_header)
      min_cleared_header = cleared_header[iread];
  }

  if (verbose)
  {
    fprintf (stderr,"IREAD,TOTAL,FULL,CLEAR,W_BUF,R_BUF,TOTAL,FULL,CLEAR,W_BUF,"
                    "R_BUF\n");
    for (iread=0; iread < n_readers; iread++)
    {
      fprintf(stderr, "%d,%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",",
              iread, ndbufs, full_data[iread], cleared_data[iread], bufs_written_data, read_data[iread]);
      fprintf(stderr,"%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64"\n",
                        ndbufs, full_header[iread], cleared_header[iread], bufs_written_header, read_header[iread]);
    }
  }
  else
  {
    fprintf(stderr,"%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64"\n",
                   ndbufs, max_full_data, min_cleared_data, bufs_written_data, max_read_data,
                   ndbufs, max_full_header, min_cleared_header, bufs_written_header, max_read_header);
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
