#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcbuf.h"

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
     "dada_dbmeminfo [options]\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -v         be verbose\n", DADA_DEFAULT_BLOCK_KEY);
}

const char * state_to_str(int state);

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

  while ((arg=getopt(argc,argv,"k:v")) != -1)
    switch (arg) {
     
    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_dbmeminfo: could not parse key from %s\n",
                         optarg);
        return -1;
      }
      break;

    case 'v':
      verbose=1;
      break;
      
    default:
      usage ();
      return 0;
      
    }

  log = multilog_open ("dada_dbmonitor", 0);

  multilog_add (log, stderr);

  multilog_serve (log, DADA_DEFAULT_DBMONITOR_LOG);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (verbose)
    printf("Connecting to data block\n");
  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  /* get a pointer to the data block */
  uint64_t bufsz, nhbufs, ndbufs;

  dada_hdu_db_addresses(hdu, &nhbufs, &bufsz);
  dada_hdu_db_addresses(hdu, &ndbufs, &bufsz);

  uint64_t total_bytes = nhbufs * bufsz;
  if (verbose) {
    fprintf(stderr,"HEADER BLOCK:\n");
    fprintf(stderr,"Number of buffers: %"PRIu64"\n",nhbufs);
    fprintf(stderr,"Buffer size: %"PRIu64"\n",bufsz);
    fprintf(stderr,"Total buffer memory: %5.0f MB\n", ((double) total_bytes) / 
                                                       (1024.0*1024.0));
  }

  total_bytes = ndbufs * bufsz;

  fprintf(stderr,"DATA BLOCK:\n");
  fprintf(stderr,"Number of buffers: %"PRIu64"\n",ndbufs);
  fprintf(stderr,"Buffer size: %"PRIu64"\n",bufsz);
  fprintf(stderr,"Total buffer memory: %5.0f MB\n", ((double) total_bytes) / 
                                                       (1024.0*1024.0));

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}


const char * state_to_str(int state) 
{

  switch (state) 
  {
    case 0:
      return "disconnected";

    case 1:
      return "connected";

    case 2:
      return "one process that writes to the buffer";

    case 3:
      return "start-of-data flag has been raised";

    case 4:
      return "next operation will change writing state";

    case 5:
      return "one process that reads from the buffer";

    case 6:
      return "start-of-data flag has been raised";

    case 7:
      return "end-of-data flag has been raised";

    case 8:
      return "currently viewing";

    case 9:
      return "end-of-data while viewer";

    default:
      return "unknown";

  }

}


