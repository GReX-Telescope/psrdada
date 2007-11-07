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
     "dada_dbmonitor [options]\n"
     " -v         be verbose\n"
     " -d         run as daemon\n");
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

  int arg = 0;

  /* TODO the amount to conduct a busy sleep inbetween clearing each sub
   * block */

  while ((arg=getopt(argc,argv,"dv")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;

    case 'v':
      verbose=1;
      break;
      
    default:
      usage ();
      return 0;
      
    }

  log = multilog_open ("dada_dbmonitor", daemon);

  if (daemon) 
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  multilog_serve (log, DADA_DEFAULT_DBMONITOR_LOG);

  hdu = dada_hdu_create (log);

  printf("connecting\n");
  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  /* get a pointer to the data block */
 
  ipcbuf_t *hb = hdu->header_block;
  ipcbuf_t *db = (ipcbuf_t *) hdu->data_block;
          
  uint64_t bufsz = ipcbuf_get_bufsz (hb);
  uint64_t nhbufs = ipcbuf_get_nbufs (hb);

  fprintf(stderr,"HEADER BLOCK:\n");
  fprintf(stderr,"Number of buffers: %"PRIu64"\n",nhbufs);
  fprintf(stderr,"Buffer size: %"PRIu64"\n",bufsz);
  uint64_t total_bytes = nhbufs * bufsz;
  fprintf(stderr,"Total buffer memory: %5.0f MB\n", ((double) total_bytes)/(1024.0*1024.0));

  bufsz = ipcbuf_get_bufsz (db);
  uint64_t ndbufs = ipcbuf_get_nbufs (db);

  fprintf(stderr,"DATA BLOCK:\n");
  fprintf(stderr,"Number of buffers: %"PRIu64"\n",ndbufs);
  fprintf(stderr,"Buffer size: %"PRIu64"\n",bufsz);
  total_bytes = ndbufs * bufsz;
  fprintf(stderr,"Total buffer memory: %5.0f MB\n", ((double) total_bytes)/(1024.0*1024.0));

  int i=0;
  for (i=0;i<IPCBUF_XFERS;i++) {
    fprintf(stderr,"%d: [%"PRIu64",%"PRIu64"]=>[%"PRIu64",%"PRIu64"] %d\n", i,
                    db->sync->s_buf[i],db->sync->s_byte[i],
                    db->sync->e_buf[i],db->sync->e_byte[i],db->sync->eod[i]);
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
