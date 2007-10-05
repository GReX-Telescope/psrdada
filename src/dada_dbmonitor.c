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


#define IPCBUF_WRITE  0   /* semaphore locks writer status */
#define IPCBUF_READ   1   /* semaphore locks reader (+clear) status */
#define IPCBUF_SODACK 2   /* acknowledgement of start of data */
#define IPCBUF_EODACK 3   /* acknowledgement of end of data */
#define IPCBUF_FULL   4   /* semaphore counts full buffers */
#define IPCBUF_CLEAR  5   /* semaphore counts emptied buffers */
#define IPCBUF_NSEM   6   /* total number of semaphores */


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

  /* Quit flag */
  char quit = 0;

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

  printf("locking read...\n");
  //if (dada_hdu_lock_read (hdu) < 0)
    //return EXIT_FAILURE;

  /* get a pointer to the header block */
  ipcbuf_t *header_block = hdu->header_block;

  /* get a pointer to the data block */
  ipcio_t* data_block = hdu->data_block;
 
  int write_buffs = 0;
  int read_buffs = 0;
  int total_buffs = 0;
  int full_buffs = 0;
  int clear_buffs = 0;
  int buff_size;

  //total_buffs = semctl (header_block->semid, IPCBUF_NSEM, GETVAL);
 
  ipcbuf_t *tricky = (ipcbuf_t *) data_block;

  total_buffs = tricky->sync->nbufs;
  buff_size = tricky->sync->bufsz;
 
  fprintf(stderr,"Number of buffers: %d\n",total_buffs);
  fprintf(stderr,"Buffer size: %d\n",buff_size);
  fprintf(stderr,"total\twrite\tread\n");

  fprintf(stderr,"FULL\tCLEAR\tWRITTEN\tREAD\n");
  while (!quit) {


    write_buffs = tricky->sync->w_buf;
    read_buffs = tricky->sync->r_buf;
    full_buffs = semctl (tricky->semid, IPCBUF_FULL, GETVAL);
    clear_buffs = semctl (tricky->semid, IPCBUF_CLEAR, GETVAL);
  
    fprintf(stderr,"%d\t%d\t%d\t%d\t%d\n",full_buffs, clear_buffs, write_buffs, read_buffs);

    sleep(1);

  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
