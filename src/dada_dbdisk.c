#include "dada.h"
#include "ipcio.h"
#include "multilog.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void usage()
{
  fprintf (stdout,
	   "dada_dbdisk [options]\n"
	   " -b <buffersize>\n"
	   " -f <filesize>\n"
	   " -F <rootfilename>\n"
	   " -d run as daemon\n"
	   " -A Adaptive sleep algorithm (slow virtual memory flushing)\n");
}

int main_loop (dada_t* dada,
	       ipcio_t* data_block,
	       ipcbuf_t* header_block,
	       multilog_t* log)
{
  char* header = 0;
  uint64_t header_size = 0;

  char* buffer = 0;
  uint64_t buffer_size = 0;

  uint64_t bytes_read = 0;

  /* Wait for the next valid header sub-block */
  header = ipcbuf_get_next_read (header_block, &header_size);
    
  // get the header size and duplicate it

  ipcbuf_mark_cleared (header_block);

  while (!ipcbuf_eod((ipcbuf_t*)data_block)) {

    bytes_read = ipcio_read (data_block, buffer, buffer_size);

    if (bytes_read < 0) {
      perror ("ipcio_read error");
      return -1;
    }

  }

  ipcio_reset (data_block);

  return 0;

}


int main (int argc, char **argv)
{
  /* DADA configuration */
  dada_t dada;

  /* DADA Data Block */
  ipcio_t data_block = IPCIO_INIT;

  /* DADA Header Block */
  ipcbuf_t header_block = IPCBUF_INIT;

  /* DADA Logger */
  multilog_t* log;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  while ((arg=getopt(argc,argv,"dv")) != -1) {
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
  }

  dada_init (&dada);

  log = multilog_open ("dada_dbdisk", daemon);

  if (daemon) {

    /* set up for daemon usage */	  
    if (fork() < 0)
      exit(EXIT_FAILURE);

    exit(EXIT_SUCCESS);

  }
  else
    multilog_add (log, stderr);

  multilog_serve (log, dada.log_port);

  /* First connect to the shared memory */
  if (ipcbuf_connect (&header_block, dada.hdr_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to header block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_lock_read (&header_block) < 0) {
    multilog (log, LOG_ERR, "Could not lock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_connect (&data_block, dada.data_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to data block\n");
    return EXIT_FAILURE;
  }

  if (ipcio_open (&data_block, 'R') < 0) {
    multilog (log, LOG_ERR, "Failed to open data block for reading\n");
    return EXIT_FAILURE;
  }


  while (!state.quit) {

    if (main_loop (dada, &data_block, &header_block, log) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

  }

  /* Disconnect from the shared memory */
  if (ipcio_close (&data_block) < 0) {
    multilog (log, LOG_ERR, "Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_disconnect (&data_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from data block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_unlock_write (&header_block) < 0) {
    multilog (log, LOG_ERR,"Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_disconnect (&header_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from header block\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

