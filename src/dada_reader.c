#include "dada.h"
#include "ipcio.h"
#include "multilog.h"

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

void usage()
{
  fprintf (stdout,
	   "dada_reader [options]\n"
	   " -d run as daemon\n");
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

  log = multilog_open ("dada_writer", daemon);

  /* set up for daemon usage */	  
  if (daemon) {

    if (fork() < 0)
      exit(EXIT_FAILURE);

    exit(EXIT_SUCCESS);

    /* Change the file mode mask */
    umask(0);

    /* Create a new SID for the child process */
    if (setsid() < 0)
      exit (EXIT_FAILURE);
              
    /* Change the current working directory */
    if (chdir("/") < 0)
      exit (EXIT_FAILURE);
        
    /* Close out the standard file descriptors */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    multilog_serve (log, dada.log_port);

  }

  /* First connect to the shared memory */
  if (ipcio_connect (&data_block, dada.data_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to data block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_lock_read ((ipcbuf_t*)&data_block) < 0) {
    multilog (log, LOG_ERR,"Could not lock designated reader status\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_connect (&header_block, dada.hdr_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to header block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_lock_read (&header_block) < 0) {
    multilog (log, LOG_ERR,"Could not lock designated reader status\n");
    return EXIT_FAILURE;
  }

  /* Disconnect from the shared memory */
  if (ipcbuf_unlock_read ((ipcbuf_t*)&data_block) < 0) {
    multilog (log, LOG_ERR,"Could not unlock designated reader status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_disconnect (&data_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from data block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_unlock_read (&header_block) < 0) {
    multilog (log, LOG_ERR,"Could not unlock designated reader status\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_disconnect (&header_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from header block\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

