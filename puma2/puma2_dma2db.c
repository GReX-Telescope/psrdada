#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <syslog.h>

#include "dada_db.h"

void usage()
{
  fprintf(stderr,
	  "puma2_dma2db [options]\n"
	  " -n <numberbufs>\n"
	  " -b <buffersize>\n"
	  " -d run as daemon\n");
}

int main (int argc, char **argv)
{
  /* DADA Data Block ring buffer */
  dada_db_t shm = DADA_DB_INIT;

  /* Name of the host on which puma2_dma2db is running */
  char* hostname = 0;

  /* Flag set in daemon mode */
  int daemon = 0;

  /* Flag set in verbose mode */
  int verbose = 0;

  int arg = 0;
  while ((arg=getopt(argc,argv,"nbdv")) != -1) {
    switch (arg) {
      case 'd':
	daemon=1;
	break;
      case 'v':
        verbose=1;
        break;
      default:
	usage ();
	return EXIT_FAILURE;
    }
  }

  /* set up for daemon usage */	  
  if (daemon==1) {
    if (fork() < 0)
      exit(EXIT_FAILURE);
    exit(EXIT_SUCCESS);

    setsid(); 
    // setuid(500);
  }
  
  openlog("puma2_dma2db",LOG_CONS,LOG_USER);

  /* First connect to the shared memory */

  if (dada_db_connect (&shm) == -1) {
    if (!daemon)
      fprintf (stderr, "Failed to connect to shared memory area\n");
    else if (verbose)
      syslog(LOG_ERR,"Failed to connect to shared memory: %m");
    return EXIT_FAILURE;
  }

  if (daemon) {
    if (verbose)
      syslog(LOG_INFO,"connected to shared memory");
  }
  else
    fprintf(stderr,"connected to shared memory");

  if (ipcbuf_lock_write ((ipcbuf_t *) &shm) < 0) {
    if (!daemon)
      fprintf (stderr, "Could not lock designated writer status\n");
    else if (verbose)
      syslog (LOG_ERR, "Could not lock designated writer status: %m\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

