#include "dada_pwc_nexus.h"
#include "daemon.h"

#include <stdlib.h>
#include <unistd.h>

void usage()
{
  fprintf (stdout,
	   "dada_nexus [options]\n"
	   " -d         run as daemon\n");
}

int main (int argc, char **argv)
{
  /* the DADA nexus command distribution interface */
  dada_pwc_nexus_t* nexus = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  /* configuration file name */
  char* dada_config = getenv ("DADA_CONFIG");

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

  if (!dada_config) {
    fprintf (stderr, "Please define the DADA_CONFIG environment variable\n");
    return -1;
  }

  log = multilog_open ("dada_nexus", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, 123);
  }
  else
    multilog_add (log, stderr);

  fprintf (stderr, "Creating DADA PWC nexus\n");
  nexus = dada_pwc_nexus_create ();

  fprintf (stderr, "Configuring DADA PWC nexus\n");
  if (dada_pwc_nexus_configure (nexus, dada_config) < 0) {
    fprintf (stderr, "Error while configuring the DADA nexus\n");
    return -1;
  }

  fprintf (stderr, "Running the DADA PWC nexus server\n");
  if (dada_pwc_nexus_serve (nexus) < 0) {
    fprintf (stderr, "Error while running the DADA PWC nexus server\n");
    return -1;
  }

  fprintf (stderr, "Destroying nexus\n");
  dada_pwc_nexus_destroy (nexus);

  return EXIT_SUCCESS;
}
