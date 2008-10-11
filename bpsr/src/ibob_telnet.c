
#include "sock.h"
#include "ibob.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

void usage()
{
  printf ("ibob_telnet [host port|-n N] \n"
	  " host        on which the ibob interface is accessible \n"
	  " port        on which the ibob interace is accessible \n"
	  " -n N        connect to %sN port %d \n"
	  " -v          verbose\n", IBOB_VLAN_BASE, IBOB_PORT );
}

int main (int argc, char** argv)
{
  char* hostname = 0;
  int port = 0;
  int ibob = 0;

  int verbose = 0;
  int arg = 0;

  while ((arg=getopt(argc,argv,"n:i:vh")) != -1)
  {
    switch (arg) {

    case 'n':
      ibob = atoi (optarg);
      break;

    case 'v':
      verbose = 1;
      break;

    case 'h':
      usage();
      return 0;

    default:
      usage();
      return 0;
    }
  }

  if (ibob > 0)
  {
    hostname = strdup (IBOB_VLAN_BASE"XXXXXXX");
    sprintf (hostname, IBOB_VLAN_BASE"%d", ibob);
    port = IBOB_PORT;
  }
  else
  {
    if ((argc - optind) != 2)
    {
      fprintf(stderr, "Error: host and port must be specified\n");
      usage();
      return EXIT_FAILURE;
    }
    hostname = argv[optind];
    port = atoi(argv[(optind+1)]);
  }

  int fd = sock_open (hostname, port);
  if ( fd < 0 )
  {
    fprintf (stderr, "could not open %s %d: %s\n",
	     hostname, port, strerror(errno));
    return -1;
  }

  FILE* sockin = 0;
  FILE* sockout = 0;

  sockin = fdopen (fd, "r");
  sockout = fdopen (fd, "w");

  fprintf (sockout, fake_telnet);

#define BUFFER 128
  char buffer [BUFFER];
  char* rgot = 0;

  do 
  {
    rgot = fgets (buffer, BUFFER, sockin);
  }
  while (rgot && !strstr(rgot, "IBOB"));

  sock_close (fd);
  return 0;
}

