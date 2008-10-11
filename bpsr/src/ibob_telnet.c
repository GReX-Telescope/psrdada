
#include "sock.h"
#include "ibob.h"

#include <ctype.h>
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

void filter (const char* out)
{
  if (!out)
    return;

  for (; *out != '\0'; out++)
    if (!isprint(*out) && *out != '\n' && *out != '\r')
      printf ("telnet code: %d received\n", (unsigned)(*out));
    else
      putchar (*out);
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

  if (verbose)
    printf ("ibob_telnet: opening %s %d\n", hostname, port);

  int fd = sock_open (hostname, port);
  if ( fd < 0 )
  {
    fprintf (stderr, "could not open %s %d: %s\n",
	     hostname, port, strerror(errno));
    return -1;
  }

#define BUFFER 128
  char buffer [BUFFER];

  int emulate_length = strlen (emulate_telnet_msg1);
  if (write (fd, emulate_telnet_msg1, emulate_length) < emulate_length)
  {
    fprintf (stderr, "could not send emulate telnet 1: %s\n",
             strerror(errno));
    sock_close (fd);
    return -1;
  }

  if (read (fd, buffer, 6) < 6)
  {
    fprintf (stderr, "could not read telnet response\n");
    sock_close (fd);
    return -1;
  }

  emulate_length = strlen (emulate_telnet_msg2);
  if (write (fd, emulate_telnet_msg2, emulate_length) < emulate_length)
  {
    fprintf (stderr, "could not send emulate telnet 2: %s\n",
             strerror(errno));
    sock_close (fd);
    return -1;
  }

  while (1)
  {
    do 
    {
      int got = read (fd, buffer, BUFFER);
      if (got < 0)
      {
        perror ("error while reading from iBoB socket");
        break;
      }
      buffer[got] = '\0';
      printf ("%s", buffer);
    }
    while (!strstr(buffer, ibob_prompt));

    fgets (buffer, BUFFER, stdin);

    if (strstr (buffer, "quit"))
      break;

    char* newline = strchr (buffer, '\n');
    if (newline)
      *newline = '\r';

    int length = strlen (buffer);
    write (fd, buffer, length);
  }

  fprintf (stderr, "Closing connection\n");
  sock_close (fd);
  return 0;
}

