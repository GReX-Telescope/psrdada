#include "sock.h"
#include "dada_def.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

void usage();

void usage() {
  fprintf(stdout,
    "ibob_rearm_trigger host port\n"
    " host        on which the ibob interface is accessible\n"
    " port        on which the ibob interace is accessible\n"
    " -v          verbose\n");

}

int main (int argc, char** argv)
{

  char * hostname;
  int port;
  int verbose = 0;
  int arg = 0;
  FILE *fptr;
  char buffer[64];
  int buffer_size = 64; 
  char buffer2[64];

  while ((arg=getopt(argc,argv,"n:i:vh")) != -1) {
    switch (arg) {

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

  if ((argc - optind) != 2) {
    fprintf(stderr, "Error: host and port must be specified\n");
    usage();
    return EXIT_FAILURE;
  } else {
    hostname = argv[optind];
    port = atoi(argv[(optind+1)]);
  }

  fptr = fopen("/lfs/data0/bpsr/logs/rearm.log","a");
  if (!fptr) {
    system("echo 'could not open log file for appending' >> /lfs/data0/bpsr/logs/rearm.log");
  }

  char command[100];
  int rval = 0;

  int fd = sock_open_quiet(hostname, port);

  if (verbose) 
    printf("opened socket to %s on port %d [fd %d]\n",hostname, port, fd);

  time_t current = time(0);

  if (fd < 0)  {
    strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
    fprintf(fptr, "[%s] Error creating socket\n", buffer);
    return -1;
  }

  struct timeval timeout;
  timeout.tv_sec=0;
  timeout.tv_usec=500000;

  current = time(0);
  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] Waiting for next second\n", buffer);

  /* Busy sleep until a second has ticked over */
  time_t prev = current;
  while (current == prev) {
    current = time(0);
  }

  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] 1 second tick, waiting for 500,000 usecs\n", buffer);

  /* Now sleep for 0.5 seconds */
  select(0,NULL,NULL,NULL,&timeout);

  current = time(0);
  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] Wait over, rearming\n", buffer);

  sprintf(command, "regwrite reg_arm 0\r\n");

  current = time(0);
  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] regwrite reg_arm 0\n", buffer);

  if (verbose) printf("<- %s",command);
  if (rval = sock_write(fd, command, strlen(command)-1) < 0 ) {
    fprintf(stderr, "could not write command %s\n",command);
    sock_close(fd);
     return(1);
  }

  timeout.tv_sec=0;
  timeout.tv_usec=10000;
  select(0,NULL,NULL,NULL,&timeout);

  current = time(0);
  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] regwrite reg_arm 1\n", buffer);

  sprintf(command, "regwrite reg_arm 1\r\n");

  if (verbose) printf("<- %s",command);
  if (rval = sock_write(fd, command, strlen(command)-1) < 0 ) {
    fprintf(stderr, "could not write command %s\n",command);
    sock_close(fd);
    return(1);
  }

  current = time(0);
  printf("%d\n",(current+1));

  strftime (buffer, buffer_size, DADA_TIMESTR, localtime(&current));
  current++;
  strftime (buffer2, buffer_size, DADA_TIMESTR, localtime(&current));
  fprintf(fptr, "[%s] returing time %d [%s]\n", buffer, (current), buffer2);

  if (verbose) printf("closing socket\n");
  sock_close (fd);
  fclose(fptr);

  return 0;
}

