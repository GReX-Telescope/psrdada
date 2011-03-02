/***************************************************************************
 *
 *   Copyright (C) 2009 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#include "bibob.h"

void usage() {
  fprintf(stdout,
    "bibob_start_observation [options] host port\n"
    " host        of the ibob telnet interface\n"
    " port        of the ibob telnet interace\n"
    "\n"
    " -m int      start on modulo int second[relative to midnight]\n"
    " -v          verbose\n"
    " -n          dont actaully rearm the bibob\n"
    " -h          print help text\n");

}

int main (int argc, char** argv)
{

  char * hostname;
  int port;
  int verbose = 0;
  int n_attempts = 3;
  int arg = 0;
  int r = 0;
  int rearm = 1;
  int modulo = 0;

  while ((arg=getopt(argc,argv,"m:nhv")) != -1) {

    switch (arg) {

    case 'h':
      usage();
      return 0;

    case 'm':
      modulo = atoi(optarg);
      break;

    case 'n':
      rearm = 0;
      break;

    case 'v':
      verbose = 1;
      break;

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

  bibob_t * bibob = bibob_construct();

  if (bibob_set_host(bibob, hostname, port) < 0)
  {
    fprintf(stderr, "could not set bibob host and port\n");
    bibob_destroy(bibob);
    return EXIT_FAILURE;
  }

  if (bibob_open(bibob) < 0) 
  {
    fprintf(stderr, "could not open connection to bibob\n");
    bibob_destroy(bibob);
    return EXIT_FAILURE;
  }

  unsigned done = 0;
  char command[100];
  int len = 0;
  size_t bytes = 0;

  /* ensure we have a connection, this command should return 321 bytes */
  sprintf(command, "help");
  bytes = bibob_send(bibob, command);
  if (verbose)
    fprintf(stdout, "bibob <- %s [%d bytes]\n", command, bytes);

  bytes = bibob_recv(bibob);
  if (verbose)
    fprintf(stdout, "bibob -> %s [%d bytes]\n", bibob->buffer, bytes);
  
  if (bytes != 321)
  {
    fprintf(stderr, "could not read expected data from bibob\n");
    bibob_close(bibob);
    bibob_destroy(bibob);
    return EXIT_FAILURE;
  }

  struct timeval t_stamp;
  char start_time[64];
  char utc_time[64];
  char local_time[64];
  /* busy sleep until a second has ticked over */
  time_t current = time(0);
  time_t prev = current;
  time_t start = 0;
  struct timeval timeout;

  // we will issue the rearm 0.5 seconds into a second
  timeout.tv_sec=0;
  timeout.tv_usec=500000;


  if (verbose) 
  {
    gettimeofday(&t_stamp, NULL);
    strftime (local_time, 64, DADA_TIMESTR, localtime(&(t_stamp.tv_sec)));
    fprintf(stderr, "[%s.%06d] entering busy wait loop for next 1sec tick\n", local_time, t_stamp.tv_usec);
    //strftime (utc_time, 64, DADA_TIMESTR, gmtime(&current));
    //fprintf(stdout, "entering busy wait loop for next time tick, current utc=%s, current local=%s\n", utc_time, local_time);
  }

  // if we have been asked to start on a second that is modulo the number of seconds since midnight
  if (modulo) {

    // get the current time
    gettimeofday(&t_stamp, NULL);

    // convert to string UTC time at midnight
    strftime (utc_time, 64, "%Y-%m-%d-00:00:00", gmtime(&(t_stamp.tv_sec)));

    // convert UTC time back to a unix time
    time_t midnight = str2utctime(utc_time);

    // determine offset from current time
    time_t remainder = (current - midnight) % modulo;

    time_t to_add = (modulo - remainder);

    // we need a few seconds to be sure we start on time
    if (to_add < 3) 
      to_add += modulo;

    start = current + to_add;

  } else {
  
    start = current + 3;
 
  }

  strftime (start_time, 64, DADA_TIMESTR, localtime(&start));

  if (verbose) {
    gettimeofday(&t_stamp, NULL);
    strftime (local_time, 64, DADA_TIMESTR, localtime(&(t_stamp.tv_sec)));
    fprintf(stderr, "[%s.%06d] start_time = %s [%d]\n", local_time, t_stamp.tv_usec, start_time, start);
  }

  /* busy sleep until we are 1 seconds behind the start time */
  while (current < (start - 1))
    current = time(0);

  if (verbose)
  {
    gettimeofday(&t_stamp, NULL);
    strftime (local_time, 64, DADA_TIMESTR, localtime(&(t_stamp.tv_sec)));
    fprintf(stderr, "[%s.%06d] exit busy wait loop for next 1sec tick\n", local_time, t_stamp.tv_usec);
  }


  // sleep for the timeout [0.5 s]
  select(0,NULL,NULL,NULL,&timeout);

  if (verbose)
  {
    gettimeofday(&t_stamp, NULL);
    strftime (local_time, 64, DADA_TIMESTR, localtime(&(t_stamp.tv_sec)));
    fprintf(stderr, "[%s.%06d] exit 0.5 second select \n", local_time, t_stamp.tv_usec);
  }

  // issue the rearm command to the bibob
  if (rearm) 
  {

    sprintf(command, "regwrite reg_arm 0");
    bytes = bibob_send(bibob, command);
    if (verbose)
      fprintf(stdout, "bibob <- %s [%d bytes]\n", command, bytes);

    bytes = bibob_recv(bibob);
    if (verbose)
      fprintf(stdout, "bibob -> %s [%d bytes]\n", bibob->buffer, bytes);

    sprintf(command, "regwrite reg_arm 1");
    bytes = bibob_send(bibob, command);
    if (verbose)
      fprintf(stdout, "bibob <- %s [%d bytes]\n", command, bytes);

    bytes = bibob_recv(bibob);
    if (verbose)
      fprintf(stdout, "bibob -> %s [%d bytes]\n", bibob->buffer, bytes);

    if (verbose) 
    {
      gettimeofday(&t_stamp, NULL);
      strftime (local_time, 64, DADA_TIMESTR, localtime(&(t_stamp.tv_sec)));
      fprintf(stderr, "[%s.%06d] rearm completed\n", local_time, t_stamp.tv_usec);
    }

  }

  current = time(0);

  if ((current+1) != start)
    fprintf(stderr, "something went wrong: current+1 [%d] != start_time [%d]\n", current+1, start);

  strftime (utc_time, 64, DADA_TIMESTR, gmtime(&start));

  fprintf (stdout, "%s\n", utc_time);

  bibob_close(bibob);
  bibob_destroy(bibob);

  return 0;
}


