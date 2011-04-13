/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson
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
#include <time.h>
#include "dada_def.h"

void usage() {
  fprintf(stdout,
    "dspsr_start_time from\n"
    " from        unix time to calculate from\n"
    "\n"
    " -m int      start on modulo int second [relative to midnight]\n"
    " -v          verbose\n"
    " -h          print help text\n");

}

int main (int argc, char** argv)
{

  int verbose = 0;
  int arg = 0;
  int r = 0;
  int modulo = 0;
  unsigned long from_time = 0;

  while ((arg=getopt(argc,argv,"m:hv")) != -1) {

    switch (arg) {

    case 'h':
      usage();
      return 0;

    case 'm':
      modulo = atoi(optarg);
      break;

    case 'v':
      verbose = 1;
      break;

    default:
      usage();
      return 0;
    }
  }

  if ((argc - optind) != 1) {
    fprintf(stderr, "Error: from time must be specified\n");
    usage();
    return EXIT_FAILURE;
  } else {
    from_time = atoi(argv[optind]);
  }

  struct timeval t_stamp;
  char start_time[64];
  char utc_time[64];
  char local_time[64];
  
  time_t current = from_time;
  time_t start = 0;

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

    // convert to string UTC time at midnight
    strftime (utc_time, 64, "%Y-%m-%d-00:00:00", gmtime(&(from_time)));

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

  strftime (utc_time, 64, DADA_TIMESTR, gmtime(&start));

  fprintf (stdout, "%s\n", utc_time);

  return 0;
}


