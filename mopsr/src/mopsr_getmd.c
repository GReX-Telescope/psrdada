/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "mopsr_delays.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <slalib.h>

void usage ()
{
	fprintf(stdout, "mopsr_getmd utc RA DEC\n"
    " utc         UTC in YYYY-MM-DD-HH:MM:SS\n"
    " RA          J2000 RA in HH:MM:SS\n"
    " DEC         J2000 DEC in DD:MM:SS\n"
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  char verbose = 0;

  int device = 0;

  while ((arg = getopt(argc, argv, "hv")) != -1) 
  {
    switch (arg)  
    {
      case 'h':
        usage ();
        return 0;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char * utc_start_str = strdup (argv[optind]);

  // convert UTC_START to a unix UTC
  time_t utc_start = str2utctime (utc_start_str);
  if (utc_start == (time_t)-1)
  {
    fprintf (stderr, "ERROR:  could not parse start time from '%s'\n", utc_start_str);
    return -1;
  }

  if (verbose)
    fprintf (stderr, "utc_start=%ld\n", utc_start);

  mopsr_source_t source;
  if (mopsr_delays_hhmmss_to_rad (argv[optind+1], &(source.raj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse RA from %s\n", argv[optind+1]);
    return -1;
  }

  if (mopsr_delays_ddmmss_to_rad (argv[optind+2], &(source.decj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse DEC from %s\n", argv[optind+2]);
    return -1;
  }

  struct timeval timestamp;
  timestamp.tv_sec = utc_start;
  timestamp.tv_usec = 0;

  calc_app_position (source.raj, source.decj, timestamp,
                      &(source.ra_curr), &(source.dec_curr));

  if (verbose)
  {
    int HMSF[4];
    int DMSF[4];
    char sign;
    int NDP = 2;    // number of decimal places

    slaCr2tf(NDP, source.ra_curr, &sign, HMSF);
    fprintf (stderr, "Apparent RA=%02d:%02d:%02d.%d\n", HMSF[0],HMSF[1],HMSF[2],HMSF[3]);

    slaDr2af(NDP, source.dec_curr, &sign, DMSF);
    fprintf (stderr, "Apparent DEC=%c%02d:%02d:%02d.%d\n", sign, DMSF[0],DMSF[1],DMSF[2],DMSF[3]);
  }

  double jer_delay = calc_jer_delay2 (source.ra_curr, source.dec_curr, timestamp);
  double md_angle = asin(jer_delay) * (180 / M_PI);

  fprintf (stdout, "%lf\n", md_angle);
  return 0;
}