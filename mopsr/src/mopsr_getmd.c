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

void usage ()
{
	fprintf(stdout, "mopsr_getmd [options] utc RA DEC\n"
    " return the Meridian Distance angle in degrees for the specified epoch and position\n"
    " utc             UTC in YYYY-MM-DD-HH:MM:SS\n"
    " RA              J2000 RA in HH:MM:SS\n"
    " DEC             J2000 DEC in DD:MM:SS\n"
    " -f fractional   add fractional seconds to the UTC\n"
    " -u fractional   add fractional UTS1 offset to the UTC [seconds]\n"
    " -h              print this help text\n" 
    " -v              verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  char verbose = 0;

  int device = 0;

  float fractional = 0;

  float ut1_offset = 0;

  char negative_dec = 0;

  while ((arg = getopt(argc, argv, "f:hnu:v")) != -1) 
  {
    switch (arg)  
    {
      case 'f':
        fractional = atof(optarg);
        if ((fractional <= 0) || (fractional >= 1))
        {
          fprintf (stderr, "ERROR: -f expects argument between 0 and 1 exclusive\n");
          usage();
          exit (EXIT_FAILURE);
        }
        break;

      case 'n':
        negative_dec = 1;
        break;

      case 'h':
        usage ();
        return 0;

      case 'u':
        ut1_offset = atof(optarg);
        break;

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
    fprintf (stdout, "utc_start=%ld\n", utc_start);

  mopsr_source_t source;
  if (mopsr_delays_hhmmss_to_rad (argv[optind+1], &(source.raj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse RA from %s\n", argv[optind+1]);
    return -1;
  }
  if (verbose)
    fprintf (stdout, "RA str=%s deg=%lf\n", argv[optind+1], source.raj);

  char dec[32];
  if (negative_dec)
  {
    dec[0] = '-';
    strcpy(dec+1, argv[optind+2]);
  }
  else
    strcpy(dec, argv[optind+2]); 

  if (mopsr_delays_ddmmss_to_rad (dec, &(source.decj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse DEC from %s\n", argv[optind+2]);
    return -1;
  }
  if (verbose)
    fprintf (stdout, "DEC str=%s deg=%lf\n", argv[optind+2], source.decj);

  struct timeval timestamp;
  timestamp.tv_sec = utc_start;
  timestamp.tv_usec = (uint64_t) (fractional * 1e6);

  if (ut1_offset != 0)
  {
    int64_t usec = (int64_t) timestamp.tv_usec;
    usec += (ut1_offset * 1e6);
    while (usec < -1e6)
    {
      timestamp.tv_sec--;
      usec += 1e6;
    }

    while (usec > 1e6)
    {
      timestamp.tv_sec++;
      usec -= 1e6;
    }

    timestamp.tv_usec = (uint64_t) usec;
  }
  
  struct tm * utc = gmtime (&utc_start);
  int rval = cal_app_pos_iau (source.raj, source.decj, utc,
                              &(source.ra_curr), &(source.dec_curr));
  if (rval < 0)
    fprintf(stderr, "ERROR: cal_app_pos_iau failed\n");

  if (verbose)
  {
    int HMSF[4];
    int DMSF[4];
    char sign;
    int NDP = 2;    // number of decimal places

    iauA2tf (NDP, source.ra_curr, &sign, HMSF);
    fprintf (stdout, "Apparent RA=%02d:%02d:%02d.%d\n", HMSF[0],HMSF[1],HMSF[2],HMSF[3]);

    iauA2af(NDP, source.dec_curr, &sign, DMSF);
    fprintf (stdout, "Apparent DEC=%c%02d:%02d:%02d.%d\n", sign, DMSF[0],DMSF[1],DMSF[2],DMSF[3]);
  }

  double jer_delay = calc_jer_delay (source.ra_curr, source.dec_curr, timestamp);
  double md_angle = asin(jer_delay) * (180.0 / M_PI);

  fprintf (stdout, "%17.15lf\n", md_angle);
  return 0;
}
