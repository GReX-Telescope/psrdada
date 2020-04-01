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
	fprintf(stdout, "mopsr_getlst [options] utc\n"
    " utc         UTC in YYYY-MM-DD-HH:MM:SS\n"
    " -f seconds  fractional seconds to add to the UTC\n"
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  char verbose = 0;

  int device = 0;

  double fractional_seconds = 0.0;
  const double seconds_per_day = 86400;

  while ((arg = getopt(argc, argv, "f:hv")) != -1) 
  {
    switch (arg)  
    {
      case 'f':
        fractional_seconds = atof(optarg);
        break;

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
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 argument is required\n");
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

  // calculate the MJD
  struct tm * local_t = gmtime (&utc_start);
  double mjd = mjd_from_utc (local_t);
  mjd += (fractional_seconds / seconds_per_day);

  if (verbose)
    fprintf (stdout, "mjd=%lf\n", mjd);

#ifdef HAVE_SLA
  // get the LST
  float lmst = (float) lmst_from_mjd (mjd);
#endif
  float last = (float) last_from_mjd (mjd);

  int HMSF[4];
  char sign;
  int NDP = 4;    // number of decimal places

  iauA2tf (NDP, last, &sign, HMSF);
  fprintf (stdout, "LAST: %02d:%02d:%02d.%04d [radians=%f]\n", HMSF[0],HMSF[1],HMSF[2],HMSF[3], last);

  return 0;
}
