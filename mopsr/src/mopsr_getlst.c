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
	fprintf(stdout, "mopsr_getlst utc\n"
    " utc         UTC in YYYY-MM-DD-HH:MM:SS\n"
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
    fprintf (stderr, "utc_start=%ld\n", utc_start);

  // calculate the MJD
  struct tm * local_t = gmtime (&utc_start);
  double mjd = mjd_from_utc (local_t);
  if (verbose)
    fprintf (stderr, "mjd=%lf\n", mjd);

  // get the LST
  float lst = (float) lmst_from_mjd (mjd);
  if (verbose)
    fprintf (stderr, "lst=%f\n", lst);

  int HMSF[4];
  char sign;
  int NDP = 2;    // number of decimal places

  slaCr2tf(NDP, lst, &sign, HMSF);
  fprintf (stderr, "%02d:%02d:%02d.%d [radians=%f]\n", HMSF[0],HMSF[1],HMSF[2],HMSF[3], lst);

  return 0;
}
