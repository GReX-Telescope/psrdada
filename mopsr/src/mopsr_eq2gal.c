/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

void usage ()
{
	fprintf(stdout, "mopsr_eq2gal [options] ra dec\n"
    " ra          J2000 right ascension [degrees]\n"
    " deg         J2000 degreesascension [radians]\n"
    " -h          print this help text\n" 
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  char verbose = 0;

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
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 command line arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  double dr, dd;

  if (sscanf(argv[optind+0], "%lf", &dr) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse %s as RA in radians\n", argv[optind+0]);
    usage ();
    exit (EXIT_FAILURE);
  }

  if (sscanf(argv[optind+1], "%lf", &dd) != 1)
  { 
    fprintf (stderr, "ERROR: failed to parse %s as DEC in radians\n", argv[optind+1]); 
    usage ();
    exit (EXIT_FAILURE);
  }

  if (verbose)
  {
    fprintf (stderr, "parsed RA=%lf DEC=%lf\n", dr, dd);
  }

  // convert to radians
  const double deg2rad = M_PI / 180.0;
  dr *= deg2rad;
  dd *= deg2rad;

  // convert to galactic
  double dl, db;
  iauIcrs2g (dr, dd, &dl, &db);

  // convert back to degrees
  dl /= deg2rad;
  db /= deg2rad;

  fprintf (stdout, "%lf %lf\n", dl, db);

  return 0;
}
