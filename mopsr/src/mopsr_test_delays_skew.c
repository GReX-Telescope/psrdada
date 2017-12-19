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
#include <sys/stat.h>
#include <cpgplot.h>
#include <float.h>

void mopsr_skew_plot (unsigned nskew, float * xvals, float * skews1, float * skews2, float * skews);

void usage ()
{
	fprintf(stdout, "mopsr_test_delays_skew header_file\n"
    " -d dec      delta declination to apply (degrees, default 1)\n" 
    " -h          print this help text\n" 
    " -v          verbose output\n\n" 
    "Plot the delay and phase as a function of time\n"
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned verbose = 0;

  double delta_dec = 0;

  while ((arg = getopt(argc, argv, "d:hv")) != -1) 
  {
    switch (arg)  
    {
      case 'd':
        delta_dec = (double) atof (optarg) * M_PI / 180;
        break;

      case 'h':
        usage ();
        return 0;

      case 'v':
        verbose ++;
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

  unsigned nskews = 100;
  float * xvals = (float *) malloc (nskews * sizeof(float));
  float * skews1 = (float *) malloc (nskews * sizeof(float));
  float * skews2 = (float *) malloc (nskews * sizeof(float));
  float * skews  = (float *) malloc (nskews * sizeof(float));

  // read the header describing the observation
  char * header_file = strdup (argv[optind]);
  struct stat buf;
  if (stat (header_file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat header_file [%s]: %s\n", header_file, strerror(errno));
    return (EXIT_FAILURE);
  }
  size_t header_size = buf.st_size + 1;
  char * header = (char *) malloc (header_size);
  if (fileread (header_file, header, header_size) < 0)
  {
    fprintf (stderr, "ERROR: could not read header from %s\n", header_file);
    return EXIT_FAILURE;
  }

  char tmp[32];
  if (ascii_header_get (header, "UTC_START", "%s", tmp) == 1)
  {
    if (verbose)
      fprintf (stderr, " UTC_START=%s\n", tmp);
  }
  else
  {
    fprintf (stderr, " UTC_START=UNKNOWN\n");
  }
  // convert UTC_START to a unix UTC
  time_t utc_start = str2utctime (tmp);
  if (utc_start == (time_t)-1)
  {
    fprintf (stderr, "ERROR:  could not parse start time from '%s'\n", tmp);
    return -1;
  }

  // now calculate the apparent RA and DEC for the current timestamp
  struct timeval timestamp;
  timestamp.tv_sec = utc_start;
  timestamp.tv_usec = 0;

  cpgopen ("/xs");
  cpgask (0);

  double start_skew = -1 * MOLONGLO_AZIMUTH_CORR;
  double end_skew   = 3 * MOLONGLO_AZIMUTH_CORR;
  double skew_step  = (end_skew - start_skew) / nskews;

  double hour_angle1 = 0.0;
  double hour_angle2 = 0.0;

  char * dec_str1 = "02:03:09.0";  // 3C273
  char * dec_str2 = "-75:07:20.0"; // CJ0408-7507

  double dec1;
  double dec2;
  mopsr_delays_ddmmss_to_rad (dec_str1, &dec1);
  mopsr_delays_ddmmss_to_rad (dec_str2, &dec2);

  fprintf (stderr, "dec1=%lf dec2=%lf\n", dec1, dec2);

  double projected_delay1, projected_delay2, md1, md2;
  unsigned i;

  for (i=0; i<nskews; i++)
  {
    double skew = start_skew + (skew_step * i);
    xvals[i] = skew;

    // first the errors for 3C273
    projected_delay1 = doc_delay (hour_angle1,
                                  dec1,
                                  MOLONGLO_ARRAY_SLOPE,
                                  MOLONGLO_AZIMUTH_CORR,
                                  MOLONGLO_LATITUDE);

    projected_delay2 = doc_delay (hour_angle1,
                                  dec1,
                                  MOLONGLO_ARRAY_SLOPE,
                                  skew,
                                  MOLONGLO_LATITUDE);
    
    md1 = asin(projected_delay1);
    md2 = asin(projected_delay2);
    skews1[i] = (float) (md1 - md2) * 180 / M_PI;

    // next the erorrs for CJ0408
    projected_delay1 = doc_delay (hour_angle2,
                                  dec2,
                                  MOLONGLO_ARRAY_SLOPE,
                                  MOLONGLO_AZIMUTH_CORR,
                                  MOLONGLO_LATITUDE);

    projected_delay2 = doc_delay (hour_angle2,
                                  dec2,
                                  MOLONGLO_ARRAY_SLOPE,
                                  skew,
                                  MOLONGLO_LATITUDE);
    md1 = asin(projected_delay1);
    md2 = asin(projected_delay2);
    skews2[i] = (float) (md1 - md2) * 180 / M_PI;

    skews[i] = skews2[i] - skews1[i];

  }

  mopsr_skew_plot (nskews, xvals, skews1, skews2, skews);

  cpgclos();

  free (skews1);
  free (skews2);
  free (skews);

  return 0;
}


// simple PGPLOT of the delays across the array
void mopsr_skew_plot (unsigned nskew, float * xvals, float * skews1, float * skews2, float * skews)
{
  int iskew;
  float xmin = xvals[0];
  float xmax = xvals[nskew-1];
  float ymax = -FLT_MAX;
  float ymin = FLT_MAX;
  float yrange;
  float y;
  for (iskew=0; iskew <nskew; iskew++)
  {
    y = skews1[iskew];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    y = skews2[iskew];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    y = skews[iskew];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
  }

  yrange = (ymax - ymin);
  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  cpgbbuf();
  cpgeras();

  char title[60];
  sprintf (title, "Delays: time = %lf", time);

  cpgswin(xmin, xmax, (float) ymin, (float) ymax);
  cpgsvp(0.10, 0.90, 0.10, 0.90);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab ("Value of Skew", "MD Error (degrees)", title);

  cpgsci(2);
  cpgline (nskew, xvals, skews1);
  cpgsci(3);
  cpgline (nskew, xvals, skews2);
  cpgsci(1);
  cpgline (nskew, xvals, skews);

  cpgebuf();
}
