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

void mopsr_docjer_plot (unsigned nmod, float * jers, float * docs);

void usage ()
{
	fprintf(stdout, "mopsr_test_delays_jerdoc header_file\n"
    " -b file     bays config file [default $DADA_ROOT/share/molonglo_bays.txt\n"
    " -c nchan    number of channels\n" 
    " -d dec      delta declination to apply (degrees, default 1)\n" 
    " -i          disable instrumental delays\n"
    " -g          disable geometric delays\n"
    " -h          print this help text\n" 
    " -m file     modules config file [default $DADA_ROOT/share/molonglo_modules.txt\n"
    " -p imod     print delay and phase for imod for each timestep\n"
    " -v          verbose output\n\n" 
    "Plot the delay and phase as a function of time\n"
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  unsigned verbose = 0;

  char bays_file[512];
  char modules_file[512];
  bays_file[0] = '\0';
  modules_file [0] = '\0';

  char apply_instrumental = 1;
  char apply_geometric = 1;

  int print_module = -1;
  float delta_dec = 1;

  while ((arg = getopt(argc, argv, "b:d:ghim:p:v")) != -1) 
  {
    switch (arg)  
    {
      case 'b':
        if (optarg)
        {
          strcpy (bays_file, optarg);
          break;
        }
        else
        {
          fprintf (stderr, "-b requires argument\n");
          return (EXIT_FAILURE);
        }

      case 'd':
        delta_dec = atof (optarg);
        break;

      case 'g':
        apply_geometric = 0;
        break;

      case 'h':
        usage ();
        return 0;

      case 'i':
        apply_instrumental = 0;
        break;

      case 'm':
        if (optarg)
        {
          strcpy (modules_file, optarg);
          break;
        }
        else
        {
          fprintf (stderr, "-m requires argument\n");
          return (EXIT_FAILURE);
        }

      case 'p':
        print_module = atoi(optarg);
        break;

      case 'v':
        verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  char * share_dir = getenv("DADA_ROOT");
  if (strlen(bays_file) == 0)
  {
    if (!share_dir)
    {
      fprintf (stderr, "ERROR: DADA_ROOT environment variable must be defined\n");
      exit (EXIT_FAILURE);
    }
    sprintf (bays_file, "%s/share/molonglo_bays.txt", share_dir);
  }

  if (strlen(modules_file) == 0)
  {
    if (!share_dir)
    {
      fprintf (stderr, "ERROR: DADA_ROOT environment variable must be defined\n");
      exit (EXIT_FAILURE);
    }
    sprintf (modules_file, "%s/share/molonglo_modules.txt", share_dir);
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 argument is required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  unsigned ichan, imod;

  int nbay;
  mopsr_bay_t * all_bays = read_bays_file (bays_file, &nbay);
  if (!all_bays)
  {
    fprintf (stderr, "ERROR: failed to read bays file [%s]\n", bays_file);
    return EXIT_FAILURE;
  }

  // second argument is the molonglo modules file
  int nmod;
  mopsr_module_t * modules = read_modules_file (modules_file, &nmod);
  if (!modules)
  {
    fprintf (stderr, "ERROR: failed to read modules file [%s]\n", modules_file);
    return EXIT_FAILURE;
  }
  nmod = 352;

  float * jers = (float *) malloc (nmod * sizeof(float));
  float * docs = (float *) malloc (nmod * sizeof(float));

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

  // read source information from header
  mopsr_source_t source;

  if (ascii_header_get (header, "SOURCE", "%s", source.name) != 1)
  {
    fprintf (stderr, "ERROR:  could not read SOURCE from header\n");
    return -1;
  }

  char position[32];
  if (ascii_header_get (header, "RA", "%s", position) != 1)
  {
    fprintf (stderr, "ERROR:  could not read RA from header\n");
    return -1;
  }

  if (verbose)
    fprintf (stderr, " RA (HMS) = %s\n", position);
  if (mopsr_delays_hhmmss_to_rad (position, &(source.raj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse RA from %s\n", position);
    return -1;
  }
  if (verbose)
    fprintf (stderr, " RA (rad) = %lf\n", source.raj);

  if (ascii_header_get (header, "DEC", "%s", position) != 1)
  {
    fprintf (stderr, "ERROR:  could not read RA from header\n");
    return -1;
  }
  if (verbose)
    fprintf (stderr, " DEC (DMS) = %s\n", position);
  if (mopsr_delays_ddmmss_to_rad (position, &(source.decj)) < 0)
  {
    fprintf (stderr, "ERROR:  could not parse DEC from %s\n", position);
    return -1;
  }
  if (verbose)
    fprintf (stderr, " DEC (rad) = %lf\n", source.decj);

  float ut1_offset;
  if (ascii_header_get (header, "UT1_OFFSET", "%f", &ut1_offset) == 1)
  {
    fprintf (stderr, " UT1_OFFSET=%f\n", ut1_offset);
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
  timestamp.tv_usec = (long) (ut1_offset * 1000000);

  struct tm * utc = gmtime (&utc_start);
  cal_app_pos_iau (source.raj, source.decj, utc,
                   &(source.ra_curr), &(source.dec_curr));

  // extract required metadata from header
  int nchan;
  if (ascii_header_get (header, "NCHAN", "%d", &nchan) != 1)
  {
    fprintf (stderr, "ERROR:  could not read NCHAN from header\n");
    return -1;
  }

  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) != 1)
  {
    fprintf (stderr, "ERROR:  could not read BW from header\n");
    return -1;
  }

  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) != 1)
  {
    fprintf (stderr, "ERROR:  could not read FREQ from header\n");
    return -1;
  }

  int chan_offset;
  if (ascii_header_get (header, "CHAN_OFFSET", "%d", &chan_offset) != 1)
  {
    fprintf (stderr, "ERROR:  could not read CHAN_OFFSET from header\n");
    return -1;
  }

  double tsamp;
  if (ascii_header_get (header, "TSAMP", "%lf", &tsamp) != 1)
  {
    fprintf (stderr, "ERROR:  could not read TSAMP from header\n");
    return -1;
  }

  char is_tracking = 1;
  if (ascii_header_get (header, "OBSERVING_TYPE", "%s", tmp) == 1)
  {
    if (verbose)
      fprintf (stderr, "OBSERVING_TYPE=%s\n", tmp);
  }
  else
  {
    fprintf (stderr, "OBSERVING_TYPE not specified in header, assuming TRACKING\n");
  }
  if (strcmp(tmp, "TRANSITING") == 0)
  {
    is_tracking = 0;
  }

  cpgopen ("/xs");
  cpgask (0);

  // advance time by 1 millisecond each plot
  const double delta_time = 10;
  double obs_offset_seconds = 0;
  unsigned nant = 352;
  double jer_delay, md_angle;
  double doc_delay, doc_md_angle;

  fprintf (stderr, "TRACKING=%d APPLY_GEOMETRIC=%d APPLY_INSTRUMENTAL=%d\n", is_tracking,
           apply_geometric, apply_instrumental);

  if (verbose && print_module >= 0)
    fprintf (stderr, "imod=%d module=%s\n", print_module, modules[print_module].name);

  int more = 1;
  float start_md_angle = 0;

  double C = 2.99792458e8;
  const double cfreq = 843e6;
  const double fac = cfreq * 2 * M_PI;
  double delay, jer_md_angle1, jer_md_angle2, doc_md_angle1, doc_md_angle2, jer_phase1, jer_phase2, doc_phase1, doc_phase2;

  while ( more )
  {
    struct timeval timestamp;
    timestamp.tv_sec = floor(obs_offset_seconds);
    timestamp.tv_usec = (obs_offset_seconds - (double) timestamp.tv_sec) * 1000000;
    timestamp.tv_sec += utc_start;

    delay = calc_jer_delay (source.ra_curr, source.dec_curr, timestamp);
    jer_md_angle1 = asin(delay);

    delay = calc_jer_delay (source.ra_curr, source.dec_curr - delta_dec, timestamp);
    jer_md_angle2 = asin(delay);

    delay = calc_doc_delay (source.ra_curr, source.dec_curr, timestamp);
    doc_md_angle1 = asin(delay);

    delay = calc_doc_delay (source.ra_curr, source.dec_curr - delta_dec, timestamp);
    doc_md_angle2 = asin(delay);

    double geom_delay, phase;
    for (imod=0; imod<nmod; imod++)
    {
      double dist = 4.42 * imod;

      jer_phase1 = fac * jer_md_angle1 * dist / C;
      jer_phase2 = fac * jer_md_angle2 * dist / C;
      doc_phase1 = fac * doc_md_angle1 * dist / C;
      doc_phase2 = fac * doc_md_angle2 * dist / C;

      jers[imod] = (float) (jer_phase1-jer_phase2);
      docs[imod] = (float) (doc_phase1-doc_phase2);
    }

     mopsr_docjer_plot (nmod, jers, docs);

    obs_offset_seconds += delta_time;

    if (timestamp.tv_sec >= 1200)
      more = 0;
    usleep (100000);
  }

  cpgclos();

  free (modules);
  free (jers);
  free (docs);

  return 0;
}


// simple PGPLOT of the delays across the array
void mopsr_docjer_plot (unsigned nmod, float * jers, float * docs)
{
  float * xvals = (float *) malloc(sizeof(float) * nmod);
  float * diff = (float *) malloc(sizeof(float) * nmod);

  int imod;
  for (imod=0; imod<nmod; imod++)
    xvals[imod] = (float) imod;

  float xmin = (float) 0;
  float xmax = (float) nmod;
  float ymax = -FLT_MAX;
  float ymin = FLT_MAX;
  float yrange;
  float y;
  for (imod=0; imod <nmod; imod++)
  {
    y = jers[imod] - docs[imod];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    diff[imod] = y;
/*
    y = jers[imod];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;

    y = docs[imod];
    if (y > ymax) ymax = y;
    if (y < ymin) ymin = y;
    */
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
  cpgbox("BCST", 0.0, 0.0, "BCNSTV", 0.0, 0.0);
  cpglab ("", "JER vs DOC", title);

  cpgsci(2);
  cpgline (nmod, xvals, jers);
  cpgsci(3);
  cpgline (nmod, xvals, docs);
  cpgsci(1);
  cpgline (nmod, xvals, diff);

  cpgebuf();

  free (xvals);
  free (diff);
}
