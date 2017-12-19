/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

//
// Compute the differential phase as a function of time for the given
// delta from the mopsr calculate_delays routine
//

#include "mopsr_def.h"
#ifdef HIRES
#include "mopsr_delays_hires.h"
#else
#include "mopsr_delays.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <cpgplot.h>
#include <complex.h>
#include <float.h>

void usage();
void mopsr_phase_delays_plot (float * phases, unsigned npt, double obs_offset1, char * device);
void mopsr_baseline_phase_delays_plot (float * phases, unsigned nmod, unsigned mod, mopsr_module_t * mods, char * device);

void usage ()
{
	fprintf(stdout, "mopsr_phase_difference_slope bays_file modules_file obs.header\n"
    " -D dev      use pgplot devices [default /xs]\n" 
    " -a angle    apply md angle offset in degrees [default 0]\n" 
    " -d nsecs    plot time unit [default 10s]\n" 
    " -l metres   apply metres delta baseline difference\n" 
    " -h          print this help text\n" 
    " -F nmod     apply Frank term for nmod offset\n"
    " -f nMhz     apply nMhz delta centre frequnecy delay\n"
    " -m mod      module to compute phase offsets for [default 0]\n"
    " -t nsecs    apply nsecs delta time delay\n"
    " -v          verbose output\n" 
  );
}

int main(int argc, char** argv) 
{
  int arg = 0;

  char verbose = 0;

  double delta_time = 0;

  double delta_time_plot = 10;

  double delta_freq = 0;

  double delta_distance = 0;

  double delta_md_angle = 0;

  char * device = "/xs";

  int frank_nmod = 0;

  int mod = 0;

  while ((arg = getopt(argc, argv, "a:D:d:f:F:hl:m:t:v")) != -1) 
  {
    switch (arg)  
    {
      case 'D':
        device = strdup (optarg);
        break; 

      case 'a':
        delta_md_angle = (double) atof (optarg);
        delta_md_angle *= (M_PI / 180.0);
        break;

      case 'd':
        delta_time_plot = (double) atof (optarg);
        break;
      
      case 'f':
        delta_freq = (double) atof (optarg);
        break;

      case 'F':
        frank_nmod = atoi (optarg);
        break;

      case 'l':
        delta_distance = (double) atof (optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

      case 'm':
        mod = atoi (optarg);
        break;

      case 't':
        delta_time = (double) atof (optarg);
        break;
      
      case 'v':
        verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  fprintf (stderr, "delta_md_angle=%le radians\n", delta_md_angle);
  fprintf (stderr, "delta_freq=%le\n", delta_freq);
  fprintf (stderr, "delta_distance=%le\n", delta_distance);
  fprintf (stderr, "frank_nmod=%d\n", frank_nmod);
  fprintf (stderr, "delta_time=%f\n", delta_time);

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }


  // read the calibration results into module_t 
  mopsr_module_t * 3c273 = read_modules_file ("3C273.delays.sorted", &nmod);
  mopsr_module_t * CJ0408_6545 = read_modules_file ("CJ0408-6545.delays.sorted", &nmod);
  mopsr_module_t * CJ0408_7507 = read_modules_file ("CJ0408-7507.delays.sorted", &nmod);
  mopsr_module_t * CJ0440_4333 = read_modules_file ("CJ0440-4333.delays.sorted", &nmod);
  mopsr_module_t * CJ1018_3144 = read_modules_file ("CJ1018-3144.delays.sorted", &nmod);
  mopsr_module_t * CJ1935_4620 = read_modules_file ("CJ1935-4620.delays.sorted", &nmod);

  unsigned ichan, imod;

  char * bays_file = strdup (argv[optind+0]);
  int nbay;
  mopsr_bay_t * bays1 = read_bays_file (bays_file, &nbay);
  if (!bays1)
  {
    fprintf (stderr, "ERROR: failed to read bays file [%s] into bays1\n", bays_file);
    return EXIT_FAILURE;
  }

  mopsr_bay_t * bays2 = read_bays_file (bays_file, &nbay);
  if (!bays2)
  {
    fprintf (stderr, "ERROR: failed to read bays file [%s] into bays2\n", bays_file);
    return EXIT_FAILURE;
  }

  unsigned i;
  for (i=0; i<nbay; i++)
  {
    bays2[i].dist += delta_distance;
  }

  char * modules_file = strdup (argv[optind+1]);
  int nmod;
  mopsr_module_t * modules1 = read_modules_file (modules_file, &nmod);
  if (!modules1)
  {
    fprintf (stderr, "ERROR: failed to read modules file [%s]\n", modules_file);
    return EXIT_FAILURE;
  }

  mopsr_module_t * modules2 = read_modules_file (modules_file, &nmod);
  if (!modules2)
  {
    fprintf (stderr, "ERROR: failed to read modules file [%s]\n", modules_file);
    return EXIT_FAILURE;
  }

  fprintf (stderr, "nmod=%d mod=%d module=%s\n", nmod, mod, modules1[mod].name);

  for (imod=0; imod<nmod; imod++)
    modules2[imod].dist += delta_distance;

  // apply frank module delay (if set)
  for (imod=0; imod<nmod; imod++)
    modules2[imod].bay_idx += frank_nmod;

  // read the header
  char * header_file = strdup (argv[optind+2]);
  size_t header_size = 4096;
  char * header = (char *) malloc (header_size);

  if (fileread (header_file, header, header_size) < 0)
  {
    fprintf (stderr, "ERROR: could not read header from %s\n", header_file);
    return EXIT_FAILURE;
  }

  const unsigned nsources = 6;
  mopsr_source_t sources[nsources];

  mopsr_delays_hhmmss_to_rad ("12:29:06.7", &(sources[0].raj));
  mopsr_delays_ddmmss_to_rad ("02:03:09.0", &(sources[0].raj));

  mopsr_delays_hhmmss_to_rad ("04:08:20.3", &(sources[1].raj));
  mopsr_delays_ddmmss_to_rad ("-65:45:08.5", &(sources[1].raj));

  mopsr_delays_hhmmss_to_rad ("04:08:48.5", &(sources[2].raj));
  mopsr_delays_ddmmss_to_rad ("-75:07:20.0", &(sources[2].raj));

  mopsr_delays_hhmmss_to_rad ("04:40:17.07", &(sources[3].raj));
  mopsr_delays_ddmmss_to_rad ("-43:33:09.0", &(sources[3].raj));

  mopsr_delays_hhmmss_to_rad ("10:18:09.19", &(sources[4].raj));
  mopsr_delays_ddmmss_to_rad ("-31:44:14.7", &(sources[4].raj));

  mopsr_delays_hhmmss_to_rad ("19:35:57.2", &(sources[5].raj));
  mopsr_delays_ddmmss_to_rad ("-46:20:43.1", &(sources[5].raj));

  float ut1_offset;
  if (ascii_header_get (header, "UT1_OFFSET", "%f", &ut1_offset) == 1)
  {
    fprintf (stderr, " UT1_OFFSET=%f\n", ut1_offset);
  }
  else 
    ut1_offset = 0;
  if (verbose)
    fprintf (stderr, "UT1_OFFSET=%f\n", ut1_offset);

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
  for (i=0; i<nsources; i++)
    cal_app_pos_iau (sources[i].raj, sources[i].decj, utc, &(sources[i].ra_curr), &(sources[i].dec_curr));

  // extract required metadata from header
  int nchan;
  if (ascii_header_get (header, "NCHAN", "%d", &(nchan)) != 1)
  {
    fprintf (stderr, "ERROR:  could not read NCHAN from header\n");
    return -1;
  }
  if (nchan != 1)
  {  
    fprintf (stderr, "ERROR: cannot support headers of NCHAN != 1\n");
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
    fprintf (stderr, "ERROR:  could not read BW from header\n");
    return -1;
  }

  float start_md_angle;
  float start_md_angle2;
  if (ascii_header_get (header, "MD_ANGLE", "%f", &start_md_angle) != 1)
  {
    fprintf (stderr, "ERROR:  could not read MD_ANGLE from header\n");
    return -1;
  }
  start_md_angle *= (M_PI / 180.0);

  fprintf (stderr, "start MD angle=%f\n", start_md_angle);

  // configure the channels
  mopsr_chan_t * channels1 = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t));
  channels1[0].number = 0;
  channels1[0].bw     = bw;
  channels1[0].cfreq  = freq;

  mopsr_chan_t * channels2 = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t));
  channels2[0].number = 0;
  channels2[0].bw     = bw;
  channels2[0].cfreq  = freq + delta_freq;

#ifdef HIRES
  mopsr_delay_hires_t ** delays1 = (mopsr_delay_hires_t **) malloc(sizeof(mopsr_delay_hires_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays1[imod] = (mopsr_delay_hires_t *) malloc (sizeof(mopsr_delay_hires_t) * nchan);

  mopsr_delay_hires_t ** delays2 = (mopsr_delay_hires_t **) malloc(sizeof(mopsr_delay_hires_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays2[imod] = (mopsr_delay_hires_t *) malloc (sizeof(mopsr_delay_hires_t) * nchan);
#else
  mopsr_delay_t ** delays1 = (mopsr_delay_t **) malloc(sizeof(mopsr_delay_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays1[imod] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);

  mopsr_delay_t ** delays2 = (mopsr_delay_t **) malloc(sizeof(mopsr_delay_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays2[imod] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);
#endif

  cpgopen (device);
  cpgask (0);

  char apply_instrumental = 0;
  char apply_geometric = 1;
  char is_tracking1 = 0;
  char is_tracking2 = 0;

  double tsamp = 1.28;
  unsigned nant = nmod;

  double ut1_time = (double) utc_start + (double) ut1_offset;

  double ut1_time1 = ut1_time;
  double ut1_time2 = ut1_time + (double) delta_time;

  struct timeval timestamp1;
  struct timeval timestamp2;

  unsigned npts = 100;
  unsigned ipt = 0;
  float * phase_error = (float *) malloc (sizeof(float) * npts);
  float * baseline_phase_error = (float *) malloc(sizeof(float) * nmod);

  fprintf (stderr, "freq1=%lf freq2=%lf delta_freq=%le\n", channels1[0].cfreq, channels2[0].cfreq, channels1[0].cfreq - channels2[0].cfreq);

  double slope_error = 0.01 * MOLONGLO_ARRAY_SLOPE;
  double slope = MOLONGLO_ARRAY_SLOPE - slope_error;
  double slope_step = slope_error * 2 / npts;

  timestamp1.tv_sec = floor (ut1_time1);
  timestamp2.tv_sec = floor (ut1_time2);
  timestamp1.tv_usec = (long) ((ut1_time1 - (double) timestamp1.tv_sec) * 1000000);
  timestamp2.tv_usec = (long) ((ut1_time2 - (double) timestamp2.tv_sec) * 1000000);

  float results[nsources][352];

  while ( ipt < npts )
  {
    slope += slope_step;

    start_md_angle2 = start_md_angle + (float) delta_md_angle;

    // calculate the effect of this slope error on 3C273, which would have been absorbed into the calibration
    if (calculate_delays_hires_slope (nbay, bays1, nant, modules1, nchan, channels1,
                          sources[0], timestamp1, delays1, start_md_angle, 
                          apply_instrumental, apply_geometric, 
                          is_tracking1, tsamp, MOLONGLO_ARRAY_SLOPE
                         ) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    if (calculate_delays_hires_slope (nbay, bays2, nant, modules2, nchan, channels2,
                          sources[0], timestamp2, delays2, start_md_angle2, 
                          apply_instrumental, apply_geometric, 
                          is_tracking2, tsamp, slope) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    for (imod=0; imod<nmod; imod++)
    {
      fringe1 = delays1[imod]->fringe_coeff;
      fringe2 = delays2[imod]->fringe_coeff;
      correction[imod] = fringe1 - fringe2;
    }

    // now subtract this "absorbed correction for each source
    for (i=0; i<nsources; i++)
    {
      calculate_delays_hires_slope (nbay, bays1, nant, modules1, nchan, channels1,
                                    sources[i], timestamp1, delays1, start_md_angle,
                                    apply_instrumental, apply_geometric,
                                    is_tracking1, tsamp, MOLONGLO_ARRAY_SLOPE);
      calculate_delays_hires_slope (nbay, bays1, nant, modules1, nchan, channels1,
                                    sources[i], timestamp1, delays1, start_md_angle,
                                    apply_instrumental, apply_geometric,
                                    is_tracking1, tsamp, slope);
      for (imod=0; imod<nmod; imod++)
      {     
        fringe1 = delays1[imod]->fringe_coeff;
        fringe2 = delays2[imod]->fringe_coeff;
        results[i][imod] = fringe1 - fringe2;
        results[i][imod] -= correction[imod];
      }
    }

    mopsr_phase_delays_plot (results[0], 352, (ut1_time1 - (double) utc_start), "1/xs");
    mopsr_phase_delays_plot (results[1], 352, (ut1_time1 - (double) utc_start), "2/xs");
    mopsr_phase_delays_plot (results[2], 352, (ut1_time1 - (double) utc_start), "3/xs");
    mopsr_phase_delays_plot (results[3], 352, (ut1_time1 - (double) utc_start), "4/xs");
    mopsr_phase_delays_plot (results[4], 352, (ut1_time1 - (double) utc_start), "5/xs");
    mopsr_phase_delays_plot (results[5], 352, (ut1_time1 - (double) utc_start), "6/xs");

    usleep (100000);
  }

  //cpgclos();

  free (channels1);
  free (channels2);
  free (bays1);
  free (bays2);
  free (modules1);
  free (modules2);

  for (imod=0; imod<nmod; imod++)
  {
    free (delays1[imod]);
    free (delays2[imod]);
  }
  free (delays1);
  free (delays2);

  return 0;
}

void mopsr_phase_delays_plot (float * phases, unsigned npts, double obs_offset, char * device)
{
  float xmin = 0;
  float xmax = (float) npts;
  float ymin = -1 * M_PI;
  float ymax =  1 * M_PI;
  float factor;

  float xvals[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    xvals[i] = ((float) i / (float) (npts - 1)) * obs_offset;

    if (phases[i] > M_PI)
      while (phases[i] > M_PI)
        phases[i] -= (2 * M_PI);

    if (phases[i] < -M_PI)
      while (phases[i] < -M_PI)
        phases[i] += (2 * M_PI);

    /*
    factor = phases[i] / M_PI;
    if (factor > 1)
      phases[i] -= (floorf(factor/2) * 2 * M_PI);

    if (factor < -1)
      phases[i] -= (floorf(factor/2) * 2 * M_PI);
    */

    if (phases[i] > ymax)
      ymax = phases[i];
    if (phases[i] < ymin)
      ymin = phases[i];
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin (xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Differential Phase", "");

  cpgsci(3);
  cpgslw(5);
  cpgpt(npts, xvals, phases, -1);
  cpgslw(1);

  cpgebuf();
  cpgend();

}

void mopsr_baseline_phase_delays_plot (float * phases, unsigned nmod, unsigned mod, mopsr_module_t * mods, char * device)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin = -1 * M_PI;
  float ymax =  1 * M_PI;

  float xvals[nmod];
  float factor;

  unsigned imod;
  for (imod=0; imod<nmod; imod++)
  {
    xvals[imod] = (mods[imod].dist - mods[mod].dist);
    //xvals[imod] = mods[imod].dist;
    //xvals[imod] =(float) imod;

    if (xvals[imod] > xmax)
      xmax = xvals[imod];
    if (xvals[imod] < xmin)
      xmin = xvals[imod];

    if (phases[imod] > M_PI)
      while (phases[imod] > M_PI)
        phases[imod] -= (2 * M_PI);

    if (phases[imod] < -M_PI)
      while (phases[imod] < -M_PI)
        phases[imod] += (2 * M_PI);

    if (phases[imod] > ymax)
      ymax = phases[imod];
    if (phases[imod] < ymin)
      ymin = phases[imod];
  }
  //fprintf(stderr, "yrange=(%f, %f)\n", ymin, ymax);
  //sleep(1);

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();


  cpgswin (xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Baseline Length", "Differential Phase [radians]", "");

  cpgsci(3);
  cpgslw(5);
  cpgpt(nmod, xvals, phases, -1);
  cpgslw(1);

  cpgebuf();
  cpgend();

}
