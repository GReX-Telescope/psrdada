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

#include "mopsr_delays.h"

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
void mopsr_baseline_phase_delays_plot (float * phases, unsigned nmod, unsigned mod, mopsr_module_t * mods, char * device);

void usage ()
{
	fprintf(stdout, "mopsr_phase_diff_transit bays_file modules_file obs.header\n"
    " -D dev      use pgplot devices [default /xs]\n" 
    " -d nsecs    time offset to add to UTC_START [default 0s]\n" 
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

  char * device = "/xs";

  int frank_nmod = 0;

  int mod = 0;

  while ((arg = getopt(argc, argv, "D:d:f:F:hl:m:t:v")) != -1) 
  {
    switch (arg)  
    {
      case 'D':
        device = strdup (optarg);
        break; 

      case 'd':
        delta_time_plot = atof (optarg);
        break;
      
      case 'f':
        delta_freq = atof (optarg);
        break;

      case 'F':
        frank_nmod = atoi (optarg);
        break;

      case 'l':
        delta_distance = atof (optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

      case 'm':
        mod = atoi (optarg);
        break;

      case 't':
        delta_time = atof (optarg);
        break;
      
      case 'v':
        verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

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
      
  // read source information from header
  mopsr_source_t source;
  //source.hour_angle = 25;

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

  float ut1_offset = 0;
  if (ascii_header_get (header, "UT1_OFFSET", "%f", &ut1_offset) == 1)
  {
    if (verbose)
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
  timestamp.tv_usec = 0;

  struct tm * utc = gmtime (&utc_start);
  //double rob, dob, sha;
  //fprintf (stderr, "main: calc_observed_pos()\n");
  //int rval = calc_observed_pos (source.raj, source.decj, utc, (double) ut1_offset, &rob, &dob, &sha);

  double rap, dap;
  cal_app_pos_iau (source.raj, source.decj, utc, &rap, &dap);

//  fprintf (stderr, "main: calc_app_position()\n");
//  calc_app_position (source.raj, source.decj, timestamp,
//                    &(source.ra_curr), &(source.dec_curr));

  if (verbose)
  {
    fprintf (stderr, "Apparent (current) positions:\n");

    int HMSF[4];
    int DMSF[4];
    char sign;
    int NDP = 4;    // number of decimal places

    /*
    iauA2tf (NDP, source.ra_curr, &sign, HMSF);
    iauA2af(NDP, source.dec_curr, &sign, DMSF)
    
    fprintf (stderr, "  RA: %d:%d:%d.%d, DEC: %c%d:%d:%d.%d\n",
            HMSF[0],HMSF[1],HMSF[2],HMSF[3],
            sign,
            DMSF[0], DMSF[1],DMSF[2],DMSF[3]);

    */
    iauA2tf (NDP, rap, &sign, HMSF);
    iauA2af (NDP, dap, &sign, DMSF);
    fprintf (stderr, "  RA: %d:%d:%d.%d, DEC: %c%d:%d:%d.%d\n",
            HMSF[0],HMSF[1],HMSF[2],HMSF[3],
            sign,
            DMSF[0], DMSF[1],DMSF[2],DMSF[3]);
  }

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

  // configure the channels
  mopsr_chan_t * channels1 = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t));
  channels1[0].number = 0;
  channels1[0].bw     = bw;
  channels1[0].cfreq  = freq;

  mopsr_chan_t * channels2 = (mopsr_chan_t *) malloc(sizeof(mopsr_chan_t));
  channels2[0].number = 0;
  channels2[0].bw     = bw;
  channels2[0].cfreq  = freq + delta_freq;

  mopsr_delay_t ** delays1 = (mopsr_delay_t **) malloc(sizeof(mopsr_delay_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays1[imod] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);

  mopsr_delay_t ** delays2 = (mopsr_delay_t **) malloc(sizeof(mopsr_delay_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays2[imod] = (mopsr_delay_t *) malloc (sizeof(mopsr_delay_t) * nchan);

  cpgopen (device);
  cpgask (0);

  char apply_instrumental = 1;
  char apply_geometric = 1;
  char is_tracking = 0;

  double tsamp = 1.28;
  unsigned nant = nmod;

  ut1_offset = 0;
  //double ut1_time  = (double) utc_start + (double) ut1_offset + (double) delta_time_plot;
  double ut1_time  = (double) utc_start + (double) ut1_offset;
  double ut1_time1 = ut1_time;
  double ut1_time2 = ut1_time + (double) delta_time;

  //double obs_offset1 = (double) ut1_offset + (double) delta_time_plot;
  //double obs_offset2 = (double) ut1_offset + (double) delta_time_plot + (double) delta_time;

  float start_md_angle = 0;

  struct timeval timestamp1;
  struct timeval timestamp2;

  float * baseline_phase_error = (float *) malloc(sizeof(float) * nmod);

  //fprintf (stderr, "freq1=%lf freq2=%lf delta_freq=%le\n", channels1[0].cfreq, channels2[0].cfreq, channels1[0].cfreq - channels2[0].cfreq);

  {
    timestamp1.tv_sec = floor (ut1_time1);
    timestamp2.tv_sec = floor (ut1_time2);

    timestamp1.tv_usec = (long) ((ut1_time1 - (double) timestamp1.tv_sec) * 1000000);
    timestamp2.tv_usec = (long) ((ut1_time2 - (double) timestamp2.tv_sec) * 1000000);

    if (verbose)
      fprintf (stderr, "t1=%lf t2=%lf\n", ut1_time1, ut1_time2);

    if (calculate_delays (nbay, bays1, nant, modules1, nchan, channels1,
                          source, timestamp1, delays1, start_md_angle, 
                          apply_instrumental, apply_geometric, is_tracking, tsamp) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    //source.hour_angle = sha;
    //source.ra_curr = rap;
    //source.dec_curr = dap;

    if (calculate_delays (nbay, bays2, nant, modules2, nchan, channels2,
                          source, timestamp2, delays2, start_md_angle, 
                          apply_instrumental, apply_geometric, is_tracking, tsamp) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    // for the specified antenna, compute the phase difference between 
    // the 2 regeimes
    double fringe1 = delays1[mod]->fringe_coeff;
    double fringe2 = delays2[mod]->fringe_coeff;
    double diff_phase = fringe1 - fringe2;

    /*
    fprintf (stderr, "fdiff=%le\n", fringe1 - fringe2);

    complex double phasor1 = cos(fringe1) + sin(fringe1) * I;
    complex double phasor2 = cos(fringe2) + sin(fringe2) * I;
    complex double diff_phasor = phasor2 - phasor1;

    fprintf (stderr, "(%lf, %lf) - (%lf, %lf) == (%le, %le) \n", creal(phasor1), cimag(phasor1), creal(phasor2), cimag(phasor2), creal(diff_phasor), cimag(diff_phasor));

    double diff_phase = atan2(cimag(diff_phasor), creal(diff_phasor));
*/
    // compute the phase error for the module in question

    // compute the phase offset for each baseline against the reference module
    for (imod=0; imod<nmod; imod++)
    {
      fringe1 = delays1[imod]->fringe_coeff;
      fringe2 = delays2[imod]->fringe_coeff;
      diff_phase = fringe1 - fringe2;
      baseline_phase_error[imod] = (float) diff_phase;
      //fprintf (stderr, "f1=%le f2=%le diff_phase=%le error=%f\n", fringe1, fringe2, diff_phase, baseline_phase_error[imod]);
    }
    
    mopsr_baseline_phase_delays_plot (baseline_phase_error, nmod, mod, modules1, device);

    fprintf (stderr, "%ld\t%f\n", timestamp1.tv_sec, baseline_phase_error[nmod-1]);

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
