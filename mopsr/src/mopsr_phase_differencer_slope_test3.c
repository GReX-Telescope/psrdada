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
#include "mopsr_delays_hires.h"

#include "config.h"

#ifdef HAVE_GSL
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
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
void mopsr_phase_delays_plot (float * dists, float * phases, unsigned npt, char * dec, char * device, double m_w, double c_w, double m_e, double c_e);

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
  int nmod;
  const unsigned nsources = 10;
  mopsr_module_t * phases[nsources];

  phases[0] = read_modules_file ("CJ1819-6345.delays.sorted", &nmod);
  phases[1] = read_modules_file ("CJ1424-4913.delays.sorted", &nmod);
  phases[2] = read_modules_file ("CJ1935-4620.delays.sorted", &nmod);
  phases[3] = read_modules_file ("CJ1935-4620_2.delays.sorted", &nmod);
  phases[4] = read_modules_file ("CJ1020-4251.delays.sorted", &nmod);
  phases[5] = read_modules_file ("CJ1020-4251_2.delays.sorted", &nmod);
  phases[6] = read_modules_file ("CJ1830-3602.delays.sorted", &nmod);
  phases[7] = read_modules_file ("CJ1154-3505.delays.sorted", &nmod);
  phases[8] = read_modules_file ("CJ1256-0547.delays.sorted", &nmod);
  phases[9] = read_modules_file ("3C273.delays.sorted", &nmod);

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
  {
    modules2[imod].dist += delta_distance;
    //fprintf (stderr, "modules[%d] name=%s dist=%f\n", imod, modules1[imod].name, modules1[imod].dist);
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

  mopsr_source_t sources[nsources];
  float grads_e[nsources];
  float grads_w[nsources];

  mopsr_delays_hhmmss_to_rad ("18:19:35.0", &(sources[0].raj));  // 2017-09-03-09:23:12
  mopsr_delays_ddmmss_to_rad ("-63:45:48.6", &(sources[0].decj));

  mopsr_delays_hhmmss_to_rad ("14:24:32.18", &(sources[1].raj));  // 2017-09-03-05:31:04
  mopsr_delays_ddmmss_to_rad ("-49:13:17.16", &(sources[1].decj)); // 2017-09-03-05:38:57

  mopsr_delays_hhmmss_to_rad ("19:35:57.2", &(sources[2].raj)); // 2017-09-03-10:26:36
  mopsr_delays_ddmmss_to_rad ("-46:20:43.1", &(sources[2].decj));

  mopsr_delays_hhmmss_to_rad ("19:35:57.2", &(sources[3].raj)); // 2017-09-04-10:32:32
  mopsr_delays_ddmmss_to_rad ("-46:20:43.1", &(sources[3].decj));

  mopsr_delays_hhmmss_to_rad ("10:20:03.5", &(sources[4].raj)); // 2017-09-04-01:28:28
  mopsr_delays_ddmmss_to_rad ("-42:51:33.0", &(sources[4].decj));

  mopsr_delays_hhmmss_to_rad ("10:20:03.5", &(sources[5].raj)); // 2017-09-05-01:15:08
  mopsr_delays_ddmmss_to_rad ("-42:51:33.0", &(sources[5].decj));

  mopsr_delays_hhmmss_to_rad ("18:30:58.8", &(sources[6].raj)); // 2017-09-04-09:22:48
  mopsr_delays_ddmmss_to_rad ("-36:02:30.3", &(sources[6].decj));

  mopsr_delays_hhmmss_to_rad ("11:54:21.9", &(sources[7].raj)); // 2017-09-04-02:52:52
  mopsr_delays_ddmmss_to_rad ("-35:05:32.2", &(sources[7].decj));

  mopsr_delays_hhmmss_to_rad ("12:56:11.15", &(sources[8].raj)); // 2017-09-04-03:54:40
  mopsr_delays_ddmmss_to_rad ("-05:47:20.1", &(sources[8].decj));

  mopsr_delays_hhmmss_to_rad ("12:29:06.7", &(sources[9].raj)); // 2017-09-04-03:28:12
  mopsr_delays_ddmmss_to_rad ("02:03:09.0", &(sources[9].decj));

  char * devices[] = { "1/xs", "2/xs", "3/xs", "4/xs", "5/xs", "6/xs", "7/xs", "8/xs", "9/xs", "10/xs"};
  char * titles[] = { "-63", "-49", "-46", "-46", "-42", "-42", "-36", "-35", "02", "-05" };

  // rough transit time for the source positions
  time_t utc_starts[nsources];
  utc_starts[0] = str2utctime("2017-09-01-09:42:10");
  utc_starts[1] = str2utctime("2017-09-03-05:38:57");
  utc_starts[2] = str2utctime("2017-09-03-10:49:36");
  utc_starts[3] = str2utctime("2017-09-04-10:45:38");
  utc_starts[4] = str2utctime("2017-09-04-01:30:41");
  utc_starts[5] = str2utctime("2017-09-05-01:26:45");
  utc_starts[6] = str2utctime("2017-09-04-09:40:38");
  utc_starts[7] = str2utctime("2017-09-04-03:04:48");
  utc_starts[8] = str2utctime("2017-09-04-04:06:10");
  utc_starts[9] = str2utctime("2017-09-04-03:39:06");

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

  // now calculate the apparent RA and DEC for the current timestamp
  struct timeval timestamps[nsources];
  for (i=0; i<nsources; i++)
  {
    fprintf (stderr, "utc_starts[%d]=%ld\n",i, utc_starts[i]);
    struct tm * utc = gmtime (utc_starts + i);
    cal_app_pos_iau (sources[i].raj, sources[i].decj, utc, &(sources[i].ra_curr), &(sources[i].dec_curr));
    fprintf (stderr, "sources[%d] (%lf, %lf) -> (%lf, %lf)\n", i, sources[i].raj, sources[i].decj, sources[i].ra_curr, sources[i].dec_curr);

    timestamps[i].tv_sec = utc_starts[i];
    timestamps[i].tv_usec = 0;
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


  mopsr_delay_hires_t ** delays1 = (mopsr_delay_hires_t **) malloc(sizeof(mopsr_delay_hires_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays1[imod] = (mopsr_delay_hires_t *) malloc (sizeof(mopsr_delay_hires_t) * nchan);

  mopsr_delay_hires_t ** delays2 = (mopsr_delay_hires_t **) malloc(sizeof(mopsr_delay_hires_t *) * nmod);
  for (imod=0; imod<nmod; imod++)
    delays2[imod] = (mopsr_delay_hires_t *) malloc (sizeof(mopsr_delay_hires_t) * nchan);

  cpgopen (device);
  cpgask (0);

  char apply_instrumental = 0;
  char apply_geometric = 1;
  char is_tracking1 = 0;
  char is_tracking2 = 0;

  double tsamp = 10.24;
  unsigned nant = nmod;

  unsigned npts = 100;
  unsigned ipt = 0;
  float * phase_error = (float *) malloc (sizeof(float) * npts);
  float * baseline_phase_error = (float *) malloc(sizeof(float) * nmod);

  fprintf (stderr, "freq1=%lf freq2=%lf delta_freq=%le\n", channels1[0].cfreq, channels2[0].cfreq, channels1[0].cfreq - channels2[0].cfreq);

  double slope_error = 0.005 * MOLONGLO_ARRAY_SLOPE;
  double slope = MOLONGLO_ARRAY_SLOPE - slope_error;
  double slope_step = slope_error * 2 / npts;

  float results[nsources][352];
  float correction[352];

  float dists[352];
  double x_e[176];
  double y_e[176];
  double w_e[176];
  double x_w[176];
  double y_w[176];
  double w_w[176];
  double c0_e, c1_e, c0_w, c1_w, cov00, cov01, cov11, chisq;

  float best_e = 1e10;
  float best_w = 1e10;

  for (i=0; i<nmod; i++)
  {
    dists[i] = (float) modules1[i].dist;
  }

  while ( ipt < npts )
  {
    slope += slope_step;

    start_md_angle2 = start_md_angle + (float) delta_md_angle;

    // calculate the effect of this slope error on 3C273, which would have been absorbed into the calibration
    if (calculate_delays_hires_slope (nbay, bays1, nant, modules1, nchan, channels1,
                          sources[0], timestamps[0], delays1, start_md_angle, 
                          apply_instrumental, apply_geometric, 
                          is_tracking1, tsamp, MOLONGLO_ARRAY_SLOPE) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    if (calculate_delays_hires_slope (nbay, bays2, nant, modules2, nchan, channels2,
                          sources[0], timestamps[0], delays2, start_md_angle2, 
                          apply_instrumental, apply_geometric, 
                          is_tracking2, tsamp, slope) < 0)
    {
      fprintf (stderr, "failed to update delays\n");
      return -1;
    }

    // correction is the additional phase change from the trial vs the original
    for (imod=0; imod<nmod; imod++)
    {
      float fringe1 = delays1[imod]->fringe_coeff;
      float fringe2 = delays2[imod]->fringe_coeff;
      correction[imod] = fringe2 - fringe1;
    }

    mopsr_phase_delays_plot (dists, correction, 352, "Slope Error", "50/xs", 0, 0, 0, 0);

    fprintf (stderr, "slope[%d]=1/%f delta=%f", ipt, 1.0/slope, MOLONGLO_ARRAY_SLOPE - slope);

    float sum_e = 0;
    float sum_w = 0;

    gsl_multifit_robust_workspace *work = gsl_multifit_robust_alloc (gsl_multifit_robust_bisquare, X_e->size1, X_e->size2);
    gsl_multifit_robust_workspace *work = gsl_multifit_robust_alloc (gsl_multifit_robust_bisquare, X_w->size1, X_w->size2);

    gsl_matrix *X_e, *X_w, *cov_e, *cov_w;
    gsl_vector *x_e, *y_e, *x_w, *y_w, *c_e, *c_w;
    X = gsl_matrix_alloc (176, 2); // 2 <- linear fit (no. of params)
    // now subtract this "absorbed correction for each source
    for (i=0; i<nsources; i++)
    {
      size_t nvalid_e = 0;
      size_t nvalid_w = 0;
      calculate_delays_hires_slope (nbay, bays1, nant, modules1, nchan, channels1,
                                    sources[i], timestamps[i], delays1, start_md_angle,
                                    apply_instrumental, apply_geometric,
                                    is_tracking1, tsamp, MOLONGLO_ARRAY_SLOPE);

      calculate_delays_hires_slope (nbay, bays2, nant, modules2, nchan, channels2,
                                    sources[i], timestamps[i], delays2, start_md_angle,
                                    apply_instrumental, apply_geometric,
                                    is_tracking2, tsamp, slope);
      for (imod=0; imod<nmod; imod++)
      {
        float fringe1 = delays1[imod]->fringe_coeff;
        float fringe2 = delays2[imod]->fringe_coeff;
        float difference = fringe2 - fringe1;

        // TODO just testing
        results[i][imod] = phases[i][imod].phase_offset;

        // now we substract this difference in phase offet to the measured phase offset 
        //results[i][imod] = phases[i][imod].phase_offset - difference;

        // and also remove the phase offset that would have been absorbed into the calibration observation
        //results[i][imod] -= correction[imod];

        if (phases[i][imod].scale > 0.6)
        {
          // -ve east, +ve west
          if (modules1[imod].dist > 0)
          {
            x_w[nvalid_w] = (double) modules1[imod].dist;
            y_w[nvalid_w] = (double) results[i][imod];
            w_w[nvalid_w] = (double) modules1[imod].scale;
            nvalid_w ++;
          }
          else
          {
            x_e[nvalid_e] = (double) modules1[imod].dist;
            y_e[nvalid_e] = (double) results[i][imod];
            w_e[nvalid_e] = (double) modules1[imod].scale;
            nvalid_e ++;
          }
          //fprintf (stderr, "modules1[imod].name=%s phases[i][imod].name=%s, dist=%f\n", modules1[imod].name, phases[i][imod].name, modules1[imod].dist);
        }
      }

      //fprintf (stderr, "[%d] nvalid=%d\n", i, nvalid);
      // design matrix (and copy data to gsl containers):
      for (unsigned iel = 0; iel < 176; ++iel) {
        gsl_matrix_set (X_e, iel, 0, 1.0);
        gsl_matrix_set (X_e, iel, 1, x_e[iel]);
        gsl_matrix_set (X_w, iel, 0, 1.0);
        gsl_matrix_set (X_w, iel, 1, x_w[iel]);

        gsl_vector_set (x_e_gsl, iel, x_w[iel]);
        gsl_vector_set (y_e_gsl, iel, y_w[iel]);
        gsl_vector_set (x_w_gsl, iel, x_w[iel]);
        gsl_vector_set (y_w_gsl, iel, y_w[iel]);
      }
      int s_e = gsl_multifit_robust(X_e, y_e, c_e, cov_e, work);
      int s_w = gsl_multifit_robust(X_w, y_w, c_w, cov_w, work);

      mopsr_phase_delays_plot (dists, results[i], 352, titles[i], devices[i], c_w[1], c_w[0], c_e[1], c_e[0]);

      grads_e[i] = (float) c1_e;
      grads_w[i] = (float) c1_w;

      sum_e += grads_e[i];
      sum_w += grads_w[i];

      //fprintf (stderr, "[%8.7f %8.7f]", grads_w[i], grads_e[i]);
    }
    gsl_multifit_robust_free (work);

    fprintf (stderr, " west=%8.7f east=%8.7f\n", sum_w, sum_e);

    //mopsr_phase_delays_plot (results[1], 352, "-65:45:08.5", "2/xs");
    //mopsr_phase_delays_plot (results[2], 352, "-75:07:20.0", "3/xs");
    //mopsr_phase_delays_plot (results[3], 352, "-43:33:09.0", "4/xs");
    //mopsr_phase_delays_plot (results[4], 352, "-46:20:43.1", "5/xs");
    //mopsr_phase_delays_plot (results[5], 352, "-31:44:14.7", "5/xs");

    //usleep (100000);
    ipt ++;
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

void mopsr_phase_delays_plot (float * dists, float * phases, unsigned npts, char * dec, char * device, double m_w, double c_w, double m_e, double c_e)
{
  float xmin = -800; 
  float xmax = 800;

  float ymin = -1 * M_PI;
  float ymax =  1 * M_PI;
  float factor;

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (phases[i] > M_PI)
      while (phases[i] > M_PI)
        phases[i] -= (2 * M_PI);

    if (phases[i] < -M_PI)
      while (phases[i] < -M_PI)
        phases[i] += (2 * M_PI);
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
  cpglab("Module", "Differential Phase", dec);

  cpgsci(1);
  cpgslw(5);
  cpgpt(npts, dists, phases, -1);
  cpgslw(1);

  float xxx[2];
  float yyy[2];

  float M, C;
  M = (float) m_w;
  C = (float) c_w;

  xxx[0] = 0;
  xxx[1] = xmax;

  yyy[0] = (M * xxx[0]) + C;
  yyy[1] = (M * xxx[1]) + C;

  cpgsci(2);
  cpgline (2, xxx, yyy);
  cpgsci(1);

  xxx[0] = xmin;
  xxx[1] = 0;

  M = (float) m_e;
  C = (float) c_e;

  yyy[0] = (M * xxx[0]) + C;
  yyy[1] = (M * xxx[1]) + C;
    
  cpgsci(3);
  cpgline (2, xxx, yyy);
  cpgsci(1);


  cpgebuf();
  cpgend();

}
