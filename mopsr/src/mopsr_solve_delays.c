/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <float.h>
#include <complex.h>
#include <math.h>
#include <cpgplot.h>
#include <fftw3.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();

float get_mean (float * data, unsigned npt);
float calculate_snr (float * data, unsigned npt);
int furtherest_from_mean (float * data, unsigned npt);
void form_histogram (unsigned * hist, unsigned nbin, float * scales, float * data, unsigned npt);

void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, char * title);
void plot_series (float * y, unsigned npts, char * device, float M, char * title);
void plot_lags (float * yvals, unsigned npts, char * device);
void plot_histogram_old (float * vals, unsigned npts, unsigned nbin, char * device, char * title);
void plot_histogram (unsigned * hist, unsigned nbin, char * device, char * title, float xline, float mean_phase);
void plot_gsl_series (double * x, double * y, unsigned npts, char * device, float M, float C, char * title);
void plot_delays (float * delays, unsigned npts, char * device, char * title, mopsr_util_t * opt);
void plot_phase_offsets (float * phase_offets, unsigned npts, char * device, char * title, mopsr_util_t * opt);
void plot_map (float * map, unsigned nx, unsigned ny, char * device, char * title, mopsr_util_t * opt);
int  pair_from_ants (unsigned iant, unsigned jant, unsigned nant);

void usage()
{
  fprintf (stdout,
     "mopsr_solve_delays npt ccfile antenna_file\n"
     " -a antenna      reference antenna to use for relative delays [default 0]\n"
     " -b baseline     plot specific baseline in great detail\n"
     " -c channel      coarse channel number [default 0]\n"
     " -d altant       alternate ref antenna\n"
     " -D device       pgplot device name\n"
     " -p              plot SNR and Delay maps\n"
     " -r file         reject channels listed in file\n"
     " -t tsamp        sampling time [default 1.28, units micro seconds]\n"
     " -h              plot this help\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = 0;

  // reference antenna to use for solving delays
  int antenna = 0;

  // alternate reference antenna
  int alt_antenna = -1;

  // coarse channel being processed
  int channel = 0;
  
  // default sampling time
  float tsamp = 1.28;

  char plot_maps = 0;

  int arg = 0;

  int baseline = -1;

  char * channel_reject_file = NULL;
  char channel_rejection = 0;

  while ((arg=getopt(argc,argv,"a:b:c:d:D:hpr:t:v")) != -1)
  {
    switch (arg)
    {
      case 'a':
        antenna = atoi(optarg);
        break;

      case 'b':
        baseline = atoi(optarg);
        break;

      case 'c':
        channel = atoi(optarg);
        break;

      case 'd':
        alt_antenna = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'p':
        plot_maps = 1;
        break;

      case 'r':
        channel_reject_file = strdup(optarg);
        channel_rejection = 1;
        break;

      case 't':
        tsamp = atof (optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'h':
      default:
        usage ();
        return 0;
    } 
  }

  if (alt_antenna == -1)
    alt_antenna = antenna + 5;

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  unsigned npt = 0;
  if (sscanf(argv[optind], "%u", &npt) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse npt from %s\n", argv[optind]);
    usage();
    return (EXIT_FAILURE);
  }

  if (verbose > 1)
    fprintf (stderr, "npt=%u\n", npt);
  char cc_file[1024];
  strcpy(cc_file, argv[optind+1]);

  size_t file_size;
  struct stat buf;
  if (stat (cc_file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", cc_file, strerror(errno));
    return (EXIT_FAILURE);
  }
  file_size = buf.st_size;
  if (verbose > 1)
    fprintf (stderr, "filesize for %s is %d bytes\n", cc_file, file_size);

  char antenna_file[1024];
  strcpy (antenna_file, argv[optind+2]);
  if (stat (cc_file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", antenna_file, strerror(errno));
    return (EXIT_FAILURE);
  }

  int num_reject_channels = 0;
  int * channel_reject = NULL;
  if (channel_rejection)
  {
    FILE * fptr = fopen (channel_reject_file, "r");
    if (!fptr)
    {
      fprintf (stderr, "ERROR: failed to open file: %s for reading: %s\n", channel_reject_file, strerror(errno));
      return (EXIT_FAILURE);
    }

    char line1[1024];
    int i;
    fgets (line1, 1024, fptr);
    num_reject_channels = atoi(line1);

    channel_reject = (int*) malloc (num_reject_channels * sizeof(int));
    for(i=0;i<num_reject_channels;i++) 
    {
      fgets(line1, 1024, fptr);
      channel_reject[i] = atoi(line1);
    }
  }

  unsigned npairs = (unsigned) (file_size / (npt * sizeof(complex float)));
  unsigned nant = (1 + sqrt(1 + (8 * npairs))) / 2;
  if (verbose)
    fprintf (stderr, "npairs=%u nant=%u \n", npairs, nant);

  char line[1024];
  char antennas[nant][6];
  float dist;
  float delay;
  unsigned iant = 0;

  FILE * fptr = fopen (antenna_file, "r");
  if (!fptr)
  {
    fprintf (stderr, "failed to open atnenna file[%s]: %s\n", antenna_file, strerror(errno));
    return (EXIT_FAILURE);
  }

  int nscanned;
  while (iant < nant && fgets(line, 1024, fptr))
  {
    nscanned = sscanf(line, "%5s %f %f", antennas[iant], &dist, &delay);
    iant++;
  }

  fclose (fptr);

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (cc_file, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", cc_file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  // the ramps will be pre-calculated to range from -1 to 1 delay
  unsigned iramp;
  unsigned nramps = 100;
  unsigned nbin = 33;

  // for creating SNR and Delay Image maps
  float * delays_map = (float *) malloc (nant * nant * sizeof(float));
  float * snrs_map   = (float *) malloc (nant * nant * sizeof(float));

  memset (delays_map, 0, nant * nant * sizeof(float));
  memset (snrs_map,   0, nant * nant * sizeof(float));

  // Pair delay and map values
  float * delays = (float *) malloc (npairs * sizeof(float));
  float * snrs   = (float *) malloc (npairs * sizeof(float));
  float * phase_offsets = (float *) malloc (npairs * sizeof(float));

  unsigned * hist = (unsigned *) malloc (nbin * sizeof(unsigned));
  float * lags = (float *) malloc (npt * sizeof(float));
  float * amps = (float *) malloc (npt * sizeof(float));
  float * phases = (float *) malloc (npt * sizeof(float));

  complex float * raw = (complex float *) malloc (npt * sizeof(complex float));
  complex float * cc = (complex float *) malloc (npt * sizeof(complex float));
  complex float * sp = (complex float *) malloc (npt * sizeof(complex float));
  complex float * ts = (complex float *) malloc (npt * sizeof(complex float));

  // for the integer sample delay
  complex float * ramp = (complex float *) malloc (npt * sizeof(complex float));
  float * ramp_peaks = (float *) malloc (nramps * sizeof(float));

  // for the fractional sample search
  complex float ** ramps = (complex float **) malloc (nramps * sizeof(complex float *));
  for (iramp=0; iramp<nramps; iramp++)
    ramps[iramp] = (complex float *) malloc (npt * sizeof(complex float));

  fftwf_complex * in = (fftwf_complex *) sp;
  fftwf_complex * ou = (fftwf_complex *) ts;
  fftwf_plan plan_bwd = fftwf_plan_dft_1d (npt, in, ou, FFTW_BACKWARD, FFTW_ESTIMATE);

  unsigned ipair;
  float fraction, theta, max, re, im, power, percent;
  int ipt, max_pt, max_ramp, shift, lag;
  char dead;

  double * coarse_phases = (double *) malloc (npt * sizeof(double));

  unsigned N_skip = npt / 20;
  unsigned N = npt - (2 * N_skip);
  double * x = (double *) malloc (npt * sizeof(double));
  double * y = (double *) malloc (npt * sizeof(double));

  for (ipt=0; ipt<npt; ipt++)
    x[ipt] =((double) ipt / (double) (npt-1)) * 2 * M_PI;

  double c0, c1, cov00, cov01, cov11, chisq;

  if (verbose > 1)
    fprintf (stderr, "created FFT plan\n");

  // ramp range is from -1 turn to +1 turn
  for (iramp=0; iramp<nramps; iramp++)
  {
    fraction = (((float) iramp / (float) (nramps-1))) - 0.5;
    for (ipt=0; ipt<npt; ipt++)
    {
      percent = ((float) ipt / (float) npt) - 0.5;
      theta = fraction * (2 * M_PI * percent);
      ramps[iramp][ipt] = cos (theta) - sin(theta) * I; 
    }
  }

  if (verbose > 1)
    fprintf (stderr, "initialized ramps\n");

  char title[1024];

  size_t bytes_to_read = npt * sizeof(complex float);
  int ramp_best;
  float ramp_max, iramp_delay, gsl_delay, ramp_delay, snr, avg_phase, phase, total_delay;
  float coarse_phase_maxbin;
  double centre_phase;
  unsigned ibin, max_bin, max_bin_val;

  unsigned i = 1;
  unsigned tri = i;
  unsigned ant_a, ant_b;

  //antenna to alt_antenna pair
  int alt_pair = -1;

  for (ipair=0; ipair<npairs; ipair++)
  {
    ant_a = i / nant;
    ant_b = i % nant;

    if (ant_a == antenna && ant_b == alt_antenna)
    {
      alt_pair = ipair;
    }

    // read in spectra pair
    size_t bytes_read = read (fd, (void *) raw, bytes_to_read);

    if (verbose > 1)
      fprintf (stderr, "read %ld bytes from file\n", bytes_read);

    // check if all non-DC bins are 0
    dead = 1;
    for (ipt=0; ipt<npt; ipt++)
    {
      cc[ipt] = raw[ipt];

      // DC bin casuses weird ripples in the iFFT (probably because
      // we have shifted in the output of the CC??) perhaps un fft-shift
      // for correctness
      //if ((ipt != npt/2) && ((cc[ipt]) != 0) && (cimag(cc[ipt]) != 0))
      if (((cc[ipt]) != 0) && (cimag(cc[ipt]) != 0))
        dead = 0;
    }
    //cc[npt/2] = 0;
    if (baseline == ipair)
      dead = 0;

    if (channel_rejection)
    {
      for (ipt=0; ipt<num_reject_channels; ipt++)
      {
        cc[channel_reject[ipt]] = 0;
      }

    }

    if (verbose > 1 || baseline == ipair)
      fprintf (stderr, "antenna=%u pair %d [%u -> %u], dead=%d\n", antenna, ipair, ant_a, ant_b, dead);

    ramp_delay = 0;
    gsl_delay = 0;
    avg_phase = 0;
    snr = 0;

    //if (ant_a != antenna)
    //  dead = 1;

    if (!dead)
    {
      // inverse FFT to find the integer delay
      in = (fftwf_complex *) cc;
      ou = (fftwf_complex *) ts;
      fftwf_execute_dft (plan_bwd, in, ou);

      for (ipt=0; ipt<npt; ipt++)
      {
        re = creal(ts[ipt]);
        im = cimag(ts[ipt]);
        power = (re * re) + (im * im);

        shift = (ipt + npt/2) % npt;
        lags[shift] = power;

        re = creal(cc[ipt]);
        im = cimag(cc[ipt]);
        amps[ipt] = (re*re) + (im*im);
        phases[ipt] = atan2f(im, re);
      }

      lag = furtherest_from_mean (lags, npt) - (npt/2);
      snr = calculate_snr (lags, npt);
      if (baseline == ipair)
        fprintf (stderr, "pair=%d lag=%d snr=%f\n", ipair, lag, snr);

      // If SNR of RAMP < 20, dont bother doing anything
      // In the high snr case > 1000, least squares fitting will get a better
      // result due to FIR edge effects, for low SNR case - just rely on ramp
      // trials for best fit

      if (snr > 20 || baseline == ipair)
      {
        if ((verbose > 1) || (baseline == ipair))
        {
          plot_lags (lags, npt, "1/xs");
          plot_cross_bin (amps, phases, npt, "2/xs", 
                          "Cross Correlation - NO delay correction");
        }
        
        // perform integer delay correction if necessary
        if (lag != 0)
        {
          fraction = (float) lag;
          for (ipt=0; ipt<npt; ipt++)
          {
            percent = ((float) ipt / (float) npt) - 0.5;
            theta = fraction * (2 * M_PI * percent);
            ramp[ipt] = cos (theta) + sin(theta) * I;
            cc[ipt] *= ramp[ipt];
          }

          if ((verbose > 1) || (baseline == ipair))
          {
            for (ipt=0; ipt<npt; ipt++)
            {
              re = creal(cc[ipt]);
              im = cimag(cc[ipt]);
              amps[ipt] = (re*re) + (im*im);
              phases[ipt] = atan2f(im, re);
            }
            plot_cross_bin (amps, phases, npt, "3/xs", 
                "Cross Correlation after Integer Delay Correction");
          }
        }

        // use trial ramps get a good estimate on fractional delay
        ramp_best = -1;
        ramp_max = 0;
        for (iramp=0; iramp<nramps; iramp++)
        {
          for (ipt=0; ipt<npt; ipt++)
          {
            sp[ipt] = cc[ipt] * ramps[iramp][ipt];
            if ((verbose > 1) || (baseline == ipair))
            {
              re = creal(sp[ipt]);
              im = cimag(sp[ipt]);
              amps[ipt] = (re*re) + (im*im);
              phases[ipt] = atan2f(im, re);
            }
          }

          if ((verbose > 1) || (baseline == ipair))
            plot_cross_bin (amps, phases, npt, "3/xs", "Cross Correlation - Trial Ramp");

          in = (fftwf_complex *) sp;
          ou = (fftwf_complex *) ts;
          fftwf_execute_dft (plan_bwd, in, ou);

          for (ipt=0; ipt<npt; ipt++)
          {
            re = creal(ts[ipt]);
            im = cimag(ts[ipt]);
            power = (re * re) + (im * im);
            shift = (ipt + npt/2) % npt;
            lags[shift] = power;
          }

          float mean = get_mean (lags, npt);
          max = powf ((lags[npt/2] - mean), 2);
          ramp_peaks[iramp] = max;

          if ((verbose > 1) || (baseline == ipair))
            plot_lags (lags, npt, "1/xs");

          if (max > ramp_max)
          {
            ramp_max = max;
            ramp_best = iramp;
            iramp_delay = (((float) iramp / (float) (nramps-1))) - 0.5;
          }
        }

        // this is the delay to be use when SNR < 1000
        ramp_delay = (float) lag - iramp_delay;

        if ((verbose) || (baseline == ipair))
          fprintf (stderr, "[%d] ramp_best=%d iramp_delay=%f ramp_delay=%f\n", 
                   ipair, ramp_best, iramp_delay, ramp_delay);

        complex float mean_value = 0;

        // calculate phases for best trial ramp
        for (ipt=0; ipt<npt; ipt++)
        {
          re = creal(cc[ipt] * ramps[ramp_best][ipt]);
          im = cimag(cc[ipt] * ramps[ramp_best][ipt]);
          amps[ipt] = re * re + im * im;
          phases[ipt] = atan2f(im, re);

          mean_value += (cc[ipt] * ramps[ramp_best][ipt]);
        }

        if ((verbose > 1) || (baseline == ipair))
        {
          plot_cross_bin (amps, phases, npt, "7/xs", "Cross Correlation - Best Trial Ramp");
        }

        form_histogram (hist, nbin, amps, phases, npt);

        // determine histogram peak
        max_bin = -1;
        max_bin_val = 0;
        for (ibin=0; ibin<nbin; ibin++)
        {
          if (hist[ibin] > max_bin_val)
          {
            max_bin = ibin;
            max_bin_val = hist[ibin];
          }
        }
        if ((verbose>1) || (baseline == ipair))
          fprintf (stderr, "max_bin=%u max_bin_val=%u\n", max_bin, max_bin_val);

        coarse_phase_maxbin = (2 * M_PI * ((float) max_bin) / nbin) - M_PI;

        if ((verbose>1) || (baseline == ipair))
          fprintf (stderr, "coarse_phase_maxbin=%f\n", coarse_phase_maxbin);

        for (ipt=0; ipt<npt; ipt++)
        {
          if (phases[ipt] > coarse_phase_maxbin + M_PI)
            coarse_phases[ipt] = (double) phases[ipt] - (2 * M_PI);
          else if (phases[ipt] < coarse_phase_maxbin - M_PI)
            coarse_phases[ipt] = (double) phases[ipt] + (2 * M_PI);
          else
            coarse_phases[ipt] = (double) phases[ipt];
        }

        gsl_sort (coarse_phases, 1, npt);
        centre_phase = gsl_stats_median_from_sorted_data (coarse_phases, 1, npt);

        //centre_phase = gsl_stats_mean (coarse_phases, 1, npt);

        if (centre_phase > M_PI)
          avg_phase = (float) centre_phase - (2 * M_PI);
        else
          avg_phase = (float) centre_phase;

        if ((verbose>1) || (baseline == ipair))
          fprintf (stderr, "centre_phase=%lf, avg_phase=%f\n", centre_phase, avg_phase);

        if ((verbose > 1) || (baseline == ipair))
        {
          float mean_phase = atan2f (cimag(mean_value), creal(mean_value));
          sprintf (title, "Uncorrected Histogram: Pair %d avg=%3.2f mean=%3.2f", ipair, avg_phase, mean_phase);
          plot_histogram (hist, nbin, "4/xs", title, avg_phase, mean_phase);
        }

        if ((verbose > 1) || (baseline == ipair))
        {
          // apply peak phase correction to the whole band to centre
          // this + integer correction should remove all wraps...
          for (ipt=0; ipt<npt; ipt++)
          {
            phases[ipt] -= avg_phase;
            if (phases[ipt] < -1 * M_PI)
              phases[ipt] += 2 * M_PI;
            if (phases[ipt] >  M_PI)
              phases[ipt] -= 2 * M_PI;
          }
  
          form_histogram (hist, nbin, amps, phases, npt);
          sprintf (title, "Corrected Histogram");
          plot_histogram (hist, nbin, "8/xs", title, 0, 0);

          fprintf (stderr, "SNR=%f\n", snr);
        }

        if ((snr > 5000) && (0))
        {
          for (ipt=0; ipt<npt; ipt++)
          {
            y[ipt] = (double) (atan2f (cimag(cc[ipt]), creal(cc[ipt])) - avg_phase);
            if (y[ipt] < -1 * M_PI)
              y[ipt] += 2 * M_PI;
            if (y[ipt] >  M_PI)
              y[ipt] -= 2 * M_PI;

          }

          y[npt/2] = (y[npt/2-1] + y[npt/2+1]) / 2;

          gsl_fit_linear (x + N_skip, 1, y + N_skip, 1, (size_t) N, &c0, &c1, &cov00, &cov01, &cov11, &chisq);
          gsl_delay = (float) lag - (float) c1;
          if (verbose)
            fprintf (stderr, "gsl_delay: %f [%d - %lf]\n", gsl_delay, lag, c1);
          total_delay = gsl_delay;
        }
        else
        {
          total_delay = ramp_delay;
        }

        if ((verbose > 1) || (baseline == ipair))
        { 
          for (ipt=0; ipt<npt; ipt++)
          {
            cc[ipt] = raw[ipt];
            //if (ipt == npt/2)
            //  cc[ipt] = 0;
            re = creal(cc[ipt]);
            im = cimag(cc[ipt]);
            amps[ipt] = (re*re) + (im*im);
            phases[ipt] = atan2f (im, re) - avg_phase;
            if (phases[ipt] < -1 * M_PI)
              phases[ipt] += 2 * M_PI;
            if (phases[ipt] >  M_PI)
              phases[ipt] -= 2 * M_PI;
          }

          if ((snr > 5000) && 0)
          {
            sprintf (title, "GSL Corrected Phase. SNR=%5.2f Delay=%5.2f", snr, gsl_delay);
            plot_series (phases, npt, "6/xs", gsl_delay, title);
          }
          sprintf (title, "Ramp Corrected Phase. SNR=%5.2f Delay=%5.2f", snr, ramp_delay);
          plot_series (phases, npt, "5/xs", ramp_delay, title);
        }
      }
    }

    if (snr < 50)
    {
      total_delay = 0;
      avg_phase = 0;
    }

    phase_offsets[ipair] = avg_phase;
    delays[ipair] = -1 * total_delay;
    delays_map[i] = -1 * total_delay;
    snrs[ipair]   = snr;
    snrs_map[i]   = snr;

    i++;
    if (i % nant == 0)
    {
      tri++;
      i += tri;
    }
  }

  close (fd);

  // now that we have an SNR and delay value for each pair, determine the
  ipair = 0;
  int jant;
  float * rel_delays = (float *) malloc (sizeof(float) * nant);
  float * rel_phases = (float *) malloc (sizeof(float) * nant);
  float * rel_snrs   = (float *) malloc (sizeof(float) * nant);

  float alt_delay = delays[alt_pair];
  float alt_phase = phase_offsets[alt_pair];

  for (iant=0; iant<nant; iant++)
  {
    rel_delays[iant] = 0;
    rel_phases[iant] = 0;
    rel_snrs[iant]   = 0;
  }

  if (verbose)
    fprintf (stderr, "computing relative delays\n");

  int jpair;
  for (iant=0; iant<nant; iant++)
  {
    for (jant=iant+1; jant<nant; jant++)
    {
      if (ipair == baseline)
        fprintf (stderr, "snrs[%d]=%f iant=%d jant=%d\n", ipair, snrs[ipair], iant, jant);
      // need to have a half decent correlation to calculate a delay
      if (snrs[ipair] > 50)
      {
        //fprintf (stderr, "[%d, %d] -> %d snr=%f delay=%f\n", iant, jant, ipair, snrs[ipair], delays[ipair]);

        if (iant == antenna)
        {
          if (rel_delays[jant] == 0)
          {
            // AJ: TO remove
            if ((abs(jant-iant) < 4) && 0)
            {
              // get the pair index for the jant to the reference
              jpair = pair_from_ants (jant, (unsigned) alt_antenna, nant);
              if (verbose > 2 || baseline == ipair)
                fprintf (stderr, "0: iant=%d jant=%d ipair=%d jpair=%d\n", iant, jant, ipair, jpair);

              // now compute the differnce of the two round trip
              rel_delays[jant] = alt_delay - delays[jpair];
              rel_phases[jant] = alt_phase - phase_offsets[jpair];
              rel_snrs[jant]   = snrs[jpair];
            }
            else
            {
              rel_delays[jant] = delays[ipair];
              rel_phases[jant] = phase_offsets[ipair];
              rel_snrs[iant]   = snrs[ipair];
              if (verbose > 2 || baseline == ipair)
                fprintf (stderr, "1: iant=%d jant=%d ipair=%d rel_delays[%d]=%f\n", iant, jant, ipair, jant, rel_delays[jant]);
            }
          }

        }
        if (jant == antenna)
        {
          if (rel_delays[iant] == 0)
          {
            rel_delays[iant] = delays[ipair] * -1;
            rel_phases[iant] = phase_offsets[ipair] * -1;
            rel_snrs[iant]   = snrs[ipair];
            if (verbose > 2 || baseline == ipair)
              fprintf (stderr, "2: iant=%d jant=%d ipair=%d rel_delays[%d]=%f\n", iant, jant, ipair, iant, rel_delays[iant]);
          }
        }
      }
      ipair ++;
    }
  }

  if (verbose)
    fprintf (stderr, "writing obs.delays\n");
  // write obs.delays file
  fptr = fopen ("obs.delays", "w");
  if (!fptr)
  {
    fprintf (stderr, "failed to open obs.delays for writing: %s\n", strerror(errno));
    return (EXIT_FAILURE);
  }

  fprintf (fptr, "ref %s\n", antennas[antenna]);
  for (iant=0; iant<nant; iant++)
  {
    fprintf (fptr, "%s %e %f %f\n", antennas[iant], (rel_delays[iant] * (tsamp / 1000000)), rel_phases[iant], snrs[iant]);
  }
  fclose (fptr);

  if (plot_maps)
  {
    if (verbose)
      fprintf (stderr, "creating plots\n");
    mopsr_util_t opt;

    // Delays plot
    sprintf (title, "Baseline Delays: CH%02d", channel);

    opt.nant = nant;
    opt.plot_log = 0;
    opt.plot_plain = 0;
    opt.ymin = -1;
    opt.ymax = -1;

    time_t now = time(0);
    char local_time[32];
    strftime (local_time, 32, DADA_TIMESTR, localtime(&now));
    char filename[1024];

    // high resolution
    if (!device)
      sprintf (filename, "%s.CH%02u.bd.%dx%d.png/png", local_time, channel, 1024, 768);
    else
      sprintf (filename, "1/xs");
    if (cpgbeg(0, filename, 1, 1) != 1)
      fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
    if (!device)
      set_resolution (1024, 768);
    plot_map (delays_map, nant, nant, filename, title, &opt);
    cpgend();

    // low resolution (unless we are doing interactive plots)
    if (!device)
    {
      sprintf (filename, "%s.CH%02u.bd.%dx%d.png/png", local_time, channel, 160, 120);
      if (cpgbeg(0, filename, 1, 1) != 1)
        fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
      set_resolution (160, 120);
      opt.plot_plain = 1;
      opt.ymin = -1;
      opt.ymax = -1;
      plot_map (delays_map, nant, nant, filename, title, &opt);
      cpgend();
    }

    /////////////////////////////////////////////////////////////////////////////
    //  
    // Plot delays for the reference antenna
    // 
    opt.plot_plain = 0;
    sprintf (title, "Antenna Delays: CH%02d, Ref Antenna: %s [%d]", channel, antennas[antenna], antenna);
    if (!device)
      sprintf (filename, "%s.CH%02u.ad.%dx%d.png/png", local_time, channel, 1024, 768);
    else
      sprintf (filename, "2/xs");
    if (cpgbeg(0, filename, 1, 1) != 1)
      fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
    if (!device)
    set_resolution (1024, 768);
    plot_delays (rel_delays, nant, filename, title, &opt);
    cpgend();

    // low resolution (unless we are doing interactive plots)
    if (!device)
    {
      sprintf (filename, "%s.CH%02u.ad.%dx%d.png/png", local_time, channel, 160, 120);
      if (cpgbeg(0, filename, 1, 1) != 1)
        fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
      set_resolution (160, 120);
      opt.plot_plain = 1;
      plot_delays (rel_delays, nant, filename, title, &opt);
      cpgend();
    }


    /////////////////////////////////////////////////////////////////////////////
    //  
    // Plot phase offsets for the reference antenna
    // 
    opt.plot_plain = 0;
    sprintf (title, "Phase Offsets: CH%02d, Ref Antenna: %s [%d]", channel, antennas[antenna], antenna);
    if (!device)
      sprintf (filename, "%s.CH%02u.po.%dx%d.png/png", local_time, channel, 1024, 768);
    else
      sprintf (filename, "3/xs");
    if (cpgbeg(0, filename, 1, 1) != 1)
      fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
    if (!device)
      set_resolution (1024, 768);
    plot_phase_offsets (rel_phases, nant, filename, title, &opt);
    cpgend();

    // low resolution (unless we are doing interactive plots)
    if (!device)
    {
      sprintf (filename, "%s.CH%02u.po.%dx%d.png/png", local_time, channel, 160, 120);
      if (cpgbeg(0, filename, 1, 1) != 1)
        fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
      set_resolution (160, 120);
      opt.plot_plain = 1;
      plot_phase_offsets (rel_phases, nant, filename, title, &opt);
      cpgend();
    }


    /////////////////////////////////////////////////////////////////////////////
    //  
    // Plot SNR map
    // 
    sprintf (title, "Baseline SNRs: CH%02d", channel);

    opt.plot_plain = 0;
    opt.plot_log = 1;
    opt.ymin = 0;
    opt.ymax = 4;

    if (!device)
      sprintf (filename, "%s.CH%02u.sn.%dx%d.png/png", local_time, channel, 1024, 768);
    else
      sprintf (filename, "4/xs");
    if (cpgbeg(0, filename, 1, 1) != 1)
      fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
    if (!device)
      set_resolution (1024, 768);
    plot_map (snrs_map, nant, nant, filename, title, &opt);
    cpgend();

    if (!device)
    {
      sprintf (filename, "%s.CH%02u.sn.%dx%d.png/png", local_time, channel, 160, 120);
     if (cpgbeg(0, filename, 1, 1) != 1)
        fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
      set_resolution (160, 120);
      opt.plot_plain = 1;
      plot_map (snrs_map, nant, nant, filename, title, &opt);
      cpgend();
    }
  }

  free (hist);
  free (lags);
  free (amps);
  free (phases);
  free (raw);

  free (x);
  free (y);
  free (rel_delays);
  free (rel_phases);
  free (delays_map);
  free (snrs_map);
  free (delays);
  free (snrs);
  free (phase_offsets);

  free (cc);
  free (sp);
  free (ts);
  free (ramp);
  free (ramps);

  fftwf_destroy_plan (plan_bwd);

  return EXIT_SUCCESS;
}

void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, char * title)
{
  float xmin = 0;
  float xmax = (float) npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  char label[64];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (amps[i] > ymax)
      ymax = amps[i];
    if (amps[i] < ymin)
      ymin = amps[i];

    xvals[i] = (float) i;
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Amplitude", title);

  cpgsci(3);
  cpgline (npts, xvals, amps);
  cpgsci(1);

  ymin = FLT_MAX;
  ymax = -FLT_MAX;

  for (i=0; i<npts; i++)
  {
    if (phases[i] > ymax)
      ymax = phases[i];
    if (phases[i] < ymin)
      ymin = phases[i];
  }

  cpgswin(xmin, xmax, -1 * M_PI, M_PI);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Phase", "");

  cpgsci(3);
  cpgpt(npts, xvals, phases, 17);
  cpgsci(1);

  cpgebuf();
  cpgend();
}

void plot_lags (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  int offset = (npts/2) + 1;

  float xmin = -1 * (float) offset;
  float xmax = 1 * (float) offset;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  int xpeak = 0;

  int i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
    {
      ymax = yvals[i];
      xpeak = i - offset;
    }

    if (yvals[i] < ymin)
      ymin = yvals[i];

    xvals[i] = (float) (i - offset);
  }

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Sample", "Strength", "Integer Sample Delays");

  cpgsci(1);
  cpgsls(5);
  cpgline(npts, xvals, yvals);

  cpgsci(5);
  cpgsls(1);
  cpgpt(npts, xvals, yvals, -5);

  cpgebuf();
  cpgend();
}

void plot_series (float * y, unsigned npts, char * device, float M, char * title)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float x[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    x[i] = (float) i;

    if (y[i] > ymax)
      ymax = y[i];
    if (y[i] < ymin)
      ymin = y[i];

    if (x[i] > xmax)
      xmax = x[i];
    if (x[i] < xmin)
      xmin = x[i];
  }
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  ymin = -1 * M_PI;
  ymax = 1 * M_PI;

  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Angle", title);

  cpgsci(2);
  cpgpt(npts, x, y, -5);
  cpgsci(1);

  float xxx[2] = {xmin, xmax};
  float yyy[2];

  float real_M = (-2 * M_PI * M) / npts;
  //float C = real_M * (npts/2);
  float C = M_PI * M;

  yyy[0] = (real_M * xxx[0]) + C;
  yyy[1] = (real_M * xxx[1]) + C;

  cpgsci(3);
  cpgline (2, xxx, yyy);
  cpgsci(1);

  cpgebuf();
  cpgend();
}

void plot_gsl_series (double * x, double * y, unsigned npts, char * device, float M, float C, char * title)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xx[npts];
  float yy[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    xx[i] = (float) x[i];
    yy[i] = (float) y[i];

    if (yy[i] > ymax)
      ymax = yy[i];
    if (yy[i] < ymin)
      ymin = yy[i];

    if (xx[i] > xmax)
      xmax = x[i];
    if (xx[i] < xmin)
      xmin = xx[i];
  }
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  ymin = -1 * M_PI;
  ymax = M_PI;
  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Angle", title);

  cpgsci(2);
  cpgpt(npts, xx, yy, -5);
  //cpgline (npts, xx, yy);
  cpgsci(1);

  for (i=0; i<npts; i++)
    yy[i] = (M * xx[i]) + C;

  cpgsci(3);
  cpgline (npts, xx, yy);
  cpgsci(1);

  cpgebuf();
  cpgend();
}


void plot_histogram_old (float * vals, unsigned npts, unsigned nbin, char * device, char * title)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;
  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (vals[i] > ymax) ymax = vals[i];
    if (vals[i] < ymin) ymin = vals[i];
  }

  ymin = -1 * M_PI;
  ymax = 1 * M_PI;

  int N = (int) npts;
  int nb = (int) nbin;
  if (nb > 128)
    nb = 128;

  int flag = 0;
  cpgbbuf();
  cpghist (N, vals, ymin, ymax, nbin, flag);
  cpglab("Phase", "Counts", title);
  cpgebuf();
  cpgclos();
}

void plot_histogram (unsigned * hist, unsigned nbin, char * device, char * title, float xline, float mean_phase)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = -M_PI;
  float xmax = M_PI;
  float xstep = (xmax - xmin) / nbin;

  float xvals[nbin];
  float yvals[nbin];

  unsigned ibin;
  float ymax = -FLT_MAX;
  for (ibin=0; ibin<nbin; ibin++)
  {
    xvals[ibin] = xmin + (ibin * xstep);
    yvals[ibin] = (float) hist[ibin];
    //xvals[ibin] = (float) ibin;
    if (yvals[ibin] > ymax)
      ymax = yvals[ibin];
  }

  int flag = 0;
  cpgbbuf();
  //xmin = 0;
  //xmax = nbin;
  cpgenv(xmin, xmax, 0, ymax * 1.1, 0, 0);
  cpglab("Phase", "Counts", title);
  cpgbin (nbin, xvals, yvals, flag);

  float x[2] = {xline, xline};
  float y[2] = {0, ymax};
  cpgline (2, x, y);

  x[0] = mean_phase;
  x[1] = mean_phase;
  cpgsci(2);
  cpgline (2, x, y);

  cpgebuf();
  cpgclos();

}

float get_mean (float * data, unsigned npt)
{
  unsigned ipt;
  double mean = 0;
  for (ipt=0; ipt<npt; ipt++)
  {
    mean += (double) data[ipt];
  }
  return (float) (mean / (double) npt);
}

int compare_floats (const float * a, const float * b)
{
  if (*a > *b)
    return 1;
  else if (*a < *b)
    return -1;
  else
    return 0;
}

float calculate_snr (float * data, unsigned npt)
{
  double ddata[npt];
  unsigned ipt;

  for (ipt=0; ipt<npt; ipt++)
    ddata[ipt] = (double) data[ipt];

  // sort the data (doubles)
  gsl_sort (ddata, 1, npt);

  double median = gsl_stats_median_from_sorted_data (ddata, 1, npt);

  // the abs value of distance from median
  for (ipt=0; ipt<npt; ipt++)
    ddata[ipt] = fabs (ddata[ipt] - median);

  // resort the data
  gsl_sort (ddata, 1, npt);

  double sum = 0;
  unsigned rms_npts = floor ((float) npt * 0.9);
  for (ipt=0; ipt<rms_npts; ipt++)
    sum += pow(ddata[ipt],2);
  double rms = sqrt( sum / rms_npts);

  double snr = ddata[npt-1] / rms;

  return (float) snr;
}

int furtherest_from_mean (float * data, unsigned npt)
{
  unsigned ipt;
  int max_pt = -1;
  float dist, max_dist;
  float mean = get_mean (data, npt);

  max_dist = 0;
  for (ipt=0; ipt<npt; ipt++)
  {
    dist = powf((data[ipt] - mean), 2); 
    if (dist > max_dist)
    {
      max_dist = dist;
      max_pt = ipt;
    }
  }

  return max_pt;
}

void form_histogram (unsigned * hist, unsigned nbin, 
                     float * scales, float * data, unsigned npt)
{
  float scaled_hist[nbin];

  unsigned ibin, ipt;
  for (ibin=0; ibin<nbin; ibin++)
    scaled_hist[ibin] = 0;

  float scale_max = 0;
  for (ipt=0; ipt<npt; ipt++)
  {
    if (scales[ipt] > scale_max)
      scale_max = scales[ipt];
  }

  float val;
  int bin;
  for (ipt=0; ipt<npt; ipt++)
  {
    val = data[ipt] + M_PI;
    bin = floor((val / (2 * M_PI)) * nbin);
    if (bin < 0)
      bin  = 0;
    if (bin >= nbin)
      bin = nbin-1;
    scaled_hist[bin] += (scales[ipt] / scale_max);
  }

  for (ibin=0; ibin<nbin; ibin++)
    hist[ibin] = rintf (scaled_hist[ibin] * 100);
}

void plot_delays (float * delays, unsigned npts, char * device, char * title, mopsr_util_t * opt)
{
  float xmin = 0;
  float xmax = (float) npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float x1[npts];
  float x2[npts];
  float yvals[npts];
  char label[64];

  unsigned i;
  unsigned ipt = 0;
  for (i=0; i<npts; i++)
  {
    if (delays[i] > ymax && delays[i] < 3)
      ymax = delays[i];
    if (delays[i] < ymin && delays[i] > 3)
      ymin = delays[i];

    x1[i] = (float) i + 0.5;
    if (fabsf (delays[i]) > 0.001)
    {
      x2[ipt] = (float) i + 0.5;
      yvals[ipt] = delays[i];
      ipt++;
    }
  }

  cpgbbuf();

  float yrange = ymax - ymin;

  ymax += (yrange / 10);
  ymin -= (yrange / 10);

  if (ymax < 1)
    ymax = 1;
  if (ymin > -1)
    ymin = -1;
  //fprintf (stderr, "ymin=%f ymax=%f yrange=%f\n", ymin, ymax, yrange);

  if (opt->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);

  cpgswin (xmin, xmax, ymin, ymax);

  if (!opt->plot_plain)
    cpglab("Antenna", "Delay [samples]", title);

  if (!opt->plot_plain)
    cpgbox("BCGNST", 16, 0, "BCNST", 0, 0);

  cpgsci(15);
  cpgslw(3);
  cpgpt(npts, x1, delays, -2);

  cpgsci(3);
  if (!opt->plot_plain)
    cpgslw(10);
  else
    cpgslw(4);
  cpgpt(ipt, x2, yvals, -2);
  cpgslw(1);
  cpgsci(1);

  cpgebuf();
}

void plot_phase_offsets (float * phase_offsets, unsigned npts, char * device, char * title, mopsr_util_t * opt)
{
  float xmin = 0;
  float xmax = (float) npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float x1[npts];
  float x2[npts];
  float yvals[npts];
  char label[64];

  ymin = -1 * M_PI;
  ymax =  1 * M_PI;
  unsigned i;
  unsigned ipt = 0;
  for (i=0; i<npts; i++)
  {
    x1[i] = (float) i + 0.5;
    if (fabsf(phase_offsets[i]) > 0)
    {
      x2[ipt] = (float) i + 0.5;
      yvals[ipt] = phase_offsets[i];
      ipt++;
    }
  }

  cpgbbuf();

  if (opt->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);

  cpgswin (xmin, xmax, ymin, ymax);

  if (!opt->plot_plain)
    cpglab("Antenna", "Phase Offsets [radians]", title);

  if (!opt->plot_plain)
    cpgbox("BCGNST", 16, 0, "BCNST", 0, 0);

  cpgsci(15);
  cpgpt(npts, x1, phase_offsets, -2);
  cpgsci(3);
  cpgpt(ipt, x2, yvals, -4);
  cpgsci(1);

  cpgebuf();
}



void plot_map (float * map, unsigned nx, unsigned ny, char * device, char * title, mopsr_util_t * opt)
{
  unsigned i;
  float * m = (float *) malloc (sizeof(float) * nx * ny);

  cpgbbuf();
  cpgsci(1);

  if (opt->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);

  cpgswin(0, (float) nx, 0, (float) ny);

  if (!opt->plot_plain)
    cpglab("Ant", "Ant", title);

  float contrast = 1.0;
  float brightness = 0.5;

  /*
  float cool_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float cool_r[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float cool_g[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float cool_b[] = {0.0, 0.5, 1.0, 1.0, 1.0};

  cpgctab (cool_l, cool_r, cool_g, cool_b, 5, contrast, brightness);
  */

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float x_min = 0;
  float x_max = (float) nx;

  float y_min = 0;
  float y_max = (float) ny;

  float x_res = (x_max-x_min)/x_max;
  float y_res = (y_max-y_min)/y_max;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = nx * ny;
  float z_min = FLT_MAX;
  float z_max = -FLT_MAX;
  float z_avg = 0;

  if (opt->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opt->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int ix, iy;
  unsigned int ndat_avg = 0;
  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      i = ix * nx + iy;
      m[i] = map[i];
      if (opt->plot_log) 
        if (map[i] > 0)
          m[i] = log10(map[i]);
        else
          m[i] = 0;

      if ((opt->ymin == -1) && (opt->ymax == -1))
      {
        if (m[i] > z_max) z_max = m[i];
        if (m[i] < z_min) z_min = m[i];
      }
      else
      {
        z_min =  opt->ymin;
        z_max =  opt->ymax;
      }

      z_avg += m[i];
      ndat_avg++;
    }
  }

  //fprintf (stderr, "z_min=%f z_max=%f\n", z_min, z_max);

  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      i = ix * nx + iy;
      if (opt->plot_log && m[i] == 0)
        m[i] = z_min;
    }
  }

  cpgimag(m, nx, ny, 1, nx, 1, ny, z_min, z_max, trf);
  if (!opt->plot_plain)
    cpgbox("BCGNST", 16, 0, "BCGNST", 16, 0);
  cpgebuf();

  free (m);
}

// compute the ipair corresponding to the specified iant and jant
int pair_from_ants (unsigned iant, unsigned jant, unsigned nant)
{
  int pair = 0;

  int a_count = (int) nant - 1;

  unsigned i;
  for (i=0; i<iant; i++)
  {
    pair += a_count;
    a_count --;
  }

  pair += ((jant-iant)-1);
  return pair;
}

