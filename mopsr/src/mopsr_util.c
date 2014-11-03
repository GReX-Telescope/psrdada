/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cpgplot.h>
#include <math.h>
#include <complex.h>
#include <float.h>
#include <fftw3.h>

#include "mopsr_util.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"

int mopsr_channelise_frame (char * buffer, uint64_t nbytes, unsigned nant, unsigned npt)
{
  size_t fft_bytes = sizeof(float) * npt * 2;
  float * fft_in  = (float *) malloc (fft_bytes);
  if (!fft_in)
  {
    fprintf (stderr, "could not allocated %ld bytes\n", fft_bytes);
    return -1;
  }
  float * fft_out = (float *) malloc (fft_bytes);
  if (!fft_out)
  {
    fprintf (stderr, "could not allocated %ld bytes\n", fft_bytes);
    return -1;
  }

  int direction_flags = FFTW_FORWARD;
  int flags = FFTW_ESTIMATE;
  fftwf_plan plan = fftwf_plan_dft_1d (npt, (fftwf_complex*) fft_in, (fftwf_complex*)fft_out, direction_flags, flags);

  uint64_t nframe = nbytes / (nant * 2 * npt);

  int8_t * in  = (int8_t *) buffer;
  int8_t * out = (int8_t *) buffer;

  uint64_t iframe;
  unsigned iant, ipt;

  // read data in
  for (iframe=0; iframe<nframe; iframe++)
  {
    for (iant=0; iant<nant; iant++)
    {
      for (ipt=0; ipt<npt; ipt++)
      {
        fft_in[2*ipt+0] = (float) (in[2*(ipt*nant+iant) + 0]) + 0.5;
        fft_in[2*ipt+1] = (float) (in[2*(ipt*nant+iant) + 1]) + 0.5;
      }

      // fft this ant/batch
      fftwf_execute_dft (plan, (fftwf_complex*) fft_in, (fftwf_complex*) fft_out);

      for (ipt=0; ipt<npt; ipt++)
      {
        out[2*(ipt*nant+iant) + 0] = (int8_t) floor((fft_out[2*ipt+0]/npt) - 0.5);
        out[2*(ipt*nant+iant) + 1] = (int8_t) floor((fft_out[2*ipt+1]/npt) - 0.5);
      }
    }
    in += (nant * npt * 2);
    out += (nant * npt * 2);
  }

  free (fft_in);
  free (fft_out);
  fftwf_destroy_plan (plan);

  return 0;
}

int mopsr_print_pfbframe (char * buffer, ssize_t bytes, mopsr_util_t * opts)
{
  char ant0r[9];
  char ant0i[9];
  char ant1r[9];
  char ant1i[9];

  unsigned int ibyte = 0;
  unsigned int size  = opts->nchan * opts->nant * opts->ndim;
  if ( size > bytes)
  {
    fprintf (stderr, "did not include enough bytes for a full frame\n");
    return -1;
  }

  fprintf(stderr, "chan\tant0imag ant0real ant1imag ant1real\n");
  for (ibyte=0; ibyte<size; ibyte += 4)
  {
    char_to_bstring (ant0i, buffer[ibyte+0]);
    char_to_bstring (ant0r, buffer[ibyte+1]);
    char_to_bstring (ant1i, buffer[ibyte+2]);
    char_to_bstring (ant1r, buffer[ibyte+3]);
    fprintf (stderr, "%d\t%s %s %s %s\t%d\t%d\t%d\t%d\n", ibyte/4, ant0i, ant0r, ant1i, ant1r, (int8_t) buffer[ibyte+0], (int8_t) buffer[ibyte+1], (int8_t) buffer[ibyte+2], (int8_t) buffer[ibyte+3]);
  }
  return 0;
}

void mopsr_sqld_pfbframe (float * power_spectra, char * buffer,
                          mopsr_util_t * opts, int flag)
{
  unsigned ichan = 0;
  unsigned iant = 0;
  int8_t * input = (int8_t *) buffer;
  int re, im;

  for (ichan=0; ichan < opts->nchan; ichan++)
  {
    for (iant=0; iant < opts->nant; iant++)
    {
      if (iant == opts->ant)
      {
        re = (int) input[0];
        im = (int) input[1];
        if (flag == 0)
          power_spectra[ichan]  = (float) ((re*re) + (im*im));
        else  
          power_spectra[ichan] += (float) ((re*re) + (im*im));
      }
      input += 2;
    }
  }
}

void mopsr_zap_channel (char * buffer, ssize_t bytes, unsigned int schan, unsigned int echan, mopsr_util_t * opts)
{
  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  unsigned nframe = bytes / (opts->nant * opts->nchan * 2);
  int8_t * in = (int8_t *) buffer;

  int64_t re_sum = 0;
  int64_t im_sum = 0;
  uint64_t count = 0;
/*
  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < opts->nchan; ichan++)
    {
      for (iant=0; iant < opts->nant; iant++)
      {
        if ((ichan < schan) || (ichan > echan))
        {
          re_sum += (int64_t) in[0];
          im_sum += (int64_t) in[1];
          count ++;
        }
        in += 2;
      }
    }
  }
  fprintf (stderr, "re_sum=%"PRIi64" im_sun=%"PRIi64"\n", re_sum, im_sum);
*/

  int8_t re_mean = (int8_t) (re_sum / count);
  int8_t im_mean = (int8_t) (im_sum / count);

  in = (int8_t *) buffer;
  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < opts->nchan; ichan++)
    {
      for (iant=0; iant < opts->nant; iant++)
      {
        if ((ichan >= schan) && (ichan <= echan))
        {
          in[0] = 0;
          in[1] = 0;
        }
        in += 2;
      }
    }
  }
}

void mopsr_form_histogram (unsigned int * histogram, char * buffer, ssize_t bytes,  mopsr_util_t * opts)
{
  unsigned ibin = 0;
  unsigned ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  unsigned nframe = bytes / (opts->nant * opts->nchan * 2);

  int8_t * in = (int8_t *) buffer;
  int8_t re, im;
  int re_bin, im_bin;

  unsigned int re_out_index;
  unsigned int im_out_index;

  // set re and im bins to 0 [idim][ibin]
  for (ibin=0; ibin < opts->nbin; ibin++)
  {
    histogram[ibin] = 0;                  // Re
    histogram[opts->nbin + ibin] = 0;     // Im
  }

  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < opts->nchan; ichan++)
    {
      for (iant=0; iant < opts->nant; iant++)
      {
        if (iant == opts->ant)
        {
          if ((opts->chans[0] == -1) || ((opts->chans[0] >= 0) && (ichan >= opts->chans[0]) && (ichan <= opts->chans[1])))
          {
            re = in[0];
            im = in[1];

            // determine the bin for each value [-128 -> 127]
            re_bin = ((int) re) + 128;
            im_bin = ((int) im) + 128;

            if (opts->nbin != 256)
            {
              // bin range is now 0 -> 256
              re_bin *= opts->nbin;
              im_bin *= opts->nbin;

              re_bin /= 256;
              im_bin /= 256;
            }

            if (re_bin > 255)
              re_bin == 255;
            if (im_bin > 255)
              im_bin == 255;

            histogram[re_bin]++;
            histogram[opts->nbin + im_bin]++;
          }
        }
        in += 2;
      }
    }
  }

  /*
  char str[9];
  int i;
  char c;
  for (i=0; i < 256; i++)
  {
    c = (char) (i - 128);
    char_to_bstring (str, c);
    fprintf(stderr, "bin[%d]= %d, c=%d binary=%s\n", i, histogram[opts->nbin + i], c, str);
  }
  */
}

void mopsr_extract_channel (float * timeseries, char * buffer, ssize_t bytes,
                            unsigned int channel, unsigned int antenna, 
                            unsigned int nchan, unsigned int nant)
{
  unsigned iframe, ichan, iant;
  int8_t * input = (int8_t *) buffer;
  float re, im;
  float * output = timeseries;
  unsigned int nframe = bytes / (nchan * nant * MOPSR_NDIM);

  for (iframe=0; iframe < nframe; iframe++)
  {
    for (ichan=0; ichan < nchan; ichan++)
    {
      for (iant=0; iant < nant; iant++)
      {
        if ((channel == ichan) && (antenna == iant))
        {
          re = (float) input[0];
          im = (float) input[1];
          output[0] = re;
          output[1] = im;
          output += 2;
        }
        input += 2;
      }
    }
  }
}

void dump_packet (char * buffer, unsigned int size)
{
 int flags = O_WRONLY | O_CREAT | O_TRUNC;
 int perms = S_IRUSR | S_IRGRP;
 int fd = open ("packet.raw", flags, perms);
 ssize_t n_wrote = write (fd, buffer, size);
 close (fd);
}

void print_packet (char * buffer, unsigned int size)
{
  char ant0r[9]; 
  char ant0i[9];
  char ant1r[9];
  char ant1i[9];
  
  unsigned ibyte = 0;
  fprintf(stderr, "chan\tant0imag ant0real ant1imag ant1real\n");
  for (ibyte=0; ibyte<size; ibyte += 4)
  {
    char_to_bstring (ant0r, buffer[ibyte+0]);
    char_to_bstring (ant0i, buffer[ibyte+1]);
    char_to_bstring (ant1r, buffer[ibyte+2]);
    char_to_bstring (ant1i, buffer[ibyte+3]);
    fprintf (stderr, "%d\t%s %s %s %s\t(%d + %dj)\t(%d+i%dj)\n", ibyte/4, ant0r, ant0i, ant1r, ant1i, (int8_t) buffer[ibyte+0], (int8_t) buffer[ibyte+1], (int8_t) buffer[ibyte+2], (int8_t) buffer[ibyte+3]);
  }
}

void mopsr_plot_time_series (float * timeseries, unsigned int chan, unsigned int nsamps, mopsr_util_t * opts)
{
  float xmin = 0.0;
  float xmax = (float) nsamps;
  float ymin = -129;
  float ymax = 129;

  if (opts->ymin == 0)
    ymin = 0;
  if (opts->ymax == 0)
    ymax = 0;

  float x[nsamps];
  float re[nsamps];
  float im[nsamps];
  char label[64];

  float re_mean = 0;
  float im_mean = 0;
  char all_zero = 1;
  float re_sum_sq = 0;
  float im_sum_sq = 0;

  unsigned int i;
  for (i=0; i<nsamps; i++)
  {
    x[i] = (float) i;
    re[i] = timeseries[2*i];
    im[i] = timeseries[2*i+1];
    re_mean += re[i];
    im_mean += im[i];
    re_sum_sq += re[i]*re[i];
    im_sum_sq += im[i]*im[i];
    if ((re[i] != 0) || (im[i] != 0))
      all_zero = 0;
    if (re[i] > ymax)
      ymax = re[i];
    if (im[i] > ymax)
      ymax = im[i];
    if (re[i] < ymin)
      ymin = re[i];
    if (im[i] < ymin)
      ymin = im[i];
  }

  re_mean /= xmax;
  im_mean /= xmax;

  float re_rms = sqrt(re_sum_sq/xmax);
  float im_rms = sqrt(im_sum_sq/xmax);

  // Now that we have the mean, recompute v^2 for average power
  // and display it in the TimeSeries plots

  // reset sum of squares
  re_sum_sq = 0.0;
  im_sum_sq = 0.0;

  for (i=0; i<nsamps; i++)
  {
    re_sum_sq += pow(re[i]-re_mean,2.0);
    im_sum_sq += pow(im[i]-im_mean,2.0);
  }

  float power_mean = (re_sum_sq + im_sum_sq)/nsamps;

  cpgbbuf();
  cpgsci(1);
  cpgslw(1);
  int symbol = -1;

  if (opts->plot_plain)
  {
    cpgpage();
    cpgswin(xmin, xmax, ymin, ymax);
    cpgsvp(0, 1, 0, 1);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  }
  else
  {
    cpgenv(xmin, xmax, (ymin - 1), (ymax + 1), 0, 0);
    sprintf (label, "Complex Timeseries for Channel %d, Module %u", chan, opts->ant_id);
    cpglab("Time", "Voltage", label);
  }

  // Real line
  cpgsci(2);
  sprintf (label, "Real: mean %5.2f error %6.2f", re_mean, re_rms/sqrt(xmax));
  cpgmtxt("T", -2.5, 0.0, 0.0, label);
  cpgslw(5);
  cpgpt (nsamps, x, re, symbol);
  cpgslw(1);

  cpgsci(3);
  sprintf (label, "Imag: mean %5.2f error %6.2f", im_mean, im_rms/sqrt(xmax));
  cpgmtxt("T", -3.5, 0.0, 0.0, label);
  cpgslw(5);
  cpgpt (nsamps, x, im, symbol);
  cpgslw(1);

  char power_str[100];
  sprintf(power_str, "Power: %6.1f",power_mean);
  if (opts->plot_plain)
    cpgsch(7.0);
  else
    cpgsch(2.0);
  cpgscf(2);
  cpgsci(1);
  cpgslw(3);
  cpgmtxt("B",-0.2,0.05, 0.0, power_str);
  cpgslw(1);
  cpgsch(1.0);
  cpgscf(1);

  float sigma_worst = 0.0;
  if (re_rms>im_rms) sigma_worst = re_rms/sqrt(xmax);
  if (re_rms<=im_rms) sigma_worst = im_rms/sqrt(xmax);

  if (fabs(re_mean-im_mean)/(sigma_worst)>5) {
    sprintf(label, "Difference worse than 5 sigma!");
    cpgsch(2.0);
    cpgsci(2);
    cpgslw(5);
    cpgscf(2);
    cpgmtxt("T",-10, 0.0, 0.0, label);
    cpgsch(1);
    cpgsci(1);
    cpgslw(1);
    cpgscf(1);
  }

  cpgsci(1);

  float yrange = (ymax-ymin);

  if (all_zero)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(nsamps/2, ymin + (yrange * 0.25), 0.0, 0.5, "All Data Zero");
    cpgsch(1);
    cpgslw(1);
  }

  if (opts->lock_flag == 0)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(nsamps/2, ymin + (yrange * 0.5), 0.0, 0.5, "No Lock");
    cpgslw(1);
    cpgsch(1);
  }

  if (opts->lock_flag_long == 0)
  {
    cpgsch(5);
    cpgslw(5);
    cpgptxt(nsamps/2, ymin + (yrange * 0.75), 0.0, 0.5, "Unstable Lock");
    cpgslw(1);
    cpgsch(1);
  }

  cpgebuf();
}

void mopsr_plot_bandpass (float * power_spectra, mopsr_util_t * opts)
{
  unsigned i=0;

  float * x_points = (float *) malloc (sizeof(float) * opts->nchan);
  for (i=0; i<opts->nchan; i++)
    x_points[i] = (float) i;

  size_t ant_label_size = 16;
  char * ant_label = (char *) malloc (sizeof(char) * ant_label_size);
  sprintf (ant_label, "Module %d", opts->ant_id);

  float xmin = 0;
  float xmax = (float) opts->nchan;
  float ymin = FLT_MAX;
  float ymax = -1 * FLT_MAX;
  char all_zero = 0;

  unsigned ichan;
  float val;
  float mean = 0;
  for (ichan=0; ichan<opts->nchan; ichan++)
  {
    val = power_spectra[ichan];
    if (opts->plot_log && val > 0)
      val = log10(val);
    if (opts->zap && ichan == 0)
      val = 0;
    if (val > ymax) ymax = val;
    if (val < ymin) ymin = val;

    power_spectra[ichan] = val;
    mean += val;
  }
  if (opts->zap)
  {
    mean /= opts->nchan;
    power_spectra[0] = mean;
  }

  if (ymin == ymax)
  {
    all_zero = 1;
    ymax = 1;
  }

  cpgbbuf();
  cpgsci(1);

  if (opts->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);
  cpgswin(xmin, xmax, ymin, ymax);

  if (opts->plot_log)
    cpgbox("BCNST", 0.0, 0.0, "BCNSTL", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);

  if (!opts->plot_plain)
    if (opts->plot_log)
      cpglab("Channel", "log\\d10\\u(Power)", "Bandpass");
    else
      cpglab("Channel", "Power", "Bandpass");

  float yrange = (ymax-ymin);

  if (all_zero)
  {
    cpgsch(5);
    cpgptxt(opts->nchan/2, ymin + (yrange * 0.25), 0.0, 0.5, "All Data Zero");
    cpgsch(1);
  }

  cpgsci(1);
  cpgmtxt("T", 1.5, 0.0, 0.0, ant_label);
  cpgslw(1);
  cpgline(opts->nchan, x_points, power_spectra);
  cpgslw(1);

  if (opts->lock_flag == 0)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(opts->nchan/2, ymin + (yrange * 0.5), 0.0, 0.5, "No Lock");
    cpgslw(1);
    cpgsch(1);
  }

  if (opts->lock_flag_long == 0)
  {
    cpgsch(5);
    cpgslw(5);
    cpgptxt(opts->nchan/2, ymin + (yrange * 0.75), 0.0, 0.5, "Unstable Lock");
    cpgslw(1);
    cpgsch(1);
  }


  cpgebuf();
  
  free (x_points);
  free (ant_label);
}

void mopsr_plot_bandpass_vertical (float * power_spectra, mopsr_util_t * opts)
{
  unsigned i=0;

  float * y_points = (float *) malloc (sizeof(float) * opts->nchan);
  for (i=0; i<opts->nchan; i++)
  for (i=0; i<opts->nchan; i++)
    y_points[i] = (float) i;

  size_t ant_label_size = 16;
  char * ant_label = (char *) malloc (sizeof(char) * ant_label_size);
  sprintf (ant_label, "Module %d", opts->ant_id);

  float ymin = 0;
  float ymax = (float) opts->nchan;
  float xmin = 0;
  float xmax = 0;

  unsigned ichan;
  float val;
  for (ichan=0; ichan<opts->nchan; ichan++)
  {
    val = power_spectra[ichan];
    if (opts->plot_log && val > 0)
      val = log10(val);
    if (opts->zap && ichan == 0)
      val = 0;
    if (val > xmax) xmax = val;
    if (val < xmin) xmin = val;
    power_spectra[ichan] = val;
  }

  cpgbbuf();
  cpgsci(1);

  if (opts->plot_plain)
  {
     cpgsvp(0.0,1.0,0.0,1.0);
  }
  else
  {
    cpgsvp(0.1,0.9,0.1,0.9);
  }
  cpgswin(xmin, xmax, ymin, ymax);

  if (opts->plot_log)
    cpgbox("BCNSTL", 0.0, 0.0, "BCNST", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);

  if (!opts->plot_plain)
    if (opts->plot_log)
      cpglab("log\\d10\\u(Power)", "Channel", "Bandpass");
    else
      cpglab("Power", "Channel", "Bandpass");

  cpgsci(2);
  cpgmtxt("T", 1.5, 0.0, 0.0, ant_label);
  cpgline(opts->nchan, power_spectra, y_points);
  cpgebuf();

  opts->ymin = xmin;
  opts->ymax = xmax;

  free (y_points);
  free (ant_label);
}


/*
 *  plot waterfall plot assuming
 */
void mopsr_plot_waterfall_vertical (float * spectra, unsigned int nsamps, mopsr_util_t * opts)
{
  cpgbbuf();
  cpgsci(1);

  char label[64];
  if (opts->plot_log)
    sprintf (label, "Waterfall: Log, Module %d", opts->ant_id);
  else
    sprintf (label, "Waterfall: Module %d", opts->ant_id);


  if (opts->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);
  cpgswin(0, (float) opts->nchan, 0, (float) nsamps);

  //cpgenv(0, (float) opts->nchan, 0, (float) nsamps, 0, 0);
  cpglab("Channel", "Time (samples)", label);

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float length = (1.08 * nsamps) / 1000000;

  float x_min = 0;
  float x_max = opts->nchan;

  float y_min = 0;
  float y_max = nsamps;

  float x_res = (x_max-x_min)/opts->nchan;
  float y_res = (y_max-y_min)/nsamps;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = opts->nchan * nsamps;
  float z_min = 100000000000;
  float z_max = 1;
  float z_avg = 0;

  if (opts->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opts->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int ichan, isamp;
  unsigned int i;
  unsigned int ndat_avg = 0;
  for (isamp=0; isamp<nsamps; isamp++)
  {
    for (ichan=0; ichan<opts->nchan; ichan++)
    {
      i = isamp * opts->nchan + ichan;
      if (opts->plot_log && spectra[i] > 0)
        spectra[i] = log10(spectra[i]);
        if ((!opts->zap) || (opts->zap && ichan != 0))
        {
          if (spectra[i] > z_max) z_max = spectra[i];
          if (spectra[i] < z_min) z_min = spectra[i];
        }
        z_avg += spectra[i];
        ndat_avg++;
    }
  }

  z_avg /= (float) ndat_avg;

  z_min = z_max - (2 * z_avg);
  z_max *= 2;

  if (opts->zap)
  {
    for (isamp=0; isamp<nsamps; isamp++)
    {
      for (ichan=0; ichan<opts->nchan; ichan++)
      {
        i = isamp * opts->nchan + ichan;
        if (ichan == 0)
          spectra[i] = z_avg;
      }
    }
  }

  cpgimag(spectra, opts->nchan, nsamps, 1, opts->nchan, 1, nsamps, z_min, z_max, trf);
  cpgebuf();
}

void mopsr_plot_delay_snrs(float * snrs, mopsr_util_t * opts)
{
  cpgbbuf();
  cpgsci(1);

  char label[64];
  if (opts->plot_log)
    sprintf (label, "SNR (log)");
  else
    sprintf (label, "SNR");

  if (opts->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);

  cpgswin(0, (float) opts->nant, 0, (float) opts->nant);

  cpglab("Antenna", "Antenna", label);

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float x_min = 0;
  float x_max = (float) opts->nant;

  float y_min = 0;
  float y_max = (float) opts->nant;

  float x_res = (x_max-x_min)/x_max;
  float y_res = (y_max-y_min)/y_max;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = opts->nant * opts->nant;
  float z_min = 100000000000;
  float z_max = 1;
  float z_avg = 0;

  if (opts->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opts->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int iant, jant; 
  unsigned int i;
  unsigned int ndat_avg = 0;
  for (iant=0; iant<opts->nant; iant++)
  {
    for (jant=0; jant<opts->nant; jant++)
    {
      i = iant * opts->nant + jant;
      if (opts->plot_log && snrs[i] > 0)
        snrs[i] = log10(snrs[i]);
      if ((!opts->zap) || (opts->zap && jant != 0))
      {
        if (snrs[i] > z_max) z_max = snrs[i];
        if (snrs[i] < z_min) z_min = snrs[i];
      }
      z_avg += snrs[i];
      ndat_avg++;
    }
  }

  z_avg /= (float) ndat_avg;
  z_min = z_max - (2 * z_avg);
  z_max *= 2;

  cpgimag(snrs, opts->nant, opts->nant, 1, opts->nant, 1, opts->nant, z_min, z_max, trf);
  cpgebuf();
}


/*
 *  plot waterfall plot assuming
 */
void mopsr_plot_waterfall (float * spectra, unsigned int nsamps, mopsr_util_t * opts)
{
  cpgbbuf();
  cpgsci(1);

  char label[64];

  if (opts->plot_plain)
  {
    cpgsvp(0, 1, 0, 1);
    cpgswin(0, (float) nsamps, 0, (float) opts->nchan);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  }
  else
  {
    if (opts->plot_log)
      sprintf (label, "Waterfall: Log, Module %d", opts->ant_id);
    else
      sprintf (label, "Waterfall:, Module %d", opts->ant_id);

    cpgenv(0, (float) nsamps, 0, (float) opts->nchan, 0, 0);
    cpglab("Time (samples)", "Channel", label);
  }

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float length = (1.08 * opts->nchan) / 1000000;

  float x_min = 0;
  float x_max = nsamps;

  float y_min = 0;
  float y_max = opts->nchan;

  float x_res = (x_max-x_min)/nsamps;
  float y_res = (y_max-y_min)/opts->nchan;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = nsamps * opts->nchan;
  float z_min = 10000000;
  float z_max = 0;
  float z_avg = 0;

  if (opts->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opts->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int ichan, isamp;
  unsigned int i;
  unsigned int ndat_avg = 0;
  float val;

  for (ichan=0; ichan<opts->nchan; ichan++)
  {
    for (isamp=0; isamp<nsamps; isamp++)
    {
      i = ichan * nsamps + isamp;
      if (opts->plot_log && spectra[i] > 0)
        spectra[i] = log10(spectra[i]);
      if ((!opts->zap) || (opts->zap && ichan != 0))
      {
        if (spectra[i] > z_max) z_max = spectra[i];
        if (spectra[i] < z_min) z_min = spectra[i];
        z_avg += spectra[i];
        ndat_avg++;
      }
    }
  }

  z_avg /= (float) ndat_avg;

  if (opts->zap)
  {
    for (ichan=0; ichan<opts->nchan; ichan++)
    {
      for (isamp=0; isamp<nsamps; isamp++)
      {
        i = ichan * nsamps + isamp;
        if (ichan == 0)
          spectra[i] = z_avg;
      }
    }
  }

  float yrange = opts->nchan;

  if (z_min == z_max)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(nsamps/2,yrange * 0.25, 0.0, 0.5, "All Data Zero");
    cpgsch(1);
  }
  else
    cpgimag(spectra, nsamps, opts->nchan, 1, nsamps, 1, opts->nchan, z_min, z_max, trf);

  if (opts->lock_flag == 0)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(nsamps/2, yrange * 0.5, 0.0, 0.5, "No Lock");
    cpgslw(1);
    cpgsch(1);
  }

  if (opts->lock_flag_long == 0)
  {
    cpgsch(5);
    cpgslw(5);
    cpgptxt(nsamps/2, yrange * 0.75, 0.0, 0.5, "Unstable Lock");
    cpgslw(1);
    cpgsch(1);
  }


  cpgebuf();
}

void mopsr_transpose (float * out, float * in, unsigned int nsamps, mopsr_util_t * opts)
{
  unsigned int isamp, ichan;
  for (isamp=0; isamp < nsamps; isamp++)
  {
    for (ichan=0; ichan < opts->nchan; ichan++)
    {
      out[ichan * nsamps + isamp] = in[isamp * opts->nchan + ichan];
    }
  }
}

void mopsr_plot_histogram (unsigned int * histogram, mopsr_util_t * opts)
{

  float x[opts->nbin];
  float re[opts->nbin];
  float im[opts->nbin];
  int ibin;
  int ichan, iant;

  float ymin = 0;
  float ymax_re= 0;
  float ymax_im = 0;
  char all_zero = 1;

  cpgeras();

  for (ibin=0; ibin<opts->nbin; ibin++)
  {
    x[ibin] = ((float) ibin * (256 / opts->nbin)) - 128;
    re[ibin] = (float) histogram[ibin];
    im[ibin] = (float) histogram[opts->nbin + ibin];

    if (re[ibin] > ymax_re)
    {
      ymax_re = re[ibin];
     }
    if (im[ibin] > ymax_im)
    {
      ymax_im = im[ibin];
    }
    if ((x[ibin] != 0) && (re[ibin] > 0 || im[ibin] > 0))
      all_zero = 0;
  }
  cpgbbuf();
  cpgsci(1);

  char title[128];
  char label[64];

  if (opts->chans[0] < 0)
    sprintf(title, "Module %d Histogram, all channels", opts->ant_id);
  else if (opts->chans[0] == opts->chans[1])
    sprintf(title, "Module %d Histogram, channel %d", opts->ant_id, opts->chans[0]);
  else
    sprintf(title, "Module %d Histogram, channels %d to %d", opts->ant_id, opts->chans[0], opts->chans[1]);

  ymax_re *= 1.1;
  ymax_im *= 1.1;

  // real
  cpgswin(-128, 128, ymin, ymax_re);
  if (opts->plot_plain)
  {
    cpgsvp(0, 1, 0.5, 1.0);
    cpgbox("BC", 0.0, 0.0, "BC", 0.0, 0.0);
    cpgsch(3);
    cpgmtxt("T", -2.0, 0.05, 0.0, "Real");
    cpgsch(1);
  }
  else
  {
    cpgsvp(0.1, 0.9, 0.5, 0.9);
    cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
    cpglab("", "Count", title);
  }

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_re);
  cpgsls(1);

  // Real line
  if (!opts->plot_plain)
    cpgmtxt("T", -2.2, 0.05, 0.0, "Real");
  cpgsci(2);
  cpgslw(3);
  cpgbin (opts->nbin, x, re, 0);
  cpgslw(1);
  cpgsci(1);

  float yrange = ymax_re;
  if (opts->lock_flag == 0)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(0, 0, 0.0, 0.5, "No Lock");
    cpgslw(1);
    cpgsch(1);
  }

  if (opts->lock_flag_long == 0)
  {
    cpgsch(5);
    cpgslw(5);
    cpgptxt(0, 0.5 * yrange, 0.0, 0.5, "Unstable Lock");
    cpgslw(1);
    cpgsch(1);
  }

  if (opts->plot_plain)
  {
    cpgsvp(0, 1, 0, 0.5);
    cpgbox("BNC", 0.0, 0.0, "BNC", 0.0, 0.0);
    cpgsch(3);
    cpgmtxt("T", -2.0, 0.05, 0.0, "Imag");
    cpgsch(1);
  }
  else
  {
    cpgsvp(0.1, 0.9, 0.1, 0.5);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
    cpglab("State", "Count", "");
  }
  cpgswin(-128, 128, ymin, ymax_im);

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_im);
  cpgsls(1);

  // Im line
  if (!opts->plot_plain)
    cpgmtxt("T", -2.2, 0.05, 0.0, "Imag");
  cpgslw(3);
  cpgsci(3);
  cpgbin (opts->nbin, x, im, 0);
  cpgsci(1);
  cpgslw(1);

  yrange = ymax_im;
  if (all_zero)
  {
    cpgsch(6);
    cpgptxt(0, yrange * 0.5, 0.0, 0.5, "All Data Zero");
    cpgsch(1);
  }

  cpgebuf();
}

void mopsr_zero_float (float * array, unsigned int size)
{
  unsigned int i;
  for (i=0; i<size; i++)
    array[i] = 0;
}

void mopsr_zero_uint (unsigned int * array, unsigned int size)
{
  unsigned int i;
  for (i=0; i<size; i++)
    array[i] = 0;
}

void get_scale (int from, int to, float * width, float * height)
{
  float j = 0;
  float fx, fy;
  cpgqvsz (from, &j, &fx, &j, &fy);

  float tx, ty;
  cpgqvsz (to, &j, &tx, &j, &ty);

  *width = tx / fx;
  *height = ty / fy;
}

void set_resolution (int width_pixels, int height_pixels)
{
  float width_scale, height_scale;
  width_pixels--;
  height_pixels--;

  get_scale (3, 1, &width_scale, &height_scale);

  float width_inches = width_pixels * width_scale;
  float aspect_ratio = height_pixels * height_scale / width_inches;

  cpgpap( width_inches, aspect_ratio );

  float x1, x2, y1, y2;
  cpgqvsz (1, &x1, &x2, &y1, &y2);
}

void set_white_on_black()
{
  cpgscr(0, 1, 1, 1);
  cpgscr(1, 0, 0, 0);
}
