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

#include "mopsr_util.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"

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

  fprintf (stderr, "sqld_pfbframe: nchan=%u, nant=%u\n", opts->nchan, opts->nant);

  for (ichan=0; ichan < opts->nchan; ichan++)
  {
    for (iant=0; iant < opts->nant; iant++)
    {
      re = (int) input[0];
      im = (int) input[1];
      if (flag == 0)
        power_spectra[(iant * opts->nchan) + ichan]  = (float) ((re*re) + (im*im));
      else  
        power_spectra[(iant * opts->nchan) + ichan] += (float) ((re*re) + (im*im));
      input += 2;
    }
  }
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
          re = (float) input[1];
          im = (float) input[0];
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


void mopsr_plot_time_series (float * timeseries, unsigned int nsamps, mopsr_util_t * opts)
{
  float xmin = 0.0;
  float xmax = (float) nsamps;

  float x[nsamps];
  float re[nsamps];
  float im[nsamps];

  float re_mean = 0;
  float im_mean = 0;

  unsigned int i;
  for (i=0; i<nsamps * MOPSR_NDIM; i++)
  {
    if (timeseries[i] > opts->ymax) opts->ymax = timeseries[i];
    if (timeseries[i] < opts->ymin) opts->ymin = timeseries[i];
  }
  for (i=0; i<nsamps; i++)
  {
    x[i] = (float) i;
    re[i] = timeseries[2*i];
    im[i] = timeseries[2*i+1];
    re_mean += re[i];
    im_mean += im[i];
  }

  re_mean /= xmax;
  im_mean /= xmax;

  cpgbbuf();
  cpgsci(1);
  cpgslw(1);
  int symbol = -1;

  char label[64];
  sprintf (label, "Complex Timeseries for Ant %u", mopsr_get_ant_number(opts->ant_code, opts->ant));

  cpgenv(xmin, xmax, (opts->ymin - 1), (opts->ymax + 1), 0, 0);
  cpglab("Time", "Volatage", label);

  // Real line
  cpgsci(2);
  sprintf (label, "Real: mean %5.2f", re_mean);
  cpgmtxt("T", 0.5, 0.0, 0.0, label);
  cpgslw(5);
  cpgpt (nsamps, x, re, symbol);
  cpgslw(1);

  cpgsci(3);
  sprintf (label, "Imag: mean %5.2f", im_mean);
  cpgmtxt("T", 1.5, 0.0, 0.0, label);
  cpgslw(5);
  cpgpt (nsamps, x, im, symbol);
  cpgslw(1);

  cpgebuf();
}

void mopsr_plot_bandpass (float * power_spectra, mopsr_util_t * opts)
{
  unsigned i=0;

  float x_points[opts->nchan];
  for (i=0; i<opts->nchan; i++)
    x_points[i] = (float) i;

  float xmin = 0;
  float xmax = (float) opts->nchan;

  unsigned ichan, iant;
  float val;
  for (iant=0; iant<opts->nant; iant++)
  {
    for (ichan=0; ichan<opts->nchan; ichan++)
    {
      val = power_spectra[(iant * opts->nchan) + ichan];
      if (opts->plot_log)
      {
        if (val > 0)
          val = log10(val);
        else
          val = 0;
        if (opts->zap && ichan == 0)
          val = 0;
        power_spectra[(iant * opts->nchan) + ichan] = val;
      }
      if (val > opts->ymax) opts->ymax = val;
      if (val < opts->ymin) opts->ymin = val;
    }
  }

  cpgbbuf();
  cpgsci(1);
  if (opts->plot_log)
  {
    cpgenv(xmin, xmax, 1.1*opts->ymin, 1.1*opts->ymax, 0, 20);
    cpglab("Channel", "log\\d10\\u(Power)", "Bandpass");
  }
  else
  {
    cpgenv(xmin, xmax, 1.1*opts->ymin, 1.1*opts->ymax, 0, 0);
    cpglab("Channel", "Power", "Bandpass");
  }

  fprintf (stderr, "xrange [%f:%f]\n", xmin, xmax);
  fprintf (stderr, "yrange [%f:%f]\n", opts->ymin, opts->ymax);

  char ant_label[8];
  for (iant=0; iant < opts->nant; iant++)
  {
    sprintf(ant_label, "Ant %d", iant);
    cpgsci(iant + 2);
    cpgmtxt("T", 1.5 + (1.0 * iant), 0.0, 0.0, ant_label);
    cpgline(opts->nchan, x_points, power_spectra + (iant * opts->nchan));
  }
  cpgebuf();
}

/*
 *  plot waterfall plot assuming [
 */
void mopsr_plot_waterfall (float * spectra, unsigned int nsamps, mopsr_util_t * opts)
{
  cpgbbuf();
  cpgsci(1);

  cpgenv(0, (float) opts->nchan, 0, (float) nsamps, 0, 0);
  cpglab("Channel", "Integration", "Waterfall");

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

  fprintf (stderr, "plot: z_min=%f z_max=%f z_avg=%f\n", z_min, z_max, z_avg);

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
