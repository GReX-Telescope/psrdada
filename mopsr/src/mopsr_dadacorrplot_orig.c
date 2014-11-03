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

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage ();
void plot_delays (float * yvals, unsigned npts, char * device);
void plot_power_spectrum (float * yvals, unsigned npts, char * device);
void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot [options] file1 file2\n"
     " files must be single antenna, single channel files\n"
     " -b nsub-band    number of sub-bands to create from single channel data [default 5]\n"
     " -c sub-band     sub-band to use in cross-correlation [default 1]\n"
     " -D device       pgplot device name\n"
     " -n npt          number of points in cross-correlation signal [default 1024]\n"
     " -t nsets        number of sets of data to add in default [1]\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int nsubband = 1024;

  unsigned int subband = 400;

  unsigned int npt = 1024;

  unsigned int nsets = 1;

  while ((arg=getopt(argc,argv,"b:c:D:hn:t:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        nsubband = atoi(optarg);
        break;

      case 'c':
        subband = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'n':
        npt = atoi(optarg);
        break;

      case 't': 
        nsets = atoi (optarg);
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

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char file1[1024];
  char file2[1024];

  strcpy(file1, argv[optind]);
  strcpy(file2, argv[optind+1]);

  size_t file1_size, file2_size;
  struct stat buf;
  if (stat (file1, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file1, strerror(errno));
    return (EXIT_FAILURE);
  }
  file1_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", file1, file1_size);

  if (stat (file2, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file2, strerror(errno));
    return (EXIT_FAILURE);
  }
  file2_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", file2, file2_size);

  if (file1_size != file2_size)
  {
    fprintf (stderr, "file sizes differed\n");
    //return (EXIT_FAILURE);
  }

  // number of bytes required to read file into memory
  size_t bytes_packed = npt * MOPSR_NDIM * sizeof(int8_t) * nsubband;
  if (verbose)
    fprintf (stderr, "bytes_packed=%ld\n", bytes_packed);
  if (bytes_packed > file1_size - 4096)
  {
    fprintf (stderr, "file size not big enough for requested operations\n");
    return (EXIT_FAILURE);
  }

  int8_t * file1_raw = (int8_t *) malloc(bytes_packed);
  int8_t * file2_raw = (int8_t *) malloc(bytes_packed);

  if (verbose)
    fprintf (stderr, "reading file1\n");

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (file1, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file1, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t bytes_read;
  off_t offset = 4096;
  // skip the 4K ascii header
  lseek (fd1, offset, SEEK_SET);

  if (verbose)
    fprintf (stderr, "reading file2\n");
  int fd2 = open (file2, flags, perms);
  if (fd2 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file2, strerror(errno));
    exit(EXIT_FAILURE);
  }

  // skip the 4K ascii header
  size_t delay = 0;
  offset = 4096 + (2 * nsubband * delay);
  lseek (fd2, offset, SEEK_SET);

  if (verbose)
    fprintf (stderr, "preparing fft memory\n");

  // max size for fft buffer
  size_t fft_buffer_bytes = sizeof(fftwf_complex) * nsubband;
  fprintf (stderr, "filterbank size bytes=%ld\n", fft_buffer_bytes);
  fftwf_complex * fb_fft_in  = (fftwf_complex*) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * fb_fft_out = (fftwf_complex*) fftwf_malloc (fft_buffer_bytes);

  float * fb1_spectrum = (float *) malloc (sizeof(float) * nsubband);
  float * fb2_spectrum = (float *) malloc (sizeof(float) * nsubband);
  memset (fb1_spectrum, 0, sizeof(float) * nsubband);
  memset (fb2_spectrum, 0, sizeof(float) * nsubband);

  int direction_flags = FFTW_FORWARD;
  flags = FFTW_ESTIMATE;

  // filterbank plan
  fftwf_plan plan_fb = fftwf_plan_dft_1d (nsubband, fb_fft_in, fb_fft_out, direction_flags, flags);

  unsigned npad = 2 * npt -1;

  // for cross-correlation we actually require 2*npt-1 points 0 padded 
  fft_buffer_bytes = sizeof(fftwf_complex) * npad;
  fftwf_complex * file1_fft_in  = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * file1_fft_out = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * file2_fft_in  = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * file2_fft_out = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * bwd_in        = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * bwd_out       = (fftwf_complex *) fftwf_malloc (fft_buffer_bytes);
  fftwf_complex * result        = (float complex *) malloc (fft_buffer_bytes);
  fftwf_complex * result_phases = (float complex *) malloc (fft_buffer_bytes);
  float * result_power          = (float *) malloc (sizeof(float) * npad);
  float * phases                = (float *) malloc (sizeof(float) * npad);

  memset (result, 0, fft_buffer_bytes);
  memset (result_phases, 0, fft_buffer_bytes);

  float * file1_spectrum = (float *) malloc (sizeof(float) * npad);
  float * file2_spectrum = (float *) malloc (sizeof(float) * npad);
  memset (file1_spectrum, 0, sizeof(float) * npad);
  memset (file2_spectrum, 0, sizeof(float) * npad);

  if (verbose)
    fprintf (stderr, "ffts for sub-bands\n");

  // memset each signal 
  if (npad == 2 * npt -1)
  {
    memset (file1_fft_in, 0, sizeof(fftwf_complex) * (npt - 1));
    memset (file2_fft_in + npt, 0, sizeof(fftwf_complex) * (npt - 1));
  }

  // now prepare a fft plan for the  fwds and inverse transform
  fftwf_plan plan_fwd = fftwf_plan_dft_1d (npad, file1_fft_in, file1_fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_bwd = fftwf_plan_dft_1d (npad, bwd_in, bwd_out, FFTW_BACKWARD, FFTW_ESTIMATE);

  fftwf_complex scale = 1.0/ ((float) npad);
  if (verbose)
    fprintf (stderr, "scale=%f + %fi\n", creal(scale), cimag(scale));

  unsigned ipt, iband, iset;
  float re, im;
  for (iset=0; iset<nsets; iset++)
  {
    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from file %s offset=%d\n", bytes_packed, file1, offset);
    bytes_read = read (fd1, (void *) file1_raw, bytes_packed);

    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from file %s offset=%d\n", bytes_packed, file2, offset);
    bytes_read = read (fd2, (void *) file2_raw, bytes_packed);

    // now perform sub-band ffts on file1 data
    for (ipt=0; ipt<npt; ipt++)
    {
      for (iband=0; iband<nsubband; iband++)
      {
        re = ((float) file1_raw[((ipt*nsubband+iband)*2)+0]) + 0.5;
        im = ((float) file1_raw[((ipt*nsubband+iband)*2)+1]) + 0.5;
        fb_fft_in[iband] = re + im * I;
      }
      fftwf_execute_dft (plan_fb, fb_fft_in, fb_fft_out);

      for (iband=0; iband<nsubband; iband++)
        fb1_spectrum[iband] += creal(fb_fft_out[iband]) * creal(fb_fft_out[iband]) + cimag(fb_fft_out[iband]) * cimag(fb_fft_out[iband]);

      if (npad == 2* npt - 1)
        file1_fft_in[(npt-1)+ipt] = fb_fft_out[subband];
      else
        file1_fft_in[ipt] = fb_fft_out[subband];

      for (iband=0; iband<nsubband; iband++)
      {
        re = ((float) file2_raw[((ipt*nsubband+iband)*2)+0]) + 0.5;
        im = ((float) file2_raw[((ipt*nsubband+iband)*2)+1]) + 0.5;
        fb_fft_in[iband] = re + im * I;
      }
      fftwf_execute_dft (plan_fb, fb_fft_in, fb_fft_out);
      for (iband=0; iband<nsubband; iband++)
        fb2_spectrum[iband] += creal(fb_fft_out[iband]) * creal(fb_fft_out[iband]) + cimag(fb_fft_out[iband]) * cimag(fb_fft_out[iband]);
      file2_fft_in[ipt] = fb_fft_out[subband];
    }

    fftwf_execute_dft (plan_fwd, file1_fft_in, file1_fft_out);
    fftwf_execute_dft (plan_fwd, file2_fft_in, file2_fft_out);

    for (ipt=0; ipt<npad; ipt++)
    {
      file1_spectrum[ipt] += (creal(file1_fft_out[ipt]) * creal(file1_fft_out[ipt]) + cimag(file1_fft_out[ipt]) * cimag(file1_fft_out[ipt]));
      file2_spectrum[ipt] += (creal(file2_fft_out[ipt]) * creal(file2_fft_out[ipt]) + cimag(file2_fft_out[ipt]) * cimag(file2_fft_out[ipt]));
    }

    for (ipt=0; ipt<npad; ipt++)
    {
      bwd_in[ipt] = file1_fft_out[ipt] * conj(file2_fft_out[ipt]) * scale;
    }

    fftwf_execute_dft (plan_bwd, bwd_in, bwd_out);

    for (ipt=0; ipt<npad; ipt++)
      result[ipt] += bwd_out[ipt];
    for (ipt=0; ipt<npad; ipt++)
      result_phases[ipt] += bwd_in[ipt];
  }

  //plot_power_spectrum (fb1_spectrum, nsubband, "1/xs");
  //plot_power_spectrum (fb2_spectrum, nsubband, "2/xs");

  plot_power_spectrum (file1_spectrum, npad, "1/xs");
  plot_power_spectrum (file2_spectrum, npad, "2/xs");

  for (ipt=0; ipt<npad; ipt++)
  {
    result_power[ipt] = creal(result[ipt]) * creal(result[ipt]) + cimag(result[ipt]) * cimag(result[ipt]);
    phases[ipt] = atanf(cimag(result_phases[ipt]) / creal(result_phases[ipt]));
  }

  plot_delays (result_power, npad, "3/xs");

  plot_power_spectrum (phases, npad, "4/xs");

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  free (file1_raw);
  free (file2_raw);
  free (file1_spectrum);
  free (file2_spectrum);
  free (fb1_spectrum);
  free (fb2_spectrum);

  fftwf_free (fb_fft_in);
  fftwf_free (fb_fft_out);
  fftwf_free (file1_fft_in);
  fftwf_free (file1_fft_out);
  fftwf_free (file2_fft_in);
  fftwf_free (file2_fft_out);
  fftwf_free (bwd_in);
  fftwf_free (bwd_out);
  free (result);
  free (result_power);
  fftwf_destroy_plan (plan_fb);
  fftwf_destroy_plan (plan_fwd);
  fftwf_destroy_plan (plan_bwd);

  close (fd1);
  close (fd2);

  return EXIT_SUCCESS;
}



void plot_delays (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float half_npts = (float) npts / 2;

  float xmin = -1 * half_npts;
  float xmax = half_npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  unsigned xpeak = 0;

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if (yvals[i] < ymin)
      ymin = yvals[i];
    xvals[i] = (float) i - half_npts;
  }

  fprintf (stderr, "xpeak=%d LAG=%d samples\n", xpeak, xpeak - (npts / 2));

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", "");

  cpgsci(2);

  cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();
}


void plot_power_spectrum (float * yvals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = (float) npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  unsigned xpeak = 0;

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if (yvals[i] < ymin)
      ymin = yvals[i];
    xvals[i] = (float) i;
  }

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", "");

  cpgsci(2);

  cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();
}


void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device)
{

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = npts;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];
  float yvals[npts];
  unsigned max_peak = 0;
  unsigned min_peak = 0;

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (creal(data[i]) > ymax) 
    {
      ymax = creal(data[i]);
      max_peak = i;
    }

    if (cimag(data[i]) > ymax)
    {
      ymax = cimag(data[i]);
      max_peak = i;
    }

    if (creal(data[i]) < ymin) 
    {
      ymin = creal(data[i]);
      min_peak = i;
    }

    if (cimag(data[i]) < ymin) 
    {
      ymin = cimag(data[i]);
      min_peak = i;
    }

    xvals[i] = (float) i;
  }

  fprintf (stderr, "peaks min=%u, max=%u\n", min_peak, max_peak);

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", "");

  cpgsci(2);
  for (i=0; i<npts; i++)
    yvals[i] = creal(data[i]);
  cpgline(npts, xvals, yvals);

  cpgsci(3);
  for (i=0; i<npts; i++)
    yvals[i] = cimag(data[i]);
  cpgline(npts, xvals, yvals);

  cpgebuf();
  cpgend();

}
