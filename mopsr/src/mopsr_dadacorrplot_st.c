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

static char padded = 1;

void usage ();
void convert_log(float * yvals, unsigned npts);
void plot_delays (float * yvals, unsigned npts, char * device);
void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, char plot_log);
void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device);
void plot_cross_power (float * cross, float * phases, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot_st [options] file\n"
     " files must be single antenna, single channel files\n"
     " -a ant1         first ant from file to compare [default 0]\n"
     " -b ant2         second ant from file to compare [default 1]\n"
     " -d delay        delay file2 by this many samples [default 0]\n"
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

  int ant1 = 0;

  int ant2 = 1;

  int npt = 1024;

  int nsets = 1;

  int nant = 4;

  int delay = 0;

  float fractional_delay = 0.0;

  while ((arg=getopt(argc,argv,"a:b:d:D:f:hn:t:v")) != -1)
  {
    switch (arg)
    {
      case 'a':
        ant1 = atoi(optarg);
        break;

      case 'b':
        ant2 = atoi(optarg);
        break;

      case 'd':
        delay = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'f':
        fractional_delay = atof(optarg);
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
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line argument is required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char file[1024];

  strcpy(file, argv[optind]);

  struct stat buf;
  if (stat (file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", file, strerror(errno));
    return (EXIT_FAILURE);
  }
  size_t file_size = buf.st_size;
  size_t ant_size = (file_size - 4096) / nant;
  if (verbose)
    fprintf (stderr, "filesize for %s is %ld bytes ant_size=%ld\n", file, file_size, ant_size);

  // number of bytes required to read file into memory
  size_t bytes_packed = npt * MOPSR_NDIM * sizeof(int8_t);
  //if (verbose)
    fprintf (stderr, "bytes_packed=%ld\n", bytes_packed);
  if (bytes_packed > file_size - 4096)
  {
    fprintf (stderr, "file size not big enough for requested operations\n");
    return (EXIT_FAILURE);
  }

  int8_t * file_raw = (int8_t *) malloc (bytes_packed);
  if (!file_raw)
  {
    fprintf (stderr, "failed to allocate %d bytes\n", bytes_packed);
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "reading file\n");

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (file, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t bytes_read;
  off_t offset = 4096;
  lseek (fd, offset, SEEK_SET);

  if (verbose)
    fprintf (stderr, "preparing fft memory\n");

  unsigned npt_pad;
  if (padded)
    npt_pad = 2 * npt -1;
  else
    npt_pad = npt;
  size_t nbytes;

  nbytes = sizeof(fftwf_complex) * npt_pad;
  fprintf (stderr, "FWD FFT npt=%d npt_pad=%u nbytes=%ld\n", npt, npt_pad, nbytes);
  fftwf_complex * fwd1_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  memset (fwd1_in, 0, nbytes);
  memset (fwd2_in, 0, nbytes);

  nbytes = sizeof(fftwf_complex) * npt_pad;
  fftwf_complex * bwd_in      = (fftwf_complex *) fftwf_malloc (nbytes);
  fftwf_complex * bwd_out     = (fftwf_complex *) fftwf_malloc (nbytes);
  fftwf_complex * result_sum  = (fftwf_complex *) fftwf_malloc (nbytes);
  fftwf_complex * phases_sum  = (fftwf_complex *) fftwf_malloc (nbytes);

  memset (result_sum, 0, nbytes);
  memset (phases_sum, 0, nbytes);

  nbytes = sizeof(float) * npt_pad;
  float * cross    = (float *) malloc (nbytes);
  float * delays   = (float *) malloc (nbytes);
  float * phases   = (float *) malloc (nbytes);
  float * file1_sp = (float *) malloc (nbytes);
  float * file2_sp = (float *) malloc (nbytes);

  double real_avg1 = 0;
  double imag_avg1 = 0;

  double real_avg2 = 0;
  double imag_avg2 = 0;

  uint64_t avg_count = 0;

  memset (file1_sp, 0, nbytes);
  memset (file2_sp, 0, nbytes);

  // FWD filterbank plan
  fftwf_plan plan_fwd  = fftwf_plan_dft_1d (npt_pad, fwd1_in, fwd1_out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_bwd = fftwf_plan_dft_1d (npt_pad, bwd_in,  bwd_out , FFTW_BACKWARD, FFTW_ESTIMATE);

  fftwf_complex scale = 1.0/ ((float) npt_pad);
  if (verbose)
    fprintf (stderr, "scale=%f + %fi\n", creal(scale), cimag(scale));

  unsigned ipt, iset, dsb_ipt;
  float re, im, theta;
  complex float ramp;
  fftwf_complex f1, f2;

  // check the maximum number of sets...
  size_t max_nsets = (ant_size - ((size_t) abs(delay))) / (MOPSR_NDIM * bytes_packed);
  if (nsets > max_nsets)
    nsets = (int) max_nsets;
  fprintf (stderr, "nsets=%d (of %ld)\n", nsets, max_nsets);

  for (iset=0; iset<nsets; iset++)
  {
    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from ant %d\n", bytes_packed, ant1);
    offset = 4096 + (ant1 * ant_size) + iset * bytes_packed;
    if (delay > 0)
      offset += delay * MOPSR_NDIM;
    lseek (fd, offset, SEEK_SET);
    bytes_read = read (fd, (void *) file_raw, bytes_packed);

    // unpack ant1 to end half of fwd1_in
    for (ipt=0; ipt<npt; ipt++)
    {
      re = ((float) file_raw[2*ipt+0]) + 0.5;
      im = ((float) file_raw[2*ipt+1]) + 0.5;

      real_avg1 += (double) (re * re);
      imag_avg1 += (double) (im * im);

      if (padded)
        fwd1_in[(npt-1) + ipt] = re + im * I;
      else
        fwd1_in[ipt] = re + im * I;
    }
    fftwf_execute_dft (plan_fwd, fwd1_in, fwd1_out);

    // now do file2
    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from ant %d\n", bytes_packed, ant2);
    offset = 4096 + (ant2 * ant_size) + iset * bytes_packed;
    if (delay < 0)
      offset += -1 * delay * MOPSR_NDIM;
    lseek (fd, offset, SEEK_SET);
    bytes_read = read (fd, (void *) file_raw, bytes_packed);

    // unpack file2 to first half of fwd2_in
    for (ipt=0; ipt<npt; ipt++)
    {
      re = ((float) file_raw[2*ipt+0]) + 0.5;
      im = ((float) file_raw[2*ipt+1]) + 0.5;

      real_avg2 += (double) (re * re);
      imag_avg2 += (double) (im * im);

      fwd2_in[ipt] = re + im * I;
    }
    fftwf_execute_dft (plan_fwd, fwd2_in, fwd2_out);

    avg_count += npt_pad; 

    if (verbose > 1)
      fprintf(stderr, "fwd ffts done, computing cross correlation\n");

    // now compute cross correlation
    for (ipt=0; ipt<npt_pad; ipt++)
    {
      dsb_ipt = (ipt + npt) % npt_pad;
      f1 = fwd1_out[dsb_ipt];
      f2 = fwd2_out[dsb_ipt];

      // setup cross correlation input
      bwd_in[ipt] = f1 * conj(f2) * scale;

      if (fractional_delay > 0)
      {
        theta = fractional_delay * 2 * M_PI * ((float) ipt / (float) npt_pad);
        ramp = sin (theta) - cos(theta) * I;
        //if (iset == 0)
        //  fprintf (stderr, "ipt=%d theta=%f ramp=(%f,%f) bwd_in=(%f,%f)", ipt, theta, creal(ramp), cimag(ramp), ipt, creal(bwd_in[ipt]), cimag(bwd_in[ipt]));
        bwd_in[ipt] *= ramp;
        //if (iset == 0)
        //  fprintf (stderr,  " bwd_in=(%f, %f)\n", creal(bwd_in[ipt]), cimag(bwd_in[ipt]));
      }

      // phases
      phases_sum[ipt] += bwd_in[ipt];

      // sub-band spectrum
      file1_sp[ipt] += (creal(f1) * creal(f1)) + (cimag(f1) * cimag(f1));
      file2_sp[ipt] += (creal(f2) * creal(f2)) + (cimag(f2) * cimag(f2));
    }

    if (verbose > 1)
      fprintf(stderr, "cross correlation computed, perform ifft\n");

    // iFFT to compute cross-correlation between signals
    fftwf_execute_dft (plan_bwd, bwd_in, bwd_out);

    if (verbose > 1)
      fprintf(stderr, "ifft done, updating result_sum\n");

    for (ipt=0; ipt<npt_pad; ipt++)
    {
      result_sum[ipt] += bwd_out[ipt];
    }
  }

  if (verbose > 1)
    fprintf(stderr, "computing averages\n");

  real_avg1 /= (double) avg_count;  
  real_avg2 /= (double) avg_count;  
  imag_avg1 /= (double) avg_count;  
  imag_avg2 /= (double) avg_count;  

  fprintf (stderr, "Input1: re=%lf, im=%f   Input2: re=%lf im=%lf\n", real_avg1, imag_avg1, real_avg2, imag_avg2);

  convert_log (file1_sp, npt_pad);
  convert_log (file2_sp, npt_pad);
  plot_power_spectrum (file1_sp, npt_pad, "1/xs", "Bandpass A", 1);
  plot_power_spectrum (file2_sp, npt_pad, "2/xs", "Bandpass B", 1);

  for (ipt=0; ipt<npt_pad; ipt++)
  {
    delays[ipt] = creal(result_sum[ipt]) * creal(result_sum[ipt]) + 
                   cimag(result_sum[ipt]) * cimag(result_sum[ipt]);
    phases[ipt] = atan2f(cimag(phases_sum[ipt]), creal(phases_sum[ipt]));
    //phases[ipt] = atanf(cimag(phases_sum[ipt])/ creal(phases_sum[ipt]));
    cross[ipt] = sqrt(creal(phases_sum[ipt]) * creal(phases_sum[ipt]) +
                        cimag(phases_sum[ipt]) * cimag(phases_sum[ipt]));
    //phases[ipt] = atanf(cimag(phases_sum[ipt]) / creal(phases_sum[ipt]));
  }

  plot_delays (delays, npt_pad, "3/xs");
  plot_cross_power (cross, phases, npt_pad, "4/xs");

  //plot_power_spectrum (phases, npt_pad, "4/xs", "Cross Phases");

  close (fd);

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");
  free (file_raw);
  free (file1_sp);
  free (file2_sp);
  free (delays);
  free (cross);
  free (phases);

  fftwf_free (fwd1_in);
  fftwf_free (fwd2_in);
  fftwf_free (fwd1_out);
  fftwf_free (fwd2_out);
  fftwf_free (bwd_in);
  fftwf_free (bwd_out);
  fftwf_destroy_plan (plan_fwd);
  fftwf_destroy_plan (plan_bwd);

  return EXIT_SUCCESS;
}

void plot_cross_power (float * cross, float * phases, unsigned npts, char * device)
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

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (cross[i] > ymax)
      ymax = cross[i];
    if (cross[i] < ymin)
      ymin = cross[i];

    xvals[i] = (float) i;
  }
  
  cpgbbuf();

  cpgswin(xmin, xmax, 0, ymax);

  cpgsvp(0.05, 0.95, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "", "Cross Power Spectrum");

  cpgsci(2);
  cpgline(npts, xvals, cross);
  cpgsci(1);

  cpgswin(xmin, xmax, (-1* M_PI), M_PI);
  cpgsvp(0.05, 0.95, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Phase", "");

  cpgsci(2);
  cpgpt(npts, xvals, phases, -2);

  cpgebuf();
  cpgend();

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
  int xpeak = 0;

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
    xvals[i] = i - half_npts;
  }

  int lag = xpeak - npts/2;
  fprintf (stderr, "xpeak=%d, LAG=%d samples\n", xpeak, lag);

  xmin =  (float) xpeak - 32;
  xmax =  (float) xpeak + 32;

  //fprintf (stderr, "xmin=%f xmax=%f\n", xmin, xmax);

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Sample", "Strength", "Integer Sample Delays");

  cpgsci(1);
  cpgsls(2);
  cpgline(npts, xvals, yvals);

  cpgsci(2);
  cpgsls(1);
  cpgpt(npts, xvals, yvals, -5);

  cpgebuf();
  cpgend();
}


void convert_log(float * yvals, unsigned npts)
{
  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i]  <= 0)
      yvals[i] = 1;
    yvals[i] = log10(yvals[i]);
  }
}

void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, char plot_log)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  char plotlog = 1;
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
  cpgsvp(0.1,0.9,0.1,0.9);
  cpgswin(xmin, xmax, ymin, ymax);

  if (plot_log)
  {
    cpgbox("BCNST", 0.0, 0.0, "BCNSTL", 0.0, 0.0);
  }
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);

  cpglab("", "", title);

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
