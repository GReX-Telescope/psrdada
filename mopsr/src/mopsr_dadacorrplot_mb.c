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
void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts);
void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device);
void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, int bin);
void plot_histogram (float * vals, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot [options] file1 file2\n"
     " files must be single antenna, single channel files\n"
     " -d delay        number of samples to delay files by [default 0]\n"
     " -f fractional   fractional sample to delay files by [default 0]\n"
     " -D device       pgplot device name\n"
     " -n npt          number of points in cross-correlation signal [default 1024]\n"
     " -p bin          plot the specified cross-correlation bin's timeseries [default npt/2]\n"
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

  unsigned int npt = 1024;

  unsigned int nsets = -1;

  unsigned int plot_bin = -1;

  int delay = 0;

  float fractional_delay = 0.0;

  while ((arg=getopt(argc,argv,"d:f:D:hn:p:t:v")) != -1)
  {
    switch (arg)
    {
      case 'd':
        delay = atoi(optarg);
        break;

      case 'f':
        fractional_delay = atof(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'n':
        npt = atoi(optarg);
        break;

      case 'p':
        plot_bin = atoi(optarg);
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

  if (plot_bin == -1)
    plot_bin = npt / 2;

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
  size_t bytes_packed = npt * MOPSR_NDIM * sizeof(int8_t);
  if (verbose)
    fprintf (stderr, "bytes_packed=%ld\n", bytes_packed);
  if (nsets * bytes_packed > file1_size - 4096)
  {
    fprintf (stderr, "file size not big enough for requested operations\n");
    return (EXIT_FAILURE);
  }

  int8_t * file_raw = (int8_t *) malloc (bytes_packed);

  if (verbose)
    fprintf (stderr, "reading file1\n");

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (file1, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file1, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t bytes_read;
  offset = 4096;
  if (delay > 0)
    offset += (2 * delay);
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
  offset = 4096;
  if (delay < 0)
    offset += (-2 * delay);
  lseek (fd2, offset, SEEK_SET);

  if (verbose)
    fprintf (stderr, "preparing fft memory\n");

  size_t nbytes;
  nbytes = sizeof(fftwf_complex) * npt;
  fprintf (stderr, "FWD FFT npt=%d nbytes=%ld\n", npt, nbytes);
  fftwf_complex * fwd1_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  nbytes = sizeof(fftwf_complex) * npt;
  fftwf_complex * bwd_in     = (fftwf_complex *) fftwf_malloc (nbytes);
  fftwf_complex * bwd_out    = (fftwf_complex *) fftwf_malloc (nbytes);
  fftwf_complex * result_sum = (fftwf_complex *) fftwf_malloc (nbytes);

  memset (result_sum, 0, nbytes);

  nbytes = sizeof(float) * npt;
  float * result   = (float *) malloc (nbytes);
  float * phases   = (float *) malloc (nbytes);
  float * file1_sp = (float *) malloc (nbytes);
  float * file2_sp = (float *) malloc (nbytes);

  fftwf_complex cross_power;

  memset (file1_sp, 0, nbytes);
  memset (file1_sp, 0, nbytes);

  // FWD filterbank plan
  fftwf_plan plan_fb  = fftwf_plan_dft_1d (npt, fwd1_in, fwd1_out, FFTW_FORWARD, FFTW_ESTIMATE);

  unsigned ipt, iset;
  float re, im;

  if (nsets == -1)
    nsets = (file1_size - 4096) / bytes_packed;

  fprintf (stderr, "Nsets= %d\n", nsets);

  float * cross_amps_bin = (float *) malloc(sizeof(float) * nsets);
  float * cross_phases_bin = (float *) malloc(sizeof(float) * nsets);
  float cross_amps_sum_squares = 0;
  float theta;
  complex float ramp;

  for (iset=0; iset<nsets; iset++)
  {
    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from file %s\n", bytes_packed, file1);
    bytes_read = read (fd1, (void *) file_raw, bytes_packed);

    // unpack file1 to end half of fwd1_in
    for (ipt=0; ipt<npt; ipt++)
    {
      re = ((float) file_raw[2*ipt+0]) + 0.5;
      im = ((float) file_raw[2*ipt+1]) + 0.5;
      fwd1_in[ipt] = re + im * I;
    }
    fftwf_execute_dft (plan_fb, fwd1_in, fwd1_out);

    // now do file2
    if (verbose > 1)
      fprintf(stderr, "reading %ld bytes from file %s\n", bytes_packed, file2);
    bytes_read = read (fd2, (void *) file_raw, bytes_packed);

    // unpack file2 to first half of fwd2_in
    for (ipt=0; ipt<npt; ipt++)
    {
      re = ((float) file_raw[2*ipt+0]) + 0.5;
      im = ((float) file_raw[2*ipt+1]) + 0.5;
      fwd2_in[ipt] = re + im * I;
    }
    fftwf_execute_dft (plan_fb, fwd2_in, fwd2_out);

    // now compute cross correlation
    for (ipt=0; ipt<npt; ipt++)
    {
      file1_sp[ipt] += creal(fwd1_out[ipt]) * creal(fwd1_out[ipt]) + cimag(fwd1_out[ipt]) * cimag(fwd1_out[ipt]);
      file2_sp[ipt] += creal(fwd2_out[ipt]) * creal(fwd2_out[ipt]) + cimag(fwd2_out[ipt]) * cimag(fwd2_out[ipt]);

      cross_power = fwd1_out[ipt] * conj(fwd2_out[ipt]);

      if (fractional_delay > 0)
      {
        theta = fractional_delay * 2 * M_PI * ((float) ipt / (float) npt);
        ramp = sin (theta) - cos(theta) * I;
        cross_power *= ramp;
      }

      result_sum[ipt] += cross_power;
      //result[ipt] += sqrt(creal(cross_power) * creal(cross_power) + cimag(cross_power) * cimag(cross_power));
      if (ipt == plot_bin)
      {
        cross_amps_bin[iset] = creal(cross_power) * creal(cross_power) + cimag(cross_power) * cimag(cross_power);
        cross_amps_sum_squares += cross_amps_bin[iset];
        cross_amps_bin[iset] = sqrt(creal(cross_power) * creal(cross_power) + cimag(cross_power) * cimag(cross_power));
        cross_phases_bin[iset] = atan2f(cimag(cross_power), creal(cross_power));
      }
    }

  }

  for (ipt=0; ipt<npt; ipt++)
  {
    phases[ipt] = atan2f(cimag(result_sum[ipt]), creal(result_sum[ipt]));
    result[ipt] = sqrt(creal(result_sum[ipt]) * creal(result_sum[ipt]) + cimag(result_sum[ipt]) * cimag(result_sum[ipt]));
  }

  cross_amps_sum_squares /= (float) nsets;

  float cross_amps_rms = sqrt(cross_amps_sum_squares);
  fprintf (stderr, "cross_amps_rms=%f\n",cross_amps_rms);

  for (iset=0; iset<nsets; iset++)
  {
    cross_amps_bin[iset] /= cross_amps_rms;
  }

  plot_power_spectrum (file1_sp, npt, "1/xs", "Bandpass A", 0, -1, 0);
  plot_power_spectrum (file2_sp, npt, "2/xs", "Bandpass B", 0, -1, 0);
  plot_power_spectrum (result, npt, "3/xs", "Cross Power Spectrum", 0, -1, 0);
  plot_power_spectrum (phases, npt, "4/xs", "Cross Phases", -1 * M_PI, M_PI, 1);
  plot_cross_bin (cross_amps_bin, cross_phases_bin, nsets, "5/xs", plot_bin);
  plot_histogram (cross_phases_bin, nsets, "6/xs");

  float ymax = -FLT_MAX;
  int bin = 0;
  unsigned i;
  for (i=0; i<npt; i++)
  {
    if ((result[i] > ymax) && (i < npt/2))
    {
      ymax = result[i];
      bin = i;
    }
  }

  fprintf (stderr, "ymax=%f bin=%d\n", ymax, bin);


  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  free (file_raw);
  free (file1_sp);
  free (file2_sp);
  free (phases);

  fftwf_free (result);
  fftwf_free (fwd1_in);
  fftwf_free (fwd2_in);
  fftwf_free (fwd1_out);
  fftwf_free (fwd2_out);
  fftwf_destroy_plan (plan_fb);

  close (fd1);
  close (fd2);

  return EXIT_SUCCESS;
}


void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, int bin)
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

  //ymin = 0;
  //ymax = 10e8;

  cpgswin(xmin, xmax, ymin, ymax);

  sprintf (label, "Cross-Correlation for bin %d", bin);

  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Amplitude", label);

  cpgsci(3);
  cpgpt(npts, xvals, amps, -2);
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

  fprintf (stderr, "ymin=%f ymax=%f\n", ymin, ymax);

  //cpgswin(xmin, xmax, (-1* M_PI), M_PI);
  cpgswin(xmin, xmax, ymin, ymax);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (samples)", "Phase", "");

  cpgsci(3);
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

  cpgpt(npts, xvals, yvals, -2);

  cpgebuf();
  cpgend();
}


void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = npts;
  float ymin, ymax;
  float xvals[npts];
  unsigned xpeak = 0;
  unsigned i;

  if (ymin_in == -1)
    ymin = FLT_MAX;
  else
    ymin = ymin_in;

  if (ymax_in == -1)
    ymax = -FLT_MAX;
  else
    ymax = ymax_in;

  for (i=0; i<npts; i++)
  {
    if ((ymax_in == -1) && (yvals[i] > ymax))
    {
      ymax = yvals[i];
      xpeak = i;
    }

    if ((ymin_in == -1) && (yvals[i] < ymin))
    {
      ymin = yvals[i];
    }

    xvals[i] = 800.0 + (((float) i / npts) * 100);
    xvals[i] = (float) i;
  }

  //xmin = 262150;
  //xmax = 262200;

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("", "", title);

  cpgsci(2);

  if (use_pts)
    cpgpt(npts, xvals, yvals, -2);
  else
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

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (creal(data[i]) > ymax) 
    {
      ymax = creal(data[i]);
    }

    if (cimag(data[i]) > ymax)
    {
      ymax = cimag(data[i]);
    }

    if (creal(data[i]) < ymin) 
    {
      ymin = creal(data[i]);
    }

    if (cimag(data[i]) < ymin) 
    {
      ymin = cimag(data[i]);
    }

    xvals[i] = (float) i;
  }

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

void plot_histogram (float * vals, unsigned npts, char * device)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

/*
  unsigned i;
  unsigned hist[256];
  float ymin = 0;
  float ymax = 2 * M_PI;
  float val;

  for (i=0; i<256; i++)
    hist[i] = 0;

  for (i=0; i<npts; i++)
  {
    val = vals[i] + M_PI;
    val /= (2*M_PI);
    val *= 256;
    bin = (unsigned) floor(val);
    fprintf (stderr, "bin %d ++ \n", bin);
    hist[bin]++;
  }

  for (i=0; i<256; i++)
  {
    if (hist[i] > ymax)
      ymax = hist[i];
  }

  float xmin = -1 * M_PI;
  float xmax = M_PI;

  cpgbbuf();
  cpgswin(ymin, ymax
*/

  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;
  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (vals[i] > ymax) ymax = vals[i];
    if (vals[i] < ymin) ymin = vals[i];
  }

  int N = (int) npts;
  int nbin = 128;
  int flag = 0;
  cpgbbuf();
  cpghist (N, vals, ymin, ymax, nbin, flag);
  cpgebuf();
  cpgend();

}


