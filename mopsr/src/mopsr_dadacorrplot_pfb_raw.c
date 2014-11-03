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
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage ();
void plot_delays (float * yvals, unsigned npts, char * device);
void plot_power_spectrum (float * yvals, unsigned npts, char * device, const char * title, float ymin_in, float ymax_in, char use_pts);
void plot_complex_spectrum (fftwf_complex * data, unsigned npts, char * device);
void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title);
void plot_cross_vs_freq (float * amps, float * phases, unsigned npts, char * device);
void plot_histogram (float * vals, unsigned npts, char * device);

void usage()
{
  fprintf (stdout,
     "mopsr_dadacorrplot_pfb [options] file1 file2\n"
     " files must be single antenna, single channel files\n"
     " -b nblocks      number of 1s blocks to process [default 10]\n"
     " -D device       pgplot device name\n"
     " -n npt          number of points in cross-correlation signal [default 256]\n"
     " -p bin          plot the specified cross-correlation bin's timeseries [default npt/2]\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int npt = 256;

  const unsigned block_size = 781250 * MOPSR_NDIM;

  unsigned nblocks = -1;

  unsigned int plot_bin = -1;

  int channel = 0;

  while ((arg=getopt(argc,argv,"b:D:hn:p:v")) != -1)
  {
    switch (arg)
    {
      case 'b':
        nblocks = atoi(optarg);
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
    return (EXIT_FAILURE);
  }

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (file1, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file1, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "reading file2\n");
  int fd2 = open (file2, flags, perms);
  if (fd2 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", file2, strerror(errno));
    exit(EXIT_FAILURE);
  }

  const size_t dada_header_size = 4096;
  char * header1 = (char *) malloc(dada_header_size);
  char * header2 = (char *) malloc(dada_header_size);

  size_t bytes_read = read (fd1, (void *) header1, dada_header_size);
  if (bytes_read != dada_header_size)
  {
    fprintf (stderr, "failed to read header from %s\n", file1);
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  bytes_read = read (fd2, (void *) header2, dada_header_size);
  if (bytes_read != dada_header_size)
  { 
    fprintf (stderr, "failed to read header from %s\n", file2);
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  int nchan = 0;
  if (ascii_header_get( header1, "NCHAN", "%d", &nchan) != 1)
  {
    fprintf (stderr, "failed to read nchan from header1\n");
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }
  assert (nchan == 1);

  float tsamp;
  if (ascii_header_get( header1, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "failed to read nchan from header1\n");
    close(fd1);
    close(fd2);
    return (EXIT_FAILURE);
  }

  size_t bytes_packed = npt * MOPSR_NDIM;
  fprintf (stderr, "bytes_packed=%ld\n", bytes_packed);

  unsigned iblock;
  if (nblocks == -1)
    nblocks = (file1_size - 4096) / block_size;
  fprintf (stderr, "main: nblocks=%u\n", nblocks);

  // width of 1 sample
  size_t samp_stride = MOPSR_NDIM;

  /*
  delay = 130 * (1000000.0 / tsamp);
  offset = delay * samp_stride;
  lseek (fd2, offset, SEEK_CUR);
  lseek (fd1, offset, SEEK_CUR);
  */

  size_t nbytes;
  nbytes = sizeof(fftwf_complex) * npt;
  fprintf (stderr, "FWD FFT npt=%d nbytes=%ld\n", npt, nbytes);
  fftwf_complex * fwd1_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd1_out = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_in  = (fftwf_complex*) fftwf_malloc (nbytes);
  fftwf_complex * fwd2_out = (fftwf_complex*) fftwf_malloc (nbytes);

  nbytes = sizeof(fftwf_complex) * npt;
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

  // number of blocks to read
  const unsigned nreads = (file1_size - 4096) / block_size;
  fprintf (stderr, "main: nreads= %d\n", nreads);

  nbytes = sizeof(int16_t) * npt;
  int16_t * raw = (int16_t *) malloc (nbytes);
  memset (raw, 0, nbytes);

  unsigned nsamps_per_block = block_size / MOPSR_NDIM;
  float seconds_per_pt = (float) nsamps_per_block * (tsamp / 1000000);

  float * cross_amps_bin = (float *) malloc(sizeof(float) * nblocks);
  float * cross_phases_bin = (float *) malloc(sizeof(float) * nblocks);
  float cross_amps_sum_squares = 0;

  fftwf_complex raw_cross;
  fftwf_complex p1, p2, cross_power_fscr;
  unsigned pt, i;

  double bytes_per_second = (double) (1000000 / tsamp) * MOPSR_NDIM;
  fprintf (stderr, "bytes_per_second=%lf\n", bytes_per_second);

  const unsigned nant= 2;
  unsigned iant;

  // get initial delays
  char title[128];

  cross_power_fscr = 0;

  int8_t * raw8;

  for (iblock=0; iblock<nblocks; iblock++)
  {
    // each block is divided up into nparts
    unsigned ipart;
    const unsigned npart = block_size / (npt * MOPSR_NDIM);
    nbytes = sizeof(int16_t) * npt;

    cross_power_fscr = 0;

    for (ipart=0; ipart < npart; ipart++)
    {
      raw8 = (int8_t *) raw;
      bytes_read = read (fd1, (void *) raw, nbytes);
      for (ipt=0; ipt<npt; ipt++)
      {
        re = (((float) raw8[0]) + 0.5) / 127.5;
        im = (((float) raw8[1]) + 0.5) / 127.5;
        fwd1_in[ipt] = (re + im * I);
        raw8 += 2;
      }

      raw8 = (int8_t *) raw;
      bytes_read = read (fd2, (void *) raw, nbytes);
      for (ipt=0; ipt<npt; ipt++)
      {
        re = (((float) raw8[0]) + 0.5) / 127.5;
        im = (((float) raw8[1]) + 0.5) / 127.5;
        fwd2_in[ipt] = (re + im * I);
        raw8 += 2;
      }

      // FFT both sets of samples
      fftwf_execute_dft (plan_fb, fwd1_in, fwd1_out);
      fftwf_execute_dft (plan_fb, fwd2_in, fwd2_out);

      // now compute cross correlation
      for (ipt=0; ipt<npt; ipt++)
      {
        // since PFB, this is dual side band and we must FFT-shift
        pt = (ipt + (npt/2)) % npt;
        if (pt == 0)
        {
          p1 = 0;
          p2 = 0;
        }
        else
        {
          p1 = fwd1_out[pt] / npt;
          p2 = fwd2_out[pt] / npt;
        }

        // this is the bandpass for each input, never reset
        file1_sp[ipt] += ((creal(p1) * creal(p1)) + (cimag(p1) * cimag(p1)));
        file2_sp[ipt] += ((creal(p2) * creal(p2)) + (cimag(p2) * cimag(p2)));

        cross_power = p1 * conj(p2); 
        cross_power_fscr += cross_power;
        result_sum[ipt] += cross_power;
      }
    }

    cross_phases_bin[iblock] = atan2f(cimag(cross_power_fscr), creal(cross_power_fscr)) * (180/M_PI);
    cross_amps_bin[iblock]   = sqrtf(creal(cross_power_fscr) * creal(cross_power_fscr) + cimag(cross_power_fscr) * cimag(cross_power_fscr));

    for (ipt=0; ipt<npt; ipt++)
    {
      re = creal(result_sum[ipt]);
      im = cimag(result_sum[ipt]);
      phases[ipt] = atan2f(im, re);
      result[ipt] = sqrt(re * re + im * im);
    }

    sprintf (title, "Cross-correlation vs time");
    plot_power_spectrum (file1_sp, npt, "1/xs", "Bandpass A", 0, -1, 0);
    plot_power_spectrum (file2_sp, npt, "2/xs", "Bandpass B", 0, -1, 0);
    plot_cross_vs_freq (result, phases, npt, "3/xs");
    plot_cross_vs_time (cross_amps_bin, cross_phases_bin, iblock, "4/xs", seconds_per_pt, title);
  }

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  free (raw);

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


void plot_cross_vs_time (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title)
{
  float xmin = 0;
  float xmax = (float) npts * seconds_per_pt;
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

    xvals[i] = (float) i * seconds_per_pt;
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
  cpgslw(5);
  cpgpt(npts, xvals, amps, -1);
  cpgline(npts, xvals, amps);
  cpgslw(1);
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

  cpgswin(xmin, xmax, -180, 180);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Phase", "");

  cpgsci(3);
  cpgslw(5);
  cpgpt(npts, xvals, phases, -1);
  cpgslw(1);

  cpgebuf();
  cpgend();
}

void plot_cross_vs_freq (float * cross, float * phases, unsigned npts, char * device)
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
  //cpgswin(xmin, xmax, ymin, ymax);
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

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Freq Channel", "Flux", title);

  cpgsci(2);

  if (use_pts)
    cpgpt(npts, xvals, yvals, -3);
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


