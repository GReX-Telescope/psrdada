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
void plot_delays (float * yvals, unsigned npts, char * device, float samps_per_point);

void usage()
{
  fprintf (stdout,
     "mopsr_dadapearsoncoeff [options] file1 file2\n"
     " files must be single antenna, single channel files\n"
     " -b nblocks      number of 1s blocks to process [default 10]\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  char * device = "/xs";

  int arg = 0;

  unsigned block_size = 781250 * MOPSR_NDIM;

  unsigned nblocks = -1;

  int channel = 0;

  while ((arg=getopt(argc,argv,"b:D:hv")) != -1)
  {
    switch (arg)
    {
      case 'b':
        block_size = atoi(optarg);
        break;

      case 'D':
        device = strdup (optarg);
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

  unsigned iblock;
  if (nblocks == -1)
    nblocks = (file1_size - 4096) / block_size;
  fprintf (stderr, "main: nblocks=%u\n", nblocks);

  nblocks = 5000;
  float seconds_per_point = (tsamp / 1000000) * block_size;

  // width of 1 sample
  size_t samp_stride = MOPSR_NDIM;

  unsigned ipt, iset;
  float re, im;

  // number of blocks to read
  const unsigned nreads = (file1_size - 4096) / block_size;
  fprintf (stderr, "main: nreads= %d\n", nreads);

  size_t nbytes = block_size;
  int8_t * raw1 = (int8_t *) malloc (nbytes);
  int8_t * raw2 = (int8_t *) malloc (nbytes);
  memset (raw1, 0, nbytes);
  memset (raw2, 0, nbytes);

  float * yvals = (float *) malloc (nblocks * sizeof(float));
  unsigned nsamps_per_block = block_size / MOPSR_NDIM;
  float seconds_per_pt = (float) nsamps_per_block * (tsamp / 1000000);

  double bytes_per_second = (double) (1000000 / tsamp) * MOPSR_NDIM;
  fprintf (stderr, "bytes_per_second=%lf\n", bytes_per_second);

  const unsigned nant= 2;
  unsigned i;
  double rsum = 0;

  for (iblock=0; iblock<nblocks; iblock++)
  {
    read (fd1, (void *) raw1, nbytes);
    read (fd2, (void *) raw2, nbytes);

    double x, y;
    double sum_xy = 0;
    double sum_xsq = 0;
    double sum_ysq = 0;
    double r;

    for (i=0; i<block_size; i++)
    {
      x = (double) raw1[i];
      y = (double) raw2[i];

      sum_xy  += (x * y);
      sum_xsq += (x * x);
      sum_ysq += (y * y);
    }
    r = sum_xy / (sqrt(sum_xsq) * sqrt(sum_ysq));
    yvals[iblock] = (float) r;

    rsum += r;
  }
  fprintf (stderr, "rsum=%lf\n", rsum);

  plot_delays (yvals, nblocks, device, seconds_per_point);

  if (verbose)
    fprintf (stderr, "freeing allocated memory\n");

  free (raw1);
  free (raw2);

  close (fd1);
  close (fd2);

  return EXIT_SUCCESS;
}

void plot_delays (float * yvals, unsigned npts, char * device, float seconds_per_point)
{
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  float xmin = 0;
  float xmax = (float) npts * seconds_per_point;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xvals[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (yvals[i] > ymax)
      ymax = yvals[i];
    if (yvals[i] < ymin)
      ymin = yvals[i];
    xvals[i] = (float) i * seconds_per_point;
  }

  ymin = -0.1;
  ymax = +0.1;

  cpgbbuf();
  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Time (s)", "Pearson Coeff", "Pearson Coeff between modules");

  cpgsci(2);

  cpgpt(npts, xvals, yvals, -2);

  cpgebuf();
  cpgend();
}
