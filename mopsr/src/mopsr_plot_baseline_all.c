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

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();
void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, char * title);
void plot_series (double * x, double * y, unsigned npts, char * device, float M);
void plot_map (float * map, unsigned nx, unsigned ny, char * device, char * title, mopsr_util_t * opts);
void unwrap (double * p, int N);

void usage()
{
  fprintf (stdout,
     "mopsr_plot_baseline baseline npt ccfile\n"
     " -D device       pgplot device name\n"
     " -h              plot this help\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  // baseline to plot instead
  int baseline = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"i:D:hv")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
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
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  if (sscanf(argv[optind], "%u", &baseline) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse baseline from %s\n", argv[optind]);
    usage();
    return (EXIT_FAILURE);
  }
  if (verbose)
    fprintf (stderr, "baseline=%u\n", baseline);

  unsigned npt = 0;
  if (sscanf(argv[optind+1], "%u", &npt) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse npt from %s\n", argv[optind+1]);
    usage();
    return (EXIT_FAILURE);
  }
  if (verbose)
    fprintf (stderr, "npt=%u\n", npt);

  char cc_file[1024];
  strcpy(cc_file, argv[optind+2]);

  size_t file_size;
  struct stat buf;
  if (stat (cc_file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", cc_file, strerror(errno));
    return (EXIT_FAILURE);
  }
  file_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", cc_file, file_size);

  unsigned npairs = (unsigned) (file_size / (npt * sizeof(complex float)));
  unsigned nant = (1 + sqrt(1 + (8 * npairs))) / 2;
  if (verbose)
    fprintf (stderr, "npairs=%u nant=%u \n", npairs, nant);

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (cc_file, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", cc_file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  float * amps   = (float *) malloc (npt * sizeof(float));
  float * phases = (float *) malloc (npt * sizeof(float));

  complex float * cc = (complex float *)  malloc (npt * sizeof(complex float));

  unsigned ipt, ipair;
  float re, im;

  char title[1024];
  size_t bytes_to_read = npt * sizeof(complex float);

  unsigned i = 1;
  unsigned tri = i;

  int anta = 0;
  int antb = 0;

  for (ipair=0; ipair<npairs; ipair++)
  {
    assert (i < (nant * nant));

    // read in spectra pair
    read (fd, (void *) cc, bytes_to_read);

    if (ipair == baseline)
    {
      for (ipt=0; ipt<npt; ipt++)
      {
        re = creal(cc[ipt]);
        im = cimag(cc[ipt]);
        if (ipt == npt/2)
          re = im = 0;

        amps[ipt] = re * re + im * im;
        phases[ipt] = atan2f(im, re);
      }

      anta = i / nant;
      antb = i % nant;
    }

    i++;
    if (i % nant == 0)
    {
      tri++;
      i += tri;
    }
  }

  mopsr_util_t opt;

  opt.nant = nant;
  opt.plot_log = 0;
  opt.ymin = 0;
  opt.ymax = 20;

  sprintf (title, "Baseline %d [%d -> %d]", baseline, anta, antb);
  plot_cross_bin (amps, phases, npt, device, title);

  close (fd);

  free (cc);
  free (amps);
  free (phases);

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

