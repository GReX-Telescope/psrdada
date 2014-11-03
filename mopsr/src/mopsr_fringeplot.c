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
#include "mopsr_udp.h"
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();
void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title);

void usage()
{
  fprintf (stdout,
     "mopsr_fringeplot phases amps\n"
     " -D device       pgplot device name\n"
     " -i factor       integrate by this factor in time\n"
     " -t secs         number of seconds per point\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  int integrate = 1;

  float secs_per_point = -1;

  while ((arg=getopt(argc,argv,"i:D:ht:v")) != -1)
  {
    switch (arg)
    {
      case 'i':
        integrate = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 't':
        secs_per_point = atof(optarg);
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

  char phases_file[1024];
  char amps_file[1024];

  strcpy(phases_file, argv[optind]);
  strcpy(amps_file, argv[optind+1]);

  size_t file_size;
  struct stat buf;
  if (stat (phases_file, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", phases_file, strerror(errno));
    return (EXIT_FAILURE);
  }
  file_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", phases_file, file_size);
  int nsamp = file_size / sizeof(float);

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd1 = open (phases_file, flags, perms);
  if (fd1 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", phases_file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "reading amps_file\n");
  int fd2 = open (amps_file, flags, perms);
  if (fd2 < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", amps_file, strerror(errno));
    exit(EXIT_FAILURE);
  }

  float * phases = (float *) malloc (file_size);
  float * amps   = (float *) malloc (file_size);

  read (fd1, (void *) phases, file_size);
  read (fd2, (void *) amps, file_size);

  close (fd1);
  close (fd2);

  int integrated_size = file_size / integrate;
  float * p_plot = (float *) malloc (integrated_size);
  float * a_plot = (float *) malloc (integrated_size);
  float secs_per_point_plot = secs_per_point * integrate;

  unsigned i, j;
  float a_sum = 0;
  float p_sum = 0;
  unsigned nsamp_plot = nsamp / integrate;
  for (i=0; i< nsamp_plot; i++)
  {
    a_plot[i] = 0;
    p_plot[i] = 0;
    for (j=0; j<integrate; j++)
    {
      a_plot[i] += amps[i*integrate + j];
      p_plot[i] += phases[i*integrate + j];
    }
    a_plot[i] /= integrate;
    p_plot[i] /= integrate;
  }

  //plot_cross_bin (amps, phases, nsamp, "1/xs", secs_per_point, "");
  plot_cross_bin (a_plot, p_plot, nsamp_plot, device, secs_per_point_plot, "Cross Correlation between West25-2 and E37-1: 1089m baseline");

  free (amps);
  free (a_plot);
  free (phases);
  free (p_plot);

  return EXIT_SUCCESS;
}

void plot_cross_bin (float * amps, float * phases, unsigned npts, char * device, float seconds_per_pt, char * title)
{
  float xmin = 0;
  float xmax = (float) npts * seconds_per_pt;
  fprintf (stderr, "npts=%u seconds_per_pt=%f, xmax=%f\n", npts, seconds_per_pt, xmax);
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

  //ymin = 0;
  //ymax = 10e8;

  ymin = 0;
  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Amplitude", title);

  cpgsci(3);
  cpgpt(npts, xvals, amps, 17);
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

  cpgswin(xmin, xmax, -180, 180);
  //cpgswin(xmin, xmax, ymin, ymax);
  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Phase", "");

  cpgsci(3);
  cpgpt(npts, xvals, phases, 17);
  cpgslw(1);

  cpgebuf();
  cpgend();
}
