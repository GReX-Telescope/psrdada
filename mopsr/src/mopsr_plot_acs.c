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
#include <gsl/gsl_fit.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();
void plot_phase_time (float * xvals, float * amps, unsigned npts, char * device, char * title);

void usage()
{
  fprintf (stdout,
     "mopsr_plot_acs acs npt acfiles\n"
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

  // plotting of meridian distance glag
  char plot_meridian_dist = 0;

  // acs to plot instead
  int acs = 0;

  int arg = 0;

  float delay = 0;

  while ((arg=getopt(argc,argv,"D:hmv")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'm':
        plot_meridian_dist = 1;
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
  if (argc-optind < 3)
  {
    fprintf(stderr, "ERROR: at least 3 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  if (sscanf(argv[optind], "%u", &acs) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse acs from %s\n", argv[optind]);
    usage();
    return (EXIT_FAILURE);
  }
  if (verbose)
    fprintf (stderr, "acs=%u\n", acs);

  unsigned npt = 0;
  if (sscanf(argv[optind+1], "%u", &npt) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse npt from %s\n", argv[optind+1]);
    usage();
    return (EXIT_FAILURE);
  }
  if (verbose)
    fprintf (stderr, "npt=%u\n", npt);

  unsigned nfiles = argc - (optind + 2);
  if (verbose)
    fprintf (stderr, "nfiles=%u\n", nfiles);
  char ** ac_files = (char **) malloc (sizeof(char *) * nfiles);
  unsigned ifile;
  for (ifile=0; ifile<nfiles; ifile++)
  {
    size_t len = strlen(argv[optind+2+ifile]);
    ac_files[ifile] = (char *) malloc (sizeof(char) * (len + 1));
    strcpy (ac_files[ifile], argv[optind+2+ifile]);
  }

  size_t file_size;
  struct stat buf;
  if (verbose)
    fprintf (stderr, "stating %s\n", ac_files[0]);
  if (stat (ac_files[0], &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", ac_files[0], strerror(errno));
    return (EXIT_FAILURE);
  }

  file_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", ac_files[0], file_size);

  unsigned nant = (unsigned) (file_size / (npt * sizeof(float)));
  if (verbose)
    fprintf (stderr, "nant=%u \n", nant);

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;

  float * amps   = (float *) malloc (npt * sizeof(float));
  float * ac = (float *)  malloc (npt * sizeof(float));
  memset (amps, 0, npt * sizeof(float));

  float * amps_t = (float *) malloc (nfiles * sizeof(float));
  float * x_t = (float *) malloc (nfiles * sizeof(float));

  unsigned ipt, iant;
  float power;

  char title[1024];
  size_t bytes_to_read = npt * sizeof(float);

  int anta = 0;
  int antb = 0;

  float theta;
  float val;
  float fsum;

  mopsr_util_t opt;
  opt.nant = nant;
  opt.plot_log = 0;
  opt.ymin = 0;
  opt.ymax = 20;

  // total power in all files, used for amp mean
  double amp_total = 0;
  double amp_mean;
  char good_file;
  unsigned good_sums = 0;
  float amp_t;
  unsigned i;

  mopsr_source_t source;
  struct timeval timestamp;
  uint64_t bytes_per_second;
  float ut1_offset;
  time_t utc_start;
  char utc_start_str[20];

  char * header_file = "obs.header";

  {
    // read the header
    size_t header_size = 4096;
    char * header = (char *) malloc (header_size);

    if (fileread (header_file, header, header_size) < 0)
    {
      fprintf (stderr, "ERROR: could not read header from %s\n", header_file);
      return EXIT_FAILURE;
    }

    char position[32];
    if (ascii_header_get (header, "RA", "%s", position) != 1)
    {
      fprintf (stderr, "ERROR:  could not read RA from header\n");
      return -1;
    }

    if (verbose)
      fprintf (stderr, "RA (HMS) = %s\n", position);
    if (mopsr_delays_hhmmss_to_rad (position, &(source.raj)) < 0)
    {
      fprintf (stderr, "ERROR:  could not parse RA from %s\n", position);
      return -1;
    }
    if (verbose)
      fprintf (stderr, " RA (rad) = %lf\n", source.raj);

    if (ascii_header_get (header, "DEC", "%s", position) != 1)
    {
      fprintf (stderr, "ERROR:  could not read RA from header\n");
      return -1;
    }
    if (verbose)
      fprintf (stderr, " DEC (DMS) = %s\n", position);
    if (mopsr_delays_ddmmss_to_rad (position, &(source.decj)) < 0)
    {
      fprintf (stderr, "ERROR:  could not parse DEC from %s\n", position);
      return -1;
    }
    if (verbose)
      fprintf (stderr, " DEC (rad) = %lf\n", source.decj);

    if (ascii_header_get (header, "UT1_OFFSET", "%f", ut1_offset) == 1)
    {
      fprintf (stderr, " UT1_OFFSET=%f\n", ut1_offset);
    }
    if (ascii_header_get (header, "UTC_START", "%s", utc_start_str) == 1)
    {
      if (verbose)
        fprintf (stderr, " UTC_START=%s\n", utc_start_str);
    }
    else
    {
      fprintf (stderr, " UTC_START=UNKNOWN\n");
    }
    // convert UTC_START to a unix UTC
    utc_start = str2utctime (utc_start_str);
    if (utc_start == (time_t)-1)
    {
      fprintf (stderr, "ERROR:  could not parse start time from '%s'\n", utc_start_str);
      return -1;
    }

    if (ascii_header_get (header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
    {
      fprintf (stderr, "could not read BYTES_PER_SECOND from header\n");
      return -1;
    }
    if (verbose)
      fprintf (stderr, "BYTES_PER_SECOND=%"PRIu64"\n", bytes_per_second);

    // now calculate the apparent RA and DEC for the current timestamp
    timestamp.tv_sec = utc_start;
    timestamp.tv_usec = (long) (ut1_offset * 1000000);

    struct tm * utc = gmtime (&utc_start);
    cal_app_pos_iau (source.raj, source.decj, utc,
                     &(source.ra_curr), &(source.dec_curr));
  }

  uint64_t start_byte, end_byte, mid_byte;
  char filename[1024];

  for (ifile=0; ifile<nfiles; ifile++)
  {
    i = 0;

    int fd = open (ac_files[ifile], flags, perms);
    if (fd < 0)
    {
      fprintf(stderr, "failed to open dada file[%s]: %s\n", ac_files[ifile], strerror(errno));
      exit(EXIT_FAILURE);
    }

    const char *sep = "/_.";
    char * saveptr;

    strcpy (filename, ac_files[ifile]);
    char * str = strtok_r(filename, sep, &saveptr);

    while (str && strcmp(str, utc_start_str) != 0)
    {
      str = strtok_r(NULL, sep, &saveptr);
    }

    str = strtok_r(NULL, sep, &saveptr);
    if (!str)
    {
      fprintf (stderr, "ac file [%s] not of expected form\n", ac_files[ifile]);
      return -1;
    }
    if (sscanf(str, "%"PRIu64, &start_byte) != 1)
    {
      fprintf (stderr, "could not parse start byte from %s\n", str);
      return -1;
    }

    str = strtok_r(NULL, sep, &saveptr);
    if (!str)
    {
      fprintf (stderr, "ac file [%s] not of expected form\n", ac_files[ifile]);
      return -1;
    }
    if (sscanf(str, "%"PRIu64, &end_byte) != 1)
    {
      fprintf (stderr, "could not parse end byte from %s\n", str);
      return -1;
    }

    mid_byte = (end_byte + start_byte) / 2;
    double mid_utc_offset = (double) mid_byte / (double) bytes_per_second;

    if (verbose > 1)
      fprintf (stderr, "[%d] start_byte=%"PRIu64", end_byte=%"PRIu64" mid_byte=%"PRIu64" seconds=%lf\n", ifile, start_byte, end_byte, mid_byte, mid_utc_offset);



    for (iant=0; iant<nant; iant++)
    {
      // read in spectra ant
      if (iant == acs)
      {
        amp_t = 0;
        read (fd, (void *) ac, bytes_to_read);

        for (ipt=0; ipt<npt; ipt++)
        {
          if (ipt == npt/2)
            val = 0;
          else
            val = ac[ipt];

          amp_t += sqrtf(val);
        }

        good_file = 1;
        if (ifile > 0)
        {
          amp_mean = amp_total / good_sums;
          if (amp_t > 4 * amp_mean)
            good_file = 0;
        }

        good_file = 1;
        if (good_file)
        {
          amp_total += amp_t;
          good_sums++;
        }
        else
        {
          amp_t = 0;
        }

        amps_t[ifile] = amp_t;
      }
      else
      {
        lseek (fd, bytes_to_read, SEEK_CUR);
      }

      i++;
    }

    if (plot_meridian_dist)
    {
      double offset_seconds = mid_utc_offset + ut1_offset;
      timestamp.tv_sec = utc_start + (long) floor(offset_seconds);
      timestamp.tv_usec = (long) (offset_seconds - floor(offset_seconds));

      double jer_delay = calc_jer_delay (source.ra_curr, source.dec_curr, timestamp);
      x_t[ifile] = (float) asin(calc_jer_delay (source.ra_curr, source.dec_curr, timestamp)) * (180 / M_PI);
    }
    else
      x_t[ifile] = (float) mid_utc_offset;

    close (fd);
  }
  sprintf (title, "Antenna %d - Amp vs Time", acs);
  plot_phase_time (x_t, amps_t, ifile, device, title);

  free (ac);
  free (amps);
  free (amps_t);


  return EXIT_SUCCESS;
}

void plot_phase_time (float * xvals, float * amps, unsigned npts, char * device, char * title)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  char label[64];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    if (amps[i] > ymax)
      ymax = amps[i];
    if (amps[i] < ymin)
      ymin = amps[i];
    if (xvals[i] > xmax)
      xmax = xvals[i];
    if (xvals[i] < xmin)
      xmin = xvals[i];
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  //ymin = 0;
  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time [s]", "Amplitude", title);

  cpgsci(3);
  cpgline (npts, xvals, amps);
  cpgsci(1);

  cpgebuf();
  cpgend();
}
