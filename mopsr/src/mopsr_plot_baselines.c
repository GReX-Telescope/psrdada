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
void plot_phase_freq (float * amps, float * phases, unsigned npts, float gradient, float offset, char * device, char * title);
void plot_phase_time (float * x, float * amps, float * phases, unsigned npts, float gradient, float offset, char * device, char * title, float nsecs_per_subint);
void plot_series (double * x, double * y, unsigned npts, char * device, char * title, float M, float C);
void unwrap (double * p, int N);

void usage()
{
  fprintf (stdout,
     "mopsr_plot_baselines baseline npt ccfiles\n"
     " -f delay        adjust fractional delay\n"
     " -D device       pgplot device name\n"
     " -h              plot this help\n"
     " -p              plot each timestep\n"
     " -r rejectfile   reject channels listed in this file\n"
     " -t file         write phase vs time output to text file instread\n"
     " -s secs         number of seconds per subint [default 20]\n"
     " -v              be verbose\n"
     " -x              print xcorr vs time amplitudes\n"
     " -z              zap channel npt/2\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  char * textfile = 0;

  char plot_all = 0;

  int baseline = 0;

  int arg = 0;

  float nsecs_per_subint = 20;

  float delay = 0;

  char plot_meridian_dist = 0;

  char zap_dc = 0;

  char just_xcorr_time_amps = 0;

  char * channel_reject_file = NULL;
  char channel_rejection = 0;


  while ((arg=getopt(argc,argv,"f:D:hmpr:s:t:vxz")) != -1)
  {
    switch (arg)
    {
      case 'f':
        delay = atof (optarg);
        break;
        
      case 'D':
        device = strdup(optarg);
        break;

      case 's':
        nsecs_per_subint = atof (optarg);
        break;

      case 't':
        textfile = strdup(optarg);
        break;

      case 'm':
        plot_meridian_dist = 1;
        break;

      case 'p':
        plot_all++;
        break;

      case 'r':
        channel_reject_file = strdup(optarg);
        channel_rejection = 1;
        break;

      case 'v':
        verbose++;
        break;

      case 'x':
        just_xcorr_time_amps = 1;
        break;

      case 'z':
        zap_dc = 1;
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

  mopsr_source_t source;
  struct timeval timestamp;

  // check if there exists a obs.baselines file in the current directory
  // if so, read the baseline pair
  char * baselines_file = "obs.baselines";
  FILE * fptr = fopen (baselines_file, "r");
  time_t utc_start;
  char utc_start_str[20];
  char anta_name[6];
  char antb_name[6];
  char baseline_name[16];
  char line[1024];
  sprintf (baseline_name, "Baseline %d", baseline);
  if (fptr)
  {
    int nscanned;
    unsigned ibaseline = 0;
    while ((ibaseline <= baseline) && fgets(line, 1024, fptr))
    {
      if (ibaseline == baseline)
      {
        nscanned = sscanf(line, "%5s %5s", anta_name, antb_name);
        sprintf (baseline_name, "%s %s", anta_name, antb_name);
      }
      ibaseline++;
    }
    fclose (fptr);
  }
  fptr = 0;

  int num_reject_channels = 0;
  int * channel_reject = NULL;
  if (channel_rejection)
  {
    FILE * fptr = fopen (channel_reject_file, "r");
    if (!fptr)
    {
      fprintf (stderr, "ERROR: failed to open file: %s for reading: %s\n", channel_reject_file, strerror(errno));
      return (EXIT_FAILURE);
    }

    char line1[1024];
    int i;
    fgets (line1, 1024, fptr);
    num_reject_channels = atoi(line1);

    channel_reject = (int*) malloc (num_reject_channels * sizeof(int));
    for(i=0;i<num_reject_channels;i++)
    {
      fgets(line1, 1024, fptr);
      channel_reject[i] = atoi(line1);
    }
  }

  uint64_t bytes_per_second;
  float ut1_offset;

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

    if (ascii_header_get (header, "UT1_OFFSET", "%f", &ut1_offset) == 1)
    {
    }
    if (verbose)
      fprintf (stderr, " UT1_OFFSET=%f\n", ut1_offset);
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

    // now calculate the apparent RA and DEC for the utc start
    struct tm * utc = gmtime (&utc_start);
    cal_app_pos_iau (source.raj, source.decj, utc,
                      &(source.ra_curr), &(source.dec_curr));

  }

  unsigned nfiles = argc - (optind + 2);
  if (verbose)
    fprintf (stderr, "nfiles=%u\n", nfiles);
  char ** cc_files = (char **) malloc (sizeof(char *) * nfiles);
  unsigned ifile;
  for (ifile=0; ifile<nfiles; ifile++)
  {
    size_t len = strlen(argv[optind+2+ifile]);
    cc_files[ifile] = (char *) malloc (sizeof(char) * (len + 1));
    strcpy (cc_files[ifile], argv[optind+2+ifile]);
  }

  size_t file_size;
  struct stat buf;
  if (verbose)
    fprintf (stderr, "stating %s\n", cc_files[0]);
  if (stat (cc_files[0], &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", cc_files[0], strerror(errno));
    return (EXIT_FAILURE);
  }

  file_size = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", cc_files[0], file_size);

  unsigned npairs = (unsigned) (file_size / (npt * sizeof(complex float)));
  unsigned nant = (1 + sqrt(1 + (8 * npairs))) / 2;
  if (verbose)
    fprintf (stderr, "npairs=%u nant=%u \n", npairs, nant);

  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;

  complex float * tmp  = (complex float *) malloc (npt * sizeof(complex float));
  complex float * sum  = (complex float *) malloc (npt * sizeof(complex float));
  float * amps   = (float *) malloc (npt * sizeof(float));
  float * phases = (float *) malloc (npt * sizeof(float));
  complex float * cc = (complex float *)  malloc (npt * sizeof(complex float));
  complex float * ramps = (complex float *)  malloc (npt * sizeof(complex float));

  memset (amps, 0, npt * sizeof(float));
  memset (phases, 0, npt * sizeof(float));
  memset (sum, 0, npt * sizeof(complex float));

  float * amps_t = (float *) malloc (nfiles * sizeof(float));
  float * phases_t = (float *) malloc (nfiles * sizeof(float));
  float * x_t = (float *) malloc (nfiles * sizeof(float));

  unsigned N_skip = npt / 20;
  unsigned N = npt - 1;

  double * x1 = (double *) malloc (npt* sizeof(double));
  double * y1 = (double *) malloc (npt* sizeof(double));
  double * x2 = (double *) malloc (nfiles * sizeof(double));
  double * y2 = (double *) malloc (nfiles* sizeof(double));

  unsigned ipt, ipair;
  float re, im;
  int shift;

  char title[1024];
  size_t bytes_to_read = npt * sizeof(complex float);

  unsigned i = 1;
  unsigned tri = i;

  int anta = 0;
  int antb = 0;

  float theta;
  complex float val;
  complex float fsum;

  for (ipt=0; ipt<npt; ipt++)
  {
    theta = delay * 2 * M_PI * (((float) ipt / (float) npt) - 0.5);
    ramps[ipt] = cos (theta) + sin(theta) * I;
  }

  for (ipt=0; ipt<npt; ipt++)
  {
    x1[ipt] = ((double) (ipt) / (double) (npt-1)) * 2 * M_PI;
    x1[ipt] = (double) ipt;
  }
  for (ipt=0; ipt<nfiles;ipt++)
  {
    x2[ipt] = (double) (ipt) * nsecs_per_subint;
  }

  double c0, c1, cov00, cov01, cov11, chisq;

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
  float phase_t, amp_t, fsum_bp;
  uint64_t start_byte, end_byte, mid_byte;
  char filename[1024];

  fptr = 0;
  if (textfile)
  {
    fptr = fopen(textfile, "w");
    if (!fptr)
    {
      fprintf (stderr, "ERROR: could not open text file: %s\n", textfile);
      exit (EXIT_FAILURE);
    }
    else
    {
      fprintf (fptr, "# Amplitude, Phase [radians]\n");
    }
  }

  for (ifile=0; ifile<nfiles; ifile++)
  {
    i = 1;
    tri = i;

    if (verbose)
      fprintf (stderr, "[%d] %s\n", ifile, cc_files[ifile]);

    const char *sep = "/_.";
    char * saveptr;

    strcpy (filename, cc_files[ifile]);
    char * str = strtok_r(filename, sep, &saveptr);

    while (str && strcmp(str, utc_start_str) != 0)
    {
      str = strtok_r(NULL, sep, &saveptr);
    }

    str = strtok_r(NULL, sep, &saveptr);
    if (!str)
    {
      fprintf (stderr, "cc file [%s] not of expected form\n", cc_files[ifile]);
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
      fprintf (stderr, "cc file [%s] not of expected form\n", cc_files[ifile]);
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

    int fd = open (cc_files[ifile], flags, perms);
    if (fd < 0)
    {
      fprintf(stderr, "failed to open dada file[%s]: %s\n", cc_files[ifile], strerror(errno));
      exit(EXIT_FAILURE);
    }

    for (ipair=0; ipair<npairs; ipair++)
    {
      assert (i < (nant * nant));

      // read in spectra pair
      if (ipair == baseline)
      {
        read (fd, (void *) cc, bytes_to_read);

        fsum = 0;
        fsum_bp = 0;

        if (channel_rejection)
        {
          for (ipt=0; ipt<num_reject_channels; ipt++)
          {
            cc[channel_reject[ipt]] = 0;
          }
        }

        for (ipt=0; ipt<npt; ipt++)
        {
          
          if (zap_dc && (ipt == npt/2))
            val = 0;
          else
            val = cc[ipt];

          if (delay != 0)
            val *= ramps[ipt];

          {
            fsum += val;
          }

          tmp[ipt] = val;
        }

        re = crealf(fsum);
        im = cimagf(fsum);

        phase_t = atan2f(im, re);
        amp_t   = sqrt((re * re) + (im * im));

        good_file = 1;
        if (ifile > 2)
        {
          amp_mean = amp_total / good_sums;
          if (verbose > 1)
            fprintf (stderr, "[%d] amp_t=%f amp_total=%lf good_sums=%d, "
                             "amp_mean=%lf\n", ifile, amp_t, amp_total, 
                             good_sums, amp_mean);
          if (amp_t > 10 * amp_mean)
            good_file = 0;
        }

        good_file = 1;

        if (good_file)
        {
          for (ipt=0; ipt<npt; ipt++)
            sum[ipt] += tmp[ipt];
          amp_total += amp_t;
          good_sums++;
        }
        else
        {
          phase_t = 0;
          amp_t = 1;
        }

        phases_t[ifile] = phase_t;
        amps_t[ifile] = amp_t;

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

        anta = i / nant;
        antb = i % nant;
      }
      else
      {
        lseek (fd, bytes_to_read, SEEK_CUR);
      }

      i++;
      if (i % nant == 0)
      {
        tri++;
        i += tri;
      }
    }

    if (fptr)
    {
      fprintf (fptr, "%e,%f\n", amps_t[ifile], phases_t[ifile]);
    }
    else
    {
      if (plot_all || (ifile == (nfiles-1)))
      {
        for (ipt=0; ipt<npt; ipt++)
        {
          re = crealf(sum[ipt]);
          im = cimagf(sum[ipt]);
          amps[ipt] = (re * re) + (im * im);
          phases[ipt] = atan2f(im, re);
        }

        for (ipt=0; ipt<npt/2; ipt++)
        {
          y1[ipt] = (double) atan2f (cimagf(sum[ipt]), crealf(sum[ipt]));
        }

        for (ipt=(npt/2)+1; ipt<npt; ipt++)
        {
          y1[ipt-1] = (double) atan2f (cimagf(sum[ipt]), crealf(sum[ipt]));
        }

        c0 = 0;

        unwrap (y1, N);

        gsl_fit_linear (x1 + N_skip, 1, y1 + N_skip, 1, (size_t) (N - 2 * N_skip), &c0, &c1, &cov00, &cov01, &cov11, &chisq);
        
        float fractional_delay = (float) c1 * (npt / (2 * M_PI));

        if (!just_xcorr_time_amps)
        {
          sprintf (title, "%s [%d -> %d] Corr vs Freq uncorrected delay %f samples", baseline_name, anta, antb, fractional_delay);
          plot_phase_freq (amps, phases, npt, (float) c1, (float) c0, "10/xs", title);
        }

        c0 = 0;
        c1 = 0;

        if (ifile > 0)
        {
          for (ipt=0; ipt<ifile; ipt++)
          {
            y2[ipt] = phases_t[ipt];
          }

          unwrap (y2, ifile+1);

          gsl_fit_linear (x2, 1, y2, 1, (size_t) ifile+1, &c0, &c1, &cov00, &cov01, &cov11, &chisq);
        }

        float turns_per_hour = (float) c1 * (3600 / (2 * M_PI));

        if (just_xcorr_time_amps)
        {
          fprintf (stdout, "%f", amps_t[0]);
          for (i=1; i<ifile+1; i++)
            fprintf (stdout, ",%f", amps_t[i]);
          fprintf (stdout, "\n");
        }
        else
        {
        sprintf (title, "%s [%d -> %d] (%d) - xCorr vs Time [%4.2f tphr, chiq=%5.2lf] ", baseline_name, anta, antb, baseline, turns_per_hour, chisq);
        if (plot_meridian_dist)
          plot_phase_time (x_t, amps_t, phases_t, ifile+1, (float) c1, (float) c0, device, title, 0);
        else
          plot_phase_time (x_t, amps_t, phases_t, ifile+1, (float) c1, (float) c0, device, title, nsecs_per_subint);
        }
        //if ((fabs(turns_per_hour) > 1) && (chisq < 200) && (abs(anta - antb) > 3))
        //  sleep (1);
      }
    }

    close (fd);

    if (plot_all)
    {
      if (plot_all > 1)
      {
        memset (sum, 0, npt * sizeof(complex float));
        usleep (100000);
      }     
    }
  }

  free (cc);
  free (amps);
  free (phases);
  free (amps_t);
  free (phases_t);
  free (ramps);

  free (x1);
  free (y1);
  free (x2);
  free (y2);

  return EXIT_SUCCESS;
}

void plot_phase_freq (float * amps, float * phases, unsigned npts, float gradient, float offset, char * device, char * title)
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
    //if (amps[i] > 0)
    //  amps[i] = log10(amps[i]);
    //else
    //  amps[i] = -6;

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
  cpglab("Channel", "Phase [radians]", "");

  cpgsci(3);
  cpgpt(npts, xvals, phases, 17);
  cpgsci(1);

  float xx[2] = { 0, 255 };
  float yy[2];
  yy[0] = gradient * xx[0] + offset;
  yy[1] = gradient * xx[1] + offset;

  cpgsci(2);
  cpgline(2, xx, yy);
  cpgsci(1);


  cpgebuf();
  cpgend();
}

void plot_phase_time (float * xvals, float * amps, float * phases, unsigned npts, float gradient, float offset, char * device, char * title, float nsecs_per_subint)
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

  if (xmin == xmax)
  {
    xmin -= 0.5;
    xmax += 0.5;
  }
  if (ymin == ymax)
  {
    ymin -= 0.5;
    ymax += 0.5;
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

  if (npts > 1)
  {
    cpgsci(3);
    cpgline (npts, xvals, amps);
    cpgsci(1);
  }
  else
  {
    cpgsci(3);
    cpgpt(npts, xvals, amps, 17);
    cpgsci(1);
  }

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
  if (nsecs_per_subint == 0)
    cpglab("Merdian Distance Angle [degrees]", "Phase [radians]", "");
  else
    cpglab("Time [ses]", "Phase [radians]", "");

  cpgsci(3);
  cpgpt(npts, xvals, phases, 17);
  cpgsci(1);

  if (nsecs_per_subint != 0)
  {
    float xx[2] = { 0, xvals[npts-1] };
    float yy[2];
    yy[0] = gradient * xx[0] + offset;
    yy[1] = gradient * xx[1] + offset;

    cpgsci(2);
    cpgline(2, xx, yy);
    cpgsci(1);
  }

  cpgebuf();
  cpgend();
}

#define MAX_LENGTH 1280
void unwrap(double * p, int N)
{
  double dp[MAX_LENGTH];
  double dps[MAX_LENGTH];
  double dp_corr[MAX_LENGTH];
  double cumsum[MAX_LENGTH];
  double cutoff = M_PI;               /* default value in matlab */
  int j;

  assert(N <= MAX_LENGTH);

 // incremental phase variation 
 // MATLAB: dp = diff(p, 1, 1);
  for (j = 0; j < N-1; j++)
    dp[j] = p[j+1] - p[j];

 // equivalent phase variation in [-pi, pi]
 // MATLAB: dps = mod(dp+dp,2*pi) - pi;
  for (j = 0; j < N-1; j++)
    dps[j] = (dp[j]+M_PI) - floor((dp[j]+M_PI) / (2*M_PI))*(2*M_PI) - M_PI;

 // preserve variation sign for +pi vs. -pi
 // MATLAB: dps(dps==pi & dp>0,:) = pi;
  for (j = 0; j < N-1; j++)
    if ((dps[j] == -M_PI) && (dp[j] > 0))
      dps[j] = M_PI;

 // incremental phase correction
 // MATLAB: dp_corr = dps - dp;
  for (j = 0; j < N-1; j++)
    dp_corr[j] = dps[j] - dp[j];

 // Ignore correction when incremental variation is smaller than cutoff
 // MATLAB: dp_corr(abs(dp)<cutoff,:) = 0;
  for (j = 0; j < N-1; j++)
    if (fabs(dp[j]) < cutoff)
      dp_corr[j] = 0;

 // Find cumulative sum of deltas
 // MATLAB: cumsum = cumsum(dp_corr, 1);
  cumsum[0] = dp_corr[0];
  for (j = 1; j < N-1; j++)
    cumsum[j] = cumsum[j-1] + dp_corr[j];

 // Integrate corrections and add to P to produce smoothed phase values
 // MATLAB: p(2:m,:) = p(2:m,:) + cumsum(dp_corr,1);
  for (j = 1; j < N; j++)
    p[j] += cumsum[j-1];
}

void plot_series (double * x, double * y, unsigned npts, char * device, char * title, float M, float C)
{
  float xmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float xx[npts];
  float yy[npts];

  unsigned i;
  for (i=0; i<npts; i++)
  {
    xx[i] = (float) x[i];
    yy[i] = (float) y[i];

    if (yy[i] > ymax)
      ymax = yy[i];
    if (yy[i] < ymin)
      ymin = yy[i];

    if (xx[i] > xmax)
      xmax = x[i];
    if (xx[i] < xmin)
      xmin = xx[i];
  }
  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin(xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Channel", "Angle", title);

  cpgsci(2);
  cpgpt (npts, xx, yy, 17);
  cpgsci(1);

  for (i=0; i<npts; i++)
    yy[i] = (M * xx[i]) + C;

  cpgsci(3);
  cpgline (npts, xx, yy);
  cpgsci(1);

  cpgebuf();
  cpgclos();
}

