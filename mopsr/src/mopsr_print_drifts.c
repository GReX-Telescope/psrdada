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
#include <gsl/gsl_fit.h>
#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_delays.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();
void plot_map (float * map, unsigned nx, unsigned ny, char * title, mopsr_util_t * opt);

void usage()
{
  fprintf (stdout,
     "mopsr_print_drifts baseline npt ccfiles\n"
     " -h              plot this help\n"
     " -s secs         number of seconds per subint [default 20]\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  int arg = 0;

  float nsecs_per_subint = 20;

  float delay = 0;

  while ((arg=getopt(argc,argv,"hs:v")) != -1)
  {
    switch (arg)
    {
      case 's':
        nsecs_per_subint = atof (optarg);
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
  if (argc-optind < 2)
  {
    fprintf(stderr, "ERROR: at least 2 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  unsigned npt = 0;
  if (sscanf(argv[optind+0], "%u", &npt) != 1)
  {
    fprintf (stderr, "ERROR: failed to parse npt from %s\n", argv[optind+0]);
    usage();
    return (EXIT_FAILURE);
  }
  if (verbose)
    fprintf (stderr, "npt=%u\n", npt);

  mopsr_source_t source;
  struct timeval timestamp;

  uint64_t bytes_per_second;
  float ut1_offset;

  char * header_file = "obs.header";
  time_t utc_start;
  char utc_start_str[20];

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

    calc_app_position (source.raj, source.decj, timestamp,
                      &(source.ra_curr), &(source.dec_curr));

  }

  unsigned nfiles = argc - (optind + 1);
  if (verbose)
    fprintf (stderr, "nfiles=%u\n", nfiles);
  char ** cc_files = (char **) malloc (sizeof(char *) * nfiles);
  unsigned ifile;
  for (ifile=0; ifile<nfiles; ifile++)
  {
    size_t len = strlen(argv[optind+1+ifile]);
    cc_files[ifile] = (char *) malloc (sizeof(char) * (len + 1));
    strcpy (cc_files[ifile], argv[optind+1+ifile]);
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


  // check if there exists a obs.baselines file in the current directory
  // if so, read the baseline pair
  char baseline_names[npairs][12];
  char * baselines_file = "obs.baselines";
  FILE * fptr = fopen (baselines_file, "r");
  char anta_name[6];
  char antb_name[6];
  char line[1024];
  if (fptr)
  {
    int nscanned;
    unsigned ipair = 0;
    while ((ipair <= npairs) && fgets(line, 1024, fptr))
    {
      nscanned = sscanf(line, "%5s %5s", anta_name, antb_name);
      sprintf (baseline_names[ipair], "%s %s", anta_name, antb_name);
      ipair++;
    }
    fclose (fptr);
  }
  fptr = 0;



  off_t offset;
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;

  //float * amps_t = (float *) malloc (nfiles * sizeof(float));
  //float * phases_t = (float *) malloc (nfiles * sizeof(float));
  //float * x_t = (float *) malloc (nfiles * sizeof(float));

  complex float * cc = (complex float *) malloc (npt * sizeof(complex float));
  double ** x = (double **) malloc (npairs * sizeof(double *));
  double ** y = (double **) malloc (npairs * sizeof(double *));

  unsigned ipair;
  for (ipair=0; ipair<npairs; ipair++)
  {
    x[ipair] = (double *) malloc (nfiles * sizeof(double));
    y[ipair] = (double *) malloc (nfiles * sizeof(double));
  }

  unsigned ipt;
  float re, im;
  int shift;

  size_t bytes_to_read = npt * sizeof(complex float);

  unsigned i = 1;
  unsigned tri = i;

  int anta = 0;
  int antb = 0;

  complex float val;
  complex float fsum;

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
  float phase_t, amp_t;
  uint64_t start_byte, end_byte, mid_byte;
  char filename[1024];

  float * drifts = (float *) malloc (nant * nant * sizeof(float));
  memset(drifts, 0, nant * nant * sizeof(float));

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
    if (verbose)
      fprintf (stderr, "opened %s\n", cc_files[ifile]);

    for (ipair=0; ipair<npairs; ipair++)
    {
      assert (i < (nant * nant));

      read (fd, (void *) cc, bytes_to_read);
      fsum = 0;

      for (ipt=0; ipt<npt; ipt++)
      {
        if (ipt == npt/2)
          val = 0;
        else
          val = cc[ipt];

        fsum += val;
      }

      re = crealf (fsum);
      im = cimagf (fsum);


      phase_t = atan2f(im, re);
      amp_t   = (re * re) + (im * im);

      good_file = 1;
      if (ifile > 0)
      {
        amp_mean = amp_total / good_sums;
        if (verbose > 1)
          fprintf (stderr, "[%d] amp_t=%f amp_total=%lf good_sums=%d, "
                           "amp_mean=%lf\n", ifile, amp_t, amp_total, 
                           good_sums, amp_mean);
        if (amp_t > 5 * amp_mean)
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
        phase_t = 0;
        amp_t = 0;
      }

      y[ipair][ifile] = (double) phase_t;
      x[ipair][ifile] = mid_utc_offset;

      drifts[i] = phase_t;

      //fprintf (stderr, "[%d][%d] = (%f, %f)\n", ipair, ifile, x[ipair][ifile], y[ipair][ifile]);

      anta = i / nant;
      antb = i % nant;


      i++;
      if (i % nant == 0)
      {
        tri++;
        i += tri;
      }
    }
    close (fd);
  }

  for (ipair=0; ipair<npairs; ipair++)
  {
    gsl_fit_linear (x[ipair], 1, y[ipair], 1, nfiles, &c0, &c1, &cov00, &cov01, &cov11, &chisq);
    float turns_per_hour = (float) c1 * (3600 / (2 * M_PI));
    if ((chisq > 0) && (chisq < 100))
      fprintf (stdout, "%d\t%s\t%f\t%f\n", ipair, baseline_names[ipair], turns_per_hour, (float) chisq);
  }

  opt.nant = nant;
  opt.plot_log = 0;
  opt.plot_plain = 0;
  opt.ymin = -0.5;
  opt.ymax = 0.5;

  if (cpgbeg(0, "/xs", 1, 1) != 1)
    fprintf (stderr, "mopsr_solve_delays: error opening plot device [%s]\n", filename);
  plot_map (drifts, nant, nant, "drifts", &opt);
  cpgend();

  for (ipair=0; ipair<npairs; ipair++)
  {
    free (x[ipair]);
    free (y[ipair]);
  }
  free (x);
  free (y);
  free (cc);

  return EXIT_SUCCESS;
}

void plot_map (float * map, unsigned nx, unsigned ny, char * title, mopsr_util_t * opt)
{
  unsigned i;
  float * m = (float *) malloc (sizeof(float) * nx * ny);

  cpgbbuf();
  cpgsci(1);

  if (opt->plot_plain)
    cpgsvp(0.0,1.0,0.0,1.0);
  else
    cpgsvp(0.1,0.9,0.1,0.9);

  cpgswin(0, (float) nx, 0, (float) ny);

  if (!opt->plot_plain)
    cpglab("Ant", "Ant", title);

  float contrast = 1.0;
  float brightness = 0.5;

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float x_min = 0;
  float x_max = (float) nx;

  float y_min = 0;
  float y_max = (float) ny;

  float x_res = (x_max-x_min)/x_max;
  float y_res = (y_max-y_min)/y_max;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = nx * ny;
  float z_min = FLT_MAX;
  float z_max = -FLT_MAX;
  float z_avg = 0;

  if (opt->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opt->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int ix, iy;
  unsigned int ndat_avg = 0;
  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      i = ix * nx + iy;
      m[i] = map[i];
      if (opt->plot_log)
        if (map[i] > 0)
          m[i] = log10(map[i]);
        else
          m[i] = 0;

      if ((opt->ymin == -1) && (opt->ymax == -1))
      {
        if (m[i] > z_max) z_max = m[i];
        if (m[i] < z_min) z_min = m[i];
      }
      else
      {
        z_min =  opt->ymin;
        z_max =  opt->ymax;
      }

      z_avg += m[i];
      ndat_avg++;
    }
  }

  for (ix=0; ix<nx; ix++)
  {
    for (iy=0; iy<ny; iy++)
    {
      i = ix * nx + iy;
      if (opt->plot_log && m[i] == 0)
        m[i] = z_min;
    }
  }

  cpgimag(m, nx, ny, 1, nx, 1, ny, z_min, z_max, trf);
  if (!opt->plot_plain)
    cpgbox("BCGNST", 16, 0, "BCGNST", 16, 0);
  cpgebuf();

  free (m);
}

