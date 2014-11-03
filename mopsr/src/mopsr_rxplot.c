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

#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_rxplot [options] dumpfile\n"
     " -m module        plot specific module [0-3]\n"
     " -p plot          produce the specified plot\n"
     " -l               plot logarithmically\n"
     " -D device        pgplot device name [default create PNG files]\n"
     " -g <xres>x<yres> plot with resolution x pixels by y pixels\n"
     " -z               zap DC channel\n"
     " -v               be verbose\n\n"
     " plot can be either:\n" 
     "   hg             Histogram\n"
     "   sp             Power Spectrum\n"
     "   ts             Time Series\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = 0;

  // Flags for plotting specified plots
  int module = -1;
  int plot_hg = 0;
  int plot_sp = 0;
  int plot_ts = 0;
  unsigned int xres = 1024;
  unsigned int yres = 768;

  // Mopsr UDP header
  mopsr_hdr_t hdr;

  // Plotting options
  mopsr_util_t opts;

  // plotting defaults
  opts.plot_log   = 0;
  opts.zap        = 0;
  opts.ant        = -1;
  opts.chans[0]   = -1;
  opts.chans[1]   = 127;
  opts.nbin       = 64;
  opts.ndim       = 2;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;
  opts.lock_flag  = -1;

  int arg = 0;

  while ((arg=getopt(argc,argv,"D:g:hlm:p:vz")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'g':
        if (sscanf (optarg, "%ux%u", &xres, &yres) != 2) 
        {
          fprintf (stderr, "ERROR: could not parse width and height from %s\n", optarg);
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'h':
        usage();
        return (EXIT_SUCCESS);

      case 'l':
        opts.plot_log = 1;
        break;

      case 'm':
        module = atoi(optarg);
        break;

      case 'p':
        if (optarg)
        {
          if (strcmp(optarg, "sp") == 0)
            plot_sp = 1;
          else if (strcmp(optarg, "hg") == 0)
            plot_hg = 1;
          else if (strcmp(optarg, "ts") == 0)
            plot_ts = 1;
          else
          {
            fprintf (stderr, "ERROR: unrecognized plot type [%s]\n", optarg);
            usage();
            return (EXIT_FAILURE);
          }
          break;
        }
        else
        {
          fprintf (stderr, "ERROR: -p requires argument\n");
          usage();
          return (EXIT_FAILURE);
        }

      case 'v':
        verbose++;
        break;

      case 'z':
        opts.zap = 1;
        break;

      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char filename[256];
  char png_file[256];
  strcpy(filename, argv[optind]);

  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dump file [%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", filename, filesize);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (filename, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dump file[%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "reading %lu bytes\n",filesize);
  void * raw = malloc (filesize);
  size_t bytes_read = read (fd, raw, filesize);

  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  // close the FD now that we are done with it.
  close(fd);

  opts.nchan = 128;
  opts.ndim = 1;
  opts.nbin = 64;
  opts.nant = 1;
  unsigned int nmod = 4;
#ifdef HAVE_TS
  unsigned npts = 512;
  float timeseries_f[npts];
#endif

  if (verbose)
    fprintf (stderr, "NANT=%u, NCHAN=%u, NDIM=%u\n", opts.nant, opts.nchan, opts.ndim);

  unsigned int imod;
  char local_time[32];

  // perform SQLD on buffer / frame
  float * bandpass;
  unsigned int * histogram;

  if (xres < 300)
    opts.plot_plain = 1;
  else
    opts.plot_plain = 0;

  void * ptr;

  unsigned int spectrum_stride = sizeof(float) * opts.nchan;
  unsigned int histogram_stride = sizeof(unsigned int) * opts.nbin;
#ifdef HAVE_TS
  unsigned int timeseries_stride = sizeof(int8_t) * npts ;
  unsigned int mod_stride = spectrum_stride + 2 * histogram_stride + 2 * timeseries_stride;
#else
  unsigned int mod_stride = spectrum_stride + 2 * histogram_stride;
#endif

  unsigned int start_module, end_module;

  if (module == -1)
  {
    start_module = 0;
    end_module = nmod;
  }
  else
  {
    start_module = module;
    end_module = start_module + 1;
  }

  // for each antenna / module
  for (imod=start_module; imod<end_module; imod++)
  {
    ptr = raw + imod * mod_stride;

    // unpack bandpass from 
    bandpass = (float*) ptr;

    opts.ant_id = imod;

    if (plot_sp)
    {
      // plot the bandpass for an antenna
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".bin");
        sprintf (str_ptr, ".%d.sp.%dx%d.png/png", imod, xres, yres);
      }
      else
        sprintf (png_file, "%s", device);
      
      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
         fprintf (stderr, "mopsr_rxplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        set_white_on_black();
        mopsr_plot_bandpass (bandpass, &opts);
        cpgend();
      }
    }

    ptr += spectrum_stride;
    histogram = (unsigned int *) ptr;

    if (plot_hg)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".bin");
        sprintf (str_ptr, ".%d.hg.%dx%d.png/png", imod, xres, yres);
      }
      else
        sprintf (png_file, "%s", device);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_rxplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        set_white_on_black();
        mopsr_plot_histogram (histogram, &opts);
        cpgend();
      }
    }

#ifdef HAVE_TS
    ptr += histrogram_stride;
    timeseries = (int8_t *) ptr;

    if (plot_ts)
    {
      // assumes Re,Im not interleave in raw
      for (ipt=0; ipt < npts; ipt++)
      {
        timeseries_f[2*ipt]   = (float) timeseries[ipt];
        timeseries_f[2*ipt+1] = (float) timeseries[npts+ipt];
      }
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".bin");
        sprintf (str_ptr, ".%d.ts.%dx%d.png/png", imod, xres, yres);
      }
      else
        sprintf (png_file, "%s", device);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_rxplot: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres, yres);
        set_white_on_black();
        mopsr_plot_time_series (timeseries_f, 0, npts, opts);
        cpgend();
      }
    }
#endif
  }

  free (raw);

  return EXIT_SUCCESS;
}
