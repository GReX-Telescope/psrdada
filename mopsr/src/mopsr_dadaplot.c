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
     "mopsr_dadaplot [options] dadafile\n"
     " -l         plot logarithmically\n"
     " -D device  pgplot device name [default create PNG files]\n"
     " -z         zap DC channel\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = 0;

  // Mopsr UDP header
  mopsr_hdr_t hdr;

  // Plotting options
  mopsr_util_t opts;

  // plotting defaults
  opts.lock_flag  = -1;
  opts.plot_log   = 0;
  opts.zap        = 0;
  opts.ant        = 0;
  opts.chans[0]   = 0;
  opts.chans[1]   = 40;
  opts.nbin       = 256;
  opts.ndim       = 2;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;

  int arg = 0;

  int ant = 0;

  while ((arg=getopt(argc,argv,"D:lvz")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'l':
        opts.plot_log = 1;
        break;

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
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
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
    fprintf(stderr, "failed to open dada file[%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  char * header = (char *) malloc (4096);
  char * raw = (char *) malloc (filesize - 4096);

  if (verbose)
    fprintf (stderr, "reading header, 4096 bytes\n");
  size_t bytes_read = read (fd, header, 4096);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  size_t data_size = filesize - 4096;
  if (verbose)
    fprintf (stderr, "reading data %lu bytes\n", data_size);
  bytes_read = read (fd, raw, data_size);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  // close the FD now that we are done with it.
  close(fd);

  hdr.nchan = 40;
  hdr.nant  = 4;
  hdr.nframe = 16384;
  hdr.ant_id = 1;
  hdr.ant_id2 = 1;

  // override the data size
  data_size = hdr.nframe * hdr.nchan * hdr.nant * 2;

  opts.nchan = hdr.nchan;
  opts.nant  = hdr.nant;

  if (verbose)
    fprintf (stderr, "NANT=%u, NCHAN=%u, NDIM=%u, NSAMP=%u\n", opts.nant, opts.nchan, opts.ndim, hdr.nframe);

  unsigned int nsamp = data_size / (hdr.nchan * hdr.nant * 2);
  size_t nbytes = data_size;

  unsigned int iframe, ichan, iant, ires;
  unsigned int nframe = nsamp;

  char local_time[32];

  // perform SQLD on buffer / frame
  float * bandpass = (float *) malloc (sizeof(float) * opts.nchan);
  float * spectra = (float *) malloc (sizeof(float) * opts.nchan);
  float * waterfall = (float *) malloc (sizeof(float) * nframe * opts.nchan);
  float * waterfall_h = (float *) malloc (sizeof(float) * nframe * opts.nchan);
  float * timeseries = (float *) malloc (sizeof(float) * nframe * opts.ndim);
  unsigned int * histogram = (unsigned int *) malloc (sizeof(unsigned int) * opts.ndim * opts.nbin);

  int xres[2];
  int yres[2];
  int plot_plains[2];
  unsigned int start_ant, end_ant, nres;

  if (!device)
  {
    start_ant = 0;
    end_ant = opts.nant;
    nres = 2;
    xres[0] = 1024;
    yres[0] = 768;
    plot_plains[0] = 0;
    xres[1] = 160;
    yres[1] = 120;
    plot_plains[1] = 1;
  }
  else
  {
    start_ant = ant;
    end_ant = ant+1;
    nres = 1;
    xres[0] = 400;
    yres[0] = 300;
    plot_plains[0] = 0;
  }

  for (iant=start_ant; iant<end_ant; iant++)
  {
    // TODO - revert! special case for 4 Input mode
    if ((iant == 0) || (iant == 1))
    {
      opts.ant_id = mopsr_get_ant_number (hdr.ant_id, iant);
      opts.lock_flag  = hdr.mf_lock;
    }
    else
    {
      opts.ant_id = mopsr_get_ant_number (hdr.ant_id2, iant-2);
      opts.lock_flag  = hdr.mf_lock2;
    }

    if (verbose)
      fprintf (stderr, "mopsr_dadaplot: iant=%d, hdr.ant_id=%d, hdr.ant_id2=%d, opts.ant_id=%d\n", iant, hdr.ant_id, hdr.ant_id2, opts.ant_id);
    opts.ant = iant;

    int chan = opts.chans[0];
    if (opts.chans[0] == -1)
      chan = 50;
    else
      chan = (opts.chans[0] + opts.chans[1]) / 2;

    mopsr_extract_channel (timeseries, raw, nbytes, chan, iant, opts.nchan, opts.nant);
    for (ires=0; ires<nres; ires++)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.ts.%dx%d.png/png", opts.ant_id, xres[ires], yres[ires]);
      }
      else
        sprintf (png_file, "%d/xs", 1);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgopen (png_file) != 1)
        fprintf (stderr, "mopsr_dadaplot: error opening plot device [%s]\n", png_file);

      set_resolution (xres[ires], yres[ires]);
      opts.plot_plain = plot_plains[ires];
      mopsr_plot_time_series (timeseries, chan, nframe, &opts);
      cpgclos();
    }

    mopsr_zero_float (bandpass, opts.nchan);

    // these operations work for all antenna [perhaps should change this?]
    for (iframe = 0; iframe < nframe; iframe++)
    {
      // SQLD this antenna
      mopsr_sqld_pfbframe (spectra, raw + (iframe * opts.nchan * opts.nant * opts.ndim), &opts, 0);

      // copy spectra into waterfall array
      memcpy (waterfall + (iframe * opts.nchan), spectra, opts.nchan * sizeof(float));

      // sum spectra into bandpass plot
      for (ichan=0; ichan < opts.nchan; ichan++)
        bandpass[ichan] += spectra[ichan];
    }

    // plot the bandpass for an antenna
    for (ires=0; ires<nres; ires++)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.bp.%dx%d.png/png", opts.ant_id, xres[ires], yres[ires]);
      }
      else
        sprintf (png_file, "%d/xs", 2);
    
      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres[ires], yres[ires]);
        opts.plot_plain = plot_plains[ires];
        mopsr_plot_bandpass(bandpass, &opts);
        cpgend();
      }
    }

    // waterfall plot
    mopsr_transpose (waterfall_h, waterfall, nframe, &opts);
    for (ires=0; ires<nres; ires++)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.wf.%dx%d.png/png", opts.ant_id, xres[ires], yres[ires]);
      }
      else
        sprintf (png_file, "%d/xs", 3);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres[ires], yres[ires]);
        opts.plot_plain = plot_plains[ires];
        mopsr_plot_waterfall (waterfall_h, nframe, &opts);
        cpgend();
      }
    }

    // count histogram statistics
    mopsr_form_histogram (histogram, raw, nbytes, &opts);
    for (ires=0; ires<nres; ires++)
    {
      if (!device)
      {
        strcpy(png_file, filename);
        char * str_ptr = strstr (png_file, ".dump");
        sprintf (str_ptr, ".%d.hg.%dx%d.png/png", opts.ant_id, xres[ires], yres[ires]);
      }
      else
        sprintf (png_file, "%d/xs", 4);

      if (verbose)
        fprintf (stderr, "creating %s\n", png_file);
      if (cpgbeg(0, png_file, 1, 1) != 1)
        fprintf (stderr, "mopsr_dbmonitor: error opening plot device [%s]\n", png_file);
      else
      {
        set_resolution (xres[ires], yres[ires]);
        opts.plot_plain = plot_plains[ires];
        mopsr_plot_histogram (histogram, &opts);
        cpgend();
      }
    }
  }

  free (raw);
  free (bandpass);
  free (spectra);
  free (waterfall);
  free (waterfall_h);
  free (timeseries);
  free (histogram);

  return EXIT_SUCCESS;
}
