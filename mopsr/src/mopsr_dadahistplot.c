/*
 * read a dada file from disk and play through histogram plot
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
#include <math.h>
#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage ();
void usage()
{
  fprintf (stdout,
     "mopsr_dadahistogramplot [options] dadafile antenna\n"
     " dadafile    PSRDada raw file (MOPSR format)\n"
     " antenna     antenna to display\n"
     " -c chan     channel to show (default all)\n"
     " -D device   pgplot device name\n"
     " -h          print usage\n"
     " -t num      number of time samples to display in each plot [default 512]\n"
     " -v          be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int nsamp = 512;

  mopsr_util_t opts;

  opts.lock_flag = 1;
  opts.lock_flag_long = 1;
  opts.ndim = 2;
  opts.plot_plain = 0;
  opts.zap = 0;

  int channel = -1;

  while ((arg=getopt(argc,argv,"c:D:hn:t:v")) != -1)
  {
    switch (arg)
    {
      case 'c':
        channel = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'h':
        usage();
        return 0;

      case 't':
        nsamp = atoi (optarg);
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 1;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  char filename[256];
  strcpy(filename, argv[optind]);
  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
    return (EXIT_FAILURE);
  }

  if (sscanf(argv[optind+1], "%d", &(opts.ant)) != 1)
  {
    fprintf (stderr, "ERROR: failed parse antenna from %s\n", argv[optind+1]);
    return (EXIT_FAILURE);
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

  size_t data_size = filesize - 4096;

  char * header = (char *) malloc (4096);
  if (verbose)
    fprintf (stderr, "reading header, 4096 bytes\n");
  size_t bytes_read = read (fd, header, 4096);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  if (ascii_header_get(header, "NCHAN", "%d", &(opts.nchan)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }
  if (channel == -1)
  {
    opts.chans[0] = 0;
    opts.chans[1] = opts.nchan-1;
  }
  else
  {
    opts.chans[0] = opts.nchan;
    opts.chans[1] = opts.nchan;
  }


  if (ascii_header_get(header, "NANT", "%d", &(opts.nant)) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }

  if (header)
    free(header);
  header = 0;

  if (verbose)
    fprintf(stderr, "mopsr_dadahistogramplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_dadahistogramplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  fprintf (stderr, "nchan=%d nsamp=%d nant=%d\n", opts.nchan, nsamp, opts.nant);

  opts.nbin = 256;

  unsigned * histogram = (unsigned *) malloc (sizeof(unsigned) * opts.nbin * opts.nchan * opts.ndim);

  size_t bytes_read_total = 0;
  const size_t bytes_to_read = nsamp * opts.nchan * opts.nant * opts.ndim;
  void * raw = malloc (bytes_to_read);
  unsigned isamp, ichan, iant; 
  size_t raw_stride = opts.nchan * opts.nant * opts.ndim;

  fprintf (stderr, "bytes_to_read=%ld\n", bytes_to_read);

  memset (histogram, 0, sizeof(unsigned ) * opts.nbin * opts.ndim);

  int8_t re, im;
  int re_bin, im_bin;

  while (bytes_read_total + bytes_to_read < data_size)
  {
    bytes_read = read (fd, raw, bytes_to_read);
    if (verbose)
      fprintf (stderr, "read %ld bytes, total now %ld\n", bytes_read, bytes_read_total);

    float re_sum = 0;
    float im_sum = 0;

    if (bytes_read == bytes_to_read)
    {
      bytes_read_total += bytes_read;
      int8_t * in = (int8_t *) raw;

      // input ordering is TFS
      for (isamp=0; isamp<nsamp; isamp++)
      {
        for (ichan=0; ichan<opts.nchan; ichan++)
        {
          for (iant=0; iant<opts.nant; iant++)
          {
            if ((channel == -1 || ichan == channel) && iant == opts.ant)
            {
              re = in[0];
              im = in[1];

              re_sum += (float) re + 0.38;
              im_sum += (float) im + 0.38;

              re_bin = ((int) re) + 128;
              im_bin = ((int) im) + 128;

              if (re_bin > 255)
                re_bin == 255;
              if (im_bin > 255)
                im_bin == 255;

              histogram[re_bin]++;
              histogram[opts.nbin + im_bin]++;
            }
            in += 2;
          }
        }
      }

      int idim, ibin;
      int blank = 70;
      for (ibin=(128-blank); ibin<(128+blank); ibin++)
      {
        histogram[ibin]= 0;
        histogram[opts.nbin + ibin] = 0;
      }

      float nval = nsamp;
      if (channel == -1)
        nval *= opts.nchan;

      float re_mean = (float) re_sum / nval;
      float im_mean = (float) im_sum / nval;

      mopsr_plot_histogram (histogram, &opts);

      fprintf (stderr, "means re=%f im=%f\n", re_mean, im_mean);

      for (idim=0; idim<opts.ndim; idim++)
        for (ibin=0; ibin<opts.nbin; ibin++)
          histogram[idim*opts.nbin + ibin] = 0;

      usleep (1000000);
    }
  }

  cpgclos();
  close (fd);

  if (histogram)
    free (histogram);
  histogram = 0;

  if (raw)
    free (raw);
  raw = 0;

  return EXIT_SUCCESS;
}

