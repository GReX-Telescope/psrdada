/*
 * read a dada file from disk and play through powermonitor plot
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
     "mopsr_dadapowermonitorplot [options] dadafile antenna\n"
     " dadafile    PSRDada raw file (MOPSR format)\n"
     " antenna     antenna to display\n"
     " -D device   pgplot device name\n"
     " -h          print usage\n"
     " -l          plot logarithmically\n"
     " -t num      number of time samples to display in each plot [default 512]\n"
     " -v          be verbose\n"
     " -z          Zap DC bin\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  mopsr_util_t opts;

  opts.lock_flag = 1;
  opts.ndim = 2;
  opts.plot_plain = 0;
  opts.zap = 0;

  unsigned tscrunch = 781250;

  while ((arg=getopt(argc,argv,"D:hln:t:vz")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'h':
        usage();
        return 0;

      case 'l':
        opts.plot_log = 1;
        break; 

      case 't':
        tscrunch = atoi (optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'z':
        opts.zap = 1;
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
    fprintf (stderr, "filesize for %s is %ld bytes\n", filename, filesize);

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

  if (ascii_header_get(header, "NANT", "%d", &(opts.nant)) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }

  float tsamp;
  if (ascii_header_get(header, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }
  tsamp /= 1000000;

  float digitization_offset;
  if (ascii_header_get(header, "DIGITZATION_OFFSET", "%f", &digitization_offset) != 1)
  {
    digitization_offset = 0.0;
  }

  const unsigned ndim = 2;
  size_t nsamp_total = data_size / (ndim * opts.nchan * opts.nant);
  size_t nsamp_plot  = nsamp_total / tscrunch;
  size_t nsamp_read = tscrunch;

  if (header)
    free(header);
  header = 0;

  if (verbose)
    fprintf(stderr, "mopsr_dadapowermonitorplot: using device %s\n", device);

  // number
  const size_t bytes_to_read = nsamp_read * opts.nchan * opts.nant * opts.ndim;

  float * power = (float *) malloc (sizeof(float) * nsamp_plot);
  if (verbose)
    fprintf(stderr, "mopsr_dadapowermonitorplot: nsamp_plot=%ld\n", nsamp_plot);

  size_t bytes_read_total = 0;
  void * raw = malloc (bytes_to_read);
  unsigned isamp, isamp_plot, ichan, iant;

  size_t spectrum_stride = opts.nchan;
  size_t raw_stride = opts.nchan * opts.nant * opts.ndim;

  fprintf (stderr, "bytes_to_read=%ld\n", bytes_to_read);

  float re, im;
  int8_t * ptr = (int8_t *) raw;

  float xvals[nsamp_plot];
  isamp_plot = 0;

  while (bytes_read + bytes_to_read < data_size)
  {
    bytes_read += read (fd, raw, bytes_to_read);
    
    power[isamp_plot] = 0;
    xvals[isamp_plot] = isamp_plot * tscrunch * tsamp;

    ptr = (int8_t *) raw;

    for (isamp=0; isamp<nsamp_read; isamp++)
    {
      for (ichan=0; ichan<opts.nchan; ichan++)
      {
        for (iant=0; iant<opts.nant; iant++)
        {
          if (iant == opts.ant && ichan == 10)
          {
            re = (float) ptr[0] - digitization_offset;
            im = (float) ptr[1] - digitization_offset;
            power[isamp_plot] += (re * re) + (im * im);
          }
          ptr += 2;
        }
      }
      isamp_plot++;
    }

  }

  float xmin = 0;
  float xmax = nsamp_plot * tsamp * tscrunch;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  unsigned i;
  for (i=0; i<nsamp_plot ; i++)
  {
    if (opts.plot_log)
      if (power[i])
        power[i] = log10(power[i]);
      else
        power[i] = 0;

    if (power[i] > ymax)
      ymax = power[i];
    if (power[i] < ymin)
      ymin = power[i];
  }

  if (cpgbeg(0, device, 1, 1) != 1)
  {
    fprintf(stderr, "error opening plot device\n");
    exit(1);
  }

  cpgbbuf();

  cpgswin (xmin, xmax, ymin, ymax);

  cpgsvp(0.1, 0.9, 0.1, 0.9);
  if (opts.plot_log)
    cpgbox("BCNST", 0.0, 0.0, "BCLNST", 0.0, 0.0);
  else
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Time (seconds)", "Total Power", "");

  cpgsci(3);
  cpgslw(5);
  cpgline(nsamp_plot, xvals, power);
  cpgslw(1);

  cpgebuf();

  cpgclos();

  close (fd);

  if (power)
    free (power);
  power = 0;

  if (raw)
    free (raw);
  raw = 0;

  return EXIT_SUCCESS;
}

