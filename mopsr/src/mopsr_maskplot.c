/*
 * read a mask file from disk and play through waterfall plot
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
     "mopsr_maskplot [options] maskfile\n"
     " maskfile    1-bit mask file\n"
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
  opts.ndim = 1;
  opts.nant = 1;
  opts.plot_plain = 0;
  opts.zap = 0;

  while ((arg=getopt(argc,argv,"D:hn:t:v")) != -1)
  {
    switch (arg)
    {
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
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line argument is required\n\n");
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
  float tsamp;
  if (ascii_header_get(header, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "could not extract TSAMP from header\n");
    return EXIT_FAILURE;
  }

  if (header)
    free(header);
  header = 0;

  if (verbose)
    fprintf(stderr, "mopsr_maskplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_maskplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  fprintf (stderr, "nchan=%d nsamp=%d nant=%d\n", opts.nchan, nsamp, opts.nant);

  unsigned nsamp_plot = nsamp;
  float * spectra = (float *) malloc (sizeof(float) * opts.nchan * nsamp_plot);
  float * waterfall = (float *) malloc (sizeof(float) * opts.nchan * nsamp_plot);

  size_t bytes_read_total = 0;
  const size_t bytes_to_read = (nsamp * opts.nchan * opts.nant * opts.ndim) / 8;
  void * raw = malloc (bytes_to_read);
  unsigned isamp, isamp_plot;

  size_t spectrum_stride = opts.nchan;
  size_t raw_stride = opts.nchan * opts.nant * opts.ndim;

  fprintf (stderr, "bytes_to_read=%ld, data_size=%ld\n", bytes_to_read, data_size);
  unsigned bin_factor = nsamp / nsamp_plot;

  memset (spectra, 0, sizeof(float) * opts.nchan * nsamp_plot);
  bytes_read = 0;
  while (bytes_read_total + bytes_to_read < data_size)
  {
    bytes_read = read (fd, raw, bytes_to_read);
    char * in = (char *) raw;
    unsigned nval = opts.nchan * nsamp_plot;
    unsigned ival;
    const unsigned short mask = 0x1;

    if (bytes_read == bytes_to_read)
    {
      bytes_read_total += bytes_read;
      if (verbose)
        fprintf (stderr, "read %ld bytes, nval=%u\n", bytes_read, nval);

      // new to unpack 1-bit TF data into spectra (TF order)
      for (ival=0; ival<nval; ival+=8)
      {
        char val = *in;
        spectra[ival+0] = (float) ((val >> 0) & mask);
        spectra[ival+1] = (float) ((val >> 1) & mask);
        spectra[ival+2] = (float) ((val >> 2) & mask);
        spectra[ival+3] = (float) ((val >> 3) & mask);
        spectra[ival+4] = (float) ((val >> 4) & mask);
        spectra[ival+5] = (float) ((val >> 5) & mask);
        spectra[ival+6] = (float) ((val >> 6) & mask);
        spectra[ival+7] = (float) ((val >> 7) & mask);
        in++;
      }

      uint64_t nzapped = 0;
      unsigned ichan, isamp;
      for (isamp=0; isamp<nsamp; isamp++)
        for (ichan=0; ichan<opts.nchan; ichan++)
          nzapped += (unsigned) spectra[isamp*opts.nchan+ichan];

      fprintf (stderr, "samples zapped=%lu, percentage zapped=%5.2lf%%\n", nzapped, (float)(nzapped * 100) / (float) (nsamp * opts.nchan));

      mopsr_transpose (waterfall, spectra, nsamp_plot, &opts);
      mopsr_plot_waterfall (waterfall, nsamp_plot, &opts);
    }
  }

  cpgclos();
  close (fd);

  if (spectra)
    free (spectra);
  spectra = 0;

  if (waterfall)
    free (waterfall);
  waterfall = 0;

  if (raw)
    free (raw);
  raw = 0;

  return EXIT_SUCCESS;
}

