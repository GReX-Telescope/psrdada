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
     "mopsr_maskplot [options] mask_files\n"
     " mask_files    1-bit mask file produced by mopsr_aqdsp\n"
     " -p            enable plotting\n"
     " -D device     pgplot device name\n"
     " -h            print usage\n"
     " -s            skip this many time samples [default 0]\n"
     " -t num        number of time samples to display in each plot [default 512]\n"
     " -v            be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int nsamp_skip = 0;
  unsigned int nsamp = 512;

  mopsr_util_t opts;

  opts.lock_flag = 1;
  opts.lock_flag_long = 1;
  opts.ndim = 1;
  opts.nant = 1;
  opts.plot_plain = 0;
  opts.zap = 0;

  char plot = 0;

  while ((arg=getopt(argc,argv,"D:hn:ps:t:v")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'h':
        usage();
        return 0;

      case 'p':
        plot = 1;
        break;

      case 's':
        nsamp_skip = atoi (optarg);
        break;

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
  if (argc-optind < 1)
  {
    fprintf(stderr, "ERROR: At least 1 command line argument is required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  int nfiles = argc - optind;
  int i;
  struct stat buf;
  size_t filesize = 0;

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int * fds = (int *) malloc(sizeof(int) * nfiles);
  char ** filenames = (char **) malloc(sizeof(char *) * nfiles);
  uint64_t * zapped = (uint64_t *) malloc(sizeof(uint64_t) * nfiles);

  for (i=0; i<nfiles; i++)
  {
    filenames[i] = (char *) malloc(strlen(argv[optind+i])+1);
    strcpy(filenames[i], argv[optind+i]);
    if (stat (filenames[i], &buf) < 0)
    {
      fprintf (stderr, "ERROR: failed to stat mask file [%s]: %s\n", filenames[i], strerror(errno));
      return (EXIT_FAILURE);
    }

    filesize = buf.st_size;
    if (verbose)
      fprintf (stderr, "filesize for %s is %d bytes\n", filenames[i], filesize);

    fds[i] = open (filenames[i], flags, perms);
    if (fds[i] < 0)
    {
      fprintf(stderr, "failed to open dada file[%s]: %s\n", filenames[i], strerror(errno));
      exit(EXIT_FAILURE);
    }

    zapped[i] = 0;
  }

  size_t data_size = filesize - 4096;

  // just read the header from the first file
  char * header = (char *) malloc (4096);
  if (verbose)
    fprintf (stderr, "reading header, 4096 bytes\n");
  size_t bytes_read = read (fds[0], header, 4096);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);
  for (i=1; i<nfiles; i++)
  {
    lseek (fds[i], 4096, SEEK_SET);
  } 

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

  if (plot)
  {
    if (cpgopen(device) != 1) {
      fprintf(stderr, "mopsr_maskplot: error opening plot device\n");
      exit(1);
    }
    cpgask(1);
  }

  if (verbose)
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

  if (verbose > 1)
    fprintf (stderr, "bytes_to_read=%ld, data_size=%ld\n", bytes_to_read, data_size);
  unsigned bin_factor = nsamp / nsamp_plot;

  bytes_read = 0;

  unsigned nval = opts.nchan * nsamp_plot;
  unsigned ival, ibit;
  const unsigned short mask = 0x1;

  uint64_t zapped_total = 0;
  uint64_t nsamp_total = 0;

  if (nsamp_skip > 0) {
    bytes_read_total += nsamp_skip * opts.nchan * opts.nant * opts.ndim / 8;
    if (bytes_read_total > data_size) {
      fprintf(stderr, "Requested skipping beyond the end of the file\n");
      fprintf(stderr, "nsamp_skip=%ld corresponding bytes=%ld data_size=%ld\n", nsamp_skip, bytes_read_total, data_size);
      exit(-1);
    }
    if (verbose > 1)
      fprintf(stderr, "Skipping %ld samples (%ld bytes) past the header\n", nsamp_skip, bytes_read_total);
    for (i=1; i<nfiles; i++) {
      lseek(fds[i], 4096+bytes_read_total, SEEK_SET);
    }
  }

  while (bytes_read_total + bytes_to_read < data_size)
  {
    memset (spectra, 0, sizeof(float) * opts.nchan * nsamp_plot);

    uint64_t zapped_block = 0;
    uint64_t nsamp_block = nfiles * nval;
    for (i=0; i<nfiles; i++)
    {
      bytes_read = read (fds[i], raw, bytes_to_read);
      char * in = (char *) raw;

      if (bytes_read == bytes_to_read)
      {
        if (verbose> 1)
          fprintf (stderr, "read %ld bytes, nval=%u\n", bytes_read, nval);

        // new to unpack 1-bit TF data into spectra (TF order)
        for (ival=0; ival<nval; ival+=8)
        {
          char val = *in;
          for (ibit=0; ibit<8; ibit++)
          {
            int b = ((val >> ibit) & mask);
            if (plot)
              spectra[ival+ibit] += (float) b;
            zapped[i] += b;
            zapped_block += b;
          }
          in++;
        }
      }
    }

    bytes_read_total += bytes_read;

    zapped_total += zapped_block;
    nsamp_total += nsamp_block;

    if (verbose)
      fprintf (stderr, "Block: zapped %lu (%5.2f%%)\n", zapped_block, (float) (zapped_block * 100) / (float) nsamp_block);

    if (plot)
    {
      mopsr_transpose (waterfall, spectra, nsamp_plot, &opts);
      mopsr_plot_waterfall (waterfall, nsamp_plot, &opts);
    }
  }

  fprintf (stderr, "Total: %5.2lf%% (%lu of %lu)\n", (float)(zapped_total * 100) / (float) nsamp_total, zapped_total, nsamp_total);

  if (plot)
    cpgclos();
  for (i=0; i<nfiles; i++)
    close (fds[i]);

  free (raw);
  free (zapped);

  if (spectra)
    free (spectra);
  spectra = 0;

  if (waterfall)
    free (waterfall);
  waterfall = 0;

  return EXIT_SUCCESS;
}

