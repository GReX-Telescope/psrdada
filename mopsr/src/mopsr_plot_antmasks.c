/*
 * read all mask files from disk and plot antenna from W to E
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
void plot_ant_time (float * spectra, unsigned int nsamps, float tsamp, mopsr_util_t * opts);

void usage()
{
  fprintf (stdout,
     "mopsr_plot_antmasks [options] mask_files\n"
     " mask_files    1-bit mask file produced by mopsr_aqdsp\n"
     " -p            enable plotting\n"
     " -D device     pgplot device name\n"
     " -h            print usage\n"
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

  unsigned int nsamp = 512;

  mopsr_util_t opts;

  opts.lock_flag = 1;
  opts.lock_flag_long = 1;
  opts.ndim = 1;
  opts.nant = 1;
  opts.plot_plain = 0;
  opts.zap = 0;

  char plot = 0;

  while ((arg=getopt(argc,argv,"D:hn:pt:v")) != -1)
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
  int nant = nfiles;
  opts.nant = nfiles;
  int i;
  struct stat buf;
  size_t filesize = 0;

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int * fds = (int *) malloc(sizeof(int) * nfiles);
  char ** filenames = (char **) malloc(sizeof(char *) * nfiles);
  size_t * filesizes = (size_t *) malloc (sizeof(size_t) * nfiles);
  uint64_t * zapped = (uint64_t *) malloc(sizeof(uint64_t) * nfiles);

  size_t min_filesize = 1e12;

  for (i=0; i<nfiles; i++)
  {
    filenames[i] = (char *) malloc(strlen(argv[optind+i])+1);
    strcpy(filenames[i], argv[optind+i]);
    if (stat (filenames[i], &buf) < 0)
    {
      fprintf (stderr, "ERROR: failed to stat mask file [%s]: %s\n", filenames[i], strerror(errno));
      return (EXIT_FAILURE);
    }

    filesizes[i] = buf.st_size;
    if (verbose)
      fprintf (stderr, "filesize for %s is %d bytes\n", filenames[i], filesizes[i]);

    if (filesizes[i] < min_filesize)
      min_filesize = filesizes[i];

    fds[i] = open (filenames[i], flags, perms);
    if (fds[i] < 0)
    {
      fprintf(stderr, "failed to open dada file[%s]: %s\n", filenames[i], strerror(errno));
      exit(EXIT_FAILURE);
    }

    zapped[i] = 0;
  }

  size_t data_size = min_filesize - 4096;

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

  int nbit;
  if (ascii_header_get(header, "NBIT", "%d", &nbit) != 1)
  {
    fprintf (stderr, "could not extract NBIT from header\n");
    return EXIT_FAILURE;
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
    fprintf(stderr, "mopsr_plot_antmasks: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_plot_antmasks: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  size_t bits_per_sample = opts.nchan * opts.ndim * nbit;
  nsamp = (data_size * 8) / bits_per_sample;
  const size_t bytes_to_read = (nsamp * bits_per_sample) / 8;
  size_t nsamp_plot = 512;

  if (verbose)
    fprintf (stderr, "nchan=%d nsamp=%d nant=%d ndim=%d nbit=%d\n", opts.nchan, nsamp, opts.nant, opts.ndim, nbit);

  size_t waterfall_size = sizeof(float) * nfiles * nsamp_plot;
  if (verbose)
    fprintf (stderr, "allocating %d bytes for waterfall\n", waterfall_size);
  float * waterfall = (float *) malloc (waterfall_size);

  size_t bytes_read_total = 0;
  void * raw = malloc (bytes_to_read);
  unsigned isamp, isamp_plot;

  if (verbose > 1)
    fprintf (stderr, "bytes_to_read=%ld, data_size=%ld\n", bytes_to_read, data_size);

  unsigned bin_factor = nsamp / nsamp_plot;
  tsamp *= bin_factor;

  bytes_read = 0;

  unsigned iant, ibit, ichan, osamp;
  const unsigned short mask = 0x1;
  const unsigned nchan = opts.nchan;

  for (iant=0; iant<nfiles; iant++)
  {
    bytes_read = read (fds[iant], raw, bytes_to_read);
    if (verbose > 1)
      fprintf (stderr, "read %ld bytes from ant %d\n", bytes_read, iant);
    char * in = (char *) raw;

    if (bytes_read == bytes_to_read)
    {
      for (osamp=0; osamp<nsamp_plot; osamp++)
      {
        float tscr = 0;
        for (isamp=0; isamp<bin_factor; isamp++)
        {
          float fscr = 0;
          for (ichan=0; ichan<nchan; ichan+=8)
          {
            const char val = *in;
            for (ibit=0; ibit<8; ibit++)
            {
              fscr += (float) ((val >> ibit) & mask);
            }
            in++;
          }
          tscr += fscr;
        }
        unsigned odx = osamp*nant + iant;
        waterfall[osamp*nant + iant] = tscr;
      }
    }
  }

  plot_ant_time (waterfall, nsamp_plot, tsamp, &opts);

  cpgclos();
  for (i=0; i<nfiles; i++)
    close (fds[i]);

  free (raw);

  if (waterfall)
    free (waterfall);
  waterfall = 0;

  return EXIT_SUCCESS;
}


/*
 *  plot waterfall plot assuming
 */
void plot_ant_time (float * spectra, unsigned int nsamps, float tsamp, mopsr_util_t * opts)
{
  cpgbbuf();
  cpgsci(1);

  char label[64];

  if (opts->plot_plain)
  { 
    cpgsvp(0, 1, 0, 1);
    cpgswin(0, (float) opts->nant, 0, (float) nsamps);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  }
  else
  { 
    cpgenv(0, (float) opts->nant, 0, (float) nsamps * tsamp / 1e6, 0, 0);
    cpglab("Antenna (W to E)", "Time (s)", label);
  }

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float x_min = 0;
  float x_max = opts->nant;

  float y_min = 0;
  float y_max = nsamps;

  float x_res = (x_max-x_min)/opts->nant;
  float y_res = (y_max-y_min)/nsamps;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  int ndat = nsamps * opts->nant;
  float z_min = 10000000;
  float z_max = 0;
  float z_avg = 0;

  if (opts->plot_log && z_min > 0)
    z_min = log10(z_min);
  if (opts->plot_log && z_max > 0)
    z_max = log10(z_max);

  unsigned int ichan, isamp, iant;
  unsigned int i;
  unsigned int ndat_avg = 0;
  float val;

  for (iant=0; iant<opts->nant; iant++)
  {
    for (isamp=0; isamp<nsamps; isamp++)
    {
      i = iant * nsamps + isamp;
      if (opts->plot_log && spectra[i] > 0)
        spectra[i] = log10(spectra[i]);
      if (spectra[i] > z_max) z_max = spectra[i];
      if (spectra[i] < z_min) z_min = spectra[i];
      z_avg += spectra[i];
      ndat_avg++;
    }
  }

  z_avg /= (float) ndat_avg;

  float yrange = nsamps; 

  if (z_min == z_max)
  {
    cpgsch(6);
    cpgslw(5);
    cpgptxt(opts->nant/2,yrange * 0.25, 0.0, 0.5, "All Data Zero");
    cpgsch(1);
  }
  else
    cpgimag(spectra, opts->nant, nsamps, 1, opts->nant, 1, nsamps, z_min, z_max, trf);

  if (!opts->plot_plain)
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);

  cpgebuf();
}



