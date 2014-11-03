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
     "mopsr_dadafftplot [options] dadafile\n"
     " -a ant      antenna to print\n"
     " -c ant      channel to print\n"
     " -t nsamp    nsamp to print\n" 
     " -v          be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  int arg = 0;

  int antenna = 0;
  int channel = 0;
  int samp_to_view = 64;

  while ((arg=getopt(argc,argv,"a:c:t:v")) != -1)
  {
    switch (arg)
    {
      case 'a':
        antenna = atoi(optarg);
        break;

      case 'c':
        channel = atoi(optarg);
        break;

      case 't':
        samp_to_view = atoi(optarg);
        break;

      case 'v':
        verbose++;
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

  size_t data_size = filesize - 4096;

  char * header = (char *) malloc (4096);

  if (verbose)
    fprintf (stderr, "reading header, 4096 bytes\n");
  size_t bytes_read = read (fd, header, 4096);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  void * raw = (void *) malloc (data_size);
  if (verbose)
    fprintf (stderr, "reading data %lu bytes\n", data_size);
  bytes_read = read (fd, raw, data_size);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  close(fd);

  int nchan, nant, ndim;;

  if (ascii_header_get(header, "NCHAN", "%d", &nchan) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "NANT", "%d", &nant) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "NDIM", "%d", &ndim) != 1)
  {
    fprintf (stderr, "could not extract NDIM from header\n");
    return EXIT_FAILURE;
  }

  uint64_t nsamp = data_size / (ndim * nant * nchan);

  if (header)
    free(header);
  header = 0;

  fprintf (stderr, "[T][F][S] (re, im)\n");
  int8_t * ptr = (int8_t *) raw;
  unsigned iant, isamp, ichan;
  int re, im, power;
  for (isamp=0; isamp<samp_to_view; isamp++)
  {
    for (ichan=0; ichan<nchan; ichan++)
    {
      for (iant=0; iant<nant; iant++)
      {
        if ((ichan == channel) && (iant == antenna))
        {
          re = (int) ptr[0];
          im = (int) ptr[1];
          power = (re * re) + (im * im);

          fprintf (stderr, "[%d][%d][%d] (%d, %d) power=%d\n", isamp, ichan, iant, re, im, power);
        }
        ptr += 2;     
      }
    }
  }

  if (raw)
    free (raw);
  raw = 0;

  return EXIT_SUCCESS;
}

