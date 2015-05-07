#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <emmintrin.h>
#include <stdint.h>

int quit_threads = 0;

void usage()
{
  fprintf (stdout,
           "mopsr_dadatfssplit [options] file\n"
           " -v        verbose mode\n"
           " file      .dada file to split\n" );
}

int main (int argc, char **argv)
{
  // DADA Logger
  multilog_t* log = 0;

  // Flag set in verbose mode
  char verbose = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hv")) != -1)
  {
    switch (arg) 
    {
      case 'h':
        usage();
        return EXIT_SUCCESS;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 0;
      
    }
  }

  if (argc - optind != 1)
  {
    fprintf (stderr, "ERROR: 1 command line argument expected\n");
    return EXIT_FAILURE;
  }

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (argv[optind], flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file [%s]: %s\n", argv[optind], strerror(errno));
    exit(EXIT_FAILURE);
  }

  char * header = (char *) malloc (sizeof(char) * 4096);
  if (!header)
  {
    fprintf(stderr, "failed to allocate memory for header\n");
    exit(EXIT_FAILURE);
  }

  size_t bytes_read = read (fd, (void *) header, 4096);

  // get the transfer size (if it is set)
  int64_t transfer_size;
  if (ascii_header_get (header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size) != 1)
  {
    transfer_size = 0;
  }

  int64_t file_size;
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIi64, &file_size) != 1)
  {
    file_size = 0;
  }

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    fprintf (stderr, "open: header with no OBS_OFFSET\n");
    return -1;
  }

  uint64_t resolution;
  if (ascii_header_get (header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_WARNING, "open: header with no RESOLUTION\n");
    resolution = 0;
  }

  // get the number of antenna
  unsigned nant;
  if (ascii_header_get (header, "NANT", "%u", &nant) != 1)
  {
    multilog (log, LOG_WARNING, "open: header with no NANT\n");
    nant = 1;
  }

  unsigned nchan;
  if (ascii_header_get (header, "NCHAN", "%u", &nchan) != 1)
  {           
    fprintf (stderr, "open: header with no NCHAN\n");
    return -1;
  }

  unsigned nbit;
  if (ascii_header_get (header, "NBIT", "%u", &nbit) != 1)
  {
    fprintf (stderr, "open: header with no NBIT\n");
    return -1;
  }

  unsigned ndim;
  if (ascii_header_get (header, "NDIM", "%u", &ndim) != 1)
  {           
    fprintf (stderr, "open: header with no NDIM\n");
    return -1;                
  }                             

  char order[6];
  if (ascii_header_get (header, "ORDER", "%s", order) != 1)
  {
    fprintf (stderr, "open: header with no ORDER\n");
    return -1;
  }

  char tmp[32];
  if (ascii_header_get (header, "UTC_START", "%s", tmp) == 1)
  {
    fprintf (stderr, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    fprintf (stderr, "open: UTC_START=UNKNOWN\n");
  }

  char pfb[5];
  if (ascii_header_get (header, "PFB_ID", "%s", pfb) != 1)
  {
    fprintf (stderr ,"open: could not read PFB_ID from header\n");
    return -1;
  }

  uint64_t bytes_per_second;
  if (ascii_header_get (header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    fprintf (stderr, "open: header with no BYTES_PER_SECOND\n");
    return -1;
  }

  int ant_ids[nant];
  unsigned i;
  for (i=0; i<nant; i++)
  {
    sprintf (tmp, "ANT_ID_%d", i);
    if (ascii_header_get (header, tmp, "%d", &(ant_ids[i])) != 1)
    {
      fprintf (stderr, "open: header with no %s\n", tmp);
      return -1;
    }
    if (ascii_header_del(header, tmp) < 0)
    {
      fprintf (stderr, "open: could not delete %s from header\n", tmp);
      return -1;
    }
  }

  uint64_t new_obs_offset = obs_offset / nant;
  uint64_t new_bytes_per_second = bytes_per_second / nant;
  uint64_t new_file_size = file_size / nant;
  uint64_t new_resolution = nchan * (nbit/8);

  fprintf (stderr, "BYTES_PER_SECOND: %"PRIu64" -> %"PRIu64"\n", bytes_per_second, new_bytes_per_second);
  fprintf (stderr, "RESOLUTION: %"PRIu64" -> %"PRIu64"\n", resolution, new_resolution);

  int fds[nant];
  flags = O_WRONLY | O_CREAT | O_TRUNC;

  char dirname[16];
  char command[64];
  char filename[1024];

  if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, new_obs_offset) < 0)
  {
    fprintf (stderr, "open: failed to write new OBS_OFFSET to header\n");
    return -1;
  }

  if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
  {
    fprintf (stderr, "open: failed to write new BYTES_PER_SECOND to header\n");
    return -1;
  }

  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, new_resolution) < 0)
  {
    fprintf (stderr, "open: failed to write new RESOLUTION to header\n");
    return -1;
  }

  if (file_size)
  {
    if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
    {
      fprintf (stderr, "open: failed to write new FILE_SIZE to header\n");
      return -1;
    }
  }

  // now set each output data block to 1 antenna
  int new_nant = 1;
  if (ascii_header_set (header, "NANT", "%d", new_nant) < 0)
  {
    fprintf (stderr, "open: failed to write NANT=%d to header\n",
             new_nant);
    return -1;
  }

  sprintf (order, "%s", "TF");
  if (ascii_header_set (header, "ORDER", "%s", order) < 0)
  {
    fprintf (stderr, "open: failed to write ORDER=%s to header\n", order);
    return -1;
  }


  size_t bytes_written;
  // setup output files for each ant
  for (i=0; i<nant; i++)
  {
    sprintf (dirname, "%s_%d", pfb, ant_ids[i]);
    sprintf (command, "mkdir -p %s", dirname);
    sprintf (filename, "%s/%s", dirname, argv[optind]);

    system (command);

    fds[i] = open (filename, flags, perms);
    if (fds[i] < 0)
    {
      fprintf(stderr, "failed to open dada file [%s]: %s\n", filename, strerror(errno));
      exit(EXIT_FAILURE);
    }

    if (ascii_header_set (header, "ANT_ID", "%d", ant_ids[i]) < 0)
    {
      fprintf (stderr, "failed to write ANT_ID=%d to header for ant %d\n", ant_ids[i], i);
      exit(EXIT_FAILURE);
    }

    bytes_written = write (fds[i], (void *) header, 4096);
    if (bytes_written != 4096)
    {
      fprintf (stderr, "failed to write header to %s: %s\n", filename, strerror(errno));
      return -1;
    }
  }

  fprintf (stderr, "resoution=%"PRIu64"\n", resolution);

  const size_t in_block_size = resolution * 1024;
  const size_t out_block_size = in_block_size / nant;

  const uint64_t nsamp = out_block_size / (ndim * (nbit/8) * nchan);

  if (verbose > 1)
    fprintf (stderr, "in_data_size=%"PRIu64", out_data_size=%"PRIu64" nsamp=%"PRIu64"\n", in_block_size, out_block_size, nsamp);

  int16_t * in  = (int16_t *) malloc (in_block_size);
  int16_t * out = (int16_t *) malloc (out_block_size);

  bytes_read = read (fd, (void *) in, in_block_size);

  unsigned ichan, iant;
  uint64_t isamp;

  fprintf (stderr, "read %ld bytes\n", bytes_read);
  uint64_t iblock = 0;

  while (bytes_read == in_block_size)
  {
    if (verbose > 1)
      fprintf (stderr, "[%"PRIu64"] nant=%u nchan=%u nsamp=%u\n",iblock, nant, nchan, nsamp);

    for (iant=0; iant<nant; iant++)
    {
      int16_t * in_ptr  = in + iant;
      int16_t * out_ptr = out;

      for (isamp=0; isamp<nsamp; isamp++)
      {
        for (ichan=0; ichan<nchan; ichan++)
        {
          *out_ptr = *in_ptr;

          out_ptr ++;
          in_ptr += nant;
        }
      }

      if (verbose > 1)
        fprintf (stderr, "[%"PRIu64"] writing %ld bytes to fds[%d]\n", iblock, out_block_size, iant);
      bytes_written = write (fds[iant], (void *) out, out_block_size);

      if (bytes_written != out_block_size)
      {
        fprintf (stderr, "failed to write %"PRIu64" bytes to output ant %d\n", out_block_size, iant);
        return -1;
      }
      if (verbose > 1)
        fprintf (stderr, "[%"PRIu64"] incrementing in_ptr %ld bytes\n", iblock, out_block_size);
    }

    bytes_read = read (fd, (void *) in, in_block_size);
    iblock++;
  }

  close (fd);
  for (iant=0; iant<nant; iant++)
    close (fds[iant]);

  free (header);
  free (in);
  free (out);

  return EXIT_SUCCESS;
}
