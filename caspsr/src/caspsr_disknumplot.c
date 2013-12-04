/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <cpgplot.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <float.h>
#include <math.h>

/* #define _DEBUG 1 */

void usage()
{
  fprintf (stdout,
	   "caspsr_disknumcheck [options] file\n"
     " file       filename to read/process\n"
     " -g num     global offset, seq no in first byte of first packet\n"
     " -o offset  skip to this byte offset (relative to start of observation)\n"
     " -h         print help\n",
     DADA_DEFAULT_BLOCK_KEY);
}

int main (int argc, char **argv)
{
  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* Quit flag for processing just 1 xfer */
  char quit = 0;

  /* global offset*/
  uint64_t global_offset = 1;

  /* local offset (for seeking through file) */
  uint64_t local_offset = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"g:ho:v")) != -1)
    switch (arg) {

    case 'g':
      if (sscanf (optarg, "%"PRIu64, &global_offset) != 1) {
        fprintf (stderr,"ERROR: could not parse global offset from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'o':
      if (sscanf (optarg, "%"PRIu64, &local_offset) != 1) {
        fprintf (stderr,"ERROR: could not parse global offset from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      // ensure this is mod 8 == 0
      local_offset -= (local_offset % 8);
      break;

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return EXIT_FAILURE;
    }

  if ((argc - optind) != 1) {
    fprintf (stderr, "filename expected as command line argument\n");
    usage();
    return (EXIT_FAILURE);
  } 

  int fd  = open(argv[optind], O_RDONLY);
  if (fd < 0)
  {
    fprintf(stderr, "Errror opening %s: %s\n", argv[optind], strerror(errno));
    return (EXIT_FAILURE);
  }

  // read 4KB ascii header
  char header[4096];
  size_t bytes_read = read (fd, header, 4096);
  if (bytes_read != 4096)
  {
    fprintf(stderr, "Errror reading header: %s\n", strerror(errno));
    close (fd);
    return (EXIT_FAILURE);
  }

  // get obs offset
  uint64_t obs_offset = 0;
  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    fprintf(stderr, "Error: header with no OBS_OFFSET\n");
    close(fd);
    return (EXIT_FAILURE);
  }
  fprintf(stderr, "OBS_OFFSET=%"PRIu64"\n", obs_offset);

  uint64_t file_size = 0;
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {
    fprintf(stderr, "Error: header with no FILE_SIZE\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  int64_t relative_offset = 0;
  if (local_offset)
  {
    relative_offset = local_offset - obs_offset;
    fprintf(stderr, "local_offset=%"PRIu64"\n", local_offset);
    fprintf(stderr, "relative_offset=%"PRIu64"\n", relative_offset);
    if (relative_offset > file_size)
    {
      fprintf(stderr, "Error: local offset would cause seek past end of file\n");
      close(fd);
      return (EXIT_FAILURE);
    }

    if (relative_offset < 0)
    {
      fprintf(stderr, "Error: local offset would cause seek before start of file\n");
      close(fd);
      return (EXIT_FAILURE);
    }
  }




  uint64_t seq_size = sizeof(uint64_t);
  uint64_t seq_count = (file_size - relative_offset) / seq_size;
  uint64_t index = ((obs_offset + relative_offset) / seq_size) + global_offset;
  uint64_t seq = 0;
  uint64_t tmp = 0;
  uint64_t i = 0;
  uint64_t j = 0;
  uint64_t k = 0;
  unsigned char * ptr = 0;
  unsigned char ch;
  uint64_t local_errors = 0;

  // read in chunks of 128 MB
  size_t data_size = 256 * 1024 * 1024;
  char * data = (char *) malloc(data_size * sizeof(char));

  fprintf(stderr, "OBS_OFFSET=%"PRIu64"\n", obs_offset);
  fprintf(stderr, "FILE_SIZE=%"PRIu64"\n", file_size);
  fprintf(stderr, "seq_count=%"PRIu64"\n", seq_count);
  fprintf(stderr, "seq_size=%"PRIu64"\n", seq_size);

  if (relative_offset)
  {
    lseek(fd, relative_offset, SEEK_CUR);
  }
  read (fd, data, data_size);

  uint64_t nvals = data_size / seq_size;
  float * xvals = malloc(nvals * sizeof(float));
  float * yvals = malloc(nvals * sizeof(float));

  float xmin = FLT_MAX;
  float xmax = 1;
  float ymin = FLT_MAX;
  float ymax = 1;

  char * device = "?";
  if (cpgopen(device) != 1) 
  {
    fprintf(stderr, "Error: opening plot device\n");
    i=seq_count;
  }
  cpgask(1);

  for (i=0; i<seq_count; i++)
  {
    // determine the pointer offset
    ptr = ((unsigned char *) data) + (k * seq_size);

    // decode the uint64_t at this point
    seq = UINT64_C (0);
    for (j = 0; j < 8; j++ )
    {
      tmp = UINT64_C (0);
      tmp = ptr[8 - j - 1];
      seq |= (tmp << ((j & 7) << 3));
    }

    xvals[k] = obs_offset + (float) (i * seq_size);
    yvals[k] = (float) seq;
    //if (yvals[k] > 0) 
    //  yvals[k] = logf(yvals[k]);

    if (xvals[k] < xmin) xmin = xvals[k];
    if (xvals[k] > xmax) xmax = xvals[k];
    if (yvals[k] < ymin) ymin = yvals[k];
    if (yvals[k] > ymax) ymax = yvals[k];

    // check the value
    if (seq != index)
    {
      local_errors++;
      if (local_errors < 20)
      {
        fprintf(stderr, "i=%d, seq[%"PRIu64"] != index[%"PRIu64"]\n",
                 i, seq, index);
      }
    }

    k++;
    index++;

    unsigned l = 0;
    // plot, and read more data
    if (k * seq_size >= data_size)
    {
      cpgbbuf();
      fprintf(stderr, "x %f to %f\n", xmin, xmax);
      fprintf(stderr, "y %f to %f\n", ymin, ymax);
      cpgsci(1);
      cpgenv(xmin, xmax, ymin, ymax, 0, 0);
      cpglab("Byte offset", "64 bit seq no", "");
      fprintf(stderr, "after cpgenv\n");

      cpgsci(3);
      fprintf(stderr, "before plot %d\n", nvals);
      //for (l=0; l<nvals; l++)
      //  fprintf(stderr, "x=%f, y=%f\n", xvals[l], yvals[l]);
      cpgline(nvals, xvals, yvals);
      fprintf(stderr, "after plot %d\n", nvals);

      cpgebuf();
      k = 0;
      read (fd, data, data_size);
      xmin = ymin = FLT_MAX;
      xmax = ymax = 1;
      
    }
  }

  cpgclos();

  free(data);
  close(fd);

  return EXIT_SUCCESS;
}

