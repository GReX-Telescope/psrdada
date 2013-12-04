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

#include <sys/types.h>
#include <sys/stat.h>

/* #define _DEBUG 1 */

void usage()
{
  fprintf (stdout,
	   "caspsr_disknumcheck [options] file\n"
     " file     filename to read/process\n"
     " -g num   global offset, seq no in first byte of first packet\n"
     " -h       print help\n",
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

  int arg = 0;

  while ((arg=getopt(argc,argv,"g:hv")) != -1)
    switch (arg) {

    case 'g':
      if (sscanf (optarg, "%"PRIu64, &global_offset) != 1) {
        fprintf (stderr,"ERROR: could not parse global offset from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
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

  uint64_t file_size = 0;
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {
    fprintf(stderr, "Error: header with no FILE_SIZE\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  uint64_t seq_size = sizeof(uint64_t);
  uint64_t seq_count = file_size / seq_size;
  uint64_t index = (obs_offset / seq_size) + global_offset;
  uint64_t seq = 0;
  uint64_t tmp = 0;
  uint64_t i = 0;
  uint64_t j = 0;
  uint64_t k = 0;
  unsigned char * ptr = 0;
  unsigned char ch;
  uint64_t local_errors = 0;

  // read in chunks of 128 MB
  size_t data_size = 128 * 1024 * 1024;
  char * data = (char *) malloc(data_size * sizeof(char));

  fprintf(stderr, "OBS_OFFSET=%"PRIu64"\n", obs_offset);
  fprintf(stderr, "FILE_SIZE=%"PRIu64"\n", file_size);
  fprintf(stderr, "seq_count=%"PRIu64"\n", seq_count);
  fprintf(stderr, "seq_size=%"PRIu64"\n", seq_size);

  read (fd, data, data_size);


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

    if (i == 0) 
      fprintf(stderr, "START SEQ=%"PRIu64"\n", seq);
    if (i == seq_count -1) 
      fprintf(stderr, "END SEQ=%"PRIu64"\n", seq);

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

    // reset and read more data
    if (k * seq_size >= data_size)
    {
      k = 0;
      read (fd, data, data_size);
    }
  }

  free(data);
  close(fd);

  return EXIT_SUCCESS;
}

