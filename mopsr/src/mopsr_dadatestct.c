/*
 * read dada files from disk, testing the cornerturn
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

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

static char padded = 1;

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_dadactestct [options] ctfile1 ctfile2 chan_files\n"
     " -v              be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  int arg = 0;

  int ct_fds[2];
  int ch_fds[20];

  while ((arg=getopt(argc,argv,"hv")) != -1)
  {
    switch (arg)
    {
      case 'v':
        verbose++;
        break;

      case 'h':
      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 22)
  {
    fprintf(stderr, "ERROR: 22 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;

  ct_fds[0] = open (argv[optind], flags, perms);
  if (!ct_fds[0])
  {
    fprintf (stderr, "ERROR: could not open %s for reading\n", argv[optind]);
    exit (EXIT_FAILURE);
  }
  ct_fds[1] = open (argv[optind+1], flags, perms);
  if (!ct_fds[1])
  {
    fprintf (stderr, "ERROR: could not open %s for reading\n", argv[optind+1]);
    exit (EXIT_FAILURE);
  }

  unsigned ichan;
  for (ichan=0; ichan<20; ichan++)
  {
    ch_fds[ichan] = open (argv[optind+2+ichan], flags, perms);
    if (!ch_fds[ichan])
    {
      fprintf (stderr, "ERROR: could not open %s for reading\n", argv[optind+2+ichan]);
      exit(EXIT_FAILURE);
    }
  }

  fprintf (stderr, "allocating memory\n");

  float * ct_data0 = (float *) malloc(5406720);
  float * ct_data1 = (float *) malloc(5406720);

  float ** ch_data = (float **) malloc(sizeof(float *) * 20);
  for (ichan=0; ichan<20; ichan++)
  {
    ch_data[ichan] = (float *) malloc(540672);
  }

  fprintf (stderr, "seeking past ascii headers\n");

  // seek forward
  lseek (ct_fds[0], 4096, SEEK_SET);
  lseek (ct_fds[1], 4096, SEEK_SET);
  for (ichan=0; ichan<20; ichan++)
    lseek (ch_fds[ichan], 4096, SEEK_SET);

  fprintf (stderr, "reading data\n");
  size_t bytes_read;

  bytes_read = read (ct_fds[0], (void *) ct_data0, 5406720);
  bytes_read = read (ct_fds[1], (void *) ct_data1, 5406720);
  for (ichan=0; ichan<20; ichan++)
    bytes_read = read (ch_fds[ichan], (void *) (ch_data[ichan]), 540672);

  unsigned ibeam, isamp;
  unsigned nsamp = 3072;

  float * ct;
  float * ch;
  float ct_val, ch_val;

  unsigned nchansamp = 20 * nsamp;

  fprintf (stderr, "comparing data\n");

  uint64_t nerrs = 0;
  const unsigned nchan = 20;

  // compare the data in each channel/beam
  for (ibeam=0; ibeam<44; ibeam++)
  {

    // get the ct pointer to the right beam
    if (ibeam < 22)
      ct = ct_data0 + (ibeam * nchan * nsamp);
    else
      ct = ct_data1 + ((ibeam-22) * nchan * nsamp);

    for (ichan=0; ichan<20; ichan++)
    {
      ch = ch_data[ichan];

      for (isamp=0; isamp<nsamp; isamp++)
      {
        ct_val = ct[ichan * nsamp + isamp];
        ch_val = ch[ibeam * nsamp + isamp];

        {
          fprintf (stderr, "ibeam=%u ichan=%u isamp=%u ct=%f ch=%f\n", ibeam, ichan, isamp, ct_val, ch_val);
          nerrs++;
        }
      }
    }
  }

  free (ct_data0);
  free (ct_data1);
  for (ichan=0; ichan<20; ichan++)
    free (ch_data[ichan]);

  close (ct_fds[0]);
  close (ct_fds[1]);
  for (ichan=0; ichan<20; ichan++)
    close (ch_fds[ichan]);
}
