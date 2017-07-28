/***************************************************************************
 *  
 *    Copyright (C) 2010 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_generator.h"
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
	   "dada_dbnum [options]\n"
     " -b num   number to inc index between xfers\n"
     " -i num   number of bytes to interleave [default 8192]\n"
     " -s num   number to start on [default 0]\n"
     " -k       hexadecimal shared memory key  [default: %x]\n"
     " -h       print help\n",
     DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  /* first number to read */
  uint64_t start_num;

  /* number of uint64_ts to skip */
  uint64_t interleave_skip_num;

  /* counter */
  uint64_t index;

  /* number to skip between xfers */
  uint64_t xfer_skip_num;

  /* flag for reading the header */
  unsigned header_read;

  unsigned verbose;

} dada_dbnum_t;

#define DADA_DBNUM_INIT { 0, 0, 0, 0, 0, 0 }

/*! Pointer to the function that transfers data to/from the target */
int64_t dada_dbnum_io (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "dada_dbnum_io %p %"PRIu64"\n", data, data_size);
#endif

  /* the dada_dbnum specific data */
  dada_dbnum_t* dbnum = (dada_dbnum_t*) client->context;

  if (!dbnum->header_read) {
    dbnum->header_read = 1;
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "dada_dbnum_io: read header\n");
#endif
    return data_size;
  }

  // data_size is the number of bytes to write, uint64_size is the number of uint64s to write
  size_t uint64_size = 8;
  uint64_t uint64_count = data_size / uint64_size;

  uint64_t seq;
  uint64_t tmp;

  unsigned j = 0;
  uint64_t i = 0;
  unsigned char * ptr = 0;

  for (i=0; i<uint64_count; i++)
  {
    // decode the uint64_t 
    ptr = (unsigned char *) data + i*uint64_size;
    seq = UINT64_C (0);
    for (j = 0; j < 8; j++ )
    {
      tmp = UINT64_C (0);
      tmp = ptr[8 - j - 1];
      seq |= (tmp << ((j & 7) << 3));
    }

    //fprintf(stderr, "seq=%"PRIu64", index=%"PRIu64": ",  seq, dbnum->index);
    if (seq != dbnum->index) 
    {
      fprintf(stderr, "seq [%"PRIu64"] != index [%"PRIu64"]\n", seq, dbnum->index);
      exit(0);
    }

    dbnum->index++;
    if (dbnum->interleave_skip_num)
    {
      if (dbnum->index % dbnum->interleave_skip_num == 0)
      {
        if (dbnum->verbose)
          fprintf(stderr, "increment %"PRIu64" -> ", dbnum->index);
        dbnum->index += dbnum->interleave_skip_num;
        if (dbnum->verbose)
          fprintf(stderr, "%"PRIu64"\n", dbnum->index);
      }
    }
  }

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "dada_dbnum_io: copied %"PRIu64" bytes\n", data_size);
#endif

  return (int64_t) data_size;
}

/*! Function that closes the data file */
int dada_dbnum_close (dada_client_t* client, uint64_t bytes_written)
{
  /* the dada_dbnum specific data */
  dada_dbnum_t* dbnum = 0;

  assert (client != 0);
  dbnum = (dada_dbnum_t*) client->context;
  assert (dbnum != 0);

  if (dbnum->xfer_skip_num) 
  {
    fprintf(stderr, "incrementing index from %"PRIu64" to ", dbnum->index);
    dbnum->index += dbnum->xfer_skip_num;
    fprintf(stderr, "%"PRIu64" [ by %"PRIu64"]\n", dbnum->index, dbnum->xfer_skip_num);
  }

  dbnum->header_read = 0;

  return 0;
}

/*! Function that opens the data transfer target */
int dada_dbnum_open (dada_client_t* client)
{

  /* the dada_dbnum specific data */
  dada_dbnum_t* dbnum = (dada_dbnum_t*) client->context;

  if ((dbnum->index == 0) || (dbnum->xfer_skip_num == 0))
    dbnum->index = dbnum->start_num;

  fprintf(stderr, "open: index=%"PRIu64"\n", dbnum->index);

  if (dbnum->interleave_skip_num)
    client->optimal_bytes = dbnum->interleave_skip_num * 8 * 8;

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_dbnum_t dbnum = DADA_DBNUM_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* start number */
  uint64_t start_num = 0;

  /* interleave bytes */
  uint64_t interleave_skip_bytes = 8192;

  /* number to skip index by betweeen xfers */
  uint64_t xfer_skip_num = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hi:k:s:v")) != -1)
    switch (arg) {

    case 'b':
      if (sscanf (optarg, "%"PRIu64, &xfer_skip_num) != 1) {
        fprintf (stderr,"ERROR: could not parse xfer skip num bytes from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'i':
      if (sscanf (optarg, "%"PRIu64, &interleave_skip_bytes) != 1) {
        fprintf (stderr, "ERROR: could not parse interleave skip bytes from %s\n", optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 's':
      if (sscanf (optarg, "%"PRIu64, &start_num) != 1) {
        fprintf (stderr, "ERROR: could not parse start_num from %s\n", optarg);
        usage(); 
        return EXIT_FAILURE;
      }
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return EXIT_FAILURE;
    }

  if ((argc - optind) != 0) {
    fprintf (stderr, "no command line arguments expected\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  log = multilog_open ("dada_dbnum", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  dbnum.interleave_skip_num = interleave_skip_bytes / 8;
  dbnum.start_num = start_num;
  dbnum.xfer_skip_num = xfer_skip_num;
  dbnum.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = dada_dbnum_open;
  client->io_function = dada_dbnum_io;
  client->close_function = dada_dbnum_close;

  client->direction = dada_client_reader;

  client->context = &dbnum;

  while (!client->quit) 
  {

    if (dada_client_read (client) < 0) {
      multilog (log, LOG_ERR, "Error during transfer\n");
      return -1;
    }

  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

