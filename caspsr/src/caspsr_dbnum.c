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
	   "caspsr_dbnum [options]\n"
     " -g num   global offset, seq no in first byte of first packet\n"
     " -k       hexadecimal shared memory key  [default: %x]\n"
     " -s       process only one xfer [default process many]\n"
     " -h       print help\n",
     DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  // sequence number corresponding to first byte of first packet
  uint64_t global_offset;

  // sequence counter
  uint64_t index;

  // flag for reading the header
  unsigned header_read;

  // number of seq errors this xfer
  uint64_t seq_errors;

  unsigned verbose;

  unsigned quit;

} caspsr_dbnum_t;

#define CASPSR_DBNUM_INIT { 0, 0, 0, 0, 0, 0 }

/*! Pointer to the function that transfers data from the target */
int64_t caspsr_dbnum_io (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "caspsr_dbnum_io %p %"PRIu64"\n", data, data_size);
#endif

  /* the caspsr_dbnum specific data */
  caspsr_dbnum_t* dbnum = (caspsr_dbnum_t*) client->context;

  if (!dbnum->header_read) 
  {
    if (dbnum->verbose)
      multilog (client->log, LOG_INFO, "io: read header\n");
    dbnum->header_read = 1;
    return data_size;
  }

  // size (in bytes) of each encoded seq number
  uint64_t seq_size  = (uint64_t) sizeof(uint64_t);

  // number of sequence numbers encoded in the data
  uint64_t seq_count = data_size / seq_size;

  // sanity check
  if (data_size % seq_size != 0)
  {
    multilog (client->log, LOG_WARNING, "io: data_size[%"PRIu64"] % "
              "seq_size[%"PRIu64"] != 0\n", data_size, seq_size);
    return 0;
  }

  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "io: processing %"PRIu64" bytes => %"PRIu64" seq nos\n",
              data_size, seq_count);

  uint64_t seq;
  uint64_t tmp;
  unsigned j = 0;
  uint64_t i = 0;
  unsigned char * ptr = 0;
  unsigned char ch;

  uint64_t local_errors = 0;

  //multilog (client->log, LOG_INFO, "io: address of data=%p\n", data);

  for (i=0; i<seq_count; i++)
  {
    // determine the pointer offset
    ptr = ((unsigned char *) data) + (i * seq_size);

    // decode the uint64_t at this point
    seq = UINT64_C (0);
    for (j = 0; j < 8; j++ )
    {
      tmp = UINT64_C (0);
      tmp = ptr[8 - j - 1];
      seq |= (tmp << ((j & 7) << 3));
    }

    // check the value
    if (seq != dbnum->index) 
    {
      dbnum->seq_errors++;
      local_errors++;
      if (local_errors < 5)
      {
        multilog(client->log, LOG_WARNING, "i=%d, seq[%"PRIu64"] != index[%"PRIu64"]\n", 
                 i, seq, dbnum->index);
      }
    }

    dbnum->index++;

  }

  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "io: copied %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t caspsr_dbnum_io_block (dada_client_t* client,
                  void* data, uint64_t data_size, uint64_t block_id)
{
 
  /* the caspsr_dbnum specific data */
  caspsr_dbnum_t* dbnum = (caspsr_dbnum_t*) client->context;

  if (!dbnum->header_read) 
  {
    multilog (client->log, LOG_WARNING, "io_block: read header - this shouldn't happen\n");
    dbnum->header_read = 1;
    return data_size;
  }

  // size (in bytes) of each encoded seq number
  uint64_t seq_size  = (uint64_t) sizeof(uint64_t);
  
  // number of sequence numbers encoded in the data
  uint64_t seq_count = data_size / seq_size;
  
  // sanity check
  if (data_size % seq_size != 0)
  {
    multilog (client->log, LOG_INFO, "io_block: data_size[%"PRIu64"] mod seq_size[%"PRIu64"] != 0\n", data_size, seq_size);
    return data_size;
  }
  
  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "io_block: processing %"PRIu64" bytes => %"PRIu64" seq nos\n",
              data_size, seq_count); 

  uint64_t seq;
  uint64_t tmp;
  unsigned j = 0;
  uint64_t i = 0;
  unsigned char * ptr = 0;
  unsigned char ch;
  
  uint64_t local_errors = 0;
  
  for (i=0; i<seq_count; i++)
  {
    // determine the pointer offset
    ptr = ((unsigned char *) data) + (i * seq_size);
    
    // decode the uint64_t at this point
    seq = UINT64_C (0);
    for (j = 0; j < 8; j++ )
    {
      tmp = UINT64_C (0);
      tmp = ptr[8 - j - 1];
      seq |= (tmp << ((j & 7) << 3));
    }

    // check the value
    if ((seq) && (seq != dbnum->index))
    {
      dbnum->seq_errors++;
      local_errors++;
      if (local_errors < 5)
      {
        multilog(client->log, LOG_WARNING, "io_block: i=%d, seq[%"PRIu64"] != index[%"PRIu64"]\n",
                 i, seq, dbnum->index);
      }
    }

    dbnum->index++;
  }

  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "io_block: copied %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}



/*! Function that closes the data file */
int caspsr_dbnum_close (dada_client_t* client, uint64_t bytes_written)
{
  /* the caspsr_dbnum specific data */
  caspsr_dbnum_t* dbnum = 0;

  assert (client != 0);
  dbnum = (caspsr_dbnum_t*) client->context;
  assert (dbnum != 0);

  multilog(client->log, LOG_INFO, "close: %"PRIu64" bytes written this xfer\n",
           bytes_written);

  if (dbnum->seq_errors)
    multilog(client->log, LOG_WARNING, "close: %"PRIu64" seq errors this xfer\n",
             dbnum->seq_errors);

  dbnum->seq_errors = 0;
  dbnum->header_read = 0;

  return 0;
}

/*! Function that opens the data transfer target */
int caspsr_dbnum_open (dada_client_t* client)
{

  // the caspsr_dbnum specific data
  caspsr_dbnum_t* dbnum = (caspsr_dbnum_t*) client->context;

  uint64_t obs_offset = 0;

  int64_t obs_xfer = -1;

  // size (in bytes) of each encoded seq number
  uint64_t seq_size  = (uint64_t) sizeof(uint64_t);

  // Get the OBS_OFFSET
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_OFFSET\n");
    obs_offset = 0;
  }

  // get the OBS_XFER
  if (ascii_header_get (client->header, "OBS_XFER", "%"PRIi64, &obs_xfer) != 1)
  {
    multilog (client->log, LOG_WARNING, "open: header with no OBS_XFER\n");
    obs_xfer= -1;
  }

  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "open: OBS_OFFSET=%"PRIu64", OBS_XFER=%"PRIi64"\n",
              obs_offset, obs_xfer);

  if (obs_xfer == -1)
  {
    multilog (client->log, LOG_INFO, "open: OBS_XFER == -1, setting quit flag\n");
    dbnum->quit = 1;
  }

  // determine the expected starting index based on the OBS_OFFSET
  dbnum->index = dbnum->global_offset + (obs_offset / seq_size);
  if (dbnum->verbose)
    multilog (client->log, LOG_INFO, "open: global_offset=%"PRIu64", total_offset=%"PRIu64"\n",
              dbnum->global_offset, (obs_offset / seq_size));

  multilog (client->log, LOG_INFO, "open: setting index=%"PRIu64"\n", dbnum->index);

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  caspsr_dbnum_t dbnum = CASPSR_DBNUM_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* Quit flag for processing just 1 xfer */
  char quit = 0;

  /* global offset*/
  uint64_t global_offset = 1;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"g:hk:sv")) != -1)
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

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 's':
      quit = 1;
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

  log = multilog_open ("caspsr_dbnum", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  dbnum.global_offset = global_offset;
  dbnum.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = caspsr_dbnum_open;
  client->io_function    = caspsr_dbnum_io;
  client->io_block_function = caspsr_dbnum_io_block;
  client->close_function = caspsr_dbnum_close;

  client->optimal_bytes  = 256000000;
  //client->transfer_bytes = client->optimal_bytes * 4;

  client->direction = dada_client_reader;

  client->context = &dbnum;

  while (!client->quit && !dbnum.quit) 
  {
    if (dada_client_read (client) < 0) {
      multilog (log, LOG_ERR, "Error during transfer\n");
      return -1;
    }
    if (quit || dbnum.quit)
      client->quit = 1;
  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

