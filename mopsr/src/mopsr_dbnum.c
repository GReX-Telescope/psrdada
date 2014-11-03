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
	   "mopsr_dbnum [options]\n"
     " -a nant    number of antenna to expect\n"
     " -c chan    channel to expect\n"
     " -e type    check type where: e=time a=ant c=chan\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -s         process only one observation\n"
     " -h         print help\n",
     DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  // channel number
  unsigned ichan;

  // nant
  unsigned nant;

  // current sequence number
  uint32_t seq;

  // flag for encoding seq numbers instead of PFB / ant
  char check_seq;

  char check_ant;

  char check_chan;

  /* flag for reading the header */
  unsigned header_read;

  unsigned verbose;

  uint64_t nerr;

} mopsr_dbnum_t;

#define DADA_DBNUM_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Pointer to the function that transfers data to/from the target */
int64_t mopsr_dbnum_io (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbnum_t * ctx = (mopsr_dbnum_t*) client->context;

  if (!ctx->header_read) 
  {
    ctx->header_read = 1;
    if (ctx->verbose)
      multilog (client->log, LOG_INFO, "mopsr_dbnum_io: read header\n");
    return data_size;
  }
}

int64_t mopsr_dbnum_io_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  mopsr_dbnum_t * ctx = (mopsr_dbnum_t*) client->context;

  const unsigned ndim = 2;
  const uint64_t nsamp = data_size / (ctx->nant * ndim);
  uint64_t isamp;
  unsigned ichan, iant;

  uint16_t * ptr16 = (uint16_t *) data;
  uint16_t aval, cval;

  cval = (uint16_t) ctx->ichan;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: data_size=%"PRIu64" block_id=%"PRIu64"\n", data_size, block_id);

  uint32_t seq_start = ctx->seq;

  for (iant=0; iant<ctx->nant; iant++)
  {
    aval = (uint16_t) iant;
    ctx->seq = seq_start;
    for (isamp=0; isamp<nsamp; isamp++)
    {
      if (ctx->check_seq)
      {
        if (ptr16[0] != (uint16_t) ctx->seq)
        {
          if (ctx->nerr < 10)
          {
            multilog (client->log, LOG_ERR, "[%"PRIu64"][%u][%"PRIu64"] data=%"PRIu16" expected=%"PRIu32" nerr=%"PRIu64"\n", block_id, iant, isamp, ptr16[0], ctx->seq, ctx->nerr);
          }
          ctx->nerr++;
        }
        ctx->seq = (ctx->seq + 1) % 65536;
      }
      else if (ctx->check_ant)
      {
        if (ptr16[0] != aval)
        {
          if (ctx->nerr < 10)
          {
            multilog (client->log, LOG_ERR, "check_ant ptr16[%"PRIu16"] != aval"
                      " [%"PRIu16"] failed\n", ptr16[0], aval);
          }
          ctx->nerr++; 
        }
      }
      else if (ctx->check_chan)
      {

        if (ptr16[0] != cval)
        {
          if (ctx->nerr < 10) 
          {
            multilog (client->log, LOG_ERR, "check_chan ptr16[%"PRIu16"] != cval [%"PRIu16"]\n", ptr16[0], cval);
          }
          ctx->nerr++;
        }
      }
      else
      {
        ;
      }
      ptr16++;
    }
  }

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "mopsr_dbnum_io: copied %"PRIu64" bytes\n", data_size);
#endif

  return (int64_t) data_size;
}

/*! Function that closes the data file */
int mopsr_dbnum_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbnum_t* ctx = 0;

  assert (client != 0);
  ctx = (mopsr_dbnum_t*) client->context;
  assert (ctx != 0);

  if (ctx->verbose || ctx->nerr > 0)
    multilog (client->log, LOG_INFO, "close: encountered %"PRIu64" errors\n", ctx->nerr);

  ctx->header_read = 0;

  return 0;
}

/*! Function that opens the data transfer target */
int mopsr_dbnum_open (dada_client_t* client)
{
  mopsr_dbnum_t* ctx = (mopsr_dbnum_t*) client->context;

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_dbnum_t ctx = DADA_DBNUM_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  ctx.check_seq = 0;
  ctx.check_ant = 1;
  ctx.check_chan = 0;


  while ((arg=getopt(argc,argv,"a:c:e:hk:n:sv")) != -1)
    switch (arg) {

    case 'a':
      ctx.nant = atoi (optarg);
      break;

    case 'c':
      ctx.ichan = atoi(optarg);
      break;

    case 'e':
      ctx.check_seq = (optarg[0] == 't');
      ctx.check_ant = (optarg[0] == 'a');
      ctx.check_chan = (optarg[0] == 'c');
      break;

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) 
      {
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

  log = multilog_open ("mopsr_dbnum", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  ctx.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = mopsr_dbnum_open;
  client->io_function = mopsr_dbnum_io;
  client->io_block_function = mopsr_dbnum_io_block;
  client->close_function = mopsr_dbnum_close;

  client->direction = dada_client_reader;

  client->context = &ctx;

  while (!client->quit) 
  {
    if (dada_client_read (client) < 0) {
      multilog (log, LOG_ERR, "Error during transfer\n");
      return -1;
    }

    if (quit)
      client->quit = 1;
  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

