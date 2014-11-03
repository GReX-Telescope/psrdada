/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "daemon.h"

#include "mopsr_def.h"

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

#define DBSELECTANTSDB_CACHE_SIZE 4096

int quit_threads = 0;

int64_t dbselectantsdb_write_block_TFS_to_TFS (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id);

void usage()
{
  fprintf (stdout,
           "mopsr_dbselectantsdb [options] in_key out_key ant_indicies+\n"
           " -s             1 transfer, then exit\n"
           " -t             transpose output order from TS to ST [requires -z]\n"
           " -z             use zero copy transfers\n"
           " -v             verbose mode\n"
           " in_key         DADA key for input data block\n"
           " out_key        DADA keys for output data blocks\n"
           " ant_indicies   channel number to choose\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbselectantsdb_hdu_t;

typedef struct {

  mopsr_dbselectantsdb_hdu_t output;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  uint64_t in_block_size;

  // verbose output
  int verbose;

  unsigned int nant;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit;

  unsigned int nant_out;
  unsigned int output_ants[MOPSR_MAX_MODULES_PER_PFB];

  unsigned quit;

  int16_t * in_buf;
  int16_t * out_buf;

} mopsr_dbselectantsdb_t;


/*! Function that opens the data transfer target */
int dbselectantsdb_open (dada_client_t* client)
{
  // the mopsr_dbselectantsdb specific data
  mopsr_dbselectantsdb_t* ctx = (mopsr_dbselectantsdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbselectantsdb_open()\n");

  // lock writer status on the out HDU
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", ctx->output.key);

  if (dada_hdu_lock_write (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size;
  if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size) != 1)
  {
    transfer_size = 0;
  }

  int64_t file_size;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIi64, &file_size) != 1)
  {
    file_size = 0;
  }

  uint64_t obs_offset;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no OBS_OFFSET\n");
    return -1;
  }

  uint64_t bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BYTES_PER_SECOND\n");
    return -1;
  }

  // get the number of antenna
  if (ascii_header_get (client->header, "NANT", "%u", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NANT\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NBIT", "%u", &(ctx->nbit)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NBIT\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NDIM", "%u", &(ctx->ndim)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NDIM\n");
    return -1;                
  }                             

  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;                
  }

  char tmp[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", tmp) == 1)
  {
    multilog (log, LOG_INFO, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    multilog (log, LOG_INFO, "open: UTC_START=UNKNOWN\n");
  }

  uint64_t new_obs_offset = (obs_offset * ctx->nant_out) / ctx->nant;
  uint64_t new_bytes_per_second = (bytes_per_second  * ctx->nant_out) / ctx->nant;
  uint64_t new_file_size = (file_size * ctx->nant_out) / ctx->nant;

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "open: OBS_OFFSET %"PRIu64" -> %"PRIu64"\n", obs_offset, new_obs_offset);
  }

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: writing HDU %x\n",  ctx->output.key);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: enabling HDU %x\n", ctx->output.key);
  assert( header_size == ipcbuf_get_bufsz (ctx->output.hdu->header_block) );

  header = ipcbuf_get_next_write (ctx->output.hdu->header_block);
  if (!header) 
  {
    multilog (log, LOG_ERR, "open: could not get next header block\n");
    return -1;
  }

  // copy the header from the in to the out
  memcpy (header, client->header, header_size);

  if (ascii_header_set (header, "NANT", "%u", ctx->nant_out) < 0)
  {   
    multilog (log, LOG_ERR, "open: failed to write new NANT to header\n");
    return -1;  
  }               

  if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, new_obs_offset) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new OBS_OFFSET to header\n");
    return -1;
  }

  if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new BYTES_PER_SECOND to header\n");
    return -1;
  }

  if (file_size)
  {
    if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
    {
      multilog (log, LOG_ERR, "open: failed to write new FILE_SIZE to header\n");
      return -1;
    }
  }

  ctx->in_buf  = (int16_t *) malloc(DBSELECTANTSDB_CACHE_SIZE);
  ctx->out_buf = (int16_t *) malloc((DBSELECTANTSDB_CACHE_SIZE * ctx->nant_out) / ctx->nant);

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->output.hdu->header_block, header_size) < 0)
  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbselectantsdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbselectantsdb_t* ctx = (mopsr_dbselectantsdb_t*) client->context;
  
  multilog_t* log = client->log;

  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  // close the block if it is open
  if (ctx->output.block_open)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: ipcio_close_block_write bytes_written=%"PRIu64"\n");
    if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
    {
      multilog (log, LOG_ERR, "dbselectantsdb_close: ipcio_close_block_write failed\n");
      return -1;
    }
    ctx->output.block_open = 0;
    ctx->output.bytes_written = 0;
  }

  // unlock write on the datablock (end the transfer)
  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: dada_hdu_unlock_write\n");

  if (dada_hdu_unlock_write (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbselectantsdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  if (ctx->in_buf)
    free (ctx->in_buf);
  ctx->in_buf = 0;
  if (ctx->out_buf)
    free (ctx->out_buf);
  ctx->out_buf = 0;

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbselectantsdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbselectantsdb_t* ctx = (mopsr_dbselectantsdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write (ctx->output.hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += (data_size / ctx->nchan);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}

/*
 *  reorder the input samples from TFS order to TS order
 */
int64_t dbselectantsdb_write_block_TFS_to_TFS (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbselectantsdb_t* ctx = (mopsr_dbselectantsdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TFS data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = (in_data_size * ctx->nant_out) / ctx->nant;
  unsigned ichunk, isamp, iant, ichan;
  const unsigned nsamp = DBSELECTANTSDB_CACHE_SIZE / (ctx->ndim * ctx->nchan * ctx->nant);
  const unsigned chunk_size_in = DBSELECTANTSDB_CACHE_SIZE;
  const unsigned chunk_size_out = (DBSELECTANTSDB_CACHE_SIZE * ctx->nant_out) / ctx->nant;
  const unsigned nchunk = (unsigned) (in_data_size / chunk_size_in);
  const unsigned nchunk_left = (unsigned) (in_data_size % chunk_size_in);
  uint64_t out_block_id;

  // tmp ptrs to cache buffers
  int16_t * in, * out;

  // output pointer
  void * outdat;

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_TFS_to_TFS [%x] ipcio_open_block_write()\n", ctx->output.key);
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_TFS_to_TFS [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    outdat = (void *) ctx->output.curr_block;
  }
  else
    outdat = (void *) ctx->output.curr_block + ctx->output.bytes_written;

  for (ichunk=0; ichunk<nchunk; ichunk++)
  {
    memcpy ((void *) ctx->in_buf, in_data, chunk_size_in);

    in  = ctx->in_buf;
    out = ctx->out_buf;

    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        for (iant=0; iant<ctx->nant_out; iant++)
        {
          out[iant] = in[ctx->output_ants[iant]];
        }
        out += ctx->nant_out;
        in  += ctx->nant;
      }
    }

    // now write this chunk to the output block
    memcpy (outdat, (void *) &(ctx->out_buf), chunk_size_out);

    in_data += chunk_size_in;
    outdat += chunk_size_out;
  }

  // handle any remainder
  if (nchunk_left)
  {
    memcpy ((void *) ctx->in_buf, in_data, nchunk_left);
    unsigned nsamp_left = nchunk_left / (ctx->ndim * ctx->nchan * ctx->nant);
    unsigned chunk_size_out_left = nsamp_left * ctx->ndim;

    for (isamp=0; isamp<nsamp_left; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        for (iant=0; iant<ctx->nant_out; iant++)
        {
          out[iant] = in[ctx->output_ants[iant]];
        }
        out += ctx->nant_out;
        in  += ctx->nant;
      }
    }

    memcpy (outdat, (void *) &(ctx->out_buf), chunk_size_out_left);
  }

  ctx->output.bytes_written += out_data_size;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_TFS_to_TFS [%x] output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TFS [%x] bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_TFS_to_TFS [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_TFS_to_TFS [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_TFS_to_TFS [%x] ipcio_update_block_write failed\n", ctx->output.key);
        return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TFS_to_TFS [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_TFS_to_TFS [%x] ipcio_close_block_write failed\n", ctx->output.key);
        return -1;
      }
    }
    ctx->output.block_open = 0;
    ctx->output.bytes_written = 0;
  }
  else
  {
    if (ctx->output.bytes_written == 0)
      ctx->output.bytes_written = 1;
  }
  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TFS read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbselectantsdb_t dbselectantsdb;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  char single_transfer = 0;

  // use zero copy transfers
  char zero_copy= 0;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dhsvz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'h':
        usage();
        return (EXIT_SUCCESS);

      case 's':
        single_transfer = 1;
        break;

      case 'v':
        verbose++;
        break;
        
      case 'z':
        zero_copy = 1;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  dbselectantsdb.verbose = verbose;
  dbselectantsdb.quit = 0;

  int num_args = argc-optind;
  int i = 0;
      
  if (num_args < 3)
  {
    fprintf (stderr, "mopsr_dbselectantsdb: at least 3 arguments required\n");
    usage();
    return (EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbselectantsdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbselectantsdb.output.key)) != 1)
  {
    fprintf (stderr, "mopsr_dbselectantsdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  int iarg;
  dbselectantsdb.nant_out = (num_args - 2);
  for (iarg=2; iarg<num_args; iarg++)
  {
    if (sscanf (argv[optind+iarg], "%u", &(dbselectantsdb.output_ants[iarg])) != 1)
    {
      fprintf (stderr, "mopsr_dbselectantsdb: could not parse out ant %d from %s\n", iarg-2, argv[optind+iarg]);
      return EXIT_FAILURE;
    }
  }

  log = multilog_open ("mopsr_dbselectantsdb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbselectantsdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbselectantsdb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  dbselectantsdb.in_block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  dbselectantsdb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbselectantsdb.output.hdu, dbselectantsdb.output.key);
  if (dada_hdu_connect (dbselectantsdb.output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbselectantsdb.output.key);
    return -1;
  }
  dbselectantsdb.output.curr_block = 0;
  dbselectantsdb.output.bytes_written = 0;
  dbselectantsdb.output.block_open = 0;
  dbselectantsdb.output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbselectantsdb.output.hdu->data_block);

  if (dbselectantsdb.in_block_size != dbselectantsdb.output.block_size / dbselectantsdb.nant_out)
  {
    multilog (log, LOG_ERR, "input block size [%"PRIu64"] must be %d times "
              "larger than the output block size [%"PRIu64"]\n", dbselectantsdb.in_block_size, 
              dbselectantsdb.nant_out, dbselectantsdb.output.block_size);
    return (EXIT_FAILURE);
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbselectantsdb_open;
  client->io_function   = dbselectantsdb_write;

  client->io_block_function = dbselectantsdb_write_block_TFS_to_TFS;

  client->close_function = dbselectantsdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbselectantsdb;
  client->quiet = (verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || dbselectantsdb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
