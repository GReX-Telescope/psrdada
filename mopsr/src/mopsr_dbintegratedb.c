/***************************************************************************
 *  
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "daemon.h"

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

int quit_threads = 0;

int64_t dbintegratedb_write_block_SFT_to_SFT (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id);

void usage()
{
  fprintf (stdout,
           "mopsr_dbintegratedb [options] in_key out_key\n"
           " -s           1 transfer, then exit\n"
           " -t <factor>  integrate in time by this factor [default 1]\n"
           " -z           use zero copy transfers\n"
           " -v           verbose mode\n"
           " in_key    DADA key for input data block\n"
           " out_key   DADA keys for output data blocks\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbintegratedb_hdu_t;

typedef struct {

  mopsr_dbintegratedb_hdu_t output;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  uint64_t in_block_size;

  // verbose output
  int verbose;

  unsigned nant;
  unsigned nchan;
  unsigned ndim; 
  unsigned nbit;
  unsigned nbeam; 

  unsigned tscr;

  unsigned quit;

  unsigned control_port;

} mopsr_dbintegratedb_t;


/*! Function that opens the data transfer target */
int dbintegratedb_open (dada_client_t* client)
{
  // the mopsr_dbintegratedb specific data
  mopsr_dbintegratedb_t* ctx = (mopsr_dbintegratedb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbintegratedb_open()\n");

  // lock writer status on the out HDU
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", ctx->output.key);

  if (dada_hdu_lock_write (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  // parameters that will need to be adjusted due to transformation
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

  uint64_t resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no RESOLUTION\n");
    return -1;
  }

  double tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%lf", &tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no TSAMP\n");
    return -1;    
  }

  // parameters of relevance to transformation
  if (ascii_header_get (client->header, "NANT", "%u", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NANT\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NBEAM", "%u", &(ctx->nbeam)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NBEAM\n");
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

  // integration can only occur on detected powers (i.e. not voltages)
  if (ctx->ndim == 2)
  {
    multilog (log, LOG_ERR, "Cannot integrate voltage data (i.e. NDIM == 2)\n");
    return -1;
  } 

  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;                
  }

  char order[4];
  if (ascii_header_get (client->header, "ORDER", "%s", order) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    if (strcmp(order, "SFT") == 0)
    {
      client->io_block_function = dbintegratedb_write_block_SFT_to_SFT;
    }
    else
    {
      multilog (log, LOG_ERR, "open: unsupported input order\n");
      return -1;
    }
  }

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "open: NCHAN=%u NDIM=%u NBIT=%u\n", ctx->nchan,
              ctx->ndim, ctx->nbit);
    multilog (log, LOG_INFO, "open: NBEAM=%u NANT=%u ORDER=%s\n", ctx->nbeam,
              ctx->nant, order);
  }

  // changed quantities in outgoing header
  uint64_t new_obs_offset = obs_offset / ctx->tscr;
  uint64_t new_bytes_per_second = bytes_per_second / ctx->tscr;
  uint64_t new_file_size = file_size / ctx->tscr;
  uint64_t new_resolution = resolution / ctx->tscr;
  double   new_tsamp = tsamp * ctx->tscr;

  // changes due to integration
  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "open: BYTES_PER_SECOND %"PRIu64" -> %"PRIu64"\n", bytes_per_second, new_bytes_per_second);
    multilog (log, LOG_INFO, "open: OBS_OFFSET %"PRIu64" -> %"PRIu64"\n", obs_offset, new_obs_offset);
    multilog (log, LOG_INFO, "open: FILE_SIZE %"PRIu64" -> %"PRIu64"\n", file_size, new_file_size);
    multilog (log, LOG_INFO, "open: RESOLUTION %"PRIu64" -> %"PRIu64"\n", resolution, new_resolution);
    multilog (log, LOG_INFO, "open: TSAMP %lf -> %lf\n", tsamp, new_tsamp);
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

  if (ascii_header_set (header, "TSAMP", "%lf", new_tsamp) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new TSAMP to header\n");
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

  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, new_resolution) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new RESOLUTION to header\n");
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

int dbintegratedb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbintegratedb_t* ctx = (mopsr_dbintegratedb_t*) client->context;
  
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
      multilog (log, LOG_ERR, "dbintegratedb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbintegratedb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbintegratedb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbintegratedb_t* ctx = (mopsr_dbintegratedb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write (ctx->output.hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += (data_size / ctx->tscr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}

int64_t dbintegratedb_write_block_SFT_to_SFT (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  
  mopsr_dbintegratedb_t* ctx = (mopsr_dbintegratedb_t*) client->context;

  multilog_t * log = client->log;

  const uint64_t out_data_size = in_data_size / ctx->tscr;
  unsigned nsamp = (unsigned) (in_data_size / (ctx->nchan * (ctx->nbit/8) * ctx->ndim * ctx->nbeam));
  unsigned nscr  = nsamp / ctx->tscr;

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: data_size=%lu out_data_size=%lu nsamp=%u nscr=%u\n", in_data_size, out_data_size, nsamp, nscr);
  }

  uint64_t out_block_id;
  float * in, * out;
  char * outdat;

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_SFT: ipcio_open_block_write()\n");
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_SFT_to_SFT: ipcio_open_block_write failed: %s\n", strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    outdat = ctx->output.curr_block;
  }
  else
    outdat = ctx->output.curr_block + ctx->output.bytes_written;

  out = (float *) outdat;
  in  = (float *) in_data;
  float sum;

  unsigned ichan, ibeam, iscr, isamp;;

  for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (iscr=0; iscr<nscr; iscr++)
      {
        sum = 0;
        for (isamp=0; isamp<ctx->tscr; isamp++)
        {
          sum += in[isamp];
        }
        out[iscr] = sum;

        in += ctx->tscr;
      }
      out += nscr;
    }
  }

  ctx->output.bytes_written += out_data_size;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_SFT_to_SFT: output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_SFT: block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_SFT_to_SFT: update_block_write written=%"PRIu64"\n", ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_SFT: ipcio_update_block_write failed\n");
        return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_SFT_to_SFT: close_block_write written=%"PRIu64"\n", ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_SFT: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}


int main (int argc, char **argv)
{
  mopsr_dbintegratedb_t dbintegratedb;

  mopsr_dbintegratedb_t * ctx = &dbintegratedb;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  ctx->verbose = 0;

  // number of transfers
  char single_transfer = 0;

  // use zero copy transfers
  char zero_copy = 0;

  // time scrunch factor
  ctx->tscr = 1;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dhst:vz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'h':
        usage();
        return (EXIT_SUCCESS);

      case 't':
        if (optarg)
        {
          ctx->tscr = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "mopsr_dbintegratedb: -t requires argument\n");
          usage();
          return (EXIT_FAILURE);
        }

      case 's':
        single_transfer = 1;
        break;

      case 'v':
        ctx->verbose++;
        break;
        
      case 'z':
        zero_copy = 1;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  ctx->quit = 0;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf (stderr, "mopsr_dbintegratedb: 2 arguments required\n");
    usage();
    return (EXIT_FAILURE);
  } 

  if (ctx->verbose > 1)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+0]);
  if (sscanf (argv[optind+0], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbintegratedb: could not parse in key from %s\n", argv[optind+0]);
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(ctx->output.key)) != 1)
  {
    fprintf (stderr, "mopsr_dbintegratedb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbintegratedb", 0);

  multilog_add (log, stderr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbintegratedb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbintegratedb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  ctx->in_block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  ctx->output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (ctx->output.hdu, ctx->output.key);
  if (dada_hdu_connect (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }
  ctx->output.curr_block = 0;
  ctx->output.bytes_written = 0;
  ctx->output.block_open = 0;
  ctx->output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) ctx->output.hdu->data_block);

  if (ctx->in_block_size != ctx->output.block_size * ctx->tscr)
  {
    multilog (log, LOG_ERR, "input block size [%"PRIu64"] was not %u x output block size [%"PRIu64"]\n", ctx->in_block_size, ctx->tscr, ctx->output.block_size);
    return -1;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbintegratedb_open;
  client->io_function   = dbintegratedb_write;

  if (zero_copy)
    client->io_block_function = dbintegratedb_write_block_SFT_to_SFT;

  client->close_function = dbintegratedb_close;
  client->direction      = dada_client_reader;

  client->context = &dbintegratedb;
  client->quiet = (ctx->verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (ctx->verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || ctx->quit)
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
