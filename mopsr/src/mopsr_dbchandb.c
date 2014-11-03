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

int64_t dbchandb_write_block_TFS_to_TS (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id);
int64_t dbchandb_write_block_TFS_to_ST (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id);
int64_t dbchandb_write_block_FST_to_ST (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id);

void usage()
{
  fprintf (stdout,
           "mopsr_dbchandb [options] chan in_key out_key\n"
           " -s        1 transfer, then exit\n"
           " -t        transpose output order from TS to ST [requires -z]\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n"
           " chan      channel number to choose\n"
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
} mopsr_dbchandb_hdu_t;

typedef struct {

  mopsr_dbchandb_hdu_t output;

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

  unsigned int channel;

  unsigned quit;

  unsigned control_port;

  char transpose;

  char input_order[4];

  char output_order[3];

} mopsr_dbchandb_t;


/*! Function that opens the data transfer target */
int dbchandb_open (dada_client_t* client)
{
  // the mopsr_dbchandb specific data
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbchandb_open()\n");

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

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->input_order)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    multilog (log, LOG_INFO, "open: ORDER=%s\n", ctx->input_order);
    if (strcmp(ctx->input_order, "FST") == 0)
      client->io_block_function = dbchandb_write_block_FST_to_ST;
    else if (strcmp(ctx->input_order, "TFS") != 0)
    {
      ;
    }
    else
    {
      multilog (log, LOG_ERR, "open: unsupported input order\n");
      return -1;
    }
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

  float bw;
  if (ascii_header_get (client->header, "BW", "%f", &bw) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BW\n");
    return -1;
  }

  float freq;
  if (ascii_header_get (client->header, "FREQ", "%f", &freq) != 1)
  {     
    multilog (log, LOG_ERR, "open: header with no FREQ\n");
    return -1;    
  }

  unsigned chan_offset;
  if (ascii_header_get (client->header, "CHAN_OFFSET", "%d", &chan_offset) != 1)
  {
    multilog (log, LOG_WARNING, "open: header with no CHAN_OFFSET, assuming 0\n");
    chan_offset = 0;
  }

  if (ctx->in_block_size / ctx->nchan != ctx->output.block_size)
  {
    multilog (log, LOG_ERR, "open: input block size / nchan != output block size\n");
    return -1;
  }

  uint64_t new_obs_offset = obs_offset / ctx->nchan;
  uint64_t new_bytes_per_second = bytes_per_second / ctx->nchan;
  uint64_t new_file_size = file_size / ctx->nchan;
  float    new_bw = bw / (float) ctx->nchan;
  float    base_freq = freq - (bw / 2.0);
  float    new_freq = base_freq + (ctx->channel * new_bw) + (new_bw / 2.0);
  unsigned new_chan_offset = chan_offset + ctx->channel;

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

  if (ascii_header_set (header, "FREQ", "%f", new_freq) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new FREQ to header\n");
    return -1;
  }

  if (ascii_header_set (header, "BW", "%f", new_bw) < 0)
  {   
    multilog (log, LOG_ERR, "open: failed to write new BW to header\n");
    return -1;  
  }               

  if (ascii_header_set (header, "NCHAN", "%d", 1) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new NCHAN to header\n");
    return -1;
  }

  if (ascii_header_set (header, "CHAN_OFFSET", "%d", new_chan_offset) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new CHAN_OFFSET to header\n");
    return -1;
  }

  if (strcmp(ctx->input_order, "TFS") == 0)
  {
    if (ctx->transpose)
      sprintf (ctx->output_order, "%s", "ST");
    else
      sprintf (ctx->output_order, "%s", "TS");
  }
  else
    sprintf (ctx->output_order, "%s", "ST");

  if (ascii_header_set (header, "ORDER", "%s", ctx->output_order) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write ORDER=%s  to header\n", ctx->output_order);
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

int dbchandb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t*) client->context;
  
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
      multilog (log, LOG_ERR, "dbchandb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbchandb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbchandb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t*) client->context;

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

int64_t dbchandb_write_block_FST_to_ST (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_ST data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->nchan;
  const uint64_t offset = ctx->channel * out_data_size;

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "write_block_FST_to_ST: out_data_size=%lu\n", out_data_size);
    multilog (log, LOG_INFO, "write_block_FST_to_ST: channel offset=%lu\n", offset);
  }


  uint64_t out_block_id, isamp;
  void * in, * out;
  char * outdat;

  {
    if (!ctx->output.block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_ST [%x] ipcio_open_block_write()\n", ctx->output.key);
      ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
      if (!ctx->output.curr_block)
      {
        multilog (log, LOG_ERR, "write_block_FST_to_ST [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
        return -1;
      }
      ctx->output.block_open = 1;
      outdat = ctx->output.curr_block;
    }
    else
      outdat = ctx->output.curr_block + ctx->output.bytes_written;

    out = (void *) outdat;
    in  = (void *) in_data + offset;

    memcpy (out, in, out_data_size);

    ctx->output.bytes_written += out_data_size;

    if (ctx->output.bytes_written > ctx->output.block_size)
      multilog (log, LOG_ERR, "write_block_FST_to_ST [%x] output block overrun by "
                "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_FST_to_ST [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if the output block is now full
    if (ctx->output.bytes_written >= ctx->output.block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_ST [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_FST_to_ST [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_ST [%x] ipcio_update_block_write failed\n", ctx->output.key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_FST_to_ST [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_ST [%x] ipcio_close_block_write failed\n", ctx->output.key);
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
  }
  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_ST read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}


/*
 *  reorder the input samples from TFS order to TS order
 */
int64_t dbchandb_write_block_TFS_to_TS (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TS data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->nchan;
  const uint64_t nsplit = out_data_size / ctx->ndim;
  const uint64_t nsamp = nsplit / ctx->nant;
  const size_t chan_stride = ctx->nant * ctx->ndim;
  const size_t samp_stride = ctx->nchan * chan_stride;

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "write_block_TFS_to_TS: nsamp=%ld\n", nsamp);
    multilog (log, LOG_INFO, "write_block_TFS_to_TS: chan_stride=%ld\n", chan_stride);
    multilog (log, LOG_INFO, "write_block_TFS_to_TS: samp_stride=%ld\n", samp_stride);
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TS in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  uint64_t out_block_id, isamp;
  void * in, * out;
  char * outdat;

  {
    if (!ctx->output.block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TFS_to_TS [%x] ipcio_open_block_write()\n", ctx->output.key);
      ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
      if (!ctx->output.curr_block)
      {
        multilog (log, LOG_ERR, "write_block_TFS_to_TS [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
        return -1;
      }
      ctx->output.block_open = 1;
      outdat = ctx->output.curr_block;
    }
    else
      outdat = ctx->output.curr_block + ctx->output.bytes_written;

    out = outdat;
    in  = in_data + (ctx->channel * chan_stride);

    for (isamp=0; isamp<nsamp; isamp++)
    {
      memcpy (out, in, chan_stride);
      out += chan_stride;
      in  += samp_stride;
    }
      
    ctx->output.bytes_written += out_data_size;

    if (ctx->output.bytes_written > ctx->output.block_size)
      multilog (log, LOG_ERR, "write_block_TFS_to_TS [%x] output block overrun by "
                "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_TFS_to_TS [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if the output block is now full
    if (ctx->output.bytes_written >= ctx->output.block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TFS_to_TS [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_TFS_to_TS [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TFS_to_TS [%x] ipcio_update_block_write failed\n", ctx->output.key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_TFS_to_TS [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TFS_to_TS [%x] ipcio_close_block_write failed\n", ctx->output.key);
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
  }
  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_TS read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

/*
 *  reorder the input samples from TFS order to ST order
 */
int64_t dbchandb_write_block_TFS_to_ST (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbchandb_t* ctx = (mopsr_dbchandb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_ST data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->nchan;
  const uint64_t nsamp = out_data_size / (ctx->ndim * ctx->nant);
  const size_t chan_stride = ctx->nant; // 16 bit qtys
  const size_t samp_stride = ctx->nchan * chan_stride; // 16 bit qtys

  if (ctx->verbose)
  {
    multilog (log, LOG_INFO, "write_block_TFS_to_ST: nsamp=%ld\n", nsamp);
    multilog (log, LOG_INFO, "write_block_TFS_to_ST: chan_stride=%ld\n", chan_stride);
    multilog (log, LOG_INFO, "write_block_TFS_to_ST: samp_stride=%ld\n", samp_stride);
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_ST in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  uint64_t out_block_id, isamp;
  int16_t * in, * out;
  char * outdat;
  unsigned iant, ichan;

  {
    if (!ctx->output.block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TFS_to_ST [%x] ipcio_open_block_write()\n", ctx->output.key);
      ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
      if (!ctx->output.curr_block)
      {
        multilog (log, LOG_ERR, "write_block_TFS_to_ST [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
        return -1;
      }
      ctx->output.block_open = 1;
      outdat = ctx->output.curr_block;
    }
    else
      outdat = ctx->output.curr_block + ctx->output.bytes_written;

    out = (int16_t *) outdat;
    in  = (int16_t *) in_data; 
    //in  = (int16_t *) (in_data + (ctx->channel * chan_stride));

    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        for (iant=0; iant<ctx->nant; iant++)
        {
          if (ichan == ctx->channel)
          {
            out[iant*nsamp+isamp] = in[0];
            //out[iant*nsamp+isamp] = (int16_t) (isamp % 8);
          }
          in++;
        }
      }
    }

/*
    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (iant=0; iant<ctx->nant; iant++)
      {
        //if ((block_id == 0) && (isamp < 10))
        //  fprintf (stderr, "out[%"PRIu64"] = in[%"PRIu64"]\n", (isamp + iant*nsamp), (ctx->channel * chan_stride) + ((isamp * samp_stride) + iant));
        out[iant*nsamp] = in[iant];
      }
      in += samp_stride;
      out ++;
    }
*/
      
    ctx->output.bytes_written += out_data_size;

    if (ctx->output.bytes_written > ctx->output.block_size)
      multilog (log, LOG_ERR, "write_block_TFS_to_ST [%x] output block overrun by "
                "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_TFS_to_ST [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if the output block is now full
    if (ctx->output.bytes_written >= ctx->output.block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TFS_to_ST [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_TFS_to_ST [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TFS_to_ST [%x] ipcio_update_block_write failed\n", ctx->output.key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_TFS_to_ST [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
        if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TFS_to_ST [%x] ipcio_close_block_write failed\n", ctx->output.key);
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
  }
  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFS_to_ST read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbchandb_t dbchandb;

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

  char transpose = 0;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dhp:stvz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'h':
        usage();
        return (EXIT_SUCCESS);

      case 'p':
        if (optarg)
        {
          dbchandb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "mopsr_dbchandb: -p requires argument\n");
          usage();
          return (EXIT_FAILURE);
        }

      case 's':
        single_transfer = 1;
        break;

      case 't':
        transpose = 1;

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

  dbchandb.verbose = verbose;
  dbchandb.transpose = transpose;
  dbchandb.quit = 0;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 3)
  {
    fprintf (stderr, "mopsr_dbchandb: 3 arguments required\n");
    usage();
    return (EXIT_FAILURE);
  } 

  if (transpose && !zero_copy)
  {
    fprintf (stderr, "mopsr_dbchandb: transpose requires zero copy flag\n");
    usage();
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "parsing channel from %s\n", argv[optind]);
  if (sscanf (argv[optind], "%d", &(dbchandb.channel)) != 1)
  {
    fprintf (stderr, "mopsr_dbchandb: could not parse channel from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }
  if (verbose)
    fprintf (stderr, "channel=%d\n", dbchandb.channel);


  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbchandb: could not parse in key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+2]);
  if (sscanf (argv[optind+2], "%x", &(dbchandb.output.key)) != 1)
  {
    fprintf (stderr, "mopsr_dbchandb: could not parse out key from %s\n", argv[optind+2]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbchandb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbchandb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbchandb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  dbchandb.in_block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  dbchandb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbchandb.output.hdu, dbchandb.output.key);
  if (dada_hdu_connect (dbchandb.output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbchandb.output.key);
    return -1;
  }
  dbchandb.output.curr_block = 0;
  dbchandb.output.bytes_written = 0;
  dbchandb.output.block_open = 0;
  dbchandb.output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbchandb.output.hdu->data_block);

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbchandb_open;
  client->io_function   = dbchandb_write;

  if (zero_copy)
    if (transpose)
      client->io_block_function = dbchandb_write_block_TFS_to_ST;
    else
      client->io_block_function = dbchandb_write_block_TFS_to_TS;

  client->close_function = dbchandb_close;
  client->direction      = dada_client_reader;

  client->context = &dbchandb;
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

    if (single_transfer || dbchandb.quit)
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
