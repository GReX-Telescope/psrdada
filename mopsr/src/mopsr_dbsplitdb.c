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

#define DBSPLITDB_CACHE_SIZE 4096

int quit_threads = 0;

void control_thread (void *);

int64_t dbsplitdb_write_block (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsplitdb_write_block_16 (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsplitdb_write_block_FST_to_FT (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsplitdb_write_block_TS_to_T (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsplitdb_write_block_FST_to_TF (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsplitdb_write_block_STF_to_TF (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbsplitdb [options] in_key out_keys+\n"
           " -b <core>     bind compuation to CPU core\n"
           " -s        1 transfer, then exit\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n"
           " in_key    DADA key for input data block\n"
           " out_keys  DADA keys for output data blocks\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbsplitdb_hdu_t;


typedef struct {

  mopsr_dbsplitdb_hdu_t * outputs;

  unsigned n_outputs; 

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  unsigned int nant;
  unsigned int nbeam;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit;

  unsigned quit;

  char order[4];

  void ** outs;

  int16_t in_buf[DBSPLITDB_CACHE_SIZE];
  int16_t out_buf[DBSPLITDB_CACHE_SIZE];

} mopsr_dbsplitdb_t;

#define DADA_DBSPLITDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbsplitdb_open (dada_client_t* client)
{
  // the mopsr_dbsplitdb specific data
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  mopsr_dbsplitdb_hdu_t * o = 0;  
 
  unsigned i = 0;

  char output_order[4];

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbsplitdb_open()\n");

  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);
    // lock writer status on the out HDU
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", o->key);

    if (dada_hdu_lock_write (o->hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", o->key);
      return -1;
    }
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

  uint64_t resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_WARNING, "open: header with no RESOLUTION\n");
    resolution = 0;
  }

  // get the number of antenna/beams
  if (ascii_header_get (client->header, "NANT", "%u", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NANT\n");
    return -1;
  }
  if (ascii_header_get (client->header, "NBEAM", "%u", &(ctx->nbeam)) != 1)
  {
    multilog (log, LOG_WARNING, "open: header with no NBEAM\n");
    ctx->nbeam = 1;
  }

  if ((ctx->nant != ctx->n_outputs) && (ctx->nbeam != ctx->n_outputs))
  {
    multilog (log, LOG_ERR, "open: header specified NANT=%u, NBEAM=%u, but only %d data blocks configured\n",
              ctx->nant, ctx->nbeam, ctx->n_outputs);
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

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    sprintf (output_order, "%s", "TF");
    multilog (log, LOG_INFO, "open: ORDER=%s\n", ctx->order);
    if ((strcmp(ctx->order, "FST") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from FST to TF\n");
      client->io_block_function = dbsplitdb_write_block_FST_to_TF;
    }
    if ((strcmp(ctx->order, "STF") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from STF to TF\n");
      client->io_block_function = dbsplitdb_write_block_STF_to_TF;
    }
    if ((strcmp(ctx->order, "ST") == 0) && client->io_block_function)
    {
      sprintf (output_order, "%s", "T");
      multilog (log, LOG_INFO, "open: changing order from ST to T\n");
      client->io_block_function = dbsplitdb_write_block_STF_to_TF;
    }
    if ((strcmp(ctx->order, "TS") == 0) && client->io_block_function)
    {
      sprintf (output_order, "%s", "T");
      multilog (log, LOG_INFO, "open: changing order from TS to T\n");
      client->io_block_function = dbsplitdb_write_block_TS_to_T;
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

  uint64_t bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BYTES_PER_SECOND\n");
    return -1;
  }

  uint64_t new_obs_offset = obs_offset / ctx->n_outputs;
  uint64_t new_bytes_per_second = bytes_per_second / ctx->n_outputs;
  uint64_t new_file_size = file_size / ctx->n_outputs;
  uint64_t new_resolution = resolution / ctx->n_outputs;

  multilog (log, LOG_INFO, "open: old_bytes_per_second=%"PRIu64" new_bytes_per_second=%"PRIu64" n_outputs=%d\n", bytes_per_second, new_bytes_per_second, ctx->n_outputs);

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  unsigned ant_id;

  // setup headers for all active HDUs
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: writing HDU %x\n",  o->key);

    assert( header_size == ipcbuf_get_bufsz (o->hdu->header_block) );

    header = ipcbuf_get_next_write (o->hdu->header_block);
    if (!header) 
    {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    // copy the header from the in to the out
    memcpy ( header, client->header, header_size );

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

    if (ctx->nant > 1)
    {
      // now set each output data block to 1 antenna
      int nant = 1;
      if (ascii_header_set (header, "NANT", "%d", nant) < 0)
      {
        multilog (log, LOG_ERR, "open: failed to write NANT=%d to header\n",
                                 nant);
        return -1;
      }

      sprintf (tmp, "ANT_ID_%d", i);
      if (ascii_header_get (client->header, tmp, "%u", &ant_id) != 1)
      {
        multilog (log, LOG_ERR, "open: header with no %s\n", tmp);
        return -1;
      }

      if (ascii_header_set (header, "ANT_ID", "%u", ant_id) < 0)
      {
        multilog (log, LOG_ERR, "open: failed to write ANT_ID=%u to header\n", ant_id);
        return -1;
      }
      if (ascii_header_set (header, "ORDER", "%s", "TF") < 0)
      {
        multilog (log, LOG_ERR, "open: failed to write ORDER=TF to header\n");
        return -1;
      }
    }
    if (ctx->nbeam > 1)
    {
      // now set each output data block to 1 antenna
      int nbeam = 1;
      if (ascii_header_set (header, "NBEAM", "%d", nbeam) < 0)
      {
        multilog (log, LOG_ERR, "open: failed to write NBEAM=%d to header\n",
                                 nbeam);
        return -1;
      }

      if (ascii_header_set (header, "ORDER", "%s", "T") < 0)
      {
        multilog (log, LOG_ERR, "open: failed to write ORDER=T to header\n");
        return -1;
      }
    }

    // mark the outgoing header as filled
    if (ipcbuf_mark_filled (o->hdu->header_block, header_size) < 0)  {
      multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
      return -1;
    }
    if (ctx->verbose) 
      multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", o->key);
  }

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbsplitdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  for (i=0; i<ctx->n_outputs; i++)
  { 
    o = &(ctx->outputs[i]);

    // close the block if it is open
    if (o->block_open)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "close: ipcio_close_block_write bytes_written=%"PRIu64"\n");
      if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "dbsplitdb_close: ipcio_close_block_write failed\n");
        return -1;
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }

    // unlock write on the datablock (end the transfer)
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: dada_hdu_unlock_write\n");

    if (dada_hdu_unlock_write (o->hdu) < 0)
    {
      multilog (log, LOG_ERR, "dbsplitdb_close: cannot unlock DADA HDU (key=%x)\n", o->key);
      return -1;
    }

    // mark this output's current state as inactive
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbsplitdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  unsigned i = 0;
  for (i=0; i<ctx->n_outputs; i++)
  {
    ipcio_write (ctx->outputs[i].hdu->data_block, data, data_size);
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}


/* default order is TFS */
int64_t dbsplitdb_write_block (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  uint64_t isplit;
  const uint64_t out_data_size = in_data_size / ctx->n_outputs;
  const uint64_t nsplit = out_data_size / ctx->ndim;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  uint64_t out_block_id;

  // 2 bytes / sample
  int16_t * in;
  int16_t * out;
  char * outdat;
  const unsigned nout = ctx->n_outputs;

  unsigned i = 0;
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);

    if (!o->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] ipcio_open_block_write()\n", o->key);
      o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
      if (!o->curr_block)
      { 
        multilog (log, LOG_ERR, "write_block: [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
        return -1;
      }
      o->block_open = 1;
      outdat = o->curr_block;
    }
    else
      outdat = o->curr_block + o->bytes_written;

    out = (int16_t *) outdat;
    in  = (int16_t *) in_data;
    in += i;

    for (isplit=0; isplit<nsplit; isplit++)
      out[isplit] = in[nout*isplit]; 
  
    o->bytes_written += out_data_size;

    if (o->bytes_written > o->block_size)
      multilog (log, LOG_ERR, "write_block: [%x] output block overrun by "
                "%"PRIu64" bytes\n", o->key, o->bytes_written - o->block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block: [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block: [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block: [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block: [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block: [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

/* default order is TFS - optimized 16-antenna case */
int64_t dbsplitdb_write_block_16 (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t * ctx = (mopsr_dbsplitdb_t*) client->context;
  multilog_t * log = client->log;
  mopsr_dbsplitdb_hdu_t * o;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block:_16 data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  uint64_t out_block_id;
  const uint64_t out_data_size = in_data_size / ctx->n_outputs;
  const unsigned nout = ctx->n_outputs;

  // we do 256 samples in a chunk

  unsigned ichunk, isamp, iout;
  const unsigned nsamp = DBSPLITDB_CACHE_SIZE / nout;
  const unsigned chunk_size_in = nsamp * ctx->ndim * nout;
  const unsigned chunk_size_out = nsamp * ctx->ndim;
  const unsigned nchunk = (unsigned) (in_data_size / chunk_size_in);
  const unsigned nchunk_left = (unsigned) (in_data_size % chunk_size_in);

  for (iout=0; iout<nout; iout++)
  {
    o = &(ctx->outputs[iout]);
    ctx->outs[iout] = (void *) ipcio_open_block_write(o->hdu->data_block, &out_block_id);
    o->curr_block = ctx->outs[iout];
    if (!o->curr_block)
    {
      multilog (log, LOG_ERR, "write_block: [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
      return -1;
    }
    o->block_open = 1;
  }

  for (ichunk=0; ichunk<nchunk; ichunk++)
  {
    memcpy ((void *) ctx->in_buf, in_data, chunk_size_in);

    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (iout=0; iout<nout; iout++)
      {
        ctx->out_buf[iout*nsamp + isamp] = ctx->in_buf[isamp*nout + iout];
      }
    }

    for (iout=0; iout<nout; iout++)
    {
      memcpy ((void *) (ctx->outs[iout] + ichunk*chunk_size_out), (void *) &(ctx->out_buf[nsamp*iout]), chunk_size_out); 
    }

    in_data += chunk_size_in;
  }

  // handle any remainder
  if (nchunk_left)
  {
    memcpy ((void *) ctx->in_buf, in_data, nchunk_left);
    unsigned nsamp_left = nchunk_left / (nout * ctx->ndim);
    unsigned chunk_size_out_left = nsamp_left * ctx->ndim;

    for (isamp=0; isamp<nsamp_left; isamp++)
    {
      for (iout=0; iout<nout; iout++)
      {
        ctx->out_buf[iout*nsamp + isamp] = ctx->in_buf[isamp*nout + iout];
      }
    }

    for (iout=0; iout<nout; iout++)
    {
      memcpy ((void *) (ctx->outs[iout] + nchunk*chunk_size_out), (void *) &(ctx->out_buf[nsamp*iout]), chunk_size_out_left); 
    }
  }

  for (iout=0; iout<nout; iout++)
  {
    o = &(ctx->outputs[iout]);
    o->bytes_written += out_data_size;
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block: [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block: [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block: [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block: [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}




/*
 *  reorder the input samples from FST order to S x TF order
 */
int64_t dbsplitdb_write_block_FST_to_TF (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_TF data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->n_outputs;
  const uint64_t nsplit = out_data_size / ctx->ndim;
  const uint64_t nsamp = nsplit / ctx->nchan;
  const uint64_t chan_stride = nsamp * ctx->n_outputs;
  const uint64_t ant_stride = nsamp;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_TF in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  uint64_t out_block_id, isamp;
  unsigned iant, ichan;

  // 2 bytes / sample
  int16_t * in, * out;
  char * outdat;

  for (iant=0; iant<ctx->n_outputs; iant++)
  {
    o = &(ctx->outputs[iant]);

    if (!o->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_TF [%x] ipcio_open_block_write()\n", o->key);
      o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
      if (!o->curr_block)
      {
        multilog (log, LOG_ERR, "write_block_FST_to_TF [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
        return -1;
      }
      o->block_open = 1;
      outdat = o->curr_block;
    }
    else
      outdat = o->curr_block + o->bytes_written;

    out = (int16_t *) outdat;
    in  = (int16_t *) in_data + (iant * ant_stride);

    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      //in = (int16_t *) in_data + (ichan * chan_stride) + (iant * ant_stride);
      //out = (int16_t *) outdat + ichan;
 
      for (isamp=0; isamp<nsamp; isamp++)
      {
        out[isamp*ctx->nchan] = in[isamp];
      }
      out++;
      in += chan_stride;
    }

    o->bytes_written += out_data_size;

    if (o->bytes_written > o->block_size)
      multilog (log, LOG_ERR, "write_block_FST_to_TF [%x] output block overrun by "
                "%"PRIu64" bytes\n", o->key, o->bytes_written - o->block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_FST_to_TF [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_TF [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_FST_to_TF [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_TF [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_FST_to_TF [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_TF [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }
  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_TF read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;

}

int64_t dbsplitdb_write_block_STF_to_TF (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->n_outputs;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: in_data_size=%"PRIu64", "
              "out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  void * in = in_data;
  void * out;
  
  uint64_t out_block_id;
  unsigned iant;

  for (iant=0; iant<ctx->n_outputs; iant++)
  {
    o = &(ctx->outputs[iant]);

    if (!o->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] ipcio_open_block_write()\n", o->key);
      o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
      if (!o->curr_block)
      {
        multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
        return -1;
      }
      o->block_open = 1;
      o->bytes_written = 0;
      out = o->curr_block;
    }
    else
      out = o->curr_block + o->bytes_written;

    memcpy (out, in, out_data_size);
    in += out_data_size;

    o->bytes_written += out_data_size;

    if (o->bytes_written > o->block_size)
      multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] output block overrun by "
                "%"PRIu64" bytes\n", o->key, o->bytes_written - o->block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;

}

int64_t dbsplitdb_write_block_FST_to_FT (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FT: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->n_outputs;
  const uint64_t nsplit = out_data_size / ctx->ndim;
  const uint64_t nsamp = nsplit / ctx->nchan;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FT in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  uint64_t out_block_id;

  void * in;
  void * out;
  unsigned iant, ichan;

  const uint64_t ant_stride = nsamp * ctx->ndim;
  const uint64_t out_stride = ant_stride;
  const uint64_t in_stride  = ctx->nant * ant_stride;

  for (iant=0; iant<ctx->n_outputs; iant++)
  {
    o = &(ctx->outputs[iant]);

    if (!o->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] ipcio_open_block_write()\n", o->key);
      o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
      if (!o->curr_block)
      {
        multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
        return -1;
      }
      o->block_open = 1;
      out = (void *) o->curr_block;
    }
    else
      out = (void *) o->curr_block + o->bytes_written;

    in = in_data + (iant * ant_stride);

    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      memcpy (out, in, out_stride);
      out += out_stride;
      in += in_stride;
    }

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FT read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

// 32-bit mode!!
int64_t dbsplitdb_write_block_TS_to_T (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsplitdb_t* ctx = (mopsr_dbsplitdb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbsplitdb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TS_to_T: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->n_outputs;
  const uint64_t nsamp = out_data_size / 4;

  uint64_t out_block_id;
  int32_t * in;
  int32_t * out;

  unsigned iant, isamp;

  for (iant=0; iant<ctx->n_outputs; iant++)
  {
    o = &(ctx->outputs[iant]);

    if (!o->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TS_to_T [%x] ipcio_open_block_write()\n", o->key);
      o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
      if (!o->curr_block)
      {
        multilog (log, LOG_ERR, "write_block_TS_to_T [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
        return -1;
      }
      o->block_open = 1;
      out = (int32_t *) o->curr_block;
    }
    else
      out = (int32_t *) (o->curr_block + o->bytes_written);

    in = ((int32_t *) in_data) + iant;

    for (isamp=0; isamp<nsamp; isamp++)
      out[isamp] = in[isamp*ctx->n_outputs];

    o->bytes_written += out_data_size;

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_TS_to_T [%x] block now full bytes_written=%"
                  PRIu64", block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
      {
        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_TS_to_T [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TS_to_T [%x] ipcio_update_block_write failed\n", o->key);
          return -1;
        }
      }
      else
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_TS_to_T [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
        if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
        {
          multilog (log, LOG_ERR, "write_block_TS_to_T [%x] ipcio_close_block_write failed\n", o->key);
          return -1;
        }
      }
      o->block_open = 0;
      o->bytes_written = 0;
    }
    else
    {
      if (o->bytes_written == 0)
        o->bytes_written = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TS_to_T read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}




int main (int argc, char **argv)
{
  mopsr_dbsplitdb_t dbsplitdb = DADA_DBSPLITDB_INIT;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  int core = -1;

  pthread_t control_thread_id;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hsvz")) != -1)
  {
    switch (arg) 
    {
      case 'b':
        if (optarg)
        {
          core = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "ERROR: -b requires argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 'h':
        usage();
        return EXIT_SUCCESS;

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

  if (core >= 0)
    if (dada_bind_thread_to_core(core) < 0)
      multilog(log, LOG_WARNING, "main: failed to bind to core %d\n", core);


  dbsplitdb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) < 2)
  {
    fprintf(stderr, "mopsr_dbsplitdb: at least 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbsplitdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  dbsplitdb.n_outputs = (unsigned) num_args - 1;
  dbsplitdb.outs = (void ** ) malloc (sizeof (void *) * dbsplitdb.n_outputs);
  dbsplitdb.outputs = (mopsr_dbsplitdb_hdu_t *) malloc (sizeof(mopsr_dbsplitdb_hdu_t) * dbsplitdb.n_outputs);
  if (!dbsplitdb.outputs)
  {
    fprintf (stderr, "mopsr_dbsplitdb: could not allocate memory\n");
    return EXIT_FAILURE;
  }

  // read output DADA keys from command line arguments
  for (i=1; i<num_args; i++)
  {
    if (verbose)
      fprintf (stderr, "parsing output key %d=%s\n", i-1, argv[optind+i]);
    if (sscanf (argv[optind+i], "%x", &(dbsplitdb.outputs[i-1].key)) != 1) {
      fprintf (stderr, "mopsr_dbsplitdb: could not parse out key %d from %s\n", i, argv[optind+i]);
      return EXIT_FAILURE;
    }
  }

  log = multilog_open ("mopsr_dbsplitdb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbsplitdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbsplitdb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  if (verbose)
    multilog (log, LOG_INFO, "main: n_outputs=%u\b", dbsplitdb.n_outputs);

  // setup output data blocks
  for (i=0; i<dbsplitdb.n_outputs; i++)
  {
    dbsplitdb.outputs[i].hdu = dada_hdu_create (log);
    dada_hdu_set_key (dbsplitdb.outputs[i].hdu, dbsplitdb.outputs[i].key);
    if (dada_hdu_connect (dbsplitdb.outputs[i].hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbsplitdb.outputs[i].key);
      return -1;
    }
    dbsplitdb.outputs[i].curr_block = 0;
    dbsplitdb.outputs[i].bytes_written = 0;
    dbsplitdb.outputs[i].block_open = 0;
    dbsplitdb.outputs[i].block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbsplitdb.outputs[i].hdu->data_block);
    if (verbose)
      multilog (log, LOG_INFO, "main: dbsplitdb.outputs[%d].block_size=%"PRIu64"\n", i, dbsplitdb.outputs[i].block_size);
    if (zero_copy && ((block_size / dbsplitdb.n_outputs) != dbsplitdb.outputs[i].block_size))
    {
      multilog (log, LOG_ERR, "for zero copy, all DADA buffer block sizes must "
                              "be the same size\n");
      return EXIT_FAILURE;
    }
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbsplitdb_open;
  client->io_function   = dbsplitdb_write;

  if (zero_copy)
  {
    client->io_block_function = dbsplitdb_write_block_16;
  }

  client->close_function = dbsplitdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbsplitdb;
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

    if (single_transfer || dbsplitdb.quit)
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

  free (dbsplitdb.outs);
  free (dbsplitdb.outputs);

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
