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
#include <float.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>

int quit_threads = 0;

int compare (const void * a, const void * b);
int compare (const void * a, const void * b)
{
  return ( *(float *)a - *(float *)b );
}

int64_t dbrescaledb_write_block_FST_to_FST (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbrescaledb_write_block_SFT_to_SFT (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbrescaledb [options] in_key out_key\n"
           "              data type change from 32-bit float to 8-bit unsigned int\n"
           " -f val       apply val as global scale\n" 
           " -i           individually rescale reach signal\n" 
           " -o val       apply val as global offset to data\n"
           " -t secs      calculate scales over secs\n"
           " -s           1 transfer, then exit\n"
           " -z           use zero copy transfers\n"
           " -v           verbose mode\n"
           " in_key       DADA key for input data block\n"
           " out_key      DADA key for output data blocks\n");
}

typedef struct {

  dada_hdu_t *  out_hdu;
  key_t         out_key;
  uint64_t      out_block_size;
  char *        out_block;

  // number of bytes read
  uint64_t bytes_in;
  uint64_t bytes_out;
  unsigned bitrate_factor;

  // verbose output
  int verbose;

  unsigned int nsig;
  unsigned int nant;
  unsigned int nbeam;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit_in;
  unsigned int nbit_out;

  double * sums;
  double * sums_sq;
  float * work;
  float * offsets;
  float * scales;
  uint64_t * nclipped;

  unsigned quit;
  char order[4];
  uint64_t n_errors;

  uint64_t nsamps_integrated;
  uint64_t nsamps_to_integrate;
  float seconds_to_integrate;

  float offset;
  float scale;

  char global_rescale;
  float global_offset;
  float global_scale;

  char indiv_rescale;

} mopsr_dbrescaledb_t;

#define DADA_DBREBLOCKDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbrescaledb_open (dada_client_t* client)
{
  // the mopsr_dbrescaledb specific data
  mopsr_dbrescaledb_t* ctx = (mopsr_dbrescaledb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbrescaledb_open()\n");

  char output_order[4];

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", ctx->out_key);
  if (dada_hdu_lock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: dada_hdu_lock_write\n");

  float bw;
  if (ascii_header_get (client->header, "BW", "%f", &bw) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BW\n");
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  // get the number of antenna and beams
  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    ctx->nant = 1;
  }
  if (ascii_header_get (client->header, "NBEAM", "%d", &(ctx->nbeam)) != 1)
  {
    ctx->nbeam = 1;
  }
  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;                
  }

  /*
  // this code injects an FRB into a specific output beam
  if (ascii_header_get (client->header, "FRB_INJECTION", "%d", &(ctx->frb_inj)) == 1)
  {
    if (ascii_header_get (client->header, "FRB_INJECTION_BEAM", "%d", &(ctx->frb_inj_beam)) != 1)
    {
      multilog (log, LOG_ERR, "open: FRB_INJECTION with no FRB_INJECTION_BEAM\n");
      return -1;
    }

    if (ascii_header_get (client->header, "FRB_INJECTION_WIDTH", "%f", &(ctx->frb_inj_width)) != 1)
    {
      multilog (log, LOG_ERR, "open: FRB_INJECTION with no FRB_INJECTION_WIDTH\n");
      return -1;
    }

    if (ascii_header_get (client->header, "FRB_INJECTION_DM", "%f", &(ctx->frb_inj_dm)) != 1)
    {
      multilog (log, LOG_ERR, "open: FRB_INJECTION with no FRB_INJECTION_DM\n");
      return -1;
    }

    if (ascii_header_get (client->header, "FRB_INJECTION_SNR", "%f", &(ctx->frb_inj_snr)) != 1)
    {
      multilog (log, LOG_ERR, "open: FRB_INJECTION with no FRB_INJECTION_SNR\n");
      return -1;
    }

    if (ascii_header_get (client->header, "FRB_INJECTION_UTC", "%s", &(ctx->frb_inj_utc)) != 1)
    {
      multilog (log, LOG_ERR, "open: FRB_INJECTION with no FRB_INJECTION_SNR\n");
      return -1;
    }
  }
  */

  if (ctx->nant == 1 && ctx->nbeam > 1)
    ctx->nsig = ctx->nbeam;
  else if (ctx->nant > 1 && ctx->nbeam == 1)
    ctx->nsig = ctx->nant;
  else
  {
    multilog (log, LOG_ERR, "open: cannot rescale when nant=%d and nbeam=%d\n", ctx->nant, ctx->nbeam);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: using nant=%d, nbeam=%d, nsig=%d nchan=%d\n", ctx->nant, ctx->nbeam, ctx->nsig, ctx->nchan);

  if (ctx->scales)
    free (ctx->scales);
  ctx->scales = (float *) malloc (sizeof(float) * ctx->nsig * ctx->nchan);

  if (ctx->offsets)
    free (ctx->offsets);
  ctx->offsets = (float *) malloc (sizeof(float) * ctx->nsig * ctx->nchan);

  if (ctx->work)
    free (ctx->work);
  ctx->work = (float *) malloc (sizeof(float) * (ctx->nsig-1));

  if (ctx->sums)
    free (ctx->sums);
  ctx->sums = (double *) malloc (sizeof(double) * ctx->nsig * ctx->nchan);

  if (ctx->sums_sq)
    free (ctx->sums_sq);
  ctx->sums_sq = (double *) malloc (sizeof(double) * ctx->nsig * ctx->nchan);

  if (ctx->nclipped)
    free (ctx->nclipped);
  ctx->nclipped = (uint64_t *) malloc (sizeof(uint64_t) * ctx->nsig * ctx->nchan);

  if (ascii_header_get (client->header, "NBIT", "%u", &(ctx->nbit_in)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NBIT\n");
    return -1;
  }
  ctx->nbit_out = 8;

  if (ascii_header_get (client->header, "NDIM", "%u", &(ctx->ndim)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NDIM\n");
    return -1;                
  }                             

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    // for summing of data blocks we always want TF output mode
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: ORDER=%s\n", ctx->order);
    if ((strcmp(ctx->order, "SFT") == 0) && client->io_block_function)
    {
      client->io_block_function = dbrescaledb_write_block_SFT_to_SFT;
      strcpy (output_order, "SFT");
    }
    else if ((strcmp(ctx->order, "ST") == 0) && client->io_block_function)
    {
      client->io_block_function = dbrescaledb_write_block_FST_to_FST;
      strcpy (output_order, "ST");
    }
    else
    {
      multilog (log, LOG_ERR, "open: input ORDER=%s is not supported\n", ctx->order);
      return -1;
    }
  }

  char tmp[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", tmp) == 1)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    multilog (log, LOG_INFO, "open: UTC_START=UNKNOWN\n");
  }

  uint64_t resolution, new_resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no RESOLUTION\n");
    return -1;
  }

  uint64_t bytes_per_second, new_bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BYTES_PER_SECOND\n");
    return -1;
  }

  uint64_t file_size, new_file_size;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no FILE_SIZE\n");
    return -1;
  }

  float tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%f", &tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: failed to read TSAMP from header\n");
    return -1;
  }

  float samples_per_second = 1000000.0f/ tsamp;

  // TSAMP origin
  ctx->nsamps_to_integrate = (uint64_t) (ctx->seconds_to_integrate * samples_per_second);
  ctx->nsamps_integrated = 0;

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  assert( header_size == ipcbuf_get_bufsz (ctx->out_hdu->header_block) );

  header = ipcbuf_get_next_write (ctx->out_hdu->header_block);
  if (!header) 
  {
    multilog (log, LOG_ERR, "open: could not get next header block\n");
    return -1;
  }

  // copy the header from the in to the out
  memcpy ( header, client->header, header_size );

  if (ascii_header_set (header, "ORDER", "%s", output_order) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write ORDER=%s to header\n", output_order);
    return -1;
  }

  new_bytes_per_second = bytes_per_second / ctx->bitrate_factor;
  if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write BYTES_PER_SECOND=%"PRIu64" to header\n", bytes_per_second);
    return -1;
  }

  new_file_size = file_size / ctx->bitrate_factor;
  if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write FILE_SIZE=%"PRIu64" to header\n", file_size);
    return -1;
  }

  new_resolution = resolution / ctx->bitrate_factor;
  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, new_resolution) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write RESOLUITON=%"PRIu64" to header\n", new_resolution);
    return -1;
  }

  if (ascii_header_set (header, "NBIT", "%"PRIu16, ctx->nbit_out) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NBIT=%"PRIu16" to header\n", ctx->nbit_out);
    return -1;
  }

  // TODO fix corner turn to deal with this
  if (ascii_header_set (header, "NBEAM", "%d", ctx->nsig) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NBEAM=%d to header\n", ctx->nsig);
    return -1;
  }

  int nant = 1;
  if (ascii_header_set (header, "NANT", "%d", nant) < 0)
  {     
    multilog (log, LOG_ERR, "open: failed to write NBEAM=%d to header\n", ctx->nsig);
    return -1;    
  }                 

  unsigned ichan, isig, i;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (isig=0; isig<ctx->nsig; isig++)
    {
      i = ichan * ctx->nsig + isig;
      ctx->sums[i] = 0;
      ctx->sums_sq[i] = 0;
      ctx->nclipped[i] = 0;
      ctx->offsets[i] = ctx->global_offset;
      ctx->scales[i] = ctx->global_scale;
    }
  }
  ctx->offset = 0;
  ctx->scale = 1;

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->out_hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }
  if (ctx->verbose) 
    multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", ctx->out_key);

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbrescaledb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbrescaledb_t* ctx = (mopsr_dbrescaledb_t*) client->context;
  
  multilog_t* log = client->log;

  unsigned ichan, isig, i;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (isig=0; isig<ctx->nsig; isig++)
    {
      i = ichan * ctx->nsig + isig;
      if (ctx->nclipped[i] && ctx->verbose)
        multilog (log, LOG_INFO, "close: samples from chan=%u sig=%u that exceed power limits=%"PRIu64"\n", ichan, isig, ctx->nclipped[i]);
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  // unlock write on the datablock (end the transfer)
  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: dada_hdu_unlock_write\n");

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbrescaledb_close: cannot unlock DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }

  if (ctx->scales)
    free (ctx->scales);
  ctx->scales = 0;

  if (ctx->offsets)
    free (ctx->offsets);
  ctx->offsets = 0;

  if (ctx->sums)
    free (ctx->sums);
  ctx->sums = 0;

  if (ctx->sums_sq)
    free (ctx->sums_sq);
  ctx->sums_sq = 0;

  if (ctx->work)
    free (ctx->work);
  ctx->work = 0;

  if (ctx->nclipped)
    free (ctx->nclipped);
  ctx->nclipped = 0;

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbrescaledb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbrescaledb_t* ctx = (mopsr_dbrescaledb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write (ctx->out_hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}

// rescale the ST data from 32-bit floats to 8-bit unsigned integers.
int64_t dbrescaledb_write_block_FST_to_FST (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  mopsr_dbrescaledb_t* ctx = (mopsr_dbrescaledb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FST: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp  = data_size / (ctx->nchan * ctx->nsig * ctx->ndim * (ctx->nbit_in/8));

  //  assume 32-bit, detected (ndim == 1) data
  float * in = (float *) in_data;
  uint64_t out_block_id, isamp;
  double val;
  unsigned ichan, isig, i;
  float mean, mean_sq, variance;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FST: nsamp=%lu\n", nsamp);

  // if the input power levels are to be measured 
  while ((!ctx->global_rescale) && (ctx->nsamps_integrated < ctx->nsamps_to_integrate))
  {
    ctx->nsamps_integrated += nsamp;

    // compute the mean power level across all beams and channels
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      for (isig=0; isig<ctx->nsig; isig++)
      {
        // ichansig
        i = ichan * ctx->nsig + isig;

        uint64_t total_power = 0;
        for (isamp=0; isamp<nsamp; isamp++)
        {
          val = (double) in[i * nsamp + isamp];
          ctx->sums[i] += val;
          ctx->sums_sq[i] += (val * val);
        }

        mean = (float) (ctx->sums[i] / ctx->nsamps_integrated);
        mean_sq = (float) (ctx->sums_sq[i] / ctx->nsamps_integrated);
        variance = mean_sq - (mean * mean);

        ctx->offsets[i] = mean;
        if (variance == 0)
          ctx->scales[i] = 1.0;
        else
          ctx->scales[i] = 1.0 / sqrt(variance);

        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_FST_to_FST: beam=%u mean=%f "
                    "variance=%f scale=%f [%d - %d]\n", i, mean, variance, 
                    ctx->scales[i], i*nsamp, (i+1)*nsamp);
      }
    }

    // use all signals within a channel to calculate the median
    unsigned nwork = ctx->nsig - 1;
    unsigned midsig = (nwork - 1) / 2;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "write_block_FST_to_FST: calculate medians: nwork=%u midsig=%u\n", nwork, midsig);

    // N.B. avoid signal 0 [incoherrent beam]

    // compute the offset and scale for each channel
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      unsigned base = ichan * ctx->nsig;
  
      // get the median offset
      for (isig=1; isig<ctx->nsig; isig++)
        ctx->work[isig-1] = ctx->offsets[base + isig];
      qsort ((void *) ctx->work, nwork, sizeof(float), compare);
      ctx->offset = ctx->work[midsig];

      // get the median scale
      for (isig=1; isig<ctx->nsig; isig++)
        ctx->work[isig-1] = ctx->scales[base + isig];
      qsort ((void *) ctx->work, nwork, sizeof(float), compare);
      ctx->scale = ctx->work[midsig];

      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_FST: ichan=%u ctx->offset=%f ctx->scale=%f\n", ichan, ctx->offset, ctx->scale);

      // Fan beams, beam0 is the incoherrent beam, use an independent scale for that one
      if (ctx->nbeam > 1)
      {
        for (isig=1; isig<ctx->nsig; isig++)
        {
          i = ichan * ctx->nsig + isig;
          ctx->offsets[i] = ctx->offset;
          ctx->scales[i] = ctx->scale;
        }
      }

      if (ctx->nsamps_integrated >= ctx->nsamps_to_integrate && ctx->verbose)
        multilog (log, LOG_INFO, "ichan=%u offset=%e scale=%e\n", ichan, ctx->offset, ctx->scale);
    } 
  }
   
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FST: ipcio_open_block_write()\n");
  ctx->out_block = ipcio_open_block_write(ctx->out_hdu->data_block, &out_block_id);
  if (!ctx->out_block)
  {
    multilog (log, LOG_ERR, "write_block_FST_to_FST: ipcio_open_block_write failed %s\n", strerror(errno));
    return -1;
  }

  // 8 it values from Jenet and Anderson / digifil
  const float digi_sigma = 6;
  const float digi_mean = 127.5;
  const float digi_scale = digi_mean / digi_sigma;
  const float digi_min = 0;
  const float digi_max = 255;

  uint8_t * out = (uint8_t *) ctx->out_block;

  unsigned idx = 0;

  // we need to perform the following operations in the input FP data
  // subtract mean
  // multiply by scale to get to unit variance
  // multply by digi_scale to ideal 8-bit FP variance
  // add digi_mean to get to ideal 8-bit mean

  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    for (isig=0; isig<ctx->nsig; isig++)
    {
      i = ichan * ctx->nsig + isig;

      const float combined_scale = ctx->scales[i] * digi_scale;
      const float mean = ctx->offsets[i];

      for (isamp=0; isamp<nsamp; isamp++)
      {
        int result = ((in[idx] - mean) * combined_scale) + digi_mean + 0.5;

        // clip the results
        if (result < digi_min)
        {
          result = digi_min;
          ctx->nclipped[i]++;
        }
        if (result > digi_max)
        {
          result = digi_max;
          ctx->nclipped[i]++;
        }
        out[idx] = (uint8_t) result;
        idx++;
      }
    }
  }

  uint64_t out_data_size = data_size / ctx->bitrate_factor;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FST close_block_write written=%"PRIu64"\n", out_data_size);
  if (ipcio_close_block_write (ctx->out_hdu->data_block, out_data_size) < 0)
  {
    multilog (log, LOG_ERR, "write_block_FST_to_FST ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FST read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);

  return data_size;
}

// rescale the SFT data from 32-bit floats to 8-bit unsigned integers.
int64_t dbrescaledb_write_block_SFT_to_SFT (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  mopsr_dbrescaledb_t* ctx = (mopsr_dbrescaledb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp  = data_size / (ctx->nchan * ctx->nsig * ctx->ndim * (ctx->nbit_in/8));

  //  assume 32-bit, detected (ndim == 1) data
  float * in = (float *) in_data;
  uint64_t out_block_id, isamp;
  double val;
  unsigned ichan, isig, i;
  float mean, mean_sq, variance;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: nsamp=%lu\n", nsamp);

  // 
  while ((!ctx->global_rescale) && (ctx->nsamps_integrated < ctx->nsamps_to_integrate))
  {
    ctx->nsamps_integrated += nsamp;

    // compute the statistics of each beam and channel
    for (isig=0; isig<ctx->nsig; isig++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        // isigchan
        i = isig * ctx->nchan + ichan;

        uint64_t total_power = 0;
        for (isamp=0; isamp<nsamp; isamp++)
        {
          val = (double) in[i * nsamp + isamp];
          ctx->sums[i] += val;
          ctx->sums_sq[i] += (val * val);
        }

        mean = (float) (ctx->sums[i] / ctx->nsamps_integrated);
        mean_sq = (float) (ctx->sums_sq[i] / ctx->nsamps_integrated);
        variance = mean_sq - mean * mean;

        ctx->offsets[i] = mean;
        if (variance == 0)
          ctx->scales[i] = 1.0;
        else
          ctx->scales[i] = 1.0 / sqrt(variance);

        if (ctx->verbose)
          multilog (log, LOG_INFO, "write_block_SFT_to_SFT: beam=%u chan=%u mean=%f variance=%f scale=%f [%d - %d]\n", isig, ichan, mean, variance, ctx->scales[i], i*nsamp, (i+1)*nsamp);
      }
    }

    // use all signals within a channel to calculate the median
    unsigned nwork = ctx->nsig - 1;
    unsigned midsig = (nwork - 1) / 2;
    if (ctx->verbose)
      multilog (log, LOG_INFO, "write_block_SFT_to_SFT: calculate medians: nwork=%u midval=%u\n", nwork, midsig);

    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      // N.B. avoid signal 0 [incoherrent beam]

      // get the median offset for all signals within this channel
      for (isig=1; isig<ctx->nsig; isig++)
      {
        ctx->work[isig-1] = ctx->offsets[isig * ctx->nchan + ichan];
      }

      qsort ((void *) ctx->work, nwork, sizeof(float), compare);
      ctx->offset = ctx->work[midsig];

      // get the median scale
      for (isig=1; isig<ctx->nsig; isig++)
        ctx->work[isig-1] = ctx->scales[isig * ctx->nchan + ichan];
      qsort ((void *) ctx->work, nwork, sizeof(float), compare);
      ctx->scale = ctx->work[midsig];

      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_SFT_to_SFT: ichan=%u ctx->offset=%e ctx->scale=%e\n", ichan, ctx->offset, ctx->scale);

      // Fan beams, beam0 is the incoherrent beam
      if (ctx->nbeam > 1 && !ctx->indiv_rescale)
      {
        for (isig=1; isig<ctx->nsig; isig++)
        {
          i = isig * ctx->nchan + ichan;
          ctx->offsets[i] = ctx->offset;
          ctx->scales[i] = ctx->scale;
        }
      }

      if (ctx->nsamps_integrated >= ctx->nsamps_to_integrate)
        multilog (log, LOG_INFO, "ichan=%u offset=%e scale=%e\n", ichan, ctx->offset, ctx->scale);
    } 
  }
   
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: ipcio_open_block_write()\n");
  ctx->out_block = ipcio_open_block_write(ctx->out_hdu->data_block, &out_block_id);
  if (!ctx->out_block)
  {
    multilog (log, LOG_ERR, "write_block_SFT_to_SFT: ipcio_open_block_write failed %s\n", strerror(errno));
    return -1;
  }

  const float digi_sigma = 6;
  const float digi_mean = 127.5;
  const float digi_scale = digi_mean / digi_sigma;
  const float digi_min = 0;
  const float digi_max = 255;

  uint8_t * out = (uint8_t *) ctx->out_block;

  uint64_t idx = 0;

  // we need to perform the following operations in the input FP data
  // subtract mean
  // multiply by scale to get to unit variance
  // multply by digi_scale to ideal 8-bit FP variance
  // add digi_mean to get to ideal 8-bit mean

  for (isig=0; isig<ctx->nsig; isig++)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      i = isig * ctx->nchan + ichan;

      const float combined_scale = ctx->scales[i] * digi_scale;
      const float mean = ctx->offsets[i];

      for (isamp=0; isamp<nsamp; isamp++)
      {
        int result = ((in[idx] - mean) * combined_scale) + digi_mean + 0.5;

        // clip the results
        if (result < digi_min)
        {
          result = digi_min;
          ctx->nclipped[i]++;
        }
        if (result > digi_max)
        {
          result = digi_max;
          ctx->nclipped[i]++;
        }
        out[idx] = (uint8_t) result;
        idx++;
      }
    }
  }

  uint64_t out_data_size = data_size / ctx->bitrate_factor;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT close_block_write written=%"PRIu64"\n", out_data_size);
  if (ipcio_close_block_write (ctx->out_hdu->data_block, out_data_size) < 0)
  {
    multilog (log, LOG_ERR, "write_block_SFT_to_SFT ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbrescaledb_t dbrescaledb = DADA_DBREBLOCKDB_INIT;

  mopsr_dbrescaledb_t * ctx = &dbrescaledb;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  ctx->verbose = 0;
  ctx->bitrate_factor = 4;
  ctx->global_offset = 0;
  ctx->global_scale = 0;
  char global_rescale = 0;
  ctx->indiv_rescale = 0;

  float nsecs = 30;

  while ((arg=getopt(argc,argv,"df:io:st:vz")) != -1)
  {
    switch (arg) 
    {
      case 'd':
        daemon = 1;
        break;

      case 'f':
        global_rescale |= 0x01;
        ctx->global_scale = atof(optarg);
        break;

      case 'i':
        ctx->indiv_rescale = 1;
        break;

      case 'o':
        global_rescale |= 0x10;
        ctx->global_offset = atof(optarg);
        break;

      case 's':
        single_transfer = 1;
        break;

      case 't':
        nsecs = atof(optarg);
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

  ctx->global_rescale = (global_rescale == 0x11);

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "mopsr_dbrescaledb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (ctx->verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbrescaledb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (ctx->verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(ctx->out_key)) != 1) {
    fprintf (stderr, "mopsr_dbrescaledb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }



  log = multilog_open ("mopsr_dbrescaledb", 0);

  multilog_add (log, stderr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbrescaledb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbrescaledb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  ctx->out_hdu = dada_hdu_create (log);
  dada_hdu_set_key (ctx->out_hdu, ctx->out_key);
  if (dada_hdu_connect (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }
  ctx->out_block = 0;
  ctx->out_block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) ctx->out_hdu->data_block);
  ctx->seconds_to_integrate = nsecs;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: ctx->out_block_size=%"PRIu64"\n", ctx->out_block_size);
  if (zero_copy && (ctx->out_block_size != (block_size / ctx->bitrate_factor)))
  {
    multilog (log, LOG_ERR, "output block size [%"PRIu64"]  must be input block size [%"PRIu64"] / %u\n", ctx->out_block_size, block_size, ctx->bitrate_factor);
   return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbrescaledb_open;
  client->io_function   = dbrescaledb_write;

  if (zero_copy)
  {
    client->io_block_function = dbrescaledb_write_block_FST_to_FST;
  }
  else
  {
    multilog (log, LOG_ERR, "currently zero copy must be used\n");
    return EXIT_FAILURE;
  }

  client->close_function = dbrescaledb_close;
  client->direction      = dada_client_reader;

  client->context = &dbrescaledb;
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
