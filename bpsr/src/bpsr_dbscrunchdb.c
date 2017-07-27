#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "daemon.h"
#include "bpsr_def.h"

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

int64_t dbscrunchdb_write_block_TFP_to_TF (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbscrunchdb_write_block_TFP_to_TF_scaled (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "bpsr_dbscrdb [options] in_key out_key\n"
           "              Pscrunch and Tscrunch BPSR spectra\n"
           " -c secs      calculate scales over secs, then hold constant\n"
           " -f val       apply val as global scale\n" 
           " -i           individually rescale reach signal\n" 
           " -o val       apply val as global offset to data\n"
           " -s           1 transfer, then exit\n"
           " -t num       Tscrunch num samples\n"
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
  unsigned db_factor;

  // verbose output
  int verbose;

  unsigned int nchan;
  unsigned int ndim; 
  unsigned int npol; 
  unsigned int nbit;

  unsigned tscrunch;

  float * sums;
  float * sums_sq;
  float * work;
  float * offsets;
  float * scales;
  uint64_t * nclipped;

  unsigned quit;

  uint64_t nsamps_integrated;
  uint64_t nsamps_to_integrate;
  float seconds_to_integrate;

  float offset;
  float scale;

  char global_rescale;
  float global_offset;
  float global_scale;

  char indiv_rescale;

} bpsr_dbscrunchdb_t;

#define DADA_DBREBLOCKDB_INIT { 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbscrunchdb_open (dada_client_t* client)
{
  // the bpsr_dbscrunchdb specific data
  bpsr_dbscrunchdb_t* ctx = (bpsr_dbscrunchdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbscrunchdb_open()\n");

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

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  // read the number of channels
  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;                
  }

  // read the number of polarisations 
  if (ascii_header_get (client->header, "NPOL", "%u", &(ctx->npol)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NPOL\n");
    return -1;
  }

  // read the number of bits per sample
  if (ascii_header_get (client->header, "NBIT", "%u", &(ctx->nbit)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NBIT\n");
    return -1;
  }

  // read the number of dimensions of each time sample
  if (ascii_header_get (client->header, "NDIM", "%u", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NDIM\n");
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: INPUT npol=%d nchan=%d nbit=%d ndim=%d\n", ctx->npol, ctx->nchan, ctx->nbit, ctx->ndim);

  if (ctx->scales)
    free (ctx->scales);
  ctx->scales = (float *) malloc (sizeof(float) * ctx->nchan);

  if (ctx->offsets)
    free (ctx->offsets);
  ctx->offsets = (float *) malloc (sizeof(float) * ctx->nchan);

  if (ctx->sums)
    free (ctx->sums);
  ctx->sums = (float *) malloc (sizeof(float) * ctx->nchan);

  if (ctx->sums_sq)
    free (ctx->sums_sq);
  ctx->sums_sq = (float *) malloc (sizeof(float) * ctx->nchan);

  if (ctx->nclipped)
    free (ctx->nclipped);
  ctx->nclipped = (uint64_t *) malloc (sizeof(uint64_t) * ctx->nchan);

  // reduction in ctx->npol
  ctx->bitrate_factor = ctx->npol;

  // if tcrunching, this affects bitrate factor too
  ctx->bitrate_factor *= ctx->tscrunch;

  if (ctx->bitrate_factor != ctx->db_factor)
  {
    multilog (log, LOG_ERR, "open: In/Out data block factor [%u] != scrunch factor [%u]\n", ctx->db_factor, ctx->bitrate_factor);
    return -1;
  }

  char tmp[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", tmp) == 1)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: UTC_START=%s\n", tmp);
  }
  else
  {
    multilog (log, LOG_ERR, "open: header with no UTC_START\n");
    return -1;
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

  float tsamp, new_tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%f", &tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: failed to read TSAMP from header\n");
    return -1;
  }

  float samples_per_second = 1000000.0f / tsamp;

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

  unsigned new_npol = 1;
  if (ascii_header_set (header, "NPOL", "%u", new_npol) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NPOL=%u to header\n", new_npol);
    return -1;
  }

  new_tsamp = tsamp * ctx->tscrunch;
  if (ascii_header_set (header, "TSAMP", "%f", new_tsamp) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write TSAMP=%f to header\n", new_tsamp);
    return -1;
  }

  if (ascii_header_set (header, "STATE", "%s", "Intensity") < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write STATE=Intensity to header\n");
    return -1;
  }

  unsigned ichan;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    ctx->sums[ichan] = 0;
    ctx->sums_sq[ichan] = 0;
    ctx->nclipped[ichan] = 0;
    ctx->offsets[ichan] = ctx->global_offset;
    ctx->scales[ichan] = ctx->global_scale;
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

int dbscrunchdb_close (dada_client_t* client, uint64_t bytes_written)
{
  bpsr_dbscrunchdb_t* ctx = (bpsr_dbscrunchdb_t*) client->context;
  
  multilog_t* log = client->log;

  uint64_t nclipped_total = 0;
  uint64_t namps_total = ctx->bytes_out;

  unsigned ichan, isig, i;
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    nclipped_total += ctx->nclipped[ichan];
    if (ctx->nclipped[ichan] > 0 && ctx->verbose > 1)
      multilog (log, LOG_INFO, "close: samples from chan=%u that exceed power limits=%"PRIu64"\n", ichan, ctx->nclipped[ichan]);
  }

  float percent_clipped = ((float) nclipped_total / (float) namps_total) * 100;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: nclipped total=%lu of %lu (%5.3f)\n", nclipped_total, ctx->bytes_out, percent_clipped);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  // unlock write on the datablock (end the transfer)
  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: dada_hdu_unlock_write\n");

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbscrunchdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->out_key);
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
int64_t dbscrunchdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  bpsr_dbscrunchdb_t* ctx = (bpsr_dbscrunchdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  uint64_t out_data_size = data_size / ctx->bitrate_factor;

  // write dat to all data blocks
  ipcio_write (ctx->out_hdu->data_block, data, out_data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);
 
  return data_size;
}

// pscrunch the data inverting the channel ordering
int64_t dbscrunchdb_write_block_TFP_to_TF_scaled (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  bpsr_dbscrunchdb_t* ctx = (bpsr_dbscrunchdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp  = data_size / (ctx->nchan * ctx->npol * ctx->ndim * (ctx->nbit/8));

  //  assume 8-bit, detected (ndim == 1) data
  uint8_t * in = (uint8_t *) in_data;
  uint64_t out_block_id, isamp;
  float val;
  unsigned ichan, i;
  float mean, mean_sq, variance;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: nsamp=%lu\n", nsamp);

  // 8 bit values from Jenet and Anderson / digifil
  const float digi_sigma = 6;
  const float digi_mean = 128;
  const float digi_scale = digi_mean / digi_sigma;
  const float digi_min = 0;
  const float digi_max = 255;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: ctx->nsamps_integrated=%lu, ctx->nsamps_to_integrate=%lu\n", ctx->nsamps_integrated, ctx->nsamps_to_integrate);

  // if the input power levels of the pscrunch data are to be measured 
  if ((!ctx->global_rescale) && (ctx->nsamps_integrated < ctx->nsamps_to_integrate))
  {
    ctx->nsamps_integrated += nsamp;

    uint64_t idat = 0;
    // compute the mean power level across all beams and channels
    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan+=2)
      {
        val = (float) in[idat] + (float) in[idat+2];
        ctx->sums[ichan] += val;
        ctx->sums_sq[ichan] += (val * val);

        val = (float) in[idat+1] + (float) in[idat + 3];
        ctx->sums[ichan+1] += val;
        ctx->sums_sq[ichan+1] += (val * val);

        idat += (ctx->npol * 2);
      }
    }

    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      mean = (float) (ctx->sums[ichan] / ctx->nsamps_integrated);
      mean_sq = (float) (ctx->sums_sq[ichan] / ctx->nsamps_integrated);
      variance = mean_sq - (mean * mean);

      ctx->offsets[ichan] = mean;
      if (variance == 0)
        ctx->scales[ichan] = 1.0;
      else
        ctx->scales[ichan] = 1.0 / sqrt(variance);

      if (ctx->verbose > 2)
        multilog (log, LOG_INFO, "write_block_TFP_to_TF: channel=%u mean=%f "
                  "variance=%f scale=%f\n", ichan, mean, variance, ctx->scales[ichan]);

      // now adjust the scales for the optimal 8-bit distribution
      ctx->scales[ichan] *= digi_scale;
    }
  }
   
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: ipcio_open_block_write()\n");
  ctx->out_block = ipcio_open_block_write(ctx->out_hdu->data_block, &out_block_id);
  if (!ctx->out_block)
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF: ipcio_open_block_write failed %s\n", strerror(errno));
    return -1;
  }

  // we need to perform the following operations in the input data
  // subtract mean
  // multiply by scale to get to unit variance
  // multply by digi_scale to ideal 8-bit FP variance
  // add digi_mean to get to ideal 8-bit mean

  uint8_t * out = (uint8_t *) ctx->out_block;
  uint64_t idat = 0;

  float pscr, result;

  if (ctx->npol == 4 || ctx->npol == 2)
  {
    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan+=2)
      {
        pscr   = (float) in[idat] + (float) in[idat + 2];
        result = (pscr - ctx->offsets[ichan]) * ctx->scales[ichan] + digi_mean;
        if (result > digi_max)
        {
          result = digi_max;
          ctx->nclipped[ichan]++;
        }
        if (result < digi_min)
        {
          result = digi_min;
          ctx->nclipped[ichan]++;
        }

        out[ichan] = (uint8_t) result;

        pscr   = (float) in[idat + 1] + (float) in[idat + 3];
        result = (pscr - ctx->offsets[ichan+1]) * ctx->scales[ichan+1] + digi_mean;
        if (result > digi_max)
        {
          result = digi_max;
          ctx->nclipped[ichan+1]++;
        }
        if (result < digi_min)
        {
          result = digi_min;
          ctx->nclipped[ichan+1]++;
        }

        // do allow for reverse channel ordering
        out[ichan] = (uint8_t) result;

        idat += (2 * ctx->npol);
      }
      out += ctx->nchan;
    }
  }
  else
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF: npol=%d not supported\n", ctx->npol);
    return -1;
  }

  uint64_t out_data_size = data_size / ctx->bitrate_factor;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF close_block_write written=%"PRIu64"\n", out_data_size);
  if (ipcio_close_block_write (ctx->out_hdu->data_block, out_data_size) < 0)
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);

  return data_size;
}

// pscrunch the data inverting the channel ordering
int64_t dbscrunchdb_write_block_TFP_to_TF (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  bpsr_dbscrunchdb_t* ctx = (bpsr_dbscrunchdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp  = data_size / (ctx->nchan * ctx->npol * ctx->ndim * (ctx->nbit/8));

  //  assume 8-bit, detected (ndim == 1) data
  uint8_t * in = (uint8_t *) in_data;
  uint64_t out_block_id, isamp;
  float val;
  unsigned ichan, i;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: nsamp=%lu\n", nsamp);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF: ipcio_open_block_write()\n");
  ctx->out_block = ipcio_open_block_write(ctx->out_hdu->data_block, &out_block_id);
  if (!ctx->out_block)
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF: ipcio_open_block_write failed %s\n", strerror(errno));
    return -1;
  }

  // the input 8bit data are summed in polarisation and tscrunched and divided by the number of sums
  uint8_t * out = (uint8_t *) ctx->out_block;
  
  unsigned iscr;
  uint64_t idat = 0;
  const float scr_factor = 2 * ctx->tscrunch;

  if (ctx->npol == 4 || ctx->npol == 2)
  {
    // for reversing the channel ordering
    for (isamp=0; isamp<nsamp; isamp += ctx->tscrunch)
    {
      bzero (ctx->sums, sizeof(float) * ctx->nchan);

      // form a P (and optionally T) scrunched spectrum
      for (iscr=0; iscr<ctx->tscrunch; iscr++)
      {
        for (ichan=0; ichan<ctx->nchan; ichan+=2)
        {
          // BPSR data is packed C0P0 C1P0 C0P1 C1P2
          ctx->sums[ichan+0] += (float) in[idat+0] + (float) in[idat + 2];
          ctx->sums[ichan+1] += (float) in[idat+1] + (float) in[idat + 3];
          idat += (2*ctx->npol);
        }
      }

      // reverse the channel ordering
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        float decimated = rintf (ctx->sums[ichan] / scr_factor);
        if (decimated > 255)
        {
          ctx->nclipped[ichan]++;
          decimated = 255;
        }
        out[ichan] = (uint8_t) decimated;
      }
      out += ctx->nchan; 
    }
  }
  else
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF: npol=%d not supported\n", ctx->npol);
    return -1;
  }

  uint64_t out_data_size = data_size / ctx->bitrate_factor;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF close_block_write written=%"PRIu64"\n", out_data_size);
  if (ipcio_close_block_write (ctx->out_hdu->data_block, out_data_size) < 0)
  {
    multilog (log, LOG_ERR, "write_block_TFP_to_TF ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_TFP_to_TF read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  bpsr_dbscrunchdb_t dbscrunchdb = DADA_DBREBLOCKDB_INIT;

  bpsr_dbscrunchdb_t * ctx = &dbscrunchdb;

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
  ctx->bitrate_factor = 1;
  ctx->global_offset = 0;
  ctx->global_scale = 0;
  char global_rescale = 0;
  ctx->indiv_rescale = 0;
  ctx->tscrunch = 1;

  float nsecs = 30;

  while ((arg=getopt(argc,argv,"c:df:io:st:vz")) != -1)
  {
    switch (arg) 
    {
      case 'c':
        nsecs = atof(optarg);
        break;

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
        ctx->tscrunch = atoi(optarg);
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

  if (ctx->verbose)
    fprintf (stderr, "bpsr_dbscrunchdb: global_rescale=%d\n", ctx->global_rescale);

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "bpsr_dbscrunchdb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (ctx->verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "bpsr_dbscrunchdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (ctx->verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(ctx->out_key)) != 1) {
    fprintf (stderr, "bpsr_dbscrunchdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("bpsr_dbscrunchdb", 0);

  multilog_add (log, stderr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "bpsr_dbscrunchdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "bpsr_dbscrunchdb: could not lock read on input data block\n");
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
  if (zero_copy && (block_size % ctx->out_block_size != 0))
  {
    multilog (log, LOG_ERR, "output block size [%"PRIu64"] must be an integer factor of input block size [%"PRIu64"]a\n", ctx->out_block_size, block_size);
   return EXIT_FAILURE;
  }
  ctx->db_factor = block_size / ctx->out_block_size;

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbscrunchdb_open;
  client->io_function   = dbscrunchdb_write;

  if (zero_copy)
  {
    if (ctx->indiv_rescale || ctx->global_scale)
      client->io_block_function = dbscrunchdb_write_block_TFP_to_TF_scaled;
    else
      client->io_block_function = dbscrunchdb_write_block_TFP_to_TF;
  }
  else
  {
    multilog (log, LOG_ERR, "currently zero copy must be used\n");
    return EXIT_FAILURE;
  }

  client->close_function = dbscrunchdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbscrunchdb;
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
