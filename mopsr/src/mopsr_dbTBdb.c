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
#include <complex.h>
#include <float.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <xmmintrin.h>
#include <mmintrin.h>

int quit_threads = 0;

void control_thread (void *);

int64_t dbTBdb_write_block (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbTBdb_write_block_ST_to_T (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbTBdb [options] in_key out_key\n"
           " -s        1 transfer, then exit\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n"
           " in_key    DADA key for input data block\n"
           " out_key   DADA key for output data block\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbTBdb_hdu_t;


typedef struct {

  mopsr_dbTBdb_hdu_t output;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  unsigned int nant;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit;

  unsigned quit;

  unsigned control_port;

  char order[4];

  int8_t in_buf[1024];
  float ou_buf[1024];

} mopsr_dbTBdb_t;

#define DADA_DBSUMDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

// unpacked 8x8-bit values to 8x32bit values vectorized
/*
static inline __m256 unpack_8samples (__m64 packed)
{
  __m256 result;

  __m128 unpacked = _mm_cvtpi8_ps (packed);

    // unpack lower 32-bits of 64-bit value
    *dest = _mm_cvtpi8_ps (packed);

    // shift upper 32-bits to lower 32-bits
    _mm_srli_pi64 (packed, 32);

    // increment dest by 4 floats (128-bits)
    dest ++;

    // unpack the lower 32-bits
    *dest = _mm_cvtpi8_ps (packed);

    // incrememt src by 64 bits (8x-bit values)
    src++;
*/

/*! Function that opens the data transfer target */
int dbTBdb_open (dada_client_t* client)
{
  // the mopsr_dbTBdb specific data
  mopsr_dbTBdb_t* ctx = (mopsr_dbTBdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbTBdb_open()\n");

  char output_order[4];

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", ctx->output.key);

  if (dada_hdu_lock_write (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

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

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    // for summing of data blocks we always want TF output mode
    multilog (log, LOG_INFO, "open: ORDER=%s\n", ctx->order);
    if ((strcmp(ctx->order, "ST") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from ST to T\n");
      client->io_block_function = dbTBdb_write_block_ST_to_T;
      strcpy (output_order, "T");
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
  uint64_t new_bytes_per_second = 4 * (bytes_per_second / ctx->nant);

  uint64_t resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no RESOLUTION\n");
    return -1;
  }
  uint64_t new_resolution = 4 * (resolution / ctx->nant);
  // actually new resolution should be == NDIM now!
  new_resolution = 4 * ctx->ndim;

  uint64_t file_size;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {     
    multilog (log, LOG_ERR, "open: header with no FILE_SIZE\n");
    return -1;    
  }                 
  uint64_t new_file_size = 4 * (file_size / ctx->nant);

  uint64_t obs_offset;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {     
    multilog (log, LOG_ERR, "open: header with no OBS_OFFSET\n");
    return -1;    
  }
  uint64_t new_obs_offset = 4 * (obs_offset / ctx->nant);

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  // setup header for output HDU
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

  // now set each output data block to 1 antenna
  int nant = 1;
  if (ascii_header_set (header, "NANT", "%d", nant) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NANT=%d to header\n",
                             nant);
    return -1;
  }
  
  int nbit = 32;
  if (ascii_header_set (header, "NBIT", "%d", nbit) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NBIT=%d to header\n",
                             nbit);
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
  if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, new_obs_offset) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new OBS_OFFSET to header\n");
    return -1;
  }
  if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new FILE_SIZE to header\n");
    return -1;
  }

  if (ascii_header_set (header, "ORDER", "%s", output_order) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write ORDER=%s to header\n", output_order);
    return -1;
  }

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->output.hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }
  if (ctx->verbose) 
    multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", ctx->output.key);

  fprintf (stderr, "transfer_size=%"PRIu64"\n", transfer_size);

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbTBdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbTBdb_t* ctx = (mopsr_dbTBdb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbTBdb_hdu_t * o = 0;

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
      multilog (log, LOG_ERR, "dbTBdb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbTBdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbTBdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbTBdb_t* ctx = (mopsr_dbTBdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write (ctx->output.hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}

int64_t dbTBdb_write_block_ST_to_T (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbTBdb_t* ctx = (mopsr_dbTBdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = 4 * (in_data_size / ctx->nant);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: in_data_size=%"PRIu64", "
              "out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  void * out;
  
  uint64_t out_block_id;
  unsigned iant;

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] ipcio_open_block_write()\n", ctx->output.key);
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    ctx->output.bytes_written = 0;
    out = ctx->output.curr_block;
  }
  else
    out = ctx->output.curr_block + ctx->output.bytes_written;

#define USE_SSE 1

#ifdef USE_SSE
  float * dest = (float *) out;

  __m128 packed;
  __m64 * parts = (__m64 *) &packed; 
  __m128 sum0, sum1, sum2, sum3;
  __m128 unpacked;

  // each vectorized operation will unpack 8 data points
  const uint64_t nops = in_data_size / (ctx->nant * 16);
  uint64_t iop;
  unsigned ant_stride_float = nops * 4;

  float * in_op = (float *) in_data;
  float * src;

  const unsigned nval_per_ant = nops * 16;
  int8_t * in = (int8_t *) in_data;

  // simply unpack iant 0 directly to the output
  for (iop=0; iop<nops; iop++)
  {
    src = in_op;

    // load 8-bit packed data to register
    packed = _mm_loadu_ps (src);
    parts = (__m64 *) &packed; 

    // unpack each 32-bit segment into 128-bit vectors
    sum0 = _mm_cvtpi8_ps (parts[0]);
    sum2 = _mm_cvtpi8_ps (parts[1]);
    packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
    sum1 = _mm_cvtpi8_ps (parts[0]);
    sum3 = _mm_cvtpi8_ps (parts[1]);

    src += ant_stride_float;

    for (iant=1; iant<ctx->nant; iant++)
    {
      packed = _mm_loadu_ps (src);
      parts = (__m64 *) &packed; 

      sum0 = _mm_add_ps(sum0, _mm_cvtpi8_ps (parts[0]));
      sum2 = _mm_add_ps(sum2,_mm_cvtpi8_ps (parts[1]));
      packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
      sum1 = _mm_add_ps(sum1,_mm_cvtpi8_ps (parts[0]));
      sum3 = _mm_add_ps(sum3,_mm_cvtpi8_ps (parts[1]));

      src += ant_stride_float;
    }

    _mm_storeu_ps (dest, sum0);
    _mm_storeu_ps (dest+4, sum1);
    _mm_storeu_ps (dest+8, sum2);
    _mm_storeu_ps (dest+12, sum3);

    // increment output by 16 floats
    dest += 16;
    in_op += 4;
  }

#else

  const unsigned ndim = 2;
  const unsigned nsamp = (unsigned) (in_data_size / (ndim * ctx->nant));

  const unsigned max_bytes = 4096;
  const unsigned chunk_nsamps = max_bytes / (ndim * sizeof(float)); // read 512 samples at a time (1024 bytes)
  const unsigned chunk_nvals = chunk_nsamps * ndim;
  const unsigned nchunk = nsamp / chunk_nsamps;

  unsigned ichunk, ival;
  //const complex float phase = 1 + 2 * I;

#ifdef SSE_CHUNK

  float * dest = (float *) out;

  __m128 packed;
  __m64 * parts = (__m64 *) &packed;
  __m128 sum0, sum1, sum2, sum3;
  __m128 unpacked;

  // each vectorized operation will unpack 8 data points
  const unsigned nops = chunk_nvals / (ctx->nant * 16);
  unsigned iop;
  unsigned ant_stride_float = nops * 4;

  float * in_op = (float *) in_data;
  float * src;

  const unsigned nval_per_ant = nops * 16;
  int8_t * in = (int8_t *) in_data;

#endif


  for (ichunk=0; ichunk<nchunk; ichunk++)
  {
    int8_t * indat = ((int8_t *) in_data) + (ichunk * chunk_nsamps * ndim);

    for (iant=0; iant<ctx->nant; iant++)
    {
      memcpy ((void *) ctx->in_buf, indat, chunk_nvals);

#if SSE_CHUNK

      in_op = (float *) in_data

      // simply unpack iant 0 directly to the output
      unsigned iop;
      for (iop=0; iop<nops; iop++)
      {
        src = in_op;

        // load 8-bit packed data to register
        packed = _mm_loadu_ps (src);
        //parts = (__m64 *) &packed;

        if (iant == 0)
        {
          sum0 = _mm_setzero_ps();
          sum1 = _mm_setzero_ps();
          sum2 = _mm_setzero_ps();
          sum3 = _mm_setzero_ps();
        }
        else
        {
          sum0 = _mm_load_ps (dest);
          sum1 = _mm_load_ps (dest+4);
          sum2 = _mm_load_ps (dest+8);
          sum3 = _mm_load_ps (dest+12);
        }

        sum0 = _mm_add_ps(sum0, _mm_cvtpi8_ps (parts[0]));
        _mm_storeu_ps (dest, sum0);

        sum2 = _mm_add_ps(sum2,_mm_cvtpi8_ps (parts[1]));
        _mm_storeu_ps (dest+8, sum2);

        packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);

        sum1 = _mm_add_ps(sum1,_mm_cvtpi8_ps (parts[0]));
        _mm_storeu_ps (dest+4, sum1);

        sum3 = _mm_add_ps(sum3,_mm_cvtpi8_ps (parts[1]));
        _mm_storeu_ps (dest+12, sum3);

        // unpack each 32-bit segment into 128-bit vectors
        sum0 = _mm_cvtpi8_ps (parts[0]);
        sum2 = _mm_cvtpi8_ps (parts[1]);
        packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
        sum1 = _mm_cvtpi8_ps (parts[0]);
        sum3 = _mm_cvtpi8_ps (parts[1]);

        src += ant_stride_float;

        for (iant=1; iant<ctx->nant; iant++)
        {
          packed = _mm_loadu_ps (src);
          parts = (__m64 *) &packed;

          sum0 = _mm_add_ps(sum0, _mm_cvtpi8_ps (parts[0]));
          sum2 = _mm_add_ps(sum2,_mm_cvtpi8_ps (parts[1]));
          packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
          sum1 = _mm_add_ps(sum1,_mm_cvtpi8_ps (parts[0]));
          sum3 = _mm_add_ps(sum3,_mm_cvtpi8_ps (parts[1]));

          src += ant_stride_float;
        }

        _mm_storeu_ps (dest, sum0);
        _mm_storeu_ps (dest+4, sum1);
        _mm_storeu_ps (dest+8, sum2);
        _mm_storeu_ps (dest+12, sum3);

        // increment output by 16 floats
        dest += 16;
        in_op += 4;
      }

#else

      if (iant == 0)
        for (ival=0; ival<chunk_nvals; ival++)
          ctx->ou_buf[ival] = (float) ctx->in_buf[ival];
      else
        for (ival=0; ival<chunk_nvals; ival++)
          ctx->ou_buf[ival] += (float) ctx->in_buf[ival];
#endif

      /*
      for (ival=0; ival<chunk_nsamps; ival++)
      {
        //complex float val = ((float) ctx->in_buf[2*ival]) + ((float) ctx->in_buf[2*ival+1]) * I;
        //ctx->ou_buf[ival] += val * phase;
        ctx->ou_buf[ival] = ((float) ctx->in_buf[2*ival]) + ((float) ctx->in_buf[2*ival+1]) * I;
      }
      */
      indat += (nsamp * ndim);
    }

    memcpy (out, (void *) ctx->ou_buf, chunk_nvals * sizeof(float));
    out += chunk_nvals * sizeof(float);
  }
#endif

  ctx->output.bytes_written += out_data_size;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_update_block_write failed\n", ctx->output.key);
         return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_STF_to_TF [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_STF_to_TF [%x] ipcio_close_block_write failed\n", ctx->output.key);
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
    multilog (log, LOG_INFO, "write_block_STF_to_TF read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbTBdb_t dbTBdb = DADA_DBSUMDB_INIT;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  pthread_t control_thread_id;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dp:svz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'p':
        if (optarg)
        {
          dbTBdb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "mopsr_dbTBdb: -p requires argument\n");
          usage();
          return EXIT_FAILURE;
        }

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

  dbTBdb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "mopsr_dbTBdb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbTBdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbTBdb.output.key)) != 1) {
    fprintf (stderr, "mopsr_dbTBdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbTBdb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbTBdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbTBdb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  dbTBdb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbTBdb.output.hdu, dbTBdb.output.key);
  if (dada_hdu_connect (dbTBdb.output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbTBdb.output.key);
    return -1;
  }
  dbTBdb.output.curr_block = 0;
  dbTBdb.output.bytes_written = 0;
  dbTBdb.output.block_open = 0;
  dbTBdb.output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbTBdb.output.hdu->data_block);

  // we cannot strictly test this until we receive the header and get the number of antenna
  // it must at least be a factor of the input block size (
  if (verbose)
    multilog (log, LOG_INFO, "main: dbTBdb.output.block_size=%"PRIu64"\n", dbTBdb.output.block_size);
  if (zero_copy && (block_size % dbTBdb.output.block_size != 0))
  {
    multilog (log, LOG_ERR, "for zero copy, all DADA buffer block sizes must "
                            "be matching\n");
   return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbTBdb_open;
  client->io_function   = dbTBdb_write;

  if (zero_copy)
  {
    client->io_block_function = dbTBdb_write_block_ST_to_T;
  }

  client->close_function = dbTBdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbTBdb;
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

    if (single_transfer || dbTBdb.quit)
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
