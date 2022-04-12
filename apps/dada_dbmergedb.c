#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "tmutil.h"

#include <pthread.h>
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

typedef struct
{
  time_t  tv_sec;
  uint64_t tv_psec;
} dada_timeval_t;

typedef struct dada_block_t dada_block_t;
struct dada_block_t
{
  dada_hdu_t *    hdu;
  key_t           key;
  dada_timeval_t  utc_start;
  uint64_t        bufsz;
  uint64_t        bytes;
  uint64_t        block_id;
  char            block_open;
  char            error;
  char *          header;
  char *          curr_buf;
  float           max_percent_full;
  dada_block_t *  other_pol;
  pthread_mutex_t * mutex;
};

typedef struct {

  dada_block_t pola;
  dada_block_t polb;
  dada_block_t output;
  multilog_t * log;
  uint64_t resolution;
  int verbose;
  unsigned quit;
  char wait_for_both;
  pthread_t pola_thread_id;
  pthread_t polb_thread_id;
  pthread_mutex_t   mutex;

} dada_dbmergedb_t;

int64_t sample_difference (dada_timeval_t time1, dada_timeval_t time2, uint64_t tsamp_pico);
void add_to_dadatime (dada_timeval_t * time1, uint64_t psecs);

int dbmergedb_init_hdu (dada_dbmergedb_t * ctx, dada_block_t * hdu, char reader);
int dbmergedb_init (dada_dbmergedb_t * ctx);
void dbmergedb_destroy (dada_dbmergedb_t * ctx);

void usage()
{
  fprintf (stdout,
           "dada_dbmergedb [options] pola_key polb_key out_key\n"
           " -s        1 transfer, then exit\n"
           " -w        wait for both pols before aligning\n"
           " -v        verbose mode\n"
           " pola_key  DADA key for polarisation A\n"
           " polb_key  DADA key for polarisation B\n"
           " out_key   DADA key for combined output \n");
}

// return the difference in dada_timeval_t's in seconds
int64_t sample_difference (dada_timeval_t time1, dada_timeval_t time2, uint64_t tsamp_pico)
{
  // compute the difference in samples between time1 and time2
  int64_t diff_secs = (int64_t) time2.tv_sec - (int64_t) time1.tv_sec;
  int64_t diff_psecs = (int64_t) time2.tv_psec - (int64_t) time1.tv_psec;

  int64_t diff_samps = (diff_psecs + (diff_secs * 1e12)) / tsamp_pico;

  return diff_samps;
}

void add_to_dadatime (dada_timeval_t * time1, uint64_t psecs)
{
  // add to fractional component
  time1->tv_psec += psecs;

  unsigned seconds_to_add = (unsigned) (time1->tv_psec / 1e12);

  time1->tv_sec += seconds_to_add;
  time1->tv_psec -= (seconds_to_add * 1e12);
}
                      
int dbmergedb_init_hdu (dada_dbmergedb_t * ctx, dada_block_t * db, char reader)
{
  db->hdu = dada_hdu_create (ctx->log);
  dada_hdu_set_key (db->hdu, db->key);
  if (dada_hdu_connect (db->hdu) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not connect to data block with key %x\n", db->key);
    return -1;
  }

  if (reader)
  {
    if (dada_hdu_lock_read (db->hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "could not lock read on hdu %x\n", db->key);
      return -1;
    }
  }
  else
  {
    if (dada_hdu_lock_write (db->hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "could not lock write on hdu %x\n", db->key);
      return -1;
    }
  }

  db->bufsz = ipcbuf_get_bufsz((ipcbuf_t *) (db->hdu->data_block));
  db->block_open = 0;
  db->bytes = 0;
  db->curr_buf = 0;
  db->header = 0;
  db->mutex = &(ctx->mutex);
  if (ctx->wait_for_both)
    db->max_percent_full = 1.1;
  else
    db->max_percent_full = 0.5;

  db->utc_start.tv_sec = 0;
  db->utc_start.tv_psec = 0;

  return 0;
}

int dbmergedb_init (dada_dbmergedb_t * ctx)
{
  if (dbmergedb_init_hdu (ctx, &(ctx->pola), 1) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not initialize HDU for pol A\n");
    return -1;
  }

  if (dbmergedb_init_hdu (ctx, &(ctx->polb), 1) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not initialize HDU for pol B\n");
    return -1;
  }

  if (dbmergedb_init_hdu (ctx, &(ctx->output), 0) < 0)
  {
    multilog (ctx->log, LOG_ERR, "could not initialize HDU for output\n");
    return -1;
  }

  ctx->pola.other_pol = &(ctx->polb);
  ctx->polb.other_pol = &(ctx->pola);
  ctx->output.other_pol = 0;

  // check that the input block sizes are half that of the output block size
  if (ctx->pola.bufsz + ctx->polb.bufsz != ctx->output.bufsz)
  {
    multilog (ctx->log, LOG_ERR, "pola [%lu] + polb [%lu] bufsz did not equal output [%lu] bufsz\n", ctx->pola.bufsz, ctx->polb.bufsz, ctx->output.bufsz);
    return -1;
  }

  multilog (ctx->log, LOG_INFO, "init: pthread_mutex_init()\n");
  pthread_mutex_init(&(ctx->mutex), NULL);

  multilog (ctx->log, LOG_INFO, "init: complete\n");
  return 0;
}

void dbmergedb_destroy (dada_dbmergedb_t * ctx)
{
  pthread_mutex_lock (&(ctx->mutex));
  pthread_mutex_unlock (&(ctx->mutex));
  pthread_mutex_destroy (&(ctx->mutex));
}

/*! Function that opens the data transfer target */
int dbmergedb_open (dada_client_t* client)
{
  // the dada_dbmergedb specific data
  dada_dbmergedb_t* ctx = (dada_dbmergedb_t *) client->context;

  // wait for both input polarisations to be aligned onto a UTC
  // second boundary, join input threads first
  void* result = 0;
  pthread_join (ctx->pola_thread_id, &result);
  pthread_join (ctx->polb_thread_id, &result);

  // copy the header form pol A to the output header block
  memcpy (client->header, ctx->pola.header, client->header_size);

  // get the aligned timestamp
  double tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%lf", &tsamp) != 1)
  {
    multilog (ctx->log, LOG_ERR, "Header did not contain TSAMP\n");
    return -1;
  }
  uint64_t tsamp_psec = (uint64_t) (tsamp * 1e6);
  // the two polarisations are not aligned to within half a sample, bork
  if (sample_difference (ctx->pola.utc_start, ctx->polb.utc_start, tsamp_psec) != 0)
  {
    multilog (ctx->log, LOG_ERR, "Failed to align polarisations\n");
    return -1;
  }

  // determine the new UTC_START for the output
  char utc_start[20];
  strftime (utc_start, 20, DADA_TIMESTR, gmtime(&(ctx->pola.utc_start.tv_sec)));
  if (ascii_header_set (client->header, "UTC_START", "%s", utc_start) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write UTC_START=%s to header\n", utc_start);

  double micro_seconds = ((double) ctx->pola.utc_start.tv_psec) / 1e6;
  if (ascii_header_set (client->header, "MICROSECONDS", "%lf", micro_seconds) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write MICROSECONDS=%lf to header\n", micro_seconds);

  uint64_t obs_offset = 0;
  if (ascii_header_set (client->header, "OBS_OFFSET", "%"PRIu64, obs_offset) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write OBS_OFFSET=%"PRIu64" to header\n", obs_offset);

  unsigned npol = 2;
  if (ascii_header_set (client->header, "NPOL", "%u", npol) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write NPOL=%u to header\n", npol);

  uint64_t bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
    multilog (ctx->log, LOG_WARNING, "Could not read BYTES_PER_SECOND from header\n");
  bytes_per_second *= npol;
  if (ascii_header_set (client->header, "BYTES_PER_SECOND", "%"PRIu64, bytes_per_second) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write BYTES_PER_SECOND=%"PRIu64" to header\n", bytes_per_second);

  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &(ctx->resolution)) != 1)
    multilog (ctx->log, LOG_WARNING, "Could not read RESOLUTION from header\n");
  ctx->resolution *= npol;
  if (ascii_header_set (client->header, "RESOLUTION", "%"PRIu64, ctx->resolution) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write RESOLUTION=%"PRIu64" to header\n", ctx->resolution);

  //client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  client->header_transfer = 0;

  return 0;
}

int dbmergedb_close (dada_client_t* client, uint64_t bytes_written)
{
  dada_dbmergedb_t * ctx = (dada_dbmergedb_t*) client->context;
  
  multilog_t* log = client->log;

  // close the input
  if (ctx->pola.block_open)
  {
    if (ipcio_close_block_read (ctx->pola.hdu->data_block, ctx->pola.bufsz) < 0)
      multilog (log, LOG_ERR, "close: failed to close block for polA\n");
    ctx->pola.block_open = 0;
  }

  if (ctx->polb.block_open)
  {
    if (ipcio_close_block_read (ctx->polb.hdu->data_block, ctx->polb.bufsz) < 0)
      multilog (log, LOG_ERR, "close: failed to close block for polB\n");
    ctx->polb.block_open = 0;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbmergedb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  multilog_t * log = client->log;

  multilog (log, LOG_ERR, "write: should not be called\n");
  return -1;
}

int64_t dbmergedb_write_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  dada_dbmergedb_t* ctx = (dada_dbmergedb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: data_size=%"PRIu64", block_id=%"PRIu64"\n",
                data_size, block_id);

  // the number of output chunks
  const uint64_t nchunk = data_size / ctx->resolution;
  const size_t pol_chunk_size = ctx->resolution / 2;
  uint64_t ochunk;

  char * out = (char *) data;
  uint64_t out_off = 0;

  if (data_size % ctx->resolution)
  {
    multilog (log, LOG_INFO, "write_block: block size was not a multiple of resolution\n");
    return -1;
  }

  char drain_pola = 0;
  char drain_polb = 0;

  for (ochunk=0; ochunk<nchunk; ochunk++)
  {
    if (!ctx->pola.block_open)
    {
      if (ipcbuf_eod((ipcbuf_t*) ctx->pola.hdu->data_block))
      {
        multilog (log, LOG_INFO, "write_block: EOD from pol A\n");
        drain_polb = 1;
      }
      else
      {
        ctx->pola.curr_buf = ipcio_open_block_read (ctx->pola.hdu->data_block, 
                                                    &(ctx->pola.bytes),
                                                    &(ctx->pola.block_id));
        ctx->pola.block_open = 1;
        if (ctx->pola.bytes != ctx->pola.bufsz)
        {
          multilog (log, LOG_INFO, "write_block: EOD on polA\n");
          drain_pola = drain_polb = 1;
        }
        else
          ctx->pola.bytes = 0;
      }
    }
     
    if (!ctx->polb.block_open)
    {
      if (ipcbuf_eod((ipcbuf_t*) ctx->polb.hdu->data_block))
      {
        multilog (log, LOG_INFO, "write_block: EOD from pol B\n");
        drain_pola = 1;
      }
      else
      {
        ctx->polb.curr_buf = ipcio_open_block_read (ctx->polb.hdu->data_block, 
                                                    &(ctx->polb.bytes),
                                                    &(ctx->polb.block_id));
        ctx->polb.block_open = 1;
        if (ctx->polb.bytes != ctx->polb.bufsz)
        {
          multilog (log, LOG_INFO, "write_block: EOD on polB\n");
          drain_pola = drain_polb = 1;
        }
        else
          ctx->polb.bytes = 0;
      }
    }

    if (drain_pola)
    {
      multilog (log, LOG_INFO, "write_block: draining pol A\n");
      int rval = 0;
      if (ctx->pola.block_open)
      {
        if (ctx->pola.bytes == 0)
          ctx->pola.bytes = ctx->pola.bufsz;
        rval = ipcio_close_block_read (ctx->pola.hdu->data_block, ctx->pola.bytes);
      }
      while (! ipcbuf_eod((ipcbuf_t*) ctx->pola.hdu->data_block) && rval == 0)
      {
        ipcio_open_block_read (ctx->pola.hdu->data_block,
                               &(ctx->pola.bytes),
                               &(ctx->pola.block_id));
        rval = ipcio_close_block_read (ctx->pola.hdu->data_block, ctx->pola.bytes);
      }
      ctx->pola.block_open = 0;
    }

    if (drain_polb)
    {
      multilog (log, LOG_INFO, "write_block: draining pol B\n");
      int rval = 0;
      if (ctx->polb.block_open)
      {
        if (ctx->polb.bytes == 0)
          ctx->polb.bytes = ctx->polb.bufsz;
        rval = ipcio_close_block_read (ctx->polb.hdu->data_block, ctx->polb.bytes);
      }
      while (! ipcbuf_eod((ipcbuf_t*) ctx->polb.hdu->data_block) && rval == 0)
      {
        ipcio_open_block_read (ctx->polb.hdu->data_block,
                               &(ctx->polb.bytes),
                               &(ctx->polb.block_id));
        rval = ipcio_close_block_read (ctx->polb.hdu->data_block, ctx->polb.bytes);
      }
      ctx->polb.block_open = 0;
    }

    if (drain_pola || drain_polb)
    {
      multilog (log, LOG_INFO, "write_block: returning 0 bytes, obs ending\n");
      return 0;
    }

    memcpy (out + out_off, ctx->pola.curr_buf + ctx->pola.bytes, pol_chunk_size);
    out_off += pol_chunk_size;
    ctx->pola.bytes += pol_chunk_size;

    memcpy (out + out_off, ctx->polb.curr_buf + ctx->polb.bytes, pol_chunk_size);
    out_off += pol_chunk_size;
    ctx->polb.bytes += pol_chunk_size;

    if (ctx->pola.bytes >= ctx->pola.bufsz)
    {
      ipcio_close_block_read (ctx->pola.hdu->data_block, ctx->pola.bufsz);
      ctx->pola.block_open = 0;
    } 

    if (ctx->polb.bytes >= ctx->polb.bufsz)
    {
      ipcio_close_block_read (ctx->polb.hdu->data_block, ctx->polb.bufsz);
      ctx->polb.block_open = 0;
    }
  }

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}


/*
 * Thread to align the input polarisation to the same UTC second as the other
 * polarisation
 */
void input_thread (void * arg)
{
  dada_block_t * db = (dada_block_t*) arg;

  fprintf (stderr, "input[%x]: STARTING\n", db->key);
  // the input data block is locked for reading
  uint64_t header_size;

  // wait for the header to appear
  char * header = ipcbuf_get_next_read (db->hdu->header_block, &header_size);
  if (!header) 
  {
    fprintf (stderr, "input[%x]: could not get next header block\n", db->key);
    db->error = 1;
    return;
  }
  
  fprintf (stderr, "input[%x]: get_next_read returned\n", db->key);

  // allocate aligned memory for the header
  if (posix_memalign ( (void **) &(db->header), 512, header_size) < 0)
  {
    fprintf (stderr, "input[%x]: failed to allocate %lu bytes for header\n", db->key, header_size);
    db->error = 1;
    return ;
  }

  // make a copy of the ascii header into our header buffer
  memcpy (db->header, header, header_size);
  fprintf (stderr, "input[%x]: header copied\n", db->key);

  // mark the header as cleared
  ipcbuf_mark_cleared (db->hdu->header_block);

  fprintf (stderr, "input[%x]: header mark cleared\n", db->key);
  // read the key parameters for this header to work out the epoch of
  // the data flow
  char buffer[20];
  if (ascii_header_get (header, "UTC_START", "%s", buffer) != 1) 
  {
    fprintf (stderr, "input[%x]: header did not contain UTC_START\n", db->key);
    db->error = 1;
    return ;
  }
  else
    db->utc_start.tv_sec = str2utctime(buffer);

  double micro_seconds;
  if (ascii_header_get (header, "MICROSECONDS", "%lf", &micro_seconds) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain MICROSECONDS\n", db->key);
    db->error = 1;
    return;
  }
  db->utc_start.tv_psec = (uint64_t) (micro_seconds * 1e6);

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%lu", &obs_offset) != 1)
  {     
    fprintf (stderr, "input[%x]: header did not contain OBS_OFFSET\n", db->key);
    db->error = 1;
    return;        
  }
  // TODO fix this with chris
  obs_offset -= 2147483648;

  uint64_t bytes_per_second;
  if (ascii_header_get (header, "BYTES_PER_SECOND", "%lu", &bytes_per_second) != 1)
  { 
    fprintf (stderr, "input[%x]: header did not contain BYTES_PER_SECOND\n", db->key);
    db->error = 1;
    return;
  }
  // TODO remove and remind chris to fix this
  bytes_per_second = 800000000;

  unsigned nchan, ndim, nbit;
  if (ascii_header_get (header, "NCHAN", "%u", &nchan) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain NCHAN\n", db->key);
    db->error = 1;
    return;
  }

  if (ascii_header_get (header, "NDIM", "%u", &ndim) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain NDIM\n", db->key);
    db->error = 1;
    return;
  }

  if (ascii_header_get (header, "NBIT", "%u", &nbit) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain NBIT\n", db->key);
    db->error = 1;
    return;
  }
  unsigned bytes_per_sample = nchan * ndim * nbit / 8;

  uint64_t resolution;
  if (ascii_header_get (header, "RESOLUTION", "%lu", &resolution) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain RESOLUTION\n", db->key);
    db->error = 1;
    return;
  }

  double tsamp;
  if (ascii_header_get (header, "TSAMP", "%lf", &tsamp) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain TSAMP\n", db->key);
    db->error = 1;
    return;
  }
  uint64_t tsamp_psec = (uint64_t) (tsamp * 1e6);

  uint64_t psecs_obs_offset = (obs_offset / bytes_per_sample) * tsamp_psec;
  fprintf (stderr, "input[%x]: adding %lu samples offset to utc_start\n", db->key, (obs_offset / bytes_per_sample));

  add_to_dadatime (&(db->utc_start), psecs_obs_offset);


  const uint64_t psecs_per_buffer = ((uint64_t) db->bufsz * 1e12) / bytes_per_second;
  char both_aligned = 0;

  fprintf (stderr, "input[%x]: aligning\n", db->key);
  while (!both_aligned)
  {
    // determine the next second on which this input CAN start 
    pthread_mutex_lock (db->mutex);

    // if the other thread has not yet started, we must wait for it so 
    // that we which know which thread needs to delay itself
    if (db->other_pol->utc_start.tv_sec == 0)
    {
      fprintf (stderr, "input[%x]: other pol not yet started\n", db->key);
      if (ipcio_percent_full (db->hdu->data_block) > db->max_percent_full)
      {
        db->curr_buf = ipcio_open_block_read (db->hdu->data_block, &(db->bytes), &(db->block_id));
        fprintf (stderr, "input[%x]: block %5.2f full [limit=%5.2f], clearing block %lu\n", db->key,
                 ipcio_percent_full (db->hdu->data_block), db->max_percent_full, db->block_id);
        ipcio_close_block_read (db->hdu->data_block, db->bufsz);
        fprintf (stderr, "input[%x]: adding %lu psec to utc_start\n", db->key, psecs_per_buffer);
        add_to_dadatime (&(db->utc_start), psecs_per_buffer);
      }
      pthread_mutex_unlock (db->mutex);
      usleep(1000000);
    }
    // both threads have started, check the time difference between them
    else
    {
      // difference = other_start - this_start. If +ve this thread started first and
      // must consume some data to align with other pol
      int64_t nsamp_diff = sample_difference (db->utc_start, db->other_pol->utc_start, tsamp_psec); 

      fprintf (stderr, "input[%x]: other pol started, sample diff=%ld\n", db->key, nsamp_diff);

      if (nsamp_diff == 0)
      {
        both_aligned = 1;
        pthread_mutex_unlock (db->mutex);
      }
      // if this thread started before the other, wait for the other thread
      else if (nsamp_diff > 0)
      {
        fprintf (stderr, "input[%x]: THIS= %ld %lu\n", db->key, db->utc_start.tv_sec, db->utc_start.tv_psec);
        fprintf (stderr, "input[%x]: THAT= %ld %lu\n", db->key, db->other_pol->utc_start.tv_sec, db->other_pol->utc_start.tv_psec);
        fprintf (stderr, "input[%x]: nsamp_diff=%lu\n", db->key, nsamp_diff);

        // round the difference to the nearest integer byte
        uint64_t bytes_to_read = (uint64_t) nsamp_diff * bytes_per_sample;
        double   time_offset = (double) bytes_to_read / (double) bytes_per_second;

        fprintf (stderr, "input[%x]: bytes_to_read=%lu time_offset=%lf\n", db->key, bytes_to_read, time_offset);

        if (bytes_to_read % resolution != 0)
        {
          fprintf (stderr, "input[%x]: difference between polA and polB was not an "
                           "integer multiple of RESOLUTION\n", db->key);
          db->error = 1;
          return;
        }

        // discard all data up to the first startable second
        while (bytes_to_read > 0)
        {
          if (!db->block_open)
          {
            if (ipcbuf_eod((ipcbuf_t*) db->hdu->data_block))
            {
              fprintf (stderr, "input[%x]: EOD occured prior to alignment\n", db->key);
              db->error = 1;
              return;
            }

            db->curr_buf = ipcio_open_block_read (db->hdu->data_block, &(db->bytes), &(db->block_id));
            if (db->curr_buf == NULL)
            {
              fprintf (stderr, "input[%x]: failed to open block\n", db->key);
              db->error = 1;
              return;
            }

            assert (db->bytes == db->bufsz);
            db->block_open = 1;
          }

          // if all the bytes in this buffer will be discarded
          if (db->bytes < bytes_to_read)
          {
            bytes_to_read -= db->bytes;
            ipcio_close_block_read (db->hdu->data_block, db->bufsz);
            db->block_open = 0;
            db->bytes = 0;
          }
          // discard only part of this buffer
          else
          {
            db->bytes = db->bufsz - bytes_to_read;
            bytes_to_read = 0;
          }
        }

        uint64_t psec_to_add = nsamp_diff * tsamp_psec;
        fprintf (stderr, "input[%x]: adding %lu psecs to utc_start\n", db->key, psec_to_add);
        add_to_dadatime (&(db->utc_start), psec_to_add);
        pthread_mutex_unlock (db->mutex);
        usleep (1000000);
      }
      else
      {
        fprintf (stderr, "input[%x]: this pol behind, sleeping\n", db->key);
        // this thread is behind the other one, wait for it to chew
        // through some data
        pthread_mutex_unlock (db->mutex);
        usleep (1000000);
      }
    }
  }
  fprintf (stderr, "input[%x]: aligned %lu %lu\n", db->key, db->utc_start.tv_sec, db->utc_start.tv_psec);
      
  return; 
}

int main (int argc, char **argv)
{
  dada_dbmergedb_t dbmergedb;

  dada_client_t* client = 0;

  // Flag set in verbose mode
  char verbose = 0;

  // Flag for waiting for both pols
  char wait_for_both = 0;

  // number of transfers
  unsigned single_transfer = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"svw")) != -1)
  {
    switch (arg) 
    {
      case 's':
        single_transfer = 1;
        break;

      case 'w':
        wait_for_both = 1;
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 0;
      
    }
  }

  dbmergedb.verbose = verbose;
  dbmergedb.wait_for_both = wait_for_both;

  int num_args = argc-optind;
  if (num_args != 3)
  {
    fprintf(stderr, "dada_dbmergedb: 3 arguments required, %d provided\n", num_args);
    usage();
    exit(EXIT_FAILURE);
  } 

  if (sscanf (argv[optind], "%x", &(dbmergedb.pola.key)) != 1)
  {
    fprintf (stderr, "dada_dbmergedb: could not parse pola key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }
  if (sscanf (argv[optind+1], "%x", &(dbmergedb.polb.key)) != 1)
  {
    fprintf (stderr, "dada_dbmergedb: could not parse polb key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }
  if (sscanf (argv[optind+2], "%x", &dbmergedb.output.key) != 1)
  {
    fprintf (stderr, "dada_dbmergedb: could not parse out key from %s\n", argv[optind+2]);
    return EXIT_FAILURE;
  }

  dada_dbmergedb_t * ctx = &dbmergedb;

  ctx->log = multilog_open ("dada_dbmergedb", 0);

  multilog_add (ctx->log, stderr);

  // initialize data block connections and memory structures
  if (dbmergedb_init (ctx) < 0)
  {
    fprintf (stderr, "dada_dbmergedb: failed to initialize\n");
    return EXIT_FAILURE;
  }

  multilog (ctx->log, LOG_INFO, "main: dada_client_create()\n");
  client = dada_client_create ();

  client->log               = ctx->log;
  client->data_block        = ctx->output.hdu->data_block;
  client->header_block      = ctx->output.hdu->header_block;
  client->open_function     = dbmergedb_open;
  client->io_function       = dbmergedb_write;
  client->io_block_function = dbmergedb_write_block;
  client->close_function    = dbmergedb_close;
  client->direction         = dada_client_writer;

  client->context = &dbmergedb;
  client->quiet = (verbose > 0) ? 0 : 1;

  multilog (ctx->log, LOG_INFO, "main: while (!%d)\n", client->quit);
  while (!client->quit)
  {
    multilog (ctx->log, LOG_INFO, "main: pthread_create (POLA)\n");
    // start the input threads to align the input data streams
    int rval;
    rval = pthread_create (&(ctx->pola_thread_id), 0, (void *) input_thread, (void *) &(ctx->pola));
    if (rval < 0)
    {
      multilog (ctx->log, LOG_ERR, "Could not create pol A thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }

    multilog (ctx->log, LOG_INFO, "main: pthread_create (POLB)\n");
    rval = pthread_create (&(ctx->polb_thread_id), 0, (void *) input_thread, (void *) &(ctx->polb));
    if (rval < 0)
    {
      multilog (ctx->log, LOG_ERR, "Could not create pol A thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }

    multilog (ctx->log, LOG_INFO, "main: dada_client_read()\n");
    if (dada_client_write (client) < 0)
      multilog (ctx->log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (ctx->log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_write (ctx->output.hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || dbmergedb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_write (ctx->output.hdu) < 0)
      {
        multilog (ctx->log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (ctx->output.hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
