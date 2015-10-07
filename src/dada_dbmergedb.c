#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "daemon.h"

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
  double  tv_usec;
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
  uint64_t bytes_in;
  uint64_t bytes_out;
  int verbose;
  unsigned quit;
  char wait_for_both;
  pthread_t pola_thread_id;
  pthread_t polb_thread_id;
  pthread_mutex_t   mutex;

} dada_dbmergedb_t;

double diff_dada_time ( dada_timeval_t time1, dada_timeval_t time2 );
double diff_dada_timeval ( dada_timeval_t time1, struct timeval time2 );
int dbmergedb_init_hdu (dada_dbmergedb_t * ctx, dada_block_t * hdu, char reader);
int dbmergedb_init (dada_dbmergedb_t * ctx);
int dbmergedb_destroy (dada_dbmergedb_t * ctx);

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
double diff_dada_time ( dada_timeval_t time1, dada_timeval_t time2 )
{
  double difference = (((double) time2.tv_sec) - ((double) time1.tv_sec)) +
                      ((time2.tv_usec - time1.tv_usec) / 1000000.0);

  return difference;
}

// return the difference in dada_timeval_t's in seconds
double diff_dada_timeval (dada_timeval_t time1, struct timeval time2 )
{
  double difference = (((double) time2.tv_sec) - ((double) time1.tv_sec)) +
                      ((((double) time2.tv_usec) - time1.tv_usec) / 1000000.0);

  return difference;
}

void add_to_dadatime (dada_timeval_t * time1, double seconds)
{
  // add to fractional component
  time1->tv_usec += (seconds * 1000000.0);

  unsigned seconds_to_add = (unsigned) floor(time1->tv_usec / 1000000.0);

  time1->tv_sec += seconds_to_add;
  time1->tv_usec -= ((double) seconds_to_add * 1000000.0);
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

  db->bufsz = ipcbuf_get_bufsz((ipcbuf_t *)db->hdu->data_block);
  db->block_open = db->bytes = db->bytes = 0;
  db->curr_buf = 0;
  db->header = 0;
  db->mutex = &(ctx->mutex);
  if (ctx->wait_for_both)
    db->max_percent_full = 1.1;
  else
    db->max_percent_full = 0.5;

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
    multilog (ctx->log, LOG_ERR, "pola and polb bufsz did not equal output bufsz\n");
    return -1;
  }

  pthread_mutex_init(&(ctx->mutex), NULL);

  return 0;
}

int dbmergedb_destroy (dada_dbmergedb_t * ctx)
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

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;
  
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

  // the two polarisations are not aligned to within half a sample, bork
  if (abs (diff_dada_time (ctx->pola.utc_start, ctx->polb.utc_start)) > tsamp / 2.0)
  {
    multilog (ctx->log, LOG_ERR, "Failed to align polarisations\n");
    return -1;
  }

  // determine the new UTC_START for the output
  char utc_start[20];
  strftime (utc_start, 20, DADA_TIMESTR, gmtime(&(ctx->pola.utc_start.tv_sec)));
  if (ascii_header_set (client->header, "UTC_START", "%s", utc_start) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write UTC_START=%s to header\n", utc_start);

  if (ascii_header_set (client->header, "MICROSECONDS", "%lf", ctx->pola.utc_start.tv_usec) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write MICROSECONDS=%lf to header\n", ctx->pola.utc_start.tv_usec);

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

  uint64_t resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
    multilog (ctx->log, LOG_WARNING, "Could not read RESOLUTION from header\n");
  resolution *= npol;
  if (ascii_header_set (client->header, "RESOLUTION", "%"PRIu64, resolution) < 0)
    multilog (ctx->log, LOG_WARNING, "Could not write RESOLUTION=%"PRIu64" to header\n", resolution);

  //client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;

  client->header_transfer = 0;

  return 0;
}

int dbmergedb_close (dada_client_t* client, uint64_t bytes_written)
{
  dada_dbmergedb_t * ctx = (dada_dbmergedb_t*) client->context;
  
  multilog_t* log = client->log;

  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  // close the input
  if (ctx->pola.block_open)
  {
    ipcio_close_block_read (ctx->pola.hdu->data_block, ctx->pola.bufsz);
    ctx->pola.block_open = 0;
  }

  if (ctx->polb.block_open)
  {
    ipcio_close_block_read (ctx->polb.hdu->data_block, ctx->polb.bufsz);
    ctx->polb.block_open = 0;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbmergedb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  dada_dbmergedb_t* ctx = (dada_dbmergedb_t*) client->context;

  multilog_t * log = client->log;

  multilog (log, LOG_ERR, "write: should not be called\n");
  return -1;
}

int64_t dbmergedb_write_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  dada_dbmergedb_t* ctx = (dada_dbmergedb_t*) client->context;

  multilog_t * log = client->log;

  dada_block_t * o = 0;

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
   
  for (ochunk=0; ochunk<nchunk; ochunk++)
  {
    if (!ctx->pola.block_open)
    {
      ctx->pola.curr_buf = ipcio_open_block_read(ctx->pola.hdu->data_block, &(ctx->pola.bytes),  &(ctx->pola.block_id));
      ctx->pola.block_open = 1;
      ctx->pola.bytes = 0;
    }
     
    if (!ctx->polb.block_open)
    {
      ctx->polb.curr_buf = ipcio_open_block_read(ctx->polb.hdu->data_block, &(ctx->polb.bytes),  &(ctx->polb.block_id));
      ctx->polb.block_open = 1;
      ctx->polb.bytes = 0;
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

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}


/*
 * Thread to align the input polarisation to the same UTC second as the other
 * polarisation
 */
void * input_thread (void * arg)
{
  dada_block_t * db = (dada_block_t*) arg;

  // the input data block is locked for reading
  uint64_t header_size;

  // wait for the header to appear
  char * header = ipcbuf_get_next_read (db->hdu->header_block, &header_size);
  if (!header) 
  {
    fprintf (stderr, "input[%x]: could not get next header block\n", db->key);
    return;
  }
  
  // allocate aligned memory for the header
  posix_memalign ( (void **) &(db->header), 512, header_size);

  // make a copy of the ascii header into our header buffer
  memcpy (db->header, header, header_size);

  // mark the header as cleared
  ipcbuf_mark_cleared (db->hdu->header_block);

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

  if (ascii_header_get (header, "MICROSECONDS", "%lf", &(db->utc_start.tv_usec)) != 1)
  {
    fprintf (stderr, "input[%x]: header did not contain MICROSECONDS\n", db->key);
    db->error = 1;
    return;
  }

  uint64_t obs_offset;
  if (ascii_header_get (header, "OBS_OFFSET", "%lu", &obs_offset) != 1)
  {     
    fprintf (stderr, "input[%x]: header did not contain OBS_OFFSET\n", db->key);
    db->error = 1;
    return;        
  }

  uint64_t bytes_per_second;
  if (ascii_header_get (header, "BYTES_PER_SECOND", "%lu", &bytes_per_second) != 1)
  { 
    fprintf (stderr, "input[%x]: header did not contain BYTES_PER_SECOND\n", db->key);
    db->error = 1;
    return;
  }

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

  const double seconds_per_buffer = (double) db->bufsz / (double) bytes_per_second;
  char both_aligned = 0;

  while (!both_aligned)
  {
    // determine the next second on which this input CAN start 
    pthread_mutex_lock (db->mutex);

    // if the other thread has not yet started, we must wait for it so 
    // that we which know which thread needs to delay itself
    if (db->other_pol->utc_start.tv_sec == 0)
    {
      if (ipcio_percent_full (db->hdu->data_block) > db->max_percent_full)
      {
        db->curr_buf = ipcio_open_block_read (db->hdu->data_block, &(db->bytes), &(db->block_id));
        ipcio_close_block_read (db->hdu->data_block, db->bufsz);
        add_to_dadatime (&(db->utc_start), seconds_per_buffer);
      }
      pthread_mutex_unlock (db->mutex);
    }
    // both threads have started, check the time difference between them
    else
    {
      // difference = this - other. If +ve this thread is ahead of other pol
      double difference = diff_dada_time (db->other_pol->utc_start, db->utc_start);

      if (abs(difference) < tsamp / 2.0)
      {
        both_aligned = 1;
        pthread_mutex_unlock (db->mutex);
      }
      // if this thread started before the other, wait for the other thread
      else if (difference > 0)
      {
        // round the difference to the nearest integer byte
        uint64_t bytes_to_read = (uint64_t) rint(difference * bytes_per_second);

        if (bytes_to_read % resolution != 0)
        {
          fprintf (stderr, "input[%x]: difference between polA and polB was not an "
                           "integer multiple of RESOlUTION\n", db->key);
          db->error = 1;
          return;
        }

        // discard all data up to the first startable second
        while (bytes_to_read > 0)
        {
          if (!db->block_open)
          {
            db->curr_buf = ipcio_open_block_read (db->hdu->data_block, &(db->bytes), &(db->block_id));
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
        add_to_dadatime (&(db->utc_start), abs(difference));
        pthread_mutex_unlock (db->mutex);
      }
      else
      {
        // this thread is behind the other one, wait for it to chew
        // through some data
        pthread_mutex_unlock (db->mutex);
        usleep (100000);
      }
    }
  }
      
  return; 
}

int main (int argc, char **argv)
{
  dada_dbmergedb_t dbmergedb;

  dada_client_t* client = 0;

  // DADA Logger
  multilog_t* log = 0;

  // Flag set in daemon mode
  char daemon = 0;

  // Flag set in verbose mode
  char verbose = 0;

  // Flag for waiting for both pols
  char wait_for_both = 0;

  // number of transfers
  unsigned single_transfer = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dsv")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

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
  int i = 0;
      
  if ((argc-optind) != 3)
  {
    fprintf(stderr, "dada_dbmergedb: 3 arguments required\n");
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
  if (sscanf (argv[optind], "%x", &dbmergedb.output.key) != 1)
  {
    fprintf (stderr, "dada_dbmergedb: could not parse out key from %s\n", argv[optind+2]);
    return EXIT_FAILURE;
  }

  dada_dbmergedb_t * ctx = &dbmergedb;

  ctx->log = multilog_open ("dada_dbmergedb", 0);

  multilog_add (ctx->log, stderr);

  // initialize data block connections and memory structures
  dbmergedb_init (ctx);

  client = dada_client_create ();

  client->log               = log;
  client->data_block        = ctx->output.hdu->data_block;
  client->header_block      = ctx->output.hdu->header_block;
  client->open_function     = dbmergedb_open;
  client->io_function       = dbmergedb_write;
  client->io_block_function = dbmergedb_write_block;
  client->close_function    = dbmergedb_close;
  client->direction         = dada_client_writer;

  client->context = &dbmergedb;
  client->quiet = (verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    // start the input threads to align the input data streams
    int rval;
    rval = pthread_create (&(ctx->pola_thread_id), 0, (void *) input_thread, (void *) &(ctx->pola));
    if (rval < 0)
    {
      multilog(log, LOG_ERR, "Could not create pol A thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }

    rval = pthread_create (&(ctx->polb_thread_id), 0, (void *) input_thread, (void *) &(ctx->polb));
    if (rval < 0)
    {
      multilog(log, LOG_ERR, "Could not create pol A thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_write (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_write (ctx->output.hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || dbmergedb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_write (ctx->output.hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (ctx->output.hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
