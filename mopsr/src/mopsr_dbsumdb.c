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
#include <xmmintrin.h>

int quit_threads = 0;

void control_thread (void *);

int64_t dbsumdb_write_block (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbsumdb_write_block_STF_to_TF (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbsumdb [options] in_key out_keys\n"
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
} mopsr_dbsumdb_hdu_t;


typedef struct {

  mopsr_dbsumdb_hdu_t output;

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

} mopsr_dbsumdb_t;

#define DADA_DBSUMDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

/*! Function that opens the data transfer target */
int dbsumdb_open (dada_client_t* client)
{
  // the mopsr_dbsumdb specific data
  mopsr_dbsumdb_t* ctx = (mopsr_dbsumdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbsumdb_open()\n");

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
    if ((strcmp(ctx->order, "STF") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from STF to TF\n");
      client->io_block_function = dbsumdb_write_block_STF_to_TF;
      strcpy (output_order, "TF");
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

  uint64_t new_bytes_per_second = bytes_per_second / ctx->nant;

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
  memcpy ( header, client->header, header_size );

  // now set each output data block to 1 antenna
  int nant = 1;
  if (ascii_header_set (header, "NANT", "%d", nant) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NANT=%d to header\n",
                             nant);
    return -1;
  }

  if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, bytes_per_second) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new BYTES_PER_SECOND to header\n");
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

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbsumdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbsumdb_t* ctx = (mopsr_dbsumdb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbsumdb_hdu_t * o = 0;

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
      multilog (log, LOG_ERR, "dbsumdb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbsumdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbsumdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbsumdb_t* ctx = (mopsr_dbsumdb_t*) client->context;

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

int64_t dbsumdb_write_block_STF_to_TF (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsumdb_t* ctx = (mopsr_dbsumdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->nant;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_STF_to_TF: in_data_size=%"PRIu64", "
              "out_data_size=%"PRIu64"\n", in_data_size, out_data_size);

  void * in = in_data;
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

  // copy the first antenna from input to output
  memcpy (out, in, out_data_size);

  __m128i * dest = out;
  __m128i * src;

  // _mm_add_pi8 does 8 operations at once
  uint64_t nops = out_data_size / 128;
  uint64_t iop;
  for (iant=1; iant < ctx->nant; iant++)
  {
    src = (__m128i *) (in + (iant * out_data_size));
    for (iop=0; iop<nops; iop++)
      dest[iop] = _mm_add_epi8 (dest[iop], src[iop]);
  }

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

//
// shouldn't need this unless multiplex combined FT data
//
int64_t dbsumdb_write_block_FST_to_FT (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t block_id)
{
  mopsr_dbsumdb_t* ctx = (mopsr_dbsumdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_FST_to_FT: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              in_data_size, block_id);

  const uint64_t out_data_size = in_data_size / ctx->nant;
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

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] ipcio_open_block_write()\n", ctx->output.key);
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    out = (void *) ctx->output.curr_block;
  }
  else
    out = (void *) ctx->output.curr_block + ctx->output.bytes_written;

  in = in_data;

  __m128i * dest = out;
  __m128i * src;

  // _mm_add_pi8 does 8 operations at once
  uint64_t nops = out_data_size / (ctx->nchan * 128);
  uint64_t iop;

  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    // copy 1 channel for the first antenna from input to output
    memcpy (out, in, ant_stride);

    dest = (__m128i *) out;

    for (iant=1; iant < ctx->nant; iant++)
    {
      src = (__m128i *) (in + (iant * ant_stride));
      for (iop=0; iop<nops; iop++)
        dest[iop] = _mm_add_epi8 (dest[iop], src[iop]);
    }

    out += out_stride;
    in  += in_stride;
  }

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + in_data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_update_block_write failed\n", ctx->output.key);
        return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_FST_to_FT [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_FST_to_FT [%x] ipcio_close_block_write failed\n", ctx->output.key);
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
    multilog (log, LOG_INFO, "write_block_FST_to_FT read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return in_data_size;
}


int main (int argc, char **argv)
{
  mopsr_dbsumdb_t dbsumdb = DADA_DBSUMDB_INIT;

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
          dbsumdb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "mopsr_dbsumdb: -p requires argument\n");
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

  dbsumdb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "mopsr_dbsumdb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbsumdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbsumdb.output.key)) != 1) {
    fprintf (stderr, "mopsr_dbsumdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbsumdb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbsumdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbsumdb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  dbsumdb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbsumdb.output.hdu, dbsumdb.output.key);
  if (dada_hdu_connect (dbsumdb.output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbsumdb.output.key);
    return -1;
  }
  dbsumdb.output.curr_block = 0;
  dbsumdb.output.bytes_written = 0;
  dbsumdb.output.block_open = 0;
  dbsumdb.output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbsumdb.output.hdu->data_block);

  // we cannot strictly test this until we receive the header and get the number of antenna
  // it must at least be a factor of the input block size (
  if (verbose)
    multilog (log, LOG_INFO, "main: dbsumdb.output.block_size=%"PRIu64"\n", dbsumdb.output.block_size);
  if (zero_copy && (block_size % dbsumdb.output.block_size != 0))
  {
    multilog (log, LOG_ERR, "for zero copy, all DADA buffer block sizes must "
                            "be matching\n");
   return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbsumdb_open;
  client->io_function   = dbsumdb_write;

  if (zero_copy)
  {
    client->io_block_function = dbsumdb_write_block_STF_to_TF;
  }

  client->close_function = dbsumdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbsumdb;
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

    if (single_transfer || dbsumdb.quit)
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
