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

int quit_threads = 0;

void control_thread (void *);

int64_t dbtransposedb_write_block (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbtransposedb_write_block_ST_to_TS (dada_client_t *, void *, uint64_t, uint64_t);
int64_t dbtransposedb_write_block_SFT_to_STF (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbtransposedb [options] in_key out_key\n"
           " -s        1 transfer, then exit\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n"
           " in_key    DADA key for input data block\n"
           " out_key   DADA key for output data blocks\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbtransposedb_hdu_t;


typedef struct {

  mopsr_dbtransposedb_hdu_t output;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  unsigned int nsig;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit;

  unsigned quit;

  unsigned control_port;

  char order[4];

} mopsr_dbtransposedb_t;

#define DADA_DBSUMDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

/*! Function that opens the data transfer target */
int dbtransposedb_open (dada_client_t* client)
{
  // the mopsr_dbtransposedb specific data
  mopsr_dbtransposedb_t* ctx = (mopsr_dbtransposedb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbtransposedb_open()\n");

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

  int nant;
  int nbeam;
  // get the number of antenna
  if (ascii_header_get (client->header, "NANT", "%u", &nant) != 1)
  {
    nant = 1;
  }

  if (ascii_header_get (client->header, "NBEAM", "%u", &nbeam) != 1)
  {
    nbeam = 1;
  }

  if ((nant == 1) && (nbeam == 1))
  {
    multilog (log, LOG_ERR, "open: cannot transpose from ST to T with NANT=%d && NBEAM=%d\n", nant, nbeam);
    return -1;
  }
  ctx->nsig = nant > nbeam ? nant : nbeam;

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
  if (ctx->nchan != 1 && nbeam == 1)
  {
    multilog (log, LOG_ERR, "open: cannot transpose from ST to T with NCHAN=%d\n", ctx->nchan);
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
      multilog (log, LOG_INFO, "open: changing order from ST to TS\n");
      client->io_block_function = dbtransposedb_write_block_ST_to_TS;
      strcpy (output_order, "TS");
    }
    else if ((strcmp(ctx->order, "SFT") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from SFT to STF\n");
      client->io_block_function = dbtransposedb_write_block_SFT_to_STF;
      strcpy (output_order, "STF");
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

int dbtransposedb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbtransposedb_t* ctx = (mopsr_dbtransposedb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbtransposedb_hdu_t * o = 0;

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
      multilog (log, LOG_ERR, "dbtransposedb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbtransposedb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbtransposedb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbtransposedb_t* ctx = (mopsr_dbtransposedb_t*) client->context;

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

int64_t dbtransposedb_write_block_ST_to_TS (dada_client_t* client, void* in_data, uint64_t data_size, uint64_t block_id)
{
  mopsr_dbtransposedb_t* ctx = (mopsr_dbtransposedb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_ST_to_TS: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  int16_t * in = (int16_t *) in_data;
  int16_t * out;
 
  const uint64_t nsamp = data_size / (ctx->nsig * ctx->nchan * ctx->ndim); 
  uint64_t out_block_id;
  unsigned isig, isamp;

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_ST_to_TS [%x] ipcio_open_block_write()\n", ctx->output.key);
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_ST_to_TS [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    ctx->output.bytes_written = 0;
    out = (int16_t *) ctx->output.curr_block;
  }
  else
    out = (int16_t *) (ctx->output.curr_block + ctx->output.bytes_written);

  /*
  unsigned ichunk;
  const unsigned cache_size = 4096;
  const unsigned chunk_size = nsamp * ctx->ndim * nout;
  const unsigned nchunk = (unsigned) (data_size / chunk_size_in);
  const unsigned nchunk_left = (unsigned) (data_size % chunk_size_in);
  */

  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (isig=0; isig<ctx->nsig; isig++)
    {
      out[isamp*ctx->nsig + isig] = in[isig*nsamp + isamp];
    }
  }

  ctx->output.bytes_written += data_size;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_ST_to_TS [%x] output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_ST_to_TS [%x] bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_ST_to_TS [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_ST_to_TS [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_ST_to_TS [%x] ipcio_update_block_write failed\n", ctx->output.key);
         return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_ST_to_TS [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_ST_to_TS [%x] ipcio_close_block_write failed\n", ctx->output.key);
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

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_ST_to_TS read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}

int64_t dbtransposedb_write_block_SFT_to_STF (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  mopsr_dbtransposedb_t* ctx = (mopsr_dbtransposedb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  //  assume 32-bit, detected data
  uint32_t * in = (int32_t *) in_data;
  uint32_t * out; 
 
  const uint64_t nsamp = data_size / (ctx->nsig * ctx->ndim * ctx->nchan); 
  const size_t sig_stride = nsamp * ctx->ndim * ctx->nchan;
  uint64_t out_block_id;
  unsigned isig, isamp, ichan, ochan;

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_STF [%x] ipcio_open_block_write()\n", ctx->output.key);
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_SFT_to_STF [%x] ipcio_open_block_write failed %s\n", ctx->output.key, strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    ctx->output.bytes_written = 0;
    out = (uint32_t *) ctx->output.curr_block;
  }
  else
    out = (uint32_t *) (ctx->output.curr_block + ctx->output.bytes_written);

  for (isig=0; isig<ctx->nsig; isig++)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      ochan = (ctx->nchan - 1) - ichan;
      for (isamp=0; isamp<nsamp; isamp++)
      {
        out[isamp * ctx->nchan + ochan] = in[ichan * nsamp + isamp];
      }
    }
    out += sig_stride;
  }

  ctx->output.bytes_written += data_size;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_SFT_to_STF [%x] output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.key, ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF [%x] bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_STF [%x] block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_SFT_to_STF [%x] update_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_STF [%x] ipcio_update_block_write failed\n", ctx->output.key);
         return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_SFT_to_STF [%x] close_block_write written=%"PRIu64"\n", ctx->output.key, ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_STF [%x] ipcio_close_block_write failed\n", ctx->output.key);
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

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbtransposedb_t dbtransposedb = DADA_DBSUMDB_INIT;

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
          dbtransposedb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "mopsr_dbtransposedb: -p requires argument\n");
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

  dbtransposedb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "mopsr_dbtransposedb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbtransposedb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(dbtransposedb.output.key)) != 1) {
    fprintf (stderr, "mopsr_dbtransposedb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbtransposedb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbtransposedb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbtransposedb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  dbtransposedb.output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (dbtransposedb.output.hdu, dbtransposedb.output.key);
  if (dada_hdu_connect (dbtransposedb.output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbtransposedb.output.key);
    return -1;
  }
  dbtransposedb.output.curr_block = 0;
  dbtransposedb.output.bytes_written = 0;
  dbtransposedb.output.block_open = 0;
  dbtransposedb.output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbtransposedb.output.hdu->data_block);

  if (verbose)
    multilog (log, LOG_INFO, "main: dbtransposedb.output.block_size=%"PRIu64"\n", dbtransposedb.output.block_size);
  if (zero_copy && (block_size != dbtransposedb.output.block_size))
  {
    multilog (log, LOG_ERR, "for zero copy, input and output block sizes must match\n");
   return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbtransposedb_open;
  client->io_function   = dbtransposedb_write;

  if (zero_copy)
  {
    client->io_block_function = dbtransposedb_write_block_ST_to_TS;
  }

  client->close_function = dbtransposedb_close;
  client->direction      = dada_client_reader;

  client->context = &dbtransposedb;
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

    if (single_transfer || dbtransposedb.quit)
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
