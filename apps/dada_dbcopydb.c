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

int quit_threads = 0;

void control_thread (void *);

void usage()
{
  fprintf (stdout,
           "dada_dbcopydb [options] in_key out_keys+\n"
           " -p port   control port to use\n"
           " -s        1 transfer, then exit\n"
           " -S        1 observation with multiple transfers, then exit\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n"
           " in_key    DADA key for input data block\n"
           " out_keys  DADA keys for output data block\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} dada_dbcopydb_hdu_t;


typedef struct {

  dada_dbcopydb_hdu_t * outputs;

  unsigned n_outputs; 

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  unsigned quit;

  unsigned control_port;

} dada_dbcopydb_t;

#define DADA_DBCOPYDB_INIT { 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbdecidb_open (dada_client_t* client)
{
  // the dada_dbcopydb specific data
  dada_dbcopydb_t* ctx = (dada_dbcopydb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  dada_dbcopydb_hdu_t * o = 0;  
 
  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbdecidb_open()\n");

  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);
    // lock writer status on the out HDU
    if (dada_hdu_lock_write (o->hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", o->key);
      return -1;
    }
  }

  int64_t obs_xfer = 0;
  if (ascii_header_get (client->header, "OBS_XFER", "%"PRIi64, &obs_xfer) != 1)
     multilog (log, LOG_WARNING, "header with no OBS_XFER\n");

  int64_t transfer_size = 0;
  if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size) != 1)
     multilog (log, LOG_WARNING, "header with no TRASFER_SIZE\n");

  // signal main that this is the final xfer
  if (obs_xfer == -1)
    ctx->quit = 1;

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  // setup headers for all active HDUs
  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);
    if (ctx->verbose)
              multilog (log, LOG_INFO, "open: enabling HDU %x\n", o->key);
    assert( header_size == ipcbuf_get_bufsz (o->hdu->header_block) );

    header = ipcbuf_get_next_write (o->hdu->header_block);
    if (!header)  {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    // copy the header from the in to the out
    memcpy ( header, client->header, header_size );

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

int dbdecidb_close (dada_client_t* client, uint64_t bytes_written)
{
  dada_dbcopydb_t* ctx = (dada_dbcopydb_t*) client->context;
  
  multilog_t* log = client->log;

  dada_dbcopydb_hdu_t * o = 0;

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
        multilog (log, LOG_ERR, "dbdecidb_close: ipcio_close_block_write failed\n");
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
      multilog (log, LOG_ERR, "dbdecidb_close: cannot unlock DADA HDU (key=%x)\n", o->key);
      return -1;
    }

    // mark this output's current state as inactive
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbdecidb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  dada_dbcopydb_t* ctx = (dada_dbcopydb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write decimated data to the active output data blocks
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

int64_t dbdecidb_write_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  dada_dbcopydb_t* ctx = (dada_dbcopydb_t*) client->context;

  multilog_t * log = client->log;

  dada_dbcopydb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: data_size=%"PRIu64", block_id=%"PRIu64"\n",
                data_size, block_id);

  uint64_t out_block_id;
  char * indat = (char *) data;
  char * outdat = 0;

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

    memcpy (outdat, indat, data_size);
  
    o->bytes_written += data_size;

    if (o->bytes_written > o->block_size)
      multilog (log, LOG_ERR, "write_block: [%x] output block overrun by "
                "%"PRIu64" bytes\n", o->key, o->bytes_written - o->block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block: [%x] bytes_written=%"PRIu64", "
                "block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

    // check if the output block is now full
    if (o->bytes_written >= o->block_size)
    {
      // check if this is the end of data
      if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
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

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  dada_dbcopydb_t dbdecidb = DADA_DBCOPYDB_INIT;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // single transfer with multiple xfers
  unsigned quit_xfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"p:sSvz")) != -1)
  {
    switch (arg) 
    {
      case 'p':
        if (optarg)
        {
          dbdecidb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "dada_dbcopydb: -p requires argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 's':
        single_transfer = 1;
        break;

      case 'S':
        quit_xfer = 1;
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

  dbdecidb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) < 2)
  {
    fprintf(stderr, "dada_dbcopydb: at least 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "dada_dbcopydb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  dbdecidb.n_outputs = (unsigned) num_args - 1;
  dbdecidb.outputs = (dada_dbcopydb_hdu_t *) malloc (sizeof(dada_dbcopydb_hdu_t) * dbdecidb.n_outputs);
  if (!dbdecidb.outputs)
  {
    fprintf (stderr, "dada_dbcopydb: could not allocate memory\n");
    return EXIT_FAILURE;
  }

  // read output DADA keys from command line arguments
  for (i=1; i<num_args; i++)
  {
    if (verbose)
      fprintf (stderr, "parsing output key %d=%s\n", i, argv[optind+i]);
    if (sscanf (argv[optind+i], "%x", &(dbdecidb.outputs[i-1].key)) != 1) {
      fprintf (stderr, "dada_dbcopydb: could not parse out key %d from %s\n", i, argv[optind+i]);
      return EXIT_FAILURE;
    }
  }

  log = multilog_open ("dada_dbcopydb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "dada_dbcopydb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "dada_dbcopydb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data blocks
  for (i=0; i<dbdecidb.n_outputs; i++)
  {
    dbdecidb.outputs[i].hdu = dada_hdu_create (log);
    dada_hdu_set_key (dbdecidb.outputs[i].hdu, dbdecidb.outputs[i].key);
    if (dada_hdu_connect (dbdecidb.outputs[i].hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbdecidb.outputs[i].key);
      return -1;
    }
    dbdecidb.outputs[i].curr_block = 0;
    dbdecidb.outputs[i].bytes_written = 0;
    dbdecidb.outputs[i].block_open = 0;
    dbdecidb.outputs[i].block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbdecidb.outputs[i].hdu->data_block);
    if (zero_copy && block_size != dbdecidb.outputs[i].block_size)
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
  client->open_function = dbdecidb_open;
  client->io_function   = dbdecidb_write;

  if (zero_copy)
  {
    client->io_block_function = dbdecidb_write_block;
  }

  client->close_function = dbdecidb_close;
  client->direction      = dada_client_reader;

  client->context = &dbdecidb;
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

    if (single_transfer || (quit_xfer && dbdecidb.quit))
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
