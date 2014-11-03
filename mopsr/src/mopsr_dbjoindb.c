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

int64_t dbjoindb_write_block_TF_to_TFS (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbjoindb [options] in_keys+ out_key\n"
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
  uint64_t      bytes_read;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbjoindb_hdu_t;


typedef struct {

  mopsr_dbjoindb_hdu_t * inputs;

  unsigned n_inputs; 

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

} mopsr_dbjoindb_t;

#define DADA_DBSPLITDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ""}

/*! Function that opens the data transfer target */
int dbjoindb_open (dada_client_t* client)
{
  // the mopsr_dbjoindb specific data
  mopsr_dbjoindb_t* ctx = (mopsr_dbjoindb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  mopsr_dbjoindb_hdu_t * in = 0;  
 
  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbjoindb_open()\n");

  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_inputs; i++)
  {
    in = &(ctx->inputs[i]);
    // lock writer status on the out HDU
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", in->key);

    if (dada_hdu_lock_read(in->hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", in->key);
      return -1;
    }
  }

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  char have_header = 0;

  // setup headers for all active HDUs
  // for all outputs marked ACTIVE, lock read access on them
  for (i=0; i<ctx->n_inputs; i++)
  {
    in = &(ctx->inputs[i]);

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: reading header from HDU %x\n",  in->key);

    if (ctx->verbose)
              multilog (log, LOG_INFO, "open: enabling HDU %x\n", in->key);
    assert( header_size == ipcbuf_get_bufsz (in->hdu->header_block) );

    uint64_t read_buf_size;
    header = ipcbuf_get_next_read (in->hdu->header_block, &read_buf_size);
    if (!header) 
    {
      multilog (log, LOG_ERR, "open: could not get next header block\n");
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: got full header block from HDU %x\n",  in->key);

    if (!have_header)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "open: copying header block from HDU %x\n",  in->key);
      // copy the header from the in to the out
      memcpy ( client->header, header, header_size );
      have_header = 1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: marking header cleared from HDU %x\n",  in->key);

    // mark the outgoing header as filled
    if (ipcbuf_mark_cleared (in->hdu->header_block) < 0)
    {
      multilog (log, LOG_ERR, "Could not input header block cleared\n");
      return -1;
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: processing output header\n");

  // get the transfer size (if it is set)
  int64_t transfer_size;
  if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size) != 1)
    transfer_size = 0;
  transfer_size *= ctx->n_inputs;

  int64_t file_size;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIi64, &file_size) != 1)
    file_size = 0;

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
    multilog (log, LOG_WARNING, "open: header with no ORDER, ASSUMING TF\n");
  }
  else
  {
    if (strcmp(ctx->order, "TF") != 0)
    {
      multilog (log, LOG_ERR, "open: input datablock order MUST be TF\n");
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
  uint64_t new_bytes_per_second = bytes_per_second * ctx->n_inputs;

  // now set each output data block to 1 antenna
  ctx->nant = ctx->n_inputs;
  if (ascii_header_set (client->header, "NANT", "%d", ctx->nant) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NANT=%d to header\n",
                             ctx->nant);
    return -1;
  }

  if (ascii_header_set (client->header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write new BYTES_PER_SECOND to header\n");
    return -1;
  }

  int64_t obs_offset;
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_ERR, "open: failed to read OBS_OFFSET from header\n");
    return -1;
  }

  obs_offset *= ctx->nant;
  if (ascii_header_set (client->header, "OBS_OFFSET", "%"PRIu64, obs_offset) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write OBS_OFFSET to header\n");
    return -1;
  }

  if (file_size)
  {
    file_size *= ctx->nant;
    if (ascii_header_set (client->header, "FILE_SIZE", "%"PRIi64, file_size) < 0)
    {
      multilog (log, LOG_ERR, "open: failed to write FILE_SIZE to header\n");
      return -1;
    }
  }

  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbjoindb_close (dada_client_t* client, uint64_t bytes_read)
{
  mopsr_dbjoindb_t* ctx = (mopsr_dbjoindb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbjoindb_hdu_t * in = 0;

  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  for (i=0; i<ctx->n_inputs; i++)
  { 
    in = &(ctx->inputs[i]);

    // close the block if it is open
    if (in->block_open)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "close: ipcio_close_block_read bytes_read=%"PRIu64"\n");
      if (ipcio_close_block_read (in->hdu->data_block, in->bytes_read) < 0)
      {
        multilog (log, LOG_ERR, "dbjoindb_close: ipcio_close_block_read failed\n");
        return -1;
      }
      in->block_open = 0;
      in->bytes_read = 0;
    }

    // unlock write on the datablock (end the transfer)
    if (ctx->verbose)
      multilog (log, LOG_INFO, "close: dada_hdu_unlock_read\n");

    if (dada_hdu_unlock_read (in->hdu) < 0)
    {
      multilog (log, LOG_ERR, "dbjoindb_close: cannot unlock DADA HDU (key=%x)\n", in->key);
      return -1;
    }

    // mark this output's current state as inactive
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbjoindb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbjoindb_t* ctx = (mopsr_dbjoindb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: HEADER!!!!!! to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  unsigned i = 0;
  for (i=0; i<ctx->n_inputs; i++)
  {
    ipcio_read (ctx->inputs[i].hdu->data_block, data, data_size);
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}


int64_t dbjoindb_write_block_TF_to_TFS (dada_client_t* client, void* out_data, uint64_t out_data_size, uint64_t block_id)
{
  mopsr_dbjoindb_t* ctx = (mopsr_dbjoindb_t*) client->context;

  multilog_t * log = client->log;

  mopsr_dbjoindb_hdu_t * in = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block: out_data_size=%"PRIu64", block_id=%"PRIu64"\n",
              out_data_size, block_id);

  uint64_t ijoin;
  uint64_t in_data_size = 0;
  uint64_t njoin = 0;

  int64_t in_block_size;
  uint64_t in_block_id;

  // 2 bytes / sample
  int16_t * in_ptr;
  int16_t * out_ptr;
  char * in_data;
  const unsigned nin = ctx->n_inputs;

  unsigned i = 0;
  for (i=0; i<ctx->n_inputs; i++)
  {
    in = &(ctx->inputs[i]);

    if (!in->block_open)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] ipcio_open_block_read()\n", in->key);
      in->curr_block = ipcio_open_block_read(in->hdu->data_block, &in_block_size, &in_block_id);
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] in_block_size=%"PRIi64", in_block_id=%"PRIi64"\n", in_block_size, in_block_id);
      if (in_data_size == 0)
      {
        in_data_size = in_block_size;
        out_data_size = in_data_size * ctx->nant;
        njoin = in_data_size / ctx->ndim;

        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block: in_data_size=%"PRIu64", out_data_size=%"PRIu64"\n", in_data_size, out_data_size);
      }
      if (!in->curr_block)
      { 
        multilog (log, LOG_ERR, "write_block: [%x] ipcio_open_block_read failed %s\n", in->key, strerror(errno));
        return -1;
      }
      in->block_open = 1;
      in_data = in->curr_block;
    }
    else
      in_data = in->curr_block + in->bytes_read;

    out_ptr = (int16_t *) out_data;
    in_ptr  = (int16_t *) in_data;
    out_ptr += i;

    for (ijoin=0; ijoin<njoin; ijoin++)
      out_ptr[nin*ijoin] = in_ptr[ijoin]; 
  
    in->bytes_read += in_data_size;

    if (in->bytes_read > in->block_size)
      multilog (log, LOG_ERR, "write_block: [%x] output block overrun by "
                "%"PRIu64" bytes\n", in->key, in->bytes_read - in->block_size);

    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block: [%x] bytes_read=%"PRIu64", "
                "block_size=%"PRIu64"\n", in->key, in->bytes_read, in->block_size);

    // check if the output block is now full
    if (in->bytes_read >= in->block_size)
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] block now full bytes_read=%"PRIu64", block_size=%"PRIu64"\n", in->key, in->bytes_read, in->block_size);

      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block: [%x] close_block_read written=%"PRIu64"\n", in->key, in->bytes_read);
      if (ipcio_close_block_read (in->hdu->data_block, in->bytes_read) < 0)
      {
        multilog (log, LOG_ERR, "write_block: [%x] ipcio_close_block_read failed\n", in->key);
        return -1;
      }
      in->block_open = 0;
      in->bytes_read = 0;
    }
    else
    {
      if (in->bytes_read == 0)
        in->bytes_read = 1;
    }
  }

  ctx->bytes_in += in_data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block: read %"PRIu64", wrote %"PRIu64" bytes\n", in_data_size, out_data_size);

  return out_data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbjoindb_t dbjoindb = DADA_DBSPLITDB_INIT;

  // output HDU
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

  int arg = 0;

  while ((arg=getopt(argc,argv,"svz")) != -1)
  {
    switch (arg) 
    {
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

  dbjoindb.verbose = verbose;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) < 2)
  {
    fprintf(stderr, "mopsr_dbjoindb: at least 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  dbjoindb.n_inputs = (unsigned) num_args - 1;
  dbjoindb.inputs = (mopsr_dbjoindb_hdu_t *) malloc (sizeof(mopsr_dbjoindb_hdu_t) * dbjoindb.n_inputs);
  if (!dbjoindb.inputs)
  {
    fprintf (stderr, "mopsr_dbjoindb: could not allocate memory\n");
    return EXIT_FAILURE;
  }

  // read input DADA keys from command line arguments
  for (i=0; i<num_args-1; i++)
  {
    if (verbose)
      fprintf (stderr, "parsing output key %d=%s\n", i, argv[optind+i]);
    if (sscanf (argv[optind+i], "%x", &(dbjoindb.inputs[i].key)) != 1) {
      fprintf (stderr, "mopsr_dbjoindb: could not parse out key %d from %s\n", i, argv[optind+i]);
      return EXIT_FAILURE;
    }
  }

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+num_args-1]);
  if (sscanf (argv[optind+num_args-1], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbjoindb: could not parse in key from %s\n", argv[optind + num_args-1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbjoindb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbjoindb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock write key=%x\n", in_key);
  if (dada_hdu_lock_write (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbjoindb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data blocks
  for (i=0; i<dbjoindb.n_inputs; i++)
  {
    dbjoindb.inputs[i].hdu = dada_hdu_create (log);
    dada_hdu_set_key (dbjoindb.inputs[i].hdu, dbjoindb.inputs[i].key);
    if (dada_hdu_connect (dbjoindb.inputs[i].hdu) < 0)
    {
      multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", dbjoindb.inputs[i].key);
      return -1;
    }
    dbjoindb.inputs[i].curr_block = 0;
    dbjoindb.inputs[i].bytes_read = 0;
    dbjoindb.inputs[i].block_open = 0;
    dbjoindb.inputs[i].block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) dbjoindb.inputs[i].hdu->data_block);
    if (verbose)
      multilog (log, LOG_INFO, "main: dbjoindb.inputs[%d].block_size=%"PRIu64"\n", i, dbjoindb.inputs[i].block_size);
    if (zero_copy && ((block_size * dbjoindb.n_inputs) != dbjoindb.inputs[i].block_size))
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
  client->open_function = dbjoindb_open;
  client->io_function   = dbjoindb_write;

  //if (zero_copy)
  {
    client->io_block_function = dbjoindb_write_block_TF_to_TFS;
  }

  client->close_function = dbjoindb_close;
  client->direction      = dada_client_writer;

  client->context = &dbjoindb;
  client->quiet = (verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_write()\n");

    if (dada_client_write (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_write()\n");

    if (dada_hdu_unlock_write (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || dbjoindb.quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_write (hdu) < 0)
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
