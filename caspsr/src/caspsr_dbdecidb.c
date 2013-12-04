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

#define DBDECIDB_INACTIVE 0
#define DBDECIDB_ACTIVE 1

int quit_threads = 0;

void control_thread (void *);

void usage()
{
  fprintf (stdout,
           "caspsr_dbdecidb [options] in_key out_keys+\n"
           " in_key    DADA key for input data block\n"
           " out_keys  DADA keys for output data block\n"
           "\n"
           " -n zone   move to the specifed nyquist zone [default 0]\n"
           " -p port   control port for control of out/aux datablocks [default no control]\n"
           " -t num    decimation factor [default 2]\n"
           " -s        1 transfer, then exit\n"
           " -S        1 observation with multiple transfers, then exit\n"
           " -z        use zero copy transfers\n"
           " -v        verbose mode\n");
}

typedef struct 
{
  dada_hdu_t * hdu;
  key_t key;
  unsigned state;
  unsigned new_state;
  uint64_t block_size;
  uint64_t bytes_written;
  unsigned block_open;
  char * curr_block;
} caspsr_dbdecidb_hdu_t;


typedef struct
{
  double bw;
  double cf;
  double tsamp;
  double bytesps;
  uint64_t filesize;
  uint64_t obs_offset;
  uint64_t transfer_size;
  int64_t obs_xfer;
} caspsr_dbdecidb_hdr_t;

typedef struct {

  caspsr_dbdecidb_hdu_t * outputs;

  unsigned n_outputs; 

  // decimation factor
  unsigned tdec;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  // verbose output
  int verbose;

  // number of bytes of consective pol
  int resolution;

  // nyquist zone to move to
  int nyquist_zone;

  uint64_t outdat;

  unsigned idec;

  unsigned ocnt;

  unsigned quit;

  unsigned control_port;

} caspsr_dbdecidb_t;

#define CASPSR_DBDECIDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int dbdecidb_open (dada_client_t* client)
{

  // the caspsr_dbdecidb specific data
  caspsr_dbdecidb_t* ctx = (caspsr_dbdecidb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  // incoming header params
  caspsr_dbdecidb_hdr_t old;

  // outgoing header params
  caspsr_dbdecidb_hdr_t new;

  caspsr_dbdecidb_hdu_t * o = 0;  
 
  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbdecidb_open()\n");

  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);
    if (o->new_state == DBDECIDB_ACTIVE)
    {
      // lock writer status on the out HDU
      if (dada_hdu_lock_write (o->hdu) < 0)
      {
        multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", o->key);
        return -1;
      }
      o->state = DBDECIDB_ACTIVE;
    }
  }

  read_header_params(client->header, &old, log);

  // signal main that this is the final xfer
  if (old.obs_xfer == -1)
    ctx->quit = 1;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: parsed old BANDWIDTH=%lf, CFREQ=%lf, TSAMP=%lf,"
                             " BYTES_PER_SECOND=%lf, OBS_OFFSET=%"PRIu64", "
                             " TRANSFER_SIZE=%"PRIu64" OBS_XFER=%"PRIi64"\n", old.bw, old.cf, old.tsamp, 
                             old.bytesps, old.obs_offset, old.transfer_size, old.obs_xfer);

  // generate the new BANDWIDTH, CFREQ and TSAMP
  if (ctx->tdec == 1)
  {
    new.bw = old.bw;
    new.cf = old.cf;
  } 
  else 
  {
    new.bw = (old.bw / ctx->tdec) * powf(-1, ctx->nyquist_zone);
    new.cf = old.cf + old.bw * ((1.0 / ctx->tdec) * (0.5 + ctx->nyquist_zone) - 0.5);
  }
  new.tsamp = old.tsamp * ctx->tdec;
  new.bytesps = old.bytesps / ctx->tdec;
  new.filesize = old.filesize / ctx->tdec;
  new.obs_offset = old.obs_offset / ctx->tdec;
  new.transfer_size = old.transfer_size / ctx->tdec; 
  new.obs_xfer = old.obs_xfer;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: setting new BANDWIDTH=%lf, CFREQ=%lf, TSAMP=%lf,"
                             " BYTES_PER_SECOND=%lf, OBS_OFFSET=%"PRIu64","
                             " TRANSFER_SIZE=%"PRIu64" OBS_XFER=%"PRIi64"\n", new.bw, new.cf, 
                             new.tsamp, new.bytesps, new.obs_offset, new.transfer_size,
                             new.obs_xfer);

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);

  // setup headers for all active HDUs
  // for all outputs marked ACTIVE, lock write access on them
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);
    if (o->state == DBDECIDB_ACTIVE)
    {
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

      // adjust the header params
      write_header_params (header, new, log);

      // mark the outgoing header as filled
      if (ipcbuf_mark_filled (o->hdu->header_block, header_size) < 0)  {
        multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
        return -1;
      }
      if (ctx->verbose) 
        multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", o->key);
    }
  }
  client->transfer_bytes = old.transfer_size; 
  client->optimal_bytes = 64*1024*1024 * ctx->tdec;

  ctx->idec = ctx->tdec;
  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbdecidb_close (dada_client_t* client, uint64_t bytes_written)
{
  caspsr_dbdecidb_t* ctx = (caspsr_dbdecidb_t*) client->context;
  
  multilog_t* log = client->log;

  caspsr_dbdecidb_hdu_t * o = 0;

  unsigned i = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  for (i=0; i<ctx->n_outputs; i++)
  { 
    o = &(ctx->outputs[i]);

    if (o->state == DBDECIDB_ACTIVE)
    {
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
      o->state = DBDECIDB_INACTIVE;
    }
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbdecidb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  caspsr_dbdecidb_t* ctx = (caspsr_dbdecidb_t*) client->context;

  multilog_t * log = client->log;

  // decimate the data down inplace
  char * d = (char *) data;
  uint64_t indat = 0;
  uint64_t outdat = 0;
  uint64_t deci_data_size = data_size / ctx->tdec;

  unsigned idec = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  for (indat=0; indat < data_size; indat++)
  {
    idec = idec % ctx->tdec;

    if (indat && (indat % 4 == 0))
      indat += 4;

    if (idec == 0)
    {
      if (outdat && (outdat % 4 == 0))
        outdat += 4;

      d[outdat] = d[indat];     // pol0
      d[outdat+4] = d[indat+4]; // pol1
      outdat++;
    }
    idec++; 
  }

  // write decimated data to the active output data blocks
  unsigned i = 0;
  for (i=0; i<ctx->n_outputs; i++)
  {
    if (ctx->outputs[i].state == DBDECIDB_ACTIVE)
      ipcio_write (ctx->outputs[i].hdu->data_block, data, deci_data_size);
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += deci_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, deci_data_size);
 
  return data_size;
}

int64_t dbdecidb_write_block_1 (dada_client_t* client, void* data, uint64_t data_size,
                                uint64_t block_id)
{
  caspsr_dbdecidb_t* ctx = (caspsr_dbdecidb_t*) client->context;

  multilog_t * log = client->log;

  caspsr_dbdecidb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_1: data_size=%"PRIu64", block_id=%"PRIu64"\n",
                data_size, block_id);

  uint64_t out_block_id;
  uint64_t idat = 0;
  const uint64_t ndat = data_size;
  char * indat = (char *) data;
  char * outdat = 0;

  unsigned i = 0;
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);

    // if this output is currently active
    if (o->state == DBDECIDB_ACTIVE)
    {
      if (!o->block_open)
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_1: [%x] ipcio_open_block_write()\n", o->key);
  
        o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
        if (!o->curr_block)
        { 
          multilog (log, LOG_ERR, "write_block_1: [%x] ipcio_open_block_write failed %s\n", o->key, strerror(errno));
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
        multilog (log, LOG_ERR, "write_block_1: [%x] output block overrun by "
                  "%"PRIu64" bytes\n", o->key, o->bytes_written - o->block_size);

      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_1: [%x] bytes_written=%"PRIu64", "
                  "block_size=%"PRIu64"\n", o->key, o->bytes_written, o->block_size);

      // check if the output block is now full
      if (o->bytes_written >= o->block_size)
      {
        // check if this is the end of data
        if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
        {
          if (ctx->verbose)
            multilog (log, LOG_INFO, "write_block_1: [%x] update_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
          if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
          {
            multilog (log, LOG_ERR, "write_block_1: [%x] ipcio_update_block_write failed\n", o->key);
            return -1;
          }
        }
        else
        {
          if (ctx->verbose > 1)
            multilog (log, LOG_INFO, "write_block_1: [%x] close_block_write written=%"PRIu64"\n", o->key, o->bytes_written);
          if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
          {
            multilog (log, LOG_ERR, "write_block_1: [%x] ipcio_close_block_write failed\n", o->key);
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
    else
    {
      if (ctx->verbose > 2)
        multilog (log, LOG_INFO, "write_block_1: [%x] skipping inactive HDUx\n", o->key);
    }
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_1: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;

}

/*! Optimized write_block for tdec = 2 */
int64_t dbdecidb_write_block_2 (dada_client_t* client, void* data, uint64_t data_size,
                                  uint64_t block_id)
{
  caspsr_dbdecidb_t* ctx = (caspsr_dbdecidb_t*) client->context;

  multilog_t * log = client->log;

  caspsr_dbdecidb_hdu_t * o = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_2: write %"PRIu64" bytes, block=%"PRIu64"\n",
                data_size, block_id);

  uint64_t out_block_id;
  uint64_t idat = 0;
  const uint64_t ndat = data_size;
  char * indat = (char *) data;
  char * outdat = (char *) data;

  // decimate the input data inplace
  for (idat=0; idat<ndat; idat+=16)
  {
     // pol 0
    outdat[0] = indat[0];
    outdat[1] = indat[2];
    outdat[2] = indat[8];
    outdat[3] = indat[10];

    // pol 1
    outdat[4] = indat[4];
    outdat[5] = indat[6];
    outdat[6] = indat[12];
    outdat[7] = indat[14];

    // increment input by 16 samples
    indat += 16;
    // incremenet output by 8 samples
    outdat += 8;
  }

  // copy the decimated data to each active output datablock
  indat = (char *) data;
  unsigned i = 0;
  for (i=0; i<ctx->n_outputs; i++)
  {
    o = &(ctx->outputs[i]);

    if (o->state == DBDECIDB_ACTIVE)
    {
      // if we dont have an active block open
      if (!o->block_open)
      {
        if (ctx->verbose > 1)
          multilog (log, LOG_INFO, "write_block_2: ipcio_open_block_write()\n");
  
        o->curr_block = ipcio_open_block_write(o->hdu->data_block, &out_block_id);
        if (!o->curr_block)
        {
          multilog (log, LOG_ERR, "write_block_2: ipcio_open_block_write error %s\n", strerror(errno));
          return -1;
        }
        o->block_open = 1;
        outdat = o->curr_block;
      }
      else
        outdat = o->curr_block + o->bytes_written;

      // copy the deciamted data
      memcpy (outdat, indat, (size_t) (ndat / 2));

      o->bytes_written += ndat / 2;

      // check if the output block is now full
      if (o->bytes_written >= o->block_size)
      {
        if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
        {
          if (ctx->verbose)
            multilog (log, LOG_INFO, "write_block_2: update_block_write written=%"PRIu64"\n", o->bytes_written);
          if (ipcio_update_block_write (o->hdu->data_block, o->bytes_written) < 0)
          {
            multilog (log, LOG_ERR, "write_block_2: ipcio_update_block_write failed\n");
            return -1;
          }
        }
        else
        {
          if (ctx->verbose)
            multilog (log, LOG_INFO, "write_block_2: close_block_write written=%"PRIu64"\n", o->bytes_written);
            if (ipcio_close_block_write (o->hdu->data_block, o->bytes_written) < 0)
          {
            multilog (log, LOG_ERR, "write_block_2: ipcio_close_block_write failed\n");
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
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size / 2;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_2: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size / 2);

  return data_size;

}

int main (int argc, char **argv)
{
  caspsr_dbdecidb_t dbdecidb = CASPSR_DBDECIDB_INIT;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // decimation factor
  unsigned tdec = 1;

  // number of transfers
  unsigned single_transfer = 0;

  // single transfer with multiple xfers
  unsigned quit_xfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  pthread_t control_thread_id;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dn:p:sSt:vz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'n':
        if (optarg)
        {
          dbdecidb.nyquist_zone = atoi(optarg);
          break;
        }
        else
        {
          fprintf (stderr, "caspsr_dbdecidb: -n requires argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 'p':
        if (optarg)
        {
          dbdecidb.control_port = atoi(optarg);
          break;
        }
        else
        {
          fprintf(stderr, "caspsr_dbdecidb: -p requires argument\n");
          usage();
          return EXIT_FAILURE;
        }

      case 's':
        single_transfer = 1;
        break;

      case 'S':
        quit_xfer = 1;
        break;

      case 't':
        if (!optarg)
        { 
          fprintf (stderr, "caspsr_dbdecidb: -t requires argument\n");
          usage();
          return EXIT_FAILURE;
        }
        if (sscanf (optarg, "%d", &tdec) != 1) 
        {
          fprintf (stderr ,"caspsr_dbdecidb: could not parse tdec from %s\n", optarg);
          usage();
          return EXIT_FAILURE;
        } 
        if ((tdec != 1) && (tdec != 2))
        {
          fprintf (stderr ,"caspsr_dbdecidb: tdec can only be 1 or 2\n");
          usage();
          return EXIT_FAILURE;
        }
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
  dbdecidb.tdec = tdec;

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) < 2)
  {
    fprintf(stderr, "caspsr_dbdecidb: at least 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "caspsr_dbdecidb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  dbdecidb.n_outputs = (unsigned) num_args - 1;
  dbdecidb.outputs = (caspsr_dbdecidb_hdu_t *) malloc (sizeof(caspsr_dbdecidb_hdu_t) * dbdecidb.n_outputs);
  if (!dbdecidb.outputs)
  {
    fprintf (stderr, "caspsr_dbdecidb: could not allocate memory\n");
    return EXIT_FAILURE;
  }

  // read output DADA keys from command line arguments
  for (i=1; i<num_args; i++)
  {
    if (verbose)
      fprintf (stderr, "parsing output key %d=%s\n", i, argv[optind+i]);
    if (sscanf (argv[optind+i], "%x", &(dbdecidb.outputs[i-1].key)) != 1) {
      fprintf (stderr, "caspsr_dbdecidb: could not parse out key %d from %s\n", i, argv[optind+i]);
      return EXIT_FAILURE;
    }
  }

  log = multilog_open ("caspsr_dbdecidb", 0);

  multilog_add (log, stderr);

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "caspsr_dbdecidb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "caspsr_dbdecidb: could not lock read on input data block\n");
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
    dbdecidb.outputs[i].state = DBDECIDB_INACTIVE;
    dbdecidb.outputs[i].new_state = DBDECIDB_INACTIVE;
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

  // by default only the first output data block is meant to be active
  dbdecidb.outputs[0].new_state = DBDECIDB_ACTIVE;

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbdecidb_open;
  client->io_function   = dbdecidb_write;

  if (zero_copy)
  {
    // we have optimized io_block functions for TDEC 1 and 2
    if (tdec == 2)
      client->io_block_function = dbdecidb_write_block_2;
    else
      client->io_block_function = dbdecidb_write_block_1;
  }

  client->close_function = dbdecidb_close;
  client->direction      = dada_client_reader;

  client->context = &dbdecidb;
  client->quiet = (verbose > 0) ? 0 : 1;


  if (verbose)
    multilog(log, LOG_INFO, "starting control_thread()\n");
  int rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) client);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
    return -1;
  }

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

int read_header_params(char * header, caspsr_dbdecidb_hdr_t * p, multilog_t * log)
{

  p->bw = 400;
  p->cf = 628;
  p->tsamp = 0.00125;
  p->bytesps = 1600000000;
  p->obs_offset = 0;
  p->filesize = 0;
  p->transfer_size = 0;
  p->obs_xfer = 0;

  if (ascii_header_get (header, "BANDWIDTH", "%lf", &p->bw) != 1) {
    if (ascii_header_get (header, "BW", "%lf", &p->bw) != 1) {
      multilog (log, LOG_WARNING, "header with no BANDWIDTH, using %lf\n", p->bw);
    }
  }

  if (ascii_header_get (header, "CFREQ", "%lf", &p->cf) != 1) {
    p->cf = 628.00;
    multilog (log, LOG_WARNING, "header with no CFREQ, using %lf\n", p->cf);
  }

  if (ascii_header_get (header, "TSAMP", "%lf", &p->tsamp) != 1) {
    p->tsamp = 0.00125;
    if (p->bw != 0)
      p->tsamp = 1 / (p->bw * 2.0);
    multilog (log, LOG_WARNING, "header with no TSAMP, using %lf\n", p->tsamp);
  }

  if (ascii_header_get (header, "BYTES_PER_SECOND", "%lf", &p->bytesps) != 1) {
    p->bytesps = 1600000000;
    multilog (log, LOG_WARNING, "header with no BYTES_PER_SECOND, using %lf\n", p->bytesps);
  }

  if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &p->obs_offset) != 1) {
    p->obs_offset = 0;
    multilog (log, LOG_WARNING, "header with no OBS_OFFSET, using %"PRIu64"\n", p->obs_offset);
  }

  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &p->filesize) != 1) {
    multilog (log, LOG_INFO, "header with no FILE_SIZE, ignoring param\n");
  }

  if (ascii_header_get (header, "TRANSFER_SIZE", "%"PRIu64, &p->transfer_size) != 1) {
    multilog (log, LOG_INFO, "header with no TRANSFER_SIZE, ignoring param\n");
    p->transfer_size = 0;
  }

  if (ascii_header_get (header, "OBS_XFER", "%"PRIi64, &p->obs_xfer) != 1) {
    multilog (log, LOG_WARNING, "header with no OBS_XFER\n");
  }

  return 0;
}

int write_header_params(char * header, caspsr_dbdecidb_hdr_t p, multilog_t * log)
{
  if (ascii_header_set (header, "CFREQ", "%lf", p.cf) < 0) {
    multilog (log, LOG_WARNING, "failed to set CFREQ in outgoing header\n");
  }
  if (ascii_header_set (header, "FREQ", "%lf", p.cf) < 0) {
    multilog (log, LOG_WARNING, "failed to set FREQ in outgoing header\n");
  }
  if (ascii_header_set (header, "BANDWIDTH", "%lf", p.bw) < 0) {
    multilog (log, LOG_WARNING, "failed to set BANDWIDTH in outgoing header\n");
  }
  if (ascii_header_set (header, "BW", "%lf", p.bw) < 0) {
    multilog (log, LOG_WARNING, "failed to set BW in outgoing header\n");
  }
  if (ascii_header_set (header, "TSAMP", "%lf", p.tsamp) < 0) {
    multilog (log, LOG_WARNING, "failed to set TSAMP in outgoing header\n");
  }
  if (ascii_header_set (header, "BYTES_PER_SECOND", "%lf", p.bytesps) < 0) {
    multilog (log, LOG_WARNING, "failed to set BYTES_PER_SECOND in outgoing header\n");
  }
  if (ascii_header_set (header, "OBS_OFFSET", "%"PRIu64, p.obs_offset) < 0) {
    multilog (log, LOG_WARNING, "failed to set OBS_OFFSET in outgoing header\n");
  }
  if (p.filesize)
    if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, p.filesize) < 0)
      multilog (log, LOG_WARNING, "failed to set FILE_SIZE in outgoing header\n");
  if (p.transfer_size) 
    if (ascii_header_set (header, "TRANSFER_SIZE", "%"PRIu64, p.transfer_size) < 0)
      multilog (log, LOG_WARNING, "failed to set TRANSFER_SIZE in outgoing header\n");
}


/*
 *  Thread to provide TCPIP control 
 */
void control_thread (void * arg)
{
  dada_client_t * client = (dada_client_t *) arg;

  caspsr_dbdecidb_t * ctx = (caspsr_dbdecidb_t *) client->context;

  multilog_t * log = client->log;

  // port on which to listen for control commands
  int port = ctx->control_port;

  int thread_result = 0;

  // check if remote control was requested
  if (port == 0)
  {
    if (ctx->verbose)
      multilog(log, LOG_INFO, "control_thread: no port specified, exiting\n");
    pthread_exit((void *) &thread_result);
  }

  if (ctx->verbose)
    multilog(log, LOG_INFO, "control_thread: starting\n");

  // buffer for incoming command strings
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  assert (buffer != 0);

  const char* whitespace = " \r\t\n";
  char * command = 0;
  char * args = 0;

  FILE *sockin = 0;
  FILE *sockout = 0;
  int listen_fd = 0;
  int fd = 0;
  char *rgot = 0;
  int readsocks = 0;
  fd_set socks;
  struct timeval timeout;
  unsigned i = 0;

  // create a socket on which to listen
  if (ctx->verbose)
    multilog(log, LOG_INFO, "control_thread: creating socket on port %d\n", port);

  listen_fd = sock_create (&port);
  if (listen_fd < 0)  {
    multilog(log, LOG_ERR, "Failed to create socket for control commands: %s\n", strerror(errno));
    free (buffer);
    return;
  }

  while (!quit_threads) {

    // reset the FD set for selecting  
    FD_ZERO(&socks);
    FD_SET(listen_fd, &socks);
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    readsocks = select(listen_fd+1, &socks, (fd_set *) 0, (fd_set *) 0, &timeout);

    // error on select
    if (readsocks < 0) 
    {
      perror("select");
      exit(EXIT_FAILURE);
    }

    // no connections, just ignore
    else if (readsocks == 0) 
    {
    } 

    // accept the connection  
    else 
    {
   
      if (ctx->verbose) 
        multilog(log, LOG_INFO, "control_thread: accepting conection\n");

      fd =  sock_accept (listen_fd);
      if (fd < 0)  {
        multilog(log, LOG_WARNING, "control_thread: Error accepting "
                                        "connection %s\n", strerror(errno));
        break;
      }

      sockin = fdopen(fd,"r");
      if (!sockin)
        multilog(log, LOG_WARNING, "control_thread: error creating input "
                                        "stream %s\n", strerror(errno));


      sockout = fdopen(fd,"w");
      if (!sockout)
        multilog(log, LOG_WARNING, "control_thread: error creating output "
                                        "stream %s\n", strerror(errno));

      setbuf (sockin, 0);
      setbuf (sockout, 0);

      rgot = fgets (buffer, bufsize, sockin);

      if (rgot && !feof(sockin)) {

        buffer[strlen(buffer)-2] = '\0';

        args = buffer;

        // parse the command and arguements
        command = strsep (&args, whitespace);

        if (ctx->verbose)
        {
          multilog(log, LOG_INFO, "control_thread: command=%s\n", command);
          if (args != NULL)
            multilog(log, LOG_INFO, "control_thread: args=%s\n", args);
        }

        if (strcmp(command, "STATE") == 0)
        {
          for (i=0; i<ctx->n_outputs; i++)
          {
            if ((ctx->outputs[i].state == DBDECIDB_ACTIVE) && (ctx->outputs[i].new_state == DBDECIDB_ACTIVE))
              fprintf(sockout, "%x active active\r\n", ctx->outputs[i].key);
            if ((ctx->outputs[i].state == DBDECIDB_ACTIVE) && (ctx->outputs[i].new_state == DBDECIDB_INACTIVE))
              fprintf(sockout, "%x active inactive\r\n", ctx->outputs[i].key);
            if ((ctx->outputs[i].state == DBDECIDB_INACTIVE) && (ctx->outputs[i].new_state == DBDECIDB_ACTIVE))
              fprintf(sockout, "%x inactive active\r\n", ctx->outputs[i].key);
            if ((ctx->outputs[i].state == DBDECIDB_INACTIVE) && (ctx->outputs[i].new_state == DBDECIDB_INACTIVE))
              fprintf(sockout, "%x inactive inactive\r\n", ctx->outputs[i].key);
          }
          fprintf (sockout, "ok\r\n");
        }
      
        else if (strcmp(command, "ACTIVE") == 0) 
        {
          fprintf(sockout, "active:");
          for (i=0; i<ctx->n_outputs; i++)
          {
            if (ctx->outputs[i].state == DBDECIDB_ACTIVE)
              fprintf(sockout, " %x", ctx->outputs[i].key);
          }
          fprintf(sockout, "\r\n");
          fprintf (sockout, "ok\r\n");
        }

        else if (strcmp(command, "INACTIVE") == 0)
        {
          fprintf(sockout, "inactive:");
          for (i=0; i<ctx->n_outputs; i++)
          {
            if (ctx->outputs[i].state == DBDECIDB_INACTIVE)
              fprintf(sockout, " %x", ctx->outputs[i].key);
          }
          fprintf(sockout, "\r\n");
          fprintf (sockout, "ok\r\n");
        }

        else if (strcmp(command, "ACTIVATE") == 0)
        {
          key_t key = 0;
          int activated = 0;
          if (sscanf (args, "%x", &key) != 1) {
            fprintf (sockout, "could not parse key\r\n");
            activated = -1;
          }
          else
          {
            for (i=0; i<ctx->n_outputs; i++)
            {
              if (key == ctx->outputs[i].key)
              {
                if (ctx->outputs[i].new_state == DBDECIDB_ACTIVE)
                {
                  fprintf (sockout, "DB %x already queued to become active\r\n", ctx->outputs[i].key);
                  activated = -1;
                }
                else
                {
                  ctx->outputs[i].new_state = DBDECIDB_ACTIVE;
                  fprintf (sockout, "DB %x activated for next transfer\r\n", ctx->outputs[i].key);
                  activated = 1;
                }
              }
            }
          }  
          if (activated == 0)
            fprintf (sockout, "Could not find DB to match %x\n", key);

          if (activated == 1)
            fprintf (sockout, "ok\r\n");
          else
            fprintf (sockout, "fail\r\n");
        }

        else if (strcmp(command, "DEACTIVATE") == 0)
        {
          key_t key = 0;
          int deactivated = 0;
          if (sscanf (args, "%x", &key) != 1) {
            fprintf (sockout, "could not parse key\r\n");
            deactivated = -1;
          }
          else
          {
            for (i=0; i<ctx->n_outputs; i++)
            {
              if (key == ctx->outputs[i].key)
              {
                if (ctx->outputs[i].new_state == DBDECIDB_INACTIVE)
                {
                  fprintf (sockout, "DB %x already queued to become deactive\r\n", ctx->outputs[i].key);
                  deactivated = -1;
                }
                else
                {
                  ctx->outputs[i].new_state = DBDECIDB_INACTIVE;
                  fprintf (sockout, "DB %x deactivated for next transfer\r\n", ctx->outputs[i].key);
                  deactivated = 1;
                }
              }
            }
          }
          if (deactivated == 0)
            fprintf (sockout, "Could not find DB to match %x\r\n", key);

          if (deactivated == 1)
            fprintf (sockout, "ok\r\n");
          else
            fprintf (sockout, "fail\r\n");
        }

        // QUIT COMMAND, immediately exit 
        else if (strcmp(command, "QUIT") == 0) 
        {
          multilog(log, LOG_INFO, "control_thread: QUIT command received, exiting\n");
          quit_threads = 1;
          fprintf(sockout, "ok\r\n");
        }

        // UNRECOGNISED COMMAND
        else 
        {
          multilog(log, LOG_WARNING, "control_thread: unrecognised command: %s\n", buffer);
          fprintf(sockout, "fail\r\n");
        }
      }
    }

    close(fd);
  }
  close(listen_fd);

  free (buffer);

  if (ctx->verbose)
    multilog(log, LOG_INFO, "control_thread: exiting\n");

}
