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

int64_t dbreblockdb_write_block_SFT_to_STF (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbreblockdb [options] in_key out_key\n"
           "              data type change from 32-bit float to 16-bit uint\n"
           " -r factor    reblock by factor from input to output\n"
           " -s           1 transfer, then exit\n"
           " -z           use zero copy transfers\n"
           " -v           verbose mode\n"
           " in_key       DADA key for input data block\n"
           " out_key      DADA key for output data blocks\n");
}

typedef struct 
{
  dada_hdu_t *  hdu;
  key_t         key;
  uint64_t      block_size;
  uint64_t      bytes_written;
  unsigned      block_open;
  char *        curr_block;
} mopsr_dbreblockdb_hdu_t;


typedef struct {

  mopsr_dbreblockdb_hdu_t output;

  // number of bytes read
  uint64_t bytes_in;

  // number of bytes written
  uint64_t bytes_out;

  unsigned reblock_factor;
  unsigned reblock_curr;

  unsigned bitrate_factor;

  // verbose output
  int verbose;

  unsigned int nsig;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit_in;
  unsigned int nbit_out;
  
  float scale;

  unsigned quit;

  char order[4];


} mopsr_dbreblockdb_t;

#define DADA_DBREBLOCKDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

/*! Function that opens the data transfer target */
int dbreblockdb_open (dada_client_t* client)
{
  // the mopsr_dbreblockdb specific data
  mopsr_dbreblockdb_t* ctx = (mopsr_dbreblockdb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbreblockdb_open()\n");

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

  float bw;
  if (ascii_header_get (client->header, "BW", "%f", &bw) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BW\n");
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
    if ((strcmp(ctx->order, "SFT") == 0) && client->io_block_function)
    {
      multilog (log, LOG_INFO, "open: changing order from SFT to STF\n");
      client->io_block_function = dbreblockdb_write_block_SFT_to_STF;
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

  // since we are inverting the channel order, flip the bandwidth
  bw *= -1;
  if (ascii_header_set (header, "BW", "%f", bw) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write BW=%f to header\n", bw);
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

  new_resolution = (resolution * ctx->reblock_factor) / ctx->bitrate_factor;
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
  ctx->scale = -1;

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

int dbreblockdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbreblockdb_t* ctx = (mopsr_dbreblockdb_t*) client->context;
  
  multilog_t* log = client->log;

  mopsr_dbreblockdb_hdu_t * o = 0;

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
      multilog (log, LOG_ERR, "dbreblockdb_close: ipcio_close_block_write failed\n");
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
    multilog (log, LOG_ERR, "dbreblockdb_close: cannot unlock DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbreblockdb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbreblockdb_t* ctx = (mopsr_dbreblockdb_t*) client->context;

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

//
// reblock the data into a larger block size. This involes:
//   transpose input from SFT to STF 
//   reblocking from input nsamps to output nsamps
//   reordering channels from (Flo -> Fhi) to (Fhi -> Flo)
// 
int64_t dbreblockdb_write_block_SFT_to_STF (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  mopsr_dbreblockdb_t* ctx = (mopsr_dbreblockdb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp_in  = data_size / (ctx->nsig * ctx->ndim * ctx->nchan * (ctx->nbit_in/8));
  const uint64_t nsamp_out = ctx->output.block_size / (ctx->nsig * ctx->ndim * ctx->nchan * (ctx->nbit_out/8));

  const uint64_t sig_stride_in  = nsamp_in * ctx->ndim * ctx->nchan;
  const uint64_t sig_stride_out = nsamp_out * ctx->ndim * ctx->nchan;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF: sig_stride_in=%"PRIu64", sig_stride_out=%"PRIu64"\n", sig_stride_in, sig_stride_out);

  //  assume 32-bit, detected (ndim == 1) data
  float * in = (float *) in_data;
  // since we are converting from 32-bit float to 16-bit unsigned int
  uint8_t * out; 
 
  //const uint64_t nsamp = data_size / (ctx->nsig * ctx->ndim * ctx->nchan); 
  //const size_t sig_stride = nsamp * ctx->ndim * ctx->nchan;
  uint64_t out_block_id;
  unsigned isig, isamp, ichan, ochan;

  if (ctx->scale < 0)
  {
    // compute the mean power level across all beams and channels
    uint64_t total_power = 0;
    for (isig=0; isig<ctx->nsig; isig++)
    {
      for (ichan=0; ichan<ctx->nchan; ichan++)
      {
        for (isamp=0; isamp<nsamp_in; isamp++)
        {
          total_power += (uint64_t) in[ichan * nsamp_in + isamp];
        }
      }
      in += sig_stride_in;
    }

    // reset the pointer
    in = (float *) in_data;

    uint64_t avg_power = total_power / (ctx->nsig * ctx->nchan * nsamp_in);

    // we want the average power to be around 12-bits of unsigned integer (4096)
    //ctx->scale = 4096.0f / avg_power;
    // we want the average power to be around the 6-bit
    ctx->scale = 64.0f / avg_power;

    multilog (log, LOG_INFO, "write_block_SFT_to_STF: total_power=%"PRIu64" avg_power=%"PRIu64" scale=%e\n", total_power, avg_power, ctx->scale);
  }

  if (!ctx->output.block_open)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_STF: ipcio_open_block_write()\n");
    ctx->output.curr_block = ipcio_open_block_write(ctx->output.hdu->data_block, &out_block_id);
    if (!ctx->output.curr_block)
    {
      multilog (log, LOG_ERR, "write_block_SFT_to_STF: ipcio_open_block_write failed %s\n", strerror(errno));
      return -1;
    }
    ctx->output.block_open = 1;
    ctx->output.bytes_written = 0;
    ctx->reblock_curr = 0;
    out = (uint8_t *) ctx->output.curr_block;
  }
  else
    out = (uint8_t *) ctx->output.curr_block;

  // advance forward on the datablock pointer for the current input block offset
  //multilog (log, LOG_INFO, "write_block_SFT_to_STF: incrementing output pointer by %"PRIu64" samples\n", nsamp_in * ctx->reblock_curr);
  out += (nsamp_in * ctx->nchan * ctx->reblock_curr);

  //multilog (log, LOG_INFO, "write_block_SFT_to_STF: in_stride=%"PRIu64" out_stride=%"PRIu64"\n", sig_stride_in, sig_stride_out);

  for (isig=0; isig<ctx->nsig; isig++)
  {
    //multilog (log, LOG_INFO, "write_block_SFT_to_STF: transposing signal %u\n", isig);
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      ochan = (ctx->nchan - 1) - ichan;
      //multilog (log, LOG_INFO, "write_block_SFT_to_STF: ichan=%u ochan=%u [%ld]\n", ichan, ochan, out - ((uint16_t *) ctx->output.curr_block));
      for (isamp=0; isamp<nsamp_in; isamp++)
      {
        //multilog (log, LOG_INFO, "write_block_SFT_to_STF: ioff=%u off=%u\n", ichan * nsamp_in + isamp, isamp * ctx->nchan + ochan);
        out[isamp * ctx->nchan + ochan] = (uint8_t) (in[ichan * nsamp_in + isamp] * ctx->scale);
        //if (isig == 0 && ichan == 0 && isamp < 10)
        //  multilog (log, LOG_INFO, "write_block_SFT_to_STF: [%u] out=%"PRIu16" in=%e scale=%e\n", isamp, out[isamp * ctx->nchan + ochan], in[ichan * nsamp_in + isamp], ctx->scale);
      }
    }
    out += sig_stride_out;
    in += sig_stride_in;
  }

  ctx->reblock_curr++;
  ctx->output.bytes_written += data_size / ctx->bitrate_factor;

  if (ctx->output.bytes_written > ctx->output.block_size)
    multilog (log, LOG_ERR, "write_block_SFT_to_STF output block overrun by "
              "%"PRIu64" bytes\n", ctx->output.bytes_written - ctx->output.block_size);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF bytes_written=%"PRIu64", "
              "block_size=%"PRIu64"\n", ctx->output.bytes_written, ctx->output.block_size);

  // check if the output block is now full
  if (ctx->output.bytes_written >= ctx->output.block_size)
  {
    if (ctx->verbose > 1)
      multilog (log, LOG_INFO, "write_block_SFT_to_STF block now full bytes_written=%"PRIu64", block_size=%"PRIu64"\n", ctx->output.bytes_written, ctx->output.block_size);

    // check if this is the end of data
    if (client->transfer_bytes && ((ctx->bytes_in + data_size) == client->transfer_bytes))
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "write_block_SFT_to_STF update_block_write written=%"PRIu64"\n", ctx->output.bytes_written);
      if (ipcio_update_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_STF ipcio_update_block_write failed\n");
         return -1;
      }
    }
    else
    {
      if (ctx->verbose > 1)
        multilog (log, LOG_INFO, "write_block_SFT_to_STF close_block_write written=%"PRIu64"\n", ctx->output.bytes_written);
      if (ipcio_close_block_write (ctx->output.hdu->data_block, ctx->output.bytes_written) < 0)
      {
        multilog (log, LOG_ERR, "write_block_SFT_to_STF ipcio_close_block_write failed\n");
        return -1;
      }
    }
    ctx->output.block_open = 0;
    ctx->output.bytes_written = 0;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += (data_size / ctx->bitrate_factor);

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "write_block_SFT_to_STF read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbreblockdb_t dbreblockdb = DADA_DBREBLOCKDB_INIT;

  mopsr_dbreblockdb_t * ctx = &dbreblockdb;

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
  ctx->reblock_factor = 1;
  ctx->bitrate_factor = 4;

  while ((arg=getopt(argc,argv,"dp:r:svz")) != -1)
  {
    switch (arg) 
    {
      
      case 'd':
        daemon = 1;
        break;

      case 'r':
        ctx->reblock_factor = atoi(optarg);
        break;

      case 's':
        single_transfer = 1;
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

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 2)
  {
    fprintf(stderr, "mopsr_dbreblockdb: 2 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  if (ctx->verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind]);
  if (sscanf (argv[optind], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbreblockdb: could not parse in key from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (ctx->verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &(ctx->output.key)) != 1) {
    fprintf (stderr, "mopsr_dbreblockdb: could not parse out key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  log = multilog_open ("mopsr_dbreblockdb", 0);

  multilog_add (log, stderr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbreblockdb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbreblockdb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  ctx->output.hdu = dada_hdu_create (log);
  dada_hdu_set_key (ctx->output.hdu, ctx->output.key);
  if (dada_hdu_connect (ctx->output.hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", ctx->output.key);
    return -1;
  }
  ctx->output.curr_block = 0;
  ctx->output.bytes_written = 0;
  ctx->output.block_open = 0;
  ctx->output.block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) ctx->output.hdu->data_block);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: ctx->output.block_size=%"PRIu64"\n", ctx->output.block_size);
  if (zero_copy && (ctx->output.block_size != (block_size * ctx->reblock_factor / ctx->bitrate_factor)))
  {
    multilog (log, LOG_ERR, "output block size [%"PRIu64"]  must be input block size [%"PRIu64"] * reblocking factor [%u]\n", ctx->output.block_size, block_size, ctx->reblock_factor);
   return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbreblockdb_open;
  client->io_function   = dbreblockdb_write;

  if (zero_copy)
  {
    client->io_block_function = dbreblockdb_write_block_SFT_to_STF;
  }
  else
  {
    multilog (log, LOG_ERR, "currently zero copy must be used\n");
    return EXIT_FAILURE;
  }

  client->close_function = dbreblockdb_close;
  client->direction      = dada_client_reader;

  client->context = &dbreblockdb;
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
