#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_generator.h"
#include "ascii_header.h"

#include "stopwatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

#define USE_FST

static void fsleep (double seconds)
{
  struct timeval t ;

  t.tv_sec = seconds;
  seconds -= t.tv_sec;
  t.tv_usec = seconds * 1e6;
  select (0, 0, 0, 0, &t) ;
}

void usage()
{
  fprintf (stdout,
	   "mopsr_numdb [options] header_file\n"
     " -a nant    number of antenna [default 4]\n"
     " -c schan   start channel number [default 0]\n"
     " -e type    encode type where: t=time a=ant c=chan\n"
     " -h         print help\n"
     " -i iant    starting index number for antenna [default 0]\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -n nchan   number of channels [default 40]\n"
     " -s secs    generate secs worth of data\n"
     " -v         verbose\n",
     DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  // header file to use
  char * header_file;

  // index of first channel
  unsigned schan;

  // number of channels
  unsigned nchan;

  // number of antennae / modules
  unsigned nant;

  // start antenna number for encoding
  unsigned sant;

  // flags
  char encode_seq;
  char encode_ant;
  char encode_chan;

  // flag for reading the header 
  unsigned header_read;

  unsigned verbose;

  uint64_t seq;

  uint64_t bytes_per_second;

  unsigned seconds;

  stopwatch_t wait_sw;

} mopsr_numdb_t;

#define DADA_NUMDB_INIT { "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

int64_t mopsr_numdb_io (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_numdb_t* ctx = (mopsr_numdb_t *) client->context;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io: dada=%p data_size=%"PRIu64"\n", 
              data, data_size);
  if (!ctx->header_read)
  {
    if (ctx->verbose)
    {
      multilog (client->log, LOG_INFO, "io: read 'header'\n");
    }
    ctx->header_read = 1;
  }
  return data_size;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t mopsr_numdb_io_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  mopsr_numdb_t* ctx = (mopsr_numdb_t *) client->context;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: dada=%p data_size=%"PRIu64" "
              "block_id=%"PRIu64"\n", data, data_size, block_id);

  uint16_t * ptr16 = (uint16_t *) data;
  const unsigned ndim = 2;
  const uint64_t nsamp = data_size / (ctx->nant * ctx->nchan * ndim);

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: nant=%d nchan=%d ndim=%d nsamp=%lu\n", 
              ctx->nant, ctx->nchan, ndim, nsamp);

  // encoding is FST order
  unsigned ichan, iant;
  uint64_t isamp;

  uint16_t val, aval, cval;
  uint32_t seq_start = ctx->seq;

#ifdef USE_FST
  for (ichan=0; ichan<ctx->nchan; ichan++)
  {
    cval = (uint16_t) ichan;
    for (iant=0; iant<ctx->nant; iant++)
    {
      aval = (uint16_t) ctx->sant + iant;
      //fprintf (stderr, "[%d][%d] nsamp=%lu aval=%u\n", ichan, iant, nsamp, aval);
      ctx->seq = seq_start;
      for (isamp=0; isamp<nsamp; isamp++)
      {
        if (ctx->encode_seq)
        {
          ptr16[0] = (uint16_t) ctx->seq;
          ctx->seq = (ctx->seq + 1) % 65536;
        }
        else if (ctx->encode_ant)
          ptr16[0] = aval;
        else if (ctx->encode_chan)
          ptr16[0] = cval;
        else
          multilog (client->log, LOG_WARNING, "io_block: flag mismatch\n");

        ptr16++;
      }
    }
  }
#else

  // TFS
  ctx->seq = seq_start;
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      cval = (uint16_t) ichan;
      for (iant=0; iant<ctx->nant; iant++)
      {
        aval = (uint16_t) ctx->sant + iant;
        if (ctx->encode_seq)
        {
          ptr16[0] = (uint16_t) ctx->seq;
        }
        else if (ctx->encode_ant)
          ptr16[0] = aval;
        else if (ctx->encode_chan)
          ptr16[0] = cval;
        else
          multilog (client->log, LOG_WARNING, "io_block: flag mismatch\n");

        ptr16++;
      }
    }
    ctx->seq = (ctx->seq + 1) % 65536;
  }
#endif

  double data_secs = ((double) data_size) / ((double) ctx->bytes_per_second);

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: data_secs=%lf\n", data_secs);

  // delay the requisiste time
  DelayTimer(&(ctx->wait_sw), data_secs);

  // start it again 
  StartTimer(&(ctx->wait_sw));

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: processed %"PRIu64" bytes\n", data_size);

  return (int64_t) data_size;
}

/*! Function that closes the data file */
int mopsr_numdb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_numdb_t * ctx = (mopsr_numdb_t *) client->context;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "close: bytes_written=%"PRIu64"\n", bytes_written);

  StopTimer(&(ctx->wait_sw));
  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "close: stopwatch stopped\n");

  return 0;
}

/*! Function that opens the data transfer target */
int mopsr_numdb_open (dada_client_t* client)
{
  mopsr_numdb_t* ctx = (mopsr_numdb_t*) client->context;

  // read the header file
  if (fileread (ctx->header_file, client->header, client->header_size) < 0) 
  {
    multilog (client->log, LOG_ERR, "Could not read header from %s\n", ctx->header_file);
  }

  // set flag so that the io function knows to expect a header read...
  ctx->header_read = 0;

  

  uint64_t hdr_size;
  if (ascii_header_get (client->header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1)
  {
    multilog (client->log, LOG_WARNING, "Header with no HDR_SIZE\n");
    hdr_size = DADA_DEFAULT_HEADER_SIZE;
  }

  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {     
    multilog (client->log, LOG_ERR, "Header with no BYTES_PER_SECOND\n");
    return -1;
  }
  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: BYTES_PER_SECOND=%"PRIu64"\n", ctx->bytes_per_second);

  // ensure that the incoming header fits in the client header buffer
  if (hdr_size > client->header_size) {
    multilog (client->log, LOG_ERR, "HDR_SIZE=%u > Block size=%"PRIu64"\n",
              hdr_size, client->header_size);
    return -1;
  }

  client->header_size = hdr_size;

  client->transfer_bytes = ctx->seconds * ctx->bytes_per_second;
  if (ascii_header_set (client->header, "TRANSFER_SIZE", "%"PRIu64, client->transfer_bytes) < 0) 
  {
    multilog (client->log, LOG_ERR, "open: failed to write TRANSFER_SIZE=%"PRIu64" to header\n", client->transfer_bytes);
    return -1;
  }
  
  ctx->seq = 0;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: starting stopwatch\n");
  StartTimer(&(ctx->wait_sw));

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: setting transfer_bytes to %"PRIu64"\n", client->transfer_bytes);

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_numdb_t ctx = DADA_NUMDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* Quit flag */
  char quit = 0;

  // hexadecimal shared memory key
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  ctx.schan = 0;
  ctx.nchan = 40;
  ctx.nant = 8;
  ctx.sant = 0;
  ctx.seconds = 10;
  ctx.encode_ant  = 1;
  ctx.encode_chan = 0;
  ctx.encode_seq  = 0;

  while ((arg=getopt(argc,argv,"a:c:e:hi:k:n:s:v")) != -1)
    switch (arg) {

    case 'a':
      ctx.nant = atoi(optarg);
      break;

    case 'c':
      ctx.schan = atoi(optarg);
      break;

    case 'e':
      ctx.encode_seq = (optarg[0] == 't');
      ctx.encode_ant = (optarg[0] == 'a');
      ctx.encode_chan = (optarg[0] == 'c');
      break;

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'i':
      ctx.sant = atoi(optarg);
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'n':
      ctx.nchan = atoi(optarg);
      break;

    case 's':
      ctx.seconds = atoi(optarg);
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return EXIT_FAILURE;
    }

  if ((argc - optind) != 1) {
    fprintf (stderr, "Error: a header file must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  ctx.header_file = strdup(argv[optind]);

  // test that the header file can be read
  FILE* fptr = fopen (ctx.header_file, "r");
  if (!fptr)
  {
    fprintf (stderr, "Error: could not open '%s' for reading: %s\n", 
             ctx.header_file, strerror(errno));
    return(EXIT_FAILURE);
  }
  fclose(fptr);

  if (verbose)
    fprintf (stdout, "Using header file: %s\n", ctx.header_file);

  log = multilog_open ("mopsr_numdb", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  ctx.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = mopsr_numdb_open;
  client->io_function = mopsr_numdb_io;
  client->io_block_function = mopsr_numdb_io_block;
  client->close_function = mopsr_numdb_close;

  client->direction = dada_client_writer;

  client->context = &ctx;

  if (dada_client_write (client) < 0) {
    multilog (log, LOG_ERR, "Error during transfer\n");
    return -1;
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

