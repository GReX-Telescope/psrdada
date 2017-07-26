#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_generator.h"
#include "ascii_header.h"

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

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
     " -r seed    random number generator seed [default 0]\n"
     " -h         print help\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -s secs    generate secs worth of data\n"
     " -v         verbose\n",
     DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  // header file to use
  char * header_file;

  // flags
  int seed;

  // flag for reading the header 
  unsigned header_read;

  unsigned verbose;

  uint64_t bytes_per_second;

  unsigned seconds;

  unsigned nchan;

  unsigned nant;

  StopWatch wait_sw;

  char * rand_buffer;

  size_t rand_buffer_size;

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


  const unsigned ndim = 2;
  if (ctx->verbose > 1)
    multilog (client->log, LOG_INFO, "io_block: nant=%u nchan=%u ndim=%u\n", ctx->nant, ctx->nchan, ndim);

  const uint64_t nsamp = data_size / (ctx->nant * ctx->nchan * ndim);
  const uint64_t nval = data_size / ctx->nant;
  const uint64_t rand_bytes = nval;

  if (ctx->rand_buffer_size < rand_bytes)
  {
    ctx->rand_buffer = (char *) malloc (rand_bytes);
    ctx->rand_buffer_size = rand_bytes;
  }

  if (ctx->verbose > 1)
    multilog (client->log, LOG_INFO, "io_block: nsamp=%lu nval=%lu\n", nsamp, nval);

  // generate a random sequence for nchan * ndim * nsamp
  fill_gaussian_chars (ctx->rand_buffer, nval, 8, 500);

  //#ifdef USE_FST
  int8_t * ptr8 = (int8_t *) data;
  
  unsigned ichan, iant, isamp, idim;
  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (ichan=0; ichan<ctx->nchan; ichan++)
    {
      const unsigned offset = ichan * nsamp * ndim + isamp*ndim;
      for (iant=0; iant<ctx->nant; iant++)
      {
        for (idim=0; idim<ndim; idim++)
        {
          ptr8[0] = (int8_t) ctx->rand_buffer[offset + idim];
          ptr8++;
        } 
      }
    }
  }

  double data_secs = ((double) data_size) / ((double) ctx->bytes_per_second);
  double data_usecs = data_secs * 1000000;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "io_block: data_secs=%lf\n", data_secs);

  // delay the requisiste time
  StopWatch_Delay(&(ctx->wait_sw), data_usecs);

  // start it again 
  StopWatch_Start(&(ctx->wait_sw));

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

  StopWatch_Stop(&(ctx->wait_sw));
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

  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no NCHAN\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NANT", "%u", &(ctx->nant)) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no NANT\n");
    return -1;
  }

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
  
  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: starting stopwatch\n");
  StopWatch_Start(&(ctx->wait_sw));

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: setting transfer_bytes to %"PRIu64"\n", client->transfer_bytes);

  ctx->rand_buffer_size = 0;
  ctx->rand_buffer = NULL;

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

  ctx.seconds = 10;
  ctx.seed = 0;

  while ((arg=getopt(argc,argv,"hk:r:s:v")) != -1)
    switch (arg) {

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'r':
      ctx.seed = atoi (optarg);
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

  // initialise data rate timing library 
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

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

