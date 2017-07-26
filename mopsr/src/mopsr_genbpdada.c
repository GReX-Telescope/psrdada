#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_generator.h"
#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

/* #define _DEBUG 1 */

#define DEFAULT_DATA_RATE 64.000
#define DEFAULT_WRITE_TIME 10 

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
	   "dada_genbpdada [options] header_file\n"
     " -b bytes   total number of bytes to generate\n"
     " -m mean    use mean for gaussian data\n"
     " -s stddev  use stddev for gaussian data\n"
     " -k         hexadecimal shared memory key  [default: %x]\n"
     " -r rate    data rate MB/s [default %f]\n"
     " -R rate    data rate MiB/s [default %f]\n"
     " -t secs    length of time to write [default %d s]\n"
     " -z         use zero copy direct shm access\n"
     " -h         print help\n",
     DADA_DEFAULT_BLOCK_KEY, DEFAULT_DATA_RATE, DEFAULT_DATA_RATE, DEFAULT_WRITE_TIME);
}

typedef struct {

  /* header file to use */
  char * header_file;

  /* number of beam */
  unsigned nbeam;

  /* number of channels */
  unsigned nchan;

  /* number of bits per sample */
  unsigned nbit;

  /* number of polarisations */
  unsigned npol;

  /* data rate B/s */
  uint64_t rate;

  /* mean of gaussian data */
  double mean;

  /* stddev of gaussian data */
  double stddev;

  /* length of time to write for */
  uint64_t write_time;

  /* pre generated data to write */
  char * data;

  /* length of each array */ 
  uint64_t data_size;

  /* transfer/file size */
  uint64_t transfer_size;

  uint64_t bytes_to_copy;

  time_t curr_time;

  time_t prev_time;

  unsigned header_read;

  unsigned verbose;

} mopsr_genbpdada_t;

#define DADA_JUNKDB_INIT { "", 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0, 0, 0 }

/*! Pointer to the function that transfers data to/from the target */
int64_t transfer_data (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "transfer_data %p %"PRIu64"\n", data, data_size);
#endif

  /* the mopsr_genbpdada specific data */
  mopsr_genbpdada_t* ctx = (mopsr_genbpdada_t*) client->context;

  if (!ctx->header_read) {
    ctx->header_read = 1;
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "transfer_data: read header\n");
#endif
    return data_size;
  }

  uint64_t bytes_copied = 0;
  uint64_t bytes = 0;

  while (bytes_copied < data_size) {

    if (ctx->bytes_to_copy == 0) {
      fsleep(0.1);
    } 

    ctx->prev_time = ctx->curr_time;
    ctx->curr_time = time(0);
    if (ctx->curr_time > ctx->prev_time) {
      ctx->bytes_to_copy += ctx->rate;
    }

    if (ctx->bytes_to_copy) {

      if (data_size-bytes_copied < ctx->bytes_to_copy)
        bytes = data_size-bytes_copied;
      else 
        bytes = ctx->bytes_to_copy-bytes_copied;

      if (bytes > ctx->data_size)
        bytes = ctx->data_size;

#ifdef _DEBUG
      multilog (client->log, LOG_INFO, "[%"PRIu64" / %"PRIu64"] bytes=%"PRIu64", btc=%"PRIu64"\n", bytes_copied, data_size,bytes, ctx->bytes_to_copy);
#endif
      memcpy (data+bytes_copied, ctx->data, bytes);
      bytes_copied += bytes;
      ctx->bytes_to_copy -= bytes;
    }

  }
#ifdef _DEBUG 
  multilog (client->log, LOG_INFO, "transfer_data: copied %"PRIu64" bytes\n", bytes_copied);
#endif

  return (int64_t) bytes_copied;
}

int64_t transfer_data_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  mopsr_genbpdada_t* ctx = (mopsr_genbpdada_t*) client->context;

  memcpy (data, ctx->data, data_size );
  int64_t bytes_copied = (int64_t) data_size;

/*
  uint64_t bytes;

  while (bytes_copied < data_size)
  {
    if (ctx->bytes_to_copy == 0) {
      fsleep(0.1);
    }

    ctx->prev_time = ctx->curr_time;
    ctx->curr_time = time(0);
    if (ctx->curr_time > ctx->prev_time) {
      ctx->bytes_to_copy += ctx->rate;
    }

    if (ctx->bytes_to_copy) 
    {
  
      if (data_size-bytes_copied < ctx->bytes_to_copy)
        bytes = data_size-bytes_copied;
      else
        bytes = ctx->bytes_to_copy-bytes_copied;
  
      if (bytes > ctx->data_size)
        bytes = ctx->data_size;
  
      bytes_copied += bytes;
      ctx->bytes_to_copy -= bytes;
    }
  }
  */

  return (int64_t) bytes_copied;
}


/*! Function that closes the data file */
int mopsr_genbpdada_close (dada_client_t* client, uint64_t bytes_written)
{
  assert (client != 0);

  mopsr_genbpdada_t* ctx = (mopsr_genbpdada_t*) client->context;
  assert (ctx != 0);

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "close: wrote %"PRIu64" bytes\n", bytes_written);

  free(ctx->data);

  return 0;
}

/*! Function that opens the data transfer target */
int mopsr_genbpdada_open (dada_client_t* client)
{
  assert (client != 0);
  assert (client->header != 0);

  mopsr_genbpdada_t* ctx= (mopsr_genbpdada_t*) client->context;
  assert (ctx != 0);

  uint64_t hdr_size = 0;
  time_t current_time = 0;
  time_t prev_time = 0;
  
  // read the header
  if (fileread (ctx->header_file, client->header, client->header_size) < 0) {
    multilog (client->log, LOG_ERR, "Could not read header from %s\n", ctx->header_file);
  }

  ctx->header_read = 0;

  if (ascii_header_get (client->header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1)
  {
    multilog (client->log, LOG_WARNING, "Header with no HDR_SIZE\n");
    hdr_size = DADA_DEFAULT_HEADER_SIZE;
  }

  // ensure that the incoming header fits in the client header buffer
  if (hdr_size > client->header_size) {
    multilog (client->log, LOG_ERR, "HDR_SIZE=%u > Block size=%"PRIu64"\n",
	      hdr_size, client->header_size);
    return -1;
  }

  if (ascii_header_get (client->header, "NBEAM", "%u", &(ctx->nbeam)) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no NBEAM\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  { 
    multilog (client->log, LOG_ERR, "Header with no NCHAN\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NPOL", "%u", &(ctx->npol)) != 1)
  { 
    multilog (client->log, LOG_ERR, "Header with no NPOL\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NBIT", "%u", &(ctx->nbit)) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no NBIT\n");
    return -1;
  }

  client->header_size = hdr_size;
  client->optimal_bytes = 32 * 1024 * 1024;
  client->transfer_bytes = ctx->transfer_size;
  ctx->bytes_to_copy = 0;

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "FILE_SIZE=%"PRIu64"\n", ctx->transfer_size);
  if (ascii_header_set(client->header, "FILE_SIZE", "%"PRIu64, ctx->transfer_size) < 0)
  {
    multilog (client->log, LOG_WARNING, "Failed to set FILE_SIZE header attribute to %"PRIu64"\n",
              ctx->transfer_size);
  }

  if (ctx->verbose)
    multilog (client->log, LOG_INFO, "open: setting transfer_bytes=%"PRIu64"\n", ctx->transfer_size);

  // seed the RNG
  srand ( time(NULL) );

  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) client->data_block);
  ctx->data_size = block_size;
  ctx->data = (char *) malloc (sizeof(char) * block_size);
  multilog (client->log, LOG_INFO, "open: setting beam_size=%"PRIu64"\n", block_size);

  uint64_t beam_size = ctx->data_size / ctx->nbeam;
  unsigned ibeam;
  for (ibeam=0; ibeam<ctx->nbeam; ibeam++)
  {
    double offset = (ibeam % 2) ? -10 : 10;
    double mean = ctx->mean + offset;
    double stddev = 10;
    multilog (client->log, LOG_INFO, "open: generating beam %u gaussian data %"PRIu64" mean=%lf stdev=%lf\n", ibeam, beam_size, mean, stddev);
    fill_gaussian_data(ctx->data + (ibeam * beam_size), beam_size, mean, stddev);
    multilog (client->log, LOG_INFO, "open: data generated\n");
  }

  multilog (client->log, LOG_INFO, "open: data generated\n");

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_genbpdada_t genbpdada = DADA_JUNKDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* header to use for the data block */
  char * header_file = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  /* data rate [MB/s | MiB/s] */
  float rate = DEFAULT_DATA_RATE;

  /* write time [s] */
  unsigned write_time = DEFAULT_WRITE_TIME;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* zero copy direct shm access */
  char zero_copy = 0;

  uint64_t rate_base = 1024*1024;

  int64_t total_bytes = -1;

  int arg = 0;

  genbpdada.mean = 127;
  genbpdada.stddev = 10;

  while ((arg=getopt(argc,argv,"b:hk:m:r:R:s:t:vz")) != -1)
    switch (arg) {

    case 'b':
      if (sscanf(optarg, "%"PRIi64, &total_bytes) != 1)
      {
        fprintf (stderr, "ERROR: could not parse total_bytes from %s\n", optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

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

    case 'm':
      if (sscanf (optarg, "%lf", &(genbpdada.mean)) != 1) {
        fprintf (stderr, "ERROR: could not parse mean from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'r':
      if (sscanf (optarg, "%f", &rate) != 1) {
        fprintf (stderr,"ERROR: could not parse data rate from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      rate_base = 1024*1024;
      break;

    case 'R':
      if (sscanf (optarg, "%f", &rate) != 1) {
        fprintf (stderr,"ERROR: could not parse data rate from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      rate_base = 1000000;
      break;

    case 's':
      if (sscanf (optarg, "%lf", &(genbpdada.stddev)) != 1) {
        fprintf (stderr, "ERROR: could not parse stddev from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 't':
      if (sscanf (optarg, "%u", &write_time) != 1) {
        fprintf (stderr,"ERROR: could not parse write time from %s\n",optarg);
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
      return EXIT_FAILURE;
    }

  if ((argc - optind) != 1) {
    fprintf (stderr, "Error: a header file must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  header_file = strdup(argv[optind]);

  /* test that the header file can be read */
  FILE* fptr = fopen (header_file, "r");
  if (!fptr) {
    fprintf (stderr, "Error: could not open '%s' for reading: %s\n", header_file, strerror(errno));
    return(EXIT_FAILURE);
  }
  fclose(fptr);

  log = multilog_open ("mopsr_genbpdada", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  // placeholder for now
  genbpdada.data_size = 1024;

  genbpdada.rate = (uint64_t) rate * rate_base;
  genbpdada.write_time = write_time;
  genbpdada.transfer_size = write_time * genbpdada.rate;
  fprintf (stderr, "transfer_size=%lu\n", genbpdada.transfer_size);
  if (total_bytes > 0)
    genbpdada.transfer_size = (uint64_t) total_bytes;
  genbpdada.header_file = strdup(header_file);
  genbpdada.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = mopsr_genbpdada_open;

  if (zero_copy)
    client->io_block_function = transfer_data_block;

  client->io_function = transfer_data;
  client->close_function = mopsr_genbpdada_close;
  client->direction = dada_client_writer;

  client->context = &genbpdada;

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

