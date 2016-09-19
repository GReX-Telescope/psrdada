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
#include <emmintrin.h>

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
	   "dada_zero [options] header_file\n"
     " -k       hexadecimal shared memory key  [default: %x]\n"
     " -r rate  data rate MB/s [default %f]\n"
     " -R rate  data rate MiB/s [default %f]\n"
     " -t secs  length of time to write [default %d s]\n"
     " -z       use zero copy direct shm access\n"
     " -h       print help\n",
     DADA_DEFAULT_BLOCK_KEY, DEFAULT_DATA_RATE, DEFAULT_DATA_RATE, DEFAULT_WRITE_TIME);
}

typedef struct {

  /* header file to use */
  char * header_file;

  /* data rate B/s */
  uint64_t rate;

  /* length of time to write for */
  uint64_t write_time;

  /* transfer/file size */
  uint64_t transfer_size;

  uint64_t bytes_to_copy;

  time_t curr_time;

  time_t prev_time;

  unsigned header_read;

  unsigned verbose;

} dada_zero_t;

#define DADA_ZERODB_INIT { "", 0, 0, 0, 0, 0, 0, 0, 0 }


/*! Pointer to the function that transfers data to/from the target */
int64_t transfer_data (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "transfer_data %p %"PRIu64"\n", data, data_size);
#endif

  /* the dada_zero specific data */
  dada_zero_t* zerodb = (dada_zero_t*) client->context;

  if (!zerodb->header_read) {
    zerodb->header_read = 1;
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "transfer_data: read header\n");
#endif
    return data_size;
  }

#define MEMSET
#ifdef MEMSET
  bzero (data, data_size);
  //memset (data, 0, data_size);
#else
  const unsigned nblock = data_size / (sizeof(__m128i));
  __m128i zero = {0};
  __m128i * p = (__m128i *) data;

  unsigned i;
  for (i=0; i<nblock; i++)
  {
    _mm_stream_si128(&p[i], zero);
  }

#ifdef _DEBUG 
  multilog (client->log, LOG_INFO, "transfer_data: copied %"PRIu64" bytes\n", bytes_copied);
#endif
#endif

  return (int64_t) data_size;
}

int64_t transfer_data_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  return transfer_data(client, data, data_size);
}


/*! Function that closes the data file */
int dada_zero_close (dada_client_t* client, uint64_t bytes_written)
{
  assert (client != 0);

  dada_zero_t* zerodb = (dada_zero_t*) client->context;
  assert (zerodb != 0);

  if (zerodb->verbose)
    multilog (client->log, LOG_INFO, "close: wrote %"PRIu64" bytes\n", bytes_written);

  return 0;
}

/*! Function that opens the data transfer target */
int dada_zero_open (dada_client_t* client)
{

  assert (client != 0);
  assert (client->header != 0);

  dada_zero_t* zerodb = (dada_zero_t*) client->context;
  assert (zerodb != 0);

  uint64_t hdr_size = 0;
  time_t current_time = 0;
  time_t prev_time = 0;
  
  // read the header
  if (fileread (zerodb->header_file, client->header, client->header_size) < 0) {
    multilog (client->log, LOG_ERR, "Could not read header from %s\n", zerodb->header_file);
  }

  if (zerodb->verbose > 1)
  fprintf (stderr, "===========HEADER START=============\n"
                   "%s"
                   "============HEADER END==============\n", client->header);
  zerodb->header_read = 0;

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

  client->header_size = hdr_size;
  client->optimal_bytes = 32 * 1024 * 1024;
  client->transfer_bytes = zerodb->transfer_size;
  zerodb->bytes_to_copy = 0;

  if (ascii_header_set(client->header, "FILE_SIZE", "%"PRIu64, zerodb->transfer_size) < 0)
  {
    multilog (client->log, LOG_WARNING, "Failed to set FILE_SIZE header attribute to %"PRIu64"\n",
              zerodb->transfer_size);
  }

  if (zerodb->verbose)
    multilog (client->log, LOG_INFO, "open: setting filesize %"PRIu64"\n", zerodb->transfer_size);

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_zero_t zerodb = DADA_ZERODB_INIT;

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

  int arg = 0;

  while ((arg=getopt(argc,argv,"hk:r:R:t:vz")) != -1)
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

  log = multilog_open ("dada_zero", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  zerodb.rate = (uint64_t) rate * rate_base;
  zerodb.write_time = write_time;

  zerodb.transfer_size = write_time * zerodb.rate;
  zerodb.header_file = strdup(header_file);
  zerodb.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = dada_zero_open;

  if (zero_copy)
    client->io_block_function = transfer_data_block;

  client->io_function = transfer_data;
  client->close_function = dada_zero_close;
  client->direction = dada_client_writer;

  client->context = &zerodb;

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

