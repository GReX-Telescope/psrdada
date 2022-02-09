#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_generator.h"
#include "ascii_header.h"
#include "tmutil.h"
#include "futils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

#define DEFAULT_DATA_RATE 64.000
#define DEFAULT_WRITE_TIME 10 

void usage()
{
  fprintf (stdout,
	   "dada_junkdb [options] header_file\n"
     " -b bytes   total number of bytes to generate\n"
     " -c char    fill data with char\n"
     " -g         write gaussian distributed data\n"
     " -m mean    use mean for gaussian data\n"
     " -n         skip writing to data to ring buffer, for performance testing only\n"
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

  /* data rate B/s */
  uint64_t rate;

  /* generate gaussian data ? */
  unsigned write_gaussian;

  /* mean of gaussian data */
  double mean;

  /* stddev of gaussian data */
  double stddev;

  char fill_char;

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

  unsigned perform_memcpy;

} dada_junkdb_t;

#define DADA_JUNKDB_INIT { "", 0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0, 0, 0, 1 }


/*! Pointer to the function that transfers data to/from the target */
int64_t transfer_data (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "transfer_data %p %"PRIu64"\n", data, data_size);
#endif

  /* the dada_junkdb specific data */
  dada_junkdb_t* junkdb = (dada_junkdb_t*) client->context;

  if (!junkdb->header_read) {
    junkdb->header_read = 1;
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "transfer_data: read header\n");
#endif
    return data_size;
  }

  // copy data_size bytes at the specified rate into data  
  int64_t bytes_remaining = (int64_t) data_size;
  uint64_t bytes_copied = 0;
  uint64_t bytes = 0;

  while (bytes_remaining > 0)
  {
    if (junkdb->bytes_to_copy == 0) {
      float_sleep(0.1);
    } 

    // busy loop wait until the next second ticks over, then 
    // increment by bytes_per_second
    junkdb->prev_time = junkdb->curr_time;
    junkdb->curr_time = time(0);

    if (junkdb->curr_time > junkdb->prev_time)
    {
      //fprintf (stderr, "incrementing %lu by %lu\n", junkdb->bytes_to_copy, junkdb->rate);
      junkdb->bytes_to_copy += junkdb->rate;
    }

    // if we have some data to copy now (due to rate)
    if (junkdb->bytes_to_copy > 0) 
    {
      // how much data to memcpy into the ring buffer
      if (bytes_remaining < junkdb->bytes_to_copy)
        bytes = bytes_remaining;
      else 
        bytes = junkdb->bytes_to_copy;

      // if the number of bytes to be copied is more than our
      // internal buffer, just truncate it
      if (bytes > junkdb->data_size)
        bytes = junkdb->data_size;

      // copy
      if (junkdb->perform_memcpy)
        memcpy (data + bytes_copied, junkdb->data, bytes);

      // increment 
      bytes_copied += bytes;            // destination offset
      junkdb->bytes_to_copy -= bytes;   // data_rate offset
      bytes_remaining -= bytes;         // bytes left to copy to destination

#ifdef _DEBUG
      fprintf(stderr, "bytes=%lu bytes_copied=%lu bytes_remaining=%lu data_size=%lu\n", bytes, bytes_copied, bytes_remaining, data_size);
#endif
    }
  }
  
#ifdef _DEBUG 
  multilog (client->log, LOG_INFO, "transfer_data: copied %"PRIu64" bytes\n", bytes_copied);
#endif

  return (int64_t) bytes_copied;
}

int64_t transfer_data_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  return transfer_data(client, data, data_size);
}


/*! Function that closes the data file */
int dada_junkdb_close (dada_client_t* client, uint64_t bytes_written)
{
  assert (client != 0);

  dada_junkdb_t* junkdb = (dada_junkdb_t*) client->context;
  assert (junkdb != 0);

  if (junkdb->verbose)
    multilog (client->log, LOG_INFO, "close: wrote %"PRIu64" bytes\n", bytes_written);

  free (junkdb->data);
  return 0;
}

/*! Function that opens the data transfer target */
int dada_junkdb_open (dada_client_t* client)
{

  assert (client != 0);
  assert (client->header != 0);

  dada_junkdb_t* junkdb = (dada_junkdb_t*) client->context;
  assert (junkdb != 0);

  uint64_t hdr_size = 0;
  
  // read the header
  if (fileread (junkdb->header_file, client->header, client->header_size) < 0) {
    multilog (client->log, LOG_ERR, "Could not read header from %s\n", junkdb->header_file);
  }

  if (junkdb->verbose > 1)
  fprintf (stderr, "===========HEADER START=============\n"
                   "%s"
                   "============HEADER END==============\n", client->header);
  junkdb->header_read = 0;

  if (ascii_header_get (client->header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1)
  {
    multilog (client->log, LOG_WARNING, "Header with no HDR_SIZE\n");
    hdr_size = DADA_DEFAULT_HEADER_SIZE;
  }

  size_t buffer_size = 64;
  char buffer[buffer_size];
  if (ascii_header_get (client->header, "UTC_START", "%s", buffer) != 1)
  {
    time_t utc = time(0);
    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime(&utc));
    multilog (client->log, LOG_INFO, "Setting UTC_START=%s\n", buffer);
    if (ascii_header_set (client->header, "UTC_START", "%s", buffer) < 0)
    {
      multilog (client->log, LOG_WARNING, "failed to set UTC_START in header\n");
    }
  }

  int nbit;
  if (ascii_header_get (client->header, "NBIT", "%d", &nbit) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no NBIT\n");
    return -1;
  }

  // ensure that the incoming header fits in the client header buffer
  if (hdr_size > client->header_size) {
    multilog (client->log, LOG_ERR, "HDR_SIZE=%u > Block size=%"PRIu64"\n",
	      hdr_size, client->header_size);
    return -1;
  }

  client->header_size = hdr_size;
  client->optimal_bytes = 32 * 1024 * 1024;
  client->transfer_bytes = junkdb->transfer_size;
  junkdb->bytes_to_copy = 0;

  if (junkdb->verbose)
    multilog (client->log, LOG_INFO, "FILE_SIZE=%"PRIu64"\n", junkdb->transfer_size);
  if (ascii_header_set(client->header, "FILE_SIZE", "%"PRIu64, junkdb->transfer_size) < 0)
  {
    multilog (client->log, LOG_WARNING, "Failed to set FILE_SIZE header attribute to %"PRIu64"\n",
              junkdb->transfer_size);
  }

  if (junkdb->verbose)
    multilog (client->log, LOG_INFO, "open: setting transfer_bytes=%"PRIu64"\n", junkdb->transfer_size);

  // seed the RNG
  srand ( time(NULL) );
  junkdb->data = (char *) malloc (sizeof(char) * junkdb->data_size);

  if (junkdb->fill_char)
  {
    multilog (client->log, LOG_INFO, "setting all data to %c\n", junkdb->fill_char);
    memset ((void *) junkdb->data, (int) junkdb->fill_char, junkdb->data_size);
  }

  if (junkdb->write_gaussian) 
  {
    multilog (client->log, LOG_INFO, "open: generating gaussian data %"PRIu64"\n", junkdb->data_size);
    if (nbit == 8)
    {
      fill_gaussian_data(junkdb->data, junkdb->data_size, junkdb->mean, junkdb->stddev);
    }
    else if (nbit == 32)
    {
      fill_gaussian_float ((float *) junkdb->data, junkdb->data_size / sizeof(float), junkdb->mean, junkdb->stddev);
    }
    multilog (client->log, LOG_INFO, "open: data generated\n");
  }

  multilog (client->log, LOG_INFO, "open: data generated\n");

  // wait for a 1s tick
  junkdb->curr_time = time(0);
  junkdb->prev_time = junkdb->curr_time;
  while (junkdb->prev_time == junkdb->curr_time)
    junkdb->curr_time = time(0);
  junkdb->bytes_to_copy += junkdb->rate;

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_junkdb_t junkdb = DADA_JUNKDB_INIT;

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

  /* ascii char to fill data with */
  char fill_char = 0;

  unsigned write_gaussian = 0;

  /* data rate [MB/s | MiB/s] */
  float rate = DEFAULT_DATA_RATE;

  /* write time [s] */
  unsigned write_time = DEFAULT_WRITE_TIME;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* zero copy direct shm access */
  char zero_copy = 0;

  /* perform memcpy for junk data */
  unsigned perform_memcpy = 1;

  uint64_t rate_base = 1024*1024;

  int64_t total_bytes = -1;

  int arg = 0;

  junkdb.mean = 0;
  junkdb.stddev = 10;

  while ((arg=getopt(argc,argv,"b:c:ghk:m:nr:R:s:t:vz")) != -1)
    switch (arg) {

    case 'b':
      if (sscanf(optarg, "%"PRIi64, &total_bytes) != 1)
      {
        fprintf (stderr, "ERROR: could not parse total_bytes from %s\n", optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'c':
      if (sscanf(optarg, "%c", &fill_char) != 1)
      {
        fprintf (stderr, "ERROR: could not parse fill char from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'g':
      write_gaussian = 1;
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
      if (sscanf (optarg, "%lf", &(junkdb.mean)) != 1) {
        fprintf (stderr, "ERROR: could not parse mean from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'n':
      perform_memcpy = 0;
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
      if (sscanf (optarg, "%lf", &(junkdb.stddev)) != 1) {
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

  log = multilog_open ("dada_junkdb", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  junkdb.rate = (uint64_t) rate * rate_base;
  junkdb.write_time = write_time;
  junkdb.write_gaussian = write_gaussian;

  junkdb.data_size = (uint64_t) rate * rate_base;
  junkdb.transfer_size = write_time * junkdb.data_size;
  junkdb.perform_memcpy = perform_memcpy;

  // never "generate" more than 4 MB
  if (junkdb.data_size > 4 * 1024 * 1024)
    junkdb.data_size = 4 * 1024 * 1024;

  if (total_bytes > 0)
    junkdb.transfer_size = (uint64_t) total_bytes;

  junkdb.header_file = strdup(header_file);
  junkdb.verbose = verbose;
  junkdb.fill_char = fill_char;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = dada_junkdb_open;

  if (zero_copy)
    client->io_block_function = transfer_data_block;

  client->io_function = transfer_data;
  client->close_function = dada_junkdb_close;
  client->direction = dada_client_writer;

  client->context = &junkdb;

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

