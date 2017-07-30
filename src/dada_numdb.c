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
#define DEFAULT_WRITE_BYTES 128000000


void usage()
{
  fprintf (stdout,
	   "dada_numdb [options] header_file\n"
     " -b bytes number of bytes to write [default %d]\n"
     " -i num   number of bytes to interleave [default 8192]\n"
     " -s num   number to start on [default 0]\n"
     " -k       hexadecimal shared memory key  [default: %x]\n"
     " -h       print help\n",
     DEFAULT_WRITE_BYTES, DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  /* header file to use */
  char * header_file;

  /* first number to write */
  uint64_t start_num;

  /* number of uint64_ts to interleave */
  uint64_t interleave_num;

  /* counter */
  uint64_t index;

  /* number of bytes to write */
  uint64_t write_bytes;

  /* flag for reading the header */
  unsigned header_read;

  unsigned verbose;

} dada_numdb_t;

#define DADA_NUMDB_INIT { "", 0, 0, 0, 0, 0, 0 }

/*! Pointer to the function that transfers data to/from the target */
int64_t dada_numdb_io (dada_client_t* client, void* data, uint64_t data_size)
{

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "dada_numdb_io %p %"PRIu64"\n", data, data_size);
#endif

  /* the dada_numdb specific data */
  dada_numdb_t* numdb = (dada_numdb_t*) client->context;

  if (!numdb->header_read) {
    numdb->header_read = 1;
#ifdef _DEBUG
    multilog (client->log, LOG_INFO, "dada_numdb_io: read header\n");
#endif
    return data_size;
  }

  // data_size is the number of bytes to write, uint64_size is the number of uint64s to write
  size_t uint64_size = 8;
  uint64_t uint64_count = data_size / uint64_size;

  unsigned j = 0;
  uint64_t i = 0;
  char * ptr = 0;
  unsigned char ch;

  for (i=0; i<uint64_count; i++)
  {
    // set the ptr to the correct place in the array
    ptr = data + i*uint64_size;
    for (j = 0; j < 8; j++ ) 
    {
      ch = (numdb->index >> ((j & 7) << 3)) & 0xFF; 
      ptr[8 - j - 1] = ch; 
    }
    
    numdb->index++;
    if (numdb->index % numdb->interleave_num == 0)
    {
      if (numdb->verbose)
        fprintf(stderr, "increment %"PRIu64" -> ", numdb->index);
      numdb->index += numdb->interleave_num;
      if (numdb->verbose)
        fprintf(stderr, "%"PRIu64"\n", numdb->index);
    }
  }

#ifdef _DEBUG
  multilog (client->log, LOG_INFO, "dada_numdb_io: copied %"PRIu64" bytes\n", data_size);
#endif

  return (int64_t) data_size;
}

/*! Function that closes the data file */
int dada_numdb_close (dada_client_t* client, uint64_t bytes_written)
{
  /* the dada_numdb specific data */
  dada_numdb_t* numdb = 0;

  assert (client != 0);
  numdb = (dada_numdb_t*) client->context;
  assert (numdb != 0);

  fprintf(stderr, "close: end number was=%"PRIu64"\n", numdb->index);

  return 0;
}

/*! Function that opens the data transfer target */
int dada_numdb_open (dada_client_t* client)
{

  /* the dada_numdb specific data */
  dada_numdb_t* numdb = (dada_numdb_t*) client->context;

  numdb->index = numdb->start_num;

  fprintf(stderr, "open: start number=%"PRIu64"\n", numdb->index);
  fprintf(stderr, "open: end number=%"PRIu64"\n", (numdb->write_bytes / 8) * 2);
  fprintf(stderr, "open: total bytes=%"PRIu64"\n", numdb->write_bytes);
  fprintf(stderr, "open: interleave_bytes=%"PRIu64"\n", numdb->interleave_num * 8);

  client->optimal_bytes = numdb->interleave_num * 8 * 8;
  client->transfer_bytes = numdb->write_bytes;
  fprintf(stderr, "open: setting transfer_bytes to %"PRIu64"\n", client->transfer_bytes);

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_numdb_t numdb = DADA_NUMDB_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* header to use for the data block */
  char * header_file = 0;

  /* Flag set in verbose mode */
  unsigned verbose = 0;

  /* start number */
  uint64_t start_num = 0;

  /* interleave bytes */
  uint64_t interleave_bytes = 8192;

  /* total number of bytes to write */
  uint64_t write_bytes = DEFAULT_WRITE_BYTES;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"b:hi:k:s:v")) != -1)
    switch (arg) {

    case 'b':
      if (sscanf (optarg, "%"PRIu64, &write_bytes) != 1) {
        fprintf (stderr,"ERROR: could not parse write bytes from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'h':
      usage();
      return (EXIT_SUCCESS);

    case 'i':
      if (sscanf (optarg, "%"PRIu64, &interleave_bytes) != 1) {
        fprintf (stderr, "ERROR: could not parse interleave bytes from %s\n", optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "ERROR: could not parse key from %s\n",optarg);
        usage();
        return EXIT_FAILURE;
      }
      break;

    case 's':
      if (sscanf (optarg, "%"PRIu64, &start_num) != 1) {
        fprintf (stderr, "ERROR: could not parse start_num from %s\n", optarg);
        usage(); 
        return EXIT_FAILURE;
      }
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

  header_file = strdup(argv[optind]);

  /* test that the header file can be read */
  FILE* fptr = fopen (header_file, "r");
  if (!fptr) {
    fprintf (stderr, "Error: could not open '%s' for reading: %s\n", header_file, strerror(errno));
    return(EXIT_FAILURE);
  }
  fclose(fptr);

  if (verbose)
    fprintf (stdout, "Using header file: %s\n", header_file);

  log = multilog_open ("dada_numdb", 0);

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;
  numdb.interleave_num = interleave_bytes / 8;
  numdb.start_num = start_num;
  numdb.write_bytes = write_bytes;
  numdb.header_file = strdup(header_file);
  numdb.verbose = verbose;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function = dada_numdb_open;
  client->io_function = dada_numdb_io;
  client->close_function = dada_numdb_close;

  client->direction = dada_client_writer;

  client->context = &numdb;

  if (dada_client_write (client) < 0) {
    multilog (log, LOG_ERR, "Error during transfer\n");
    return -1;
  }

  if (verbose)
    multilog (log, LOG_INFO, "dada_hdu_unlock_write()\n");
  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "dada_hdu_disconnect()\n");
  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "dada_hdu_destroy()\n");
  dada_hdu_destroy (hdu);

  return EXIT_SUCCESS;
}

