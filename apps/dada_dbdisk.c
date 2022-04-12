/* To enable the use of O_DIRECT */
#define _GNU_SOURCE

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "disk_array.h"
#include "ascii_header.h"
#include "daemon.h"
#include "dada_affinity.h"

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

void usage()
{
  fprintf (stdout,
           "dada_dbdisk [options]\n"
           " -b <core>  bind process to CPU core\n"
           " -k         hexadecimal shared memory key  [default: %x]\n"
           " -D <path>  add a disk to which data will be written\n"
           " -o         use O_DIRECT flag to bypass kernel buffering\n"
           " -W         over-write exisiting files\n"
           " -s         single transfer only\n"
           " -t bytes   set optimal bytes\n"
           " -z         use zero copy transfers\n"
           " -d         run as daemon\n", DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  /* the set of disks to which data may be written */
  disk_array_t* array;

  /* current utc start, as defined by UTC_START attribute */
  char utc_start[64];

  uint64_t obs_offset;

  /* current filename */
  char file_name [FILENAME_MAX];

  /* file offset from start of data, as defined by FILE_NUMBER attribute */
  unsigned file_number;

  char o_direct;

  uint64_t optimal_bytes;

  unsigned verbose;

} dada_dbdisk_t;

#define DADA_DBDISK_INIT { 0, "", 0, "", 0, 0, 0, 0 }

/*! Function that opens the data transfer target */
int file_open_function (dada_client_t* client)
{
  /* the dada_dbdisk specific data */
  dada_dbdisk_t* dbdisk = 0;
  
  /* status and error logging facility */
  multilog_t* log;

  /* the header */
  char* header = 0;

  /* utc start, as defined by UTC_START attribute */
  char utc_start[64] = "";

  /* size of each file to be written in bytes, subject to % 512*RESOLUTION */
  /* use  FILE_SIZE is defined, else 10*BYTES_PER_SECOND */
  uint64_t file_size = 0;

  /* BYTES_PER_SECOND from the header */
  uint64_t bytes_ps= 0;

  /* the optimal buffer size for writing to file */
  uint64_t optimal_bytes = 0;

  /* the open file descriptor */
  int fd = -1;

  /* observation offset from the UTC_START */
  uint64_t obs_offset = 0;

  /* resolution of data [from packet size] defined in RESOLUTION */
  uint64_t resolution = 0;

  assert (client != 0);

  dbdisk = (dada_dbdisk_t*) client->context;
  assert (dbdisk != 0);
  assert (dbdisk->array != 0);

  log = client->log;
  assert (log != 0);

  header = client->header;
  assert (header != 0);

  // get the UTC_START 
  if (ascii_header_get (client->header, "UTC_START", "%s", utc_start) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no UTC_START\n");
    strcpy (utc_start, "UNKNOWN");
  }

  // get the OBS_OFFSET
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no OBS_OFFSET\n");
    obs_offset = 0;
  }

  // check to see if we are still working with the same observation
  if ((strcmp (utc_start, dbdisk->utc_start) != 0) || (obs_offset != dbdisk->obs_offset))
  {
    dbdisk->file_number = 0;
    if (dbdisk->verbose)
      multilog (log, LOG_INFO, "New UTC_START=%s, OBS_OFFSET=%"PRIu64" -> "
               "file number=0\n", utc_start, obs_offset);
  }
  else
  {
    dbdisk->file_number++;
    if (dbdisk->verbose)
      multilog (log, LOG_INFO, "Continue UTC_START=%s, OBS_OFFSET=%"PRIu64" -> "
                "file number=%lu\n", utc_start, obs_offset, dbdisk->file_number);
  }

  // Set the file number to be written to the header
  if (ascii_header_set (header, "FILE_NUMBER", "%u", dbdisk->file_number) < 0)
  {
    multilog (log, LOG_ERR, "Error writing FILE_NUMBER\n");
    return -1;
  }

  // set the current observation id
  strcpy (dbdisk->utc_start, utc_start);
  dbdisk->obs_offset = obs_offset;

  // create the current file name
  snprintf (dbdisk->file_name, FILENAME_MAX, "%s_%016"PRIu64".%06u.dada", 
            utc_start, obs_offset, dbdisk->file_number);

  // check if the file size is defined in the header
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
    file_size = 0;

  if (dbdisk->verbose)
    multilog (log, LOG_INFO, "open: advertised FILE_SIZE=%"PRIu64"\n", file_size);

  // get the bytes per second from the header
  if (ascii_header_get (header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_ps) != 1) 
  {
    multilog (log, LOG_WARNING, "Header with no BYTES_PER_SECOND\n");
    bytes_ps = 64000000;
  }

  // get the resolution
  if (ascii_header_get (header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_INFO, "Header with no RESOLUTION\n");
    resolution = 0;
  }

  // if file_size is not defined, then use 10 * bytes_ps
  if (file_size == 0) 
    file_size = bytes_ps * 10;

  uint64_t byte_multiple = 1;

  // If case we use O_DIRECT, must be aligned to 512 bytes
#ifdef O_DIRECT
  if (dbdisk->o_direct)
    byte_multiple = 512;
#endif

  if (dbdisk->verbose)
    multilog (log, LOG_INFO, "open: RESOULTION=%"PRIu64"\n", resolution);

  // If we have RESOLUTION, include this in byte_multiple
  if (resolution > 0)
    byte_multiple *= resolution;

  uint64_t gap = (obs_offset + file_size) % byte_multiple;

  file_size -= gap;

  if (gap > byte_multiple / 2) 
    file_size += byte_multiple;

  if (dbdisk->verbose) 
    multilog (log, LOG_INFO, "OBS_OFFSET=%"PRIu64", FILE_SIZE=%"PRIu64","
              " BYTE_MULTIPLE=%"PRIu64"\n", obs_offset, file_size, 
              byte_multiple);

  /* update header with actual FILE_SIZE */
  if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64"", file_size) < 0) {
    multilog (log, LOG_WARNING, "Could not write FILE_SIZE to header\n");
  }

#ifdef _DEBUG
  fprintf (stderr, "dbdisk: open the file\n");
#endif

  /* Open the file */
  int flags = 0;

#ifdef O_DIRECT
  if (dbdisk->o_direct)
    flags = O_DIRECT;
#endif

  if (dbdisk->verbose)
    multilog (log, LOG_INFO, "open: dbdisk->file_name=%s file_size=%"
              PRIu64"\n", dbdisk->file_name, file_size);

  fd = disk_array_open (dbdisk->array, dbdisk->file_name,
        		file_size, &optimal_bytes, flags);

  if (fd < 0)
  {
    multilog (log, LOG_ERR, "Error opening %s: %s\n", dbdisk->file_name,
              strerror (errno));
    return -1;
  }

#ifndef O_DIRECT
  if (dbdisk->o_direct)
    fcntl (fd, F_NOCACHE, 1);
#endif

  client->fd = fd;
  client->transfer_bytes = file_size;
  client->optimal_bytes = 512 * optimal_bytes;
  
  multilog (log, LOG_INFO, "%s opened for writing %"PRIu64" bytes in %"PRIu64" chunks\n",
            dbdisk->file_name, client->transfer_bytes, client->optimal_bytes);


  return 0;
}

/*! Function that closes the data file */
int file_close_function (dada_client_t* client, uint64_t bytes_written)
{
  /* the dada_dbdisk specific data */
  dada_dbdisk_t* dbdisk = 0;

  /* status and error logging facility */
  multilog_t* log;

  assert (client != 0);

  dbdisk = (dada_dbdisk_t*) client->context;
  assert (dbdisk != 0);

  log = client->log;
  assert (log != 0);

  if (dbdisk->verbose)
    multilog (log, LOG_INFO, "close: wrote %"PRIu64" bytes\n", bytes_written);

  if (close (client->fd) < 0)
    multilog (log, LOG_ERR, "Error closing %s: %s\n", 
              dbdisk->file_name, strerror(errno));

  if (!bytes_written)
  {
    multilog (log, LOG_ERR, "Removing empty file: %s\n", dbdisk->file_name);

    if (chmod (dbdisk->file_name, S_IRWXU) < 0)
      multilog (log, LOG_ERR, "Error chmod (%s, rwx): %s\n", 
        	dbdisk->file_name, strerror(errno));
    
    if (remove (dbdisk->file_name) < 0)
      multilog (log, LOG_ERR, "Error remove (%s): %s\n", 
        	dbdisk->file_name, strerror(errno));
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t file_write_function (dada_client_t* client, 
        		     void* data, uint64_t data_size)
{
  dada_dbdisk_t * dbdisk = (dada_dbdisk_t*) client->context;

  if (dbdisk->verbose)
    multilog (client->log, LOG_INFO, "write: %"PRIu64" bytes\n", data_size);

  uint64_t written  = 0;
  uint64_t to_write = 0;
  uint64_t wrote    = 0;
  

  while (written < data_size)
  {
    if (dbdisk->optimal_bytes)
    {
      to_write = dbdisk->optimal_bytes;
      if (written + to_write > data_size)
        to_write = data_size - written;
    }
    else
      to_write = data_size;

    // Need to check we that to_write is a 512 byte multiple if using o_direct
    if (dbdisk->o_direct && (to_write % 512 != 0)) 
    {

      multilog (client->log, LOG_WARNING, "Using O_DIRECT and data_size not mod 512\n");

      // write all except the mod 512 bytes;
      //uint64_t end = (data_size / 512) * 512;
      uint64_t end = (to_write / 512) * 512;
      wrote = write (client->fd, data + written, end);
      written += wrote;

      client->fd = disk_array_reopen(dbdisk->array, client->fd, dbdisk->file_name);

      if (client->fd < 0) 
      {
        multilog (client->log, LOG_ERR, "Could not reopen file: fd=%d\n",client->fd);
        return written;
      } 
      else 
      {
        // ensure O_DIRECT is now off 
        dbdisk->o_direct = 0;
        wrote = write (client->fd, data+written, to_write-wrote);
        written += wrote;
        return written;
      }
    }
    wrote = write (client->fd, data + written, to_write);
    if (wrote != to_write)
      multilog (client->log, LOG_ERR, "write: wrote[%"PRIu64"] != to_write[%"PRIu64"]\n", wrote, to_write);
    written += wrote;
  }

  return written;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t file_write_block_function (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  return file_write_function(client, data, data_size);
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_dbdisk_t dbdisk = DADA_DBDISK_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Quit flag */
  char quit = 0;

  /* Zero copy flag */
  char zero_copy = 0;

  /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  uint64_t optimal_bytes = 0;

  int cpu_core = -1;

  dbdisk.array = disk_array_create ();

  while ((arg=getopt(argc,argv,"b:k:dD:ot:vWsz")) != -1)
    switch (arg) {

    case 'b':
      cpu_core = atoi (optarg);
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_dbdisk: could not parse key from %s\n",optarg);
        return -1;
      }
      break;

    case 'd':
      daemon=1;
      break;
      
    case 'D':
      if (disk_array_add (dbdisk.array, optarg) < 0) {
        fprintf (stderr, "Could not add '%s' to disk array\n", optarg);
        return EXIT_FAILURE;
      }
      break;
      
    case 'o':
      dbdisk.o_direct = 1;
      disk_array_set_overwrite (dbdisk.array, 1);
      break;

    case 't':
      optimal_bytes = atoi(optarg);
      break;

    case 'v':
      dbdisk.verbose = 1;

      break;
      
    case 'W':
      disk_array_set_overwrite (dbdisk.array, 1);
      break;

    case 's':
      quit = 1;
      break;

    case 'z':
      zero_copy = 1;
      break;

    default:
      usage ();
      return 0;
      
    }

  if (cpu_core >= 0)
    dada_bind_thread_to_core(cpu_core);

  log = multilog_open ("dada_dbdisk", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_DBDISK_LOG);
  }
  else
    multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = file_open_function;

  if (zero_copy)
    client->io_block_function    = file_write_block_function;

  client->io_function    = file_write_function;
  client->close_function = file_close_function;
  client->direction      = dada_client_reader;

  dbdisk.optimal_bytes   = optimal_bytes;

  client->context = &dbdisk;

  while (!client->quit) {

    if (dada_client_read (client) < 0)
    {
      multilog (log, LOG_ERR, "dada_client_read failed\n");
      quit = 1;
    }

    if (quit) {
      client->quit = 1;
    }

  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
