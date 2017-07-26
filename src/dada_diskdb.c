#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "disk_array.h"
#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>

//#define _DEBUG 1
#define MAX_FILES 1024

void usage()
{
  fprintf (stdout,
	   "dada_diskdb [options]\n"
     " -h   print this help text\n"
     " -k   hexadecimal shared memory key  [default: %x]\n"
     " -f   file to write to the ring buffer \n"
     " -o   bytes  number of bytes to seek into the file\n"
     " -s   single file then exit\n"
     " -d   run as daemon\n"
     " -z   use zero copy shm access\n", DADA_DEFAULT_BLOCK_KEY);
}

typedef struct {

  /* current utc start, as defined by UTC_START attribute */
  char utc_start[20];

  /* current observation offset, as defined by OBS_OFFSET attribute */
  uint64_t obs_offset;

  /* current filename */
  char ** filenames;

  /* flag for whether the current file continues into the next file */
  char * continues;

  /* file sizes for each of files */
  uint64_t * file_sizes;

  /* file offset from start of data, as defined by FILE_NUMBER attribute */
  unsigned file_number;

  /* number of bytes to artificially seek into the file */
  uint64_t seek_bytes;

  /* remove files after they have been transferred to the Data Block */
  char remove_files;

  char header_read;

  char verbose;

  /* the set of disks from which data files will be read */
  disk_array_t * array;

} dada_diskdb_t;

int check_contiguity (multilog_t * log, dada_diskdb_t * ctx);
int open_next_contiguous_file (dada_client_t* client);

/* Number of files to load */
static int n_files = 0;
/* Current file loading */
static int cur_file = 0;

/*! Pointer to the function that transfers data to/from the target */
int64_t file_read_function (dada_client_t* client, 
			    void* data, uint64_t data_size)
{
#ifdef _DEBUG
  fprintf (stderr, "file_read_function %p %"PRIu64"\n", data, data_size);
#endif
  dada_diskdb_t* ctx = (dada_diskdb_t*) client->context;

  size_t bytes_read;

  if (ctx->header_read == 0)
  {
    ctx->header_read = 1;
    sprintf ((char *) data, client->header, strlen(client->header));
    lseek (client->fd, client->header_size, SEEK_SET);
    bytes_read = data_size;
  }
  else
  {
    if (ctx->seek_bytes > 0)
    {
      fprintf (stderr, "file_read_function: seeking %"PRIu64" bytes into the file\n", ctx->seek_bytes);
      off_t offset = (off_t) ctx->seek_bytes;
      lseek (client->fd, offset, SEEK_CUR);
      ctx->seek_bytes = 0;
    }

    bytes_read = read (client->fd, data, data_size);

    if (bytes_read < data_size && ctx->continues[cur_file])
    {
      if (open_next_contiguous_file (client) < 0)
      {
        fprintf (stderr, "error opening next contiguous file\n");
        return -1;
      }
      bytes_read += read (client->fd, data + bytes_read, (data_size - bytes_read));
    }
  }

  return bytes_read;
}

int64_t file_read_block_function (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  return file_read_function (client, data, data_size);
}



/*! Function that closes the data file */
int file_close_function (dada_client_t* client, uint64_t bytes_written)
{
  /* the dada_diskdb specific data */
  dada_diskdb_t* diskdb = 0;

  assert (client != 0);
  diskdb = (dada_diskdb_t*) client->context;
  assert (diskdb != 0);

  if (close (client->fd) < 0) {
    multilog (client->log, LOG_ERR, "Could not close file: %s\n",
              strerror(errno));
    return -1;
  }

  cur_file++;
  if (cur_file >= n_files)
  {
	  //diskdb->filenames[cur_file][0] = '\0';
	  client->fd = -1;
	  client->quit=1;
  }

  return 0;
}

/*! Function that opens the data transfer target */
int file_open_function (dada_client_t* client)
{
  /* the dada_diskdb specific data */
  dada_diskdb_t* diskdb = 0;
  
  /* status and error logging facility */
  multilog_t* log;

  /* utc start, as defined by UTC_START attribute */
  char utc_start[20];

  /* observation offset, as defined by OBS_OFFSET attribute */
  uint64_t obs_offset = 0;

  /* size of each file to be written in bytes, as determined by FILE_SIZE */
  uint64_t file_size = 0;

  /* the size of the header, as determined by HDR_SIZE */
  uint64_t hdr_size = 0;

  int64_t ret = 0;

  assert (client != 0);

  diskdb = (dada_diskdb_t*) client->context;
  assert (diskdb != 0);
  assert (diskdb->array != 0);

  assert (client->header != 0);

  log = client->log;

  /*
  while (diskdb->filenames[cur_file][0] == '\0') 
  {
    if (n_files > 0)
    {
      client->quit = 1;
      return 0;
    }

    // look for a new file in the disk array
    fprintf (stderr, "WRITE FILE SEARCH\n");
    return -1;
  }
  */

  client->fd = open (diskdb->filenames[cur_file], O_RDONLY);

  if (client->fd < 0) {
    multilog (client->log, LOG_ERR, "Error opening %s: %s\n",
              diskdb->filenames[cur_file], strerror(errno));
    return -1;
  } 

  ret = read (client->fd, client->header, client->header_size);

  if (ret < client->header_size) {
    multilog (client->log, LOG_ERR, 
	      "read %d out of %d bytes from Header: %s\n", 
	      ret, client->header_size, strerror(errno));
    file_close_function (client, 0);
    return -1;
  }

#ifdef _DEBUG
fprintf (stderr, "read HEADER START\n%sHEADER END\n", client->header);
#endif

  /* Get the header size */
  if (ascii_header_get (client->header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1)
  {
    multilog (log, LOG_WARNING, "Header with no HDR_SIZE\n");
    hdr_size = DADA_DEFAULT_HEADER_SIZE;
  }
  if (diskdb->verbose)
    multilog (log, LOG_INFO, "HDR_SIZE=%"PRIu64"\n", hdr_size);

  /* Ensure that the incoming header fits in the client header buffer */
  if (hdr_size > client->header_size) {
    multilog (client->log, LOG_ERR, "HDR_SIZE=%u > Block size=%"PRIu64"\n",
	      hdr_size, client->header_size);
    return -1;
  }

  file_size = diskdb->file_sizes[cur_file];
  if (ascii_header_set (client->header, "FILE_SIZE", "%"PRIu64, file_size) < 0)
  {
    multilog (log, LOG_WARNING, "Could not set FILE_SIZE in outgoing header\n");
    return -1;
  }

  /* Get the file size */
  /*
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIu64, &file_size)!=1)
  {
    multilog (log, LOG_WARNING, "Header with no FILE_SIZE\n");

    struct stat buf;
    if (fstat (client->fd, &buf) < 0)  {
      multilog (log, LOG_ERR, "Error fstat %s\n", strerror(errno));
      return -1;
    }
    file_size = buf.st_size - hdr_size;
  }
  */
  if (diskdb->verbose)
    multilog (log, LOG_INFO, "FILE_SIZE=%"PRIu64"\n", file_size);

  client->header_size = hdr_size;
  client->optimal_bytes = 1024 * 1024;
  client->transfer_bytes = (file_size - diskdb->seek_bytes);

  /* Get the observation ID */
  if (ascii_header_get (client->header, "UTC_START", "%s", utc_start) != 1) {
    multilog (log, LOG_WARNING, "Header with no UTC_START\n");
    strcpy (utc_start, "UNKNOWN");
  }
  if (diskdb->verbose)
    multilog (log, LOG_INFO, "UTC_START=%s\n", utc_start);

  /* Get the observation offset */
  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &obs_offset) 
      != 1) {
    multilog (log, LOG_WARNING, "Header with no OBS_OFFSET\n");
    obs_offset = 0;
  }
  if (diskdb->verbose)
    multilog (log, LOG_INFO, "OBS_OFFSET=%"PRIu64"\n", obs_offset);

  /* set the current observation id */
  strcpy (diskdb->utc_start, utc_start);
  diskdb->obs_offset = obs_offset;

#ifdef _DEBUG
  fprintf (stderr, "file_open_function returns\n");
#endif

  lseek (client->fd, 0, SEEK_SET);

  return 0;
}

int open_next_contiguous_file (dada_client_t* client)
{
  dada_diskdb_t * diskdb = (dada_diskdb_t *) client->context;

  // close current file, incrementing static counter (cur_file)
  if (file_close_function(client, 0) < 0)
  {
    multilog (client->log, LOG_ERR, "open_next_contiguous_file: file_close_function failed\n");
    return -1;
  }

  // open next file
  client->fd = open (diskdb->filenames[cur_file], O_RDONLY);

  if (client->fd < 0) 
  {
    multilog (client->log, LOG_ERR, "Error opening %s: %s\n",
              diskdb->filenames[cur_file], strerror(errno));
    return -1;
  }

  // not strictly necessary to read file into header, could just lseek
  lseek (client->fd, client->header_size, SEEK_SET);
}


int check_contiguity (multilog_t * log, dada_diskdb_t * diskdb)
{
  unsigned ifile = 0;
  int fd, ret;
  uint64_t prev_obs_offset, prev_file_size;
  uint64_t curr_obs_offset, curr_file_size;

  // get the header size for the first file
  size_t hdr_size = ascii_header_get_size (diskdb->filenames[0]);
  char * header = (char *) malloc (hdr_size + 1);

  char * curr_utc_start = (char *) malloc (sizeof(char) * 20);
  char * prev_utc_start = (char *) malloc (sizeof(char) * 20);
  int rval = -1;

  // read the require meta-data from the first file
  if (n_files > 0)
  {
    fd = open (diskdb->filenames[0], O_RDONLY);
    if (fd)
    {
      ret = read (fd, header, hdr_size);

      if (ascii_header_get (header, "UTC_START", "%s", prev_utc_start) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain UTC_START\n", diskdb->filenames[0]);
        return -1;
      }

      if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &prev_obs_offset) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain OBS_OFFSET\n", diskdb->filenames[0]);
        return -1;
      }

      if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &prev_file_size) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain FILE_SIZE\n", diskdb->filenames[0]);
        struct stat buf;
        if (fstat (fd, &buf) < 0)  
        {
          multilog (log, LOG_ERR, "check_contiguity: error during fstat on %s: %s\n", diskdb->filenames[0], strerror(errno));
          return -1;
        }
        prev_file_size = buf.st_size - hdr_size;
      }
    }
    diskdb->file_sizes[0] = prev_file_size;
  }

  for (ifile=1; ifile<n_files; ifile++)
  {
    // by default all are not-continuous
    diskdb->continues[ifile-1] = 0;
    fd = open (diskdb->filenames[ifile], O_RDONLY);
    if (fd)
    {
      ret = read (fd, header, hdr_size);
      close (fd);

      if (ascii_header_get (header, "UTC_START", "%s", curr_utc_start) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain UTC_START\n", diskdb->filenames[ifile]);
        break;
      }
  
      if (ascii_header_get (header, "OBS_OFFSET", "%"PRIu64, &curr_obs_offset) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain OBS_OFFSET\n", diskdb->filenames[ifile]);
        break;
      }

      if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &curr_file_size) != 1)
      {
        multilog (log, LOG_WARNING, "check_contiguity: header from %s did not contain FILE_SIZE\n", diskdb->filenames[0]);
        struct stat buf;
        if (fstat (fd, &buf) < 0)
        {
          multilog (log, LOG_ERR, "check_contiguity: error during fstat on %s: %s\n", diskdb->filenames[0], strerror(errno));
          break;
        }
        curr_file_size = buf.st_size - hdr_size;
      }

      // save this file size
      diskdb->file_sizes[ifile] = curr_file_size;

      if ((strcmp (prev_utc_start, curr_utc_start) == 0) && (curr_obs_offset == (prev_obs_offset + prev_file_size)))
      {
        int i = (int) ifile - 1;
        diskdb->continues[i] = 1;
        while (i >= 0 && diskdb->continues[i] == 1)
        {
          diskdb->file_sizes[i] += curr_file_size;
          i--;
        }
      }
      if (diskdb->verbose)
        multilog (log, LOG_INFO, "contiguity=%d for %s and %s\n", diskdb->continues[ifile-1], diskdb->filenames[ifile-1], diskdb->filenames[ifile]);
      strncpy (prev_utc_start, curr_utc_start, 19);

      prev_obs_offset = curr_obs_offset;
      prev_file_size = curr_file_size;
    }
  }

  if (ifile == n_files)
    rval = 0;

  free (prev_utc_start);
  free (curr_utc_start);
  free (header);

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  dada_diskdb_t diskdb;

  diskdb.filenames = (char **) malloc (sizeof (char *) * MAX_FILES);
  diskdb.continues = (char *) malloc (sizeof (char) * MAX_FILES);
  diskdb.file_sizes = (uint64_t *) malloc (sizeof (uint64_t) * MAX_FILES);
  diskdb.file_number = 0;
  diskdb.seek_bytes = 0;
  diskdb.remove_files = 0;
  diskdb.header_read = 0;
  diskdb.verbose = 0;

  memset ((void *) diskdb.filenames, 0, sizeof (char *) * MAX_FILES);
  memset ((void *) diskdb.continues, 0, sizeof (char) * MAX_FILES);
  memset ((void *) diskdb.file_sizes , 0, sizeof (uint64_t) * MAX_FILES);

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
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

  disk_array_t * tmp = disk_array_create ();
  diskdb.array = tmp;

  n_files = 0;

  while ((arg=getopt(argc,argv,"hk:df:o:vsz")) != -1)
    switch (arg) {

    case 'h':
      usage();
      return 0;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"dada_diskdb: could not parse key from %s\n",optarg);
        return -1;
      }
      break;
      
    case 'd':
      daemon=1;
      break;

    case 'f':
      if (optarg)
      {
        diskdb.filenames[n_files] = (char *) malloc (sizeof(char) * (strlen(optarg) + 1));
        memset ((void *) diskdb.filenames[n_files], 0, (sizeof(char) * (strlen(optarg) + 1)));
        strncpy (diskdb.filenames[n_files], optarg, strlen(optarg));
        n_files++;
        break;
      }
      else
      {
        fprintf (stderr, "ERROR: missing argument for -f\n");
        return -1;
      }

    case 'o':
      fprintf (stderr, "seek bytes huh?\n");
      if (sscanf (optarg, "%"PRIu64, &(diskdb.seek_bytes)) != 1) {
        fprintf (stderr,"dada_diskdb: could not parse seek bytes from %s\n",optarg);
        return -1;
      }
      break;

    case 'v':
      diskdb.verbose = 1;
      break;
      
    case 's':
      quit=1;
      break;

    case 'z':
      zero_copy=1;
      break;

    default:
      usage ();
      return 0;
      
    }

  log = multilog_open ("dada_diskdb", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_DISKDB_LOG);
  }
  else
    multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  // check the contiguity of the files
  if (check_contiguity (log, &diskdb) < 0)
  {
    fprintf (stderr, "failed to check file contiguity\n");
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = file_open_function;
  client->io_function    = file_read_function;
  if (zero_copy)
    client->io_block_function = file_read_block_function;
  else
    client->io_block_function = 0;

  client->close_function = file_close_function;
  client->direction      = dada_client_writer;

  client->context = &diskdb;

  while (!client->quit) {
#ifdef _DEBUG
	  fprintf (stderr, "call dada_client_write\n");
#endif
    if (dada_client_write (client) < 0) {
      multilog (log, LOG_ERR, "Error during transfer\n");
      return -1;
    }

    if (quit)
      client->quit = 1;

  }
  if (dada_client_close (client) < 0) {
	  multilog (log, LOG_ERR, "Error closing data block\n");
	  return -1;
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

