#include "dada.h"
#include "ipcio.h"
#include "multilog.h"
#include "disk_array.h"
#include "ascii_header.h"
#include "diff_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

void usage()
{
  fprintf (stdout,
	   "dada_dbdisk [options]\n"
	   " -D <path>  add a disk to which data will be written\n"
	   " -W         over-write exisiting files\n"
	   " -d         run as daemon\n");
}

int64_t write_loop (ipcio_t* data_block, multilog_t* log,
		    int fd, uint64_t bytes_to_write, uint64_t optimal_bufsz)
{
  /* The buffer used for file I/O */
  static char* buffer = 0;
  static uint64_t buffer_size = 0;

  /* counters */
  uint64_t bytes_written = 0;
  ssize_t bytes = 0;

  buffer_size = 512 * optimal_bufsz;
  buffer = (char*) malloc (buffer_size);
  assert (buffer != 0);

  while (!ipcbuf_eod((ipcbuf_t*)data_block) && bytes_to_write) {

    if (buffer_size > bytes_to_write)
      bytes = bytes_to_write;
    else
      bytes = buffer_size;

    bytes = ipcio_read (data_block, buffer, bytes);
    if (bytes < 0) {
      multilog (log, LOG_ERR, "ipcio_read error %s\n", strerror(errno));
      return -1;
    }

    bytes = write (fd, buffer, bytes);
    if (bytes < 0) {
      multilog (log, LOG_ERR, "write error %s\n", strerror(errno));
      return bytes_written;
    }

    bytes_to_write -= bytes;
    bytes_written += bytes;

  }

  return bytes_written;
}

int main_loop (dada_t* dada,
	       ipcio_t* data_block,
	       ipcbuf_t* header_block,
	       disk_array_t* array,
	       multilog_t* log)
{
  /* The duplicate of the header from the ring buffer */
  static char* dup = 0;
  static uint64_t dup_size = 0;

  /* The header from the ring buffer */
  char* header = 0;
  uint64_t header_size = 0;

  /* header size, as defined by HDR_SIZE attribute */
  uint64_t hdr_size = 0;
  /* byte offset from start of data, as defined by OBS_OFFSET attribute */
  uint64_t obs_offset = 0;

  /* file size, as defined by FILE_SIZE attribute */
  uint64_t file_size = 0;
  /* file offset from start of data, as defined by FILE_NUMBER attribute */
  unsigned file_number = 0;

  /* observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN] = "";

  /* Optimal buffer size, as returned by disk array */
  uint64_t optimal_buffer_size;

  /* Byte count */
  int64_t bytes_written = 0;

  /* Time at start and end of write loop */
  struct timeval start_loop, end_loop;

  /* Time required to write data */
  double write_time = 0;

  /* File name */
  static char* file_name = 0;

  /* File descriptor */
  int fd = -1;

  while (!header_size) {

    /* Wait for the next valid header sub-block */
    header = ipcbuf_get_next_read (header_block, &header_size);

    if (!header) {
      multilog (log, LOG_ERR, "Could not get next header\n");
      return -1;
    }

    if (!header_size) {

      ipcbuf_mark_cleared (header_block);

      if (ipcbuf_eod (header_block)) {
	multilog (log, LOG_INFO, "End of data on header block\n");
	ipcbuf_reset (header_block);
      }
      else {
	multilog (log, LOG_ERR, "Empty header block\n");
	return -1;
      }

    }


  }

  header_size = ipcbuf_get_bufsz (header_block);

  /* Check that header is of advertised size */
  if (ascii_header_get (header, "HDR_SIZE", "%"PRIu64"", &hdr_size) != 1) {
    multilog (log, LOG_ERR, "Header block does not have HDR_SIZE\n");
    return -1;
  }

  if (hdr_size < header_size)
    header_size = hdr_size;

  else if (hdr_size > header_size) {
    multilog (log, LOG_ERR, "HDR_SIZE=%"PRIu64" is greater than hdr bufsz=%"PRIu64"\n",
              hdr_size, header_size);
    multilog (log, LOG_DEBUG, "ASCII header dump\n%s", header);
    return -1;
  }

  /* Duplicate the header */
  if (header_size > dup_size) {
    dup = realloc (dup, header_size);
    assert (dup != 0);
    dup_size = header_size;
  }
  memcpy (dup, header, header_size);

  ipcbuf_mark_cleared (header_block);

  /* Get the observation ID */
  if (ascii_header_get (dup, "OBS_ID", "%s", obs_id) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have OBS_ID\n");
    strcpy (obs_id, "UNKNOWN");
  }

  /* Get the header offset */
  if (ascii_header_get (dup, "OBS_OFFSET", "%"PRIu64"", &obs_offset) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have OBS_OFFSET\n");
    obs_offset = 0;
  }

  /* Get the file size */
  if (ascii_header_get (dup, "FILE_SIZE", "%"PRIu64"", &file_size) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have FILE_SIZE\n");
    file_size = DADA_DEFAULT_FILESIZE;
  }

  /* Get the file number */
  if (ascii_header_get (dup, "FILE_NUMBER", "%u", &file_number) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have FILE_NUMBER\n");
    file_number = 0;
  }

  if (!file_name)
    file_name = malloc (FILENAME_MAX);
  assert (file_name != 0);

  /* Write data until the end of the data stream */
  while (!ipcbuf_eod((ipcbuf_t*)data_block)) {

    snprintf (file_name, FILENAME_MAX, "%s.%06u.dada", obs_id, file_number);

    fd = disk_array_open (array, file_name, file_size, &optimal_buffer_size);

    if (fd < 0) {
      multilog (log, LOG_ERR, "Error opening %s\n", file_name);
      return -1;
    }

    multilog (log, LOG_INFO, "%s opened for writing %"PRIu64" bytes\n",
	      file_name, file_size);

    /* Set the header offset */
    if (ascii_header_set (dup, "OBS_OFFSET", "%"PRIu64"", obs_offset) < 0) {
      multilog (log, LOG_ERR, "Error writing OBS_OFFSET\n");
      return -1;
    }

    /* Set the file number */
    if (ascii_header_set (dup, "FILE_NUMBER", "%u", file_number) < 0) {
      multilog (log, LOG_ERR, "Error writing FILE_NUMBER\n");
      return -1;
    }

    if (write (fd, dup, header_size) < header_size) {
      multilog (log, LOG_ERR, "Error writing header: %s\n", strerror(errno));
      return -1;
    }

    gettimeofday (&start_loop, NULL);

    /* Write data until the end of the file or data stream */
    bytes_written = write_loop (data_block, log, 
				fd, file_size, optimal_buffer_size);

    gettimeofday (&end_loop, NULL);

    if (close (fd) < 0)
      multilog (log, LOG_ERR, "Error closing %s: %s\n", 
		file_name, strerror(errno));

    if (bytes_written < 0)
      return -1;



    if (bytes_written == 0)  {

      if (chmod (file_name, S_IRWXU) < 0)
	multilog (log, LOG_ERR, "Error chmod (%s, rwx): %s\n", 
		  file_name, strerror(errno));

      if (remove (file_name) < 0)
	multilog (log, LOG_ERR, "Error remove (%s): %s\n", 
		  file_name, strerror(errno));

    }
    else {
      
      write_time = diff_time(start_loop, end_loop);
      multilog (log, LOG_INFO, "%"PRIu64" bytes written to %s in %lfs (%lg MB/s)\n",
		bytes_written, file_name, write_time,
		bytes_written/(1e6*write_time));
      
      obs_offset += bytes_written;

    }

    file_number ++;

  }

  ipcbuf_reset ((ipcbuf_t*)data_block);

  return 0;

}

struct {
  char quit;
} state;

int main (int argc, char **argv)
{
  /* DADA configuration */
  dada_t dada;

  /* DADA Data Block */
  ipcio_t data_block = IPCIO_INIT;

  /* DADA Header Block */
  ipcbuf_t header_block = IPCBUF_INIT;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Array of disks to which data will be written */
  disk_array_t* disk_array = 0;

  /* Flag set in daemon mode */
  char daemon = 0;
  pid_t pid;

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;

  disk_array = disk_array_create ();

  while ((arg=getopt(argc,argv,"dD:vW")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;
      
    case 'D':
      if (disk_array_add (disk_array, optarg) < 0) {
	fprintf (stderr, "Could not add '%s' to disk array\n", optarg);
	return EXIT_FAILURE;
      }
      break;
      
    case 'v':
      verbose=1;
      break;
      
    case 'W':
      disk_array_set_overwrite (disk_array, 1);
      break;

    default:
      usage ();
      return 0;
      
    }


  dada_init (&dada);

  log = multilog_open ("dada_dbdisk", daemon);

  if (daemon) {

    pid = fork();

    if (pid < 0)
      exit(EXIT_FAILURE);

    if (pid > 0)
      exit(EXIT_SUCCESS);

    /* Create a new SID for the child process */
    if (setsid() < 0)
      exit (EXIT_FAILURE);
              
    /* Change the current working directory */
    if (chdir("/") < 0)
      exit (EXIT_FAILURE);
        
    /* Close out the standard file descriptors */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    multilog_serve (log, dada.log_port);

  }
  else
    multilog_add (log, stderr);

  /* First connect to the shared memory */
  if (ipcbuf_connect (&header_block, dada.hdr_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to header block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_lock_read (&header_block) < 0) {
    multilog (log, LOG_ERR, "Could not lock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_connect (&data_block, dada.data_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to data block\n");
    return EXIT_FAILURE;
  }

  if (ipcio_open (&data_block, 'R') < 0) {
    multilog (log, LOG_ERR, "Failed to open data block for reading\n");
    return EXIT_FAILURE;
  }


  while (!state.quit) {

    if (main_loop (&dada, &data_block, &header_block, disk_array, log) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

  }

  /* Disconnect from the shared memory */
  if (ipcio_close (&data_block) < 0) {
    multilog (log, LOG_ERR, "Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_disconnect (&data_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from data block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_unlock_write (&header_block) < 0) {
    multilog (log, LOG_ERR,"Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_disconnect (&header_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from header block\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
