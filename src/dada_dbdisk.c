#include "dada.h"
#include "ipcio.h"
#include "multilog.h"
#include "disk_array.h"
#include "ascii_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

void usage()
{
  fprintf (stdout,
	   "dada_dbdisk [options]\n"
	   " -b <buffersize>\n"
	   " -f <filesize>\n"
	   " -F <rootfilename>\n"
	   " -d run as daemon\n");
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
	       disk_array_t* disk_array,
	       multilog_t* log)
{
  /* The header from the ring buffer */
  char* header = 0;
  uint64_t header_size = 0;

  /* Expected header size, as read from HDR_SIZE attribute */
  uint64_t hdr_size = 0;
  uint64_t hdr_offset = 0;

  /* The duplicate of the header */
  static char* dup = 0;
  static uint64_t dup_size = 0;

  /* Optimal buffer size, as returned by disk array */
  uint64_t opt_buffer_size;

  /* Byte counts */
  uint64_t total_bytes_written = 0;
  int64_t bytes_written = 0;

  /* Wait for the next valid header sub-block */
  header = ipcbuf_get_next_read (header_block, &header_size);

  /* Check that header is of advertised size */
  if (ascii_header_get (header, "HDR_SIZE", "%llu", &hdr_size) != 1) {
    multilog (log, LOG_ERR, "Header block does not have HDR_SIZE");
    return -1;
  }

  if (hdr_size < header_size)
    header_size = hdr_size;

  else if (hdr_size > header_size) {
    multilog (log, LOG_ERR, "HDR_SIZE is greater than header block size");
    return -1;
  }

  /* Duplicate the header */
  if (header_size > dup_size) {
    dup = realloc (dup, header_size);
    dup_size = header_size;
  }
  memcpy (dup, header, header_size);

  ipcbuf_mark_cleared (header_block);

  /* Get the header offset */
  if (ascii_header_get (dup, "OBS_OFFSET", "%llu", &hdr_offset) != 1) {
    multilog (log, LOG_ERR, "Header block does not have OBS_OFFSET");
    return -1;
  }

  /* Write data until the end of the data stream */
  while (!ipcbuf_eod((ipcbuf_t*)data_block)) {

    OPEN a file on the disk_array;

    hdr_offset += total_bytes_written;

    /* Set the header offset */
    if (ascii_header_set (dup, "OBS_OFFSET", "%llu", hdr_offset) != 1) {
      multilog (log, LOG_ERR, "Error writing OBS_OFFSET");
      return -1;
    }

    if (write (fd, dup, header_size) < header_size) {
      multilog (log, LOG_ERR, "Error writing header: %s", sterror(errno));
      return -1;
    }

    /* Write data until the end of the file or data stream */
    bytes_written = write_loop (data_block, multilog, 
				fd, file_size, opt_buffer_size);

    if (bytes_written < 0)
      return -1;

    total_bytes_written += bytes_written;

    if (close (fd) < 0) {
      multilog (log, LOG_ERR, "Error closing %s: %s", 
		filename, strerror(errno));
    }

  }

  ipcbuf_reset ((ipcbuf_t*)data_block);

  return 0;

}


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

  /* Flag set in verbose mode */
  char verbose = 0;

  int arg = 0;
  while ((arg=getopt(argc,argv,"dv")) != -1) {
    switch (arg) {

      case 'd':
	daemon=1;
	break;

      case 'v':
        verbose=1;
        break;

      default:
	usage ();
	return 0;

    }
  }

  dada_init (&dada);

  log = multilog_open ("dada_dbdisk", daemon);

  if (daemon) {

    /* set up for daemon usage */	  
    if (fork() < 0)
      exit(EXIT_FAILURE);

    exit(EXIT_SUCCESS);

  }
  else
    multilog_add (log, stderr);

  multilog_serve (log, dada.log_port);

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
