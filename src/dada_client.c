#include "dada_prc_main.h"
#include "ascii_header.h"
#include "diff_time.h"

#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

/*! Create a new DADA primary write client main loop */
dada_prc_main_t* dada_prc_main_create ()
{
  dada_prc_main_t* prcm = malloc (sizeof(dada_prc_main_t));
  assert (prcm != 0);

  prcm -> log = 0;
  prcm -> data_block = 0;
  prcm -> header_block = 0;

  prcm -> start_function = 0;
  prcm -> buffer_function = 0;
  prcm -> stop_function = 0;

  prcm -> context = 0;

  return prcm;
}

/*! Destroy a DADA primary write client main loop */
void dada_prc_main_destroy (dada_prc_main_t* prcm)
{
  free (prcm);
}

/*! Read from the Data Block and write to the open target */
int64_t dada_prc_write_loop (dada_prc_main_t* prcm, uint64_t bytes_to_write)
{
  /* The buffer used for file I/O */
  static char* buffer = 0;
  static uint64_t buffer_size = 0;

  /* counters */
  uint64_t bytes_written = 0;
  ssize_t bytes = 0;

  multilog_t* log = 0;
  int fd = -1;

  assert (prcm != 0);

  if (buffer_size < prcm->optimal_bytes) {
    buffer_size = 512 * prcm->optimal_bytes;
    buffer = (char*) realloc (buffer, buffer_size);
    assert (buffer != 0);
  }

  log = prcm->log;
  assert (log != 0);

  fd = prcm->fd;
  assert (fd >= 0);

  while (!ipcbuf_eod((ipcbuf_t*)prcm->data_block) && bytes_to_write) {

    if (buffer_size > bytes_to_write)
      bytes = bytes_to_write;
    else
      bytes = buffer_size;

    bytes = ipcio_read (prcm->data_block, buffer, bytes);
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

int dada_prc_main (dada_prc_main_t* prcm)
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


int main_loop (dada_t* dada,
	       ipcio_t* data_block,
	       ipcbuf_t* header_block,
	       disk_array_t* array,
	       multilog_t* log)











  while (!header_size) {

    /* Wait for the next valid header sub-block */
    header = ipcbuf_get_next_read (prcm->header_block, &header_size);

    if (!header) {
      multilog (log, LOG_ERR, "Could not get next header\n");
      return -1;
    }

    if (!header_size) {

      ipcbuf_mark_cleared (prcm->header_block);

      if (ipcbuf_eod (prcm->header_block)) {
	multilog (log, LOG_INFO, "End of data on header block\n");
	ipcbuf_reset (prcm->header_block);
      }
      else {
	multilog (log, LOG_ERR, "Empty header block\n");
	return -1;
      }

    }


  }

  header_size = ipcbuf_get_bufsz (prcm->header_block);

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

  ipcbuf_mark_cleared (prcm->header_block);

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


  /* Write data until the end of the data stream */
  while (!ipcbuf_eod((ipcbuf_t*)prcm->data_block)) {

    /* Set the header offset */
    if (ascii_header_set (dup, "OBS_OFFSET", "%"PRIu64"", obs_offset) < 0) {
      multilog (log, LOG_ERR, "Error writing OBS_OFFSET\n");
      return -1;
    }

    open function

    if (write (prcm->fd, dup, header_size) < header_size) {
      multilog (log, LOG_ERR, "Error writing header: %s\n", strerror(errno));
      return -1;
    }

    gettimeofday (&start_loop, NULL);

    /* Write data until the end of the file or data stream */
    bytes_written = write_loop (prcm->data_block, file_size);

    gettimeofday (&end_loop, NULL);

    close function

    if (bytes_written > 0) {
      
      write_time = diff_time (start_loop, end_loop);
      multilog (log, LOG_INFO, "%"PRIu64" bytes written to %s in %lfs (%lg MB/s)\n",
		bytes_written, file_name, write_time,
		bytes_written/(1e6*write_time));
      
      obs_offset += bytes_written;

    }


  }

  ipcbuf_reset ((ipcbuf_t*)prcm->data_block);

  return 0;

}
