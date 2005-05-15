#include "dada_prc_main.h"
#include "dada_def.h"

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

  prcm -> open_function = 0;
  prcm -> write_function = 0;
  prcm -> close_function = 0;

  prcm -> context = 0;

  prcm -> header = 0;
  prcm -> header_size = 0;

  prcm -> fd = -1;
  prcm -> transfer_bytes = 0;
  prcm -> optimal_bytes = 0;

  return prcm;
}

/*! Destroy a DADA primary write client main loop */
void dada_prc_main_destroy (dada_prc_main_t* prcm)
{
  free (prcm);
}

/*! Read from the Data Block and write to the open target */
int64_t dada_prc_write_loop (dada_prc_main_t* prcm)
{
  /* The buffer used for transfer from Data Block to target */
  static char* buffer = 0;
  static uint64_t buffer_size = 0;

  /* counters */
  uint64_t bytes_to_write = 0;
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

  bytes_to_write = prcm->transfer_bytes;

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
  /* pointer to the status and error logging facility */
  multilog_t* log = 0;

  /* The header from the ring buffer */
  char* header = 0;
  uint64_t header_size = 0;

  /* header size, as defined by HDR_SIZE attribute */
  uint64_t hdr_size = 0;
  /* byte offset from start of data, as defined by OBS_OFFSET attribute */
  uint64_t obs_offset = 0;

  /* Byte count */
  int64_t bytes_written = 0;

  /* Time at start and end of write loop */
  struct timeval start_loop, end_loop;

  /* Time required to write data */
  double write_time = 0;

  assert (prcm != 0);

  log = prcm->log;
  assert (log != 0);


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
  if (ascii_header_get (header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1) {
    multilog (log, LOG_ERR, "Header block does not have HDR_SIZE\n");
    return -1;
  }

  if (hdr_size < header_size)
    header_size = hdr_size;

  else if (hdr_size > header_size) {
    multilog (log, LOG_ERR, "HDR_SIZE=%"PRIu64
	      " is greater than hdr bufsz=%"PRIu64"\n", hdr_size, header_size);
    multilog (log, LOG_DEBUG, "ASCII header dump\n%s", header);
    return -1;
  }

  /* Duplicate the header */
  if (header_size > prcm->header_size) {
    prcm->header = realloc (prcm->header, header_size);
    assert (prcm->header != 0);
    prcm->header_size = header_size;
  }
  memcpy (prcm->header, header, header_size);

  ipcbuf_mark_cleared (prcm->header_block);

  /* Get the header offset */
  if (ascii_header_get (prcm->header, "OBS_OFFSET",
			"%"PRIu64, &obs_offset) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have OBS_OFFSET\n");
    obs_offset = 0;
  }

  /* Write data until the end of the data stream */
  while (!ipcbuf_eod((ipcbuf_t*)prcm->data_block)) {

    /* Set the header offset */
    if (ascii_header_set (prcm->header, "OBS_OFFSET",
			  "%"PRIu64, obs_offset) < 0) {
      multilog (log, LOG_ERR, "Error writing OBS_OFFSET\n");
      return -1;
    }

    if (prcm->open_function (prcm) < 0) {
      multilog (log, LOG_ERR, "Error calling open function\n");
      return -1;
    }

    if (write (prcm->fd, prcm->header, header_size) < header_size) {
      multilog (log, LOG_ERR, "Error writing header: %s\n", strerror(errno));
      return -1;
    }

    gettimeofday (&start_loop, NULL);

    /* Write data until the end of the transfer */
    bytes_written = dada_prc_write_loop (prcm);

    gettimeofday (&end_loop, NULL);

    if (prcm->close_function (prcm, bytes_written) < 0) {
      multilog (log, LOG_ERR, "Error calling close function\n");
      return -1;
    }

    if (bytes_written > 0) {
      
      write_time = diff_time (start_loop, end_loop);
      multilog (log, LOG_INFO, "%"PRIu64" bytes written in %lfs "
		"(%lg MB/s)\n", bytes_written, write_time,
		bytes_written/(1e6*write_time));
      
      obs_offset += bytes_written;

    }


  }

  ipcbuf_reset ((ipcbuf_t*)prcm->data_block);

  return 0;

}
