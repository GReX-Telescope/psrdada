#include "dada_client.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "diff_time.h"

#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

/*! Create a new DADA primary write client main loop */
dada_client_t* dada_client_create ()
{
  dada_client_t* client = malloc (sizeof(dada_client_t));
  assert (client != 0);

  client -> log = 0;
  client -> data_block = 0;
  client -> header_block = 0;

  client -> open_function = 0;
  client -> write_function = 0;
  client -> close_function = 0;

  client -> context = 0;

  client -> header = 0;
  client -> header_size = 0;

  client -> fd = -1;
  client -> transfer_bytes = 0;
  client -> optimal_bytes = 0;

  return client;
}

/*! Destroy a DADA primary write client main loop */
void dada_client_destroy (dada_client_t* client)
{
  free (client);
}

/*! Read from the Data Block and write to the open target */
int64_t dada_prc_write_loop (dada_client_t* client)
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

  assert (client != 0);

  if (buffer_size < client->optimal_bytes) {
    buffer_size = 512 * client->optimal_bytes;
    buffer = (char*) realloc (buffer, buffer_size);
    assert (buffer != 0);
  }

  log = client->log;
  assert (log != 0);

  fd = client->fd;
  assert (fd >= 0);

  bytes_to_write = client->transfer_bytes;

  while (!ipcbuf_eod((ipcbuf_t*)client->data_block) && bytes_to_write) {

    if (buffer_size > bytes_to_write)
      bytes = bytes_to_write;
    else
      bytes = buffer_size;

    bytes = ipcio_read (client->data_block, buffer, bytes);
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

int dada_client (dada_client_t* client)
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

  assert (client != 0);

  log = client->log;
  assert (log != 0);


  while (!header_size) {

    /* Wait for the next valid header sub-block */
    header = ipcbuf_get_next_read (client->header_block, &header_size);

    if (!header) {
      multilog (log, LOG_ERR, "Could not get next header\n");
      return -1;
    }

    if (!header_size) {

      ipcbuf_mark_cleared (client->header_block);

      if (ipcbuf_eod (client->header_block)) {
	multilog (log, LOG_INFO, "End of data on header block\n");
	ipcbuf_reset (client->header_block);
      }
      else {
	multilog (log, LOG_ERR, "Empty header block\n");
	return -1;
      }

    }

  }

  header_size = ipcbuf_get_bufsz (client->header_block);

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
  if (header_size > client->header_size) {
    client->header = realloc (client->header, header_size);
    assert (client->header != 0);
    client->header_size = header_size;
  }
  memcpy (client->header, header, header_size);

  ipcbuf_mark_cleared (client->header_block);

  /* Get the header offset */
  if (ascii_header_get (client->header, "OBS_OFFSET",
			"%"PRIu64, &obs_offset) != 1) {
    multilog (log, LOG_WARNING, "Header block does not have OBS_OFFSET\n");
    obs_offset = 0;
  }

  /* Write data until the end of the data stream */
  while (!ipcbuf_eod((ipcbuf_t*)client->data_block)) {

    /* Set the header offset */
    if (ascii_header_set (client->header, "OBS_OFFSET",
			  "%"PRIu64, obs_offset) < 0) {
      multilog (log, LOG_ERR, "Error writing OBS_OFFSET\n");
      return -1;
    }

    if (client->open_function (client) < 0) {
      multilog (log, LOG_ERR, "Error calling open function\n");
      return -1;
    }

    if (write (client->fd, client->header, header_size) < header_size) {
      multilog (log, LOG_ERR, "Error writing header: %s\n", strerror(errno));
      return -1;
    }

    gettimeofday (&start_loop, NULL);

    /* Write data until the end of the transfer */
    bytes_written = dada_prc_write_loop (client);

    gettimeofday (&end_loop, NULL);

    if (client->close_function (client, bytes_written) < 0) {
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

  ipcbuf_reset ((ipcbuf_t*)client->data_block);

  return 0;

}
