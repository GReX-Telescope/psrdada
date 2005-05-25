#include "dada_pwc_main.h"
#include "dada_def.h"

#include "ascii_header.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*! Create a new DADA primary write client main loop */
dada_pwc_main_t* dada_pwc_main_create ()
{
  dada_pwc_main_t* pwcm = malloc (sizeof(dada_pwc_main_t));
  assert (pwcm != 0);

  pwcm -> pwc = 0;
  pwcm -> log = 0;

  pwcm -> data_block = 0;
  pwcm -> header_block = 0;

  pwcm -> start_function = 0;
  pwcm -> buffer_function = 0;
  pwcm -> stop_function = 0;

  pwcm -> context = 0;
  pwcm -> header = 0;
  pwcm -> header_size = 0;

  return pwcm;
}

/*! Destroy a DADA primary write client main loop */
void dada_pwc_main_destroy (dada_pwc_main_t* pwcm)
{
  free (pwcm);
}

/*! prepare for data transfer */
int dada_pwc_main_prepare (dada_pwc_main_t* pwcm);

/*! start the data transfer */
int dada_pwc_main_start_transfer (dada_pwc_main_t* pwcm);

/*! do the data transfer */
int dada_pwc_main_transfer_data (dada_pwc_main_t* pwcm);

/*! stop the data transfer */
int dada_pwc_main_stop_transfer (dada_pwc_main_t* pwcm);

/*! Run the DADA primary write client main loop */
int dada_pwc_main (dada_pwc_main_t* pwcm)
{
  if (!pwcm) {
    fprintf (stderr, "dada_pwc_main no main!\n");
    return EXIT_FAILURE;
  }

  if (!pwcm->pwc) {
    fprintf (stderr, "dada_pwc_main no PWC command connection\n");
    return EXIT_FAILURE;
  }

  if (!pwcm->log) {
    fprintf (stderr, "dada_pwc_main no logging facility\n");
    return EXIT_FAILURE;
  }

  if (!pwcm->start_function) {
    fprintf (stderr, "dada_pwc_main no start function\n");
    return EXIT_FAILURE;
  }

  if (!pwcm->buffer_function) {
    fprintf (stderr, "dada_pwc_main no buffer function\n");
    return EXIT_FAILURE;
  }

  if (!pwcm->stop_function) {
    fprintf (stderr, "dada_pwc_main no stop function\n");
    return EXIT_FAILURE;
  }

  while (!dada_pwc_quit (pwcm->pwc)) {

    /* Enter the idle/prepared state. */
    if (dada_pwc_main_prepare (pwcm) < 0)
      return EXIT_FAILURE;

    /* Start the data transfer. */
    if (dada_pwc_main_start_transfer (pwcm) < 0)
      return EXIT_FAILURE;

    /* Enter the clocking/recording state. */
    if (dada_pwc_main_transfer_data (pwcm) < 0)
      return EXIT_FAILURE;

    /* Stop the data transfer. */
    if (dada_pwc_main_stop_transfer (pwcm) < 0)
      return EXIT_FAILURE;

  } 

  return 0;
}

/*! The idle and prepared states of the DADA primary write client main loop */
int dada_pwc_main_prepare (dada_pwc_main_t* pwcm)
{
  /* get next available Header Block */
  if (pwcm->header_block) {
    pwcm->header_size = ipcbuf_get_bufsz (pwcm->header_block);
    pwcm->header = ipcbuf_get_next_write (pwcm->header_block);
    if (!pwcm->header) {
      multilog (pwcm->log, LOG_ERR, "Could not get next header block\n");
      return EXIT_FAILURE;
    }
  }

  /* ensure that Data Block is closed */
  if (pwcm->data_block && ipcio_is_open (pwcm->data_block)) {
    if (ipcio_close (pwcm->data_block) < 0)
    {
      multilog (pwcm->log, LOG_ERR, "Could not close Data Block\n");
      return EXIT_FAILURE;
    }
  } 

  while (!dada_pwc_quit (pwcm->pwc)) {

    pwcm->command = dada_pwc_command_get (pwcm->pwc);

    if (pwcm->command.code == dada_pwc_header)  {

      multilog (pwcm->log, LOG_INFO, 
		"HEADER START\n%s\nHEADER END\n", pwcm->command.header);

      if (pwcm->header_block)
	strncpy (pwcm->header, pwcm->command.header, pwcm->header_size);

      dada_pwc_set_state (pwcm->pwc, dada_pwc_prepared, 0);

    }

    else if (pwcm->command.code == dada_pwc_clock)  {

      multilog (pwcm->log, LOG_INFO, "Start clocking data\n");

      if (pwcm->command.byte_count) {
	multilog (pwcm->log, LOG_ERR, "dada_pwc_main_idle internal error.  "
		 "byte count specified in CLOCK command\n");
	return EXIT_FAILURE;
      }

      /* Open the Data Block in clocking mode */
      if (pwcm->data_block && ipcio_open (pwcm->data_block, 'w') < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not open data block\n");
        return EXIT_FAILURE;
      }

      /* leave the idle state loop */
      return 0;
	
    }

    else if (pwcm->command.code == dada_pwc_start)  {

      multilog (pwcm->log, LOG_INFO, "Start recording data\n");

      if (pwcm->command.byte_count)
	multilog (pwcm->log, LOG_INFO,
		  "Will record %"PRIu64" bytes\n", pwcm->command.byte_count);

      /* Open the Data Block in recording mode */
      if (pwcm->data_block && ipcio_open (pwcm->data_block, 'W') < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not open data block\n");
        return EXIT_FAILURE;
      }
      
      /* leave the idle state loop */
      return 0;

    }
    else {
      multilog (pwcm->log, LOG_ERR, "dada_pwc_main_prepare internal error = "
	       "unexpected command code %d\n", pwcm->command.code);
      return EXIT_FAILURE;
    }

  }

  return 0;
}

/*! The transit to clocking/recording state */
int dada_pwc_main_start_transfer (dada_pwc_main_t* pwcm)
{
  /* If utc != 0, the start function should attempt to start the data
     transfer at the specified utc.  Otherwise, start as soon as
     possible.  Regardless, the start function should return the UTC
     of the first time sample to be transfered to the Data Block.
  */

  time_t utc = pwcm->start_function (pwcm, pwcm->command.utc);

  unsigned buffer_size = 64;
  static char* buffer = 0;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

  if (utc <= 0) {
    multilog (pwcm->log, LOG_ERR, "start_function returned invalid UTC\n");
    return EXIT_FAILURE;
  }

  strftime (buffer, buffer_size, DADA_TIMESTR, gmtime (&utc));
  multilog (pwcm->log, LOG_INFO, "UTC_START = %s\n", buffer);

  /* make header available on Header Block */
  if (pwcm->header_block) {

    /* write UTC_START to the header */
    if (ascii_header_set (pwcm->header, "UTC_START", "%s", buffer) < 0) {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return EXIT_FAILURE;
    }

    if (ipcbuf_mark_filled (pwcm->header_block, pwcm->header_size) < 0)  {
      multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
      return EXIT_FAILURE;
    }

  }

  if (pwcm->command.code == dada_pwc_clock)
    return dada_pwc_set_state (pwcm->pwc, dada_pwc_clocking, utc);

  else if (pwcm->command.code == dada_pwc_start) 
    return dada_pwc_set_state (pwcm->pwc, dada_pwc_recording, utc);

  multilog (pwcm->log, LOG_ERR, "dada_pwc_main_start_transfer"
	    " internal error = invalid state\n");
  return EXIT_FAILURE;
}


/*! The clocking and recording states of the DADA PWC main loop */
int dada_pwc_main_transfer_data (dada_pwc_main_t* pwcm)
{
  /* the number of bytes written to the Data Block */
  uint64_t total_bytes_written = 0;

  /* the byte at which the state will change */
  uint64_t transit_byte = pwcm->command.byte_count;

  /* the number of bytes to write to the Data Block */
  uint64_t bytes_to_write = pwcm->command.byte_count;

  /* the number of bytes to be copied from buffer */
  uint64_t buf_bytes = 0;

  /* pointer to the data buffer */
  char* buffer = 0;

  /* size of the data buffer */
  uint64_t buffer_size = 0;

  while (!dada_pwc_quit (pwcm->pwc)) {

    /* check to see if a new command has been registered */
    if (dada_pwc_command_check (pwcm->pwc))  {

      pwcm->command = dada_pwc_command_get (pwcm->pwc);

      multilog (pwcm->log, LOG_INFO, "command code %d\n", pwcm->command.code);
      
      if (pwcm->command.code == dada_pwc_record_stop)
	multilog (pwcm->log, LOG_INFO, "recording->clocking");
      else if (pwcm->command.code == dada_pwc_record_start)
	multilog (pwcm->log, LOG_INFO, "clocking->recording");
      else if (pwcm->command.code == dada_pwc_stop)
	multilog (pwcm->log, LOG_INFO, "stopping");
      else {
	multilog (pwcm->log, LOG_ERR,
		  "dada_pwc_main_transfer data internal error = "
		  "unexpected command code %d\n", pwcm->command.code);
	return EXIT_FAILURE;
      }

      if (pwcm->command.byte_count > total_bytes_written) {
	transit_byte = pwcm->command.byte_count;
	bytes_to_write = pwcm->command.byte_count - total_bytes_written;
	multilog (pwcm->log, LOG_INFO, 
		  " in %"PRIu64" bytes\n", bytes_to_write);
      }

      else {
	multilog (pwcm->log, LOG_INFO, " immediately\n");
	transit_byte = total_bytes_written;
	bytes_to_write = 0;

	if (pwcm->command.byte_count &&
	    pwcm->command.byte_count < total_bytes_written)
	  multilog (pwcm->log, LOG_NOTICE,
		    "requested transit byte=%"PRIu64" passed\n",
		    pwcm->command.byte_count);
      }
	
    }
      
    if (pwcm->pwc->state == dada_pwc_recording)
      multilog (pwcm->log, LOG_INFO, "recording\n");
    else if (pwcm->pwc->state == dada_pwc_clocking)
      multilog (pwcm->log, LOG_INFO, "clocking\n");

    if (transit_byte) {

      if (pwcm->command.code ==  dada_pwc_record_stop)
	multilog (pwcm->log, LOG_INFO, "record stop in %"PRIu64" bytes\n",
		  bytes_to_write);
      else if (pwcm->command.code ==  dada_pwc_record_start)
	multilog (pwcm->log, LOG_INFO, "record start in %"PRIu64" bytes\n",
		  bytes_to_write);
      else if (pwcm->command.code ==  dada_pwc_record_stop)
	multilog (pwcm->log, LOG_INFO, "stop in %"PRIu64" bytes\n",
		  bytes_to_write);

    }

    if (!transit_byte || bytes_to_write) {

      /* get the next data buffer */
      buffer = pwcm->buffer_function (pwcm, &buffer_size);

      if (!buffer || !buffer_size) {
        multilog (pwcm->log, LOG_ERR, "buffer function error\n");
        return EXIT_FAILURE;
      }

      /* If the transit_byte is set, do not write more than the requested
	 amount of data to the Data Block */
      buf_bytes = buffer_size;
      if (transit_byte && buf_bytes > bytes_to_write)
	buf_bytes = bytes_to_write;

      /* write the bytes to the Data Block */
      if (pwcm->data_block &&
	  ipcio_write (pwcm->data_block, buffer, buf_bytes) < buf_bytes) {

	multilog (pwcm->log, LOG_ERR, "Cannot write %"PRIu64
		  " bytes to Data Block\n", buf_bytes);

	return EXIT_FAILURE;

      }

      total_bytes_written += buf_bytes;

      if (bytes_to_write)
	bytes_to_write -= buf_bytes;

      multilog (pwcm->log, LOG_INFO, "Written %"PRIu64" bytes\n",
		total_bytes_written);

    }

    fprintf (stderr, "transit=%"PRIu64" total=%"PRIu64"\n",
	     transit_byte, total_bytes_written);

    if (transit_byte == total_bytes_written) {

      /* The transit_byte has been reached, it is now time to change state */

      /* reset the transit_byte */ 
      transit_byte = 0;

      if (pwcm->command.code ==  dada_pwc_record_stop) {

	multilog (pwcm->log, LOG_INFO, "record stop\n");
	
	/* stop valid data on the Data Block at the last byte written */
	if (pwcm->data_block && ipcio_stop (pwcm->data_block) < 0)  {
	  multilog (pwcm->log, LOG_ERR, "Could not stop data block\n");
	  return EXIT_FAILURE;
	}
	
	dada_pwc_set_state (pwcm->pwc, dada_pwc_clocking, 0);
	
      }

      else if (pwcm->command.code ==  dada_pwc_record_start) {

	multilog (pwcm->log, LOG_INFO, "record start\n");
	
	/* start valid data on the Data Block at the requested byte */
	if (pwcm->data_block && 
	    ipcio_start (pwcm->data_block, pwcm->command.byte_count) < 0)  {
	  multilog (pwcm->log, LOG_ERR, "Could not start data block"
		    " at %"PRIu64"\n", pwcm->command.byte_count);
	  return EXIT_FAILURE;
	}
	
	dada_pwc_set_state (pwcm->pwc, dada_pwc_recording, 0);
	
      }

      else {

	multilog (pwcm->log, LOG_INFO, "stopping\n");

	/* enter the idle state */
	return 0;

      }

      /* When changing between clocking and recording states, there
	 may remain unwritten data in the buffer.  Write this data to
	 the Data Block */

      /* offset the buffer pointer by the amount already written */
      buffer += buf_bytes;
      /* set the number of remaining samples */
      buf_bytes = buffer_size - buf_bytes;

      if (buf_bytes && pwcm->data_block) {

	/* write the bytes to the Data Block */
	if (ipcio_write (pwcm->data_block, buffer, buf_bytes) < buf_bytes) {

	  multilog (pwcm->log, LOG_ERR, "Cannot write %"PRIu64
		    " bytes to Data Block\n", buf_bytes);
	  return EXIT_FAILURE;

	}

	total_bytes_written += buf_bytes;

      }

    }

  }

  return 0;

}

/*! The transit to idle/prepared state */
int dada_pwc_main_stop_transfer (dada_pwc_main_t* pwcm)
{
  if (pwcm->stop_function (pwcm) < 0)  {
    multilog (pwcm->log, LOG_ERR, "dada_pwc_main_stop_transfer"
	      " stop function returned error code\n");
    return EXIT_FAILURE;
  }

  /* close the Data Block */
     
  if (pwcm->data_block && ipcio_close (pwcm->data_block) < 0)  {
    multilog (pwcm->log, LOG_ERR, "Could not close Data Block\n");
    return EXIT_FAILURE;
  }
  
  dada_pwc_set_state (pwcm->pwc, dada_pwc_idle, 0);

  return 0;
}
