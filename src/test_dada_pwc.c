#include "dada_pwc.h"
#include "multilog.h"

#include <unistd.h>

int main ()
{
  dada_pwc_t* pwc = 0;
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;

  /* the number of bytes written to the Data Block */
  uint64_t total_bytes_written = 0;

  /* the number of bytes to write to the Data Block */
  uint64_t bytes_to_write = 0;

  /* the number of bytes written each time */
  uint64_t write_bytes = 0;

  fprintf (stderr, "Creating dada_pwc\n");
  pwc = dada_pwc_create ();

  if (dada_pwc_serve (pwc) < 0) {
    fprintf (stderr, "test_dada_pwc: could not start server\n");
    return -1;
  }

  while (!dada_pwc_quit (pwc)) {

    /* currently in idle state ... */

    total_bytes_written = 0;
    bytes_to_write = 0;

    while (!dada_pwc_quit (pwc)) {

      command = dada_pwc_command_get (pwc);

      if (command.code == dada_pwc_header)  {

        fprintf (stderr, "HEADER START\n%s\nHEADER END\n", command.header);

        /* here is where the header would be written to the Header Block e.g. 

        memcpy (header_buf, command.header, header_size);
        if (ipcbuf_mark_filled (&header_block, header_size) < 0)  {
	multilog (log, LOG_ERR, "Could not mark filled header block\n");
	return EXIT_FAILURE;
        }

        */

        dada_pwc_set_state (pwc, dada_pwc_prepared, time(0));

      }

      else if (command.code == dada_pwc_clock)  {

        fprintf (stderr, "Start clocking data\n");

        if (command.byte_count)
          fprintf (stderr, "There shouldn't be a byte count for clocking\n");

        /* here is where the Data Block would be set to over-write e.g.

        if (ipcio_open (data_block, 'w') < 0)  {
	multilog (log, LOG_ERR, "Could not open data block\n");
	return EXIT_FAILURE;
        }

        */

        /* enter the write loop */
        break;

      }

      else if (command.code == dada_pwc_start)  {

        fprintf (stderr, "Start recording data\n");

        if (command.byte_count) {
	  bytes_to_write = command.byte_count;
          fprintf (stderr, "Will record %"PRIu64" bytes\n", bytes_to_write);
	}

        /* here is where the Data Block would be set to over-write e.g.

        if (ipcio_open (data_block, 'W') < 0)  {
	multilog (log, LOG_ERR, "Could not open data block\n");
	return EXIT_FAILURE;
        }

        */

        /* enter the write loop */
        break;

      }

    }


    /* If a utc is specified, the process should sleep until the
       specified UTC.  Otherwise, start immediately. */
    if (command.utc) {


    }      

    /* here is where the PiC should be armed. */

    /* here time(0) should be replaced by the actual clock start time */ 

    if (command.code == dada_pwc_clock) 
      dada_pwc_set_state (pwc, dada_pwc_clocking, time(0));
    else
      dada_pwc_set_state (pwc, dada_pwc_recording, time(0));

    /* currently in the clocking/recording state */

    while (!dada_pwc_quit (pwc)) {

      /* check to see if a command has been registered */
      if (dada_pwc_command_check (pwc))  {

	command = dada_pwc_command_get (pwc);

	fprintf (stderr, "got command code %d\n", command.code);

	if (command.code == dada_pwc_record_stop ||
	    command.code == dada_pwc_record_start)  {

	  if (command.code == dada_pwc_record_stop)
	    fprintf (stderr, "recording->clocking\n");
	  else
	    fprintf (stderr, "clocking->recording\n");

	  if (!command.byte_count)
	    fprintf (stderr, "Error no byte count\n");

	  else if (command.byte_count < total_bytes_written) {
	    fprintf (stderr, "Already passed byte_count=%"PRIu64"\n",
		     command.byte_count);
	    bytes_to_write = 0;
	  }

	  else {
	    bytes_to_write = command.byte_count - total_bytes_written;
	    fprintf (stderr, "Will occur in %"PRIu64" bytes\n", bytes_to_write);
	  }

	}

	else if (command.code == dada_pwc_stop)  {

	  fprintf (stderr, "stopping\n");
 
	  /* here is where the Data Block would be closed e.g.

	  if (ipcio_close (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not close data block\n");
	  return EXIT_FAILURE;
	  }

	  */

	  dada_pwc_set_state (pwc, dada_pwc_idle, time(0));

	  /* enter the idle state */
	  break;

	}

      }

      if (pwc->state == dada_pwc_recording) {
	fprintf (stderr, "recording\n");
	if (command.code ==  dada_pwc_record_stop)
	  fprintf (stderr, "record stop in %"PRIu64" bytes\n", bytes_to_write);
      }
      else if (pwc->state == dada_pwc_clocking) {
	fprintf (stderr, "clocking\n");
	if (command.code ==  dada_pwc_record_start)
	  fprintf (stderr, "record start in %"PRIu64" bytes\n", bytes_to_write);
      }

      /* Actual buffer size will be the size of the EDT buffer.  Here,
	 we simulate writing one second of data */
      write_bytes = pwc->bytes_per_second;

      /* Ensure that the buffer size is not greater than the number of
	 bytes left to write. */
      if (bytes_to_write && write_bytes > bytes_to_write)
	write_bytes = bytes_to_write;


      /* here is where the writing would occur e.g.

      if ( (data = edt_wait_for_buffers(edt_p, 1)) == NULL){
      edt_perror("edt_wait");
      multilog(log,LOG_ERR,"edt_wait error\n");
      return EXIT_FAILURE;
      }

      if (( ipcio_write(&data_block, data, write_bytes) ) < write_bytes ){
      multilog(log, LOG_ERR, "Cannot write requested bytes to SHM\n");
      return EXIT_FAILURE;
      }

      */

      total_bytes_written += write_bytes;

      fprintf (stderr, "Written %"PRIu64" bytes\n", total_bytes_written);

      if (bytes_to_write && write_bytes == bytes_to_write) {

	/* it is now time to change state */

	if (command.code ==  dada_pwc_record_stop) {

	  fprintf (stderr, "record stop\n");
	
	  /* here is where the Data Block would be stopped e.g.
	   
	  if (ipcio_stop (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not stop data block\n");
	  return EXIT_FAILURE;
	  }
	   
	  */
	
	  dada_pwc_set_state (pwc, dada_pwc_clocking, time(0));
	
	}

	else if (command.code ==  dada_pwc_record_start) {

	  fprintf (stderr, "record start\n");
	
	  /* here is where the Data Block would be started e.g.
	   
	  if (ipcio_start, (data_block, command.byte_count) < 0)  {
	  multilog (log, LOG_ERR, "Could not stop data block\n");
	  return EXIT_FAILURE;
	  }
	   
	  */
	
	  dada_pwc_set_state (pwc, dada_pwc_recording, time(0));
	
	}

	else if (pwc->state == dada_pwc_recording) {

	  fprintf (stderr, "stopping\n");
 
	  /* here is where the Data Block would be closed e.g.

	  if (ipcio_close (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not close data block\n");
	  return EXIT_FAILURE;
	  }

	  */

	  dada_pwc_set_state (pwc, dada_pwc_idle, time(0));

	  /* enter the idle state */
	  break;

	}

	/* If we are changing between clocking and recording (et vice versa),
	   then there still may remain data in the EDT buffer.  Write this
	   data to the Data Block */

	/* Actual buffer size will be the size of the EDT buffer.  Here,
	   we simulate writing one second of data */
	write_bytes = pwc->bytes_per_second - bytes_to_write;

	/* here is where the writing would occur e.g.

	offset the buffer by the number of bytes already written
	data += bytes_to_write;

	if (( ipcio_write(&data_block, data, write_bytes) ) < write_bytes ){
	multilog(log, LOG_ERR, "Cannot write requested bytes to SHM\n");
	return EXIT_FAILURE;
	}

	*/

	total_bytes_written += write_bytes;

	/* the transition in state is completed */
	bytes_to_write = 0;

      }
      else 
	bytes_to_write -= write_bytes;

      sleep (1);

    }

  }

  fprintf (stderr, "Destroying pwc\n");
  dada_pwc_destroy (pwc);

  return 0;
}

