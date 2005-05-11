#include "dada_pwc.h"
#include "multilog.h"

/*! Create a new DADA primary write client main loop */
dada_pwc_main_t* dada_pwc_main_create ()
{
  dada_pwc_main_t* main = malloc (sizeof(dada_pwc_main_t));

  main -> pwc = 0;

  return main;
}

/*! Destroy a DADA primary write client main loop */
void dada_pwc_main_destroy (dada_pwc_main_t* main)
{
  free (main);
}

/*! The idle and prepared states of the DADA primary write client main loop */
int dada_pwc_main_idle_prepared (dada_pwc_main_t* main)
{
  while (!dada_pwc_quit (main->pwc)) {

    main->command = dada_pwc_command_get (main->pwc);

    if (main->command.code == dada_pwc_header)  {

      fprintf (stderr, "HEADER START\n%s\nHEADER END\n", main->command.header);

      /* here is where the header would be written to the Header Block e.g. 

      memcpy (header_buf, command.header, header_size);
      if (ipcbuf_mark_filled (&header_block, header_size) < 0)  {
        multilog (log, LOG_ERR, "Could not mark filled header block\n");
        return EXIT_FAILURE;
      }

      */

      dada_pwc_set_state (main->pwc, dada_pwc_prepared, time(0));

    }

    else if (main->command.code == dada_pwc_clock)  {

      fprintf (stderr, "Start clocking data\n");

      if (main->command.byte_count) {
	fprintf (stderr, "dada_pwc_main_idle internal error.  "
		 "byte count specified in CLOCK command\n");
	return -1;
      }

      /* here is where the Data Block would be set to over-write e.g.
	 
      if (ipcio_open (data_block, 'w') < 0)  {
        multilog (log, LOG_ERR, "Could not open data block\n");
        return EXIT_FAILURE;
      }
      
      */

      /* leave the idle state loop */
      break;
	
    }

    else if (main->command.code == dada_pwc_start)  {

      fprintf (stderr, "Start recording data\n");

      if (main->command.byte_count)
	fprintf (stderr, "Will record %"PRIu64" bytes\n", bytes_to_write);

      /* here is where the Data Block would be set to over-write e.g.

      if (ipcio_open (data_block, 'W') < 0)  {
        multilog (log, LOG_ERR, "Could not open data block\n");
        return EXIT_FAILURE;
      }
      
      */

      /* leave the idle state loop */
      break;

    }
    else {
      fprintf (stderr, "dada_pwc_main_idle internal error.  "
	       "unexpected command code %d\n", main->command.code);
      return -1;
    }

  }

  return 0;
}

/*! The idle and prepared states of the DADA primary write client main loop */
int dada_pwc_main_start_transfer (dada_pwc_main_t* main)
{
  time_t utc = main->command.utc;

  /* If utc != 0, the start function should attempt to start the data
     transfer at the specified utc.  Otherwise, start as soon as
     possible.  Regardless, the start function should return the UTC
     of the first time sample to be transfered to the Data Block
  */

  utc = *(main->command->start_function) (main, utc, main->context);

  if (utc <= 0) {


    /* here is where the PiC should be armed. */

    /* here time(0) should be replaced by the actual clock start time */ 

    if (command.code == dada_pwc_clock) 
      dada_pwc_set_state (main->pwc, dada_pwc_clocking, time(0));
    else
      dada_pwc_set_state (main->pwc, dada_pwc_recording, time(0));


/*! Run the DADA primary write client main loop */
int dada_pwc_main (dada_pwc_main_t* main)
{

  if (!main) {
    fprint (stderr, "dada_pwc_main no main!\n");
    return -1;
  }

  if (!main->pwc) {
    fprint (stderr, "dada_pwc_main no PWC command connection!\n");
    return -1;
  }

  while (!dada_pwc_quit (main->pwc)) {

    /* Enter the idle state. */
    if (dada_pwc_main_idle_prepared (main) < 0)
      return -1;

    /* Start the data transfer. */
    if (dada_pwc_main_start_transfer (main) < 0)
      return -1;



  /* the number of bytes written to the Data Block */
  uint64_t total_bytes_written = 0;

  /* the number of bytes to write to the Data Block */
  uint64_t bytes_to_write = 0;

  /* the number of bytes written each time */
  uint64_t write_bytes = 0;

  /* flag set when a state change has been requested */
  int change_state = 0;

    /* currently in the clocking/recording state */

    while (!dada_pwc_quit (main->pwc)) {

      /* check to see if a command has been registered */
      if (dada_pwc_command_check (main->pwc))  {

	command = dada_pwc_command_get (main->pwc);

	fprintf (stderr, "got command code %d\n", command.code);

	if (command.code == dada_pwc_record_stop ||
	    command.code == dada_pwc_record_start)  {

	  if (command.code == dada_pwc_record_stop)
	    fprintf (stderr, "recording->clocking\n");
	  else
	    fprintf (stderr, "clocking->recording\n");

	  if (!command.byte_count)  {
	    fprintf (stderr, "Effective immediately\n");
            bytes_to_write = 0;
          }

	  else if (command.byte_count < total_bytes_written) {
	    fprintf (stderr, "Already passed byte_count=%"PRIu64"\n",
		     command.byte_count);
	    bytes_to_write = 0;
	  }

	  else {
	    bytes_to_write = command.byte_count - total_bytes_written;
	    fprintf (stderr, "Will occur in %"PRIu64" bytes\n", bytes_to_write);
	  }

          change_state = 1;

	}

	else if (command.code == dada_pwc_stop)  {

	  fprintf (stderr, "stopping\n");
 
	  /* here is where the Data Block would be closed e.g.

	  if (ipcio_close (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not close data block\n");
	  return EXIT_FAILURE;
	  }

	  */

	  dada_pwc_set_state (main->pwc, dada_pwc_idle, time(0));

	  /* enter the idle state */
	  break;

	}

      }

      if (main->pwc->state == dada_pwc_recording) {
	fprintf (stderr, "recording\n");
	if (command.code ==  dada_pwc_record_stop)
	  fprintf (stderr, "record stop in %"PRIu64" bytes\n", bytes_to_write);
      }
      else if (main->pwc->state == dada_pwc_clocking) {
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

      if ((bytes_to_write && write_bytes == bytes_to_write) ||
          (!bytes_to_write && change_state))
      {
        change_state = 0;

	/* it is now time to change state */

	if (command.code ==  dada_pwc_record_stop) {

	  fprintf (stderr, "record stop\n");
	
	  /* here is where the Data Block would be stopped e.g.
	   
	  if (ipcio_stop (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not stop data block\n");
	  return EXIT_FAILURE;
	  }
	   
	  */
	
	  dada_pwc_set_state (main->pwc, dada_pwc_clocking, time(0));
	
	}

	else if (command.code ==  dada_pwc_record_start) {

	  fprintf (stderr, "record start\n");
	
	  /* here is where the Data Block would be started e.g.
	   
	  if (ipcio_start, (data_block, command.byte_count) < 0)  {
	  multilog (log, LOG_ERR, "Could not stop data block\n");
	  return EXIT_FAILURE;
	  }
	   
	  */
	
	  dada_pwc_set_state (main->pwc, dada_pwc_recording, time(0));
	
	}

	else if (main->pwc->state == dada_pwc_recording) {

	  fprintf (stderr, "stopping\n");
 
	  /* here is where the Data Block would be closed e.g.

	  if (ipcio_close (data_block) < 0)  {
	  multilog (log, LOG_ERR, "Could not close data block\n");
	  return EXIT_FAILURE;
	  }

	  */

	  dada_pwc_set_state (main->pwc, dada_pwc_idle, time(0));

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
  dada_pwc_destroy (main->pwc);

  return 0;
}

