#include "dada_pwc.h"
#include "multilog.h"

#include <unistd.h>

int main ()
{
  dada_pwc_t* pwc = 0;
  dada_pwc_command_t command = DADA_PWC_COMMAND_INIT;

  fprintf (stderr, "Creating dada_pwc\n");
  pwc = dada_pwc_create ();

  if (dada_pwc_serve (pwc) < 0) {
    fprintf (stderr, "test_dada_pwc: could not start server\n");
    return -1;
  }

  while (!dada_pwc_quit (pwc)) {

    /* currently in idle state ... */

    while (!dada_pwc_quit (pwc)) {

      command = dada_pwc_command_get (pwc);

      if (command.code == dada_pwc_header)  {

        fprintf (stderr, "HEADER=%s", command.header);

        /* here is where the header would be written to the Header Block e.g. 

        memcpy (header_buf, command.header, header_size);
        if (ipcbuf_mark_filled (&header_block, header_size) < 0)  {
          multilog (log, LOG_ERR, "Could not mark filled header block\n");
          return EXIT_FAILURE;
        }

        */

        dada_pwc_command_ack (pwc, dada_pwc_prepared, time(0));

      }

      else if (command.code == dada_pwc_clock)  {

        fprintf (stderr, "Start clocking data\n");

        /* here is where the Data Block would be set to over-write e.g.

        if (ipcio_open (data_block, 'w') < 0)  {
          multilog (log, LOG_ERR, "Could not open data block\n");
          return EXIT_FAILURE;
        }

        */

        /* here time(0) should be replaced by the actual clock start time */ 
        dada_pwc_command_ack (pwc, dada_pwc_clocking, time(0));

        /* enter the write loop */
        break;

      }

      else if (command.code == dada_pwc_start)  {

        fprintf (stderr, "Start recording data\n");

        /* here is where the Data Block would be set to over-write e.g.

        if (ipcio_open (data_block, 'W') < 0)  {
          multilog (log, LOG_ERR, "Could not open data block\n");
          return EXIT_FAILURE;
        }

        */

        /* here time(0) should be replaced by the actual record start time */
        dada_pwc_command_ack (pwc, dada_pwc_recording, time(0));

        /* enter the write loop */
        break;

      }

    }

    /* currently in the clocking/recording state */

    while (!dada_pwc_quit (pwc)) {

      /* check to see if a command has been registered */
      if (dada_pwc_command_check (pwc))  {

        command = dada_pwc_command_get (pwc);

        if (command.code == dada_pwc_record_stop)  {

          fprintf (stderr, "recording->clocking\n");

          /* here is where the Data Block would be stopped e.g.

          if (ipcio_stop (data_block) < 0)  {
            multilog (log, LOG_ERR, "Could not stop data block\n");
            return EXIT_FAILURE;
          }

          */

          dada_pwc_command_ack (pwc, dada_pwc_clocking, time(0));

        }

        if (command.code == dada_pwc_record_start)  {

          fprintf (stderr, "clocking->recording\n");

          /* here is where the Data Block would be started e.g.

          if (ipcio_start, (data_block, command.byte_count) < 0)  {
            multilog (log, LOG_ERR, "Could not stop data block\n");
            return EXIT_FAILURE;
          }

          */

          dada_pwc_command_ack (pwc, dada_pwc_recording, time(0));

        }

        else if (command.code == dada_pwc_stop)  {

          fprintf (stderr, "stopping\n");
 
          /* here is where the Data Block would be closed e.g.

          if (ipcio_close (data_block) < 0)  {
            multilog (log, LOG_ERR, "Could not close data block\n");
            return EXIT_FAILURE;
          }

          */

          dada_pwc_command_ack (pwc, dada_pwc_idle, time(0));

          /* enter the idle state */
          break;

        }

      }

      if (pwc->state == dada_pwc_recording)
        fprintf (stderr, "recording\n");
      else if (pwc->state == dada_pwc_clocking)
        fprintf (stderr, "clocking\n");

      sleep (1);

      /* here is where the writing would occur e.g.

      if ( (data = edt_wait_for_buffers(edt_p, 1)) == NULL){
        edt_perror("edt_wait");
        multilog(log,LOG_ERR,"edt_wait error\n");
        return EXIT_FAILURE;
      }

      if (( ipcio_write(&data_block, data, bufsize) ) < bufsize ){
        multilog(log, LOG_ERR, "Cannot write requested bytes to SHM\n");
        return EXIT_FAILURE;
      }

      */

    }

  }

  fprintf (stderr, "Destroying pwc\n");
  dada_pwc_destroy (pwc);

  return 0;
}

