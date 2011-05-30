#include "dada_pwc_main.h"
#include "dada_def.h"

#include "ascii_header.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

// #define _DEBUG 1

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
  pwcm -> block_function = 0;
  pwcm -> stop_function = 0;
  pwcm -> error_function = 0;
  pwcm -> header_valid_function = 0;
  pwcm -> xfer_pending_function = 0;
  pwcm -> new_xfer_function = 0;

  pwcm -> context = 0;
  pwcm -> header = 0;
  pwcm -> header_size = 0;
  pwcm -> verbose = 0;
  pwcm -> header_valid = 0;

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

/*! do the data transfer, via zero copy block operations */
int dada_pwc_main_transfer_data_block (dada_pwc_main_t* pwcm);

/*! stop the data transfer */
int dada_pwc_main_stop_transfer (dada_pwc_main_t* pwcm);

/*! Run the DADA primary write client main loop */
int dada_pwc_main (dada_pwc_main_t* pwcm)
{
 
  if (!pwcm) {
    fprintf (stderr, "dada_pwc_main no main!\n");
    return -1;
  }

  if (!pwcm->pwc) {
    fprintf (stderr, "dada_pwc_main no PWC command connection\n");
    return -1;
  }

  if (!pwcm->log) {
    fprintf (stderr, "dada_pwc_main no logging facility\n");
    return -1;
  }

  if (!pwcm->start_function) {
    fprintf (stderr, "dada_pwc_main no start function\n");
    return -1;
  }

  if (!pwcm->buffer_function) {
    fprintf (stderr, "dada_pwc_main no buffer function\n");
    return -1;
  }

  if (!pwcm->stop_function) {
    fprintf (stderr, "dada_pwc_main no stop function\n");
    return -1;
  }

  if (!pwcm->pwc->log)
    pwcm->pwc->log = pwcm->log;


  /* the return value can be
   *  0:  no error
   * -1:  soft error   pwc_command will try to restart
   * -2:  hard error   fatal, but data block intact, dada clients ok
   * -3:  fatal error  fatal, data block damaged, dada clients hung
   */
  int rval = 0;

  while (!dada_pwc_quit (pwcm->pwc))
  {
    /* Enter the idle/prepared state. */
    rval = dada_pwc_main_prepare (pwcm);

    if (dada_pwc_quit (pwcm->pwc))
      break;

    if (rval < 0)
      dada_pwc_main_process_error (pwcm, rval);
    else
    {
      /* Start the data transfer. */
      rval = dada_pwc_main_start_transfer (pwcm);
      if (rval < 0) 
        dada_pwc_main_process_error (pwcm, rval);
      else {

        /* Enter the clocking/recording state. */
        if (pwcm->block_function)
          rval = dada_pwc_main_transfer_data_block (pwcm);
        else
          rval = dada_pwc_main_transfer_data (pwcm);
        if (rval < 0) 
          dada_pwc_main_process_error (pwcm, rval); 

      }

      /* Stop the data transfer. */
      rval = dada_pwc_main_stop_transfer (pwcm);
      if (rval < 0) 
        dada_pwc_main_process_error (pwcm, rval);

    } 

    /* If we ever reach a fatal state, try to exit */
    if (pwcm->pwc->state == dada_pwc_fatal_error) 
      pwcm->pwc->quit = 1;

  }

  return rval;

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
      return DADA_ERROR_HARD;
    }
  }

  /* ensure that Data Block is closed */
  if (pwcm->data_block && ipcio_is_open (pwcm->data_block)
      && ipcio_close (pwcm->data_block) < 0)
  {
    multilog (pwcm->log, LOG_ERR, "Could not close Data Block\n");
    return DADA_ERROR_HARD;
  }

  while (!dada_pwc_quit (pwcm->pwc))
  {
    pwcm->command = dada_pwc_command_get (pwcm->pwc);

    if (dada_pwc_quit (pwcm->pwc))
      break;

    if (pwcm->command.code == dada_pwc_reset)
    {
      dada_pwc_set_state (pwcm->pwc, dada_pwc_idle, 0);
    }
    
    else if (pwcm->command.code == dada_pwc_header)
    {
#ifdef _DEBUG 
      multilog (pwcm->log, LOG_INFO, 
                "HEADER START\n%s\nHEADER END\n", pwcm->command.header);
#endif

      if (pwcm->header_block)
        strncpy (pwcm->header, pwcm->command.header, pwcm->header_size);

      dada_pwc_set_state (pwcm->pwc, dada_pwc_prepared, 0);
    }

    else if (pwcm->command.code == dada_pwc_clock)
    {
      /* multilog (pwcm->log, LOG_INFO, "Start clocking data\n"); */

      if (pwcm->command.byte_count) {
        multilog (pwcm->log, LOG_ERR, "dada_pwc_main_idle internal error.  "
                 "byte count specified in CLOCK command\n");
        return DADA_ERROR_SOFT;
      }

      /* Open the Data Block in clocking mode */
      if (pwcm->data_block && ipcio_open (pwcm->data_block, 'w') < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not open data block\n");
        return DADA_ERROR_HARD;
      }

      /* leave the idle state loop */
      return 0;
        
    }
    else if (pwcm->command.code == dada_pwc_start)
    {
#ifdef _DEBUG            
      multilog (pwcm->log, LOG_INFO, "Start recording data\n");
#endif

      if (pwcm->command.byte_count)
        multilog (pwcm->log, LOG_INFO,
                  "Will record %"PRIu64" bytes\n", pwcm->command.byte_count);

      /* Open the Data Block in recording mode */
      if (pwcm->data_block && ipcio_open (pwcm->data_block, 'W') < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not open data block\n");
        return DADA_ERROR_HARD;
      }
      
      /* leave the idle state loop */
      return 0;
    }
    else if (pwcm->command.code == dada_pwc_stop) 
    {
      if (pwcm->pwc->state == dada_pwc_soft_error) 
        multilog (pwcm->log, LOG_WARNING, "Resetting soft_error to idle\n");
      else
        multilog (pwcm->log, LOG_WARNING, "dada_pwc_main_prepare: Unexpected stop command\n");

      dada_pwc_set_state (pwcm->pwc, dada_pwc_idle, 0);
    }
    else
    {
      multilog (pwcm->log, LOG_ERR, "dada_pwc_main_prepare internal error = "
               "unexpected command code %s\n", 
               dada_pwc_cmd_code_string(pwcm->command.code));
      return DADA_ERROR_HARD;
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

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_main_start_transfer: call start function\n");
#endif

  time_t utc = pwcm->start_function (pwcm, pwcm->command.utc);

  unsigned buffer_size = 64;
  static char* buffer = 0;

  if (!buffer)
    buffer = malloc (buffer_size);
  assert (buffer != 0);

  if (utc < 0) {
    multilog (pwcm->log, LOG_ERR, "start_function returned invalid UTC\n");
    return DADA_ERROR_HARD;
  }

  /* If the start function, cannot provide the utc start, 0 
   * indicates that the UTC will be provided by the command
   * interface */
  if (utc == 0) 
    buffer = "UNKNOWN";
  else 
    strftime (buffer, buffer_size, DADA_TIMESTR, gmtime (&utc));

  multilog (pwcm->log, LOG_INFO, "UTC_START = %s\n", buffer);

  /* make header available on Header Block */
  if (pwcm->header_block) {

    /* write UTC_START to the header */
    if (ascii_header_set (pwcm->header, "UTC_START", "%s", buffer) < 0) {
      multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
      return DADA_ERROR_SOFT;
    }

    /* Set the primary UTC_START in the pwc if we have it */
    if (utc > 0) {

      /* Set the base utc from which all commands are calculated */
      pwcm->pwc->utc_start = utc;
      multilog(pwcm->log, LOG_INFO, "Setting pwcm->pwc->utc_start = %d\n",pwcm->pwc->utc_start);

    }

    /* We can only mark the header filled if we have a start command */
    if (pwcm->command.code == dada_pwc_start) {

      /* only mark header block filled if header is valid */
      if (pwcm->header_valid_function)
        pwcm->header_valid = pwcm->header_valid_function(pwcm);
      else 
        pwcm->header_valid = 1;

      if (pwcm->header_valid) {
#ifdef _DEBUG
          multilog(pwcm->log, LOG_INFO, "dada_pwc_main_start_transfer: Marking header filled\n");
#endif
        if ( ipcbuf_mark_filled (pwcm->header_block, pwcm->header_size) < 0)  {
          multilog (pwcm->log, LOG_ERR, "Could not marked header filled or command.code != start\n");
          return DADA_ERROR_HARD;
        }
      }
    }
  } 

#ifdef _DEBUG
  fprintf (stderr, "dada_pwc_main_start_transfer: exit change state\n");
#endif

  ipcbuf_set_soclock_buf ((ipcbuf_t*) pwcm->data_block);

  if (pwcm->command.code == dada_pwc_clock)
    return dada_pwc_set_state (pwcm->pwc, dada_pwc_clocking, utc);

  else if (pwcm->command.code == dada_pwc_start) 
    return dada_pwc_set_state (pwcm->pwc, dada_pwc_recording, utc);
 

  multilog (pwcm->log, LOG_ERR, "dada_pwc_main_start_transfer"
            " internal error = invalid state\n");
  return DADA_ERROR_HARD;
}

/*! Switch from clocking to recording states */
int dada_pwc_main_record_start (dada_pwc_main_t* pwcm)
{

  /* the minimum offset at which recording may start */
  uint64_t minimum_record_start = 0;

  /* the byte to which the UTC_START corresponds */
  uint64_t utc_start_byte = 0;

  /* the byte to which command will correspond */
  uint64_t command_start_byte = 0;

  /* the next header to be written */
  char* header = 0;

  /* Find the earliest byte recording can start on. NB this may
   * have wrapped around if we have been clocking for longer than
   * the data block length... */
  minimum_record_start = ipcio_get_start_minimum (pwcm->data_block);

  /* If the first data received did not occur on buf 0, then
   * we need to know this to correctly calculate the command
   * offset */
  utc_start_byte = ipcio_get_soclock_byte(pwcm->data_block);

  // multilog(pwcm->log, LOG_INFO, "utc_start_byte = %"PRIu64"\n",utc_start_byte);

  /* The actual byte this command corresponds to */
  command_start_byte = utc_start_byte + pwcm->command.byte_count;

  // multilog (pwcm->log, LOG_INFO, "minimum_record_start=%"PRIu64"\n",
  //           minimum_record_start);

  /* If the command is scheduled to occur earlier than is possible, then
   * we must delay that command to the earliest time */
  if (command_start_byte < minimum_record_start) {
    multilog (pwcm->log, LOG_ERR, "Requested start byte=%"PRIu64
              " reset to minimum=%"PRIu64"\n", command_start_byte, 
              minimum_record_start);

    /* This is the byte the command corresponds to */
    command_start_byte = minimum_record_start;

    /* This is the offset of the command from the utc_start (OBS_OFFSET) */
    pwcm->command.byte_count = minimum_record_start - utc_start_byte;
  }

  multilog (pwcm->log, LOG_INFO, "REC_START\n");
  multilog (pwcm->log, LOG_INFO, "pwcm->command.utc = %d\n",pwcm->command.utc);
  multilog (pwcm->log, LOG_INFO, "pwcm->pwc->utc_start = %d\n",pwcm->pwc->utc_start);

  /* Special case for rec_stop/rec_start toggling */
  header = ipcbuf_get_next_write (pwcm->header_block);
  if (header != pwcm->header) {
    /* the case if there is more than one sub-block in Header Block */
    memcpy (header, pwcm->header, pwcm->header_size);
    pwcm->header = header;
  }

  /* a UTC_START may be different for multiple PWC's, but the REC_START will 
   * be uniform, so on a REC_START we will change the UTC_START in the outgoing 
   * header to be UTC_START + OBS_OFFSET, to ensure uniformity across PWCs */

  time_t utc = pwcm->command.utc;
  int buffer_size = 64;
  char buffer[buffer_size];
  strftime (buffer, buffer_size, DADA_TIMESTR, gmtime (&utc));

  multilog (pwcm->log, LOG_INFO, "dada_pwc_main_record_start: UTC_START reset to REC_START = %s\n", buffer);

  if (ascii_header_set (pwcm->header, "UTC_START", "%s", buffer) < 0) {
    multilog (pwcm->log, LOG_ERR, "fail ascii_header_set UTC_START\n");
    return DADA_ERROR_HARD;
  }

  multilog (pwcm->log, LOG_INFO, "dada_pwc_main_record_start: OBS_OFFSET = 0\n");
  if (ascii_header_set (pwcm->header, "OBS_OFFSET", "%"PRIu64, 0) < 0) {
    multilog (pwcm->log, LOG_ERR, "fail ascii_header_set OBS_OFFSET\n");
    return DADA_ERROR_HARD;
  }

  multilog (pwcm->log, LOG_INFO,"command_start_byte = %"PRIu64", command.byte_"
            "count = %"PRIu64"\n",command_start_byte,pwcm->command.byte_count);

  /* start valid data on the Data Block at the requested byte */
  if (ipcio_start (pwcm->data_block, command_start_byte) < 0)  {
    multilog (pwcm->log, LOG_ERR, "Could not start data block"
              " at %"PRIu64"\n", command_start_byte);
    return DADA_ERROR_HARD;
  }

  /* If the header has not yet been made valid */
  if (!(pwcm->header_valid)) {

    if (pwcm->header_valid_function) 
      pwcm->header_valid = pwcm->header_valid_function(pwcm);
    else
      pwcm->header_valid = 1;

    if (pwcm->header_valid) {
#ifdef _DEBUG
        multilog(pwcm->log, LOG_INFO, "dada_pwc_main_record_start: Marking header filled\n");
#endif
      if (ipcbuf_mark_filled (pwcm->header_block, pwcm->header_size) < 0)  {
        multilog (pwcm->log, LOG_ERR, "Could not mark filled header\n");
        return DADA_ERROR_HARD;
      }
    } else {
      multilog (pwcm->log, LOG_ERR, "Cannot transit from clocking to recoding "
                "if when header is invalid\n");;
      return DADA_ERROR_HARD;
    }
  }

  return 0;

}

/*! The clocking and recording states of the DADA PWC main loop */
int dada_pwc_main_transfer_data (dada_pwc_main_t* pwcm)
{
  /* total number of bytes written to the Data Block */
  uint64_t total_bytes_written = 0;

  /* the byte at which the state will change */
  uint64_t transit_byte = pwcm->command.byte_count;

  /* the number of bytes to write to the Data Block */
  uint64_t bytes_to_write = pwcm->command.byte_count;

  /* the number of bytes to be copied from buffer */
  uint64_t buf_bytes = 0;

  /* the number of bytes written to the Data Block */
  int64_t bytes_written = 0;

  /* pointer to the data buffer */
  char* buffer = 0;

  /* size of the data buffer */
  int64_t buffer_size = 0;

  /* error state due to a 0 byte buffer function return */
  uint64_t data_transfer_error_state = 0;

  char* command_string = 0;

  /* flags and buffers for setting of UTC_START */
  uint64_t utc_start_set = 0;
  int utc_size = 1024;
  char utc_buffer[utc_size];

  int64_t first_byte_of_xfer = 1;

  /* if the pwcm supports xfers, get the first byte of data for the first xfer */
  if (pwcm->new_xfer_function) {
    total_bytes_written = pwcm->new_xfer_function(pwcm);
    if (pwcm->verbose)
      multilog (pwcm->log, LOG_INFO, "transfer_data: first byte expected on %"PRIu64"\n", total_bytes_written);
  }

  /* Check if the UTC_START is valid */
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    multilog (pwcm->log, LOG_ERR, "Could not read UTC_START from header\n");
    return DADA_ERROR_HARD;
  }

  if (strcmp(utc_buffer,"UNKNOWN") != 0) 
    utc_start_set = 1;

#ifdef _DEBUG
  fprintf (stderr, "transfer_data: enter main loop\n");
#endif

  while ((!dada_pwc_quit (pwcm->pwc)) && (data_transfer_error_state <= 1)) {

#ifdef _DEBUG
    fprintf (stderr, "transfer_data: check for command\n");
#endif

    /* check to see if a new command has been registered */
    if (dada_pwc_command_check (pwcm->pwc))  {

#ifdef _DEBUG
      fprintf (stderr, "transfer_data: get command\n");
#endif

      pwcm->command = dada_pwc_command_get (pwcm->pwc);

      /* special case for set_utc_start */
      if (pwcm->command.code == dada_pwc_set_utc_start) {

        assert(pwcm->command.utc > 0);

        if (utc_start_set) { 
          multilog (pwcm->log, LOG_WARNING, "WARNING, UTC_START was already "
                    "set. Ignoring set_utc_start command\n");
        } else {

          strftime (utc_buffer, utc_size, DADA_TIMESTR, 
                    (struct tm*) gmtime(&(pwcm->command.utc)));

#ifdef _DEBUG
          fprintf (stderr,"transfer_data: set UTC_START in header to : %s\n",
                   utc_buffer);
#endif
          multilog (pwcm->log, LOG_INFO,"UTC_START = %s\n",utc_buffer);

          // write UTC_START to the header
          if (ascii_header_set (pwcm->header,"UTC_START","%s",utc_buffer) < 0) {
            multilog (pwcm->log,LOG_ERR,"failed ascii_header_set UTC_START\n");
            return DADA_ERROR_HARD;
          }

          /* We only marked the header block as filled if we are already 
           * recording this can only happen via a "start" command. if we
           * are clocking, then the record_start command will mark the 
           * header block as filled */

#ifdef _DEBUG
          fprintf (stderr,"transfer_data: header block filled\n");
#endif
          if (pwcm->pwc->state == dada_pwc_recording) {

            if (pwcm->header_valid_function) 
              pwcm->header_valid = pwcm->header_valid_function(pwcm);
            else
              pwcm->header_valid = 1;

            if (pwcm->header_valid) {
#ifdef _DEBUG
              multilog (pwcm->log, LOG_INFO, "transfer_data: marking header valid\n");
#endif

              if (ipcbuf_mark_filled (pwcm->header_block,pwcm->header_size) < 0) {
                multilog (pwcm->log, LOG_ERR, "Could not mark filled header\n");
                return DADA_ERROR_HARD;
              }
            }
          }
      
          /* Set to true */
          utc_start_set = 1;
        }

        /* Signal the command control interface */
        pthread_mutex_lock (&(pwcm->pwc->mutex));
        pthread_cond_signal (&(pwcm->pwc->cond));
        pthread_mutex_unlock (&(pwcm->pwc->mutex));

      } else {

        if (pwcm->command.code == dada_pwc_record_stop)
          command_string = "recording->clocking";
        else if (pwcm->command.code == dada_pwc_record_start)
          command_string = "clocking->recording";
        else if (pwcm->command.code == dada_pwc_stop)
          command_string = "stopping";
        else {
          multilog (pwcm->log, LOG_ERR, "transfer_data: internal error = "
                    "unexpected command code %d\n", pwcm->command.code);
          return DADA_ERROR_HARD;
        }

        if (pwcm->verbose)
          multilog (pwcm->log, LOG_INFO, "transfer_data: %s byte count = %"PRIu64" bytes\n",
                    command_string, pwcm->command.byte_count);

        if (pwcm->command.byte_count > total_bytes_written) {
          /* command is dated, happens in the future... */
          transit_byte = pwcm->command.byte_count;
          bytes_to_write = pwcm->command.byte_count - total_bytes_written;

          multilog (pwcm->log, LOG_INFO, "%s in %"PRIu64" bytes\n", 
                    command_string, bytes_to_write);

        } else {
          /* command is immediate */
          multilog (pwcm->log, LOG_INFO, "%s immediately\n", command_string);
          if (total_bytes_written == 0)
          {
            multilog (pwcm->log, LOG_WARNING, "Received 0 bytes, setting to 1\n");
            total_bytes_written = 1;
          }
          transit_byte = total_bytes_written;
          bytes_to_write = 0;

          if (pwcm->command.byte_count &&
              pwcm->command.byte_count < total_bytes_written)
            multilog (pwcm->log, LOG_NOTICE,
                      "requested transit byte=%"PRIu64" passed\n",
                      pwcm->command.byte_count);
        }
        
      }

      if (pwcm->verbose) {
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
          else
            ;

        }
      }
    }

    if (!transit_byte || bytes_to_write) {

#ifdef _DEBUG
      fprintf (stderr, "transfer_data: call buffer function\n");
#endif

      /* get the next data buffer */
      buffer = pwcm->buffer_function (pwcm, &buffer_size);

      /* If the buffer_function had an error */
      if (buffer_size < 0) 
        return buffer_size;

      if ((data_transfer_error_state) && (buffer_size != 0)) {
        data_transfer_error_state = 0;
        if (total_bytes_written) 
          multilog (pwcm->log, LOG_WARNING, "pwc buffer_function "
                    "recovered from error state\n");
      }


      if (buffer_size == 0) {

        // if the pwc supports multiple xfers within the 1 observation
        if (pwcm->xfer_pending_function) 
        {
          if (pwcm->verbose)       
            multilog (pwcm->log, LOG_INFO, "transfer_data: buffer_size=0, checking xfer status\n");
          first_byte_of_xfer = dada_pwc_main_process_xfer (pwcm, 0, 0);
          if (first_byte_of_xfer == -2) 
          {
            multilog (pwcm->log, LOG_ERR, "transfer_data: process_xfer failed\n");
            return DADA_ERROR_HARD;
          } 
          else if (first_byte_of_xfer == -1) 
          {
            if (pwcm->verbose)
              multilog (pwcm->log, LOG_INFO, "transfer_data: process_xfer indicates end of observation\n");
            return 0;
          }
          else if (first_byte_of_xfer == 0) 
          {
            if (pwcm->verbose)
              multilog (pwcm->log, LOG_INFO, "transfer_data: process_xfer indicates waiting for SOD on xfer\n");
          }
          else
          {
            if (pwcm->verbose)
              multilog (pwcm->log, LOG_INFO, "transfer_data: process_xfer indicates next byte is %"PRIi64"\n", first_byte_of_xfer);
            total_bytes_written = (uint64_t) first_byte_of_xfer;
          }

        } else {

          /* If we are experiencing an read error that we may be able to 
           * recover from, run the error_function, if it returns:
           * 0, we have recovered from error 
           * 1, we keep trying to recover
           * 2, we will not be able to recover from error, exit gracefully */
          if (pwcm->error_function) {
            /* If defined */
            data_transfer_error_state = pwcm->error_function(pwcm);
          } else {
            /* give up */        
            data_transfer_error_state = 2;
          }

          if (data_transfer_error_state == 2) {
            multilog (pwcm->log, LOG_ERR, "pwc buffer_function returned 0 bytes."
                                          " Stopping\n");

            /* Ensure that the ipcio_write function is called with 1 bytes
             * so that a reader may acknwowledge the SoD and subsequent EoD */
            bytes_written = ipcio_write (pwcm->data_block, buffer, 1);

            /* If we are clocking, then its not the end of the world, signal
             * a soft error */
            return DADA_ERROR_HARD;
          }

          if ((data_transfer_error_state == 1) && total_bytes_written) {
            multilog (pwcm->log, LOG_INFO, "PWC buffer function "
                      "returned 0 bytes, trying to continue\n");
          }
        }
      }
      
      if (!buffer) {
        multilog (pwcm->log, LOG_ERR, "buffer function error\n");
        return DADA_ERROR_HARD;   
        //TODO CHECK WHAT buffer_function actually returns...
      }

      /* Check if we need to mark the header as valid after this - not possible
       * if we are clocking */
      if ((!pwcm->header_valid) && (pwcm->pwc->state == dada_pwc_recording)) {
        pwcm->header_valid = pwcm->header_valid_function(pwcm);
      
        /* If the header is NOW valid, flag the header as filled */  
        if (pwcm->header_valid) {
#ifdef _DEBUG
            multilog (pwcm->log, LOG_INFO,"transfer_data: marking header filled\n");
#endif
          if (ipcbuf_mark_filled (pwcm->header_block,pwcm->header_size) < 0) {
            multilog (pwcm->log, LOG_ERR, "Could not mark filled header\n");
            return DADA_ERROR_HARD;
          }
        }
      }

      /* If the transit_byte is set, do not write more than the requested
         amount of data to the Data Block */
      buf_bytes = buffer_size;
      
      if (transit_byte && buf_bytes > bytes_to_write)
        buf_bytes = bytes_to_write;
     
      /* write the bytes to the Data Block */
      if (pwcm->data_block) {

#ifdef _DEBUG
        fprintf (stderr, "transfer_data: write to data block buffer=%p bytes=%"
                         PRIu64"\n", buffer, buf_bytes);
#endif

        /* If we run out of empty buffers and don't yet have a UTC_START, 
         * fail */
        if ((!utc_start_set) && (ipcio_space_left (pwcm->data_block) < buf_bytes)) {
          multilog (pwcm->log, LOG_ERR, "Data block full and UTC_START not "
                    "set.\n");
          return DADA_ERROR_FATAL;
        }
        
        bytes_written = ipcio_write (pwcm->data_block, buffer, buf_bytes);

#ifdef _DEBUG
        fprintf (stderr, "transfer_data: return from write\n");
#endif

        if (bytes_written < 0 || bytes_written < buf_bytes) {
          multilog (pwcm->log, LOG_ERR, "Cannot write %"PRIu64
                    " bytes to Data Block\n", buf_bytes);
          return DADA_ERROR_FATAL;
        }

      }

      total_bytes_written += buf_bytes;

      if (bytes_to_write)
        bytes_to_write -= buf_bytes;

      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "Written %"PRIu64" bytes\n",
                  total_bytes_written);

    }

    if (transit_byte && pwcm->verbose)
      multilog (pwcm->log, LOG_INFO, "transit=%"PRIu64" total=%"PRIu64"\n",
                transit_byte, total_bytes_written);

    // total_bytes_written can jump ahead of a transit byte (with XFERS)
    if (transit_byte && (transit_byte <= total_bytes_written)) {

#ifdef _DEBUG
      fprintf (stderr, "transfer_data: transit state\n");
#endif

      /* The transit_byte has been reached, it is now time to change state */
      /* reset the transit_byte */ 
      transit_byte = 0;

      if (pwcm->command.code ==  dada_pwc_record_stop) {

        multilog (pwcm->log, LOG_INFO, "record stop\n");

        /* stop valid data on the Data Block at the last byte written */
        if (pwcm->data_block && ipcio_stop (pwcm->data_block) < 0){
          multilog (pwcm->log, LOG_ERR, "Could not stop data block\n");
          return DADA_ERROR_FATAL;
        }
        
        dada_pwc_set_state (pwcm->pwc, dada_pwc_clocking, 0);
        
      }

      else if (pwcm->command.code ==  dada_pwc_record_start) {

        multilog (pwcm->log, LOG_INFO, "record start\n");

        if (dada_pwc_main_record_start (pwcm) < 0)
          return DADA_ERROR_HARD;
        
        dada_pwc_set_state (pwcm->pwc, dada_pwc_recording, 0);

      } 

      else if (pwcm->command.code == dada_pwc_stop) {

        if (pwcm->xfer_pending_function) 
        {
          // CASPSR CHANGE
          // If we haven't stopped already due to a packet timeout, we must
          // write the additional 1 byte transfer and set END_OF_OBS=-1
          dada_pwc_main_process_xfer(pwcm, 2, 0);
        }

#ifdef _DEBUG
        multilog (pwcm->log, LOG_INFO, "stopping... entering idle state\n");
#endif
        return 0;
      }
      
      else if (pwcm->command.code ==  dada_pwc_set_utc_start) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected set_utc_start\n");
          return DADA_ERROR_HARD;
        }

      } else if (pwcm->command.code == dada_pwc_clock) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected clock command\n");
          return DADA_ERROR_HARD;
        }

      } else if (pwcm->command.code == dada_pwc_start) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected start command\n");
          return DADA_ERROR_HARD;
        }
        
      } else  {
        /* enter the idle state */
        multilog (pwcm->log, LOG_ERR, "Error. unpected command: %d\n",
                  pwcm->command.code);
        return DADA_ERROR_HARD;

      }

      /* When changing between clocking and recording states, there
         may remain unwritten data in the buffer.  Write this data to
         the Data Block */

      /* offset the buffer pointer by the amount already written */
      buffer += buf_bytes;
      /* set the number of remaining samples */
      buf_bytes = buffer_size - buf_bytes;

      if (buf_bytes && pwcm->data_block) {

        // If we run out of empty buffers and don't yet have a UTC_START, fail
        if (ipcio_space_left(pwcm->data_block) < buf_bytes) 
        {
          if (!utc_start_set) {
            multilog (pwcm->log, LOG_ERR, "Data block full and UTC_START not set.\n");
            return DADA_ERROR_FATAL;
          }
          else
            multilog(pwcm->log, LOG_WARNING, "Data block full, waiting for space\n");
        }

        // write the bytes to the Data Block
        if (ipcio_write (pwcm->data_block, buffer, buf_bytes) < buf_bytes) {

          multilog (pwcm->log, LOG_ERR, "Cannot write %"PRIu64
                    " bytes to Data Block\n", buf_bytes);
          return DADA_ERROR_FATAL;

        }

        total_bytes_written += buf_bytes;

      }

    }

  }

  return 0;

}

/*! The clocking and recording states for zero copy of the DADA PWC main loop */
int dada_pwc_main_transfer_data_block (dada_pwc_main_t* pwcm)
{
  /* total number of bytes written to the Data Block */
  uint64_t total_bytes_written = 0;

  /* the byte at which the state will change */
  uint64_t transit_byte = pwcm->command.byte_count;

  /* the number of bytes to write to the Data Block */
  uint64_t bytes_to_write = pwcm->command.byte_count;

  /* the number of bytes to be copied from buffer */
  uint64_t buf_bytes = 0;

  /* the number of bytes written to the Data Block */
  int64_t bytes_written = 0;

  /* the number of bytes written this xfer */
  uint64_t bytes_written_this_xfer = 0;

  /* pointer to the data buffer */
  char* buffer = 0;

  /* error state due to a 0 byte buffer function return */
  uint64_t data_transfer_error_state = 0;

  /* block id of the current block */
  uint64_t block_id = 0;

  /* size of the buffer in the data block */
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t*) pwcm->data_block);

  char* command_string = 0;

  /* flags and buffers for setting of UTC_START */
  uint64_t utc_start_set = 0;
  int utc_size = 1024;
  char utc_buffer[utc_size];

  int64_t first_byte_of_xfer = 1;

  /* if the pwcm supports xfers, get the first byte of data for the first xfer */
  if (pwcm->new_xfer_function) {
    total_bytes_written = pwcm->new_xfer_function(pwcm);
    if (pwcm->verbose)
      multilog (pwcm->log, LOG_INFO, "first byte expected on %"PRIu64"\n", total_bytes_written);
  }

  /* Check if the UTC_START is valid */
  if (ascii_header_get (pwcm->header, "UTC_START", "%s", utc_buffer) < 0) {
    multilog (pwcm->log, LOG_ERR, "Could not read UTC_START from header\n");
    return DADA_ERROR_HARD;
  }

  if (strcmp(utc_buffer,"UNKNOWN") != 0) 
    utc_start_set = 1;

#ifdef _DEBUG
  fprintf (stderr, "transfer_data_block: enter main loop\n");
#endif

  while ((!dada_pwc_quit (pwcm->pwc)) && (data_transfer_error_state <= 1)) {

#ifdef _DEBUG
    fprintf (stderr, "transfer_data_block: check for command\n");
#endif

    /* check to see if a new command has been registered */
    if (dada_pwc_command_check (pwcm->pwc))  {

#ifdef _DEBUG
      fprintf (stderr, "transfer_data_block: get command\n");
#endif

      pwcm->command = dada_pwc_command_get (pwcm->pwc);

      // multilog (pwcm->log, LOG_INFO, "pwcm->command.utc = %d\n",pwcm->command.utc);
      // multilog (pwcm->log, LOG_INFO, "pwcm->pwc->utc_start = %d\n",pwcm->pwc->utc_start);

      /* special case for set_utc_start */
      if (pwcm->command.code == dada_pwc_set_utc_start) {

        assert(pwcm->command.utc > 0);

        if (utc_start_set) { 
          multilog (pwcm->log, LOG_WARNING, "WARNING, UTC_START was already "
                    "set. Ignoring set_utc_start command\n");
        } else {

          strftime (utc_buffer, utc_size, DADA_TIMESTR, 
                    (struct tm*) gmtime(&(pwcm->command.utc)));

#ifdef _DEBUG
          fprintf (stderr,"transfer_data_block: set UTC_START in "
                          "header to : %s\n",utc_buffer);
#endif
          multilog (pwcm->log, LOG_INFO,"UTC_START = %s\n",utc_buffer);

          // write UTC_START to the header
          if (ascii_header_set (pwcm->header,"UTC_START","%s",utc_buffer) < 0) {
            multilog (pwcm->log,LOG_ERR,"failed ascii_header_set UTC_START\n");
            return DADA_ERROR_HARD;
          }

          /* We only marked the header block as filled if we are already 
           * recording this can only happen via a "start" command. if we
           * are clocking, then the record_start command will mark the 
           * header block as filled */

#ifdef _DEBUG
          fprintf (stderr,"transfer_data_block: header block filled\n");
#endif
          if (pwcm->pwc->state == dada_pwc_recording) {

            if (pwcm->header_valid_function) 
              pwcm->header_valid = pwcm->header_valid_function(pwcm);
            else
              pwcm->header_valid = 1;

            if (pwcm->header_valid) {
#ifdef _DEBUG
              multilog (pwcm->log, LOG_INFO, "transfer_data_block: marking header valid\n");
#endif

              if (ipcbuf_mark_filled (pwcm->header_block,pwcm->header_size) < 0) {
                multilog (pwcm->log, LOG_ERR, "Could not mark filled header\n");
                return DADA_ERROR_HARD;
              }
            }
          }
      
          /* Set to true */
          utc_start_set = 1;
        }

        /* Signal the command control interface */
        pthread_mutex_lock (&(pwcm->pwc->mutex));
        pthread_cond_signal (&(pwcm->pwc->cond));
        pthread_mutex_unlock (&(pwcm->pwc->mutex));

      } else {

        if (pwcm->command.code == dada_pwc_record_stop)
          command_string = "recording->clocking";
        else if (pwcm->command.code == dada_pwc_record_start)
          command_string = "clocking->recording";
        else if (pwcm->command.code == dada_pwc_stop)
          command_string = "stopping";
        else {
          multilog (pwcm->log, LOG_ERR,
                    "dada_pwc_main_transfer data internal error = "
                    "unexpected command code %d\n", pwcm->command.code);
          return DADA_ERROR_HARD;
        }

        if (pwcm->verbose)
          multilog (pwcm->log, LOG_INFO, "%s byte count = %"PRIu64" bytes\n",
                    command_string, pwcm->command.byte_count);

        if (pwcm->command.byte_count > total_bytes_written) {
          /* command is dated, happens in the future... */
          transit_byte = pwcm->command.byte_count;
          bytes_to_write = pwcm->command.byte_count - total_bytes_written;

          multilog (pwcm->log, LOG_INFO, "%s in %"PRIu64" bytes\n", 
                    command_string, bytes_to_write);

        } else {
          /* command is immediate */
          multilog (pwcm->log, LOG_INFO, "%s immediately\n", command_string);
          transit_byte = total_bytes_written;
          bytes_to_write = 0;

          if (pwcm->command.byte_count &&
              pwcm->command.byte_count < total_bytes_written)
            multilog (pwcm->log, LOG_NOTICE,
                      "requested transit byte=%"PRIu64" passed\n",
                      pwcm->command.byte_count);
        }
        
      }

      if (pwcm->verbose) {
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
          else
            ;

        }
      }
    }

    // if we dont have a specific byte to stop on or we have
    // some bytes to write
    if (!transit_byte || bytes_to_write) {

      // get the pointer to the next empty block and its ID
#ifdef _DEBUG
      fprintf (stderr, "transfer_data_block: ipcio_open_block_write()\n");
#endif
      buffer = ipcio_open_block_write (pwcm->data_block, &block_id);
      if (!buffer) 
      {
        multilog (pwcm->log, LOG_ERR, "ipcio_open_block_write error %s\n", strerror(errno));
        return DADA_ERROR_FATAL;
      }

      // call the block function to fill the buffer with data
#ifdef _DEBUG
      fprintf (stderr, "transfer_data_block: block_function()\n");
#endif
      bytes_written = pwcm->block_function (pwcm, buffer, block_size, block_id);

      // check the result of the block function
      if (bytes_written < 0)
      {
        multilog (pwcm->log, LOG_ERR, "transfer_data_block: block_function failed\n");
        ipcio_close_block_write (pwcm->data_block, 0);
        return bytes_written; // TODO check this behaviuor
      }

      /* If the transit_byte is set, do not write more than the requested 
         amount of data to the Data Block */
      buf_bytes = bytes_written;

      if (transit_byte && bytes_written > bytes_to_write)
        buf_bytes = bytes_to_write;

      if ((bytes_written == 0) && (pwcm->verbose))
      {
        multilog (pwcm->log, LOG_INFO, "transfer_data_block: block_function returned 0 bytes, buf_bytes=%"PRIu64"\n", buf_bytes);
        multilog (pwcm->log, LOG_INFO, "transfer_data_block: bytes_written this xfer=%"PRIu64"\n", 
                  bytes_written_this_xfer);
      }

      // close the block if 0 or more bytes were written
#ifdef _DEBUG
      fprintf (stderr, "transfer_data_block: ipcio_close_block_write(%"PRIu64")\n", (uint64_t) buf_bytes);
#endif

      if (ipcio_close_block_write (pwcm->data_block, (uint64_t) buf_bytes) < 0) {
        multilog (pwcm->log, LOG_ERR, "transfer_data_block: ipcio_close_block_write error %s\n", strerror(errno));
        return DADA_ERROR_FATAL;
      }

      // if we wrote 0 bytes, its the end of an xfer or end of obs
      if (bytes_written == 0) {

        // if the pwc supports multiple xfers within the 1 observation
        if (pwcm->xfer_pending_function) 
        {
          if (pwcm->verbose)
            multilog (pwcm->log, LOG_INFO, "transfer_data_block: dada_pwc_main_process_xfer(0, %"PRIu64")\n",
                      bytes_written_this_xfer);
          first_byte_of_xfer = dada_pwc_main_process_xfer(pwcm, 0, bytes_written_this_xfer);

          // check the result of processing the XFER
          if (first_byte_of_xfer == -2) 
          {
            multilog (pwcm->log, LOG_ERR, "dada_pwc_main_process_xfer failed\n");
            return DADA_ERROR_HARD;
          } 
          else if (first_byte_of_xfer == -1) 
          {
            multilog (pwcm->log, LOG_ERR, "transfer_data_block: OBS ended, this shouldn't happen\n");
            //return 0;
          }
          else if (first_byte_of_xfer == 0) 
          {
            multilog (pwcm->log, LOG_INFO, "transfer_data_block: waiting for SOD on xfer\n");
          }
          else
          {
            if (pwcm->verbose)
              multilog (pwcm->log, LOG_INFO, "transfer_data_block: new XFER, next byte=%"PRIi64"\n", first_byte_of_xfer);
            total_bytes_written = (uint64_t) first_byte_of_xfer;
            bytes_written_this_xfer = 0;
          }
        }
      }
      
      /* Check if we need to mark the header as valid after this - not possible
       * if we are clocking */
      if ((!pwcm->header_valid) && (pwcm->pwc->state == dada_pwc_recording)) {
        pwcm->header_valid = pwcm->header_valid_function(pwcm);
      
        /* If the header is NOW valid, flag the header as filled */  
        if (pwcm->header_valid) {
          multilog (pwcm->log, LOG_INFO,"transfer_data_block: marking header filled\n");
          if (ipcbuf_mark_filled (pwcm->header_block,pwcm->header_size) < 0) {
            multilog (pwcm->log, LOG_ERR, "Could not mark filled header\n");
            return DADA_ERROR_HARD;
          }
        }
      }

      total_bytes_written += buf_bytes;
      bytes_written_this_xfer += buf_bytes;

      if (bytes_to_write)
        bytes_to_write -= buf_bytes;

      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "Written %"PRIu64" bytes\n",
                  total_bytes_written);

    } else {
      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "transfer_data_block: tranist byte was set, block_function ignored\n");
    }

    if (pwcm->verbose)
      multilog (pwcm->log, LOG_INFO, "transit=%"PRIu64" total=%"PRIu64"\n",
                transit_byte, total_bytes_written);

    // CASPSR change - total_bytes_written can now jump ahead of a transit byte
    if (transit_byte && (transit_byte <= total_bytes_written)) {

#ifdef _DEBUG
      fprintf (stderr, "transfer_data_block: transit state\n");
#endif

      /* The transit_byte has been reached, it is now time to change state */
      /* reset the transit_byte */ 
      transit_byte = 0;

      if (pwcm->command.code ==  dada_pwc_record_stop) {

        multilog (pwcm->log, LOG_INFO, "record stop\n");

        /* stop valid data on the Data Block at the last byte written */
        if (pwcm->data_block && ipcio_stop (pwcm->data_block) < 0){
          multilog (pwcm->log, LOG_ERR, "Could not stop data block\n");
          return DADA_ERROR_FATAL;
        }
        
        dada_pwc_set_state (pwcm->pwc, dada_pwc_clocking, 0);
        
      }

      else if (pwcm->command.code ==  dada_pwc_record_start) {

        multilog (pwcm->log, LOG_INFO, "record start\n");

        if (dada_pwc_main_record_start (pwcm) < 0)
          return DADA_ERROR_HARD;
        
        dada_pwc_set_state (pwcm->pwc, dada_pwc_recording, 0);

      } 

      else if (pwcm->command.code == dada_pwc_stop) {

        if (pwcm->xfer_pending_function)
        {
          // CASPSR CHANGE
          // If we haven't stopped already due to a packet timeout, we must
          // write the additional 1 byte transfer and set END_OF_OBS=-1
          if (pwcm->verbose)
            multilog (pwcm->log, LOG_INFO, "transfer_data_block: [STOP] "
                      "dada_pwc_main_process_xfer(1, %"PRIu64")\n", 
                      bytes_written_this_xfer);
          dada_pwc_main_process_xfer (pwcm, 1, bytes_written_this_xfer);
        }

#ifdef _DEBUG
        multilog (pwcm->log, LOG_INFO, "stopping... entering idle state\n");
#endif
        return 0;
      }
      
      else if (pwcm->command.code ==  dada_pwc_set_utc_start) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected set_utc_start\n");
          return DADA_ERROR_HARD;
        }

      } else if (pwcm->command.code == dada_pwc_clock) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected clock command\n");
          return DADA_ERROR_HARD;
        }

      } else if (pwcm->command.code == dada_pwc_start) {

        if (total_bytes_written) {
          multilog (pwcm->log, LOG_ERR, "Error. unexpected start command\n");
          return DADA_ERROR_HARD;
        }
        
      } else  {
        /* enter the idle state */
        multilog (pwcm->log, LOG_ERR, "Error. unpected command: %d\n",
                  pwcm->command.code);
        return DADA_ERROR_HARD;

      }
    }
  }
  return 0;
}


/*! The transit to idle/prepared state */
int dada_pwc_main_stop_transfer (dada_pwc_main_t* pwcm)
{

  /* Reset the header so that it is not valid anymore */
  pwcm->header_valid = 0;
        
  if (pwcm->stop_function (pwcm) < 0)  {
    multilog (pwcm->log, LOG_ERR, "dada_pwc_main_stop_transfer"
              " stop function returned error code\n");
    return DADA_ERROR_HARD;
  }

  /* close the Data Block */
  if (pwcm->data_block && ipcio_close (pwcm->data_block) < 0)  {
    multilog (pwcm->log, LOG_ERR, "Could not close Data Block\n");
    return DADA_ERROR_FATAL;
  }

  if (pwcm->pwc->state != dada_pwc_soft_error &&
      pwcm->pwc->state != dada_pwc_hard_error &&
      pwcm->pwc->state != dada_pwc_fatal_error)  
    dada_pwc_set_state (pwcm->pwc, dada_pwc_idle, 0);

  return 0;
}

void dada_pwc_main_process_error (dada_pwc_main_t* pwcm, int rval) 
{

  int new_state = pwcm->pwc->state;

  switch (rval) 
  {
    case DADA_ERROR_SOFT:
      if ( (pwcm->pwc->state != dada_pwc_hard_error) && 
           (pwcm->pwc->state != dada_pwc_fatal_error) )
        new_state = dada_pwc_soft_error;
      break;

    case DADA_ERROR_HARD:
      if (pwcm->pwc->state != dada_pwc_fatal_error)
        new_state = dada_pwc_hard_error;
      break;

    case DADA_ERROR_FATAL:
      new_state = dada_pwc_fatal_error;
      break;

    default:
      multilog (pwcm->log, LOG_ERR, "Unknown error state: %d\n",rval);
      new_state = dada_pwc_fatal_error;
  }

  multilog(pwcm->log, LOG_WARNING, "PWC entering error state: %s\n",
           dada_pwc_state_to_string(new_state));

  if (dada_pwc_set_state (pwcm->pwc, new_state, 0) < 0) 
    multilog(pwcm->log, LOG_ERR, "Failed to change state from %s to %s\n",
             dada_pwc_state_to_string(pwcm->pwc->state),
             dada_pwc_state_to_string(new_state));


}

/*
 * dada_pwc_main_process_xfer
 *
 * check the current xfer status of the pwc. If required, move to a new transfer.
 * Return values:
 *   -2 error
 *   -1 at end of observation
 *    0 at beginning of observation and waiting for Start of Data
 *   >0 first byte of the next xfer
 *
 */
int64_t dada_pwc_main_process_xfer (dada_pwc_main_t* pwcm, int finalize, uint64_t bytes_this_xfer)
{

  // temporary header required for next xfer
  char * tmp_header = NULL;

  // a zerod memory buffer for 1 byte transfers
  char zerod_char = 'c';
  
  // status of the pending transfer;
  int xfer_pending_status = 0;

  // first byte of the next xfer
  int64_t first_byte = 0;

  // flag for when the final 1 byte xfer has been written
  int written_closing_block = 0;

  tmp_header = (char *) malloc (sizeof(char) * pwcm->header_size);
  if (!tmp_header) 
  {
    multilog (pwcm->log, LOG_ERR, "process_xfer: malloc failed for temp header\n");
    return -2;
  }

  // set the zerod char to 0
  memset(&zerod_char,0,sizeof(zerod_char));

  // copy the current header
  memcpy(tmp_header, pwcm->header, pwcm->header_size);

  // get the pwc's current xfer pending status
  xfer_pending_status = pwcm->xfer_pending_function(pwcm);

  if (pwcm->verbose)
    multilog(pwcm->log, LOG_INFO, "process_xfer: xfer_pending_status=%d\n", xfer_pending_status);

  // if we are finalizing the observation (i.e. we have passed the stopping byte)
  if (finalize) 
  {
    if (pwcm->verbose)
      multilog(pwcm->log, LOG_INFO, "process_xfer: finalizing bytes_this_xfer=%"PRIu64"\n", bytes_this_xfer);

    if (bytes_this_xfer)
    {
      // if we have received data and are at the end of an XFER or OBS
      if (!pwcm->data_block) {
        multilog (pwcm->log, LOG_ERR, "process_xfer: data block not connected\n");
        return -2;
      }

      void * buffer = 0;

      // allow the block function to cleanup
      pwcm->block_function (pwcm, buffer, 0, 0);

      // close the current data/header block
      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "process_xfer: closing header/data block\n");

      if (ipcio_close (pwcm->data_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not unlock Data Block write\n");
        return -2;
      }
      if (ipcbuf_unlock_write (pwcm->header_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not unlock Header Block write\n");
        return -2;
      }

      // re-open the data/header block
      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "process_xfer: opening header/data block\n");

      if (ipcbuf_lock_write (pwcm->header_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not lock Header Block for writing\n");
        return -2;
      }
      if (ipcio_open (pwcm->data_block, 'W') < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not lock Data Block for writing\n");
        return -2;
      }

      // get the next header block 
      pwcm->header = ipcbuf_get_next_write (pwcm->header_block);

      // copy the header_buf to the header block
      memcpy(pwcm->header, tmp_header, pwcm->header_size);

      if (pwcm->verbose)
        multilog(pwcm->log, LOG_INFO, "process_xfer: pwcm->new_xfer_function()\n");
      first_byte = (int64_t) pwcm->new_xfer_function(pwcm);
      if (pwcm->verbose)
        multilog(pwcm->log, LOG_INFO, "process_xfer: next XFER on byte=%"PRIu64"\n", first_byte);


      /* marked the header as filled */
      if (pwcm->verbose)
        multilog(pwcm->log, LOG_INFO, "process_xfer: marking header as filled\n");
      if (ipcbuf_mark_filled (pwcm->header_block, pwcm->header_size) < 0) {
        multilog (pwcm->log, LOG_ERR, "process_xfer: Could not mark header block filled\n");
        return -2;
      }

    }

    if (pwcm->verbose)
      multilog (pwcm->log, LOG_INFO, "process_xfer: writing 1 byte to empty DB during finalization\n");
    if (ipcio_write(pwcm->data_block, &zerod_char, 1) != 1)
    {
      multilog(pwcm->log, LOG_ERR, "main: failed to write 1 byte to empty datablock at end of OBS\n");
      return -2;
    }

    free(tmp_header);
    return 0;
  }

  // otherwise we are processing a regular xfer
  if (xfer_pending_status != 1)
    multilog(pwcm->log, LOG_ERR, "process_xfer: not finalizing and xfer_pending_status != 1\n");

  if (xfer_pending_status > 0) 
  {

    // if we have 0 bytes written, then the end of obs will come soon, so do nothing!
    if (bytes_this_xfer == 0)
    {
      multilog(pwcm->log, LOG_ERR, "process_xfer: not finalizing and bytes_written == 0\n");
    } 
    else
    {

      // if we have received data and are at the end of an XFER or OBS
      if (!pwcm->data_block) {
        multilog (pwcm->log, LOG_ERR, "process_xfer: data block not connected\n");
        return -2;
      } 

      // close the current data/header block
      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "process_xfer: closing header/data block\n");

      if (ipcio_close (pwcm->data_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not unlock Data Block write\n");
        return -2;
      }
      if (ipcbuf_unlock_write (pwcm->header_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not unlock Header Block write\n");
        return -2;
      }

      // re-open the data/header block
      if (pwcm->verbose)
        multilog (pwcm->log, LOG_INFO, "process_xfer: opening header/data block\n");

      if (ipcbuf_lock_write (pwcm->header_block) < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not lock Header Block for writing\n");
        return -2;
      }
      if (ipcio_open (pwcm->data_block, 'W') < 0) {
        multilog (pwcm->log, LOG_ERR, "Could not lock Data Block for writing\n");
        return -2;
      }

      // get the next header block 
      pwcm->header = ipcbuf_get_next_write (pwcm->header_block);

      // copy the header_buf to the header block
      memcpy(pwcm->header, tmp_header, pwcm->header_size);

      // call pwc's new xfer function. This will set the OBS_XFER count, the OBS_OFFSET and return 
      // the first byte expected of the next XFER

      if (pwcm->verbose)
        multilog(pwcm->log, LOG_INFO, "process_xfer: pwcm->new_xfer_function()\n");
      first_byte = (int64_t) pwcm->new_xfer_function(pwcm);
      if (pwcm->verbose)
        multilog(pwcm->log, LOG_INFO, "process_xfer: next XFER on byte=%"PRIu64"\n", first_byte);

      /* marked the header as filled */
      if (ipcbuf_mark_filled (pwcm->header_block, pwcm->header_size) < 0) {
        multilog (pwcm->log, LOG_ERR, "process_xfer: Could not mark header block filled\n");
        return -2;
      }

    }

    free (tmp_header);

  }
  return first_byte;
}
