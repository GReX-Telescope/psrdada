#ifndef __DADA_PWC_H
#define __DADA_PWC_H

/* ************************************************************************

   dada_pwc_t - a struct and associated routines for creation and
   management of a dada primary write client control connection

   ************************************************************************ */

#include "command_parse_server.h"
#include "multilog.h"

#include <inttypes.h> 
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*! The states in which the primary write client may be found */
  enum {

    /*! idle: no data is being recorded */
    dada_pwc_idle,

    /*! prepared: no data is being recorded, header received */
    dada_pwc_prepared,

    /*! clocking: data is being clocked in over-write mode */
    dada_pwc_clocking,

    /*! recording: data is being recorded */
    dada_pwc_recording

  };

  /*! The states in which the primary write client may be found */
  enum {

    /*! none: no command available */
    dada_pwc_no_command,

    /*! header: configuration parameters are available */
    dada_pwc_header,

    /*! clock: enter the clocking data state */
    dada_pwc_clock,

    /*! record start: enter the recording state (from clocking state) */
    dada_pwc_record_start,

    /*! record stop: enter the clocking state (from recording state) */
    dada_pwc_record_stop,

    /*! start: enter the recording state */
    dada_pwc_start,

    /*! stop: enter the idle state */
    dada_pwc_stop

  };

  typedef struct {

    /*! The command code */
    int code;

    /*! The UTC associated with the command */
    time_t utc;

    /*! The duration associated with the command */
    uint64_t duration;

    /*! The ASCII header associated with the command */
    char* header;

  } dada_pwc_command_t;

#define DADA_PWC_COMMAND_INIT {0,0,0,0}

  typedef struct {

    /*! The name of the host on which primary write client is running */
    char* host;

    /*! The port on which primary write client control is listening */
    int port;

    /*! The primary write client identifier */
    int id;

    /*! The state of the primary write client */
    int state;

    /*! The last command received */
    dada_pwc_command_t command;

    /*! The ASCII header sent/received via the connection */
    char* header;

    /*! The size of the ASCII header */
    unsigned header_size;

    /*! The command parse server */ 
    command_parse_server_t* server;

    /*! The command parser */
    command_parse_t* parser;

    /* for multi-threaded use of the struct */
    pthread_mutex_t mutex;

    /* for multi-threaded polling */
    pthread_cond_t cond;

  } dada_pwc_t;

  /*! Create a new DADA primary write client connection */
  dada_pwc_t* dada_pwc_create ();

  /*! Destroy a DADA primary write client connection */
  int dada_pwc_destroy (dada_pwc_t* primary);

  /*! Start the command parsing server */
  int dada_pwc_serve (dada_pwc_t* primary);

  /*! Primary write client should exit when this is true */
  int dada_pwc_quit (dada_pwc_t* primary);

  /*! Check to see if a command has arrived */
  int dada_pwc_command_check (dada_pwc_t* primary);

  /*! Get the next command from the connection; wait until command received */
  dada_pwc_command_t dada_pwc_command_get (dada_pwc_t* primary);

  /*! Acknowledge the last command received */
  int dada_pwc_command_ack (dada_pwc_t* primary, int state);

#ifdef __cplusplus
	   }
#endif

#endif
