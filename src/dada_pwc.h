#ifndef __DADA_PRIMARY_H
#define __DADA_PRIMARY_H

/* ************************************************************************

   dada_primary_t - a struct and associated routines for creation and
   management of a dada primary write client control connection

   ************************************************************************ */

#include "command_parse_server.h"
#include "multilog.h"
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*! The states in which the primary write client may be found */
  enum {

    /*! idle: no data is being recorded */
    dada_primary_idle,

    /*! prepared: no data is being recorded, header received */
    dada_primary_prepared,

    /*! recording invalid: data is being recorded in over-write mode */
    dada_primary_recording_invalid,

    /*! recording valid: data is being recorded in lock-step mode */
    dada_primary_recording_valid

  };

  /*! The states in which the primary write client may be found */
  enum {

    /*! none: no command available */
    dada_primary_no_command = 0x00,

    /*! header: configuration parameters are available */
    dada_primary_header = 0x01,

    /*! invalid start: enter the recording invalid state */
    dada_primary_invalid_start = 0x02,

    /*! valid start: enter the recording valid state at the specified time */
    dada_primary_valid_start = 0x03,

    /*! start: enter the recording valid state */
    dada_primary_start = 0x04,

    /*! stop: stop recording */
    dada_primary_stop = 0x05,

    /*! time specified: a UTC is associated with the command */
    dada_primary_time_specified = 0x08

  };

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
    int command;

    /*! The ASCII header sent/received via the connection */
    char* header;

    /*! The UTC associated with the command */
    time_t utc;

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

  } dada_primary_t;

  /*! Create a new DADA primary write client connection */
  dada_primary_t* dada_primary_create ();

  /*! Destroy a DADA primary write client connection */
  int dada_primary_destroy (dada_primary_t* primary);

  /*! Check to see if a command has arrived */
  int dada_primary_command_check (dada_primary_t* primary);

  /*! Get the next command from the connection; wait until command received */
  int dada_primary_command_get (dada_primary_t* primary);

  /*! Reply to the last command received */
  int dada_primary_command_reply (dada_primary_t* primary);

#ifdef __cplusplus
	   }
#endif

#endif
