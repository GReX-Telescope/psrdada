#ifndef __DADA_PRIMARY_H
#define __DADA_PRIMARY_H

/* ************************************************************************

   dada_primary_t - a struct and associated routines for creation and
   management of a dada primary write client control connection

   ************************************************************************ */

#include "command_parse_server.h"
#include "multilog.h"
//#include <stdio.h>
//#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    /*! The name of the host on which primary write client is running */
    char* host;

    /*! The port on which primary write client control is listening */
    int port;

    /*! The primary write client identifier */
    int id;

    /*! The ASCII header sent/received via the connection */
    char* header;

    /*! The size of the ASCII header */
    unsigned header_size;

    /* for multi-threaded use of the connection */
    pthread_mutex_t mutex;

    /*! The command parse server */ 
    command_parse_server_t* server;

    /*! The command parser */
    command_parse_t* parser;

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
