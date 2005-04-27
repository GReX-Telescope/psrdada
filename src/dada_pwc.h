#ifndef __DADA_PWC_H
#define __DADA_PWC_H

/* ************************************************************************

   dada_pwc_t - a struct and associated routines for creation and
   management of a dada primary write client control connection

   ************************************************************************ */

#include <stdio.h>
#include <pthread.h>

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

    /*! The I/O stream to the other end of the connection */
    FILE* to;

    /*! The I/O stream from the other end of the connection */
    FILE* from;

    /*! The ASCII header sent/received via the connection */
    char* header;

    /*! The size of the ASCII header */
    unsigned header_size;

    /* for multi-threaded use of the connection */
    pthread_mutex_t mutex;

  } dada_pwc_t;

  /*! Create a new DADA primary write client connection */
  dada_pwc_t* dada_pwc_create ();

  /*! Destroy a DADA primary write client connection */
  int dada_pwc_destroy (dada_pwc_t* pwc);

  /*! Check to see if a command has arrived */
  int dada_pwc_command_check (dada_pwc_t* pwc);

  /*! Get the next command from the connection; wait until command received */
  int dada_pwc_command_get (dada_pwc_t* pwc);

  /*! Reply to the last command received */
  int dada_pwc_command_reply (dada_pwc_t* pwc);

#ifdef __cplusplus
	   }
#endif

#endif
