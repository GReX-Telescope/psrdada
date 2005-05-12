#ifndef __DADA_PWC_MAIN_H
#define __DADA_PWC_MAIN_H

/* ************************************************************************

   dada_pwc_main_t - a struct and associated routines for creation and
   execution of a dada primary write client main loop

   ************************************************************************ */

#include "dada_pwc.h"
#include "multilog.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct dada_pwc_main {

    /*! The primary write client control connection */
    dada_pwc_t* pwc;

    /*! The current command from the PWC control connection */
    dada_pwc_command_t command;

    /*! The status and error logging interface */
    multilog_t* log;

    /*! Pointer to the function that starts data transfer */
    time_t (*start_function) (struct dada_pwc_main*, time_t, void* context);

    /*! Additional context information to be passed to start_function */
    void* context;

  } dada_pwc_main_t;

  /*! Create a new DADA primary write client main loop */
  dada_pwc_main_t* dada_pwc_main_create ();

  /*! Destroy a DADA primary write client main loop */
  void dada_pwc_main_destroy (dada_pwc_main_t* primary);

  /*! Run the DADA primary write client main loop */
  int dada_pwc_main (dada_pwc_main_t* main);

#ifdef __cplusplus
	   }
#endif

#endif
