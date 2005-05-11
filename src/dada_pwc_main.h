#ifndef __DADA_PWC_H
#define __DADA_PWC_H

/* ************************************************************************

   dada_pwc_main_t - a struct and associated routines for creation and
   execution of a dada primary write client main loop

   ************************************************************************ */

#include "dada_pwc.h"
#include "multilog.h"


  typedef struct {

    /* The primary write client control connection */
    dada_pwc_t* pwc;

    /* The current command from the control connection */
    dada_pwc_command_t command;

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
