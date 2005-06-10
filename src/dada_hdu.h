#ifndef __DADA_HDU_H
#define __DADA_HDU_H

/* ************************************************************************

   dada_hdu_t - a struct and associated routines for creation and
   management of a DADA Header plus Data Unit

   ************************************************************************ */

#include "multilog.h"
#include "ipcio.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct dada_hdu {

    /*! The status and error logging interface */
    multilog_t* log;

    /*! The Data Block interface */
    ipcio_t* data_block;

    /*! The Header Block interface */
    ipcbuf_t* header_block;

  } dada_hdu_t;

  /*! Create a new DADA Header plus Data Unit */
  dada_hdu_t* dada_hdu_create (multilog_t* log);

  /*! Destroy a DADA Header plus Data Unit */
  void dada_hdu_destroy (dada_hdu_t* hdu);

  /*! Connect the DADA Header plus Data Unit */
  int dada_hdu_connect (dada_hdu_t* hdu);

  /*! Connect the DADA Header plus Data Unit */
  int dada_hdu_disconnect (dada_hdu_t* hdu);

  /*! Lock DADA Header plus Data Unit designated reader */
  int dada_hdu_lock_read (dada_hdu_t* hdu);

  /*! Unlock DADA Header plus Data Unit designated reader */
  int dada_hdu_unlock_read (dada_hdu_t* hdu);

  /*! Lock DADA Header plus Data Unit designated writer */
  int dada_hdu_lock_write (dada_hdu_t* hdu);

  /*! Unlock DADA Header plus Data Unit designated writer */
  int dada_hdu_unlock_write (dada_hdu_t* hdu);

  /*! Lock DADA Header plus Data Unit designated writer */
  int dada_hdu_lock_write_spec (dada_hdu_t* hdu, char writemode);

#ifdef __cplusplus
	   }
#endif

#endif
