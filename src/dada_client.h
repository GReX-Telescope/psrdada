#ifndef __DADA_PRC_MAIN_H
#define __DADA_PRC_MAIN_H

/* ************************************************************************

   dada_prc_main_t - a struct and associated routines for creation and
   execution of a dada primary read client main loop

   ************************************************************************ */

#include "multilog.h"
#include "ipcio.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct dada_prc_main {

    /*! The status and error logging interface */
    multilog_t* log;

    /*! The Data Block interface */
    ipcio_t* data_block;

    /*! The Header Block interface */
    ipcbuf_t* header_block;

    /*! The optimal size in bytes for each data transfer to the target */
    uint64_t optimal_bytes;

    /*! Pointer to the function that opens the data transfer target */
    int (*open_function) (struct dada_prc_main*);

    /*! Pointer to the function that writes data to the transfer target */
    int64_t (*write_function) (struct dada_prc_main*, 
			       void* data, uint64_t* size);

    /*! Pointer to the function that closes the data transfer target */
    int (*close_function) (struct dada_prc_main*);

    /*! The file descriptor of the data transfer target */
    int fd;

    /*! Additional context information */
    void* context;

  } dada_prc_main_t;

  /*! Create a new DADA primary write client main loop */
  dada_prc_main_t* dada_prc_main_create ();

  /*! Destroy a DADA primary write client main loop */
  void dada_prc_main_destroy (dada_prc_main_t* pwcm);

  /*! Run the DADA primary write client main loop */
  int dada_prc_main (dada_prc_main_t* pwcm);

#ifdef __cplusplus
	   }
#endif

#endif
