#ifndef __IPCUTIL_H
#define __IPCUTIL_H

/* ************************************************************************

   ipcbuf_t - a struct and associated routines for creation and management
   of a ring buffer in shared memory

   ************************************************************************ */

#include <sys/types.h> 
#include "environ.h"

#define IPCUTIL_PERM 0666 /* default: read/write permissions for all */

#ifdef __cplusplus
extern "C" {
#endif

  /* allocate size bytes in shared memory with the specified flags and key.
     returns the pointer to the base address and the shmid, id */
  void* ipc_alloc (key_t key, int size, int flag, int* id);

  /* operate on the specified semaphore */
  int ipc_semop (int semid, short num, short incr, short flag);

#ifdef __cplusplus
	   }
#endif

#endif
