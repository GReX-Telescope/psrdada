#ifndef __IPCBUF_H
#define __IPCBUF_H

/* ************************************************************************

   ipcbuf_t - a struct and associated routines for creation and management
   of a ring buffer in shared memory

   ************************************************************************ */

#include <sys/types.h> 
#include "environ.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    key_t semkey;    /* semaphore key */
    key_t shmkey;    /* shared memory key */

    uint64 nbufs;    /* the number of buffers in the ring */
    uint64 bufsz;    /* the size of the buffers in the ring */

    uint64 readbuf;  /* count of next buffer to read */
    uint64 writebuf; /* count of next buffer to write */

    uint64 s_buf;    /* the first valid buffer when sod is raised */
    uint64 s_byte;   /* the first valid byte when sod is raised */

    int eod;         /* end of data flag */
    uint64 e_buf;    /* the last valid buffer when sod is raised */
    uint64 e_byte;   /* the last valid byte when sod is raised */

  } ipcsync_t;

  typedef struct {

    int state;       /* the state of the process: writer, reader, etc. */

    int semid;       /* semaphore id */
    int shmid;       /* ring buffer shared memory id */
    int syncid;      /* sync struct shared memory id */

    ipcsync_t* sync; /* pointer to sync structure in shared memory */
    char* base;      /* base address of ring buffer in shared memory */

    uint64 viewbuf;  /* count of next buffer to look at (non-reader) */
    uint64 waitbuf;  /* count of buffers on which to wait */

  } ipcbuf_t;

#define IPCBUF_INIT {0, -1,-1,-1, 0,0, 0,0}

  /* ////////////////////////////////////////////////////////////////////
     
     FUNCTIONS USED TO CREATE/CONNECT/DESTROY SHARED MEMORY

     //////////////////////////////////////////////////////////////////// */

  /*! Initialize an ipcbuf_t struct, creating shm and sem */
  int ipcbuf_create (ipcbuf_t* ringbuf, int key, uint64 nbufs, uint64 bufsz);

  /*! Connect to an already created ipcsync_t struct in shm */
  int ipcbuf_connect (ipcbuf_t* ringbuf, int key);

  /*! Disconnect from a previously connected ipcsync_t struct in shm */
  int ipcbuf_disconnect (ipcbuf_t* ringbuf);

  /*! Destroy the ring buffer space, semaphores, and shared memory ipcbuf_t */
  int ipcbuf_destroy (ipcbuf_t* id);

  /* ////////////////////////////////////////////////////////////////////
     
     FUNCTIONS USED BY THE PROCESS WRITING TO SHARED MEMORY

     //////////////////////////////////////////////////////////////////// */

  /*! Lock this process in as the data writer */
  int ipcbuf_lock_write (ipcbuf_t* id);

  /*! Unlock this process in as the data writer */
  int ipcbuf_unlock_write (ipcbuf_t* id);

  /*! Disable the start-of-data flag */
  int ipcbuf_disable_sod (ipcbuf_t* id);

  /*! Enable the start-of-data flag */
  int ipcbuf_enable_sod (ipcbuf_t* id, uint64 st_buf, uint64 st_byte);

  /*! Get the next empty buffer available for writing.  The
    calling process must have locked "data writer" status with a call
    to ipcbuf_lock_write. */
  char* ipcbuf_get_next_write (ipcbuf_t* id);

  /*! Declare that the last buffer to be returned by
    ipcbuf_get_next_write has been filled with nbytes bytes.  The
    calling process must have locked "data writer" status with a call
    to ipcbuf_lock_write.  If nbytes is less than bufsz, then end of
    data is implicitly set. */
  int ipcbuf_mark_filled (ipcbuf_t* id, uint64 nbytes);

  /* ////////////////////////////////////////////////////////////////////
     
     FUNCTIONS USED BY THE PROCESS READING FROM SHARED MEMORY

     //////////////////////////////////////////////////////////////////// */

  /*! Lock this process in as the data reader */
  int ipcbuf_lock_read (ipcbuf_t* id);

  /*! Unlock this process in as the data reader */
  int ipcbuf_unlock_read (ipcbuf_t* id);

  /*! Get the next full buffer, and the number of bytes in it */
  char* ipcbuf_get_next_read (ipcbuf_t* id, uint64* bytes);

  /*! Declare that the last buffer to be returned by
    ipcbuf_get_next_read has been cleared and can be recycled.  The
    process must have locked "data reader" status with a call to
    ipcbuf_lock_read */
  int ipcbuf_mark_cleared (ipcbuf_t* id);

  /*! Return the state of the start-of-data flag */
  int ipcbuf_sod (ipcbuf_t* id);

  /*! Test if the current buffer is the last buffer containing data */
  int ipcbuf_eod (ipcbuf_t* id);

  /*! Return the number of bytes written to the ring buffer */
  uint64 ipcbuf_get_write_count (ipcbuf_t* id);

  /*! Return the number of bytes read from the ring buffer */
  uint64 ipcbuf_get_read_count (ipcbuf_t* id);

  /*! Return the number of buffers in the ring buffer */
  uint64 ipcbuf_get_nbufs (ipcbuf_t* id);

  /*! Return the size of each buffer in the ring */
  uint64 ipcbuf_get_bufsz (ipcbuf_t* id);

  /*! Reset the buffer count and end of data flags */
  int ipcbuf_reset (ipcbuf_t* id);

  /*! Reset the buffer count and end of data flags, with extreme prejudice */
  int ipcbuf_hard_reset (ipcbuf_t* id);

  /*! Lock the shared memory into physical RAM (must be su) */
  int ipcbuf_lock_shm (ipcbuf_t* id);

  /*! Unlock the shared memory from physical RAM (allow swap) */
  int ipcbuf_unlock_shm (ipcbuf_t* id);

  /*! Useful utility */
  void* shm_alloc (key_t key, size_t size, int flag, int* id);

#ifdef __cplusplus
	   }
#endif

#endif
