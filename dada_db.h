#ifndef __DADA_DB_H
#define __DADA_DB_H

/* ************************************************************************

   dada_db_t - a struct and associated routines for creating, managing,
   and reading/writing to/from the Data Block in shared memory

   ************************************************************************ */

#include "ipcio.h"

#define DADA_KEY  0x00c2

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    ipcio_t ipcio;   /* ipcio_t struct on which this is based */

    int semid;       /* semaphore to protect header */
    int shmid;       /* shared memory id of header */

    char* header;    /* base address of header in shared memory */
    
  } dada_db_t;

#define DADA_DB_INIT { IPCIO_INIT, -1,-1,0 }

  /* create a new shared memory block and initialize an db_t struct */
  int dada_db_create (dada_db_t* db, uint64 hdrsz, uint64 nbufs, uint64 bufsz);

  /* connect to an already created dbbuf_t struct in shared memory */
  int dada_db_connect (dada_db_t* db);

  /* disconnect to an already created dbbuf_t struct in shared memory */
  int dada_db_disconnect (dada_db_t* db);
  
  /* attach to existing ring buffer essentially a copy of the base address */
  int dada_db_attach (dada_db_t* db, ipcbuf_t* base);

  /* destroy the dada ring buffer and header */
  int dada_db_destroy (dada_db_t* db);

  /* wait until the header has been read */
  char* dada_db_get_header_write (dada_db_t* db);

  /* wait until the header has been written */
  char* dada_db_get_header_read (dada_db_t* db);

  /* mark that the header has been written */
  int dada_db_mark_header_written (dada_db_t* db);

  /* mark that the header has been read */
  int dada_db_mark_header_read (dada_db_t* db);

#ifdef __cplusplus
	   }
#endif

#endif
