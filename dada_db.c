#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/ipc.h> 
#include <sys/shm.h>
#include <sys/sem.h>

#include "dada_db.h"
#include "ipcutil.h"

int dada_db_construct (dada_db_t* db, uint64 header_size, int flag)
{
  db->header = ipc_alloc (DADA_KEY, header_size, flag, &(db->shmid));

  if (db->header == 0) {
    fprintf (stderr, "dada_db_construct: ipc_alloc error\n");
    return -1;
  }

  db->semid = semget (DADA_KEY, 1, flag);

  if (db->semid < 0) {
    perror ("dada_db_construct: semget");
    return -1;
  }

  return 0;
}

int dada_db_semop (dada_db_t* db, int incr)
{
  struct sembuf semopbuf;

  semopbuf.sem_num = 0;
  semopbuf.sem_flg = 0;
  semopbuf.sem_op = incr;
  
  if (semop (db->semid, &semopbuf, 1) < 0)  {
    perror ("dada_db_semop: semop");
    return -1;
  }
  return 0;
}

int dada_db_create (dada_db_t* db, uint64 hdrsz, uint64 nbufs, uint64 bufsz)
{
  if (ipcio_create (&(db->ipcio), DADA_KEY+1, nbufs, bufsz) < 0) {
    fprintf (stderr, "dada_db_create: ipcio_create error\n");
    return -1;
  }

  if (dada_db_construct (db, hdrsz, IPCUTIL_PERM|IPC_CREAT|IPC_EXCL) < 0) {
    fprintf (stderr, "dada_db_create: ipcio_create error\n");
    return -1;
  }

  /* initialize the semaphore to one */
  return dada_db_semop (db, 1);
}


/* connect to an already created ipcbuf_t struct in shared memory */
int dada_db_connect (dada_db_t* db)
{
  if (ipcio_connect (&(db->ipcio), DADA_KEY+1) < 0) {
    fprintf (stderr, "dada_db_connect: ipcio_connect error\n");
    return -1;
  }

  return dada_db_construct (db, 0, IPCUTIL_PERM);
}

int dada_db_disconnect (dada_db_t* db)
{
  if (ipcio_disconnect(&(db->ipcio)) < 0) {
    fprintf (stderr, "dada_db_disconnect: ipcio_disconnect error\n");
    return -1;
  }
  return EXIT_SUCCESS;
}
 

/* destroy the cpsr2 ring buffer and header */
int dada_db_destroy (dada_db_t* db)
{
  if (!db) {
    fprintf (stderr, "dada_db_destroy: invalid dada_db_t\n");
    return -1;
  }

  if (ipcio_destroy (&(db->ipcio)) < 0)
    fprintf (stderr, "dada_db_destroy: error ipcio_destroy\n");

  if (db->shmid>-1 && shmctl (db->shmid, IPC_RMID, 0) < 0)
    perror ("dada_db_destroy: shmctl");

  db->shmid = -1;
  db->header = 0;

  if (db->semid>-1 && semctl (db->semid, 0, IPC_RMID) < 0)
    perror ("dada_db_destroy: semctl");

  db->semid = -1;

  return 0;
}

char* dada_db_get_header_write (dada_db_t* db)
{
  if (dada_db_semop (db, -1) < 0)
    return 0;
  return db->header;
}

int dada_db_mark_header_written (dada_db_t* db)
{
  return dada_db_semop (db, 2);
}

char* dada_db_get_header_read (dada_db_t* db)
{
  if (dada_db_semop (db, -2) < 0)
    return 0;
  return db->header;
}

int dada_db_mark_header_read (dada_db_t* db)
{
  return dada_db_semop (db, 1);
}
