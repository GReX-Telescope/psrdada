#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

#include "ipcbuf.h"
#include "ipcutil.h"

#define _DEBUG 0

/* semaphores */

#define IPCBUF_WRITE  0   /* semaphore locks writer status */
#define IPCBUF_READ   1   /* semaphore locks reader (+clear) status */

#define IPCBUF_SODACK 2   /* acknowledgement of start of data */
#define IPCBUF_EODACK 3   /* acknowledgement of end of data */

#define IPCBUF_FULL   4   /* semaphore counts full buffers */
#define IPCBUF_CLEAR  5   /* semaphore counts emptied buffers */

#define IPCBUF_NSEM   6   /* total number of semaphores */

/* process states */

#define IPCBUF_DISCON  0  /* disconnected */
#define IPCBUF_VIEWER  1  /* connected */

#define IPCBUF_WRITER  2  /* one process that writes to the buffer */
#define IPCBUF_WRITING 3  /* start-of-data flag has been raised */
#define IPCBUF_WCHANGE 4  /* next operation will change writing state */

#define IPCBUF_READER  5  /* one process that reads from the buffer */
#define IPCBUF_READING 6  /* start-of-data flag has been raised */
#define IPCBUF_RSTOP   7  /* end-of-data flag has been raised */

static void fsleep (double seconds)
{
  struct timeval t ;
  
  t.tv_sec = seconds;
  seconds -= t.tv_sec;
  t.tv_usec = seconds * 1e6;
  select (0, 0, 0, 0, &t) ;
}

/* *************************************************************** */
/*!
  creates the shared memory block that may be used as an ipcsync_t struct

  \param key the shared memory key
  \param flag the flags to pass to shmget
*/
int ipcsync_get (ipcbuf_t* id, key_t key, uint64_t nbufs, int flag)
{
  int required = sizeof(ipcsync_t) + sizeof(key_t) * nbufs;

  if (!id) {
    fprintf (stderr, "ipcsync_get: invalid ipcbuf_t*\n");
    return -1;
  }

  id->sync = ipc_alloc (key, required, flag, &(id->syncid));
  if (id->sync == 0) {
    fprintf (stderr, "ipcsync_get: ipc_alloc error\n");
    return -1;
  }

  id -> sync -> shmkey = (key_t*) (id->sync + 1);
  id -> state = 0;
  id -> viewbuf = 0;
  id -> waitbuf = 0;

  return 0;
}

int ipcbuf_get (ipcbuf_t* id, int flag)
{
  int retval = 0;
  ipcsync_t* sync = 0;
  uint64_t ibuf = 0;

  if (!id)  {
    fprintf (stderr, "ipcbuf_get: invalid ipcbuf_t*\n");
    return -1;
  }

  sync = id -> sync;

#if _DEBUG
  fprintf (stderr, "ipcbuf_get: semkey=0x%x shmkey=0x%x\n",
	   sync->semkey, sync->shmkey[0]);
#endif

  /* all semaphores are created in this id */
  id->semid = semget (sync->semkey, IPCBUF_NSEM, flag);

  if (id->semid < 0) {
    fprintf (stderr, "ipcbuf_get: semget(0x%x, %d, 0x%x): %s\n",
	     sync->semkey, IPCBUF_NSEM, flag, strerror(errno));
    retval = -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_get: semid=%d\n", id->semid);
#endif

  id->buffer = (char**) malloc (sizeof(char*) * sync->nbufs);
  id->shmid = (int*) malloc (sizeof(int) * sync->nbufs);

  for (ibuf=0; ibuf < sync->nbufs; ibuf++) {

    id->buffer[ibuf] = ipc_alloc (sync->shmkey[ibuf], sync->bufsz, 
				  flag, id->shmid + ibuf);

    if ( id->buffer[ibuf] == 0 ) {
      perror ("ipcbuf_get: ipc_alloc");
      retval = -1;
      break;
    }
    
  }

  return retval;
}


/* *************************************************************** */
/*!
  Creates a new ring buffer in shared memory

  \return pointer to a new ipcbuf_t ring buffer struct
  \param nbufs
  \param bufsz
  \param key
*/

/* start with some random key for all of the pieces */
static int curkey = 0xc2;

int ipcbuf_create (ipcbuf_t* id, int key, uint64_t nbufs, uint64_t bufsz)
{
  uint64_t ibuf = 0;
  int flag = IPCUTIL_PERM | IPC_CREAT | IPC_EXCL;

#if _DEBUG
  fprintf (stderr, "ipcbuf_create: key=%d nbufs=%llu bufsz=%llu\n",
	   key, nbufs, bufsz);
#endif

  if (ipcsync_get (id, key, nbufs, flag) < 0) {
    fprintf (stderr, "ipcbuf_create: ipcsync_get error\n");
    return -1;
  }

  id -> sync -> nbufs  = nbufs;
  id -> sync -> bufsz  = bufsz;

  id -> sync -> s_buf  = 0;
  id -> sync -> s_byte = 0;

  id -> sync -> e_buf  = 0;
  id -> sync -> e_byte = 0;

  id -> sync -> semkey = curkey; curkey ++;

  for (ibuf = 0; ibuf < nbufs; ibuf++) {
    id -> sync -> shmkey[ibuf] = curkey;
    curkey ++;
  }

  id -> sync -> readbuf = 0;
  id -> sync -> writebuf = 0;

  id -> sync -> eod = 0;

  id -> buffer  = 0;
  id -> waitbuf = 0;

  if (ipcbuf_get (id, flag) < 0) {
    fprintf (stderr, "ipcbuf_create: ipcbuf_get error\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_create: syncid=%d semid=%d\n",
	   id->syncid, id->semid);
#endif

  /* ready to be locked by writer and reader processes */
  if (ipc_semop (id->semid, IPCBUF_WRITE, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_create: error incrementing IPCBUF_WRITE\n");
    return -1;
  }
  if (ipc_semop (id->semid, IPCBUF_READ, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_create: error incrementing IPCBUF_READ\n");
    return -1;
  }

  /* ready for writer to decrement when it needs to set SOD/EOD */
  if (ipc_semop (id->semid, IPCBUF_SODACK, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_create: error incrementing IPCBUF_SODACK\n");
    return -1;
  }
  if (ipc_semop (id->semid, IPCBUF_EODACK, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_create: error incrementing IPCBUF_EODACK\n");
    return -1;
  }

  id->state = IPCBUF_VIEWER;
  return 0;
}

int ipcbuf_connect (ipcbuf_t* id, int key)
{
  int flag = IPCUTIL_PERM;

  if (ipcsync_get (id, key, 0, flag) < 0) {
    fprintf (stderr, "ipcbuf_connect: ipcsync_get error\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_connect: key=0x%x nbufs=%llu bufsz=%llu\n",
	   key, id->sync->nbufs, id->sync->bufsz);
#endif

  id -> buffer = 0;

  if (ipcbuf_get (id, flag) < 0) {
    fprintf (stderr, "ipcbuf_connect: ipcbuf_get error\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_connect: syncid=%d semid=%d\n",
	   id->syncid, id->semid);
#endif

  id -> state = IPCBUF_VIEWER;
  return 0;
}


int ipcbuf_disconnect (ipcbuf_t* id)
{
  uint64_t ibuf = 0;

  if (!id) {
    fprintf (stderr, "ipcbuf_disconnect: invalid ipcbuf_t\n");
    return -1;
  }

  for (ibuf = 0; ibuf < id->sync->nbufs; ibuf++)
    if (id->buffer[ibuf] && shmdt (id->buffer[ibuf]) < 0)
      perror ("ipcbuf_disconnect: shmdt(buffer)");

  if (id->buffer) free (id->buffer); id->buffer = 0;
  if (id->shmid) free (id->shmid); id->shmid = 0;

  if (id->sync && shmdt (id->sync) < 0)
    perror ("ipcbuf_disconnect: shmdt(sync)");

  id->sync = 0;

  id->state = IPCBUF_DISCON; 

  return 0;
}

int ipcbuf_destroy (ipcbuf_t* id)
{
  uint64_t ibuf = 0;

  if (!id) {
    fprintf (stderr, "ipcbuf_destroy: invalid ipcbuf_t\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_destroy: semid=%d\n", id->semid);
#endif

  if (id->semid>-1 && semctl (id->semid, 0, IPC_RMID) < 0)
    perror ("ipcbuf_destroy: semctl");
  id->semid = -1;

  for (ibuf = 0; ibuf < id->sync->nbufs; ibuf++) {

#if _DEBUG
    fprintf (stderr, "ipcbuf_destroy: id[%llu]=%x\n", ibuf, id->shmid[ibuf]);
#endif

    if (id->buffer)
      id->buffer[ibuf] = 0;

    if (id->shmid[ibuf]>-1 && shmctl (id->shmid[ibuf], IPC_RMID, 0) < 0)
      perror ("ipcbuf_destroy: buf shmctl");

  }

  if (id->buffer) free (id->buffer); id->buffer = 0;
  if (id->shmid) free (id->shmid); id->shmid = 0;

#if _DEBUG
  fprintf (stderr, "ipcbuf_destroy: syncid=%d\n", id->syncid);
#endif

  if (id->syncid>-1 && shmctl (id->syncid, IPC_RMID, 0) < 0)
    perror ("ipcbuf_destroy: sync shmctl");
  id->sync = 0;
  id->syncid = -1;

  return 0;
}

/*! Lock this process in as the designated writer */
int ipcbuf_lock_write (ipcbuf_t* id)
{
  if (id->state != IPCBUF_VIEWER) {
    fprintf (stderr, "ipcbuf_lock_write: not connected\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_lock_write: decrement WRITE=%d\n",
	   semctl (id->semid, IPCBUF_WRITE, GETVAL));
#endif

  /* decrement the write semaphore (only one can) */
  if (ipc_semop (id->semid, IPCBUF_WRITE, -1, SEM_UNDO) < 0) {
    fprintf (stderr, "ipcbuf_lock_write: error decrement IPCBUF_WRITE\n");
    return -1;
  }

  /* WCHANGE is a special state that means the process will change into the
     WRITING state on the first call to get_next_write */

  id->state = IPCBUF_WCHANGE;

  return 0;
}

int ipcbuf_unlock_write (ipcbuf_t* id)
{
  if (id->state != IPCBUF_WRITER) {
    fprintf (stderr, "ipcbuf_unlock_write: state != WRITER\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_unlock_write: increment WRITE=%d\n",
	   semctl (id->semid, IPCBUF_WRITE, GETVAL));
#endif

  if (ipc_semop (id->semid, IPCBUF_WRITE, 1, SEM_UNDO) < 0) {
    fprintf (stderr, "ipcbuf_unlock_write: error increment IPCBUF_WRITE\n");
    return -1;
  }

  id->state = IPCBUF_VIEWER;

  return 0;
}

int ipcbuf_enable_eod (ipcbuf_t* id)
{
  /* must be the designated writer */
  if (id->state != IPCBUF_WRITING) {
    fprintf (stderr, "ipcbuf_enable_eod: not writing\n");
    return -1;
  }

  id->state = IPCBUF_WCHANGE;

  return 0;
}

int ipcbuf_disable_sod (ipcbuf_t* id)
{
  /* must be the designated writer */
  if (id->state != IPCBUF_WCHANGE) {
    fprintf (stderr, "ipcbuf_disable_sod: not able to change writing state\n");
    return -1;
  }

  id->state = IPCBUF_WRITER;

  return 0;
}

int ipcbuf_enable_sod (ipcbuf_t* id, uint64_t start_buf, uint64_t start_byte)
{
  ipcsync_t* sync = id -> sync;
  unsigned new_bufs = 0;

  /* must be the designated writer */
  if (id->state != IPCBUF_WRITER && id->state != IPCBUF_WCHANGE) {
    fprintf (stderr, "ipcbuf_enable_sod: not writer\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_enable_sod: start buf=%llu writebuf=%llu\n",
	   start_buf, sync->writebuf);
#endif

  /* start_buf must be less than or equal to the number of buffers written */
  if (start_buf > sync->writebuf) {
    fprintf (stderr,
	     "ipcbuf_enable_sod: start_buf=%llu > writebuf=%llu\n",
	     start_buf, sync->writebuf);
    return -1;
  }

  if (sync->writebuf >= sync->nbufs &&
      start_buf <= sync->writebuf - sync->nbufs ) {
    fprintf (stderr,
	     "ipcbuf_enable_sod: start_buf=%llu <= start_min=%llu\n",
	     start_buf, sync->writebuf-sync->nbufs);
    return -1;
  }

  /* start_byte must be less than or equal to the size of the buffer */
  if (start_byte > sync->bufsz) {
    fprintf (stderr,
	     "ipcbuf_enable_sod: start_byte=%llu > bufsz=%llu\n",
	     start_byte, sync->bufsz);
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_enable_sod: decrement SODACK=%d\n",
	   semctl (id->semid, IPCBUF_SODACK, GETVAL));
#endif

  /* decrement the start-of-data acknowlegement semaphore */
  if (ipc_semop (id->semid, IPCBUF_SODACK, -1, 0) < 0) {
    fprintf (stderr, "ipcbuf_enable_sod: error decrement SODACK\n");
    return -1;
  }

  sync->s_buf  = start_buf;
  sync->s_byte = start_byte;

#if _DEBUG
  fprintf (stderr, "ipcbuf_enable_sod: start buf=%llu byte=%llu\n",
           sync->s_buf, sync->s_byte);
#endif

  new_bufs = sync->writebuf - start_buf;

#if _DEBUG
  fprintf (stderr, "ipcbuf_enable_sod: waitbuf=%llu newbufs=%u\n",
           id->waitbuf, new_bufs);
#endif

  id->waitbuf += new_bufs;
  id->state = IPCBUF_WRITING;

#if _DEBUG
  fprintf (stderr, "ipcbuf_enable_sod: increment FULL=%d by %u\n",
	   semctl (id->semid, IPCBUF_FULL, GETVAL), new_bufs);
#endif

  /* increment the buffers written semaphore */
  if (new_bufs && ipc_semop (id->semid, IPCBUF_FULL, new_bufs, 0) < 0) {
    fprintf (stderr, "ipcbuf_enable_sod: error increment FULL\n");
    return -1;
  }

  return 0;
}

char ipcbuf_is_writer (ipcbuf_t* id)
{
  int who = id->state;
  return who==IPCBUF_WRITER || who==IPCBUF_WCHANGE || who==IPCBUF_WRITING;
}

char* ipcbuf_get_next_write (ipcbuf_t* id)
{
  uint64_t bufnum = 0;
  ipcsync_t* sync = id -> sync;

  /* must be the designated writer */
  if (!ipcbuf_is_writer(id))  {
    fprintf (stderr, "ipcbuf_get_next_write: process is not writer\n");
    return NULL;
  }

  if (id->state == IPCBUF_WCHANGE) {

#if _DEBUG
  fprintf (stderr, "ipcbuf_get_next_write: WCHANGE->WRITING\n");
#endif

    if (ipcbuf_enable_sod (id, 0, 0) < 0) {
      fprintf (stderr, "ipcbuf_get_next_write: ipcbuf_enable_sod error\n");
      return NULL;
    }

  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_get_next_write: waitbuf=%llu\n", id->waitbuf);
#endif

  if (id->waitbuf > sync->nbufs+1) {
    fprintf (stderr, "ipcbuf_get_next_write: waitbuf=%llu > nbufs+1=%llu\n",
	     id->waitbuf, sync->nbufs+1);
    return NULL;
  }

  /* waitbuf can be greater than the number of buffers when the last
   buffer of the previous session is equal to the first buffer of the
   current session */
  while (id->waitbuf >= sync->nbufs) {

#if _DEBUG
    fprintf (stderr, "ipcbuf_get_next_write: decrement CLEAR=%d\n",
	     semctl (id->semid, IPCBUF_CLEAR, GETVAL));
#endif

    /* decrement the buffers read semaphore */
    if (ipc_semop (id->semid, IPCBUF_CLEAR, -1, 0) < 0) {
      fprintf (stderr, "ipcbuf_get_next_write: error decrement CLEAR\n");
      return NULL;
    }

    id->waitbuf --;

  }

  bufnum = sync->writebuf % sync->nbufs;

  return id->buffer[bufnum];
}

int ipcbuf_mark_filled (ipcbuf_t* id, uint64_t nbytes)
{
  /* must be the designated writer */
  if (!ipcbuf_is_writer(id))  {
    fprintf (stderr, "ipcbuf_mark_filled: process is not writer\n");
    return -1;
  }

  /* increment the buffers written semaphore only if WRITING */
  if (id->state == IPCBUF_WRITER)  {
    id->sync->writebuf ++;
    return 0;
  }

  if (id->state == IPCBUF_WCHANGE || nbytes < id->sync->bufsz) {

#if _DEBUG
    if (id->state == IPCBUF_WCHANGE)
      fprintf (stderr, "ipcbuf_mark_filled: WCHANGE->WRITER\n");
#endif

#if _DEBUG
    fprintf (stderr, "ipcbuf_mark_filled: decrement EODACK=%d\n",
	     semctl (id->semid, IPCBUF_EODACK, GETVAL));
#endif
    
    if (ipc_semop (id->semid, IPCBUF_EODACK, -1, 0) < 0) {
      fprintf (stderr, "ipcbuf_mark_filled: error decrementing EODACK\n");
      return -1;
    }

    id->sync -> e_buf  = id->sync->writebuf;
    id->sync -> e_byte = nbytes;
    id->sync -> eod = 1;
    id->state = IPCBUF_WRITER;

#if _DEBUG
      fprintf (stderr, "ipcbuf_mark_filled: end buf=%llu byte=%llu\n",
	       id->sync->e_buf, id->sync->e_byte);
#endif

    if (nbytes == id->sync->bufsz)
      id->sync->writebuf ++;

  }
  else
    id->sync->writebuf ++;

#if _DEBUG
  fprintf (stderr, "ipcbuf_mark_filled: increment FULL=%d\n",
	   semctl (id->semid, IPCBUF_FULL, GETVAL));
#endif
  
  if (ipc_semop (id->semid, IPCBUF_FULL, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_mark_filled: error increment FULL\n");
    return -1;
  }
  
  id->waitbuf ++;
  
  return 0;
}

/*! Lock this process in as the designated reader */
int ipcbuf_lock_read (ipcbuf_t* id)
{
  if (id->state != IPCBUF_VIEWER) {
    fprintf (stderr, "ipcbuf_lock_read: not connected\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_lock_read: decrement READ=%d\n",
	   semctl (id->semid, IPCBUF_READ, GETVAL));
#endif

  /* decrement the read semaphore (only one can) */
  if (ipc_semop (id->semid, IPCBUF_READ, -1, SEM_UNDO) < 0) {
    fprintf (stderr, "ipcbuf_lock_read: error decrement READ\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_lock_read: reader status locked\n");
#endif

  id->state = IPCBUF_READER;
  return 0;
}

int ipcbuf_unlock_read (ipcbuf_t* id)
{
  if (id->state != IPCBUF_READER && id->state != IPCBUF_RSTOP) {
    fprintf (stderr, "ipcbuf_unlock_read: state != READER\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_unlock_read: increment READ=%d\n",
	   semctl (id->semid, IPCBUF_READ, GETVAL));
#endif

  if (ipc_semop (id->semid, IPCBUF_READ, 1, SEM_UNDO) < 0) {
    fprintf (stderr, "ipcbuf_unlock_read: error increment READ\n");
    return -1;
  }

  id->state = IPCBUF_VIEWER;
  return 0;
}

char ipcbuf_is_reader (ipcbuf_t* id)
{
  int state = id->state;
  return state==IPCBUF_READER || state==IPCBUF_READING || state==IPCBUF_RSTOP;
}


char* ipcbuf_get_next_read (ipcbuf_t* id, uint64_t* bytes)
{
  uint64_t bufnum;
  uint64_t start = 0;

  if (ipcbuf_eod (id))
    return NULL;

  if (ipcbuf_is_reader (id)) {

#if _DEBUG
    fprintf (stderr, "ipcbuf_get_next_read: decrement FULL=%d\n", 
	     semctl (id->semid, IPCBUF_FULL, GETVAL));
#endif

    /* decrement the buffers written semaphore */
    if (ipc_semop (id->semid, IPCBUF_FULL, -1, 0) < 0) {
      fprintf (stderr, "ipcbuf_get_next_read: error decrement FULL\n");
      return NULL;
    }

    if (id->state == IPCBUF_READER) {

#if _DEBUG
      fprintf (stderr, "ipcbuf_get_next_read: start buf=%llu byte=%llu\n",
	       id->sync->s_buf, id->sync->s_byte);
#endif

      id->state = IPCBUF_READING;
      id->sync->readbuf = id->sync->s_buf;
      start = id->sync->s_byte;

#if _DEBUG
    fprintf (stderr, "ipcbuf_get_next_read: increment SODACK=%d\n",
	     semctl (id->semid, IPCBUF_SODACK, GETVAL));
#endif

      /* increment the start-of-data acknowlegement semaphore */
      if (ipc_semop (id->semid, IPCBUF_SODACK, 1, 0) < 0) {
	fprintf (stderr, "ipcbuf_get_next_read: error increment SODACK\n");
	return NULL;
      }

    }

    bufnum = id->sync->readbuf;

  }
  else {

    /* KLUDGE!  wait until writebuf is incremented without sem operations */
    while (id->sync->writebuf <= id->viewbuf)
      fsleep (0.1);

    if (id->viewbuf + id->sync->nbufs < id->sync->writebuf)
      id->viewbuf = id->sync->writebuf - id->sync->nbufs + 1;

    bufnum = id->viewbuf;
    id->viewbuf ++;

  }

  bufnum %= id->sync->nbufs;

  if (bytes) {
    if (id->sync->eod && id->sync->readbuf == id->sync->e_buf)
      *bytes = id->sync->e_byte - start;
    else
      *bytes = id->sync->bufsz - start;
  }

  return id->buffer[bufnum] + start;
}


int ipcbuf_mark_cleared (ipcbuf_t* id)
{
  if (id->state != IPCBUF_READING)  {
    fprintf (stderr, "ipcbuf_mark_cleared: not reading\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_mark_cleared: increment CLEAR=%d\n",
	   semctl (id->semid, IPCBUF_CLEAR, GETVAL));
#endif

  /* increment the buffers cleared semaphore */
  if (ipc_semop (id->semid, IPCBUF_CLEAR, 1, 0) < 0)
    return -1;

  if (id->sync->eod && id->sync->readbuf == id->sync->e_buf)  {

#if _DEBUG
    fprintf (stderr, "ipcbuf_mark_cleared: increment EODACK=%d; CLEAR=%d\n",
	     semctl (id->semid, IPCBUF_EODACK, GETVAL),
	     semctl (id->semid, IPCBUF_CLEAR, GETVAL));
#endif

    id->state = IPCBUF_RSTOP;
    id->sync->eod = 0;
    if (ipc_semop (id->semid, IPCBUF_EODACK, 1, 0) < 0) {
      fprintf (stderr, "ipcbuf_mark_cleared: error incrementing EODACK\n");
      return -1;
    }

  }
  else
    id->sync->readbuf ++;

  return 0;
}

int ipcbuf_reset (ipcbuf_t* id)
{
  ipcsync_t* sync = id -> sync;

  /* if the reader has reached end of data, reset the state */
  if (id->state == IPCBUF_RSTOP) {
    id->state = IPCBUF_READER;
    return 0;
  }

  /* otherwise, must be the designated writer */
  if (!ipcbuf_is_writer(id))  {
    fprintf (stderr, "ipcbuf_mark_filled: invalid state=%d\n", id->state);
    return -1;
  }

  if (sync->writebuf == 0)
    return 0;

#if _DEBUG
  fprintf (stderr, "ipcbuf_reset: decrement CLEAR=%d by %llu\n",
	   semctl (id->semid, IPCBUF_CLEAR, GETVAL), id->waitbuf);
#endif

  /* decrement the buffers cleared semaphore */
  if (ipc_semop (id->semid, IPCBUF_CLEAR, -id->waitbuf, 0) < 0) {
    fprintf (stderr, "ipcbuf_reset: error decrementing CLEAR\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_reset: decrement SODACK=%d\n",
	   semctl (id->semid, IPCBUF_SODACK, GETVAL));
#endif

  if (ipc_semop (id->semid, IPCBUF_SODACK, -1, 0) < 0) {
    fprintf (stderr, "ipcbuf_reset: error decrementing SODACK\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_reset: decrement EODACK=%d\n",
	   semctl (id->semid, IPCBUF_EODACK, GETVAL));
#endif

  if (ipc_semop (id->semid, IPCBUF_EODACK, -1, 0) < 0) {
    fprintf (stderr, "ipcbuf_reset: error decrementing EODACK\n");
    return -1;
  }

  sync->writebuf = 0;
  sync->readbuf = 0;
  sync->eod = 0;
  id->waitbuf = 0;

#if _DEBUG
  fprintf (stderr, "ipcbuf_reset: increment SODACK=%d\n",
	   semctl (id->semid, IPCBUF_SODACK, GETVAL));
#endif

  /* ready for writer to decrement when it needs to set SOD/EOD */
  if (ipc_semop (id->semid, IPCBUF_SODACK, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_reset: error incrementing IPCBUF_SODACK\n");
    return -1;
  }

#if _DEBUG
  fprintf (stderr, "ipcbuf_reset: increment EODACK=%d\n",
	   semctl (id->semid, IPCBUF_EODACK, GETVAL));
#endif

  if (ipc_semop (id->semid, IPCBUF_EODACK, 1, 0) < 0) {
    fprintf (stderr, "ipcbuf_reset: error incrementing IPCBUF_EODACK\n");
    return -1;
  }

  return 0;
}

/* reset the buffer count and end of data flags, without prejudice */
int ipcbuf_hard_reset (ipcbuf_t* id)
{
  ipcsync_t* sync = id -> sync;
  int val = 0;

  sync->writebuf = 0;
  sync->readbuf = 0;
  sync->eod = 0;

  if (semctl (id->semid, IPCBUF_FULL, SETVAL, val) < 0) {
    perror ("ipcbuf_hard_reset: semctl (IPCBUF_FULL, SETVAL)");
    return -1;
  }

  if (semctl (id->semid, IPCBUF_CLEAR, SETVAL, val) < 0) {
    perror ("ipcbuf_hard_reset: semctl (IPCBUF_FULL, SETVAL)");
    return -1;
  }

  return 0;
}

int ipcbuf_lock (ipcbuf_t* id)
{
  uint64_t ibuf = 0;

  if (id->syncid < 0 || id->shmid == 0)
    return -1;

#ifdef SHM_LOCK
  if (shmctl (id->syncid, SHM_LOCK, 0) < 0) {
    perror ("ipcbuf_lock: shmctl (syncid, SHM_LOCK)");
    return -1;
  }

  for (ibuf = 0; ibuf < id->sync->nbufs; ibuf++)
    if (shmctl (id->shmid[ibuf], SHM_LOCK, 0) < 0) {
      perror ("ipcbuf_lock: shmctl (shmid, SHM_LOCK)");
      return -1;
    }

#else
  fprintf(stderr, "Warning: ipcbuf_lock called but does nothing on this platform!\n");
#endif

  return 0;
}

int ipcbuf_unlock (ipcbuf_t* id)
{
  uint64_t ibuf = 0;

  if (id->syncid < 0 || id->shmid == 0)
    return -1;

#ifdef SHM_UNLOCK
  if (shmctl (id->syncid, SHM_UNLOCK, 0) < 0) {
    perror ("ipcbuf_lock: shmctl (syncid, SHM_UNLOCK)");
    return -1;
  }

  for (ibuf = 0; ibuf < id->sync->nbufs; ibuf++)
    if (shmctl (id->shmid[ibuf], SHM_UNLOCK, 0) < 0) {
      perror ("ipcbuf_lock: shmctl (shmid, SHM_UNLOCK)");
      return -1;
    }

#else
  fprintf(stderr, "Warning: ipcbuf_unlock does nothing on this platform!\n");
#endif

  return 0;
}


int ipcbuf_eod (ipcbuf_t* id)
{
  return id->state == IPCBUF_RSTOP;
}


int ipcbuf_sod (ipcbuf_t* id)
{
  return id->state == IPCBUF_READING;
}

uint64_t ipcbuf_get_write_count (ipcbuf_t* id)
{
  return id -> sync -> writebuf;
}

uint64_t ipcbuf_get_read_count (ipcbuf_t* id)
{
  return id -> sync -> readbuf;
}

uint64_t ipcbuf_get_nbufs (ipcbuf_t* id)
{
  return id -> sync -> nbufs;
}

uint64_t ipcbuf_get_bufsz (ipcbuf_t* id)
{
  return id -> sync -> bufsz;
}
