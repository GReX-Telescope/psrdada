#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

#include "ipcutil.h"

#define _DEBUG 0

/* *************************************************************** */
/*!
  Returns a shared memory block and its shmid
*/
void* ipc_alloc (key_t key, int size, int flag, int* shmid)
{
  void* buf = 0;
  int id = 0;

  id = shmget (key, size, flag);
  if (id < 0) {
    fprintf (stderr, "ipc_alloc: shmget (key=%x, size=%d, flag=%x) %s",
             key, size, flag, strerror(errno));
    return 0;
  }

#if _DEBUG
  fprintf (stderr, "ipc_alloc: shmid=%d\n", id);
#endif

  buf = shmat (id, 0, flag);

  if (buf == (void*)-1) {
    fprintf (stderr, "ipc_alloc: shmat (shmid=%d=shmget "
                     "(key=%x, size=%d, flag=%x)) = %p - %s",
                     id, key, size, flag, buf, strerror(errno));
    return 0;
  }

#if _DEBUG
  fprintf (stderr, "ipc_alloc: shmat=%p\n", buf);
#endif

  if (shmid)
    *shmid = id;

  return buf;
}

int ipc_semop (int semid, short num, short op, short flag)
{
  struct sembuf semopbuf;

  semopbuf.sem_num = num;
  semopbuf.sem_op = op;
  semopbuf.sem_flg = flag;
 
  if (semop (semid, &semopbuf, 1) < 0)  {
    perror ("ipc_semop: semop");
    return -1;
  }
  return 0;
}
