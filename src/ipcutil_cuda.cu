#include "ipcutil_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>

#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

// #define _DEBUG 1

/* *************************************************************** */
/*!
  Returns a device pointer to a shared memory block and its shmid
*/
void* ipc_alloc_cuda (key_t key, size_t size, int flag, int* shmid, int device_id)
{
  void * devPtr = 0;
  void * buf = 0;
  cudaIpcMemHandle_t * handlePtr = 0;
  int id = 0;
  size_t handle_size = sizeof(cudaIpcMemHandle_t);
  cudaError_t error;

#ifdef _DEBUG
  fprintf (stderr, "ipc_alloc_cuda: shmget(key=%x size=%ld, flag=%x\n", 
           key, handle_size, flag);
#endif

  // we want to extract the IPC handle
  id = shmget (key, handle_size, flag);
  if (id < 0) 
  {
     fprintf (stderr, "ipc_alloc_cuda: shmget (key=%x, size=%ld, flag=%x) %s\n",
              key, handle_size, flag, strerror(errno));
     return 0;
  }

#ifdef _DEBUG
  fprintf (stderr, "ipc_alloc_cuda: shmid=%d\n", id);
#endif

  // pointer to cudaIpcMemHandle_t
  buf = shmat (id, 0, flag);
  if (buf == (void *)-1) 
  {
    fprintf (stderr,
       "ipc_alloc_cuda: shmat (shmid=%d) %s\n"
       "ipc_alloc_cuda: after shmget (key=%x, size=%ld, flag=%x)\n",
       id, strerror(errno), key, size, flag);
    return 0;
  }

  handlePtr = (cudaIpcMemHandle_t *) buf;

#ifdef _DEBUG
  fprintf (stderr, "ipc_alloc_cuda: buf=%p handlePtr=%p\n", buf, handlePtr);
  fprintf (stderr, "ipc_alloc_cuda: selecting device %d\n", device_id);
#endif

  cudaSetDevice (device_id);

  // if we are wanting to create a shared memory segment of size bytes
  if (flag & IPC_CREAT)
  {
    // allocate device memory
#ifdef _DEBUG
    fprintf (stderr, "ipc_alloc_cuda: allocating device memory of size %ld bytes\n", size);
#endif
    error = cudaMalloc (&devPtr, size);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "failed to allocate %ld bytes on device %d: %s\n", 
               size, device_id, cudaGetErrorString (error));
      return 0;
    }
#ifdef _DEBUG
    fprintf (stderr, "ipc_alloc_cuda: cudaIpcGetMemHandle (%p, %p)\n", handlePtr, devPtr);
#endif

    // get an event handle associated with that memory, writing to handlePtr
    error = cudaIpcGetMemHandle (handlePtr, devPtr);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "failed to get IPC memory handle for devPtr=%p on device %d: %s\n", 
               devPtr, device_id, cudaGetErrorString (error));
      return 0;
    }
  }
  else
  {
    error = cudaIpcOpenMemHandle (&devPtr, *handlePtr, cudaIpcMemLazyEnablePeerAccess);
    if (error != cudaSuccess)
    {
      fprintf (stderr, "ipc_alloc_cuda: failed to open memory handle to segment: %s\n",
              cudaGetErrorString (error));
      return 0;
    }
  }

#ifdef _DEBUG
  fprintf (stderr, "ipc_alloc_cuda: devPtr=%p\n", devPtr);
#endif

  if (shmid)
    *shmid = id;

  return devPtr;
}

// detach from memory segment
int ipc_disconnect_cuda (void * devPtr)
{
  cudaError_t error = cudaIpcCloseMemHandle (devPtr);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "ipc_disconnect_cuda: failed to close memory handle to segment: %s\n",
             cudaGetErrorString (error));
    return -1;
  }

  return 0;
}

// deallocate memory segment
int ipc_dealloc_cuda (void * devPtr, int device_id)
{
  cudaSetDevice (device_id);
  cudaError_t error = cudaFree (devPtr);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "ipc_disconnect_cuda: failed to free device memory: %s\n",
             cudaGetErrorString (error));
    return -1;
  }

  return 0;
}

int ipc_zero_buffer_cuda (void * devPtr, size_t nbytes)
{
  cudaError_t error = cudaMemset (devPtr, 0, nbytes);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "ipc_zero_buffer_cuda: failed to zero device memory: %s\n",
             cudaGetErrorString (error));
    return -1;
  }

  return 0;
}

