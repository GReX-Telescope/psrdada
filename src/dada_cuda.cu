/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_cuda.h"

inline void start_profiling(const cudaEvent_t& start_event,
                            const cudaStream_t& stream=0)
{
  cudaEventRecord(start_event, stream);

}
inline float stop_profiling(const cudaEvent_t& start_event,
                            const cudaEvent_t& stop_event,
                            const cudaStream_t& stream=0)
{
  float elapsed = 0.f;
  cudaEventRecord(stop_event, stream);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&elapsed, start_event, stop_event);
  return elapsed;
}


/*! select the specified GPU as the active device */
int dada_cuda_select_device (int index)
{
  cudaError_t error_id = cudaSetDevice (index);
  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "dada_cuda_select_device: cudaSetDevice failed: %s\n",
                      cudaGetErrorString(error_id));
    return -1;
  }
  return 0;
}

/*! get the number of CUDA devices */
int dada_cuda_get_device_count ()
{
  int device_count = 0;
  cudaError_t error_id = cudaGetDeviceCount(&device_count);
  if (error_id != cudaSuccess) 
  {
    fprintf (stderr, "dada_cuda_get_device_count: cudaGetDeviceCount failed: %s\n", 
	                   cudaGetErrorString(error_id) );
    return -1;
  }
  return device_count;
}

/*! get the name of the specified CUDA device */
char * dada_cuda_get_device_name (int index)
{
  cudaDeviceProp device_prop;
  cudaError_t error_id = cudaGetDeviceProperties(&device_prop, index);
  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "dada_cuda_get_device_name: cudaGetDeviceProperties failed: %s\n",
                     cudaGetErrorString(error_id) );
    return 0;
  }

  return strdup(device_prop.name);
}

/*! register the data_block in the hdu via cudaHostRegister */
int dada_cuda_dbregister (dada_hdu_t * hdu)
{

  ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;

  // ensure that the data blocks are SHM locked
  if (ipcbuf_lock (db) < 0)
  {
    perror("dada_dbregister: ipcbuf_lock failed\n");
    return -1;
  }

  size_t bufsz = db->sync->bufsz;
  unsigned int flags = 0;
  cudaError_t rval;

  // lock each data block buffer as cuda memory
  uint64_t ibuf;
  for (ibuf = 0; ibuf < db->sync->nbufs; ibuf++)
  {
    rval = cudaHostRegister ((void *) db->buffer[ibuf], bufsz, flags);
    if (rval != cudaSuccess)
    {
      perror("dada_dbregister:  cudaHostRegister failed\n");
      return -1;
    }
  }
  
  return 0;
}

/*! unregister the data_block in the hdu via cudaHostUnRegister */
int dada_cuda_dbunregister (dada_hdu_t * hdu)
{

  ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;
  cudaError_t error_id;

  // lock each data block buffer as cuda memory
  uint64_t ibuf;
  for (ibuf = 0; ibuf < db->sync->nbufs; ibuf++)
  {
    error_id = cudaHostUnregister ((void *) db->buffer[ibuf]);
    if (error_id != cudaSuccess)
    {
      fprintf (stderr, "dada_dbunregister: cudaHostUnregister failed: %s\n",
               cudaGetErrorString(error_id));
      return -1;
    }
  }

  return 0;
}

/*! return a pointer to GPU device memory of bytes size */
void * dada_cuda_device_malloc ( size_t bytes)
{
  cudaError_t error_id;
  void * device_memory;
  error_id = cudaMalloc (&device_memory, bytes);
  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "dada_cuda_device_malloc: could not allocate %d bytes: %s\n", 
                      bytes, cudaGetErrorString(error_id));
    return 0;
  }
  return device_memory;
}

/*! free the specified GPU device memory */
int dada_cuda_device_free (void * memory)
{
  cudaError_t error_id;
  error_id = cudaFree (memory);
  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "dada_cuda_device_free: could not free memory: %s\n",
                      cudaGetErrorString(error_id));
    return -1;
  }
  return 0;

}

/*! transfer the supplied buffer to the GPU */
float dada_cuda_device_transfer ( void * from, void * to, size_t size, memory_mode_t mode, dada_cuda_profile_t * timer)
{

  cudaError_t error_id;

  start_profiling (timer->start_event, 0);

  if (mode == PINNED)
    error_id = cudaMemcpyAsync (to , from, size, cudaMemcpyHostToDevice, 0);
  else
    error_id = cudaMemcpy (to, from, size, cudaMemcpyHostToDevice);

  float elapsed = stop_profiling(timer->start_event, timer->stop_event, 0);

  if (error_id != cudaSuccess)
  {
    fprintf (stderr, "dada_cuda_device_transfer: memcpy failed: %s\n",
                      cudaGetErrorString(error_id));
    return -1;
  }

  return elapsed;
}


int dada_cuda_profiler_init (dada_cuda_profile_t * timer)
{
  cudaEventCreate (&(timer->start_event));
  cudaEventCreate (&(timer->stop_event));
  timer->elapsed = 0;
  return 0;
}
