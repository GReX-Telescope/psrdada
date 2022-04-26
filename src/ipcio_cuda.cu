#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "ipcio_cuda.h"
#include "ipcbuf_cuda.h"

// #define _DEBUG

/* read bytes from ipcbuf, writing to device memory via H2D transfer*/
extern "C"
ssize_t ipcio_read_cuda (ipcio_t* ipc, char* ptr, size_t bytes, cudaStream_t stream)
{
#ifdef _DEBUG
  fprintf (stderr, "ipcio_read_cuda (%p, %p, %ld, stream)\n", ipc, ptr, bytes);
#endif
  return ipcio_read_cuda_work (ipc, ptr, NULL, bytes, stream);
}

extern "C"
ssize_t ipcio_read_zero_cuda (ipcio_t* ipc, char* ptr, char * zero_ptr,  size_t bytes, cudaStream_t stream)
{
#ifdef _DEBUG
  fprintf (stderr, "ipcio_read_zero_cuda (%p, %p, %p, %ld, stream)\n", ipc, ptr, zero_ptr, bytes);
#endif
  return ipcio_read_cuda_work (ipc, ptr, zero_ptr, bytes, stream);
}

ssize_t ipcio_read_cuda_work (ipcio_t* ipc, char* ptr, char * zero_ptr, size_t bytes, cudaStream_t stream)
{
#ifdef _DEBUG
  fprintf (stderr, "ipcio_read_cuda_work (%p, %p, %p, %ld, stream)\n", ipc, ptr, zero_ptr, bytes);
#endif
  size_t space = 0;
  size_t toread = bytes;

  if (ipc -> rdwrt != 'r' && ipc -> rdwrt != 'R')
  {
    fprintf (stderr, "ipcio_read_cuda_work: invalid ipcio_t (rdwrt=%c)\n", ipc->rdwrt);
    return -1;
  }
 
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  cudaMemcpyKind invkind = cudaMemcpyDeviceToHost; 

  // >= 0 if buffers are located in device memory
  if (ipcbuf_get_device((ipcbuf_t*)ipc) >= 0)
  {
    kind = cudaMemcpyDeviceToDevice;
    invkind = cudaMemcpyDeviceToDevice;
  }

  while (!ipcbuf_eod((ipcbuf_t*)ipc))
  {
    if (!ipc->curbuf)
    {
      ipc->curbuf = ipcbuf_get_next_read ((ipcbuf_t*)ipc, &(ipc->curbufsz));

#ifdef _DEBUG
      fprintf (stderr, "ipcio_read_cuda_work: buffer: %lu bytes. buf[0]=%p\n",
               ipc->curbufsz, (void *) ipc->curbuf);
#endif

      if (!ipc->curbuf)
      {
        fprintf (stderr, "ipcio_read_cuda error ipcbuf_next_read\n");
        return -1;
      }

      ipc->bytes = 0;
    }
    if (bytes)
    {
      space = ipc->curbufsz - ipc->bytes;
      if (space > bytes)
        space = bytes;

      if (ptr)
      {
        if (stream)
          cudaMemcpyAsync (ptr, ipc->curbuf + ipc->bytes, space, kind, stream);
        else
          cudaMemcpy (ptr, ipc->curbuf + ipc->bytes, space, kind);

        if (zero_ptr)
        {
          if (stream)
            cudaMemcpyAsync (ipc->curbuf + ipc->bytes, zero_ptr, space, invkind, stream);
          else
            cudaMemcpy (ipc->curbuf + ipc->bytes, zero_ptr, space, invkind);
        }
        ptr += space;
      }

      ipc->bytes += space;
      bytes -= space;
    }
    if (ipc->bytes == ipc->curbufsz)
    {
      if (stream)
        cudaStreamSynchronize (stream);
      else
        cudaDeviceSynchronize ();

      if (ipc -> rdwrt == 'R' && ipcbuf_mark_cleared ((ipcbuf_t*)ipc) < 0)
      {
        fprintf (stderr, "ipcio_read_cuda_work: error ipcbuf_mark_filled\n");
        return -1;
      }

      ipc->curbuf = 0;
      ipc->bytes = 0;
    }
    else if (!bytes)
      break;
  }

#ifdef _DEBUG
  fprintf (stderr, "ipcio_read_cuda_work: to_read=%ld bytes=%ld return=%ld\n", 
           toread, bytes, (toread - bytes));
#endif
  return toread - bytes;
}

extern "C"
ssize_t ipcio_zero_next_block_cuda (ipcio_t* ipc, char * dev_ptr, size_t dev_bytes, cudaStream_t stream)
{
  if (ipc -> rdwrt != 'W')
  {
    fprintf(stderr, "ipcio_zero_next_block_cuda: ipc -> rdwrt != W\n");
    return -1;
  }

  return ipcbuf_zero_next_block_cuda ((ipcbuf_t*)ipc, dev_ptr, dev_bytes, stream);
}

