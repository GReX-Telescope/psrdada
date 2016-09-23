#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipcio_cuda.h"
#include "ipcbuf_cuda.h"

/* read bytes from ipcbuf, writing to device memory via H2D transfer*/
extern "C"
ssize_t ipcio_read_cuda (ipcio_t* ipc, char* ptr, size_t bytes, cudaStream_t stream)
{
  size_t space = 0;
  size_t toread = bytes;

  if (ipc -> rdwrt != 'r' && ipc -> rdwrt != 'R')
  {
    fprintf (stderr, "ipcio_read_cuda invalid ipcio_t (rdwrt=%c)\n", ipc->rdwrt);
    return -1;
  }

  while (!ipcbuf_eod((ipcbuf_t*)ipc))
  {
    if (!ipc->curbuf)
    {
      ipc->curbuf = ipcbuf_get_next_read ((ipcbuf_t*)ipc, &(ipc->curbufsz));

#ifdef _DEBUG
      fprintf (stderr, "ipcio_read buffer:%"PRIu64" %"PRIu64" bytes. buf[0]=%x\n",
               ipc->buf.sync->r_buf, ipc->curbufsz, ipc->curbuf[0]);
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
          cudaMemcpyAsync (ptr, ipc->curbuf + ipc->bytes, space, cudaMemcpyHostToDevice, stream);
        else
          cudaMemcpy (ptr, ipc->curbuf + ipc->bytes, space, cudaMemcpyHostToDevice);
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
        cudaThreadSynchronize ();

      if (ipc -> rdwrt == 'R' && ipcbuf_mark_cleared ((ipcbuf_t*)ipc) < 0)
      {
        fprintf (stderr, "ipcio_read_cuda error ipcbuf_mark_filled\n");
        return -1;
      }

      ipc->curbuf = 0;
      ipc->bytes = 0;
    }
    else if (!bytes)
      break;
  }

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

