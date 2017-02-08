#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#include "tmutil.h"
#include "ipcbuf_cuda.h"

/* zero bytes in an ipcbuf, by reading from previously zerod device memory */
extern "C"
ssize_t ipcbuf_zero_next_block_cuda (ipcbuf_t* id, char * dev_ptr, size_t dev_bytes, cudaStream_t stream)
{
  ipcsync_t* sync = id->sync;

  /* must be the designated writer */
  if (!ipcbuf_is_writer(id))
  {
    fprintf (stderr, "ipcbuf_zero_next_block_cuda: process is not writer\n");
    return -1;
  }

  // get the next buffer to be written
  uint64_t next_buf = (sync->w_buf + 1) % sync->nbufs;

  char have_cleared = 0;
  unsigned iread;
  while (!have_cleared)
  {
    have_cleared = 1;
    // check that each reader has 1 clear buffer at least
    for (iread = 0; iread < sync->n_readers; iread++ )
    {
      if (semctl (id->semid_data[iread], IPCBUF_CLEAR, GETVAL) == 0)
        have_cleared = 0;
    }
    if (!have_cleared)
      float_sleep((double)0.01);
  }

  uint64_t bytes_zeroed = 0;

  while (bytes_zeroed < id->sync->bufsz)
  {
    uint64_t bytes_to_zero = id->sync->bufsz - bytes_zeroed;
    if (bytes_to_zero > dev_bytes)
      bytes_to_zero = dev_bytes;

    if (stream)
      cudaMemcpyAsync (id->buffer[next_buf], dev_ptr, bytes_to_zero, cudaMemcpyDeviceToHost, stream);
    else
      cudaMemcpy (id->buffer[next_buf], dev_ptr, bytes_to_zero, cudaMemcpyDeviceToHost);
    bytes_zeroed += bytes_to_zero;
  }

  // NB explicitly do not synchronize, expect the called to do this before using the buffer[next_buf]
  //if (stream)
  //  cudaStreamSynchronize (stream);

  return 0;
}
