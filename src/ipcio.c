#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ipcio.h"

#define _DEBUG 0

void ipcio_init (ipcio_t* ipc)
{
  ipc -> bytes = 0;
  ipc -> rdwrt = 0;
  ipc -> curbuf = 0;
}

/* create a new shared memory block and initialize an ipcio_t struct */
int ipcio_create (ipcio_t* ipc, int key, uint64 nbufs, uint64 bufsz)
{
  if (ipcbuf_create (&(ipc->buf), key, nbufs, bufsz) < 0) {
    fprintf (stderr, "ipcio_create: ipcbuf_create error\n");
    return -1;
  }
  ipcio_init (ipc);
  return 0;
}

/* connect to an already created ipcbuf_t struct in shared memory */
int ipcio_connect (ipcio_t* ipc, int key)
{
  if (ipcbuf_connect (&(ipc->buf), key) < 0) {
    fprintf (stderr, "ipcio_connect: ipcbuf_connect error\n");
    return -1;
  }
  ipcio_init (ipc);
  return 0;
}

/* disconnect from an already connected ipcbuf_t struct in shared memory */
int ipcio_disconnect (ipcio_t* ipc)
{
  if (ipcbuf_disconnect (&(ipc->buf)) < 0) {
    fprintf (stderr, "ipcio_disconnect: ipcbuf_disconnect error\n");
    return -1;
  }
  ipcio_init (ipc);
  return 0;
}

int ipcio_destroy (ipcio_t* ipc)
{
  ipcio_init (ipc);
  return ipcbuf_destroy (&(ipc->buf));
}

/* start reading/writing to an ipcbuf */
int ipcio_open (ipcio_t* ipc, char rdwrt)
{
  if (rdwrt != 'R' && rdwrt != 'r' && rdwrt != 'w' && rdwrt != 'W') {
    fprintf (stderr, "ipcio_open: invalid rdwrt = '%c'\n", rdwrt);
    return -1;
  }
  ipc -> rdwrt = rdwrt;
  ipc -> bytes = 0;
  ipc -> curbuf = 0;

  if (rdwrt == 'w' || rdwrt == 'W') {
    /* read from file, write to shm */
    if (ipcbuf_lock_write (&(ipc->buf)) < 0) {
      fprintf (stderr, "ipcio_open: error ipcbuf_lock_write\n");
      return -1;
    }
    ipc -> rdwrt = 'W';
    return ipcbuf_reset (&(ipc->buf));
  }

  if (rdwrt == 'R') {
    if (ipcbuf_lock_read (&(ipc->buf)) < 0) {
      fprintf (stderr, "ipcio_open: error ipcbuf_lock_read\n");
      return -1;
    }
  }

  return 0;
}

/* start writing valid data to an ipcbuf */
int ipcio_start (ipcio_t* ipc, uint64 sample)
{
  uint64 bufsz   = ipcbuf_get_bufsz(&(ipc->buf));
  uint64 st_buf  = sample / bufsz;
  uint64 st_byte = sample % bufsz;

  return ipcbuf_enable_sod (&(ipc->buf), st_buf, st_byte);
}


/* stop reading/writing to an ipcbuf */
int ipcio_stop_close (ipcio_t* ipc, char unlock)
{
  if (ipc -> rdwrt == 'W') {

#if _DEBUG
    if (ipc->curbuf)
      fprintf (stderr, "ipcio_close:W buffer:"UI64" "UI64
	       " bytes. buf[0]=%x\n",
	       ipc->buf.sync->writebuf, ipc->bytes, ipc->curbuf[0]);
#endif

    if (ipcbuf_mark_filled (&(ipc->buf), ipc->bytes) < 0) {
      fprintf (stderr, "ipcio_close:W error ipcbuf_mark_filled\n");
      return -1;
    }

    if (ipc->bytes == ipcbuf_get_bufsz(&(ipc->buf))) {

#if _DEBUG
      fprintf (stderr, "ipcio_close:W last buffer\n");
#endif

      if (ipcbuf_mark_filled (&(ipc->buf), 0) < 0)  {
	fprintf (stderr, "ipcio_close:W error ipcbuf_mark_filled EOF\n");
	return -1;
      }

    }

    if (unlock) {

#if _DEBUG
      fprintf (stderr, "ipcio_close:W calling ipcbuf_reset\n");
#endif

      if (ipcbuf_reset (&(ipc->buf)) < 0) {
	fprintf (stderr, "ipcio_close:W error ipcbuf_reset\n");
	return -1;
      }

      if (ipcbuf_unlock_write (&(ipc->buf)) < 0) {
	fprintf (stderr, "ipcio_close:W error ipcbuf_unlock_write\n");
	return -1;
      }

    }

    return 0;

  }

  if (ipc -> rdwrt == 'R') {

    if (ipcbuf_unlock_read (&(ipc->buf)) < 0) {
      fprintf (stderr, "ipcio_close:R error ipcbuf_unlock_read\n");
      return -1;
    }

    return 0;

  }

  fprintf (stderr, "ipcio_close: invalid ipcio_t\n");
  return -1;
}

/* stop writing valid data to an ipcbuf */
int ipcio_stop (ipcio_t* ipc)
{
  return ipcio_stop_close (ipc, 0);
}

/* stop reading/writing to an ipcbuf */
int ipcio_close (ipcio_t* ipc)
{
  return ipcio_stop_close (ipc, 1);
}

/* write bytes to ipcbuf */
ssize_t ipcio_write (ipcio_t* ipc, char* ptr, size_t bytes)
{
  size_t space = 0;
  size_t towrite = bytes;

  if (ipc -> rdwrt != 'W') {
    fprintf (stderr, "ipcio_write: invalid ipcio_t\n");
    return -1;
  }

  while (bytes) {

    if (!ipc->curbuf) {
      ipc->curbuf = ipcbuf_get_next_write (&(ipc->buf));

      if (!ipc->curbuf) {
	fprintf (stderr, "ipcio_write: ipcbuf_next_write\n");
	return -1;
      }

      ipc->bytes = 0;
    }

    space = ipcbuf_get_bufsz(&(ipc->buf)) - ipc->bytes;
    if (space > bytes)
      space = bytes;

    if (space > 0) {
      memcpy (ipc->curbuf + ipc->bytes, ptr, space);
      ipc->bytes += space;
      ptr += space;
      bytes -= space;
    }
    else {

#if _DEBUG
      fprintf (stderr, "ipcio_write buffer:"UI64" "UI64" bytes. buf[0]=%x\n",
	       ipc->buf.sync->writebuf, ipc->bytes, ipc->curbuf[0]);
#endif

      /* the buffer has been filled */
      if (ipcbuf_mark_filled (&(ipc->buf), ipc->bytes) < 0) {
	fprintf (stderr, "ipcio_write: ipcbuf_mark_filled\n");
	return -1;
      }


      ipc->curbuf = 0;
    }

  }

  return towrite;
}


/* read bytes from ipcbuf */
ssize_t ipcio_read (ipcio_t* ipc, char* ptr, size_t bytes)
{
  size_t space = 0;
  size_t toread = bytes;

  if (ipc -> rdwrt != 'r' && ipc -> rdwrt != 'R') {
    fprintf (stderr, "ipcio_read: invalid ipcio_t\n");
    return -1;
  }

  while (bytes && ! ipcbuf_eod(&(ipc->buf))) {

    if (!ipc->curbuf) {

      ipc->curbuf = ipcbuf_get_next_read (&(ipc->buf), &(ipc->curbufsz));

#if _DEBUG
      fprintf (stderr, "ipcio_read buffer:"UI64" "UI64" bytes. buf[0]=%x\n",
	       ipc->buf.sync->readbuf, ipc->curbufsz, ipc->curbuf[0]);
#endif

      if (!ipc->curbuf) {
	fprintf (stderr, "ipcio_read: error ipcbuf_next_read\n");
	return -1;
      }

      ipc->bytes = 0;

    }

    space = ipc->curbufsz - ipc->bytes;
    if (space > bytes)
      space = bytes;

    if (space > 0) {

      /* fprintf (stderr, "space=%d curbufsz="UI64" bytes"UI64"\n",
	 space, ipc->curbufsz, ipc->bytes); */

      memcpy (ptr, ipc->curbuf + ipc->bytes, space);
      
      ipc->bytes += space;
      ptr += space;
      bytes -= space;

    }
    else {

      if (ipc -> rdwrt == 'R') {
	if (ipcbuf_mark_cleared (&(ipc->buf)) < 0) {
	  fprintf (stderr, "ipcio_write: error ipcbuf_mark_filled\n");
	  return -1;
	}
      }

      ipc->curbuf = 0;

    }
  }

  return toread - bytes;
}

int64 ipcio_seek (ipcio_t* ipc, int64 offset, int whence)
{
  /* the current absolute byte count position in the ring buffer */
  uint64 current = 0;
  /* the absolute value of the offset */
  uint64 abs_offset = 0;
  /* space left in the current buffer */
  uint64 space = 0;
  /* end of current buffer flag */
  int eobuf = 0;

  uint64 bufsz = ipcbuf_get_bufsz (&(ipc->buf));
  uint64 nbuf = ipcbuf_get_read_count (&(ipc->buf));
  if (nbuf > 0)
    nbuf -= 1;

  current = bufsz * nbuf + ipc->bytes;

  if (whence == SEEK_SET)
    offset -= current;

  if (offset < 0) {
    /* can only go back to the beginning of the current buffer ... */
    abs_offset = (uint64) -offset;
    if (abs_offset > ipc->bytes) {
      fprintf (stderr, "ipcio_seek: "UI64" > max backwards "UI64"\n",
	       abs_offset, ipc->bytes);
      return -1;
    }
    ipc->bytes -= abs_offset;
  }

  else {
    /* ... but can seek forward until end of data */

    while (offset && ! (ipcbuf_eod(&(ipc->buf)) && eobuf)) {

      if (!ipc->curbuf || eobuf) {
	ipc->curbuf = ipcbuf_get_next_read (&(ipc->buf), &(ipc->curbufsz));
	if (!ipc->curbuf) {
	  fprintf (stderr, "ipcio_seek: error ipcbuf_next_read\n");
	  return -1;
	}
	ipc->bytes = 0;
	eobuf = 0;
      }

      space = ipc->curbufsz - ipc->bytes;
      if (space > offset)
	space = offset;

      if (space > 0) {
	ipc->bytes += space;
	offset -= space;
      }
      else
	eobuf = 1;
    }

  }

  bufsz = ipcbuf_get_bufsz (&(ipc->buf));
  nbuf = ipcbuf_get_read_count (&(ipc->buf));
  if (nbuf > 0)
    nbuf -= 1;

  return bufsz * nbuf + ipc->bytes;
}

