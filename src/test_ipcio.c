#include "ipcio.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

int main (int argc, char** argv)
{
  key_t key = 0x69;       /* some id number like a port number, i guess */
  int   arg;

  uint64_t nbufs = 4;
#ifdef _SC_PAGE_SIZE
  uint64_t bufsz = sysconf (_SC_PAGE_SIZE);
#else
  uint64_t bufsz = 1000000;
#endif

  char* read = NULL;
  char* write = NULL;

  FILE* fptr = 0;
  char* filename = 0;
  char perm[2];

  ipcio_t ringbuf = IPCIO_INIT;

  char* smbuf = 0;
  int smbufsz = 0;

  size_t bytesio = 0;

  int debug = 0;
  while ((arg = getopt(argc, argv, "db:n:k:r:w:")) != -1) {

    switch (arg)  {
    case 'h':
      fprintf (stderr, "test_ipcio -[r|w] filename\n");
      return 0;

    case 'd':
      debug = 1;
      break;

    case 'r':
      read = optarg;
      break;

    case 'w':
      write = optarg;
      break;

    case 'k':
      key = atoi (optarg);
      break;

    }
  }

  if ((read && write) || !(read || write)) {
    fprintf (stderr, "Please specify one of -r or -w\n");
    return -1;
  }

  if (read) {
    filename = read;
    strcpy (perm, "r");
  }
  else {
    filename = write;
    strcpy (perm, "w");
  }

  fptr = fopen (filename, perm);
  if (!fptr) {
    fprintf (stderr, "Could not open %s\n", filename);
    perror ("");
    return -1;
  }

  smbufsz = bufsz/3 + 7;
  smbuf = malloc (smbufsz);
  assert (smbuf != 0);

  fprintf (stderr, "small buf size = %d\n", smbufsz);

  if (read) {

    /* this process is reading from the file and creates the shared memory */
    fprintf (stderr, "Creating shared memory ring buffer."
	     " nbufs=%llu bufsz=%llu\n", nbufs, bufsz);

    if (ipcio_create (&ringbuf, key, nbufs, bufsz) < 0) {
      fprintf (stderr, "Error creating shared memory ring buffer\n");
      return -1;
    }

    ipcio_open (&ringbuf, 'w');

    /* read from the file into the buffer */
    while ( (bytesio = fread (smbuf, 1, smbufsz, fptr)) > 0)
      /* get the next buffer to be filled */
      bytesio = ipcio_write (&ringbuf, smbuf, bytesio);

    fprintf (stderr, "Flagging eod\n");
    ipcio_close (&ringbuf);

    sleep (1);

    fprintf (stderr, "Scheduling IPC resources for destruction\n");
    ipcio_destroy (&ringbuf);
  }

  else {

    fprintf (stderr, "Connecting to shared memory ring buffer\n");

    if (ipcio_connect (&ringbuf, key) < 0) {
      fprintf (stderr, "Error connecting to shared memory ring buffer\n");
      return -1;
    }

    ipcio_open (&ringbuf, 'R');

    while ( (int) (bytesio = ipcio_read (&ringbuf, smbuf, smbufsz)) > 0)
      fwrite (smbuf, bytesio, 1, fptr);

    fprintf (stderr, "Closing...\n");
    ipcio_close (&ringbuf);
  }

  fclose (fptr);

  return 0;
}
