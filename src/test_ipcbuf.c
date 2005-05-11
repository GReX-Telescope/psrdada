#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "ipcbuf.h"

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

  uint64_t offset = 0;

  char* read = NULL;
  char* write = NULL;

  FILE* fptr = 0;
  char* filename = 0;
  char perm[2];

  ipcbuf_t ringbuf = IPCBUF_INIT;

  char* buf = 0;
  size_t bytesio = 0;

  int debug = 0;
  while ((arg = getopt(argc, argv, "db:n:o:k:r:w:")) != -1) {

    fprintf (stderr, "arg=%c\n", arg);

    switch (arg)  {

    case 'h':
      fprintf (stderr, "test_ipcbuf -[r|w] filename\n");
      return 0;

    case 'b':
      if (sscanf (optarg, "%"PRIu64"", &bufsz) < 1) {
	fprintf (stderr, "test_ipcbuf could not parse -b %s", optarg);
	return -1;
      }
      break;

    case 'd':
      debug = 1;
      break;

    case 'n':
      if (sscanf (optarg, "%"PRIu64"", &nbufs) < 1) {
	fprintf (stderr, "test_ipcbuf could not parse -n %s", optarg);
	return -1;
      }
      break;

    case 'o':
      if (sscanf (optarg, "%"PRIu64"", &offset) < 1) {
	fprintf (stderr, "test_ipcbuf could not parse -o %s", optarg);
	return -1;
      }
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

  fprintf (stderr, "arg=%d\n", arg);

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

  if (read) {
    
    /* this process is reading from the file and creates the shared memory */
    fprintf (stderr, "Creating shared memory ring buffer."
	     " nbufs=%"PRIu64" bufsz=%"PRIu64"\n", nbufs, bufsz);

    if (ipcbuf_create (&ringbuf, key, nbufs, bufsz) < 0) {
      fprintf (stderr, "Error creating shared memory ring buffer\n");
      return -1;
    }

    /* read from file, write to shm */
    if (ipcbuf_lock_write (&ringbuf) < 0) {
      fprintf (stderr, "Error ipcbuf_lock_write\n");
      return -1;
    }

    if (offset)
      ipcbuf_disable_sod (&ringbuf);

    do {

      /* get the next buffer to be filled */
      buf = ipcbuf_get_next_write (&ringbuf);
      if (!buf) {
	fprintf (stderr, "error ipcbuf_get_next_write\n");
	return -1;
      }

      /* read from the file into the buffer */
      bytesio = fread (buf, 1, bufsz, fptr);

      fprintf (stderr, "buffer:%"PRIu64" %d bytes. buf[0]='%c'\n",
	       ipcbuf_get_write_count(&ringbuf), bytesio, buf[0]);

      if (ipcbuf_mark_filled (&ringbuf, bytesio) < 0) {
	fprintf (stderr, "error ipcbuf_mark_filled\n");
	return -1;
      }

      if (!ipcbuf_sod (&ringbuf)) {

	if (offset < bufsz)
	  ipcbuf_enable_sod (&ringbuf, 
			     ipcbuf_get_write_count(&ringbuf)-1, offset);
	else
	  offset -= bufsz;

      }

      fprintf (stderr, "End of data: %d %d\n",
	       ringbuf.sync->eod, ipcbuf_eod (&ringbuf));

      if (debug) {
	if (ipcbuf_get_write_count(&ringbuf) == ipcbuf_get_nbufs(&ringbuf)) {
	  fprintf (stderr, "Pause\n");
	  getchar();
	}
      }

    } while (bytesio == bufsz);

    if (debug) {
      fprintf (stderr, "Pause before reset\n");
      getchar();
    }

    fprintf (stderr, "Waiting for read to finish\n");
    if (ipcbuf_reset (&ringbuf) < 0)
      fprintf (stderr, "Error ipcbuf_reset\n");

    if (debug) {
      fprintf (stderr, "Pause before destroy\n");
      getchar();
    }

    fprintf (stderr, "Scheduling IPC resources for destruction\n");
    ipcbuf_destroy (&ringbuf);
  }

  else {

    fprintf (stderr, "Connecting to shared memory ring buffer\n");
    if (ipcbuf_connect (&ringbuf, key) < 0) {
      fprintf (stderr, "Error connecting to shared memory ring buffer\n");
      return -1;
    }

    /* read from  shm */
    if (ipcbuf_lock_read (&ringbuf) < 0) {
      fprintf (stderr, "Error ipcbuf_lock_read\n");
      return -1;
    }

    nbufs = ipcbuf_get_nbufs(&ringbuf);
    bufsz = ipcbuf_get_bufsz(&ringbuf);

    fprintf (stderr, " nbufs=%"PRIu64" bufsz=%"PRIu64"\n", nbufs, bufsz);

    while ( ! ipcbuf_eod (&ringbuf) ) { 

      buf = ipcbuf_get_next_read (&ringbuf, &bufsz);

      if (!buf) {
	fprintf (stderr, "Error ipcbuf_get_next_read\n");
	return -1;
      }

      fprintf (stderr, "buffer:%"PRIu64" %"PRIu64" bytes. buf[0]='%c'\n",
	       ipcbuf_get_read_count (&ringbuf), bufsz, buf[0]);
      
      bytesio = fwrite (buf, bufsz, 1, fptr);

      if (ipcbuf_mark_cleared (&ringbuf) < 0) {
	fprintf (stderr, "error ipcbuf_mark_cleared\n");
	return -1;
      }
	
      fprintf (stderr, "End of data: %d %d\n",
	       ringbuf.sync->eod, ipcbuf_eod (&ringbuf));

    }

  }

  fclose (fptr);

  return 0;
}
