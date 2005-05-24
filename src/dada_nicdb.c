#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "sock.h"
#include "ascii_header.h"
#include "daemon.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/socket.h>

void usage()
{
  fprintf (stdout,
	   "dada_nicdb [options]\n"
	   " -p   port on which to listen\n"
	   " -d   run as daemon\n");
}

int64_t sock_recv (int fd, char* buffer, uint64_t size, int flags)
{
  int64_t received = 0;
  uint64_t total_received = 0;

  fd_set* readset = 0;
  fd_set* excpset = 0;

  fd_set set;
  FD_ZERO (&set);
  FD_SET (fd,&set);

fprintf (stderr, "sock_recv (%d, %p, %lu, %d)\n", fd, buffer, 
         (unsigned long) size, flags);

  if (flags & MSG_OOB)  {
fprintf (stderr, "OOB\n");
    excpset = &set;
}
  else  {
fprintf (stderr, "NORMAL\n");
    readset = &set;
}

  while (size) {

#if 0
    if (select (fd+1, readset, NULL, excpset, NULL) < 0) {
      perror ("sock_recv select");
      return -1;
    }

    if (FD_ISSET (fd, &set)) {
#endif

      received = recv (fd, buffer, size,
		       MSG_NOSIGNAL | MSG_WAITALL | flags);
fprintf (stderr, "received=%lu\n", (unsigned long) received);

#if 0
}
    else  {
fprintf (stderr, "FD_ISNOTSET!\n");
      received = 0;
}
#endif

    if (received < 0) {
      perror ("sock_recv recv");
      return -1;
    }
    else if (!received)
      break;

    size -= received;
    buffer += received;
    total_received += received;
  }

  return total_received;
}


/*! Pointer to the function that transfers data to/from the target */
int64_t sock_recv_function (dada_client_t* client, 
			    void* data, uint64_t data_size)
{
  int64_t received = sock_recv (client->fd, data, data_size, 0);

  if (received < 0)
    return -1;

  if (received < data_size) {

    /* The transmission was cut short; most likely due to out-of-band
       data awaiting reception.  If this is the case, then it is a message
       from dbnic with an updated header; recv this header and parse
       the new TRANSFER_LENGTH attribute from it. */

    if (sock_recv (client->fd, client->header, client->header_size, MSG_OOB)
	< client->header_size) {
      multilog (client->log, LOG_ERR, 
		"Could not recv out-of-band Header: %s\n", strerror(errno));
      return -1;
    }

    /* Get the transfer size */
    if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIu64,
			  &(client->transfer_bytes)) != 1)
    {
      multilog (client->log, LOG_ERR, "Header with no TRANSFER_SIZE\n");
      return -1;
    }

  }

  return received;
}

/*! Function that closes the data file */
int sock_close_function (dada_client_t* client, uint64_t bytes_written)
{
  /* don't close the socket; just send the header back as a handshake */

  if (send (client->fd, client->header, client->header_size, 
	    MSG_NOSIGNAL | MSG_WAITALL) < client->header_size) {
    multilog (client->log, LOG_ERR, 
	      "Could not send acknowledgment Header: %s\n", strerror(errno));
    return -1;
  }

  return 0;
}

/*! Function that opens the data transfer target */
int sock_open_function (dada_client_t* client)
{
  unsigned hdr_size = 0;
  int ret = 0;

  ret = sock_recv (client->fd, client->header, client->header_size, MSG_OOB);

  if (ret < client->header_size) {
    multilog (client->log, LOG_ERR, 
	      "recv %d out of %d peek at the Header: %s\n", 
	      ret, client->header_size, strerror(errno));
    return -1;
  }

#ifdef _DEBUG
fprintf (stderr, "HEADER START\n%s\nHEADER END\n", client->header);
#endif

  /* Get the transfer size */
  if (ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIu64,
			&(client->transfer_bytes)) != 1)  {
    multilog (client->log, LOG_ERR, "Header with no TRANSFER_SIZE\n");
    return -1;
  }

  /* Get the header size */
  if (ascii_header_get (client->header, "HDR_SIZE", "%"PRIu64, &hdr_size) != 1)
  {
    multilog (client->log, LOG_ERR, "Header with no HDR_SIZE\n");
    return -1;
  }

  /* Ensure that the incoming header fits in the client header buffer */
  if (hdr_size > client->header_size) {
    multilog (client->log, LOG_ERR, "HDR_SIZE=%u > Block size=%"PRIu64"\n",
	      hdr_size, client->header_size);
    return -1;

  }

  client->header_size = hdr_size;

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Secondary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* port on which to listen for incoming connections */
  int port = DADA_DEFAULT_NICDB_PORT;

  /* file descriptor of passive socket */
  int listen_fd = 0;

  /* file descriptor of active socket */
  int comm_fd = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"dp:v")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;

    case 'p':
      port = atoi (optarg);
      break;

    case 'v':
      verbose=1;
      break;
      
    default:
      usage ();
      return 0;
      
    }

  log = multilog_open ("dada_nicdb", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, DADA_DEFAULT_NICDB_LOG);
  }
  else
    multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = sock_open_function;
  client->io_function    = sock_recv_function;
  client->close_function = sock_close_function;
  client->direction      = dada_client_writer;

  signal (SIGPIPE, SIG_IGN);

  listen_fd = sock_create (&port);
  if (listen_fd < 0)  {
    multilog (log, LOG_ERR, "Error creating socket: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  while (!quit) {

    comm_fd = sock_accept (listen_fd);
    if (comm_fd < 0)  {
      multilog (log, LOG_ERR, "Error accepting connection: %s\n",
		strerror(errno));
      return -1;
    }

    client->fd = comm_fd;

    if (dada_client_write (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    multilog (log, LOG_INFO, "Closing socket connection\n");

    if (sock_close (comm_fd) < 0) {
      multilog (log, LOG_ERR, "Error accepting connection: %s\n",
		strerror(errno));
      return -1;
    }

  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
