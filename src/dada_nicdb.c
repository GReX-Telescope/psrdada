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

void usage()
{
  fprintf (stdout,
	   "dada_nicdb [options]\n"
	   " -p   port on which to listen\n"
	   " -d   run as daemon\n");
}

/*! Pointer to the function that transfers data to/from the target */
int64_t sock_recv_function (dada_client_t* client, 
			    void* data, uint64_t data_size)
{
  /* this function will call select to see if there is any out-of-band
     data awaiting reception.  if so, this marks a premature end to
     the expected transfer_bytes length.  read the new header, and
     update the transfer_bytes attribute. if not, then recv the data.  */
  return 0;
}

/*! Function that closes the data file */
int sock_close_function (dada_client_t* client, uint64_t bytes_written)
{
  /* don't close the socket, maybe send the header back as a handshake */
  return 0;
}

/*! Function that opens the data transfer target */
int sock_open_function (dada_client_t* client)
{
  /* wait for incoming data on the socket.  recv peek the header and
     set the header_size attribute (ensuring that it is less than the
     header_size ???) */
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
