#include "monitor.h"

#include <stdlib.h>
#include <assert.h>

// #define _DEBUG 1

/*! Create a new monitor */
monitor_t* monitor_create ()
{
  monitor_t* monitor = (monitor_t*) malloc (sizeof(monitor_t));
  assert (monitor != 0);

  monitor->nexus = 0;
  monitor->log = 0;
  monitor->handle_message = 0;

  return monitor;
}

/*! Destroy a monitor */
int monitor_destroy (monitor_t* monitor)
{
  free (monitor);
  return 0;
}

void* monitor_thread (void* context)
{
  monitor_t* monitor = (monitor_t*) context;
  unsigned nnode = 0;
  unsigned inode = 0;
  node_t* node = 0;

  fd_set readset;
  int maxfd = 0;
  int fd = 0;

  unsigned buffer_size = 1024;
  char* buffer = malloc (buffer_size);
  assert (buffer != 0);

#ifdef _DEBUG
  fprintf (stderr, "monitor_thread start nexus=%p\n", monitor->nexus);
#endif

  while (monitor->nexus) {

    FD_ZERO (&readset);
    maxfd = 0;

    pthread_mutex_lock (&(monitor->nexus->mutex));
    nnode = monitor->nexus->nnode;

#ifdef _DEBUG
  fprintf (stderr, "monitor_thread %u nodes\n", nnode);
#endif

    for (inode = 0; inode < nnode; inode++) {
      node = monitor->nexus->nodes[inode];
#ifdef _DEBUG
      fprintf (stderr, "monitor_thread add %d\n", fileno(node->from));
#endif
      if (node->from) {
	fd = fileno(node->from);
	if (fd > maxfd)
	  maxfd = fd;
	FD_SET (fd, &readset);
      }
    }

    pthread_mutex_unlock (&(monitor->nexus->mutex));

    if (select (maxfd + 1, &readset, NULL, NULL, NULL) < 0) {
      perror ("monitor_thread: select");
      free (buffer);
      return 0;
    }

    pthread_mutex_lock (&(monitor->nexus->mutex));
    nnode = monitor->nexus->nnode;
    for (inode = 0; inode < nnode; inode++) {
      node = monitor->nexus->nodes[inode];
#ifdef _DEBUG
  fprintf (stderr, "monitor_thread check %d\n", fileno(node->from));
#endif
      if (FD_ISSET (fileno(node->from), &readset))
	break;
    }
    pthread_mutex_unlock (&(monitor->nexus->mutex));

    if (inode == nnode)
      fprintf (stderr, "monitor_thread: select returns, but no FD_ISSSET\n");
    else {
      fgets (buffer, buffer_size, node->from);

#ifdef _DEBUG
      fprintf (stderr, "%u: %s", inode, buffer);
#endif
      if (monitor->log)
	multilog (monitor->log, LOG_INFO, "%u: %s", inode, buffer);

      if (monitor->handle_message)
	monitor->handle_message (monitor->context, inode, buffer);

    }

  }
  
#ifdef _DEBUG
  fprintf (stderr, "monitor_thread exit\n");
#endif

  free (buffer);
  return 0;
}


/*! Start another thread to monitor messages from the nexus */
int monitor_launch (monitor_t* monitor)
{
  pthread_t tmp_thread;

  if (pthread_create (&tmp_thread, 0, monitor_thread, monitor) < 0) {
    perror ("monitor_launch: Error creating new thread");
    return -1;
  }

  /* thread cannot be joined; resources will be destroyed on exit */
  pthread_detach (tmp_thread);

  return 0;
}

