#include "nexus.h"
#include "ascii_header.h"
#include "futils.h"
#include "sock.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

void node_init (node_t* node)
{
  node -> host = 0;
  node -> port = 0;
  node -> id = -1;
  node -> to = 0;
  node -> from = 0;
}

/*! Create a new node */
node_t* node_create ()
{
  node_t* node = (node_t*) malloc (sizeof(node_t));
  node_init (node);
  return node;
}

// #define _DEBUG 1
void nexus_init (nexus_t* nexus)
{
  /* default command port */
  nexus -> node_port = 0xdada;

  /* default polling interval */
  nexus -> polling_interval = 10;

  /* no nodes */
  nexus -> nodes = 0;
  nexus -> nnode = 0;

  /* default creator */
  nexus -> node_create = &node_create;

  pthread_mutex_init(&(nexus->mutex), NULL);
}

/*! Create a new nexus */
nexus_t* nexus_create ()
{
  nexus_t* nexus = (nexus_t*) malloc (sizeof(nexus_t));
  nexus_init (nexus);
  return nexus;
}


/*! Destroy a nexus */
int nexus_destroy (nexus_t* nexus)
{
  node_t* node = 0;
  unsigned inode = 0;

  if (!nexus)
    return -1;

  for (inode = 0; inode < nexus->nnode; inode++) {
    node = (node_t*) nexus->nodes[inode];
    if (node->host)
      free (node->host);
    if (node->to)
      fclose (node->to);
    if (node->from)
      fclose (node->from);
    free (node);
  }

  if (nexus->nodes)
    free (nexus->nodes);

  pthread_mutex_destroy (&(nexus->mutex));

  return 0;
}

unsigned nexus_get_nnode (nexus_t* nexus)
{
  return nexus->nnode;
}

int nexus_parse (nexus_t* n, const char* buffer)
{
  char node_name [16];
  char host_name [64];

  unsigned inode, nnode = 0;

  if (ascii_header_get (buffer, "COM_PORT", "%d", &(n->node_port)) < 0) {
    fprintf (stderr, "nexus_parse: using default COM_PORT\n");
    n->node_port = 0xdada;
  }

  if (ascii_header_get (buffer, "COM_POLL", "%d", &(n->polling_interval)) <0) {
    fprintf (stderr, "nexus_parse: using default COM_POLL\n");
    n->polling_interval = 10;
  }

  if (ascii_header_get (buffer, "NUM_NODE", "%u", &nnode) < 0)
    nnode = 0;

  if (!nnode)
    fprintf (stderr, "nexus_parse: WARNING no Nodes!\n");

  for (inode=0; inode < nnode; inode++) {

    sprintf (node_name, "NODE_%u", inode);

    if (ascii_header_get (buffer, node_name, "%s", host_name) < 0)
      fprintf (stderr, "nexus_parse: WARNING no host name for %s\n", 
	       node_name);
    else if (nexus_add (n, inode, host_name) < 0) 
      fprintf (stderr, "nexus_parse: Error adding %s %s\n", 
	       node_name, host_name);
    
  }

  return 0;
}


int nexus_configure (nexus_t* nexus, const char* filename)
{
  int error = 0;
  char* buffer = 0;
  long fsize = filesize (filename);

  buffer = (char *) malloc (fsize + 1);

#ifdef _DEBUG
  fprintf (stderr, "nexus_configure filename='%s'\n", filename);
#endif

  if (fileread (filename, buffer, fsize+1) < 0)
    return -1;

#ifdef _DEBUG
  fprintf (stderr, "nexus_configure call nexus_parse\n");
#endif

  error = nexus_parse (nexus, buffer);
  free (buffer);

  return error;
}

typedef struct {
    nexus_t* nexus;
    int id;
} node_open_t;

void* node_open_thread (void* context)
{
  node_open_t* request = (node_open_t*) context;
  nexus_t* nexus = request->nexus;
  node_t* node = 0;
  int id = request->id;

  /* Name of the host on which the node is running */
  char* host_name = 0;

  /* Port on which the NODE is listening */
  int port = nexus->node_port;

  int fd = -1;
  FILE* to = 0;
  FILE* from = 0;

  unsigned inode = 0;

  free (context);

  pthread_mutex_lock (&(nexus->mutex));
  for (inode = 0; inode < nexus->nnode; inode++) {
    node = (node_t*) nexus->nodes[inode];
    if (id == node->id)
      host_name = strdup (node->host);
  }
  pthread_mutex_unlock (&(nexus->mutex));

  if (!host_name) {
    fprintf (stderr, "node_open_thread: no NODE with id=%d\n", id);
    return 0;
  }

  while (fd < 0) {

#ifdef _DEBUG
  fprintf (stderr, "nexus_open_thread: call sock_open (%s,%d)\n",
	   host_name, port);
#endif

    fd = sock_open (host_name, port);

    if (fd < 0)  {
      fprintf (stderr, "open_thread: Error connecting with %s on %d\n"
	       "\tsleeping %u seconds before retrying\n",
	       host_name, port, nexus->polling_interval);
      sleep (nexus->polling_interval);
    }

  }

  free (host_name);

  from = fdopen (fd, "r");
  if (!from)  {
    perror ("node_open_thread: Error creating input stream");
    return 0;
  }

  to = fdopen (fd, "w");
  if (!to)  {
    perror ("node_open_thread: Error creating output stream");
    return 0;
  }

  /* do not buffer the output */
  setbuf (to, 0);

  pthread_mutex_lock (&(nexus->mutex));
  for (inode = 0; inode < nexus->nnode; inode++) {
    node = (node_t*) nexus->nodes[inode];
    if (id == node->id) {
      node->to = to;
      node->from = from;
      to = from = 0;
    }
  }
  pthread_mutex_unlock (&(nexus->mutex));

  if (to || from)
    fprintf (stderr, "node_open_thread: no NODE with id=%d\n", id);

#ifdef _DEBUG
  fprintf (stderr, "nexus_open_thread: return\n");
#endif

  return 0;
}

int nexus_connect (nexus_t* nexus, unsigned inode)
{
  node_open_t* request = 0;
  pthread_t tmp_thread;
  node_t* node = 0;

  if (!nexus)
    return -1;

  if (inode >= nexus->nnode) {
    fprintf (stderr, "nexus_connect: inode=%d >= nnode=%d",
	     inode, nexus->nnode);
    return -1;
  }
  
  node = (node_t*) nexus->nodes[inode];

  /* start a new thread to open a socket connection with the host */
  request = (node_open_t*) malloc (sizeof(node_open_t));
  request->nexus = nexus;
  request->id = node->id;

#ifdef _DEBUG
  fprintf (stderr, "nexus_connect: pthread_create node_open_thread\n");
#endif

  if (pthread_create (&tmp_thread, 0, node_open_thread, request) < 0) {
    perror ("nexus_add: Error creating new thread");
    return -1;
  }

  /* thread cannot be joined; resources will be deallocated on exit */
  pthread_detach (tmp_thread);

  return 0;
}

/*! Add a node to the nexus */
int nexus_add (nexus_t* nexus, int id, char* host_name)
{
  node_t* node = 0;
  unsigned inode = 0;

  /* ensure that each node in nexus has a unique id */
  for (inode = 0; inode < nexus->nnode; inode++) {
    node = (node_t*) nexus->nodes[inode];
    if (id == node->id) {
      fprintf (stderr, "nexus_add: id=%d equals that of %s\n",
	       id, node->host);
      return -1;
    }
  }

  pthread_mutex_lock (&(nexus->mutex));

  nexus->nodes = realloc (nexus->nodes, (nexus->nnode+1)*sizeof(void*));
  node = nexus->node_create();
  
  node->host = strdup (host_name);
  node->id = id;
  node->to = 0;
  node->from = 0;

  nexus->nodes[nexus->nnode] = node;
  nexus->nnode ++;

  pthread_mutex_unlock (&(nexus->mutex));

  return nexus_connect (nexus, nexus->nnode-1);
}



int nexus_restart (nexus_t* nexus, unsigned inode)
{
  node_t* node = (node_t*) nexus->nodes[inode];

  if (node->to)
    fclose (node->to);
  node->to = 0;

  if (node->from)
    fclose (node->from);
  node->from = 0;

  return nexus_connect (nexus, inode);
}

/*! Send a command to the specified node */
int nexus_send_node (nexus_t* nexus, unsigned inode, char* command)
{
  FILE* to = 0;
  node_t* node = 0;

  if (!nexus)
    return -1;

  if (inode >= nexus->nnode) {
    fprintf (stderr, "nexus_send_node: node %d >= nnode=%d",
	     inode, nexus->nnode);
    return -1;
  }

  node = (node_t*) nexus->nodes[inode];
  to = node->to;

  if (!to) {
#ifdef _DEBUG
    fprintf (stderr, "nexus_send_node: node %d not online\n", inode);
#endif
    return -1;
  }

  if (ferror (to)) {
#ifdef _DEBUG
    fprintf (stderr, "nexus_send_node: error on node %d", inode);
#endif
    if (nexus_restart (nexus, inode) < 0)
      fprintf (stderr, "nexus_send_node: error restart node %d\n", inode);
    return -1;
  }

#ifdef _DEBUG
  fprintf (stderr, "nexus_send: sending to node %d\n", inode);
#endif

  if (fprintf (node->to, "%s\n", command) < 0) {
    fprintf (stderr, "nexus_send_node: node %d '%s'\n\t%s",
	     inode, command, strerror(errno));
    return -1;
  }

  return 0;
}

/*! Send a command to the specified node */
int nexus_recv_node (nexus_t* nexus, unsigned inode)
{
  FILE* from = 0;
  node_t* node = 0;

  if (!nexus)
    return -1;

  if (inode >= nexus->nnode) {
    fprintf (stderr, "nexus_send_node: node %d >= nnode=%d",
	     inode, nexus->nnode);
    return -1;
  }
    
  node = (node_t*) nexus->nodes[inode];
  from = node->from;

  if (!from) {
#ifdef _DEBUG
    fprintf (stderr, "nexus_send_node: node %d not online\n", inode);
#endif
    return -1;
  }

  if (ferror (from)) {
#ifdef _DEBUG
    fprintf (stderr, "nexus_send_node: error on node %d", inode);
#endif
    if (nexus_restart (nexus, inode) < 0)
      fprintf (stderr, "nexus_send_node: error restart node %d\n", inode);
    return -1;
  }

  //#ifdef _DEBUG
  fprintf (stderr, "nexus_send: receiving from node %d not implemented\n", inode);
  //#endif

  

  return 0;
}

/*! Send a command to all selected nodes */
int nexus_send (nexus_t* nexus, char* command)
{
  unsigned inode = 0;
  int status = 0;

  pthread_mutex_lock (&(nexus->mutex));

  for (inode = 0; inode < nexus->nnode; inode++) {
    if (nexus_send_node (nexus, inode, command) < 0) {
      fprintf (stderr, "nexus_send error");
    }
  }

  for (inode = 0; inode < nexus->nnode; inode++) {
    if (nexus_recv_node (nexus, inode) < 0) {
      fprintf (stderr, "nexus_send error inode=%d\n", inode);
      status = -1;
    }
  }

  pthread_mutex_unlock (&(nexus->mutex));

  return status;
}
