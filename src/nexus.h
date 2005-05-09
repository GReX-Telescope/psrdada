#ifndef __NEXUS_H
#define __NEXUS_H

/* ************************************************************************

   nexus_t - a struct and associated routines for creation and
   management of a dada network

   ************************************************************************ */

#include <stdio.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    /*! The name of the host on which node is running */
    char* host;

    /*! The port on which node is listening */
    int port;

    /*! The node identifier */
    int id;

    /*! The I/O stream to the node */
    FILE* to;

    /*! The I/O stream from the node */
    FILE* from;

  } node_t;

  /*! Create a new node */
  node_t* node_create ();

  /*! For use by derived classes during construction */
  void node_init (node_t* node);

  typedef struct nexus_struct {

    /*! The default port on which node is listening */
    int node_port;

    /*! The polling interval for connecting with nodes */
    unsigned polling_interval;

    /*! The nodes */
    void** nodes;

    /*! The number of nodes */
    unsigned nnode;

    /*! Pointer to function that creates new nodes */
    node_t* (*node_create) ();

    /*! Pointer to function that parses configuration */
    int (*nexus_parse) (struct nexus_struct* n, const char* buffer);

    /* for multi-threaded use of the nexus */
    pthread_mutex_t mutex;
    
  } nexus_t;

  /*! Create a new nexus */
  nexus_t* nexus_create ();

  /*! Destroy a nexus */
  int nexus_destroy (nexus_t* nexus);

  /*! Read the nexus configuration from the specified filename */
  int nexus_configure (nexus_t* nexus, const char* filename);

  /*! Add a node to the nexus */
  int nexus_add (nexus_t* nexus, int id, char* host_name);

  /*! Send a command to all selected nodes */
  int nexus_send (nexus_t* nexus, char* command);

  /*! Send a command to the selected node */
  int nexus_send_node (nexus_t* nexus, unsigned inode, char* command);

  /*! Get the number of nodes in the nexus */
  unsigned nexus_get_nnode (nexus_t* nexus);

  /*! For use by derived classes during construction */
  void nexus_init (nexus_t* nexus);

  /*! For use by derived classes during configuration */
  int nexus_parse (nexus_t* n, const char* buffer);

#ifdef __cplusplus
	   }
#endif

#endif
