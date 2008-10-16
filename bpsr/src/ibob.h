/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source$
   $Revision$
   $Date$
   $Author$ */

#ifndef __DADA_IBOB_H
#define __DADA_IBOB_H

#include <time.h>
#include <sys/types.h>

#define IBOB_VLAN_BASE "169.254.128."
#define IBOB_PORT 23

/* ************************************************************************

   ibob_t - a struct and associated routines for creating and managing,
   as well as reading and writing to and from an iBoB

   ************************************************************************ */

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    //! Host name of iBoB interface
    char* host;

    //! Port number of iBob interface
    int port;

    //! File descriptor of open socket; -1 otherwise
    int fd;

    //! Buffer used for ibob communication
    char* buffer;

    //! Size of buffer reserved for ibob communication
    unsigned buffer_size;

    //! Emulate telnet flag must be set before call to ibob_open
    char emulate_telnet;

  } ibob_t;

  /*! allocate and initialize a new ibob_t struct */
  ibob_t* ibob_construct ();

  /*! free all resources reserved for ibob communications */
  int ibob_destroy (ibob_t* bob);

  /*! set the host and port number of the ibob */
  int ibob_set_host (ibob_t* bob, const char* host, int port);

  /*! set the ibob number (use default IP base and port) */
  int ibob_set_number (ibob_t* bob, unsigned number);

  /*! open socket connection with currently configured ibob */
  int ibob_open (ibob_t* bob);

  /*! close socket connection with currently open ibob */
  int ibob_close (ibob_t* bob);

  /*! return true if already open */
  int ibob_is_open (ibob_t* bob);

  /*! return true if iBoB is alive */
  int ibob_ping (ibob_t* bob);

  /*! reset packet counter on next UTC second, returned */
  time_t ibob_arm (ibob_t* bob);

  /*! configure the ibob */
  int ibob_configure (ibob_t* bob, const char* mac_address);

  /*! send message to ibob */
  ssize_t ibob_send (ibob_t* bob, const char* message);

  /*! receive response from ibob, up to bytes characters */
  ssize_t ibob_recv (ibob_t* bob, char* response, size_t bytes);

  /*! write bytes to ibob */
  ssize_t ibob_send_async (ibob_t* bob, const char* message);

  /*! if emulating telnet, receive echoed characters */
  int ibob_recv_echo (ibob_t* bob, size_t bytes);

#ifdef __cplusplus
	   }
#endif

#endif
