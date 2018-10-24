/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __COMAP_UDP_H
#define __COMAP_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#include "dada_udp.h"
#include "comap_def.h"

#define COMAP_UDPDB_BUF_CLEAR = 0
#define COMAP_UDPDB_BUF_FULL = 1

/* socket buffer for receiving udp data */
typedef struct {

  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  char *        buf;          // the socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received

} comap_sock_t;

comap_sock_t * comap_init_sock ();

void comap_free_sock(comap_sock_t* b);

void comap_decode_header (unsigned char * b, uint64_t *seq_no, uint8_t * ant_id);

void comap_encode_header (char *b, uint64_t seq_no, uint8_t ant_id);

#endif

