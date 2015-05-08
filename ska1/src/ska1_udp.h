/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __SKA1_UDP_H
#define __SKA1_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#include "dada_udp.h"
#include "ska1_def.h"

#define SKA1_UDPDB_BUF_CLEAR = 0
#define SKA1_UDPDB_BUF_FULL = 1

/* socket buffer for receiving udp data */
typedef struct {

  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  char *        buf;          // the socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received

} ska1_sock_t;

ska1_sock_t * ska1_init_sock ();

void ska1_free_sock(ska1_sock_t* b);

void ska1_decode_header (unsigned char * b, uint64_t *seq_no, uint16_t * ant_id);

void ska1_encode_header (char *b, uint64_t seq_no, uint16_t ant_id);

#endif

