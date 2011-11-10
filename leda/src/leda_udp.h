/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __LEDA_UDP_H
#define __LEDA_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#include "dada_udp.h"
#include "leda_def.h"

/* socket buffer for receiving udp data */
typedef struct {

  int        fd;            // FD of the socket
  char     * buffer;        // the socket buffer
  size_t     size;          // size of the buffer
  unsigned   have_packet;   // is there a packet in the buffer
  size_t     got;           // amount of data received

} leda_sock_t;

leda_sock_t * leda_init_sock();

void leda_free_sock(leda_sock_t* b);

void leda_decode_header (unsigned char * b, uint64_t *seq_no, uint64_t * ch_id);

void leda_encode_header (char *b, uint64_t seq_no, uint64_t ch_id);

#endif

