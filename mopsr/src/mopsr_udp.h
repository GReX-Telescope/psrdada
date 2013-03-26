/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __MOPSR_UDP_H
#define __MOPSR_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#include "dada_udp.h"
#include "mopsr_def.h"

#define MOPSR_UDPDB_BUF_CLEAR = 0
#define MOPSR_UDPDB_BUF_FULL = 1

/* socket buffer for receiving udp data */
typedef struct {

  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  char *        buf;          // the socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received

} mopsr_sock_t;

mopsr_sock_t * mopsr_init_sock ();

void mopsr_free_sock(mopsr_sock_t* b);

void mopsr_decode_header (unsigned char * b, uint64_t *seq_no);

void mopsr_encode_header (unsigned char * b, uint64_t seq_no);


#define MOPSR_UDP_COUNTER_BYTES  8          // size of header/sequence number
#define MOPSR_UDP_DATASIZE_BYTES 2048       // obs bytes per packet
#define MOPSR_UDP_PAYLOAD_BYTES  2056       // counter + datasize
#define MOPSR_UDP_INTERFACE      "10.0.0.4" // default interface

#endif
