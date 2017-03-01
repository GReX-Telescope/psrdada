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

  char *        buf;           // the socket buffer
  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received
  uint64_t      prev_seq;      // previous seq_no
  int64_t       seq_offset;    // sequence offset to first input
  int64_t       block_count;    // sequence offset to first input

} mopsr_sock_t;

typedef struct {

  // sequence number of packet
  uint64_t seq_no;

  // 16-bit flags for 16 antenna MGT locks (instaneous)
  uint16_t mgt_locks_long;

  // 16-bit flags for long term lock
  uint16_t mgt_locks;

  // first channel offset
  unsigned int start_chan;

  unsigned int nbit;

  unsigned int nant;

  unsigned int nchan;

  unsigned int nframe;

  // nb - not really used except in flow through mode
  unsigned int nsamp;

  unsigned int ant_id;
  unsigned int ant_id2;

  unsigned char mf_lock2;
  unsigned char mf_lock;

} mopsr_hdr_t;

mopsr_sock_t * mopsr_init_sock ();
void mopsr_reset_sock (mopsr_sock_t* b);
void mopsr_free_sock(mopsr_sock_t* b);

int mopsr_get_bit_from_16 (uint16_t n, unsigned bit);
void mopsr_decode_header (unsigned char * b, uint64_t *seq_no, unsigned int * ant_id);
void mopsr_decode (unsigned char * b, mopsr_hdr_t * hdr);
void mopsr_decode_v2 (unsigned char * b, mopsr_hdr_t * hdr);
void mopsr_decode_v1 (unsigned char * b, mopsr_hdr_t * hdr);
void mopsr_decode_seq (unsigned char * b, mopsr_hdr_t * hdr);
void mopsr_decode_seq_fast (unsigned char * b, mopsr_hdr_t * hdr);

void mopsr_encode_header (unsigned char * b, uint64_t seq_no);
void mopsr_encode (unsigned char * b, mopsr_hdr_t * hdr);
void mopsr_encode_v2 (unsigned char * b, mopsr_hdr_t * hdr);

unsigned int mopsr_get_ant_number (unsigned int id, unsigned int index);
void mopsr_print_header(mopsr_hdr_t * hdr);

unsigned int mopsr_get_new_ant_number (unsigned int index);
unsigned int mopsr_get_new_ant_index (unsigned int number);


#define MOPSR_UDP_COUNTER_BYTES  8          // size of header/sequence number
#define MOPSR_UDP_DATASIZE_BYTES 2048       // obs bytes per packet
#define MOPSR_UDP_PAYLOAD_BYTES  2056       // counter + datasize
#define MOPSR_UDP_INTERFACE      "10.0.0.4" // default interface

#endif
