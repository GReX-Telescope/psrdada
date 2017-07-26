/***************************************************************************
 *  
 *    Copyright (C) 2016 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __SNAPSR_UDP_H
#define __SNAPSR_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#include "dada_udp.h"

#define SNAPSR_DEFAULT_UDP_PORT    50000
#define SNAPSR_UDP_HEADER_BYTES    8         // size of header/sequence number
#define SNAPSR_UDP_DATA_BYTES      6144      // obs bytes per packet
#define SNAPSR_UDP_PAYLOAD_BYTES   6152      // counter + datasize

#define SNAPSR_NSUBBAND           10
#define SNAPSR_NCHAN_PER_SUBBAND  32
#define SNAPSR_NANT_PER_PACKET    12
#define SNAPSR_NFRAME_PER_PACKET  8
#define SNAPSR_BYTES_PER_SAMPLE   SNAPSR_UDP_DATA_BYTES / SNAPSR_NFRAME_PER_PACKET

#define SNAPSR_UDPDB_BUF_CLEAR = 0
#define SNAPSR_UDPDB_BUF_FULL = 1

// header within a SNAP udp packet
typedef struct {
  unsigned subband_id;
  unsigned snap_id;
  uint64_t seq_no;
  unsigned nframe;                  // always SNAPSR_NFRAME_PER_PACKET
  unsigned nant;                    // always SNAPSR_NANT_PER_PACKET
  unsigned nchan;                   // always SNAPSR_NCHAN_PER_SUBBAND
} snapsr_udp_hdr_t;


/* socket buffer for receiving udp data */
typedef struct {

  char *             buf;           // the socket buffer
  int                fd;            // FD of the socket
  size_t             bufsz;         // size of socket buffer
  int                have_packet;   // 
  size_t             got;           // amount of data received
  uint64_t           prev_seq;      // previous seq_no
  snapsr_udp_hdr_t   hdr;

} snapsr_sock_t;

typedef struct {

  unsigned nsubband;

  unsigned * snapsr_subband_t;

  // sequence number of packet
  uint64_t seq_no;

  unsigned start_chan;

  unsigned end_chan;

  unsigned nbit;

  unsigned nant;

  unsigned nchan;

} snapsr_hdr_t;

snapsr_sock_t * snapsr_init_sock ();

void snapsr_reset_sock (snapsr_sock_t* b);
void snapsr_free_sock(snapsr_sock_t* b);

void snapsr_decode (unsigned char * b, snapsr_udp_hdr_t * hdr);
void inline snapsr_decode_seq (unsigned char * b, snapsr_udp_hdr_t * hdr);

void snapsr_encode_header (unsigned char * b, uint64_t seq_no);
void snapsr_encode (unsigned char * b, snapsr_udp_hdr_t * hdr);

void snapsr_print_header(snapsr_udp_hdr_t * hdr);

#endif
