/***************************************************************************
 *        
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *      
 ****************************************************************************/

#ifndef __HISPEC_UDP_H
#define __HISPEC_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include "dada_udp.h"
#include "dada_generator.h"

#define TEST_MODE 0

#ifdef TEST_MODE

#define HISPEC_UDP_COUNTER_BYTES  8     // size of header/sequence number
#define HISPEC_UDP_DATASIZE_BYTES 2048  // obs bytes per packet
#define HISPEC_UDP_PAYLOAD_BYTES  2056  // counter + datasize

#else

#define HISPEC_UDP_COUNTER_BYTES  8          // size of header/sequence number
#define HISPEC_UDP_DATASIZE_BYTES 4096       // obs bytes per packet
#define HISPEC_UDP_PAYLOAD_BYTES  4104       // counter + datasize
#define HISPEC_UDP_INTERFACE      "10.0.0.4" // default interface

#endif

typedef struct
{
  unsigned char version;
  unsigned char beam_id;
#ifdef TEST_MODE 
  uint64_t pkt_cnt;
#else
  uint32_t pkt_cnt;
#endif
  unsigned char diode_state;
  unsigned char freq_state;
} hispec_udp_header_t;

void hispec_decode_header (unsigned char * buffer, hispec_udp_header_t * header);
void hispec_encode_header (unsigned char * buffer, hispec_udp_header_t header);

#endif /* __HISPEC_UDP_H */
