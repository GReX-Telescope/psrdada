/***************************************************************************
 *        
 *    Copyright (C) 2012 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 *      
 ****************************************************************************/

#include "hispec_udp.h"

// Encode the counter into the first8 bytes of the buffer array
void encode_header (unsigned char * buffer, hispec_udp_header_t hdr)
{
  buffer[0] = hdr.version;
  buffer[1] = hdr.beam_id;
  buffer[2] = (uint8_t) (hdr.pkt_cnt>>24);
  buffer[3] = (uint8_t) (hdr.pkt_cnt>>16);
  buffer[4] = (uint8_t) (hdr.pkt_cnt>>8);
  buffer[5] = (uint8_t) (hdr.pkt_cnt);
  buffer[6] = hdr.diode_state;
  buffer[7] = hdr.freq_state;

}

/* Reads the first 8 bytes of the buffer array, interpreting it as
 * a 64 bit interger */
void decode_header(unsigned char * buffer, hispec_udp_header_t * hdr)
{
  // this is the correct head DECODE, but for testing, we are using
  // parspec which has just a single

  // decode the pkt counter
#ifdef TEST_MODE
  uint64_t tmp = 0;
  unsigned i = 0;
  hdr->pkt_cnt = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = buffer[8 - i - 1];
    hdr->pkt_cnt |= (tmp << ((i & 7) << 3));
  }
  hdr->version = 1;
  hdr->beam_id = 1;
  hdr->diode_state = 1;
  hdr->freq_state = 1;
#else

  hdr->version     = buffer[0];
  hdr->beam_id     = buffer[1];
  hdr->diode_state = buffer[6];
  hdr->freq_state  = buffer[7];

  uint32_t tmp = 0;
  unsigned i = 0;
  hdr->pkt_cnt = UINT32_C (0);
  for (i = 0; i < 4; i++ )
  {
    tmp = UINT32_C (0);
    tmp = buffer[6 - i - 1];
    hdr->pkt_cnt |= (tmp << ((i & 7) << 3));
  }
#endif
}

