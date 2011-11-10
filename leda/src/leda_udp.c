/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>

#include "leda_udp.h"

leda_sock_t * leda_init_sock()
{
  leda_sock_t * b = (leda_sock_t *) malloc(sizeof(leda_sock_t));

  assert(b != NULL);

  b->size = sizeof(char) * UDP_PAYLOAD;
  b->buffer = (char *) malloc(b->size);

  assert(b->buffer != NULL);

  b->fd = 0;
  b->have_packet = 0;

  return b;
}

void leda_free_sock(leda_sock_t* b)
{
  b->fd = 0;
  b->size = 0;
  b->have_packet = 0;
  free(b->buffer);
}

void leda_decode_header (unsigned char * b, uint64_t *seq_no, uint64_t * ch_id)
{
  uint64_t tmp = 0;
  unsigned i = 0;
  *seq_no = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[8 - i - 1];
    *seq_no |= (tmp << ((i & 7) << 3));
  }

  *ch_id = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    *ch_id |= (tmp << ((i & 7) << 3));
  }
}

void leda_encode_header (char *b, uint64_t seq_no, uint64_t ch_id)
{
  b[0] = (uint8_t) (seq_no>>56);
  b[1] = (uint8_t) (seq_no>>48);
  b[2] = (uint8_t) (seq_no>>40);
  b[3] = (uint8_t) (seq_no>>32);
  b[4] = (uint8_t) (seq_no>>24);
  b[5] = (uint8_t) (seq_no>>16);
  b[6] = (uint8_t) (seq_no>>8);
  b[7] = (uint8_t) (seq_no);

  b[8]  = (uint8_t) (ch_id>>56);
  b[9]  = (uint8_t) (ch_id>>48);
  b[10] = (uint8_t) (ch_id>>40);
  b[11] = (uint8_t) (ch_id>>32);
  b[12] = (uint8_t) (ch_id>>24);
  b[13] = (uint8_t) (ch_id>>16);
  b[14] = (uint8_t) (ch_id>>8);
  b[15] = (uint8_t) (ch_id);
}
