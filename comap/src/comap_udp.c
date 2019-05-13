/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>

#include "comap_udp.h"

/*
 * create a socket with the specified number of buffers
 */
comap_sock_t * comap_init_sock ()
{
  comap_sock_t * b = (comap_sock_t *) malloc(sizeof(comap_sock_t));
  assert(b != NULL);

  b->bufsz = sizeof(char) * UDP_PAYLOAD;

	b->buf = (char *) malloc (b->bufsz);
  assert(b->buf != NULL);

  b->have_packet = 0;
  b->fd = 0;

  return b;
}

void comap_free_sock(comap_sock_t* b)
{
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
}

void comap_decode_header (unsigned char * b, uint64_t *seq_no, uint8_t * ant_id)
{
  uint64_t tmp = 0;
  uint8_t rid = 0;
  unsigned i = 0;
  *seq_no = UINT64_C (0);
  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[8 - i - 1];
    *seq_no |= (tmp << ((i & 7) << 3));
  }

  rid = b[8];
  *ant_id = rid;
}

void comap_encode_header (char *b, uint64_t seq_no, uint8_t ant_id)
{
  b[0] = (uint8_t) (seq_no>>56);
  b[1] = (uint8_t) (seq_no>>48);
  b[2] = (uint8_t) (seq_no>>40);
  b[3] = (uint8_t) (seq_no>>32);
  b[4] = (uint8_t) (seq_no>>24);
  b[5] = (uint8_t) (seq_no>>16);
  b[6] = (uint8_t) (seq_no>>8);
  b[7] = (uint8_t) (seq_no);

  b[8]  = (uint8_t) (ant_id);
}
