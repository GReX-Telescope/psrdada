/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>

#include "mopsr_udp.h"

/*
 * create a socket with the specified number of buffers
 */
mopsr_sock_t * mopsr_init_sock ()
{
  mopsr_sock_t * b = (mopsr_sock_t *) malloc(sizeof(mopsr_sock_t));
  assert(b != NULL);

  b->bufsz = sizeof(char) * UDP_PAYLOAD;
    b->buf = (char *) malloc (b->bufsz);
  assert(b->buf != NULL);

  b->have_packet = 0;
  b->fd = 0;

  return b;
}

void mopsr_free_sock(mopsr_sock_t* b)
{
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
}

/*
 * test design has 64 bits constart, then 64 bits seq no
 */
void mopsr_decode_header (unsigned char * b, uint64_t *seq_no)
{
  uint64_t tmp = 0;
  unsigned i = 0;
  *seq_no = UINT64_C (0);

  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    *seq_no |= (tmp << ((i & 7) << 3));
  }
}

void mopsr_encode_header (unsigned char *b, uint64_t seq_no)
{
  b[8] = (uint8_t) (seq_no>>56);
  b[9] = (uint8_t) (seq_no>>48);
  b[10] = (uint8_t) (seq_no>>40);
  b[11] = (uint8_t) (seq_no>>32);
  b[12] = (uint8_t) (seq_no>>24);
  b[13] = (uint8_t) (seq_no>>16);
  b[14] = (uint8_t) (seq_no>>8);
  b[15] = (uint8_t) (seq_no);
}

