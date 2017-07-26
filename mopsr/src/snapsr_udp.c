/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>
#include <errno.h>
#include "snapsr_udp.h"

/*
 * create a socket with the specified number of buffers
 */
snapsr_sock_t * snapsr_init_sock ()
{
  snapsr_sock_t * b = (snapsr_sock_t *) malloc(sizeof(snapsr_sock_t));
  assert(b != NULL);

  b->bufsz = sizeof(char) * SNAPSR_UDP_PAYLOAD_BYTES;
  b->buf = (char *) malloc ( b->bufsz);
  assert(b->buf != NULL);

  b->have_packet = 0;
  b->fd = 0;
  b->prev_seq = 0;

  return b;
}

void snapsr_reset_sock (snapsr_sock_t* b)
{
  b->have_packet = 0;
  b->prev_seq = 0;
}

void snapsr_free_sock(snapsr_sock_t* b)
{
  if (b->fd)
    close(b->fd);
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
}

void snapsr_decode (unsigned char * b, snapsr_udp_hdr_t * hdr)
{
  hdr->snap_id = (unsigned) b[0];

  hdr->seq_no = UINT64_C (0);
  uint64_t tmp = 0;
  unsigned i;
  for (i = 0; i < 6; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[7 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }

  hdr->subband_id = (unsigned) b[7];

  hdr->nframe = SNAPSR_NFRAME_PER_PACKET;
  hdr->nant = SNAPSR_NANT_PER_PACKET;
  hdr->nchan = SNAPSR_NCHAN_PER_SUBBAND;

  // TODO remove
  hdr->snap_id = 0;
}

/* just decodes the sequence number, leaving other fields blank */
void inline snapsr_decode_seq (unsigned char * b, snapsr_udp_hdr_t * hdr)
{
  hdr->seq_no = UINT64_C (0);
  uint64_t tmp = 0;
  unsigned i;
  for (i = 0; i < 6; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[7 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }
}

void snapsr_encode (unsigned char * b, snapsr_udp_hdr_t * hdr)
{
  b[0] = (uint8_t) hdr->snap_id;
  b[1] = (uint8_t) (hdr->seq_no>>40);
  b[2] = (uint8_t) (hdr->seq_no>>32);
  b[3] = (uint8_t) (hdr->seq_no>>24);
  b[4] = (uint8_t) (hdr->seq_no>>16);
  b[5] = (uint8_t) (hdr->seq_no>>8);
  b[6] = (uint8_t) (hdr->seq_no);
  b[7] = (uint8_t) hdr->subband_id;
}

void snapsr_print_header(snapsr_udp_hdr_t * hdr)
{
  fprintf (stderr, "SNAP UDP_HEADER  %u  %lu  %u \n", hdr->snap_id, hdr->seq_no, hdr->subband_id);
}

