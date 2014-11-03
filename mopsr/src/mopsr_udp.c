/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
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
  b->buf = (char *) malloc ( b->bufsz);
  assert(b->buf != NULL);

  b->have_packet = 0;
  b->fd = 0;
  b->prev_seq = 0;
  b->seq_offset = -1;
  b->block_count = 0;

  return b;
}

void mopsr_reset_sock (mopsr_sock_t* b)
{
  b->seq_offset = -1;
  b->have_packet = 0;
  b->prev_seq = 0;
  b->block_count = 0;
}

void mopsr_free_sock(mopsr_sock_t* b)
{
  if (b->fd)
    close(b->fd);
  b->fd = 0;
  b->bufsz = 0;
  b->have_packet =0;
  if (b->buf)
    free (b->buf);
  b->buf = 0;
  b->seq_offset = 0;
  b->block_count = 0;
}


int mopsr_get_bit_from_16 (uint16_t n, unsigned bit)
{
  return (n & ( 1 << bit )) >> bit;
}

// version 3 of UDP header for PFB format
// byte [  nibble 1  |  nibble 1  ]
// 0    [ mgtlocks_l | mgtlocks_l ]
// 1    [ mgtlocks_l | mgtlocks_l ]
// 2    [  mgtlocks  |  mgtlocks  ]
// 3    [  mgtlocks  |  mgtlocks  ]
// 4    [   schan    |   schan    ]
// 5    [     nbit   |    nant    ]
// 6    [   nchan    |   nchan    ]
// 7    [   nframe   |     0x0    ]
void mopsr_decode (unsigned char * b, mopsr_hdr_t * hdr)
{
  hdr->mgt_locks_long = (uint16_t) ((b[0] << 8) | b[1]);

  hdr->mgt_locks = (uint16_t) ((b[2] << 8) | b[3]);

  hdr->start_chan = (unsigned int) b[4] + 1;

  hdr->nbit = (unsigned int) (b[5] >> 4) + 1;
  hdr->nant = (unsigned int) (b[5] & 0x0f) + 1;

  hdr->nchan = (unsigned int) b[6] + 1;

  hdr->nframe = (unsigned int) (b[7] >> 4) + 1;

  uint64_t tmp = 0;
  unsigned i = 0;
  hdr->seq_no = UINT64_C (0);

  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }
}

// byte [  nibble 1  |  nibble 1  ]
// 0    [     0      |      1     ]
// 1    [     2      |  mgtlocks  ]
// 2    [  mgtlocks  |  mgtlocks  ]
// 3    [  mgtlocks  |   schan    ]
// 4    [   schan    |    nbit    ]
// 5    [    nant    |   nchan    ]
// 6    [   nchan    |   nframe   ]
// 7    [ magic/b1   |  magic/b2  ]


void mopsr_decode_v2 (unsigned char * b, mopsr_hdr_t * hdr)
{
  // bits 0-12 are x012
  hdr->mgt_locks = ((b[1] << 12) | (b[2] << 4) | (b[3] >> 4));

  unsigned char schan = ((b[3] << 4) | (b[4] >> 4)) + 1;
  hdr->start_chan = (unsigned int) schan;

  //hdr->start_chan = ((unsigned int) (b[3] & 0x0f) | (b[4] >> 4)) + 1;
  //hdr->start_chan = ((unsigned int) (b[3] & 0x0f) | (b[4] >> 4)) + 1;

  hdr->nbit  = (unsigned int) (b[4] & 0x0f) + 1;
  hdr->nant  = (unsigned int) (b[5] >> 4) + 1;

  unsigned char nchan = ((b[5] << 4) | (b[6] >> 4)) + 1;
  hdr->nchan = (unsigned int) nchan;

  hdr->nframe = (unsigned int) (b[6] & 0x0f) + 1;

  hdr->ant_id  = (unsigned int) ((b[7] & 0x0f) & 0x07);
  hdr->ant_id2 = (unsigned int) ((b[7] >> 4) & 0x07);

  hdr->mf_lock = (unsigned char) (b[7] & 0x80) >> 7;
  hdr->mf_lock2 = (unsigned char) (b[7] & 0x08) >> 3;

  // special case for flow through mode
  if (b[0] == 0x12)
  {
    uint16_t nsamp = ((b[2] << 12) ^ (b[3] << 4) ^ (b[4] >> 4));
    hdr->nsamp = (unsigned int) nsamp + 1;

    hdr->ant_id = 2;
    hdr->ant_id2 = 3;
  }

  uint64_t tmp = 0;
  unsigned i = 0;
  hdr->seq_no = UINT64_C (0);

  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }

}


void mopsr_decode_v1 (unsigned char * b, mopsr_hdr_t * hdr)
{
  if (b[0] == 0x12)
  {
#ifdef _DEBUG
    char str[9];
    char_to_bstring(str, b[2]);
    fprintf (stderr, "b[2]=%s\n", str);
    char_to_bstring(str, b[3]);
    fprintf (stderr, "b[3]=%s\n", str);
    char_to_bstring(str, b[4]);
    fprintf (stderr, "b[4]=%s\n", str);
#endif

    uint16_t nsamp = ((b[2] << 12) ^ (b[3] << 4) ^ (b[4] >> 4));
    hdr->nsamp = (unsigned int) nsamp + 1;

    hdr->ant_id = 2;
    hdr->ant_id2 = 3;
  }
  else
  {
    hdr->nsamp = 0;
    hdr->ant_id2 = (unsigned int) ((b[7] >> 4) & 0x07);
    hdr->ant_id  = (unsigned int) ((b[7] & 0x0f) & 0x07);
  }

  hdr->nbit  = (unsigned int) (b[4] & 0x0f) + 1;
  hdr->nant  = (unsigned int) (b[5] >> 4) + 1;

  //unsigned char nchan = ((b[5] << 4) ^ (b[6] >> 4)) + 1;
  unsigned char nchan = ((b[5] << 4) ^ (b[6] >> 4));
  hdr->nchan = (unsigned int) nchan;

  hdr->nframe = (unsigned int) (b[6] & 0x0f) + 1;

  hdr->mf_lock2 = (unsigned char) (b[7] & 0x08) >> 3;
  hdr->mf_lock = (unsigned char) (b[7] & 0x80) >> 7;

  uint64_t tmp = 0;
  unsigned i = 0;
  hdr->seq_no = UINT64_C (0);

  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }
}

/* just decodes the sequence number, leaving other fields blank */
void inline mopsr_decode_seq (unsigned char * b, mopsr_hdr_t * hdr)
{
  uint64_t tmp = 0;
  unsigned i = 0;
  hdr->seq_no = UINT64_C (0);

  for (i = 0; i < 8; i++ )
  {
    tmp = UINT64_C (0);
    tmp = b[16 - i - 1];
    hdr->seq_no |= (tmp << ((i & 7) << 3));
  }
}

// 4    [   schan    |   schan    ]
// 5    [     nbit   |    nant    ]
// 6    [   nchan    |   nchan    ]
// 7    [   nframe   |     0x0    ]

void mopsr_encode (unsigned char * b, mopsr_hdr_t * hdr)
{
  b[0] = (uint8_t) (hdr->mgt_locks_long >> 8);
  b[1] = (uint8_t) (hdr->mgt_locks_long & 0x00ff);

  b[2] = (uint8_t) (hdr->mgt_locks >> 8);
  b[3] = (uint8_t) (hdr->mgt_locks & 0x00ff);

  b[4] = (uint8_t) (hdr->start_chan - 1);

  b[5]  = ((uint8_t) (hdr->nbit - 1) << 4) & 0xf0;
  b[5] |= ((uint8_t) (hdr->nant - 1)) & 0x0f;

  b[6] = (uint8_t) (hdr->nchan - 1);
  
  b[7] = ((uint8_t) (hdr->nframe - 1) << 4) & 0xf0;

  b[8] = (uint8_t) (hdr->seq_no>>56);
  b[9] = (uint8_t) (hdr->seq_no>>48);
  b[10] = (uint8_t) (hdr->seq_no>>40);
  b[11] = (uint8_t) (hdr->seq_no>>32);
  b[12] = (uint8_t) (hdr->seq_no>>24);
  b[13] = (uint8_t) (hdr->seq_no>>16);
  b[14] = (uint8_t) (hdr->seq_no>>8);
  b[15] = (uint8_t) (hdr->seq_no);
}

void mopsr_encode_v2 (unsigned char * b, mopsr_hdr_t * hdr)
{
  // PFB mode
  if (hdr->nchan > 1)
  {
    b[0] = 0x01;
    b[1] = 0x20;

    b[1] |= ((uint8_t) (hdr->mgt_locks >> 12)) & 0x0f;
    b[2] = (uint8_t) (hdr->mgt_locks >> 4);
    b[3] = (uint8_t) (hdr->mgt_locks << 4) & 0xf0;

    b[3] |= ((uint8_t) (hdr->start_chan - 1) >> 4) & 0x0f;
    b[4] =  ((uint8_t) (hdr->start_chan - 1) << 4) & 0xf0;
  }
  // Flow Through mode
  else
  {
    b[0] = 0x12;
    b[1] = 0x34;
    b[2] = 0x50;
    uint16_t nsamp = (uint16_t) (hdr->nsamp - 1);
    b[2] |= ((uint8_t) (nsamp >> 12)) & 0x0f;
    b[3] = (uint8_t) (nsamp >> 4);
    b[4] = (uint8_t) (nsamp << 4) & 0xf0;
  }

  b[4] |= ((uint8_t) (hdr->nbit -1)) & 0x0f;

  b[5] =  ((uint8_t) (hdr->nant - 1) << 4) & 0xf0;

  b[5] |= ((uint8_t) (hdr->nchan - 1) >> 4) & 0x0f;
  b[6] =  ((uint8_t) (hdr->nchan - 1) << 4) & 0xf0;

  b[6] |= ((uint8_t) (hdr->nframe - 1)) & 0x0f;

  b[7] =  (uint8_t) ((hdr->ant_id2) << 4) & 0xf0;
  b[7] |= (uint8_t)  ((hdr->mf_lock2 & 0x01) << 7);
  b[7] |= (uint8_t) (hdr->ant_id & 0x0f);
  b[7] |= (uint8_t)  ((hdr->mf_lock & 0x01) << 3);


  b[8] = (uint8_t) (hdr->seq_no>>56);
  b[9] = (uint8_t) (hdr->seq_no>>48);
  b[10] = (uint8_t) (hdr->seq_no>>40);
  b[11] = (uint8_t) (hdr->seq_no>>32);
  b[12] = (uint8_t) (hdr->seq_no>>24);
  b[13] = (uint8_t) (hdr->seq_no>>16);
  b[14] = (uint8_t) (hdr->seq_no>>8);
  b[15] = (uint8_t) (hdr->seq_no);
}

void mopsr_encode_v1 (unsigned char * b, mopsr_hdr_t * hdr)
{
  if (hdr->nchan > 1)
  {
    b[0] = 0x01;
    b[1] = 0x23;
    b[2] = 0x40;
    b[3] = 0x00;
    b[4] = 0x00;
  }
  else
  {
    b[0] = 0x12;
    b[1] = 0x34;
    b[2] = 0x50;

    uint16_t nsamp = (uint16_t) (hdr->nsamp - 1);
    b[2] |= ((uint8_t) (nsamp >> 12)) & 0x0f;

    b[3] = (uint8_t) (nsamp >> 4);

    b[4] = (uint8_t) (nsamp << 4) & 0xf0;

  }

  b[4] |= ((uint8_t) (hdr->nbit -1)) & 0x0f;

  b[5] =  ((uint8_t) (hdr->nant - 1) << 4) & 0xf0;
  b[5] |= ((uint8_t) (hdr->nchan -1) >> 4) & 0x0f;

  b[6] =  ((uint8_t) (hdr->nchan - 1) << 4) & 0xf0;
  b[6] |= ((uint8_t) (hdr->nframe - 1)) & 0x0f;

  b[7] =  (uint8_t) ((hdr->ant_id2) << 4) & 0xf0;
  b[7] |= (uint8_t)  ((hdr->mf_lock2 & 0x01) << 7);
  b[7] |= (uint8_t) (hdr->ant_id & 0x0f);
  b[7] |= (uint8_t)  ((hdr->mf_lock & 0x01) << 3);

  b[8] = (uint8_t) (hdr->seq_no>>56);
  b[9] = (uint8_t) (hdr->seq_no>>48);
  b[10] = (uint8_t) (hdr->seq_no>>40);
  b[11] = (uint8_t) (hdr->seq_no>>32);
  b[12] = (uint8_t) (hdr->seq_no>>24);
  b[13] = (uint8_t) (hdr->seq_no>>16);
  b[14] = (uint8_t) (hdr->seq_no>>8);
  b[15] = (uint8_t) (hdr->seq_no);
}


/*
 * test design has 64 bits constart, then 64 bits seq no
 */
void mopsr_decode_header (unsigned char * b, uint64_t *seq_no, unsigned int * ant_id)
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

  // antenna ID is encoded in last nibble of first 8bytes
  *ant_id = (unsigned int) (b[7] & 0x0f);
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

unsigned int mopsr_get_ant_number (unsigned int id, unsigned int index)
{
  return ((2 * id) + index);
}

int mopsr_new_ant_mapping[] = { 0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15 };

unsigned int mopsr_get_new_ant_number (unsigned int index)
{
  return mopsr_new_ant_mapping[index];

  //if (index < 8)
  //  return (index * 2);
  //else
  //  return ((index - 8) * 2) + 1;
}

unsigned int mopsr_get_new_ant_index (unsigned int number) 
{
  unsigned i, index;

  index = 0;
  for (i=0; i<16; i++)
    if (mopsr_new_ant_mapping[i] == number)
      index = i;

  return index;
  //if (number % 2 == 0)
  //  return number / 2;
  //else
  //  return (number / 2) + 8;
}


void mopsr_print_header(mopsr_hdr_t * hdr)
{
  fprintf (stderr, "nsamp=%u\n", hdr->nsamp);
  fprintf (stderr, "nbit=%u nant=%u nchan=%u nframe=%u\n",
           hdr->nbit, hdr->nant, hdr->nchan, hdr->nframe);
  fprintf (stderr, "ant_id=%u ant_id2=%u seq=%"PRIu64"\n",
           hdr->ant_id, hdr->ant_id2, hdr->seq_no);
}

