/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <assert.h>

#include "caspsr_udp.h"

socket_buffer_t * init_socket_buffer(unsigned size)
{
  socket_buffer_t * b = (socket_buffer_t *) malloc(sizeof(socket_buffer_t));

  assert(b != NULL);

  b->size = size;
  b->have_packet = 0;
  b->buffer = (char *) malloc(sizeof(char) * size);
  assert(b->buffer != NULL);

  return b;
}

void free_socket_buffer(socket_buffer_t* b)
{
  b->size = 0;
  b->have_packet = 0;
  free(b->buffer);
}


caspsr_buffer_t * init_caspsr_buffer(uint64_t size)
{
  caspsr_buffer_t * b = (caspsr_buffer_t *) malloc(sizeof(caspsr_buffer_t));

  assert(b != NULL);
  b->size = size;
  b->count = 0;
  b->min = 0;
  b->max = 0;
  b->buffer = (char *) malloc(sizeof(char) * size);
  assert(b->buffer != NULL);
  b->ids = (uint64_t *) malloc(sizeof(uint64_t) * size);
  assert(b->ids != NULL);

  return b;
}

caspsr_header_t * init_caspsr_header()
{
  caspsr_header_t * h = (caspsr_header_t *) malloc(sizeof(caspsr_header_t));
  assert(h != NULL);
  h->seq_no = 0;
  h->ch_id = 0;
  return h;
}

void zero_caspsr_buffer(caspsr_buffer_t * b)
{
  char zerodchar = 'c';
  memset(&zerodchar,0,sizeof(zerodchar));
  memset(b->buffer, zerodchar, b->size);
}

void free_caspsr_buffer(caspsr_buffer_t * b)
{
  b->size = 0;
  b->count = 0;
  b->min = 0;
  b->max = 0;
  free(b->buffer);
  free(b->ids);
}


/* copy b_length data from s to b, subject to the s_offset, wrapping if
 * necessary and return the new s_offset */
unsigned int caspsr_encode_data(char * b, char * s, unsigned int b_length, 
                              unsigned int s_length, unsigned int s_offset) {
#ifdef _DEBUG
  fprintf(stderr, "caspsr_encode_data: b_length=%d, s_length=%d, s_offset=%d\n",
                  b_length, s_length, s_offset);
#endif

  if (s_offset + b_length > s_length) {

    unsigned part = s_length - s_offset;
    memcpy(b, s + s_offset, part);
    memcpy(b + part, s, b_length-part);
    return (b_length-part);

  } else {

    memcpy(b, s + s_offset, b_length);
    return (s_offset + b_length);

  }

}

/* platform independant uint64_t to char array converter */
void uint64ToByteArray (uint64_t num, size_t bytes, unsigned char *arr, int type)  
{  
  size_t i;  
  unsigned char ch;  
  for (i = 0; i < bytes; i++ )  
  {  
    ch = (num >> ((i & 7) << 3)) & 0xFF;  
    if (type == LITTLE)  
      arr[i] = ch;  
    else if (type == BIG)  
      arr[bytes - i - 1] = ch;  
  }  
} 

/* platform independant char array to uint64_t converter */
uint64_t byteArrayToUInt64 (unsigned char *arr, size_t bytes, int type) 
{

  uint64_t num = UINT64_C (0);
  uint64_t tmp;

  size_t i;
  for (i = 0; i < bytes; i++ )
  {

    tmp = UINT64_C (0);
    if (type == LITTLE)
      tmp = arr[i];
    else if (type == BIG)
      tmp = arr[bytes - i - 1];

    num |= (tmp << ((i & 7) << 3));
  }

  return num;
}

/* Encode the counter into the first8 bytes of the buffer array */
void caspsr_encode_header (char *b, caspsr_header_t * h) {

  uint64ToByteArray (h->seq_no, (size_t) 8, (unsigned char*) b, (int) BIG);
  uint64ToByteArray (h->ch_id,  (size_t) 8, (unsigned char*) b+8, (int) BIG);

}

/* Reads the first 8 bytes of the buffer array, interpreting it as
 *  * a 64 bit interger */
void caspsr_decode_header(caspsr_header_t * h, char * b) {

  h->seq_no = (byteArrayToUInt64 ((unsigned char*) b, 8, (int) BIG));
  h->ch_id  = byteArrayToUInt64 ((unsigned char*) b+8, 8, (int) BIG);

}
