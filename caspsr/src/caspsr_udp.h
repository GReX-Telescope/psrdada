#ifndef __CASPSR_UDP_H
#define __CASPSR_UDP_H

/* 
 * CASPSR udp specific functions 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dada_udp.h"

#define CASPSR_UDP_HEADER   16         // size of header/sequence number
#define CASPSR_UDP_DATA     1024       // obs bytes per packet
#define CASPSR_UDP_PAYLOAD  1040       // header + datasize
#define CASPSR_UDP_NPACKS   102400     // 100 megabytes worth (1 second)
#define CASPSR_UDP_IFACE    "10.0.0.4" // default interface

#define CASPSR_BIBOB_UDP_PAYLOAD 1024
#define CASPSR_BIBOB_UDP_HEADER 8
#define CASPSR_BIBOB_UDP_PACKET 1032

#define CASPSR_IBOB_CLOCK         400        // MHz
#define CASPSR_IBOB_NCHANNELS     1024

/* Temp constants for the conversion function */
# define LITTLE 0
# define BIG 1

//#define _DEBUG

/* a buffer that can be filled with data */
typedef struct {

  char * buffer;    // data buffer itself
  uint64_t * ids;   // packet ids in the buffer
  uint64_t count;   // number of packets in this buffer
  uint64_t size;    // size in bytes of the buffer
  uint64_t min;     // minimum seq number acceptable
  uint64_t max;     // maximum seq number acceptable

} caspsr_buffer_t;


/* socket buffer for receiving udp data */
typedef struct {

  char * buffer;          // the buffer itself
  unsigned have_packet;   // is there a packet in the buffer
  unsigned size;          // size of the buffer
  size_t got;           // amount of data received

} socket_buffer_t;


/* a caspsr udp header */
typedef struct {

  uint64_t seq_no;
  uint64_t ch_id;

} caspsr_header_t;



int      caspsr_create_udp_socket(multilog_t* log, const char* iface, int port, 
                                int verbose);
void     caspsr_decode_header (caspsr_header_t * h, char * b);
void     caspsr_encode_header (char *b, caspsr_header_t * h);
void     uint64ToByteArray (uint64_t num, size_t bytes, unsigned char *arr, 
                            int type);
uint64_t byteArrayToUInt64 (unsigned char *arr, size_t bytes, int type);
unsigned int caspsr_encode_data(char * b, char * s, unsigned int b_length,
                              unsigned int s_length, unsigned int s_offset); 

socket_buffer_t * init_socket_buffer(unsigned size);
caspsr_buffer_t * init_caspsr_buffer(uint64_t size);
caspsr_header_t * init_caspsr_header(void);
void free_socket_buffer(socket_buffer_t* b);
void zero_caspsr_buffer(caspsr_buffer_t * b);
void free_caspsr_buffer(caspsr_buffer_t * b);



#endif /* __CASPSR_UDP_H */
