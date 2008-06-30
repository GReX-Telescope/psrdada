#ifndef __DADA_APSR_UDP_H
#define __DADA_APSR_UDP_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "udp.h"

/* Maximum size of a UDP packet */
#define UDPBUFFSIZE 16384

/* Size of header component of the data packet */
#define UDPHEADERSIZE 24

/* Size of data component of the data packet */
/* 8948 == 9000 - 28 (udp hdr) - 24 (our hdr) */
#define DEFAULT_UDPDATASIZE 8948

/* header struct for UDP packet from board */ 
typedef struct {
  unsigned char length;
  unsigned char source;
  unsigned int sequence;
  unsigned char bits;
  unsigned char channels;
  unsigned char bands;
  unsigned char bandID[4];
  unsigned int pollength;
} header_struct;

void decode_header(char *buffer, header_struct *header);
void encode_header(char *buffer, header_struct *header);
void print_header(header_struct *header);

void encode_header(char *buffer, header_struct *header) {

  if (header->bits == 0) header->bits = 8;

  int temp = header->sequence;
       
  buffer[7] = 0x00;
  buffer[6] = header->bits;
  buffer[5] = temp & 0xff;
  buffer[4] = (temp >> 8) & 0xff;
  buffer[3] = (temp >> 16) & 0xff;
  buffer[2] = (temp >> 24) & 0xff;
  buffer[1] = header->source;
  buffer[0] = header->length;
  buffer[15] = 0x00;
  buffer[14] = 0x00;
  buffer[13] = header->bandID[3];
  buffer[12] = header->bandID[2];
  buffer[11] = header->bandID[1];
  buffer[10] = header->bandID[0];
  buffer[9] = header->bands;
  buffer[8] = header->channels;

}

void decode_header(char *buffer, header_struct *header) {
    
  int temp;

  /* header decode */
  header->length    = buffer[0];
  header->source    = buffer[1];
  header->sequence  = buffer[2]; 
  header->sequence  <<= 24;
  temp              = buffer[3];
  header->sequence  |= ((temp << 16) & 0xff0000);
  temp              = buffer[4];
  header->sequence  |= (temp << 8) & 0xff00;
  header->sequence  |=  (buffer[5] & 0xff);
    
  // header->packetbitsum  = buffer                                                            
  // header->bits      = buffer[6];    
  // zeros	    = buffer[7];
  header->channels  = buffer[8];
  header->bands     = buffer[9];
                                                 
  header->bandID[0] = buffer[10];
  header->bandID[1] = buffer[11];
  header->bandID[2] = buffer[12];
  header->bandID[3] = buffer[13];
  // zeros          = buffer[14];
  // zeros          = buffer[14];

}

void print_header(header_struct *header) {

  fprintf(stderr,"length = %d\n",header->length);
  fprintf(stderr,"source = %d\n",header->source);
  fprintf(stderr,"sequence= %d\n",header->sequence);
  fprintf(stderr,"bits = %d\n",header->bits);
  fprintf(stderr,"channels = %d\n",header->channels);
  fprintf(stderr,"bands = %d\n",header->bands);
}

#endif /* __DADA_APSR_UDP_H */
