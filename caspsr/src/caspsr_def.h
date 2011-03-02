#ifndef __CASPSR_DEF_H
#define __CASPSR_DEF_H

#include "dada_def.h"

/* default port to connect to udpdb command interface */
#define CASPSR_DEFAULT_UDPDB_PORT   4002
#define CASPSR_DEFAULT_PWC_PORT     49122
#define CASPSR_DEFAULT_PWC_LOGPORT  49123

#define UDP_HEADER   16             // size of header/sequence number
#define UDP_DATA     8192           // obs bytes per packet
#define UDP_PAYLOAD  8208           // header + datasize
#define UDP_NPACKS   65536          // 512 MB worth
#define UDP_IFACE    "192.168.4.14" // default interface

#endif // __CASPSR_DEF_H

