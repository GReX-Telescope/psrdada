#ifndef __SNAPSR_DEF_H
#define __SNAPSR_DEF_H

#include "dada_def.h"

#define SNAPSR_DEFAULT_UDPDB_PORT   4001
#define SNAPSR_DEFAULT_PWC_PORT     49122
#define SNAPSR_DEFAULT_PWC_LOGPORT  49123
#define SNAPSR_NUM_UDP_PACKETS      8192

#define SNAPSR_NDIM   2

#define HIRES
#ifdef HIRES

#define SNAPSR_MAX_MODULES_PER_PFB  8
#define SNAPSR_MAX_PFBS 44
#define SNAPSR_MAX_BAYS 88
#define SNAPSR_MAX_MODULES 384
#define SNAPSR_MAX_CHANS   1024

#define UDP_HEADER   16             // size of header/sequence number
#define UDP_DATA     5120           // obs bytes per packet [missing 8 bytes atm]
#define UDP_PAYLOAD  5136           // header + datasize
#define UDP_NPACKS   65536          // 512 MB worth

#else

#define SNAPSR_MAX_MODULES_PER_PFB  16
#define SNAPSR_MAX_PFBS 24
#define SNAPSR_MAX_BAYS 88
#define SNAPSR_MAX_MODULES 384
#define SNAPSR_MAX_CHANS   128

#endif

#define UDP_IFACE    "192.168.4.14" // default interface

#endif // __SNAPSR_DEF_H

