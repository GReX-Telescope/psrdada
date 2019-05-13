#ifndef __COMAP_DEF_H
#define __COMAP_DEF_H

#include "dada_def.h"

// default port to connect to udpdb command interface
#define COMAP_DEFAULT_UDPDB_PORT   4002
#define COMAP_DEFAULT_PWC_LOGPORT  40123

#define UDP_HEADER   9             // size of header/sequence number
#define UDP_DATA     6144           // obs bytes per packet
#define UDP_PAYLOAD  6153           // header + datasize
#define UDP_IFACE    "192.168.41.8" // default interface

#endif // __COMAP_DEF_H

