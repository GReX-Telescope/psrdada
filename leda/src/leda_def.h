#ifndef __LEDA_DEF_H
#define __LEDA_DEF_H

#include "dada_def.h"

// default port to connect to udpdb command interface
#define LEDA_DEFAULT_UDPDB_PORT   4002

#define UDP_HEADER   16             // size of header/sequence number
#define UDP_DATA     8192           // obs bytes per packet
#define UDP_PAYLOAD  8208           // header + datasize
#define UDP_IFACE    "192.168.4.14" // default interface

#endif // __LEDA_DEF_H

