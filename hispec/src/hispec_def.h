#ifndef __HISPEC_DEF_H
#define __HISPEC_DEF_H

/* default port to connect to udpdb command interface */
#define HISPEC_DEFAULT_UDPDB_PORT   4001

#define HISPEC_DEFAULT_PWC_PORT     51022
#define HISPEC_DEFAULT_PWC_LOGPORT  51023

/* default port to connect to udpdb statistics interface */
#define HISPEC_DEFAULT_UDPDB_STATS  51012

/* Number of UDP packets to be recived for a called to buffer_function
 * 32 MB = 2048 * 16384*/
#define HISPEC_NUM_UDP_PACKETS      16384

#endif

