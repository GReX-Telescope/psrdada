#ifndef __DADA_DEF_H
#define __DADA_DEF_H

#include <unistd.h>

/* ************************************************************************

   dada default definitions

   ************************************************************************ */

/* key to connect to header block shared memory */
#define DADA_HEADER_BLOCK_KEY   0xdada

/* key to connect to data block shared memory */
#define DADA_DATA_BLOCK_KEY     0xdadb

/* default number of blocks in Data Block */
#define DADA_DEFAULT_BLOCK_NUM  ((uint64_t) 32)

/* default size of blocks in Data Block */
#define DADA_DEFAULT_BLOCK_SIZE  ((uint64_t) sysconf (_SC_PAGE_SIZE) * 128)

/* default size of block in Header Block */
#define DADA_DEFAULT_HEADER_SIZE ((uint64_t) sysconf (_SC_PAGE_SIZE))

/* default port to connect to primary write client command interface */
#define DADA_DEFAULT_PWC_PORT   56026

/* default port to connect to primary write client logging interface */
#define DADA_DEFAULT_PWC_LOG    56027

/* default port to connect to dbdisk logging interface */
#define DADA_DEFAULT_DBDISK_LOG 56037

/* default port to connect to diskdb logging interface */
#define DADA_DEFAULT_DISKDB_LOG 56039

/* default port to connect to dbnic logging interface */
#define DADA_DEFAULT_DBNIC_LOG  56047

/* default port to connect to primary write client command interface */
#define DADA_DEFAULT_NICDB_PORT 56056

/* default port to connect to primary write client logging interface */
#define DADA_DEFAULT_NICDB_LOG  56057

/* default port to connect to udpdb command interface */
#define DADA_DEFAULT_UDPDB_PORT 56058

/* default port to connect to udpdb logging interface */
#define DADA_DEFAULT_UDPDB_LOG  56059

/* default port to connect to udpdb statistics interface */
#define DADA_DEFAULT_UDPDB_STATS  56060

/* default port to connect to dbull client logging interface */
#define DADA_DEFAULT_DBNULL_LOG  56061

/* default port to connect to dbtritest client logging interface */
#define DADA_DEFAULT_DBTRITEST_LOG  56062

/* default port to connect to dada_dbmonitor logging interface */
#define DADA_DEFAULT_DBMONITOR_LOG  56063

/* default port to connect to dada_dbudp logging interface */
#define DADA_DEFAULT_DBUDP_LOG  56064

/* default file size of 1 GB */
#define DADA_DEFAULT_FILESIZE 1073741824

#define DADA_DEFAULT_XFERSIZE DADA_DEFAULT_FILESIZE

/* maximum length of observation id string */
#define DADA_OBS_ID_MAXLEN 64

/* the format of the UTC_START string used in call to strftime */
#define DADA_TIMESTR "%Y-%m-%d-%H:%M:%S"


#endif
