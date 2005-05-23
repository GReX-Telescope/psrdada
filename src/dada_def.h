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

/* default port to connect to dbnic logging interface */
#define DADA_DEFAULT_DBNIC_LOG  56047

/* default port to connect to primary write client command interface */
#define DADA_DEFAULT_NICDB_PORT 56056

/* default port to connect to primary write client logging interface */
#define DADA_DEFAULT_NICDB_LOG  56057

/* default file size of 1 GB */
#define DADA_DEFAULT_FILESIZE 1073741824

#define DADA_DEFAULT_XFERSIZE DADA_DEFAULT_FILESIZE

/* maximum length of observation id string */
#define DADA_OBS_ID_MAXLEN 64

#endif
