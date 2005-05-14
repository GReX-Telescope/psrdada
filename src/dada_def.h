#ifndef __DADA_H
#define __DADA_H

/* ************************************************************************

   dada default definitions

   ************************************************************************ */

/* default file size of 1 GB */
#define DADA_DEFAULT_FILESIZE 1073741824

/* default header size of 4kB */
#define DADA_DEFAULT_HDR_SIZE 4096

/* key to connect to header block shared memory */
#define DADA_HEADER_BLOCK_KEY   0xdada

/* key to connect to data block shared memory */
#define DADA_DATA_BLOCK_KEY     0xdadb

/* default port to connect to primary write client command interface */
#define DADA_DEFAULT_PWC_PORT   56026

/* default port to connect to primary write client logging interface */
#define DADA_DEFAULT_PWC_LOG    56027

/* default port to connect to dbdisk logging interface */
#define DADA_DEFAULT_DBDISK_LOG 56037

/* default port to connect to dbnic logging interface */
#define DADA_DEFAULT_DBNIC_LOG  56047

/* maximum length of observation id string */
#define DADA_OBS_ID_MAXLEN 64

#endif
