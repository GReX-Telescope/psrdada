#ifndef __DADA_DISK_ARRAY_H
#define __DADA_DISK_ARRAY_H

/* ************************************************************************

   ************************************************************************ */

#include <pthread.h>
#include <sys/vfs.h>
#include <sys/types.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {

    struct statfs info;
    dev_t device;

    char*  path;

  } disk_t;

  typedef struct {
    
    disk_t*  disks;    /* disks to which data will be written */
    unsigned ndisk;    /* number of disks */
    uint64_t space;    /* number of bytes total */

    /* for multi-threaded use of the dbdisk struct */
    pthread_mutex_t mutex;
    
  } disk_array_t;

  /*! Create a new disk array */
  disk_array_t* disk_array_create ();

  /*! Destroy a disk array */
  int disk_array_destroy (disk_array_t*);

  /*! Add a disk to the array */
  int disk_array_add (disk_array_t*, char* path);

  /*! Get the total amount of disk space */
  uint64_t disk_array_get_total (disk_array_t*);

  /*! Get the available amount of disk space */
  uint64_t disk_array_get_available (disk_array_t*);

#ifdef __cplusplus
}
#endif

#endif
