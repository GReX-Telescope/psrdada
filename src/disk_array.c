#include "disk_array.h"

#include <sys/stat.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// #define _DEBUG 1

/*! Create a new disk array */
disk_array_t* disk_array_create ()
{
  disk_array_t* array = (disk_array_t*) malloc (sizeof(disk_array_t));
  array -> disks = 0;
  array -> ndisk = 0;
  array -> space = 0;

  pthread_mutex_init(&(array->mutex), NULL);

  return array;
}

/*! Destroy a disk array */
int disk_array_destroy (disk_array_t* array)
{
  if (!array)
    return -1;

  if (array->disks)
    free (array->disks);

  pthread_mutex_destroy (&(array->mutex));

  return 0;
}

static void fprintf_bytes (FILE* fptr, void* id, unsigned nbyte)
{
  char* byte = (char*) id;
  unsigned ibyte = 0;

  for (ibyte=0; ibyte < nbyte; ibyte++)
    fprintf (stderr, "%x ", byte[ibyte]);
  fprintf (stderr, "\n");
}

/*! Add a disk to the array */
int disk_array_add (disk_array_t* array, char* path)
{
  uint64_t new_space = 0;
  unsigned idisk;
  int diff;

  struct statfs info;
  struct stat buf;

  if (statfs (path, &info) < 0) {
    fprintf (stderr, "disk_array_add error statfs(%s)", path);
    perror ("");
    return -1;
  }

  if (stat (path, &buf) < 0) {
    fprintf (stderr, "disk_array_add error stat(%s)", path);
    perror ("");
    return -1;
  }

#ifdef _DEBUG
  fprintf(stderr,"DISK_FREE_SPACE: Debug info:\n");
  fprintf(stderr,"file system block size f_bsize: %lu\n",info.f_bsize );
  fprintf(stderr,"number of blocks f_blocks: %lu\n",info.f_blocks );
  fprintf(stderr,"number of free blocks f_bfree: %lu\n",info.f_bfree );
  fprintf(stderr,"number of free blocks f_bavail: %lu\n",info.f_bavail );
  fprintf(stderr,"number of file nodes (inodes) f_files: %lu\n",info.f_files );
  fprintf(stderr,"number of free file nodes f_ffree: %lu\n",info.f_ffree );
  fprintf(stderr,"file system id f_fsid: ");
  fprintf_bytes (stderr, &(info.f_fsid), sizeof(fsid_t) );
  fprintf(stderr,"device: ");
  fprintf_bytes (stderr, &(buf.st_dev), sizeof(dev_t) );
#endif

  new_space = info.f_blocks;
  new_space *= info.f_bsize;

  pthread_mutex_lock (&(array->mutex));

  /* ensure that each disk in array is a unique device */
  for (idisk = 0; idisk < array->ndisk; idisk++) {
    diff = memcmp(&(buf.st_dev), &(array->disks[idisk].device), sizeof(dev_t));
    if (!diff) {
      fprintf (stderr, "disk_array_add: %s is on same device as %s\n",
	       path, array->disks[idisk].path);
      pthread_mutex_unlock (&(array->mutex));
      return -1;
    }
  }

  array->disks = realloc (array->disks, (array->ndisk+1)*sizeof(disk_t));

  array->disks[array->ndisk].info = info;
  array->disks[array->ndisk].device = buf.st_dev;
  array->disks[array->ndisk].path = strdup (path);

  array->ndisk ++;
  array->space += new_space;

  pthread_mutex_unlock (&(array->mutex));

  return 0;
}


uint64_t disk_array_get_total (disk_array_t* array)
{
  return array->space;
}

/*! Get the available amount of disk space */
uint64_t disk_array_get_available (disk_array_t* array)
{
  uint64_t available_space = 0;
  uint64_t new_space = 0;
  struct statfs info;
  unsigned idisk;

  pthread_mutex_lock (&(array->mutex));

  for (idisk = 0; idisk < array->ndisk; idisk++) {

    if (statfs (array->disks[idisk].path, &info) < 0) {
      fprintf (stderr, "disk_array_add error statfs(%s)", 
	       array->disks[idisk].path);
      perror ("");
      continue;
    }

    new_space = info.f_bfree;
    new_space *= info.f_bsize;

    available_space += new_space;
  }

  pthread_mutex_unlock (&(array->mutex));

  return available_space;
}
