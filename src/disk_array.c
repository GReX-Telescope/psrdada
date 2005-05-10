#include "disk_array.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

// #define _DEBUG 1

/*! Create a new disk array */
disk_array_t* disk_array_create ()
{
  disk_array_t* array = (disk_array_t*) malloc (sizeof(disk_array_t));
  assert (array != 0);

  array -> disks = 0;
  array -> ndisk = 0;
  array -> space = 0;
  array -> overwrite = 0;

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

#ifdef _DEBUG
static void fprintf_bytes (FILE* fptr, void* id, unsigned nbyte)
{
  char* byte = (char*) id;
  unsigned ibyte = 0;

  for (ibyte=0; ibyte < nbyte; ibyte++)
    fprintf (stderr, "%x ", byte[ibyte]);
  fprintf (stderr, "\n");
}
#endif

/*! Add a disk to the array */
int disk_array_add (disk_array_t* array, char* path)
{
  /* free space on new disk */
  uint64_t new_space = 0;
  /* pointer to new disk_t structure in array */
  disk_t* new_disk = 0;
  /* array counter */
  unsigned idisk;
  /* used while ensuring that two disks are different */
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
  assert (array->disks != 0);

  new_disk = array->disks + array->ndisk;
  array->ndisk ++;

  new_disk->info = info;
  new_disk->device = buf.st_dev;
  new_disk->path = strdup (path);
  assert (new_disk->path != 0);

  array->space += new_space;

  pthread_mutex_unlock (&(array->mutex));

  return 0;
}

int disk_array_set_overwrite (disk_array_t* array, char value)
{
  return array->overwrite = value;
}

uint64_t disk_array_get_total (disk_array_t* array)
{
  return array->space;
}

uint64_t get_available (const char* filename)
{
  struct statfs info;
  uint64_t space = 0;

  if (statfs (filename, &info) < 0) {
    fprintf (stderr, "get_available error statfs(%s): %s",
	     filename, strerror(errno));
    return 0;
  }
  
  space = info.f_bfree;
  space *= info.f_bsize;

  return space;
}

/*! Get the available amount of disk space */
uint64_t disk_array_get_available (disk_array_t* array)
{
  uint64_t available_space = 0;
  unsigned idisk;

  pthread_mutex_lock (&(array->mutex));

  for (idisk = 0; idisk < array->ndisk; idisk++)
    available_space += get_available (array->disks[idisk].path);

  pthread_mutex_unlock (&(array->mutex));

  return available_space;
}

/*! Open a file on the disk array, return the open file descriptor */
int disk_array_open (disk_array_t* array, char* filename, uint64_t filesize,
		     uint64_t* optimal_buffer_size)
{
  static char* fullname = 0;
  unsigned idisk;

  int fd = -1;

  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;

  if (array->overwrite)
    perms |= S_IWUSR;
  else
    flags |= O_EXCL;

  pthread_mutex_lock (&(array->mutex));

  for (idisk = 0; idisk < array->ndisk; idisk++)
    if (get_available (array->disks[idisk].path) > filesize) {

      if (!fullname)
	fullname = malloc (FILENAME_MAX);
      assert (fullname != 0);

      strcpy (fullname, array->disks[idisk].path);
      strcat (fullname, "/");
      strcat (fullname, filename);
      strcpy (filename, fullname);

      fd = open (fullname, flags, perms);

      if (optimal_buffer_size)
	*optimal_buffer_size = array->disks[idisk].info.f_bsize;

      break;
    }

  pthread_mutex_unlock (&(array->mutex));

  return fd;

}
