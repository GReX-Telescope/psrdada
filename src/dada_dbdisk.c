#include "dada_prc_main.h"
#include "dada.h"

#include "disk_array.h"
#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>

void usage()
{
  fprintf (stdout,
	   "dada_dbdisk [options]\n"
	   " -D <path>  add a disk to which data will be written\n"
	   " -W         over-write exisiting files\n"
	   " -d         run as daemon\n");
}

typedef struct {

  /* the set of disks to which data may be written */
  disk_array_t* array;

  /* current observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN];

  /* current filename */
  char file_name [FILENAME_MAX];

  /* file offset from start of data, as defined by FILE_NUMBER attribute */
  unsigned file_number;

} dada_dbdisk_t;

#define DADA_DBDISK_INIT { 0, "", "", 0 }

/*! Function that opens the data transfer target */
int file_open_function (dada_prc_main_t* prcm)
{
  /* the dada_dbdisk specific data */
  dada_dbdisk_t* dbdisk = 0;
  
  /* status and error logging facility */
  multilog_t* log;

  /* the header */
  char* header = 0;

  /* observation id, as defined by OBS_ID attribute */
  char obs_id [DADA_OBS_ID_MAXLEN] = "";

  /* size of each file to be written in bytes, as determined by FILE_SIZE */
  uint64_t file_size = 0;

  /* the optimal buffer size for writing to file */
  uint64_t optimal_bytes = 0;

  /* the open file descriptor */
  int fd = -1;

  assert (prcm != 0);

  dbdisk = (dada_dbdisk_t*) prcm->context;
  assert (dbdisk != 0);

  log = prcm->log;
  assert (log != 0);

  header = prcm->header;
  assert (header != 0);

  /* Get the observation ID */
  if (ascii_header_get (prcm->header, "OBS_ID", "%s", obs_id) != 1) {
    multilog (log, LOG_WARNING, "Header block does not define OBS_ID\n");
    strcpy (obs_id, "UNKNOWN");
  }

  /* check to see if we are still working with the same observation */
  if (strcmp (obs_id, dbdisk->obs_id) != 0) {
    dbdisk->file_number = 0;
    multilog (log, LOG_INFO, "New OBS_ID=%s -> file number=0\n", obs_id);
  }
  else {
    dbdisk->file_number++;
    multilog (log, LOG_INFO, "Continue OBS_ID=%s -> file number=%lu\n",
	      obs_id, dbdisk->file_number);
  }

  /* Set the file number to be written to the header */
  if (ascii_header_set (header, "FILE_NUMBER", "%u", dbdisk->file_number)<0) {
    multilog (log, LOG_ERR, "Error writing FILE_NUMBER\n");
    return -1;
  }

  /* set the current observation id */
  strcpy (dbdisk->obs_id, obs_id);

  /* create the current file name */
  snprintf (dbdisk->file_name, FILENAME_MAX,
	    "%s.%06u.dada", obs_id, dbdisk->file_number);

  /* Get the file size */
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &file_size) != 1) {
    multilog (log, LOG_WARNING, "Header block does not define FILE_SIZE\n");
    file_size = DADA_DEFAULT_FILESIZE;
  }

  /* Open the file */
  fd = disk_array_open (dbdisk->array, dbdisk->file_name,
			file_size, &optimal_bytes);

  if (fd < 0) {
    multilog (log, LOG_ERR, "Error opening %s\n", dbdisk->file_name);
    return -1;
  }

  multilog (log, LOG_INFO, "%s opened for writing %"PRIu64" bytes\n",
	    dbdisk->file_name, file_size);

  prcm->fd = fd;
  prcm->transfer_bytes = file_size;
  prcm->optimal_bytes = 512 * optimal_bytes;

  return 0;
}

/*! Function that closes the data file */
int file_close_function (dada_prc_main_t* prcm, uint64_t bytes_written)
{
  /* the dada_dbdisk specific data */
  dada_dbdisk_t* dbdisk = 0;

  /* status and error logging facility */
  multilog_t* log;

  assert (prcm != 0);

  dbdisk = (dada_dbdisk_t*) prcm->context;
  assert (dbdisk != 0);

  log = prcm->log;
  assert (log != 0);

  if (close (prcm->fd) < 0)
    multilog (log, LOG_ERR, "Error closing %s: %s\n", 
	      dbdisk->file_name, strerror(errno));

  if (!bytes_written)  {

    multilog (log, LOG_ERR, "Removing empty file: %s\n", dbdisk->file_name);

    if (chmod (dbdisk->file_name, S_IRWXU) < 0)
      multilog (log, LOG_ERR, "Error chmod (%s, rwx): %s\n", 
		dbdisk->file_name, strerror(errno));
    
    if (remove (dbdisk->file_name) < 0)
      multilog (log, LOG_ERR, "Error remove (%s): %s\n", 
		dbdisk->file_name, strerror(errno));
    
  }

  return 0;
}

int main (int argc, char **argv)
{
  /* DADA configuration */
  dada_t dada;

  /* DADA Data Block to Disk configuration */
  dada_dbdisk_t dbdisk = DADA_DBDISK_INIT;

  /* DADA Data Block */
  ipcio_t data_block = IPCIO_INIT;

  /* DADA Header Block */
  ipcbuf_t header_block = IPCBUF_INIT;

  /* DADA Primary Read Client main loop */
  dada_prc_main_t* prcm = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Array of disks to which data will be written */
  disk_array_t* disk_array = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  int arg = 0;

  disk_array = disk_array_create ();

  while ((arg=getopt(argc,argv,"dD:vW")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;
      
    case 'D':
      if (disk_array_add (disk_array, optarg) < 0) {
	fprintf (stderr, "Could not add '%s' to disk array\n", optarg);
	return EXIT_FAILURE;
      }
      break;
      
    case 'v':
      verbose=1;
      break;
      
    case 'W':
      disk_array_set_overwrite (disk_array, 1);
      break;

    default:
      usage ();
      return 0;
      
    }


  dada_init (&dada);

  log = multilog_open ("dada_dbdisk", daemon);

  if (daemon) {
    be_a_daemon ();
    multilog_serve (log, dada.log_port);
  }
  else
    multilog_add (log, stderr);

  /* First connect to the shared memory */
  if (ipcbuf_connect (&header_block, dada.hdr_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to header block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_lock_read (&header_block) < 0) {
    multilog (log, LOG_ERR, "Could not lock designated reader status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_connect (&data_block, dada.data_key) < 0) {
    multilog (log, LOG_ERR, "Failed to connect to data block\n");
    return EXIT_FAILURE;
  }

  if (ipcio_open (&data_block, 'R') < 0) {
    multilog (log, LOG_ERR, "Failed to open data block for reading\n");
    return EXIT_FAILURE;
  }

  prcm = dada_prc_main_create ();

  prcm->log = log;

  prcm->data_block = &data_block;
  prcm->header_block = &header_block;

  prcm->open_function = file_open_function;
  prcm->close_function = file_close_function;

  prcm->context = &dbdisk;

  while (!quit) {

    if (dada_prc_main (prcm) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

  }

  /* Disconnect from the shared memory */
  if (ipcio_close (&data_block) < 0) {
    multilog (log, LOG_ERR, "Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcio_disconnect (&data_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from data block\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_unlock_write (&header_block) < 0) {
    multilog (log, LOG_ERR,"Could not unlock designated writer status\n");
    return EXIT_FAILURE;
  }

  if (ipcbuf_disconnect (&header_block) < 0) {
    multilog (log, LOG_ERR, "Failed to disconnect from header block\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
