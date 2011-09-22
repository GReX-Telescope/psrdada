/* To enable the use of O_DIRECT */
#define _GNU_SOURCE


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>

#define DEFAULT_CHUNK_SIZE 8225280
#define DEFAULT_FILE_SIZE_GB 24

void stats_thread(void * arg);
double diff_time ( struct timeval time1, struct timeval time2 );


/* #define _DEBUG 1 */
int quit_threads = 0;

void usage()
{
  fprintf (stdout,
    "test_diskperf [options] file\n"
    " file       file to read/write\n"
    " -o         use O_DIRECT flag to bypass kernel buffering\n"
    " -r         read back file [default write file]\n"
    " -c size    write file in size byte chunks [default %d]\n"
    " -b size    write file of size GB [default %d]\n"
    " -s         print realtime statistics\n"
    " -v         verbose output\n", DEFAULT_CHUNK_SIZE, DEFAULT_FILE_SIZE_GB);
}

typedef struct {

  /* file descriptor */
  int fd;

  /* number of bytes currently read/written */
  uint64_t bytes;

  /* total number of bytes to read/write */
  uint64_t total_bytes;

  /* number of gigabytes to write */
  int file_size_gbytes;

  /* file name of the file */
  char * file_name; 

  /* O_DIRECT flag */ 
  int o_direct;

  /* size of each write operation */ 
  long chunk_size;

  /* read/write flag */
  int read;

  /* buffer for read/write data */
  char * buffer;

  /* verbosity flag */
  int verbose;

} dada_diskperf_t;

#define DADA_DISKPERF_INIT { 0,0,0,0,0,0,0,0,0,0 }


/*! Function that opens the data transfer target */
int file_open_function (dada_diskperf_t* diskperf)
{

  /* malloc memory for read/write operations */
  assert (diskperf->chunk_size > 0);
  if (posix_memalign ( (void **) &(diskperf->buffer), 512, diskperf->chunk_size) != 0) {
    if (! diskperf->buffer ) {
      fprintf(stderr, "Failed to allocated %d bytes of aligned "
                "memory: %s\n", diskperf->chunk_size, strerror(errno));
      return -1;
    }
  }

  diskperf->fd = -1;

  /* flags/permissions for file operations */
  int flags = 0;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP;

  if (diskperf->read) {
    flags = O_RDONLY;

  } else {
    flags = O_WRONLY | O_CREAT | O_TRUNC;
    if (diskperf->o_direct) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
      if (diskperf->chunk_size % 512 != 0) {
        fprintf(stderr, "chunk size must be a multiple of 512 bytes\n");
        return -1;
      }
    }
  }

  diskperf->fd = open(diskperf->file_name, flags, perms);

  if (diskperf->fd < 0) {
    fprintf(stderr, "Error opening %s: %s\n",
              diskperf->file_name, strerror(errno));
    return -1;
  }

#ifndef O_DIRECT
  if (diskperf->o_direct)
    fcntl (diskperf->fd, F_NOCACHE, 1);
#endif

  /* determine the file size */
  if (diskperf->read) {

    struct stat filestat;
    int status = 0;
    status = fstat(diskperf->fd, &filestat);
    if (status != 0) {
      fprintf(stderr, "Failed to stat '%s': %s\n", 
                diskperf->file_name, strerror(errno));
      close (diskperf->fd);
      return -1;
    }
    diskperf->total_bytes = (uint64_t) filestat.st_size;
    if (diskperf->verbose) {
      fprintf(stderr, "%s opened for reading %"PRIu64" bytes\n",
                diskperf->file_name, diskperf->total_bytes);
    }

  } else {

    diskperf->total_bytes = ((uint64_t) diskperf->file_size_gbytes) * (1024*1024*1024);

    if (diskperf->total_bytes % diskperf->chunk_size != 0) {
      diskperf->total_bytes = (diskperf->total_bytes / diskperf->chunk_size) * diskperf->chunk_size;
      if (diskperf->verbose) {
        fprintf(stderr, "Adjusting file size to %"PRIu64" bytes so that it is "
                  "a multiple of the chunk size\n", diskperf->total_bytes);
      }
    }
      
    if (diskperf->verbose) {
      fprintf(stderr, "%s opened for writing %"PRIu64" bytes in "
                "%d byte chunks\n", diskperf->file_name, diskperf->total_bytes,
                diskperf->chunk_size);
    }
  }
  diskperf->bytes = 0;

  return 0;
}

/*! Function that closes the data file */
int file_close_function (dada_diskperf_t* diskperf)
{

  assert (diskperf != 0);

  if (close (diskperf->fd) < 0) {
    fprintf(stderr, "Error closing %s: %s\n", diskperf->file_name, strerror(errno));
    return -1;
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t file_io_function (dada_diskperf_t* diskperf)
{
  ssize_t bytes_iod  = 0;
  size_t bytes_to_io = 0;

  if (diskperf->read) {

    while ( diskperf->bytes < diskperf->total_bytes ) {

      bytes_to_io = diskperf->total_bytes - diskperf->bytes;

      if (bytes_to_io > diskperf->chunk_size)
        bytes_to_io = diskperf->chunk_size;

      bytes_iod = read (diskperf->fd, diskperf->buffer, bytes_to_io);

      if (bytes_iod == bytes_to_io) {
        diskperf->bytes += bytes_iod;

      } else if (bytes_iod < 0) {
        fprintf(stderr, "failed to read %d bytes: %s\n",
                  bytes_to_io, strerror(errno));
        return -1;

      } else {
        fprintf(stderr, "unexpected EOF\n");
        diskperf->bytes = diskperf->total_bytes;
      }
    }

  } else { 

    while ( diskperf->bytes < diskperf->total_bytes ) {

      bytes_to_io = diskperf->chunk_size;

      bytes_iod = write (diskperf->fd, diskperf->buffer, bytes_to_io);

      if (bytes_iod == bytes_to_io) {
        diskperf->bytes += bytes_iod;
  
      } else if (bytes_iod < 0) {
        fprintf(stderr, "failed to write %d bytes: %s\n",
                  bytes_to_io, strerror(errno));
        return -1;

      } else {
        fprintf(stderr, "write returned 0 bytes\n");
        diskperf->bytes = diskperf->total_bytes;
      }
    }
  }

  if (diskperf->verbose) {
    fprintf(stderr, "IO loop complete\n");
  }

  return 0;
}


int main (int argc, char **argv)
{

  dada_diskperf_t diskperf = DADA_DISKPERF_INIT;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* chunk size */
  int chunk_size_bytes = DEFAULT_CHUNK_SIZE;

  /* total file size [GB] */
  int file_size_gbytes = DEFAULT_FILE_SIZE_GB;

  /* flag to read file instead of write */
  int read = 0;

  /* O_DIRECT flag */
  int o_direct = 0;

  int arg = 0;

  /* stats thread flag */
  int stats = 0;
  pthread_t stats_thread_id;


  while ((arg=getopt(argc,argv,"b:c:orsv")) != -1)
    switch (arg) {

    case 'b':
      file_size_gbytes= atoi(optarg);
      break;

    case 'c':
      chunk_size_bytes = atoi(optarg);
      break;

    case 'o':
      o_direct = 1;
      break;

    case 'r':
      read = 1;
      break;

    case 's':
      stats = 1;
      break;

    case 'v':
      verbose = 1;
      break;
      
    default:
      usage ();
      return 0;
    }

  /* check for filename */
  if ((argc - optind) != 1) {
    fprintf(stderr, "Error: file to read/write must be specified\n\n");
    usage();
    return EXIT_FAILURE;
  } else {
    diskperf.file_name = strdup(argv[optind]);
  }

  diskperf.verbose = verbose;
  diskperf.o_direct = o_direct;
  diskperf.read = read;
  diskperf.chunk_size = (long) chunk_size_bytes;
  diskperf.file_size_gbytes = file_size_gbytes;

  if (file_open_function(&diskperf) < 0) {
    fprintf (stderr, "failed to open file: %s\n", diskperf.file_name);
    return (EXIT_FAILURE);
  }

  if (stats) {
    int rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &diskperf);
    if (rval != 0) {
       fprintf(stderr, "Error creating stats_thread: %s\n", strerror(rval));
       stats_thread_id = 0;
    }
  }

  struct timeval start_time;
  struct timeval end_time;

  gettimeofday (&start_time, 0);

  if (file_io_function(&diskperf) < 0) {
    fprintf(stderr, "IO function failed\n");
  }

  gettimeofday (&end_time, 0);

  if (file_close_function(&diskperf) < 0) {
    fprintf(stderr, "Failed to close file\n");
    return EXIT_FAILURE;
  }

  quit_threads = 1;

  if (stats && stats_thread_id) {
    void * result = 0;
    pthread_join (stats_thread_id, &result);
  }

  double time_taken = diff_time (start_time, end_time);
  fprintf(stderr, "Time %5.3f s, Rate: %5.3f [10^6B/s] %5.3f [MB/s]\n", time_taken, (((double)diskperf.total_bytes) / time_taken)/(1000000), (((double)diskperf.total_bytes) / time_taken)/(1024*1024));

  return EXIT_SUCCESS;
}

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  dada_diskperf_t * ctx = (dada_diskperf_t *) arg;

  uint64_t b_io_total = 0;
  uint64_t b_io_curr = 0;
  uint64_t b_io_1sec = 0;
  double mb_io_ps = 0;
  double ten6b_io_ps = 0;
  double pc_done = 0.0;

  while (!quit_threads)
  { 
    /* get a snapshot of the data as quickly as possible */
    b_io_curr    = ctx->bytes;

    /* calc the values for the last second */
    b_io_1sec =  b_io_curr - b_io_total;

    /* update the totals */
    b_io_total = b_io_curr;

    ten6b_io_ps = (double) b_io_1sec / 1000000;
    mb_io_ps = (double) b_io_1sec / (1024*1024);
    pc_done = ((double) b_io_curr / (double) ctx->total_bytes) * 100;

    fprintf(stderr, "Rate %04.1f [10^6B/s] %4.1f [MB/s] %5.2f%%\n", ten6b_io_ps, mb_io_ps, pc_done);

    sleep(1);
  }
}

/* 
 *  Compute difference in timeval
 */
double diff_time ( struct timeval time1, struct timeval time2 )
{
  return ( (double)( time2.tv_sec - time1.tv_sec ) +
            ( (double)( time2.tv_usec - time1.tv_usec ) / 1000000.0 ) );
}


