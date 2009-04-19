//#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <inttypes.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

void usage(void);

#define DEFAULT_DELETION_RATE 64

int main (int argc, char **argv)
{

  /* flag set in verbose mode */
  char verbose = 0;

  /* list of file names to delete */
  char ** f_names = 0;

  /* file sizes of file names */
  uint64_t *f_sizes = 0;

  /* file descriptors of files */
  int *fds = 0;

  /* deletion rate in MB/s */
  unsigned rate = DEFAULT_DELETION_RATE;

  /* Amount to delete between 1 second sleeps */
  uint64_t truncate_chunk = rate*1024*1024;

  /* text file of filenames to be deleted */
  char* metafile = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"r:hM:v")) != -1)
    switch (arg) {
     
    case 'r': 
      rate = atoi(optarg);
      truncate_chunk = rate*1024*1024;
      break; 

    case 'M':
      metafile = optarg;
      break;

    case 'h':
      usage();
      return(0);

    case 'v':
      verbose=1;
      break;

    default:
      usage ();
      return 0;
      
  }

  if ((rate <= 0) || (rate > 1024)) {
    fprintf(stderr, "rate must be between 0 and 1024\n");
    usage();
    exit(EXIT_FAILURE);
  }

  if ((argc - optind) < 1 && !metafile) {
    fprintf(stderr, "slow_rm: at least one file must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  }

  struct stat filestat;
  int status = 0;

  unsigned n_files = 0;
  int file_i = 0;
  int i = 0;
  unsigned error = 0;

  if (!metafile) {
    n_files = argc-optind;

    if (verbose)
      fprintf(stdout, "%d files provided on the command line\n", n_files);

    fds = (int *) malloc(sizeof(int) * n_files);
    f_names = (char **) malloc(sizeof(char *) * n_files);
    f_sizes = (uint64_t *) malloc(sizeof(uint64_t) * n_files);

    if ((!fds) || (!f_names) || (!f_sizes)) {
      fprintf(stderr, "Could not malloc memory\n");
      return(EXIT_FAILURE);
    }

    for (file_i=optind; file_i<argc; file_i++, i++)
      f_names[i] = strdup(argv[file_i]);
  }
  else {

    FILE* fptr = fopen (metafile, "r");
    if (!fptr) {
      fprintf (stderr, "Could not open '%s' for reading: %s\n", f_names[i], strerror(errno));
      return(EXIT_FAILURE);
    }

    char name [FILENAME_MAX+1];
    while (fgets (name, FILENAME_MAX, fptr)) {
      n_files ++;
      f_names = (char **) realloc (f_names, sizeof(char *) * n_files);
      if (!f_names) {
	fprintf(stderr, "Could not realloc memory\n");
	return(EXIT_FAILURE);
      }
      char* newline = strchr (name, '\n');
      if (newline) *newline = '\0';

      f_names[i] = strdup(name);
      i++;
    }

    fds = (int *) malloc(sizeof(int) * n_files);
    f_sizes = (uint64_t *) malloc(sizeof(uint64_t) * n_files);
    if ((!fds) || (!f_sizes)) {
      fprintf(stderr, "Could not malloc memory\n");
      return(EXIT_FAILURE);
    }

  }

  unsigned file_start = 0;
  unsigned file_end = 0;

  while (file_start < n_files)
  {
    for (i=file_start; i < n_files; i++)
    {
      /* try to open the file for writing */
      fds[i] = open(f_names[i], O_WRONLY);
      if (fds[i] < 0) {

	if (errno == EMFILE)
	  break;

	fprintf(stderr, "Failed to open '%s' for writing: %s\n", f_names[i], strerror(errno));
	return(EXIT_FAILURE);
      }

      /* stat the file to check we can delete it */
      status = fstat(fds[i], &filestat);
      if (status != 0) {
	fprintf(stderr, "Failed to stat '%s': %s\n", f_names[i], strerror(errno));
	return(EXIT_FAILURE);
      }

      if (S_ISDIR(filestat.st_mode)) {
	printf("Cannot handle %s (directory)\n", argv[file_i]);
	return(EXIT_FAILURE);
      }

      f_sizes[i] = (uint64_t) filestat.st_size;
      if (verbose) {
	fprintf(stdout, "%s: %"PRIu64" bytes\n", f_names[i], f_sizes[i]);
      }

      file_end ++;
    }

    int64_t new_length = 0;
    unsigned n_ops = 0;
    unsigned i_op = 0;
    int j = 0;

    if (verbose) 
      fprintf(stdout, "Truncating %d files at %d MB/s\n", file_end-file_start, rate);

  /* Now begin truncating each file */
    for (i=file_start; i<file_end; i++) {

      if (verbose) 
	fprintf(stderr, "%s [", f_names[i]);

    i_op = 0;
    n_ops = f_sizes[i] / truncate_chunk;
    if (n_ops == 0) 
      n_ops = 1;
    if (verbose) {
      for (j=0; j<n_ops; j++) {
        fprintf(stderr, " ");  
      }
      fprintf(stderr, "]");
    }

    while (f_sizes[i] > 0) { 

      new_length = f_sizes[i] - truncate_chunk; 
      if (new_length < 0)
        new_length = 0;
      //if (verbose) {
      //  fprintf(stdout, "truncating %s from %"PRIu64" to %"PRIu64"\n",
      //                  f_names[i], f_sizes[i], new_length);
      //}

      status = ftruncate(fds[i], new_length);
      if (status < 0) {
        fprintf(stderr, "truncating failed\n");
      }

      f_sizes[i] = new_length;

      i_op++;

      if (verbose) {
        fprintf(stderr, "\r%s [", f_names[i]);
        for (j=0; j<n_ops; j++) {
          if (j < i_op)
            fprintf(stderr, ".");
          else
            fprintf(stderr, " ");
        }
        fprintf(stderr, "]");
      }

      if (f_sizes[i] > 0) 
        sleep(1);
    }
    if (verbose)
      fprintf(stderr, "\n");
  }

  /* close all the files */
  for (i=file_start; i<file_end; i++) {
    if (fds[i])
      close(fds[i]);
  }

  for (i=file_start; i<file_end; i++) {
    if (f_sizes[i] == 0) {
      status = unlink(f_names[i]);
      if (status < 0) 
        fprintf(stderr, "could not delete truncated file: %s\n", f_names[i]);
    } else {
      fprintf(stderr, "file %s was not truncated correctly\n", f_names[i]);
    }
  }

  file_start = file_end;
  }

  return EXIT_SUCCESS;
}


void usage()
{
  fprintf (stdout,
     "slow_rm [options] files\n"
     " -h       print usage\n"
     " -r rate  deletion rate in MB/s [default %u]\n"
     " -v       verbose output\n", DEFAULT_DELETION_RATE);
}
