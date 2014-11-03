/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <float.h>

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_corr_sum [options] output_file\n"
     " -a file      auto-correlation file to sum\n"
     " -c file      cross-correlation file to sum\n"
     " -h           plot this help\n"
     " -v           be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  int arg = 0;

  char ** files = NULL; 

  unsigned nfiles = 0;
  unsigned nacfiles = 0;
  unsigned nccfiles = 0;

  while ((arg=getopt(argc,argv,"a:c:hv")) != -1)
  {
    switch (arg)
    {
      case 'a':
        nfiles++;
        nacfiles++;
        files = (char **) realloc ((void *) files, nfiles * sizeof(char *));
        files[nfiles-1] = (char *) malloc (sizeof(char) * strlen(optarg));
        strcpy (files[nfiles-1], optarg);
        break;

      case 'c':
        nfiles++;
        nccfiles++;
        files = (char **) realloc ((void *) files, nfiles * sizeof(char *));
        files[nfiles-1] = (char *) malloc (sizeof(char) * strlen(optarg));
        strcpy (files[nfiles-1], optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'h':
      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line argument is required\n");
    usage();
    return (EXIT_FAILURE);
  }

  if ((nacfiles > 0) && (nccfiles > 0))
  {
    fprintf (stderr, "ERROR: cannot mix ac and cc files together\n");
    usage();
    return (EXIT_FAILURE);
  }

  if ((nacfiles == 0) && (nccfiles == 0))
  {
    fprintf (stderr, "ERROR: need at least 1 ac or cc file\n");
    usage();
    return (EXIT_FAILURE);
  }

  int flags, perms, fd;
  unsigned ifile, nchan; 

  // the output file to be written
  char ofile[1024];
  strcpy (ofile, argv[optind]);

  // all input files must be the same size
  struct stat buf;
  size_t file_size;
  for (ifile=0; ifile<nfiles; ifile++)
  {
    if (stat (files[ifile], &buf) < 0)
    {
      fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", files[ifile], strerror(errno));
      return (EXIT_FAILURE);
    }

    if (ifile == 0)
      file_size = buf.st_size;
    else
      if (file_size != buf.st_size)
      {
        fprintf (stderr, "ERROR: input file[%d] size [%ld] did not equal size of file 0 [%d]\n", 
                 ifile, buf.st_size, file_size);
        return (EXIT_FAILURE);
      } 
  }

  if (verbose)
    fprintf (stderr, "input filesize s is %d bytes\n", file_size);

  // determine the number of channels 
  size_t element_size = sizeof (float);

  // cc files are complex floats, ac are just floats
  if (nccfiles > 0)
    element_size *= 2;

  nchan = file_size / element_size; 

  float * sum = malloc (nchan * element_size);
  float * in = malloc (nchan * element_size);

  memset(sum, 0, nchan * element_size);

  flags = O_RDONLY;

  size_t nfloat = file_size / sizeof (float);
  size_t i;

  for (ifile=0; ifile<nfiles; ifile++)
  {
    fd = open (files[ifile], flags, perms);
    if (fd < 0)
    {
      fprintf(stderr, "ERROR: failed to open file[%d] %s: %s\n", ifile, files[ifile], strerror(errno));
      free (sum);
      free (in);
      exit(EXIT_FAILURE);
    }

    // read the file to memory
    if (read (fd, (void *) in, file_size) != file_size)
    {
      fprintf(stderr, "ERROR: failed to read %ld bytes from file %s\n", file_size, files[ifile]);
      free (sum);
      free (in);
      close (fd);
      return (EXIT_FAILURE);
    }

    // now simply add each float in file2 to file1
    for (i=0; i<nfloat; i++)
    {
      sum[i] += in[i];
    }
    close (fd);
  }

  flags = O_WRONLY | O_CREAT | O_TRUNC;
  perms = S_IRUSR | S_IWUSR | S_IRGRP;
  fd = open (ofile, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "ERROR: failed to open output file[%s] for writing: %s\n", ofile, strerror(errno));
    return (EXIT_FAILURE);
  }

  if (write (fd, (void *) sum, file_size) != file_size)
  {
    fprintf(stderr, "ERROR: failed to write to %s: %s\n", ofile, strerror(errno));
    close(fd);
    return (EXIT_FAILURE);
  }

  close(fd);
  return (EXIT_SUCCESS);
}
