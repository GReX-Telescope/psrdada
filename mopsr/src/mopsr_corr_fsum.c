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
     "mopsr_corr_fsum [options] npt output_file\n"
     " sums AC or CC files in frequency\n"
     " npt          number of fine channels in input files\n"
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
  if (argc-optind != 2)
  {
    fprintf(stderr, "ERROR: 2 command line arguments are required\n");
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

  int flags, perms;
  unsigned ifile, nchan; 

  int npt;
  if (sscanf(argv[optind], "%d", &npt) != 1)  
  {
    fprintf (stderr, "ERROR: could not parse npt from %s\n", argv[optind]);
    return (EXIT_FAILURE);
  }

  // the output file to be written
  char ofile[1024];
  strcpy (ofile, argv[optind+1]);

  // all input files must be the same size
  struct stat buf;
  size_t file_size;

  flags = O_RDONLY;
  perms = S_IRUSR | S_IRGRP;

  int fds[nfiles];
  
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

    fds[ifile] = open (files[ifile], flags, perms);
    if (!fds[ifile])
    {
      fprintf (stderr, "ERROR: failed to open file[%d] %s for reading\n", ifile, files[ifile]);
      return (EXIT_FAILURE);
    }
  }

  if (verbose)
    fprintf (stderr, "input filesize is %d bytes\n", file_size);

  size_t out_file_size = file_size * nfiles;

  // output file
  flags = O_WRONLY | O_CREAT | O_TRUNC;
  perms = S_IRUSR | S_IWUSR | S_IRGRP;
  int ofd = open (ofile, flags, perms);
  if (ofd < 0)
  {
    fprintf(stderr, "ERROR: failed to open output file[%s] for writing: %s\n", ofile, strerror(errno));
    return (EXIT_FAILURE);
  }

  flags = O_RDONLY;

  unsigned ichunk;
  unsigned nchunk;
  unsigned chunk_size;

  if (nccfiles > 0)
  {
    chunk_size = npt * sizeof(float) * 2;
  }
  else
  {
    chunk_size = npt * sizeof(float);
  }

  nchunk = (unsigned) (file_size / (chunk_size));

  float * in = malloc (chunk_size);

  for (ichunk=0; ichunk<nchunk; ichunk++)
  {
    for (ifile=0; ifile<nfiles; ifile++)
    {
      // read the file to memory
      if (read (fds[ifile], (void *) in, chunk_size) != chunk_size)
      {
        fprintf(stderr, "ERROR: failed to read %ld bytes from file %s\n", chunk_size, files[ifile]);
        free (in);
        for (ifile=0; ifile<nfiles; ifile++)
          close(fds[ifile]);
        close(ofd);
        return (EXIT_FAILURE);
      }

      if (write (ofd, (void *) in, chunk_size) != chunk_size)
      {
        fprintf(stderr, "ERROR: failed to write %ld bytes to %s: %s\n", chunk_size, ofile, strerror(errno));
        free (in);
        for (ifile=0; ifile<nfiles; ifile++)
          close(fds[ifile]);
        close(ofd);
        return (EXIT_FAILURE);
      }
    }
  }
  for (ifile=0; ifile<nfiles; ifile++)
    close(fds[ifile]);
  close (ofd);

  return (EXIT_SUCCESS);
}
