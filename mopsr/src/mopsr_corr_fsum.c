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
     "mopsr_corr_fsum [options] nant output_file\n"
     " sums AC or CC files in frequency\n"
     " nant         number of nant in CC or AC files\n"
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
  size_t filename_len;

  while ((arg=getopt(argc,argv,"a:c:hv")) != -1)
  {
    switch (arg)
    {
      case 'a':
        nfiles++;
        nacfiles++;
        files = (char **) realloc ((void *) files, nfiles * sizeof(char *));
        filename_len = sizeof(char) * (strlen(optarg) + 1);
        files[nfiles-1] = (char *) malloc (filename_len);
        strncpy (files[nfiles-1], optarg, filename_len);
        break;

      case 'c':
        nfiles++;
        nccfiles++;
        files = (char **) realloc ((void *) files, nfiles * sizeof(char *));
        filename_len = sizeof(char) * (strlen(optarg) + 1);
        files[nfiles-1] = (char *) malloc (filename_len);
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

  if (verbose)
    fprintf (stderr, "parsed command line arguments and options\n");

  int flags, perms;
  unsigned ifile, nchan; 

  int nant;
  if (sscanf(argv[optind], "%d", &nant) != 1)  
  {
    fprintf (stderr, "ERROR: could not parse nant from %s\n", argv[optind]);
    return (EXIT_FAILURE);
  }
  
  size_t sample_stride;
  size_t nbaselines;
  if (nccfiles > 0)
  {
    sample_stride = sizeof(float) * 2;
    nbaselines = (nant * (nant-1)) / 2;
  }
  else
  {
    sample_stride  = sizeof(float);
    nbaselines = nant; 
  }

  // the output file to be written
  char ofile[1024];
  strcpy (ofile, argv[optind+1]);

  // all input files must be the same size
  struct stat buf;
  size_t file_size;
  size_t max_chunk_size = 0;

  flags = O_RDONLY;
  perms = S_IRUSR | S_IRGRP;

  int fds[nfiles];
  size_t chunk_sizes[nfiles];
  int nchan_total = 0;
  
  for (ifile=0; ifile<nfiles; ifile++)
  {
    if (stat (files[ifile], &buf) < 0)
    {
      fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", files[ifile], strerror(errno));
      return (EXIT_FAILURE);
    }

    file_size = buf.st_size;

    // number of channels in this file
    nchan = file_size / (sample_stride * nbaselines);

    // size of a single baseline
    chunk_sizes[ifile] = sample_stride * nchan;
    if (chunk_sizes[ifile] > max_chunk_size)
      max_chunk_size = chunk_sizes[ifile];

#ifdef _DEBUG
    fprintf (stderr, "file_size=%ld nchan=%ld sample_stride=%d nbaselines=%d chunk_size=%ld\n",
              file_size, nchan, sample_stride, nbaselines, chunk_sizes[ifile]);
#endif

    nchan_total += nchan;

    fds[ifile] = open (files[ifile], flags, perms);
    if (!fds[ifile])
    {
      fprintf (stderr, "ERROR: failed to open file[%d] %s for reading\n", ifile, files[ifile]);
      return (EXIT_FAILURE);
    }
  }

  if (verbose)
    fprintf (stderr, "total input channels=%d\n", nchan_total);

  size_t out_file_size = nchan_total * sample_stride * nbaselines;

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

  float * in = (float *) malloc (max_chunk_size);
  unsigned i;
  for (i=0; i<nbaselines; i++)
  {
    for (ifile=0; ifile<nfiles; ifile++)
    {
      const size_t chunk_size = chunk_sizes[ifile];

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
