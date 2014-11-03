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
#include <complex.h>
#include <float.h>

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_corr_fscr [options] nant input output\n"
     " nant         number of antenna in input file\n"
     " input        input filename\n"
     " output       output filename\n"
     " -a           interpret files as auto-correlation data\n"
     " -F nchan     fscrunch the output to nchan channels [default 1 channel]\n"
     " -h           plot this help\n"
     " -v           be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  int arg = 0;

  int nchan_out = 1;

  char auto_corr = 0;
  
  while ((arg=getopt(argc,argv,"aF:hv")) != -1)
  {
    switch (arg)
    {
      case 'a':
        auto_corr = 1;
        break;
        
      case 'F':
        nchan_out = atoi(optarg); 
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
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n");
    usage();
    return (EXIT_FAILURE);
  }

  const int nant = atoi(argv[optind]);
  const int nbaselines = (nant * (nant-1))/2;

  // the output file to be written
  char ifile[1024];
  strcpy (ifile, argv[optind+1]);

  char ofile[1024];
  strcpy (ofile, argv[optind+2]);
  if (verbose)
    fprintf (stderr, "input=%s output=%s\n", ifile, ofile);

  // all input files must be the same size
  struct stat buf;
  size_t file_size_in;
  if (stat (ifile, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat file [%s]: %s\n", ifile, strerror(errno));
    return (EXIT_FAILURE);
  }
  file_size_in = buf.st_size;

  if (verbose)
    fprintf (stderr, "input filesize is %d bytes\n", file_size_in);

  // determine the number of channels 
  unsigned nchan_in;
  size_t element_size;

  if (auto_corr)
  {
    element_size = sizeof (float);
    nchan_in = file_size_in / (nant * element_size); 
  }
  else
  {
    element_size = sizeof (complex float);
    nchan_in = file_size_in / (nbaselines * element_size); 
  }

  if (nchan_in % nchan_out != 0)
  {
    fprintf (stderr, "ERROR: output nchan must be a divisor of the input nchan [%d]\n", nchan_in);
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "nchan_in=%u nchan_out=%u\n", nchan_in, nchan_out);

  // check the output can be opened for writin
  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;

  int fd = open (ifile, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "ERROR: failed to open input file [%s]: %s\n", ifile, strerror(errno));
    exit(EXIT_FAILURE);
  }
  
  void * in = malloc (file_size_in);
  if (!in)
  {
    fprintf (stderr, "ERROR: failed to allocate %d bytes for out\n", file_size_in);
    return (EXIT_FAILURE);
  }

  // read in the first file to memory
  if (read (fd, in, file_size_in) != file_size_in)
  {
    fprintf(stderr, "ERROR: failed to read %ld bytes from file %s\n", file_size_in, ifile);
    free (in);
    close (fd);
    return (EXIT_FAILURE);
  }

  close (fd);

  unsigned ichan, iscr;
  const unsigned scrunch_factor = nchan_in / nchan_out;
  const size_t file_size_out = file_size_in / scrunch_factor;
  if (verbose)
  {
    fprintf (stderr, "file_size_out=%ld\n", file_size_out);
  }

  void * out = malloc (file_size_out);
  if (!out)
  {
    fprintf (stderr, "ERROR: failed to allocate %d bytes for out\n", file_size_out);
    return (EXIT_FAILURE);
  }

  if (auto_corr)
  {
    float sum;
    float * in_ptr = (float *) in;
    float * out_ptr = (float *) out;

    unsigned iant;
    for (iant=0; iant<nant; iant++)
    {
      // zap DC channel
      in_ptr[nchan_in/2] = 0;
      for (ichan=0; ichan<nchan_out; ichan++)
      {
        sum = 0;
        for (iscr=0; iscr<scrunch_factor; iscr++)
        {
          sum += *in_ptr;
          in_ptr++;
        }
        out_ptr[ichan] = sum;
      }
      out_ptr += nchan_out;
    }
  }
  else
  {
    complex float sum;
    complex float * in_ptr = (complex float *) in;
    complex float * out_ptr = (complex float *) out;

    unsigned ibaseline;
    for (ibaseline=0; ibaseline<nbaselines; ibaseline++)
    {
      // zap DC channel
      in_ptr[nchan_in/2] = 0;
      for (ichan=0; ichan<nchan_out; ichan++)
      {
        sum = 0;
        for (iscr=0; iscr<scrunch_factor; iscr++)
        {
          sum += (*in_ptr);
          in_ptr++;
        }
        out_ptr[ichan] = sum;
      }
      out_ptr += nchan_out;
    }
  }
  
  free (in);
  in = 0;

  flags = O_WRONLY | O_CREAT | O_TRUNC;
  perms = S_IRUSR | S_IWUSR | S_IRGRP;
  fd = open (ofile, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "ERROR: failed to open output file [%s] for writing: %s\n", ofile, strerror(errno));
    free (out);
    out = 0;
    return (EXIT_FAILURE);
  }

  if (write (fd, (void *) out, file_size_out) != file_size_out)
  {
    fprintf(stderr, "ERROR: failed to write file[%s] for writing: %s\n", ofile, strerror(errno));
    free (out);
    out = 0;
    close (fd);
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf (stderr, "closing output file\n");
  close (fd);

  if (verbose)
    fprintf (stderr, "freeing out [%p]\n", (void *) out);
  free (out);

  return (EXIT_SUCCESS);
}
