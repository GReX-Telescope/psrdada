/*
   installs ascii header information in dada files
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "futils.h"

#define APPEND    0
#define MERGE     1
#define OVERWRITE 2

void usage ()
{
  printf ("dada_install_header -H header_filename [-o] data_filename[s]\n"
          "header_filename contains ascii header\n"
          "data_filename contains binary data\n"
          "specify -o to overwrite existing header [default=append]\n");
}

int main (int argc, char** argv)
{
  unsigned header_size = 4096;
  char* new_header = 0;
  char* old_header = 0;
  char* header_filename = 0;
  unsigned todo = APPEND;

  int arg;
  int fd;

  while ((arg = getopt(argc, argv, "hH:o")) != -1) {

    switch (arg)  {
    case 'h':
      usage ();
      return 0;

    case 'H':
      header_filename = optarg;
      break;

    case 'o':
      todo = OVERWRITE;
      break;

    default:
      fprintf (stderr, "unrecognized option\n");
      return -1;

    }

  }

  if (!header_filename)  {
    fprintf (stderr, "Please specify header filename with -H\n");
    return -1;
  }

  new_header = (char*) malloc (header_size);
  old_header = (char*) malloc (header_size);

  if (fileread (header_filename, new_header, header_size) < 0)  {
    fprintf (stderr, "Could not read header from %s\n", header_filename);
    return -1;
  }

  for (arg=optind; arg<argc; arg++)  {

    fprintf (stderr, "Opening %s\n", argv[arg]);
    fd = open(argv[arg], O_RDWR);
    if (fd < 0) {
      fprintf (stderr, "Could not open %s: %s\n", argv[arg], strerror(errno));
      continue;
    }

    fprintf (stderr, "Reading header: %u bytes\n", header_size);

    if (read (fd, old_header, header_size) < header_size) {
      fprintf (stderr, "Could not read header from %s: %s\n", 
               argv[arg], strerror(errno));
      close (fd);
      continue;
    }

    if (todo == APPEND)  {
      strcat (old_header, "\n");
      strcat (old_header, new_header);
    }
    else if (todo == OVERWRITE)
      strcpy (old_header, new_header);
    else {
      fprintf (stderr, "Unsupported mode: internal error\n");
      return -1;
    }

    if (lseek (fd, 0, SEEK_SET) != 0) {
      fprintf (stderr, "Could not seek to start of %s: %s\n",
               argv[arg], strerror(errno));
      close (fd);
      continue;
    }


    fprintf (stderr, "Writing header: %u bytes\n", header_size);

    if (write (fd, old_header, header_size) < header_size) {
      fprintf (stderr, "Could not write header to %s: %s\n",
               argv[arg], strerror(errno));
    }

    close (fd);
  }

  fprintf (stderr, "All files completed\n");
  return 0;

}

