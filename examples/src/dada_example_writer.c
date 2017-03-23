/***************************************************************************
 *  
 *    Copyright (C) 2017 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

/*
 * example_dada_writer
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>


// DADA includes for this example
#include "futils.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ascii_header.h"

void usage()
{
  fprintf (stdout,
     "example_dada_writer [options] header_file\n"
     " -k key     shared memory key [default %x]\n"
     " -h         print help text\n",
     DADA_DEFAULT_BLOCK_KEY);
}

int main (int argc, char **argv)
{
  /* DADA Logger */ 
  multilog_t * log = 0;

  // default shared memory key
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  // DADA Header + Data unit
  dada_hdu_t * hdu = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hk:")) != -1)
  {
    switch (arg)
    {
      case 'h':
        usage();
        return 0;
        
      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1)
        {
          fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
          return -1;
        }

      default:
        usage ();
        return 0;
        
    }
  }

  // check the header file was supplied
  if ((argc - optind) != 1)
  {
    fprintf (stderr, "ERROR: header must be specified\n");
    usage();
    return (EXIT_FAILURE);
  }

  char * header_file = strdup(argv[optind]);
  char * obs_header = (char *) malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);
  if (!obs_header)
  {
    fprintf (stderr, "ERROR: could not allocate memory\n");
    return (EXIT_FAILURE);
  }
    
  // read the ASCII DADA header from the file
  if (fileread (header_file, obs_header, DADA_DEFAULT_HEADER_SIZE) < 0)
  {
    free (obs_header);
    fprintf (stderr, "ERROR: could not read ASCII header from %s\n", header_file);
    return (EXIT_FAILURE);
  }

  // create a multilogger
  log = multilog_open ("example_dada_writer", 0);

  // set the destination for multilog to stderr
  multilog_add (log, stderr);

  // create the HDU struct
  hdu = dada_hdu_create (log);

  // set the key to connecting to the HDU
  dada_hdu_set_key (hdu, dada_key);

  // connect to HDU
  if (dada_hdu_connect (hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not connect to HDU\n");
    return EXIT_FAILURE;
  }

  // lock as writer on the HDU
  if (dada_hdu_lock_write (hdu) < 0)
  {
    multilog (log, LOG_ERR, "could not lock write on HDU\n");
    return EXIT_FAILURE;
  }

  // write the obs_header to the header_block
  {
    uint64_t header_size = ipcbuf_get_bufsz (hdu->header_block);
    char * header = ipcbuf_get_next_write (hdu->header_block);
    memcpy (header, obs_header, header_size);

    // flag the header block for this "obsevation" as filled
    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)
    {
      multilog (log, LOG_ERR, "could not mark filled Header Block\n");
      return EXIT_FAILURE;
    }
  }

  // the size of 1 block (buffer element) in the data block
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t*) hdu->data_block);

  // write 1 block worth of data block via the "block" method
  {
    uint64_t block_id;
    char * block = ipcio_open_block_write (hdu->data_block, &block_id);
    if (!block)
    {
      multilog (log, LOG_ERR, "ipcio_open_block_write failed\n");
      return EXIT_FAILURE;
    }

    memset (block, 0, block_size);

    if (ipcio_close_block_write (hdu->data_block, block_size) < 0)
    {
      multilog (log, LOG_ERR, "ipcio_close_block_write failed\n");
      return EXIT_FAILURE;
    }
  }

  // write 1.5 blocks wort of data via the standard method
  {
    uint64_t local_buffer_size = (block_size * 3) / 2;
    char * local_buffer = (char *) malloc (local_buffer_size);
    if (!local_buffer)
    {
      multilog (log, LOG_ERR, "failed to allocate %lu bytes\n", local_buffer_size);
      return EXIT_FAILURE;
    }

    memset (local_buffer, 0, local_buffer_size);
    ipcio_write (hdu->data_block, local_buffer, local_buffer_size); 
    free (local_buffer);
  }

  // unlock write access from the HDU, performs implicit EOD
  if (dada_hdu_unlock_write (hdu) < 0)
  {
    multilog (log, LOG_ERR, "dada_hdu_unlock_write failed\n");
    return -1;
  }

  // disconnect from HDU
  if (dada_hdu_disconnect (hdu) < 0)
    multilog (log, LOG_ERR, "could not unlock write on hdu\n");

  return EXIT_SUCCESS;
}
    
