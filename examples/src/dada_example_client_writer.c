/***************************************************************************
 *  
 *    Copyright (C) 2017 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/


#include <time.h>
#include <assert.h>
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
#include "dada_client.h"
#include "multilog.h"
#include "ipcio.h"
#include "ascii_header.h"

int example_dada_client_writer_open (dada_client_t* client);
int64_t example_dada_client_write (dada_client_t* client, void* data, uint64_t data_size);
int64_t example_dada_client_writer_write_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id);
int example_dada_client_writer_close (dada_client_t* client, uint64_t bytes_written);

typedef struct 
{
  dada_hdu_t * hdu;
  multilog_t * log;      // logging interface
  char * header_file;    // file containing DADA header
  char * obs_header;     // file containing DADA header
  char header_written;   // flag for header I/O
} example_client_writer_t;

void usage()
{
  fprintf (stdout,
     "example_dada_client_writer [options] header\n"
     " -k key       hexadecimal shared memory key  [default: %x]\n"
     "header        DADA header file contain obs metadata\n",
     DADA_DEFAULT_BLOCK_KEY);
}

/*! Function that opens the data transfer target */
int example_dada_client_writer_open (dada_client_t* client)
{
  assert (client != 0);
  example_client_writer_t * ctx = (example_client_writer_t *) client->context;
  assert(ctx != 0);

  ctx->obs_header = (char *) malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);
  if (!ctx->obs_header)
  {
    multilog (ctx->log, LOG_ERR, "could not allocate memory\n");
    return (EXIT_FAILURE);
  }

  // read the ASCII DADA header from the file
  if (fileread (ctx->header_file, ctx->obs_header, DADA_DEFAULT_HEADER_SIZE) < 0)
  {
    free (ctx->obs_header);
    multilog (ctx->log, LOG_ERR, "could not read ASCII header from %s\n", ctx->header_file); 
    return (EXIT_FAILURE);
  }
  ctx->header_written = 0;
}


/*! Transfer header/data to data block */
int64_t example_dada_client_writer_write (dada_client_t* client, void* data, uint64_t data_size)
{
  assert (client != 0);
  example_client_writer_t * ctx = (example_client_writer_t *) client->context;
  assert(ctx != 0);

  if (!ctx->header_written)
  {
    // write the obs_header to the header_block
    uint64_t header_size = ipcbuf_get_bufsz (ctx->hdu->header_block);
    char * header = ipcbuf_get_next_write (ctx->hdu->header_block);
    memcpy (header, ctx->obs_header, header_size);

    // flag the header block for this "obsevation" as filled
    if (ipcbuf_mark_filled (ctx->hdu->header_block, header_size) < 0)
    {
      multilog (ctx->log, LOG_ERR, "could not mark filled Header Block\n");
      return -1;
    }
    ctx->header_written = 0;
  }
  else
  {
    // write data to the data_size bytes to the data_block
    memset (data, 0, data_size); 
  }
  return data_size;
}

/*! Transfer data to data block, 1 block only */
int64_t example_dada_client_writer_write_block (dada_client_t* client, void* data, uint64_t data_size, uint64_t block_id)
{
  assert (client != 0);
  example_client_writer_t * ctx = (example_client_writer_t *) client->context;
  assert(ctx != 0);

  // write 1 block of data
  memset (data, 0, data_size);

  return data_size;
}


/*! Function that closes socket */
int example_dada_client_writer_close (dada_client_t* client, uint64_t bytes_written)
{
  assert (client != 0);
  example_client_writer_t * ctx = (example_client_writer_t *) client->context;
  assert(ctx != 0);

  free (ctx->obs_header);
  return 0;
}


int main (int argc, char **argv)
{
  // contextual struct
  example_client_writer_t ctx;

  // DADA Primary Read Client main loop
  dada_client_t* client = 0;

  // dada key for SHM 
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;

  while ((arg=getopt(argc,argv,"hk:")) != -1)
  {
    switch (arg) 
    {
      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) {
          fprintf (stderr, "ERROR: could not parse key from %s\n", optarg);
          return EXIT_FAILURE;
        }
        break;

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
    exit(EXIT_FAILURE);
  }

  ctx.header_file = strdup(argv[optind]);

  // create a multilogger
  ctx.log = multilog_open ("example_dada_client_writer", 0);

  // set the destination for multilog to stderr
  multilog_add (ctx.log, stderr);

  // create the HDU struct
  ctx.hdu = dada_hdu_create (ctx.log);

  // set the key to connecting to the HDU
  dada_hdu_set_key (ctx.hdu, dada_key);

  // connect to HDU
  if (dada_hdu_connect (ctx.hdu) < 0)
  {
    multilog (ctx.log, LOG_ERR, "could not connect to HDU\n");
    return EXIT_FAILURE;
  }

  // lock as writer on the HDU
  if (dada_hdu_lock_write (ctx.hdu) < 0)
  {
    multilog (ctx.log, LOG_ERR, "could not lock write on HDU\n");
    return EXIT_FAILURE;
  }

  client = dada_client_create ();

  client->context = &ctx;
  client->log = ctx.log;

  client->data_block   = ctx.hdu->data_block;
  client->header_block = ctx.hdu->header_block;

  client->open_function     = example_dada_client_writer_open;
  client->io_function       = example_dada_client_writer_write;
  client->io_block_function = example_dada_client_writer_write_block;
  client->close_function    = example_dada_client_writer_close;

  client->direction         = dada_client_writer;

  if (dada_client_write (client) < 0)
  {
    multilog (ctx.log, LOG_ERR, "Error during transfer\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_unlock_write (ctx.hdu) < 0)
  {
    multilog (ctx.log, LOG_ERR, "could not unlock read on hdu\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_disconnect (ctx.hdu) < 0)
  {
    multilog (ctx.log, LOG_ERR, "could not disconnect from HDU\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

