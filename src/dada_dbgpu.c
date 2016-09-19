/***************************************************************************
 *  
 *    Copyright (C) 2011 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_cuda.h"

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

typedef struct {

  memory_mode_t mode;

  int device;

  unsigned verbose;

  uint64_t bytes_transferred;

  uint64_t time_msec;

  void * device_memory;

  size_t device_memory_bytes;

  cudaStream_t stream;

} dada_dbgpu_t;

#define DADA_DBGPU_INIT { PINNED, 0, 0, 0, 0, 0, 0 }

int dbgpu_init (dada_dbgpu_t* ctx, multilog_t* log);
int dbgpu_open (dada_client_t* client);
int dbgpu_close (dada_client_t* client, uint64_t bytes_written);
int64_t dbgpu_send (dada_client_t* client, void * buffer, uint64_t bytes);
int64_t dbgpu_send_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id);
int64_t dbgpu_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode);

void usage ()
{
	fprintf(stdout,
    "dada_dbgpu - transfer data from DADA buffer to CUDA gpu\n"
    "\n"
    " -d id   use GPU device with id, default 0\n"
    " -k key  hexadecimal shared memory key, default %x\n"
    " -m      used pageable memory, default pinned\n"
    " -s      process single DADA transfer then exit\n"
    " -v      verbose output\n", DADA_DEFAULT_BLOCK_KEY
  );

}


int main(int argc, char** argv) 
{

  /* dbgpu contextual struct */
  dada_dbgpu_t dbgpu = DADA_DBGPU_INIT;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  int arg = 0;

  unsigned quit = 0;

  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  // default values
  dbgpu.mode = PINNED;
  dbgpu.verbose = 0;
  dbgpu.device = 0;
  
  while ((arg = getopt(argc, argv, "d:ihk:msv")) != -1) 
  {
    switch (arg)  
    {
      case 'd':
        dbgpu.device = atoi(optarg);
        break;
      
      case 'h':
        usage ();
        return 0;

      case 'k':
        if (sscanf (optarg, "%x", &dada_key) != 1) 
        {
          fprintf (stderr, "dada_dbgpu: could not parse key from %s\n", optarg);
          return -1;
        }
        break;
    
      case 'm':
        dbgpu.mode = PAGEABLE;
        break;

      case 's':
        quit = 1;
        break;

      case 'v':
        dbgpu.verbose ++;
        break;

      default:
        usage ();
        return 0;
    }
  }

  log = multilog_open ("dada_dbgpu", 0); 

  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  dada_hdu_set_key(hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  dbgpu_init (&dbgpu, log);

  // ensure that we register the DADA DB buffers as Cuda Host memory
  if (dada_cuda_dbregister(hdu) < 0)
  {
    fprintf (stderr, "failed to register DADA buffers as pinned memory\n");
    return EXIT_FAILURE;
  }

  client->log = log;

  client->data_block        = hdu->data_block;
  client->header_block      = hdu->header_block;

  client->open_function     = dbgpu_open;
  client->io_function       = dbgpu_send;
  client->io_block_function = dbgpu_send_block;
  client->close_function    = dbgpu_close;
  client->direction         = dada_client_reader;

  client->context = &dbgpu;

  while (!client->quit)
  {
    if (dbgpu.verbose)
      multilog(client->log, LOG_INFO, "main: dada_client_read()\n");
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (dbgpu.verbose)
      multilog(client->log, LOG_INFO, "main: dada_unlock_read()\n");
    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      quit = 1;
    }

    if (quit)
      client->quit = 1;
    else
    {
      if (dbgpu.verbose)
        multilog(client->log, LOG_INFO, "main: dada_lock_read()\n");
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dbgpu.verbose)
    multilog(client->log, LOG_INFO, "main: dada_hdu_disconnect()\n");

  if (dada_cuda_dbunregister (hdu) < 0)
  {
    multilog (client->log, LOG_ERR, "failed to unregister DADA DB buffers\n");
    return EXIT_FAILURE;
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

/*! Perform initialization */
int dbgpu_init ( dada_dbgpu_t* ctx, multilog_t* log )
{
  // select the gpu device
  int n_devices = dada_cuda_get_device_count();
  multilog (log, LOG_INFO, "Detected %d CUDA devices\n", n_devices);

  if ((ctx->device < 0) && (ctx->device >= n_devices))
  {
    multilog (log, LOG_ERR, "dbgpu_init: no CUDA devices available [%d]\n",
              n_devices);
    return -1;
  }

  if (dada_cuda_select_device (ctx->device) < 0)
  {
    multilog (log, LOG_ERR, "dbgpu_init: could not select requested device [%d]\n",
              ctx->device);
    return -1;
  }

  char * device_name = dada_cuda_get_device_name (ctx->device);
  if (!device_name)
  {
    multilog (log, LOG_ERR, "dbgpu_init: could not get CUDA device name\n");
    return -1;
  }
  multilog (log, LOG_INFO, "Using device %d : %s\n", ctx->device, device_name);

  // create a stream for these operations
  cudaError_t error = cudaStreamCreate(&(ctx->stream));
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "aqdsp_init: could not create CUDA stream\n");
    return -1;
  }
  multilog (log, LOG_INFO, "Using stream %d\n", ctx->stream);

  free(device_name);
}
 

int dbgpu_open (dada_client_t* client)
{
  assert (client != 0);
  dada_dbgpu_t* dbgpu = (dada_dbgpu_t*) client->context;

  if (dbgpu->verbose)
    multilog (client->log, LOG_INFO, "dbgpu_open()\n");

  client->transfer_bytes = 0;
  client->optimal_bytes = 0;

  return 0;

}

int dbgpu_close (dada_client_t* client, uint64_t bytes_written)
{

  assert (client != 0);
  dada_dbgpu_t* dbgpu = (dada_dbgpu_t*) client->context;

  if (dbgpu->verbose)
    multilog (client->log, LOG_INFO, "dbgpu_close()\n");

  multilog (client->log, LOG_INFO, "Host to Device Bandwidth for device %d\n", dbgpu->device);

  if (dbgpu->mode == PAGEABLE)
    multilog (client->log, LOG_INFO, "Paged memory\n");
  else
    multilog (client->log, LOG_INFO, "Pinned memory\n");

  multilog (client->log, LOG_INFO, "   bytes_transferred = %"PRIu64"\n", dbgpu->bytes_transferred);
  multilog (client->log, LOG_INFO, "   time_msec = %"PRIu64"\n", dbgpu->time_msec);

  multilog (client->log, LOG_INFO, "   Transfer Size (GB)\tBandwidth(MB/s)\n");

  double mb_transferred = ((double) dbgpu->bytes_transferred / (1024*1024));

  double seconds_elapsed = ((double) dbgpu->time_msec / 1000);

  double bandwidthInMBs =  mb_transferred / seconds_elapsed;

  double gbytes_transferred = mb_transferred / 1024.0; 
  
  multilog (client->log, LOG_INFO, "   %lf\t\t%lf\n", gbytes_transferred, bandwidthInMBs);

  return 0;
}

/*! used for transferring the header, and uses pageable memory */
int64_t dbgpu_send (dada_client_t* client, void * buffer, uint64_t bytes)
{
  return dbgpu_transfer (client, buffer, (size_t) bytes, PAGEABLE);
}

int64_t dbgpu_send_block (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  return dbgpu_transfer (client, buffer, (size_t) bytes, PINNED);
}

int64_t dbgpu_transfer (dada_client_t* client, void * buffer, size_t bytes, memory_mode_t mode)
{
  assert (client != 0);
  dada_dbgpu_t* dbgpu = (dada_dbgpu_t*) client->context;

  // check that we have enough space on device to write to
  if (bytes > dbgpu->device_memory_bytes)
  {
    if (dbgpu->device_memory)
      if (dada_cuda_device_free (dbgpu->device_memory) < 0)
      {
        multilog (client->log, LOG_ERR, "dbgpu_send: dada_cuda_device_free failed\n");
        return -1;
      }

    dbgpu->device_memory = dada_cuda_device_malloc ((size_t) bytes);
    dbgpu->device_memory_bytes = bytes;
    if (!dbgpu->device_memory)
    {
      multilog (client->log, LOG_ERR, "dbgpu_send: dada_cuda_device_malloc failed\n");
      return -1;
    }
  }

  float msec_elapsed = dada_cuda_device_transfer (buffer, dbgpu->device_memory, bytes, mode, dbgpu->stream);

  if (mode == PINNED)
  {
    if (dbgpu->verbose)
      multilog (client->log, LOG_INFO, "dbgpu_transfer: %ld bytes in %f mili seconds\n", bytes, msec_elapsed);
    dbgpu->bytes_transferred += bytes;
    dbgpu->time_msec += (uint64_t) msec_elapsed;
  }

  return (int64_t) bytes;
}
