/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson & Ewan Barr
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/
   
#define DSB 1

#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_cuda.h"
#include "mopsr_def.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include "xgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>
#include <complex.h>
#include <float.h>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>

#include <cufft.h>

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage()
{
  fprintf (stdout,
           "mopsr_dbxgpu [options]\n"
           " -c core   bind process to CPU core\n"
           " -d id     run on GPU device\n"
           " -h        display usage\n"
           " -k key    input DADA shm key [default %x]\n"
           " -t dump   number of seconds per correlation dump [default 30]\n"
           " -s        1 transfer, then exit\n"
           " -v        verbose mode\n",
           DADA_DEFAULT_BLOCK_KEY
           );
}

typedef struct {

  multilog_t * log;

  char order[4];

  // verbose output
  int verbose;

  unsigned nchan;

  unsigned npol;

  unsigned ndim;

  unsigned nant;

  uint64_t obs_offset;

  uint64_t block_size;

  int device;             // cuda device to use

  XGPUContext xgpu_context;

  uint64_t bytes_per_second; 
 
  unsigned dump_time;       // how often (s) to dump CC and AC spectra 
  unsigned blocks_per_dump; // number of blocks in each file dump
  unsigned dump_counter;  // counter for dumps or something!
  uint64_t byte_counter; //counter for total number of bytes read

  char utc_start[20]; //time stamp for output files

  Complex * tmp;
  size_t tmp_size;

  size_t nbaselines;

  float *   ac_data;
  size_t    ac_data_size;

  Complex * cc_data;
  size_t    cc_data_size;

  ComplexInput * inbuf;
  size_t inbuf_size;

  unsigned xgpu_npol;
  unsigned xgpu_nstation;
  unsigned xgpu_nfrequencies;
  unsigned xgpu_ntime;
  unsigned xgpu_npulsar;

} mopsr_dbxgpu_t;


int dbxgpu_init (mopsr_dbxgpu_t * ctx, dada_hdu_t * in_hdu);
unsigned dbxgpu_parse_info (char * line);
int dbxgpu_destroy (mopsr_dbxgpu_t * ctx, dada_hdu_t * in_hdu);

#define MOPSR_DBXGPU_INIT { 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

//function to write output in raw format
int dbxgpu_dump_spectra (dada_client_t* client, uint64_t start_byte, uint64_t end_byte)
{
  mopsr_dbxgpu_t * ctx = (mopsr_dbxgpu_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;
  cudaError_t error;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra()\n");

  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  char filename_tmp[512];
  char filename[512];
  sprintf (filename_tmp,"%s_%020lu_%020lu.xc.tmp", ctx->utc_start, start_byte, end_byte);
  sprintf (filename,"%s_%020lu_%020lu.xc", ctx->utc_start, start_byte, end_byte);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opening %s\n", filename_tmp);
  int fd = open (filename_tmp, flags, perms);
  if (!fd)
  {
    multilog (log, LOG_ERR, "Could not open file: %s\n", filename_tmp);
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opened file %s\n", filename_tmp);

  write (fd, ctx->xgpu_context.matrix_h, ctx->tmp_size * sizeof(Complex));

  close (fd);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: closed file %s\n", filename_tmp);

  rename (filename_tmp, filename);
  return 0;
}

//function to write ac and cc files
int dbxgpu_dump_spectra_old (dada_client_t* client, uint64_t start_byte, uint64_t end_byte)
{
  mopsr_dbxgpu_t * ctx = (mopsr_dbxgpu_t *) client->context;

  // status and error logging facilty 
  multilog_t* log = client->log;
  cudaError_t error;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra()\n");

  int acfile, ccfile;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;
  char acfilename [512];
  char ccfilename [512];
  sprintf(acfilename,"%s_%020lu_%020lu.ac.tmp", ctx->utc_start, start_byte, end_byte);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opening %s\n", acfilename);
  acfile = open (acfilename, flags, perms);
  if (!acfile)
  {
    multilog (log, LOG_ERR, "Could not open file: %s\n", acfilename);
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Opened file %s\n",acfilename);

  sprintf (ccfilename, "%s_%020lu_%020lu.cc.tmp", ctx->utc_start, start_byte, end_byte);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opening %s\n", ccfilename);
  ccfile = open(ccfilename, flags, perms);
  if (!ccfile)
  {
    multilog (log, LOG_ERR, "Could not open file: %s\n", ccfilename);
    return -1;
  }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Opened file %s\n",ccfilename);

  // here is where the trickery begins to understand the xGPU internal format!
 
  // xGPU will use it's internal REGISTER_TILE_TRIANGULAR_ORDERordering, so this
  // should be converted to the AC and CC formats we are used to

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: reordering REGISTER_TILE -> TRIANGLE\n");
  int f, i, rx, j, ry, pol1, pol2;
  size_t matLength = ctx->xgpu_nfrequencies * ((ctx->xgpu_nstation+1)*(ctx->xgpu_nstation/2)*ctx->xgpu_npol*ctx->xgpu_npol) * (ctx->xgpu_npulsar + 1);
  for (i=0; i<ctx->xgpu_nstation/2; i++) 
  {
    for (rx=0; rx<2; rx++) 
    {
      for (j=0; j<=i; j++) 
      {
        for (ry=0; ry<2; ry++) 
        {
          int k = f*(ctx->xgpu_nstation+1)*(ctx->xgpu_nstation/2) + (2*i+rx)*(2*i+rx+1)/2 + 2*j+ry;
          int l = f*4*(ctx->xgpu_nstation/2+1)*(ctx->xgpu_nstation/4) + (2*ry+rx)*(ctx->xgpu_nstation/2+1)*(ctx->xgpu_nstation/4) + i*(i+1)/2 + j;
          for (pol1=0; pol1<ctx->xgpu_npol; pol1++)
          {
            for (pol2=0; pol2<ctx->xgpu_npol; pol2++)
            {
              size_t tri_index = (k*ctx->xgpu_npol+pol1)*ctx->xgpu_npol+pol2;
              size_t reg_index = (l*ctx->xgpu_npol+pol1)*ctx->xgpu_npol+pol2;
              ctx->tmp[tri_index].real = ((float*) (ctx->xgpu_context.matrix_h))[reg_index];
              ctx->tmp[tri_index].imag = ((float*) (ctx->xgpu_context.matrix_h))[reg_index+matLength];
            }
          }
        }
      }
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: reordering TRIANGLE -> CUSTOM\n");
  // now we have auto and cross correlations in ctx->tmp in xGPU TRIANGULAR ORDER
  for (i=0; i<ctx->xgpu_nstation; i++)
  {
    for (j=0; j<=i; j++)
    {
      for (pol1=0; pol1<ctx->xgpu_npol; pol1++)
      {
        for (pol2=0; pol2<ctx->xgpu_npol; pol2++)
        {
          for(f=0; f<ctx->xgpu_nfrequencies; f++)
          {
            int k = f*(ctx->xgpu_nstation+1)*(ctx->xgpu_nstation/2) + i*(i+1)/2 + j;
            int index = (k*ctx->xgpu_npol+pol1)*ctx->xgpu_npol+pol2;

            // convert i + pol1 and j + pol2 to our single pol antenna number
            int ant1 = (i * 2) + pol1;
            int ant2 = (j * 2) + pol2;

            // if auto-correlation
            if (ant1 == ant2)
            {
              ctx->ac_data[ant1] = ctx->tmp[index].real;
            }
            else
            {
              int baseline = 0;
              int ib, jb;
              for (ib=0; ib<ctx->nant; ib++)
              {
                for (jb=0; jb<ctx->nant; jb++)
                {
                  if (ib != jb)
                  {
                    if (ib == ant1 && jb == ant2)
                    {
                      ctx->cc_data[baseline] = ctx->tmp[index];
                    }
                    baseline++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: writing AC\n");
  write (acfile, ctx->ac_data, ctx->ac_data_size);

  close(acfile);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Closed file %s\n",acfilename);
  
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: writing CC\n");
  write (ccfile, ctx->cc_data, ctx->cc_data_size);
  close(ccfile);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Closed file %s\n",ccfilename);

  // rename files from temp names to real names
  char command[1024];
  sprintf (command,  "mv %s_%020lu_%020lu.ac.tmp %s_%020lu_%020lu.ac", ctx->utc_start, start_byte, end_byte, ctx->utc_start, start_byte, end_byte);
  system (command);

  sprintf (command,  "mv %s_%020lu_%020lu.cc.tmp %s_%020lu_%020lu.cc", ctx->utc_start, start_byte, end_byte, ctx->utc_start, start_byte, end_byte);
  system (command);
  
  return 0;
}

/*! Function that opens the data transfer target */
int dbxgpu_open (dada_client_t* client)
{
  mopsr_dbxgpu_t * ctx = (mopsr_dbxgpu_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "dbxgpu_open()\n");

  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_ERR, "header had no NCHAN\n");
    return -1;
  }
  if (ctx->nchan != 1)
  {
    multilog (log, LOG_ERR, "header specified NCHAN == %d, should be 1\n", ctx->nchan);
    return -1;
  } 

  if (ascii_header_get (client->header, "ctx->xgpu_npol", "%d", &(ctx->npol)) != 1)
  {
    multilog (log, LOG_ERR, "header had no ctx->xgpu_npol\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    multilog (log, LOG_ERR, "header had no NANT\n");
    return -1;
  }

  if (ascii_header_get (client->header, "NDIM", "%d", &(ctx->ndim)) != 1)
  {
    multilog (log, LOG_WARNING, "header had no NDIM\n");
    return -1;
  }

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "header had no ORDER\n");
    return -1;
  }

  if (ascii_header_get (client->header, "OBS_OFFSET", "%"PRIu64, &(ctx->obs_offset)) != 1)
  {
    multilog (log, LOG_ERR, "header had no OBS_OFFSET\n");
    return -1;
  }

  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &(ctx->bytes_per_second)) != 1)
  {
    multilog (log, LOG_ERR, "header had no BYTES_PER_SECOND\n");
    return -1;
  }

  if (ascii_header_get (client->header, "UTC_START", "%s", &(ctx->utc_start)) != 1)
  {
    multilog (log, LOG_ERR, "header had no UTC_START\n");
    return -1;
  }

  if (strcmp(ctx->order, "ST") != 0)
  {
    multilog (log, LOG_ERR, "ORDER [%s] was not ST\n", ctx->order);
    return -1;
  }

  if (ctx->nant != (ctx->xgpu_nstation * ctx->xgpu_npol))
  {
    multilog (log, LOG_ERR, "NANT [%u] does not match xGPUs compiled size [%u]\n",
              ctx->nant, (ctx->xgpu_nstation * ctx->xgpu_npol));
    return -1;
  }
 
  unsigned nbytes_per_samp = ctx->nchan * ctx->nant * ctx->ndim * ctx->npol;
  if ((ctx->block_size / nbytes_per_samp) % ctx->xgpu_ntime != 0)
  {
    multilog (log, LOG_ERR, "open: not an even number of xGPU gulps per block\n");
    return -1;
  }

  
  ctx->ac_data_size = ctx->nant * sizeof(float);
  ctx->ac_data = (float *) malloc (ctx->ac_data_size);

  ctx->nbaselines = (ctx->nant * (ctx->nant-1)) / 2;
  ctx->cc_data_size = ctx->nbaselines * sizeof(Complex);
  ctx->cc_data = (Complex *) malloc (ctx->cc_data_size);

  ctx->tmp_size = ctx->xgpu_nfrequencies * ((ctx->xgpu_nstation/2+1)*(ctx->xgpu_nstation/4)*ctx->xgpu_npol*ctx->xgpu_npol*4) * (ctx->xgpu_npulsar + 1);
  ctx->tmp = (Complex *) malloc (ctx->tmp_size * sizeof(Complex));

  ctx->inbuf_size = ctx->xgpu_ntime * ctx->nant * ctx->ndim * ctx->npol;
  ctx->inbuf = (ComplexInput *) malloc (ctx->inbuf_size);

  float block_length_s = ((float) ctx->block_size) / (float) ctx->bytes_per_second;
  ctx->blocks_per_dump = (unsigned ) (ctx->dump_time / block_length_s + 0.5);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NCHAN=%d NANT=%d, NDIM=%d ctx->xgpu_npol=%d\n", ctx->nchan, ctx->nant, ctx->ndim, ctx->npol);

  client->transfer_bytes = 0;
  client->optimal_bytes = 64*1024*1024;
  client->header_transfer = 0;

  return 0;
}

unsigned dbxgpu_parse_info (char * line)
{
  unsigned value;
  const char * sep = ":";
  char * saveptr;
  char * str;

  str = strtok_r (line, sep, &saveptr);
  if (str == NULL)
    return 0;
  str = strtok_r (NULL, sep, &saveptr);
  if (str == NULL)
    return 0;
  if (sscanf(str, " %u", &value) != 1)
    return 0;
  else
    return value;
}

/*! Function that closes the data transfer */
int dbxgpu_close (dada_client_t* client, uint64_t bytes_read)
{
  // the mopsr_dbxgpu specific data
  mopsr_dbxgpu_t* ctx = 0;

  // status and error logging facility
  multilog_t* log;

  cudaError_t error;

  assert (client != 0);

  ctx = (mopsr_dbxgpu_t*) client->context;

  assert (ctx != 0);

  log = client->log;
  assert (log != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_read=%"PRIu64"\n", bytes_read);
  
  if (ctx->dump_counter%ctx->blocks_per_dump != 0)
    {
      //dump remainder of data to file
      uint64_t start_byte = ctx->obs_offset + ctx->byte_counter;
      uint64_t end_byte = start_byte + ctx->block_size * (ctx->dump_counter % ctx->blocks_per_dump); 
      if (dbxgpu_dump_spectra(client,start_byte,end_byte) != 0){
        return -1;
      }
    }
  return 0;
}

/*! Pointer to the function that transfers data to/from the target via direct block IO*/
int64_t dbxgpu_write_block (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t in_block_id)
{
  assert (client != 0);
  mopsr_dbxgpu_t* ctx = (mopsr_dbxgpu_t*) client->context;
  multilog_t * log = client->log;

  if (ctx->verbose) 
    multilog (log, LOG_INFO, "write_block: processing %"PRIu64" bytes\n", in_data_size);

  int64_t bytes_read = in_data_size;

  const unsigned frame_size = (ctx->ndim * ctx->nant);

  const uint64_t nframe = in_data_size / frame_size;

  //fprintf (stderr, "nframe=%d\n", nframe);

  if (in_data_size % frame_size)
    multilog (log, LOG_ERR, "input block size [%"PRIu64"] % %d = %d, mismatch!!\n", in_data_size, frame_size, in_data_size % frame_size);
    
  return bytes_read;
}

int64_t dbxgpu_block_gpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_dbxgpu_t* ctx = (mopsr_dbxgpu_t*) client->context;

  multilog_t * log = client->log;
  
  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: buffer=%p, bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, bytes, block_id);

  // if this is not a full block, then just return
  if (bytes != ctx->block_size)
  {
    multilog (log, LOG_INFO, "block_gpu: non-full block of %"PRIu64" bytes, ignoring\n", bytes);
    return (int64_t) bytes;
  } 

  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: nant=%u ndim=%u\n", ctx->nant, ctx->ndim);
  const uint64_t nsamp = bytes / (ctx->nant * ctx->ndim);
  const unsigned nant = ctx->nant;

  ComplexInput * buf = ctx->xgpu_context.array_h;
  ComplexInput * in =  (ComplexInput *) buffer;

  ctx->dump_counter++;
  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "block_gpu: dump_counter=%lu blocks_per_dump=%lu\n", ctx->dump_counter, ctx->blocks_per_dump);
  const char dump_at_end = (ctx->dump_counter % ctx->blocks_per_dump == 0);

  uint64_t isamp, osamp, iblock;
  unsigned iant;
  int xgpu_error;

  const unsigned block_nantsamp = nant * ctx->xgpu_ntime;

  // prepare the specified number of time samples for xGPU to consume
  const uint64_t nblock = bytes / (block_nantsamp * ctx->ndim);
  for (iblock=0; iblock<nblock; iblock++)
  {
    // copy the input from main memory to a smallish buffer
    for (iant=0; iant<nant; iant++)
    {
      memcpy (ctx->inbuf, in + (iant * nsamp), ctx->xgpu_ntime * ctx->ndim);

      // transpose
      for (isamp=0; isamp<ctx->xgpu_ntime; isamp++)
      {
        buf[isamp*nant + iant] = ctx->inbuf[isamp];
      }
    }

    in += ctx->xgpu_ntime;

#ifdef _DEBUG
    multilog (log, LOG_INFO, "block_gpu: reordered %d samps, caling CudaXengine\n", osamp);
#endif
    if (dump_at_end && iblock == (nblock - 1))
    {
      xgpu_error = xgpuCudaXengine (&(ctx->xgpu_context), SYNCOP_DUMP);
    }
    else
    {
      xgpu_error = xgpuCudaXengine (&(ctx->xgpu_context), SYNCOP_SYNC_COMPUTE);
    }
#ifdef _DEBUG
    multilog (log, LOG_INFO, "block_gpu: CudaXengine done, xgpu_error=%d\n", xgpu_error);
#endif
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: finished processing block %lu, dump=%d\n", block_id, dump_at_end);

  //test if we want to dump
  if (dump_at_end)
  {
    uint64_t start_byte = ctx->obs_offset + ctx->byte_counter;
    uint64_t end_byte = start_byte + ctx->block_size * ctx->blocks_per_dump;

    if (dbxgpu_dump_spectra(client, start_byte, end_byte) !=0)
    {
      multilog (log, LOG_ERR, "dbxgpu_block_gpu: could not dump spectra\n");
      return -1;
    }

    // zero the integration buffer in xGPU library
    xgpuClearDeviceIntegrationBuffer (&(ctx->xgpu_context));
    ctx->byte_counter += ctx->block_size * ctx->blocks_per_dump;
  }


  return (int64_t) bytes;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbxgpu_write (dada_client_t* client, void* data, uint64_t data_size)
{
  fprintf(stderr, "dbxgpu_write should be disabled!!!!!\n");

  return data_size;
}

int dbxgpu_init (mopsr_dbxgpu_t * ctx, dada_hdu_t * in_hdu)
{
  multilog_t * log = ctx->log;

  ctx->dump_counter = 0;

  // output DB block size must be bit_p times the input DB block size
  ctx->block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);

  if (ctx->device >= 0)
  {
    int xgpu_error = 0;

    // parse xgpuinfo from compiled program
    FILE *fp = popen ("xgpuinfo", "r");
    if (fp == NULL)
    {
      multilog (log, LOG_ERR, "Failed to run xgpuinfo\n");
      return -1;
    }

    char line[160];

    // parse key information from xGPU library configuration
    while (fgets(line, sizeof(line)-1, fp) != NULL)
    {
      if (strcmp (line, "Number of polarizations:") == 0)
        ctx->xgpu_npol = dbxgpu_parse_info (line);
      if (strcmp (line, "Number of stations:") == 0)
        ctx->xgpu_nstation = dbxgpu_parse_info (line);
      if (strcmp (line, "Number of frequencies:") == 0)
        ctx->xgpu_nfrequencies = dbxgpu_parse_info (line);
      if (strcmp (line, "Number of time samples per GPU integration:") == 0)
        ctx->xgpu_ntime = dbxgpu_parse_info (line);
    }
    pclose (fp);
    ctx->xgpu_npulsar = 1;

    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: xGPU configured for %u antenna "
                "and integration length of %u\n", 
                ctx->xgpu_nstation * ctx->xgpu_npol,
                ctx->xgpu_ntime);

    // note this is pinned memory allocated by xGPU library
    ctx->xgpu_context.array_h = NULL;
    ctx->xgpu_context.matrix_h = NULL;

    xgpu_error = xgpuInit(&(ctx->xgpu_context), ctx->device);
    if (xgpu_error)
    {
      multilog (log, LOG_ERR, "xgpuInit returned error code %d\n", xgpu_error);
      return -1;
    }


    // ensure that we register the DADA DB buffers as Cuda Host memory
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: registering input HDU buffers\n");
    if (dada_cuda_dbregister(in_hdu) < 0)
    {
      fprintf (stderr, "failed to register in_hdu DADA buffers as pinned memory\n");
      return -1;
    }
  }

  return 0;
}

int dbxgpu_destroy (mopsr_dbxgpu_t * ctx, dada_hdu_t * in_hdu)
{
  if (ctx->device >= 0)
  {
    // free gpu memory
    xgpuFree(&(ctx->xgpu_context));

    if (dada_cuda_dbunregister (in_hdu) < 0)
    {
      multilog (ctx->log, LOG_ERR, "failed to unregister input DADA buffers\n");
      return -1;
    }
  }
  return 0;
}


int main (int argc, char **argv)
{
  /* DADA Data Block to Disk configuration */
  mopsr_dbxgpu_t dbxgpu = MOPSR_DBXGPU_INIT;

  // input data block HDU key
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // number of seconds per dump
  // this is rounded to an integer number of blocks
  unsigned dump_time = 30;

  // number of ffts to perform on each channel
  unsigned nfft = 1024;

  // core to run on
  int core = -1;

  // cuda device to use, default CPU
  int device = -1;

  int arg = 0;


  while ((arg=getopt(argc,argv,"c:d:t:hk:sv")) != -1)
  {
    switch (arg) 
    {
    case 'c':
      core = atoi(optarg);
      break;
      
    case 'd':
      device = atoi(optarg);
      break;

    case 't':        
      dump_time = atoi(optarg);
      break;
      
    case 'h':
      usage();
      return (EXIT_SUCCESS);
      
    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr, "dada_dbxgpu: could not parse key from %s\n", optarg);
        return -1;
      }
      break;
      
    case 's':
      single_transfer = 1;
      break;

    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return 0;
      
    }
  }
  
  dbxgpu.dump_time = dump_time;
  dbxgpu.verbose = verbose;
  dbxgpu.device = device;

  int num_args = argc-optind;
  unsigned i = 0;
   
  if (num_args != 0)
  {
    fprintf(stderr, "mopsr_dbxgpu: no command line arguments expected\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  log = multilog_open ("mopsr_dbxgpu", 0);
  multilog_add (log, stderr);

  dbxgpu.log = log;

  if (verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // open connection to the in/read DB
  hdu = dada_hdu_create (log);

  dada_hdu_set_key (hdu, dada_key);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", dada_key);

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (verbose > 1)
    multilog (log, LOG_INFO, "main: dbxgpu_init()\n");
  if (dbxgpu_init (&dbxgpu, hdu) < 0)
  {
    multilog (log, LOG_ERR, "failed to initialized required resources\n");
    dada_hdu_disconnect (hdu);
    return EXIT_FAILURE;
  }

  if (verbose > 1)
    multilog (log, LOG_INFO, "main: preparing dada client\n");
  client = dada_client_create ();

  client->log = log;

  client->data_block   = hdu->data_block;
  client->header_block = hdu->header_block;

  client->open_function  = dbxgpu_open;
  client->io_function    = dbxgpu_write;
  if (device >= 0)
    client->io_block_function = dbxgpu_block_gpu;
  else
    client->io_block_function = dbxgpu_write_block;

  client->close_function = dbxgpu_close;
  client->direction      = dada_client_reader;

  client->context = &dbxgpu;
  client->quiet = 0;

  while (!client->quit)
  {
    if (verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

