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
#include "mopsr_calib_cuda.h"

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
#include <cuda_runtime.h>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>

#include <cufft.h>

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

void usage()
{
  fprintf (stdout,
           "mopsr_dbcalib [options]\n"
           " -c core   bind process to CPU core\n"
           " -d id     run on GPU device\n"
           " -h        display usage\n"
           " -k key    input DADA shm key [default %x]\n"
           " -n nfft   batch size for ffts [default 1024]\n"
           " -p ant    add specified ant as primary ant [default no primaries]\n"
           " -t dump   number of seconds per correlation dump [default 30]\n"
           " -s        1 transfer, then exit\n"
           " -v        verbose mode\n"
           " -z        use spectral kurtosis [M=1024]\n",
           DADA_DEFAULT_BLOCK_KEY
           );
}

typedef struct {

  multilog_t * log;

  char order[4];

  // verbose output
  int verbose;

  // number of points in fft
  unsigned batch_size;

  cufftHandle fft_plan_forward;

  unsigned nchan;

  unsigned npol;

  unsigned ndim;

  unsigned nant;

  uint64_t obs_offset;

  uint64_t block_size;

  int device;             // cuda device to use

  cudaStream_t stream;    // cuda stream for engine

#ifndef PER_CHANNEL_TRANSFERS
  void * d_raw;            // device memory for input
#endif

  void * d_in;            // device memory for input

  void * d_unpacked;      // device memory for unpacked data
  size_t d_unpacked_bytes;

  void * d_fftd;           // device memory for output

  void * d_cross_power;    // device memory for cross power spectra
  size_t d_cross_power_bytes;  

  void * h_cross_power; // host memory for cross power spectra
  
  unsigned npairs;            // number of baseline pairs to correlate 
  unsigned npairs_total;      // total possible npairs
  mopsr_baseline_t* d_pairs;  // device memory for baseline pairs 
  mopsr_baseline_t* h_pairs;  // host memory for baseline pairs

  //EBARR 2014 edit for user defined baselines
  // if nprimaries == 0 then all baselines will be computed
  unsigned * primary_antennas;  // host array for parsing primary antennas from command line
  unsigned nprimaries;          // size of primary_antennas array
  unsigned * baseline_mask; // 0 or 1 mask

  // This is a bit fudgy, as the autocorr mask is required to identify when 
  // not correlated baselines (zero arrays) should be written to the ac or
  // the cc file.
  unsigned * autocorr_mask; // 0 = cc and 1 = ac
  unsigned nmasked; // number of masked baselines

  uint64_t bytes_per_second; 
 
  unsigned dump_time;       // how often (s) to dump CC and AC spectra 
  unsigned blocks_per_dump; // number of blocks in each file dump
  unsigned dump_counter;  // counter for dumps or something!
  uint64_t byte_counter; //counter for total number of bytes read

  char skzap;

  char utc_start[20]; //time stamp for output files

  void * d_buffer;
  size_t d_buffer_size;

} mopsr_dbcalib_t;


int dbcalib_init (mopsr_dbcalib_t * ctx, dada_hdu_t * in_hdu);
int dbcalib_destroy (mopsr_dbcalib_t * ctx, dada_hdu_t * in_hdu);

#define MOPSR_DBCALIB_INIT { 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0}

//function to write ac and cc files
int dbcalib_dump_spectra (dada_client_t* client, uint64_t start_time, uint64_t end_time)
{
  mopsr_dbcalib_t * ctx = (mopsr_dbcalib_t *) client->context;

  // status and error logging facilty 
  multilog_t* log = client->log;
  cudaError_t error;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra()\n");

  size_t bytes_to_copy = sizeof(cuFloatComplex) * ctx->npairs * ctx->batch_size * ctx->nchan;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: cudaMemcpyAsync copying back %ld bytes\n", bytes_to_copy);
  error = cudaMemcpyAsync (ctx->h_cross_power, ctx->d_cross_power, bytes_to_copy, cudaMemcpyDeviceToHost, ctx->stream);
  if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "cudaMemcpyAsync D2H failed: %s\n", cudaGetErrorString(error));
      return -1;
    }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: cudaStreamSynchronize()\n");
  cudaStreamSynchronize(ctx->stream);

  FILE *acfile, *ccfile;
  char acfilename [512];
  char ccfilename [512];
  sprintf(acfilename,"%s_%010lu_%010lu.ac.tmp", ctx->utc_start, start_time, end_time);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opening %s\n", acfilename);
  acfile = fopen(acfilename, "w");
  if (!acfile)
    {
      multilog (log, LOG_ERR, "Could not open file: %s\n", acfilename);
      return -1;
    }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Opened file %s\n",acfilename);

  sprintf (ccfilename, "%s_%010lu_%010lu.cc.tmp", ctx->utc_start, start_time, end_time);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: opening %s\n", ccfilename);
  ccfile = fopen(ccfilename, "w");
  if (!ccfile)
    {
      multilog (log, LOG_ERR, "Could not open file: %s\n", ccfilename);
      return -1;
    }
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Opened file %s\n",ccfilename);

  int ii,jj,kk;
  cuFloatComplex * cp_spectra = (cuFloatComplex *) ctx->h_cross_power;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dump_spectra: writing output data..\n");

  int pair_idx = 0;

#ifdef DSB

  const unsigned half_batch = ctx->batch_size / 2;

#ifdef HIRES
  for (ii=0;ii<ctx->npairs_total;ii++)
  {
    // In the case we want to write empty data
    if (ctx->baseline_mask[ii] == 0)
    {
      if (ctx->autocorr_mask[ii] == 1)
      {
        float zero = 0.0;
        for (jj=0; jj<ctx->nchan; jj++)
          for (kk=0; kk<ctx->batch_size; kk++)
            fwrite( &zero, sizeof(float), 1, acfile);
      }
      else
      {
        cuFloatComplex zero = make_cuFloatComplex(0.0,0.0);
        for (jj=0; jj<ctx->nchan; jj++)
          for (kk=0; kk<ctx->batch_size; kk++)
            fwrite(&zero, sizeof(cuFloatComplex), 1, ccfile);
      }
    }
    else
    {
      if (ctx->autocorr_mask[ii] == 1)
      {
        for (jj=0; jj<ctx->nchan; jj++)
        {
          uint64_t offset = jj * ctx->batch_size * ctx->npairs;
          for (kk=half_batch; kk<ctx->batch_size; kk++)
            fwrite( &(cp_spectra[offset + ctx->batch_size*pair_idx+kk].x), sizeof(float), 1, acfile);
          for (kk=0; kk<half_batch; kk++)
            fwrite( &(cp_spectra[offset + ctx->batch_size*pair_idx+kk].x), sizeof(float), 1, acfile);
        }
      }
      else
      {
        // offset within the channel for this baseline
        uint64_t offset = ctx->batch_size * pair_idx;
        uint64_t chan_stride = ctx->batch_size * ctx->npairs;

        for (jj=0; jj<ctx->nchan; jj++)
        {
          // zap the DC bin of each channel
          //cp_spectra[offset + half_batch].x = 0;
          //cp_spectra[offset + half_batch].y = 0;

          //multilog (log, LOG_INFO, "dump_spectra: full cross [%d][%d] offset=%lu size=%lu\n", ii, jj, offset, ctx->batch_size);

          //multilog (log, LOG_INFO, "dump_spectra: full cross [%d][%d] offset=%lu\n", ii, jj, offset + half_batch);
          fwrite(&(cp_spectra[offset + half_batch]), sizeof(cuFloatComplex), half_batch, ccfile);
          //multilog (log, LOG_INFO, "dump_spectra: full cross [%d][%d] offset=%lu\n", ii, jj, offset);
          fwrite(&(cp_spectra[offset]), sizeof(cuFloatComplex), half_batch, ccfile);

          offset += chan_stride;
        }
      }
      pair_idx++;
    }
  }
#else
  for (ii=0;ii<ctx->npairs_total;ii++)
  {
    // In the case we want to write empty data
    if (ctx->baseline_mask[ii] == 0)
    {
      if (ctx->autocorr_mask[ii] == 1)
      {
        float zero = 0.0;
        for (jj=0; jj<ctx->batch_size; jj++)
          fwrite( &zero, sizeof(float), 1, acfile);
      }
      else
      {
        cuFloatComplex zero = make_cuFloatComplex(0.0,0.0);
        for (jj=0; jj<ctx->batch_size; jj++)
          fwrite(&zero, sizeof(cuFloatComplex), 1, ccfile);
      }
    }
    else
    {
      if (ctx->autocorr_mask[ii] == 1)
      {
        for (jj=ctx->batch_size/2; jj<ctx->batch_size; jj++)
          fwrite( &(cp_spectra[ctx->batch_size*pair_idx+jj].x), sizeof(float), 1, acfile);
        for (jj=0; jj<ctx->batch_size/2; jj++)
          fwrite( &(cp_spectra[ctx->batch_size*pair_idx+jj].x), sizeof(float), 1, acfile);
      }
      else
      {
        fwrite(&(cp_spectra[ctx->batch_size*pair_idx + ctx->batch_size/2]), sizeof(cuFloatComplex), ctx->batch_size/2, ccfile);
        fwrite(&(cp_spectra[ctx->batch_size*pair_idx]), sizeof(cuFloatComplex), ctx->batch_size/2, ccfile);
      }
      pair_idx++;
    }
  }
#endif
#else
  for (ii=0;ii<ctx->npairs_total;ii++)
    if (ctx->baseline_mask[ii] == 0)
    {
      if (ctx->autocorr_mask[ii] == 1)
      {
        float zero = 0.0;
        for (jj=0; jj<ctx->batch_size; jj++)
          fwrite( &zero, sizeof(float), 1, acfile);
      }
      else
      {
        cuFloatComplex zero = make_cuFloatComplex(0.0,0.0);
        for (jj=0; jj<ctx->batch_size; jj++)
          fwrite(&zero, sizeof(cuFloatComplex), 1, ccfile);
      }
    }
    else
    {
      if (ctx->autocorr_mask == 1)
      {
        for (jj=0; jj<ctx->batch_size; jj++)
          fwrite( &(cp_spectra[ctx->batch_size*pair_idx+jj].x), sizeof(float), 1, acfile);
      }
      else
        fwrite(&(cp_spectra[ctx->batch_size*pair_idx]), sizeof(cuFloatComplex), ctx->batch_size, ccfile);
      pair_idx++;
    }
#endif

  fclose(acfile);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Closed file %s\n",acfilename);
  
  fclose(ccfile);
  if (ctx->verbose)
    multilog (log, LOG_INFO, "Closed file %s\n",ccfilename);

  // rename files from temp names to real names
  char command[1024];
  sprintf (command,  "mv %s_%010lu_%010lu.ac.tmp %s_%010lu_%010lu.ac", ctx->utc_start, start_time, end_time, ctx->utc_start, start_time, end_time);
  system (command);

  sprintf (command,  "mv %s_%010lu_%010lu.cc.tmp %s_%010lu_%010lu.cc", ctx->utc_start, start_time, end_time, ctx->utc_start, start_time, end_time);
  system (command);
  
  return 0;
}





/*! Function that opens the data transfer target */
int dbcalib_open (dada_client_t* client)
{
  mopsr_dbcalib_t * ctx = (mopsr_dbcalib_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose > 1)
    multilog (log, LOG_INFO, "dbcalib_open()\n");

  if (ascii_header_get (client->header, "NCHAN", "%d", &(ctx->nchan)) != 1)
  {
    multilog (log, LOG_ERR, "header had no NCHAN\n");
    return -1;
  }

#ifdef HIRES
  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbcalib_open: nchan=%d\n", ctx->nchan); 
#else
  if (ctx->nchan != 1)
  {
    multilog (log, LOG_ERR, "header specified NCHAN == %d, should be 1\n", ctx->nchan);
    return -1;
  } 
#endif

  if (ascii_header_get (client->header, "NPOL", "%d", &(ctx->npol)) != 1)
  {
    multilog (log, LOG_ERR, "header had no NPOL\n");
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

  if ((strcmp(ctx->order, "ST") != 0) && (strcmp(ctx->order, "FST") != 0))
  {
    multilog (log, LOG_ERR, "ORDER [%s] was not ST\n", ctx->order);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: NCHAN=%d NANT=%d, NDIM=%d NPOL=%d\n", ctx->nchan, ctx->nant, ctx->ndim, ctx->npol);

  client->transfer_bytes = 0;
  client->optimal_bytes = 64*1024*1024;
  client->header_transfer = 0;

  // EBARR 2014 EDIT
  // Below this point there needs to be a way to generate arbitrary baseline combinations
  // These will be specified via a command line option identifying primary antennas only
  // Given a number of primary antennas (B) from the full set of antennas (N), the 
  // total number of baselines is given by:
  // ctx->npairs = B(N-1) - B(B-1)/2
  
  //UPDATE
  // Generate baseline combinations
  cudaError_t error;
  ctx->npairs_total = (ctx->nant * ctx->nant + ctx->nant)/2;
  
  if (ctx->nprimaries==0)
    ctx->npairs = ctx->npairs_total;
  else
    ctx->npairs = (int)(ctx->nprimaries * ((ctx->nant+1)-(ctx->nprimaries+1)/2.0));
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: npairs_total=%u npairs=%u\n", ctx->npairs_total, ctx->npairs);
      
  // allocate host memory for generating baseline combinations
  ctx->h_pairs = (mopsr_baseline_t*) malloc(sizeof(mopsr_baseline_t) * ctx->npairs);
  if (ctx->h_pairs == NULL)
  {
    multilog (log, LOG_ERR, "dbcalib_open: could not create allocated %ld "
              "bytes of host memory\n", sizeof(mopsr_baseline_t) * ctx->npairs);
    return -1;
  }

  // allocate memory for baseline mask
  ctx->baseline_mask = (unsigned*) malloc(sizeof(unsigned) * ctx->npairs_total);
  if (ctx->baseline_mask == NULL)
  {
    multilog (log, LOG_ERR, "dbcalib_open: could not create allocated %ld "
              "bytes of host memory\n", sizeof(unsigned) * ctx->npairs_total);
    return -1;
  }
  memset (ctx->baseline_mask, 0, sizeof(unsigned) * ctx->npairs_total);
  
  // allocate memory for autocorr mask
  ctx->autocorr_mask = (unsigned*) malloc(sizeof(unsigned) * ctx->npairs_total);
  if (ctx->autocorr_mask == NULL)
  {
    multilog (log, LOG_ERR, "dbcalib_open: could not create allocated %ld "
              "bytes of host memory\n", sizeof(unsigned) * ctx->npairs_total);
    return -1;
  }
  memset (ctx->autocorr_mask, 0, sizeof(unsigned) * ctx->npairs_total);

  // allocate device memory for baseline combinations                                                  
  error = cudaMalloc ((void**)&ctx->d_pairs, sizeof(mopsr_baseline_t) * ctx->npairs);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "dbcalib_open: could not create allocated %ld "
              "bytes of device memory\n", sizeof(mopsr_baseline_t) * ctx->npairs);
    return -1;
  }

  // generate all baseline combinations on host                                                                          
  int pair_idx,mask_idx,idx,ii,jj;
  pair_idx = mask_idx = idx = 0;
  while (idx < ctx->nant)
  {
    for (ii=idx; ii < ctx->nant; ii++)
    {
      unsigned use = 0;
      unsigned a;
      
      if (ctx->nprimaries > 0)
      {
        use = 0;
        for (jj=0;jj<ctx->nprimaries;jj++)
        {
          a = ctx->primary_antennas[jj];
          if ((a==idx) || (a==ii))
          {
            use=1;
            break;
          }
        }
      }
      else 
      {
        use = 1;
      }


      if (idx==ii)
        ctx->autocorr_mask[mask_idx] = 1;
      else
        ctx->autocorr_mask[mask_idx] = 0;
      
      ctx->baseline_mask[mask_idx] = use;
      if (use)
      {
        ctx->h_pairs[pair_idx].a = idx;
        ctx->h_pairs[pair_idx].b = ii;
        pair_idx++;
      }
      mask_idx++;
    }
    idx++;
  }

  
  // copy baseline combinations to device
  error = cudaMemcpyAsync (ctx->d_pairs, ctx->h_pairs, sizeof(mopsr_baseline_t) * ctx->npairs, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  ctx->d_cross_power_bytes = ctx->batch_size * ctx->npairs * sizeof(cuFloatComplex) * ctx->nchan;
  if (!ctx->d_cross_power)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: cudaMalloc(%"PRIu64") for d_cross_power\n", ctx->d_cross_power_bytes);
    error = cudaMalloc (&(ctx->d_cross_power), ctx->d_cross_power_bytes);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "open: could not create allocated %ld bytes of device memory\n", ctx->d_cross_power_bytes);
      return -1;
    }
  }
  error = cudaMemsetAsync (ctx->d_cross_power, 0, ctx->d_cross_power_bytes, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "open: could not memset array\n");
    return -1;
  }
  
  if (!ctx->h_cross_power)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: cudaMallocHost(%"PRIu64") for h_cross_power\n", ctx->d_cross_power_bytes);
    error = cudaMallocHost (&(ctx->h_cross_power), ctx->d_cross_power_bytes);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "open: could not create allocated %ld bytes of host memory\n", ctx->d_cross_power_bytes);
        return -1;
    }
  }

  // instantiate fft_plan_forward for data accululation
  cufftResult cufft_error;
  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: creating cufftPlan1d NX=%d BATCH=%d\n", ctx->batch_size, (ctx->block_size / ctx->ndim / ctx->batch_size));
  
  cufft_error = cufftPlan1d(&(ctx->fft_plan_forward), ctx->batch_size, CUFFT_C2C, ctx->block_size / ctx->ndim / ctx->batch_size);
  if (cufft_error != CUFFT_SUCCESS)
  {
    multilog (log, LOG_ERR, "dbcalib_open: cufftPlan1d returned error state %d\n",
              cufft_error);
    return -1;
  }

  cufft_error = cufftSetStream(ctx->fft_plan_forward, ctx->stream);
  if (cufft_error != CUFFT_SUCCESS)
  {
    multilog (log, LOG_ERR, "dbcalib_open: cufftSetStream returned error state %d \n",
              cufft_error);
    return -1;
  }

  float block_length_s = ((float) ctx->block_size) / (float) ctx->bytes_per_second;
  ctx->blocks_per_dump = (unsigned ) (ctx->dump_time / block_length_s + 0.5);

  return 0;
}

/*! Function that closes the data transfer */
int dbcalib_close (dada_client_t* client, uint64_t bytes_read)
{
  // the mopsr_dbcalib specific data
  mopsr_dbcalib_t* ctx = 0;

  // status and error logging facility
  multilog_t* log;

  cudaError_t error;

  assert (client != 0);

  ctx = (mopsr_dbcalib_t*) client->context;

  assert (ctx != 0);

  log = client->log;
  assert (log != 0);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_read=%"PRIu64"\n", bytes_read);
  
  if (ctx->dump_counter%ctx->blocks_per_dump != 0)
  {
    // dump remainder of data to file
    uint64_t start_time = (ctx->obs_offset + ctx->byte_counter) / ctx->bytes_per_second;
    uint64_t end_time   = start_time + ((ctx->block_size * ctx->blocks_per_dump) / ctx->bytes_per_second); 
    if (dbcalib_dump_spectra(client, start_time, end_time) != 0){
      return -1;
    }
  }
  return 0;
}

/*! Pointer to the function that transfers data to/from the target via direct block IO*/
int64_t dbcalib_write_block (dada_client_t* client, void* in_data, uint64_t in_data_size, uint64_t in_block_id)
{
  assert (client != 0);
  mopsr_dbcalib_t* ctx = (mopsr_dbcalib_t*) client->context;
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

int64_t dbcalib_block_gpu (dada_client_t* client, void * buffer, uint64_t bytes, uint64_t block_id)
{
  assert (client != 0);
  mopsr_dbcalib_t* ctx = (mopsr_dbcalib_t*) client->context;

  multilog_t * log = client->log;
  
  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: buffer=%p, bytes=%"PRIu64", block_id=%"PRIu64"\n", buffer, bytes, block_id);

  // if this is not a full block, then just return
  if (bytes != ctx->block_size)
  {
    multilog (log, LOG_INFO, "block_gpu: non-full block of %"PRIu64" bytes, ignoring\n", bytes);
    return (int64_t) bytes;
  } 

  cudaError_t error;
  cufftResult cufft_error;

#ifdef PER_CHANNEL_TRANSERS
  // copy from the [pinned] shared memory block to d_in
  error = cudaMemcpyAsync (ctx->d_in, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }
#else
  // copy from the [pinned] shared memory block to d_in
  error = cudaMemcpyAsync (ctx->d_raw, buffer, bytes, cudaMemcpyHostToDevice, ctx->stream);
  if (error != cudaSuccess)
  {
    multilog (log, LOG_ERR, "cudaMemcpyAsync H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  mopsr_transpose_BFST_FST (ctx->stream, ctx->d_raw, ctx->d_in, bytes, ctx->nant, ctx->nchan);
#endif

  // we must ensure that the data is copied across before continuing...
  cudaStreamSynchronize(ctx->stream);

  unsigned nsamps = ctx->block_size/ctx->nant/ctx->ndim/ctx->nchan;
  unsigned nbatch = nsamps/ctx->batch_size;
  
  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: mopsr_byte_to_float (block_size=%"PRIu64")\n", ctx->block_size);
  mopsr_byte_to_float ((int16_t *) ctx->d_in, (cuFloatComplex *) ctx->d_unpacked, nsamps, ctx->nant, ctx->nchan, ctx->stream);

  if (ctx->skzap)
  {
    //multilog (log, LOG_INFO, "block_gpu: block_id=%"PRIu64"\n", block_id);
    // apply spectral kurtosis zapping [AJ]
    const unsigned M = 512;
    const float sk_lower = 0.775046;  // 3 sigma
    const float sk_upper = 1.132542;   // 3 sigma
    //const float sk_lower = 0.786484;    // 4 sigma
    //const float sk_upper = 1.31218;     // 4 sigma
    //uint64_t nbytes = ctx->block_size * sizeof(float);
    if (ctx->verbose)
      multilog (log, LOG_INFO, "block_gpu: mopsr_skzap (%p, %"PRIu64", %u, %u, %f, %f)\n",
                (void *) ctx->d_unpacked, ctx->d_unpacked_bytes, ctx->nant, M, sk_lower, sk_upper);
    //mopsr_skzap ((float *) ctx->d_unpacked, ctx->d_unpacked_bytes, ctx->nant, M, sk_lower, sk_upper, ctx->stream);

    mopsr_skzap2 ((float *) ctx->d_unpacked, &(ctx->d_buffer), &(ctx->d_buffer_size), ctx->d_unpacked_bytes, ctx->nant, M, ctx->stream);
  }

  // FFT
  cufft_error = cufftExecC2C(ctx->fft_plan_forward, ctx->d_unpacked, ctx->d_unpacked, CUFFT_FORWARD);
  if (cufft_error != CUFFT_SUCCESS)
  {
    multilog (log, LOG_ERR, "dbcalib_block_gpu: cufftExecC2C returned error state %d: %s \n",
              cufft_error, cudaGetErrorString(cudaGetLastError()));
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "block_gpu: mopsr_accumulate_cp_spectra(%p %p %p %d %d %d %d %d %p)\n",
              ctx->d_unpacked, ctx->d_cross_power, ctx->d_pairs, ctx->batch_size,
              nbatch, ctx->npairs, nsamps, ctx->nchan, ctx->stream);
  mopsr_accumulate_cp_spectra (ctx->d_unpacked, ctx->d_cross_power, ctx->d_pairs, ctx->batch_size,
                               nbatch, ctx->npairs, nsamps, ctx->nchan, ctx->nant, ctx->stream);

  //test if we want to dump
  ctx->dump_counter++;
  if (ctx->dump_counter%ctx->blocks_per_dump == 0)
  {
    uint64_t start_time = (ctx->obs_offset + ctx->byte_counter) / ctx->bytes_per_second;
    uint64_t end_time   = start_time + ((ctx->block_size * ctx->blocks_per_dump) / ctx->bytes_per_second);
    if (dbcalib_dump_spectra(client, start_time, end_time) != 0)
    {
      multilog (log, LOG_ERR, "dbcalib_block_gpu: could not dump spectra\n");
      return -1;
    }
     
    // zero the cross power accumulation array
    error = cudaMemsetAsync (ctx->d_cross_power, 0, ctx->d_cross_power_bytes, ctx->stream);
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "dbcalib_block_gpu: could not memset accumulation array\n");
      return -1;
    }
    ctx->byte_counter += ctx->block_size * ctx->blocks_per_dump;
  }

  return (int64_t) bytes;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbcalib_write (dada_client_t* client, void* data, uint64_t data_size)
{
  fprintf(stderr, "dbcalib_write should be disabled!!!!!\n");

  return data_size;
}

int dbcalib_init (mopsr_dbcalib_t * ctx, dada_hdu_t * in_hdu)
{
  multilog_t * log = ctx->log;

  ctx->dump_counter = 0;

  // output DB block size must be bit_p times the input DB block size
  ctx->block_size = ipcbuf_get_bufsz ((ipcbuf_t *) in_hdu->data_block);

  if (ctx->device >= 0)
  {
    // select the gpu device
    int n_devices = dada_cuda_get_device_count();
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: detected %d CUDA devices\n", n_devices);

    if ((ctx->device < 0) && (ctx->device >= n_devices))
    {
      multilog (log, LOG_ERR, "dbcalib_init: no CUDA devices available [%d]\n",
                n_devices);
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: selecting cuda device %d\n", ctx->device);
    if (dada_cuda_select_device (ctx->device) < 0)
    {
      multilog (log, LOG_ERR, "dbcalib_init: could not select requested device [%d]\n",
                ctx->device);
      return -1;
    }

    char * device_name = dada_cuda_get_device_name (ctx->device);
    if (!device_name)
    {
      multilog (log, LOG_ERR, "dbcalib_init: could not get CUDA device name\n");
      return -1;
    }
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: using device %d : %s\n", ctx->device, device_name);
    free(device_name);

    // setup the cuda stream for operations
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: creating cudaStream\n");
    cudaError_t error = cudaStreamCreate (&(ctx->stream));
    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: stream created\n");
    if (error != cudaSuccess)
    {
      multilog (log, LOG_ERR, "init: could not create CUDA stream\n");
      return -1;
    }

    if (ctx->verbose)
      multilog (log, LOG_INFO, "init: stream=%p\n", (void *) ctx->stream);

#ifndef PER_CHANNEL_TRANSFERS
    if (!ctx->d_raw)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: cudaMalloc(%"PRIu64") for d_raw\n", ctx->block_size);
      error = cudaMalloc (&(ctx->d_raw), ctx->block_size);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "dbcalib_init: could not create allocated %ld bytes of device memory\n", ctx->block_size);
        return -1;
      }
    }
#endif

    // d_in should be same size as PSRDADA block
    if (!ctx->d_in)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: cudaMalloc(%"PRIu64") for d_in\n", ctx->block_size);
      error = cudaMalloc (&(ctx->d_in), ctx->block_size);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "dbcalib_init: could not create allocated %ld bytes of device memory\n", ctx->block_size);
        return -1;
      }
    }

    // d_unpacked, should be 4 times larger than the PSRDADA block (8->32bit) this also includes the transpose
    ctx->d_unpacked_bytes = ctx->block_size * 4;
    if (!ctx->d_unpacked)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: cudaMalloc(%"PRIu64") for d_unpacked\n", ctx->d_unpacked_bytes);
      error = cudaMalloc (&(ctx->d_unpacked), ctx->d_unpacked_bytes);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "init: could not create allocated %ld bytes of device memory\n", ctx->d_unpacked_bytes);
        return -1;
      }
    }

    // d_fft'd same as d_unpacked
    if (!ctx->d_fftd)
    {
      if (ctx->verbose)
        multilog (log, LOG_INFO, "init: cudaMalloc(%"PRIu64") for d_fftd\n", ctx->d_unpacked_bytes);
      error = cudaMalloc (&(ctx->d_fftd), ctx->d_unpacked_bytes);
      if (error != cudaSuccess)
      {
        multilog (log, LOG_ERR, "init: could not create allocated %ld bytes of device memory\n", ctx->d_unpacked_bytes);
        return -1;
      }
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

int dbcalib_destroy (mopsr_dbcalib_t * ctx, dada_hdu_t * in_hdu)
{
  if (ctx->device >= 0)
  {
#ifndef PER_CHANNEL_TRANSFERS
    if (ctx->d_raw)
      cudaFree (ctx->d_raw);
    ctx->d_raw = 0;
#endif

    if (ctx->d_in)
      cudaFree (ctx->d_in);
    ctx->d_in = 0;

    if (ctx->d_unpacked)
      cudaFree (ctx->d_unpacked);
    ctx->d_unpacked = 0;

    if (ctx->d_fftd)
      cudaFree (ctx->d_fftd);
    ctx->d_fftd = 0;

    if (ctx->d_cross_power)
      cudaFree (ctx->d_cross_power);
    ctx->d_cross_power = 0;

    if (ctx->d_pairs)
      cudaFree (ctx->d_pairs);
    ctx->d_pairs = 0;

    if (ctx->h_pairs)
      free (ctx->h_pairs);
    ctx->h_pairs = 0;

    if (ctx->d_buffer)
      cudaFree (ctx->d_buffer);
    ctx->d_buffer = 0;

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
  mopsr_dbcalib_t dbcalib = MOPSR_DBCALIB_INIT;

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

  //EBARR edit for primary antenna control
  int p_ant_ar_idx = 0; //current primary antenna idx
  int p_ant_ar_size = 5; // current primary antenna size

  dbcalib.primary_antennas = (unsigned*) malloc(sizeof(unsigned)*p_ant_ar_size);
  if (dbcalib.primary_antennas == NULL){
    fprintf (stderr, "mopsr_dbcalib: cannot allocate memory for primary antenna array\n");
    return -1;
  }
    

  while ((arg=getopt(argc,argv,"b:c:d:t:hk:n:p:svz")) != -1)
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
        fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
        return -1;
      }
      break;
      
    case 'n':
      nfft = atoi(optarg);
      break;
      
    case 's':
      single_transfer = 1;
      break;

    case 'p':
      // Here populate an array with primary antenna choices
      if (p_ant_ar_idx+1 >= p_ant_ar_size)
      {
        p_ant_ar_size+=p_ant_ar_size;
        dbcalib.primary_antennas = (unsigned*) realloc(dbcalib.primary_antennas, sizeof(unsigned)*p_ant_ar_size);
        if (dbcalib.primary_antennas == NULL)
        {
          fprintf (stderr, "mopsr_dbcalib: cannot allocate memory for primary antenna array\n");
          // Error message of choice
          return -1;
        }
      }
      dbcalib.primary_antennas[p_ant_ar_idx] = atoi(optarg);
      p_ant_ar_idx++;
      break;
      
    case 'v':
      verbose++;
      break;

    case 'z':
      dbcalib.skzap = 1;
      break;
      
    default:
      usage ();
      return 0;
      
    }
  }
  
  // AJ: this was moved to the open function (thats where we find out the BPS)
  //float block_length_s = ((float)dbcalib.block_size)/dbcalib.bytes_per_second;
  //dbcalib.blocks_per_dump = (int)(dump_time/block_length_s + 0.5);
  dbcalib.dump_time = dump_time;
  dbcalib.verbose = verbose;
  dbcalib.device = device;
  dbcalib.batch_size = nfft;
  dbcalib.nprimaries = p_ant_ar_idx;

  int num_args = argc-optind;
  unsigned i = 0;
   
  if (num_args != 0)
  {
    fprintf(stderr, "mopsr_dbcalib: no command line arguments expected\n");
    usage();
    exit(EXIT_FAILURE);
  } 

  log = multilog_open ("mopsr_dbcalib", 0);
  multilog_add (log, stderr);

  dbcalib.log = log;

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

  if (verbose)
    multilog (log, LOG_INFO, "main: batch_size=%d\n", dbcalib.batch_size);

  if (core >= 0)
    if (dada_bind_thread_to_core(core) < 0)
       multilog(log, LOG_WARNING, "mopsr_dbcalib: failed to bind to core %d\n", core);

  if (verbose > 1)
    multilog (log, LOG_INFO, "main: dbcalib_init()\n");
  if (dbcalib_init (&dbcalib, hdu) < 0)
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

  client->open_function  = dbcalib_open;
  client->io_function    = dbcalib_write;
  if (device >= 0)
    client->io_block_function = dbcalib_block_gpu;
  else
    client->io_block_function = dbcalib_write_block;

  client->close_function = dbcalib_close;
  client->direction      = dada_client_reader;

  client->context = &dbcalib;
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

