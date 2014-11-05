
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <inttypes.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "mopsr_cuda.h"

#define WARP_SIZE       32
//#define _GDEBUG         1

// each thread loads 16 x 2bytes into shm
__global__ void input_transpose_TFS_to_FST (
     const int16_t * input, int16_t * output,
     const unsigned nchan, const unsigned nant,
     const unsigned nval, const unsigned nval_per_thread,
     const unsigned in_block_stride, const unsigned nsamp_per_block, 
     const unsigned out_chanant_stride)
{
  extern __shared__ int16_t sdata[];

  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;
  const unsigned offset = (warp_num * (WARP_SIZE * nval_per_thread)) + warp_idx;

  unsigned in_idx  = (blockIdx.x * blockDim.x * nval_per_thread) + offset;
  unsigned sin_idx = offset;

  unsigned ival;
  for (ival=0; ival<nval_per_thread; ival++)
  {
    if (in_idx < nval * nval_per_thread)
      sdata[sin_idx] = input[in_idx];
    else
      sdata[sin_idx] = 0;

    in_idx += WARP_SIZE;
    sin_idx += WARP_SIZE;
  }

  __syncthreads();

  // at this point we have had 1024 threads each load 40 bytes (20 * 2) 40960 bytes / block
  // the sdata is order as TFS
  // For 40 channels, 16 ant, this is 32 time samples (nice)
  // for 40 channels, 4 ant, this is 128 time samples

  // each thread in a warp will write out 20 sets of time samples (40 chan, 16 ant => 640)

  unsigned nchanant = nchan * nant;
  // starting (first of the 20 ichan ant)
  unsigned ichanant = warp_num * nval_per_thread;
  // the time sample this thread will write out
  unsigned isamp    = warp_idx;

  // determine which shared memory index for this output ichan and isamp
  unsigned sout_idx = (isamp * nchanant) + ichanant;

  // determine the output index for this thread
  unsigned out_idx = (ichanant * out_chanant_stride) + (blockIdx.x * nsamp_per_block) + (isamp);

  for (ival=0; ival<nval_per_thread; ival++)
  {
    output[out_idx] = sdata[sout_idx];
    sout_idx ++;
    out_idx += out_chanant_stride;
  }

  return;
}

/*
 *  Transpose a block of data from TFS to FST
 */
void mopsr_input_transpose_TFS_to_FST (cudaStream_t stream, 
      void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;

  // have issues with this kernel if the nant is 1 and nchan 40, try 
  // changing nthread to be a divisor of nval_per_block

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = 32;
  const unsigned nval_per_block  = nsamp_per_block * nchan * nant;

  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }
  unsigned nval_per_thread = nval_per_block / nthread;
  
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  // the total number of values we have to process is
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  const size_t sdata_bytes = nthread * ndim * nval_per_thread;
  const unsigned in_block_stride = nthread * nval_per_thread;
  const unsigned out_chanant_stride = ndat;

#ifdef _GDEBUG
  fprintf (stderr, "input_transpose_TFS_to_FST: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "input_transpose_TFS_to_FST: nbytes=%lu, ndat=%lu, nval=%lu\n", nbytes, ndat, nval);
  fprintf (stderr, "input_transpose_TFS_to_FST: nthread=%d, nblocks=%d\n", nthread, nblocks);
  fprintf (stderr, "input_transpose_TFS_to_FST: input=%p output=%p sdata_bytes=%ld, in_block_stride=%d, nsamp_per_block=%u out_chan_stride=%u\n", d_in, d_out, sdata_bytes, in_block_stride, nsamp_per_block, out_chanant_stride);
#endif

  input_transpose_TFS_to_FST<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *)d_in, (int16_t *) d_out, nchan, nant, nval, nval_per_thread, in_block_stride, nsamp_per_block, out_chanant_stride);
}

// this will work best in FST format
__global__ void mopsr_input_ant_sum_kernel (const int16_t * input, int16_t * output, const uint64_t nsamp, const unsigned nchan, const unsigned nant)
{
  const unsigned ichan = blockIdx.y;
  const uint64_t isamp = blockIdx.x * blockDim.x + threadIdx.x;

  if (isamp >= nsamp)
    return;

  unsigned i_idx       = (ichan * nsamp * nant) + isamp;
  const unsigned o_idx = (ichan * nsamp) + isamp;

  // each thread will load data for nant into register memory

  int16_t ant16[MOPSR_MAX_NANT_PER_AQ];
  unsigned iant;
  for (iant=0; iant<nant; iant++)
  {
    ant16[iant] = input[i_idx];
    i_idx += nsamp;
  }

  float re = 0;
  float im = 0;

  int8_t * ant8 = (int8_t *) ant16;

  for (iant=0; iant<nant; iant++)
  {
#if 1
    if (iant % 2 == 0)
    {
      re += ((float) ant8[2*iant]) + 0.5;
      im += ((float) ant8[2*iant+1]) + 0.5;
    }
#else
    re += ((float) ant8[2*iant]) + 0.5;
    im += ((float) ant8[2*iant+1]) + 0.5;
#endif
  }

#if 0
  ant8[0] = (int8_t) (re / 2);
  ant8[1] = (int8_t) (im / 2);
#else
  //ant8[0] = (int8_t) (re / nant);
  //ant8[1] = (int8_t) (im / nant);
  ant8[0] = (int8_t) rintf ((re) - 0.5);
  ant8[1] = (int8_t) rintf ((im) - 0.5);
#endif

  output[o_idx] = ant16[0];
}

//
// sum all modules in stream together [ORDER FST -> FT]
//
void mopsr_input_sum_ant (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  const uint64_t nsamp = nbytes / (nchan * nant * ndim);

  // number of threads that actually load data
  const unsigned nthread = 1024;

  // need shared memory to load the ntap coefficients + nthread_load data points
  //const size_t   sdata_bytes = nant * ndim * nthread * sizeof(int8_t);

  dim3 blocks (nsamp / nthread, nchan);
  if (nsamp % nthread)
    blocks.x++;

#ifdef _GDEBUG
  fprintf (stderr, "input_ant_sum: bytes=%lu nsamp=%lu\n", nbytes, nsamp);
  fprintf (stderr, "input_ant_sum: nchan=%u nant=%u\n", nchan, nant);
  fprintf (stderr, "input_ant_sum: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
#endif

  mopsr_input_ant_sum_kernel<<<blocks, nthread, 0, stream>>>((const int16_t *) d_in, (int16_t *) d_out, nsamp, nchan, nant);

#ifdef _GDEBUG
  check_error_stream("mopsr_input_ant_sum_kernel", stream);
#endif
}

__global__ void input_transpose_FT_to_TF_kernel (
     const int16_t * input, int16_t * output, const uint64_t nsamp,
     const unsigned nchan, const unsigned nval, const unsigned nval_per_thread,
     const unsigned nsamp_per_block)
{
  extern __shared__ int16_t sdata[];

  const unsigned warp_num   = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx   = threadIdx.x % WARP_SIZE;
  //const unsigned nwarp      = blockDim.x  / WARP_SIZE;

  const unsigned nwarp_chunk_per_chan = nsamp_per_block / WARP_SIZE;
  const unsigned iwarp_chunk = warp_num * nval_per_thread;

  unsigned ichan  = iwarp_chunk / nwarp_chunk_per_chan;
  unsigned ichunk = iwarp_chunk % nwarp_chunk_per_chan;

  // offset from base pointer to the chanant this warp starts at
  uint64_t in_idx  = (ichan * nsamp) + (blockIdx.x * nsamp_per_block) + (ichunk * WARP_SIZE) + warp_idx;

  // to avoid shm bank conflicts add some padding 
  unsigned sin_idx = (warp_num * WARP_SIZE * nval_per_thread) + warp_idx + (2 * ichan);
  unsigned ival;
  //int8_t * tmp = (int8_t*) sdata;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    if (in_idx < nval * nval_per_thread)
      sdata[sin_idx] = input[in_idx];
    else
      sdata[sin_idx] = 0;

    //if ((blockIdx.x == 0) && (warp_num == 1))
    //  printf ("%d.%d.%d sdata[%d]=%d ichunk=%u ichan0=%u\n", blockIdx.x, threadIdx.x, ival, sin_idx, tmp[2*sin_idx], ichunk, ichan);
    // shared memory increases linearly
    sin_idx += WARP_SIZE;
    in_idx += WARP_SIZE;

    ichunk++;
    // if we are moving channel
    if (ichunk >= nwarp_chunk_per_chan)  
    {
      in_idx += (nsamp - nsamp_per_block);
      sin_idx += 2;
      ichunk = 0;
    }
  }

  __syncthreads();

  // starting ichan and isamp or this thread/warp to write out
  const unsigned ichan0 = warp_idx;
  const unsigned isamp0 = (warp_num * WARP_SIZE * nval_per_thread) / nchan;
  const unsigned nchansamp_block = nchan * nsamp_per_block;

  //                   block offset isamp               warp offset       thread offset
  uint64_t out_idx  = (blockIdx.x * nchansamp_block) + (isamp0 * nchan) + ichan0;

  //                   chan_offset                 sample offset
  unsigned sout_idx = (ichan0 * nsamp_per_block) + isamp0;


  const unsigned thread_stride = WARP_SIZE * nsamp_per_block;
  const unsigned thread_rewind = nchansamp_block - 1;

  unsigned warp_idat = warp_idx;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    ichan = warp_idat % nchan;

    //if ((blockIdx.x == 0) && (warp_num == 31))
    //  printf ("%d.%d.%d out_idx=%lu sout_idx=%u ichan0=%u isamp0=%u\n", blockIdx.x, threadIdx.x, ival, out_idx, sout_idx, ichan0, isamp0);
    output[out_idx] = sdata[sout_idx + (2*ichan)];
    //if ((blockIdx.x == 0) && (warp_num == 0))
    //  printf ("%d.%d.%d output[%lu] = sdata[%d]=%d\n", blockIdx.x, threadIdx.x, ival, out_idx, sout_idx + (2*ichan), tmp[2*(sout_idx+(2*ichan))]);

    // update the output index
    out_idx += WARP_SIZE;

    // update our warp idat so we can keep track of ichan
    warp_idat += WARP_SIZE;

    // update our shared memory output index
    sout_idx += thread_stride;
    if (sout_idx >= nchansamp_block)
      sout_idx -= thread_rewind;
  }
}

void mopsr_input_transpose_FT_to_TF (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = WARP_SIZE * 4;
  const unsigned nval_per_block  = nsamp_per_block * nchan;

  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }
  unsigned nval_per_thread = nval_per_block / nthread;

  const uint64_t nsamp = nbytes / (ndim * nchan);
  // the total number of values we have to process is
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  const size_t sdata_bytes = nthread * ndim * nval_per_thread + (2 * nchan);

#ifdef _GDEBUG
  fprintf (stderr, "input_transpose_FT_to_TF: nsamp_per_block=%u nval_per_block=%u, nval_per_thread=%u\n", nsamp_per_block, nval_per_block, nval_per_thread);
  fprintf (stderr, "input_transpose_FT_to_TF: nbytes=%lu, nsamp=%lu, nval=%lu\n", nbytes, nsamp, nval);
  fprintf (stderr, "input_transpose_FT_to_TF: nthread=%d, nblocks=%d\n", nthread, nblocks);
  fprintf (stderr, "input_transpose_FT_to_TF: input=%p output=%p sdata_bytes=%ld\n", d_in, d_out, sdata_bytes);
#endif

  input_transpose_FT_to_TF_kernel<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in, (int16_t *) d_out, nsamp, nchan, nval, nval_per_thread, nsamp_per_block);

#ifdef _GDEBUG
  check_error_stream ("input_transpose_FT_to_TF", stream);
#endif
}

#if 0
int mopsr_transpose_delay_alloc (transpose_delay_t * ctx,
                                 uint64_t block_size, unsigned nchan,
                                 unsigned nant, unsigned ntap)
{
  ctx->nchan = nchan;
  ctx->nant = nant;
  ctx->ntap = ntap;
  const unsigned nchanant = nchan * nant;
  const unsigned ndim = 2;

  cudaError_t error;

  // malloc data for H2D transfer of TFS input data
  error = cudaMalloc (&(ctx->d_in), block_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_alloc: cudaMalloc failed for %ld bytes\n", block_size);
    return -1;
  }

  ctx->half_ntap = ntap / 2;
  
  ctx->curr = (transpose_delay_buf_t *) malloc (sizeof(transpose_delay_buf_t));
  ctx->next = (transpose_delay_buf_t *) malloc (sizeof(transpose_delay_buf_t));

  ctx->buffer_size = block_size + (ndim * nchanant * ctx->half_ntap * 2);

  size_t counter_size = ctx->nchan * ctx->nant * sizeof(unsigned);
  if (mopsr_transpose_delay_buf_alloc (ctx->curr, ctx->buffer_size, counter_size) < 0)
  {
    fprintf (stderr, "mopsr_transpose_delay_alloc: mopsr_transpose_delay_buf_alloc failed\n");
    return -1;
  }

  if (mopsr_transpose_delay_buf_alloc (ctx->next, ctx->buffer_size, counter_size) < 0)
  {
    fprintf (stderr, "mopsr_transpose_delay_alloc: mopsr_transpose_delay_buf_alloc failed\n");
    return -1;
  }

  ctx->first_kernel = 1;
  
  return 0;
}

int mopsr_transpose_delay_buf_alloc (transpose_delay_buf_t * buf, size_t buffer_size, size_t counter_size)
{
  cudaError_t error; 

  // allocate the buffer for data
  error = cudaMalloc (&(buf->d_buffer), buffer_size);
  fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc( %p %ld)\n", buf->d_buffer, buffer_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buffer_size);
    return -1;
  }

  buf->counter_size = counter_size;

  error = cudaMalloc (&(buf->d_out_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMalloc (&(buf->d_in_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMalloc (&(buf->d_in_to), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMallocHost (&(buf->h_out_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMallocHost (&(buf->h_in_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMallocHost (&(buf->h_in_to), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  buf->h_off = (unsigned *) malloc(buf->counter_size);
  buf->h_delays = (unsigned *) malloc(buf->counter_size);

  return 0;
}


int mopsr_transpose_delay_dealloc (transpose_delay_t * ctx)
{
  if (ctx->d_in)
    cudaFree (ctx->d_in);
  ctx->d_in = 0;

  mopsr_transpose_delay_buf_dealloc (ctx->curr);
  mopsr_transpose_delay_buf_dealloc (ctx->next);

  return 0;
}

int mopsr_transpose_delay_buf_dealloc (transpose_delay_buf_t * ctx)
{
  if (ctx->d_out_from)
    cudaFree (ctx->d_out_from);
  ctx->d_out_from = 0;

  if (ctx->d_in_from)
    cudaFree (ctx->d_in_from);
  ctx->d_in_from = 0;

  if (ctx->d_in_to)
    cudaFree (ctx->d_in_to);
  ctx->d_in_to = 0;

  if (ctx->h_out_from)
    cudaFreeHost (ctx->h_out_from);
  ctx->h_out_from = 0;

  if (ctx->h_in_from)
    cudaFreeHost (ctx->h_in_from);
  ctx->h_in_from = 0;

  if (ctx->h_in_to)
    cudaFreeHost (ctx->h_in_to);
  ctx->h_in_to = 0;
  
  if (ctx->h_off)
    free(ctx->h_off);
  ctx->h_off = 0;

  if (ctx->h_delays)
    free(ctx->h_delays);
  ctx->h_delays = 0;

  return 0;
}

__global__ void mopsr_transpose_delay_kernel (
     int16_t * in,
     int16_t * curr, unsigned * c_out_from, unsigned * c_in_from, unsigned * c_in_to,
     int16_t * next, unsigned * n_out_from, unsigned * n_in_from, unsigned * n_in_to,
     const unsigned nchan, const unsigned nant, const unsigned nval, 
     const unsigned nval_per_thread, const unsigned in_block_stride, 
     const unsigned nsamp_per_block, const unsigned out_chanant_stride)
{
  // for loaded data samples
  extern __shared__ int16_t sdata[];

  const unsigned nchanant = nchan * nant;

  unsigned * curr_out_from = (unsigned *) (sdata + (nval_per_thread * blockDim.x));
  unsigned * curr_in_from  = curr_out_from + nchanant;
  unsigned * curr_in_to    = curr_in_from  + nchanant;
  unsigned * next_out_from = curr_in_to + nchanant;
  unsigned * next_in_from  = next_out_from + nchanant;
  unsigned * next_in_to    = next_in_from + nchanant;

  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;
  const unsigned offset = (warp_num * (WARP_SIZE * nval_per_thread)) + warp_idx;

  unsigned in_idx  = (blockIdx.x * blockDim.x * nval_per_thread) + offset;
  unsigned sin_idx = offset;

  unsigned ival;
  for (ival=0; ival<nval_per_thread; ival++)
  {
    if (in_idx < nval * nval_per_thread)
      sdata[sin_idx] = in[in_idx];
    else
      sdata[sin_idx] = 0;

    in_idx += WARP_SIZE;
    sin_idx += WARP_SIZE;
  }

  //__syncthreads();

  if (threadIdx.x < nchanant)
  {
    curr_out_from[threadIdx.x] = c_out_from[threadIdx.x];
    curr_in_from[threadIdx.x] = c_in_from[threadIdx.x];
    curr_in_to[threadIdx.x] = c_in_to[threadIdx.x];
    next_out_from[threadIdx.x] = n_out_from[threadIdx.x];
    next_in_from[threadIdx.x] = n_in_from[threadIdx.x];
    next_in_to[threadIdx.x] = n_in_to[threadIdx.x];
  }

  __syncthreads();

  // at this point we have had 1024 threads each load 40 bytes (20 * 2) 40960 bytes / block
  // the sdata is order as TFS
  // For 40 channels, 16 ant, this is 32 time samples (nice)
  // for 40 channels, 4 ant, this is 128 time samples

  // each thread in a warp will write out 20 sets of time samples (40 chan, 16 ant => 640)

  // our thread number within the warp [0-32], also the time sample this will write each time
  const unsigned isamp = warp_idx;

  // starting ichan/ant)
  unsigned ichanant = warp_num * nval_per_thread;

  // determine which shared memory index for this output ichan and isamp
  unsigned sout_idx = (isamp * nchanant) + ichanant;

  // which sample number in the kernel this thread is writing
  const unsigned isamp_kernel = (blockIdx.x * nsamp_per_block) + (isamp);

  // vanilla output index for this thread
  uint64_t out_idx = (ichanant * out_chanant_stride) + isamp_kernel;

  int64_t curr_idx, next_idx;

  //int8_t * tmp = (int8_t *) sdata;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    //unsigned ichan = ichanant / nant;
    //unsigned iant  = ichanant % nant;

    //printf ("isamp=%u  curr [%d -> %d]\n", isamp_kernel, curr_in_from[ichanant], curr_in_to[ichanant]);

    if ((curr_in_from[ichanant] <= isamp_kernel) && (isamp_kernel < curr_in_to[ichanant]))
    {
      curr_idx = (int64_t) out_idx + curr_out_from[ichanant] - curr_in_from[ichanant];
      //if (threadIdx.x > 1000)
      //  printf ("%d.%d ichanant=%d ichan=%d iant=%d isamp=%d isamp_kernel=%u out_idx=%lu curr_idx=%ld\n", blockIdx.x, threadIdx.x, ichanant, ichan, iant, isamp, isamp_kernel, out_idx, curr_idx);
      curr[curr_idx] = sdata[sout_idx];
      //if ((iant == 2) && (ichan == 39))
      //  printf ("isamp=%u curr[%ld] = sdata[%u] == %d blockIdx.x=%d\n", isamp_kernel, curr_idx, sout_idx, tmp[2*sout_idx], blockIdx.x);
    }

    if ((next_in_from[ichanant] <= isamp_kernel) && (isamp_kernel < next_in_to[ichanant]))
    {
      next_idx = (int64_t) out_idx + next_out_from[ichanant] - next_in_from[ichanant];
      //if (threadIdx.x > 1000)
      //  printf ("%d.%d ichanant=%d ichan=%d iant=%d isamp=%d out_idx=%lu next_idx=%ld\n", blockIdx.x, threadIdx.x, ichanant, ichan, iant, isamp, out_idx, next_idx);
      next[next_idx] = sdata[sout_idx];
      //if ((iant == 2) && (ichan == 39))
      //  printf ("isamp=%u next[%ld] = sdata[%u] == %d blockIdx.x=%d\n", isamp_kernel, next_idx, sout_idx, tmp[2*sout_idx], blockIdx.x);
    }

    sout_idx ++;
    out_idx += out_chanant_stride;
    ichanant++;
  }
}

int mopsr_transpose_delay_sync (cudaStream_t stream, transpose_delay_buf_t * ctx)
{
  cudaError_t error;
  error = cudaMemcpyAsync (ctx->d_out_from, ctx->h_out_from,
                           ctx->counter_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsyc H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  error = cudaMemcpyAsync (ctx->d_in_from, ctx->h_in_from,
                           ctx->counter_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsyc H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  error = cudaMemcpyAsync (ctx->d_in_to, ctx->h_in_to,
                           ctx->counter_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "cudaMemcpyAsyc H2D failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

void * mopsr_transpose_delay (cudaStream_t stream, transpose_delay_t * ctx, uint64_t nbytes, mopsr_delay_t ** delays)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;

  // have issues with this kernel if the nant is 1 and nchan 40, try 
  // changing nthread to be a divisor of nval_per_block

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = 32;
  const unsigned nval_per_block  = nsamp_per_block * ctx->nchan * ctx->nant;
  const uint64_t nsamp = nbytes / (ctx->nchan * ctx->nant * ndim);

  unsigned ichan, iant;
  unsigned ichanant = 0;
  int shift;

  for (ichan=0; ichan < ctx->nchan; ichan++)
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      if (ctx->first_kernel)
      {
        ctx->curr->h_delays[ichanant]   = delays[iant][ichan].samples;
        ctx->next->h_delays[ichanant]   = delays[iant][ichan].samples;

        ctx->curr->h_out_from[ichanant] = 0;
        ctx->curr->h_in_from[ichanant]  = ctx->curr->h_delays[ichanant] - ctx->half_ntap;
        ctx->curr->h_in_to[ichanant]    = nsamp;
        ctx->curr->h_off[ichanant]      = ctx->curr->h_in_to[ichanant] - ctx->curr->h_in_from[ichanant];

        // should never be used on first iteration
        ctx->next->h_out_from[ichanant] = 0;
        ctx->next->h_in_from[ichanant]  = nsamp;
        ctx->next->h_in_to[ichanant]    = 2 * nsamp;

      }
      else
      {
        // curr always uses delays from previous iteration
        ctx->curr->h_out_from[ichanant] = ctx->curr->h_off[ichanant];
        ctx->curr->h_in_from[ichanant]  = 0;
        ctx->curr->h_in_to[ichanant]    = nsamp + (2 * ctx->half_ntap) - ctx->curr->h_off[ichanant];
        //ctx->curr->h_off[ichanant]      = 0;  // no longer required

        // next always uses new delays
        ctx->next->h_out_from[ichanant] = 0;
        ctx->next->h_in_from[ichanant]  = ctx->curr->h_in_to[ichanant] - (2 * ctx->half_ntap);
        ctx->next->h_in_to[ichanant]    = nsamp;

        // handle a change in sample level delay
        shift = delays[iant][ichan].samples - ctx->next->h_delays[ichanant];
        ctx->next->h_in_from[ichanant] += shift;
        ctx->next->h_delays[ichanant]   = delays[iant][ichan].samples;

        ctx->next->h_off[ichanant]      = ctx->next->h_in_to[ichanant] - ctx->next->h_in_from[ichanant];

/*
        if (ichan == 0)
          fprintf (stderr, "ichanant=%d delay=%d CURR: in_from=%d in_to=%d off=%d, NEXT: in_from=%d in_to=%d off=%d\n", 
                  ichanant, delays[iant][ichan].samples, 
                  ctx->curr->h_in_from[ichanant], ctx->curr->h_in_to[ichanant],ctx->curr->h_off[ichanant],
                  ctx->next->h_in_from[ichanant], ctx->next->h_in_to[ichanant],ctx->next->h_off[ichanant]);
*/
      }

      ichanant++;
    }
  }

  mopsr_transpose_delay_sync (stream, ctx->curr);
  mopsr_transpose_delay_sync (stream, ctx->next);

  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }
  unsigned nval_per_thread = nval_per_block / nthread;

  const uint64_t ndat = nbytes / (ctx->nchan * ctx->nant * ndim);
  // the total number of values we have to process is
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  const size_t sdata_bytes = (nthread * ndim * nval_per_thread) + (6 * ctx->nchan * ctx->nant * sizeof(unsigned));
  const unsigned in_block_stride = nthread * nval_per_thread;
  const unsigned out_chanant_stride = ndat + (2 * ctx->half_ntap);

#ifdef _GDEBUG
  fprintf (stderr, "transpose_delay: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "transpose_delay: nbytes=%lu, ndat=%lu, nval=%lu\n", nbytes, ndat, nval);
  fprintf (stderr, "transpose_delay: nthread=%d, nblocks=%d sdata_bytes=%d\n", nthread, nblocks, sdata_bytes);
  fprintf (stderr, "transpose_delay: out_chanant_stride=%u\n", out_chanant_stride);
#endif

  mopsr_transpose_delay_kernel<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *) ctx->d_in, 
        (int16_t *) ctx->curr->d_buffer, ctx->curr->d_out_from, ctx->curr->d_in_from, ctx->curr->d_in_to,
        (int16_t *) ctx->next->d_buffer, ctx->next->d_out_from, ctx->next->d_in_from, ctx->next->d_in_to,
        ctx->nchan, ctx->nant, nval, nval_per_thread, in_block_stride, nsamp_per_block, out_chanant_stride);

  check_error_stream("mopsr_transpose_delay_kernel", stream);


  if (ctx->first_kernel)
  {
    ctx->first_kernel = 0;
    return 0;
  }
  else
  {
    transpose_delay_buf_t * save = ctx->curr;
    ctx->curr = ctx->next;
    ctx->next = save;
    return save->d_buffer;
  }
}
#endif


__global__ void input_transpose_FST_to_STF (
     const int16_t * input, int16_t * output,
     const unsigned nchan, const unsigned nant,
     const unsigned nval, const unsigned nval_per_thread,
     const unsigned nsamp_per_block, const uint64_t out_ant_stride)
{
  extern __shared__ int16_t sdata[];

  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;
  const uint64_t nsamp    = nsamp_per_block * gridDim.x;
  unsigned isamp          = warp_idx;

  // offset from base pointer to the chanant this warp starts at
  uint64_t in_idx  = (blockIdx.x * WARP_SIZE) + (warp_num * nsamp * nval_per_thread) + warp_idx;

  unsigned sin_dat = warp_num * nsamp_per_block * nval_per_thread + isamp;

  const unsigned nantsamp_block = nant * nsamp_per_block;
  const unsigned nchansamp_block = nchan * nsamp_per_block;
  const unsigned nchanantsamp_block = nant * nchan * nsamp_per_block;

  unsigned ival, ichan, iant, sin_idx;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    ichan = sin_dat / nantsamp_block;
    iant  = (sin_dat % nantsamp_block) / nsamp_per_block;

    // note that we add ichan to the shm index to avoid shm bank conflicts on shm read (later)
    sin_idx = (ichan * nantsamp_block) + (iant * nsamp_per_block) + isamp + (2 * ichan);

    if (in_idx < nval * nval_per_thread)
      sdata[sin_idx] = input[in_idx];
    else
      sdata[sin_idx] = 0;

    sin_dat += nsamp_per_block;
    in_idx += nsamp;
  }

  __syncthreads();


  // antenna for this WARP
  iant = (warp_num * nant) / WARP_SIZE;

  // shared memory strides
  const unsigned swarp_stride = nval_per_thread * WARP_SIZE;  // number of dats per warp
  const unsigned sant_base    = iant * nsamp_per_block;

  // starting ichan and isamp or this thread/warp to write out
  const unsigned ichan0 = warp_idx;
  const unsigned nchansamp_per_warp = (WARP_SIZE * nval_per_thread) / nchan;
  const unsigned isamp0 = (warp_num * nchansamp_per_warp) % nsamp_per_block;
  const unsigned out_warp_offset = warp_num % (WARP_SIZE / nant);

  //                   ant offset               block offset isamp               warp offset
  uint64_t out_idx  = (iant * out_ant_stride) + (blockIdx.x * nchansamp_block) + (out_warp_offset * swarp_stride) + ichan0;

  //                   chan_offset                ant offset   sample offset
  unsigned sout_idx = (ichan0 * nantsamp_block) + sant_base  + isamp0;

  const unsigned thread_stride = WARP_SIZE * nsamp_per_block * nant;
  const unsigned thread_rewind = nchanantsamp_block - 1;

  unsigned warp_idat = warp_idx;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    ichan = warp_idat % nchan;

    if ((blockIdx.x == 16) && (threadIdx.x < 32))
      printf ("[%u] output[%u] = sdata[%d]\n", threadIdx.x, out_idx, sout_idx + 2*ichan);
 
    //output[out_idx] = sdata[sout_idx + 2*ichan];

    // update the output index
    out_idx += WARP_SIZE;

    // update our warp idat so we can keep track of ichan
    warp_idat += WARP_SIZE;

    // update our shared memory output index
    sout_idx += thread_stride;
    if (sout_idx >= nchanantsamp_block)
      sout_idx -= thread_rewind;
  }
}

void mopsr_input_transpose_FST_to_STF (cudaStream_t stream,
      void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;

  // have issues with this kernel if the nant is 1 and nchan 40, try 
  // changing nthread to be a divisor of nval_per_block

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = WARP_SIZE;
  const unsigned nval_per_block  = nsamp_per_block * nchan * nant;

  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }
  unsigned nval_per_thread = nval_per_block / nthread;

  // the total number of values we have to process is
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  const size_t sdata_bytes = nthread * ndim * nval_per_thread + (2 * nchan);
  const unsigned out_ant_stride = nbytes / (nant * ndim);
  // TODO might need to pass nsamp to kernel!!!

#ifdef _GDEBUG
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  fprintf (stderr, "input_transpose_FST_to_STF: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "input_transpose_FST_to_STF: nbytes=%lu, ndat=%lu, nval=%lu\n", nbytes, ndat, nval);
  fprintf (stderr, "input_transpose_FST_to_STF: nthread=%d, nblocks=%d\n", nthread, nblocks);
  fprintf (stderr, "input_transpose_FST_to_STF: input=%p output=%p sdata_bytes=%ld,nsamp_per_block=%u out_ant_stride=%u\n", d_in, d_out, sdata_bytes, nsamp_per_block, out_ant_stride);
#endif

  input_transpose_FST_to_STF<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *)d_in, (int16_t *) d_out, nchan, nant, nval, nval_per_thread, nsamp_per_block, out_ant_stride);

#ifdef _GDEBUG
  check_error_stream ("input_transpose_FST_to_STF", stream);
#endif
}

// scaling factors for antenna
__device__ __constant__ float d_ant_scales [MOPSR_MAX_NANT_PER_AQ];

__global__ void input_rephase (int16_t * input, cuFloatComplex const * __restrict__ corrections,
                                   uint64_t nbytes, const unsigned chan_stride, const unsigned ant_stride)
{
  extern __shared__ cuFloatComplex corr_sh[];

  const unsigned isamp = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned iant  = blockIdx.y;
  const unsigned ichan = blockIdx.z;
  const unsigned idx   = (ichan * chan_stride + iant*ant_stride + isamp);
  const unsigned icorr = isamp % MOPSR_UNIQUE_CORRECTIONS;

  // all threads in this block will be using the same ant and channel, so we only need
  // to load into shm the 32 co-efficients for this ant and channel
  if (threadIdx.x < MOPSR_UNIQUE_CORRECTIONS)
    corr_sh[icorr] = corrections[ichan*MOPSR_UNIQUE_CORRECTIONS + icorr];

  __syncthreads();

  // coalesced int16_t read from global memory
  int16_t load16 = input[idx];
  int8_t * load8 = (int8_t *) &load16;

  cuFloatComplex val = make_cuComplex((float) load8[0], (float) load8[1]);
  cuFloatComplex res = cuCmulf(val, corr_sh[icorr]);

  const float scale =  d_ant_scales[iant];
  load8[0] = (int8_t) (cuCrealf(res) * scale);
  load8[1] = (int8_t) (cuCimagf(res) * scale);

  input[idx]   = load16;
}


void mopsr_input_rephase (cudaStream_t stream, void * d_data, void * d_corrections,
                          uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  const size_t sdata_bytes = MOPSR_UNIQUE_CORRECTIONS * sizeof(cuFloatComplex);

  const unsigned nthread = 1024;
  dim3 blocks (ndat / nthread, nant, nchan);
  if (ndat % nthread)
    blocks.x++;

#ifdef _GDEBUG
  fprintf (stderr, "input_rephase: bytes=%lu ndat=%lu\n", nbytes, ndat);
  fprintf (stderr, "input_rephase: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
#endif

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

  input_rephase<<<blocks, nthread, sdata_bytes, stream>>>((int16_t*) d_data, 
      (cuFloatComplex *) d_corrections, nbytes, chan_stride, ant_stride);

#ifdef _GDEBUG
  check_error_stream("input_rephase", stream);
#endif
}

__global__ void input_rephase_TFS (int16_t * input, uint64_t nval, const unsigned nchan, 
                                   const unsigned nant, const unsigned chan_offset)
{
  const unsigned samp_stride = nchan * nant;

  const unsigned idx  = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nval)
    return;

  const unsigned isamp = idx / samp_stride;           // sample number
  const unsigned ipos  = isamp % 32;                   // sample position for FIR filter
  const unsigned ichan = (idx % samp_stride) / nant;  // 
  const unsigned iant  = idx % nant;

  // load the 16 bit value from global memory
  int16_t load16 = input[idx];
  int8_t * load8 = (int8_t *) &load16;

  // calculate the rephasing factor for this channel and sample
  float ratio = 2 * M_PI * (5.0 / 32.0);
  float theta = (chan_offset + ichan) * ratio * ipos;
  cuFloatComplex rephase = make_cuComplex(sinf(theta), -1 * cos(theta));

  cuFloatComplex val = make_cuComplex((float) load8[0] + 0.5, (float) load8[1] + 0.5);
  cuFloatComplex res = cuCmulf(val, rephase);

  const float scale =  d_ant_scales[iant];
  load8[0] = (int8_t) rintf ((cuCrealf(res) * scale) - 0.5);
  load8[1] = (int8_t) rintf ((cuCimagf(res) * scale) - 0.5);

  // write out to global memory (in-place)
  input[idx] = load16;
}


void mopsr_input_rephase_TFS (cudaStream_t stream, void * d_data, uint64_t nbytes, 
                              unsigned nchan, unsigned nant, unsigned chan_offset)
{
  const unsigned ndim = 2;
  const uint64_t nval = nbytes / ndim;
  const unsigned nthread = 1024;
  unsigned nblocks = (unsigned) (nval / nthread);
  if (nval % nthread)
    nblocks++;

#ifdef _GDEBUG
  fprintf (stderr, "input_rephase_TFS: bytes=%lu nval=%lu\n", nbytes, nval);
  fprintf (stderr, "input_rephase_TFS: blocks=%u\n", nblocks);
#endif

  input_rephase_TFS<<<nblocks, nthread, 0, stream>>>((int16_t*) d_data, nval, nchan, nant, chan_offset);

#ifdef _GDEBUG
  check_error_stream("input_rephase_TFS", stream);
#endif
}


void mopsr_input_rephase_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes)
{
  cudaMemcpyToSymbolAsync (d_ant_scales, (void *) h_ant_scales, nbytes, 0, cudaMemcpyHostToDevice, stream);

}

// apply a fractional delay correction to a channel / antenna, warps will always 
__global__ void input_delay (int8_t * input, int8_t * output, int8_t * overlap, float * delays,
                             unsigned nthread_run, uint64_t ndat, const unsigned chan_stride, 
                             const unsigned ant_stride, const unsigned ntap)
{
  extern __shared__ float sdata_delay[];

  float * filter = sdata_delay;
  float * reals = filter + ntap;
  float * imags = reals + blockDim.x;

  const unsigned half_ntap = (ntap / 2);
  const unsigned in_offset = 2 * half_ntap;
  const unsigned isamp = blockIdx.x * nthread_run + threadIdx.x;
  const unsigned iant  = blockIdx.y;
  const unsigned nant  = blockDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ndim  = 2;

  if (threadIdx.x < ndat)
    return;

  // calculate the filter coefficients for the delay
  if (threadIdx.x < ntap)
  {
    float x = (threadIdx.x - half_ntap) + delays[ichan*nant * iant];
    if (x == 0)
      filter[threadIdx.x] = 1;
    else
    {
      x *= M_PI;
      filter[threadIdx.x] = sinf(x) / x;
    }
  }
  
  // each thread must also load its data from main memory
  unsigned data_idx = ichan*chan_stride + iant*ant_stride + isamp;

  // the first block needs to load data from the overlap buffer, not from input block - in_offset
  if (blockIdx.x == 0) 
  {
    if (threadIdx.x < in_offset)
    {
      const unsigned overlap_idx = (ichan*nant*ntap + iant*ntap + isamp) * ndim;
      reals[threadIdx.x] = (float) overlap[overlap_idx + 0];
      imags[threadIdx.x] = (float) overlap[overlap_idx + 1];
    }
    else
    {
      reals[threadIdx.x] = (float) input[2*(data_idx - in_offset)];
      imags[threadIdx.x] = (float) input[2*(data_idx - in_offset)+1];
    }
  }
  else
  {
    reals[threadIdx.x] = (float) input[2*(data_idx - in_offset)];
    imags[threadIdx.x] = (float) input[2*(data_idx - in_offset)+1];
  }

  __syncthreads();

  // there are 2 * half_ntap threads that dont calculate anything
  if (threadIdx.x < nthread_run)
  {
    float re = 0;
    float im = 0;
    unsigned i;
    for (i=0; i<ntap; i++)
    {
      re += reals[i] * filter[i];
      im += imags[i] * filter[i];
    }

    output[2*data_idx]   = (int8_t) floor(re + 0.5);
    output[2*data_idx+1] = (int8_t) floor(im + 0.5);
  }
}

// 
// Perform fractional delay correction, out-of-place
//
void mopsr_input_delay_fractional (cudaStream_t stream, void * d_in, 
                                   void * d_out, void * d_overlap,
                                   float * d_delays, uint64_t nbytes, 
                                   unsigned nchan, unsigned nant, unsigned ntap)
{
  fprintf (stderr, "mopsr_input_delay_fractional()\n");

  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  const unsigned half_ntap = ntap / 2;

  // number of threads that actually load data
  const unsigned nthread_load = 1024;
  const unsigned nthread_run  = nthread_load - (2 * half_ntap);

  // need shared memory to load the ntap coefficients + nthread_load data points
  const size_t   sdata_bytes = (ntap * 2) + (nthread_load * 2);

  dim3 blocks (ndat / nthread_run, nant, nchan);
  if (ndat % nthread_run)
    blocks.x++;

  fprintf (stderr, "mopsr_input_delay: bytes=%lu ndat=%lu\n", nbytes, ndat);
  fprintf (stderr, "mopsr_input_delay: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

  input_delay<<<blocks, nthread_load, sdata_bytes, stream>>>((int8_t *) d_in, (int8_t *) d_out, 
                (int8_t *) d_overlap, (float *) d_delays, nthread_run, ndat, chan_stride, ant_stride, ntap);

#ifdef _GDEBUG
  check_error_stream("input_delay", stream);
#endif
}

__global__ void tile_beams_kernel (int16_t * input, float * output, 
                                   float * beam_sin_thetas,
                                   float * ant_factors,
                                   unsigned nbeam, uint64_t ndat, unsigned nant)
{
  extern __shared__ int16_t sdata_tb[];
  float * sh_ant_factors = (float *) (sdata_tb + (32 * nant));

  //const unsigned ndim = 2;
  const unsigned sample = blockIdx.x * blockDim.x + threadIdx.x;

  //unsigned warp_idx = threadIdx.x % WARP_SIZE;
  unsigned warp_num = threadIdx.x / WARP_SIZE;
  unsigned iant  = warp_num;

  unsigned i_idx = (iant * ndat) + sample;
  unsigned s_idx = threadIdx.x;

  const uint64_t in_stride = WARP_SIZE * ndat;
  
  while (iant < nant)
  {
    if (sample < ndat)
    {
      //if ((blockIdx.x == 0) && (warp_num == 0))
      //  printf ("[%d][%d] reading [%d] = [%d]\n", blockIdx.x, threadIdx.x, s_idx, i_idx);

      sdata_tb[s_idx] = input[i_idx];
      s_idx += blockDim.x;
      i_idx += in_stride;
    }
    iant += WARP_SIZE;
  }

  // load the antenna factors into shared memory
  if (threadIdx.x < nant)
  {
   sh_ant_factors[threadIdx.x] = ant_factors[threadIdx.x]; 
  }

  // load input data to shared memory such that
  // [s0t0 s0t1 s0t2 ...  s0t31]
  // [s1t0 s1t1 t1t2 ...  s1t31]
  // [           ...           ]
  // [s351t0 s351t1 ... s351t31]

  __syncthreads();

  // Form tied array beams, detecting and summing as we go, 
  // only use as many beams as there are threads

  int8_t * sdata_tb_re = (int8_t *) sdata_tb;
  int8_t * sdata_tb_im = sdata_tb_re + 1;

  cuFloatComplex phasor, samp_sum;

  // TODO change beam_thetas to be sin(beam_thetas on CPU)
  unsigned ibeam = threadIdx.x;

  if (ibeam < nbeam)
  {
    // a simple 1 time load from gmem, coalesced
    const float sin_theta = beam_sin_thetas[ibeam];
    cuFloatComplex beam_sum = make_cuComplex(0,0);
    s_idx = 0;

    for (unsigned iant=0; iant<nant; iant++)
    {
      sincosf(sin_theta * sh_ant_factors[iant], &(phasor.x), &(phasor.y));
      samp_sum = make_cuComplex(0,0);

      for (unsigned isamp=0; isamp<32; isamp++)
      {
        samp_sum.x += (float) sdata_tb_re[2*isamp];
        samp_sum.y += (float) sdata_tb_im[2*isamp];
      }
      s_idx += 128;
      beam_sum = cuCaddf( beam_sum, cuCmulf (samp_sum, phasor)); 
    }
    output[(blockIdx.x * nbeam) + ibeam] = cuCabsf(beam_sum);
  }
}

// 32 beams per block
__device__ __constant__ float d_beam_phasors [MOPSR_MAX_NANT_PER_AQ];

__global__ void tile_beams_kernel_32 (int32_t * input, float * output,
                                      float * beam_sin_thetas,
                                      float * ant_factors,
                                      unsigned nbeam, unsigned ndat, unsigned nant)
{
  extern __shared__ int32_t sdata_tb_a[];
  float * sh_ant_factors = (float *) (sdata_tb_a + (16 * nant));

  const int half_ndat = ndat / 2;
  int iant = threadIdx.x / WARP_SIZE;  // warp_num
  const int warp_idx = threadIdx.x & 0x1F;

  int idx = (iant * half_ndat) + (blockIdx.x * 16) + threadIdx.x;
  int sdx = (iant * 16) + threadIdx.x;

  const int in_stride = half_ndat * 32;

  if (warp_idx < 16)
  {
    while (sdx < nant * 16)
    {
      sdata_tb_a[sdx] = input[idx];

      sdx += 512;
      idx += in_stride;
      iant += 32;
    } 
  }

  // load the antenna factors into shared memory
  if (threadIdx.x < nant)
  {
   sh_ant_factors[threadIdx.x] = ant_factors[threadIdx.x];
  }

  __syncthreads();

#ifdef TWOATATIME
  cuFloatComplex phasor, samp_sum;

  unsigned ibeam = threadIdx.x;
  if (ibeam < nbeam)
  {
    // a simple 1 time load from gmem, coalesced
    const float sin_theta = beam_sin_thetas[ibeam];
    float beam_sum = 0;
    cuFloatComplex s1_sum, s2_sum, val;
    sincosf(sin_theta * sh_ant_factors[iant], &(phasor.x), &(phasor.y));

    for (unsigned isamp=0; isamp<16; isamp++)
    {
      s1_sum = make_cuComplex(0,0);
      s2_sum = make_cuComplex(0,0);

      sdata_tb_32 = sdata_tb_a + isamp;

      for (unsigned iant=0; iant<nant; iant++)
      {
        int32_t inval = *sdata_tb_32;

        val = make_cuComplex ((float) ((inval >> 0) & 0xFF), (float) ((inval >> 8) & 0xFF));
        s1_sum = cuCfmaf (phasor, val, s1_sum);

        val = make_cuComplex ((float) ((inval >> 16) & 0xFF), (float) ((inval >> 24) & 0xFF));
        s2_sum = cuCfmaf (phasor, val, s2_sum);

        sdata_tb_32 += 16;
      }

      beam_sum += cuCabsf(s1_sum);
      beam_sum += cuCabsf(s2_sum);
    }
    output[(blockIdx.x * nbeam) + ibeam] = beam_sum;
  }
#endif

  cuFloatComplex phasor;
  unsigned ibeam = threadIdx.x;

  if (ibeam < nbeam)
  {
    // a simple 1 time load from gmem, coalesced
    //const float sin_theta = beam_sin_thetas[ibeam];
    float beam_sum = 0;
    cuFloatComplex sum, val;
    phasor.x = 2;
    phasor.y = 3;
    //sincosf(sin_theta * sh_ant_factors[iant], &(phasor.x), &(phasor.y));

    int16_t * sdata_tb_16 = (int16_t *) sdata_tb_a; 
    int16_t val16;
    int8_t * val8 = (int8_t *) &val16;

    for (unsigned isamp=0; isamp<32; isamp++)
    {
      sdata_tb_16 = ((int16_t *) sdata_tb_a) + isamp;
      sum = make_cuComplex(0,0);

      for (unsigned iant=0; iant<nant; iant++)
      {
        val16 = sdata_tb_16[isamp];

        val = make_cuComplex ((float) val8[0], (float) val8[1]);
        sum = cuCfmaf (phasor, val, sum);

        sdata_tb_16 += 32;
      }
      beam_sum += cuCabsf(sum);
    }
    output[(blockIdx.x * nbeam) + ibeam] = beam_sum;
  }
}

__global__ void tile_beams_kernel_32_blk (
        int32_t * input, float * output,
        float * beam_phasors,
        unsigned nbeam, unsigned ndat, unsigned nant)
{
  extern __shared__ int32_t sdata_tb_a[];

  const int half_ndat = ndat / 2;
  const int warp_num = threadIdx.x / WARP_SIZE;
  int iant = warp_num;
  const int warp_idx = threadIdx.x & 0x1F;

  int idx = (iant * half_ndat) + (blockIdx.x * 16) + threadIdx.x;
  int sdx = (iant * 16) + threadIdx.x;

  const int in_stride = half_ndat * 32;

  if (warp_idx < 16)
  {
    while (sdx < nant * 16)
    {
      sdata_tb_a[sdx] = input[idx];

      sdx += 512;
      idx += in_stride;
      iant += 32;
    }
  }

  __syncthreads();

  // if uniform, 352 * 352
  const unsigned nbeamant = nbeam * nant;

  unsigned iblock = 0;
  for (iblock=0; iblock<12; iblock++)
  {
    unsigned ibeam = blockIdx.y * blockDim.y + warp_num;
    unsigned isamp = warp_idx;

    cuFloatComplex phasor;

    int8_t * sdata_tb_8 = ((int8_t *) sdata_tb_a) + 2 * isamp;

    cuFloatComplex sum = make_cuComplex(0,0);

    unsigned count = 0;
    unsigned iant = warp_idx;

    for (unsigned i=0; i<nant; i++)
    {
      // every thread in this warp is on the same ibeam
      // so each thread will share this computation with 
      // a shuffle every 32 antenna
      if (count == 32)
      {
        unsigned beam_ant = ibeam * nant + iant;
        phasor.x = beam_phasors[beam_ant];
        beam_ant += nbeamant;
        phasor.y = beam_phasors[beam_ant];

        iant += 32;
        count = 0;
      }
      else
      {
#if HAVE_CUDA_SHUFFLE
        phasor.x = __shfl_down(phasor.x,1);
        phasor.y = __shfl_down(phasor.y,1);
#endif
        count++;
      }
      
      {
        cuFloatComplex val = make_cuComplex ((float) sdata_tb_8[0], (float) sdata_tb_8[1]);
        sum = cuCfmaf (phasor, val, sum);
      }
      sdata_tb_8 += 64;
    }

    // get the magnitude of the power
    float beam_sum = sum.x * sum.x + sum.y * sum.y;

#if HAVE_CUDA_SHUFFLE

    beam_sum += __shfl_down (beam_sum, 16);
    beam_sum += __shfl_down (beam_sum, 8);
    beam_sum += __shfl_down (beam_sum, 4);
    beam_sum += __shfl_down (beam_sum, 2);
    beam_sum += __shfl_down (beam_sum, 1);

#endif

    output[ibeam] = beam_sum;
  }
}




void mopsr_tile_beams (cudaStream_t stream, void * d_in, void * d_fbs,
                       float * beam_sin_thetas, float * ant_factors,
                       uint64_t bytes, unsigned nbeam, unsigned nant, unsigned tdec)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nant * ndim);
  const unsigned nthread = 1024;
  const unsigned ndat_per_block = 32;

  unsigned nblocks = ndat / ndat_per_block;
  if (ndat % ndat_per_block)
    nblocks++;
  //dim3 blocks (ndat / ndat_per_block, nant/32);
  //if (ndat % ndat_per_block)
  //  blocks.x++;
  //if (nant % 32)
  //  blocks.y++;
 
  const size_t sdata_bytes = (nant * ndat_per_block * ndim) + (nant * sizeof(float)); 
  fprintf (stderr, "bytes=%lu ndat=%lu blocks=%u shm=%ld\n", bytes, ndat, nblocks, sdata_bytes);

  // forms fan beams integrating 32 time samples for efficiency
  tile_beams_kernel_32<<<nblocks, nthread, sdata_bytes, stream>>>((int32_t *) d_in, (float *) d_fbs, beam_sin_thetas, ant_factors, nbeam, ndat, nant);
}

void mopsr_tile_beams_precomp (cudaStream_t stream, void * d_in, void * d_fbs, float * d_phasors,
                       uint64_t bytes, unsigned nbeam, unsigned nant, unsigned tdec)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nant * ndim);
  const unsigned nthread = 1024;
  const unsigned ndat_per_block = 32;

  unsigned nblocks = ndat / ndat_per_block;
  if (ndat % ndat_per_block)
    nblocks++;

  const size_t sdata_bytes = nant * ndat_per_block * ndim;
  fprintf (stderr, "bytes=%lu ndat=%lu blocks=%u shm=%ld\n", bytes, ndat, nblocks, sdata_bytes);

  // forms fan beams integrating 32 time samples for efficiency
  tile_beams_kernel_32_blk<<<nblocks, nthread, sdata_bytes, stream>>>((int32_t *) d_in, (float *) d_fbs,
d_phasors, nbeam, ndat, nant);
}

// process 

/*
void mopsr_input_transpose_ST_to_TS (cudaStream_t stream, void * d_in, 
                                     void * d_out, uint64_t nbytes, 
                                     unsigned nchan)
{
  unsigned nthread = 1024;
  unsigned nsamp_per_block = 32;

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = WARP_SIZE * 4;
  const unsigned nval_per_block  = nsamp_per_block * nchan;

  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }
  unsigned nval_per_thread = nval_per_block / nthread;

  const uint64_t nsamp = nbytes / (ndim * nchan);
  // the total number of values we have to process is
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;

  const size_t sdata_bytes = nthread * ndim * nval_per_thread + (2 * nchan);

#ifdef _GDEBUG
  fprintf (stderr, "input_transpose_FT_to_TF: nsamp_per_block=%u nval_per_block=%u, nval_per_thread=%u\n", nsamp_per_block, nval_per_block, nval_per_thread);
  fprintf (stderr, "input_transpose_FT_to_TF: nbytes=%lu, nsamp=%lu, nval=%lu\n", nbytes, nsamp, nval);
  fprintf (stderr, "input_transpose_FT_to_TF: nthread=%d, nblocks=%d\n", nthread, nblocks);
  fprintf (stderr, "input_transpose_FT_to_TF: input=%p output=%p sdata_bytes=%ld\n", d_in, d_out, sdata_bytes);
#endif

  input_transpose_FT_to_TF_kernel<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in, (int16_t *) d_out, nsamp, nchan, nval, nval_per_thread, nsamp_per_block);

#ifdef _GDEBUG
  check_error_stream ("input_transpose_FT_to_TF", stream);
#endif
}
*/

#if 0
//
// Perform sample delay correction, out-of-place
//
void mopsr_input_delay_sample (cudaStream_t stream, void * d_in,  void * d_out, 
                               int64_t ** byte_offsets, int64_t ** sample_offsets,
                               mopsr_delay_t ** delays, char first_time, 
                               const unsigned nchan, const unsigned nant, 
                               uint64_t bytes, char * block_full)
{
  assert (d_in != d_out);

  // flag for whether d_out contains a full block of data on this call
  *block_full = 1;

  const unsigned nchanant = nchan * nant;
  size_t chanant_stride = bytes / nchanant;
  size_t to_copy, i_offset, o_offset, shift;

  for (ichan=0; ichan < nchan; ichan++)
  {
    for (iant=0; iant < nant; iant++)
    {
      // sample offset of the input [initially 0]
      i_offset = i_byte_offsets[ichan][iant];

      // bytes to copy back for this chanant
      to_copy = chanant_stride - i_offset;

      // shift is new offset - old offset [samples]
      shift = sample_offsets[ichan][iant] - delays[iant][ichan].samples;
      if (shift != 0)
      {
        to_copy  -= shift * 2;
        i_offset += shift * 2;
      }

      // o_byte_offsets are [initially 0]
      o_offset = o_byte_offsets[ichan][iant];

      // dont overfill the buffer
      if (o_offset + to_copy > chanant_stride)
        to_copy = chanant_stride - o_offset;

      // bytes delayed can differ for *every* channel and antenna
      error = cudaMemcpyAsync (d_out + o_offset, d_in + i_offset, to_copy, cudaMemcpyDeviceToDevice, ctx->stream);
      if (error != cudaSuccess)
      {
        fprintf (stderr, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
        return -1;
      }

      // record how many byters we have copied
      o_byte_offsets[ichan][iant] += to_copy;
      i_byte_offsets[ichan][iant] = d_offset + to_copy;

      if (o_byte_offsets[ichan][iant] < chanant_stride)
        *block_full = 0;

      d_out += chanant_stride;
      d_in  += chanant_stride;
    }
  }

  if (*block_full)
  {
    ipcio_close_block_write (ctx->out_hdu->data_block, bytes);
    ctx->block_open = 0;

    if (!ctx->block_open)
    {
      ctx->curr_block = ipcio_open_block_write (ctx->out_hdu->data_block, &out_block_id);
      ctx->block_open = 1;
    }

    h_ptr = ctx->curr_block;
    d_ptr = ctx->d_out;

    // copy anything remaining in the d_ptr
    for (ichan=0; ichan < nchan; ichan++)
    {
      for (iant=0; iant < nant; iant++)
      {
        d_offset = ctx->d_byte_offsets[ichan][iant];
        to_copy = chanant_stride - d_offset;

        // no shifting this time! [TODO check this]

        // new block so h_offset = 0
        h_offset = ctx->h_byte_offsets[ichan][iant] = 0;

        error = cudaMemcpyAsync (h_ptr + h_offset, d_ptr + d_offset, to_copy, cudaMemcpyDeviceToHost, ctx->stream);
        if (error != cudaSuccess)
        {
          fprintf (stderr, "cudaMemcpyAsyc D2H failed: %s\n", cudaGetErrorString(error));
          return -1;
        }

        // record how many byters we have copied
        ctx->h_byte_offsets[ichan][iant] += to_copy;
        ctx->d_byte_offsets[ichan][iant] = d_offset + to_copy;
      }
    }
  }
}

void check_error_stream (const char* method, cudaStream_t stream)
{
  if (!stream)
  {
    fprintf (stderr, "called check_error_stream on invalid stream\n");
    exit (1);
  }
  else
  {
    cudaStreamSynchronize (stream);

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
    {
      fprintf (stderr,  "method=%s, cudaGetLastError=%s\n", method, cudaGetErrorString (error));
      exit (1);
    }
  }
}
#endif

