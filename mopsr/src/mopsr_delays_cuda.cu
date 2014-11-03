#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <inttypes.h>
#include <stdio.h>

#include "mopsr_cuda.h"
#include "mopsr_delays_cuda.h"

// maximum number of channels * antenna from 1 PFB 128 * 16
#define MOPSR_PFB_CHANANT_MAX 640 
#define MOPSR_MAX_ANT         352
#define WARP_SIZE             32
#define SPECTRAL_DELAYS       1
//#define _GDEBUG               1

__constant__ float d_ant_scales_delay [MOPSR_MAX_NANT_PER_AQ];

int mopsr_transpose_delay_alloc (transpose_delay_t * ctx,
                                 uint64_t block_size, unsigned nchan,
                                 unsigned nant, unsigned ntap)
{
  ctx->nchan = nchan;
  ctx->nant = nant;
  ctx->ntap = ntap;
  ctx->half_ntap = ntap / 2;
  const unsigned nchanant = nchan * nant;
  const unsigned ndim = 2;

  ctx->curr = (transpose_delay_buf_t *) malloc (sizeof(transpose_delay_buf_t));
  ctx->next = (transpose_delay_buf_t *) malloc (sizeof(transpose_delay_buf_t));
  ctx->buffer_size = block_size + (ndim * nchanant * ctx->half_ntap * 2);

#ifdef SPECTRAL_DELAYS
  size_t counter_size = ctx->nchan * ctx->nant * sizeof(unsigned);
#else
  size_t counter_size = ctx->nant * sizeof(unsigned);
#endif

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
  if (error != cudaSuccess)
  {
    fprintf (stderr, "mopsr_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buffer_size);
    return -1;
  }

  buf->counter_size = counter_size;

/*
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
*/

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

void mopsr_transpose_delay_reset (transpose_delay_t * ctx)
{
  ctx->first_kernel = 1;
}


int mopsr_transpose_delay_dealloc (transpose_delay_t * ctx)
{
  mopsr_transpose_delay_buf_dealloc (ctx->curr);
  mopsr_transpose_delay_buf_dealloc (ctx->next);

  return 0;
}

int mopsr_transpose_delay_buf_dealloc (transpose_delay_buf_t * ctx)
{
/*
  if (ctx->d_out_from)
    cudaFree (ctx->d_out_from);
  ctx->d_out_from = 0;

  if (ctx->d_in_from)
    cudaFree (ctx->d_in_from);
  ctx->d_in_from = 0;

  if (ctx->d_in_to)
    cudaFree (ctx->d_in_to);
  ctx->d_in_to = 0;
*/
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

  if (ctx->d_buffer)
    cudaFree(ctx->d_buffer);
  ctx->d_buffer =0;

  return 0;
}

#ifdef SPECTRAL_DELAYS
__constant__ unsigned curr_out_from[MOPSR_PFB_CHANANT_MAX];
__constant__ unsigned curr_in_from[MOPSR_PFB_CHANANT_MAX];
__constant__ unsigned curr_in_to[MOPSR_PFB_CHANANT_MAX];
__constant__ unsigned next_out_from[MOPSR_PFB_CHANANT_MAX];
__constant__ unsigned next_in_from[MOPSR_PFB_CHANANT_MAX];
__constant__ unsigned next_in_to[MOPSR_PFB_CHANANT_MAX];
#else
__constant__ unsigned curr_out_from[MOPSR_MAX_NANT_PER_AQ];
__constant__ unsigned curr_in_from[MOPSR_MAX_NANT_PER_AQ];
__constant__ unsigned curr_in_to[MOPSR_MAX_NANT_PER_AQ];
__constant__ unsigned next_out_from[MOPSR_MAX_NANT_PER_AQ];
__constant__ unsigned next_in_from[MOPSR_MAX_NANT_PER_AQ];
__constant__ unsigned next_in_to[MOPSR_MAX_NANT_PER_AQ];
#endif

__global__ void mopsr_transpose_delay_kernel (
     int16_t * in,
     int16_t * curr,
     int16_t * next,
     const unsigned nchan, const unsigned nant, const unsigned nval, 
     const unsigned nval_per_thread, const unsigned in_block_stride, 
     const unsigned nsamp_per_block, const unsigned out_chanant_stride)
{
  // for loaded data samples
  extern __shared__ int16_t sdata[];

  const unsigned nchanant = nchan * nant;

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

    //if (blockIdx.x == 0)
    //  printf ("[%d][%d] read data[%u]=%d [in_idx=%u]\n", blockIdx.x, threadIdx.x, sin_idx, sdata[sin_idx], in_idx);
    in_idx += WARP_SIZE;
    sin_idx += WARP_SIZE;
  }

  __syncthreads();

  // our thread number within the warp [0-32], also the time sample this will write each time
  const unsigned isamp = warp_idx;

  // starting ichan/ant
  unsigned ichanant = warp_num * nval_per_thread;

  // determine which shared memory index for this output ichan and isamp
  unsigned sout_idx = (isamp * nchanant) + ichanant;

  // which sample number in the kernel this thread is writing
  const unsigned isamp_kernel = (blockIdx.x * nsamp_per_block) + (isamp);

  // vanilla output index for this thread
  uint64_t out_idx = (ichanant * out_chanant_stride) + isamp_kernel;

  int64_t curr_idx, next_idx;

#ifdef SPECTRAL_DELAYS
  for (ival=0; ival<nval_per_thread; ival++)
  {
    if ((curr_in_from[ichanant] <= isamp_kernel) && (isamp_kernel < curr_in_to[ichanant]))
    {
      curr_idx = (int64_t) out_idx + curr_out_from[ichanant] - curr_in_from[ichanant];
      curr[curr_idx] = sdata[sout_idx];
      //if ((ichanant == 0) && (blockIdx.x > 1000 && blockIdx.x < 1010 )) 
      //  printf ("[%d][%d] wrote curr[%ld] = sdata[%u] == %d\n", blockIdx.x, threadIdx.x, curr_idx, sout_idx, sdata[sout_idx]); 
    }

    if ((next_in_from[ichanant] <= isamp_kernel) && (isamp_kernel < next_in_to[ichanant]))
    {
      next_idx = (int64_t) out_idx + next_out_from[ichanant] - next_in_from[ichanant];
      next[next_idx] = sdata[sout_idx];
      //if ((ichanant == 0) && (blockIdx.x > 1000 && blockIdx.x < 1010 )) 
      //  printf ("[%d][%d] wrote next[%ld] = sdata[%u] == %d\n", blockIdx.x, threadIdx.x, next_idx, sout_idx, sdata[sout_idx]); 
    }

    sout_idx ++;
    out_idx += out_chanant_stride;
    ichanant++;
  }
#else
  unsigned iant;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    iant = ichanant % nant;

    if ((curr_in_from[iant] <= isamp_kernel) && (isamp_kernel < curr_in_to[iant]))
    {
      curr_idx = (int64_t) out_idx + curr_out_from[iant] - curr_in_from[iant];
      curr[curr_idx] = sdata[sout_idx];
    }

    if ((next_in_from[iant] <= isamp_kernel) && (isamp_kernel < next_in_to[iant]))
    {
      next_idx = (int64_t) out_idx + next_out_from[iant] - next_in_from[iant];
      next[next_idx] = sdata[sout_idx];
    }

    sout_idx ++;
    out_idx += out_chanant_stride;
    ichanant++;
  }
#endif
}

void * mopsr_transpose_delay (cudaStream_t stream, transpose_delay_t * ctx, void * d_in, uint64_t nbytes, mopsr_delay_t ** delays)
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
  int shift;

#ifdef SPECTRAL_DELAYS
  unsigned ichanant = 0;

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
        if (nsamp + (2 * ctx->half_ntap) < ctx->curr->h_off[ichanant])
          ctx->curr->h_in_to[ichanant] = 0;

        // next always uses new delays
        ctx->next->h_out_from[ichanant] = 0;
        ctx->next->h_in_from[ichanant]  = ctx->curr->h_in_to[ichanant] - (2 * ctx->half_ntap);
        ctx->next->h_in_to[ichanant]    = nsamp;

        // handle a change in sample level delay this should be right
        shift = delays[iant][ichan].samples - ctx->curr->h_delays[ichanant];

        ctx->next->h_in_from[ichanant] += shift;
        ctx->next->h_delays[ichanant]   = delays[iant][ichan].samples;
        ctx->next->h_off[ichanant]      = ctx->next->h_in_to[ichanant] - ctx->next->h_in_from[ichanant];

      }
      ichanant++;
    }
  }

#else

  ichan = 0;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if (ctx->first_kernel)
    {
      ctx->curr->h_delays[iant]   = delays[iant][0].samples;
      ctx->next->h_delays[iant]   = delays[iant][0].samples;

      ctx->curr->h_out_from[iant] = 0;
      ctx->curr->h_in_from[iant]  = ctx->curr->h_delays[iant] - ctx->half_ntap;
      ctx->curr->h_in_to[iant]    = nsamp;
      ctx->curr->h_off[iant]      = ctx->curr->h_in_to[iant] - ctx->curr->h_in_from[iant];

      // should never be used on first iteration
      ctx->next->h_out_from[iant] = 0;
      ctx->next->h_in_from[iant]  = nsamp;
      ctx->next->h_in_to[iant]    = 2 * nsamp;

      //if (iant == 0)
      //  fprintf (stderr, "1: h_out_from=%u h_in_from=%u h_in_to=%u h_off=%u\n", 
      //            ctx->curr->h_out_from[0], ctx->curr->h_in_from[0],
      //            ctx->curr->h_in_to[0], ctx->curr->h_off[0]);

    }
    else
    {
      // curr always uses delays from previous iteration
      ctx->curr->h_out_from[iant] = ctx->curr->h_off[iant];
      ctx->curr->h_in_from[iant]  = 0;
      ctx->curr->h_in_to[iant]    = nsamp + (2 * ctx->half_ntap) - ctx->curr->h_off[iant];
      if (nsamp + (2 * ctx->half_ntap) < ctx->curr->h_off[iant])
        ctx->curr->h_in_to[iant] = 0;
      //ctx->curr->h_off[iant]      = 0;  // no longer required

      //if (iant == 0)
      //  fprintf (stderr, "2: h_out_from=%u h_in_from=%u h_in_to=%u h_off=%u\n", 
      //            ctx->curr->h_out_from[0], ctx->curr->h_in_from[0],
      //            ctx->curr->h_in_to[0], ctx->curr->h_off[0]);

      // next always uses new delays
      ctx->next->h_out_from[iant] = 0;
      ctx->next->h_in_from[iant]  = ctx->curr->h_in_to[iant] - (2 * ctx->half_ntap);
      ctx->next->h_in_to[iant]    = nsamp;

      // handle a change in sample level delay
      shift = delays[iant][ichan].samples - ctx->curr->h_delays[iant];

      ctx->next->h_in_from[iant] += shift;
      ctx->next->h_delays[iant]   = delays[iant][ichan].samples;
      ctx->next->h_off[iant]      = ctx->next->h_in_to[iant] - ctx->next->h_in_from[iant];
    }
  }
#endif

/*
 */

  cudaMemcpyToSymbolAsync(curr_out_from, (void *) ctx->curr->h_out_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(curr_in_from, (void *) ctx->curr->h_in_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(curr_in_to, (void *) ctx->curr->h_in_to, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_out_from, (void *) ctx->next->h_out_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_in_from, (void *) ctx->next->h_in_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_in_to, (void *) ctx->next->h_in_to, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);

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

  const size_t sdata_bytes = (nthread * ndim * nval_per_thread);// + (6 * ctx->nchan * ctx->nant * sizeof(unsigned));
  const unsigned in_block_stride = nthread * nval_per_thread;
  const unsigned out_chanant_stride = ndat + (2 * ctx->half_ntap);

#ifdef _GDEBUG
  fprintf (stderr, "transpose_delay: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "transpose_delay: nbytes=%lu, ndat=%lu, nval=%lu\n", nbytes, ndat, nval);
  fprintf (stderr, "transpose_delay: nthread=%d, nblocks=%d sdata_bytes=%d\n", nthread, nblocks, sdata_bytes);
  fprintf (stderr, "transpose_delay: out_chanant_stride=%u\n", out_chanant_stride);
#endif

  mopsr_transpose_delay_kernel<<<nblocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in, 
        (int16_t *) ctx->curr->d_buffer, //ctx->curr->d_out_from, ctx->curr->d_in_from, ctx->curr->d_in_to,
        (int16_t *) ctx->next->d_buffer, //ctx->next->d_out_from, ctx->next->d_in_from, ctx->next->d_in_to,
        ctx->nchan, ctx->nant, nval, nval_per_thread, in_block_stride, nsamp_per_block, out_chanant_stride);

#if _GDEBUG
  check_error_stream("mopsr_transpose_delay_kernel", stream);
#endif

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



// fringe co-efficients are fast in constant memory here
__constant__ float fringe_coeffs[MOPSR_PFB_CHANANT_MAX];
__constant__ float delays_ds[MOPSR_PFB_CHANANT_MAX];
__constant__ float fringe_coeffs_ds[MOPSR_PFB_CHANANT_MAX];

// apply a fractional delay correction to a channel / antenna, warps will always
__global__ void mopsr_fringe_rotate_kernel (int16_t * input, uint64_t ndat)
{
  const unsigned isamp = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned iant  = blockIdx.y;
  const unsigned nant  = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;
  const uint64_t idx = ichanant * ndat + isamp;

  if (isamp >= ndat)
    return;

  // using constant memory should result in broadcast for this block/half warp
  float fringe_coeff = fringe_coeffs[ichanant];
  cuComplex fringe_phasor = make_cuComplex (cosf(fringe_coeff), sinf(fringe_coeff));

  int16_t val16 = input[idx];
  int8_t * val8ptr = (int8_t *) &val16;
  const float scale = d_ant_scales_delay[iant];

  float re = ((float) (val8ptr[0]) + 0.5) * scale;
  float im = ((float) (val8ptr[1]) + 0.5) * scale;
  cuComplex val = make_cuComplex (re, im);
  cuComplex rotated = cuCmulf(val, fringe_phasor);

  //if ((blockIdx.x == 0) && (threadIdx.x <= 10) && (iant == 1) && (ichan == 30))
  //  printf ("[%d][%d] val (%f, %f) phasor(%f, %f) fringe=%f\n", ichan, threadIdx.x, cuCrealf(val), cuCimagf(val), cuCrealf(fringe_phasor), cuCimagf(fringe_phasor), fringe_coeff);

  val8ptr[0] = (int8_t) rintf (cuCrealf(rotated) - 0.5);
  val8ptr[1] = (int8_t) rintf (cuCimagf(rotated) - 0.5);

  input[idx] = val16;
}

//
// Perform fractional delay correction, out-of-place
//
void mopsr_fringe_rotate (cudaStream_t stream, void * d_in,
                          float * h_fringes, size_t fringes_size,
                          uint64_t nbytes, unsigned nchan,
                          unsigned nant)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);

  // number of threads that actually load data
  unsigned nthread = 1024;

  dim3 blocks (ndat / nthread, nant, nchan);
  if (ndat % nthread)
    blocks.x++;

  cudaMemcpyToSymbolAsync(fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);

#if _GDEBUG
  fprintf (stderr, "fringe_rotate: bytes=%lu ndat=%lu\n", nbytes, ndat);
  fprintf (stderr, "fringe_rotate: nthread=%d, blocks.x=%d, blocks.y=%d, blocks.z=%d\n", nthread, blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "fringe_rotate: d_in=%p h_fringes=%p\n", (void *) d_in, (void*) h_fringes);
#endif

  mopsr_fringe_rotate_kernel<<<blocks, nthread, 0, stream>>>((int16_t *) d_in, ndat);

#if _GDEBUG
  check_error_stream("mopsr_fringe_rotate_kernel", stream);
#endif
}


/*
__global__ void mopsr_print_ant_scales (unsigned nant)
{
  unsigned iant;
  for (iant=0; iant<nant; iant++)
  {
    printf("d_ant_scales[%d]=%f\n", iant, d_ant_scales_delay[iant]); 
  }
}
*/

void mopsr_delay_copy_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes)
{
  cudaMemcpyToSymbolAsync (d_ant_scales_delay, (void *) h_ant_scales, nbytes, 0, cudaMemcpyHostToDevice, stream);
  //int blocks_test = 1;
  //int threads_test = 1;
  //mopsr_print_ant_scales<<<blocks_test, threads_test, 0, stream>>>(nbytes/sizeof(float));
}


// apply a fractional delay correction to a channel / antenna, warps will always 
__global__ void mopsr_delay_fractional_kernel (int16_t * input, int16_t * output, 
                                               float * delays,
                                               unsigned nthread_run, 
                                               uint64_t nsamp_in, 
                                               const unsigned chan_stride, 
                                               const unsigned ant_stride, 
                                               const unsigned ntap)
{
  //extern __shared__ float fk_shared[];

  // the input data for block are stored in blockDim.x values
  extern __shared__ cuComplex fk_shared16[];
  
  // the FIR filter stored in the final NTAP values
  float * filter = (float *) (fk_shared16 + blockDim.x);

  //const unsigned ndim = 2;
  const unsigned half_ntap = (ntap / 2);
  const unsigned in_offset = 2 * half_ntap;
  //const unsigned idx = blockIdx.x * nthread_run + threadIdx.x;
  const unsigned isamp = blockIdx.x * nthread_run + threadIdx.x;
  // iant  = blockIdx.y;
  // nant  = gridDim.y;
  // ichan = blockIdx.z;
  const unsigned ichanant = blockIdx.z * gridDim.y + blockIdx.y;

  const unsigned nsamp_out = nsamp_in - in_offset;

#if 0
  const float isamp_offset = (float) isamp - ((float) nsamp_out) / 2;

  // using constant memory should result in broadcast for this block/half warp
  // handle change in delay across the block
  float delay = delays[ichanant] + (delays_ds[ichanant] * isamp_offset);
  float fringe_coeff = fringe_coeffs[ichanant] + (fringe_coeffs_ds[ichanant] * isamp_offset);
#else
  float delay = delays[ichanant];
  float fringe_coeff = fringe_coeffs[ichanant];
#endif

  cuComplex fringe_phasor = make_cuComplex (cosf(fringe_coeff), sinf(fringe_coeff));

  // calculate the filter coefficients for the delay
  if (threadIdx.x < ntap)
  {
    float x = ((float) threadIdx.x) - delay;
    float window = 0.54 - 0.46 * cos(2.0 * M_PI * (x+0.5) / ntap);
    float sinc = 1;
    if (x != half_ntap)
    {
      x -= half_ntap;
      x *= M_PI;
      sinc = sinf(x) / x;
    }
    //filter[threadIdx.x] = sinc;
    filter[threadIdx.x] = sinc * window;
  }
  
  if (isamp >= nsamp_in)
  {
    return;
  }

  // each thread must also load its data from main memory here chan_stride + ant_stride
  // const unsigned in_data_idx  = ichanant * nsamp_in + isamp;
  // const unsigned out_data_idx = ichanant * nsamp_out + isamp;

  int16_t val16 = input[ichanant * nsamp_in + isamp];
  int8_t * val8ptr = (int8_t *) &val16;

  {
    const float scale = d_ant_scales_delay[blockIdx.y];
    cuComplex val = make_cuComplex ((float) (val8ptr[0]) + 0.5, (float) (val8ptr[1]) + 0.5);
    val.x *= scale;
    val.y *= scale;
    fk_shared16[threadIdx.x] = cuCmulf(val, fringe_phasor);
  }

  __syncthreads();

  // there are 2 * half_ntap threads that dont calculate anything
  if ((threadIdx.x < nthread_run) && (isamp < nsamp_out))
  {
    float re = 0;
    float im = 0;
    for (unsigned i=0; i<ntap; i++)
    {
      re += cuCrealf(fk_shared16[threadIdx.x + i]) * filter[i];
      im += cuCimagf(fk_shared16[threadIdx.x + i]) * filter[i];
    }
    
    val8ptr[0] = (int8_t) rintf (re - 0.5);
    val8ptr[1] = (int8_t) rintf (im - 0.5);

    output[ichanant * nsamp_out + isamp] = val16;
  }
}

// apply a fractional delay correction to a channel / antenna, warps will always 
__global__ void mopsr_delay_fractional_float_kernel (int16_t * input, 
                    float * output, float * delays, 
                    unsigned nthread_run, uint64_t nsamp_in, 
                    const unsigned chan_stride, const unsigned ant_stride, 
                    const unsigned ntap)
{
  extern __shared__ float fk_shared[];

  cuComplex * data = (cuComplex *) fk_shared;
  float * filter = (float *) (data + blockDim.x);

  const unsigned ndim = 2;
  const unsigned half_ntap = ntap / 2;

  // iant  : blockIdx.y
  // ichan : blockIdx.z
  // nant  : gridDim.y
  const unsigned ichanant  = (blockIdx.z * gridDim.y) + blockIdx.y;
  const unsigned isamp     = (blockIdx.x * nthread_run) + threadIdx.x;
  const unsigned nsamp_out = nsamp_in - (2 * half_ntap);
  const float isamp_offset = (float) isamp - ((float) nsamp_out) / 2;

  // using constant memory should result in broadcast for this 
  // block/half warp.  handle change in delay across the block
  float fringe_coeff      = fringe_coeffs[ichanant] + (fringe_coeffs_ds[ichanant] * isamp_offset);
  cuComplex fringe_phasor = make_cuComplex (cosf(fringe_coeff), sinf(fringe_coeff));

  // each thread loads a complex pair from input
  int16_t val16 = input[(ichanant * nsamp_in) + isamp];
  int8_t * val8ptr = (int8_t *) &val16;
    
  cuComplex val = make_cuComplex ((float) (val8ptr[0]) + 0.5, (float) (val8ptr[1]) + 0.5);
  data[threadIdx.x] = cuCmulf(val, fringe_phasor);

  // calculate the filter coefficients for the delay
  if (threadIdx.x < ntap)
  {
    const float delay = delays[ichanant] + (delays_ds[ichanant] * isamp_offset);
    float x           = ((float) threadIdx.x) - delay;
    float window      = 0.54 - 0.46 * cos(2.0 * M_PI * (x+0.5) / ntap);
    float sinc        = 1;
    if (x != half_ntap)
    {
      x -= half_ntap;
      x *= M_PI;
      sinc = sinf(x) / x;
    }
    filter[threadIdx.x] = sinc * window;
  }

  __syncthreads();

  // there are 2 * half_ntap threads that dont calculate anything
  if (threadIdx.x < nthread_run)
  {
    unsigned osamp = (blockIdx.x * nthread_run * ndim) + threadIdx.x;
    const unsigned nfloat_out = nsamp_out * ndim;

    // increment the base pointed to the right block
    output += (ichanant * nfloat_out);

    if (osamp < nfloat_out)
    {
      float * dataf = (float *) data;

      // pointer to first value in shared memory
      dataf += threadIdx.x;

      // compute sinc delayed float
      float val = 0;
      for (unsigned i=0; i<ntap; i++)
        val += dataf[2*i] * filter[i];

      //if (ichanant == 0 && blockIdx.x == 60)
      //  printf ("[%d][%d] output[%u]=%f\n", blockIdx.x, threadIdx.x, osamp, val);
      
      // write output to gmem
      output[osamp] = val;

      // increment shared memory pointer by number of active threads
      dataf += nthread_run;

      // increment output pointer by number of active threads
      osamp += nthread_run;

      if (osamp < nfloat_out)
      {
        // compute sinc delayed float
        val = 0;
        for (unsigned i=0; i<ntap; i++)
          val += dataf[2*i] * filter[i];

        //if (ichanant == 0 && blockIdx.x == 60)
        //  printf ("[%d][%d] output[%u]=%f\n", blockIdx.x, threadIdx.x, osamp, val);

        // write output to gmem
        output[osamp] = val;
      }
    }
  }
}

// 
// Perform fractional delay correction, out-of-place
//
void mopsr_delay_fractional (cudaStream_t stream, void * d_in, void * d_out,
                             float * d_delays, float * h_fringes, 
                             float * h_delays_ds, float * h_fringe_coeffs_ds, 
                             size_t fringes_size, 
                             uint64_t nbytes, unsigned nchan, 
                             unsigned nant, unsigned ntap)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  const unsigned half_ntap = ntap / 2;

  // number of threads that actually load data
  unsigned nthread_load = 1024;
  if (ndat < nthread_load)
    nthread_load = ndat;
  unsigned nthread_run  = nthread_load - (2 * half_ntap);

  // need shared memory to load the ntap coefficients + nthread_load data points
  const size_t   sdata_bytes = (nthread_load * ndim + ntap) * sizeof(float);

  dim3 blocks (ndat / nthread_run, nant, nchan);
  if (ndat % nthread_load)
    blocks.x++;

  //fprintf (stderr, "delay_fractional: copying fringe's to symbold (%ld bytes)\n", fringes_size);
  cudaMemcpyToSymbolAsync (fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync (delays_ds, (void *) h_delays_ds, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync (fringe_coeffs_ds, (void *) h_fringe_coeffs_ds, fringes_size, 0, cudaMemcpyHostToDevice, stream);

#if _GDEBUG
  fprintf (stderr, "delay_fractional: bytes=%lu ndat=%lu sdata_bytes=%ld\n", nbytes, ndat, sdata_bytes);
  fprintf (stderr, "delay_fractional: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "delay_fractional: nthread_load=%d nthread_run=%d ntap=%d\n", nthread_load, nthread_run, ntap);
#endif

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

  mopsr_delay_fractional_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in, (int16_t *) d_out, 
                (float *) d_delays, nthread_run, ndat, chan_stride, ant_stride, ntap);

#if _GDEBUG
  check_error_stream("mopsr_delay_fractional_kernel", stream);
#endif
}


//
// Compute the S1 and S2 sums for blocks of input data, writing the S1 and S2 sums out to Gmem
//
__global__ void mopsr_skcompute_kernel (cuFloatComplex * in, cuFloatComplex * sums, const unsigned nval_per_thread, const uint64_t ndat)
{
  extern __shared__ float sdata_skc[];

  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;
  const uint64_t in_offset = ndat * ichanant;

  // offset into the block for the current channel and antenna
  cuFloatComplex * indat = in + in_offset;

  unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) * nval_per_thread;
  const unsigned s1 = (threadIdx.x*2);
  const unsigned s2 = (threadIdx.x*2) + 1;

  cuFloatComplex val;
  float s1_sum = 0;
  float s2_sum = 0;
  float power;

  for (unsigned ival=0; ival<nval_per_thread; ival++)
  {
    if (idx < ndat)
    {
      val = indat[idx];
      power = (val.x * val.x) + (val.y * val.y);
      s1_sum += power;
      s2_sum += (power * power);
    }
    idx += blockDim.x;
  }

  sdata_skc[s1] = s1_sum;
  sdata_skc[s2] = s2_sum;

  __syncthreads();

  // This is a parallel reduction. On kepler+ cards this could be done better using
  // shuf_down(), but lets keep it generic for now

  int last_offset = blockDim.x/2 + blockDim.x % 2;

  for (int offset = blockDim.x/2; offset > 0;  offset >>= 1)
  {
    // add a partial sum upstream to our own
    if (threadIdx.x < offset)
    {
      sdata_skc[s1] += sdata_skc[s1 + (2*offset)];
      sdata_skc[s2] += sdata_skc[s2 + (2*offset)];
    }

    __syncthreads();

    // special case for non power of 2 reductions
    if ((last_offset % 2) && (last_offset > 2) && (threadIdx.x == offset))
    {
      sdata_skc[0] += sdata_skc[s1 + (2*offset)];
      sdata_skc[1] += sdata_skc[s2 + (2*offset)];
    }

    last_offset = offset;

    // wait until all threads in the block have updated their partial sums
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    // FST ordered
    const unsigned out_idx = (ichanant * gridDim.x) +  blockIdx.x;
    sums[out_idx].x = sdata_skc[0];
    sums[out_idx].y = sdata_skc[1];
    //if ((blockIdx.x == 0) && (ichan == 1) && (iant == 1))
    //  printf ("s1_sum=%f s2_sum=%f\n", sdata_skc[0], sdata_skc[1]);
  }
}

#if HAVE_CUDA_SHUFFLE
__inline__ __device__
float warpReduceSumF(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSumF(float val) {

  __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumF(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSumF(val); //Final reduce within first warp

  return val;
}

__inline__ __device__
int warpReduceSumI(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
int blockReduceSumI(int val) {

  __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumI(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSumI(val); //Final reduce within first warp

  return val;
}
#endif

//
// take the S1 and S2 values in sums.x and sums.y that were computed 
//  from M samples, and integrate of nsums blocks to
// compute a sk mask and zap
//
__global__ void mopsr_skmask_kernel (float * in, int8_t * out, cuFloatComplex * sums, 
                                     curandState * rstates, float * sigmas,
                                     unsigned nsums, unsigned M, unsigned nval_per_thread, 
                                     unsigned nsamp_per_thread, uint64_t ndat)
{
  // Pearson Type IV SK limits for 3sigma RFI rejection, based on 2^index

  // maximum to be 16384 samples (20.97152 ms)
  unsigned sk_idx_max = 20;

  const float sk_low[20]  = { 0, 0, 0, 0, 0,
                              0.387702, 0.492078, 0.601904, 0.698159, 0.775046,
                              0.834186, 0.878879, 0.912209, 0.936770, 0.954684,
                              0.967644, 0.976961, 0.983628, 0.988382, 0.991764 };
  const float sk_high[20] = { 0, 0, 0, 0, 0,
                              2.731480, 2.166000, 1.762970, 1.495970, 1.325420,
                              1.216950, 1.146930, 1.100750, 1.069730, 1.048570,
                              1.033980, 1.023850, 1.016780, 1.011820, 1.008340 };
/*
  const float sk_low[20]  = { 0, 0, 0, 0, 0,
                              0.274561, 0.363869, 0.492029, 0.613738, 0.711612,
                              0.786484, 0.843084, 0.885557, 0.917123, 0.940341,
                              0.957257, 0.969486, 0.978275, 0.984562, 0.989046 };
  const float sk_high[20] = { 0, 0, 0, 0, 0,
                              4.27587, 3.11001, 2.29104, 1.784, 1.48684,
                              1.31218, 1.20603, 1.13893, 1.0951, 1.06577,
                              1.0458, 1.03204, 1.02249, 1.01582, 1.01115 };
*/

  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;

  const unsigned id = ichanant * blockDim.x + threadIdx.x;
  curandState localState = rstates[id];

  // zap mask for each set of M samples
  extern __shared__ char smask[];

  // initialize zap mask to 0
  {
    unsigned idx = threadIdx.x;
    for (unsigned ival=0; ival<nval_per_thread; ival++)
    {
      if (idx < nsums)
      {
        smask[idx] = 0;
        idx += blockDim.x;
      }
    }
  }

  __syncthreads();

  const unsigned log2_M = (unsigned) log2f (M);
  unsigned idx = threadIdx.x;

  // 1 standard deviation for the input data, 0 indicates not value yet computed
  float sigma = sigmas[ichanant];
  float s1_thread = 0;
  int s1_count = 0;

  // sums data stored as FST
  sums += (ichanant * nsums);

  for (unsigned ival=0; ival<nval_per_thread; ival++)
  {
    if (idx < nsums)
    {
      for (unsigned sk_idx = log2_M; sk_idx < sk_idx_max; sk_idx ++)
      {
        unsigned powers_to_add = sk_idx - log2_M;
        unsigned to_add = (unsigned) exp2f(powers_to_add);

        if (idx + to_add <= nsums)
        {
          const float m = M * to_add;
          const float m_fac = (m + 1) / (m - 1);
          float s1 = 0;
          float s2 = 0;

          for (unsigned ichunk=0; ichunk < to_add; ichunk++)
          {
            s1 += sums[idx + ichunk].x;
            s2 += sums[idx + ichunk].y;
          }

          const float sk_estimate = m_fac * (m * (s2 / (s1 * s1)) - 1);

          //if ((iant == 0) && (ichan == 0))
          //  printf ("m=%f m_fac=%f s1=%f, s2=%f [%f < %f < %f]\n", m, m_fac, s1, s2, sk_low[sk_idx], sk_estimate, sk_high[sk_idx]); 
          if ((sk_estimate < sk_low[sk_idx]) || (sk_estimate > sk_high[sk_idx]))
          {
            for (unsigned ichunk=0; ichunk < to_add; ichunk++)
            {
              smask[idx+ichunk] = 1;
            }
          }
          else
          {
            if (sk_idx == log2_M)
            {
              s1_thread += s1;
              s1_count++;
            }
          }
        }
      }
      //if ((iant == 1) && (ichan == 1))
      //  printf ("%u: S1=%f s1_thread=%f\n", threadIdx.x, sums[idx].x, s1_thread);
      idx += blockDim.x;
    }
  }

  if (sigma == 0)
  {
    // since s1 will have twice the variance of the Re/Im components, / 2
    s1_thread /= (2 * M);

    // compute the sum of the sums[].x for all the block
#if HAVE_CUDA_SHUFFLE
    s1_thread = blockReduceSumF (s1_thread);
    s1_count = blockReduceSumI (s1_count);
#endif

    // sync here to be sure the smask is now updated
    __syncthreads();

    __shared__ float block_sigma;

    if (threadIdx.x == 0)
    {
      sigma = 0;
      if (s1_count > 0)
        sigma = sqrtf (s1_thread / s1_count);

      sigmas[ichanant] = sigma;

      block_sigma = sigma;
    }

    __syncthreads();

    sigma = block_sigma;
  }

  // Jenet & Anderson 1998, 6-bit (2-bits for RFI) spacing
  //const float spacing = 0.09925;    // 6-bit
  const float spacing = 0.02957;      // 8-bit

  // dont do antenna scaling here anymore for the moment, unless it is zero
  const float ant_scale = d_ant_scales_delay[iant];
  float data_factor = ant_scale / (sigma * spacing);
  if (ant_scale < 0.01)
    data_factor = 0;
  const float rand_factor = ant_scale / spacing;

  // now we want to zap all blocks of input that have an associated mask
  // note that this kernel has only 1 block, with blockDim.x threads that may not match
  const unsigned ndim = 2;
  const unsigned nval_per_sum = M * ndim;
  unsigned block_offset = (ichanant * ndat * ndim);
  float * indat = in + block_offset;
  int8_t * outdat = out + block_offset;

  // foreach block of M samples (i.e. 1 sum)
  for (unsigned isum=0; isum<nsums; isum++)
  {
    // use the threads to write out the int8_t scaled value (or zapped value)
    // back to global memory. There are 2 * M values to write each iteration

    unsigned idx = threadIdx.x;

    if (smask[isum] == 1)
    {
      for (unsigned isamp=0; isamp<nsamp_per_thread; isamp++)
      {
        if (idx < nval_per_sum)
        {
          const float inval = curand_normal (&localState);
          outdat[idx] = (int8_t) rintf(inval * rand_factor);
        }
        idx += blockDim.x;
      }
    }
    else
    {
      for (unsigned isamp=0; isamp<nsamp_per_thread; isamp++)
      {
        if (idx < nval_per_sum)
        {
          outdat[idx] = (int8_t) rintf ((indat[idx] * data_factor) - 0.5);
        }
        idx += blockDim.x;
      }
    }
    outdat += ndim * M;
    indat += ndim * M;
  }

  rstates[id] = localState;
}

__global__ void mopsr_srand_setup_kernel (unsigned long long seed, curandState *states)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init (seed, id, 0, &states[id]);
}

void mopsr_init_rng (cudaStream_t stream, unsigned long long seed, unsigned nrngs, void * states)
{
  unsigned nthreads = 1024;
  unsigned nblocks = nrngs / nthreads;

#if _GDEBUG
  fprintf (stderr, "rand_setup: nblocks=%u nthreads=%u\n", nblocks_r, nthreads_r);
#endif

  mopsr_srand_setup_kernel<<<nblocks, nthreads, 0, stream>>>(seed, (curandState *) states);

#if _GDEBUG
  check_error_stream("mopsr_srand_setup_kernel", stream);
#endif
}

//
// Perform fractional delay correction, compute SK, ZAP, rescale
// out-of-place
//
void mopsr_delay_fractional_sk_scale (cudaStream_t stream, 
     void * d_in, void * d_out, void * d_fbuf, void * d_rstates,
     void * d_sigmas, float * d_delays, float * h_fringes,
     float * h_delays_ds, float * h_fringe_coeffs_ds,
     size_t fringes_size,
     uint64_t nbytes, unsigned nchan,
     unsigned nant, unsigned ntap)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  const unsigned half_ntap = ntap / 2;

  // number of threads that actually load data
  unsigned nthread_load = 1024;
  if (ndat < nthread_load)
    nthread_load = ndat;
  unsigned nthread_run  = nthread_load - (2 * half_ntap);

  // need shared memory to load the ntap coefficients + nthread_load data points
  const size_t   sdata_bytes = (nthread_load * ndim + ntap) * sizeof(float);

  dim3 blocks (ndat / nthread_run, nant, nchan);
  if (ndat % nthread_load)
    blocks.x++;

  cudaMemcpyToSymbolAsync (fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync (delays_ds, (void *) h_delays_ds, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync (fringe_coeffs_ds, (void *) h_fringe_coeffs_ds, fringes_size, 0, cudaMemcpyHostToDevice,
stream);

#if _GDEBUG
  fprintf (stderr, "delay_fractional: bytes=%lu ndat=%lu sdata_bytes=%ld\n", nbytes, ndat, sdata_bytes);
  fprintf (stderr, "delay_fractional: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "delay_fractional: nthread_load=%d nthread_run=%d ntap=%d\n", nthread_load, nthread_run, ntap);
#endif

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

  mopsr_delay_fractional_float_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in, 
                (float *) d_fbuf, (float *) d_delays, nthread_run, 
                ndat, chan_stride, ant_stride, ntap);

#if _GDEBUG
  check_error_stream("mopsr_delay_fractional_float_kernel", stream);
#endif
  ///////////////////////////////////////////////////////// 
  // Calculate kurtosis sums

  // TODO fix this configuration
  unsigned M = 1024;
  unsigned nthreads = 1024;
  const uint64_t ndat_sk = ndat - (ntap - 1);

  unsigned nval_per_thread = 1;
  if (M > nthreads)
    nval_per_thread = M / nthreads;
  else
    nthreads = M;

  // each block is a single integration
  blocks.x = ndat_sk / M;
  size_t shm_bytes = M * ndim * sizeof(float);

#if _GDEBUG
  fprintf (stderr, "mopsr_skcompute_kernel: bytes=%lu ndat=%lu shm_bytes=%ld\n", nbytes, ndat_sk, shm_bytes);
  fprintf (stderr, "mopsr_skcompute_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d, nthreads=%u\n", blocks.x, blocks.y, blocks.z, nthreads);
  fprintf (stderr, "mopsr_skcompute_kernel: d_fbuf=%p d_in=%p, nval_per_thread=%u, ndat_sk=%lu\n", d_fbuf, d_in, nval_per_thread, ndat_sk);
#endif

  // reuse d_in as a temporary work buffer for the S1 and S2 sums
  mopsr_skcompute_kernel<<<blocks, nthreads, shm_bytes, stream>>>( (cuFloatComplex *) d_fbuf, 
                (cuFloatComplex *) d_in, nval_per_thread, ndat_sk);

#if _GDEBUG
  check_error_stream("mopsr_skcompute_kernel", stream);
#endif

  //////////////////////////////////////////////////////////
  // mask the input data
  unsigned nsums = blocks.x;
  shm_bytes = nsums;
  dim3 blocks_skm (1, nant, nchan);
  nthreads = 1024;

  nval_per_thread = 1;
  if (nsums > nthreads)
  {
    nval_per_thread = nsums / nthreads;
    if (nsums % nthreads)
      nval_per_thread++;
  }
  else
    nthreads = nsums;

  unsigned nsamp_per_thread = (M  * ndim) / nthreads;
  if (M % nthreads)
    nsamp_per_thread++;

#if _GDEBUG
  fprintf (stderr, "mopsr_skmask_kernel: blocks_skm.x=%d, blocks_skm.y=%d, blocks_skm.z=%d\n", blocks_skm.x, blocks_skm.y, blocks_skm.z);
  fprintf (stderr, "mopsr_skmask_kernel: nthreads=%u shm_bytes=%ld\n", nthreads, shm_bytes);
  fprintf (stderr, "mopsr_skmask_kernel: d_fbuf=%p d_out=%p, d_in=%p, nsums=%u M=%u, nval_per_thread=%u, nsamp_per_thread=%u ndat_sk=%lu\n", d_fbuf, d_out, d_in, nsums, M, nval_per_thread, nsamp_per_thread, ndat_sk);
#endif

  mopsr_skmask_kernel<<<blocks_skm, nthreads, shm_bytes, stream>>>((float *) d_fbuf, (int8_t *) d_out, 
                (cuFloatComplex *) d_in, (curandState *) d_rstates, (float *) d_sigmas, 
                nsums, M, nval_per_thread, nsamp_per_thread, ndat_sk);

#if _GDEBUG
  check_error_stream("mopsr_skmask_kernel", stream);
#endif
}

// wrapper for getting curandState_t size
size_t mopsr_curandState_size()
{
  return sizeof(curandState_t);
}
