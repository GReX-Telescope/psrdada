#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>

#include "mopsr_cuda.h"
#include "mopsr_delays_cuda_hires.h"

// maximum number of channels [320] * antenna [16] from 1 PFB
#define MOPSR_PFB_ANT_MAX     8
#define MOPSR_PFB_CHANANT_MAX 2560
#define MOPSR_MAX_ANT         352
#define WARP_SIZE             32
#define MEDIAN_FILTER         1
#define TWO_SIGMA             
//#define ONE_SIGMA
//#define SK_FREQ_AVG
//#define SHOW_MASK             // this puts the SK/TP masks into the output data!
//#define _GDEBUG               1

#ifdef USE_CONSTANT_MEMORY
__constant__ float d_ant_scales_delay [MOPSR_MAX_NANT_PER_AQ];
#endif

int hires_transpose_delay_alloc (transpose_delay_hires_t * ctx,
                                 uint64_t block_size, unsigned nchan,
                                 unsigned nant, unsigned ntap)
{
  ctx->nchan = nchan;
  ctx->nant = nant;
  ctx->ntap = ntap;
  ctx->half_ntap = ntap / 2;
  const unsigned nchanant = nchan * nant;
  const unsigned ndim = 2;

  ctx->curr = (transpose_delay_hires_buf_t *) malloc (sizeof(transpose_delay_hires_buf_t));
  ctx->next = (transpose_delay_hires_buf_t *) malloc (sizeof(transpose_delay_hires_buf_t));
  ctx->buffer_size = block_size + (ndim * nchanant * ctx->half_ntap * 2);

  size_t counter_size = ctx->nant * sizeof(unsigned);

  if (hires_transpose_delay_buf_alloc (ctx->curr, ctx->buffer_size, counter_size) < 0)
  {
    fprintf (stderr, "hires_transpose_delay_alloc: hires_transpose_delay_buf_alloc failed\n");
    return -1;
  }

  if (hires_transpose_delay_buf_alloc (ctx->next, ctx->buffer_size, counter_size) < 0)
  {
    fprintf (stderr, "hires_transpose_delay_alloc: hires_transpose_delay_buf_alloc failed\n");
    return -1;
  }

  ctx->first_kernel = 1;
  
  return 0;
}

int hires_transpose_delay_buf_alloc (transpose_delay_hires_buf_t * buf, size_t buffer_size, size_t counter_size)
{
  cudaError_t error; 

  // allocate the buffer for data
  error = cudaMalloc (&(buf->d_buffer), buffer_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "hires_transpose_delay_buf_alloc: cudaMalloc failed for %ld bytes\n", buffer_size);
    return -1;
  }

  buf->counter_size = counter_size;
  buf->counter_bytes = counter_size * 3;

#ifdef USE_CONSTANT_MEMORY
  error = cudaMallocHost (&(buf->h_out_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "hires_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMallocHost (&(buf->h_in_from), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "hires_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }

  error = cudaMallocHost (&(buf->h_in_to), buf->counter_size);
  if (error != cudaSuccess)
  {
    fprintf (stderr, "hires_transpose_delay_buf_alloc: cudaMallocHost failed for %ld bytes\n", buf->counter_size);
    return -1;
  }
#else

  // allocate host memory for counters
  error = cudaMallocHost (&(buf->h_base), buf->counter_bytes);

  // setup 3 pointers for host memory
  buf->h_out_from = (unsigned *) (buf->h_base + 0 * counter_size);
  buf->h_in_from  = (unsigned *) (buf->h_base + 1 * counter_size);
  buf->h_in_to    = (unsigned *) (buf->h_base + 2 * counter_size);

  error = cudaMalloc (&(buf->d_base), buf->counter_bytes);

  buf->d_out_from = (unsigned *) (buf->d_base + 0 * counter_size);
  buf->d_in_from  = (unsigned *) (buf->d_base + 1 * counter_size);
  buf->d_in_to    = (unsigned *) (buf->d_base + 2 * counter_size);

#endif

  buf->h_off = (unsigned *) malloc(buf->counter_size);
  buf->h_delays = (unsigned *) malloc(buf->counter_size);

  return 0;
}

void hires_transpose_delay_reset (transpose_delay_hires_t * ctx)
{
  ctx->first_kernel = 1;
}


int hires_transpose_delay_dealloc (transpose_delay_hires_t * ctx)
{
  hires_transpose_delay_buf_dealloc (ctx->curr);
  hires_transpose_delay_buf_dealloc (ctx->next);
  free (ctx->curr);
  free (ctx->next);

  return 0;
}

int hires_transpose_delay_buf_dealloc (transpose_delay_hires_buf_t * ctx)
{
#ifdef USE_CONSTANT_MEMORY
  if (ctx->h_out_from)
    cudaFreeHost (ctx->h_out_from);
  ctx->h_out_from = 0;

  if (ctx->h_in_from)
    cudaFreeHost (ctx->h_in_from);
  ctx->h_in_from = 0;

  if (ctx->h_in_to)
    cudaFreeHost (ctx->h_in_to);
  ctx->h_in_to = 0;
#else
  if (ctx->h_base)
    cudaFreeHost (ctx->h_base);
  ctx->h_base = 0;

  if (ctx->d_base)
    cudaFree (ctx->d_base);
  ctx->d_base = 0;
#endif
  
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

#ifdef USE_CONSTANT_MEMORY
__constant__ unsigned curr_out_from[MOPSR_PFB_ANT_MAX];
__constant__ unsigned curr_in_from[MOPSR_PFB_ANT_MAX];
__constant__ unsigned curr_in_to[MOPSR_PFB_ANT_MAX];
__constant__ unsigned next_out_from[MOPSR_PFB_ANT_MAX];
__constant__ unsigned next_in_from[MOPSR_PFB_ANT_MAX];
__constant__ unsigned next_in_to[MOPSR_PFB_ANT_MAX];
#endif

// major transpose kernel
// each block will process 32 time samples for 16 channels for all antenna
#ifdef USE_CONSTANT_MEMORY
__global__ void hires_transpose_delay_kernel (
     int16_t * in,
     int16_t * curr,
     int16_t * next,
     const unsigned nchan, const unsigned nant, 
     const unsigned nval, const unsigned nval_per_thread,
     const unsigned samp_stride, const unsigned chan_block_stride, 
     const unsigned out_chanant_stride)
#else
__global__ void hires_transpose_delay_kernel (
     int16_t * in,
     int16_t * curr,
     int16_t * next,
     unsigned * curr_counter,
     unsigned * next_counter,
     const unsigned nchan, const unsigned nant,
     const unsigned nval, const unsigned nval_per_thread,
     const unsigned samp_stride, const unsigned chan_block_stride,
     const unsigned out_chanant_stride)
#endif
{
  // for loaded data samples
  extern __shared__ int16_t sdata[];

  const int nsamp_per_block = 32;
  const int nchan_per_block = 16;
  const int nchanant_per_block = nant * nchan_per_block;

  const int warp_num = threadIdx.x / 32;
  const int warp_idx = threadIdx.x & 0x1F;  // % 32

  // each warp reads a time sample, with the warp threads each reading the antenna and channels required

  // offsets      time sample offset                                      +  channel block offset            + the chanant
  unsigned idx = (blockIdx.x * nsamp_per_block + warp_num) * samp_stride  + (blockIdx.y * chan_block_stride) + warp_idx;

  //              the right time sample in shm     the chanant  bank conflict trick
  unsigned sdx = (nchanant_per_block * warp_num) + warp_idx;// + (warp_num * 2);

  // read the TFS input to TFS shared memory
  for (unsigned i=0; i<nval_per_thread; i++)
  {
    if (idx < nval)
    {
      sdata[sdx] = in[idx];
      idx += 32;
      sdx += 32; 
    }
  }

  __syncthreads();

  // each warp will write out 32 time samples for a single antenna, for a number of channels
  const int ant = warp_num % nant;
  int ichan = nval_per_thread * (warp_num / nant);
  int ichanant = ichan * nant + ant; 

#ifdef USE_CONSTANT_MEMORY
  // TODO try removing these references
  const int curr_from = curr_in_from[ant];
  const int curr_to   = curr_in_to[ant];
  const int curr_out  = curr_out_from[ant] - curr_from;
  const int next_from = next_in_from[ant];
  const int next_to   = next_in_to[ant];
  const int next_out  = next_out_from[ant] - next_from;
#else
  const int curr_to   = curr_counter[2*nant + ant];
  const int curr_from = curr_counter[nant + ant];
  const int curr_out  = curr_counter[ant] - curr_from;

  const int next_to   = next_counter[2*nant + ant];
  const int next_from = next_counter[nant + ant];
  const int next_out  = next_counter[ant] - next_from;
#endif

  // offset for this thread in shared memory
  //         sample   * sample_stride_in_shm    + chanant offset + shm bank trick
  sdx = (warp_idx * nant * nchan_per_block) + ichanant;// + (warp_idx * 2);

  // output chanant for this warp
  const int ochanant = (blockIdx.y * nchan_per_block * nant) + ichanant;
  int osamp = (blockIdx.x * nsamp_per_block) + warp_idx;
  int64_t odx = ochanant * out_chanant_stride + osamp;

  // loop over channels
  for (unsigned i=0; i<nval_per_thread; i++)
  {
    if (curr_from <= osamp && osamp < curr_to)
      curr[odx + curr_out] = sdata[sdx];
    if (next_from <= osamp && osamp < next_to)
      next[odx + next_out] = sdata[sdx];

    sdx += nant;
    odx += out_chanant_stride * nant;
  }
}

void * hires_transpose_delay (cudaStream_t stream, transpose_delay_hires_t * ctx, void * d_in, uint64_t nbytes, mopsr_delay_hires_t ** delays)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;
  
  // process 32 samples and 16 channels in a block
  const unsigned nsamp_per_block = 32;
  const unsigned nchan_per_block = 16;
  const unsigned nchanblocks = ctx->nchan / nchan_per_block;

  const unsigned nval_per_block  = nsamp_per_block * nchan_per_block * ctx->nant;
  const uint64_t nsamp = nbytes / (ctx->nchan * ctx->nant * ndim);

  unsigned iant;
  int shift;

  const unsigned ichan = 0;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if (delays[iant][ichan].samples < ctx->half_ntap)
    {
      fprintf (stderr, "ERROR: [%d] delay in samples[%u] is less than ntap/2[%u]\n", iant, delays[iant][ichan].samples, ctx->half_ntap);
      return 0;
    }

    if (ctx->first_kernel)
    {
      ctx->curr->h_delays[iant]   = delays[iant][ichan].samples;
      ctx->next->h_delays[iant]   = delays[iant][ichan].samples;

      ctx->curr->h_out_from[iant] = 0;
      ctx->curr->h_in_from[iant]  = ctx->curr->h_delays[iant] - ctx->half_ntap;
      ctx->curr->h_in_to[iant]    = nsamp;
      ctx->curr->h_off[iant]      = ctx->curr->h_in_to[iant] - ctx->curr->h_in_from[iant];

      // should never be used on first iteration
      ctx->next->h_out_from[iant] = 0;
      ctx->next->h_in_from[iant]  = nsamp;
      ctx->next->h_in_to[iant]    = 2 * nsamp;
    }

    else
    {
      // curr always uses delays from previous iteration
      ctx->curr->h_out_from[iant] = ctx->curr->h_off[iant];
      ctx->curr->h_in_from[iant]  = 0;
      ctx->curr->h_in_to[iant]    = nsamp + (2 * ctx->half_ntap) - ctx->curr->h_off[iant];
      if (nsamp + (2 * ctx->half_ntap) < ctx->curr->h_off[iant])
        ctx->curr->h_in_to[iant] = 0;

      // next always uses new delays
      ctx->next->h_out_from[iant] = 0;
      ctx->next->h_in_from[iant]  = ctx->curr->h_in_to[iant] - (2 * ctx->half_ntap);
      ctx->next->h_in_to[iant]    = nsamp;

      // handle a change in sample level delay this should be right
      shift = delays[iant][ichan].samples - ctx->curr->h_delays[iant];

      ctx->next->h_in_from[iant] += shift;
      ctx->next->h_delays[iant]   = delays[iant][ichan].samples;
      ctx->next->h_off[iant]      = ctx->next->h_in_to[iant] - ctx->next->h_in_from[iant];
    }
  }

/*
 */

#ifdef USE_CONSTANT_MEMORY
  cudaMemcpyToSymbolAsync(curr_out_from, (void *) ctx->curr->h_out_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(curr_in_from, (void *) ctx->curr->h_in_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(curr_in_to, (void *) ctx->curr->h_in_to, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_out_from, (void *) ctx->next->h_out_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_in_from, (void *) ctx->next->h_in_from, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(next_in_to, (void *) ctx->next->h_in_to, ctx->curr->counter_size, 0, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
#else
  cudaMemcpyAsync (ctx->curr->d_base, ctx->curr->h_base, ctx->curr->counter_bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (ctx->next->d_base, ctx->next->h_base, ctx->next->counter_bytes, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
#endif

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
  const uint64_t nval = nbytes / ndim;

  // the total number of samples is 
  dim3 blocks = dim3 (nsamp / nsamp_per_block, nchanblocks);
  if (nsamp % nsamp_per_block)
    blocks.x++;

  const size_t sdata_bytes = (nsamp_per_block * nchan_per_block * ctx->nant * ndim) + 256;

  // nbytes of bytes different (for input) between each block of data
  const unsigned samp_stride = ctx->nchan * ctx->nant;
  const unsigned chan_block_stride = nchan_per_block * ctx->nant;
  const unsigned out_chanant_stride = nsamp + (2 * ctx->half_ntap);

#ifdef _GDEBUG
  fprintf (stderr, "transpose_delay: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "transpose_delay: nbytes=%lu, nsamp=%lu, nval=%lu\n", nbytes, nsamp, nval);
  fprintf (stderr, "transpose_delay: nthread=%d, blocks=(%d,%d,%d) sdata_bytes=%d\n", nthread, blocks.x, blocks.y, blocks.z, sdata_bytes);
  fprintf (stderr, "transpose_delay: out_chanant_stride=%u\n", out_chanant_stride);
#endif
#ifdef USE_CONSTANT_MEMORY
  hires_transpose_delay_kernel<<<blocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in,
        (int16_t *) ctx->curr->d_buffer, (int16_t *) ctx->next->d_buffer,
        ctx->nchan, ctx->nant, nval, nval_per_thread, samp_stride, chan_block_stride, out_chanant_stride);
#else
  hires_transpose_delay_kernel<<<blocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in,
        (int16_t *) ctx->curr->d_buffer, (int16_t *) ctx->next->d_buffer,
        (unsigned *) ctx->curr->d_base, (unsigned *) ctx->next->d_base,
        ctx->nchan, ctx->nant, nval, nval_per_thread, samp_stride, chan_block_stride, out_chanant_stride);
#endif

#if _GDEBUG
  check_error_stream("hires_transpose_delay_kernel", stream);
#endif

  if (ctx->first_kernel)
  {
    ctx->first_kernel = 0;
    return 0;
  }
  else
  {
    transpose_delay_hires_buf_t * save = ctx->curr;
    ctx->curr = ctx->next;
    ctx->next = save;
    return save->d_buffer;
  }
}


#ifdef USE_CONSTANT_MEMORY

// fringe co-efficients are fast in constant memory here
__constant__ float fringe_coeffs[MOPSR_PFB_CHANANT_MAX];

// apply a fractional delay correction to a channel / antenna, warps will always
__global__ void hires_fringe_rotate_kernel (int16_t * input, uint64_t ndat)
#else
__global__ void hires_fringe_rotate_kernel (int16_t * input, uint64_t ndat, 
                                            const float * __restrict__ d_fringes,
                                            const float * __restrict__ d_ant_scales_delay)
#endif
{
  const unsigned isamp = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned iant  = blockIdx.y;
  const unsigned nant  = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;
  const uint64_t idx = ichanant * ndat + isamp;

  if (isamp >= ndat)
    return;

  cuFloatComplex fringe_phasor;
#ifdef USE_CONSTANT_MEMORY
  // using constant memory should result in broadcast for this block/half warp
  sincosf (fringe_coeffs[ichanant], &(fringe_phasor.y), &(fringe_phasor.x));
#else
  sincosf (d_fringes[ichanant], &(fringe_phasor.y), &(fringe_phasor.x));
#endif

  int16_t val16 = input[idx];
  int8_t * val8ptr = (int8_t *) &val16;
  const float scale = d_ant_scales_delay[iant];

  float re = ((float) (val8ptr[0]) + 0.38) * scale;
  float im = ((float) (val8ptr[1]) + 0.38) * scale;
  cuComplex val = make_cuComplex (re, im);
  cuComplex rotated = cuCmulf(val, fringe_phasor);

  // output from signal processing, should have 0 mean data
  // i.e. we range from -128 to 127
  val8ptr[0] = (int8_t) rintf (cuCrealf(rotated));
  val8ptr[1] = (int8_t) rintf (cuCimagf(rotated));

  input[idx] = val16;
}

//
// Perform fractional delay correction, out-of-place
//
void hires_fringe_rotate (cudaStream_t stream, void * d_in,
#ifdef USE_CONSTANT_MEMORY
                          float * h_fringes, size_t fringes_size,
#else
                          void * d_fringes,
                          void * d_ant_scales,
#endif
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

#ifdef USE_CONSTANT_MEMORY
  cudaMemcpyToSymbolAsync(fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
#endif

#if _GDEBUG
  fprintf (stderr, "fringe_rotate: bytes=%lu ndat=%lu\n", nbytes, ndat);
  fprintf (stderr, "fringe_rotate: nthread=%d, blocks.x=%d, blocks.y=%d, blocks.z=%d\n", nthread, blocks.x, blocks.y, blocks.z);
#endif

#ifdef USE_CONSTANT_MEMORY
  hires_fringe_rotate_kernel<<<blocks, nthread, 0, stream>>>((int16_t *) d_in, ndat);
#else
  hires_fringe_rotate_kernel<<<blocks, nthread, 0, stream>>>((int16_t *) d_in, ndat, (float *) d_fringes, (float *) d_ant_scales);
#endif

#if _GDEBUG
  check_error_stream("hires_fringe_rotate_kernel", stream);
#endif
}


#ifdef USE_CONSTANT_MEMORY
void hires_delay_copy_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes)
{
  cudaMemcpyToSymbolAsync (d_ant_scales_delay, (void *) h_ant_scales, nbytes, 0, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}
#endif

// apply a fractional delay correction to a channel / antenna, warps will always 
__global__ void hires_delay_fractional_kernel (int16_t * input, int16_t * output, 
                                               const float * __restrict__ fir_coeffs,
#ifndef USE_CONSTANT_MEMORY
                                               const float * __restrict__ d_fringes,
                                               const float * __restrict__ d_ant_scales_delay,
#endif
                                               unsigned nthread_run, 
                                               uint64_t nsamp_in, 
                                               const unsigned chan_stride, 
                                               const unsigned ant_stride, 
                                               const unsigned ntap)
{
  // the input data for block are stored in blockDim.x values
  extern __shared__ cuComplex fk_shared1[];

  // the FIR filter stored in the final NTAP values
  float * filter = (float *) (fk_shared1 + blockDim.x);

  const unsigned half_ntap = (ntap / 2);
  //const unsigned in_offset = 2 * half_ntap;
  // iant blockIdx.y
  // ichan blockIDx.z

  const unsigned isamp = blockIdx.x * nthread_run + threadIdx.x;
  const unsigned ichanant = blockIdx.z * gridDim.y + blockIdx.y;
  const unsigned nsamp_out = nsamp_in - ( 2 * half_ntap);

  cuComplex fringe_phasor;
#ifdef USE_CONSTANT_MEMORY
  sincosf (fringe_coeffs[ichanant], &(fringe_phasor.y), &(fringe_phasor.x));
#else
  sincosf (d_fringes[ichanant], &(fringe_phasor.y), &(fringe_phasor.x));
#endif

  // read in the FIR cofficients
  if (threadIdx.x < ntap)
    filter[threadIdx.x] = fir_coeffs[(ichanant * ntap) + threadIdx.x];
  
  if (isamp >= nsamp_in)
  {
    return;
  }

  // each thread must also load its data from main memory here chan_stride + ant_stride
  const unsigned in_data_idx  = (ichanant * nsamp_in) + isamp;
  // const unsigned out_data_idx = ichanant * nsamp_out + isamp;

  int16_t val16 = input[in_data_idx];
  int8_t * val8ptr = (int8_t *) &val16;

  {
    const float scale = d_ant_scales_delay[blockIdx.y];
    cuComplex val = make_cuComplex (((float) (val8ptr[0])) + 0.38, ((float) (val8ptr[1])) + 0.38);
    val.x *= scale;
    val.y *= scale;
    fk_shared1[threadIdx.x] = cuCmulf(val, fringe_phasor);
  }

  __syncthreads();

  // there are 2 * half_ntap threads that dont calculate anything
  if ((threadIdx.x < nthread_run) && (isamp < nsamp_out))
  {
    float re = 0;
    float im = 0;
    for (unsigned i=0; i<ntap; i++)
    {
      re += cuCrealf(fk_shared1[threadIdx.x + i]) * filter[i];
      im += cuCimagf(fk_shared1[threadIdx.x + i]) * filter[i];
    }
    
    // input is -127.5 to -127.5, output is -128 to 127
    val8ptr[0] = (int8_t) rintf (re);
    val8ptr[1] = (int8_t) rintf (im);

    output[ichanant * nsamp_out + isamp] = val16;
  }
}

// calculate the filter coefficients for each channel and antenna
__global__ void hires_calculate_fir_coeffs (float * delays, float * fir_coeffs, unsigned ntap)
{
  const unsigned half_ntap = ntap / 2;

  const unsigned ichanant = blockIdx.x;
  const float itap = (float) threadIdx.x;
  const float filter_order = ntap - 1;

  float x = itap - delays[ichanant];
  // Hamming window filter http://users.spa.aalto.fi/vpv/publications/vesan_vaitos/ch3_pt1_fir.pdf
  float window = 0.54 - 0.46 * cos (2.0 * M_PI * x / filter_order);
  float sinc   = 1;
  if (x != half_ntap)
  {
    x -= half_ntap;
    x *= M_PI;
    sinc = sinf(x) / x;
  }

  fir_coeffs[(ichanant * ntap) + threadIdx.x] = sinc * window;
}

// apply a fractional delay correction to a channel / antenna, warps will always
__global__ void hires_delay_fractional_float_kernel (int16_t * input,
                    cuFloatComplex * output, float * fir_coeffs,
#ifndef USE_CONSTANT_MEMORY
                    float * fringe_coeffs,
#endif
                    unsigned nthread_run, uint64_t nsamp_in,
                    const unsigned chan_stride, const unsigned ant_stride,
                    const unsigned ntap)
{
  extern __shared__ float fk_shared_filter[];
  cuFloatComplex * in_shm = (cuFloatComplex *) (fk_shared_filter + ntap + 1);

  const unsigned half_ntap = ntap / 2;
  const unsigned in_offset = 2 * half_ntap;

  const unsigned isamp = blockIdx.x * nthread_run + threadIdx.x;
  const unsigned iant  = blockIdx.y;
  const unsigned nant  = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant  = ichan * nant + iant;

  const unsigned nsamp_out = nsamp_in - in_offset;

  // compute the complex term required for fringe stopping
  cuFloatComplex fringe_phasor;
  sincosf (fringe_coeffs[ichanant], &(fringe_phasor.y), &(fringe_phasor.x));

  // read in the FIR cofficients
  if (threadIdx.x < ntap)
  {
    fk_shared_filter[threadIdx.x] = fir_coeffs[(ichanant * ntap) + threadIdx.x];
  }

  // final block check for data input (not data output!)
  if (isamp >= nsamp_in)
  {
    return;
  }

  // each thread must also load its data from main memory here chan_stride + ant_stride
  const unsigned in_data_idx  = (ichanant * nsamp_in) + isamp;

  int16_t val16 = input[in_data_idx];
  int8_t * val8ptr = (int8_t *) &val16;

  cuFloatComplex val = make_cuComplex ((float) (val8ptr[0]) + 0.33, (float) (val8ptr[1]) + 0.33);
  in_shm[threadIdx.x] = cuCmulf (val, fringe_phasor);

  __syncthreads();

  const unsigned osamp = (blockIdx.x * nthread_run) + threadIdx.x;

  // there are 2 * half_ntap threads that dont calculate anything
  if (threadIdx.x < nthread_run && osamp < nsamp_out)
  {
    cuFloatComplex sum = make_cuComplex(0,0);
    for (unsigned i=0; i<ntap; i++)
    {
      val = in_shm[threadIdx.x + i];
      val.x *= fk_shared_filter[i];
      val.y *= fk_shared_filter[i];
      sum = cuCaddf(sum, val);
    }

    unsigned ou_data_idx = (ichanant * nsamp_out) + osamp;
    output[ou_data_idx] = sum;
  }
}

// 
// Perform fractional delay correction, out-of-place
//
void hires_delay_fractional (cudaStream_t stream, void * d_in, void * d_out,
                             float * d_delays, float * d_fir_coeffs, 
#ifdef USE_CONSTANT_MEMORY
                             float * h_fringes, size_t fringes_size, 
#else
                             void * d_fringes, void * d_ant_scales,
#endif
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

#ifdef USE_CONSTANT_MEMORY
  cudaMemcpyToSymbolAsync (fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
#endif

  // calculate the FIR co-efficients to be use in the fractional delay
  unsigned nthread = ntap;
  unsigned nblock = nchan * nant;
  hires_calculate_fir_coeffs<<<nblock,nthread,0,stream>>>((float *) d_delays, (float *) d_fir_coeffs, ntap);

#if _GDEBUG
  check_error_stream("hires_calculate_fir_coeffs", stream);
#endif


#if _GDEBUG
  fprintf (stderr, "delay_fractional: bytes=%lu ndat=%lu sdata_bytes=%ld\n", nbytes, ndat, sdata_bytes);
  fprintf (stderr, "delay_fractional: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "delay_fractional: nthread_load=%d nthread_run=%d ntap=%d\n", nthread_load, nthread_run, ntap);
#endif

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

#ifdef USE_CONSTANT_MEMORY
  hires_delay_fractional_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in, (int16_t *) d_out, 
        (float *) d_fir_coeffs, nthread_run, ndat, chan_stride, ant_stride, ntap);
#else
  hires_delay_fractional_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in, (int16_t *) d_out, 
        (float *) d_fir_coeffs, (float *) d_fringes, (float *) d_ant_scales, nthread_run, ndat, chan_stride, ant_stride, ntap);
#endif

#if _GDEBUG
  check_error_stream("hires_delay_fractional_kernel", stream);
#endif
}


#ifdef HAVE_CUDA_SHUFFLE
__inline__ __device__
float warpReduceSumF(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSumF(float  val) 
{
  static __shared__ float shared[32]; // Shared mem for 32 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumF(val);      // Each warp performs partial reduction

  if (lane==0) shared[wid] = val; // Write reduced value to shared memory

  __syncthreads();                // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSumF(val); //Final reduce within first warp

  return val;
}

__inline__ __device__
float blockReduceSumFS(float * vals)
{
  float val = vals[threadIdx.x];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumF(val);      // Each warp performs partial reduction

  if (lane==0) vals[wid] = val;   // Write reduced value to shared memory

  __syncthreads();                // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? vals[lane] : 0;

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

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
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


// Compute the mean of the re and imginary compoents for 
__global__ void hires_measure_means_kernel (cuFloatComplex * in, cuFloatComplex * means, const unsigned nval_per_thread, const uint64_t ndat)
{
  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;
  const uint64_t in_offset = ichanant * ndat;

  cuFloatComplex * indat = in + in_offset;

  unsigned idx = threadIdx.x * nval_per_thread;

  cuFloatComplex val;
  float sum_re = 0;
  float sum_im = 0;
  int count = 0;

  for (unsigned ival=0; ival<nval_per_thread; ival++)
  {
    if (idx < ndat)
    {
      val = indat[idx];
      sum_re += val.x;
      sum_im += val.y;
      count++;
    }
    idx += blockDim.x;
  }

#ifdef HAVE_CUDA_SHUFFLE
  // compute via block reduce sum
  sum_re = blockReduceSumF(sum_re);
  sum_im = blockReduceSumF(sum_im);
  count = blockReduceSumI(count);
#endif

  if (threadIdx.x == 0)
  {
    means[ichanant].x = sum_re / count;
    means[ichanant].y = sum_im / count;
  }
}

//
// Compute the S1 and S2 sums for blocks of input data, writing the S1 and S2 sums out to Gmem
//
__global__ void hires_skcompute_kernel (cuFloatComplex * in, float * s1s, float * s2s, const unsigned nval_per_thread, const uint64_t ndat)
{
  extern __shared__ float skc_shm[];

  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;
  const uint64_t in_offset = ichanant * ndat;

  // offset into the block for the current channel and antenna
  cuFloatComplex * indat = in + in_offset;

  unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) * nval_per_thread;

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

#ifdef HAVE_CUDA_SHUFFLE
  const unsigned warp_idx = threadIdx.x % 32;
  const unsigned warp_num = threadIdx.x / 32;

  s1_sum += __shfl_down (s1_sum, 16);
  s1_sum += __shfl_down (s1_sum, 8);
  s1_sum += __shfl_down (s1_sum, 4);
  s1_sum += __shfl_down (s1_sum, 2);
  s1_sum += __shfl_down (s1_sum, 1);

  s2_sum += __shfl_down (s2_sum, 16);
  s2_sum += __shfl_down (s2_sum, 8);
  s2_sum += __shfl_down (s2_sum, 4);
  s2_sum += __shfl_down (s2_sum, 2);
  s2_sum += __shfl_down (s2_sum, 1);

  if (warp_idx == 0)
  {
    skc_shm [warp_num]    = s1_sum;
    skc_shm [32+warp_num] = s2_sum;
  }

  __syncthreads();

  if (warp_num == 0)
  {
    s1_sum = skc_shm [warp_idx];
    s2_sum = skc_shm [32 + warp_idx];

    s1_sum += __shfl_down (s1_sum, 16);
    s1_sum += __shfl_down (s1_sum, 8);
    s1_sum += __shfl_down (s1_sum, 4);
    s1_sum += __shfl_down (s1_sum, 2);
    s1_sum += __shfl_down (s1_sum, 1);

    s2_sum += __shfl_down (s2_sum, 16);
    s2_sum += __shfl_down (s2_sum, 8);
    s2_sum += __shfl_down (s2_sum, 4);
    s2_sum += __shfl_down (s2_sum, 2);
    s2_sum += __shfl_down (s2_sum, 1);
  }
#endif

  if (threadIdx.x == 0)
  {
    // FST ordered
    const unsigned out_idx = (ichanant * gridDim.x) +  blockIdx.x;
    //if (iant == 0 && ichan == 168)
    //  printf ("s1s[%u]=%f\n", out_idx, s1_sum);
    s1s[out_idx] = s1_sum;
    s2s[out_idx] = s2_sum;
  }
}


void hires_test_skcompute (cudaStream_t stream, void * d_in, void * d_s1s_out, void * d_s2s_out, unsigned nchan, unsigned nant, unsigned nbytes)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim * sizeof(float));
  const unsigned nthreads = 1024;
  const unsigned nval_per_thread = 1;
  size_t shm_bytes = 64 * sizeof(float);

  dim3 blocks (ndat / nthreads, nant, nchan);
  if (ndat % nthreads)
    blocks.x++;

//#ifdef _GDEBUG
  fprintf (stderr, "hires_skcompute_kernel: bytes=%lu ndat=%lu shm_bytes=%ld\n", nbytes, ndat, shm_bytes);
  fprintf (stderr, "hires_skcompute_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d, nthreads=%u\n", blocks.x, blocks.y, blocks.z, nthreads);
  fprintf (stderr, "hires_skcompute_kernel: d_in=%p d_s1s_out=%p, d_s2s_out=%p nval_per_thread=%u, ndat_sk=%lu\n", d_in, d_s1s_out, d_s2s_out, nval_per_thread, ndat);
//#endif

  hires_skcompute_kernel<<<blocks, nthreads, shm_bytes, stream>>>( (cuFloatComplex *) d_in, (float *) d_s1s_out, (float *) d_s2s_out, nval_per_thread, ndat);

  check_error_stream("hires_skcompute_kernel", stream);
}

__device__ inline void Comparator(
    float &valA,
    float &valB,
    uint dir
)
{
  float k;
  if ((valA > valB) == dir)
  {
    k = valA;
    valA = valB;
    valB = k;
  }
}

__device__ inline void shm_merge_sort (unsigned length, float * keys)
{
  const unsigned maxthread = length / 2;
  for (uint size = 2; size <= length; size <<= 1)
  {
    uint stride = size / 2;
    uint offset = threadIdx.x & (stride - 1);

    {
      __syncthreads();
      if (threadIdx.x < maxthread)
      {
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator (keys[pos + 0], keys[pos + stride], 1);
      }
      stride >>= 1;
    }

    for (; stride > 0; stride >>= 1)
    {
      __syncthreads();
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

      if (threadIdx.x < maxthread)
      {
        if (offset >= stride)
        {
          Comparator( keys[pos - stride], keys[pos + 0], 1);
        }
      }
    }
  }
  __syncthreads();
}

// simplistic block wide shared memory sum, 1 val per thread
__device__ inline float shm_sum_thread (unsigned length, float * keys)
{
  for (unsigned size=length/2; size>0; size >>= 1)
  {
    if (threadIdx.x < size)
      keys[threadIdx.x] += keys[threadIdx.x + size];
    __syncthreads();
  }

  return keys[0];
} 

__global__ void shm_merge_sort_kernel2 (float *d_Dst,
                                        float *d_Src,
                                        unsigned arrayLength,
                                        unsigned dir)
{
  //Shared memory storage for one or more small vectors
  __shared__ float keys[1024];

  keys[threadIdx.x] = d_Src[threadIdx.x];

  __syncthreads();
  
  shm_merge_sort (arrayLength, keys);

  __syncthreads();

  d_Dst[threadIdx.x] = keys[threadIdx.x];
}


void test_merge_sort2 (cudaStream_t stream, float * d_key_out, float * d_key_in, unsigned length, unsigned direction) { 
  unsigned nblocks = 1;
  unsigned nthreads = length;
  
  shm_merge_sort_kernel2<<<nblocks, nthreads>>> (d_key_out, d_key_in, length, direction);
  check_error_stream("shm_merge_sort_kernel2", stream);
  return;
}


__global__ void hires_compute_sigmas_kernel (float * in, cuFloatComplex * thresholds,
         float * voltage_sigmas, unsigned nsums)
{
  extern __shared__ float csk_keys[];
  // iant = blockIdx.y;
  // nant = gridDim.y;
  // ichan = blockIdx.z;
  // nchan = gridDim.z;

  const unsigned ichanant = (blockIdx.z * gridDim.y) + blockIdx.y;

  float s1 = in[(ichanant * nsums) + threadIdx.x];

  // read the 16 input values into shared memory
  csk_keys[threadIdx.x] = s1;

  __syncthreads();

  // sort using shared memory
  shm_merge_sort (nsums, csk_keys);

  __syncthreads();

  float median = csk_keys[nsums / 2];

  __syncthreads();

  // now subtract median from s1 value in key and take abs value
  csk_keys[threadIdx.x] = fabsf(csk_keys[threadIdx.x] - median);

  __syncthreads();

  // now sort again
  shm_merge_sort (nsums, csk_keys);

  __syncthreads();

  // convert median absolute deviation to standard deviation
  float sigma = csk_keys[nsums / 2] * 1.4826;

  // set the thresholds
  if (threadIdx.x == 0)
  {
    thresholds[ichanant].x = median;
    thresholds[ichanant].y = sigma;
  }

  csk_keys[threadIdx.x] = s1;
  __syncthreads();

  // simple sum whereby nsums == nthreads
  s1 = shm_sum_thread (nsums, csk_keys);

  if (threadIdx.x == 0)
  {
    voltage_sigmas[ichanant] = sqrtf (s1 / (nsums * 1024 * 2));
  }
}


/*
 */
__global__ void hires_compute_power_limits_kernel (float * in, cuFloatComplex * thresholds, 
         float * voltage_sigmas, int8_t * mask, curandStatePhilox4_32_10_t * rstates, unsigned nsums, unsigned valid_memory, 
         unsigned nsigma, unsigned iblock)
{
  // iant = blockIdx.y;
  // nant = gridDim.y;
  // ichan = blockIdx.z;
  // nchan = gridDim.z;

  //const unsigned n_elements = nsums * valid_memory;

  // 1024 threads, 16 samples/block, 64 memory blocks, each thread does 1 samples
  const unsigned mem_block = threadIdx.x / 16;
  const unsigned mem_element = threadIdx.x % 16;

  const unsigned ichanant = (blockIdx.z * gridDim.y) + blockIdx.y;
  const unsigned nchanant = gridDim.z * gridDim.y;

  // get the generator for this channel and antenna [gridDim.x == 1]
  const unsigned id = ichanant * blockDim.x + threadIdx.x;

  // a maximum of 32 * 32 keys [1024] will be handled by 1024 threads.
  __shared__ float keys[1024];

  // existing median and sigma for the S1s
  float median = thresholds[ichanant].x;
  float sigma  = thresholds[ichanant].y;

  float s1 = 0;
  float s1_count = 0;

  // S1 values stored as 64 sets of FST in blocks that are each 16 samples
  //    ichanant offset + offset into the memory [0-16]

  if (mem_block < valid_memory)
  {
    s1 = in[(mem_block * nchanant * 16) + (ichanant * 16) + mem_element];
    s1_count = 1;

    // if skdetect has determined this sample is bad, generate something similar
    if ((mem_block == iblock) && (mask[ichanant * 16 + mem_element] > 0))
    {
      s1 = median + (curand_normal (&(rstates[id])) * sigma);
      in[(mem_block * nchanant * 16) + (ichanant * 16) + mem_element] = s1;
    }
  }

  // now find the median and median absolute deviation (stddev)

  keys[threadIdx.x] = s1;

  __syncthreads();

  // sort the nelements values using shared memory
  shm_merge_sort (1024, keys);

  __syncthreads();

  unsigned centre = 1024 - ((valid_memory * 16) / 2);

  median = keys[centre];

  __syncthreads();

  // now subtract median from s1 value in key and take abs value
  if (s1 > 0)
    keys[threadIdx.x] = fabsf(s1 - median);
  else
    keys[threadIdx.x] = 0;

  __syncthreads();

  // now sort again
  shm_merge_sort (1024, keys);

  __syncthreads();

  // convert median absolute deviation to standard deviation
  sigma = keys[centre] * 1.4826;

  //if (blockIdx.z == 210 && blockIdx.y == 0 && iblock == 0 && threadIdx.x < 16)
  //  printf ("[%d] s1=%f centre=%u median=%f sigma=%f\n", threadIdx.x, s1, centre, median, sigma);

  // now sum S1 across threads
  s1 += __shfl_down (s1, 16);
  s1 += __shfl_down (s1, 8);
  s1 += __shfl_down (s1, 4);
  s1 += __shfl_down (s1, 2);
  s1 += __shfl_down (s1, 1);

  s1_count += __shfl_down (s1_count, 16);
  s1_count += __shfl_down (s1_count, 8);
  s1_count += __shfl_down (s1_count, 4);
  s1_count += __shfl_down (s1_count, 2);
  s1_count += __shfl_down (s1_count, 1);

  unsigned warp_idx = threadIdx.x % 32;
  unsigned warp_num = threadIdx.x / 32;

  if (warp_idx == 0)
  {
    keys[warp_num] = s1;
    keys[32+warp_num] = s1_count;
  }

  __syncthreads();

  if (warp_num == 0)
  {
    s1 = keys[warp_idx];
    s1_count = keys[32+warp_idx];

    s1 += __shfl_down (s1, 16);
    s1 += __shfl_down (s1, 8);
    s1 += __shfl_down (s1, 4);
    s1 += __shfl_down (s1, 2);
    s1 += __shfl_down (s1, 1);

    s1_count += __shfl_down (s1_count, 16);
    s1_count += __shfl_down (s1_count, 8);
    s1_count += __shfl_down (s1_count, 4);
    s1_count += __shfl_down (s1_count, 2);
    s1_count += __shfl_down (s1_count, 1);

    // this sigma is the stddev of the voltages (hence 1024 * 2)
    if (warp_idx == 0)
    {
      //voltage_sigmas[ichanant] = sqrtf(s1 / (s1_count * 2048));
      voltage_sigmas[ichanant] = sqrtf(median / 2048);

      // now we have the median and sigma for the memory blocks of S1, compute the
      // total power thresholds
      thresholds[ichanant].x = median;
      thresholds[ichanant].y = sigma;
    }
  }
}

void hires_test_compute_power_limits (cudaStream_t stream, void * d_s1s, void * d_sigmas, 
                          void * d_thresh, void * d_mask,  unsigned nsums, unsigned nant, unsigned nchan, uint64_t ndat,
                          uint64_t s1_count, unsigned s1_memory, void * d_rstates)
{
  dim3 blocks_skm (1, nant, nchan);

  unsigned nthreads = 1024;
  const unsigned nsigma = 4;

  unsigned valid_memory = s1_memory;
  if (s1_count < s1_memory)
    valid_memory = (unsigned) s1_count;

#ifdef _DEBUG
  fprintf (stderr, "test_compute_power_limits: d_s1s=%p d_thresh=%p\n", d_s1s, d_thresh);
  fprintf (stderr, "test_compute_power_limits: nant=%u nchan=%u ndat=%lu\n", nant, nchan, ndat);
  fprintf (stderr, "test_compute_power_limits: nsums=%u nmemory=%u nsigma=%u\n", nsums, valid_memory, nsigma);
#endif

  hires_compute_power_limits_kernel<<<blocks_skm,nthreads,0,stream>>>((float *) d_s1s,
                  (cuFloatComplex *) d_thresh, (float *) d_sigmas, (int8_t *) d_mask, (curandStatePhilox4_32_10_t *) d_rstates, nsums, valid_memory, nsigma, 0);

  check_error_stream("hires_compute_power_limits_kernel", stream);
}

//
// take the S1 and S2 values in sums.x and sums.y that were computed 
//  from M samples, and integrate of nsums blocks to
// compute a sk mask and zap
//
__global__ void hires_skdetect_kernel (float * s1s, float * s2s, cuFloatComplex * power_thresholds,
                                       int8_t * mask, float * sigmas, 
                                       unsigned nchan_sum, unsigned sk_nsigma,
                                       unsigned nsums, unsigned M, unsigned nval_per_thread)
                                       
{
  // zap mask for each set of M samples
  extern __shared__ int8_t smask_det[];

  // maximum to be 16384 samples (20.97152 ms)
  // unsigned sk_idx_max = 16;

  // given the buffer sizes of 16384 samples, we shall not exceed 2^14
  // 2^11 is an important one: 10.24us * 2048 samples == 20.97152 ms
  // maximum to be 2048 samples (20.97152 ms)
  unsigned sk_idx_max = 14;

  // 3 sigma
  const float sk_low[15]  = { 0, 0, 0, 0, 0,
                              0.387702, 0.492078, 0.601904, 0.698159, 0.775046,
                              0.834186, 0.878879, 0.912209, 0.936770, 0.954684};
  const float sk_high[15] = { 0, 0, 0, 0, 0,
                              2.731480, 2.166000, 1.762970, 1.495970, 1.325420,
                              1.216950, 1.146930, 1.100750, 1.069730, 1.048570};

  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned nchan = gridDim.z;
  const unsigned ichanant = (ichan * nant) + iant;

  // ASSUME! nsums == nthreads
  
  // initialize zap mask to 0 in shared memory [FT order]
  unsigned idx = threadIdx.x;
  for (unsigned i=0; i<nchan_sum; i++)
  {
    smask_det[idx] = 0;
    idx += nsums;
  }

  __syncthreads();

  // log2 of 1024
  const unsigned log2_M = (unsigned) log2f (M);

  // S1 and S2 sums are stored as FST
  s1s += (ichanant * nsums);
  s2s += (ichanant * nsums);

  idx = threadIdx.x;

  if (!((ichan == 54 || ichan == 105 || ichan == 155 || ichan == 204)))
    nchan_sum = 1;
 
  // for each different boxcar width
  for (unsigned sk_idx = log2_M; sk_idx < sk_idx_max; sk_idx ++)
  {
    // the number of S1 (1024 powers) to add to this boxcar
    const unsigned to_add = (unsigned) exp2f (sk_idx - log2_M);

    // prevent running over the end of the array
    if (idx + to_add <= nsums)
    {
      const float m = (float) (M * to_add);
      const float m_fac = (m + 1) / (m - 1);

      // the number of channels that are considered bad
      // 2 sigma == 9 channels
      // 1 sigma == 25 channels
      unsigned nchan_bad_count = 0;
      const unsigned nchan_bad_limit = 12;
      float sk_avg = 0;
      unsigned cdx = idx;

      // loop over the channels in our sum
      for (unsigned i=ichan; i<(ichan+nchan_sum); i++)
      {
        const unsigned ica = i * nant + iant;
        const float median = power_thresholds[ica].x;
        const float sigma  = power_thresholds[ica].y; // / sqrtf(to_add);
        const float chan_sum_limit = 2 * sigma;
        const float power_limit = 3 * sigma;

        // compute the SK estimate for this boxcar width and channel
        float s1 = 1e-10; 
        float s2 = 1e-10;
        for (unsigned ichunk=0; ichunk < to_add; ichunk++)
        {
          s1 += s1s[cdx + ichunk];
          s2 += s2s[cdx + ichunk];
        }
        float sk_estimate = m_fac * (m * (s2 / (s1 * s1)) - 1);

        sk_avg += sk_estimate;
        float s1_avg = s1 / to_add;

        // test the SK estimate for only the current channel
        if (i == ichan)
        {
          if ((sk_estimate < sk_low[sk_idx]) || (sk_estimate > sk_high[sk_idx])) 
          {
            for (unsigned ichunk=0; ichunk < to_add; ichunk++)
            {
              smask_det[idx+ichunk] = (int8_t) 1;
            } 
          }
        
          // test if the average S1 power exceeds the 3sigma limits from the long running median/sigma
          if ((s1_avg > (median + power_limit)) || (s1_avg < (median - power_limit)))
          {
            for (unsigned ichunk=0; ichunk < to_add; ichunk++)
            {
              smask_det[idx+ichunk] = 3;
            }
          }
        }

        // phone call detector
        // test if the average S1 power exceeds the special limit for channel summation
        if (s1_avg > (median + chan_sum_limit))
          nchan_bad_count ++;

        // increment by 1 channel
        cdx += (nant * nsums);
      }

      // if we this is a phone call band, check the limits on the SK Average and nchan_bad
      if (nchan_sum == 50)
      {
#ifdef SKAVG_METHOD
        float mu2 = (4 * m * m) / ((m-1) * (m + 2) * (m + 3));
        float one_sigma_idat = sqrtf(mu2 / nchan_sum);
        float upper = 1 + (sk_nsigma * one_sigma_idat);
        float lower = 1 - (sk_nsigma * one_sigma_idat);
        sk_avg /= nchan_sum;

        if ((sk_avg < lower) || (sk_avg > upper) || (nchan_bad_count > nchan_bad_limit))
#else
        if (nchan_bad_count > nchan_bad_limit)
#endif
        {
          cdx = idx;
          for (unsigned i=0; i<nchan_sum; i++)
          {
            for (unsigned ichunk=0; ichunk < to_add; ichunk++)
            {
              smask_det[cdx+ichunk] = 2;
            }
            cdx += nsums;
          }
        }
      }
    }
  }

  // now write out the SK mask to gmem
  for (unsigned i=0; i < nchan_sum; i++)
  {
    if ((ichan + i) < nchan)
    {
      unsigned odx = (((ichan + i) * nant) + iant) * nsums + threadIdx.x;
      unsigned sdx = i * nsums + threadIdx.x;
      if ((sdx < nchan_sum * nsums) && (smask_det[sdx] > 0))
      {
        mask[odx] = smask_det[sdx];
      }
    }
  }
}

void hires_test_skdetect (cudaStream_t stream, void * d_s1s, void * d_s2s, void * d_thresh, 
                          void * d_mask, void * d_sigmas, unsigned nsums, unsigned nant, 
                          unsigned nchan, uint64_t ndat)
{
  unsigned M = 1024;
  //////////////////////////////////////////////////////////
  // mask the input data
  dim3 blocks (1, nant, nchan);
  unsigned nthreads = 1024;
  unsigned nval_per_thread = 1;
  if (nsums > nthreads)
  {
    nval_per_thread = nsums / nthreads;
    if (nsums % nthreads)
      nval_per_thread++;
  }
  else
    nthreads = nsums;

  unsigned nchan_sum = 50;
  unsigned sk_nsigma = 4;

  size_t shm_bytes = (nchan_sum + 1) * nsums * sizeof(uint8_t);
  size_t mask_size = nsums * nchan * nant * sizeof(uint8_t);
  cudaMemsetAsync (d_mask, 0, mask_size, stream);
  cudaStreamSynchronize(stream);

  fprintf (stderr, "hires_skdetect_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "hires_skdetect_kernel: nthreads=%u shm_bytes=%ld\n", nthreads, shm_bytes);
  fprintf (stderr, "hires_skdetect_kernel: d_s1s=%p, d_s2s=%p, d_masks=%p, nsums=%u, M=%u, nval_per_thread=%u\n", d_s1s, d_s2s, d_mask, nsums, M, nval_per_thread);

  hires_skdetect_kernel<<<blocks, nthreads, shm_bytes, stream>>>((float *) d_s1s, (float *) d_s2s, (cuFloatComplex *) d_thresh, (int8_t *) d_mask, (float *) d_sigmas, nchan_sum, sk_nsigma, nsums, M, nval_per_thread);

  check_error_stream("hires_skdetect_kernel", stream);
}

//
// take the S1 and S2 values in sums.x and sums.y that were computed 
//  from M samples, and integrate of nsums blocks to
// compute a sk mask and zap
//
__global__ void hires_skmask_kernel (float * in, int8_t * out, int8_t * mask, 
                                     curandStatePhilox4_32_10_t * rstates, float * sigmas,
#ifndef USE_CONSTANT_MEMORY
                                     const float * __restrict__ d_ant_scales_delay,
#endif
                                     unsigned nsums, unsigned M, unsigned nval_per_thread, 
                                     unsigned nsamp_per_thread, uint64_t ndat,
                                     char replace_noise)
{
  const unsigned iant = blockIdx.y;
  const unsigned nant = gridDim.y;
  const unsigned ichan = blockIdx.z;
  const unsigned ichanant = (ichan * nant) + iant;

  const unsigned id = ichanant * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t localState = rstates[id];

  float sigma = sigmas[ichanant];
  int8_t * chanant_mask = mask + (ichanant * nsums);

  // Jenet & Anderson 1998, 6-bit (2-bits for RFI) spacing
  const float spacing = 0.09925;    // 6-bit
  //const float spacing = 0.02957;      // 8-bit

  // dont do antenna scaling here anymore for the moment, unless it is zero
  const float ant_scale = d_ant_scales_delay[iant];
  float data_factor = ant_scale / (sigma * spacing);
  float rand_factor = ant_scale / spacing;
  if (!replace_noise)
    rand_factor = 0;

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

#ifdef SHOW_MASK
    for (unsigned isamp=0; isamp<nsamp_per_thread; isamp++)
    {
      if (idx < nval_per_sum)
      {
        outdat[idx] = (int8_t) chanant_mask[isum];
      }
      idx += blockDim.x;
    }
#else
    if (chanant_mask[isum] > 0)
    {
      // it is more efficient to generate 4 floats at a time
      for (unsigned isamp=0; isamp<nsamp_per_thread; isamp+=4)
      {
        const float4 inval = curand_normal4 (&localState);
        if (idx < nval_per_sum)
          outdat[idx] = (int8_t) rintf(inval.x * rand_factor);
        idx += blockDim.x;

        if (idx < nval_per_sum)
          outdat[idx] = (int8_t) rintf(inval.y * rand_factor);
        idx += blockDim.x;

        if (idx < nval_per_sum)
          outdat[idx] = (int8_t) rintf(inval.z * rand_factor);
        idx += blockDim.x;

        if (idx < nval_per_sum)
          outdat[idx] = (int8_t) rintf(inval.w * rand_factor);
        idx += blockDim.x;
      }
    }
    else
    {
      for (unsigned isamp=0; isamp<nsamp_per_thread; isamp++)
      {
        if (idx < nval_per_sum)
        {
          outdat[idx] = (int8_t) rintf (indat[idx] * data_factor);
        }
        idx += blockDim.x;
      }
    }
#endif

    outdat += ndim * M;
    indat += ndim * M;
  }

  rstates[id] = localState;
}

void hires_test_skmask (cudaStream_t stream, void * d_in, void * d_out, void * d_mask, void * d_rstates, void * d_sigmas, 
#ifndef USE_CONSTANT_MEMORY
                        void * d_ant_scales_delay,
#endif

unsigned nsums, unsigned nchan, unsigned nant, uint64_t ndat, char replace_noise)
{
  unsigned M = 1024;
  unsigned ndim = 2;
  //////////////////////////////////////////////////////////
  // mask the input data
  dim3 blocks (1, nant, nchan);
  unsigned nthreads = 1024;
  unsigned nval_per_thread = 1;
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

  size_t shm_bytes = 0;

  fprintf (stderr, "hires_skmask_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "hires_skmask_kernel: nthreads=%u shm_bytes=%ld\n", nthreads, shm_bytes);
  fprintf (stderr, "hires_skmask_kernel: d_in=%p d_out=%p, d_mask=%p, nsums=%u M=%u, nval_per_thread=%u, nsamp_per_thread=%u ndat=%lu\n", d_in, d_out, d_in, nsums, M, nval_per_thread, nsamp_per_thread, ndat);

  hires_skmask_kernel<<<blocks, nthreads, shm_bytes, stream>>>((float *) d_in, (int8_t *) d_out,
                (int8_t *) d_mask, (curandStatePhilox4_32_10_t *) d_rstates, (float *) d_sigmas,
#ifndef USE_CONSTANT_MEMORY
                (float *) d_ant_scales_delay,
#endif
                nsums, M, nval_per_thread, nsamp_per_thread, ndat, replace_noise);

  check_error_stream("hires_skmask_kernel", stream);
}

__global__ void hires_srand_setup_kernel_sparse (unsigned long long seed, unsigned pfb_idx, unsigned nrngs, unsigned nval_per_thread, curandStatePhilox4_32_10_t * rstates)
{
  unsigned id = threadIdx.x;
  unsigned long long sequence = (blockDim.x * pfb_idx) + threadIdx.x;
  unsigned long long local_seed = seed;
  unsigned long long offset = 0;
  unsigned long long skip = nrngs;

  curandStatePhilox4_32_10_t local_state;
  curand_init (local_seed, sequence, offset, &local_state);

  rstates[id] = local_state;
  id += blockDim.x;

  for (unsigned i=1; i<nval_per_thread; i++)
  {
    skipahead_sequence (skip, &local_state);
    rstates[id] = local_state;
    id += blockDim.x;
  }
}

void hires_init_rng_sparse (cudaStream_t stream, unsigned long long seed, unsigned nrngs, unsigned pfb_idx, unsigned npfb, void * states)
{
  unsigned nthreads = 1024;
  unsigned nval_per_thread = nrngs / nthreads;

#if _GDEBUG
  fprintf (stderr, "rand_setup: nrngs=%u nval_per_thread=%u nthreads=%u pfb_idx=%u\n", nrngs, nval_per_thread, nthreads, pfb_idx);
#endif

  hires_srand_setup_kernel_sparse<<<1, nthreads, 0, stream>>>(seed, pfb_idx, nrngs, nval_per_thread, (curandStatePhilox4_32_10_t *) states);

#if _GDEBUG
  check_error_stream("hires_srand_setup_kernel", stream);
#endif
}

__global__ void hires_srand_setup_kernel (unsigned long long seed, unsigned pfb_idx, unsigned npfb, curandStatePhilox4_32_10_t *states)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  // local seed will be different for each of NCHAN * NANT * 1024 generators
  // sequence 0, since each sequence increment involves 2^67 steps through the RNG sequence!
  // offset is the step through the random sequence
  unsigned long long sequence   = 0;
  unsigned long long offset     = id;
  unsigned long long local_seed = (seed << 20) + id;

  // more efficient, since moving along the sequence of a seed is expensive
  curand_init (local_seed, sequence, offset, &states[id]);
  //curand_init( (seed << 20) + id, 0, 0, &states[id]);
}

void hires_init_rng (cudaStream_t stream, unsigned long long seed, unsigned nrngs, unsigned pfb_idx, unsigned npfb, void * states)
{
  unsigned nthreads = 1024;
  unsigned nblocks = nrngs / nthreads;

#if _GDEBUG
  fprintf (stderr, "rand_setup: nblocks=%u nthreads=%u\n", nblocks, nthreads);
#endif

  hires_srand_setup_kernel<<<nblocks, nthreads, 0, stream>>>(seed, pfb_idx, npfb, (curandStatePhilox4_32_10_t *) states);

#if _GDEBUG
  check_error_stream("hires_srand_setup_kernel", stream);
#endif
}


// out-of-place
//
void hires_delay_fractional_sk_scale (cudaStream_t stream, 
     void * d_in, void * d_out, void * d_fbuf, void * d_rstates,
     void * d_sigmas, void * d_mask, float * d_delays, void * d_fir_coeffs,
#ifndef USE_CONSTANT_MEMORY
     void * d_fringes, void * d_ant_scales,
#endif
     void * d_s1s, void * d_s2s, void * d_thresh,
#ifdef USE_CONSTANT_MEMORY
     float * h_fringes, size_t fringes_size, 
#endif
     uint64_t nbytes, unsigned nchan, unsigned nant, unsigned ntap, 
     unsigned s1_memory, uint64_t s1_count, char replace_noise)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim);
  const unsigned half_ntap = ntap / 2;

#ifdef USE_CONSTANT_MEMORY
  // copy the fringe coeffs and delays to GPU memory
  cudaMemcpyToSymbolAsync (fringe_coeffs, (void *) h_fringes, fringes_size, 0, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream); 
#endif

  // calculate the FIT co-efficients to be use in the fractional delay
  unsigned nthread = ntap;
  unsigned nblock = nchan * nant;
  hires_calculate_fir_coeffs<<<nblock,nthread,0,stream>>>((float *) d_delays, (float *) d_fir_coeffs, ntap);
#if _GDEBUG
  check_error_stream("hires_calculate_fir_coeffs", stream);
#endif

  // number of threads that actually load data
  unsigned nthread_load = 1024;
  if (ndat < nthread_load)
    nthread_load = ndat;
  unsigned nthread_run  = nthread_load - (2 * half_ntap);

  // need shared memory to load the ntap coefficients + nthread_load data points
  //const size_t   sdata_bytes = (nthread_load * ndim + ntap) * sizeof(float);
  const size_t   sdata_bytes = (nthread_load * ndim + ntap + 1) * sizeof(float);

  dim3 blocks (ndat / nthread_run, nant, nchan);
  if (ndat % nthread_load)
    blocks.x++;

#if _GDEBUG
  fprintf (stderr, "hires_delay_fractional_float_kernel: bytes=%lu ndat=%lu sdata_bytes=%ld\n", nbytes, ndat, sdata_bytes);
  fprintf (stderr, "hires_delay_fractional_float_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d\n", blocks.x, blocks.y, blocks.z);
  fprintf (stderr, "hires_delay_fractional_float_kernel: nthread_load=%d nthread_run=%d ntap=%d\n", nthread_load, nthread_run, ntap);
#endif

  const unsigned chan_stride = nant * ndat;
  const unsigned ant_stride  = ndat;

#ifdef USE_CONSTANT_MEMORY
  hires_delay_fractional_float_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in,
              (cuFloatComplex *) d_fbuf, (float *) d_fir_coeffs, nthread_run,
              ndat, chan_stride, ant_stride, ntap);
#else
  hires_delay_fractional_float_kernel<<<blocks, nthread_load, sdata_bytes, stream>>>((int16_t *) d_in,
              (cuFloatComplex *) d_fbuf, (float *) d_fir_coeffs, (float *) d_fringes,
              nthread_run, ndat, chan_stride, ant_stride, ntap);
#endif

#if _GDEBUG
  check_error_stream("hires_delay_fractional_float_kernel", stream);
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
  size_t shm_bytes;

  // each block is a single integration
  //shm_bytes = M * ndim * sizeof(float);

  ///////////////////////////////////////////////////////
  // compute the means of each antenna / channel
  //blocks.x = 1;
  //shm_bytes = 0;
  //unsigned nval_per_thread_mean = ndat_sk / 1024;
  //hires_measure_means_kernel <<<blocks, nthreads, shm_bytes, stream>>>( (cuFloatComplex *) d_fbuf, 
  //          (cuFloatComplex *) d_means, nval_per_thread_mean, ndat_sk);
  
  ///////////////////////////////////////////////////////
  // compute the S1 and S2 values from the input
  //
  blocks.x = ndat_sk / M;
  shm_bytes = 64 * sizeof(float);
#if _GDEBUG
  fprintf (stderr, "hires_skcompute_kernel: bytes=%lu ndat=%lu shm_bytes=%ld\n", nbytes, ndat_sk, shm_bytes);
  fprintf (stderr, "hires_skcompute_kernel: blocks.x=%d, blocks.y=%d, blocks.z=%d, nthreads=%u\n", blocks.x, blocks.y, blocks.z, nthreads);
  fprintf (stderr, "hires_skcompute_kernel: d_fbuf=%p d_in=%p, nval_per_thread=%u, ndat_sk=%lu\n", d_fbuf, d_in, nval_per_thread, ndat_sk);
#endif

  unsigned s1_idx = (unsigned) ((s1_count-1) % s1_memory);
  float * d_s1s_curr = ((float *) d_s1s) + (s1_idx * blocks.x * nchan * nant);

  // reuse d_in as a temporary work buffer for the S1 and S2 sums
  hires_skcompute_kernel<<<blocks, nthreads, shm_bytes, stream>>>( (cuFloatComplex *) d_fbuf, (float *) d_s1s_curr, (float *) d_s2s, nval_per_thread, ndat_sk);

#if _GDEBUG
  check_error_stream("hires_skcompute_kernel", stream);
#endif

  //
  unsigned nsums = blocks.x;
  dim3 blocks_skm (1, nant, nchan);

#ifdef MEDIAN_FILTER
  /////////////////////////////////////////////////////////
  // compute the power limits based on the S1 and S2 values
#ifdef _GDEBUG
  fprintf (stderr, "ndat=%lu ndat_sk=%lu nsums=%u\n", ndat, ndat_sk, nsums);
  fprintf (stderr, "s1_idx=%u s1_count=%u\n", s1_idx, s1_count);
#endif

  const unsigned nsigma = 3;
  unsigned valid_memory = s1_memory;
  if (s1_count < s1_memory)
    valid_memory = (unsigned) s1_count;
  shm_bytes = 0;

  // on first iteration, compute sigmas and thresholds
  if (s1_count == 1)
  {
    nthreads = nsums;
    shm_bytes = nthreads * sizeof(float);
    hires_compute_sigmas_kernel<<<blocks_skm,nthreads,shm_bytes,stream>>>((float *) d_s1s, (cuFloatComplex *) d_thresh,  (float *) d_sigmas, nsums);
#if _GDEBUG
    check_error_stream("hires_compute_sigmas_kernel", stream);
#endif
  }


#endif

  //////////////////////////////////////////////////////////
  // mask the input data
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

  unsigned nchan_sum = 50;
  unsigned sk_nsigma = 4;

  shm_bytes = nchan_sum * nsums * sizeof(uint8_t);
  size_t mask_size = nsums * nchan * nant * sizeof(uint8_t);
  cudaMemsetAsync (d_mask, 0, mask_size, stream);
  cudaStreamSynchronize(stream);

#if _GDEBUG
  fprintf (stderr, "hires_skdetect_kernel: blocks_skm.x=%d, blocks_skm.y=%d, blocks_skm.z=%d\n", blocks_skm.x, blocks_skm.y, blocks_skm.z);
  fprintf (stderr, "hires_skdetect_kernel: nthreads=%u shm_bytes=%ld\n", nthreads, shm_bytes);
  fprintf (stderr, "hires_skdetect_kernel: d_fbuf=%p d_out=%p, d_in=%p, nsums=%u M=%u, nval_per_thread=%u\n", d_fbuf, d_out, d_thresh, nsums, M, nval_per_thread);
#endif

  hires_skdetect_kernel<<<blocks_skm, nthreads, shm_bytes, stream>>>(d_s1s_curr, (float *) d_s2s, (cuFloatComplex *) d_thresh, (int8_t *) d_mask, (float *) d_sigmas, nchan_sum, sk_nsigma, nsums, M, nval_per_thread);

#if _GDEBUG
  check_error_stream("hires_skdetect_kernel", stream);
#endif

  shm_bytes = nchan_sum * nsums;

#if _GDEBUG
  fprintf (stderr, "hires_skmask_kernel: blocks_skm.x=%d, blocks_skm.y=%d, "
                   "blocks_skm.z=%d\n", blocks_skm.x, blocks_skm.y, blocks_skm.z);
  fprintf (stderr, "hires_skmask_kernel: nthreads=%u shm_bytes=%ld\n", 
                   nthreads, shm_bytes);
  fprintf (stderr, "hires_skmask_kernel: d_fbuf=%p d_out=%p, d_in=%p, nsums=%u "
                   "M=%u, nval_per_thread=%u, nsamp_per_thread=%u ndat_sk=%lu\n", 
                    d_fbuf, d_out, d_in, nsums, M, nval_per_thread, nsamp_per_thread, 
                    ndat_sk);
#endif

  // now compute the power limits for a kernel, taking the mask into account, updating the thresholds and sigmas
  unsigned nthreads_cpl = 1024;
  shm_bytes = 0;
  hires_compute_power_limits_kernel<<<blocks_skm,nthreads_cpl,shm_bytes,stream>>>((float *) d_s1s, (cuFloatComplex *) d_thresh, (float *) d_sigmas, (int8_t *) d_mask, (curandStatePhilox4_32_10_t *) d_rstates, nsums, valid_memory, nsigma, s1_idx);

  shm_bytes = 0;
  hires_skmask_kernel<<<blocks_skm, nthreads, shm_bytes, stream>>>((float *) d_fbuf, (int8_t *) d_out, 
                (int8_t *) d_mask, (curandStatePhilox4_32_10_t *) d_rstates, (float *) d_sigmas, 
#ifndef USE_CONSTANT_MEMORY
                (float *) d_ant_scales,
#endif
                nsums, M, nval_per_thread, nsamp_per_thread, ndat_sk, replace_noise);

#if _GDEBUG
  check_error_stream("hires_skmask_kernel", stream);
#endif
}

// wrapper for getting curandStatePhilox4_32_10_t_t size
size_t hires_curandState_size()
{
  return sizeof(curandStatePhilox4_32_10_t);
}
