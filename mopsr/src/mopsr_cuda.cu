
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <inttypes.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "mopsr_cuda.h"

#define BEAMS_PER_LOOP 8
#define WARP_SIZE      32
#define NWARPS_PER_BLOCK 32
#define NSUM 32
//#define _GDEBUG      1

#ifdef __CUDA_ARCH__
    #if (__CUDA_ARCH__ >= 300)
        #define HAVE_SHFL
    #else
        #define NO_SHFL
    #endif
#endif

// large parts of these kernels require SHFL instructions that are 
// only available in sm_30 (kepler) or greater

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

__global__ void input_transpose_TFS_to_FST_hires (
     int16_t * in, int16_t * out,
     const unsigned nchan, const unsigned nant,
     const unsigned nval, const unsigned nval_per_thread,
     const unsigned samp_stride, const unsigned chan_block_stride,
     const unsigned out_chanant_stride)
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
    out[odx] = sdata[sdx];
    sdx += nant;
    odx += out_chanant_stride * nant;
  }
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

void mopsr_input_transpose_TFS_to_FST_hires (cudaStream_t stream,
      void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  unsigned nthread = 1024;

  // process 32 samples and 16 channels in a block
  const unsigned nsamp_per_block = 32;
  const unsigned nchan_per_block = 16;
  const unsigned nchanblocks = nchan / nchan_per_block;

  const unsigned nval_per_block  = nsamp_per_block * nchan_per_block * nant;
  const uint64_t nsamp = nbytes / (nchan * nant * ndim);

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

  const size_t sdata_bytes = (nsamp_per_block * nchan_per_block * nant * ndim) + 256;

  // nbytes of bytes different (for input) between each block of data
  const unsigned samp_stride = nchan * nant;
  const unsigned chan_block_stride = nchan_per_block * nant;


#ifdef _GDEBUG
  fprintf (stderr, "input_transpose_TFS_to_FST: nval_per_block=%u, nval_per_thread=%u\n", nval_per_block, nval_per_thread);
  fprintf (stderr, "input_transpose_TFS_to_FST: nbytes=%lu, nval=%lu\n", nbytes, nval);
  fprintf (stderr, "input_transpose_TFS_to_FST: input=%p output=%p sdata_bytes=%ld, nsamp_per_block=%u\n", d_in, d_out, sdata_bytes, nsamp_per_block);
#endif

  input_transpose_TFS_to_FST_hires<<<blocks,nthread,sdata_bytes,stream>>>((int16_t *) d_in,
        (int16_t *) d_out, nchan, nant, nval, nval_per_thread, samp_stride, chan_block_stride, nsamp);

#ifdef _GDEBUG
  check_error_stream("input_transpose_TFS_to_FST_hires", stream);
#endif

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
    if (iant % 2 == 0)
    {
      re += (float) ant8[2*iant];
      im += (float) ant8[2*iant+1];
    }
  }

  ant8[0] = (int8_t) rintf (re);
  ant8[1] = (int8_t) rintf (im);

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

    output[out_idx] = sdata[sout_idx + (2*ichan)];

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


// perform a transpose from FST to FTS order, but with 2 samples paired
// with each other in 32-bit word
__global__ void input_transpose_FST_to_FTS (
     const int32_t * input, int32_t * output,
     const unsigned nchan, const unsigned nant,
     const unsigned nsamp_per_block, const unsigned chan_stride)
{
  // to hold 16+1 32-bit packed samples, for all antenna
  extern __shared__ int32_t sdata_32t[];

  const unsigned warp_num = threadIdx.x / nsamp_per_block;
  const unsigned warp_idx = threadIdx.x % nsamp_per_block;
  const int warps_per_block = blockDim.x / nsamp_per_block;
  const uint64_t nsamp    = nsamp_per_block * gridDim.x;

  // input offset (ichan      * chan_stride) + (iant * nsamp)     + (iblock * nsamp_per_block)     + isamp
  uint64_t idx  = (blockIdx.y * chan_stride) + (warp_num * nsamp) + (blockIdx.x * nsamp_per_block) + warp_idx;

  // shm offset  (iant     * nsamp per block) + isamp
  unsigned sdx = (warp_num * nsamp_per_block) + warp_idx;

  // loop offsets
  const unsigned ant_stride = nsamp * warps_per_block;
  //const unsigned sin_stride = (WARP_SIZE + 1);
  const unsigned sin_stride = nsamp_per_block * warps_per_block;

  // read ST data straight into shm, with 1 sample stride offset for bank conflicts
  for (unsigned iant=warp_num; iant<nant; iant+=warps_per_block)
  {
    sdata_32t[sdx] = input[idx];
    idx += ant_stride;
    sdx += sin_stride;
  }

  // ensure all blocks have loaded nant * nsamp_per_block samples
  __syncthreads();

  // write out transposed data, each warp writing out nsamp_per_block antenna
  idx = (blockIdx.y * chan_stride) + (blockIdx.x * nsamp_per_block * nant) + threadIdx.x;

  // assume nthread == nantenna, output antenna = threadId.x
  sdx = threadIdx.x * nsamp_per_block;
  //sdx = threadIdx.x * (WARP_SIZE + 1);

  for (unsigned isamp=0; isamp<nsamp_per_block; isamp++)
  {
    output[idx] = sdata_32t[sdx];
    // increment to next output sample
    idx += nant;
    sdx++;
  }
}


// transpose each channel from ST to TS order
void mopsr_input_transpose_FST_to_FTS (cudaStream_t stream,
      void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant)
{
  const unsigned ndim = 2;
  const uint64_t ndat = nbytes / (nchan * nant * ndim) / 2;
  unsigned nthread = nant;

  // each block reads 16x2 time samples for each antenna
  const unsigned nsamp_per_block = 16;
  const unsigned chan_stride = nant * ndat;

  dim3 blocks = dim3 (ndat / nsamp_per_block, nchan, 1);
  const size_t sdata_bytes = nant * sizeof(int32_t) * (nsamp_per_block + 1);

  fprintf (stderr, "input_transpose_FST_to_FTS: nbytes=%lu, ndat=%lu, nant=%u nchan=%u \n", nbytes, ndat, nant, nchan);
  fprintf (stderr, "input_transpose_FST_to_FTS: nthread=%d, blocks.x=%d\n", nthread, blocks.x);
  fprintf (stderr, "input_transpose_FST_to_FTS: input=%p output=%p sdata_bytes=%ld nsamp_per_block=%u chan_stride=%u\n", d_in, d_out, sdata_bytes, nsamp_per_block, chan_stride);

  input_transpose_FST_to_FTS<<<blocks,nthread,sdata_bytes,stream>>>((int32_t *)d_in, (int32_t *) d_out, nchan, nant, nsamp_per_block, chan_stride);

#ifdef _GDEBUG
  check_error_stream ("input_transpose_FST_to_FTS", stream);
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
      sincosf(sin_theta * sh_ant_factors[iant], &(phasor.y), &(phasor.x));
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

#ifdef HAVE_SHFL
__device__ __forceinline__ cuFloatComplex shflComplex( cuFloatComplex r, int lane )
{
  return make_cuComplex ( __shfl( r.x, lane ), __shfl( r.y, lane ) );
}

__device__ __forceinline__ cuFloatComplex shfl_xor_Complex ( cuFloatComplex r, int lane )
{
  return make_cuComplex ( __shfl_xor( r.x, lane ), __shfl_xor( r.y, lane ) );
}
#endif

#ifdef EIGHT_BIT_PHASORS 
__global__ void tile_beams_kernel_2048(
        const __restrict__ int32_t * input, float * output, int8_t * phasors,
        unsigned nbeam, unsigned ndat, unsigned nant)
#else
__global__ void tile_beams_kernel_2048(
        const __restrict__ int32_t * input, float * output, float * phasors,
        unsigned nbeam, unsigned ndat, unsigned nant)
#endif
{
  extern __shared__ float sdata_tb_c[];

  float * re_phasors = sdata_tb_c + (2 * 32 * BEAMS_PER_LOOP);
  float * im_phasors = re_phasors + (nant * BEAMS_PER_LOOP);

#ifdef HAVE_SHFL
  const int warp_num = threadIdx.x / WARP_SIZE;
  const int warp_idx = threadIdx.x & 0x1F;
  float power;
#endif

  int32_t val32;
  int8_t * ptr8 = (int8_t *) &val32;
  const unsigned nbeamant = nbeam * nant;
  //const float scale = 127.5; 
  cuFloatComplex b1s[BEAMS_PER_LOOP];
  cuFloatComplex b2s[BEAMS_PER_LOOP];
  cuFloatComplex val;

  // this kernel exectutes for computes 4 beams at a time for 1024 samples
  for (unsigned ibeam=0; ibeam<nbeam; ibeam += BEAMS_PER_LOOP)
  {
    unsigned ibeamant = ibeam * nant;

    // use all threads in the warp to load the load phasors for this beam
    // and all antenna into shared memory.
    for (unsigned i=threadIdx.x; i<nant*BEAMS_PER_LOOP; i+=blockDim.x)
    {
      re_phasors[i] = (float) phasors[ibeamant + i];
      im_phasors[i] = (float) phasors[nbeamant + ibeamant + i];
    }

    __syncthreads();

    // for all the antenna, perform complex multiplications on  the 4 beams
    unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    //for (unsigned i=0; i<nbeam_loop; i++)
    for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
    {
      b1s[i].x = 0;
      b1s[i].y = 0;
      b2s[i].x = 0;
      b2s[i].y = 0;
    }

    for (unsigned iant=0; iant<nant; iant++)
    {
      // load 4 x 8bit values (2 complex samples) for this time sample and antenna
      val32 = input[idx];

      // make a complex float from this input 
      //val = make_cuComplex (((float) ptr8[0]) / scale, ((float) ptr8[1]) / scale);
      val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);

      unsigned pidx = iant;
      for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
      {
        // multiply by phasor and add to the beam (yes this is a += operation)
        if (ibeam == 0 && i == 0)
          b1s[0].x = fmaf( val.x, val.x, fmaf (val.y, val.y, b1s[0].x));
        else
          b1s[i] = cuCfmaf (make_cuComplex(re_phasors[pidx], im_phasors[pidx]), val, b1s[i]);
        pidx += nant;
      }

      //val = make_cuComplex (((float) ptr8[2] / scale), ((float) ptr8[3]) / scale);
      val = make_cuComplex ((float) ptr8[2], (float) ptr8[3]);
      pidx = iant;
      for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
      {
        // multiply by phasor and add to the beam (yes this is a += operation)
        if (ibeam == 0 && i == 0)
          b2s[0].x = fmaf( val.x, val.x, fmaf (val.y, val.y, b2s[0].x));
        else
          b2s[i] = cuCfmaf (make_cuComplex(re_phasors[pidx],im_phasors[pidx]), val, b2s[i]);
        pidx += nant;
      }

      idx += ndat/2;
    }

#ifdef HAVE_SHFL
    // detect each sample and integrate across the warp (factor of 32 in time)
    // this takes us from 1.28us to 81.92
    unsigned sdx = warp_num;
    for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
    {
      if (ibeam == 0 && i == 0)
        power = b1s[i].x;
      else
        power = (b1s[i].x * b1s[i].x) + (b1s[i].y * b1s[i].y);
      power += __shfl_down (power, 16);
      power += __shfl_down (power, 8);
      power += __shfl_down (power, 4);
      power += __shfl_down (power, 2);
      power += __shfl_down (power, 1);

      // power now contains the integrated power for this warp (i.e. 40.96 us samples
      // we write these to shared memory in ST order. T=64, S=8
      if (warp_idx == 0)
      {
        sdata_tb_c[sdx] = power;
      }

      if (ibeam == 0 && i == 0)
        power = b2s[i].x;
      else
        power = (b2s[i].x * b2s[i].x) + (b2s[i].y * b2s[i].y);
      power += __shfl_down (power, 16);
      power += __shfl_down (power, 8);
      power += __shfl_down (power, 4);
      power += __shfl_down (power, 2);
      power += __shfl_down (power, 1);

      // power now contains the integrated power for this warp (i.e. 40.96 us samples
      // we write these to shared memory in ST order. T=32, S=8
      if (warp_idx == 0)
      {
        sdata_tb_c[sdx] += power;
        sdx += 32;
      }
    }

    __syncthreads();

    // one warp per output beam
    if (warp_num < BEAMS_PER_LOOP)
    {
      // threads to generate 4 x 655.36 time samples from 32 x 81.92)
      power = sdata_tb_c[(warp_num * WARP_SIZE) + warp_idx];
      power += __shfl_down (power, 4);
      power += __shfl_down (power, 2);
      power += __shfl_down (power, 1);

      // warp_idxs 0, 8, 16, 24 have the 4 time samples for beam warp_num
      if (warp_idx % 8 == 0)
      {
        const unsigned obeam = ibeam + warp_num;
        const unsigned ndat_out = ndat / 512;
        const unsigned osamp = warp_idx / 8;
        // output is in ST format
        unsigned out_idx = (obeam * ndat_out) + (blockIdx.x * 4) + osamp;
        output[out_idx] = power;
      }
    }
#endif
  }
}

// load 2048 samples per block, form beams, scrunch down x32 to write out 
// 64 samples. input FST output SFT
__global__ void tile_beams_kernel_2048_32scr (
        const __restrict__ int32_t * input, float * output, float * phasors,
        unsigned nbeam, unsigned ndat, unsigned nant)
{

  extern __shared__ float sdata_tb_c[];
  //const float scale = 127.5;

  float * re_phasors = sdata_tb_c + (2 * 32 * BEAMS_PER_LOOP);
  float * im_phasors = re_phasors + (nant * BEAMS_PER_LOOP);

  const int warp_num = threadIdx.x / WARP_SIZE;
  const int warp_idx = threadIdx.x & 0x1F;
#ifdef HAVE_SHFL
  float power;
#endif

  int32_t val32;
  int8_t * ptr8 = (int8_t *) &val32;
  const unsigned nbeamant = nbeam * nant;

  cuFloatComplex b1s[BEAMS_PER_LOOP];
  cuFloatComplex b2s[BEAMS_PER_LOOP];
  cuFloatComplex val;

  // shift phasors pointer by ichan * chan_stride
  phasors += blockIdx.y * nant * nbeam * 2;

  // shift input by ndat/2 (since int32_t *)
  input   += blockIdx.y * nant * (ndat / 2);

  // shift output by output_ndat to align to right channel
  output  += blockIdx.y * (ndat / 32);

  // this kernel exectutes for computes 4 beams at a time for 1024 samples
  for (unsigned ibeam=0; ibeam<nbeam; ibeam += BEAMS_PER_LOOP)
  {
    unsigned ibeamant = ibeam * nant;

    // use all threads in the warp to load the load phasors for this beam
    // and all antenna into shared memory.
    for (unsigned i=threadIdx.x; i<nant*BEAMS_PER_LOOP; i+=blockDim.x)
    {
      re_phasors[i] = (float) phasors[ibeamant + i];
      im_phasors[i] = (float) phasors[nbeamant + ibeamant + i];
    }

    __syncthreads();

    // for all the antenna, perform complex multiplications on the 4 beams
    unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
    {
      b1s[i].x = 0;
      b1s[i].y = 0;
      b2s[i].x = 0;
      b2s[i].y = 0;
    }

    for (unsigned iant=0; iant<nant; iant++)
    {
      // load 4 x 8bit values (2 complex samples) for this time sample and antenna
      val32 = input[idx];

      // make a complex float from this input 
      //val = make_cuComplex ((float) ptr8[0] / scale, (float) ptr8[1] / scale);
      val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);

      unsigned pidx = iant;
      for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
      {
        // multiply by phasor and add to the beam (yes this is a += operation)
        if (ibeam == 0 && i == 0)
          b1s[0].x = fmaf( val.x, val.x, fmaf (val.y, val.y, b1s[0].x));
        else
        {
          //b1s[i] = cuCaddf (val, b1s[i]);
          b1s[i] = cuCfmaf (make_cuComplex(re_phasors[pidx], im_phasors[pidx]), val, b1s[i]);
        }
        //if (blockIdx.x == 0 && ibeam == 0 && i == 1 && threadIdx.x == 0)
        //  printf("OLD: samp=%d ant=%d antenna=(%f,%f) weight=(%f,%f) sum=(%f,%f)\n", 
        //      2*threadIdx.x, iant,
        //      val.x, val.y, re_phasors[pidx], im_phasors[pidx], b1s[i].x, b1s[i].y);

        pidx += nant;
      }

      //val = make_cuComplex ((float) ptr8[2] / scale, (float) ptr8[3] / scale);
      val = make_cuComplex ((float) ptr8[2], (float) ptr8[3]);
      pidx = iant;
      for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
      {
        // multiply by phasor and add to the beam (yes this is a += operation)
        if (ibeam == 0 && i == 0)
          b2s[0].x = fmaf( val.x, val.x, fmaf (val.y, val.y, b2s[0].x));
        else
        {
          //b2s[i] = cuCaddf (val, b2s[i]);
          b2s[i] = cuCfmaf (make_cuComplex(re_phasors[pidx],im_phasors[pidx]), val, b2s[i]);
        }
        //if (blockIdx.x == 0 && ibeam == 0 && i == 1)
        //  printf("OLD: samp=%d ant=%d antenna=(%f,%f) sum=(%f,%f)\n", 
        //      2*threadIdx.x+1, iant,
        //      val.x, val.y, b2s[i].x, b2s[i].y);
        pidx += nant;
      }

      idx += ndat/2;
    }

    // detect each sample and integrate across the warp (factor of 32 in time)
    // this takes us from 10.24 to 327.68 us
#ifdef HAVE_SHFL
    unsigned sdx = 2 * warp_num;
    for (unsigned i=0; i<BEAMS_PER_LOOP; i++)
    {
      if (ibeam == 0 && i == 0)
        power = b1s[i].x;
      else
        power = (b1s[i].x * b1s[i].x) + (b1s[i].y * b1s[i].y);

      //if ((blockIdx.y == 0) && (ibeam == 0) && (i == 1) && (blockIdx.x == 0) && (warp_num == 0))
      //  printf("GPU %d %f %f %f\n", 2*threadIdx.x+0, b1s[i].x, b1s[i].y, power);

      // since the consecutive samples are spread across b1 and b2 
      power += __shfl_down (power, 8);
      power += __shfl_down (power, 4);
      power += __shfl_down (power, 2);
      power += __shfl_down (power, 1);

      // power now contains the integrated power for this warp (i.e. 327.68 us 
      // samples. Write these to shared memory in ST order. T=64, S=8
      if (warp_idx == 0 || warp_idx == 16)
      {
        sdata_tb_c[sdx + warp_idx/16] = power;  // sample sdx + 0 and 1
      }

      if (ibeam == 0 && i == 0)
        power = b2s[i].x;
      else
        power = (b2s[i].x * b2s[i].x) + (b2s[i].y * b2s[i].y);

      //if ((blockIdx.y == 0) && (ibeam == 0) && (i == 1) && (blockIdx.x == 0) && (warp_num == 0))
      //  printf("GPU %d %f %f %f\n", 2*threadIdx.x+1, b2s[i].x, b2s[i].y, power);

      power += __shfl_down (power, 8);
      power += __shfl_down (power, 4);
      power += __shfl_down (power, 2);
      power += __shfl_down (power, 1);

      if (warp_idx == 0 || warp_idx == 16)
      {
        sdata_tb_c[sdx+warp_idx/16] += power;
        //if ((blockIdx.y == 0) && (ibeam == 0) && (i == 1) && (blockIdx.x == 0) && (warp_num == 0))
        //  printf("GPU %d sdata_tb_c[%d]==%f\n", warp_idx, sdx+warp_idx/16, sdata_tb_c[sdx+warp_idx/16]);
        sdx += 64;
      }
    }

    __syncthreads();

#endif



    // there are now 8beams * (2 * 32) samples in SHM to write
    // out to gmem, do 1 warp per beam, 
    if (warp_num < BEAMS_PER_LOOP)
    {
      //              ibeam * 64                + warp_idx
      unsigned sdx = (warp_num * WARP_SIZE * 2) + warp_idx;

      const unsigned obeam = ibeam + warp_num;
      //              obeam * nchan * ndat_out         +  osamp_block      + osamp
      unsigned odx = (obeam * gridDim.y * (ndat / 32)) + (blockIdx.x * 64) + warp_idx;

      // write out the 64 samples for this beam
      output[odx] = sdata_tb_c[sdx];
      output[odx+32] = sdata_tb_c[sdx+32];
    }
  }
}

/*
__host__ __device__
cuFloatComplex ComplexInt8_to_cuFloatComplex(ComplexInt8 input)
{
  return make_cuFloatComplex((float)(input.x),(float)(input.y));
}

__host__ __device__
cuFloatComplex char2_to_cuFloatComplex(char2 input)
{
  return make_cuFloatComplex(float(input.x)/127.0,float(input.y)/127.0);
}

// each block proceses 1024 samples, producing 32 output samples
// and 32 beams
// each warp converts 32 samples -> 1 sample, and each block
// therefore processes 32 x 32 = 1024 samples
__global__ void tile_beams_kernel_32scr_TS (
        const int * __restrict__ input, 
        float * output,
        const half2 * __restrict__ phasors,
        unsigned nbeam, unsigned ndat, unsigned nant)
{
  extern __shared__ half2 shared_phasors[];

  const int warp_idx = threadIdx.x / WARP_SIZE; // isamp
  const int lane_idx = threadIdx.x & 0x1F;      // ibeam

  // the beam this thread is acuumulating
  const unsigned ibeam = (blockIdx.z * WARP_SIZE) + lane_idx;

  // beam-forming phasors offset for this block
  int phasors_offset = (blockIdx.y * nant * nbeam) + ibeam;

  int shared_phasors_offset = lane_idx;

  // each iteration of the loop, loads phasors for these beams + ants
  // phasors are read/stored in ant,beam order
  for (int iant = lane_idx; iant < nant; iant+=WARP_SIZE)
  {
    // storage     iant * nbeam     + ibeam
    //shared_phasors[shared_phasors_offset] = phasors[phasors_offset];
    shared_phasors[shared_phasors_offset] = __floats2half2_rn(1.0f, 0.0f);
    phasors_offset += nbeam;
    shared_phasors_offset += WARP_SIZE;
  }
  __syncthreads();

  // output sample offset, each warp loads a set of 32 samples that are condensed to 1 sample
  const int osamp = (blockIdx.x * NWARPS_PER_BLOCK) + warp_idx;

  // input sample offset /2 since loading 2 antenna at a time
  int sample_offset = NSUM * osamp / 2;

  // accumulated output power
  float power = 0.0f;

  int two_samples;

  // pointer to access antenna
  // TODO check wither other_ant is really necessary
  char4 * samples = (char4 *) &two_samples;

  // samples to be processed
  for (int isamp = sample_offset; isamp < sample_offset + (NSUM/2); ++isamp)
  {
    // compute the offset of this block on the input
    //                channel offset                +  samp offset     
    //                (blockIdx.y * nant * nsamp)/2 + (isamp * nant)/2 + lane_idx;
    const int idx = nant * (blockIdx.y * ndat + isamp) / 2 + lane_idx;

    // set the complex accumulator for the tied-array beam
    cuFloatComplex tb = make_cuComplex(0.0f, 0.0f);

    // loop over nant (/2 since dual load)
    for (int iant=0; iant<nant; iant+=WARP_SIZE)
    {
      // load 2 complex samples at once
      two_samples = input[idx + iant];

      cuFloatComplex sample1 = make_cuFloatComplex(float(samples->x),float(samples->y));
      cuFloatComplex sample2 = make_cuFloatComplex(float(samples->z),float(samples->w));
      cuFloatComplex sample;

      //if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && warp_idx == 0 && iant < 64)
      //  printf("[%d][%d] load isamp=%d offset=%d\n", threadIdx.x, iant, isamp, idx + iant);

      // other antenna pair for shuffling
      //int other_two_samples;
      // use warp shuffle to sum these antenna into the TB result and share
      for (int other_lane_idx=0; other_lane_idx < WARP_SIZE; ++other_lane_idx)
      {
        cuFloatComplex weight = __half22float2(shared_phasors[iant*WARP_SIZE+lane_idx]);
        //cuFloatComplex weight = make_cuFloatComplex(1.0f,0.0f);

#ifdef HAVE_SHFL
        // shuffle the first sample across
        sample.x = __shfl(sample1.x, other_lane_idx);
        sample.y = __shfl(sample1.y, other_lane_idx);
        //other_two_samples = __shfl(two_samples, other_lane_idx);
#endif
        // first antenna pair
        if (ibeam == 0)
          fmaf( sample.x, sample.x, fmaf (sample.y, sample.y, tb.x));
        else
          tb = cuCfmaf(sample, weight, tb);

#ifdef HAVE_SHFL
        sample.x = __shfl(sample2.x, other_lane_idx);
        sample.y = __shfl(sample2.y, other_lane_idx);
#endif
        // first antenna pair
        if (ibeam == 0)
          fmaf( sample.x, sample.x, fmaf (sample.y, sample.y, tb.x));
        else
          tb = cuCfmaf(sample, weight, tb);

          //if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && ibeam == 1 && sample_offset == 0)
          //  printf("NEW: samp=%d iant=%d sample=(%d,%d) weight=(%f,%f) sum=(%f,%f)\n", 
          //         2*isamp, iant+other_lane_idx, samples->x, samples->y, weight.x, weight.y, tb.x, tb.y);

          //if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && ibeam == 1 && sample_offset == 0)
          //  printf("NEW: samp=%d iant=%d sample=(%d,%d) weight=(%f,%f) sum=(%f,%f)\n", 
          //         2*isamp+1, iant+other_lane_idx, samples->w, samples->z, weight.x, weight.y, tb.x, tb.y);
      }
    }

    // now that all antennae are summed, detected and accumulate the powers for this sample / beam

    float amplitude;
    if (ibeam == 0)
      amplitude = tb.x;
    else
      amplitude = (tb.x * tb.x) + (tb.y * tb.y);
    power += amplitude;
   // if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && ibeam == 1 && threadIdx.x == 1)
   //   printf("NEW amplitude=%f power=%f\n", amplitude, power);
  }

  // the warp_idx corresponds to the output sample 
  // the lane_idx correponds to the output beam
  // order SFT for Fan Beams
  //             (ibeam * nchan * n_osamp) + (ichan * n_osamp) + osamp

  //unsigned odx = (ndat/32) * (ibeam * gridDim.y + blockIdx.y) + osamp;
  unsigned odx = threadIdx.x;
  //if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && ibeam == 1 && threadIdx.x == 1)
  //printf("OUTPUT %u %u %u %f ibeam=%u nchan=%u\n", odx, blockIdx.x, threadIdx.x, power, ibeam, gridDim.y);

  output[odx] = power;
}
*/

void mopsr_tile_beams_precomp (cudaStream_t stream, void * d_in, void * d_fbs, void * d_phasors,
                       uint64_t bytes, unsigned nbeam, unsigned nant, unsigned nchan)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nchan * nant * ndim);
  const unsigned nthread = 1024;

  const unsigned nbeam_block = BEAMS_PER_LOOP;
  unsigned ndat_per_block = 2048;
  dim3 blocks = dim3 (ndat / ndat_per_block, nchan, 1);
  //unsigned nblocks = ndat / ndat_per_block;
  if (ndat % ndat_per_block)
    fprintf (stderr, "WARNING: ndat not divisible by %d\n", ndat_per_block);

  size_t sdata_bytes = (nbeam_block * (nthread/WARP_SIZE) * sizeof(float) * 2) +
                       (nbeam_block * nant * sizeof (float _Complex));

#ifdef _GDEBUG
  fprintf (stderr, "bytes=%lu ndat=%lu blocks=(%u,%u,%u) threads=%u shm=%ld\n", bytes, ndat, blocks.x, blocks.y, blocks.z, nthread, sdata_bytes);
#endif

#ifdef EIGHT_BIT_PHASORS
  tile_beams_kernel_2048<<<blocks, nthread, sdata_bytes, stream>>>((int32_t *) d_in, (float *) d_fbs, (int8_t *) d_phasors, nbeam, ndat, nant);
#else
#ifdef HIRES
  tile_beams_kernel_2048_32scr<<<blocks, nthread, sdata_bytes, stream>>>((int32_t *) d_in, (float *) d_fbs, (float *) d_phasors, nbeam, ndat, nant);
#else
  tile_beams_kernel_2048<<<blocks, nthread, sdata_bytes, stream>>>((int32_t *) d_in, (float *) d_fbs, (float *) d_phasors, nbeam, ndat, nant);
#endif
#endif

#ifdef _GDEBUG
  check_error_stream("tile_beams_kernel_2048", stream);
#endif

}

/*
// assumes data has been transposed to FTS order
void mopsr_tile_beams_transpose (cudaStream_t stream, void * d_in, void * d_fbs, void * d_phasors,
                       uint64_t bytes, unsigned nbeam, unsigned nant, unsigned nchan)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nchan * nant * ndim);
  const unsigned nthread = 1024;
  const unsigned nwarps_per_block = nthread / WARP_SIZE;
  const unsigned nbeam_per_block = 32;
  const uint64_t ndat_per_block = nwarps_per_block * 32;

  dim3 blocks = dim3 (ndat / ndat_per_block, nchan, nbeam / nbeam_per_block);
  if (ndat % ndat_per_block)
    fprintf (stderr, "WARNING: ndat not divisible by %d\n", ndat_per_block);
  if (nbeam % 32)
    fprintf (stderr, "WARNING: nbeam not divisible by 32\n");

  size_t sdata_bytes = 32 * nant * sizeof(half2);

#ifdef _GDEBUG
  fprintf (stderr, "mopsr_tile_beams_transpose: bytes=%lu ndat=%lu blocks=(%u,%u,%u) threads=%u shm=%ld\n", bytes, ndat, blocks.x, blocks.y, blocks.z, nthread, sdata_bytes);
#endif

#ifdef HIRES
  tile_beams_kernel_32scr_TS<<<blocks, nthread, sdata_bytes, stream>>>((int *) d_in, (float *) d_fbs, (half2 *) d_phasors, nbeam, ndat, nant);
#endif

#ifdef _GDEBUG
  check_error_stream("tile_beams_kernel_32scr_TS", stream);
#endif
}
*/

__global__ void tie_beam_kernel (int16_t * in, cuFloatComplex * out, cuFloatComplex * d_phasors, uint64_t ndat, unsigned nant)
{
  extern __shared__ cuFloatComplex s_phasors[];

  const unsigned ichan = blockIdx.y;

  int16_t val16;
  int8_t * ptr8 = (int8_t *) &val16;

  cuFloatComplex tied_beam = make_cuComplex(0.0,0.0);

  // load nant phasors for the ichan
  if (threadIdx.x < nant)
  {
    s_phasors[threadIdx.x] = d_phasors[ichan*nant + threadIdx.x];
  }

  __syncthreads();

  // idat
  const unsigned idx  = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= ndat)
    return;

  //const uint64_t out_off = ndat * ichan;

  // output in TF order (isamp * nchan) + ichan
  const uint64_t odx = idx * gridDim.y + ichan;
  const uint64_t in_off  = ndat * ichan * nant; 

  // increment to the right channel
  in  += in_off;
  //out += out_off;

  // step through the antennas, forming tied array beam
  for (unsigned iant=0; iant<nant; iant++)
  {
    val16 = in[idx];
    
    // make a complex float from this input
    cuFloatComplex val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);

    tied_beam = cuCfmaf (s_phasors[iant], val, tied_beam);
    
    in += ndat;
  }

  // output ordering is good!
  //out[idx] = tied_beam;
  out[odx] = tied_beam;
}

/*
 * create steer a tied array beam to the target position
 */
void mopsr_tie_beam (cudaStream_t stream, void * d_in, void * d_out, void * d_phasors,
                     uint64_t bytes, unsigned nant, unsigned nchan)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nchan * nant * ndim);

  // input order is FST, output is FT
  unsigned nthreads = 1024;
  dim3 blocks = dim3 (ndat / nthreads, nchan, 1);
  size_t shm_bytes = nant * sizeof(cuFloatComplex);
  if (ndat % nthreads)
    blocks.x++;

#ifdef _GDEBUG
  fprintf (stderr, "bytes=%lu ndat=%lu nthreads=%u blocks=(%u,%u,%u) shm_bytes=%ld\n", 
           bytes, ndat, nthreads, blocks.x, blocks.y, blocks.z, shm_bytes);
#endif

  tie_beam_kernel<<<blocks, nthreads, shm_bytes, stream>>>((int16_t *) d_in, (cuFloatComplex *) d_out, (cuFloatComplex *) d_phasors, ndat, nant);

#ifdef _GDEBUG
  check_error_stream("tie_beam_kernel", stream);
#endif

}

// integrate 64 samples together from each antenna
__global__ void mod_beam_kernel_64 (int16_t * in, float * out, uint64_t ndat, unsigned nant)
{
  extern __shared__ float block_power_sums[];

  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;

  // offset [ant_offset + block_offset]
  const unsigned offset = (blockIdx.y * ndat) + (blockIdx.x * blockDim.x);

  const unsigned idx  = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= ndat)
    return;

  int16_t val16;
  int8_t * ptr8 = (int8_t *) &val16;

  // load the value from this times sample into a local variable
  val16 = in[offset + threadIdx.x];

  // make a complex float from this input
  cuFloatComplex val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);

  // detect
  float power = val.x * val.x + val.y * val.y;

  // add all of the time samples from this warp together
#ifdef HAVE_SHFL
  power += __shfl_down (power, 16);
  power += __shfl_down (power, 8);
  power += __shfl_down (power, 4);
  power += __shfl_down (power, 2);
  power += __shfl_down (power, 1);
#endif

  if (warp_idx == 0)
    block_power_sums[warp_num] = power;

  __syncthreads();

  if (warp_num == 0)
  {
    power = block_power_sums[warp_idx];
#ifdef HAVE_SHFL
    power += __shfl_down (power, 16);
#endif

    if (warp_idx < 16)
    {
      out[offset/64 + warp_idx] = power;
    }
  }
}

// integrate 512 time samples together from each antenna
__global__ void mod_beam_kernel_512 (int16_t * in, float * out, uint64_t ndat)
{
  extern __shared__ float block_power_sums[];
  
  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;

  // offset [ant_offset + block_offset]
  const unsigned offset = (blockIdx.z * ndat) + (blockIdx.x * blockDim.x);

  const unsigned idx  = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= ndat)
    return;
  
  int16_t val16;
  int8_t * ptr8 = (int8_t *) &val16;
  
  // load the value from this times sample into a local variable
  val16 = in[offset + threadIdx.x];
  
  // make a complex float from this input
  cuFloatComplex val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);
  
  // detect
  float power = val.x * val.x + val.y * val.y;
  
  // add all of the time samples from this warp together
#ifdef HAVE_SHFL
  power += __shfl_down (power, 16);
  power += __shfl_down (power, 8);
  power += __shfl_down (power, 4);
  power += __shfl_down (power, 2);
  power += __shfl_down (power, 1);
#endif

  if (warp_idx == 0)
    block_power_sums[warp_num] = power;
    
  __syncthreads();

  if (warp_num == 0)
  { 
    power = block_power_sums[warp_idx];
#ifdef HAVE_SHFL
    power += __shfl_down (power, 8);
    power += __shfl_down (power, 4);
    power += __shfl_down (power, 2);
    power += __shfl_down (power, 1);
#endif
  
    if (warp_idx == 0 || warp_idx == 16)
    {
      out[offset/512 + warp_idx/16] = power;
    }
  }
}

// integrate 32 time samples together from each antenna, FST -> SFT
__global__ void mod_beam_kernel_32 (int16_t * in, float * out, uint64_t ndat)
{
  extern __shared__ float block_power_sums[];
  
  const unsigned warp_num = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx = threadIdx.x % WARP_SIZE;
  const unsigned ichan = blockIdx.y;
  const unsigned nchan = gridDim.y;
  const unsigned iant  = blockIdx.z;
  const unsigned nant = gridDim.z;

  // input sample number
  const unsigned idx  = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= ndat)
    return;

  const uint64_t ndat_out = ndat / 32;

  // input offset in FST  [chan_offset + ant_offset + block_offset]
  const unsigned in_offset = (ichan * ndat * nant) + (iant * ndat) + (blockIdx.x * blockDim.x);

  // output offset in SFT [ant_offset + chan_offset + block_offset]
  const unsigned out_offset = (iant * nchan * ndat_out) + (ichan * ndat_out) + (blockIdx.x * blockDim.x/32);

  int16_t val16;
  int8_t * ptr8 = (int8_t *) &val16;
  
  // load the value from this times sample into a local variable
  val16 = in[in_offset + threadIdx.x];
  
  // make a complex float from this input
  cuFloatComplex val = make_cuComplex ((float) ptr8[0], (float) ptr8[1]);
  
  // detect
  float power = val.x * val.x + val.y * val.y;
  
  // add all of the time samples from this warp together
#ifdef HAVE_SHFL
  power += __shfl_down (power, 16);
  power += __shfl_down (power, 8);
  power += __shfl_down (power, 4);
  power += __shfl_down (power, 2);
  power += __shfl_down (power, 1);
#endif

  if (warp_idx == 0)
    block_power_sums[warp_num] = power;
    
  __syncthreads();

  if (warp_num == 0)
    out[out_offset + warp_idx] = block_power_sums[warp_idx];
}

void mopsr_mod_beams (cudaStream_t stream, void * d_in, void * d_out, uint64_t bytes, 
                      unsigned nant, unsigned nchan, unsigned tdec)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nchan * nant * ndim);

  unsigned threads = 1024;
  dim3 blocks (ndat / threads, nchan, nant);
  size_t shm_bytes = sizeof(float) * WARP_SIZE;

  if (ndat % threads)
    blocks.x++;

#ifdef _GDEBUG
  fprintf (stderr, "ndat=%lu threads=%u blocks=(%u,%u)\n", ndat, threads, blocks.x, blocks.y);
#endif

  if (tdec == 512)
    mod_beam_kernel_512<<<blocks, threads, shm_bytes, stream>>>((int16_t *) d_in, (float *) d_out, ndat);
  else if (tdec == 32)
    mod_beam_kernel_32<<<blocks, threads, shm_bytes, stream>>>((int16_t *) d_in, (float *) d_out, ndat);
  else
    fprintf (stderr, "mopsr_mod_beams: unrecognized TDEC\n");
}

__global__ void mopsr_transpose_BFST_FST_kernel (int16_t * in, int16_t * out, unsigned ndat)
{
  const unsigned idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  const unsigned in_chan_stride = ndat * 8;
  //              iblock * nchan * in_chan_stride  +  ichan * in_chan_stride       + idat
  uint64_t idx = (blockIdx.z * gridDim.y * in_chan_stride) + (blockIdx.y * in_chan_stride) + idat;

  //                               ndat * nant (total)
  const unsigned out_chan_stride = ndat * gridDim.z * 8;
  //              ichan      * out_chan_stride  +  iant * out_ant_stride  + idat
  uint64_t odx = (blockIdx.y * out_chan_stride) + (blockIdx.z * 8 * ndat) + idat;

  for (unsigned i=0; i<8; i++)
  {
    out[odx] = in[idx];
    idx += ndat;
    odx += ndat;
  }
}


/* transpose from FST (in 8antenna blocks) to FST */
void mopsr_transpose_BFST_FST (cudaStream_t stream, void * d_in, void * d_out, uint64_t bytes,
                               unsigned nant, unsigned nchan)
{
  const unsigned ndim = 2;
  const uint64_t ndat = bytes / (nchan * nant * ndim);

  unsigned nthreads = 1024;
  dim3 blocks (ndat / nthreads, nchan, nant / 8);
  mopsr_transpose_BFST_FST_kernel<<<blocks, nthreads, 0, stream>>>((int16_t *) d_in, (int16_t *) d_out, ndat);
}

/*
using namespace half_float;
void mopsr_convert_float_to_half(void * in, void * out, size_t num)
{
  float * in_ptr = (float *) in;
  half_float::half * out_ptr = (half_float::half *) out;

  for (unsigned i=0; i<num; i++)
  {
    out_ptr[i] = half_cast<half_float::half,std::numeric_limits<float>::round_style>(in_ptr[i]);
  }
}
*/
