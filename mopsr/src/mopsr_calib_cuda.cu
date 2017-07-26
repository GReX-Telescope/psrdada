/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson & Ewan Barr
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <inttypes.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "mopsr_calib_cuda.h"

#define WARP_SIZE       32
//#define _GDEBUG         1

/*
  cuFloatComplex * in:
  Input data stream. Structured as freq/time/ant order (freq changing fastest) 

  cuFloatComplex * out:
  Output data steam. Structured as freq/pair (freq changing fastest)

  mopsr_baseline_t * pairs:
  All pair combinations
  
  unsigned batch_size:
  Number of frequency channels
  
  unsigned nbatch:
  Number of time samples
  
  unsigned npairs:
  Number of pairs

  unsigned nsamps:
  batch_size * nbatch
  
  unsigned nread:
  Number of frequencies to read at a time into shared memory

__global__ acc_cp_spectra_kernel(cuFloatComplex * in, cuFloatComplex * out,
				 mopsr_baseline_t * pairs,
				 unsigned batch_size, unsigned nbatch,
				 unsigned npairs, unsigned nsamps, 
				 unsigned nread)
{
  extern __shared__ cuFloatComplex shared []; // size of nread*nant 
  
  int ii,jj,kk,ll;
  int freq_idx, ant_idx;
  int nsamps_by_nread = nsamps/nread;

  
  //Each block deals with nread samples from each antenna
  for (ii=blockIdx.x; ii<nsamps_by_nread; ii+=gridDim.x)
    {
      freq_idx = threadIdx.x % nread; //0-nread
      samp_idx = ii*nread;
      
      for (ant_idx = threadIdx.x/nread;
	   ant_idx < nant;
	   ant_idx += blockDim.x/nread)
	{
	  
	  
	  
	}
    }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  int freq_idx = threadIdx.x%batch_size; //0-batch_size
}*/


__global__
void accumulate_cp_spectra_kernel(cuFloatComplex * in, cuFloatComplex * out,
                  mopsr_baseline_t * pairs,
                  unsigned batch_size, unsigned nbatch,
                  unsigned npairs, unsigned nsamps,
                  unsigned in_stride,
                  unsigned out_stride)
{
  mopsr_baseline_t pair;
  unsigned ii,jj,kk;
  cuFloatComplex val, val_a, val_b;
  unsigned pair_idx;
  unsigned pair_pos_a;
  unsigned pair_pos_b;
  unsigned bin_idx;
  unsigned out_idx;
  unsigned idx_a,idx_b;

  in += (blockIdx.y * in_stride);
  out += (blockIdx.y * out_stride);
  
  //loop over npairs (should only ever execute one loop)
  for (ii=0; ii<npairs; ii+=gridDim.x)
  {
    // each block operates on a single pair
    pair_idx = ii + blockIdx.x;
    pair = pairs[pair_idx];
    pair_pos_a = pair.a * nsamps;
    pair_pos_b = pair.b * nsamps;

    // each thread operates on a bin in the CP spectrum

    //loop over each bin in the cross power spectrum
    for (kk=0; kk<batch_size; kk+=blockDim.x)
    {
      val = make_cuFloatComplex(0.0,0.0);
      bin_idx = threadIdx.x+kk;

      //loop over each batch in the fft
      for (jj=0; jj<nbatch; jj++)
      {
        idx_a = pair_pos_a + bin_idx + (jj * batch_size);
        idx_b = pair_pos_b + bin_idx + (jj * batch_size);
        val_a = in[idx_a];
        val_b = in[idx_b];
        val_a.x /= batch_size;
        val_a.y /= batch_size;
        val_b.x /= batch_size;
        val_b.y /= batch_size;
        val = cuCaddf(val, cuCmulf(cuConjf(val_a),val_b));
      }
      out_idx = pair_idx*batch_size+bin_idx;
      out[out_idx] = cuCaddf(out[out_idx], val);
    }
  }
}

int mopsr_accumulate_cp_spectra(cuFloatComplex * in, cuFloatComplex * out,
                mopsr_baseline_t * pairs,
                unsigned batch_size, unsigned nbatch,
                unsigned npairs, unsigned nsamps, 
                unsigned nchan, unsigned nant,
                cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  dim3 blocks = dim3(props.maxGridSize[0], nchan, 1);
  if (batch_size<nthreads)
    nthreads = batch_size;
  if (npairs<blocks.x)
    blocks.x = npairs;

  // stride in bytes for each channel 
  int in_stride = batch_size * nbatch * nant;
  int out_stride = batch_size * npairs;
  //fprintf (stderr, "blocks=(%d,%d,%d) nthreads=%d in_stride=%d out_stride=%d\n", blocks.x, blocks.y, blocks.z, nthreads, in_stride, out_stride);
  
  accumulate_cp_spectra_kernel<<< blocks, nthreads, 0, stream >>>(in, out, pairs, batch_size, nbatch, npairs, nsamps, in_stride, out_stride);
#if _GDEBUG
  check_error_stream( "mopsr_accumulate_cp_spectra", stream);
#endif
  return 0;
}


__global__ 
void multiply_baselines_kernel (cuFloatComplex * out, const cuFloatComplex * in,
                                const mopsr_baseline_t * pairs, 
                                const unsigned batch_size, 
                                const unsigned npairs) 
{
  // element-wise multiplication of baseline pairs
  // primary component is multiplied by complex conjugate
  // of secondary component.
  mopsr_baseline_t pair;
  int ii,jj,out_idx,idx_a,idx_b,pair_idx;
  for (ii=0; ii<npairs; ii+=gridDim.x)
  {
    pair_idx = ii + blockIdx.x;
    if (pair_idx<npairs)
    {
      pair = pairs[pair_idx];
      for (jj=0; jj<batch_size; jj+=blockDim.x)
      {
        idx_a = pair.a * batch_size + threadIdx.x + jj;
        idx_b = pair.b * batch_size + threadIdx.x + jj;
        out_idx = pair_idx * batch_size + threadIdx.x + jj;
        out[out_idx] = cuCmulf(in[idx_a],cuConjf(in[idx_b]));
      }
    }
  }
}

int mopsr_multiply_baselines(cuFloatComplex * out, cuFloatComplex * in,
                             mopsr_baseline_t * pairs, unsigned batch_size,
                             unsigned npairs, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock; // get max threads
  int nblocks = props.maxGridSize[0]; // get max blocks
  if (batch_size<nthreads)
    nthreads = batch_size;
  if (npairs<nblocks)
    nblocks = npairs;
  
  multiply_baselines_kernel<<< nblocks, nthreads, 0, stream >>>(out, in, pairs, batch_size, npairs);
#if _GDEBUG
  check_error_stream( "mopsr_multiply_baselines", stream);
#endif
  // sync and error check?
  return 0;
}

__global__
void static_delay_kernel(cuFloatComplex * in, float * out, float * out_errors,
                         unsigned npairs, unsigned batch_size)
{
  // Quick and dirty implementation
  // With some thought, this could be converted to a parallel reduction
  // each thread does the max calculation for batch_size points
  unsigned pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned idx;
  unsigned max_pos=0;
  float max_val=0;
  float val;
  int x1_bin,x2_bin,x3_bin;
  float x1,x2,x3;
  float y1,y2,y3;
  cuFloatComplex tmp;
  float mean;
  float std;
  float sn;
  float variance;
  float sum = 0.0;
  float sq_sum = 0.0;
  double scaling;

  if (pair_idx < npairs)
  {
    idx = pair_idx * batch_size;
    tmp = in[idx];
    scaling = tmp.x*tmp.x + tmp.y*tmp.y;

    for (int ii=0;ii<batch_size;ii++)
    {
      idx = pair_idx * batch_size + ii;
      tmp = in[idx];
      val = tmp.x*tmp.x + tmp.y*tmp.y;
      val /= scaling;

      sum += val;
      sq_sum += val*val;

      if (val>max_val)
      {
        max_pos = ii;
        max_val = val;
      }
    }
      
    //float n = (float) batch_size-1;
    int n = batch_size-1;
    sum -= max_val;
    sq_sum -= max_val*max_val;
    mean = sum / n;
    variance = sq_sum/n - mean*mean;
    std = sqrt(variance);
    
    sn = (max_val-mean)/std;

    x1_bin = (max_pos-1)%batch_size;
    x2_bin =  max_pos;
    x3_bin = (max_pos+1)%batch_size;

    x1 = max_pos-1;
    x2 = max_pos;
    x3 = max_pos+1;
        
    y1 = (float) cuCabsf(in[pair_idx * batch_size + x1_bin]);
    y2 = (float) cuCabsf(in[pair_idx * batch_size + x2_bin]);
    y3 = (float) cuCabsf(in[pair_idx * batch_size + x3_bin]);

    float denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    float A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    float B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
    //float C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
    //printf("thread %d: %.4f\n",threadIdx.x,-B / (2*A));

    //float xv = -B / (2*A);
    float xv = fmodf((-B / (2*A) + batch_size/2), (float) batch_size) - batch_size/2;
    //yv = C - B*B / (4*A);
    //out[pair_idx] = fmodf((-batch_size/2 + xv) - batch_size/2,batch_size);
    out[pair_idx] = xv;
    out_errors[pair_idx] = sn;
    //out[pair_idx] = sq_sum/n;
    //out_errors[pair_idx] = mean*mean;
  }
}

int mopsr_static_delay(cuFloatComplex * in, float * out, float * out_errors,
                       unsigned npairs, unsigned batch_size,
                       cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  int nblocks = npairs/nthreads + 1;
  static_delay_kernel<<<nblocks,nthreads,0,stream>>>(in, out, out_errors, npairs, batch_size);
#if _GDEBUG
  check_error_stream( "static_delay_kernel", stream);
#endif
  return 0;
}

__global__
void accumulate_bandpass_kernel(cuFloatComplex* in, cuFloatComplex* accumulator,
                                unsigned nant, unsigned nsamps,
                                unsigned batch_size)
{
  // in data should be in ST order
  //each block sums all batches for a given antenna
  unsigned ant = blockIdx.x;
  unsigned offset = ant * nsamps; 
  unsigned pos = threadIdx.x;
  unsigned out_idx = ant * batch_size + pos;
  cuFloatComplex val = make_cuFloatComplex(0.0,0.0);
  //cuFloatComplex inval;
  //float fft_scale = (float) batch_size;

  // loop over all time samples for given antenna
  for (int ii=pos; ii<nsamps; ii+=batch_size)
  {
    // loop over batch_size to account for the condition batch_size > blockDim.x
    for (int jj=0; jj<batch_size; jj+=blockDim.x)
      //inval = in[ offset + ii + jj];
      //inval.x /= fft_scale;
      //inval.y /= fft_scale;
      //val = cuCaddf( val, inval);
      val = cuCaddf( val, in[ offset + ii + jj] );
  }
  // add result to the accumulator array
  accumulator[out_idx] = cuCaddf(val, accumulator[out_idx]);
}

int mopsr_accumulate_bandpass(cuFloatComplex* in, cuFloatComplex* accumulator,
                              unsigned nant, unsigned nsamps, 
                              unsigned batch_size, cudaStream_t stream)
{
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  int nthreads = props.maxThreadsPerBlock;
  if (batch_size < nthreads)
    nthreads = batch_size;

#ifdef _GDEBUG
  fprintf (stderr, "mopsr_accumulate_bandpass: nant=%u, nsamps=%u, batch_size=%u\n", nant, nsamps, batch_size);
  fprintf (stderr, "mopsr_accumulate_bandpass: nblocks=%u, nthreads=%d\n", nant, nthreads);
#endif

  accumulate_bandpass_kernel<<< nant, nthreads, 0, stream>>>(in, accumulator, nant, nsamps, batch_size);

#if _GDEBUG
  check_error_stream("accumulate_bandpass_kernel", stream);
#endif

  return 0;
}

__global__ void byte_to_float_kernel (const char * input, float * output, uint64_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    unsigned ii;
    for (ii=idx; ii<size; ii+=gridDim.x*blockDim.x)
    {
      output[ii] = ((float) input[ii]) / 127.0;
    }
  }
}

// convert complex 8-bit input to 32-bit 
int mopsr_byte_to_float (int16_t * input, cuFloatComplex * output, unsigned nsamp, unsigned nant, unsigned nchan, cudaStream_t stream)
{

  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int nthreads = props.maxThreadsPerBlock;

  const unsigned ndim = 2;
  uint64_t size = uint64_t(nsamp) * nant * nchan * ndim;
  int max_blocks = props.maxGridSize[0];
  int nblocks;
  if (size/nthreads > max_blocks)
    nblocks = max_blocks;
  else
    nblocks = size/nthreads;
#ifdef _GDEBUG
  fprintf (stderr, "mopsr_byte_to_float: nsamp=%lu nant=%u nchan=%u\n", nsamp, nant, nchan);
  fprintf (stderr, "mopsr_byte_to_float: blocks=(%d,%d,%d) nthreads=%d\n", blocks.x, blocks.y, blocks.z, nthreads);
#endif
  byte_to_float_kernel<<<nblocks,nthreads,0,stream>>>((const char*) input, (float *)output, size);
  
#if _GDEBUG
  check_error_stream("byte_to_float_kernel", stream);
#endif
  return 0;
}


__global__ void transpose_TS_to_ST_kernel (const int16_t * input, cuFloatComplex * output,
                                           const uint64_t nsamp, const unsigned nant,
                                           const uint64_t nval, const unsigned nval_per_thread,
                                           const unsigned nsamp_per_block)
{
  extern __shared__ int16_t sdata[];
  const unsigned warp_num   = threadIdx.x / WARP_SIZE;
  const unsigned warp_idx   = threadIdx.x % WARP_SIZE;

  const unsigned offset = (warp_num * (WARP_SIZE * nval_per_thread)) + warp_idx;

  unsigned in_idx  = (blockIdx.x * blockDim.x * nval_per_thread) + offset;
  unsigned sin_idx = offset;

  // for use in access each 8bit value
  int8_t * sdata8 = (int8_t *) sdata;
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

  // our thread number within the warp [0-32], also the time sample this will write each time
  const unsigned isamp = warp_idx;

  unsigned iant = warp_num * nval_per_thread;

  unsigned sout_idx = (isamp * nant) + iant;

  //                   block offset isamp              warp offset      thread offset
  uint64_t out_idx  = (blockIdx.x * nsamp_per_block) + (nsamp * iant) + isamp;
  float imag, real;

  for (ival=0; ival<nval_per_thread; ival++)
  {
    imag = (float) sdata8[2*sout_idx];
    real = (float) sdata8[2*sout_idx + 1];

    //if (out_idx < nval * nval_per_thread)
    output[out_idx] = make_cuFloatComplex(real,imag);

    // update the output index 
    out_idx += nsamp;

    sout_idx++;
  }
}

int mopsr_transpose_TS_to_ST(void * d_in, void * d_out, uint64_t nbytes, 
                             unsigned nant, cudaStream_t stream)
{
  const unsigned ndim = 2;
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props,0);
  unsigned nthread = props.maxThreadsPerBlock;

  // since we want a warp of 32 threads to write out just 1 chunk
  const unsigned nsamp_per_block = WARP_SIZE;
  const unsigned nval_per_block  = nsamp_per_block * nant;
  
  // special case where not a clean multiple [TODO validate this!]
  if (nval_per_block % nthread)
  {
    unsigned numerator = nval_per_block;
    while ( numerator > nthread )
      numerator /= 2;
    nthread = numerator;
  }

  unsigned nval_per_thread = nval_per_block / nthread;
  
  const uint64_t nsamp = nbytes / (ndim * nant);
  // the total number of values we have to process is 
  const uint64_t nval = nbytes / (ndim * nval_per_thread);
  int nblocks = nval / nthread;
  if (nval % nthread)
    nblocks++;
  const size_t sdata_bytes = nthread * ndim * nval_per_thread + (2 * nant);

#ifdef _GDEBUG
  fprintf (stderr, "mopsr_transpose_TS_to_ST: nthread=%u\n", nthread);
  fprintf (stderr, "mopsr_transpose_TS_to_ST: nsamp=%lu, nant=%u, nval=%lu\n", nsamp, nant, nval);
  fprintf (stderr, "nsamp_per_block=%u  nval_per_block=%u nval_per_thread=%u\n", nsamp_per_block, nval_per_block, nval_per_thread);
  fprintf (stderr, "nblocks=%d, sdata_bytes=%ld\n", nblocks, sdata_bytes);
#endif

  transpose_TS_to_ST_kernel<<<nblocks,nthread,sdata_bytes,stream>>>
    ((int16_t *) d_in, (cuFloatComplex *) d_out, nsamp, nant, nval, nval_per_thread, nsamp_per_block);

#if _GDEBUG
  check_error_stream("transpose_TS_to_ST_kernel", stream);
#endif

  return 0;
}

// Perform compute the SK estimator for each block of M samples, zapping them if
// they exceed the threshold. 
__global__ void mopsr_skzap_kernel (cuFloatComplex * in, const uint64_t ndat, float M_fac, float sk_lower, float sk_upper)
{
  extern __shared__ float sdata_sk[];

  const unsigned M = blockDim.x;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned s1 = (threadIdx.x*2);
  const unsigned s2 = (threadIdx.x*2) + 1;

  cuFloatComplex val;
  if (i < ndat)
    val = in[i];

  const float power = (val.x * val.x) + (val.y * val.y);
  sdata_sk[s1] = power;
  sdata_sk[s2] = power * power;

  __syncthreads();

  int last_offset = blockDim.x/2 + blockDim.x % 2;

  for (int offset = blockDim.x/2; offset > 0;  offset >>= 1)
  {
    // add a partial sum upstream to our own
    if (threadIdx.x < offset)
    {
      sdata_sk[s1] += sdata_sk[s1 + (2*offset)];
      sdata_sk[s2] += sdata_sk[s2 + (2*offset)];
    }

    __syncthreads();

    // special case for non power of 2 reductions
    if ((last_offset % 2) && (last_offset > 2) && (threadIdx.x == offset))
    {
      sdata_sk[0] += sdata_sk[s1 + (2*offset)];
      sdata_sk[1] += sdata_sk[s2 + (2*offset)];
    }

    last_offset = offset;

    // wait until all threads in the block have updated their partial sums
    __syncthreads();
  }

  // all threads read the S1 and S2 sums
  const float S1 = sdata_sk[0];
  const float S2 = sdata_sk[1];
  const float SK_estimate = M_fac * (M * (S2 / (S1 * S1)) - 1);

  if ((i < ndat) && ((SK_estimate > sk_upper) || (SK_estimate < sk_lower)))
  {
    in[i] = make_cuFloatComplex(0.0,0.0);
  }
}

//
// Compute the S1 and S2 sums for blocks of input data, writing the S1 and S2 sums out to Gmem
//
__global__ void mopsr_skcompute_kernel (cuFloatComplex * in, cuFloatComplex * sums, const unsigned nval_per_thread, const uint64_t ndat, unsigned iant)
{
  extern __shared__ float sdata_skc[];

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
      val = in[idx];
      //if ((iant == 3) && (blockIdx.x == 0))
      //  printf ("%u %f %f\n", idx, val.x, val.y);
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
    sums[blockIdx.x].x = sdata_skc[0];
    sums[blockIdx.x].y = sdata_skc[1];
    //if ((iant == 3) && (blockIdx.x == 0))
    //  printf ("SUM %f %f\n", sdata_skc[0], sdata_skc[1]);
  }
}

//__inline__ __device__ int warpReduceSum (int val) 
//{
//  for (int offset = warpSize/2; offset > 0; offset /= 2) 
//    val += __shfl_down(val, offset);
//  return val;
//}

//
// take the S1 and S2 values in sums.x and sums.y that were computed from M samples, and integrate of nsums blocks to 
// compute a sk mask and zap
//
__global__ void mopsr_skmask_kernel (float * in, cuFloatComplex * sums, unsigned nsums, unsigned M, unsigned nval_per_thread, unsigned nsamp_per_thread, unsigned iant)
{
  // Pearson Type IV SK limits for 3sigma RFI rejection, based on 2^index
  //const unsigned sk_idx_max = 20;

  const float sk_low[20]  = { 0, 0, 0, 0, 0, 
                              0.387702, 0.492078, 0.601904, 0.698159, 0.775046, 
                              0.834186, 0.878879, 0.912209, 0.936770, 0.954684, 
                              0.967644, 0.976961, 0.983628, 0.988382, 0.991764 };
  const float sk_high[20] = { 0, 0, 0, 0, 0, 
                              2.731480, 2.166000, 1.762970, 1.495970, 1.325420, 
                              1.216950, 1.146930, 1.100750, 1.069730, 1.048570, 
                              1.033980, 1.023850, 1.016780, 1.011820, 1.008340 };
/*
  // 4 sigma limits
  const float sk_low[20]  = { 0, 0, 0, 0, 0,
                              0.274561, 0.363869, 0.492029, 0.613738, 0.711612,
                              0.786484, 0.843084, 0.885557, 0.917123, 0.940341,
                              0.957257, 0.969486, 0.978275, 0.984562, 0.989046 };
  const float sk_high[20] = { 0, 0, 0, 0, 0,
                              4.27587, 3.11001, 2.29104, 1.784, 1.48684,
                              1.31218, 1.20603, 1.13893, 1.0951, 1.06577,
                              1.0458, 1.03204, 1.02249, 1.01582, 1.01115 };
*/
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
  unsigned sk_idx_max = 20;

  unsigned idx = threadIdx.x;
  for (unsigned ival=0; ival<nval_per_thread; ival++)
  {
    if (idx < nsums)
    {
      for (unsigned sk_idx = log2_M; sk_idx < sk_idx_max; sk_idx ++)
      {
        unsigned powers_to_add = sk_idx - log2_M;
        unsigned to_add = (unsigned) exp2f(powers_to_add);

        //if ((iant == 0) && (threadIdx.x == 0))
        //{
        //  printf ("[%d] sk_idx=%u powers_to_add=%u to_add=%u\n", ival, sk_idx, powers_to_add, to_add);
        //}

        //printf("to_add=%u\n", to_add); 
        if (idx + to_add <= nsums)
        {
          //float s1 = sums[idx].x;
          //float s2 = sums[idx].y;

          const float m = M * to_add;
          const float m_fac = (m + 1) / (m - 1);

          //if ((iant == 0) && (threadIdx.x == 0))
          //  printf ("[%d] sums[%d] = (%f, %f)\n", ival, idx, s1, s2);

          float s1 = 0;
          float s2 = 0;

          for (unsigned ichunk=0; ichunk < to_add; ichunk++)
          {
            s1 += sums[idx + ichunk].x;
            s2 += sums[idx + ichunk].y;
            //if ((iant == 0) && (threadIdx.x == 0))
            //  printf ("[%d] sums[%d] = (%f, %f)\n", ival, idx+ichunk, sums[idx + ichunk].x, sums[idx + ichunk].y);
          }

          float sk_estimate = m_fac * (m * (s2 / (s1 * s1)) - 1);
          //if ((iant == 0) && (threadIdx.x == 0))
          //  printf ("[%d] total = (%f, %f), m=%f, sk=%f\n", ival, s1, s2, m, sk_estimate);

          //if (threadIdx.x == 0)
          //{
          //  printf ("ival=%d idx=%d s1=%e s2=%e sk=%f\n", ival, idx, s1, s2, sk_estimate);
          //}
          if ((sk_estimate < sk_low[sk_idx]) || (sk_estimate > sk_high[sk_idx]))
          {
            //if (iant == 0)
            //  printf ("[%d][%d] s1=%e s2=%e sk_estimate=%e\n", 
            //          iant, idx, s1, s2, sk_estimate);
            for (unsigned ichunk=0; ichunk < to_add; ichunk++)
            {
              //if (iant == 0)
              //  printf ("MASK: ant=%u block=%d\n", iant, idx+ichunk);
              smask[idx+ichunk] = 1;
            }
          }
        }
      }
      idx += blockDim.x;
    }
  }

  // sync here to be sure the smask is now updated
  __syncthreads();

  // now we want to zap all blocks of input that have an associated mask 
  // note that this kernel has only 1 block, with blockDim.x threads that may not match
  float * indat = in;
  nsamp_per_thread *= 2;

  for (unsigned isum=0; isum<nsums; isum++)
  {
    if (smask[isum] == 1)
    {
      //if ((iant == 0) && (threadIdx.x == 0))
      //  printf ("zapping chunk %d\n", isum);
      unsigned idx = threadIdx.x;
      for (unsigned isamp=0; isamp<nsamp_per_thread; isamp++)
      {
        if (idx < nsums)
        { 
          indat[idx] = 0;
          idx += blockDim.x;
        }
      }
    }
    indat += 2 * M;
  }
}

//
// relies on ST ordering of the data
//
void mopsr_skzap (float * in, uint64_t nbytes , unsigned nant, unsigned tscrunch, 
                  float sk_lower, float sk_upper, cudaStream_t stream)
{
  //printf ("mopsr_skzap (%p, %lu, %u, %u, %f, %f)\n", in, nbytes, nant, tscrunch, sk_lower, sk_upper);

  const float M = (float) tscrunch;
  const float M_fac = (M+1) / (M-1);
  const unsigned ndim = 2;

  uint64_t ndat = nbytes / (nant * ndim * sizeof(float));
  unsigned block_size = tscrunch;
  uint64_t nblocks  = ndat / block_size;
  uint64_t ndat_proc = nblocks * block_size;

  //printf ("mopsr_skzap: ndat=%u ndat_proc=%lu block_size=%u nblocks=%lu\n", ndat, ndat_proc, block_size, nblocks);
  size_t shm_bytes = block_size * ndim * sizeof(float);

  unsigned iant;
  float * indat = in;

  for (iant=0; iant<nant; iant++)
  {
    // foreach block reduce to S1, S2 sums [out of place]
    //printf ("mopsr_skzap: iant=%d offset=%u M=%f M_fac=%f\n", iant, (iant * ndat * ndim), M, M_fac);
    mopsr_skzap_kernel<<<nblocks,block_size,shm_bytes, stream>>> ((cuFloatComplex *) indat, ndat_proc, M_fac, sk_lower, sk_upper);
    
#ifdef _GDEBUG
    check_error_stream ("mopsr_skzap", stream);
#endif
    indat += (ndat * ndim);
  }
}

//
// relies on ST ordering of the data
//
void mopsr_skzap2 (float * in, void ** work_buffer, size_t * work_buffer_size, 
                   uint64_t nbytes , unsigned nant, unsigned tscrunch,
                   cudaStream_t stream)
{
#ifdef _GDEBUG
  fprintf (stderr, "mopsr_skzap2 (%p, %p, %ld, %lu, %u, %u)\n", in, *work_buffer, *work_buffer_size, nbytes, nant, tscrunch);
#endif

  unsigned nthreads = 1024;
  unsigned nval_per_thread = 1;
  if (tscrunch > nthreads)
    nval_per_thread = tscrunch / nthreads;
  else
    nthreads = tscrunch;

  // each block is a single integration  
  const unsigned ndim = 2;
  uint64_t ndat = nbytes / (nant * ndim * sizeof(float));
  uint64_t nblocks = ndat / tscrunch;
  size_t shm_bytes = tscrunch * ndim * sizeof(float);
  size_t bytes_req = nblocks * 2 * sizeof(float);

#ifdef _GDEBUG
  fprintf (stderr, "mopsr_skzap2: work_buffer_size=%ld bytes_req=%ld\n", *work_buffer_size, bytes_req);
#endif
  if (*work_buffer_size < bytes_req)
  {
    if (*work_buffer != NULL) 
    {
#ifdef _GDEBUG
      fprintf (stderr, "freeing work_buffer\n");
#endif
      cudaFree (*work_buffer);
    }
    cudaMalloc (work_buffer, bytes_req);
#ifdef _GDEBUG
    fprintf (stderr, "mopsr_skzap2: allocated %ld bytes, ptr=%p\n", bytes_req, *work_buffer);
#endif
    *work_buffer_size = bytes_req;
  }

#ifdef _GDEBUG
  fprintf (stderr, "ndat=%lu\n", ndat);
#endif

  unsigned nthread_mask = 1024;
  unsigned nval_per_thread_mask = 1;
  if (nblocks > nthread_mask)
  {
    nval_per_thread_mask = nblocks / nthread_mask;
    if (nblocks % nthread_mask)
      nval_per_thread_mask++;
  }
  else
    nthread_mask = nblocks;
  unsigned shm_bytes_mask = nblocks;
  unsigned nsamp_per_thread_mask = tscrunch / nthread_mask;
  if (tscrunch % nthread_mask)
    nsamp_per_thread_mask++;

  unsigned iant;
  float * indat = in;
  for (iant=0; iant<nant; iant++)
  {
    // foreach block reduce to S1, S2 sums [out of place]
#ifdef _GDEBUG
    fprintf (stderr, "nblocks=%u, nthreads=%u, shm_bytes=%u nval_per_thread=%u ndat=%u work_buffer=%p\n",
                      nblocks, nthreads, shm_bytes, nval_per_thread, ndat, *work_buffer);
#endif
    mopsr_skcompute_kernel<<<nblocks, nthreads, shm_bytes, stream>>>((cuFloatComplex *) indat, (cuFloatComplex *) *work_buffer, nval_per_thread, ndat, iant);

#ifdef _GDEBUG
    check_error_stream ("mopsr_skcompute_kernel", stream);
#endif

#ifdef _GDEBUG
    fprintf (stderr, "nthread_mask=%u shm_bytes_mask=%u nval_per_thread_mask=%u nsamp_per_thread_mask=%u\n",
                      nthread_mask, shm_bytes_mask, nval_per_thread_mask, nsamp_per_thread_mask);
#endif
    mopsr_skmask_kernel<<<1, nthread_mask, shm_bytes_mask, stream>>>(indat, (cuFloatComplex *) *work_buffer, nblocks, tscrunch, nval_per_thread_mask, nsamp_per_thread_mask, iant);
#ifdef _GDEBUG
    check_error_stream ("mopsr_skmask_kernel", stream);
#endif
    indat += (ndat * ndim);
  }
}
