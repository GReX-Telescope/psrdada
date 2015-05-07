
#ifndef __MOPSR_DELAYS_CUDA_H
#define __MOPSR_DELAYS_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mopsr_cuda.h"
#include "mopsr_delays.h"
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
# if (__CUDA_ARCH__ >= 300)
# define HAVE_CUDA_SHUFFLE 1
# else
# define HAVE_CUDA_SHUFFLE 0
# endif
#else
# define HAVE_CUDA_SHUFFLE 0
#endif

#define MOPSR_MEMORY_BLOCKS 4

typedef struct {

  // buffer on GPU where samples reside
  void * d_buffer;

  size_t counter_size;

  // current sample offsets
  unsigned * h_out_from;
  unsigned * h_in_from;
  unsigned * h_in_to;
 
  unsigned * h_off;
  unsigned * h_delays;
  uint64_t utc_offset; 

} transpose_delay_buf_t;

typedef struct {

  unsigned nchan;
  unsigned nant;
  unsigned ntap;
  unsigned half_ntap;

  void * d_in;
  size_t buffer_size;

  transpose_delay_buf_t * curr;
  transpose_delay_buf_t * next;

  char first_kernel;

} transpose_delay_t;

int mopsr_transpose_delay_alloc (transpose_delay_t * ctx, size_t nbytes, 
                                 unsigned nchan, unsigned nant, unsigned ntap);

int mopsr_transpose_delay_dealloc (transpose_delay_t * ctx);

int mopsr_transpose_delay_buf_alloc (transpose_delay_buf_t * buf, 
                                     size_t buffer_size, size_t counter_size);

int mopsr_transpose_delay_buf_dealloc (transpose_delay_buf_t * ctx);

void mopsr_transpose_delay_reset (transpose_delay_t * buf);

int mopsr_tranpose_delay_sync (cudaStream_t stream, 
                               transpose_delay_buf_t * ctx, uint64_t nbytes, 
                               mopsr_delay_t ** delays);

void * mopsr_transpose_delay (cudaStream_t stream, transpose_delay_t * ctx, 
                              void * d_in, uint64_t nbytes, mopsr_delay_t ** delays);

void mopsr_delay_copy_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes);

void mopsr_delay_fractional (cudaStream_t stream, void * d_in, void * d_out, 
                             float * d_delays, float * h_fringes, float * h_delays_ds, 
                             float * h_fringe_coeffs_ds, size_t fringes_size,
                             uint64_t nbytes, unsigned nchan, unsigned nant, unsigned ntap);

void mopsr_init_rng (cudaStream_t stream, unsigned long long seed, unsigned nrngs, void * states);

void mopsr_delay_fractional_sk_scale (cudaStream_t stream,
                             void * d_in, void * d_out, void * d_fbuf, void * d_rstates, 
                             void * d_sigmas, void * d_mask, float * d_delays, void * d_s1s,
                             void * d_s2s, void * d_thresh, float * h_fringes, float * h_delays_ds, 
                             float * h_fringe_coeffs_ds, size_t fringes_size,
                             uint64_t nbytes, unsigned nchan,
                             unsigned nant, unsigned ntap,
                             unsigned s1_memory, uint64_t s1_count);

void mopsr_fringe_rotate (cudaStream_t stream, void * d_in, float * h_fringes, size_t fringes_size,
                          uint64_t nbytes, unsigned nchan, unsigned nant);

size_t mopsr_curandState_size ();

void mopsr_test_skcompute (cudaStream_t stream, void * d_in, void * d_s1s_out, void * d_s2s_out, unsigned nchan, unsigned nant, unsigned nbytes);
void mopsr_test_compute_power_limits (cudaStream_t stream, void * d_s1s, void * d_thresh,
                              unsigned nsums, unsigned nant, unsigned nchan, uint64_t ndat,
                              uint64_t s1_count, unsigned s1_memory);

void mopsr_test_skdetect (cudaStream_t stream, void * d_s1s, void * d_s2s, void * d_thesh, void * d_mask, void * d_sigmas, unsigned nsums, unsigned nant, unsigned nchan, uint64_t ndat);
void mopsr_test_skmask (cudaStream_t stream, void * d_in, void * d_out, void * d_mask, void * d_rstates, void * d_sigmas, unsigned nsums, unsigned nchan, unsigned nant, uint64_t ndat);

#ifdef __cplusplus
}
#endif

#endif
