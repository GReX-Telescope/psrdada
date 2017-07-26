
#ifndef __MOPSR_DELAYS_CUDA_HIRES_H
#define __MOPSR_DELAYS_CUDA_HIRES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mopsr_cuda.h"
#include "mopsr_delays_hires.h"
#include <cuda_runtime.h>

//#define USE_CONSTANT_MEMORY
#define MOPSR_MEMORY_BLOCKS 16

typedef struct {

  // buffer on GPU where samples reside
  void * d_buffer;

  size_t counter_size;
  size_t counter_bytes;

  // current sample offsets
  char * h_base;
  unsigned * h_out_from;
  unsigned * h_in_from;
  unsigned * h_in_to;
 
  unsigned * h_off;
  unsigned * h_delays;

  char * d_base;
  unsigned * d_out_from;
  unsigned * d_in_from;
  unsigned * d_in_to;

  uint64_t utc_offset; 

} transpose_delay_hires_buf_t;

typedef struct {

  unsigned nchan;
  unsigned nant;
  unsigned ntap;
  unsigned half_ntap;

  void * d_in;
  size_t buffer_size;

  transpose_delay_hires_buf_t * curr;
  transpose_delay_hires_buf_t * next;

  char first_kernel;

} transpose_delay_hires_t;

int hires_transpose_delay_alloc (transpose_delay_hires_t * ctx, size_t nbytes, 
                                 unsigned nchan, unsigned nant, unsigned ntap);

int hires_transpose_delay_dealloc (transpose_delay_hires_t * ctx);

int hires_transpose_delay_buf_alloc (transpose_delay_hires_buf_t * buf, 
                                     size_t buffer_size, size_t counter_size);

int hires_transpose_delay_buf_dealloc (transpose_delay_hires_buf_t * ctx);

void hires_transpose_delay_reset (transpose_delay_hires_t * buf);

int hires_tranpose_delay_sync (cudaStream_t stream, 
                               transpose_delay_hires_buf_t * ctx, uint64_t nbytes, 
                               mopsr_delay_t ** delays);

void * hires_transpose_delay (cudaStream_t stream, transpose_delay_hires_t * ctx, 
                              void * d_in, uint64_t nbytes, mopsr_delay_hires_t ** delays);

#ifdef USE_CONSTANT_MEMORY
void hires_delay_copy_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes);
#endif

void hires_delay_fractional (cudaStream_t stream, void * d_in, void * d_out, 
                             float * d_delays, float * d_fir_coeffs, 
#ifdef USE_CONSTANT_MEMORY
                             float * h_fringes, size_t fringes_size,
#else
                             void * d_fringes, void * d_ant_scales,
#endif
                             uint64_t nbytes, unsigned nchan, unsigned nant, unsigned ntap);

void hires_init_rng (cudaStream_t stream, unsigned long long seed, unsigned nrngs, unsigned pfb_idx, unsigned npfb, void * states);

void hires_init_rng_sparse (cudaStream_t stream, unsigned long long seed, unsigned nrngs, unsigned pfb_idx, unsigned npfb, void * states);

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
                             uint64_t nbytes, unsigned nchan,
                             unsigned nant, unsigned ntap,
                             unsigned s1_memory, uint64_t s1_count, char replace_noise);

void hires_fringe_rotate (cudaStream_t stream, void * d_in,
#ifdef USE_CONSTANT_MEMORY
                          float * h_fringes, size_t fringes_size,
#else
                          void * d_fringes, void * d_ant_scales,
#endif
                          uint64_t nbytes, unsigned nchan, unsigned nant);

size_t hires_curandState_size ();

void hires_test_skcompute (cudaStream_t stream, void * d_in, void * d_s1s_out, void * d_s2s_out, unsigned nchan, unsigned nant, unsigned nbytes);
void hires_test_compute_power_limits (cudaStream_t stream, void * d_s1s, void * d_sigmas, void * d_thresh, void * d_mask,
                              unsigned nsums, unsigned nant, unsigned nchan, uint64_t ndat,
                              uint64_t s1_count, unsigned s1_memory, void * d_rstates);

void hires_test_skdetect (cudaStream_t stream, void * d_s1s, void * d_s2s, void * d_thesh, void * d_mask, void * d_sigmas, unsigned nsums, unsigned nant, unsigned nchan, uint64_t ndat);
void hires_test_skmask (cudaStream_t stream, void * d_in, void * d_out, void * d_mask, void * d_rstates, void * d_sigmas, 
#ifndef USE_CONSTANT_MEMORY
                        void * d_ant_scales,
#endif
                        unsigned nsums, unsigned nchan, unsigned nant, uint64_t ndat, char replace_noise);

void test_merge_sort (cudaStream_t stream, float * d_key_out, float * d_key_in, unsigned length, unsigned direction);
void test_merge_sort2 (cudaStream_t stream, float * d_key_out, float * d_key_in, unsigned length, unsigned direction);


#ifdef __cplusplus
}
#endif

#endif
