/***************************************************************************
 *  
 *    Copyright (C) 2014 by Andrew Jameson & Ewan Barr
 *    Licensed under the Academic Free License version 2.1
 * 
 *****************************************************************************/

#ifndef __MOPSR_CALIB_CUDA_H
#define __MOPSR_CALIB_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuComplex.h>

typedef struct {//antenna indexes for a single baseline pair                
  int a;
  int b;
} mopsr_baseline_t;

typedef struct{//formating for writing out binary delays
  int primary_antenna;
  int secondary_antenna;
  float delay;
  float delay_error;
} mopsr_calib_delay_t;

int mopsr_accumulate_cp_spectra(cuFloatComplex * in, cuFloatComplex * out,
				mopsr_baseline_t * pairs,
				unsigned batch_size, unsigned nbatch,
				unsigned npairs, unsigned nsamps,
        unsigned nchan, unsigned nant,
				cudaStream_t stream);
  
int mopsr_byte_to_float (int16_t * input, cuFloatComplex * output,
			unsigned nsamp, unsigned nant, unsigned nchan, float scale, cudaStream_t stream);
  
int mopsr_multiply_baselines(cuFloatComplex * out, cuFloatComplex * in,
                             mopsr_baseline_t * pairs, unsigned batch_size,
                             unsigned npairs, cudaStream_t stream);
  
int mopsr_static_delay(cuFloatComplex * in, float * out, float * out_errors,
                       unsigned npairs, unsigned batch_size,
                       cudaStream_t stream);
  
int mopsr_accumulate_bandpass(cuFloatComplex* in, cuFloatComplex* accumulator,
                              unsigned nant, unsigned nsamps,
                              unsigned batch_size, cudaStream_t stream);

int mopsr_transpose_TS_to_ST(void * d_in, void * d_out, uint64_t nbytes,
                             unsigned nant, cudaStream_t stream);

void mopsr_skzap (float * in, uint64_t nbytes , unsigned nant, unsigned tscrunch,
                   float sk_lower, float sk_upper, cudaStream_t stream);
void mopsr_skzap2 (float * in, void ** work_buffer, size_t * work_buffer_size,
                   uint64_t nbytes , unsigned nant, unsigned tscrunch,
                   cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
