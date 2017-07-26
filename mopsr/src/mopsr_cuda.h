
#ifndef __MOPSR_CUDA_H
#define __MOPSR_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mopsr_delays.h"
#include <cuda_runtime.h>

#define MOPSR_UNIQUE_CORRECTIONS 32
#define MOPSR_MAX_NANT_PER_AQ    16
#define MOPSR_MEDIAN_FILTER_MEMORY 8
//#define EIGHT_BIT_PHASORS

#ifdef __CUDA_ARCH__
      #if (__CUDA_ARCH__ >= 300)
          #define HAVE_SHFL
      #else
          #define NO_SHFL
      #endif
#endif

void mopsr_input_transpose_TFS_to_FST (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant);
void mopsr_input_transpose_TFS_to_FST_hires (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant);
void mopsr_input_transpose_FST_to_STF (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant);
void mopsr_input_transpose_FT_to_TF (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan);

// for fixing the frequency ordering inside a coarse channel
void mopsr_input_rephase (cudaStream_t stream, void * d_in, void * d_corr, uint64_t nbytes, unsigned nchan, unsigned nant);
void mopsr_input_rephase_TFS (cudaStream_t stream, void * d_in, uint64_t nbytes, unsigned nchan, unsigned nant, unsigned chan_offset);
void mopsr_input_rephase_scales (cudaStream_t stream, float * h_ant_scales, size_t nbytes);

void mopsr_input_delay_fractional (cudaStream_t stream, void * d_in, void * d_out, void * d_overlap,
                        float * d_delays, uint64_t nbytes, unsigned nchan, unsigned nant, unsigned ntap);
void mopsr_input_sum_ant (cudaStream_t stream, void * d_in, void * d_out, uint64_t nbytes, unsigned nchan, unsigned nant);

void mopsr_tile_beams_precomp (cudaStream_t stream, void * d_in, void * d_fbs, void * d_phasors, uint64_t bytes, unsigned nbeam, unsigned nant, unsigned nchan);

void mopsr_tie_beam (cudaStream_t stream, void * d_in, void * d_out, void * d_phasors, uint64_t bytes, unsigned nant, unsigned nchan);

void mopsr_mod_beams (cudaStream_t stream, void * d_in, void * d_out, uint64_t bytes, unsigned nant, unsigned nchan, unsigned tdec);


void check_error_stream (const char* method, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
