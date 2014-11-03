/***************************************************************************
 *  
 *    Copyright (C) 2013 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 *****************************************************************************/

#ifndef __MOPSR_DBCALIB_H
#define __MOPSR_DBCALIB_H

#include <stdlib.h>
#include <stdint.h>
//#include <math.h>
//#include <time.h>
//#include <inttypes.h>
//#include <mopsr_def.h>
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

int mopsr_byte_to_float(char* input, float* output,
                        unsigned size, cudaStream_t stream);

int mopsr_multiply_baselines(cuComplex * out, cuComplex * in,
                             mopsr_baseline_t * pairs, unsigned batch_size,
                             unsigned npairs, cudaStream_t stream);

int mopsr_static_delay(cuComplex * in, float * out, float * out_errors,
                       unsigned npairs, unsigned batch_size,
		       cudaStream_t stream);

int mopsr_accumulate_bandpass(cuComplex* in, cuComplex* accumulator,
                              unsigned nant, unsigned nsamps,
                              unsigned batch_size, cudaStream_t stream);

int mopsr_transpose_TS_to_ST(void * d_in, void * d_out, uint64_t nbytes,
                             unsigned nant, cudaStream_t stream);
  
#endif
