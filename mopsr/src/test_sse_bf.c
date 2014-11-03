#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>

#include <inttypes.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <xmmintrin.h>

void convert (int8_t * in, float * out, uint64_t in_data_size, unsigned nant);

int main (int argc, char **argv)
{
  int nant = 4;
  int nsamp = 32;
  int ndim = 2;

  unsigned ninput = ndim*nant*nsamp;
  unsigned noutput = ndim*nsamp;

  int8_t input[ninput];
  float output[noutput];

  memset (output, 0, noutput*4);

  unsigned iant, isamp, idim;
  for (iant=0; iant<nant; iant++)
  {
    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (idim=0; idim<ndim; idim++)
      {
        fprintf (stderr, "[%d][%d][%d][%d] =%u\n", iant, isamp, idim, (iant*nsamp*ndim)+(isamp*ndim) + idim, (int8_t) (iant + isamp + idim));
        input[(iant*nsamp*ndim)+(isamp*ndim) + idim] = (int8_t) (iant + isamp + idim);
      }
    }
  }

  convert (input, output, ninput, nant);

  fprintf (stderr, "conversion done!\n");

  for (isamp=0; isamp<nsamp; isamp++)
  {
    for (idim=0; idim<ndim; idim++)
    {
      fprintf (stdout, "[%u][%u][", isamp, idim);
      for (iant=0; iant<nant; iant++)
      {
        fprintf (stdout, "%u ", input[iant*nsamp*ndim+isamp*ndim+idim]);
      }
      fprintf (stdout, "] == %f\n", output[isamp*ndim + idim]);
    }
  }
}

void convert (int8_t * in, float * out, uint64_t in_data_size, unsigned nant)
{
  float * dest = (float *) out;

  __m128 packed;
  __m64 * parts = (__m64 *) &packed;
  __m128 sum0, sum1, sum2, sum3;
  __m128 unpacked;

  // each vectorized operation will unpack 8 data points
  const uint64_t nops = in_data_size / (nant * 16);
  uint64_t iop;
  unsigned ant_stride_float = nops * 4;

  float * in_op = (float *) in;
  float * src;
  unsigned iant;

  float tmp[4];

  fprintf (stderr, "in_data_size=%"PRIu64", nops=%"PRIu64" ant_stride_float=%u\n", in_data_size, nops, ant_stride_float);

  // simply unpack iant 0 directly to the output
  for (iop=0; iop<nops; iop++)
  {
    src = in_op;
    fprintf (stderr, "iop=%"PRIu64", iant=0 src=%p\n", iop, (void *)src);

    // load 8-bit packed data to register
    packed = _mm_loadu_ps (src);
    parts = (__m64 *) &packed;

    // unpack each 32-bit segment into 128-bit vectors
    sum0 = _mm_cvtpi8_ps (parts[0]);
    sum2 = _mm_cvtpi8_ps (parts[1]);
    packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
    sum1 = _mm_cvtpi8_ps (parts[0]);
    sum3 = _mm_cvtpi8_ps (parts[1]);

    src += ant_stride_float;

    for (iant=1; iant<nant; iant++)
    {
      fprintf (stderr, "iop=%"PRIu64", iant=%u src=%p\n", iop, iant, src);

      packed = _mm_loadu_ps (src);

      sum0 = _mm_add_ps(sum0, _mm_cvtpi8_ps (parts[0]));
      sum2 = _mm_add_ps(sum2,_mm_cvtpi8_ps (parts[1]));
      packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);
      sum1 = _mm_add_ps(sum1,_mm_cvtpi8_ps (parts[0]));
      sum3 = _mm_add_ps(sum3,_mm_cvtpi8_ps (parts[1]));

      src += ant_stride_float;
    }

    _mm_storeu_ps (dest,    sum0);
    _mm_storeu_ps (dest+4,  sum1);
    _mm_storeu_ps (dest+8,  sum2);
    _mm_storeu_ps (dest+12, sum3);

    // increment output by 16 floats
    dest += 16;
    in_op += 4;
  }
}

#if 0
void disabled ()
{
  __m128 packed = _mm_load_ps ((float *) input);

  __m64 * parts = (__m64 *) &packed;

  __m128 unpacked;

  unpacked = _mm_cvtpi8_ps (parts[0]);
  _mm_store_ps (output, unpacked);

  unpacked = _mm_cvtpi8_ps (parts[1]);
  _mm_store_ps (output+8, unpacked);

  packed = (__m128) _mm_srli_epi64 ((__m128i) packed, 32);

  unpacked = _mm_cvtpi8_ps (parts[0]);
  _mm_store_ps (output+4, unpacked);

  unpacked = _mm_cvtpi8_ps (parts[1]);
  _mm_store_ps (output+12, unpacked);
}
#endif
