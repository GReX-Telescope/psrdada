/******************************************************
This is a simple software correlator. The input is a stream of data samples (e.g. bytes)
with one group of bytes for each input per time sample. I.e. for 4 inputs, at time t,
the input would be:
i0i1i2i3 for time t0, then i0i1i2i3 for time t1 etc.
The output is a file of binary float data, real or complex, with adjacent values being the
spectral channels of the correlation products. I.e for channel c, product p the output would be
c0c1...cnchan-1 for p then
c0c1...cnchan-1 for p+1 etc.

Author: Randall Wayth, Oct 2007.

Requires:
- fftw3

to compile:
gcc -Wall -O3 -D_FILE_OFFSET_BITS=64 -o corr_cpu corr_cpu.c -lfftw3f -lm

O3 optimisation gives the fastest code, but is impossible to debug with gdb since 
the compiler inlines everything.
Switch to -O -g optimisation or no optimisation for debugging.

Use:
start the program with no arguments for a summary of command line options.
Command-line options control the number of inputs, number of freq channels,
type of correlation product to write (auto, cross or both),
the number of averages before writing output and the input and output file names.
(stdin,stdout can also be used in some cases)
Debugging/timing output can also be generated with the command line option '-d'.

The code reads blocks of ninp*2*nchan bytes and re-shuffles them into ninp arrays of 2*nchan floats.
(for nchan spectral channels, not including the total power term, one needs 2*nchan values to FFT.)
Each input array is then FFT'd into a complex voltage spectrum. Correlation products (auto and/or cross)
are then generated for each of these spectra and accumulated.

After the specified number of spectra have been added into
the accumulations (the number of averages), they are written to output after being
averaged and normalised for the FFT. In this way, one or more averages can be written per output file.
If the end of the input is reached and nothing has yet been written, one average is written regardless
of the command-line '-a' argument. If one or more averages have already been written and the end of
the input has been reached, then partially accumulated data is discarded.

By default, two output files are created with the suffixes ".LACSPC" and ".LCCSPC" for the
auto and cross correlation products respectively. If only one type of correlation product is generated,
then stdout can be used.
The output format of autocorrelation is binary single precision floats,
one spectrum for each correlation product, per average.
The output is ordered by channel, then product, then average. I.e. for 4 output products with 512 channels,
the data will be channel1, channel2... channel512 all for product 1, then the same for
channel 2 etc. Then the whole thing is repeated again if there are more than 1 averages.
The output format of cross correlation is binary single precision complexes,
one spectrum for each correlation product, per average. Same idea with data ordering.

It is advisable to use input files that are an integer multiple of ninp*2*nchan bytes long, otherwise there
will be unused data at the end of the file. If more than one average put input file is desired,
it is also advisable to use files that are ninp*2*nchan*naver long so that all data is used in an output average.

How many averages do I need to average for t seconds?
- for Nyquist sampled byte data with bandwidth B Hz, there are 2B*n_inputs samples per second.
- for n_channels output, we consume 2*n_chan*n_input samples per input batch.
- therefore B/nchan (batches/sec) gives the number of averages required for 1 second of data.

Input data size:
this code was written for MWA ED data which used a 1-byte sample size. For other data
sizes, the function readData() will need to be modified to read and unpack the
appropriate number of bits.

******************************************************/

#if USE_DADA
#include <ipcio.h>
#include <dada_hdu.h>
#include <dada_def.h>
#include <multilog.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>
#include <complex.h>
#include <fftw3.h>
#include <getopt.h>

#if USE_GPU
#include <cufft.h>
#include "cuda_poly.h"
#endif

#define DEFAULT_NBATCH 16  /* Batch number, used for GPU execution, determines how many blocks of output to be produced at once */
#define DEFAULT_NAV 243712 /* Better to be a multiple of DEFAULT_NBATCH */
#define DEFAULT_NCHAN 512
#define DEFAULT_NINP 1
#define DEFAULT_WINDOW_BLOCK 8
#define SAMPLING_RATE 256 * 1024 * 1024


enum wordtypes {UNSIGNED_BYTE, SIGNED_BYTE, UNSIGNED_4BIT, SIGNED_4BIT};

/* function prototypes */
void perform_forward_fft(float *in, fftwf_complex *out, int size);
void perform_inverse_fft(fftwf_complex *in, float *out, int size);

void make_window(float *window_buf, int ntaps, int windowSize, int windowBlocks, char windowType);

void cpu_corr();
void polyphase_cpu(float **out, float **in, float *window, int size, int windowBlocks, int ninp, int qHead);

void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[]);

int readData(int nchan,int ninp,FILE *fpin,float **inp_buf);
int readDataTail(int nchan, int ninp, FILE *fpin, float **inp_buf, int tail);

#if USE_DADA
int openFiles(key_t dada_key, char *outfilename, int prod_type, dada_hdu_t **hdu, FILE **fout_ac, FILE **fout_cc);
#else
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc);
#endif
void do_FFT_fftw(int nchan, int ninp, float **inp_buf, fftwf_complex **ft_buf);
void setup_FFT_fftw(int nchan, int ninp, float **inp_buf, fftwf_complex **ft_buf);

void do_CMAC(int nchan, int ninp, int prod_type, float **inp_buf, fftwf_complex **ft_buf, fftwf_complex **corr_buf);

void writeOutput(FILE *fout_ac,FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,fftwf_complex **buf, float normaliser);

/* global vars */
int nchan=DEFAULT_NCHAN;/* number of channels in output spectra */
int ninp=DEFAULT_NINP;  /* input of inputs */
int debug=0;            /* turn debugging info on */
int naver=DEFAULT_NAV;  /* number of input spectra to accumulate before writing an average */
int prod_type='A';      /* correlation product type: B: both, C: cross, A: auto */
int wordtype=1,bits_per_samp=0;
char infilename[BUFSIZ], outfilename[BUFSIZ];
static fftwf_plan *plans;
key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

char windowType = 'q';	/* The type of window function for polyphase filter */
int windowBlocks = DEFAULT_WINDOW_BLOCK;  /* Default number of blocks of window */
int windowSize = 0;
char polyMethod[BUFSIZ] = "weighted-overlap-add"; /* Polyphase method */
int complexinput = 0; /* 0 for non-complex input, 1 for complex input */
int yaxis_size = 128;
int rows_per_refresh = 16;
int cuda_device_num = 0;

#ifndef M_PI
double M_PI;
#endif

int main(int argc, char * const argv[]) 
{
  int filedone=0,res=0,i, ncorr,ncross=0,iter=0,nav_written=0;
#if USE_DADA
  dada_hdu_t *hdu = NULL;
#else
  FILE *finp=NULL;
#endif
  FILE *fout_ac = NULL, *fout_cc = NULL;
  float **inp_buf=NULL, normaliser=1.0;
  float read_time=0, fft_time=0, cmac_time=0, write_time=0;
  float poly_time=0;
  fftwf_complex **ft_buf;
  fftwf_complex **corr_buf;
  struct timeval thetime;

  int qHead = 0;

  /* Defining pi, if it is not defined in the maths library */
#ifndef M_PI
  M_PI = acos(-1);
#endif

  float sumOfWindow = 0;
  float *window_buf;
  float **poly_buf;

#if USE_GPU
  /* Variables for CUDA GPU executions */
  int nbatch = DEFAULT_NBATCH;  /* Determine how many data chunks to be processed per step */
  float *cuda_inp_buf;
  cufftComplex *cuda_complexinp_buf;
  float *cuda_window_buf;
  
  float *cuda_poly_buf;	
  cufftComplex *cuda_complexpoly_buf;
  cufftComplex *cuda_ft_buf;
  
  /* Outputs of the correlator, auto and cross separated for more efficient processing */
  float *cuda_auto_corr = NULL;
  cufftComplex *cuda_cross_corr = NULL;

  cudaSetDevice(cuda_device_num);
  /*************************************/
#endif
  
  /* process command line args */
  if (argc < 2) 
    print_usage(argv);
  parse_cmdline(argc,argv);

  //naver = 16;/*SAMPLING_RATE / nchan / 2;*/
  
  ncorr = ninp*(ninp+1)/2;  /* Number of all correlation */
  ncross = ncorr - ninp; /* Number of cross correlation */

  if (debug) {
    fprintf(stderr,"Num inp:\t%d. Num corr products: %d\n",ninp,ncorr);
    fprintf(stderr,"Num chan:\t%d\n",nchan);
    fprintf(stderr,"Correlation product type:\t%c\n",prod_type);
#if USE_DADA
    fprintf(stderr,"DADA key: \t%x\n", dada_key);
#else
    fprintf(stderr,"infile: \t%s\n",infilename);
#endif
    fprintf(stderr,"outfile:\t%s\n",outfilename);
    if( windowType == 'q' )
      fprintf( stderr, "Window type: \tQuadrature Mirror Filter\n" );
    else if( windowType == 'h' )
      fprintf( stderr, "Window type: \tHamming Window Function\n" );
    else
      fprintf( stderr, "Window type: \tunknown\n" );
  }
  
  /* open input and output files */
#if USE_DADA
  openFiles(dada_key, outfilename, prod_type, &hdu, &fout_ac, &fout_cc );
  char *header = 0;
  uint64_t header_size = 0;
  header = ipcbuf_get_next_read( hdu->header_block, &header_size );
  if( !header )
  {
    fprintf( stderr, "header cannot be null\n" );
    exit(1);
  }
  ipcbuf_mark_cleared( hdu->header_block );
#else
  openFiles(infilename,outfilename,prod_type, &finp,&fout_ac,&fout_cc);
#endif

  if( !complexinput )
    windowSize = nchan * 2;
  else
    /* Complex input will need window with half the size */
    windowSize = nchan;
    
  window_buf = (float *)calloc( windowSize * windowBlocks, sizeof(float) );
  make_window( window_buf, 16, windowSize, windowBlocks, windowType );
  
  /* Calculate the sum of window for normalisation, which I (shinkee) don't quite understand */
  for( i = 0; i < windowSize * windowBlocks; i++ )
    sumOfWindow += window_buf[i];
  if(debug)
    fprintf( stderr, "Sum of window: %e\n", sumOfWindow );

  for( i = 0; i < windowSize * windowBlocks; i++ )
  {
    window_buf[i] = window_buf[i] * nchan * 2 / sumOfWindow;
  }
  /* Finish normalising the window */

#if USE_GPU
  /* Verify the nbatch */
  if( nbatch < windowBlocks )
  {
    fprintf( stderr, "Error nbatch: %d\nMust be larger than number of window blocks: %d\n", nbatch, windowBlocks );
    exit(1);
  }
  /* Allocate GPU memory, then initialise or copy data */

  /* This input buffer needs to have a different size 
   * due to the algorithm design, essentially have 
   * nbatch + windowBlocks - 1 blocks of input saved */
  if( !complexinput )
  {
    cudaMalloc( (void **)&cuda_inp_buf, 
       nchan * 2 * ninp * (nbatch + windowBlocks-1) * sizeof(float) );
    cudaMemset( cuda_inp_buf, 0, 
       nchan * 2 * ninp * (nbatch + windowBlocks-1) * sizeof(float) );

    cudaMalloc( (void **)&cuda_poly_buf, nchan * 2 * ninp * nbatch * sizeof(float) );
    cudaMemset( cuda_poly_buf, 0, nchan * 2 * ninp * nbatch * sizeof(float) );

  }
  else
  {
    cudaMalloc( (void **)&cuda_complexinp_buf,
	nchan * ninp * (nbatch + windowBlocks-1) *
	sizeof(cufftComplex) );
    cudaMemset( cuda_complexinp_buf, 0, 
	nchan * ninp * (nbatch + windowBlocks-1) * 
	sizeof(cufftComplex) );

    cudaMalloc( (void **)&cuda_complexpoly_buf, 
	nchan * ninp * nbatch * sizeof(cufftComplex) );
    cudaMemset( cuda_complexpoly_buf, 0, 
	nchan * ninp * nbatch * sizeof(cufftComplex) );
  }
  
  /* Copy the window coefficient */
  cudaMalloc( (void **)&cuda_window_buf, windowSize * windowBlocks * sizeof(float) );
  cudaMemcpy( cuda_window_buf, window_buf, windowSize * windowBlocks * sizeof(float), cudaMemcpyHostToDevice );

  /* Data buffer for the fourier transform */
  cudaMalloc( (void **)&cuda_ft_buf, (nchan+1) * ninp * nbatch * sizeof(cufftComplex) );
  cudaMemset( cuda_ft_buf, 0, (nchan+1) * ninp * nbatch * sizeof(cufftComplex) );

  /* The correlations are 1 size smaller than the fourier transform */
  if( prod_type == 'B' || prod_type == 'A' )
  {
    /* allocate memory for auto correlation output */
    cudaMalloc( (void **)&cuda_auto_corr, (nchan) * ninp * sizeof(float) );
    cudaMemset( cuda_auto_corr, 0, (nchan) * ninp * sizeof(float) );
  }
  
  if( prod_type == 'B' || prod_type == 'C' )
  {
    /* allocate memory for cross correlation output */
    cudaMalloc( (void **)&cuda_cross_corr, (nchan) * ncross * sizeof(cufftComplex) );
    cudaMemset( cuda_cross_corr, 0, (nchan) * ncross * sizeof(cufftComplex) );
  }


#else
  /* Without GPU */
  /* allocate buffers */
  /* for input and FFT results */
  inp_buf = (float **) calloc(ninp,sizeof(float *));
  poly_buf = (float **) calloc( ninp, sizeof(float *) );
  ft_buf = (fftwf_complex **) calloc(ninp,sizeof(fftwf_complex *));
  if (inp_buf == NULL || ft_buf==NULL) {
    fprintf(stderr,"malloc failed\n");
    exit(1);
  }

  for (i=0; i< ninp; i++) {
    inp_buf[i]= (float *) calloc(nchan*2*windowBlocks,sizeof(float));
    ft_buf[i] = (fftwf_complex *) fftwf_malloc((nchan)*sizeof(fftwf_complex));
    poly_buf[i] = (float *) calloc( nchan * 2, sizeof(float) );
  }

  /* for correlation products */
  corr_buf = (fftwf_complex **) calloc(ncorr,sizeof(fftwf_complex *));
  for(i=0; i<ncorr;i++) {
    corr_buf[i] = (fftwf_complex *) fftwf_malloc((nchan)*sizeof(fftwf_complex));
    /* init to zero  */
    memset(corr_buf[i],'\0',(nchan)*sizeof(fftwf_complex));
  }

  /* set up FFT plans. Note because we use FFTW_MEASURE, this actually modifies data in the array, so have to do this
      before reading data */
  setup_FFT_fftw(nchan,ninp,poly_buf,ft_buf);
#endif


  /* process file */
  while (!filedone) {
  
    /* read time chunk into buffers */
    gettimeofday(&thetime,NULL);
    
#if USE_GPU
    /* If using GPU, read data straight into GPU memory */

    if( !complexinput )
    {
#if USE_DADA
      res = readDataToGPU( nchan, ninp, windowBlocks, nbatch, bits_per_samp, hdu, 
	  cuda_inp_buf, debug, wordtype );
#else
      res = readDataToGPU( nchan, ninp, windowBlocks, nbatch, bits_per_samp, finp, 
	  cuda_inp_buf, debug, wordtype );
#endif
    }
    else
    {
#if USE_DADA
      res = readComplexDataToGPU( nchan, ninp, windowBlocks, nbatch, bits_per_samp, hdu, 
	  cuda_complexinp_buf, debug, wordtype );
#else
      res = readComplexDataToGPU( nchan, ninp, windowBlocks, nbatch, bits_per_samp, finp, 
	  cuda_complexinp_buf, debug, wordtype );
#endif
    }
    if( res != 0 )
      filedone = 1;

#else
    /* Read data into CPU memory */
    res = readDataTail(nchan, ninp,finp,inp_buf, 
	(qHead+windowBlocks-1) % windowBlocks);
    if (res !=0) 
      filedone=1;
#endif

    read_time += elapsed_time(&thetime);
    
    if (!filedone) {

#if USE_GPU
      /***************************************************/
      /* GPU correlator execution */
      /***************************************************/
      if( !complexinput )
	gpu_corr( nchan, ninp, ncross, windowBlocks, nbatch, prod_type, 
	    polyMethod, cuda_inp_buf, cuda_window_buf, cuda_poly_buf, 
	    cuda_ft_buf, cuda_cross_corr, cuda_auto_corr,
	    &poly_time, &fft_time, &cmac_time );
      else
	gpu_corr_complex( nchan, ninp, ncross, windowBlocks, 
	    nbatch, prod_type, 
	    polyMethod, cuda_complexinp_buf, cuda_window_buf, 
	    cuda_complexpoly_buf, 
	    cuda_ft_buf, cuda_cross_corr, cuda_auto_corr,
	    &poly_time, &fft_time, &cmac_time );

      /* Only the GPU execution will be batched up */
      iter += nbatch;

#else
      /***************************************************/
      /* CPU execution */
      /***************************************************/
      gettimeofday(&thetime, NULL);
      for( i = 0; i < ninp; i++ )
	memset( poly_buf[i], 0, nchan * 2 * sizeof(float) );
      /* Multiply the data with the window function, polyphase filter */
      polyphase_cpu(poly_buf, inp_buf, window_buf, nchan * 2, windowBlocks, 
	  ninp, qHead);
      poly_time += elapsed_time(&thetime);

     
      /* do the FFT */
      gettimeofday(&thetime,NULL);
      do_FFT_fftw(nchan,ninp,poly_buf,ft_buf);
      fft_time += elapsed_time(&thetime);

      /* do the CMAC */
      gettimeofday(&thetime,NULL);
      do_CMAC(nchan,ninp,prod_type, inp_buf,ft_buf,corr_buf);
      cmac_time += elapsed_time(&thetime);

      iter++;
#endif

    }

    /* write and average if it is time to do so */
    if ((filedone && nav_written==0) || iter >= naver) {
      gettimeofday(&thetime,NULL);
      normaliser = 1.0/(nchan*iter);

#if USE_GPU
      /* write output from GPU */
      writeGPUOutput(fout_ac, fout_cc, ninp, nchan, ncross, iter, 
	  prod_type, nbatch, filedone, normaliser, 
	  yaxis_size, rows_per_refresh, 
	  cuda_cross_corr, cuda_auto_corr);
#else
      /* CPU output writing */
      writeOutput(fout_ac,fout_cc,ninp,nchan,iter,prod_type,corr_buf,normaliser);
#endif

      if(debug) fprintf(stderr,"writing average of %d chunks\n",iter);
      iter=0;
      nav_written++;
      write_time += elapsed_time(&thetime);
    }

    qHead = (qHead+1) % windowBlocks;
  }
  
  if (debug) 
  {
    fprintf( stderr, "wrote %d averages. unused chunks: %d\n", nav_written,iter );
    fprintf( stderr, "Time reading:\t%g ms\n", read_time );
    fprintf( stderr, "Time of polyphase:\t%g ms\n", poly_time );
    fprintf( stderr, "Time FFTing:\t%g ms\n", fft_time );
    fprintf( stderr, "Time CMACing:\t%g ms\n", cmac_time );
    fprintf( stderr, "Time writing:\t%g ms\n", write_time );
    fprintf( stderr, "Total:\t%g ms\n", read_time + poly_time + fft_time + cmac_time + write_time );
  }
  
  /* clean up */
#if USE_DADA
  if( dada_hdu_unlock_read(hdu) < 0 )
  {
    exit(1);
  }
  if( dada_hdu_disconnect(hdu) < 0 )
    exit(1);
#else
  fclose(finp);
#endif
  /* Again, unfreeable memory or uncloseable files for some reason */
  //if(fout_ac !=NULL) fclose(fout_ac);
  //if(fout_cc !=NULL) fclose(fout_cc);
#if USE_GPU
  /* Free GPU memory */
  if( !complexinput )
  {
    cudaFree(cuda_inp_buf);
    cudaFree(cuda_poly_buf);
  }
  else
  {
    cudaFree( cuda_complexinp_buf );
    cudaFree( cuda_complexpoly_buf );
  }
  cudaFree(cuda_ft_buf);
  if( cuda_auto_corr != NULL ) cudaFree(cuda_auto_corr);
  if( cuda_cross_corr != NULL ) cudaFree(cuda_cross_corr);
#else
  for (i=0; i<ninp; i++) {
    if (inp_buf[i]!=NULL) free(inp_buf[i]);
    if (ft_buf[i]!=NULL) fftwf_free(ft_buf[i]);
  }
  free(inp_buf); free(ft_buf);
#endif

  return 0;
}

/* Calculate the polyphase values by multiplying the data 
 * with a window function.
 */
void polyphase_cpu(float **out, float **in, float *window, int size, int windowBlocks, int ninp, int qHead)
{
  int i, j, k;

  for( k = 0; k < ninp; k++ )
  {
    for( i = 0; i < size; i++ )
    {
      for( j = 0; j < windowBlocks; j++ )
      {
	out[k][i] = out[k][i] + window[j * size + i] * in[k][(qHead+j)%windowBlocks*size + i];
      }
    }
  }
}

/* Perform a single FFT, intended to be used very limited times */
void perform_forward_fft(float *in, fftwf_complex *out, int size)
{
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(size, in, out, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

/* Perform a single inverse FFT, intended to be used very limited times */
void perform_inverse_fft(fftwf_complex *in, float *out, int size)
{
  fftwf_plan plan = fftwf_plan_dft_c2r_1d(size, in, out, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

/*
 * Create the window function for FFT
 */
void make_window(float *window_buf, int ntaps, int windowSize, int windowBlocks, char windowType)
{
  int i;

  /* Use the quadrature mirror filter function */
  if( windowType == 'q' )
  {
    /* For now use 16A filter, using more taps will not affect the time performance,
     * it's the number of windowBlocks that will */
    float halftap[] = {0.4810284,0.9779817e-1,
      -0.9039223e-1,-0.9666376e-2,
      0.276414e-1,-0.2589756e-2,
      -0.5054526e-2,0.1050167e-2};
    float fulltap[ntaps];
    fftwf_complex *tapfft;
    fftwf_complex *windowTemp;
    float *temp;
    
    int numPoints = windowSize * windowBlocks;
    int shiftIndex;

    /* Allocate the temporary array for the window function initilized to 0's */
    tapfft = (fftwf_complex *)calloc( ntaps/2 + 1, sizeof(fftwf_complex) );
    windowTemp = (fftwf_complex *)calloc( numPoints/2 + 1, sizeof(fftwf_complex) );
    temp = (float *)malloc( numPoints * sizeof(float) );

    for( i = 0; i < ntaps/2; i++ )
    {
      /* Obtain the full tap by mirroring the half tap */
      fulltap[i] = halftap[ntaps/2-1-i];
      fulltap[ntaps/2+i] = halftap[i];
    }
    
    /* Do FFT on the critically sampled window function */
    perform_forward_fft(fulltap, tapfft, ntaps);

    /* Put the taps into an array with intended size and prepare for inverse fft */     
    memcpy(windowTemp, tapfft, (ntaps/2+1) * sizeof(fftwf_complex));
    perform_inverse_fft(windowTemp, temp, numPoints);

    /* Normalisation, multiplied by numPoints / ntaps for loss of amplitude in 
     * interpolation, the factor numPoints is canceled out by inverse FFTW normalisation
     */
    for( i = 0; i < numPoints; i++ )
      temp[i] = temp[i] / ntaps;

    /* Shift the points to center the peak of the window function */
    shiftIndex = numPoints / ntaps / 2;
    memcpy( window_buf, &temp[numPoints - shiftIndex], shiftIndex * sizeof(float) );
    memcpy( &window_buf[shiftIndex], temp, (numPoints - shiftIndex) * sizeof(float) );

    if( debug )
    {
      FILE *fpwin = fopen( "quadraturewin.csv", "w" );
      for( i = 0; i < numPoints; i++ )
      {
	fprintf( fpwin, "%e\n", window_buf[i] );
      }
      fclose(fpwin);
    }
    
    /* Free up the memory */
    free(temp);
    
    /* FIXME: Bug, not sure why, these complex type variables 
     * cannot be freed 
     */
    //free(tapfft);
    //free(windowTemp);
  }
  /* Use the hamming/hanning window function */
  else if( windowType == 'h' )
  {
    /* hamming window function */
    float *hamming;
    float *sincfunc;
    int M = windowSize * windowBlocks; /* Total number of window points */
    int N = windowSize; /* Intended FFT length */

    /* Quite an inefficient way to do it, but it is easier to plot/debug */
    hamming = (float *)malloc( M * sizeof(float) );
    sincfunc = (float *)malloc( M * sizeof(float) );

    for( i = 0; i < M; i++ )
    {
      hamming[i] = 0.5 - 0.5 * cos(2 * M_PI * i / M);
      if( i == M/2 ) sincfunc[i] = 1;
      else
      {
	float x = (i - M/2) / (float) N;
	sincfunc[i] = sin(M_PI * x) / (M_PI * x);
      }
      window_buf[i] = hamming[i] * sincfunc[i];
    }

    if( debug )
    {
      FILE *fpwin = fopen( "hammingwin.csv", "w" );
      for( i = 0; i < M; i++ )
      {
	fprintf( fpwin, "%e\n", window_buf[i] );
      }
      fclose(fpwin);
    }

    free(hamming);
    free(sincfunc);
  }
}


/* write out correlation products, including DC components
   Apply a normalisation factor that depends on the FFT length and the number
   of averages */
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                fftwf_complex **buf,float normaliser) {
    int inp1,inp2,cprod=0,chan;
    float *temp_buffer=NULL;
    
    temp_buffer=malloc(sizeof(float)*(nchan));
    
    for(inp1=0; inp1<ninp; inp1++) {
        for (inp2=inp1; inp2<ninp; inp2++) {
            /* make an average by dividing by the number of chunks that went into the total */
            for (chan=0; chan<nchan; chan++) {
                buf[cprod][chan] *= normaliser;
                /* convert the autocorrelation numbers into floats, since the imag parts will be zero*/
                if (inp1==inp2 && (prod_type == 'A' || prod_type=='B')){
                    temp_buffer[chan] = creal(buf[cprod][chan]);
                }
            }
            if(inp1==inp2 && (prod_type == 'A' || prod_type=='B')) {
                /* write the auto correlation product */
                fwrite(temp_buffer,sizeof(float),nchan,fout_ac);
            }
            if(inp1!=inp2 && (prod_type == 'C' || prod_type=='B')) {
                /* write the cross correlation product */
                fwrite(buf[cprod],sizeof(fftwf_complex),nchan,fout_cc);
            }

            /* reset the correlation products to zero */
            memset(buf[cprod],'\0',(nchan)*sizeof(fftwf_complex));
            cprod++;
        }
    }
    if (temp_buffer!=NULL) free(temp_buffer);
}


/* accumulate correlation products */
void do_CMAC(int nchan, int ninp, int prod_type, float **inp_buf, fftwf_complex **ft_buf, fftwf_complex **corr_buf) 
{
    int inp1,inp2;
    register int chan,cprod=0;
    
    for(inp1=0; inp1<ninp; inp1++) 
    {
        for(inp2=inp1; inp2<ninp; inp2++) 
	{
            if(prod_type=='B' || ((prod_type=='C' && inp1!=inp2) || (prod_type=='A' && inp1==inp2))) 
	    {
                for(chan=0;chan<nchan;chan++) 
		{
                    corr_buf[cprod][chan] += ft_buf[inp1][chan]*conjf(ft_buf[inp2][chan]);
                }
            }
            cprod++;         
        }
    }    
}


/* make the FFTW execution plans */
/* since each input and output buffer is different, we can't re-use the same plan, which is clunky
   but there's not much we can do about it, short of moving data around unnecesarily. We create one
   plan for each input stream, just once, and use it many times */
void setup_FFT_fftw(int nchan, int ninp, float **inp_buf, fftwf_complex **ft_buf) {
    int i;
    
    plans = (fftwf_plan *) calloc(ninp,sizeof(fftwf_plan));
    for (i=0; i<ninp; i++) {
        plans[i] = fftwf_plan_dft_r2c_1d(nchan*2, inp_buf[i], ft_buf[i],FFTW_MEASURE);
    }
}


/* do the FFT */
void do_FFT_fftw(int nchan, int ninp, float **inp_buf, fftwf_complex **ft_buf) {
    int i;
    
    for (i=0; i<ninp; i++) {
        fftwf_execute(plans[i]);
    }
}


/* read in a batch of input data, typically 2*n_channels*n_inputs samples, convert to float
   and pack into the input buffers for later FFT */
int readData(int nchan,int ninp,FILE *fpin,float **inp_buf) {
    static int init=0,ntoread=0;
    static unsigned char *buffer=NULL;

    int inp,chan,nread;

    if (!init) {

        ntoread = ninp*nchan*2*bits_per_samp/8;
        init=1;
        buffer = (unsigned char *) malloc(ntoread);
        if (debug) {
            fprintf(stderr,"size of read buffer: %d bytes\n",ntoread);
        }
    }

    nread = fread(buffer,ntoread,1,fpin);
    if(nread < 1) return 1;
    
    for(inp=0; inp<ninp; inp++) {
        if(wordtype==0) {
          for(chan=0; chan<nchan*2; chan++) {
              inp_buf[inp][chan] = (float)(buffer[ninp*chan + inp] - 128);
          }
        }
        if(wordtype==1) {
          for(chan=0; chan<nchan*2; chan++) {
              inp_buf[inp][chan] = ((char *)buffer)[ninp*chan + inp];
              //printf("%g ",inp_buf[inp][chan]);
          }
          //printf("\n");
        }
    }
 
    return 0;
}


/* read in a batch of input data, typically 2*n_channels*n_inputs samples, 
 * convert to float and save it to the tail of the queue
 * and pack into the input buffers for later FFT */
int readDataTail(int nchan,int ninp,FILE *fpin, float **inp_buf, int tail) {
    static int init=0,ntoread=0;
    static unsigned char *buffer=NULL;

    int inp,chan,nread;


    if (!init) {

        ntoread = ninp*nchan*2*bits_per_samp/8;
        init=1;
        buffer = (unsigned char *) malloc(ntoread);
        if (debug) {
            fprintf(stderr,"size of read buffer: %d bytes\n",ntoread);
        }
    }

    nread = fread(buffer,1,ntoread,fpin);

    if(nread < 1) return 1;
    
    for(inp=0; inp<ninp; inp++) {
        if(wordtype==0) {
          for(chan=0; chan<nchan * 2; chan++) {
              inp_buf[inp][nchan*2*tail + chan] = (float)(buffer[ninp*chan + inp] - 128);
          }
        }
        if(wordtype==1) {
          for(chan=0; chan<nchan; chan++) {
              inp_buf[inp][nchan*2*tail + chan] = ((char *)buffer)[ninp*chan + inp];
              //printf("%g ",inp_buf[inp][chan]);
          }
          //printf("\n");
        }
    }

    return 0;
}

/* open the input and output files */
#if USE_DADA
int openFiles(key_t dada_key, char *outfilename, int prod_type, dada_hdu_t **hdu, FILE **fout_ac, FILE **fout_cc) {
#else
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc) {
#endif
    char tempfilename[FILENAME_MAX];
#if !USE_DADA
    if (infilename == NULL) {
        fprintf(stderr,"No input file specified\n");
        exit(1);
    }
#endif
    if (outfilename == NULL) {
        fprintf(stderr,"No output file specified\n");
        exit(1);
    }
 
#if !USE_DADA   
    /* sanity check to see if there was a typo on the command line */
    if (infilename[0] != '-' && (strcmp(infilename,outfilename)==0)) {
        fprintf(stderr,"Input and output file names are the same. You really don't want to do that.\n");
        exit(1);
    }
#endif
    
    /* sanity check: can only use stdout for one type of output */
    if((prod_type=='B') && strcmp(outfilename,"-")==0) {
        fprintf(stderr,"Can only use stdout for either auto or cross correlations, not both\n");
        exit(1);
    }
    
#if USE_DADA
    multilog_t *mlog = multilog_open( "corr_poly_gpu_dada", 0 );
    multilog_add( mlog, stderr );
    *hdu = dada_hdu_create(mlog);
    dada_hdu_set_key( *hdu, dada_key);
    if( dada_hdu_connect(*hdu) < 0 ) 
    {
      fprintf( stderr, "dada connect error\n" );
      exit(1);
    }
    if( dada_hdu_lock_read(*hdu) < 0 )
    {
      fprintf( stderr, "dada lock read error\n" );
      exit(1);
    }
#else
    /* check for special file name: "-", which indicates to use stdin/stdout */
    if (strcmp(infilename,"-")==0) {
        *fin = stdin;
    } else {
        *fin = fopen(infilename,"r");
        if (*fin ==NULL) {
            fprintf(stderr,"failed to open input file name: <%s>\n",infilename);
            exit(1);
        }        
    }
#endif
    
    if ((prod_type=='A') && strcmp(outfilename,"-")==0) {
        *fout_ac = stdout;
    } else if ((prod_type=='C') && strcmp(outfilename,"-")==0) {
        *fout_cc = stdout;
    } else {
        if (prod_type=='A' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LACSPC");
            *fout_ac = fopen(tempfilename,"w");
            if (*fout_ac ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
        if (prod_type=='C' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LCCSPC");
            *fout_cc = fopen(tempfilename,"w");
            if (*fout_cc ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
    }
    
    return 0;
}


void parse_cmdline(int argc, char * const argv[]) {
#if USE_DADA
    char optstring[]="dc:k:o:n:a:g:p:w:W:b:m:y:r:z";
#else
    char optstring[]="dc:i:o:n:a:g:p:w:W:b:m:y:r:z";
#endif
    int c;
    
    while ((c=getopt(argc,argv,optstring)) != -1) {
        switch(c) {
            case 'c':
                nchan = atoi(optarg);
                if (nchan <=0 || nchan > 65536) {
                    fprintf(stderr,"bad number of channels: %d\n",nchan);
                    print_usage(argv);
                }
                break;
            case 'n':
                ninp = atoi(optarg);
                if (ninp <=0 || ninp > 128) {
                    fprintf(stderr,"bad number of inputs: %d\n",ninp);
                    print_usage(argv);
                }
                break;
            case 'a':
                naver = atoi(optarg);
                if (naver <=0 || naver > 9999999) {
                    fprintf(stderr,"bad number of averages: %d\n",naver);
                    print_usage(argv);
                }
                break;
            case 'p':
                prod_type = toupper(optarg[0]);
                if (prod_type!='A' && prod_type !='B' && prod_type != 'C') {
                    fprintf(stderr,"bad correlation product type: %c\n",prod_type);
                    print_usage(argv);
                }
                break;
#if USE_DADA
            case 'k':
                if (sscanf(optarg, "%x", &dada_key) != 1)
                {
                  fprintf(stderr, "bad dada key: %s\n", optarg);
                  print_usage(argv);
                }
                break;
#else
            case 'i':
                strcpy(infilename, optarg);
                break;
#endif
            case 'd':
                debug=1;
                break;
	    case 'g':
		cuda_device_num = atoi(optarg);
		break;
            case 'o':
                strcpy(outfilename, optarg);
                break;
            case 'w':
                wordtype=atoi(optarg);
                break;
	    case 'W':
		windowType = optarg[0];
		break;
	    case 'b':
		windowBlocks = atoi(optarg);
		break;
	    case 'm':
		strcpy(polyMethod, optarg);
		break;
	    case 'y':
		yaxis_size = atoi(optarg);
		break;
	    case 'r':
		rows_per_refresh = atoi(optarg);
		break;
	    case 'z':
		complexinput = 1;
		break;
            default:
                fprintf(stderr,"unknown option %c\n",c);
                print_usage(argv);
        }
    }
    if(wordtype==0 || wordtype==1) bits_per_samp = 8;
    if(wordtype==2 || wordtype==3) bits_per_samp = 4;
    if(bits_per_samp != 8) {
      fprintf(stderr,"ERROR: only byte word types currently supported\n");
      exit(1);
    }

    /* Check for the compatibility of the monitoring graph parameters */
    if( yaxis_size < rows_per_refresh || yaxis_size % rows_per_refresh != 0 )
    {
      fprintf( stderr, "Error: yaxis_size: %d, must be larger than and divisible by rows_per_refresh: %d\n", yaxis_size, rows_per_refresh );
      exit(1);
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-p type\t\tspecify correlation product type(s). A: auto, C: cross, B: both. default: %c\n",prod_type);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels. default: %d\n",DEFAULT_NCHAN);
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. detault: %d\n",DEFAULT_NINP);
    fprintf(stderr,"\t-a num\t\tspecify min number of averages before output. Default: %d\n",DEFAULT_NAV);
    fprintf( stderr, "\t-g num\t\tspecify which GPU to be used for processing.\n" );
#if USE_DADA
    fprintf(stderr,"\t-k key\t\tPSRDADA hexidecimal shared memory key [default %x]\n", DADA_DEFAULT_BLOCK_KEY);
#else
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
#endif
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");

    fprintf(stderr,"\t-w wordtype\tspecify type of input data. Default: %d\n",UNSIGNED_BYTE);
    fprintf(stderr,"\t   \t0: byte, unsigned\n");
    fprintf(stderr,"\t   \t1: byte, signed\n");
    fprintf(stderr,"\t   \t2: 4bit, unsigned\n");
    fprintf(stderr,"\t   \t3: 4bit, signed\n");

    fprintf(stderr, "\t-m method\tmethod used in polyphase filtering\n");
    fprintf(stderr, "\t   \toversample-decimate \toversample the data and decimate\n");
    fprintf(stderr, "\t   \tweighted-overlap-add \tsum up the weighted time segments\n");

    fprintf( stderr, "\t-W windowType\ttype of window used for polyphase filtering\n" );
    fprintf( stderr, "\t   \tq: Quadrature Mirror Filter window\n" );
    fprintf( stderr, "\t   \th: Hamming window with sinc function\n" );

    fprintf(stderr,"\t-d \twrite debug and runtime info to stderr\n");
    fprintf( stderr, "\t-y yaxis_size\tspecify the size of y-axis when doing waterfall plotting\n" );
    fprintf( stderr, "\t-r rows_per_refresh\tspecify the stepping size for waterfall plotting\n" );
    fprintf( stderr, "\t-z \texpect complex type input (implementation not yet completed)\n" );
    exit(0);
}

