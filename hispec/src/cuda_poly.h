#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_corr( int nchan, int ninp, int ncross, int windowBlocks,
    int nbatch, int prod_type, char *polyMethod,
    float *cuda_inp_buf, float *cuda_window_buf,
    float *cuda_poly_buf, cufftComplex *cuda_ft_buf,
    cufftComplex *cuda_cross_corr, float *cuda_auto_corr,
    float *poly_time, float *fft_time, float *cmac_time );

void gpu_corr_complex( int nchan, int ninp, int ncross, 
    int windowBlocks, int nbatch, 
    int prod_type, char *polyMethod, cufftComplex *cuda_complexinp_buf, 
    float *cuda_window_buf, cufftComplex *cuda_copmlexpoly_buf, 
    cufftComplex *cuda_ft_buf, cufftComplex *cuda_cross_corr, 
    float *cuda_auto_corr,
    float *poly_time, float *fft_time, float *cmac_time );

float elapsed_time(struct timeval *start);

void time_stamp(char *str);

#if USE_DADA
int readDataToGPU(int nchan, int ninp, int windowBlocks, int nbatch, int bits_per_samp, dada_hdu_t *hdu, float *cuda_inp_buf, int debug, int wordtype);
#else
int readDataToGPU(int nchan, int ninp, int windowBlocks, int nbatch, int bits_per_samp, FILE *fpin, float *cuda_inp_buf, int debug, int wordtype);
#endif

int unpackDigitisedDataToGPU(int nchan, int ninp, int windowBlocks, int nbatch, int bits_per_samp, unsigned char *digitised_data, float *cuda_inp_buf, int debug, int wordtype);

void writeGPUOutput(FILE *fout_ac, FILE *fout_cc, int ninp, int nchan, int ncross, int naver, int prod_type, int nbatch, int isLast, float normaliser, int yaxis_size, int rows_per_refresh, cufftComplex *cuda_cross_corr, float *cuda_auto_corr);

/* Function prototypes */
void polyphase_gpu(int ninp, int windowBlocks, int size, int batch,
    char *polyMethod, float *cuda_poly_buf, float *cuda_inp_buf, float *cuda_window_buf);

void complex_polyphase_gpu(int ninp, int windowBlocks, int size, 
    int nbatch, char *polyMethod, 
    cufftComplex *cuda_poly_buf, cufftComplex *cuda_inp_buf, 
    float *cuda_window_buf);

void do_complex_CUFFT(int nchan, int ninp, int nbatch, cufftComplex *cuda_poly_buf, cufftComplex *cuda_ft_buf);

void do_CUFFT(int nchan, int ninp, int nbatch, float *cuda_poly_buf, cufftComplex *cuda_ft_buf);

void do_CUDA_CMAC(int nchan, int ninp, int ncross, int nbatch, int prod_type, cufftComplex *cuda_ft_buf, cufftComplex *cuda_cross_corr, float *cuda_auto_corr);

#ifdef __cplusplus
}
#endif
