#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>

#define NUM_THREADS 2
#define GULP_SIZE 156252

#include "mkl.h"

/* DGESVD prototype
extern void sgesvd_( char* jobu, char* jobvt, int* m, int* n, float* a,
                int* lda, float* s, float* u, int* ldu, float* vt, int* ldvt,
                float* work, int* lwork, int* info );
*/

typedef struct svdzap_data
{
  int tid;
  int n_files;
  char ** file_names;
  unsigned gulp_size;
  int * flags;
  int result;
  int verbose;
} svdzap_data_t;

svdzap_data_t * init_pol_data(unsigned num_files, unsigned gulp_size, unsigned tid, int verbose);

long create_xcorr(float** datagulp, long gulpsize, FILE **currentFile,int numbeams,float ** xcorr_matrix);

//void close_files(FILE **currentFile, int numbeams, int numpol);

int  choose_pivot(int low_index,int high_index);

void swap(float *x,float *y); 

void quicksort(float* list,int low_index,int high_index);

float mean(float* gulp, long gulpsize);

float median(float* gulp, long gulpsize);

float mad(float* gulp,int mode,long gulpsize);

void * svdzap_thread (void *);

double diff_time ( struct timeval time1, struct timeval time2 );

void usage()
{
  fprintf (stdout,
     "dgesvd_aux_p [options] [m+n files]\n"
     " -h         print help text\n"
     " -m <num>   number of pol0 files to expect\n"
     " -n <num>   number of pol1 files to expect\n"
     " -o <file>  write output file mask\n"
     " -v         verbose output\n"
     "\n"
     "  If the output file exists [-o] the RFI mask will be\n"
     "  appened, otherwise a new file will be created. If no\n"
     "  output file is specified, `pwd`/rfi.mask till be used\n");
}

/* Main program */
int main(int argc, char *argv[]) 
{

  /* append flag for output file,
   *  -1 no output file
   *  0  create
   *  1 == append */
  int append = -1;

  int num_pol0 = 0;

  int num_pol1 = 0;

  int verbose = 0;

  unsigned length = 1024;

  char output_file[length];

  int arg = 0;

  sprintf(output_file, "rfi.mask");

  while ((arg=getopt(argc,argv,"hm:n:o:v")) != -1) 
  {
    switch (arg) 
    {

      case 'h':
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'm':
        if (optarg)
          num_pol0 = atoi(optarg);
        else
        {
          fprintf (stderr, "-m requires num arg\n");
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'n':
        if (optarg)
          num_pol1 = atoi(optarg);
        else
        {
          fprintf (stderr, "-n requires num arg\n");
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'o':
        if (optarg)
        {
          sprintf(output_file, "%s", optarg);
          append = 0;
        }
        else
         {
          fprintf (stderr, "-o requires file arg\n");
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return (EXIT_FAILURE);
    }
  }

  int num_files = argc - optind;

  if (num_files != num_pol1 + num_pol0)
  {
    fprintf (stderr, "ERROR: found %d files, expected %d\n", num_files, (num_pol1 + num_pol0));
    return (EXIT_FAILURE);
  }

  // check if the specified file exists
  int mask = R_OK;
  if (access(output_file, mask) < 0)
    append = 0;
  else
    append = 1;

  // check the specified file is writable
  mask = W_OK;
  if (append == 1 && access(output_file, mask) < 0)
  {
    fprintf (stderr, "ERROR: could not write to output file [%s] %s\n", output_file, strerror(errno));
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "output_file = %s\n", output_file);

  unsigned i = 0;

  svdzap_data_t * p0 = 0;
  svdzap_data_t * p1 = 0;
  pthread_t pol0_thread_id = 0;
  pthread_t pol1_thread_id = 0;

  if (num_pol0) 
  {
    p0 = init_pol_data(num_pol0, GULP_SIZE, 0, verbose);
    for (i=0; i<num_pol0; i++)
      p0->file_names[i] = strdup(argv[optind+i]);
  }

  if (num_pol1)
  {
    p1 = init_pol_data(num_pol1, GULP_SIZE, 1, verbose);
    for (i=0; i<num_pol1; i++)
      p1->file_names[i] = strdup(argv[optind+num_pol0+i]);
  }

  // create pthread[s]

  int rval = 0;
  if (num_pol0)
  {
    if (verbose)
      fprintf (stderr, "main: launching pol0 thread with data=%x\n", p0);
    rval = pthread_create (&pol0_thread_id, 0, (void *) svdzap_thread, (void *) p0);
    if (rval != 0) {
      fprintf (stderr, "Error creating pol0_thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }
  }

  if (num_pol1)
  {
    if (verbose)
      fprintf (stderr, "main: launching pol1 thread\n");
    rval = pthread_create (&pol1_thread_id, 0, (void *) svdzap_thread, (void *) p1);
    if (rval != 0) {
      fprintf (stderr, "Error creating pol1_thread: %s\n", strerror(rval));
      return EXIT_FAILURE;
    }
  }
 
  if (verbose)
    fprintf (stderr, "main: waiting for threads to finish\n");

  // join thread[s]
  if (num_pol0)
  {
    void* result = 0;
    pthread_join (pol0_thread_id, &result);
  }

  if (num_pol1)
  {
    void* result = 0;
    pthread_join (pol1_thread_id, &result);
  }

  if (verbose)
    fprintf (stderr, "main: writing output to : %s\n", output_file);

  FILE * fptr = fopen(output_file, "a");
  if (fptr == NULL)
  {
    fprintf(stderr, "could not open output file: %s\n", output_file);
    return EXIT_FAILURE;
  }

  if (append == 0)
    fprintf(fptr, "#0 1\n");

  int value = 0;
  for (i=0; i<GULP_SIZE; i++)
  {
    value = 1;
    if (num_pol0 && p0->result == 0 && p0->flags[i] == 0)
      value = 0; 
    if (num_pol1 && p1->result == 0 && p1->flags[i] == 0)
      value = 0; 

    fprintf(fptr, "%d\n", value );
  }

  fclose(fptr);

  if (num_pol0)
  {
    free(p0->flags);
    free(p0->file_names);
    free(p0);
  }

  if (num_pol1)
  {
    free(p1->flags);
    free(p1->file_names);
    free(p1);
  }

  return EXIT_SUCCESS;
}

svdzap_data_t * init_pol_data(unsigned num_files, unsigned gulp_size, unsigned tid, int verbose)
{

  if (verbose)
    fprintf (stderr, "init_pol_data(%d, %d, %d, %d)\n", num_files, gulp_size, tid, verbose);

  svdzap_data_t * data = (svdzap_data_t *) malloc(sizeof(svdzap_data_t));
  assert(data != 0);

  data->tid = tid;
  data->n_files = num_files;
  data->gulp_size = gulp_size;
  data->verbose = verbose;
  data->result = -1;

  data->file_names = (char **) malloc(sizeof(char *) * num_files);
  assert(data->file_names != 0);

  data->flags = (int *) malloc(sizeof(int) * gulp_size);
  assert(data->flags != 0);

  return data;
}


/*
 *  process 1 polns worth of mon files
 */
void * svdzap_thread (void * arg)
{

  svdzap_data_t * svdzap = (svdzap_data_t *) arg;

  assert(svdzap != 0);

  struct timeval start;
  struct timeval split;
  struct timeval curr;
  gettimeofday (&start, 0);
  gettimeofday (&split, 0);
  gettimeofday (&curr, 0);
  double split_time = 0;
  double total_time = 0;

  if (svdzap->verbose)
    fprintf (stderr, "svdzap_thread[%d]: preparing\n", svdzap->tid);

  FILE ** fptrs = (FILE **) malloc(sizeof(FILE *) * svdzap->n_files);
  assert (fptrs);
  int i = 0;

  // check that all the files can be read
  int mask = R_OK;
  int readable_wait = 10;
  int all_readable = 0;

  while (!all_readable && readable_wait > 0)
  {
    all_readable = 1;
    for (i=0; i<svdzap->n_files; i++)
    {
      if (access(svdzap->file_names[i], mask) < 0)
      {
        fprintf(stderr, "svdzap_thread[%d]: %s not readable, waiting...\n", svdzap->tid, svdzap->file_names[i]);
        all_readable = 0;
      }
      else
      {
        if (svdzap->verbose)
          fprintf(stderr, "svdzap_thread[%d]: %s was readable\n", svdzap->tid, svdzap->file_names[i]);
      }
    }
    if (!all_readable)
    {
      sleep(1);
      readable_wait--;
    }
  }

  for (i=0; i<svdzap->n_files; i++)
  {
    fptrs[i] = fopen(svdzap->file_names[i], "rb");
    assert(fptrs[i]);
  }
  
  long gulpsize = svdzap->gulp_size;
  int numbeams = svdzap->n_files; 
  long gulpread;
  int j,ia,ib,aindex;
  int m = numbeams,n = numbeams;
  int lda = numbeams,ldu = numbeams,ldvt = numbeams;
  int info;
  int lwork;
  int final;
  float workOpt,madOne,madZero,flags_sone;
  float* work;
  float* tmpDat;
  float* sOne;
  float* sTwo;
  float* sFour;
  float* sThirteen;
 
  float * s = (float *) malloc(sizeof(float) * numbeams);
  float * a = (float *) malloc(sizeof(float) * numbeams*numbeams);
  float * u = (float *) malloc(sizeof(float) * numbeams*numbeams);
  float * vt = (float *) malloc(sizeof(float) * numbeams*numbeams);

  float** datagulp;
  float** xcorr_matrix;

  datagulp = (float **) malloc( numbeams * sizeof(float *) );
  assert (datagulp != NULL);

  for (i=0 ; i < numbeams ; i++ ) {
    datagulp[i] = (float *) malloc( gulpsize * sizeof(float) );
    assert(datagulp[i] != NULL);
  }

  xcorr_matrix = (float **) malloc (numbeams*numbeams * sizeof(float *));
  assert (xcorr_matrix != NULL);
  for (i=0; i< numbeams*numbeams; i++) {
    xcorr_matrix[i] = (float *) malloc (gulpsize * sizeof(float) );
    assert(xcorr_matrix[i] != NULL);
  }

  sOne = (float *) malloc (gulpsize * sizeof(float) );
  assert(sOne != NULL);

  sTwo = (float *) malloc (gulpsize * sizeof(float) );
  assert(sTwo != NULL);

  sFour = (float *) malloc (gulpsize * sizeof(float) );
  assert(sFour != NULL);

  sThirteen = (float *) malloc (gulpsize * sizeof(float) );
  assert(sThirteen != NULL);

  final = 0;
  
  for (i=0;i<numbeams*numbeams;i++) {
    a[i] = 0;
  }

  lwork=-1;
  // get best parameters
  sgesvd_("N","N",&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,&workOpt,&lwork,&info);
  lwork =(int)workOpt;
  work = (float*)malloc(lwork*sizeof(float));

  // timing
  if (svdzap->verbose)
  {
    split.tv_sec = curr.tv_sec;
    split.tv_usec = curr.tv_usec;
    gettimeofday (&curr, 0);
    split_time = diff_time (split, curr); 
    total_time = diff_time (start, curr); 
    fprintf (stderr, "svdzap_thread[%d]: sgesvd_ [%f of %f]\n", svdzap->tid, split_time, total_time);
  }

  gulpread = create_xcorr(datagulp,gulpsize,fptrs,numbeams,xcorr_matrix);

  if (svdzap->verbose)
  {
    split.tv_sec = curr.tv_sec;
    split.tv_usec = curr.tv_usec;
    gettimeofday (&curr, 0);
    split_time = diff_time (split, curr);      
    total_time = diff_time (start, curr);
    fprintf (stderr, "svdzap_thread[%d]: svd [%f of %f]\n", svdzap->tid, split_time, total_time);
  }

  if (gulpread < gulpsize) {
    final = 1;
  } 
    
  for (j=0;j<gulpread;j++) {
    
    for (i=0;i<numbeams*numbeams;i++) {
      a[i] = 0;
    }
    
    for (ia=0;ia<numbeams;ia++) {
      for (ib=ia;ib<numbeams;ib++) {
        a[ia*numbeams+ib] = xcorr_matrix[ia*numbeams+ib][j];
      }
    }
    
    sgesvd_("N","N",&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&lwork,&info);

    // check successful
    if( info > 0 ) {
      fprintf(stderr, "svdzap_thread[%d]: sgesvd_ failed\n", svdzap->tid);
      svdzap->result = -1;
      pthread_exit((void *) &(svdzap->result));
    }
    
    sOne[j] = s[0];
    sTwo[j] = s[1];
    sFour[j] = s[3];
    sThirteen[j] = s[12]; 
  }

  if (svdzap->verbose)
  {
    split.tv_sec = curr.tv_sec;
    split.tv_usec = curr.tv_usec;
    gettimeofday (&curr, 0);
    split_time = diff_time (split, curr);      
    total_time = diff_time (start, curr);
    fprintf (stderr, "svdzap_thread[%d]: mad [%f, %f]\n", svdzap->tid, split_time, total_time);
  }

  madOne = mad(sThirteen,1,gulpread);
  madZero = mad(sThirteen,0,gulpread);
  
  madOne = madOne*100;
  madZero = madZero*100;
  
  flags_sone = mean(sOne,gulpread);
  // normally we would double this, but the data is so noisy 
  // too much RFI is left behind in the 1-bit version.
  flags_sone = flags_sone*2;
  
  for(i=0;i<gulpread;i++) {
    if (sThirteen[i] > madZero) {
      svdzap->flags[i] = 0;
    }
    else if (sThirteen[i] > madOne) {
      if (sTwo[i] > flags_sone) {
        svdzap->flags[i] = 0;
      }
      else {
        svdzap->flags[i] = 1;
      }
    }
    else if (sFour[i] > flags_sone) {        
      // there are 4 eigenvalues greater than the first is supposed to be
      // even though it was not detected in the lower eigenvalues
      // this is probably RFI
      svdzap->flags[i] = 0;
    }
    else {
      svdzap->flags[i] = 1;
    }
    //fprintf(kfile_w,"%d\n",flags[i]);
  }

  free(s);
  free(a);
  free(u);
  free(vt);

  free(work);

  for (i=0; i<svdzap->n_files; i++)
    fclose(fptrs[i]);
  free(fptrs);

  free(datagulp);

  if (svdzap->verbose)
  {
    split.tv_sec = curr.tv_sec;
    split.tv_usec = curr.tv_usec;
    gettimeofday (&curr, 0);
    split_time = diff_time (split, curr);      
    total_time = diff_time (start, curr);
    fprintf (stderr, "svdzap_thread[%d]: finished [%f of %f]\n", svdzap->tid, split_time, total_time);
  }

  svdzap->result = 0;
  pthread_exit((void *) &(svdzap->result));

} 


long create_xcorr(float** datagulp, long gulpsize, FILE **currentFile,int numbeams, float** xcorr_matrix) {
  int i,j;
  int ia,ib;
  int some_inf;
  long num_read;
  float mean_dataia;
  float mean_dataib;
  float* datagulp_tmpia;
  float* datagulp_tmpib;
  float tmpia;
  float tmpib;
  
  datagulp_tmpia = (float *) malloc( gulpsize * sizeof(float) );
  assert(datagulp_tmpia != NULL);
  datagulp_tmpib = (float *) malloc( gulpsize * sizeof(float) );
  assert(datagulp_tmpib != NULL);

  //printf("initializing xcorr\n");
  for (i=0;i<numbeams*numbeams;i++) {
    for (j=0;j<gulpsize;j++) {
      xcorr_matrix[i][j] = 0;
    }
  }

  for (i=0;i<gulpsize;i++) {
    datagulp_tmpia[i] = 0;
    datagulp_tmpib[i] = 0;
  }

  //printf("reading data\n");
  for (i=0;i<numbeams;i++) {
    num_read = fread(datagulp[i],sizeof(float),gulpsize,currentFile[i]);
    some_inf = 0;
    for (j=0; j<gulpsize; j++)
    {
      if (isinf(datagulp[i][j]))
        some_inf = 1; 
    }
    if (some_inf)
      bzero(datagulp[i], sizeof(float) * gulpsize);
  }
  //printf("correlating\n");
  for (ia=0;ia<numbeams;ia++) {
    for (ib=ia;ib<numbeams;ib++) {
      for (i=0;i<num_read;i++) {
        datagulp_tmpia[i] = (float) datagulp[ia][i];
        datagulp_tmpib[i] = (float) datagulp[ib][i];
      }
      mean_dataia = mean(datagulp_tmpia,num_read);
      mean_dataib = mean(datagulp_tmpib,num_read);
      j = ia*numbeams + ib;
      for (i=0;i<num_read;i++) {
        tmpia = datagulp_tmpia[i] - mean_dataia;
        tmpib = datagulp_tmpib[i] - mean_dataib;
              xcorr_matrix[j][i] = tmpia*tmpib;
      }
    }
  }

  free(datagulp_tmpia);
  free(datagulp_tmpib);
  
  return num_read;
}

void close_files(FILE ** currentFile,int numbeams, int numpol) {
  int i;
  for (i=0;i<numbeams*numpol;i++) {
    fclose(currentFile[i]);
  }
} 


float mean(float* gulp, long gulpsize) {
  int i;
  float total = 0;

  for (i=0;i<gulpsize;i++) {
    total += gulp[i];
  }
  return total/gulpsize;

}

float mad(float* gulp,int mode,long gulpsize) {
  float* y;
  float* z;
  float tmp;
  int i;

  y = (float *) malloc (gulpsize * sizeof(float) );
  z = (float *) malloc (gulpsize * sizeof(float) );

  if (mode == 0) {
    tmp = mean(gulp,gulpsize);

    for (i=0;i<gulpsize;i++) {      
      y[i] = fabs(gulp[i] - tmp);
    }
    tmp = mean(y,gulpsize);
    free(y);
    free(z);
    return tmp;
  }
  
  for (i=0;i<gulpsize;i++) {
    z[i] = gulp[i];
  }

  tmp = median(z,gulpsize);


  for (i=0;i<gulpsize;i++) {
    y[i] = fabs(gulp[i] - tmp);

  }
  tmp = median(y,gulpsize);


  free(y);
  free(z);
  return tmp;
}

float median(float* gulp, long gulpsize) {

  quicksort(gulp,0,gulpsize-1);
  

  if (gulpsize % 2 == 0) {
    return (gulp[gulpsize/2] + gulp[gulpsize/2-1])/2;
  }
  return gulp[gulpsize/2];
  
}

int  choose_pivot(int  low_index,int high_index ) {
  return((low_index+high_index) /2);
}
 
void  swap(float *x,float *y)
{
   float temp;
   temp = *x;
   *x = *y;
   *y = temp;
}

void quicksort(float* list,int low_index,int high_index)
{
   int i,j,k;
   float key;

   if( low_index < high_index)
   {
     k = choose_pivot(low_index,high_index);
     swap(&list[low_index],&list[k]);
     key = list[low_index];
     i = low_index+1;
     j = high_index;

     while(i <= j)
     {       
       while((i <= high_index) && (list[i] <= key)) {
         i++;
       }
       while((j >= low_index) && (list[j] > key)) {
         j--;
       }
       if( i < j) {
         swap(&list[i],&list[j]);
       }

     }
     swap(&list[low_index],&list[j]);
     // recursively sort the lesser list
     quicksort(list,low_index,j-1);
     quicksort(list,j+1,high_index);
   }
}

double diff_time ( struct timeval time1, struct timeval time2 )
{
   return ( (double)( time2.tv_sec - time1.tv_sec ) +
             ( (double)( time2.tv_usec - time1.tv_usec ) / 1000000.0 ) );
}

