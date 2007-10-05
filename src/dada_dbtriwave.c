#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "node_array.h"
#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

#include <sys/types.h>
#include <sys/socket.h>

void generate_triwave(char *array, int array_size, int nbits);
int verify_tri_data(char * data, char * tri_data, uint64_t compare_size);
void print_triwave(char *array, int array_size, int nbits);
static void fsleep (double seconds);
int myusleep (double useconds);

/* structures dbtritest datatype  */
typedef struct{
  int verbose;              /* verbosity flag */
  int expect_header;        /* verbosity flag */
  char *tri_data;           // triangle wave form data expected
  uint64_t tri_data_length; // size of tri wave from array
  uint64_t tri_data_nbits;  // size of tri wave from array
  uint64_t tri_data_counter;  // counter to keep track of verifying
  dada_hdu_t *hdu;           // pointer to the hdu
}dbtritest_t;

void usage()
{
  fprintf (stdout,
     "dada_dbtritest [options]\n"
     " -v         be verbose\n"
     " -l num     period of triangular wave form data [default 512]\n"
     " -b num     number of bits in triangular wave form data [default 8]\n"
     " -d         run as daemon\n");
}


/*! Function that opens the data transfer target */
int dbtritest_open_function (dada_client_t* client)
{

  dbtritest_t *dbtritest = client->context;
  dbtritest->expect_header = 1;
  if (dbtritest->verbose) fprintf(stderr,"dbtritest_open_function\n");
  /* status and error logging facility */
  multilog_t* log;

  /* the header */
  char* header = 0;

  assert (client != 0);

  log = client->log;
  assert (log != 0);

  header = client->header;
  assert (header != 0);

  //dada_hdu_lock_read(dbtritest->hdu); 

  return 0;
}

/*! Function that closes the data file */
int dbtritest_close_function (dada_client_t* client, uint64_t bytes_written)
{

  /* status and error logging facility */
  multilog_t* log;

  assert (client != 0);

  log = client->log;
  assert (log != 0);

  if (bytes_written < client->transfer_bytes) {
    multilog (log, LOG_INFO, "Transfer stopped early at %"PRIu64" bytes\n",
	      bytes_written);
  }

  //dada_hdu_unlock_read(dbtritest->hdu);

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbtritest_io_function (dada_client_t* client, 
			    void* data, uint64_t data_size)
{

  /* keep track of the data as we loop through it... */
  uint64_t data_counter = 0;
  int retval = 0;
  uint64_t compare_size = 0;

  dbtritest_t *dbtritest = client->context;

  if (dbtritest->verbose) fprintf(stderr,"dbtritest_io_function: %"PRIu64"\n",
                                  data_size);

  /* If we are asked to read the header, just ignore and return */
  if (dbtritest->expect_header) {
    if (dbtritest->verbose) fprintf(stderr,"dbtritest_io_function: read the "
                                    "header of %"PRIu64" bytes\n", data_size);
    dbtritest->expect_header = 0;
    data_counter = data_size;

  } else {

    /* If the tri wave is larger than the data block... */   
    if (dbtritest->tri_data_length > data_size) {

      /* If the comparison would over run the data block, then do 2 compares */
      if (dbtritest->tri_data_counter + data_size > dbtritest->tri_data_length) 
      {

        /* compare up to the end of the tri_data block */
        compare_size = dbtritest->tri_data_length - dbtritest->tri_data_counter;
        
        retval = verify_tri_data(data, (dbtritest->tri_data + 
                                 dbtritest->tri_data_counter), compare_size);

        if (retval == -1) {
          fprintf(stderr,"1: Error in the triangular wave form data\n");
        }
        
        uint64_t data_compared = compare_size;
        
        /* increment the counter */
        dbtritest->tri_data_counter = 0;

        /* compare from the start of the tri_data block */
        compare_size = data_size - compare_size;
        retval = verify_tri_data(data+data_compared, dbtritest->tri_data, 
                                 compare_size); 
        dbtritest->tri_data_counter += compare_size;
        
        if (retval == -1) {
          fprintf(stderr,"2: Error in the triangular wave form data\n");
        }
              
      } else {

        compare_size = data_size;
        retval = verify_tri_data(data, (dbtritest->tri_data + 
                                 dbtritest->tri_data_counter), compare_size);
        dbtritest->tri_data_counter += compare_size;
      }

      /* we processed the memory block */
      data_counter = data_size;
    
    /* the data block is larger than the tri wave */  
    } else {

      while ((data_counter + dbtritest->tri_data_length) <= data_size) {

        /* compare any data from last buffer */
        compare_size = dbtritest->tri_data_length - dbtritest->tri_data_counter;

        retval = verify_tri_data(data+data_counter, dbtritest->tri_data + 
                                 dbtritest->tri_data_counter, compare_size);
        
        if (retval == -1) {
          fprintf(stderr,"3: Error in the triangular wave form data\n");
        } 

        data_counter += compare_size;
        dbtritest->tri_data_counter = 0;
        
      }
    
      /* if we have some left over data in that hasn't been compared due to
       * array size differences */
      if (data_counter != data_size) {

        assert(data_counter < data_size);
        compare_size = data_size - data_counter;

        retval = verify_tri_data(data+data_counter, dbtritest->tri_data +                                        dbtritest->tri_data_counter, compare_size);

        if (retval == -1) {
          fprintf(stderr,"4: Error in the triangular wave form data\n");
        }

        /* since there is not enough space in data to compare all of tri_data,
         * we need to record the tri_data_counter info for the next call to 
         * this function */

        dbtritest->tri_data_counter = compare_size;
        data_counter += compare_size;
      }
      
    }
  }    

  time_t temptime = time(0);
  //printf("time before = %u\n",temptime);
  
  if (myusleep(9000.0) < 0)
    perror("select error");

  temptime = time(0);
  //printf("time after  = %u\n",temptime);

  assert(data_counter == data_size);
  return data_counter;
}

/* Checks that the data in the data array matches the data in the tri_data
 * array up to the compare size index */
int verify_tri_data(char * data, char * tri_data, uint64_t compare_size) {
       
  int i = 0;
  int retval = 1;
  char zerodchar;
  memset(&zerodchar,0,sizeof(char));

  for (i=0; ((retval > 0) && (i < compare_size)); i++) {
    if ((data[i] != tri_data[i]) && (retval == 1)) {
      //fprintf(stderr,"data[%d] = %d, %d,%p\n",i,data[i], 
      //        zerodchar,&(data[i]));
      retval = -1;
    }
    if ((data[i] == zerodchar) && (retval == -1)){
      retval = 0;
    }
  }
  return retval;
}

int main (int argc, char **argv)
{

  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Primary Read Client main loop */
  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* Quit flag */
  char quit = 0;

  int arg = 0;

  /* Number of bits in triangular wave form data */
  int tri_nbits = 8;

  /* Period of triangular wave form data */
  int tri_period = 512;

  /* actual struct with info */
  dbtritest_t dbtritest;

  while ((arg=getopt(argc,argv,"d:l:b:v")) != -1)
    switch (arg) {
      
    case 'd':
      daemon=1;
      break;

    case 'l':
      tri_period = atoi(optarg);
      break;
      
    case 'b':
      tri_nbits = atoi(optarg);
      break;

    case 'v':
      verbose=1;
      break;
      
    default:
      usage ();
      return 0;
      
    }

  log = multilog_open ("dada_dbtritest", daemon);

  if (daemon) {
    be_a_daemon ();
  }
  //else
    //multilog_add (log, stderr);

  multilog_serve (log, DADA_DEFAULT_DBTRITEST_LOG);

  hdu = dada_hdu_create (log);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_read (hdu) < 0)
    return EXIT_FAILURE;

  client = dada_client_create ();

  client->log = log;

  client->data_block = hdu->data_block;
  client->header_block = hdu->header_block;

  client->context        = &dbtritest;
  client->open_function  = dbtritest_open_function;
  client->io_function    = dbtritest_io_function;
  client->close_function = dbtritest_close_function;
  client->direction      = dada_client_reader;
  client->fd             = 1;                 // not used

  client->transfer_bytes = 1024*1024;
  client->optimal_bytes = client->transfer_bytes*8;
  
  dbtritest.tri_data_length = tri_period;
  dbtritest.tri_data_nbits = tri_nbits;
  dbtritest.tri_data = malloc(sizeof(char) * dbtritest.tri_data_length);
  dbtritest.tri_data_counter = 0;
  dbtritest.expect_header = 1;
  dbtritest.verbose = verbose;
  dbtritest.hdu = hdu;

  generate_triwave(dbtritest.tri_data, dbtritest.tri_data_length,
                                       dbtritest.tri_data_nbits);

  fprintf(stderr,"entering main loop\n");

  while (!quit) {
    
    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

  }

  if (dada_hdu_unlock_read (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

void print_triwave(char *array, int array_size, int nbits) {
                                                                                
  int i = 0;
  unsigned int val = 0;
  printf("[");
  for (i=0; i < array_size; i++) {
    //val << (sizeof(val)*8)-1;
    val = (unsigned int) array[i];
    val &= 0x000000ff;
    printf("%u", val);
    if (i != (array_size - 1)) printf(",");
  }
  printf("]\n");
                                                                                
}



void generate_triwave(char *array, int array_size, int nbits) {

  // The triangle wave will fill the array, and the mid value
  // will be 2*nbits -  (i.e. for 8 bit 0 > 255)

  int i = 0;

  float max_value = powf(2.0,(float) nbits) - 1.0;
  float mid_point = ((float) array_size) / 2.0;

  //printf("max_value = %f\n",max_value);
  //printf("mid_point = %f\n",mid_point);
  //printf("array_size = %d\n",array_size);

  unsigned int val;

  for (i=0; i <= mid_point; i++) {

    val = (unsigned int) (max_value * (((float) i) / mid_point));
    array[i] = val;
    array[array_size-i] = array[i];

    //displayBits(val);
    //printf("%d: [%u] [%u]\n",i,val,array[i]);
    //displayCharBits(array[i]);
  }

}

static void fsleep (double seconds)
{
  struct timeval t ;
                                                                                
  t.tv_sec = seconds;
  seconds -= t.tv_sec;
  t.tv_usec = seconds * 1e6;
  select (0, 0, 0, 0, &t) ;
}

int myusleep (double useconds)
{
  struct timeval t ;
  t.tv_sec = 0;
  t.tv_usec = useconds;
  return select (0, 0, 0, 0, &t) ;
}


