/* dada, ipc stuff */

#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_pwc_main.h"
#include "udp.h"

#include "ipcio.h"
#include "multilog.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>

/* for semaphore reading */
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

/* Maximum size of a UDP packet */
#define UDPBUFFSIZE 16384

/* Number of UDP packets to be recived for a called to buffer_function */
#define NUMUDPPACKETS 2000

/* Size of header component of the data packet */
#define UDPHEADERSIZE 14

/* Size of data component of the data packet */
#define UDPDATASIZE 1458

/* structures dmadb datatype  */
typedef struct{
  int verbose;           /* verbosity flag */
  int fd;                /* udp socket file descriptor */
  int port;              /* port to receive UDP data */
  int check_udp_headers; /* whether to check the UDP header against header */
  char *data;            /* data buffer for sotring of multiple udp packets */
  int datasize;          /* size of *data array */
  char *buffer;          /* buffer for a single udp packet */
  int buffer_full;       /* flag if buffer overrun occured */
  uint64_t received;     /* number of bytes received */
  uint64_t expected_sequence_no;
  uint64_t packets_dropped;           // Total dropped
  uint64_t packets_dropped_this_run;  // Dropped b/w start and stop
  uint64_t packets_received;          // Total dropped
  uint64_t packets_received_this_run; // Dropped b/w start and stop
  uint64_t packets_received_last_sec; // Dropped b/w start and stop
  uint64_t bytes_received;            // Total dropped
  uint64_t bytes_received_this_run;   // Dropped b/w start and stop
  uint64_t bytes_received_last_sec;   // Dropped b/w start and stop
  uint64_t dropped_packets_to_fill;   // 
  unsigned int expected_header_bits;  // Expected bits/sample
  unsigned int expected_nchannels;    // Expected channels 
  unsigned int expected_nbands;       // Expected number of bands
  float current_bandwidth;            // Bandwidth currently being received
  multilog_t* statslog;               // special log for statistics
  time_t current_time;
  time_t prev_time; 
  int quit;
   
}udpdb_t;

void print_udpbuffer(char * buffer, int buffersize);
void print_header(header_struct header);
void check_udpdata(char * buffer, int buffersize, int value);
int create_udp_socket(dada_pwc_main_t* pwcm);
void check_header(header_struct header, udpdb_t *udpdb, multilog_t *log); 


