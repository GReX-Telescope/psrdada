/* dada, ipc stuff */

#include "config.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "bpsr_def.h"
#include "bpsr_udp.h"
#include "multilog.h"
#include "futils.h"

#ifdef HAVE_CUDA
#include "dada_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>

/* Number of UDP packets to be recived for a called to buffer_function */
#define NUMUDPPACKETS 2000
#define NOTRECORDING 0
#define RECORDING 1


/* structures dmadb datatype  */
typedef struct{

  multilog_t *      log;    // multilog pointer
  bpsr_sock_t *     sock;   // socket for UDP data

  uint64_t          packet_header_size;   // number of bytes in the custom header of packet
  uint64_t          packet_data_size;     // number of bytes in the packet of signal data
  uint64_t          packet_payload_size;  // total payload size
  uint64_t          block_size;           // size of a datablock buffer
  uint64_t          packets_per_block;    // number of packets in a block

  int verbose;           // verbosity flag
  int port;              // port to receive UDP data
  char *next_buffer;
  char *curr_buffer;
  char *zero_buffer;
  int packet_in_buffer;  // flag if buffer overrun occured
  uint64_t curr_buffer_count;
  uint64_t next_buffer_count;
  uint64_t zero_buffer_count;
  ssize_t received;     /* number of bytes received */

  // sequence numbers
  uint64_t expected_sequence_no;
  uint64_t curr_sequence_no;
  uint64_t min_sequence;
  uint64_t mid_sequence;
  uint64_t max_sequence;

  // iBob specification configuration
  char *   interface;
  uint64_t acc_len;
  uint64_t sequence_incr;
  uint64_t spectra_per_second;
  uint64_t bytes_per_second;
                                                                                
  // Packet/byte stats
  stats_t * packets;
  stats_t * bytes;

  uint64_t packets_late_this_sec;     //
  unsigned int expected_header_bits;  // Expected bits/sample
  unsigned int expected_nchannels;    // Expected channels 
  unsigned int expected_nbands;       // Expected number of bands
  float current_bandwidth;            // Bandwidth currently being received
  time_t current_time;                
  time_t prev_time; 

  /* total number of bytes to transfer */
  uint64_t transfer_bytes;

  /* optimal number of bytes to transfer in 1 write call */
  uint64_t optimal_bytes;

  uint64_t select_sleep;

#if HAVE_CUDA
  cudaStream_t stream;
  int device;
  void * gpu_buffer;
#endif

}udpheader_t;

void    udpheader_init (udpheader_t* udpheader);

/* Re-implemented functions from dada_pwc */
time_t  udpheader_start_function (udpheader_t* udpheader, time_t start_utc);
void*   udpheader_read_function (udpheader_t* udpheader, uint64_t* size);
int     udpheader_stop_function (udpheader_t* udpheader);

/* Utility functions */
void quit (udpheader_t* udpheader);
void signal_handler (int signalValue); 


