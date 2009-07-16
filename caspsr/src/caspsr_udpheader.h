
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
#include <signal.h>

#include "dada_hdu.h"
#include "caspsr_def.h"
#include "caspsr_udp.h"
#include "multilog.h"


/* Number of UDP packets to be recived for a called to buffer_function */
#define NUMUDPPACKETS 2000
#define NOTRECORDING 0
#define RECORDING 1

typedef struct{

  multilog_t* log;
  int state;
  int verbose;           /* verbosity flag */
  int udpfd;             /* udp socket file descriptor */
  int port;              /* port to receive UDP data */
  char *   interface;   /* NIC/Interface to open socket on */

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;

  /* Current and Next buffers */
  caspsr_buffer_t * curr;
  caspsr_buffer_t * next;
  caspsr_buffer_t * temp;

  /* socket used to receive a packet */
  socket_buffer_t * sock;

  /* decoded header of a caspsr packet */
  caspsr_header_t * header;

  uint64_t next_seq;        // Next packet we are expecting
  //size_t got;               // number of bytes received by recvfrom
  struct   timeval timeout; 
  unsigned timer;
  time_t   current_time;                
  time_t   prev_time; 
  unsigned got_enough;      // flag for having received enough packets
  uint64_t transfer_bytes;  // total number of bytes to transfer
  uint64_t optimal_bytes;   // optimal number of bytes to transfer in 1 write

}udpheader_t;

/* Re-implemented functinos from dada_pwc */
time_t  udpheader_start_function (udpheader_t* udpheader, time_t start_utc);
void*   udpheader_read_function (udpheader_t* udpheader, uint64_t* size);
int     udpheader_stop_function (udpheader_t* udpheader);

/* Utility functions */
void quit (udpheader_t* udpheader);
void signal_handler (int signalValue); 


