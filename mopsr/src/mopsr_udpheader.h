/* dada, ipc stuff */

#include "dada_hdu.h"
#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"
#include "multilog.h"

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

/* structures dmadb datatype  */
typedef struct {

  multilog_t *    log;

  unsigned        ninputs;
  char **         interfaces;  
  int *           ports;
  mopsr_sock_t ** socks;

  uint64_t        n_sleeps;
  int             capture_started;
  int             verbose;

  stats_t *       packets;
  stats_t *       bytes;

  time_t          prev_time;
  time_t          curr_time;

  uint64_t        seq_incr;
  uint64_t        seq_max;

} udpheader_t;

/* Re-implemented functinos from dada_pwc */
time_t  udpheader_start_function (udpheader_t* udpheader, time_t start_utc);
void*   udpheader_read_function (udpheader_t* udpheader, uint64_t* size);
int     udpheader_stop_function (udpheader_t* udpheader);

/* Utility functions */
void quit (udpheader_t* udpheader);
void signal_handler (int signalValue); 


