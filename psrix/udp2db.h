/* dada, ipc stuff */

#include "dada_hdu.h"
#include "dada_def.h"
#include "dada_pwc_main.h"

#include "ipcio.h"
#include "multilog.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

/* includes taken from test program */
/* remove those that are not needed */
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <string.h>
/* buffer size is defined by packet size here */
//#define BUFSIZE 8208
#define BUFSIZE 8192

/* structures dmadb datatype  */
typedef struct{
  int verbose; /* verbosity flag */
  unsigned long long prev_pkt_cnt, bad_pkt_cnt, pkt_cnt;
  int bound;
  int FirstFlag;
  int pktdiff;
  int marker_enable;
  int fSize; /* file size of data */
  int nSecs; /* number of seconds to acquire */
  int numip;
  int port;
  long long buf;
  char daemon; /*daemon mode */
  char *ObsId;
  char *data;
  char *dataFromPrevPkt;
  int markerOffset;
  int sock;
  struct sockaddr_in sa;
}udp2db_t;
  
