#include <stdlib.h> 
#include <stdio.h> 
#include <errno.h> 
#include <string.h> 
#include <sys/types.h> 
#include <netinet/in.h> 
#include <netdb.h> 
#include <sys/socket.h> 
#include <sys/wait.h> 
#include <sys/timeb.h> 
#include <math.h>
#include <pthread.h>
#include <assert.h>

#include "daemon.h"
#include "multilog.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)

#define MOPSR_UDPGEN_LOG  49200

void signal_handler(int signalValue);

void usage();

int main(int argc, char *argv[])
{

  /* number of microseconds between packets */
  double sleep_time = 22;
 
  /* be verbose */ 
  int verbose = 0;

  /* udp port to send data to */
  int dest_port = MOPSR_DEFAULT_UDPDB_PORT;

  /* UDP socket struct */
  struct sockaddr_in dagram;

  /* total time to transmit for */ 
  uint64_t transmission_time = 5;   

  /* DADA logger */
  multilog_t *log = 0;

  /* Hostname to send UDP packets to */
  char *dest_host;

  /* udp file descriptor */
  int udpfd;

  /* The generated signal arrays */
  unsigned char packet[UDP_PAYLOAD];

  // data rate in gbits
  float data_rate_gbits = 1;

  /* number of packets to send every second */
  uint64_t packets_ps = 0;

  /* start of transmission */
  time_t start_time;

  /* end of transmission */
  time_t end_time;

  // sequence number
  uint64_t seq_no = 0;

  opterr = 0;
  int c;
  while ((c = getopt(argc, argv, "hn:p:r:v")) != EOF) {
    switch(c) {

      case 'h':
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'n':
        transmission_time = atoi(optarg);
        break;

      case 'p':
        dest_port = atoi(optarg);
        break;

      case 'r':
        data_rate_gbits = atof(optarg);
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  // Check arguments
  if ((argc - optind) != 1) 
  {
    fprintf(stderr,"ERROR: 1 command line argument expected [destination host]\n");
    usage();
    return EXIT_FAILURE;
  }

  // destination host
  dest_host = strdup(argv[optind]);

  signal(SIGINT, signal_handler);

  // create a long array of gaussian noise
  srand ( time(NULL) );
  char * g_array;
  unsigned int g_array_length = UDP_DATA * 100;
  g_array = (char *) malloc (sizeof(char) * g_array_length);
  if (!g_array)
  {
    fprintf (stderr, "could not allocate memory for gaussian array\n");
    return 1;
  }

  // do not use the syslog facility
  log = multilog_open ("mopsr_udpgen", 0);

  multilog_add(log, stderr);

  if (verbose)
    multilog(log, LOG_INFO, "udpgen: generating gaussian stream of %d bytes\n", g_array_length);

  fill_gaussian_chars (g_array, g_array_length, 8, 500);

  if (verbose)
    multilog(log, LOG_INFO, "udpgen: gaussian stream generated\n");

  // assume gbits is base 1000 
  double gbits_to_bytes = 1000000000.0 / 8.0;

  // data rate in bytes / second
  double data_rate = (double) data_rate_gbits * gbits_to_bytes;

  if (verbose)
  {
    multilog(log, LOG_INFO, "sending UDP data to %s:%d\n", dest_host, dest_port);
    if (data_rate)
      multilog(log, LOG_INFO, "data rate: %5.2f MB/s \n", data_rate/(1024*1024));
    else
      multilog(log, LOG_INFO, "data_rate: fast as possible\n");
    multilog(log, LOG_INFO, "transmission length: %d seconds\n", transmission_time);
  }

  // create the socket for outgoing UDP data
  dada_udp_sock_out (&udpfd, &dagram, dest_host, dest_port, 0, "192.168.3.255");

  uint64_t data_counter = 0;

  // initialise data rate timing library 
  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  /* If we have a desired data rate, then we need to adjust our sleep time
   * accordingly */
  if (data_rate > 0)
  {
    packets_ps = floor(((double) data_rate) / ((double) UDP_PAYLOAD));
    sleep_time = (1.0/packets_ps) * 1000000.0;

    if (verbose)
    {
      multilog(log, LOG_INFO, "packets/sec %"PRIu64"\n",packets_ps);
      multilog(log, LOG_INFO, "sleep_time %f us\n",sleep_time);
    }
  }

  uint64_t total_bytes_to_send = data_rate * transmission_time;

  // assume 10GbE speeds
  if (data_rate == 0)
    total_bytes_to_send = 1*1024*1024*1024 * transmission_time;

  size_t bytes_sent = 0;
  uint64_t total_bytes_sent = 0;

  uint64_t bytes_sent_thistime = 0;
  uint64_t prev_bytes_sent = 0;
  
  time_t current_time = time(0);
  time_t prev_time = time(0);

  multilog(log,LOG_INFO,"Total bytes to send = %"PRIu64"\n",total_bytes_to_send);
  multilog(log,LOG_INFO,"UDP payload = %"PRIu64" bytes\n",UDP_PAYLOAD);
  multilog(log,LOG_INFO,"UDP data size = %"PRIu64" bytes\n",UDP_DATA);
  multilog(log,LOG_INFO,"Wire Rate\t\tUseful Rate\tPacket\tSleep Time\n");

  unsigned int s_off = 0;

  while (total_bytes_sent < total_bytes_to_send) 
  {
    if (data_rate)
      StopWatch_Start(&wait_sw);

    // choose a start index in gaussian array
    float rand_ratio = ((float) rand()) / ((float) RAND_MAX);
    int index = (int) (rand_ratio * UDP_DATA * 99);

    // copy the pseudo random data of gausian noise
    memcpy (packet + UDP_HEADER, g_array + index, UDP_DATA);

    // write the custom header into the packet
    mopsr_encode_header(packet, seq_no);

    bytes_sent = dada_sock_send(udpfd, dagram, packet, (size_t) UDP_PAYLOAD); 

    if (bytes_sent != UDP_PAYLOAD) 
      multilog(log,LOG_ERR,"Error. Attempted to send %d bytes, but only "
                           "%"PRIu64" bytes were sent\n",UDP_PAYLOAD,
                           bytes_sent);

    // this is how much useful data we actaully sent
    total_bytes_sent += (bytes_sent - UDP_HEADER);

    data_counter++;
    prev_time = current_time;
    current_time = time(0);
    
    if (prev_time != current_time) 
    {
      double complete_udp_packet = (double) bytes_sent;
      double useful_data_only = (double) (bytes_sent - UDP_HEADER);
      double complete_packet = 28.0 + complete_udp_packet;

      double wire_ratio = complete_packet / complete_udp_packet;
      double useful_ratio = useful_data_only / complete_udp_packet;
        
      uint64_t bytes_per_second = total_bytes_sent - prev_bytes_sent;
      prev_bytes_sent = total_bytes_sent;
      double rate = ((double) bytes_per_second) / (1024*1024);

      double wire_rate = rate * wire_ratio;
      double useful_rate = rate * useful_ratio;
             
      multilog(log,LOG_INFO,"%5.2f MB/s  %5.2f MB/s  %"PRIu64
                            "  %5.2f, %"PRIu64"\n",
                            wire_rate, useful_rate, data_counter, sleep_time,
                            bytes_sent);
    }

    seq_no++;

    if (data_rate)
      StopWatch_Delay(&wait_sw, sleep_time);
  }

  uint64_t packets_sent = seq_no;

  multilog(log, LOG_INFO, "Sent %"PRIu64" bytes\n",total_bytes_sent);
  multilog(log, LOG_INFO, "Sent %"PRIu64" packets\n",packets_sent);

  close(udpfd);
  free (dest_host);

  return 0;
}


void signal_handler(int signalValue) {
  exit(EXIT_SUCCESS);
}

void usage() 
{
  fprintf(stdout,
    "mopsr_udpgen [options] host\n"
    "-h            print this help text\n"
    "-n secs       number of seconds to transmit [default 5]\n"
    "-p port       destination udp port [default %d]\n"
    "-r rate       transmit at rate Gbits/s [default 1]\n"
    "-v            verbose output\n"
    "host          destination host name\n\n"
    ,MOPSR_DEFAULT_UDPDB_PORT);
}
