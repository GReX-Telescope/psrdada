#include "dada_def.h"
#include "apsr_udpdb.h"
#include "ascii_header.h"
#include "udp.h"
#include "sock.h"
#include "daemon.h"
#include "multilog.h"

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
#include <arpa/inet.h>

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

int sendPacket(int sockfd, struct sockaddr_in addr, char *data, int size);
void encode_data(char * udp_data, int size_of_frame, int value);
struct in_addr *atoaddr(char *address);
void signal_handler(int signalValue);
void usage();
void quit();
void generate_triwave(char *tri_data, int tri_datasize, int tri_nbits);
void print_triwave(char *tri_data, int tri_datasize, int tri_nbits);
void displayBits(unsigned int value);
void displayCharBits(char value);
int encode_tri_data(char * udp_data, int size_of_frame, char * tri_data,
                    int tri_datasize, int tri_counter);

/* UDP file descriptor */
int sockDatagramFd;

/* UDP socket */
struct sockaddr_in dagram_socket;

int main(int argc, char *argv[])
{

  /* custom header size of size UDPHEADERSIZE */
  header_struct header;

  /* number of microseconds between packets */
  double sleep_time = 22;
 
  /* Size of UDP payload. Optimally 1472 bytes */
  uint64_t size_of_frame = UDPHEADERSIZE + DEFAULT_UDPDATASIZE;

  /* be verbose? */ 
  int verbose = 0;

  /* udp port to send data to */
  int udp_port = DADA_DEFAULT_UDPDB_PORT;

  /* total time to transmit for */ 
  uint64_t transmission_time = 5;   

  /* Desired data rate in MB per second */
  double data_rate = 64.0;       

  /* be a daemon? */
  int daemon = 0;

  /* NBANDS header value */
  int nband = 1;

  /* NBIT header value */
  int nbit = 8;

  /* NCHANNEL header value */
  int nchannel = 2;

  /* NPOL header value */
  int npol = 2;

  /* DADA logger */
  multilog_t *log = 0;

  /* Hostname to send UDP packets to */
  char *udpdb_client;

  /* udp file descriptor */
  int udpfd;

  /* triangular wave data array */
  char * tri_data;

  /* length of triangular wave in bytes */
  uint64_t tri_datasize = 512;

  /* number of bits in the triangular wave */
  int tri_nbits = 8;
  
  /* number of bits in the triangular wave */
  uint64_t udp_data_size = DEFAULT_UDPDATASIZE;

  opterr = 0;
  int c;
  while ((c = getopt(argc, argv, "ds:p:n:l:r:v")) != EOF) {
    switch(c) {

      case 's':
        size_of_frame = atoi(optarg);
        udp_data_size = size_of_frame - UDPHEADERSIZE;
        if ((size_of_frame <= 0) || (size_of_frame >= UDPBUFFSIZE)) {
          fprintf(stderr,"packet size must be between > 0 and < %d bytes\n",
                          UDPBUFFSIZE);
          usage();
          return EXIT_FAILURE;
        }
        break;

      case 'n':
        transmission_time = atoi(optarg);
        break;

      case 'd':
        daemon = 1;
        break;
        
      case 'v':
        verbose = 1;
        break;

      case 'p':
        udp_port = atoi(optarg);
        break;

      case 'l':
        tri_datasize = atoi(optarg);
        break;

      case 'r':
        data_rate = atof(optarg);
        break;

      default:
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  if ((argc - optind) != 1) {
    fprintf(stderr,"no udpdb_client was specified\n");
    usage();
    return EXIT_FAILURE;
  }

  log = multilog_open ("test_triwaveudp", daemon);

  if (daemon) {
    be_a_daemon();
  } else {
    multilog_add(log, stderr);
  }
  multilog_serve (log, DADA_TEST_TRIWAVEUDP_LOG);

  udpdb_client = (char *) argv[optind];
  signal(SIGINT, signal_handler);

  /* Get the client hostname */
  struct hostent * client_hostname;
  client_hostname = gethostbyname(udpdb_client);

  /* Setup the socket for UDP packets */
  udpfd = socket(PF_INET, SOCK_DGRAM, 0);
  if (udpfd < 0) {  
    perror("udp socket");
    return EXIT_FAILURE;
  }

  /* Setup the UDP socket parameters*/
  struct in_addr *addr;
  dagram_socket.sin_family = AF_INET;        /* host byte order */
  dagram_socket.sin_port = htons(udp_port);  /* short, network byte order */
  addr = atoaddr(udpdb_client);
  dagram_socket.sin_addr.s_addr = addr->s_addr;
  bzero(&(dagram_socket.sin_zero), 8);       /* zero the rest of the struct */

  if (verbose) 
    multilog(log,LOG_INFO,"udp socket setup\n");

  char udp_data[(UDPHEADERSIZE+udp_data_size)];
  uint64_t data_counter = 0;
  uint64_t bytes_sent = 0;
  uint64_t bytes_sent_thistime = 0;
  uint64_t prev_bytes_sent = 0;

  StopWatch wait_sw;

  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  /* Setup udp header information */
  header.length = (int) UDPHEADERSIZE;
  header.source = (char) 3;     /* This is not used by udpdb */
  header.sequence = 0;
  header.bits = nbit;
  header.channels = nchannel;
  header.bands = nband;    
  header.bandID[0] = 1;         /* This is not used by udpdb */
  header.bandID[1] = 2;         /* This is not used by udpdb */
  header.bandID[2] = 3;         /* This is not used by udpdb */
  header.bandID[3] = 4;         /* This is not used by udpdb */

  if (verbose) 
    multilog(log,LOG_INFO,"data structures setup\n");

  /* If we have a desired data rate, then we need to adjust our sleep time
   * accordingly */
  if (data_rate > 0) {
    double packets_per_second = (((double)data_rate)*(1024*1024)) /
                                 ((double)udp_data_size);
    sleep_time = (1.0/packets_per_second)*1000000.0;
  }

  /* Create and allocate the triangular data array */
  tri_data = malloc(sizeof(char) * tri_datasize);
  generate_triwave(tri_data, tri_datasize, tri_nbits);

  if ((verbose) && (!daemon)) {
    print_triwave(tri_data, tri_datasize, tri_nbits);
    print_header(&header);
  }

  int tri_counter = 0;
  int bufsize = 4096;
  char* buffer = (char*) malloc ( bufsize);
  assert (buffer != 0);

  /* If we are running as a daemon, then we must wait for a "start"
   * message, and then return the UTC time of the first packet. Hence
   * we would start on "second" time boundary based on NTP */

  time_t start_time;
  time_t end_time;
  if (daemon) {

    FILE *sockin = 0;
    FILE *sockout = 0;

    /* Port for control from tcs simulator */
    int port = DADA_TEST_TRIWAVEUDP_COMMAND;

    /* create a socket on which to listen */
    int listen_fd = sock_create (&port);
    if (listen_fd < 0)  {
      multilog(log,LOG_ERR,"Error creating socket: %s\n", strerror(errno));
      return EXIT_FAILURE;
    }

    multilog(log,LOG_INFO,"waiting for command connection\n");

    /* wait for a connection */
    int tcpfd =  sock_accept (listen_fd);
    if (tcpfd < 0)  {
      multilog(log,LOG_ERR,"Error accepting connection: %s\n", strerror(errno));
      return EXIT_FAILURE;
    }

    sockin = fdopen(tcpfd,"r");
    sockout = fdopen(tcpfd,"w");
    setbuf (sockin, 0);
    setbuf (sockout, 0);

    char* rgot = NULL;
    rgot = fgets (buffer, bufsize, sockin);
    if (rgot && !feof(sockin)) {
      buffer[strlen(buffer)-1] = '\0';
    }

    /* we will start in +1 seconds of the current time */
    start_time = time(0) + 1;
    strftime (buffer, bufsize, DADA_TIMESTR,
              (struct tm*) localtime(&start_time));
    fprintf(sockout,"%s\n",buffer);
    fprintf(sockout,"ok\n");
    
    close(tcpfd);
    close(listen_fd);

  } else {
    start_time = time(0);
    strftime (buffer, bufsize, DADA_TIMESTR, (struct tm*) localtime(&start_time));
  }

  multilog(log,LOG_INFO,"udp data starting at %s\n",buffer);

  /* End time for transfer */
  end_time = start_time+transmission_time;
  strftime (buffer, bufsize, DADA_TIMESTR, (struct tm*) localtime(&end_time));
  multilog(log,LOG_INFO,"udp data will end at %s\n",buffer);
  
  time_t current_time = time(0);
  time_t prev_time = time(0);

  multilog(log,LOG_INFO,"Wire Rate\tUseful Rate\tPacket\tSleep Time\n");

  while (current_time <= end_time) {

    if (daemon)  {
      while (current_time >= start_time) {
        current_time = time(0);
      }
      daemon=0;
    }

    StopWatch_Start(&wait_sw);

    header.sequence++;
    encode_header(udp_data, &header);
    
    tri_counter = encode_tri_data(udp_data+UDPHEADERSIZE,
                                  (size_of_frame-UDPHEADERSIZE), tri_data,
                                  tri_datasize, tri_counter);

    bytes_sent_thistime = sendPacket(udpfd, dagram_socket, udp_data, 
                                     size_of_frame);

    if (bytes_sent_thistime != size_of_frame) {
      multilog(log,LOG_ERR,"Error. Attempted to send %"PRIu64" bytes, but only "
                           "%"PRIu64" bytes were sent\n",size_of_frame, 
                           bytes_sent_thistime);
    } 

    bytes_sent += bytes_sent_thistime;

    data_counter++;
    prev_time = current_time;
    current_time = time(0);
    
    if (prev_time != current_time) {

      double complete_udp_packet = (double) size_of_frame;
      double useful_data_only = (double) (size_of_frame - UDPHEADERSIZE);
      double complete_packet = 28.0 + complete_udp_packet;

      double wire_ratio = complete_packet / complete_udp_packet;
      double useful_ratio = useful_data_only / complete_udp_packet;
        
      uint64_t bytes_per_second = bytes_sent - prev_bytes_sent;
      prev_bytes_sent = bytes_sent;
      double rate = ((double) bytes_per_second) / (1024*1024);

      double wire_rate = rate * wire_ratio;
      double useful_rate = rate * useful_ratio;
             
      multilog(log,LOG_INFO,"%5.2f MB/s\t%5.2f MB/s\t%"PRIu64"\t%5.2f\n",
                            wire_rate, useful_rate,data_counter,sleep_time);
    }

    StopWatch_Delay(&wait_sw, sleep_time);

  }

  close(udpfd);
  return 0;
}


void signal_handler(int signalValue) {

  exit(EXIT_SUCCESS);

}

int sendPacket(int sockfd, struct sockaddr_in their_addr, char *udp_data, int size_of_frame) {

  int numbytes;

  if ((numbytes=sendto(sockfd, udp_data, size_of_frame, 0, (struct sockaddr *)&their_addr, sizeof(struct sockaddr))) < 0) {
    perror("sendto");
    exit(1);
  }
  
  return(numbytes);
}

void usage() {
  fprintf(stdout,
    "test_boardudp [options] udpdb_client\n"
    "\t-s n          send frames of size n bytes* [default %d]\n"
    "\t-p n          udp port to send packets to [default %d]\n"
    "\t-n n          number of seconds to transmit [default 5]\n"
    "\t-l n          period (in bytes) of triangular wave [default 512]\n"
    "\t-r n          data rate (MB/s) not including headers [default 64]\n"
    "\t-v            verbose output\n"
    "\t-d            daemon mode. expects TCP/IP control\n"
    "\n\tudpdb_client is the hostname of a machine running dada_udpdb\n\n"
    ,
    (UDPHEADERSIZE + DEFAULT_UDPDATASIZE), DADA_DEFAULT_UDPDB_PORT);
}


struct in_addr *atoaddr(char *address) {
  struct hostent *host;
  static struct in_addr saddr;

  /* First try it as aaa.bbb.ccc.ddd. */
  saddr.s_addr = inet_addr(address);
  if ((int) saddr.s_addr != -1) {
    return &saddr;
  }
  host = gethostbyname(address);
  if (host != NULL) {
    return (struct in_addr *) *host->h_addr_list;
  }
  return NULL;
}

void encode_data(char * udp_data, int size_of_frame, int value) {

  int i = 0;
  char c = (char) value;
  //printf("sending int %d as char %d\n",value,c);
  for (i=0; i<size_of_frame; i++) {
    udp_data[i] = c;
  }
}


int encode_tri_data(char * udp_data, int size_of_frame, char * tri_data, 
                    int tri_datasize, int tri_counter) {

  int i = 0;
  int j = tri_counter;

  for (i=0; i < size_of_frame; i++) {

    /* If we have reached the end of the tri array */
    if (j == tri_datasize) {
      j = 0;
    }

   udp_data[i] = tri_data[j]; 

   j++;
  }

  return j;

}



void generate_triwave(char *array, int array_size, int nbits) {

  // The triangle wave will fill the array, and the mid value
  // will be 2*nbits -  (i.e. for 8 bit 0 > 255) 

  // assume 8 bits for the moment
        
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


void print_triwave(char *array, int array_size, int nbits) {

  int i = 0;
  unsigned int val = 0;

  printf("["); 
  for (i=0; i < array_size; i++) {
    //val << ((sizeof(val)*8)-1);

    val = (unsigned int) array[i];
    val &= 0x000000ff;
    printf("%u", val);
    if (i != (array_size - 1)) printf(",");
  }
  printf("]\n");

}


void displayBits(unsigned int value) {
  unsigned int c, displayMask = 1 << 31;

  printf("%7u = ", value);

  for (c=1; c<= 32; c++) {
    putchar(value & displayMask ? '1' : '0');
    value <<= 1;

    if (c% 8 == 0)
      putchar(' ');
  }    
  putchar('\n');
}

void displayCharBits(char value) {
  char c, displayMask = 1 << 7;
                                                                                
  printf("    %c = ", value);
                                                                                
  for (c=1; c<= 8; c++) {
    putchar(value & displayMask ? '1' : '0');
    value <<= 1;
                                                                                
  }
  putchar('\n');
}



