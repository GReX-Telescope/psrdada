#include "dada_def.h"
#include "dada_udpdb.h"
#include "ascii_header.h"
#include "udp.h"

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
void print_header(header_struct header);
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

int sockDatagramFd;
struct sockaddr_in dagram_socket;

union convert {
  int integer;
  char chars[4];
};
        
int main(int argc, char *argv[])
{

  int c;
  int numClients = 0;           // Number of clients we will use
  header_struct header;

  int time_per_frame = 20;
  int size_of_frame = UDPHEADERSIZE + UDPDATASIZE;
  int verbose = 0;
  int udp_port = DADA_DEFAULT_UDPDB_PORT;
  int transmission_time = 5;   // num secs to transmit for
  int tri_datasize = 512;

  /* the filename from which the header will be read */
  char* header_file = 0;
  char* header_buf = 0;
  uint64_t header_size = 8192;
  int fSize = 800000000;

  opterr = 0;
  while ((c = getopt(argc, argv, "t:s:p:n:l:H:S:v")) != EOF) {
    switch(c) {
      case 't':
        time_per_frame = atoi(optarg);
        if (time_per_frame < 0) {
          fprintf(stderr,"time per packet must be >= 0 micro seconds\n");
          usage();
          return EXIT_FAILURE;
        }
        break;

      case 's':
        size_of_frame= atoi(optarg);
        if ((size_of_frame <= 0) || (size_of_frame >= 4096)) {
          fprintf(stderr,"packet size must be between > 0 and < 4096 bytes\n");
          usage();
          return EXIT_FAILURE;
        }
        if (size_of_frame != (UDPHEADERSIZE + UDPDATASIZE)) {
          fprintf(stderr,"Warning this program was compiled with an expected"
                         " frame size of %d bytes\n", 
                         (UDPHEADERSIZE + UDPDATASIZE));
        }
        break;

      case 'n':
        transmission_time = atoi(optarg);
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

      case 'H':
        if (optarg) {
          header_file = optarg;
        } else {
          fprintf(stderr,"Specify a header file\n");
          usage();
        }
        break;

      case 'S':
        if (optarg){
          fSize = atoi(optarg);
          fprintf(stderr,"File size will be %d\n",fSize);
        }
        else usage();
        break;

      default:
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  if (verbose) fprintf(stderr,"Command line  args parsed\n");
  
  numClients = argc - optind;

  if (numClients != 1) {
    fprintf(stderr,"no udpdb_client was specified\n");
    usage();
    return EXIT_FAILURE;
  }

  char *client_name;     // Hostname of the udpdb_client
  client_name = (char *) argv[optind];
  signal(SIGINT, signal_handler);

  // Get the client hostnames...
  struct hostent * client_hostname;
  client_hostname = gethostbyname(client_name);


  // Setup the socket for UDP packets
  int udpfd = socket(PF_INET, SOCK_DGRAM, 0);
  if (udpfd < 0) {  
    perror("udp socket");
    exit(1);
  }

  // Setup the UDP Broadcast address
  struct in_addr *addr;
  dagram_socket.sin_family = AF_INET;      /* host byte order */
  dagram_socket.sin_port = htons(udp_port);  /* short, network byte order */
  addr = atoaddr(client_name);
  dagram_socket.sin_addr.s_addr = addr->s_addr;
  bzero(&(dagram_socket.sin_zero), 8);     /* zero the rest of the struct */

  if (verbose) fprintf(stderr,"udp socket setup\n");

  time_t end_time = time(0)+transmission_time;    // end of transfer
  time_t current_time = time(0);
  time_t prev_time = time(0);

  char udp_data[(UDPHEADERSIZE+UDPDATASIZE)];

  int data_counter = 0;
  int bytes_sent = 0;
  int bytes_sent_thistime = 0;
  int prev_bytes_sent = 0;

  double sleep_time= (double) time_per_frame;
  StopWatch wait_sw;

  int ret;
  int quiet = 1;
  ret = RealTime_Initialise(quiet);
  ret = StopWatch_Initialise(quiet);

  /* Setup udp header information */
  header.length = (int) UDPHEADERSIZE;
  header.source = (char) 3;    // Not sure what this would be
  header.sequence = 0;
  header.bits = 8;
  header.channels = 2;
  header.bands = 1;
  header.bandID[0] = 1;
  header.bandID[1] = 2;
  header.bandID[2] = 3;
  header.bandID[3] = 4;

  if (verbose) fprintf(stderr,"data structures setup\n");

  header_buf = malloc(sizeof(char)*header_size);
  if (header_file) {
    if (fileread (header_file, header_buf, header_size) < 0)  {
      fprintf(stderr,"Could not read header file correctly\n");
      return EXIT_FAILURE; 
    }

    if (ascii_header_get (header_buf, "NBIT", "%d",&(header.bits)) != 1) {
      fprintf(stderr,"Could not read NBIT from header\n");
      exit(EXIT_FAILURE);
    }
    if (ascii_header_get (header_buf, "NCHANNEL", "%d",&header.channels) != 1) {
      fprintf(stderr,"Could not read NCHANNELS from header\n");
      exit(EXIT_FAILURE);
    }
    if (ascii_header_get (header_buf, "NBAND", "%d",&header.bands) != 1) {
      fprintf(stderr,"Could not read NBITS from header\n");
      exit(EXIT_FAILURE);
    }

  }

  char * tri_data = malloc(sizeof(char) * tri_datasize);
  int tri_nbits = 8;
  generate_triwave(tri_data, tri_datasize, tri_nbits);
  print_triwave(tri_data, tri_datasize, tri_nbits);
  int tri_counter = 0;
  
  if (verbose) fprintf(stderr,"Rate\t\tPacket\n");
  while (current_time <= end_time) {

    StopWatch_Start(&wait_sw);

    header.sequence++;
    encode_header(udp_data, &header);
    //print_header(header);
    tri_counter = encode_tri_data(udp_data+UDPHEADERSIZE,(size_of_frame-UDPHEADERSIZE),tri_data,tri_datasize, tri_counter);

    bytes_sent_thistime = sendPacket(udpfd,dagram_socket,udp_data, size_of_frame);
    if (bytes_sent_thistime != size_of_frame) {
      fprintf(stderr,"Did not send correct number of bytes\n");
    } 

    bytes_sent += bytes_sent_thistime;

    data_counter++;
    prev_time = current_time;
    current_time = time(0);
    
    if (verbose) {
      if (prev_time != current_time) {
        int bytes_per_second = bytes_sent - prev_bytes_sent;
        prev_bytes_sent = bytes_sent;
        float rate = (float)bytes_per_second / (1024*1024);
             
        fprintf(stderr,"%5.2f MB/s\t%d\t%5.2f\n",rate,data_counter,sleep_time);
       }
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
    "\t-t n          send a frame every n micro seconds [default 20]\n"
    "\t-s n          send frames of size n bytes* [default %d]\n"
    "\t-p n          udp port to send packets to [default %d]\n"
    "\t-n n          number of seconds to transmit [default 5]\n"
    "\t-l n          period (in bytes) of triangular wave [default 512]\n"
    "\t-H filename   header filename to use for udp header\n"
    "\t-S n          size of header buffer\n"
    "\t-v            verbose output\n"
    "\n\tudpdb_client is the hostname of a machine running dada_udpdb\n\n"
    ,
    (UDPHEADERSIZE + UDPDATASIZE), DADA_DEFAULT_UDPDB_PORT);
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

void print_header(header_struct header) {
  fprintf(stdout,"length = %d\n",header.length);
  fprintf(stdout,"source = %d\n",header.source);
  fprintf(stdout,"sequence= %d\n",header.sequence);
  fprintf(stdout,"bits = %d\n",header.bits);
  fprintf(stdout,"channels = %d\n",header.channels);
  fprintf(stdout,"bands = %d\n",header.bands);
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



