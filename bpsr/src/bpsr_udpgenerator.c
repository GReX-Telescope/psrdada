#include <stdlib.h> 
#include <stdio.h> 
#include <errno.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/stat.h>
#include <netinet/in.h> 
#include <netdb.h> 
#include <sys/socket.h> 
#include <sys/wait.h> 
#include <sys/timeb.h> 
#include <math.h>
#include <pthread.h>
#include <fcntl.h>


#include "ascii_header.h"
#include "sock.h"
#include "daemon.h"
#include "multilog.h"
#include "bpsr_def.h"
#include "bpsr_udpdb.h"
#include "bpsr_udp.h"

#include "arch.h"
#include "Statistics.h"
#include "RealTime.h"
#include "StopWatch.h"

#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)

#define IBOB_MANAGER_PORT 52013
#define BROADCAST_IP "192.168.1.255"

typedef struct {

  char ** hosts;

  char ** beams;

  char * ibob_state;

  unsigned n_ibob;

  unsigned verbose;

  unsigned int ** pol0_accs;

  unsigned int ** pol1_accs;

  unsigned int current_array;

} udpgen_t;

// Globals for thread control
unsigned quit_threads = 0;
unsigned int scale0 = 4096;
unsigned int scale1 = 4096;
int bit_value = 0;
int regenerate_signal = 0;
int reselect_bits = 0;
int reset_sequence = 0;
int num_arrays = 1;


// Generation & encoding functions
void generate_dfb_vals(int noise, unsigned int ** p0d, unsigned int ** p1d);

void scale_dfb_vals (int noise, unsigned int scale0, unsigned int scale1,
                     unsigned int ** p0_out, unsigned int ** p1_out,
                     unsigned int ** p0_in, unsigned int ** p1_in);

void sql_vals (unsigned int ** p0_out, unsigned int ** p1_out,
               unsigned int ** p0_in, unsigned int ** p1_in);

void accumulate_vals (unsigned int acc_len,
                      unsigned int ** p0_out, unsigned int ** p1_out,
                      unsigned int ** p0_in, unsigned int ** p1_in);

void bit_select(int bit_value,
                char ** p0_out, char ** p1_out,
                unsigned int ** p0_in, unsigned int ** p1_in);

void pack_pols (char ** packed, char ** p0_in, char ** p1_in);

void build_udp_arrays(uint64_t seq_no, char ** udp_datas, char ** packed_signals);

/* Debug / display functions */
void print_pol(unsigned int * pol);
void print_pol_chars(char * pol);
void displayBits(unsigned int value);
void displayCharBits(char value);
void convert_to_bits(unsigned int value, char * binary_string);
void print_bram_line(FILE * fp, unsigned int i, unsigned int value, char * binary);

int sendPacket(int sockfd, struct sockaddr_in addr, char *data, int size);
//struct in_addr *atoaddr(char *address);
void signal_handler(int signalValue);
void usage();
void quit();
void multibob_simulator (void * arg);
void create_udpout_socket(int *fd, struct sockaddr_in * dagram_socket,
                       char *client, int port, int broadcast);

int main(int argc, char *argv[])
{

  /* number of microseconds between packets */
  double sleep_time = 22;
 
  /* Size of UDP payload. Optimally 1472 (1500 MTU) or 8948 (9000 MTU) bytes */
  uint64_t size_of_frame = BPSR_UDP_PAYLOAD_BYTES;

  /* Accumulation length */
  int acc_len = BPSR_DEFAULT_ACC_LEN;

  /* be verbose? */ 
  int verbose = 0;

  /* udp port to send data to */
  int udp_port = BPSR_DEFAULT_UDPDB_PORT;

  /* total time to transmit for */ 
  uint64_t transmission_time = 5;   

  /* be a daemon? */
  int daemon = 0;

  /* DADA logger */
  multilog_t *log = 0;

  /* The generated signal arrays */
  char * pol0_signal;
  char * pol1_signal;

  // produce noise
  int produce_noise = 0;

  // broadcast udp packets
  int broadcast = 0;

  double tsamp = 0;
  double spectra_per_usec = 0;
  double data_rate= 0;

  /* start of transmission */
  time_t start_time;

  /* end of transmission */
  time_t end_time;

  opterr = 0;
  int c;
  while ((c = getopt(argc, argv, "a:bdjp:t:v")) != EOF) {
    switch(c) {

      case 'a':
        acc_len = atoi(optarg);
        break;

      case 'b':
        broadcast = 1;
        break;

      case 'd':
        daemon = 1;
        break;
        
      case 'j':
        produce_noise = 1;
        num_arrays = 10;
        break;

      case 'p':
        udp_port = atoi(optarg);
        break;

      case 't':
        transmission_time = atoi(optarg);
        break;

      case 'v':
        verbose = 1;
        break;

      default:
        usage();
        return EXIT_FAILURE;
        break;
    }
  }

  int num_args = (argc - optind);

  // Check arguements
  if (num_args < 1) {
    fprintf(stderr,"no dest host:ports were specified\n");
    usage();
    return EXIT_FAILURE;
  }

  char ** hosts = (char **) malloc (sizeof(char *) * num_args);
  int ports[num_args];
  int i = 0;

  // parse the hosts:ports for destination UDP data
  for (i=0; i<num_args; i++)
  {
    hosts[i] = (char *) malloc(sizeof(char) * HOST_NAME_MAX);
    if (sscanf(argv[optind+i], "%[^:]:%d", hosts[i], &(ports[i])) != 2)
    {
      fprintf(stderr, "could not parse host:port [%s,%d] from %s\n", hosts[i], ports[i], argv[optind+i]);
      usage();
      return EXIT_FAILURE;
    } 
  }

  int num_ports = num_args;

  // if broadcast, check for duplicate ports
  if (broadcast)
  {
    int j = 0;
    num_ports = 0;
    for (i=0; i<num_args; i++)
    {
      for (j=0; j<num_args; j++)
      {
        if ((i != j) && (ports[i] > 0) && (ports[i] == ports[j]))
          ports[j] = 0;
      }
    }
    // now setup numports
    for (i=0; i<num_args; i++)
      if (ports[i] > 0)
        num_ports++;
  }

  // Determine the tsamp for given acc_len
  spectra_per_usec= ((double) BPSR_IBOB_CLOCK) / (double) (BPSR_IBOB_NCHANNELS * acc_len);
  tsamp = 1 / spectra_per_usec;
  data_rate = spectra_per_usec * 1000000 * BPSR_UDP_DATASIZE_BYTES;

  if (verbose)
    fprintf(stderr, "Tsamp=%10.8f, data_rate=%f total_rate=%f\n",tsamp, data_rate, (data_rate * num_ports)/(1024*1024));

  if (data_rate * num_ports > 120000000)
  {
    fprintf(stderr, "Data rate too high for a 1Gbit connection: %f bytes per second * %d streams\n", data_rate, num_ports);
    return EXIT_FAILURE;
  }

  log = multilog_open ("bpsr_udpgenerator", daemon);

  if (daemon) {
    be_a_daemon();
  } else {
    multilog_add(log, stderr);
  }

  multilog_serve (log, BPSR_TEST_TRIWAVEUDP_LOG);

  signal(SIGINT, signal_handler);

  int fds[num_ports];
  struct sockaddr_in dagram_sockets[num_ports];

  // UDP socket
  //struct sockaddr_in dagram_socket;

  /* create the socket for outgoing UDP data */
  for (i=0; i<num_ports; i++)
    dada_udp_sock_out (&(fds[i]), &(dagram_sockets[i]), hosts[i], ports[i], broadcast, BROADCAST_IP);

  if (verbose) 
    multilog(log,LOG_INFO,"udp socket setup\n");

  uint64_t data_counter = 0;

  // initialise data rate timing library
  StopWatch wait_sw;
  RealTime_Initialise(1);
  StopWatch_Initialise(1);

  if (verbose)
    multilog(log,LOG_INFO,"Tsamp = %10.8f, data_rate = %f\n",tsamp, data_rate);

  // If we have a desired data rate, then we need to adjust our sleep time
  if (data_rate > 0) 
  {
    double packets_per_second = ((double) data_rate) / ((double) BPSR_UDP_PAYLOAD_BYTES);
    sleep_time = (1.0/packets_per_second)*1000000.0;
  }

  /* Generate signals */
  srand ( time(NULL) );

  // allocate memory for arrays
  unsigned int ** pol0_dfbs    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol1_dfbs    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol0_scaleds = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol1_scaleds = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol0_slds    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol1_slds    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol0_accs    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  unsigned int ** pol1_accs    = (unsigned int **) malloc(sizeof(unsigned int *) * num_arrays);
  char ** pol0_bits            = (char **) malloc(sizeof(char *) * num_arrays);
  char ** pol1_bits            = (char **) malloc(sizeof(char *) * num_arrays);
  char ** packed_signals       = (char **) malloc(sizeof(char *) * num_arrays);
  char ** udp_datas            = (char **) malloc(sizeof(char *) * num_arrays);

  for (i=0; i<num_arrays; i++)
  {
    pol0_dfbs[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol1_dfbs[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol0_scaleds[i]   = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol1_scaleds[i]   = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol0_slds[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol1_slds[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol0_accs[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol1_accs[i]      = (unsigned int *) malloc(sizeof(unsigned int) * BPSR_IBOB_NCHANNELS);
    pol0_bits[i]      = (char *) malloc(sizeof(char *) * BPSR_IBOB_NCHANNELS);
    pol1_bits[i]      = (char *) malloc(sizeof(char *) * BPSR_IBOB_NCHANNELS);
    packed_signals[i] = (char *) malloc(sizeof(char *) * BPSR_IBOB_NCHANNELS * 2);
    udp_datas[i]      = (char *) malloc(sizeof(char *) * BPSR_UDP_PAYLOAD_BYTES);
  }
 
  // generate initial 8 bit samples 
  generate_dfb_vals(produce_noise, pol0_dfbs, pol1_dfbs);

  // scale with scaling factor
  //scale_dfb_vals(produce_noise, scale0, scale1, pol0_scaleds, pol1_scaleds, pol0_dfbs, pol1_dfbs);

  // square law detect
  //sql_vals(pol0_slds, pol1_slds, pol0_scaleds, pol1_scaleds);

  //accumulate_vals(acc_len, pol0_accs, pol1_accs, pol0_slds, pol1_slds);

  //bit_select(bit_value, pol0_bits, pol1_bits, pol0_accs, pol1_accs);

  bit_select(bit_value, pol0_bits, pol1_bits, pol0_dfbs, pol1_dfbs);

  pack_pols(packed_signals, pol0_bits, pol1_bits);

  // Always start the sequence no nice and high 
  uint64_t sequence_incr = 512 * (uint64_t) acc_len;
  uint64_t sequence_no = 100000000 * sequence_incr;

  build_udp_arrays(sequence_no, udp_datas, packed_signals);

  // If we are running as a daemon, then we must wait for a "start" message
  int64_t total_bytes_to_send = data_rate * transmission_time;

  uint64_t bytes_sent = 0;
  uint64_t total_bytes_sent = 0;

  uint64_t bytes_sent_thistime = 0;
  uint64_t prev_bytes_sent = 0;
  
  time_t current_time = time(0);
  time_t prev_time = time(0);

  if (verbose)
  {
    multilog(log,LOG_INFO,"Total bytes to send = %"PRIi64"\n",total_bytes_to_send);
    multilog(log,LOG_INFO,"Packet size = %"PRIu64" bytes\n",BPSR_UDP_PAYLOAD_BYTES);
    multilog(log,LOG_INFO,"Data size = %"PRIu64" bytes\n",BPSR_UDP_DATASIZE_BYTES);
    multilog(log,LOG_INFO,"Wire Rate\t\tUseful Rate\tPacket\tSleep Time\n");
  }

  // Start the external control thread
  pthread_t gain_thread;

  udpgen_t udpgen;
  udpgen.n_ibob = num_args;
  udpgen.hosts = (char **) malloc (sizeof(char *) * udpgen.n_ibob);
  udpgen.beams = (char **) malloc (sizeof(char *) * udpgen.n_ibob);
  for (i=0; i<udpgen.n_ibob; i++)
  {
    udpgen.hosts[i] = (char *) malloc (sizeof (char) * 64);
    sprintf(udpgen.hosts[i], "roach%02d", (i+1));
    udpgen.beams[i] = (char *) malloc (sizeof (char) * 3);
    sprintf(udpgen.beams[i], "%02d", (i+1));
  }
  udpgen.ibob_state = (char *) malloc (sizeof (char) * 64);
  udpgen.verbose = verbose;
  udpgen.pol0_accs = pol0_dfbs;
  udpgen.pol1_accs = pol1_dfbs;
  udpgen.current_array = 0;

  if (pthread_create (&gain_thread, 0, (void *) multibob_simulator, (void *) &udpgen) < 0) 
  {
    perror ("nexus_connect: Error creating new thread");
    return -1;
  }
                                                                                                        
  pthread_detach(gain_thread);

  while (!quit_threads && ((total_bytes_to_send == 0) || (total_bytes_sent < total_bytes_to_send)))
  {
    if (daemon) 
    {
      while (current_time >= start_time)
        current_time = time(0);
      daemon=0;
    }

    StopWatch_Start(&wait_sw);

    // If more than one option, choose the array to use
    if (num_arrays > 1)
      udpgen.current_array = (int) (num_arrays * (rand() / (RAND_MAX + 1.0)));

    /* If the control thread has asked us to reset the sequence number */
    if (reset_sequence)
    {
      sequence_no = 0;
      reset_sequence = 0;
    }

    // put the sequence no in the first 8 bytes
    encode_header(udp_datas[udpgen.current_array], sequence_no);

/*
    if (regenerate_signal) {
      //fprintf(stderr, "regenerating signal\n");
      generate_dfb_vals(produce_noise);
      scale_dfb_vals(scale0, scale1); // Can be updated by the gain thread 
      sql_vals();
      accumulate_vals(acc_len);
      bit_select(bit_value);
      pack_pols();
      build_udp_arrays(sequence_no);
      //fprintf(stderr, "done\n");
      regenerate_signal = 0;
    }

    if (reselect_bits) {
      //fprintf(stderr, "reselecting bits\n");
      bit_select(bit_value);
      pack_pols();
      build_udp_arrays(sequence_no);
      //fprintf(stderr, "done\n");
      reselect_bits = 0;
    }
*/

    for (i=0; i<num_ports; i++)
      bytes_sent = sendPacket(fds[i], dagram_sockets[i], udp_datas[udpgen.current_array], BPSR_UDP_PAYLOAD_BYTES); 

    if (bytes_sent != BPSR_UDP_PAYLOAD_BYTES) 
      multilog(log,LOG_ERR,"Error. Attempted to send %d bytes, but only "
                           "%"PRIu64" bytes were sent\n",BPSR_UDP_PAYLOAD_BYTES,
                           bytes_sent);

    // This is how much useful data we actaully sent
    total_bytes_sent += (bytes_sent - BPSR_UDP_COUNTER_BYTES);

    data_counter++;
    prev_time = current_time;
    current_time = time(0);
    
    if (verbose && (prev_time != current_time))
    {
      double complete_udp_packet = (double) bytes_sent;
      double useful_data_only = (double) (bytes_sent - BPSR_UDP_COUNTER_BYTES);
      double complete_packet = 28.0 + complete_udp_packet;

      double wire_ratio = complete_packet / complete_udp_packet;
      double useful_ratio = useful_data_only / complete_udp_packet;
        
      uint64_t bytes_per_second = total_bytes_sent - prev_bytes_sent;
      prev_bytes_sent = total_bytes_sent;
      double rate = ((double) bytes_per_second) / (1024*1024);

      double wire_rate = rate * wire_ratio;
      double useful_rate = rate * useful_ratio;
             
      multilog(log,LOG_INFO,"%"PRIu64": %5.2f MB/s  %5.2f MB/s  %"PRIu64"  %5.2f, %"PRIu64"\n",
                            (sequence_no/sequence_incr), wire_rate, useful_rate,data_counter,sleep_time,bytes_sent);
    }

    sequence_no += sequence_incr;

    StopWatch_Delay(&wait_sw, sleep_time);

  }

  uint64_t packets_sent = sequence_no / sequence_incr;

  multilog(log, LOG_INFO, "Sent %"PRIu64" bytes\n",total_bytes_sent);
  multilog(log, LOG_INFO, "Sent %"PRIu64" packets\n",packets_sent);

  // free memory 
  for (i=0; i<num_arrays; i++)
  {
    free(pol0_dfbs[i]);
    free(pol1_dfbs[i]);
    free(pol0_scaleds[i]);
    free(pol1_scaleds[i]);
    free(pol0_slds[i]);
    free(pol1_slds[i]);
    free(pol0_accs[i]);
    free(pol1_accs[i]);
    free(pol0_bits[i]);
    free(pol1_bits[i]);
    free(packed_signals[i]);
    free(udp_datas[i]);
  }
  free(pol0_dfbs);
  free(pol1_dfbs);
  free(pol0_scaleds);
  free(pol1_scaleds);
  free(pol0_slds);
  free(pol1_slds);
  free(pol0_accs);
  free(pol1_accs);
  free(pol0_bits);
  free(pol1_bits);
  free(packed_signals);
  free(udp_datas);

  for (i=0; i<udpgen.n_ibob; i++)
  {
    free(udpgen.hosts[i]);
    free(udpgen.beams[i]);
  }
  free(udpgen.hosts);
  free(udpgen.beams);
  free(udpgen.ibob_state);

  for (i=0; i<num_ports; i++)
    close(fds[i]);

  for (i=0; i<num_args; i++)
    free(hosts[i]);
  free(hosts);
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
    "bpsr_udpgenerator [options] host1:port1 [host2:port2]\n"
    "\t-a acc_len    accumulation length [default %d]\n"
    "\t-b            broadcast packets to %s\n"
    "\t-j            produce random noise [default off]\n"
    "\t-t secs       number of seconds to transmit [default 5]\n"
    "\t-v            verbose output\n"
    "\t-d            daemon mode. expects TCP/IP control\n"
    "\n\tdest_host     hostname of a machine running dada_udpdb\n\n"
    ,BPSR_DEFAULT_ACC_LEN, BROADCAST_IP);
}

/*
 * Debug & Display functions
 */
void print_pol(unsigned int * pol) {

  int i = 0;
  unsigned int val = 0;

  fprintf(stderr,"["); 

  for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {

    fprintf(stderr,"%u,",pol[i]);

    //val = (unsigned int) pol[i];
    //val &= 0x000000ff;
    //fprintf(stderr,"%u.", val);

  }

  fprintf(stderr,"]\n");

}

void print_pol_chars(char * pol) {

  int i = 0;
  unsigned int val = 0;

  fprintf(stderr,"[");

  for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {

    val = (unsigned int) pol[i];
    val &= 0x000000ff;
    fprintf(stderr,"%u.", val);

  }
 
  fprintf(stderr,"]\n");
  
}


void convert_to_bits(unsigned int value, char * binary_string) {

  unsigned int c, displayMask = 1 << 31;
  for (c=1; c<= 32; c++) {
    binary_string[(c-1)] = value & displayMask ? '1' : '0';
    value <<= 1;
  }
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
                                                                                
  //printf("    %c = ", value);
                                                                                
  for (c=1; c<= 8; c++) {
    putchar(value & displayMask ? '1' : '0');
    //if (c%2 == 0) printf(" ");
    value <<= 1;
                                                                                
  }
  //putchar('\n');
}

/*
 * Provide a telnet style interface to allow high level IBOB commands:
 *
 *  acclen <unsigned>   Controls data rate
 *  levels              Performs level setting 
 *  arm                 reset the sequence number
 *  state               returns state string of form IBOBXX host:port [dead|closed|alive|active]
 */
void multibob_simulator (void * ctx) 
{

  udpgen_t * udpgen = (udpgen_t *) ctx;

  unsigned verbose = udpgen->verbose;

  FILE *sockin = 0;
  FILE *sockout = 0;

  // Port for control from tcs simulator
  int port = IBOB_MANAGER_PORT;

  int listen_fd;
  int fd;
  char *rgot;

  int bufsize = 4096;
  char* buffer = (char*) malloc ( bufsize);
  assert (buffer != 0);
  char * binary_string = (char*) malloc(sizeof(char) * 33);
  binary_string[32] = '\0';

  const char* whitespace = " \r\t\n";
  char * command = 0;
  char * args = 0;

  int acc_len = 0;
  int trying_to_reset = 0;
  int i=0;
  int have_connection = 0;

  // for selecting on socket
  int ret = 0;
  int to_read = 0;
  int to_write = 0;

  time_t curr = time(0);
  time_t prev = curr - 5;

  // create a socket on which to listen
  if (verbose)
    fprintf(stderr, "multibob_simulator: creating socket\n");
  listen_fd = sock_create (&port);
  if (listen_fd < 0)
  {
    fprintf(stderr,"Error creating socket: %s\n", strerror(errno));
    quit_threads = 1;
  }

  if (verbose)
    fprintf(stderr, "multibob_simulator: listening on port %d\n", port);

  sprintf(udpgen->ibob_state, "alive");

  while (!quit_threads) 
  {
    if (!have_connection) 
    {
      to_read = 0;
      to_write = 0;
      ret = sock_ready (listen_fd, &to_read, &to_write, 1);

      if (ret && to_read)
      {
        // wait for a connection to accept
        fd =  sock_accept (listen_fd);
        if (fd < 0)  {
          perror("Error accepting connection");
          quit_threads = 1;
        }

        if (verbose)
          fprintf(stderr, "multibob_simulator: accepted connection\n");

        sockin = fdopen(fd,"r");
        if (!sockin) 
          perror("Error creating input stream");

        sockout = fdopen(fd,"w");
        if (!sockout) 
          perror("Error creating output stream");

        if (verbose)
          fprintf(stderr, "multibob_simulator: fds opened\n");

        setbuf (sockin, 0);
        setbuf (sockout, 0);
  
        have_connection = 1;

        fprintf(sockout, "welcome to bpsr_udpgenerator!\n");
      }
    }

    // wait for input on sockin for 1 second
    ret = 0;
    to_read = 0;
    to_write = 0;

    if (have_connection)
    {
      ret = sock_ready (fd, &to_read, &to_write, 1);
    }

    // if there is something to be read on the socket, read it
    if (ret && to_read)
    {
      // wait for text based commands on sockin
      rgot = fgets (buffer, bufsize, sockin);

      if (rgot && !feof(sockin)) 
      {

        // assume telnet style \r\n terminator
        buffer[strlen(buffer)-2] = '\0';
        if (verbose)
          fprintf(stderr, "multibob_simulator: <- '%s'\n", buffer);
  
        args = buffer;
        command = strsep (&args, whitespace);

        // acc_len command - set the data rate based on this
        if (strcmp(command, "acclen") == 0) 
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing acc_len command\n");
          if (sscanf(args, "%d", &acc_len) != 1)
          {
            fprintf (stderr, "multibob_simulator: failed to parse acc_len from %s\n", buffer);
            acc_len = 24;
          }
          fprintf(sockout, "ok\n");
        }

        // levels command - doesn't do anything, just here for conformance
        else if (strcmp(command, "levels") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing levels command\n");
          for (i=0; i<udpgen->n_ibob; i++)
            fprintf(sockout, "%s : %u [%u,%u]\n", udpgen->hosts[i], 0, 4096, 4096); 
          fprintf(sockout, "ok\n");
        }

        else if (strcmp(command, "arm") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing arm command\n");
          time_t arm_curr = time(0);
          time_t arm_prev = arm_curr;
          while (arm_curr == arm_prev)
            arm_curr = time(0);

          // sleep for precisely 0.5 seconds
          struct timeval timeout;
          timeout.tv_sec=0;
          timeout.tv_usec=500000;

          // sleep for 0.5 seconds
          select(0,NULL,NULL,NULL,&timeout);

          // increment by 1 and return UTC_START in DADA time format
          arm_curr++;
          char date[64];
          strftime (date, 64, DADA_TIMESTR, gmtime(&arm_curr));
          fprintf (sockout, "%s\n", date);
          fprintf(sockout, "ok\n");

          // wait for next 1s tick before resetting packet sequence
          while (arm_curr == arm_prev)
            arm_curr = time(0);
          reset_sequence = 1;
        }

        else if (strcmp(command, "state") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing state command\n");
          int port = 40400;
          for (i=0; i<udpgen->n_ibob; i++)
            fprintf(sockout, "IBOB%02d %s:%d %s\n", (i+1), udpgen->hosts[i], (port + i), udpgen->ibob_state);
          fprintf(sockout, "ok\n");
        }

        else if (strcmp(command, "close") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing close command\n");
          sprintf(udpgen->ibob_state, "closed");
          fprintf(sockout, "ok\n");
        }

        else if (strcmp(command, "open") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing close command\n");
          sprintf(udpgen->ibob_state, "alive");
          fprintf(sockout, "ok\n");
        }

        else if (strcmp(command, "quit") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing quit command\n");
          quit_threads = 1;
          fprintf(sockout, "ok\n");
        }

        else if (strcmp(command, "exit") == 0)
        {
          if (verbose)
            fprintf(stderr, "multibob_simulator: processing exit command\n");
          fprintf(sockout, "ok\n");
        }

        else 
        {
          fprintf(stderr, "multibob_simulator: unrecognized command %s\n", buffer);
          fprintf(sockout, "unrecognized command\n");      
          fprintf(sockout, "fail\n");      
        }
      }
      else
      {
        if (verbose)
          fprintf(stderr, "multibob_simulator: lost connection\n");
        have_connection = 0 ;
      }
    } 

    // check if its time to bramdump or not
    curr = time(0);
    if (curr == prev + 10)
    {
      if (verbose)
        fprintf(stderr, "multibob_simulator: bramdisk()\n");
      bramdisk(udpgen);
      prev = curr;
    } 

  }

  close(fd);
  close(listen_fd);

}

/*
 * Generates 8 bit values for the DFB stage
 */
void generate_dfb_vals(int noise, unsigned int ** p0, unsigned int ** p1) 
{
  float avg_value = 32;
  float max_value = 8;
  int i,h;

  for (h=0; h < num_arrays; h++) {
  
    for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {
    
      if (noise) {
        p0[h][i] = (int) (avg_value + (64* (float) ((rand() / (RAND_MAX + 1.0))))); 
        p1[h][i] = (int) (avg_value + (64 * (float) ((rand() / (RAND_MAX + 1.0))))); 

        // fake filter
        if ((i < 100) || (i > BPSR_IBOB_NCHANNELS - 100))
        {
          p0[h][i] = 0;
          p1[h][i] = 0;
        }

      } else {
        float pos = ((float) i) / ((float) BPSR_IBOB_NCHANNELS);
        float y = 2 * (0.5 - fabs(pos - 0.5));
        p0[h][i] = (unsigned int) (max_value * (y+0.5/max_value));
        p1[h][i] = (max_value - p0[h][i]);
      }
    }
  }
}

/*
 * Generates pol?_scaled arrays from pol?_dfb arrays, multipling
 * by the scale factors for each polarisation
 */
void scale_dfb_vals(int noise, unsigned int scale0, unsigned int scale1, 
                    unsigned int ** p0_out, unsigned int ** p1_out, 
                    unsigned int ** p0_in, unsigned int ** p1_in)
{
  int i=0;
  int h=0;
  int s = 0;
  for (h=0; h < num_arrays; h++) {
    for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {
      if (noise)
        s = num_arrays * ((int) (rand() / (RAND_MAX + 1.0)));
      else
        s = h;

      p0_out[h][i] = (int) ( ((float) scale0 / 4096.0) * p0_in[s][i]);
      p1_out[h][i] = (int) ( ((float) scale1 / 4096.0) * p1_in[s][i]);
    }
  }
}

/*
 * Generates sld arrays. simply squares the scaled arrays.
 */
void sql_vals(unsigned int ** p0_out, unsigned int ** p1_out,
              unsigned int ** p0_in, unsigned int ** p1_in)
{

  int i=0;
  int h=0;

  for (h=0; h < num_arrays; h++) {
    for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {
      p0_out[h][i] = p0_in[h][i] * p0_in[h][i];
      p1_out[h][i] = p1_in[h][i] * p1_in[h][i];
    }
  }
}

/*
 * Accumulates the values by the specified acc_len
 * simply multiply the values by the acclen
 */
void accumulate_vals (unsigned int acc_len,
                      unsigned int ** p0_out, unsigned int ** p1_out,
                      unsigned int ** p0_in, unsigned int ** p1_in)
{
  int i=0;
  int h=0;
  for (h=0; h < num_arrays; h++) {
    for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {
      p0_out[h][i] = acc_len * p0_in[h][i];
      p1_out[h][i] = acc_len * p1_in[h][i];
    }
  }

}

/*
 * Selects the relevant 8 bit segment from the 32 bit
 * arrays. bit_value can be 0,1,2,3 which corresponds
 * to bits 0-7, 8-15, 16-23, 24-31 respectively
 */
void bit_select(int bit_value,
                char ** p0_out, char ** p1_out,
                unsigned int ** p0_in, unsigned int ** p1_in)
{
  unsigned int shifted;
  char val;
  int i=0;
  int h=0;

  for (h=0; h < num_arrays; h++)
  {
    for (i=0; i < BPSR_IBOB_NCHANNELS; i++)
    {
      shifted = p0_in[h][i];
      shifted >>= (8*bit_value);
      shifted &= 0x000000ff;
      p0_out[h][i] = (char) shifted;;

      shifted = p1_in[h][i];
      shifted >>= (8*bit_value);
      shifted &= 0x000000ff;
      p1_out[h][i] = (char) shifted;;

      //if ((h == 0) && (i < 10))
      //{
      //  fprintf(stderr, "p0_out[%d][%d]=%d\n", h, i, (int) p0_out[h][i]);
      //  fprintf(stderr, "p1_out[%d][%d]=%d\n", h, i, (int) p1_out[h][i]);
      //}
    }
  }
}

/*
 * Packs the 2 polarisations into the char 
 * buffer ready for udp despatch
 */
void pack_pols(char ** packed, 
               char ** p0_in, char ** p1_in)
{

  int i=0; 
  int h=0;
                                                                                                         
  for (h=0; h < num_arrays; h++) {
    for (i=0; i < (BPSR_IBOB_NCHANNELS/2); i++) {
      packed[h][4*i]     = p0_in[h][2*i];
      packed[h][(4*i)+1] = p0_in[h][(2*i)+1];
      packed[h][(4*i)+2] = p1_in[h][2*i];
      packed[h][(4*i)+3] = p1_in[h][(2*i)+1];

      //fprintf(stderr, "packed[%d][%d] = %d\n", h, (4*i), (int) packed[h][4*i]);
      //fprintf(stderr, "packed[%d][%d] = %d\n", h, (4*i)+1, (int) packed[h][(4*i)+1]);
      //fprintf(stderr, "packed[%d][%d] = %d\n", h, (4*i)+2, (int) packed[h][(4*i)+1]);
      //fprintf(stderr, "packed[%d][%d] = %d\n", h, (4*i)+3, (int) packed[h][(4*i)+1]);
    }
  }

}

/* 
 * Pack the UDP arrays
 */
void build_udp_arrays(uint64_t seq_no, char ** udp_datas, char ** packed_signals)
{
  int h=0;

  for (h=0; h < num_arrays; h++) 
  {
    // Encode the 2 pols into the udp_data char array
    encode_header((udp_datas[h]), seq_no);
    memcpy((udp_datas[h])+BPSR_UDP_COUNTER_BYTES, packed_signals[h], (size_t) BPSR_UDP_DATASIZE_BYTES);
  }

}

/*
 * Prints a line to FP in the same format as the ibob's bramdump command 
 */
void print_bram_line(FILE *fp, unsigned int i, unsigned int value, char * binary)
{
  fprintf(fp, "0x%04X / %05d -> 0x%08X / 0b%s / %010u\r\n",i,i,i,binary,value);
}

int bramdisk(udpgen_t * ctx)
{
  int fd = 0;
  int flags = O_CREAT | O_WRONLY | O_TRUNC;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;

  char fname[128];
  char time_str[64];

  time_t now = time(0);
  strftime (time_str, 64, DADA_TIMESTR, localtime(&now));

  unsigned bit_window = 0;
  float * pol1 = (float *) malloc(sizeof(float) * (BPSR_IBOB_NCHANNELS/2));
  float * pol2 = (float *) malloc(sizeof(float) * (BPSR_IBOB_NCHANNELS/2));
  int h = ctx->current_array;

  if (!pol1 || !pol2)
  {
    fprintf(stderr, "multibob_simulator: could not malloc, oh dear me\n");
    return -1;
  }

  unsigned int i = 0;
  for (i=0; i<ctx->n_ibob; i++)
  {
    sprintf(fname, "%s_%s.bramdump.tmp", time_str, ctx->beams[i]);

    if (ctx->verbose)
      fprintf(stderr, "bramdisk: creating %s\n", fname);
    fd = open(fname, flags, perms);
    if (fd < 0)
    {
      fprintf (stderr, "Error opening bramdump file: %s\n", fname);
      return -1;
    }

    unsigned int j=0;
    for (j=0; j<(BPSR_IBOB_NCHANNELS/2); j++)
    {
      pol1[j] = (float) ctx->pol0_accs[h][2*j];
      pol2[j] = (float) ctx->pol1_accs[h][2*j];
    }


    write(fd, &bit_window, sizeof(unsigned));
    write(fd, pol1, sizeof(float) * (BPSR_IBOB_NCHANNELS/2));
    write(fd, pol2, sizeof(float) * (BPSR_IBOB_NCHANNELS/2));

    close(fd);
  }

  free(pol1);
  free(pol2);

  // rename the tmp bramdump file to the correct name
  char real_fname[128];

  for (i=0; i<ctx->n_ibob; i++)
  { 
    sprintf(fname, "%s_%s.bramdump.tmp", time_str, ctx->beams[i]);
    sprintf(real_fname, "%s_%s.bramdump", time_str, ctx->beams[i]);

    // Sometimes it takes a while for the file to appear, wait for it
    flags =  R_OK | W_OK;
    int max_wait = 10;

    while (access(fname, flags) != 0)
    {
      fprintf(stderr, "bramdisk: %s waiting 500ms for bramdump files\n",
                       ctx->beams[i]);
      ibob_pause(500);
      max_wait--;

      if (max_wait == 0)
      {
        fprintf(stderr, "bramdisk: file not on disk after 5 sec\n");
        break;
      }
    }

    if (max_wait)
    {
      char command[256];
      int rval = rename(fname, real_fname);

      if (rval != 0)
      {
        fprintf(stderr, "bramdisk: %s: rename failed: %d\n", ctx->beams[i], rval);
        unlink (fname);
        return -1;
      }
    }
  }
}
