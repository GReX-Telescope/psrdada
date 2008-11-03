#ifdef __APPLE__

#define OSX

#define OPENOPTIONS O_RDONLY

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENOPTIONS O_RDONLY|O_LARGEFILE

#endif

#include <sys/types.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>     
#include <sys/timeb.h>
#include <sys/time.h>       
#include <sys/socket.h>  
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <getopt.h>

typedef struct disktime {
  double         time;
  unsigned long  size;
} disktime;

#define NBUF       3   // Must not be less than 3
#define DEFAULT_BUFSIZE      2
#define DEFAULT_TOTALSIZE  1000
#define MAXSTR              200 

#define DEBUG(x)

double tim(void);
void gettime(disktime **, long, int *, int *);
void kill_signal (int);
int setup_net(int isserver, char *hostname, int port, int window_size, int *sock);
void *netwrite(void *arg);
void throttle_rate (double firsttime, float datarate, 
                    unsigned long long totalsize, double *tbehind);
void usage(void);

#define PRINTSPEED(k, file) { \
  speed = times[k].size/(times[k].time-times[k-1].time)/1000/1000; /* MB/s */\
  fprintf(file, "%6d  %4lld  %9.3f   %6.2f Mbps ( %ld  %6.3f )\n", \
           k, totalsize/1000/1000, times[k].time-firsttime, \
           speed*8, times[k].size, (times[k].time-times[k-1].time)); \
}

volatile int time_to_quit = 0;
volatile int sig_received = 0;

/* Globals needed by network threads */
int  nfullbuf;
int bwrote;        /* Amount of net data written between updates */
int bufsize;
int write_failure;

char *buf[NBUF], *ptr;
ssize_t nbufsize[NBUF];
pthread_mutex_t bufmutex[NBUF];
pthread_mutex_t globalmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t startwait_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  start_wait  = PTHREAD_COND_INITIALIZER;

int main (int argc, char * const argv[]) {
  unsigned short fnamesize;
  int i, status, file, opt, tmp, ntime, maxtimes, ibuf, jbuf, nfile;
  int bread, nupdate;
  //  int foff;
  int sock;
  char msg[MAXSTR+50], *fname;
  double firsttime, dtime;
  disktime *times;
  float ftmp, speed;
  double tbehind;
  unsigned long long accum, totalsize;
  pthread_t netthread;

  int debug = 0;  
  int server = 0;
  float timeupdate = 50; /* Time every 50 MiB by default */
  int port = 8000;     /* TCP port to use */
  int window_size = -1;        
  float datarate = 0;  /* Limit read/write to this data rate */
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */
  struct stat filestat;
  unsigned long long filesize;

  // ajameson
  int read_stdin = 0;
  unsigned long long lltmp; 
  
  struct option options[] = {
    {"port", 1, 0, 'p'},
    {"host", 1, 0, 'H'},
    {"hostname", 1, 0, 'H'},
    {"time", 1, 0, 't'},
    {"window", 1, 0, 'w'},
    {"blocksize", 1, 0, 'b'},
    {"rate", 1, 0, 'z'},
    {"server", 0, 0, 's'},
    {"debug", 0, 0, 'D'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  bufsize   = DEFAULT_BUFSIZE * 1000*1000;
  file = -1;
  i = -1;
  write_failure = 0;

  //nread = 0;

  srand(time(NULL));

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "r:n:DdhH:i:", 
                           options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 'b':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
        fprintf(stderr, "Bad blocksize option %s\n", optarg);
      else 
        bufsize = ftmp * 1000*1000;
     break;

    case 'w':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
        fprintf(stderr, "Bad window option %s\n", optarg);
      else 
        window_size = ftmp * 1024;
     break;
     
    case 'p':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
        fprintf(stderr, "Bad port option %s\n", optarg);
      else {
        port = tmp;
      }
      break;
      
    case 't':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1 || ftmp <= 0.0)
        fprintf(stderr, "Bad time update option %s\n", optarg);
      else 
        timeupdate = ftmp;
      break;
      
    case 'H':
      if (strlen(optarg)>MAXSTR) {
        fprintf(stderr, "Hostname too long\n");
        return(1);
      }
      strcpy(hostname, optarg);
      break;

    case 'r':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
        fprintf(stderr, "Bad rate option %s\n", optarg);
      else 
        datarate = ftmp * 1000*1000/8;
      break;

    case 'D':
      debug = 1;
      break;

    case 's':
      server = 1;
      break;
      
    case 'h':
      usage();
      return(1);
    break;

    case 'i':
      read_stdin = 1;
      status = sscanf(optarg, "%llu", &lltmp);
      if (status!=1) {
        fprintf(stderr, "bad stdin filesize:%s\n", optarg);
        usage();
        return(1);
      }
      else
        filesize = lltmp;
      break;
    
    case '?':
    default:
      break;
    }
  }

  if ((read_stdin) && (argc-optind != 1))
  {
    fprintf(stderr, "  -i flag allows 1 file only (%d files specified)\n", (argc-optind));
    usage();
    return(1);
  }

  if (strlen(hostname) == 0) {
    strcpy(hostname, "localhost");
  }

  nupdate = timeupdate/(bufsize/1e6);
    
  // Initialise buffers and syncronisation variables
  for (i=0; i<NBUF; i++) {
    buf[i] = malloc(bufsize);
    if (buf[i]==NULL) {
      sprintf(msg, "Trying to allocate %d MB", bufsize/1000/1000);
      perror(msg);
      return(1);
    }

    nbufsize[i] = 0;
    pthread_mutex_init(&bufmutex[i], NULL);
  }
  nfullbuf=0;
  
  maxtimes = 500;
  times = malloc(sizeof(disktime)*maxtimes);
  if (times==NULL) {
    perror(NULL);
    return(1);
  }
  
  ntime = 0;
  accum = 0;

  status = setup_net(server, hostname, port, window_size, &sock);
    
  if (status) {
    //free(mem);
    return(1);
  }

  /* Install a ^C catcher */
  signal (SIGINT, kill_signal);
  
  totalsize=0;

  ibuf = 0;

  // Lock the first and last mutex
  pthread_mutex_lock( &bufmutex[ibuf] );
  DEBUG(printf("MAIN:   Locked %d\n", ibuf));
  pthread_mutex_lock( &bufmutex[NBUF-1] );
  DEBUG(printf("MAIN:   Locked %d\n", NBUF-1));

  /* Need to start netwrite thread if network connection */
  status = pthread_create( &netthread, NULL, netwrite, (void *)&sock);
  if (status) {
    printf("Failed to start network write thread: %d\n", status);
    exit(1);    
  }

  printf("Reading in chunks of %d MB\n", bufsize/(1000*1000));

  gettime(&times, 0, &ntime, &maxtimes);
  firsttime = times[0].time;
  tbehind = 0;

  for (nfile=optind; nfile<argc; nfile++) {

    // Remove directories if present
    fname = rindex(argv[nfile], '/');
    if (fname==NULL) {
      fname = argv[nfile];
    } else {
      fname++; // Skip over '/' 
    }

    fnamesize = strlen(fname)+1;
    if (fnamesize==1) {
      printf("Skipping %s\n", argv[nfile]);
      continue;
    }

    if (read_stdin) 
    {
      fprintf(stderr, "reading from stdin\n");
      file = open("/dev/stdin", OPENOPTIONS);
    }
    else 
      file = open(argv[nfile], OPENOPTIONS);

    if (file==-1) {
      if (read_stdin)
        sprintf(msg, "Failed to open /dev/stdin for file (%s) [%d]", 
                argv[nfile], errno);
      else 
        sprintf(msg, "Failed to open input file (%s) [%d]", 
                argv[nfile], errno);
      perror(msg);
      continue;
    }

    /* Fill first buffer with filesize and filename */
    /* 8 byte file size
       2 byte filename size
       n bytes filename 
    */

    if (read_stdin) 
    {
      // filesize set in command line
    }
    else
    {
      status =  fstat(file, &filestat);
      if (status != 0) {
        sprintf(msg, "Trying to stat %s", argv[nfile]);
        perror(msg);
        return(1);
      }

      if (S_ISDIR(filestat.st_mode)) {
        printf("Skipping %s (directory)\n", argv[nfile]);
        close(file);
        continue;
      }

      filesize = filestat.st_size;
    }

    //if (strlen(fname)+1>MAXSHORT) {
    //  fprintf(stderr, "Passed filename too long\n");
    //  return(1);
    //}

    printf("sending %s\n", fname);

    ptr = buf[ibuf];
    memcpy(ptr, &filesize, sizeof(long long));
    ptr += sizeof(long long);
    memcpy(ptr, &fnamesize, sizeof(short));
    ptr += sizeof(short);
    memcpy(ptr, fname, fnamesize);
    nbufsize[ibuf] = (ptr-buf[ibuf])+fnamesize;

    // Increment the number of full buffers
    pthread_mutex_lock(&globalmutex);
    nfullbuf++;
    DEBUG(printf("MAIN:   Nfullbuf=%d\n", nfullbuf));
    pthread_mutex_unlock(&globalmutex);

    jbuf = ibuf;
    ibuf = (ibuf+1)%NBUF;

    // Grab the next mutex, unlock the current one to allow the
    // Net thread to start sending 
    DEBUG( printf("MAIN:   Lock %d\n", ibuf));
    pthread_mutex_lock(&bufmutex[ibuf]);
    DEBUG(printf("MAIN:     and unlock %d\n", (jbuf)));
    pthread_mutex_unlock(&bufmutex[jbuf]);
            
    bread = 0;
    bwrote = 0;

    i=0;

    int bytes_read = 0;
    int have_eof = 0;

    /* Loop over buffers */
    while (1) { // Loop until EOF

      if (time_to_quit || write_failure) break;

      bytes_read = 0;

      if (read_stdin) {
        nbufsize[ibuf] = 0;

        DEBUG( printf("MAIN:   Readinto buffer %d\n", ibuf));
        while ((nbufsize[ibuf] < bufsize) && (!have_eof)) {

          bytes_read = read(file, (buf[ibuf]+(nbufsize[ibuf])), (bufsize-nbufsize[ibuf]));
          if (bytes_read == 0) 
            have_eof = 1;

          nbufsize[ibuf] += bytes_read;

        }

      } else {

        DEBUG( printf("MAIN:   Readinto buffer %d\n", ibuf));
        nbufsize[ibuf] = read(file, buf[ibuf], bufsize);

      } 

      if (nbufsize[ibuf]==0) {
        DEBUG(printf("EOF detected\n"));
        break;
      } else if (nbufsize[ibuf]==-1) {
        perror("Error reading file");
        close(file);
        //free(mem);
        return(1);
      }

      //if (nbufsize[ibuf]!=bufsize) 
      //        printf("Warning: did not read full buffer (%d/%d)\n", 
      //               nbufsize[ibuf], bufsize);

      totalsize += nbufsize[ibuf];
      bread += nbufsize[ibuf];

      // Increment the number of full buffers
      pthread_mutex_lock(&globalmutex);
      nfullbuf++;
      DEBUG(printf("MAIN:   Nfullbuf=%d\n", nfullbuf));
      pthread_mutex_unlock(&globalmutex);

      /* Fixed data rate if requested */
      if (datarate != 0.0) {
        throttle_rate (firsttime, datarate, totalsize, &tbehind);
      }
      
      if (!((i+1) % nupdate)) {
        pthread_mutex_lock(&globalmutex);
        gettime(&times, bread, &ntime, &maxtimes);
        bread = 0;
        bwrote = 0;

        PRINTSPEED(ntime-1, stdout);
        pthread_mutex_unlock(&globalmutex);
      }

      i++;
      jbuf = ibuf;
      ibuf = (ibuf+1)%NBUF;

      // Grab the next mutex, unlock the current one to allow the
      // Net thread to start sending 
      DEBUG( printf("MAIN:   Lock %d\n", ibuf));
      pthread_mutex_lock(&bufmutex[ibuf]);
      DEBUG( printf("MAIN:     and unlock %d\n", (jbuf)));
      pthread_mutex_unlock(&bufmutex[jbuf]);

    } // Loop over buffers

    /* Close the file */
    DEBUG(printf("MAIN: Closing file\n"));
    status = close(file);
    if (status!=0) {
      perror("Error closing input file");
    }

    if (bread>0) {
      gettime(&times, bread, &ntime, &maxtimes);
      pthread_mutex_lock(&globalmutex);
      PRINTSPEED(ntime-1, stdout);
      pthread_mutex_unlock(&globalmutex);
    }

    if (time_to_quit) break;
  }

  // Unlock the buffer
  pthread_mutex_unlock(&bufmutex[ibuf]);

  // Wait on netthread, if appropriate
  pthread_mutex_lock(&globalmutex);
  if (nfullbuf>0) { 
    // Buffers still write 
    pthread_mutex_unlock(&globalmutex);
    printf("Waiting for netthread to finish\n");
    status = pthread_join(netthread, NULL);
    if (status) {
      printf("ERROR: Return code from pthread_join is %d\n", status);
    }
    
  } else 
    pthread_mutex_unlock(&globalmutex);

  //free(mem);
  status = close(sock);
  if (status!=0) {
    perror("Error closing socket");
  }
    
  dtime = times[ntime-1].time-firsttime;
  if (dtime==0.0) 
    speed = 0.0;
  else 
    speed = totalsize/(dtime)/1000/1000;
  printf("  \nRate = %.2f Mbps/s (%.1f sec)\n\n", speed*8, 
         times[ntime-1].time-firsttime);
  
  /* A signal told us to quit. Raise this signal with the default
     handling */
  
  signal (sig_received, SIG_DFL);
  if (time_to_quit) raise (sig_received);
  
  return(0);
}
  
double tim(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;

  return t;
}

void gettime(disktime **times, long size, int *ntimes, int *maxtimes) {
  disktime *newtimes;
  if (*ntimes>=*maxtimes) {
    /* Reallocates some more space */
    *maxtimes +=1000;
    newtimes = realloc(*times, sizeof(disktime)* *maxtimes);

    if (newtimes==NULL) {
      fprintf(stderr, "Realloc failed!\n");
      perror(NULL);
      exit(1);
    }
    *times = newtimes;
  }

  (*times)[*ntimes].time = tim();
  (*times)[*ntimes].size = size;
  (*ntimes)++;

  return;
}

int setup_net(int isserver, char *hostname, int port, int window_size, 
              int *sock) {
  int ssock, status;
  unsigned long ip_addr;
  socklen_t client_len, winlen;
  struct hostent     *hostptr;
  struct sockaddr_in server, client;    /* Socket address */

  *sock = 0;

  memset((char *) &server, 0, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port); /* Which port number to use */

  /* Create the initial socket */
  ssock = socket(AF_INET,SOCK_STREAM,0); 
  if (ssock==-1) {
    perror("Error creating socket");
    return(1);
  }

  if (window_size>0) {
    status = setsockopt(ssock, SOL_SOCKET, SO_SNDBUF,
                        (char *) &window_size, sizeof(window_size));
    if (status!=0) {
      perror("Error setting socket send buffer");
      close(ssock);
      return(1);
    } 

    status = setsockopt(ssock, SOL_SOCKET, SO_RCVBUF,
                        (char *) &window_size, sizeof(window_size));
    
    if (status!=0) {
      perror("Error setting socket receive buffer");
      close(ssock);
      return(1);
    }

    /* Check what the window size actually was set to */
    winlen = sizeof(window_size);
    status = getsockopt(ssock, SOL_SOCKET, SO_SNDBUF,
                        (char *) &window_size, &winlen);
    if (status!=0) {
      close(ssock);
      perror("Getting socket options");
      return(1);
    }
    printf("Sending buffersize set to %d Kbytes\n", window_size/1024);
    
  }

  if (isserver) {

    /* Initialise server's address */
    server.sin_addr.s_addr = htonl(INADDR_ANY); /* Anyone can connect */
  
    status = bind(ssock, (struct sockaddr *)&server, sizeof(server));
    if (status!=0) {
      perror("Error binding socket");
      close(ssock);
      return(1);
    } 
  
    /* We are willing to receive conections, using the maximum
       back log of 1 */
    status = listen(ssock,1);
    if (status!=0) {
      perror("Error binding socket");
      close(ssock);
      return(1);
    }

    printf("Waiting for connection\n");

    /* Accept connection */
    client_len = sizeof(client);
    *sock = accept(ssock, (struct sockaddr *)&client, &client_len);
    if (*sock==-1) {
      perror("Error connecting to client");
      close(ssock);
      return(1);
    }
      
    printf("Got a connection from %s\n",inet_ntoa(client.sin_addr));

  } else {  // Acting as client

    hostptr = gethostbyname(hostname);
    if (hostptr==NULL) {
      fprintf(stderr,"Failed to look up hostname %s\n", hostname);
      close(ssock);
      return(1);
    }
  
    memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));
    server.sin_addr.s_addr = ip_addr;
  
    printf("Connecting to %s\n",inet_ntoa(server.sin_addr));

    status = connect(ssock, (struct sockaddr *) &server, sizeof(server));
    if (status!=0) {
      perror("Failed to connect to server");
      close(ssock);
      return(1);
    }
    *sock = ssock;
  }

  return(0);
}

void kill_signal (int sig) {
  /* We may get called twice */
  if (time_to_quit) {
    fprintf(stderr, "kill_signal called second time\n");
    return;
  }
  time_to_quit = 1;
  sig_received = sig;

  signal (sig, kill_signal); /* Re-install ourselves to disable double signals */
}  

void *netwrite(void *arg) {

  char *poff;
  int ntowrite, nwrote, sock;
  int ibuf = 0;
  int jbuf = NBUF-1;

  sock = *(int*)arg;

  while (1) {

    DEBUG(printf("THREAD: Wait on %d\n", ibuf));
    pthread_mutex_lock( &bufmutex[ibuf] );
    DEBUG(printf("THREAD: Got it\n"));
    DEBUG(printf("THREAD: Release buf %d\n", jbuf));
    pthread_mutex_unlock( &bufmutex[jbuf] );

    pthread_mutex_lock(&globalmutex);
    DEBUG(printf("THREAD: Nfullbuf = %d\n", nfullbuf));
    if (nfullbuf==0) { 
      DEBUG(printf("THREAD: Quitting no more buffers\n"));

      // No more buffers to write - must be done
      //time_to_quit = 1;
      pthread_mutex_unlock(&globalmutex);
      
      pthread_mutex_unlock( &bufmutex[ibuf] );
      pthread_exit(NULL);
    }
    pthread_mutex_unlock(&globalmutex);

    ntowrite = nbufsize[ibuf];
    poff = buf[ibuf];

    while (ntowrite>0) {
      DEBUG(printf("THREAD: Send from buffer %d\n", ibuf));

      nwrote = send(sock, poff, ntowrite, 0);
      if (nwrote==-1) {
        if (errno == EINTR) continue;
        perror("Error writing to network");

        pthread_mutex_lock(&globalmutex);
        write_failure = 1;
        pthread_mutex_unlock(&globalmutex);
        pthread_mutex_unlock( &bufmutex[ibuf] );
        pthread_exit(NULL);
      } else if (nwrote==0) {
        fprintf(stderr, "Warning: Did not write any bytes!\n");
        //close(sock);
        //time_to_quit = 1;
        pthread_mutex_lock(&globalmutex);
        write_failure = 1;
        pthread_mutex_unlock(&globalmutex);
        pthread_mutex_unlock( &bufmutex[ibuf] );
        pthread_exit(NULL);
      } else {
        ntowrite -= nwrote;
        poff += nwrote;
      }
    }
    pthread_mutex_lock(&globalmutex);
    nfullbuf--;
    DEBUG(printf("THREAD: nfullbuf %d\n", ibuf));
    pthread_mutex_unlock(&globalmutex);

    jbuf = ibuf;
    ibuf  = (ibuf+1)%NBUF;
  }
}

void throttle_rate (double firsttime, float datarate, 
                    unsigned long long totalsize, double *tbehind) {
  /* Sleep until time catches up with us, assuming we are writing faster than
     requested */
  int status;
  double t2, dt, twait, expected_time;

  t2 = tim();
  dt = t2-firsttime;

  expected_time = totalsize/(float)datarate;
  twait = (expected_time-dt);

  if (twait>0) {
    *tbehind = 0;
    status = usleep(twait*1e6);
    if ((status!=0) & (status!= EINTR)) {
      perror("Calling usleep\n");
      exit(1);
    }
  } else {
    twait *= -1;
    //if ((-twait>1) & (abs(twait-*tbehind)>0.1)) { 
    if ((abs(twait-*tbehind)>1)) { 
      /* More than a second difference */ 
      *tbehind = twait;
      printf(" Dropping behind %.1f seconds\n", twait);
    }
  }
  return;
}

void usage(void) {

  printf("Usage: vsib_send [options]\n");
  printf("  -p/-port <PORT>       Port number for transfer\n");
  printf("  -H/-host <HOST>       Remote host to connect to\n");
  printf("  -w/-window <SIZE>     TCP network window size (kB)\n");
  printf("  -t/-time              Number of blocks (-b) to average timing statistics\n");
  printf("  -s/-server            Run as server, not client (ie wait for remote connection)\n");
  printf("  -b/-blocksize <SIZE>  Read/write block size\n");
  printf("  -rate <RATE>          Fixed data rate (Mbps) to use (default as fast as possible)\n");
  printf("  -i <size>             Read from STDIN. 1 file only, <size> is file size in bytes\n");
  printf("  -h/-help              This list\n");

}

int sock_block (int fd)
{
  int flags;
  flags = fcntl(fd,F_GETFL);
  flags &= ~(O_NONBLOCK);
  return fcntl(fd,F_SETFL,flags);
}

