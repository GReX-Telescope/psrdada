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

#define NBUF       3    /* Minimum 3 buffers or locking will fail */
#define DEFAULT_BUFSIZE      2
#define MAXSTR              200 

#define DEBUG(x)

double tim(void);
void gettime(disktime **, long, int *, int *);
void kill_signal (int);

int setup_net(int port, int window_size, int *listensock);
int waitconnection (struct hostent *hostptr, int window_size, int port, int listensock, 
		    int *serversock, int *clientsock);
void throttle_rate (double firsttime, float datarate, 
		    unsigned long long totalsize, double *tbehind);
void *netwrite(void *arg);
void *netread(void *arg);

volatile int time_to_quit = 0;
volatile int sig_received = 0;

/* Globals needed by network threads */
int  nfullbuf;
int bwrote;        /* Amount of net data written between updates */
int bufsize;
int write_failure;

char *buf[NBUF];
ssize_t nbufsize[NBUF];
pthread_mutex_t bufmutex[NBUF];
pthread_mutex_t globalmutex = PTHREAD_MUTEX_INITIALIZER;

int main (int argc, char * const argv[]) {
  int i, j, status, ofile, opt, tmp, nwrote;
  int ibuf, jbuf, nupdate, ntowrite;
  int serversock, clientsock, listensock;
  char msg[MAXSTR+50], *poff;

  double firsttime, t1, t2;
  float ftmp, speed;
  double tbehind;
  unsigned long long totalsize;
  pthread_t netthread;

  int debug = 0;  
  float timeupdate = 50; /* Time every 50 MiB by default */
  int portserver = 8000;    /* Port to listen on */
  int portclient = 8000;    /* Port to connect to */
  int window_size1 = -1;	/* 256 kilobytes */
  int window_size2 = -1;	/* 256 kilobytes */
  float datarate = 0; /* Limit read/write to this data rate */
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */
  struct hostent     *hostptr;
  
  struct option options[] = {
    {"rate", 1, 0, 'z'},
    {"port", 1, 0, 'p'},
    {"port1", 1, 0, '1'},
    {"port2", 1, 0, '2'},
    {"p1", 1, 0, '1'},
    {"p2", 1, 0, '2'},
    {"host", 1, 0, 'H'},
    {"time", 1, 0, 't'},
    {"window", 1, 0, 'w'},
    {"window1", 1, 0, '3'},
    {"window2", 1, 0, '4'},
    {"w1", 1, 0, '3'},
    {"w2", 1, 0, '4'},
    {"blocksize", 1, 0, 'b'},
    {"debug", 0, 0, 'D'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  bufsize   = DEFAULT_BUFSIZE * 1000*1000;
  ofile = -1;
  i = -1;
  write_failure = 0;

  //nread = 0;

  srand(time(NULL));
  
  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "p:rn:DdhH:", 
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
      else {
	window_size1 = ftmp * 1024;
	window_size2 = ftmp * 1024;
      }
     break;
     
    case '3':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
	fprintf(stderr, "Bad window1 option %s\n", optarg);
      else {
	window_size1 = ftmp * 1024;
      }
     break;

    case '4':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
	fprintf(stderr, "Bad window2 option %s\n", optarg);
      else {
	window_size2 = ftmp * 1024;
      }
     break;

    case 'z':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
	fprintf(stderr, "Bad rate option %s\n", optarg);
      else 
	datarate = ftmp * 1000*1000/8;
      break;

    case 'p':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else {
	portserver = tmp;
	portclient = tmp;
      }
      break;

    case '1':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else {
	portserver = tmp;
      }
      break;

    case '2':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else {
	portclient = tmp;
      }
      break;
      
    case 't':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad time update option %s\n", optarg);
      else 
	timeupdate = tmp;
      break;
      
    case 'H':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Hostname too long\n");
	return(1);
      }
      strcpy(hostname, optarg);
      break;

    case 'D':
      debug = 1;
      break;
      
    case 'h':
      printf("Usage: diskspeed [options]\n");
      printf("  -p/-port <PORT>       Port number for transfer (server and client)\n");
      printf("  -p1/-port1 <PORT>     Port number to listen on\n");
      printf("  -p2/-port2 <PORT>     Port number to connect to remote end\n");
      printf("  -H/-host <HOST>       Host to connect to\n");
      printf("  -t/-time <TIME>       Update time (in MBytes) for network statistics\n");
      printf("  -w/-window <SIZE>     Network window size (kB)\n");
      printf("  -w1/-window1 <SIZE>   Network window size on listening side (kB)\n");
      printf("  -w2/-window2 <SIZE>   Network window size on sending side (kB)\n");
      printf("  -t/-time <TIME>       Number of blocks (-b) to average timing statistics\n");
      printf("  -b/-blocksize <SIZE>  Read/write block size\n");
      printf("  -rate <RATE>          Fixed data rate (Mbps) to use (default as fast as possible)\n");
      printf("  -h/-help              This list\n");
      return(1);
    break;
    
    case '?':
    default:
      break;
    }
  }

  if (strlen(hostname) == 0) {
    strcpy(hostname, "localhost");
  }

  printf("Listening on port %d\n", portserver);
  printf("Will connect to port %d\n", portclient);

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
  
  /* Check we can lookup remote host name before we start */
  hostptr = gethostbyname(hostname);
  if (hostptr==NULL) {
    fprintf(stderr,"Failed to look up hostname %s\n", hostname);
    exit(1);
  }

  status =  setup_net(portserver, window_size1, &listensock);
  if (status) return(1);

  while (!time_to_quit) { // Loop over multiple connections

    status =  waitconnection(hostptr, window_size2, portclient, listensock, 
			     &serversock, &clientsock);

    if (status) break;

    /* Install a ^C catcher */
    signal (SIGINT, kill_signal);
  
    bwrote = 0;
    totalsize=0;
    firsttime = tim();
    t1 = firsttime;
    ibuf = 0;
    jbuf = NBUF-1;
    j = 0;
    
    /* Need to start net thread */

    // Lock the last mutex for the netread buffer
    DEBUG(printf("MAIN:   Locked %d for thread\n", jbuf));
    pthread_mutex_lock( &bufmutex[jbuf] );
    
    status = pthread_create( &netthread, NULL, netread, (void *)&serversock);
    if (status) {
      printf("Failed to start net read thread: %d\n", status);
      exit(1);    
    }
    
    /* Now wait on netread to lock the first buffer and unlock this one */
    DEBUG(printf("MAIN:   Wait on lock for buf %d\n", jbuf));
    pthread_mutex_lock( &bufmutex[jbuf] );
    DEBUG(printf("MAIN:   Got it (%d)\n", jbuf));
    
    while (1) {
 
      tbehind = 0;
      
      /* Loop over buffers */
      
      DEBUG(printf("MAIN:   Try and get lock on %d\n", ibuf));
      pthread_mutex_lock( &bufmutex[ibuf] );
      DEBUG(printf("MAIN:   Got it (%d)\n", ibuf));

      // Unlock the previous lock
      DEBUG(printf("MAIN:   Release lock %d\n", jbuf));
      pthread_mutex_unlock( &bufmutex[jbuf] );
      
      pthread_mutex_lock(&globalmutex);
      DEBUG(printf("MAIN:    Nfullbuf = %d\n", nfullbuf));
      if (nfullbuf==0) {
	pthread_mutex_unlock(&globalmutex);
	break;
      }
      pthread_mutex_unlock(&globalmutex);
      
      DEBUG(printf("MAIN: nbufsize = %d\n", nbufsize[ibuf]));
      
      ntowrite = nbufsize[ibuf];
      poff = buf[ibuf];
      
      while (ntowrite>0) {
	DEBUG(printf("MAIN:   Send from buffer %d\n", ibuf));
	
	nwrote = send(clientsock, poff, ntowrite, 0);
	if (nwrote==-1) {
	  if (errno == EINTR) continue;
	  perror("Error writing to network");
	  pthread_mutex_lock(&globalmutex);
	  write_failure = 1;
	  pthread_mutex_unlock( &bufmutex[ibuf] );
	  break;
	  
	} else if (nwrote==0) {
	  fprintf(stderr, "Warning: Did not write any bytes!\n");
	  pthread_mutex_lock(&globalmutex);
	  write_failure = 1;
	  pthread_mutex_unlock(&globalmutex);
	  pthread_mutex_unlock( &bufmutex[ibuf] );
	  break;
	  
	} else {
	  ntowrite -= nwrote;
	  poff += nwrote;
	  totalsize += nwrote;
	  bwrote += nwrote;
	}
      }
	  
      pthread_mutex_lock(&globalmutex);
      nfullbuf--;
      pthread_mutex_unlock(&globalmutex);
      /* Fixed data rate if requested */
      if (datarate != 0.0) {
	throttle_rate (firsttime, datarate, totalsize, &tbehind);
      }
    
      
      if (!((i+1) % nupdate)) {
	t2 = tim();

	speed = bwrote/(t2-t1)/1e6*8; /* MB/s */

	printf("%8.1f %7.1f %7.2f Mbps ( %d %7.3f )\n", 
 	  (double)totalsize/1.0e6, t2-firsttime, speed, bwrote, t2-t1); 
	t1 = t2;
	
	bwrote = 0;

      }
      
      DEBUG(printf("MAIN: ibuf++\n"));
      jbuf = ibuf;
      ibuf = (ibuf+1)%NBUF;
      
      pthread_mutex_lock(&globalmutex);
      DEBUG(printf("MAIN:    Nfullbuf = %d\n", nfullbuf));
      pthread_mutex_unlock(&globalmutex);


      if (time_to_quit) break;
      i++;
    } // Loop over disks

    status = close(serversock);
    if (status!=0) {
      perror("Error closing socket");
    }
    status = close(clientsock);
    if (status!=0) {
    perror("Error closing socket");
    }
    
    t2 = tim();
    speed = totalsize/(t2-firsttime)/1e6*8;
    printf("  \nRate = %.2f Mbps/s (%.1f sec)\n\n", speed, 
	   t2-firsttime);
    
    DEBUG(printf("MAIN:   Release lock %d\n", ibuf));
    pthread_mutex_unlock(&bufmutex[ibuf]);


    printf("Re-enabling signal\n");
    signal (SIGINT, SIG_DFL);
    if (time_to_quit) {
      raise (sig_received);
      signal (sig_received, SIG_DFL);
    }
  }

  return(0);
}
  
double tim(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;

  return t;
}

int setup_net(int port, int window_size, int *listensock) {
  int status;
  struct sockaddr_in server;    /* Socket address */
  /* Open a server connection for reading */

  /* Initialise server's address */
  memset((char *)&server,0,sizeof(server));
  server.sin_family = AF_INET;
  server.sin_addr.s_addr = htonl(INADDR_ANY); /* Anyone can connect */
  server.sin_port = htons((unsigned short)port); /* Which port number to use */

  /* Create a server to listen with */
  *listensock = socket(AF_INET,SOCK_STREAM,0); 
  if (*listensock==-1) {
    perror("Error creating socket");
    return(1);
  }

  /* Set the TCP window size */

  if (window_size>0) {
    status = setsockopt(*listensock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      perror("Error setting socket options");
      close(*listensock);
      return(1);
    } 

    status = setsockopt(*listensock, SOL_SOCKET, SO_RCVBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      perror("Error setting socket options");
      close(*listensock);
      return(1);
    } 
  }

  status = bind(*listensock, (struct sockaddr *)&server, sizeof(server));
  if (status!=0) {
    perror("Error binding socket");
    close(*listensock);
    return(1);
  } 

  /* We are willing to receive conections, using the maximum
     back log of 1 */
  status = listen(*listensock,1);
  if (status!=0) {
    perror("Error binding socket");
    close(*listensock);
    return(1);
  }

  return(0);
}

int waitconnection (struct hostent *hostptr, int window_size, int port, int listensock, 
		    int *serversock, int *clientsock) {
  int status;
  unsigned long ip_addr;
  socklen_t client_len;
  struct sockaddr_in server, client;    /* Socket address */

  printf("Waiting for connection\n");
  
  /* Accept connection */
  client_len = sizeof(client);
  *serversock = accept(listensock, (struct sockaddr *)&client, &client_len);
  if (*serversock==-1) {
    perror("Error connecting to client");
    close(listensock);
    return(1);
  }
      
  printf("Got a connection from %s\n",inet_ntoa(client.sin_addr));

  /* Now try and connect to remote end */

  memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));
  memset((char *) &server, 0, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port); 
  server.sin_addr.s_addr = ip_addr;
  
  printf("Connecting to %s\n",inet_ntoa(server.sin_addr));
    
  *clientsock = socket(AF_INET, SOCK_STREAM, 0);
  if (*clientsock==-1) {
    perror("Failed to allocate socket");
    return(1);
  }

  if (window_size>0) {
    status = setsockopt(*clientsock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      close(*clientsock);
      perror("Setting socket options");
      return(1);
    }
    status = setsockopt(*clientsock, SOL_SOCKET, SO_RCVBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      close(*clientsock);
      perror("Setting socket options");
      return(1);
    }
  }

  status = connect(*clientsock, (struct sockaddr *) &server, sizeof(server));
  if (status!=0) {
    perror("Failed to connect to server");
    return(1);
  }
  return(0);
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

void *netread (void *arg) {
  int sock;
  int ibuf = 0;
  int jbuf = NBUF-1;
  ssize_t nr;
  char *poff;
  size_t ntoread;

  sock = *(int*)arg;
  
  while (1) {

    DEBUG(printf("THREAD: Wait on %d\n", ibuf));
    pthread_mutex_lock( &bufmutex[ibuf] );
    DEBUG(printf("THREAD: Got it (%d)\n", ibuf));
    DEBUG(printf("THREAD: Release buf %d\n", jbuf));
    pthread_mutex_unlock( &bufmutex[jbuf] );

    if (time_to_quit || write_failure) {
      DEBUG(printf("THREAD: Time to quit\n"));
      pthread_exit(NULL);
    }

    nbufsize[ibuf] = 0;
    poff = buf[ibuf];
    ntoread = bufsize;

    while (nbufsize[ibuf]<ntoread) {
    
      nr = recv(sock,poff,ntoread-nbufsize[ibuf],0);
    	  
      if (nr==-1) {
	if (errno == EINTR) continue;
	perror("Error reading socket");
	DEBUG(printf("THREAD: Release buf %d\n", ibuf));
	pthread_mutex_unlock( &bufmutex[ibuf] );
	pthread_exit(NULL);

      } else if (nr==0) {
	DEBUG(printf("THREAD: Socket connection closed\n"));
	/* Assume for now this means the socket was closed */
	pthread_mutex_lock(&globalmutex);
	if (nbufsize[ibuf]>0) nfullbuf++;
	pthread_mutex_unlock(&globalmutex);
	DEBUG(printf("THREAD: Release buf %d\n", ibuf));
	pthread_mutex_unlock(&bufmutex[ibuf]);
	pthread_exit(NULL);

      } else {
	pthread_mutex_lock(&globalmutex);
	poff += nr;
	nbufsize[ibuf] += nr;
	pthread_mutex_unlock(&globalmutex);
      }
    }

    pthread_mutex_lock(&globalmutex);
    nfullbuf++;
    DEBUG(printf("THREAD: nfullbuf =  %d\n", nfullbuf));
    pthread_mutex_unlock(&globalmutex);

    jbuf = ibuf;
    ibuf = (ibuf+1)%NBUF;
  }
}
