
#include "udp2db.h"
#include <string.h>
#include <signal.h>

#define DADA_TIMESTR "%Y-%m-%d-%H:%M:%S"

#define BSWAP_64(x)     (((uint64_t)(x) << 56) |                        \
                         (((uint64_t)(x) << 40) & 0xff000000000000ULL) | \
                         (((uint64_t)(x) << 24) & 0xff0000000000ULL) |  \
                         (((uint64_t)(x) << 8)  & 0xff00000000ULL) |    \
                         (((uint64_t)(x) >> 8)  & 0xff000000ULL) |      \
                         (((uint64_t)(x) >> 24) & 0xff0000ULL) |        \
                         (((uint64_t)(x) >> 40) & 0xff00ULL) |          \
                         ((uint64_t)(x)  >> 56))

static char* default_header =
"OBS_ID      Test.0001\n"
"OBS_OFFSET  0\n"
"FILE_SIZE   800000000\n"
"FILE_NUMBER 0\n";

static int exit_flag = 0;

long double julday(time_t seconds,long *intmjd, long double *fracmjd);

int usage(char *prg_name)
{
    fprintf (stdout,
	   "%s [options]\n"
	     " -D             disable marker\n"
	     " -I numip       number of IPs in PASP\n" 
	     " -k hex-value   hexadecimal shared memory key  [default: %x]\n"
	     " -p port        data port \n"
	     " -v             verbose messages\n"
	     " -H filename    ascii header information in file\n"
	     " -W             flag buffers readable\n",
	     prg_name, DADA_DEFAULT_BLOCK_KEY
	     );
    exit(0);
    
}

void sig_handler(int signo)
{
  if (signo == SIGUSR1)
    printf("received SIGUSR1\n");
  else if (signo == SIGINT)
    printf("received SIGINT\n");
  else if (signo == SIGTERM)
    printf("received SIGTERM\n");
  
  exit_flag = 1;

}

int init(udp2db_t* udp2db)
{
  int verbose = udp2db->verbose; /* saves some typing */
  time_t now;
  struct timeval tv;
  /* sleep until atleast 9 secs. This will allow us to 'catch' next */
  /* marker at 10sec boundary */
  fprintf(stderr,"Wait for 10-sec boundary\n");
  while ( 1 ) {                                                                                                                    
    usleep (25000);                                                                                                                            
    gettimeofday(&tv,0);
    now = tv.tv_sec;
    if ( (now%10 == 9) && (tv.tv_usec >= 500000)) {
    fprintf (stderr,"%d.%d .",(int)now % 10, (int)tv.tv_usec);
      break;
    }
  }                                                                                                                                             

  udp2db->sock = socket(AF_INET, SOCK_DGRAM, 0);
  
  memset(&udp2db->sa, 0, sizeof(udp2db->sa));
  
  udp2db->sa.sin_family = AF_UNSPEC;
  udp2db->sa.sin_addr.s_addr = htonl( INADDR_ANY );  
  udp2db->sa.sin_port = htons(udp2db->port);
  
  if (-1 == bind(udp2db->sock,(struct sockaddr *)&udp2db->sa, 
		 sizeof (udp2db->sa) )){
    perror("bind failed");
    close(udp2db->sock);
    return -1;
  }
  
  udp2db->data = malloc (100*BUFSIZE);
  assert ( udp2db->data != 0 );

  udp2db->dataFromPrevPkt = malloc (2*BUFSIZE);
  assert ( udp2db->dataFromPrevPkt != 0 );

  udp2db->markerOffset = 0;
  udp2db->bad_pkt_cnt = 0;
  udp2db->bound = 0;

  return 0;
  
}

time_t start_acq(dada_pwc_main_t* pwcm, time_t start_utc)
{
  /* Wait for the right 10-sec boundary, before reading data */
  time_t now;
  unsigned sleep_time = 0;
  
  /* set context here - all relavant parameters are here  */
  udp2db_t* udp2db = (udp2db_t*)pwcm->context;
  int verbose = udp2db->verbose; /* saves some typing */
  long long *ptr, tmpbuf=0;
  ssize_t recsize;
  socklen_t fromlen;
  char buf[8500];
  int j;
  struct timeval tv;
  char tbuf[64];

  now = time(0);
  if ( (start_utc - now) >= 10) {
    /* sleep for atleast 10 seconds before start */
    sleep_time = start_utc - now - 10;
    multilog (pwcm->log, LOG_INFO, "sleeping for %u sec\n", sleep_time);
    fprintf (stderr,"sleeping for %d secs",sleep_time);
    sleep (sleep_time);
  } 
  else {
    
    if (verbose) fprintf(stderr,"wait for marker...");
	
    while ( 1 ) {
	  
      recsize = recvfrom(udp2db->sock, (void *)buf, 8208, 0, (struct sockaddr *)&udp2db->sa, &fromlen);
      
      if (recsize < 0) {
		multilog(pwcm->log,LOG_ERR,"error reading udp packet\n");
		return 0;
      }
      
      /* swap bytes here; use this to check marker*/
      ptr = (long long *) buf;
      tmpbuf = BSWAP_64( *ptr );
      
      /* Wait for marker to arrive in the data stream. Use that to timestamp the first sample */
	  
      for (j=2;j<1026;j++) {
		if ( (*(ptr+j) & 0x0000ffffffffff00) == 0x0000dadadadada00 ) {
		  memcpy(udp2db->dataFromPrevPkt, &buf[j*8], (8208 - j*8));
		  udp2db->FirstFlag = 0xdada;
		  udp2db->markerOffset = 8208 - j*8;
		  fprintf (stderr,"Marker-check. offset=%d\n",j);
		  break;
		}
      }
      
      udp2db->prev_pkt_cnt = tmpbuf;
      
      /* if the marker was seen, get out of this loop*/
      if (udp2db->FirstFlag == 0xdada ) break;
      
    }  

    // set packet difference to zero
    udp2db->pktdiff = 0;
    udp2db->pkt_cnt = 1;
    // Get high resolution system time
    gettimeofday(&tv,0);
    
    // Use value returned the call above
    start_utc=(time_t)tv.tv_sec;

    // Round off: if we are too close to the 1-sec boundary,
    // the marker actually belongs to the next tick
    if (tv.tv_usec>500000) start_utc++;

    // Print start time                                                                                               
    strftime (tbuf, 64, DADA_TIMESTR, gmtime(&start_utc));
    fprintf(stderr,"UTC_START: %s UNIX: %ld.%06ld\n",tbuf,tv.tv_sec,tv.tv_usec);
	
    return start_utc;
  }
}

void* get_next_buf(dada_pwc_main_t* pwcm, uint64_t* size)
{
  udp2db_t* udp2db = (udp2db_t*)pwcm->context;
  int verbose = udp2db->verbose; /* saves some typing */
  char buf[8500];
  long long tmpbuf[2], *ptr;
  int i,j,npkt,off,nfill ;
  ssize_t recsize;
  socklen_t fromlen;
  
  /* Write only the data from marker onwards. See init() function */
  if (udp2db->FirstFlag == 0xdada){
    memcpy(udp2db->data, udp2db->dataFromPrevPkt, udp2db->markerOffset);
    *size = udp2db->markerOffset;
    udp2db->FirstFlag = 1;
    fprintf(stderr,"Returned %lu bytes; First call to get_next_nuf()\n",*size);
    return (void*) udp2db->data;
  }

  recsize = recvfrom(udp2db->sock, (void *)buf, 8208, 0, (struct sockaddr *)&udp2db->sa, &fromlen);
  
  if (recsize < 0) {
    multilog(pwcm->log,LOG_ERR,"error reading udp packet\n");
    return 0;
  }

  /* swap bytes here*/
  ptr = (long long *)buf;
  tmpbuf[0]=BSWAP_64( *ptr );
  tmpbuf[1]=BSWAP_64( *(ptr+1) );

  /*offset b/w successive packets  */
  off =  tmpbuf[0] - udp2db->prev_pkt_cnt;
  
  if ( (off % 1024) != 0){
    fprintf(stderr, "system counter is not mod(1024). exit.\n");
    exit(0);
  }
  
  /* number of packets depends on number of IP's in the PASP design*/
  npkt = off/(1024*udp2db->numip);

  if (npkt > 100){
    fprintf(stderr, "Lost > 100 packets. exit.\n");
    exit(0);
  }

  // fill-in missing packets and
  // replace marker by adajacent samples.
  if ( npkt > 1){
    //repeat the new packet by as many lost packets.
    //nfill = npkt - 1;
    //memset(&udp2db->data[0], 0, nfill*8192);
    //memcpy(&udp2db->data[nfill*8192 + 1], &buf[16], 8192);
    for (nfill=0 ; nfill < npkt; nfill++)
      memcpy(&udp2db->data[nfill*8192],&buf[16],8192);
    udp2db->bad_pkt_cnt += npkt;
    udp2db->prev_pkt_cnt = tmpbuf[0];
    udp2db->pktdiff += npkt;
    udp2db->pkt_cnt += npkt;
    if (udp2db->pktdiff > (udp2db->bound - 2) ) fprintf(stderr,"%d  ", npkt);
    *size = npkt*BUFSIZE;
    return (void*) udp2db->data;
  }
  else {
    // search and replace marker before copying
    // works only when FPGA is running at 250 MHz, and for 32c32ip design
    int ck = udp2db->pktdiff;
	int b0 = udp2db->bound - 2;
	int b1 = udp2db->bound + 2;
    if ( ( (ck > b0 && ck < b1 ) || ck > 2*b0 ) && (udp2db->marker_enable == 0)){
      for (j=2;j<1026;j++) {
        if ( (*(ptr+j) & 0x0000ffffffffff00) == 0x0000dadadadada00 ) {
		  fprintf(stderr,"%d ",j);
		  memcpy(&buf[j*8],&buf[j*8+64],48);
		  udp2db->pktdiff = 0;
          break;
        }
      }
    }
    
    /* copy read data to the data-buffer*/
    *size = BUFSIZE;
    memcpy(udp2db->data, &buf[16], 8192);
    udp2db->prev_pkt_cnt = tmpbuf[0];
    // in case marker was missed, simply reset the packet counter.
    udp2db->pktdiff++;
    udp2db->pkt_cnt++;
    /*return the acquired data buffer */
    return (void*) udp2db->data;
  }

}

int stop_acq(dada_pwc_main_t* pwcm)
{
  udp2db_t* udp2db = (udp2db_t*)pwcm->context;
  int verbose = udp2db->verbose; /* saves some typing */
  
  /* stop receiving data */
  if (verbose) fprintf(stderr,"will stop");
  
  /* close socket */
  return 0;
}

int main (int argc, char **argv)
{
  /* DADA header plus data */
  dada_hdu_t* hdu;
  
  /* primary write client main loop  */
  dada_pwc_main_t* pwcm; 
  
  /* header */
  char* header = default_header; 
  char* header_buf = 0;
  unsigned header_strlen = 0; /* the size of the header string */
  uint64_t header_size = 0;   /* size of the header buffer */
  char writemode='w';
  
  /* flags */
  char daemon = 0; /* daemon mode */
  char marker = 1;

  int verbose = 0; /* verbosity */
  
  /* udp2db stuff, all socket goes here */
  
  udp2db_t udp2db;
  
  time_t utc;
  
  unsigned buf_size = 64;
  
  static char* buf = 0;
  
  /* Some sensible defaults */
  char *src;
  int nSecs = 100;
  int numip = 0;
  int wrCount,rdCount;
  uint port = 56000;

  /* the filename from which the header will be read */
  char* header_file = 0;
  int header_flag = 0;
 
 /* hexadecimal shared memory key */
  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  int arg = 0;
  while ((arg=getopt(argc,argv,"dDI:k:m:p:vH:W")) != -1) {
    switch (arg) {
      
    case 'D':
      marker=0;
      break;
      
    case 'v':
      verbose=1;
      break;
      
    case 'H':
      header_file = optarg;
	  header_flag = 1;
      break;
      
	case 'I':
      if (sscanf (optarg, "%d", &numip) != 1) {
        fprintf (stderr,"cannot parse numer of IPs from %s\n",optarg);
        return -1;
      }
		
      break;
      
    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) {
        fprintf (stderr,"cannot not parse key from %s\n",optarg);
        return -1;
      }
      break;
      
	case 'p':
      if (optarg){
        port = atoi(optarg);
        fprintf(stderr,"using port %d\n",port);
      }
      else usage(argv[0]);
      break;
      
    case 'W':
      writemode='W';
      break;
      
    default:
      usage(argv[0]);
      return 0;
      
    }
  }
  
  /* register options in udp2db structure*/
  
  udp2db.verbose = verbose;
  udp2db.port = port;
  udp2db.buf = 0;
  udp2db.numip = numip;
  udp2db.marker_enable = marker;
  udp2db.FirstFlag = 0;

  if ( (numip < 4 ) || (numip > 32)) {
	fprintf (stderr,"invalid number of IPs (specify 4,8,16 or32)\n");
	return -1;
  }

  if (!header_flag ){
	fprintf(stderr,"Please specify header file with -H\n");
	usage(argv[0]);
  }
  
  
  // set up signal handlers
  if (signal(SIGUSR1, sig_handler) == SIG_ERR)
    fprintf(stderr,"\ncan't catch SIGUSR1\n");
  if (signal(SIGINT, sig_handler) == SIG_ERR)
    fprintf(stderr,"\ncan't catch SIGINT\n");
  if (signal(SIGTERM, sig_handler) == SIG_ERR)
    fprintf(stderr,"\ncan't catch SIGTERM\n");
  
  /* Initialize DADA/PWC structures*/
  pwcm = dada_pwc_main_create ();
  
  pwcm->context = &udp2db;
  
  pwcm->log = multilog_open ("udp2db", daemon);
  
  /* set up for daemon usage */	  
  
  multilog_serve (pwcm->log, DADA_DEFAULT_PWC_LOG);
  
  
  /* Init. sockets and bind*/
  
  if (verbose) fprintf(stderr,"Going to initialise sockets...");
  if ( init(&udp2db) < 0) {
    fprintf(stderr,"Unable to initialise.\n");
    return EXIT_FAILURE;
  }
  if (verbose) fprintf(stderr,"Initialisation done.\n");
  
  /* create the header/data blocks */
  
  hdu = dada_hdu_create (pwcm->log);
  
  dada_hdu_set_key(hdu, dada_key);
 
  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;    
  
  /* make data buffers readable */
  
  if (dada_hdu_lock_write_spec (hdu,writemode) < 0)
    return EXIT_FAILURE;
  
  /* if we are a stand alone application */
  
  header_size = ipcbuf_get_bufsz (hdu->header_block);
  multilog (pwcm->log, LOG_INFO, "header block size = %llu\n", header_size);
  
  header_buf = ipcbuf_get_next_write (hdu->header_block);
  
  if (!header_buf)  {
	multilog (pwcm->log, LOG_ERR, "get next header block error.\n");
	return EXIT_FAILURE;
  }
  
  /* if header file is presented, use it. If not set command line attributes */ 
  if (fileread (header_file, header_buf, header_size) < 0)  {
	multilog (pwcm->log, LOG_ERR, "cannot read header from %s\n", header_file);
	return EXIT_FAILURE;
  }
  
  /* Begin data acquisition set actual start time in header */
  
  utc = start_acq(pwcm, 0);
  
  if (!buf)  buf = malloc (buf_size);
  assert (buf != 0);
  
  strftime (buf, buf_size, DADA_TIMESTR, gmtime(&utc));
  
  /* write UTC_START to the header */
  if (ascii_header_set (header_buf, "UTC_START", "%s", buf) < 0) {
	multilog (pwcm->log, LOG_ERR, "failed ascii_header_set UTC_START\n");
	return -1;
  }
  
  /* write high resolution MJD start time*/
  
  long int intmjd;
  long double fracmjd;
  char str[50],str1[50];
  double tsamp;
  
  julday(utc, &intmjd, &fracmjd);
  memset(str,'\0',50);
  memset(str1,'\0',50);
  sprintf(str,"%ld %1.35LG",intmjd, fracmjd);
  memcpy(str1,str,5);
  memcpy(str1+5,index(str,'.'),29);
  
  if (ascii_header_set (header_buf, "MJD_START", "%s", str1 ) < 0) {
	multilog (pwcm->log, LOG_ERR, "failed ascii_header_set MJD_START\n");
	return -1;
  }

  if (ascii_header_get (header_buf, "TSAMP", "%lf", &tsamp) < 0){
	fprintf (stderr, "failed to parse TSAMP\n");
	return -1;
  }

  /* number of packets between two markers = (4 * 1000000*1/tsamp)/8192 */
  udp2db.bound = (int) 4e6/(tsamp*8192);
  
  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
	multilog (pwcm->log, LOG_ERR, "Could not mark filled header block\n");
	return EXIT_FAILURE;
  }
  
  /* The write loop: repeat for n bufs */ 
  while(1 ){
    
	uint64_t bsize = BUFSIZE;
    
	src = (char *)get_next_buf(pwcm, &bsize);
    
	/* write data to datablock */
	if ( (ipcio_write(hdu->data_block, src, bsize) ) < bsize ){
	  multilog(pwcm->log, LOG_ERR, "Cannot write requested bytes to SHM\n");
	  return EXIT_FAILURE;
	}
	
	wrCount = ipcbuf_get_write_count ((ipcbuf_t*)hdu->data_block);
	rdCount = ipcbuf_get_read_count ((ipcbuf_t*)hdu->data_block);
	if (verbose) fprintf(stderr,"%lld %d %d\n",udp2db.buf,wrCount,rdCount);
    
	// in the event of a signal, gracefully exit by reporting all
	// detials collected over the observation.
    
	if ( exit_flag ) break;
    
  }

  fprintf(stderr,"\nbad_pkt_cnt = %llu pkt_cnt = %llu\n",udp2db.bad_pkt_cnt, udp2db.pkt_cnt);
  
  if ( stop_acq(pwcm) != 0)  fprintf(stderr, "Error stoping acquisition");
  
  if (dada_hdu_unlock_write (hdu) < 0)  return EXIT_FAILURE;
  
  if (dada_hdu_disconnect (hdu) < 0) 	return EXIT_FAILURE;
  
  fprintf (stderr, "Destroying pwc main\n");
  dada_pwc_main_destroy (pwcm);
  
  return EXIT_SUCCESS;
}

