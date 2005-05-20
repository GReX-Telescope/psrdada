
#include "puma2_dmadb.h"


static char* default_header =
"OBS_ID      Test.0001\n"
"OBS_OFFSET  0\n"
"FILE_SIZE   800000000\n"
"FILE_NUMBER 0\n";

int usage(char *prg_name)
{
    fprintf (stdout,
	   "%s [options]\n"
	     " -d             run as daemon\n"
	     " -i idstr       observation id string\n"	
	     " -m mode        1 for independent mode, 2 for controlled mode\n"	
	     " -n nbuf        # of buffers to acquire\n"
	     " -s sec         # of seconds to acquire\n"
	     " -v             verbose messages\n"
             " -H filename    ascii header information in file\n"
	     " -S             file size in bytes\n"
	     " -W             flag buffers readable\n",
	     prg_name
	     );
    exit(0);
    
}

int initialize(dmadb_t* dmadb)
{
  int verbose = dmadb->verbose; /* saves some typing */

  /* initialize parameters */

  dmadb->nbufs = 0;
  dmadb->nbufs = 100;
  dmadb->fSize = 800000000;
  dmadb->ObsId = "Test001";
  /* open PiC card - makes the device available for further writes  */

  if (verbose) fprintf(stderr,"opening PiC Device");
  if ( (pic_open(&dmadb->pic, 0)) < 0 ){
    fprintf(stderr,"PiC open failed\n");
    return 0;
  }    

  /* disarm pic as a precautionary measure */
  if ( (pic_configure(&dmadb->pic, 2)) < 0 ){
    fprintf(stderr,"Cannot disarm PiC\n");
    return 0;
  }  
  if (verbose) fprintf(stderr,"...done\n");
 
  /* setup EDT DMA Card */
  
  if (verbose) fprintf(stderr,"open EDT device....");
  if ((dmadb->edt_p = edt_open(EDT_INTERFACE, 0)) == NULL)
    {
      fprintf(stderr,"edt_open error\n") ;
      return 0;
    }
  if (verbose) fprintf(stderr,"...done\n");
  
  
  if (verbose) fprintf(stderr,"Configuring EDT kernel ring buffers....");
  if (edt_configure_ring_buffers(dmadb->edt_p, BUFSIZE, 16, EDT_READ, NULL) == -1)
    {
      fprintf(stderr,"edt_configure_ring_buffers error\n") ;
      return 0;
    }
  if (verbose) fprintf(stderr,"...done\n");
  
  /* flush stale buffers in the EDT card */
  edt_flush_fifo(dmadb->edt_p) ;     
  
  /* setup EDT to do continuous transfers */
  edt_start_buffers(dmadb->edt_p, 0) ; 
  
  return 0;

}

time_t arm_pic(dada_pwc_main_t* pwcm, time_t start_utc)
{
  /* Arm the PuMaII Interface card, at the given UTC */
  /* Wait for the right 10-sec boundary, before arming PiC */
  time_t now;
  unsigned sleep_time = 0;
  /* get our context here, all required parameters are here */
  dmadb_t* dmadb = (dmadb_t*)pwcm->context;
  int verbose = dmadb->verbose; /* saves some typing */

  /* wait for atleast 10-second before start time */
  now = time(0);

  if (verbose) fprintf(stderr,"Going to start");
  if ( (start_utc - now) >= 10) {
    /* sleep for atleast 10 seconds before start */
    sleep_time = start_utc - now - 10;
    multilog (pwcm->log, LOG_INFO, "sleeping for %u sec\n", sleep_time);
    fprintf (stderr,"sleeping for %d secs",sleep_time);
    sleep (sleep_time);
  } 
  else {
    /* check if we're within the 'allowed window' , if not sleep */
    while ( (now % 10 <= 1) | (now % 10 >= 8) ) {
      usleep (250000);
      fprintf (stderr,"%d .",(int)now % 10);
      now = time(0);
    }
    fprintf(stderr,"\n");
  }  

  /*PiC is arm'ed now. When the next 10-Sec pulse arrives */
  /*EDT will receive both clock and enable data*/

  if ( (pic_configure(&dmadb->pic, 1)) < 0 ){
    multilog(pwcm->log, LOG_ERR,"Cannot arm PiC\n");
    return 0;
  }    

  /* The actual start time is the next 10-second tick UTC */
  now = time(0);
  start_utc = (1+now/10)*10; /* integer division plus one to get next 10-sec */
  return start_utc;

}

void* get_next_buf(dada_pwc_main_t* pwcm, uint64_t* size)
{
  dmadb_t* dmadb = (dmadb_t*)pwcm->context;
  int verbose = dmadb->verbose; /* saves some typing */

  /* size of buffer to be acqyured */
  *size = BUFSIZE;

  dmadb->buf = edt_done_count(dmadb->edt_p);
  if (verbose) fprintf(stderr,"buf count%d\n",dmadb->buf);

  if ( (dmadb->data = edt_wait_for_buffers(dmadb->edt_p, 1)) == NULL){
    edt_perror("edt_wait");
    multilog(pwcm->log,LOG_ERR,"edt_wait error\n");
    return 0;
  }
  
  /* check for overflows */
  if ( (edt_ring_buffer_overrun(dmadb->edt_p)) != 0){
    fprintf(stderr,"Overrun Error\n");
    multilog(pwcm->log,LOG_ERR,"EDT Overrun\n");
  }
  
  /* is it (void *) or (char *)? */
  return (void*) dmadb->data;

}

int stop_acq(dada_pwc_main_t* pwcm)
{
  dmadb_t* dmadb = (dmadb_t*)pwcm->context;
  int verbose = dmadb->verbose; /* saves some typing */

  /* stop the EDT card */
  if (verbose) fprintf(stderr,"Stopping EDT Device...");
  edt_stop_buffers(dmadb->edt_p) ; 
  edt_close(dmadb->edt_p);
  if (verbose) fprintf(stderr,"done\n");
  
  /* disarm pic now*/
  if (verbose) fprintf(stderr,"Stopping PiC Device...");
  if ( (pic_configure(&dmadb->pic, 2)) < 0 ){
    multilog(pwcm->log, LOG_ERR,"Cannot configure PiC\n");
    return -1;
  }  
  if (verbose) fprintf(stderr,"done\n");

  return 0;
}

int main (int argc, char **argv)
{
    /* DADA Header plus Data Unit */
    dada_hdu_t* hdu;
    multilog_t* log; /* Logger */
    dada_pwc_main_t* pwcm; /* primary write client main loop  */

    char* header = default_header; /* header */
    char* header_buf = 0;
    unsigned header_strlen = 0;/* the size of the header string */
    uint64_t header_size = 0;/* size of the header buffer */
    char writemode='w';

    /* flags */
    char daemon = 0; /*daemon mode */
    int verbose = 0; /* verbose mode */
    int mode = 0;

    /* dmadb stuff, all EDT/PiC stuff is in here */
    
    dmadb_t dmadb;
       
    int nSecs = 100,nbufs=0, buf=0;
    int wrCount,rdCount;
    char *ObsId = "Test";
    unsigned long fSize=800000000;

    /* the filename from which the header will be read */
    char* header_file = 0;

    int arg = 0;
    while ((arg=getopt(argc,argv,"di:m:n:s:vH:S:W")) != -1) {
	switch (arg) {
	    
	    case 'd':
		daemon=1;
		break;
		
	    case 'v':
		dmadb.verbose=1;
		verbose=1;
		break;

            case 'H':
                header_file = optarg;
                break;

	    case 'i':
		if (verbose) fprintf(stderr,"ObsId is %s \n",optarg);
		if(optarg) {
		    ObsId = strdup (optarg);
                    assert (ObsId != 0);
		}
		else usage(argv[0]);
		break;

	    case 'm':
	         if (verbose) fprintf(stderr,"Mode is %s \n",optarg);
		 if(optarg) {
		   mode = atoi (optarg);
		   if ( (mode == 1) || (mode == 2)){
		     fprintf(stderr,"Using mode %d\n",mode);
		   }
		   else {
		     fprintf(stderr,"Specify a valid mode\n");
		     usage(argv[0]);
		   }
		 }
		 break;

	    case 'n':
		if (optarg){
		    nbufs = atoi(optarg);
		    fprintf(stderr,"Acquiring %d buffers\n",nbufs);
		}
		else usage(argv[0]);
		break;
	    
	    case 's':
		if (optarg){
		    nSecs = atoi(optarg);
		    fprintf(stderr,"Will acquire for %d seconds\n",nSecs);
		}
		else usage(argv[0]);
		break;

	    case 'S':
		if (optarg){
		    fSize = (unsigned long)atol(optarg);
		    fprintf(stderr,"File size will be %lu\n",fSize);
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
    
    if (!mode ){
      fprintf(stderr,"Please specify mode\n");
      usage(argv[0]);
    }
		
    log = multilog_open ("puma2_dmadb", daemon);
    
    /* set up for daemon usage */	  
    if (daemon) {
      be_a_daemon ();
      multilog_serve (log, DADA_DEFAULT_PWC_LOG);
    }
    else
      multilog_add (log, stderr);

    /* create pwcm structure (and initialize)*/
    pwcm = dada_pwc_main_create ();

    pwcm->context = &dmadb;
    pwcm->start_function  = arm_pic;
    pwcm->buffer_function = get_next_buf;
    pwcm->stop_function   = stop_acq;

    /* create control communications interface */
    pwcm->pwc = dada_pwc_create();
   
    /* connect to the shared memory */

    hdu = dada_hdu_create (log);

    if (dada_hdu_connect (hdu) < 0)
      return EXIT_FAILURE;

    if (dada_hdu_lock_write (hdu) < 0)
      return EXIT_FAILURE;

    header_size = ipcbuf_get_bufsz (hdu->header_block);
    multilog (log, LOG_INFO, "header block size = %llu\n", header_size);
    
    header_buf = ipcbuf_get_next_write (hdu->header_block);
    
    if (!header_buf)  {
	multilog (log, LOG_ERR, "Could not get next header block\n");
	return EXIT_FAILURE;
    }

    if (header_file)  {
      if (fileread (header_file, header_buf, header_size) < 0)  {
        multilog (log, LOG_ERR, "Could not read header from %s\n", header_file);
        return EXIT_FAILURE;
      }
    }
    else {
      header_strlen = strlen(header);
      memcpy (header_buf, header, header_strlen);
      memset (header_buf + header_strlen, '\0', header_size - header_strlen);
    }

    /* Set the header attributes */ 
    if (ascii_header_set (header_buf, "HDR_SIZE", "%llu", header_size) < 0) {
	multilog (log, LOG_ERR, "Could not write HDR_SIZE to header\n");
	return -1;
    }
    
    if (verbose) fprintf(stderr,"Observation ID is %s\n",ObsId);
    if (ascii_header_set (header_buf, "OBS_ID", "%s", ObsId) < 0) {
	multilog (log, LOG_ERR, "Could not write OBS_ID to header\n");
	return -1;
    }

    if (verbose) fprintf(stderr,"File size is %lu\n",fSize);
    if (ascii_header_set (header_buf, "FILE_SIZE", "%lu", fSize) < 0) {
	multilog (log, LOG_ERR, "Could not write FILE_SIZE to header\n");
	return -1;
    }

    if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
	multilog (log, LOG_ERR, "Could not mark filled header block\n");
	return EXIT_FAILURE;
    }

   
    /* The write loop */
    /* repeat the write loop for n buffers */   
    /* if we number of buffers is not specified, */
    /* find it from number of seconds, based on 80 MB/sec sample rate*/
    if (nbufs == 0)  nbufs = (80000000 * nSecs)/BUFSIZE;

    if (verbose) fprintf(stderr,"Number of bufs is %d\n",nbufs);

    while(dmadb.buf < nbufs){

	/* get data from EDT and write to datablock */
	if (( ipcio_write(hdu->data_block, dmadb.data, BUFSIZE) ) < BUFSIZE ){
	    multilog(log, LOG_ERR, "Cannot write requested bytes to SHM\n");
	    return EXIT_FAILURE;
	}
      
        /* find out the number of buffers completed */
	buf = edt_done_count(dmadb.edt_p);
	wrCount = ipcbuf_get_write_count ((ipcbuf_t*)hdu->data_block);
	rdCount = ipcbuf_get_read_count ((ipcbuf_t*)hdu->data_block);
	fprintf(stderr,"%d %d %d\n",buf,wrCount,rdCount);
 
    }
    
    if (dada_hdu_unlock_write (hdu) < 0)
      return EXIT_FAILURE;

    if (dada_hdu_disconnect (hdu) < 0)
      return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

