#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <syslog.h>

#include "ascii_header.h"
#include "fsleep.h"
#include "dada_db.h"

void usage()
{
  fprintf(stderr,
	  "dada_db2disk [options]\n"
	  " -s <filesize>\n"
	  " -b <buffersize>\n"
	  " -f <rootfilename>\n"
	  " -d run as daemon\n"
	  " -A Adaptive sleep algorithm (slow virtual memory flushing)\n");
}

int main (int argc, char **argv)
{
  /* DADA Data Block ring buffer */
  dada_db_t shm = DADA_DB_INIT;

  /* Name of the host on which db2disk is running */
  char hostname [HOST_NAME_MAX];

  /* Names of the disks to which db2disk will write data */
  char** disks = 0;

  /* Size of the buffer used for file I/O (apart from header) */
  uint64 buffer_size = 0;

  /* The buffer used for file I/O */
  char* buffer = 0;

  /* Amount by which data in output files will overlap */
  uint64 overlap = 0;


  /* buffer details */
  int nbufs = 0;
  int bufsz = 0;
  int chunk = 0;
  int buffbacklog = 0;
  int turn_on_sync = 0;
  char *buffer = NULL;
  char *temp_buffer = NULL;



  int done=0;

  /* MBytes from ascii header */
  long mbytes;
  uint64 bytes_since_sod =0;

  /* file descriptor */
  int out_f = -1;

  /* file size counter */
  uint64 filesz=0;
  uint64 max_filesz = 0;
  uint64 edgeflag = 0;
  int rename_file = 0;
  /* write chunksize */
  size_t write_buff=0;
  size_t each_write=0;

  uint64 written = 0;
  int smart = 1;
  /* Header info pointers */
  int int_num;

  /* the UTC_START from each header */
  char obsid[256];
  char mode[256];

  long blkid=0;
  pid_t pid;  
  int daemon = 1;
  int nbits = 2;
  int nchan=2;
  char *tmp_hdr=NULL;
  uint64 ns_p_buff=0;
  uint64 ns_p_byte=0;
  uint64 nw_p_buff=0;
  uint64 tagcount=0;
  int test_sep=0;

  int testing=0;
  float adaptive_delay = 0.01;
  int write_f=0;
  char host[64];
  int delay = 0;
  int sync_count = 0;
  int donotwrite = 0;
 
/* Whoami */
  setuid(USRID);


  while ((arg=getopt(argc,argv,"f:s:w:T:n:S:dD:W")) != -1) {
    switch (arg) {
      case 'd':
	daemon=0;
	break;
      case 'f':
	strcpy(rootfile,(char *) optarg);
	fprintf(stderr,
	    "Writing buffer contents to files with root %s\n"
	    ,rootfile);
	fflush (stderr);
	smart = 0;
	write_f =1;
	break;
      case 's':
	max_filesz = atoi(optarg);
	write_f=1;
	break;
      case 'w':
	write_buff=atoi(optarg);
	write_f=1;
	break;
      case 'T':
	test_sep = atoi(optarg);
	smart = 1;
	testing = 1;

	break;
      case 'D':
        delay = atoi(optarg);
        break;
      case 'W':
	donotwrite = 1;
        break;
      default:
	usage ();
	return EXIT_FAILURE;
    }
  }
  /* set up for daemon usage */	  
  if (daemon==1) {
    if ((pid = fork()) < 0) {
      exit(EXIT_FAILURE);
    }
    else if (pid != 0) {
      exit(EXIT_SUCCESS);
    }

     setsid(); 
     setuid(500);
     /* umask(0);*/
  }
  
  openlog("dada_db2disk",LOG_CONS,LOG_USER);
  /* I have not tested the rtn value of this so be careful */

  /* First connect to the shared memory */

  if (dada_db_connect (&shm) == -1) {
    if (!daemon) {
      fprintf (stderr,
	  "Failed to connect to shared memory area\n");
      fflush (stderr);
    }
    else {
      if (VERBOSE)
        syslog(LOG_ERR,"Failed to connect to shared memory: %m");
    }
    return EXIT_FAILURE;
  }
  else {
    if (daemon) {
      if (VERBOSE)
        syslog(LOG_INFO,"connected to shared memory");
    }
    if (!daemon) {
      fprintf(stderr,"connected to shared memory");
    }
  }

  if (ipcbuf_lock_read (&(shm.ipcio.buf)) < 0) {
    if (!daemon)
      fprintf (stderr, "Could not lock designated reader status\n");
    else {
      if (VERBOSE)
        syslog (LOG_ERR, "Could not lock designated reader status: %m\n");
    }
    return EXIT_FAILURE;
  }

  nbufs = ipcbuf_get_nbufs ((ipcbuf_t *) &shm);
  bufsz = ipcbuf_get_bufsz ((ipcbuf_t *) &shm);
  
  ns_p_buff = ((uint64) bufsz/((uint64)nbits*(uint64)nchan))*8;
  ns_p_byte = (uint64) 8/((uint64)nbits*(uint64)nchan);
  nw_p_buff = (uint64) bufsz/2;
  /* get filesize */
  
  if (max_filesz == 0) { 
    if (smart) {
      /* max_filesz = _1GB + dada_header_size;*/
      max_filesz = (1.5*_1GB);
      sprintf(rootfile,BASEBAND_DIR);

    }
    else if (!smart) {
      max_filesz = (1.5*_1GB);
    }

  }

  /* get page size */

  if (write_buff == 0) {
    write_buff = 256*sysconf (_SC_PAGE_SIZE);
  } 
  if (!daemon) { 
    fprintf(stderr,
	"\nReported write buffersize(bytes): %d\n",
	(int)write_buff);
    fflush (stderr);
  } 
  /* Need a clever way to shut this down */
  done = 0; 
  while (1) {

    if (!daemon)
      fprintf(stderr,"Waiting for header\n");
    tmp_hdr = dada_db_get_header_read(&shm);
    if (!daemon)
      fprintf(stderr,"Got header:'%s'\n",tmp_hdr);

    memcpy(header,tmp_hdr,dada_header_size);;
    dada_db_mark_header_read(&shm);

    if (ascii_header_get(header,"NMBYTES",UI64,&mbytes) < 0) {
      if (!daemon) {
	fprintf(stderr,"NMBYTES key not in header.  Attached to engine.\n");
      }
      else {
        if (VERBOSE)
	  syslog(LOG_ERR,"NMBYTES key not in header.  Attached to engine.");
      }
      bytes_since_sod = 0;
    }
    else {
      bytes_since_sod = (uint64)mbytes * _1MB;
      if (!daemon)
        fprintf(stderr,"OFFSET! "UI64"\n", bytes_since_sod);
      else { 
        if (VERBOSE)
          syslog(LOG_ERR,"DB2DISK::OFFSET "UI64"\n",bytes_since_sod);
      }
    }

    /* this is an edge */
    if (ascii_header_get(header,"EDGE",UI64,&edgeflag) < 0) {
     if (!daemon) {
       fprintf(stderr,"EDGE key not in header.\n");
     }
     else {
       if (VERBOSE)
         syslog(LOG_ERR,"EDGE key not in header.");
     }
    rename_file = 0;
    edgeflag = 0;
    }
    else {
      if (edgeflag > 0) {
        rename_file=0;
        edgeflag = edgeflag* _1MB;
        max_filesz = _1GB + edgeflag + dada_header_size;
      }
      else {
        rename_file=0;
      }
    }


    while ( !(ipcbuf_eod (&shm.ipcio.buf)) ) { 

      if ( !smart && out_f < 0) {

	sprintf(outfile,"%s.%d",rootfile,done);

	if (DEBUG)
	  fprintf (stderr, "Opening data file: %s\n", outfile);

	out_f = open (outfile, O_CREAT | O_TRUNC | O_WRONLY ,0666 );

	if (out_f == -1) {
	  if (!daemon) {
	    fprintf(stderr,"Failed to open output file: '%s'\n", outfile);
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Failed to open output file");
	  }
	  return EXIT_FAILURE;

	}	  

	filesz=0;
      }


      /* yes i know this is dumb */


      else if (smart == 1 && write_f == 1 && out_f < 0 ) {

	/* need to interrogate the header for integration number and block number */
	/* assign this to a file */
	/* need a routine to do this */
	/* how about we treat the whole header as a string and use strstr to get the key*/

	/* this now needs re-writing in terms of the new paradigm of communication through a header area */ 

	if (!daemon)
	  fprintf (stderr, "HEADER='%s'\n", header);

	int_num = ascii_header_get(header,"UTC_START","%s",obsid);
	if (int_num == -1) {
	  if (!daemon) {
	    fprintf(stderr,"Cannot find UTC_START key in header\n");
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Cannot find UTC_START key in headers");
	  }
	  return EXIT_FAILURE;


	}

        /* EXPERIMENTAL CHECK FOR LOGIC AND CONSISTENCY - USED IN THE CASE OF MULITPLE FILES */

	int_num = ascii_header_get(header,"EDGE",UI64,&edgeflag);
	if (int_num == -1) {
	  if (!daemon) {
	    fprintf(stderr,"Cannot find EDGE key in header\n");
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Cannot find EDGE key in headers");
	  }
          edgeflag = 0;
	}
        else {
          edgeflag = edgeflag * _1MB;
        }
        
      
        max_filesz = edgeflag + _1GB + dada_header_size;
 
	if (ascii_header_set(header,"OFFSET",UI64,bytes_since_sod) < 0) {
	  if (!daemon) {
	    fprintf(stderr,"Setting byte offset in header\n");
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Setting byte offset in header\n");
	  }
	  return EXIT_FAILURE;
	}
        
        if (ascii_header_get(header,"MODE","%s",mode) < 0) {
          if (!daemon) {
             fprintf(stderr,"Setting byte offset in header\n");
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Setting byte offset in header\n");
	  }
        }

        if (strcmp("SEARCH",mode) == 0) {
          if (ascii_header_set(header,"MODE","SURVEY") < 0) {
            if (!daemon) {
              fprintf(stderr,"Setting mode offset in header\n");
            }
            else {
              if (VERBOSE)
                syslog(LOG_ERR,"Setting mode offset in header\n");
            }
          }
        }

  
	sprintf(outfile,"%s/%s_%ld_%s",rootfile,obsid,blkid,host);
	sprintf(tempfile,"%s.writing",outfile);

	blkid ++;
	out_f = open (tempfile,O_CREAT | O_TRUNC | O_WRONLY ,0666 );
        /* outstream = fdopen (out_f,"w");*/

	if (out_f == -1) {
	  if (!daemon) {
	    fprintf(stderr,"Failed to open output file: '%s':%s\n",
		outfile, strerror(errno));
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Failed to open output file: '%s':%m",outfile);
	  }
	  return EXIT_FAILURE;
	}
        if (donotwrite) {
          chunk = dada_header_size;
        }
        else {	
	  chunk = write (out_f, header , dada_header_size);
          /* chunk = fwrite(header,1,dada_header_size,outstream);*/
          /* fflush(outstream);*/
        }

	if (chunk != dada_header_size) {
	  if (!daemon) {
	    perror("Failed to write: ");
	    fflush (stderr);
	  }
	  else {
            if (VERBOSE)
	      syslog(LOG_ERR,"Failed to write because %m");
	  }
	  return EXIT_FAILURE;
	}
	filesz = dada_header_size;
      }



      if (write_f == 1 ) {


	while ((uint64) filesz < max_filesz && ipcbuf_eod (&(shm.ipcio.buf)) == 0) {
          
	  tagcount = bytes_since_sod;
          if (DEBUG) {
	    fprintf(stdout,"Waiting for buffer \n");
	    fflush(stdout);
	  }

	  if (buffer == NULL) {

            if (!daemon)
              fprintf(stderr,"Checking semaphores\n");
           
            readbuf = ipcbuf_get_read_count (&shm.ipcio.buf);
	    writebuf = ipcbuf_get_write_count (&shm.ipcio.buf);
            if (!daemon)
              fprintf(stderr,"Number of written buffers "UI64"\n",writebuf-readbuf);

            buffbacklog = writebuf-readbuf;
            
            if (buffbacklog < nbufs/2) {
              turn_on_sync = 1;
            }
            else {
              turn_on_sync = 0;
            }
                
            if (writebuf > readbuf) {
	      buffer = ipcbuf_get_next_read(&shm.ipcio.buf, &readbuf);
	      written = 0;
            }
            else {

              if (delay)
                fsleep(delay);

              continue;
            }
            
	  }
          if (DEBUG) {
	    fprintf(stdout,"Got buffer\n");
	    fflush(stdout);
	  }

	  if (buffer == NULL){
	    if (!daemon) {
	      fprintf(stderr,"Problems getting shared memory pointers\n");
	      return EXIT_FAILURE;
	    }
	  }
	  if (testing==1 && buffer != NULL){

	    if (yama_confirm(buffer,readbuf,tagcount,test_sep) == EXIT_FAILURE) {
	      syslog(LOG_ERR,"DB2DISK::Failed to confirm YAMASAKI: BUFFSZ: "UI64": OFFSET: "UI64"\n", readbuf, tagcount);
	      exit(EXIT_FAILURE);
	    }

	  }
	  each_write = write_buff;
          
	  tagcount = tagcount + (readbuf/2.);

	  while (written < readbuf && (uint64) filesz < max_filesz) {

	    if (readbuf - written < each_write)
	      each_write = readbuf - written;

	    
            if (filesz + each_write >= max_filesz)
	      each_write = max_filesz - filesz;
	  
            if (donotwrite) {
              chunk = each_write;
            }
            else {	
            
             if (!daemon) 
               fprintf(stderr,"About to write %d\n",each_write);    
                          
             chunk = write ( out_f, (buffer+written), each_write );
             /* chunk = fwrite ((buffer+written),1,each_write,outstream); */
	     /* fflush(outstream);*/
             /* fsleep(adaptive_delay);*/
             if (!daemon) 
               fprintf(stderr,"written %d\n",chunk);    
	    }
            if (chunk != each_write) {
	      if (!daemon) {
		perror("Failed to write: ");
		fflush (stderr);
	      }
	      else {
                if (VERBOSE)
		  syslog(LOG_ERR,"Failed to write because %m");
	      }
	      return EXIT_FAILURE;
	    }
	    written = written + chunk;
	    filesz = filesz + chunk;
	    bytes_since_sod += chunk;
	  }
	  if (written >= readbuf) {
	    
	    if (ipcbuf_eod(&(shm.ipcio.buf)) == 0) {
              
              if (turn_on_sync)
                fdatasync(out_f); 
              
              ipcbuf_mark_cleared (&(shm.ipcio.buf));
	    }

	    done=done+1; 
	  
	    if (DEBUG) {

	      fprintf(stdout,"total buffers done %d\r",done);
	      fflush(stdout);
	    }
	    buffer = NULL;
	  }
          /* this is a temporary kludge to avoid a race condition on the very last file of an observation */
          fsleep(0.01);
	}

	if (DEBUG)
	  fprintf (stderr, "Closing file\n");
        
        if (VERBOSE)
          syslog(LOG_ERR,"DB2DISK::Closing file");	
	
	close(out_f);
        /* fclose(outstream);*/
        if (!rename_file) {
          sprintf(command,"mv %s %s.cpsr2",tempfile,outfile);
        }
        else {  
          sprintf(command,"mv %s %s.edge",tempfile,outfile);
	  rename_file=0;
        }

	out_f = -1;
        
	if (smart) {
	  max_filesz = _1GB + edgeflag + dada_header_size; 
        }
      }
    }
    if (VERBOSE)
      syslog (LOG_ERR, "acknowledge end of data\n");
    ipcbuf_ack_eod (&(shm.ipcio.buf));
  }


  return EXIT_SUCCESS;
}

