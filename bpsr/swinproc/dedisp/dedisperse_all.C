/*
  dedisperse  - dedisperses raw filterbank data or folded pulse profiles
*/
#include <stdlib.h>
#include "string.h"
extern "C" {
#include "dedisperse_all.h"
};
#include "getDMtable.h"
//#include "gtools.h"

void inline_dedisperse_all_help(){
  fprintf(stderr,"dedisperse_all help\n");
  fprintf(stderr,"Usage: dedisperse_all filename [options]\n");
  fprintf(stderr,"-k killfilename    kill all channels in killfilename\n");
  fprintf(stderr,"-d st_DM end_DM    dedisperse from st_DM to end_DM\n");
  fprintf(stderr,"-g gulpsize        number of samples to dedisp at once\n");
  fprintf(stderr,"-b                 debird mode, create 1 timfile per freq channel\n");
  fprintf(stderr,"-n Nsamptotal      Only do Nsamptotal samples\n");
  fprintf(stderr,"-s Nsamps          Skip Nsamp samples before starting\n");
  fprintf(stderr,"-l                 Create logfile of exact DMs used\n");
  fprintf(stderr," \n");
  fprintf(stderr,"dedisperse_all uses OpenMP and 16 bit words to\n");
  fprintf(stderr,"create many dedispersed files at once in a highly\n");
  fprintf(stderr,"parallel manner. It is for use on 64 bit machines\n");
  fprintf(stderr,"but still works on 32 bit machines.\n");
  fprintf(stderr,"Currently tested on 96x1bit files, 512x1bit files, 1024x2bit files.\n");
  fprintf(stderr,"Link against OpenMP (-fopenmp with GNU on linux)\n");
}

FILE *input, *output, *outfileptr, *dmlogfileptr;
char  inpfile[80], outfile[80], ignfile[80], dmlogfilename[180];

/* global variables describing the operating mode */
int ascii, asciipol, stream, swapout, headerless, nbands, userbins, usrdm, baseline, clipping, sumifs, profnum1, profnum2, nobits, wapp_inv, wapp_off;
double refrf,userdm,fcorrect;
float clipvalue,jyf1,jyf2;
int fftshift;
#include "wapp_header.h"
#include "key.h"
struct WAPP_HEADER *wapp;
struct WAPP_HEADER head;

/*
  tsamp in seconds, f0, df in MHz
  returns the DM for a given delay between the top and
  bottom frequency channel in samples.
*/

int load_killdata(int * killdata,int nchans,char * killfile){
  FILE * kptr;
  char line[100];
  kptr = fopen(killfile,"r");
    if (kptr==NULL){
      fprintf(stderr,"Error opening file %s\n",killfile);
      exit(-2);
    }
    for (int i=0; i<nchans;i++) {
      if (fgets(line,20,kptr)!=NULL){  // Read in whole line
	int nscanned = sscanf(line,"%d",&killdata[i]);
	if (nscanned==0) {
	  fprintf(stderr,"Could not scan %s as 1 or 0\n",line);
	  exit(-1);
	}
      } else{
	fprintf(stderr,"Error reading %dth value from %s\n",i,killfile);
	exit(-1);
      }
    }
  fclose(kptr);
  return(0);
}

float get_DM(int nsamples, int nchan, double tsamp, double f0, double df){

  float DM;
  float nu1,nu2,nu1_2,nu2_2;

  //printf("5vals %d %d %lf %lf %lf\n",nsamples,nchan,tsamp,f0,df);

  if (nsamples==0) return(0.0);

  nu1 = f0;
  nu2 = f0 + (nchan-1)*df;
  nu1_2 = 1.0e6/(nu1*nu1);
  nu2_2 = 1.0e6/(nu2*nu2);

  DM = (float) nsamples * tsamp/4.15e-3 * 1.0/(nu2_2-nu1_2);
  return(DM);
}

/*
  Returns the shift in samples for a given DM. Units as above.
*/

int DM_shift(float DM, int nchan, double tsamp, double f0, double df){

  float shift;
  float nu1,nu2,nu1_2,nu2_2;

  if (nchan==0) return(0);

  nu1 = f0;
  nu2 = f0 + (nchan)*df;
  nu1_2 = 1.0e6/(nu1*nu1);
  nu2_2 = 1.0e6/(nu2*nu2);

  //printf("nu1 %f nu2 %f DM %f\n",nu1,nu2,DM);

  shift = 4.15e-3 * DM * (nu2_2-nu1_2);
  //printf("shift is %f (s)\n",shift);
  //printf("shift in samples is %f\n",shift/tsamp);
  return ((int) (shift/tsamp+0.5));
}

int main (int argc, char *argv[])
{
  /* local variables */
  char string[80];
  int i,useroutput=0,nfiles,fileidx,sigproc,scan_number,subscan=0;
  int numsamps=0;
  unsigned char * rawdata;
  unsigned short int * unpacked; //, * times;
  int nbytesraw;
  int ibyte,j,k;
  unsigned char abyte;
  unsigned short int * times;
  unsigned long int * casted_times, * casted_unpacked;
  int idelay;
  int nread;
  float DM_trial;
  int ndm=0;
  float total_MBytes = 0;
  float start_DM=0.0, end_DM;
  int counts;
  int dmlogfile=0, verbose=0;
  int debird = 0;
  int readsamp = 0;
  int nreadsamp = 0;
  int skip = 0;
  int nskip = 0;
  int ntoload;
  int ntodedisp;
  int maxdelay = 0;
  int appendable = 0;
  int ngulp; //max number of samples at a time
  int gulping = 0;
  int ngulpsize;
  char * killfile;
  int killing=0;

  /* check number of command-line arguments and print help if necessary */
  if (argc<2) {
    inline_dedisperse_all_help();
    exit(0);
  }
  
  /* print help if necessary */
  if (strcmp(argv[1],"-h")==0) {
    inline_dedisperse_all_help();
    exit(0);
  }
  /* work out how many files are on the command line */
  i=1;
  nfiles=0;
  while(fopen(argv[i],"rb")!=NULL) {
    nfiles++;
    i++;
  }
  
  /* set up default globals */
  userbins=usrdm=asciipol=stream=clipping=swapout=headerless=0;
  sumifs=wapp_inv=wapp_off=barycentric=0;
  nobits=32;
  ascii=1;
  fftshift=1;
  profnum1 = 0;
  profnum2 = 1000;
  nbands=baseline=1;
  clipvalue=refrf=userdm=fcorrect=0.0;
  refdm=-1.0;
  output=NULL;
  strcpy(ignfile,"");

  /* now parse any remaining command line parameters */
  if (argc>nfiles) {
    i=nfiles+1;
    while (i<argc) {
      if (!strcmp(argv[i],"-d")) {
	/* get DM from the user */
	start_DM=atof(argv[++i]);
	end_DM=atof(argv[++i]);
	usrdm=1;
      }
      else if (!strcmp(argv[i],"-l")) {
	/* Create a log file of the DMs */
	dmlogfile=1;
      }
      else if (!strcmp(argv[i],"-v")) {
	verbose=1;
      }
      else if (!strcmp(argv[i],"-b")) {
	debird=1;
      }
      else if (!strcmp(argv[i],"-n")) {
	/* read only X samples */
	ntodedisp=atoi(argv[++i]);
	readsamp=1;
      }
      else if (!strcmp(argv[i],"-s")) {
	/* skip first X samples */
	nskip=atoi(argv[++i]);
	skip=1;
      }
      else if (!strcmp(argv[i],"-g")) {
	ngulpsize=atoi(argv[++i]);
	gulping=1;
      }
      else if (!strcmp(argv[i],"-a")) {
	appendable=1;
      }
      else if (!strcmp(argv[i],"-k")) {
	killing = 1;
	killfile = (char *) malloc(strlen(argv[++i])+1);
	strcpy(killfile,argv[i]);
      }
      else {
	/* unknown argument passed down - stop! */
	inline_dedisperse_all_help();
	fprintf(stderr,"unknown argument (%s) passed to %s",argv[i],argv[0]);
	exit(1);
      }
      i++;
    }
  }

  if (usrdm && debird){
    fprintf(stderr,"Cannot dedisperse and debird simultaneously!\n");
    exit(-1);
  }

  verbose=1;
  
  if (!useroutput) {
    /* no output file selected, use standard output */
    output=stdout;
    strcpy(outfile,"stdout");
  }

  if (!nfiles) {
    strcpy(inpfile,"stdin");
    nfiles=1;
  }
  fileidx=1;
  if (nfiles>1) {
    fprintf(stderr,"Multi file mode not supported yet!\n");
    exit(-1);
  }
  
  /* open input datafile and get type of file from the input data */
  strcpy(inpfile,argv[fileidx]);
  input=fopen(inpfile,"rb");
  if (input==NULL){
    fprintf(stderr,"Error opening file %s\n",inpfile);
    exit(-1);
  }
  /* read in the header to establish what the input data are... */
  sigproc=read_header(input);
  if (!sigproc) {
    fprintf(stderr,"Not sigproc data\n");
    exit(-1);
  }    
  if (foff > 0.0) {
    fprintf(stderr,"dedisperse can't handle low->high frequency ordering!");
    exit(1);
  }
  if (fileidx == 1) {
    /* this is filterbank data */
    if (output!=stdout) output=fopen(outfile,"wb");
    if (output==NULL){
      fprintf(stderr,"Error opening file %s\n",output);
      exit(-1);
    }
  }
  
  numsamps = nsamples(inpfile,sigproc,nbits,nifs,nchans);	/* get numsamps */
  if (usrdm) maxdelay = DM_shift(end_DM,nchans,tsamp,fch1,foff);
  cout << "maxdelay = " << maxdelay << endl;
  
  if (readsamp) numsamps=ntodedisp+maxdelay;
  
  if (gulping) {
    if (ngulpsize>numsamps) ngulpsize = numsamps-maxdelay;
    ntodedisp=ngulpsize;
  }
  ntoload = ntodedisp + maxdelay; 
  
  nbytesraw = nchans * ntoload * nbits/8;
  if ((rawdata = (unsigned char *) malloc(nbytesraw))==NULL){
      fprintf(stderr,"Error allocating %d bytes of RAM for raw data\n",nbytesraw);
      exit(-1);
  }
  // skip either 0 or nskip samples
  fseek(input, nskip*nchans*nbits/8, SEEK_CUR);

  float * DMtable;
  double ti = 40.0;
  double tol = 1.25;
  
  getDMtable(end_DM, tsamp*1e6, ti, foff, (fch1+(nchans/2-0.5)*foff)/1000,
	     nchans, tol, &ndm, DMtable);

  int nwholegulps = (numsamps - maxdelay)/ngulpsize;
  int nleft = numsamps - ngulpsize * nwholegulps;
  int ngulps = nwholegulps + 1;

  if (!debird){
    unpacked = (unsigned short int *) malloc(nchans*ntoload*
					     sizeof(unsigned short int));
    if (unpacked==NULL) {
      fprintf(stderr,"Failed to allocate space for unpacked %d bytes\n",
	      (int)(nchans*ntoload*sizeof(unsigned short int)));
      exit(-2);
    }
    times = (unsigned short int *) 
      malloc(sizeof(unsigned short int)*ntodedisp); 
    if (times==NULL){
      fprintf(stderr,"Error allocating times array\n");
      exit(-1);
    }
    casted_times = (unsigned long int *) times;
    casted_unpacked = (unsigned long int *) unpacked;
  }

  int * killdata = new int[nchans];
  if (killing) int loaded = load_killdata(killdata,nchans,killfile);

  // Start of main loop
  for (int igulp=0; igulp<ngulps;igulp++){

    if (igulp!=0) fseek(input,-1*maxdelay*nbits*nchans/8,SEEK_CUR);
    if (igulp==nwholegulps) {
      ntoload = nleft;
      nbytesraw = ntoload*nbits*nchans/8;
      ntodedisp = nleft - maxdelay;
    }

    fprintf(stderr,"Gulp %d Loading %d samples, i. e. %d bytes of Raw Data\n",igulp, ntoload, nbytesraw);
    nread = fread(rawdata,nbytesraw,1,input);
    
    if (nread != 1){
      fprintf(stderr, "Failed to read %d nread = %d \n", nbytesraw, nread);
      exit(-1);
    }
  
    if (debird){
      refdm=0;
      nobits=8;
      float orig_fch1 = fch1;
      int orig_nchans = nchans;
      for (int ichan=0;ichan<nchans;ichan++){
	sprintf(outfile,"%s.%4.4d.tim", inpfile, ichan);
	if (igulp==0) outfileptr=fopen(outfile,"w");
	if (igulp!=0) outfileptr=fopen(outfile,"a");
	output=outfileptr;
	nchans=1;
	fch1=fch1+foff*ichan;
	if (igulp==0) dedisperse_header();
	fch1=orig_fch1;
	nchans=orig_nchans;
	
	if (nbits==1){
	  ibyte=ichan/8;
	  for (j=0;j<ntoload;j++){
	    abyte = rawdata[ibyte+j*nchans/8];
	    k=ichan-ibyte*8;
	    unsigned char databyte = (unsigned char)((abyte>>k)&1);
	    fwrite(&databyte,1,1,outfileptr);
	  }
	}
	if (nbits==2){
	  ibyte=ichan/4;
	  for (j=0;j<ntoload;j++){
	    abyte = rawdata[ibyte+j*nchans/4];
	    k=ichan-ibyte*4;
	    unsigned char databyte = (unsigned char)((abyte>>k)&3);
	    fwrite(&databyte,1,1,outfileptr);
	  }
	}
	fclose(outfileptr);
      }
    }
    
  /* Unpack it if dedispersing */
  
  if (!debird){
    if (verbose) fprintf(stderr,"Reordering data\n");
    // all time samples for a given freq channel in order in RAM
    if (nbits==1){
      for (ibyte=0;ibyte<nchans/8;ibyte++)
#pragma omp parallel for private (abyte,k,j)
	for (j=0;j<ntoload;j++){
	  abyte = rawdata[ibyte+j*nchans/8];
	  for (k=0;k<8;k++)
	    unpacked[j+(ibyte*8+k)*ntoload]=(unsigned short int)((abyte>>k)&1);
	}
    }
    
    if (nbits==2){
      for (ibyte=0;ibyte<nchans/4;ibyte++)
#pragma omp parallel for private (abyte,k,j)
	for (j=0;j<ntoload;j++){
	  abyte = rawdata[ibyte+j*nchans/4];
	  for (k=0;k<8;k+=2)
	    unpacked[j+(ibyte*4+k/2)*ntoload]=(unsigned short int)((abyte>>k)&3);
	}
    }
    
    if (killing){
      for (int i=0;i<nchans;i++){
	if (!killdata[i]){
	  cout << i << " " ;
#pragma omp parallel for private(j)	  
	  for (int j=0;j<ntoload;j++)
	    unpacked[j+i*ntoload]=0;
	}
      }
    }

    cout << endl;
    //vector <Gpulse> giant;
    //vector <Gpulse> allgiants;
    
    /* for each DM dedisperse it */
    /* Start with zero DM */
    if (!usrdm) end_DM=1e3;
    if (dmlogfile){
      sprintf(dmlogfilename,"%s.dmlog",inpfile);
      dmlogfileptr=fopen(dmlogfilename,"w");
      if (dmlogfileptr==NULL) {
	fprintf(stderr,"Error opening file %s\n",dmlogfilename);
	exit(-3);
      }
    }
    if (igulp==0) appendable=0; else appendable=1;
    for (int idm=0;idm<ndm;idm++)
      {
	//DM_trial = get_DM(idm,nchans,tsamp,fch1,foff);
	DM_trial = DMtable[idm];
	if (DM_trial>=start_DM && DM_trial<=end_DM){
	  if (verbose) fprintf(stderr,"DM trial #%d at  DM=%f ",idm,DM_trial);
	  if (dmlogfile) fprintf(dmlogfileptr,"%f %07.2f\n",DM_trial,
				 DM_trial);
	  // Name output file, dmlog file
	  sprintf(outfile,"%s.%07.2f.tim",inpfile,DM_trial);
	  // open it
	  if (appendable) outfileptr=fopen(outfile,"a");
	  if (!appendable) outfileptr=fopen(outfile,"w");
	  output=outfileptr;
	  // write header variables into globals
	  refdm = DM_trial;
	  nobits = 8;
	  // write header
	  if (!appendable) dedisperse_header();
	  // ntodedisp and numsamps??
	  for (j=0;j<(ntodedisp)/4;j++)casted_times[j]=0;
	  for (k=0;k<nchans;k++){
	    idelay = DM_shift(DM_trial,k,tsamp,fch1,foff);
#pragma omp parallel for private(j)    
	    for (j=0;j<(ntodedisp)/4;j++){
	      casted_times[j]+=*((unsigned long int *) (unpacked+(j*4+k*ntoload+idelay)));
	    }	    
	  }
	  
	  // Gsearch!
	  //giant=findgiants(numsamps-idm,times,6,30,256);
	  //allgiants.insert(allgiants.end(),giant.begin(),giant.end());
	  //printf("Giant count now %d DM is %f\n",allgiants.size(),DM_trial);
	  // write data
	  int rotate = 1;
	  if (nchans==1024 && nbits==2) rotate = 4; // max 3072/16=192
	  if (nchans==512 && nbits==1) rotate = 2;  // max 128
	  if (nchans==96) rotate = 0;               // max 96
	  for (int d=0;d<ntodedisp;d++){
	    unsigned short int twobytes = times[d]>>rotate;
	    unsigned char onebyte = twobytes;
	    fwrite(&onebyte,1,1,outfileptr);
	  }
	  // close file
	  fclose(outfileptr);
	  if (verbose) fprintf(stderr,"%d bytes to file %s\n", ntodedisp, outfile);
	  total_MBytes+= ntodedisp/1.0e6;
	} // if DM is in range
      } // DM for loop
    if (verbose) fprintf(stderr,"Wrote a total of %6.1f MB to disk\n",total_MBytes);
    /* close log files if on last input file */
    if (dmlogfile) fclose(dmlogfileptr);
  }
  }
  return(0); // exit normally
} // main
