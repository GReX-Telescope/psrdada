/***************************************************************************/
/*                                                                         */
/* main module for plot4mon                                                */
/*                                                                         */
/* Ver 1.0        AP 20 Sept 2008                                          */
/*                                                                         */
/* code for generating various  1-D or 2-D plots (standard device set by   */
/* the variable STD_DEVICE in plot4mon.h) for monitoring the BPSR data     */
/* taking. Usage is reported on-line with plot4mon -h                      */
/*                                                                         */
/* Ver 2.0 RB 04 Oct 2008
   modified for new file format and to overplot bandpass and time series for 
   both pol0 and pol1, labels and title based on file extension, default dev
   png                                                                     */ 								
/* DRAFT VERSION                                                           */
/* (REFINEMENTS IN GRAPHICS, INTERACTIVE SETTING OF PARAMETERS, ADDTIONAL  */
/*  PLOTS... ETC... ARE IN PROGRESS)                                       */ 
/*                                                                         */ 
/***************************************************************************/

#include "plot4mon.h"

int main (int argc, char *argv[])
{
  int plotnum=0,dolog=0;
  char inpfile[80],inpdev[80],outputfile[80];
  char inpfile0[80],inpfile1[80];
  char xlabel[80],ylabel[80],plottitle[80];
  char add_work[8];
  long totvaluesread,totvalues4plot;
  int nchan,ndim,firstdump_line,work_flag,nbin_x,nsub_y;
  float yscale,tsamp;
  float x_read[MAXNUMPOINTS];
  float y_read[MAXNUMPOINTS];
  float y_read1[MAXNUMPOINTS];
  float y_new[MAXNUMPOINTS];
  float y_new1[MAXNUMPOINTS];

  /* reading the command line   */
  //get_commandline(argc,argv,inpfile,inpdev,outputfile);
  get_commandline(argc,argv,inpfile0,inpfile1,inpdev,outputfile,&dolog);

  fprintf(stderr,"log bandpass %d \n",dolog);

  /* determining the relevant parameters of the data and plot */
  read_params(inpfile1,&nchan,&tsamp,
	      &ndim,&yscale,&firstdump_line,
	      &work_flag,add_work);
  /* ignore this for the moment to keep things simple */
  work_flag=0;
  strcpy(add_work,"null");

  /* reading the data and filling the y array with them */
  read_stream(inpfile0,&y_read[0],&totvaluesread);
  read_stream(inpfile1,&y_read1[0],&totvaluesread);


  /* enter the loop on the plots to be produced with the data in inpfile1 */
  while (plotnum <= work_flag) { 
   
  /* perform additional work (fft,averaging,etc...) on the data if required */ 
  work_on_data(inpfile0,&y_read[0],&y_new[0],totvaluesread,
	       &totvalues4plot,tsamp,yscale,plotnum,add_work,dolog);
  work_on_data(inpfile1,&y_read1[0],&y_new1[0],totvaluesread,
	       &totvalues4plot,tsamp,yscale,plotnum,add_work,dolog);

  /* creating the name for the output pgplot file */ 
  create_outputname(inpfile1,inpdev,outputfile,plotnum,add_work);

  /* assigning the labels of the plot */
  //create_labels(inpfile1,plotnum,xlabel,ylabel,plottitle);

  // labels and title
  if (strstr(inpfile1, "bps") != NULL) {
    strcpy(xlabel," Frequency channel ");
    strcpy(ylabel," Power level ");
    strcpy(plottitle," RMS Bandpass ");
  } else if (strstr(inpfile1,"bp") != NULL) {
    strcpy(xlabel," Frequency channel ");
    strcpy(ylabel," Power level ");
    strcpy(plottitle," Mean Bandpass ");
  } else if (strstr(inpfile1,"ts") != NULL) {
    strcpy(xlabel," Time sample ");
    strcpy(ylabel," Power level ");
    strcpy(plottitle," Zero DM Time Series");
  }

  /* plotting the data */
  if (ndim==1) 
  {
    /* filling the x array with suitable indexes */
    create_xaxis(inpfile1,plotnum,totvaluesread,totvalues4plot,
                 nchan,tsamp,&x_read[0]);  
    /* creating a 1-D plot with pgplot */
    //plot_stream_1D(&x_read[0],&y_new[0],totvalues4plot,
		//outputfile,xlabel,ylabel,plottitle);
    plot_stream_1D(&x_read[0],&y_new[0],&y_new1[0],totvalues4plot,
		outputfile,xlabel,ylabel,plottitle);
  } 
  else if (ndim==2) 
  {
    set_array_dim(totvalues4plot,nchan,&nbin_x,&nsub_y);
    /* creating a 2-D plot with pgplot */
    plot_stream_2D(&y_new[0],nbin_x,nsub_y,yscale,firstdump_line,
		outputfile,xlabel,ylabel,plottitle);  
  }
  plotnum++; 
  } // closing the loop

  /* finishing the program */
  exit(0);
}
