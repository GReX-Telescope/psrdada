/***************************************************************************/
/*                                                                         */
/* function read_params                                                    */
/*                                                                         */
/* it assignes (to be changed with a reading from a header)                */
/* various parameters of the collected data                                */
/* On the basis of input file name, it selects the dimension of the plot   */
/* to be produced (ndim) and the optional scaling of the Y-axis data       */
/* (yscale) as well as the dimension of the array [nbin_x*nsub_y]          */
/* for the case of a 2-D plot.                                             */
/* Now it also determines the blocksize for reading the data (the latter   */
/* will be fixed at the best value after some experience of using the code)*/
/*                                                                         */
/* RB 05 Sept 2008: X axes in proper units (e.g. MHz, secs, Hz)            */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"
#include <string.h>

void read_params(char inpfile[], int *nchan, float *tsamp, float *fch1,
                 float *chbw, int *ndim, float *yscale, 
		 int *firstdump_line, int *work_flag, char add_work[])
{
  printf(" Reading the parameters of the file %s \n",inpfile);

  if (strings_compare(inpfile,"bpm0.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"bpm1.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"bp0.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=2;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"bp1.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=2;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"bp0rms.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=2;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"bp1rms.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=2;
    *yscale=1.0;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  else if (strings_compare(inpfile,"time0.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=*nchan;
    *firstdump_line=1;
    *work_flag=1;
    strcpy(add_work,"fft");
  }
  else if (strings_compare(inpfile,"time1.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=*nchan;
    *firstdump_line=1;
    *work_flag=1;
    strcpy(add_work,"fft");
  }
  else if (strings_compare(inpfile,"rawstat0.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=1;
    *firstdump_line=1;
    *work_flag=1;
    strcpy(add_work,"fft");
  }
  else if (strings_compare(inpfile,"rawstat1.dat")) 
  {
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=1;
    *firstdump_line=1;
    *work_flag=1;
    strcpy(add_work,"fft");
  }
  else
  {
    printf(" Not a recognized file name!");
    printf(" Applying nchan=1024 and tsamp=64 us to the data \n");
    *fch1=1582.0;
    *chbw=0.390625;
    *nchan=1024;
    *tsamp=64e-6;
    *ndim=1;
    *yscale=1;
    *firstdump_line=1;
    *work_flag=0;
    strcpy(add_work,"null");
  }
  printf(" The number of channels = %d \n",*nchan);
  printf(" The sampling time is = %f us \n",*tsamp*1e6);
  printf(" The dimension of the plot = %d \n",*ndim);
  printf(" The Y-scaling factor of the plot = %f \n",*yscale);
  printf(" The first dump is plotted [yes=1,no=0] = %d \n",*firstdump_line);
  printf(" Number of plots produced from the data = %d \n",*work_flag+1);
}
