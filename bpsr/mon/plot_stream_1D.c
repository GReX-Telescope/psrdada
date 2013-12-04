/***************************************************************************/
/*                                                                         */
/* function plot_stream_1D                                                 */
/*                                                                         */
/* Ver 2.0 RB 06 Oct 2008                                                  */
/*              Implemented resolution (pixel dimension) mode,             */
/*              option to plot band pass on logarthmic scale,              */
/*		option to omit labels, option to omit plot box             */
/*                                                                         */
/*                                                                         */
/* Ver 4.0 RB 24 Aug 2009                                                  */
/*              display a red box warning if data samples are all zero     */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void plotdat (float* x, float* y, long nvalues, int dommm)
{
  if (!dommm)
  {
    cpgline(nvalues, x, y);
  }
  else
  {
    unsigned i,j;
    // do three lines!
    for (i=0; i<3; i++)
      {
	float* y3 = y + i;
	cpgmove (x[0],y3[0]);
	for (j=1; j<nvalues; j++)
	  cpgdraw(x[j], y3[j*3]);
      }
  }
}

int plot_stream_1D(float x[], float y[], float y1[], 
		   long nvalues,long inivalue, long endvalue,
		   char inpdev[],char xlabel[],char ylabel[],char plottitle[], 
		   int dolabel, int dobox, int dommm,
		   unsigned width_pixels, unsigned height_pixels)
{
  long kk=0;
  float max_x, min_x, max_y, min_y, max_y1, min_y1; 
  float marg_x, marg_y;
  char xopt[10],yopt[10];
  float xtick=0.0, ytick=0.0;
  int nxsub=0, nysub=0;

  /*
   *  Compute the min and max values of the plotted part of input array(s)
   */
  nvalues=endvalue-inivalue;
  compute_extremes(x,endvalue,inivalue,&max_x,&min_x);
  compute_extremes(y,endvalue,inivalue,&max_y,&min_y);
  compute_extremes(y1,endvalue,inivalue,&max_y1,&min_y1);

  // fix y range good for both y and y1
  if (max_y1 > max_y) max_y = max_y1;
  if (min_y1 < min_y) min_y = min_y1;

  if (fabs(max_y - min_y) < 0.0001) {
    fprintf(stderr, "compute_extremes returned equal min/max values\n");
    min_y = 0.0;
    max_y = 0.1;
  }

  /*
   *  Compute reasonable margins for plotting purposes
   */
  compute_margin(nvalues,max_x,min_x,&marg_x);
  compute_margin(nvalues,max_y,min_y,&marg_y);
  /*
   * Reporting number of values and extremes of the plot
   */
#ifndef PLOT4MON_QUIET
  printf("\n About to plot %ld values using pgplot \n", nvalues);
  printf(" Extremes: Xmin = %f     Xmax = %f \n",min_x,max_x);
  printf("           Ymin = %f     Ymax = %f \n",min_y,max_y);
  printf(" Margins : Xmargin = %f \n",marg_x); 
  printf(" Margins : Ymargin = %f \n",marg_y); 
#endif

  int blank=0;
  if (min_y == 0.00 && max_y == 0.00) { blank = 1; }

  // initiate plot if display device selected 
  if(cpgbeg(0, inpdev, 1, 1) != 1) return -2;

  // set resolution if pixels dimens selected
  if (width_pixels && height_pixels)
   set_dimensions (width_pixels, height_pixels);

  // omit box totally if -nobox option selected
  if(!dobox){
    strcpy(xopt," ");
    strcpy(yopt," ");
    float xleft=0.0, xright=1.0, ybot=0.0, ytop=1.0;
    cpgsvp(xleft, xright, ybot, ytop);
    cpgswin(min_x, max_x+marg_x, min_y, max_y+marg_y);
    cpgbox(xopt, xtick, nxsub, yopt, ytick, nysub);
  } else cpgenv(min_x, max_x+marg_x, min_y, max_y+marg_y, 0, 0);

  // omit labels and title if -nolabel option selected 
  if(dolabel) cpglab(xlabel,ylabel,plottitle);

  cpgsch(1.2);
  cpgslw(2);

  // plot pol0 
  cpgsci(2);
  plotdat (x, y, endvalue, dommm);

  // plot pol1
  cpgsci(10);
  plotdat (x, y1, endvalue, dommm);

  /* Display a "red box" warning if data samples are zero */

  float xc, yc, xb[4], yb[4];
  char message[100];
  strcpy(message,"Data=0.0");
  xb[0]=min_x; yb[0]=min_y;
  xb[1]=max_x; yb[1]=min_y;
  xb[2]=max_x; yb[2]=max_y;
  xb[3]=min_x; yb[3]=max_y;
  xc = 0.10*(max_x-min_x); yc = 0.50*(max_y-min_y);
  if (blank == 1) {
    printf(" blanking the plot window \n");
    cpgsci(2);
    cpgsfs(1);
    cpgpoly(4, xb, yb);
    cpgsci(1);
    cpgsch(10.0);
    cpgtext(xc, yc, message);
    cpgsch(1.2);
  }

  cpgsci(1);
  cpgend();
  //printf(" Plotting completed \n");

  return 0;
}

