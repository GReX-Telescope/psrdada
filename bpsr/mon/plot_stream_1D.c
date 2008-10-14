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
/***************************************************************************/

#include "plot4mon.h"

int plot_stream_1D(float x[], float y[], float y1[], long nvalues, char inpdev[], 
		char xlabel[], char ylabel[], char plottitle[], int dolabel,
		int dobox, unsigned width_pixels, unsigned height_pixels)
{
  long kk=0;
  float max_x, min_x, max_y, min_y, max_y1, min_y1; 
  float marg_x, marg_y;
  char xopt[10],yopt[10];
  float xtick=0.0, ytick=0.0;
  int nxsub=0, nysub=0;
 
  //fprintf (stderr, " Plot: Pixel dimensions: %d x %d \n",width_pixels,height_pixels);

  /*
   *  Compute the minimum and maximum value of the input array(s)
   */
  compute_extremes(x,nvalues,&max_x,&min_x);
  compute_extremes(y,nvalues,&max_y,&min_y);
  compute_extremes(y1,nvalues,&max_y1,&min_y1);

  // fix y range good for both y and y1
  if (max_y1 > max_y) max_y = max_y1;
  if (min_y1 < min_y) min_y = min_y1;

  /*
   *  Compute reasonable margins for plotting purposes
   */
  compute_margin(nvalues,max_x,min_x,&marg_x);
  compute_margin(nvalues,max_y,min_y,&marg_y);
  /*
   * Reporting number of values and extremes of the plot
   */
  printf("\n About to plot %ld values using pgplot \n", nvalues);
  printf(" Extremes: Xmin = %f     Xmax = %f \n",min_x,max_x);
  printf("           Ymin = %f     Ymax = %f \n",min_y,max_y);
  printf(" Margins : Xmargin = %f \n",marg_x); 
  printf(" Margins : Ymargin = %f \n",marg_y); 

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
  cpgline((int)nvalues-1, x, y);

  // plot pol1
  cpgsci(10);
  cpgline((int)nvalues-1, x, y1);

  cpgsci(1);
  cpgend();
  //printf(" Plotting completed \n");

  return 0;
}

