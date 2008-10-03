/***************************************************************************/
/*                                                                         */
/* function plot_stream_1D                                                 */
/*                                                                         */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

int plot_stream_1D(float x[], float y[], float y1[], long nvalues,
               char inpdev[], char xlabel[], char ylabel[], char plottitle[])
{
  long kk=0;
  float max_x, min_x, max_y, min_y, max_y1, min_y1; 
  float marg_x, marg_y;
 
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

  //debug lines
  //int i; 
  //for (i=1; i<nvalues; i++) { fprintf(stderr," %f %f %f \n", x[i], y[i], y1[i]); }

  /*
   * Call cpgbeg to initiate PGPLOT and open the output device; 
   */
  if(cpgbeg(0, inpdev, 1, 1) != 1) return -2;
  /*
   * Call cpgenv to specify the range of the axes and to draw a box, and
   * cpglab to label it, with line size given by cpslw (default 1)
   * and text height given by cpsch (default 1.0) 
   */
  cpgenv(min_x, max_x+marg_x, min_y, max_y+marg_y, 0, 0);
  cpgslw(4);
  cpgsch(1.2);
  cpglab(xlabel,ylabel,plottitle);
  /*
   * If needed call cpgpt to mark the points using symbol number -1 (a dot)
   */
      //  cpgpt((int)nvalues, x, y, -1);
  /*
   * Call cpgline to join the points with a line
   */

  cpgslw(2);
  printf(" Plotting pol0 \n");
  cpgsci(2);
  cpgline((int)nvalues-1, x, y);
  printf(" Plotting pol1 \n");
  cpgsci(10);
  cpgline((int)nvalues-1, x, y1);
  cpgsci(1);
  /*
   * Finally, call cpgend to terminate things properly.
   */
  cpgend();
  printf(" Plotting completed \n");
  return 0;
}

