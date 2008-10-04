/***************************************************************************/
/*                                                                         */
/* function display_help                                                   */
/*                                                                         */
/* it displays the help on command line                                    */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void display_help(char *program) 
{
  puts("");
  printf("PROGRAM \"%s\" is a code included in the BPSRMON version: %.1f \n",
           program, BPSRMON_VERSION);
  printf("         it produces various plots for monitoring BPSR activity\n");
  puts("");
  puts("Usage: plot4mon [filename0] [filename1] {-h, -g, ...}");
  puts("");
  puts("filename0   -  data file [bandpass/time series] for polarisation 0");
  puts("filename1   -  data file [bandpass/time series] for polarisation 1");
  puts("-h          -  print this help page");
  puts("-log        -  plot bandpass on log scale ");
  printf("-g <device> -  display device for plot e.g. /xs  (def=%s) \n", STD_DEVICE);
  puts("");
}
