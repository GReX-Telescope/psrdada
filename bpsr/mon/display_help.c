/***************************************************************************/
/*                                                                         */
/* function display_help                                                   */
/*                                                                         */
/* it displays the help on command line                                    */
/*                                                                         */
/* RB 06 Oct 2008:	revised 					   */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void display_help(char *program) 
{
  puts("");
  printf("PROGRAM \"%s\" is a part of BPSRMON version: %.1f ",
           program, BPSRMON_VERSION);
  printf("It produces various plots to monitor BPSR activity\n");
  puts("");
  puts("Usage: plot4mon [filename0] [filename1] {-h, -g, -t, -l ...}");
  puts("");
  puts("filename0    -  data file [bandpass/time series] for polarisation 0");
  puts("filename1    -  data file [bandpass/time series] for polarisation 1");
  puts("-h           -  print this help page");
  puts("-nolabel     -  omit plot labels and title ");
  puts("-nobox       -  omit plot box ");
  puts("-log         -  plot bandpass on log scale ");
  printf("-g <device>  -  display device for plot e.g. /xs  (def=%s) \n", STD_DEVICE);
  printf("-G <AAAxBBB> -  display resolution for plot e.g. -G 240x200 \n");
  puts("");
}
