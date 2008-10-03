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
  puts("usage: plot4mon [filename] {-h, -g, ...}");
  puts("filename    -  the binary file containing the data");
  puts("-h          -  print this help page");
  puts("-log        -  plot bandpass on log scale ");
  printf("-g <device> -  graphics device e.g. /vps      (def=%s) \n",
        STD_DEVICE);
}
