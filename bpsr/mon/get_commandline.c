/***************************************************************************/
/*                                                                         */
/* function get_commandline                                                */
/*                                                                         */
/* it reads the input values from the command line                         */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void get_commandline(int argc, char *argv[], char *inpfile0, 
		char *inpfile1, char *inpdev, char *outputfile, int *dolog)
{
  int i,j,nfiles;
  int device_selected=0; 
  char checkchar[2];

  /* check number of command-line arguments and print help if requested */
  if (argc<2) {
    display_help(argv[0]);
    exit(-1);
  } else {
    j=2;
    while (j<=argc) {
      if (strings_compare(argv[j-1],"-h")) {
	/* print the help on line */
	display_help(argv[0]);
	exit(-1);
      }
    j++;
    }
  }

  /* work out how many files are on the command line */
  i=1;
  nfiles=0;
  while(check_file_exists(argv[i])) {
        if (nfiles == 0) strcpy(inpfile0,argv[i]);
        if (nfiles == 1) strcpy(inpfile1,argv[i]);
        nfiles++;
        i++;
  }
  printf("Data files: %s %s \n",inpfile0,inpfile1); 
  if (!nfiles) {
  printf("An input file must be the FIRST argument in the command line!"); 
  printf(" Exiting...\n");
  exit(-1);
  }

  /* now parse any remaining command line parameters */
  if (argc>nfiles) {
    i=nfiles+1;
    while (i<argc) {
      if (strings_compare(argv[i],"-g")) {
        /* get graphic device for output */
	if (argc==i+1) {
	   printf("No argument specified for <device>. Exiting... \n");
	   exit(-1);
	}
        strcpy(inpdev,argv[++i]);
        strncpy(checkchar,inpdev,1);
	//if (!(strings_compare(checkchar[0],"/"))) {
	if (checkchar[0] == "/") {
    	 printf("Unknown argument %s given for <device>. Exiting...\n",inpdev);
         //strcpy(inpdev,"/png");
         exit(-1);
        }  
	device_selected=1;
      } else if (strings_compare(argv[i],"-l")) {
        dolog = 1;
        printf(" log band pass set to 1 \n");
      } else if (strings_compare(argv[i],"-h")) {
        /* print the help on line */
	display_help(argv[0]);
      } else {
        /* unknown argument passed down - stop! */
        printf("Unknown argument %s passed to plot4mon.",argv[i]);
        printf(" Exiting... \n");
	exit(-1);
      }
      i++;
    }
  }
  /* selecting the standard output device */
  if (!device_selected) strcpy(inpdev,STD_DEVICE);
}
