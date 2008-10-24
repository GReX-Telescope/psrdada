#include "bpsr_def.h"
#include "bpsr_udp.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <math.h>

#include <cpgplot.h>

void get_scale (int from, int to, float* width, float * height);

void usage()
{
  fprintf (stdout,
     "bpsr_bramplot [options] file\n"
     " file        bramdump file produced by the mulitbob_server\n"
     " -D device   pgplot device name [default ?]\n"
     " -p          plain image only, no axes, labels etc\n"
     " -v          be verbose\n"
     " -g XXXxYYY  set plot resoltion in XXX and YYY pixels\n"
     " -d          run as daemon\n");
}


int main (int argc, char **argv)
{

  /* Flag set in verbose mode */
  char verbose = 0;

  /* PGPLOT device name */
  char * device = "?";

  /* file names of each data file */
  char ** fnames;

  /* float array for eahc data file */
  float ** data;

  /* file descriptors for each data file */
  int * fds;

  /* plot JUST the data */
  int plainplot = 0;

  /* unsigned int array for reading data files */
  unsigned int * input;

  /* dimensions of plot */
  unsigned int width_pixels = 0;
  unsigned int height_pixels = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"D:vg:p")) != -1)
    switch (arg) {
      
    case 'D':
      device = strdup(optarg);
      break;

    case 'v':
      verbose=1;
      break;

    case 'g':
      if (sscanf (optarg, "%ux%u", &width_pixels, &height_pixels) != 2) {
        fprintf(stderr, "bpsr_diskplot: could not parse dimensions from %s\n",optarg);
        return -1;
      }
      break;

    case 'p':
      plainplot = 1;
      break;

    default:
      usage ();
      return 0;
      
  }

  if ((argc - optind) != 1) {
    fprintf(stderr, "bpsr_diskplot: one data file must be specified\n");
    usage();
    exit(EXIT_FAILURE);
  }

  int i=0;
  int j=0;
  int flags = O_RDONLY;

  int fd;
  char * fname = strdup(argv[optind]);

  fd = open(fname, flags);
  if (fd < 0) {
    fprintf(stderr, "Error opening %s: %s\n", fname, strerror (errno));
    free(fname);
    return (EXIT_FAILURE);
  }

  unsigned bit_window;
  long * input = malloc (sizeof(long) * BPSR_IBOB_NCHANNELS * 2);

  float * xval = malloc (sizeof(float) * BPSR_IBOB_NCHANNELS);
  float * pol0 = malloc (sizeof(float) * BPSR_IBOB_NCHANNELS);
  float * pol1 = malloc (sizeof(float) * BPSR_IBOB_NCHANNELS);

  read(fd, &bit_window, sizeof(unsigned));
  read(fd, input, (sizeof(long) * BPSR_IBOB_NCHANNELS * 2));

  close(fd);

  for (i=0; i<BPSR_IBOB_NCHANNELS; i++)
  {
    xval[i] = (float) i*2;
    pol0[i] = (float) input[i];
    pol1[i] = (float) input[BPSR_IBOB_NCHANNELS+i]
  }

  /* Open pgplot device window */
  if (cpgopen(device) != 1) {
    fprintf(stderr, "bpsr_diskplot: error opening plot device\n");
    exit(EXIT_FAILURE);
  }

  /* Resolution */
  if (width_pixels && height_pixels) {

    float width_scale, height_scale;
                                                                                                 
    get_scale (3, 1, &width_scale, &height_scale);
                                                                                                 
    float width_inches = width_pixels * width_scale;
    float aspect_ratio = height_pixels * height_scale / width_inches;
                                                                                                 
    cpgpap( width_inches, aspect_ratio );
                                                                                                 
    float x1, x2, y1, y2;
    cpgqvsz (1, &x1, &x2, &y1, &y2);
                                                                                                 
  }

  float ymin = 0;
  float ymax = 1;

  /* Get the min/max values */
  for (i=0; i < BPSR_IBOB_NCHANNELS; i++) {
    if (pol0[i] < ymax) ymax = pol0[i];
    if (pol1[i] > ymax) ymax = pol1[i];
  }
                                                                                                         
  cpgbbuf();

  if (plainplot)
    cpgenv(1024, 0, 0, (1.1*ymax), 0, -2);
  else {
    cpgenv(1024, 0, 0, (1.1*ymax), 0, 0);
    cpglab("Frequency Channel", "Intensity", "Intensity vs Frequency Channel");


  }
                                                                                                         
  cpgsci(1);

  cpgsci(2);
  if (!plainplot) 

  cpgline(IBOB_BRAM_CHANNELS, xvals, pol0);

  cpgsci(3);
  cpgline(IBOB_BRAM_CHANNELS, xvals, pol1);

  cpgsci(1);

  float bits_x[2];
  float bits_y[2];

  /* draw the bit windows cutoffs */
  for (i=1; i<=4; i++)
  {
    bits_y[0] = i*8;
    bits_y[1] = i*8;
    if (bitx_y[0] < ibob->bram_max)
    {
      cpgline(2, bits_x, bits_y);
    }
  }

  for (j=0; j < nfiles; j++) {

    cpgsci(2+j);
    if (!plainplot)
      cpgmtxt("T", (((float)j) + 0.5), 0.0, 0.0, fnames[j]);
    cpgline(1024, x_points, data[j]);

  }
                                                                                                         
  cpgebuf();

  cpgclos();

  return EXIT_SUCCESS;
}

void get_scale (int from, int to, float * width, float * height)
{

  float j = 0;
  float fx, fy;
  cpgqvsz (from, &j, &fx, &j, &fy);
                                                                                                 
  float tx, ty;
  cpgqvsz (to, &j, &tx, &j, &ty);
                                                                                                 
  *width = tx / fx;
  *height = ty / fy;
}

