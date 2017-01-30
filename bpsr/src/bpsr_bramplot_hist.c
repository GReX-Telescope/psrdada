#include "bpsr_def.h"
#include "bpsr_udp.h"
#include "ibob.h"

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
#include <float.h>

#include <cpgplot.h>
void set_dimensions (unsigned width_pixels, unsigned height_pixels);
void get_scale (int from, int to, float* width, float * height);
void createPlot (char* device, float * xvals, float * f0, float * f1,
                 unsigned width, unsigned height, int plainplot, 
                 float ymax, float mean_0, float mean_1, float variance_0, 
                 float variance_1);

extern float roundf(float);


void usage()
{
  fprintf (stdout,
     "bpsr_bramplot_hist [options] file\n"
     "  file        bramdump file produced by the mulitbob_server\n"
     "  -D device   pgplot device name [default ?]\n"
     "  -p          plain image only, no axes, labels etc\n"
     "  -v          be verbose\n"
     "  -g XXXxYYY  only produce 1 plot in XXXxYYY dimensions\n"
     "  -d          run as daemon\n"
     "\n"
     "  default produces a 112x84, 400x300 and 1024x768 plot\n"
     "  -D, -p or -g will change this behaviour\n");
}


int main (int argc, char **argv)
{

  /* Flag set in verbose mode */
  char verbose = 0;

  /* PGPLOT device name */
  char * device = "?";

  /* plot JUST the data */
  int plainplot = 0;

  /* pgplot float arrays */
  float * xval;
  unsigned * p0;
  unsigned * p1;
  float * f0;
  float * f1;

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
  int flags = O_RDONLY;

  int fd;
  char * fname = strdup(argv[optind]);

  fd = open(fname, flags);
  if (fd < 0) {
    fprintf(stderr, "Error opening %s: %s\n", fname, strerror (errno));
    free(fname);
    return (EXIT_FAILURE);
  }

  const unsigned nbin = 256;

  unsigned bit_window;
  xval = (float *) malloc (sizeof(float) * nbin);
  p0 = (unsigned *) malloc (sizeof(unsigned) * nbin);
  p1 = (unsigned *) malloc (sizeof(unsigned) * nbin);
  f0 = (float *) malloc (sizeof(float) * nbin);
  f1 = (float *) malloc (sizeof(float) * nbin);

  // for some reason P0 and P1 are opposite in the BRAM of the ADCs
  read(fd, p1, (sizeof(unsigned) * nbin));
  read(fd, p0, (sizeof(unsigned) * nbin));

  close(fd);

  float sum_0 = 0;
  float sum_1 = 0;

  /* extract data from input array */
  float ymax = -FLT_MAX;

  float npt_0 = 0;
  float npt_1 = 0;
  for (i=0; i<nbin; i++) 
  {
    xval[i] = -128 + i;
    f0[i] = (float) p0[i];
    sum_0 += f0[i] * i;
    npt_0 += f0[i];
    if (f0[i] > ymax)
      ymax = f0[i];
    f1[i] = (float) p1[i];
    sum_1 += f1[i] * i;
    npt_1 += f1[i];
    if (f1[i] > ymax)
      ymax = f1[i];
  }
  ymax *= 1.05;

  float mean_0 = sum_0 / npt_0;
  float mean_1 = sum_1 / npt_1;

  sum_0 = 0;
  sum_1 = 0;

  for (i=0; i<nbin; i++)
  {
    sum_0 += f0[i] * powf((i-mean_0),2);
    sum_1 += f1[i] * powf((i-mean_1),2);
  }

  float variance_0 = sum_0 / nbin;
  float variance_1 = sum_1 / nbin;

  if ( (plainplot == 0) && (strcmp(device, "?") == 0) && 
       (width_pixels == 0)  && (height_pixels == 0) )

  {
    char png_file[128];
    char file_base[128];
    
    // strip the .bramdump from the end of the filename
    strncpy(file_base, fname, (strlen(fname)-10));
    file_base[(strlen(fname)-4)] = '\0';

    sprintf(png_file, "%s_112x84_hist.png/png", file_base);
    createPlot(png_file, xval, f0, f1, 112, 84, 1, ymax, mean_0, mean_1, variance_0, variance_1);

    sprintf(png_file, "%s_400x300_hist.png/png", file_base);
    createPlot(png_file, xval, f0, f1, 400, 300, 0, ymax, mean_0, mean_1, variance_0, variance_1);

    sprintf(png_file, "%s_1024x768_hist.png/png", file_base);
    createPlot(png_file, xval, f0, f1, 1024, 768, 0, ymax, mean_0, mean_1, variance_0, variance_1);
  }
  else 
  {
    createPlot(device, xval, f0, f1, width_pixels, height_pixels, 
               plainplot, ymax, mean_0, mean_1, variance_0, variance_1);
  }

  free(xval);
  free(f0);
  free(f1);
  free(p0);
  free(p1);

  return EXIT_SUCCESS;
}


void createPlot (char* device, float * xvals, float * f0, float * f1, 
                 unsigned width_pixels, unsigned height_pixels, int plainplot,
                 float ymax, float mean_0, float mean_1, float variance_0, float variance_1)
{
  char label[64];

  /* Open pgplot device window */
  if (cpgopen(device) != 1) {
    fprintf(stderr, "bpsr_diskplot: error opening plot device\n");
    exit(EXIT_FAILURE);
  }

  /* Resolution */
  if (width_pixels && height_pixels) 
    set_dimensions (width_pixels, height_pixels);

  cpgsci(1);

  if (plainplot)
  {
    cpgsvp(0, 1, 0, 1.0);
    cpgswin(-128,128, 0, ymax);
    cpgbox("BC", 0.0, 0.0, "BC", 0.0, 0.0);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  }
  else
  {
    cpgsvp(0.1, 0.9, 0.1, 0.9);
    cpgswin(-128, 128, 0, ymax);
    cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
    cpglab("States", "Count", "ADC Histogram");
  }

  cpgsci (2);
  cpgbin (256, xvals, f0, 0);
  sprintf (label, "Mean: %5.2f Variance: %5.2f", mean_0, variance_0);
  if (!plainplot)
    cpgmtxt ("T", -2.5, 0.0, 0.0, label);
  cpgsci (3);
  cpgbin (256, xvals, f1, 0);
  sprintf (label, "Mean: %5.2f Variance: %5.2f", mean_1, variance_1);
  if (!plainplot)
    cpgmtxt ("T", -3.5, 0.0, 0.0, label);

  cpgclos();

}

void get_scale (int from, int to, float * width, float * height)
{

  float j = 0;
  float fx, fy;
  cpgqvsz (from, &j, &fx, &j, &fy);
                                                                                                 
  float tx, ty;
  cpgqvsz (to, &j, &tx, &j, &ty);

  /* you dont want to know why */
  fx *= 1.003;
  fy *= 1.003;
                                                                                                 
  *width = tx / fx;
  *height = ty / fy;
}


void set_dimensions (unsigned width_pixels, unsigned height_pixels)
{
  float width_scale, height_scale;
  const int Device = 0;
  const int Inches = 1;
  const int Millimetres = 2;
  const int Pixels = 3;
  const int World = 4;
  const int Viewport = 5;

  get_scale (Pixels, Inches, &width_scale, &height_scale);

  float width_inches = width_pixels * width_scale;
  float aspect_ratio = height_pixels * height_scale / width_inches;

  cpgpap( width_inches, aspect_ratio );

  float x1, x2, y1, y2;
  cpgqvsz (Pixels, &x1, &x2, &y1, &y2);

  if (roundf(x2) != width_pixels) {
    //fprintf (stderr,"set_dimensions: %f != request = %d\n",x2, width_pixels);
  }
}
