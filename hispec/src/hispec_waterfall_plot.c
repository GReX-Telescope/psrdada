#include "hispec_def.h"
#include "ascii_header.h"

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
#include <inttypes.h>

#include <cpgplot.h>

void get_scale (int from, int to, float* width, float * height);

void usage()
{
  fprintf (stdout,
     "hispec_waterfall_plot [options] file \n"
     " file        file\n"
     " -D device   pgplot device name [default ?]\n"
     " -p          plain image only, no axes, labels etc\n"
     " -l          plot logarithm\n"
     " -n chan     number of channels in file [default 512]\n"
     " -v          be verbose\n"
     " -g XXXxYYY  set plot resoltion in XXX and YYY pixels\n"
     "\n"
     "Reads a corr_poly output file and plots channel\n");
}


int main (int argc, char **argv)
{

  /* Flag set in verbose mode */
  char verbose = 0;

  /* PGPLOT device name */
  char * device = "?";

  /* file names of each data file */
  char * fname;

  /* float array for each data file */
  float * data;

  /* file descriptors for each data file */
  int fd;

  /* plot JUST the data */
  int plainplot = 0;

  /* number of channels in input file */
  int nchan = 512;

  /* plot log (z) */
  int log = 0;

  int zap_dc = 0;

  /* dimensions of plot */
  unsigned int width_pixels = 0;
  unsigned int height_pixels = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"D:ln:vg:pz")) != -1)
    switch (arg) {
      
    case 'D':
      device = strdup(optarg);
      break;

    case 'v':
      verbose=1;
      break;

    case 'g':
      if (sscanf (optarg, "%ux%u", &width_pixels, &height_pixels) != 2) {
        fprintf(stderr, "hispec_waterfall_plot: could not parse dimensions from %s\n",optarg);
        return -1;
      }
      break;

    case 'l':
      log = 1;
      break;

    case 'n':
      nchan = atoi (optarg);
      break;

    case 'p':
      plainplot = 1;
      break;

    case 'z':
      zap_dc = 1;
      break;

    default:
      usage ();
      return 0;
      
  }

  if ((argc - optind) != 1) {
    fprintf(stderr, "hispec_waterfall_plot: no data file specified\n");
    usage();
    exit(EXIT_FAILURE);
  }

  int i=0;
  int j=0;
  int flags = O_RDONLY;

  fname = strdup(argv[optind]);
  fd = open(fname, flags);
  if (fd < 0) 
  {
    fprintf(stderr, "Error opening %s: %s\n", fname, strerror (errno));
    return (EXIT_FAILURE);
  }

  if (verbose)
    fprintf(stderr, "opened file\n");

 
  // default 
  int nrows = 10; 
  data = (float *) malloc (sizeof(float) * nchan * nrows);
  size_t bytes_to_read = (size_t) (sizeof(float ) * nchan);
  size_t bytes_read = 0;
  int more_data = 1;  
  int irow = 0;

  while (more_data)
  {
    bytes_read = read (fd, &data[(nchan * irow)], bytes_to_read);
    if (!bytes_read || bytes_read < bytes_to_read)
    {
      more_data = 0;
    }
    else
    {
      irow ++;
      if (irow == nrows)
      {
        nrows += 10;  
        data = realloc (data,  (sizeof(float) * nchan * nrows));
      }
    }
  }

  close (fd);

  if (verbose)
    fprintf(stderr, "opening pgplot device\n");
  // Open pgplot device window
  if (cpgopen(device) != 1) 
  {
    fprintf(stderr, "hispec_waterfall_plot: error opening plot device\n");
    exit(EXIT_FAILURE);
  }

  if (verbose)
    fprintf(stderr, "setting pgplot resolution\n");
  // Resolution
  if (width_pixels && height_pixels) 
  {
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

  float * d;

  cpgbbuf();

  cpgsci(1);
  if (plainplot)
    cpgenv(0, nchan, 0, irow, 0, -2);
  else
  {
    cpgenv(0, nchan, 0, irow, 0, 0);
    cpglab("Channel", "Integration", "");
  }

  float heat_l[] = {0.0, 0.2, 0.4, 0.6, 1.0};
  float heat_r[] = {0.0, 0.5, 1.0, 1.0, 1.0};
  float heat_g[] = {0.0, 0.0, 0.5, 1.0, 1.0};
  float heat_b[] = {0.0, 0.0, 0.0, 0.3, 1.0};
  float contrast = 1.0;
  float brightness = 0.5;

  cpgctab (heat_l, heat_r, heat_g, heat_b, 5, contrast, brightness);

  cpgsci(1);

  float x_min = 0;
  float x_max = nchan;

  float y_min = 0;
  float y_max = irow;

  float x_res = (x_max-x_min)/nchan;
  float y_res = (y_max-y_min)/irow;

  float xoff = 0;
  float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                   y_min - 0.5*y_res,        0.0, y_res };

  float z_min = data[nchan/2];
  float z_max = data[nchan/2];
  float z_avg = 0;

  if (log && z_min > 0)
    z_min = logf(z_min);
  if (log && z_max > 0)
    z_max = logf(z_max);

  for (i=0; i<nchan * irow; i++)
  {
    if (log && data[i] > 0)
      data[i] = logf(data[i]);
    if (!zap_dc || ((zap_dc) && (i % nchan != 0)))
    {
      if (data[i] > z_max) z_max = data[i];
      if (data[i] < z_min) z_min = data[i];
    }
    else
      data[i] = 0;
  }
  z_avg = (z_max + z_min) / 2;
/*
  fprintf(stderr, "x_min=%f x_max=%f\n", x_min, x_max);
  fprintf(stderr, "y_min=%f y_max=%f\n", y_min, y_max);
  fprintf(stderr, "z_min=%f z_max=%f\n", z_min, z_max);
  fprintf(stderr, "x_res=%f y_res=%f\n", x_res, y_res);
*/
  cpgimag(data, nchan, irow, 1, nchan, 1, irow, z_min, z_max, trf);
  cpgebuf();
  cpgclos();

  free(data);

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

