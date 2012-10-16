
#include "bpsr_def.h"
#include "bpsr_udp.h"
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
     "bpsr_dadadiskplot [options] files \n"
     " files       pol0 and pol1 files\n"
     " -D device   pgplot device name [default ?]\n"
     " -p          plain image only, no axes, labels etc\n"
     " -v          be verbose\n"
     " -g XXXxYYY  set plot resoltion in XXX and YYY pixels\n"
     "\n"
     "Reads a DADA format BPSR file and plots spectra\n");
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

  /* unsigned int array for reading data files */
  unsigned char * raw;

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
    fprintf(stderr, "bpsr_dadadiskplot: no data file specified\n");
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

  unsigned header_size = 4096;
  char * header = malloc(sizeof(char) * header_size);
  read(fd, header, header_size);
  if (verbose)
    fprintf(stderr, "%s", header);

  uint64_t file_size = 0;
  if (ascii_header_get(header, "FILE_SIZE", "%"PRIu64, &file_size) < 0)
  {
    fprintf(stderr, "failed to read FILE_SIZE from header\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  unsigned npol = 0;
  if (ascii_header_get(header, "NPOL", "%d", &npol) < 0)
  {
    fprintf(stderr, "failed to read NPOL from header\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  // in us
  float tsamp = 0;
  if (ascii_header_get(header, "TSAMP", "%f", &tsamp) < 0)
  {
    fprintf(stderr, "failed to read NPOL from header\n");
    close(fd);
    return (EXIT_FAILURE);
  }

  unsigned nchan = BPSR_IBOB_NCHANNELS;
  unsigned stride = nchan * npol;
  unsigned nspectra = (1000000 / tsamp);

  nspectra = 1000;

  raw = (unsigned char *) malloc (sizeof(unsigned char) * stride);
  data = (float *) malloc(sizeof(float) * stride * nspectra);
  size_t bytes_to_read = (size_t) (sizeof(unsigned char) * stride);

  if (verbose)
    fprintf(stderr, "opening pgplot device\n");
  // Open pgplot device window
  if (cpgopen(device) != 1) 
  {
    fprintf(stderr, "bpsr_dadadiskplot: error opening plot device\n");
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

  //float x_points[1024];
  //for (i=0; i < BPSR_IBOB_NCHANNELS; i++)
  //  x_points[i] = (float) i;

  float ymin = 0;
  float ymax = 1;

  uint64_t bytes_read = 0;

  float * d;

  unsigned ispectra = 0;

  while (bytes_read < file_size)
  {
    d = data;
    ispectra = 0;
    // read 1 second worth of data
    while (bytes_read < file_size && ispectra < nspectra)
    {
      read(fd, raw, bytes_to_read); 

      for (j=0; j<BPSR_IBOB_NCHANNELS; j+=2)
      {
        if (npol == 2)
        {
          // p0
          d[j+0]  = (float) raw[(2*j)+0];
          d[j+1]  = (float) raw[(2*j)+1];

          // p1
          d[nchan+j+0]  = (float) raw[(2*j)+2];
          d[nchan+j+1]  = (float) raw[(2*j)+3];
        }
        else
        {
          d[j+0]  = (float) raw[j+0];
          d[j+1]  = (float) raw[j+1];
        }
      }
/*
      if ((npol == 1) && (ispectra % 10 == 0))
        for (j=0; j<BPSR_IBOB_NCHANNELS; j++)
          d[j] = 0;
*/
      d += stride;
      bytes_read += bytes_to_read;
      ispectra ++;
    }

    if (verbose)
      fprintf(stderr, "read data %d of %"PRIu64"\n", bytes_read, file_size);

    cpgask(1);

    cpgbbuf();

    cpgsci(1);
    if (plainplot)
      cpgenv(0, nchan, 0, nspectra, 0, -2);
    else {
      cpgenv(0, nchan, 0, nspectra, 0, 0);
      cpglab("Channel", "Time [spectra]", "");
    }
                                                                                                         
    cpgsci(1);

    float x_min = 0;
    float x_max = nspectra;

    float y_min = 0;
    float y_max = nchan;

    float nbin = nspectra;
    float nrow = nchan;

    float x_res = (x_max-x_min)/nbin;
    float y_res = (y_max-y_min)/nrow;

    float xoff = 0;
      
    float trf[6] = { xoff + x_min - 0.5*x_res, x_res, 0.0,
                     y_min - 0.5*y_res,        0.0, y_res };

    //cpgimag(&data[0], nspectra, nchan, 1, nspectra, 1, nchan, 0, 255, trf);
    cpgimag(data, nchan, nspectra, 1, nchan, 1, nspectra, 0, 255, trf);

    cpgebuf();

    //cpgeras();
    ibob_pause(150);
  }

  cpgclos();

  free(raw);
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

