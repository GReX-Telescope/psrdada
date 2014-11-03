/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>

#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define COLOUR_PLOT 1

void usage ();

void usage()
{
  fprintf (stdout,
     "mopsr_pmplot [options] log\n"
     " -c nchan           number of channels in log file [default 8]\n"
     " -D device          pgplot device name [default create PNG files]\n"
     " -g <xres>x<yres>   plot with resolution x pixels by y pixels\n"
     " -m module          module name that is being plotted\n"
     " -p                 plot with no axes or titles\n"
     " -v         be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = 0;

  char * module = 0;

  unsigned ncols = 8;

  // default image resoltion
  unsigned xres = 800;
  unsigned yres = 600;
 
  // Plotting options
  mopsr_util_t opts;

  // plotting defaults
  opts.lock_flag  = -1;
  opts.plot_log   = 0;
  opts.zap        = 0;
  opts.ant        = -1;
  opts.chans[0]   = 0;
  opts.chans[1]   = 7;
  opts.nbin       = 0;
  opts.ndim       = 1;
  opts.ant_code   = 0;
  opts.ant_id     = 0;
  opts.plot_plain = 0;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:D:g:m:pv")) != -1)
  {
    switch (arg)
    {
      case 'c':
        ncols = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'g':
        if (sscanf (optarg, "%ux%u", &xres, &yres) != 2)
        {
          fprintf (stderr, "ERROR: could not parse width and height "
                   "from %s\n", optarg);
          usage();
          return (EXIT_FAILURE);
        }
        break;

      case 'm':
        module = strdup(optarg);
        break;

      case 'p':
        opts.plot_plain = 1;
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char filename[256];
  char png_file[256];
  char buffer[1024];
  char label[64];
  strcpy(filename, argv[optind]);

  FILE * fptr = fopen(filename, "r");
  if (!fptr)
  {
    fprintf (stderr, "ERROR: failed to open %s for reading\n", filename);
    exit(EXIT_FAILURE);
  }

  // read the header row
  if (!fgets(buffer, 1024, fptr))
  {
    fprintf (stderr, "ERROR: failed to read header from %s\n", filename);
    fclose(fptr);
    exit(EXIT_FAILURE);
  }

  const char *sep = ", \n";
  char * saveptr;
  char * str;
  str = strtok_r(buffer, sep, &saveptr);
  unsigned icol;
  float cols[ncols];
  for (icol=0; icol<ncols; icol++)
  {
    str = strtok_r (NULL, sep, &saveptr);
    if (sscanf(str, "%f", &(cols[icol])) != 1)
    {
      fprintf (stderr, "ERROR: failed to read col %d\n", icol);
      fclose(fptr);
      exit(EXIT_FAILURE);
    }
  }

  // start with a big maximum
  unsigned nrows_malloc = 1024;
  unsigned irow = 0;
  float ** data = (float **) malloc (ncols * sizeof(float *));
  float * xvals = (float *) malloc (nrows_malloc * sizeof(float));
  for (icol=0; icol<ncols; icol++)
    data[icol] = (float *) malloc (nrows_malloc * sizeof(float));

  while (fgets(buffer, 1024, fptr))
  {
    // read the time offset [ignore for now]
    str = strtok_r(buffer, sep, &saveptr);
    if (sscanf(str, "%f", &(xvals[irow])) != 1)
    {
      fprintf (stderr, "ERROR: failed to time from irow=%d\n", irow);
      fclose (fptr);
      exit(EXIT_FAILURE);
    }

    for (icol=0; icol<ncols && str != NULL; icol++)
    {
      str = strtok_r (NULL, sep, &saveptr);
      if (str != NULL)
      {
        if (sscanf(str, "%f", &(data[icol][irow])) != 1)
        {
          fprintf (stderr, "ERROR: failed to read irow=%d, icol=%d\n", irow, icol);
          fclose(fptr);
          exit(EXIT_FAILURE);
        }
      }
    }

    irow++;
    if (irow >= nrows_malloc)
    {
      nrows_malloc += 1024;
      for (icol=0; icol<ncols; icol++)
        data[icol] = (float *) realloc (data[icol], nrows_malloc * sizeof(float));
      xvals = (float *) realloc (xvals, nrows_malloc * sizeof(float));
    }
  }

  fclose(fptr);

  unsigned nrows = irow;

  sprintf (png_file, "xs");

  if (verbose)
    fprintf (stderr, "creating %s\n", png_file);
  if (!device)
  {
    // generate a timestamp from localtime
    char local_time[32];
    time_t now = time(0);
    strftime (local_time, 32, DADA_TIMESTR, localtime(&now));
    if (module)
      sprintf (png_file, "%s.%s.pm.%dx%d.png/png", local_time, module, xres, yres);
    else
      sprintf (png_file, "%s.XX.pm.%dx%d.png/png", local_time, xres, yres);
  }
  else
  {
    sprintf (png_file, "%s", device);
  }

  if (cpgbeg (0, png_file, 1, 1) != 1)
    fprintf (stderr, "mopsr_dumpplot: error opening plot device [%s]\n", png_file);
  else
  {
    cpgbbuf();
    set_resolution (xres, yres);

    if (!opts.plot_plain)
    {
      if (module)
        sprintf (label, "Total Power Monitor Module %s", module);
      else
        sprintf (label, "Total Power Monitor");
#ifdef COLOUR_PLOT
      cpglab("Time [secs]", "Total Power", label);
#else
      cpglab("Time [secs]", "Freq Subband", label);
#endif
    }

    float xmin = xvals[0];
    float xmax = xvals[nrows-1];
    float ymin, ymax;
    float vp_min, vp_max;

    if (opts.plot_plain)
    {
      vp_min = 0.0;
      vp_max = 1.0;
    } 
    else
    {
      vp_min = 0.1;
      vp_max = 0.9;
    }
    float vp_range = vp_max - vp_min;
    float ysub = vp_range / (float) ncols;
    float y1 = vp_min;
    float y2 = y1 + ysub;

    ymin = 1e10;
    ymax = -1e10;

    for (icol=0; icol<ncols; icol++)
    {
      // determine min/max
      for (irow=0; irow<nrows; irow++)
      {
        if (data[icol][irow] < ymin) ymin = data[icol][irow];
        if (data[icol][irow] > ymax) ymax = data[icol][irow];
      }
    }

    float yrange = ymax - ymin;
    ymax += yrange * 0.1;
    ymin -= yrange * 0.1;

 
#ifdef COLOUR_PLOT
    cpgsvp(vp_min, vp_max, vp_min, vp_max);
    cpgswin(xmin, xmax, ymin, ymax);

    if (opts.plot_plain)
    {
      //cpgbox("BC", 0.0, 0.0, "", 0.0, 0.0);
    }
    else
    {
      cpgbox("BCNST", 0.0, 0.0, "BCNTV", 0.0, 0.0);
    }

    for (icol=0; icol<ncols; icol++)
    {
      float col_ratio = (float) icol / (float) ncols;
      //float grey = 0.5 + 0.5 * col_ratio;
      //cpgscr(icol+1, grey, grey, grey);
      cpgsci(icol+1);
      if (!opts.plot_plain)
      {
        sprintf(label, "%4.0f MHz", cols[icol]);
        cpgmtxt("RV", 0.1, 0.95 - col_ratio, 0.0, label);
      }

      cpgline(nrows, xvals, data[icol]);
    }
#else
    for (icol=0; icol<ncols; icol++)
    {
      cpgsvp(vp_min, vp_max, y1, y2);
      cpgswin(xmin, xmax, ymin, ymax);
      if (opts.plot_plain)
      {
        //cpgbox("BC", 0.0, 0.0, "", 0.0, 0.0);
      }
      else
      {
        if (icol == 0)
          cpgbox("BCNST", 0.0, 0.0, "BCNTV", 0.0, 0.0);
        else
          cpgbox("BCST", 0.0, 0.0, "BCNTV", 0.0, 0.0);
      }

      if (!opts.plot_plain)
      {
        sprintf(label, "%4.0f MHz", cols[icol]);
        cpgmtxt("RV", 0.1, 0.5, 0.0, label);
      }

      cpgline(nrows, xvals, data[icol]);
      y1 += ysub;
      y2 += ysub;
    }
#endif
    
    cpgebuf();
    cpgclos();
  }

  return EXIT_SUCCESS;
}
