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
#include <float.h>
//#include <complex.h>
#include <math.h>
#include <cpgplot.h>
#include <fftw3.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

typedef struct {

  // pgplot device
  char * device;

  // identifying code for antenna's in packet
  unsigned int ant_code;

  // number of antennae
  unsigned int nant;

  // number of input channels
  unsigned int nchan_in;

  // number of FFT points to perform on input channels
  unsigned int nfft;

  // number of output channels
  unsigned int nchan_out;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // which antenna to display [-1 for both]
  int antenna;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float *** fft_in;     // input timeseries

  float ** fft_out;     // output spectra (oversampled)

  float ** fft_in2;     // input spectra (critically sampled)

  float ** fft_out2;    // output timeseries (critically sampled)

  float ** fft_in3;     // input timeseries (critically sampled)

  float ** fft_out3;    // output spectra (critically sampled)

  unsigned int fft_count;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

  float base_freq;

  float bw;

  int zap_dc;

  float xmin;

  float xmax;

  fftwf_plan plan;

  char order[4];

  unsigned dsb;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void append_samples (udpplot_t * ctx, void * buffer, uint64_t isamp,  unsigned npt, uint64_t nbytes);
void detect_data (udpplot_t * ctx);
void fft_data (udpplot_t * ctx);
void plot_data (udpplot_t * ctx);

void usage ();


void usage()
{
  fprintf (stdout,
     "mopsr_dadafftplot [options] dadafile\n"
     " -a ant      antenna to display\n"
     " -F min,max  set the min,max x-value (e.g. frequency zoom)\n" 
     " -l          plot logarithmically\n"
     " -n npt      number of points in each coarse channel fft [default 1024]\n"
     " -D device   pgplot device name\n"
     " -t num      number of FFTs to avaerage into each plot\n"
     " -v          be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int nfft = 1024;

  unsigned int plot_log = 0;

  float xmin = 0;
  float xmax = 0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float base_freq;
  unsigned int to_integrate = 8;
  int antenna = -1;
  unsigned zap_dc = 0;


  while ((arg=getopt(argc,argv,"a:D:F:ln:t:vz")) != -1)
  {
    switch (arg)
    {
      case 'a':
        antenna = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'F':
      {
        if (sscanf (optarg, "%f,%f", &xmin, &xmax) != 2)
        {
          fprintf (stderr, "could not parse xrange from %s\n", optarg);
          return (EXIT_FAILURE);
        }
        break;
      }

      case 'l':
        plot_log = 1;
        break; 

      case 'n':
        nfft = atoi(optarg);
        break;

      case 't':
        to_integrate = atoi (optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'z':
        zap_dc  = 1;
        break;

      default:
        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char filename[256];
  char png_file[256];
  strcpy(filename, argv[optind]);

  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", filename, filesize);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (filename, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", filename, strerror(errno));
    exit (EXIT_FAILURE);
  }

  // get the size of the ascii header in this file
  size_t hdr_size = ascii_header_get_size_fd (fd);
  char * header = (char *) malloc (hdr_size + 1);

  if (verbose)
    fprintf (stderr, "reading header, %ld bytes\n", hdr_size);
  size_t bytes_read = read (fd, header, hdr_size);
  if (bytes_read != hdr_size)
  {
    fprintf (stderr, "failed to read %ld bytes of header\n", hdr_size);
    exit (EXIT_FAILURE);
  }

  size_t data_size = filesize - hdr_size;

  //size_t pkt_size = 2560;
  //void * pkt = (void *) malloc (pkt_size);

  void * raw = malloc (data_size);
  bytes_read = read (fd, raw, data_size);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  udpplot_t udpplot;
  udpplot.verbose = verbose;

  if (ascii_header_get(header, "NCHAN", "%d", &(udpplot.nchan_in)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }
  udpplot.nfft = nfft;
  udpplot.nchan_out = udpplot.nchan_in * nfft;

  if (ascii_header_get(header, "NANT", "%d", &(udpplot.nant)) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }

  float cfreq;
  if (ascii_header_get(header, "FREQ", "%f", &cfreq) != 1)
  {
    fprintf (stderr, "could not extract FREQ from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "BW", "%f", &(udpplot.bw)) != 1)
  {     
    fprintf (stderr, "could not extract BW from header\n");
    return EXIT_FAILURE; 
  }

  if (ascii_header_get(header, "ORDER", "%s", udpplot.order) != 1)
  {
    fprintf (stderr, "could not extract ORDER from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "DSB", "%u", &(udpplot.dsb)) != 1)
  {
    fprintf (stderr, "could not extract DSB from header\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "main: ORDER=%s DSB=%u\n", udpplot.order, udpplot.dsb);

  if (header)
    free(header);
  header = 0;

#if 0
  if (verbose > 1)
  {
    fprintf (stderr, "[T][F][S] (re, im)\n");
    int8_t * ptr = (int8_t *) raw;
    unsigned iant, isamp, ichan;
    for (isamp=0; isamp<4; isamp++)
    {
      for (ichan=0; ichan<udpplot.nchan_in; ichan++)
      {
        for (iant=0; iant<udpplot.nant; iant++)
        {
          fprintf (stderr, "[%d][%d][%d] (%d, %d)\n", isamp, ichan, iant, ptr[0], ptr[1]);
          ptr += 2;     
        }
      }
    }
  }
#endif

  udpplot.base_freq = cfreq - (udpplot.bw / 2);
  if ((xmin == 0) && (xmax == 0))
  {
    xmin = udpplot.base_freq;
    xmax = udpplot.base_freq + udpplot.bw;
  }

  udpplot.ant_code = 0;
  udpplot.antenna = antenna;

  udpplot.zap_dc = zap_dc;
  udpplot.num_integrated = 0;
  udpplot.fft_count = 0;
  udpplot.to_integrate = to_integrate;

  udpplot.plot_log = plot_log;
  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  if (verbose)
    fprintf (stderr, "Freq range: %f - %f MHz\n", udpplot.xmin, udpplot.xmax);

  if (verbose)
    fprintf(stderr, "mopsr_dadafftplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_dadafftplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  udpplot_t * ctx = &udpplot;

  // allocate require resources
  if (udpplot_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not alloc memory\n");
    exit(1);
  }

  // cloear packets ready for capture
  udpplot_reset (ctx);

  uint64_t isample = 0;;
  uint64_t nsamples = data_size / (MOPSR_NDIM * ctx->nchan_in * ctx->nant);
  uint64_t nsamples_per_append = ctx->nfft;

  while (isample < nsamples)
  {
    append_samples (ctx, raw, isample, ctx->nfft, data_size);
    isample += ctx->nfft;

    fft_data (ctx);
    detect_data (ctx);
    ctx->num_integrated ++;

    if (ctx->num_integrated >= ctx->to_integrate)
    {
      if (verbose)
        fprintf(stderr, "plotting %d spectra (%d pts) in %d channels\n", 
                ctx->num_integrated, ctx->nfft * ctx->num_integrated, 
                ctx->nchan_out);
      plot_data (ctx);
      udpplot_reset (ctx);

      sleep (1);
    }
  }

  udpplot_destroy (ctx);
  cpgclos();
  close(fd);

  if (raw)
    free (raw);

  return EXIT_SUCCESS;
}

int udpplot_reset (udpplot_t * ctx)
{
  unsigned ichan;
  float mhz_per_out_chan = ctx->bw / (float) ctx->nchan_out;
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    ctx->x_points[ichan] = ctx->base_freq + (((float) ichan) * mhz_per_out_chan);
  }

  unsigned iant;
  unsigned ifft;
  for (iant=0; iant < ctx->nant; iant++)
  {
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
    {
      ctx->y_points[iant][ichan] = 0;
      ctx->fft_out[iant][2*ichan+0] = 0;
      ctx->fft_out[iant][2*ichan+1] = 0;
    }
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      for (ifft=0; ifft < ctx->nfft; ifft++)
      {
        ctx->fft_in[iant][ichan][2*ifft+0] = 0;
        ctx->fft_in[iant][ichan][2*ifft+1] = 0;
      }
    }
  }
  ctx->num_integrated = 0;
  ctx->fft_count = 0;
  return 0;
}

int udpplot_destroy (udpplot_t * ctx)
{

  fftwf_destroy_plan (ctx->plan);
  unsigned int iant;
  unsigned int ichan;
  for (iant=0; iant<ctx->nant; iant++)
  {
    if (ctx->y_points[iant])
      free(ctx->y_points[iant]);
    ctx->y_points[iant] = 0;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      if (ctx->fft_in[iant][ichan])
        free (ctx->fft_in[iant][ichan]);
      ctx->fft_in[iant][ichan] = 0;
    }
    if (ctx->fft_in[iant])
      free (ctx->fft_in[iant]);
    ctx->fft_in[iant] = 0;
    if (ctx->fft_out[iant])
      free (ctx->fft_out[iant]);
    ctx->fft_out[iant] = 0;
  }

  if (ctx->fft_in)
    free (ctx->fft_in);
  ctx->fft_in = 0;

  if (ctx->fft_out)
    free (ctx->fft_out);
  ctx->fft_out = 0;

  if (ctx->y_points)
    free(ctx->y_points);
  ctx->y_points = 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;
  return 0;

}

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    fprintf(stderr, "mopsr_udpdb_init_receiver()\n");

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->nant);
  ctx->fft_in = (float ***) malloc(sizeof(float **) * ctx->nant);
  ctx->fft_out = (float **) malloc(sizeof(float *) * ctx->nant);

  unsigned int iant;
  for (iant=0; iant < ctx->nant; iant++)
  {
    ctx->y_points[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[iant] = (float **) malloc (sizeof(float *) * ctx->nchan_in);
    ctx->fft_out[iant] = (float *) malloc (sizeof(float) * ctx->nchan_out * 2);

    unsigned int ichan;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      ctx->fft_in[iant][ichan] = (float *) malloc (sizeof(float) * ctx->nfft * 2);
    }
  }

  fftwf_complex * input  = (fftwf_complex *) ctx->fft_in[0][0];
  fftwf_complex * output = (fftwf_complex *) ctx->fft_out[0];

  CHECK_ALIGN(input);
  CHECK_ALIGN(output);

  int direction_flags = FFTW_FORWARD;
  int flags = 0;

  ctx->plan = fftwf_plan_dft_1d (ctx->nfft, input, output, direction_flags, flags);

  return 0;
}

// copy data from packet in the fft input buffer
void append_samples (udpplot_t * ctx, void * buffer, uint64_t isamp,  unsigned npt, uint64_t nbytes)
{
  unsigned ichan, iant, ipt;

  if (strcmp(ctx->order, "TFS") == 0)
  {
    size_t offset = isamp * ctx->nchan_in * ctx->nant * MOPSR_NDIM;
    int8_t * in = (int8_t *) buffer + offset;
    for (ipt=0; ipt < npt; ipt++)
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        for (iant=0; iant < ctx->nant; iant++)
        { 
          ctx->fft_in[iant][ichan][(2*ipt) + 0] = (float) in[0];
          ctx->fft_in[iant][ichan][(2*ipt) + 1] = (float) in[1];
          in += 2;
        }
      }
    }
  }
  else if (strcmp(ctx->order, "ST") == 0)
  {
    uint64_t ant_stride = nbytes / ctx->nant;
    for (iant=0; iant < ctx->nant; iant++)
    {
      int8_t * in = (int8_t *) buffer + (iant * ant_stride) + (MOPSR_NDIM * isamp);
      for (ipt=0; ipt < npt; ipt++)
      {
        ctx->fft_in[iant][ichan][(2*ipt) + 0] = (float) in[0];
        ctx->fft_in[iant][ichan][(2*ipt) + 1] = (float) in[1];
        in += 2;
      }
    }
  }
  else
  {
    fprintf (stderr, "append_packet: unsupported input order\n");
  }

}

void fft_data (udpplot_t * ctx)
{
  unsigned int iant, ichan, ipt;
  float * src;
  float * dest;


  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        src = ctx->fft_in[iant][ichan];
        dest = ctx->fft_out[iant] + (ichan * ctx->nfft * 2);
        //if (iant == 0 && ichan== 39)
        {
          //for (ipt=0; ipt<ctx->nfft; ipt++)
          //{
          //  fprintf(stderr, "src[%d] (%f, %f)\n", ipt, src[2*ipt+0], src[2*ipt+1]);
          //}

          fftwf_execute_dft (ctx->plan, (fftwf_complex*) src, (fftwf_complex*) dest);

          //for (ipt=0; ipt<ctx->nfft; ipt++)
          //{
          //  fprintf(stderr, "dest[%d] (%f, %f)\n", ipt, dest[2*ipt+0], dest[2*ipt+1]);
          //}
        }
        /*
        for (ipt=0; ipt<ctx->nchan_out; ipt++)
        {
          ctx->fft_out[iant][2*ipt] = 0;
          ctx->fft_out[iant][2*ipt+1] = sqrt(ipt);
        }
        */
      }
    }
  }



}

void detect_data (udpplot_t * ctx)
{
  unsigned iant = 0;
  unsigned ichan = 0;
  unsigned ibit = 0;
  unsigned halfbit = ctx->nfft / 2;
  unsigned offset = 0;
  unsigned basechan = 0;
  unsigned newchan;
  unsigned shift;
  float a, b;

  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        offset = (ichan * ctx->nfft * 2);
        basechan = ichan * ctx->nfft;

        if (ctx->dsb == 1)
        {
          // first half [flipped - A]
          for (ibit=halfbit; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];
            newchan = (ibit-halfbit);
            ctx->y_points[iant][basechan + newchan] += ((a*a) + (b*b));
          }

          // second half [B]
          for (ibit=0; ibit<halfbit; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];
            newchan = (ibit+halfbit);
            ctx->y_points[iant][basechan + newchan] += ((a*a) + (b*b));
          }
        } else {
          for (ibit=0; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[iant][offset + (ibit*2) + 0];
            b = ctx->fft_out[iant][offset + (ibit*2) + 1];
            ctx->y_points[iant][basechan + ibit] += ((a*a) + (b*b));
          }
        }
        if (ctx->zap_dc && ichan == 0)
          ctx->y_points[iant][ichan] = 0;
      }
    }
  }
}


void plot_data (udpplot_t * ctx)
{
  if (ctx->verbose)
    fprintf(stderr, "plot_packet()\n");

  int ichan = 0;
  unsigned iant = 0;
  unsigned iframe = 0;
  float ymin = ctx->ymin;
  float ymax = ctx->ymax;

  int xchan_min = -1;
  int xchan_max = -1;

  // determined channel ranges for the x limits
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    if ((xchan_min == -1) && (ctx->x_points[ichan] >= ctx->xmin))
      xchan_min = ichan;
  }
  for (ichan=(ctx->nchan_out-1); ichan > 0; ichan--)
  {
    if ((xchan_max == -1) && (ctx->x_points[ichan] <= ctx->xmax))
      xchan_max = ichan;
  }

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    for (iant=0; iant < ctx->nant; iant++)
    {
      if ((ctx->antenna < 0) || (ctx->antenna == iant))
      {
        for (ichan=0; ichan < ctx->nchan_out; ichan++)
        {
          if (ctx->plot_log)
            ctx->y_points[iant][ichan] = (ctx->y_points[iant][ichan] > 0) ? log10(ctx->y_points[iant][ichan]) : 0;
          if ((ichan > xchan_min) && (ichan < xchan_max))
          {
            if (ctx->y_points[iant][ichan] > ymax) ymax = ctx->y_points[iant][ichan];
            if (ctx->y_points[iant][ichan] < ymin) ymin = ctx->y_points[iant][ichan];
          }
        }
      }
    }
  }
  if (ctx->verbose)
  {
    fprintf(stderr, "plot_packet: ctx->xmin=%f, ctx->xmax=%f\n", ctx->xmin, ctx->xmax);
    fprintf(stderr, "plot_packet: ymin=%f, ymax=%f\n", ymin, ymax);
  }

  cpgbbuf();
  cpgsci(1);
  if (ctx->plot_log)
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 20);
    cpglab("Channel", "log\\d10\\u(Power)", "Bandpass");
  }
  else
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 0);
    cpglab("Channel", "Power", "Bandpass");
  }

  float line_x[2];
  float line_y[2];
  float percent_chan;
  float ifreq;

  float oversampling_difference = ((5.0 / 32.0) * (ctx->bw / ctx->nchan_in)) / 2.0;
  cpgsls(2);
  for (ichan=0; ichan < ctx->nchan_in; ichan++)
  {
    line_y[0] = ymin;
    line_y[1] = ymin + (ymax - ymin);

    percent_chan = (float) ichan / (float) ctx->nchan_in;
    percent_chan *= ctx->bw;
    
    ifreq = ctx->base_freq + percent_chan;
    line_x[0] = line_x[1] = ifreq;
    cpgline(2, line_x, line_y);

    line_y[0] = ymin;
    line_y[1] = ymin + (ymax - ymin) / 4;

    line_x[0] = line_x[1] = ifreq - oversampling_difference;
    cpgline(2, line_x, line_y);

    line_x[0] = line_x[1] = ifreq + oversampling_difference;
    cpgline(2, line_x, line_y);
  }
  cpgsls(1);

  char ant_label[10];
  int ant_id = 0;
  for (iant=0; iant < ctx->nant; iant++)
  {
    if ((ctx->antenna < 0) || (ctx->antenna == iant))
    {
      sprintf(ant_label, "Ant %d", iant);
      cpgsci(iant + 2);
      cpgmtxt("T", 0.5 + (0.9 * iant), 0.0, 0.0, ant_label);
      cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[iant]);
    }
  }
  cpgebuf();
}

