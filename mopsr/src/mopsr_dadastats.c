/*
 * read a dada file from disk and look at statistical properties of the data
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"


void usage ();
int compare (const void * a, const void * b);

void usage()
{
  fprintf (stdout,
     "mopsr_dadastats [options] dadafile ant chan\n"
     " dadafile    PSRDada raw file (MOPSR format)\n"
     " ant         antenna to display\n"
     " chan        channel to display\n"
     " -D device   pgplot device name\n"
     " -h          print usage\n"
     " -t num      number of time samples to read [default 1024]\n"
     " -v          be verbose\n");
}

int compare (const void * a, const void * b)
{
  return ( *(float*)a - *(float*)b );
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  mopsr_util_t opts;

  opts.lock_flag = 1;
  opts.ndim = 2;
  opts.plot_plain = 0;
  opts.zap = 0;

  unsigned nsamp = 1024;

  while ((arg=getopt(argc,argv,"D:ht:v")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'h':
        usage();
        return 0;

      case 't':
        nsamp = atoi (optarg);
        break;

      case 'v':
        verbose++;
        break;

      default:
        usage ();
        return 1;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 3)
  {
    fprintf(stderr, "ERROR: 3 command line arguments are required\n\n");
    usage();
    return (EXIT_FAILURE);
  }

  char filename[256];
  strcpy(filename, argv[optind]);
  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
    return (EXIT_FAILURE);
  }

  if (sscanf(argv[optind+1], "%d", &(opts.ant)) != 1)
  {
    fprintf (stderr, "ERROR: failed parse antenna from %s\n", argv[optind+1]);
    return (EXIT_FAILURE);
  }

  int channel;
  if (sscanf(argv[optind+2], "%d", &channel) != 1)
  {
    fprintf (stderr, "ERROR: failed parse channelfrom %s\n", argv[optind+1]);
    return (EXIT_FAILURE);
  }
  opts.chans[0] = channel;
  opts.chans[1] = channel;

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %ld bytes\n", filename, filesize);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (filename, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t data_size = filesize - 4096;

  char * header = (char *) malloc (4096);
  if (verbose)
    fprintf (stderr, "reading header, 4096 bytes\n");
  size_t bytes_read = read (fd, header, 4096);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  if (ascii_header_get(header, "NCHAN", "%d", &(opts.nchan)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "NANT", "%d", &(opts.nant)) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }

  float tsamp;
  if (ascii_header_get(header, "TSAMP", "%f", &tsamp) != 1)
  {
    fprintf (stderr, "could not extract NANT from header\n");
    return EXIT_FAILURE;
  }
  tsamp /= 1000000;

  const unsigned ndim = 2;
  size_t nsamp_total = data_size / (ndim * opts.nchan * opts.nant);
  if (nsamp_total < nsamp)
  {
    fprintf (stderr, "ERROR: file only contained %ld samples\n", nsamp_total);
    return (EXIT_FAILURE);
  }

  size_t nsets = nsamp_total / nsamp;
  float * sks = (float *) malloc(sizeof(float) * nsets);
  float * s1s = (float *) malloc(sizeof(float) * nsets);
  float * sks2 = (float *) malloc(sizeof(float) * nsets);
  float * xvals = (float *) malloc(sizeof(float) * nsets);
  float * s1_uppers = (float *) malloc(sizeof(float) * nsets);
  float * s1_means = (float *) malloc(sizeof(float) * nsets);
  float * s1_lowers = (float *) malloc(sizeof(float) * nsets);

  size_t block_size = 768;
  size_t block_history = 8;
  float * vals = (float *) malloc(sizeof(float) * block_size * block_history);
  float * diff_block = (float *) malloc(sizeof(float) * block_size * block_history);
  float * history_block = (float *) malloc(sizeof(float) * block_size * block_history);

  unsigned i;
  for (i=0; i<nsets; i++)
  {
    sks[i] = 0;
    s1s[i] = 0;
    sks2[i] = 0;
    xvals[i] = i;
    s1_uppers[i] = 0;
    s1_lowers[i] = 0;
    s1_means[i] = 0;
  }

  if (header)
    free(header);
  header = 0;

  if (verbose)
    fprintf(stderr, "mopsr_dadastats: using device %s\n", device);

  // number
  const size_t bytes_to_read = nsamp * opts.nchan * opts.nant * opts.ndim;

  size_t bytes_read_total = 0;
  void * raw = malloc (bytes_to_read);
  unsigned isamp, isamp_plot, ichan, iant;

  float m = (float) nsamp;
  float m_fac  = (m + 1) / (m - 1);
  float m_fac2 = (m + 2) / (m - 1);

  const unsigned nbin = 256;
  opts.nbin = nbin;
  unsigned * hist = (unsigned *) malloc (sizeof(unsigned) * nbin * 2);

  // 3 sigma limits
  const float sk_low[20]  = { 0, 0, 0, 0, 0,
                              0.387702, 0.492078, 0.601904, 0.698159, 0.775046,
                              0.834186, 0.878879, 0.912209, 0.936770, 0.954684,
                              0.967644, 0.976961, 0.983628, 0.988382, 0.991764 };
  const float sk_high[20] = { 0, 0, 0, 0, 0,
                              2.731480, 2.166000, 1.762970, 1.495970, 1.325420,
                              1.216950, 1.146930, 1.100750, 1.069730, 1.048570,
                              1.033980, 1.023850, 1.016780, 1.011820, 1.008340 };

  unsigned log2_M = (unsigned) log2f (m);

  float sk_l = sk_low[log2_M];
  float sk_h = sk_high[log2_M];

  //sk_l = 2 - 3 * sqrt(24/m);
  //sk_h = 2 + 3 * sqrt(24/m);

  float x_sk[2]; 
  float y_skl[2] = {sk_l, sk_l};
  float y_skh[2] = {sk_h, sk_h};
  float y_s1[2] = {0, 0};

  float xmin = 0;
  float xmax = 0;
  float ymin = sk_l;
  float ymax = sk_h;
  float ymins1 = FLT_MAX;
  float ymaxs1 = -FLT_MAX;
  int symbol = -1;

  fprintf (stderr, "M=%f, log2_M=%u, m_fac=%f m_fac2=%f sk_low=%f sk_high=%f\n", m, log2_M, m_fac, m_fac2, sk_l, sk_h);

  double re_sum = 0;
  double im_sum = 0;
  uint64_t sum_count = 0;
  double s1_sum = 0;
  double s1_mean;

  double s1_sq_sum = 0;
  double s1_count = 0;
  double sigma;
  unsigned j;
  unsigned start_id = 0;
  unsigned batches = 0;
  unsigned valid_blocks;

  double s1_sum_total = 0;
  unsigned s1_total_count = 0;
  double avg_mean = 0;
  double avg_stddev;
  double s1_variance_sum = 0;
  double s1_variance = 0;

  float median;
  unsigned iblock = 0;
  unsigned block_idx;
  float * this_block;
  char plot;

  for (i=0; i<nsets; i++)
  {
    bytes_read = read (fd, raw, bytes_to_read);
    if (bytes_read != bytes_to_read)
    {
      fprintf (stderr, "ERROR: failed to read required number of bytes\n");
      return (EXIT_FAILURE);
    }

    // create a histogram of real and imag in the one array
    mopsr_form_histogram (hist, (char *) raw, bytes_to_read, &opts);

    //unsigned ibin=0;
    //for (ibin=0; ibin<nbin; ibin++)
   // {
    //  fprintf (stderr, "[%d] re=%u im=%u\n", ibin, hist[ibin], hist[nbin+ibin]);
    //}

    int8_t * ptr = (int8_t *) raw;
    double re, im, power, repower;

    double s1 = 0;
    double s2 = 0;

    //double s1s = 0;
    //double s2s = 0;

    //double re_mean = 0;
    //double im_mean = 0;

    for (isamp=0; isamp<nsamp; isamp++)
    {
      for (ichan=0; ichan<opts.nchan; ichan++)
      {
        for (iant=0; iant<opts.nant; iant++)
        {
          if ((iant == opts.ant) && (ichan == opts.chans[0]))
          {
            re = ((double) ptr[0]) + 0.5;
            im = ((double) ptr[1]) + 0.5;

            //re_sum += re;
            //im_sum += im;
            //sum_count ++;

            power = (re * re) + (im * im);

            //repower = (re * re);
            //s1s += repower;
            //s2s += (repower * repower);

            s1 += power;
            //s2 += (power * power);
          }
          ptr += 2;
        }
      }
    }

    //sks[i]  = (float) (m_fac * ((m * (s2 / (s1 * s1))) - 1));
    s1s[i]  = s1;

    s1_sum_total += s1;
    s1_total_count++;

    //sks2[i] = (float) (m_fac2 * ((m * (s2s / (s1s * s1s))) - 1));
    if (i - start_id > (block_size - 1))
    {
      // block idx runs from 0-7
      block_idx = iblock % block_history;
      this_block = vals + (block_idx * block_size);

      if (avg_mean > 0)
      {
        float upper = avg_mean + (5 * avg_stddev);
        for (j=0; j<block_size; j++)
        {
          if (s1s[start_id + j] > upper)
            s1s[start_id + j] = avg_mean;
        }
      }

      // copy from s1s to the vals/history block
      memcpy ((void *) this_block , (void *) (s1s + start_id), (i - start_id) * sizeof(float));

/*
      if (avg_mean > 0)
      {
        float upper = avg_mean + (4 * avg_stddev);
        //float lower = avg_mean - (5 * avg_stddev);
        for (j=0; j<block_size; j++)
        {
          if (this_block[j] > upper)
            this_block[j] = avg_mean;
        }
      }
*/

/*
      // compute the median of the last 256 S1 values
      qsort ((void *) this_block, block_size, sizeof(float), compare);
      median = this_block[block_size/2];
      fprintf (stderr,"median=%f\n",median);

      // compute the difference between the s1s and the median
      for (j=0; j<block_size; j++)
      {
        diff_block[j] = fabsf (this_block[j] - median);
      }
  
      // now compute the median of these
      qsort ((void *) diff_block, block_size, sizeof(float), compare);
      fprintf (stderr,"median of differences=%f [aka sigma!]\n",diff_block[block_size/2]);
      sigma = diff_block[block_size/2];

      for (j=0; j<block_size; j++)
      {
        s1_means[(i-block_size)+j] = median;
        s1_uppers[(i-block_size)+j] = median + (3 * sigma);
        s1_lowers[(i-block_size)+j] = median - (3 * sigma);
      }
*/

      // now compute the median across the last 8 blocks
      valid_blocks = block_history;
      if (iblock < block_history)
        valid_blocks = iblock + 1;

      // fresh copying into history block
      memcpy ((void *) history_block, (void *) vals, block_size * valid_blocks * sizeof(float));

      qsort ((void *) history_block, block_size * valid_blocks, sizeof(float), compare);

      median = history_block[(block_size * valid_blocks)/2];
      fprintf (stderr,"block history median=%f\n",median);

      for (j=0; j<block_size * valid_blocks; j++)
      {
        diff_block[j] = fabsf(history_block[j] - median);  
      }

      qsort ((void *) diff_block, block_size * valid_blocks, sizeof(float), compare);
      sigma = diff_block[(block_size * valid_blocks)/2];
      fprintf (stderr,"median of history differences=%f [aka sigma!]\n", sigma);

      avg_mean = median;
      avg_stddev = sigma;

      for (j=0; j<block_size; j++)
      {
        s1_means[(i-block_size)+j] = median;
        s1_uppers[(i-block_size)+j] = median + (3 * sigma);
        s1_lowers[(i-block_size)+j] = median - (3 * sigma);
      }

      start_id += block_size;

      s1_variance_sum += s1_variance;
      batches++;

      //avg_stddev = sqrt (s1_variance_sum / batches);
      //avg_mean = s1_sum_total / s1_total_count;

      iblock++;
      plot = 1;
    }
    else
      plot = 0;
   
/* 
    // compute mean of s1
    s1_sum = 0;
    for (j=start_id; j<i+1; j++)
    {
      s1_sum += s1s[j];
    }
    s1_mean = s1_sum / ((i-start_id)+1);

    s1_sq_sum = 0;
    for (j=start_id; j<i+1; j++)
      s1_sq_sum += (s1s[j]-s1_mean) * (s1s[j]-s1_mean);

    s1_variance = s1_sq_sum / ((i-start_id)+1);

    sigma = sqrt (s1_variance);
*/
    //s1_means[i]  = (float) s1_mean;
    //s1_uppers[i] = (float) (s1_mean + (3 * sigma));
    //s1_lowers[i] = (float) (s1_mean - (3 * sigma));

    //re_mean = re_sum / sum_count;
    //im_mean = im_sum / sum_count;

    //fprintf (stderr, "means: re=%lf im=%lf sks[%d]=%f sks2[%d]=%f\n", re_mean, im_mean, i, sks[i], i, sks2[i]);
    //fprintf (stderr, "means: s1_mean=%lf s1=%lf sks[%d]=%f sigma=%lf\n", s1_mean, s1, i, sks[i], sigma);

/*
    if (cpgbeg(0, "1/xs", 1, 1) != 1)
    {
      fprintf(stderr, "error opening plot device\n");
      exit(1);
    }

    mopsr_plot_histogram (hist, &opts);

    cpgend();

    if (cpgbeg(0, "2/xs", 0, 0) != 1)
    {
      fprintf(stderr, "error opening plot device\n");
      exit(1);
    }

    if (sks[i] > ymax)
      ymax = sks[i];
    if (sks[i] < ymin)
      ymin = sks[i];
    
    //if (sks2[i] > ymax)
    //  ymax = sks2[i];
    //if (sks2[i] < ymin)
    //  ymin = sks2[i];

    xmax = (float) i+1;

    cpgbbuf();

    float yrange = (ymax - ymin);

    cpgswin(xmin, xmax, ymin - (yrange/10), ymax + (yrange/10));
     
    cpgsvp(0.1, 0.9, 0.1, 0.9);
    cpgbox("BCSNT", 0.0, 0.0, "BCNST", 0.0, 0.0);
    cpglab("Block", "SK", "");

    x_sk[0] = 0;
    x_sk[1] = (float) (i+1);

    cpgline (2, x_sk, y_skl);
    cpgline (2, x_sk, y_skh);
    cpgsci(2);
    cpgslw(10);
    cpgpt (i+1, xvals, sks, symbol);
    cpgslw(1);
    //cpgsci(3);
    //cpgpt (i+1, xvals, sks2, symbol);
    cpgsci(1);

    cpgebuf();
    cpgend();
    */

    if (plot)
    {

    if (cpgbeg(0, "3/xs", 0, 0) != 1)
    {
      fprintf(stderr, "error opening plot device\n");
      exit(1);
    }

    if (s1s[i] > ymaxs1)
      ymaxs1 = s1s[i];
    if (s1s[i] < ymins1)
      ymins1 = s1s[i];
    xmax = (float) i+1;
    if (ymaxs1 > 450000)
      ymaxs1 = 450000;

    cpgbbuf();

    float yrange = (ymaxs1 - ymins1);

    cpgswin(xmin, xmax, ymins1 - (yrange/10), ymaxs1 + (yrange/10));

    cpgsvp(0.1, 0.9, 0.1, 0.9);
    cpgbox("BCSNT", 0.0, 0.0, "BCNST", 0.0, 0.0);
    cpglab("Block", "S1", "");

    x_sk[0] = 0;
    x_sk[1] = (float) (i+1);

    cpgline (2, x_sk, y_skl);
    cpgline (2, x_sk, y_skh);
    cpgsci(3);
    cpgslw(5);
    cpgpt (i+1, xvals, s1s, symbol);
    cpgslw(1);
    cpgsci(1);

    cpgline (i+1, xvals, s1_means);
    cpgsci(2);
    cpgline (i+1, xvals, s1_uppers);
    cpgsci(3);
    cpgline (i+1, xvals, s1_lowers);
    cpgsci(1);

    // now do the averages in blue
/*
    cpgsci(4);
    cpgslw(10);
    y_s1[0] = avg_mean;
    y_s1[1] = avg_mean;
    cpgline (2, x_sk, y_s1);
    cpgslw(1);
    y_s1[0] = avg_mean + 3 * avg_stddev;
    y_s1[1] = avg_mean + 3 * avg_stddev;
    cpgline (2, x_sk, y_s1);
    y_s1[0] = avg_mean - 3 * avg_stddev;
    y_s1[1] = avg_mean - 3 * avg_stddev;
    cpgline (2, x_sk, y_s1);
    cpgslw(1);
    cpgsci(1);
*/
    cpgebuf();
    cpgend();
    }
  }

  close (fd);

  free(xvals);
  free(sks);
  free(raw);
  free(hist);

  return EXIT_SUCCESS;
}

