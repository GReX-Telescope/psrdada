/***************************************************************************
 *  
 *    Copyright (C) 2015 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/
   
#define DSB 1

#include "dada_def.h"
#include "mopsr_def.h"

#include "xgpu.h"
#include "xgpu_info.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>
#include <complex.h>
#include <float.h>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>

#include <cufft.h>

void usage();
int xgpu_convert_dump (const char * filename, char verbose);

void usage()
{
  fprintf (stdout,
           "mopsr_xgpu_convert [options] files\n"
           " -c core   bind process to CPU core\n"
           " -h        display usage\n"
           " -v        verbose mode\n",
           DADA_DEFAULT_BLOCK_KEY
           );
}

int xgpu_convert_dump (const char * filename, char verbose)
{
  if (verbose)
    fprintf (stderr, "dump_spectra()\n");

  int acfile, ccfile;
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IRGRP;

  char acfilename[512];
  char ccfilename[512];
  strcpy (acfilename, filename);
  strcpy (ccfilename, filename);

  char * str_ptr;
  str_ptr = strstr (acfilename, ".xc");
  sprintf (str_ptr, ".ac");
  str_ptr = strstr (ccfilename, ".xc");
  sprintf (str_ptr, ".cc");
  
  char tmp[512];
  sprintf (tmp, "%s.tmp", acfilename);
  if (verbose)
    fprintf (stderr, "dump_spectra: opening %s\n", tmp);
  acfile = open (tmp, flags, perms);
  if (!acfile)
  {
    fprintf (stderr, "ERROR: could not open file: %s\n", tmp);
    return -1;
  }
  if (verbose)
    fprintf (stderr, "Opened file %s\n",tmp);

  sprintf (tmp, "%s.tmp", ccfilename);
  if (verbose)
    fprintf (stderr, "dump_spectra: opening %s\n", tmp);
  ccfile = open (tmp, flags, perms);
  if (!ccfile)
  {
    fprintf (stderr, "ERROR: could not open file: %s\n", tmp);
    return -1;
  }
  if (verbose)
    fprintf (stderr, "Opened file %s\n",tmp);

  // here is where the trickery begins to understand the xGPU internal format!
  // xGPU will use it's internal REGISTER_TILE_TRIANGULAR_ORDERordering, so this
  // should be converted to the AC and CC formats we are used to

  if (verbose)
    fprintf (stderr, "dump_spectra: reordering REGISTER_TILE -> TRIANGLE\n");
  int f, i, rx, j, ry, pol1, pol2;
  size_t matLength = NFREQUENCY * ((NSTATION/2+1)*(NSTATION/4)*NPOL*NPOL*4) * (NPULSAR + 1);
  size_t triLength = NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1);

  size_t matBytes = matLength * sizeof(Complex);
  size_t triBytes = triLength * sizeof(Complex);
  if (verbose)
    fprintf (stderr, "dump_spectra: matBytes=%ld tri_bytes=%ld\n", matBytes, triBytes);
  float * input = (float *) malloc (matBytes);
  Complex * buf = (Complex *) malloc (triBytes);

  size_t nant = NSTATION * NPOL;
  size_t ac_data_size = nant* sizeof(float);
  float * ac_data = (float *) malloc (ac_data_size);

  size_t cc_data_size = ((nant * (nant-1)) / 2) * sizeof(Complex);
  Complex * cc_data = (Complex *) malloc (cc_data_size);

  flags = O_RDONLY;
  perms = S_IRUSR | S_IRGRP;
  int fd = open (filename , flags, perms);
  ssize_t bytes_read = read (fd, input, matBytes);
  if (bytes_read < matLength * sizeof(Complex))
  {
    fprintf (stderr, "dump_spectra: failed to read %lu bytes\n", matLength * sizeof(Complex));
    return -1;
  }
  close (fd);

  for (i=0; i<NSTATION/2; i++) 
  {
    for (rx=0; rx<2; rx++) 
    {
      for (j=0; j<=i; j++) 
      {
        for (ry=0; ry<2; ry++) 
        {
          int k = f*(NSTATION+1)*(NSTATION/2) + (2*i+rx)*(2*i+rx+1)/2 + 2*j+ry;
          int l = f*4*(NSTATION/2+1)*(NSTATION/4) + (2*ry+rx)*(NSTATION/2+1)*(NSTATION/4) + i*(i+1)/2 + j;
          for (pol1=0; pol1<NPOL; pol1++)
          {
            for (pol2=0; pol2<NPOL; pol2++)
            {
              size_t tri_index = (k*NPOL+pol1)*NPOL+pol2;
              size_t reg_index = (l*NPOL+pol1)*NPOL+pol2;
              //if (reg_index + matLength > matBytes/sizeof(float))
              //  fprintf (stderr, "bad reg_index=%ld\n", reg_index);
              buf[tri_index].real = input[reg_index];
              buf[tri_index].imag = input[reg_index+matLength];
            }
          }
        }
      }
    }
  }

  if (verbose)
    fprintf (stderr, "dump_spectra: reordering TRIANGLE -> CUSTOM\n");
  // now we have auto and cross correlations in ctx->tmp in xGPU TRIANGULAR ORDER
  for (i=0; i<NSTATION; i++)
  {
    for (j=0; j<=i; j++)
    {
      for (pol1=0; pol1<NPOL; pol1++)
      {
        for (pol2=0; pol2<NPOL; pol2++)
        {
          for(f=0; f<NFREQUENCY; f++)
          {
            int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
            int index = (k*NPOL+pol1)*NPOL+pol2;

            // convert i + pol1 and j + pol2 to our single pol antenna number
            int ant1 = (i * 2) + pol1;
            int ant2 = (j * 2) + pol2;

            //fprintf (stderr, "%d\t%d\t", ant1, ant2);

            // if auto-correlation
            if (ant1 == ant2)
            {
              ac_data[ant1] = buf[index].real;
              //fprintf (stderr, "%d AC", ant1);
            }
            else
            {
              int baseline = 0;
              int ib, jb, bb;
              bb = -1;
              for (ib=0; ib<nant && bb == -1; ib++)
              {
                for (jb=ib+1; jb<nant && bb == -1; jb++)
                {
                 // fprintf (stderr, "(%d == %d && %d == %d ) || (%d == %d && %d == %d)\n",
                 //          ib, ant1, jb, ant2, ib, ant2, jb, ant1);
                  if (ib == ant1 && jb == ant2) 
                  {
                    bb = baseline;
                    //fprintf (stderr, "%d BL", bb);
                    cc_data[bb] = buf[index];
                  }

                  if (ib == ant2 && jb == ant1)
                  {
                    bb = baseline;
                    //fprintf (stderr, "%d BL", bb);
                    cc_data[bb] = buf[index];
                    //cc_data[bb].imag *= -1;
                  }

                  baseline++;
                }
              }
            }
            //fprintf (stderr, "\n");
          }
        }
      }
    }
  }

  if (verbose)
    fprintf (stderr, "dump_spectra: writing AC\n");
  write (acfile, ac_data, ac_data_size);

  close(acfile);
  if (verbose)
    fprintf (stderr, "Closed file %s\n",acfilename);
  
  if (verbose)
    fprintf (stderr, "dump_spectra: writing CC\n");
  ssize_t wrote = write (ccfile, (void *) cc_data, cc_data_size);
  if (wrote != cc_data_size)
    fprintf (stderr, "ER: dump_spectra: only wrote %lld of %ld bytes\n", wrote, cc_data_size);
  close(ccfile);
  if (verbose)
    fprintf (stderr, "Closed file %s\n",ccfilename);

  // rename files from temp names to real names
  sprintf (tmp, "%s.tmp", acfilename);
  rename (tmp, acfilename);

  sprintf (tmp, "%s.tmp", ccfilename);
  rename (tmp, ccfilename);
  
  free (buf);
  free (input);
  free (ac_data);
  free (cc_data);

  return 0;
}

int main (int argc, char **argv)
{
  // Flag set in verbose mode
  char verbose = 0;

  // core to run on
  int core = -1;

  int arg = 0;

  while ((arg=getopt(argc,argv,"c:hv")) != -1)
  {
    switch (arg) 
    {
    case 'c':
      core = atoi(optarg);
      break;
      
    case 'h':
      usage();
      return (EXIT_SUCCESS);
      
    case 'v':
      verbose++;
      break;

    default:
      usage ();
      return 0;
      
    }
  }
  
  int num_args = argc-optind;
  unsigned i = 0;
   
  if (num_args < 1)
  {
    fprintf(stderr, "mopsr_xgpu_dumpconv: at least 1 argument expected\n");
    usage();
    exit(EXIT_FAILURE);
  } 
  
  for (i=0; i<argc-optind; i++)
  {
    if (xgpu_convert_dump (argv[optind+i], verbose) < 0)
      fprintf (stderr, "mopsr_xgpu_dumpconv: failed to process file %s\n",
                argv[optind+i]);
  }

  return EXIT_SUCCESS;
}
