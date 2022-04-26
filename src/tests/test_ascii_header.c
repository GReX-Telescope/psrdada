#include "ascii_header.h"
#include "dada_def.h"
#include <stdio.h>
#include <string.h>

int main ()
{
  char * header = (char *) malloc (DADA_DEFAULT_HEADER_SIZE);
  strcpy (header, "VERSION 4.5         # the header version\n"
  "CALFREQ 1.234       # the modulation frequency of the diode\n"
  "DATA_0  0\n"
  "DATA_1  1\n"
  "DATA_10 10\n"
  "FREQ    1400.5      # the radio frequency in MHz\n");

  double frequency = 0.0;
  float version = 0.0;
  double calfreq = 0.0;

  if (ascii_header_get (header, "GARBAGE", "%lf", &frequency) != -1)
  {
    fprintf (stderr, "test_ascii_header: failed to notice garbage\n");
    return -1;
  }

  if (ascii_header_get (header, "FREQ", "%lf", &frequency) < 0 || frequency != 1400.5)
  {
    fprintf (stderr, "test_ascii_header: failed to parse FREQ = %lf\n", frequency);
    return -1;
  }

  fprintf (stderr, "=====================================================\n");
  fprintf (stderr, "%s", header);

  if (ascii_header_del (header, "DATA_1") < 0)
  { 
    fprintf (stderr, "test_ascii_header: failed to delete DATA_1\n");
    return -1;
  }

  fprintf (stderr, "=====================================================\n");
  fprintf (stderr, "%s", header);

  return 0;
}

