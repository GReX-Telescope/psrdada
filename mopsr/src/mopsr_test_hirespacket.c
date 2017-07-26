/***************************************************************************
 *  
 *    Copyright (C) 2016 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include "dada_def.h"
#include "multilog.h"
#include "mopsr_def.h"
#include "mopsr_udp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <math.h>
#include <byteswap.h>

#include <sys/types.h>
#include <sys/stat.h>

void usage()
{
  fprintf (stdout,
           "mopsr_test_hires_packet\n"
           " -v         verbose mode\n");
}


int main (int argc, char **argv)
{
  mopsr_hdr_t hdr;
  char buffer[64];

  hdr.seq_no = 1;
  hdr.start_chan = 204;
  hdr.nbit = 8;
  hdr.nant = 8;
  hdr.nchan = 320;
  hdr.nframe = 1;

  fprintf (stderr, "start_chan=%u nbit=%u nant=%u nchan=%u nframe=%u\n",
                   hdr.start_chan, hdr.nbit, hdr.nant, hdr.nchan, hdr.nframe);
  mopsr_encode (buffer, &hdr);
  mopsr_decode (buffer, &hdr);

  fprintf (stderr, "start_chan=%u nbit=%u nant=%u nchan=%u nframe=%u\n",
                   hdr.start_chan, hdr.nbit, hdr.nant, hdr.nchan, hdr.nframe);
  return 0;
}
