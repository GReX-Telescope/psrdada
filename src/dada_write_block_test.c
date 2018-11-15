#include "dada_hdu.h"
#include "multilog.h"
#include "ascii_header.h"
#include "futils.h"

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void usage()
{
  fprintf (stdout,
	   "dada_write_block_test [options]\n"
	   " -H filename    ascii header information in file\n"
	   " -g gigabytes   number of gigabytes to write\n");
}

static char* default_header =
"OBS_ID       dada_write_block_test.0001\n"
"OBS_OFFSET   100\n"
"FILE_SIZE    1073741824\n"
"FILE_NUMBER  3\n";

int main (int argc, char **argv)
{
  /* DADA Header plus Data Unit */
  dada_hdu_t* hdu = 0;

  /* DADA Logger */
  multilog_t* log;

  /* number of bytes to write */
  uint64_t bytes_to_write = (uint64_t)(1024 * 1024) * (uint64_t)(534 * 5);

  /* the header to be written to the header block */
  char* header = default_header;

  /* the header sub-block */
  char* header_buf = 0;

  /* the size of the header string */
  unsigned header_strlen = 0;

  /* the size of the header buffer */
  uint64_t header_size = 0;

  /* the filename from which the header will be read */
  char* header_file = 0;


  /* the data to copy */
  char* data = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  float gigabytes = 0.0;
  float one_gigabyte = 1024.0 * 1024.0 * 1024.0;

  int arg = 0;
  while ((arg=getopt(argc,argv,"g:hH:v")) != -1) {
    switch (arg) {

    case 'g':
      gigabytes = atof (optarg);
      bytes_to_write = (uint64_t) (gigabytes * one_gigabyte);
      break;

    case 'H':
      header_file = optarg;
      break;
      
    case 'v':
      verbose=1;
      break;

    case 'h':
    default:
      usage ();
      return 0;
      
    }
  }
  
  log = multilog_open ("dada_write_block_test", 0);
  multilog_add (log, stderr);

  hdu = dada_hdu_create (log);

  if (dada_hdu_connect (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_lock_write (hdu) < 0)
    return EXIT_FAILURE;

  header_strlen = strlen(header);

  header_size = ipcbuf_get_bufsz (hdu->header_block);
  multilog (log, LOG_INFO, "header block size = %"PRIu64"\n", header_size);

  header_buf = ipcbuf_get_next_write (hdu->header_block);

  if (!header_buf) 
  {
    multilog (log, LOG_ERR, "Could not get next header block\n");
    return EXIT_FAILURE;
  }

  if (header_file) 
  {
    multilog (log, LOG_INFO, "reading header file\n");
    if (fileread (header_file, header_buf, header_size) < 0)  {
      multilog (log, LOG_ERR, "Could not read header from %s\n", header_file);
      return EXIT_FAILURE;
    }
  }
  else { 
    header_strlen = strlen(header);
    memcpy (header_buf, header, header_strlen);
    memset (header_buf + header_strlen, '\0', header_size - header_strlen);
  }

  /* Set the header size attribute */ 
  multilog (log, LOG_INFO, "setting HDR_SIZE=%"PRIu64"\n", header_size);
  if (ascii_header_set (header_buf, "HDR_SIZE", "%"PRIu64"", header_size) < 0) {
    multilog (log, LOG_ERR, "Could not write HDR_SIZE to header\n");
    return -1;
  }

  multilog (log, LOG_INFO, "marking header filled\n");
  if (ipcbuf_mark_filled (hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled header block\n");
    return EXIT_FAILURE;
  }

  uint64_t bufsz = ipcbuf_get_bufsz ((ipcbuf_t *)hdu->data_block);
  multilog (log, LOG_INFO, "data block bufsz=%"PRIu64"\n", bufsz);

  if (verbose)
  {
    fprintf (stderr, "Writing %"PRIu64" bytes to data block in blocks of %"PRIu64"\n", bytes_to_write, bufsz);
  }

  data = (char*) malloc (bufsz);
  assert (data != 0);

  const unsigned opened = 5;
  unsigned i;
  uint64_t bytess[opened];

  uint64_t index = 0;
  uint64_t interleave_num = 1024;

  uint64_t bufid;
  char * buf;
  while (bytes_to_write)
  {
    for (i=0; i<opened; i++)
    {
      bytess[i] = bufsz;
      if (bytess[i] > bytes_to_write)
        bytess[i] = bytes_to_write;

      multilog (log, LOG_INFO, "ipcio_open_block_write()\n");
      buf = ipcio_open_block_write (hdu->data_block, &bufid);
      if (buf)
      {
        multilog (log, LOG_INFO, "buf=%p id=%"PRIu64"\n", buf, bufid);

        // data_size is the number of bytes to write, uint64_size is the 
        // number of uint64s to write
        uint64_t uint64_count = bufsz / sizeof(uint64_t);

        unsigned k = 0;
        uint64_t j = 0;
        char * ptr = 0;
        unsigned char ch;

        multilog (log, LOG_INFO, "writing %lu 64-bit uints\n", uint64_count);
        for (j=0; j<uint64_count; j++)
        {
          // set the ptr to the correct place in the array
          ptr = buf + j * sizeof(uint64_t);
          for (k = 0; k < 8; k++ )
          {
            ch = (index >> ((k & 7) << 3)) & 0xFF;
            ptr[8 - k - 1] = ch;
          }

          index++;
          if (index % interleave_num == 0)
          {
            if (verbose)
              fprintf(stderr, "increment %"PRIu64" -> ", index);
            index += interleave_num;
            if (verbose)
              fprintf(stderr, "%"PRIu64"\n", index);
          }
        }
        bytes_to_write -= bytess[i];
      }
      else
        bytess[i] = 0;
    }

    for (i=0; i<opened; i++)
    {
      if (bytess[i] > 0)
      {
        multilog (log, LOG_INFO, "ipcio_close_block_write(%"PRIu64")\n", bytess[i]);
        ipcio_close_block_write (hdu->data_block, bytess[i]);
      }
    }
  }

  if (dada_hdu_unlock_write (hdu) < 0)
    return EXIT_FAILURE;

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  dada_hdu_destroy (hdu);

  return EXIT_SUCCESS;
}

