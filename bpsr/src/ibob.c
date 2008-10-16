/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "ibob.h"
#include "sock.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

ibob_t* ibob_construct ()
{
  ibob_t* ibob = malloc (sizeof(ibob_t));

  ibob->fd = -1;
  ibob->buffer_size = 0;
  ibob->buffer = 0;
  ibob->emulate_telnet = 1;

  return ibob;
}

/*! free all resources reserved for ibob communications */
int ibob_destroy (ibob_t* ibob)
{
  if (!ibob)
    return -1;

  if (ibob->fd != -1)
    sock_close (ibob->fd);

  if (ibob->buffer)
    free (ibob->buffer);

  free (ibob);

  return 0;
}

/*! set the host and port number of the ibob */
int ibob_set_host (ibob_t* ibob, const char* host, int port)
{
  if (!ibob)
    return -1;

  if (ibob->host)
    free (ibob->host);

  ibob->host = strdup (host);
  ibob->port = port;

  return 0;
}

/*! set the ibob number (use default IP base and port) */
int ibob_set_number (ibob_t* ibob, unsigned number)
{
  if (!ibob)
    return -1;

  if (ibob->host)
    free (ibob->host);

  ibob->host = strdup (IBOB_VLAN_BASE"XXXXXX");
  sprintf (ibob->host, IBOB_VLAN_BASE"%u", number);

  ibob->port = IBOB_PORT;

  return 0;
}

int ibob_emulate_telnet (ibob_t* ibob)
{
  const char message1[] = { 255, 251, 37,
			    255, 253, 38,
			    255, 251, 38,
			    255, 253, 3,
			    255, 251, 24,
			    255, 251, 31,
			    255, 251, 32,
			    255, 251, 33,
			    255, 251, 34,
			    255, 251, 39,
			    255, 253, 5,
			    255, 251, 35,
			    0 };

  const char expected_response [] = { 255, 251, 1, 
				      255, 251, 3, 
				      0 };

  const char message2[] = { 255, 253, 1,
			    0 };

  int message_length = 0;

  if (!ibob)
    return -1;

  if (ibob->fd == -1)
    return -1;

  message_length = strlen (message1);

  if (sock_write (ibob->fd, message1, message_length) < message_length)
  {
    fprintf (stderr, "ibob_emulate_telnet: could not send message 1 - %s\n",
             strerror(errno));
    return -1;
  }

  message_length = strlen (expected_response);

  if (sock_read (ibob->fd, ibob->buffer, message_length) < message_length)
  {
    sock_close (ibob->fd);

    ibob->fd = sock_open (ibob->host, ibob->port);
    if ( ibob->fd < 0 )
    {
      fprintf (stderr, "ibob_emulate_telnet: could not reopen %s %d - %s\n",
               ibob->host, ibob->port, strerror(errno));
      return -1;
    }

    return ibob_emulate_telnet (ibob);
  }

  message_length = strlen (message2);

  if (sock_write (ibob->fd, message2, message_length) < message_length)
  {
    fprintf (stderr, "ibob_emulate_telnet: could not send message 2 - %s\n",
             strerror(errno));
    return -1;
  }

  // read the welcome message up to the iBoB prompt
  return ibob_recv (ibob, ibob->buffer, ibob->buffer_size);
}

static const char ibob_prompt[] = "IBOB % ";

/*! start reading from/writing to an ibob */
int ibob_open (ibob_t* ibob)
{
  if (!ibob)
    return -1;

  if (ibob->fd != -1)
  {
    fprintf (stderr, "ibob_open: already open\n");
    return -1;
  }

  if (ibob->host == 0)
  {
    fprintf (stderr, "ibob_open: host name not set\n");
    return -1;
  }

  if (ibob->port == 0)
  {
    fprintf (stderr, "ibob_open: port not set\n");
    return -1;
  }

  ibob->fd = sock_open (ibob->host, ibob->port);
  if ( ibob->fd < 0 )
  {
    fprintf (stderr, "ibob_open: could not open %s %d - %s\n",
	     ibob->host, ibob->port, strerror(errno));
    return -1;
  }

  if (!ibob->buffer_size)
    ibob->buffer_size = 1024;

  ibob->buffer = realloc (ibob->buffer, ibob->buffer_size);

  if (ibob->emulate_telnet)
    return ibob_emulate_telnet (ibob);

  return 0;
}

/*! stop reading from/writing to an ibob */
int ibob_close (ibob_t* ibob)
{
  if (!ibob)
    return -1;

  if (ibob->fd == -1)
    return -1;

  int fd = ibob->fd;
  ibob->fd = -1;

  return sock_close (fd);
}

/*! return true if already open */
int ibob_is_open (ibob_t* ibob)
{
  return ibob && (ibob->fd != -1);
}

/*! return true if iBoB is alive */
int ibob_ping (ibob_t* ibob)
{
  if (ibob_send (ibob, "\r") < 0)
    return -1;

  return ibob_ignore (ibob);
}

/*! reset packet counter on next UTC second, returned */
time_t ibob_arm (ibob_t* ibob)
{
  return 0;
}

int ibob_ignore (ibob_t* ibob)
{
  return ibob_recv (ibob, ibob->buffer, ibob->buffer_size);
}

/*! configure the ibob */
int ibob_configure (ibob_t* ibob, const char* mac_address)
{
  if (!ibob)
    return -1;

  if (ibob->fd == -1)
    return -1;

  int length = strlen (mac_address);
  if (length != 12)
  {
    fprintf (stderr,
	     "ibob_configure: MAC address '%s' is not 12 characters\n",
	     mac_address);
    return -1;
  }

  const char* mac_base = "x00000000";

  char* macone = strdup (mac_base);
  char* mactwo = strdup (mac_base);

  char* maclow = strdup (mac_address);

  int i = 0;
  for (i=0; i<length; i++)
    maclow[i] = tolower (mac_address[i]);

  strncpy(macone+5,maclow,4);
  strncpy(mactwo+1,maclow+4,8);

  free (maclow);

  ibob_send (ibob, "regwrite reg_ip 0x0a000004");
  ibob_ignore (ibob);

  ibob_send (ibob, "regwrite reg_10GbE_destport0 4001");
  ibob_ignore (ibob);

  ibob_send (ibob, "write l xd0000000 xffffffff");
  ibob_ignore (ibob);

  ibob_send (ibob, "setb x40000000");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb l 0 x00000060");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb l 4 xdd47e301");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb l 8 x00000000");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb l 12 x0a000001");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb b x16 x0f");
  ibob_ignore (ibob);

  ibob_send (ibob, "writeb b x17 xa0");
  ibob_ignore (ibob);

  char command [64];
  sprintf(command, "writeb l x3020 %s", macone);
  ibob_send (ibob, command);
  ibob_ignore (ibob);
  free (macone);

  sprintf(command, "writeb l x3024 %s", mactwo);
  ibob_send (ibob, command);
  ibob_ignore (ibob);
  free (mactwo);

  ibob_send (ibob, "writeb b x15 xff");
  ibob_ignore (ibob);

  ibob_send (ibob, "write l xd0000000 x0");
  ibob_ignore (ibob);

  return 0;
}

/*! write bytes to ibob */
ssize_t ibob_send (ibob_t* ibob, const char* message)
{
  ssize_t wrote = ibob_send_async (ibob, message);

  if (wrote < 0)
    return -1;

  if (ibob_recv_echo (ibob, wrote) < 0)
    return -1;

  return wrote;
}

/*! write bytes to ibob */
ssize_t ibob_send_async (ibob_t* ibob, const char* message)
{
  if (!ibob)
    return -1;

  if (ibob->fd == -1)
    return -1;

  int length = strlen (message);

  if (length+2 > ibob->buffer_size)
  {
    ibob->buffer_size = length * 2;
    ibob->buffer = realloc (ibob->buffer, ibob->buffer_size);
  }

  snprintf (ibob->buffer, ibob->buffer_size, "%s\r", message);
  length ++;

  return sock_write (ibob->fd, ibob->buffer, length);
}

/*! if emulating telnet, receive echoed characters */
int ibob_recv_echo (ibob_t* ibob, size_t length)
{
  if (!ibob->emulate_telnet)
    return length;

  /* read the echoed characters */
  ssize_t echo = ibob_recv (ibob, ibob->buffer, length);
  if (echo < length)
    return -1;

  return echo;
}

/*! read bytes from ibob */
ssize_t ibob_recv (ibob_t* ibob, char* ptr, size_t bytes)
{
  if (!ibob)
    return -1;

  if (ibob->fd == -1)
    return -1;

  size_t total_got = 0;
  ssize_t got = 0;

  while (bytes)
  {
    // do not time-out while emulating telnet (end-of-data = prompt reception)
    if (ibob->emulate_telnet)
    {
      got = sock_read (ibob->fd, ptr+total_got, bytes);
      char* prompt = strstr (ptr, ibob_prompt);
      if (prompt)
      {
	*prompt = '\0';
	return strlen(ptr);
      }
    }
    else
    {
      got = sock_tm_read (ibob->fd, ptr+total_got, bytes, 0.1);
      if (got == 0)
      {
	ptr[total_got] = '\0';
	return total_got;
      }
    }

    if (got < 0)
    {
      perror ("ibob_recv: ");
      return -1;
    }

    total_got += got;
    bytes -= got;
  }

  return total_got;
}

