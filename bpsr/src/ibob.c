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
  ibob_t* bob = malloc (sizeof(ibob_t));

  bob->fd = -1;
  bob->buffer_size = 0;
  bob->buffer = 0;
  bob->emulate_telnet = 1;

  return bob;
}

/*! free all resources reserved for ibob communications */
int ibob_destroy (ibob_t* bob)
{
  if (!bob)
    return -1;

  if (bob->fd != -1)
    sock_close (bob->fd);

  if (bob->buffer)
    free (bob->buffer);

  free (bob);

  return 0;
}

/*! set the host and port number of the ibob */
int ibob_set_host (ibob_t* bob, const char* host, int port)
{
  if (!bob)
    return -1;

  if (bob->host)
    free (bob->host);

  bob->host = strdup (host);
  bob->port = port;

  return 0;
}

/*! set the ibob number (use default IP base and port) */
int ibob_set_number (ibob_t* bob, unsigned number)
{
  if (!bob)
    return -1;

  if (bob->host)
    free (bob->host);

  bob->host = strdup (IBOB_VLAN_BASE"XXXXXX");
  sprintf (bob->host, IBOB_VLAN_BASE"%u", number);

  bob->port = IBOB_PORT;

  return 0;
}

int ibob_emulate_telnet (ibob_t* bob)
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

  if (!bob)
    return -1;

  if (bob->fd == -1)
    return -1;

  message_length = strlen (message1);

  if (sock_write (bob->fd, message1, message_length) < message_length)
  {
    fprintf (stderr, "ibob_emulate_telnet: could not send message 1 - %s\n",
             strerror(errno));
    return -1;
  }

  message_length = strlen (expected_response);

  if (sock_read (bob->fd, bob->buffer, message_length) < message_length)
  {
    fprintf (stderr, "ibob_emulate_telnet: could not receive response - %s\n",
	     strerror(errno));
    return -1;
  }

  message_length = strlen (message2);

  if (sock_write (bob->fd, message2, message_length) < message_length)
  {
    fprintf (stderr, "ibob_emulate_telnet: could not send message 2 - %s\n",
             strerror(errno));
    return -1;
  }

  // read the welcome message up to the iBoB prompt
  return ibob_recv (bob, bob->buffer, bob->buffer_size);
}

static const char ibob_prompt[] = "IBOB % ";

/*! start reading from/writing to an ibob */
int ibob_open (ibob_t* bob)
{
  if (!bob)
    return -1;

  if (bob->fd != -1)
  {
    fprintf (stderr, "ibob_open: already open\n");
    return -1;
  }

  if (bob->host == 0)
  {
    fprintf (stderr, "ibob_open: host name not set\n");
    return -1;
  }

  if (bob->port == 0)
  {
    fprintf (stderr, "ibob_open: port not set\n");
    return -1;
  }

  bob->fd = sock_open (bob->host, bob->port);
  if ( bob->fd < 0 )
  {
    fprintf (stderr, "ibob_open: could not open %s %d - %s\n",
	     bob->host, bob->port, strerror(errno));
    return -1;
  }

  if (!bob->buffer_size)
    bob->buffer_size = 1024;

  bob->buffer = realloc (bob->buffer, bob->buffer_size);

  if (bob->emulate_telnet)
    return ibob_emulate_telnet (bob);

  return 0;
}

/*! stop reading from/writing to an ibob */
int ibob_close (ibob_t* bob)
{
  if (!bob)
    return -1;

  if (bob->fd == -1)
    return -1;

  int fd = bob->fd;
  bob->fd = -1;

  return sock_close (fd);
}

/*! return true if already open */
int ibob_is_open (ibob_t* bob)
{
  return bob && (bob->fd != -1);
}

/*! reset packet counter on next UTC second, returned */
time_t ibob_arm (ibob_t* bob)
{
  return 0;
}

int ibob_ignore (ibob_t* bob)
{
  return ibob_recv (bob, bob->buffer, bob->buffer_size);
}

/*! configure the ibob */
int ibob_configure (ibob_t* bob, const char* mac_address)
{
  if (!bob)
    return -1;

  if (bob->fd == -1)
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

  ibob_send (bob, "regwrite reg_ip 0x0a000004");
  ibob_ignore (bob);

  ibob_send (bob, "regwrite reg_10GbE_destport0 4001");
  ibob_ignore (bob);

  ibob_send (bob, "write l xd0000000 xffffffff");
  ibob_ignore (bob);

  ibob_send (bob, "setb x40000000");
  ibob_ignore (bob);

  ibob_send (bob, "writeb l 0 x00000060");
  ibob_ignore (bob);

  ibob_send (bob, "writeb l 4 xdd47e301");
  ibob_ignore (bob);

  ibob_send (bob, "writeb l 8 x00000000");
  ibob_ignore (bob);

  ibob_send (bob, "writeb l 12 x0a000001");
  ibob_ignore (bob);

  ibob_send (bob, "writeb b x16 x0f");
  ibob_ignore (bob);

  ibob_send (bob, "writeb b x17 xa0");
  ibob_ignore (bob);

  char command [64];
  sprintf(command, "writeb l x3020 %s", macone);
  ibob_send (bob, command);
  ibob_ignore (bob);
  free (macone);

  sprintf(command, "writeb l x3024 %s", mactwo);
  ibob_send (bob, command);
  ibob_ignore (bob);
  free (mactwo);

  ibob_send (bob, "writeb b x15 xff");
  ibob_ignore (bob);

  ibob_send (bob, "write l xd0000000 x0");
  ibob_ignore (bob);

  return 0;
}

/*! write bytes to ibob */
ssize_t ibob_send (ibob_t* bob, const char* message)
{
  if (!bob)
    return -1;

  if (bob->fd == -1)
    return -1;

  int length = strlen (message);

  if (length+2 > bob->buffer_size)
  {
    bob->buffer_size = length * 2;
    bob->buffer = realloc (bob->buffer, bob->buffer_size);
  }

  snprintf (bob->buffer, bob->buffer_size, "%s\r", message);
  length ++;

  int wrote = sock_write (bob->fd, bob->buffer, length);
  if (wrote < length)
    return -1;

  if (bob->emulate_telnet)
  {
    /* read the echoed characters */
    int echo = ibob_recv (bob, bob->buffer, length);
    if (echo < length)
      return -1;
  }

  return 0;
}

/*! read bytes from ibob */
ssize_t ibob_recv (ibob_t* bob, char* ptr, size_t bytes)
{
  if (!bob)
    return -1;

  if (bob->fd == -1)
    return -1;

  size_t total_got = 0;
  ssize_t got = 0;

  while (bytes)
  {
    // do not time-out while emulating telnet (end-of-data = prompt reception)
    if (bob->emulate_telnet)
    {
      got = sock_read (bob->fd, ptr+total_got, bytes);
      char* prompt = strstr (ptr, ibob_prompt);
      if (prompt)
      {
	*prompt = '\0';
	return strlen(ptr);
      }
    }
    else
    {
      got = sock_tm_read (bob->fd, ptr+total_got, bytes, 0.1);
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

