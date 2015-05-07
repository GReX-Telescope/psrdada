#!/usr/bin/env perl

###############################################################################
#
# parkes_monitor.pl
#
# This script opens a UDP socket on port 54321 of the specified interface and
# listens for the Monica UDP message that is transmitted from orion each second
#

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use XML::Simple qw(:strict);
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;

#
# Constants
#
use constant DL   => 3;
use constant PORT => "54321";

#
#
#
our $quit : shared;
$quit = 0;

#
# Main
#
{
  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  my $datagram = 0;
  my $sock = 0;
  my $rh = "";
  my $string = "";
  my $handle = 0;
  my $flags = 0;
  my $xml;

  $sock = new IO::Socket::INET (
    LocalPort => PORT,
    Proto => 'udp',
    Reuse => 1
  );
  die "Could not create UDP socket: $!\n" unless $sock;

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($sock);            # add the main socket to the set

  while (!$quit)
  {
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$rh_set)
    {
      $sock->recv($datagram, 1500, $flags);
      $xml = XMLin ($datagram, ForceArray => 0, KeyAttr => 0, SuppressEmpty => 1, NoAttr => 1);
      print "========================================================\n";
      print " RA          ".$xml->{"lat"}."\n";
      print " DEC         ".$xml->{"lng"}."\n";
      print " Local Time  ".$xml->{"aest"}."\n";
      print " UTC Time    ".$xml->{"utc"}."\n";
      print " Status      ".$xml->{"fstat"}."\n";
      print " Drive Time  ".$xml->{"drv_time"}."\n";
    }
  }

  close($sock);    

  exit(0);
}

#
# Handle INT AND TERM signals
#
sub sigHandle($)
{
  my $sigName = shift;
  if ($sigName ne "PIPE")
  {
    print STDERR $0." : Received SIG".$sigName."\n";
    if ($quit)
    {
      print STDERR $0." : Exiting\n";
      exit(1);
    }
    $quit = 1;
  }
}

