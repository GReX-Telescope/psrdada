#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use IO::Socket;
use Net::hostent;
use Time::Local;
use threads;
use threads::shared;
use Dada;
use Caspsr;

Dada::preventDuplicateDaemon(basename($0));

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $current_state : shared;


#
# Global Variable Initialization
#
%cfg           = Caspsr::getConfig();
$dl            = 1;
$daemon_name   = Dada::daemonBaseName($0);
$quit_daemon = 0;
$current_state = "Unknown";

#
# Constants
#

#
# Function Prototypes
#


#
# Main
#
{

  $SIG{INT} = \&sigHandle;

  my $host = $cfg{"TCS_INTERFACE_HOST"};
  my $port = $cfg{"TCS_INTERFACE_PORT"};

  my @sources = ("J0437-4715", "J0613-0200", "J0711-6830", "J0737-3039A", "J1001-5939", "J1017-7156", "J1022+1001", "J1024-0719", "J1045-4509", "J1600-3053", "J1643-1224", "J1713+0747", "J1730-2304", "J1744-1134", "J1824-2452", "J1857+0943", "J1909-3744", "J1939+2134", "J2124-3358", "J2129-5721", "J2145-0750", "J2241-5236"); 
  my $source_index = 0;
  my $source = "";

  my %header = ();
  $header{"freq"} = "1382.00000";
  $header{"pid"} = "P999";
  $header{"band"} = "-400.0000";
  $header{"procfil"} = "dspsr.gpu";

  # these will be "guessed" from the source name
  $header{"ra"} = "";
  $header{"dec"} = "";

  my $state_info_thread = threads->new(\&stateInfoThread);

  my $handle = 0;
  my $to_sleep = 0;
  my $cmd = "";
  my $tmp = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $k = "";

  # connect to TCS interface
  Dada::logMsg(2, $dl, "connecting to ".$host." ".$port);
  $handle = Dada::connectToMachine($host, $port);
  if (!$handle)
  {
    Dada::logMsg(0, $dl, "Could not connect to ".$host.":".$port);
    $quit_daemon = 1;
    next;
  }

  while (!$quit_daemon)
  {

    # ensure we are in the Idle state before begining
    $to_sleep = 40;
    while ((!$quit_daemon) && ($to_sleep > 0) && ($current_state ne "Idle"))
    {
      $to_sleep--;
      sleep 1;
    }

    if ($quit_daemon)
    {
      Dada::logMsg(0, $dl, "Daemon asked to exit whilst waiting for Idle state");
      $quit_daemon = 1;
      next;
    }

    if ($to_sleep <= 0)
    {
      Dada::logMsg(0, $dl, "Timed out waiting for Idle state");
      $quit_daemon = 1;
      next;
    }


    # Setup the header
    $source = $sources[$source_index];
    $header{"src"} = $source;
    $tmp = $source;
    $tmp =~ s/^[A-Z]//;
    $tmp =~ s/[A-Z]$//;

    my @bits = split(/[-+]/, $tmp);
    $header{"ra"}  = substr($bits[0],0,2).":".substr($bits[0],2,2);
    $header{"dec"} = substr($bits[1],0,2).":".substr($bits[1],2,2);

    my @keys = keys %header;

    for ($i=0; $i<=$#keys; $i++)
    {
      $k = $keys[$i];
      $cmd = Dada::headerFormat($k, $header{$k});

      Dada::logMsg(0, $dl, "TCS <- ".$cmd);
      ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
      Dada::logMsg(0, $dl, "TCS <- ".$result." ".$response);
      if ($result ne "ok")
      {
        $quit_daemon = 1;
        next;
      }
    }

    $cmd = "start";
    Dada::logMsg(0, $dl, "TCS <- ".$cmd);
    ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
    Dada::logMsg(0, $dl, "TCS <- ".$result." ".$response);
    if ($result ne "ok")
    {
      $quit_daemon = 1;
      next;
    }

    $response = Dada::getLine($handle);
    Dada::logMsg(0, $dl, "TCS <- ".$response);

    $to_sleep = 3600;
    while ((!$quit_daemon) && ($to_sleep > 0))
    {
      $to_sleep--;
      sleep 1;
    }

    $cmd = "stop";
    Dada::logMsg(0, $dl, "TCS <- ".$cmd);
    ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
    Dada::logMsg(0, $dl, "TCS <- ".$result." ".$response);
    if ($result ne "ok")
    {
      $quit_daemon = 1;
      next;
    }

    $source_index ++;
    $source_index = $source_index % ($#sources+1);

  }

  if (($current_state ne "Idle") && ($current_state =~ m/Stop/)) 
  {
    $cmd = "STOP";
    Dada::logMsg(0, $dl, "TCS <- ".$cmd);
    ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
    Dada::logMsg(0, $dl, "TCS <- ".$result." ".$response);
  }

  $quit_daemon = 1;

  $handle->close();

  $state_info_thread->join();

  exit 0;
}


sub stateInfoThread()
{

  my $host = "";
  my $port = "";
  my $sock = "";
  my $cmd ="";
  my $response = ""; 

  $host = "srv0";
  $port = $cfg{"TCS_STATE_INFO_PORT"};

  Dada::logMsg(2, $dl, "stateInfoThread: Getting TCS state at ".$host.":".$port);
  $sock = Dada::connectToMachine($host, $port);
  if (!$sock)
  {
    Dada::logMsg(0, $dl, "stateInfoThread: Could not connect to ".$host.":".$port);
    $quit_daemon = 1;
    return;
  }

  $cmd = "state";
  while ((!$quit_daemon) && ($sock))
  {

    Dada::logMsg(2, $dl, "stateInfoThread: TCS <- ".$cmd);
    print $sock $cmd."\r\n";

    $response = Dada::getLine($sock);
    Dada::logMsg(2, $dl, "stateInfoThread: TCS -> ".$response);
    
    if (($response eq "") || ($response =~ m/Error/))
    {
      $quit_daemon = 1;
    } 
    else
    {
      $current_state = $response; 
      sleep 2;
    }
  }

  $sock->close();
  return;
}



#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  if ($quit_daemon)
  {
    print STDERR $daemon_name." : EXITING\n";
    exit(0);
  }
  else
  {
    print STDERR $daemon_name." : Asking daemon to exit\n";
    $quit_daemon = 1;
  }
  
}

