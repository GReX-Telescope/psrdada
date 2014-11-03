#!/usr/bin/env perl

###############################################################################
#
# server_mopsr_input_monitor.pl
#
# This script maintains sockets to all the mopsr_dbstats processes to collate
# the binary monitoring data, which it stores in memory, can also dump this data
# to disk
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Mopsr;           # Mopsr Module for configuration options
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DL           => 3;

#
# Global Variables
#
our %cfg                        = Mopsr::getConfig();
our $error                      = $cfg{"STATUS_DIR"}."/mopsr_web_monitor.error";
our $warn                       = $cfg{"STATUS_DIR"}."/mopsr_web_monitor.warn";
our $quit_daemon : shared       = 0;
our $daemon_name : shared       = Dada::daemonBaseName($0);

#
# Main
#
{

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";

  my $control_thread = 0;
  my $sock = 0;
  my $rh = "";
  my $string = "";
  my $handle = 0;

  # Sanity check for this script
  if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
   print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
   print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
   exit(1);
  }

  $sock = new IO::Socket::INET (
    LocalHost => $cfg{"SERVER_HOST"}, 
    LocalPort => $cfg{"SERVER_INPUT_MONITOR_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  die "Could not create listening socket: $!\n" unless $sock;

  # Redirect standard output and error
  # Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, DL, "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($sock);   # add the main socket to the set

  Dada::logMsg(1, DL, "Waiting for connection on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_INPUT_MONITOR_PORT"});

  # start the input monitoring thread
  my $input_monitor_thread = threads->new(\&inputMonitorThread);

  # Sleep for a few seconds to allow the threads to start and collect
  # their first iteration of data
  sleep(3);

  while (!$quit_daemon)
  {
    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 2);

    foreach $rh (@$rh_set) 
    {
      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $sock)
      {
        $handle = $rh->accept();
        $handle->autoflush();
        Dada::logMsg(2, DL, "main: accepting connection");

        # Add this read handle to the set
        $read_set->add($handle); 
        $handle = 0;
      }
      else
      {
        $string = Dada::getLine($rh);

        if (! defined $string) 
        {
          Dada::logMsg(2, DL, "main: closing a connection");
          $read_set->remove($rh);
          close($rh);
        } 
        else
        {
          Dada::logMsg(2, DL, "<- ".$string);
          my $r = "hello!";

          print $rh $r."\n";
          Dada::logMsg(2, DL, "-> ".$r);

          # experimental!
          $read_set->remove($rh);
          close($rh);
        }
      }
    }
  }

  Dada::logMsg(2, DL, "joining control_thread");
  # Rejoin our daemon control thread
  $control_thread->join();
  Dada::logMsg(2, DL, "control_thread joined");

  # Rejoin other threads
  Dada::logMsg(2, DL, "joining input_monitor_thread");
  $input_monitor_thread->join();
  Dada::logMsg(2, DL, "input_monitor_thread joined");

  # close the listening socket
  close($sock);
                                                                                  
  Dada::logMsg(0, DL, "STOPPING SCRIPT");
                                                                                  
  exit(0);
}

###############################################################################
#
# Functions
#

sub controlThread($) {

  (my $pid_file) = @_;

  Dada::logMsg(2, DL, "controlThread: starting");

  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  Dada::logMsg(2, DL, "Unlinking PID file: ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(1, DL, "controlThread: exiting");

}


#
# Handle INT AND TERM signals
#
sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  if ($quit_daemon)
  {
    print STDERR $daemon_name." : Exiting\n";
    exit(1);
  }
  $quit_daemon = 1;
}


#
# connect to all mopsr_dbstats instances to extract data for each antenna
#
sub inputMonitorThread() 
{
  Dada::logMsg(1, DL, "inputMonitorThread: starting");

  my $nant  = $cfg{"NANT"};
  my $nchan = $cfg{"NCHAN"};
  my $nant  = 1;
  my $ndim  = 2;
  my $nsamp = 0;
  my $bytes = 0;

  my $i = 0;
  my @hosts = ();
  my @raw = ();

  # SEND is the IB send and hence the 10GbE recv
  for ($i=0; $i<$cfg{"NSEND"}; $i++)
  {
    push (@hosts, $cfg{"SEND_".$i});
    push (@raw, 0);
  }

  my $sleep_time = 10;
  my $sleep_counter = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $host = ""; 
  my $port = "54321";
  my $handle = 0;

  while (!$quit_daemon) 
  {
    if ($sleep_counter > 0) 
    {
      $sleep_counter--;
      sleep(1);
    } 
    else
    {
      Dada::logMsg(2, DL, "inputMonitorThread: collecting data from ".($#hosts+1)." hosts");
      # collect data from each host
      for ($i=0; $i<=$#hosts; $i++)
      {
        $host = $hosts[$i];

        # open socket
        Dada::logMsg(2, DL, "inputMonitorThread: connecting to ".$host.":".$port);
        $handle = Dada::connectToMachine ($host , $port, DL);
        if (!$handle)
        {
          Dada::logMsg(0, DL, "could not open socket [".$handle."] to mopsr_dbstats at ".$host.":".$port);
          next;
        }

        Dada::logMsg(0, DL, $host.":".$port." <- nsamp");
        ($result, $response) = Dada::sendTelnetCommand ($handle, "nsamp");
        Dada::logMsg(0, DL, $host.":".$port." -> ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsg(0, DL, $host.":".$port." <- nsamp failed: ".$response);
          next;
        }
        $nsamp = int($response);
        
        $bytes = $nant * $nchan * $ndim * $nsamp;

        # read data into our buffer
        Dada::logMsg(0, DL, $host.":".$port." <- dump (expected ".$bytes." bytes)");
        $handle->write("dump\r\n");
        my $read = $handle->read ($raw[$i], $bytes);
        Dada::logMsg(0, DL, $host.":".$port." -> [binary] read ".$read." bytes");

        # close socket
        $handle->close();

        $handle = 0;
      }
      $sleep_counter = $sleep_time;
    }
  }

  Dada::logMsg(1, DL, "inputMonitorThread: exiting");
}
