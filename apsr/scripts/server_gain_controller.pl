#!/usr/bin/env perl

###############################################################################
#
# server_gain_controller.pl
#
# This script monitors level setting requests from PWC's and forwards them
# on to the PDFB3 interface to dynamically adjust gain levels during an observation
# to files on the server.

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Dada;           # DADA Module for configuration options
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Constants
#
use constant DEBUG_LEVEL => 1;
use constant PIDFILE     => "gain_controller.pid";
use constant LOGFILE     => "gain_controller.log";


#
# Global Variables
#
our %cfg = Dada->getDadaConfig();      # dada.cfg
our $quit_daemon : shared  = 0;


# Autoflush output
$| = 1;


# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


#
# Local Varaibles
#
my $dfb3_host = $cfg{"DFB3_HOST"};
my $dfb3_port = $cfg{"DFB3_PORT"};
my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

my $daemon_control_thread = 0;
my $result = "";
my $response = "";
my $cmd = "";
my $i = 0;
my $server_socket = "";   # Server socket for new connections
my $rh = "";
my $string = "";


# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}

$server_socket = new IO::Socket::INET (
    LocalHost => $cfg{"SERVER_HOST"}, 
    LocalPort => $cfg{"SERVER_GAIN_CONTROL_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
);

die "Could not create listening socket: $!\n" unless $server_socket;

debugMessage(2, "Trying to connect to DFB3 host: ".$dfb3_host.":".$dfb3_port);
my $dfb3_socket = Dada->connectToMachine($dfb3_host, $dfb3_port);
my $dfb3_gain = 0;
if (!$dfb3_socket) {
 debugMessage(0, "Could not connect to DFB3 Machine ".$dfb3_host.":".$dfb3_port);
}


# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

debugMessage(0, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

my $read_set = new IO::Select();  # create handle set for reading
$read_set->add($server_socket);   # add the main socket to the set

debugMessage(2, "Waiting for connection on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_GAIN_CONTROL_PORT"});

while (!$quit_daemon) {

  # If we haven't got a DFB3 connection, try to open it
  if (!$dfb3_socket) {

    debugMessage(2, "Trying to connect to the DFB3 ".$dfb3_host.":".$dfb3_port);

    $dfb3_socket = Dada->connectToMachine($dfb3_host, $dfb3_port);
    if ($dfb3_socket) {
      debugMessage(0, "Connected to DFB3 Machine ".$dfb3_host.":".$dfb3_port);
      $read_set->add($dfb3_socket);
    }
  }

  # Get all the readable handles from the server
  my ($readable_handles) = IO::Select->select($read_set, undef, undef, 2);

  foreach $rh (@$readable_handles) {

    # if it is the main socket then we have an incoming connection and
    # we should accept() it and then add the new socket to the $Read_Handles_Object
    if ($rh == $server_socket) { 

      my $handle = $rh->accept();
      $handle->autoflush();
      my $hostinfo = gethostbyaddr($handle->peeraddr);
      my $hostname = $hostinfo->name;

      debugMessage(2, "Accepting connection from ".$hostname);

      # Add this read handle to the set
      $read_set->add($handle); 

    } elsif ($rh == $dfb3_socket) {

      debugMessage(0, "Unexpected message from the DFB3 socket");
      $string = Dada->getLine($rh);
      if (! defined $string) {
        debugMessage(2, "Lost connection from DFB3 ".$dfb3_host.":".$dfb3_port.", closing socket");
        $read_set->remove($rh);
        close($rh);
        $dfb3_socket = 0;
      } else {
        debugMessage(0, "Received \"".$string."\" from the DFB3, ignoring");
      }

    } else {

      my $hostinfo = gethostbyaddr($rh->peeraddr);
      my $hostname = $hostinfo->name;
      my @parts = split(/\./,$hostname);
      my $machine = $parts[0];
      $string = Dada->getLine($rh);

      # If the string is not defined, then we have lost the connection.
      # remove it from the read_set
      if (! defined $string) {
        debugMessage(1, "Lost connection from ".$hostname.", closing socket");

        $read_set->remove($rh);
        close($rh);

      # We have a request
      } else {

        debugMessage(2, $machine.": \"".$string."\"");

        # If a client is asking what its CHANNEL base multiplier
        if ($string =~ m/CHANNEL_BASE/) {

          debugMessage(2, $machine." -> ".$string);
          for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
            if ($machine eq $cfg{"PWC_".$i}) {
              debugMessage(2, $machine." <- ".$i);
              print $rh $i."\r\n";
              debugMessage(1, "BASE: ".$machine." ".$i);
            }
          }

        # Connect to the DFB3 and get the gain
        } elsif ( $string =~ m/^APSRGAIN (\d+) (0|1)$/) {

          my ($ignore, $chan, $pol) = split(/ /, $string);
          my $dfb3_string = "APSRGAIN ".$chan." ".$pol;

          my $dfb3_gain = 0;

          debugMessage(3, "  dfb3 <- ".$dfb3_string." [get]");

          print $dfb3_socket $dfb3_string."\n";

          my $dfb3_response = Dada->getLine($dfb3_socket);

          debugMessage(2, "  dfb3 -> ".$dfb3_response);

          print $rh $dfb3_response."\r\n";

          debugMessage(1, "GET:  ".$machine." ".$dfb3_response);

          debugMessage(2, $machine." <- ".$dfb3_response);

        } elsif ($string =~ m/^APSRGAIN (\d+) (0|1) (\d)+$/) {

          debugMessage(3, $machine." -> ".$string." [set]");

          my ($ignore, $chan, $pol, $val) = split(/ /, $string);
          my $dfb3_string = "APSRGAIN ".$chan." ".$pol." ".$val;

          debugMessage(2, "  dfb3 <- ".$dfb3_string);
          print $dfb3_socket $dfb3_string."\n";
          
          my $dfb3_response = Dada->getLine($dfb3_socket);
          debugMessage(2, "  dfb3 -> ".$dfb3_response);
          
          print $rh $dfb3_response."\r\n";
          debugMessage(2, $machine." <- ".$dfb3_response);

          debugMessage(1, "SET:  ".$machine." ".$dfb3_string." ".$dfb3_response);

        } else {
          debugMessage(2, "Unknown request received");
        }
      }
      debugMessage(2, "=========================================");
    }
  }
}

# Rejoin our daemon control thread
$daemon_control_thread->join();

debugMessage(0, "STOPPING SCRIPT");

exit(0);


###############################################################################
#
# Functions
#


sub daemonControlThread() {

  debugMessage(2, "Daemon control thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});
  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  debugMessage(2, "Unlinking PID file: ".$pidfile);
  unlink($pidfile);

  debugMessage(2, "Daemon control thread ending");

}

sub debugMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    print "[".$time."] ".$message."\n";
  }
}

#
# Handle INT AND TERM signals
#
sub sigHandle($) {
                                                                                
  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting\n";
  exit(1);
                                                                                
}
