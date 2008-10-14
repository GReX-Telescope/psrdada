#!/usr/bin/env perl

###############################################################################
#
# server_ibob_manager.pl
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Bpsr;           # BPSR Module 
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Constants
#
use constant DEBUG_LEVEL => 2;
use constant PIDFILE     => "ibob_manager.pid";
use constant LOGFILE     => "ibob_manager.log";


#
# Global Variables
#
our %cfg = Bpsr->getBpsrConfig();      # Bpsr.cfg
our $quit_daemon : shared  = 0;


# Autoflush output
$| = 1;


# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


#
# Local Varaibles
#
my $logfile =         $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile =         $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;

my $daemon_control_thread = 0;
my @threads = ();
my @results = ();
my @responses = ();

my $i;
my $result;
my $response;
my $command = "config";

#
# Main
#

# Dada->daemonize($logfile, $pidfile);

logMessage(0, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);


my $server_socket = new IO::Socket::INET (
    LocalHost => "apsr17",
    LocalPort => "1999",
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
);

die "Could not create listening socket: $!\n" unless $server_socket;

my $rh;
my $read_set = new IO::Select();  # create handle set for reading
$read_set->add($server_socket);   # add the main socket to the set

while (!$quit_daemon) {

# Get all the readable handles from the server
my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1.0);

foreach $rh (@$readable_handles) {

  # if it is the main socket then we have an incoming connection and
  # we should accept() it and then add the new socket to the $Read_Handles_Object
  if ($rh == $server_socket) {

    my $handle = $rh->accept();
    $handle->autoflush();
    my $hostinfo = gethostbyaddr($handle->peeraddr);
    my $hostname = $hostinfo->name;

    logMessage(2, "Accepting connection from ".$hostname);

    # Add this read handle to the set
    $read_set->add($handle);

  } else {

    my $command = Dada->getLine($rh);

    # If the string is not defined, then we have lost the connection.
    # remove it from the read_set
    if (! defined $command) {
      logMessage(1, "Lost connection, closing socket");

      $read_set->remove($rh);
      close($rh);

    # We have a request
    } else {

      if ($command eq "set_levels") {

        # Spawn the communication threads
        for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
      
          my $pwc = $cfg{"PWC_".$i};
          my $dhost = $cfg{"IBOB_DEST_".$i};

          logMessage(2, "Calling runLevelSetter($dhost, 23)");
          @threads[$i] = threads->new(\&runLevelSetter, $dhost, "23");

        }

        @results = ();
        @responses = (); 
        $result = "ok";

        logMessage(2, "Waiting for threads to return");
        # Wait for the replies
        for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {

          logMessage(3, "Waiting for thread $i");
          (@results[$i],@responses[$i]) = $threads[$i]->join;

          if ($results[$i] ne "ok") {
            $result = "fail";
            logMessage(0, "Thread for ".$cfg{"PWC_".$i}." failed with ".$responses[$i]);
          } else {
            logMessage(3, "Thread for ".$cfg{"PWC_".$i}." returned ok");
          }
        }
        print $rh "$result\r\n";
    
      } elsif ($command eq "config") {

        # Spawn the communication threads
        for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {

          my $pwc = $cfg{"PWC_".$i};
          my $dhost = $cfg{"IBOB_DEST_".$i};
          my $mac = $cfg{"IBOB_LOCAL_MAC_ADDR_".$i};

          logMessage(2, "Calling runConfigurator($dhost, 23, $mac)");
          @threads[$i] = threads->new(\&runConfigurator, $dhost, "23", $mac);

        }

        @results = ();
        @responses = ();
        $result = "ok";

        logMessage(2, "Waiting for threads to return");
        # Wait for the replies
        for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {

          logMessage(3, "Waiting for thread $i");
          (@results[$i],@responses[$i]) = $threads[$i]->join;

          if ($results[$i] ne "ok") {
            $result = "fail";
            logMessage(0, "Thread for ".$cfg{"PWC_".$i}." failed with ".$responses[$i]);
          } else {
            logMessage(3, "Thread for ".$cfg{"PWC_".$i}." returned ok");
          }
        }
        print $rh "$result\r\n";

      } elsif ($command eq "rearm") {

        my ($result, $response) = rearmAll();
        
        if ($result eq "ok") {
          print $rh $response."\r\n";
        } else {
          print $rh "rearm failed\r\n";
        }

      } else {
        print "ignoring commmand\r\n";
      }
    }
  }
}
}


# rejoin threads
$daemon_control_thread->join();

logMessage(0, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#

#
# Manages the connection to the ibob as specified.
#

sub runLevelSetter($$) {

  my ($l_host, $l_port) = @_;

  my $result = "";
  my $response = "";

  # Check if an existing SSH tunnel exists
  my $cmd = "/home/apsr/linux_64/bin/ibob_level_setter  -n 1 ".$l_host." ".$l_port;
  logMessage(1, $cmd);

  $response = `$cmd 2>&1`;
  my $rval = $?;
  #system($cmd." 2>&1");

  if ($rval == 0) {
    return ("ok", $response);
  } else {
    return ("fail", $rval.":".$response );
  }

}

sub runConfigurator($$$) {

  my ($host, $port, $mac) = @_;
  
  my $result = "";
  my $response = "";

  my $newmac = "";
  my $i=0;
  for ($i=0; $i<6; $i++) {
    $newmac .= substr($mac,$i*3,2);
  }
  logMessage(1, $mac." -> ".$newmac);

  my $cmd = "/home/apsr/linux_64/bin/ibob_configurator ".$host." ".$port." ".$newmac;
  logMessage(1, $cmd);

  $response = `$cmd 2>&1`;
  my $rval = $?;

  if ($rval == 0) {
    return ("ok", $response);
  } else {
    return ("fail", $rval.":".$response );
  }

}

sub rearmAll() {

  my $cmd = "/home/apsr/linux_64/bin/ibob_rearm_trigger -M";
  logMessage(1, $cmd);

  my $response = `$cmd 2>&1`;
  my $rval = $?;

    if ($rval == 0) {
    return ("ok", $response);
  } else {
    return ("fail", $rval.":".$response );
  }

}


#
# Polls for the "quitdaemons" file in the control dir
#
sub daemonControlThread() {

  logMessage(2, "daemon_control: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    logMessage(3, "daemon_control: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  logMessage(2, "daemon_control: Unlinking PID file ".$pidfile);
  unlink($pidfile);

  logMessage(2, "daemon_control: exiting");

}

#
# Logs a message to the Nexus
#
sub logMessage($$) {
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
  print STDERR basename($0)." : Exiting: ".Dada->getCurrentDadaTime(0)."\n";
  exit(1);

}

