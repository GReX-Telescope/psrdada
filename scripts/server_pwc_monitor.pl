#!/usr/bin/env perl

use IO::Socket;     # Standard perl socket library
use Net::hostent;
use File::Basename;
use Dada;           # DADA Module for configuration options
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;


# Constants
#
use constant DEBUG_LEVEL        => 0;
use constant PIDFILE            => "pwc_monitor.pid";
use constant LOGFILE            => "pwc_monitor.log";

#
# Global Variables
#
our %cfg = Dada->getDadaConfig();      # dada.cfg
our $quit_daemon : shared  = 0;


#
# Local Variables
#
my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
my $daemon_control_thread = 0;
my $result;
my $response;
my $cmd;
my $i = 0;
my $rh;


# Autoflush output
$| = 1;

# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


# Connect to pwc command 
my $handle;
my $quit = 0;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}


# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

logMessage(0, "STARTING SCRIPT: ".Dada->getCurrentDadaTime(0));


# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

my @array;
my $machine = "";
my $rest = "";
my $date = "";
my $msg = "";
my $statusfile_basedir = $cfg{"STATUS_DIR"};
my $statusfile = "";
my $handle = 0;

while (!$quit_daemon) {

  # If we have lost the connection, try to reconnect to the PWCC (nexus)
  while (!$handle && !$quit_daemon) {

    logMessage(2, "Attemping connection to PWCC: ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_PWCC_LOG_PORT"});

    $handle = Dada->connectToMachine($cfg{"SERVER_HOST"},$cfg{"SERVER_PWCC_LOG_PORT"});

    if (!$handle)  {
      sleep(1);
      logMessage(2,"Could not connect to dada_pwc_command on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_PWCC_LOG_PORT"});
    } else {
      logMessage(1, "Connected to PWCC: ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_PWCC_LOG_PORT"});

    }
  }

  while ($handle && !$quit_daemon) {
 
    $result = Dada->getLineSelect($handle,1);

    # If we have lost the connection
    if (! defined $result) { 

      $handle->close();
      $handle = 0;

    } else {

      logMessage(2, "Received line: \"".$result."\"");

      # determine the source machine
      $statusfile = "";

      @array = split(/: \[/,$result);
      $machine = $array[0];
      $rest = $array[1]; 
      $date = substr($rest,0,19);
      $msg = substr($rest,21);

      # If contains a warning message
      if ($msg =~ /WARN: /) {
        $statusfile = $statusfile_basedir."/".$machine.".pwc.warn" ;

      # If contains an error message
      } elsif ($msg =~ /ERR:/) {
       $statusfile = $statusfile_basedir."/".$machine.".pwc.error" ;

      # If we are starting a new obs, delete the error and warn files 
      } elsif ($msg =~ /STATE = prepared/) {

        unlink $statusfile_basedir."/".$machine.".src.warn";
        unlink $statusfile_basedir."/".$machine.".pwc.warn";
        unlink $statusfile_basedir."/".$machine.".sys.warn";
        unlink $statusfile_basedir."/".$machine.".src.error";
        unlink $statusfile_basedir."/".$machine.".pwc.error";
        unlink $statusfile_basedir."/".$machine.".sys.error";
      } else {

        # We ignore the message
      }

      if ($statusfile ne "") {
        if (-f $statusfile) {
          open(FH,">>".$statusfile);
        } else {
          open(FH,">".$statusfile);
        }
        print FH $msg."\n";
        close FH;
        logMessage(1, "Logged: $machine, $date, $msg");
      }
    }
  }

  logMessage(1, "Lost connection with PWCC");
  sleep(1);
}

# Rejoin our daemon control thread
$daemon_control_thread->join();

logMessage(0, "STOPPING SCRIPT: ".Dada->getCurrentDadaTime(0));

exit(0);

###############################################################################
#
# Functions
#


sub daemonControlThread() {

  logMessage(2, "Daemon control thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  logMessage(2, "Unlinking PID file: ".$pidfile);
  unlink($pidfile);

  logMessage(2, "Daemon control thread ending");

}

sub logMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    print $message."\n";
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


