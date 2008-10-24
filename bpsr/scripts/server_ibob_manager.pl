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
use constant DEBUG_LEVEL  => 2;
use constant PIDFILE      => "ibob_manager.pid";
use constant LOGFILE      => "ibob_manager.log";
use constant MULTIBOB_BIN => "multibob_server";


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
my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

my $daemon_control_thread = 0;

my $uname = $cfg{"IBOB_GATEWAY_USERNAME"};
my $host  = $cfg{"IBOB_GATEWAY"};
my $port  = $cfg{"IBOB_MANAGER_PORT"};
my $npwc  = $cfg{"NUM_PWC"};

my $i;
my $result;
my $response;
my $command = "config";

#
# Main
#

Dada->daemonize($logfile, $pidfile);

logMessage(0, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

my $cmd = "ssh -n -l ".$uname." ".$host." \"".MULTIBOB_BIN." -n ".$npwc." -p ".$port."\" 2>&1";
logMessage(1, $cmd);
system($cmd);
#my $result = `$cmd`;
logMessage(1, "ssh cmd returned");
$quit_daemon = 1;
#chomp $result;
#logMessage(1, $result);

# rejoin threads
$daemon_control_thread->join();

logMessage(0, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#


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
  my $handle = Dada->connectToMachine($cfg{"IBOB_GATEWAY"}, $cfg{"IBOB_MANAGER_PORT"},1);
  if (!$handle) {
    logMessage(0, "daemon_control: could not connect to ".MULTIBOB_BIN);
  } else {
    my ($result, $response) = Dada->sendTelnetCommand($handle, "quit");
    logMessage(0, "daemon_control: ".$result.":".$response);
    close($handle);
  }

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

