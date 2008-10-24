#!/usr/bin/env perl

###############################################################################
#
# client_bpsr_results_monitor.pl 
#
# This script transfers the output of the_decimator to the results directory
# on the server

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Bpsr;               # BPSR/DADA Module for configuration options
use strict;             # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Constants
#
use constant  DEBUG_LEVEL => 1;
use constant  SLEEPTIME   => 1;
use constant  PIDFILE     => "bpsr_results_monitor.pid";
use constant  LOGFILE     => "bpsr_results_monitor.log";


#
# Global Variables
#
our $log_socket;
our $quit_daemon : shared  = 0;
our %cfg : shared = Bpsr->getBpsrConfig();      # dada.cfg in a hash


#
# Local Variables
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
my $client_archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});

# Autoflush STDOUT
$| = 1;

my $cmd;
my %opts;
getopts('h', \%opts);

if ($#ARGV!=-1) {
    usage();
    exit;
}

if ($opts{h}) {
  usage();
  exit;
}

if (!(-d($client_archive_dir))) {
  print "Error: client archive directory \"".$client_archive_dir."\" did not exist\n";
  exit(-1);
}


# Register signal handlers
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

# Open a connection to the nexus logging facility
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
if (!$log_socket) {
  print "Could not open a connection to the nexus SYS log: $log_socket\n";
}

logMessage(0,"INFO", "STARTING SCRIPT");

# Change to the local archive directory
chdir $client_archive_dir;

my $dir;


#
# Main Control Loop
#

my $daemon_control_thread = threads->new(\&daemonControlThread);

my @lines;
my $line;
my $find_result;
my $result;
my $response;

# Loop until daemon control thread asks us to quit
while (!($quit_daemon)) {

  @lines = ();

  #TODO new code
  $cmd = "find . -maxdepth 3 -regex \".*[ts|bp|bps][0|1]\" | sort";
  $find_result = `$cmd`;

  @lines = split(/\n/,$find_result);

  foreach $line (@lines) {

    $line = substr($line,2);

    logMessage(1, "INFO", "Processing decimator mon file \"".$line."\"");

    my ($utc, $beam, $file) = split( /\//, $line);

    my $file_dir = $utc."/".$beam;

    ($result, $response) = sendToServerViaNFS($file_dir."/".$file, $cfg{"SERVER_RESULTS_NFS_MNT"}, $file_dir);

    if ($result ne "ok") {
      logMessage(0, "ERROR", "NFS Copy Failed: ".$response);
      unlink($file_dir."/".$file);

    } else {

      my $aux_dir = $file_dir."/aux";

      if (! -d $aux_dir) {
        logMessage(0, "WARN", "Aux file dir for ".$utc.", beam ".$beam." did not exist, creating...");
        `mkdir -p $aux_dir`;
      }

      my $cmd  = "mv ".$file_dir."/".$file." ".$file_dir."/aux/";
      ($result, $response) = Dada->mySystem($cmd,0);
      if ($result ne "ok") {
        logMessage(0, "ERROR", "Could not move file ($file) to aux dir \"".$response."\"");
        unlink($file_dir."/".$file);
      } 
    }
  }

  if ($#lines == -1) {
    sleep(SLEEPTIME);
  }

}

# Rejoin our daemon control thread
$daemon_control_thread->join();

logMessage(0, "INFO", "STOPPING SCRIPT");

# Close the nexus logging connection
Dada->nexusLogClose($log_socket);


exit (0);

#
# Functions
#

sub sendToServerViaNFS($$$) {

  (my $file, my $nfsdir,  my $dir) = @_;

  my $result = "";
  my $response = "";

  # If the nfs dir isn't automounted, ensure it is
  if (! -d $nfsdir) {
    `ls $nfsdir >& /dev/null`;
  }

  my $cmd = "cp ".$file." ".$nfsdir."/".$dir."/";
  logMessage(2, "INFO", "NFS copy \"".$cmd."\"");
  ($result, $response) = Dada->mySystem($cmd,0);
  if ($result ne "ok") {
    return ("fail", "Command was \"".$cmd."\" and response was \"".$response."\"");
  } else {
    return "ok";
  }

}
 

sub usage() {
  print "Usage: ".basename($0)." -h\n";
  print "  -h          print this help text\n";

}

sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";

  # Tell threads to try and quit
  $quit_daemon = 1;
  sleep(3);

  if ($log_socket) {
    close($log_socket);
  }

  print STDERR basename($0)." : Exiting\n";
  exit 1;

}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $log_socket = 0;  
  $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});

}

#

# Logs a message to the Nexus
#
sub logMessage($$$) {
  (my $level, my $type, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    if (!($log_socket)) {
      $log_socket =  Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "results mon", $message);
    }
    print "[".$time."] ".$message."\n";
  }
}

sub daemonControlThread() {

  logMessage(2, "INFO", "control_thread: starting");

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;

  # Poll for the existence of the control file
  while ((!(-f $daemon_quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  logMessage(2, "INFO", "control_thread: unlinking PID file");
  unlink($pidfile);

}

