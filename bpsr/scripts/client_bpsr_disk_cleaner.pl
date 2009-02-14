#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  28 Jan 2009
# 
# This daemons deletes observations slowly from the local disk that 
# have been written to tape in all required locations
#


use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Bpsr;            # DADA Module for configuration options
use strict;          # strict mode (like -Wall)
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use Net::hostent;
use File::Basename;


#
# Constants
#
use constant DEBUG_LEVEL        => 1;
use constant PIDFILE            => "bpsr_disk_cleaner.pid";
use constant LOGFILE            => "bpsr_disk_cleaner.log";


#
# Global Variable Declarations
#
our $log_socket;
our $currently_processing : shared = 0;
our $quit_daemon : shared = 0;
our %cfg : shared = Bpsr->getBpsrConfig();	# dada.cfg in a hash

#
# Local Variable Declarations
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
my $daemon_control_thread = "";
my $result = "";
my $response = "";
my $obs = "";
my $beam = "";

#
# Register Signal handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

# Auto flush output
$| = 1;

# Open a connection to the server_sys_monitor.pl script
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
if (!$log_socket) {
  print STDERR "Could connect to log interface".$cfg{"SERVER_HOST"}.
               ":".$cfg{"SERVER_SYS_LOG_PORT"}."\n";
}

logMessage(1,"INFO", "STARTING SCRIPT");

if (! -f $daemon_quit_file ) {

  # This thread will monitor for our daemon quit file
  $daemon_control_thread = threads->new(\&daemon_control_thread);

  # Main Loop
  while ((!$quit_daemon) && (! -f $daemon_quit_file) ) {

    ($result, $response, $obs, $beam) = findCompletedBeam($cfg{"CLIENT_ARCHIVE_DIR"});

    if (($result eq "ok") && ($obs ne "none")) {
      logMessage(1, "INFO", "Deleting ".$obs."/".$beam);
      ($result, $response) = deleteCompletedBeam($cfg{"CLIENT_ARCHIVE_DIR"}, $obs, $beam);
    } else {
      logMessage(1, "INFO", "Found no beams to delete ".$obs."/".$beam);
    }

    logMessage(2, "INFO", "Sleeping 60 seconds");
    sleep(60);

  }

  logMessage(0, "INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  $daemon_control_thread->join();
  exit(0);

} else {

  logMessage(0,"INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  $daemon_control_thread->join();
  exit(1);

}


#
# Find an obs/beam that has the required on.tape.* flags set
#
sub findCompletedBeam($) {

  (my $archives_dir) = @_;

  logMessage(2, "INFO", "findCompletedBeam(".$archives_dir.")");
 
  my $found_obs = 0;
  my $result = "";
  my $response = ""; 
  my $obs = "none";
  my $beam = "none";
  my $cmd = "";
  my $i = 0;
  my $o = "";   # obs
  my $b = "";   # beam
  my $source = "";  # source
  my $s = "";       # first letter of source
  my $file = "";

  # Ensure this is NFS mounted
  $cmd = "ls ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}." >& /dev/null";
  ($result, $response) = Dada->mySystem($cmd);

  # Look for observations that are marked with obs.transferred
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}." -maxdepth 2 -name 'obs.transferred'".
         " -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  logMessage(3, "INFO", $cmd);
  ($result, $response) = Dada->mySystem($cmd);
  logMessage(3, "INFO", $result." ".$response);

  if ($result ne "ok") {
    logMessage(0, "WARN", "find command failed: ".$response);
    return ("fail", "find command failed: ".$response, $obs, $beam);
  }

  chomp $response;
  my @observations = split(/\n/, $response);

  for ($i=0; ((!$quit_daemon) && (!$found_obs) && ($i<$#observations)); $i++) {

    $o = $observations[$i];

    # Get the beam subdir
    $cmd = "ls -1d ".$archives_dir."/".$o."/?? | awk -F/ '{print \$NF}'";
    logMessage(3, "INFO", $cmd);
    ($result, $response) = Dada->mySystem($cmd);
    logMessage(3, "INFO", $result." ".$response);
    if ($result ne "ok") {
      logMessage(0, "WARN", "findCompletedBeam ".$o." could not find the beam subdir");
      next;
    }
    $b = $response;
    chomp $b;

    if (-f $archives_dir."/".$o."/".$b."/obs.deleted") {
      logMessage(0, "WARN", "Skipping ".$o."/".$b." obs.deleted existed");
      next;
    }

    # Get the type of the observation (based on SOURCE name)
    $file = $archives_dir."/".$o."/".$b."/obs.start";
    if (-f $file) {

      #$cmd = "grep SOURCE ".$file." | awk '{printf(\"\%c\",substr(\$2, 0, 1))}'";
      $cmd = "grep SOURCE ".$file." | awk '{print \$2}'";
      logMessage(3, "INFO", $cmd);
      ($result, $response) = Dada->mySystem($cmd);
      logMessage(3, "INFO", $result." ".$response);

      if ($result ne "ok") {
        logMessage(0, "WARN", "findCompletedBeam could not extract SOURCE from obs.start");
        next;
      }

      $source = $response;
      chomp $source; 
      $s = substr($source, 0, 1);
      logMessage(3, "INFO", "source = ".$source." [".$s."]");
      
      $found_obs = 1;
      if (! -f $archives_dir."/".$o."/".$b."/on.tape.swin" ) {
        logMessage(2, "INFO", $o."/".$b." [".$source."] on.tape.swin missing");
        $found_obs = 0;
      }

      if (($found_obs) && ($s eq "G") && (! -f $archives_dir."/".$o."/".$b."/on.tape.parkes")) {
        logMessage(2, "INFO", $o."/".$b." [".$source."] on.tape.parkes missing");
        $found_obs = 0;
      }

      if ($found_obs) {
        logMessage(2, "INFO", $o."/".$b." is ready to be deleted");
        $obs = $o;
        $beam = $b;
      }

    } else {
      logMessage(0, "WARN", "findCompletedBeam ".$file." did not exist");
      next;
    }

  }

  $result = "ok";
  $response = "";

  logMessage(2, "INFO", "findCompletedBeam ".$result.", ".$response.", ".$obs.", ".$beam);

  return ($result, $response, $obs, $beam);
}


#
# Delete the specified obs/beam 
#
sub deleteCompletedBeam($$) {

  my ($dir, $obs, $beam) = @_;

  my $result = "";
  my $response = ""; 
  my $path = $dir."/".$obs."/".$beam;

  logMessage(2, "INFO", "Deleting archived files in ".$path);  

  my $cmd = "find ".$path." -name '*.fil' -o -name 'aux*' -o -name '*.ar' -o -name '*.png'";

  logMessage(2, "INFO", $cmd);
  ($result, $response) = Dada->mySystem($cmd);
  logMessage(2, "INFO", $result." ".$response);

  if ($result ne "ok") {
    logMessage(0, "ERROR", "deleteCompletedBeam: find command failed: ".$response);
    return ("fail", "find command failed");
  }

  chomp $response;
  my $files = $response;

  $files =~ s/\n/ /;
  $cmd = "slow_rm ".$files;

  logMessage(2, "INFO", $cmd);
  ($result, $response) = Dada->mySystem($cmd);
  logMessage(2, "INFO", $result." ".$response);

  $cmd = "touch ".$path."/obs.deleted";
  logMessage(2, "INFO", $cmd);
  ($result, $response) = Dada->mySystem($cmd);
  logMessage(2, "INFO", $result." ".$response);

  $result = "ok";
  $response = "";

  return ($result, $response);
}


#
# Monitor for the existence of the quitfile
#
sub daemon_control_thread() {

  logMessage(2, "INFO", "control_thread: starting");

  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});

  while ((!(-f $daemon_quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  logMessage(2, "INFO", "control_thread: unlinking PID file");
  unlink($pidfile);

}


#
# Logs a message to the Nexus
#
sub logMessage($$$) {
  (my $level, my $type, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    if (!($log_socket)) {
      $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "disk cleaner", $message);
    }
    print "[".$time."] ".$message."\n";
  }
}

sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";

  # Tell threads to try and quit
  $quit_daemon= 1;
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

