#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  24 Jan 2008
# Modified: 24 Jan 2008
# 
# This daemons runs on assisting nodes and will receive data via dada_nicdb,
# process it and return the results to the server
#


#
# Include Modules
#
use Dada;             # DADA Module for configuration options
use strict;           # strict mode (like -Wall)
use threads;          # standard perl threads
use threads::shared;  # Allow shared variables for threads
use File::Basename;


#
# Constants
#
use constant DEBUG_LEVEL        => 1;
use constant DADA_NICDB_BINARY  => "dada_nicdb";
use constant PIDFILE            => "auxiliary_processor.pid";
use constant LOGFILE            => "auxiliary_processor.log";


#
# Global Variable Declarations
#
our $log_socket;
our %cfg : shared = Dada->getDadaConfig();      # dada.cfg in a hash
our $quit_daemon : shared = 0;


#
# Local Variable Declarations
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
my $prev_utc_start = "";
my $prev_obs_offset = "";
my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
my $daemon_control_thread = "";
my $proc_thread = "";


#
# Register Signal handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# Turn the script into a daemon
Dada->daemonize($logfile, $pidfile);

# Auto flush output
$| = 1;

# Open a connection to the nexus logging facility
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
if (!$log_socket) {
  print "Could not open a connection to the nexus SYS log: $log_socket\n";
}

logMessage(0, "INFO", "STARTING SCRIPT");

if (!(-f $daemon_quit_file)) {

  # This thread will monitor for our daemon quit file
  $daemon_control_thread = threads->new(\&daemon_control_thread, "dada_nicdb");
  #$daemon_control_thread->detach();

  # Main Loop
  while(!$quit_daemon) {

    # Run the processing thread once
    processing_thread();
    sleep(1);

  }

  logMessage(0, "INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  $daemon_control_thread->join();

  exit(0);

} else {

  logMessage(0,"INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  exit(1);

}


sub processing_thread() {

  my $bindir = Dada->getCurrentBinaryVersion();
  my $dada_nicdb_cmd = $bindir."/".DADA_NICDB_BINARY." -s -k eada -p 40000";
  my $processing_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $raw_data_dir = $cfg{"CLIENT_RECORDING_DIR"};
  my $utc_start = "";
  my $obs_offset = "";
  my $proc_cmd = "";
  my $proj_id = "";

  my %current_header = ();
  my @lines = ();
  my $line = "";
  my $key = "";
  my $val = "";
  my $raw_header = "";
  my $time_str;
  my $cmd = "";


  # Advertise to the server that we are available
  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"SERVER_AUX_ASSIST_PORT"};

  my $handle = Dada->connectToMachine($host, $port);
 
  if (!$handle) {
                                                                                                       
    logMessage(2, "ERROR", "Could not connect to ".$host.":".$port); 

  # We have connected to the
  } else {
    logMessage(2, "INFO", "Connected to ".$host.":".$port." offering services");
    print $handle "apsr17:40000\r\n";

    my $response = "null";
    my $break = 0;
    while ((!$break) && (!$quit_daemon)) {
      $response = Dada->getLineSelect($handle, 1);
      if ($response ne "null") {
        $break = 1;
      } 
    }

    # If the server has acknowledged our offer
    if ($response eq "ack") {

      close($handle);

      # Run dada_nicdb and

      logMessage(0, "INFO", "START $dada_nicdb_cmd");

      my $response = `$dada_nicdb_cmd`;
    
      if ($? != 0) {
        logMessage(0, "ERROR", "Aux cmd \"".$dada_nicdb_cmd."\" failed \"".$response."\"");
      }

      logMessage(0, "INFO", "END $dada_nicdb_cmd");
    } else {
      logMessage(0, "ERROR", "server_aux_manager did not ack our offer of help");
      close($handle);
    }
  }

}

sub daemon_control_thread($) {

  (my $cmd_to_kill) = @_;

  logMessage(2, "INFO", "control_thread: kill command ".$cmd_to_kill);

  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});

  while ((!(-f $daemon_quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }
  # If the daemon quit file exists, we MUST kill the dada_header process if 
  # it exists. If the processor is still processing, then we would wait for
  # it to finish.
  $quit_daemon = 1;

  logMessage(2, "INFO", "control_thread: unlinking PID file");
  unlink($pidfile);

  # It is quit possible that the processing command will be running, and 
  # in this case, the processing thread itself will exit.
  my $cmd = "killall -KILL ".$cmd_to_kill;

  logMessage(2, "INFO", "control_thread: running $cmd");
  `$cmd`;

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
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "aux mngr", $message);
    }
    print "[".$time."] ".$message."\n";
  }
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


