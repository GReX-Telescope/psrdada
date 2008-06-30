#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  6 Dec, 2007
# Modified: 3 Jan, 2008
# 
# This daemons runs continuously and launches dspsr each time
# a full header block is written
#
# 1.  Runs dada_header which will print out the next free header block
# 2.  Greps for the PROCESSING_CMD 
# 3.  Runs the PROCESSING_CMD in a system call, waiting for it to terminate
# 4.  Rinse, Lather, Repeat.


#
# Include Modules
#
use Dada;           # DADA Module for configuration options
use strict;         # strict mode (like -Wall)
use threads;        # standard perl threads
use threads::shared;        # standard perl threads
use File::Basename;
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;


#
# Constants
#
use constant DEBUG_LEVEL        => 1;
use constant DADA_HEADER_BINARY => "dada_header -k eada";
use constant NUM_CPUS           => 8;
use constant PIDFILE            => "processing_manager.pid";
use constant LOGFILE            => "processing_manager.log";


#
# Global Variable Declarations
#
our %cfg : shared = Dada->getDadaConfig();      # dada.cfg in a hash
our $log_socket;
our $dspsr_start_time : shared = 0;   # Flag to indcate if dspsr is running
our $quit_daemon : shared  = 0;


#
# Local Variable Declarations
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
my $prev_utc_start = "";
my $prev_obs_offset = "";
my $quit = 0;
my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
my $daemon_control_thread = "";
my $load_control_thread = "";
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
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SRC_LOG_PORT"});
if (!$log_socket) {
  print "Could not open a connection to the nexus SRC log: $log_socket\n";
}

logMessage(0,"INFO", "STARTING SCRIPT");

if (!(-f $daemon_quit_file)) {

  # This thread will monitor for our daemon quit file
  $daemon_control_thread = threads->new(\&daemon_control_thread, "dada_header");
  #$daemon_control_thread->detach();

  $load_control_thread = threads->new(\&load_control_thread);

  # Main Loop
  while( (!($quit)) && (!(-f $daemon_quit_file)) ) {

    # Run the processing thread once
    ($quit, $prev_utc_start, $prev_obs_offset) = processing_thread($prev_utc_start, $prev_obs_offset);

  }

  logMessage(0, "INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  $load_control_thread->join();
  $daemon_control_thread->join();
  exit(0);

} else {
  logMessage(0,"INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);

  exit(-1);
}


sub processing_thread($$) {

  (my $prev_utc_start, my $prev_obs_offset) = @_;

  my $bindir = Dada->getCurrentBinaryVersion();
  my $dada_header_cmd = $bindir."/".DADA_HEADER_BINARY;
  my $proc_log = $cfg{"CLIENT_LOGS_DIR"}."/dspsr.log";
  my $processing_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $utc_start = "";
  my $obs_offset = "";
  my $proc_cmd_file = "";
  my $proj_id = "";
  my $centre_freq = "";

  my %current_header = ();
  my @lines = ();
  my $line = "";
  my $key = "";
  my $val = "";
  my $raw_header = "";
  my $time_str;
  my $cmd = "";

  # Get the next filled header on the data block. Note that this may very
  # well hang for a long time - until the next header is written...
  logMessage(2, "INFO", "Running cmd \"".$dada_header_cmd."\"");
  my $raw_header = `$dada_header_cmd 2>&1`;

  # since the only way to currently stop this daemon is to send a kill
  # signal to dada_header_cmd, we should check the return value
  if ($? == 0) {

    @lines = split(/\n/,$raw_header);
    foreach $line (@lines) {
      ($key,$val) = split(/ +/,$line,2);
      if ((length($key) > 1) && (length($val) > 1)) {
        # Remove trailing whitespace
        $val =~ s/\s*$//g;
        $current_header{$key} = $val;
      }
    }

    $utc_start     = $current_header{"UTC_START"};
    $obs_offset    = $current_header{"OBS_OFFSET"};
    $proc_cmd_file = $current_header{"PROC_FILE"};
    $centre_freq   = $current_header{"FREQ"};
    $proj_id       = $current_header{"PID"};

    if (length($utc_start) < 5) {
      logMessage(0, "ERROR", "UTC_START was malformed or non existent");
    }
    if (length($obs_offset) < 1) {
      logMessage(0, "ERROR", "Error: OBS_OFFSET was malformed or non existent");
    }
    if (length($proc_cmd_file) < 1) {
      logMessage(0, "ERROR", "PROC_CMD_FILE was malformed or non existent");
    }

    if (($utc_start eq $prev_utc_start) && ($obs_offset eq $prev_obs_offset)) {
      logMessage(0, "ERROR", "The UTC_START and OBS_OFFSET has been repeated");
    }
                                                                                                                
    $time_str = Dada->getCurrentDadaTime();

    $processing_dir .= "/".$utc_start."/".$centre_freq;

    # This should not be requried as the observation manager should be creating
    # this directory for us
    if (! -d ($processing_dir)) {
      logMessage(0, "WARN", "The archive directory was not created by the ".
                 "observation manager: \"".$processing_dir."\"");
      `mkdir -p $processing_dir`;
      `chmod g+s $processing_dir`;
      `chgrp -R $proj_id $processing_dir`;
    }

    # Add the dada header file to the proc_cmd
    $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$proc_cmd_file;
    my %proc_cmd_hash = Dada->readCFGFile($proc_cmd_file);
    my $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

    if ($proc_cmd =~ m/dspsr/) {
      $proc_cmd .= " ".$cfg{"PROCESSING_DB_KEY"};
    }

    # Create an obs.start file in the processing dir:
    logMessage(1, "INFO", "START ".$proc_cmd);

    chdir $processing_dir;
    $dspsr_start_time = time;
    logMessage(2, "INFO", "Setting dspsr_start_time = ".$dspsr_start_time);

    $cmd = $bindir."/".$proc_cmd;
    $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_src_logger.pl";

    logMessage(1, "INFO", "Cmd: ".$cmd);

    my $returnVal = system($cmd);

    if ($returnVal != 0) {
      logMessage(0, "WARN", "Processing command dspsr failed: ".$?." ".$returnVal);
    }
    $time_str = Dada->getCurrentDadaTime();
    logMessage(1, "INFO", "END ".$proc_cmd);

    $dspsr_start_time = 0;
    logMessage(2, "INFO", "Setting dspsr_start_time = ".$dspsr_start_time);


    (return 0, $utc_start, $obs_offset);

  } else {

    logMessage(2, "INFO", "dada_header_cmd failed!, rval = ".$?);
    sleep 1;
    return (0,$utc_start, $obs_offset);

  }
}

sub daemon_control_thread($) {

  (my $cmd_to_kill) = @_;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;

  while ((!(-f $daemon_quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }
  # If the daemon quit file exists, we MUST kill the dada_header process if 
  # it exists. If the processor is still processing, then we would wait for
  # it to finish.

  # It is quit possible that the processing command will be running, and 
  # in this case, the processing thread itself will exit.
  $quit_daemon = 1;
  logMessage(2, "INFO", "control_thread: unlinking PID file");
  unlink($pidfile);


  # Kill the dada_header command
  my $cmd = "killall -KILL ".$cmd_to_kill;
  logMessage(2, "INFO", "Running kill cmd \"".$cmd."\"");
  `$cmd`;

  # Kill the dada_header command
  my $cmd = "killall -KILL dspsr";
  logMessage(2, "INFO", "Running kill cmd \"".$cmd."\"");
  `$cmd`;

}

#
# If dspsr has been running for 1 minute, return the number of free processors
# based on the 1 minute load. 
#
sub load_control_thread($) {

  my $localhost = Dada->getHostMachineName();

  my $server = new IO::Socket::INET (
    LocalHost => $localhost, 
    LocalPort => $cfg{"CLIENT_PROC_LOAD_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($server);   # add the main socket to the set
  my $rh;

  while (!$quit_daemon) {

    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 2);                                                                                 
    foreach $rh (@$readable_handles) {

      logMessage(3, "INFO", "checking a read handle");

      if ($rh == $server) {
                                                                                
        my $handle = $rh->accept();
        $handle->autoflush(1);
        my $hostinfo = gethostbyaddr($handle->peeraddr);
        my $hostname = $hostinfo->name;
        
        logMessage(2, "INFO", "Accepting connection from $hostname");
                                                                                
        # Add this read handle to the set
        $read_set->add($handle);

      } else {

        my $string = Dada->getLine($rh);

        if (! defined $string) {
          logMessage(3, "INFO", "removing read handle");
          logMessage(3, "INFO", "Socket closed");
          $read_set->remove($rh);
          close($rh);
        } else {
          logMessage(2, "INFO", "received string \"".$string."\"");

          my $num_cores_available = 0;

          # find out if we are still taking data
          my $handle2 = Dada->connectToMachine($localhost, $cfg{"CLIENT_BG_PROC_PORT"});

          # Assume we are taking data
          my $taking_data = 1;

          if ($handle2) {

            logMessage(2, "INFO", "Connection to client_observation_manager.pl established");

            print $handle2 "is data currently being received?\r\n";
            $taking_data = Dada->getLine($handle2);
            logMessage(2, "INFO", "client_observation_manager.pl returned ".$taking_data);
            $handle2->close();

          } else {
            # we assume that we are not taking data
            logMessage(0, "WARN", "Could not connect to client_observation_manager.pl on ".$localhost.":".$cfg{"CLIENT_BG_PROC_PORT"});
          }

          if ($taking_data) {
            logMessage(2, "INFO", "Currently taking data (".$taking_data.")");

            # if dspsr has been running for more than 1 minute...
            if (($dspsr_start_time) && (($dspsr_start_time+120) < time)) {

              logMessage(2, "INFO", "dspsr running for more than 2 minutes");
              my $result = "";
              my $response = "";
    
              ($result, $response) = Dada->getLoadInfo();
              my @loads = split(/,/, $response);
              my $one_minute_load = int($loads[0]);

              # always keep 1 core free
              $num_cores_available = ($cfg{"CLIENT_NUM_CORES"} - (1+$one_minute_load));

            # We have not been running for a minute, 0 cores available
            } else {
              logMessage(2, "INFO", "dspsr HAS NOT BEEN running for more than 2 minutes");
              $num_cores_available = 0;
            }
                                                                                                                                                                                           
          # we are not currently taking data, but check if dspsr is running
          } else {

            logMessage(2, "INFO", "Not taking data");

            if ($dspsr_start_time) {

              logMessage(2, "INFO", "dspsr still runing");

              my $result = "";
              my $response = "";
                                                                                                                                                                                           
              ($result, $response) = Dada->getLoadInfo();
              my @loads = split(/,/, $response);
              my $one_minute_load = int($loads[0]);
                                                                                                                                                                                           
              # always keep 1 core free
              $num_cores_available = ($cfg{"CLIENT_NUM_CORES"}- (1+$one_minute_load));


            } else {
              logMessage(2, "INFO", "dspsr not running");
              $num_cores_available = $cfg{"CLIENT_NUM_CORES"};
            }
          }
                                                                                                                                                                                           
          logMessage(2, "INFO", "Currently ".$num_cores_available." cores available");
          print $rh $num_cores_available."\r\n";
      
        }
      }

    }
  }
}

#
# Logs a message to the Nexus
#
sub logMessage($$$) {
  (my $level, my $type, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    if (!($log_socket)) {
      $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SRC_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "src", $type, "proc mngr", $message);
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

  print STDERR basename($0)." : Exiting\n";

  if ($log_socket) {
    close($log_socket);
  }

  exit 1;

}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $log_socket = 0;
  $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SRC_LOG_PORT"});

}


