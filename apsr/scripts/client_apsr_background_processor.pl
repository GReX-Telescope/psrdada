#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  1 Feb 2008
# Modified: 1 Feb 2008
# 
# This daemons runs continuously and launches dspsr on raw data files that
# have been written to disk. It will process 1 file at a time only use CPU
# such that there is always 1 core left spare


#
# Include Modules
#
use strict;         # strict mode (like -Wall)
use threads;        # standard perl threads
use threads::shared;
use File::Basename;
use Apsr;           # APSR/DADA Module for configuration options



#
# Constants
#
use constant DEBUG_LEVEL   => 1;
use constant PIDFILE       => "apsr_background_processor.pid";
use constant LOGFILE       => "apsr_background_processor.log";


#
# Global Variable Declarations
#
our %cfg : shared = Apsr->getApsrConfig();      # Apsr.cfg in a hash
our $log_socket;
our $proc_pid : shared = 0;
our $quit_daemon : shared = 0;

#
# Local Variable Declarations
#
my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
my $raw_data_dir = $cfg{"CLIENT_RECORDING_DIR"};
my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
my $localhost = Dada->getHostMachineName();

my $daemon_control_thread = "";
my $result = "";
my $response = "";
my @loads = ();
my $file_to_process;
my $raw_header;
my $one_minute_load;
my $num_cores_available;


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

logMessage(0,"INFO", "STARTING SCRIPT");

if (!(-f $daemon_quit_file)) {

  # This thread will monitor for our daemon quit file
  $daemon_control_thread = threads->new(\&daemon_control_thread);
  #$daemon_control_thread->detach();

  # Main Loop
  while (!($quit_daemon)) {

    ($file_to_process, $raw_header) = getUnprocessedFile($raw_data_dir);

    # If we have file(s) to process
    if ($file_to_process ne "none") {

      logMessage(2, "INFO", "file to process = ".$file_to_process);

      # Ask the processing daemon how many cores are currently available to 
      # process on

      my $handle = Dada->connectToMachine($localhost, $cfg{"CLIENT_PROC_LOAD_PORT"});

      if ($handle) {
    
        logMessage(2, "INFO", "Connection to client_processing_manager.pl established");

        print $handle "how many cores are available?\r\n";
        $response = Dada->getLine($handle);

        logMessage(2, "INFO", "asked for cores available = ".$response);
        $num_cores_available = $response;
        $handle->close();
      } else {
        logMessage(2, "WARN", "Could not connect to processing manager ".$localhost.":".$cfg{"CLIENT_PROC_LOAD_PORT"});
        $num_cores_available = 0;
      }

      logMessage(2, "INFO", "number of available cores =  ".$num_cores_available);

      if ($num_cores_available > 1) {

        my @lines = ();
        my $line;
        my $key;
        my $val; 
        my $proc_cmd = "dspsr -t ".$num_cores_available." ".$file_to_process;
        my $proc_file = "dspsr.default";
        my $utc_start = "unknown";
        my $centre_freq = "unknown";
        my $mode = "PSR";

        @lines = split(/\n/,$raw_header);

        foreach $line (@lines) {
          ($key,$val) = split(/ +/,$line,2);
          if ((length($key) > 1) && (length($val) > 1)) {
            # Remove trailing whitespace
            $val =~ s/\s*$//g;
            if ($key eq "PROC_FILE") {
              $proc_file = $val;
            }
            if ($key eq "UTC_START") {
              $utc_start = $val;
            }
            if ($key eq "FREQ") {
              $centre_freq = $val;
            }

            if ($key eq "MODE") {
              $mode = $val;
            }
          }
        }

        # If running a CAL do it on 1 thread only.
        #if ($mode eq "CAL") {
        #  $num_cores_available = 1;
        #}

        # Add the dada header file to the proc_cmd
        $proc_file = $cfg{"CONFIG_DIR"}."/".$proc_file;

        my %proc_cmd_hash = Dada->readCFGFile($proc_file);
        my $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

        logMessage(2, "INFO", "orig proc_cmd = ".$proc_cmd);
       
        # If it has been setup with threads
        if ($proc_cmd =~ m/-t \d/) {
 
          # Adjust the number of threads to be used by trhe proc cmd
          $proc_cmd =~ s/-t \d/-t $num_cores_available/;

        } else {
          $proc_cmd .= " -t ".$num_cores_available;
        }

        $proc_cmd .= " ".$file_to_process;

        logMessage(2, "INFO", "used proc_cmd = ".$proc_cmd);

        $archive_dir .= "/".$utc_start."/".$centre_freq;

        logMessage(1, "INFO", "Processing ".basename($file_to_process)." on ".$num_cores_available." cores");

        $result = processOneFile($archive_dir, $proc_cmd);

        # reset the client archive dir
        $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
        chdir $archive_dir;
  
        if ($result eq "ok") {
          logMessage(0, "INFO", $file_to_process." successfully processed");

          #my $cmd = "mv ".$file_to_process." ".$cfg{"CLIENT_SCRATCH_DIR"};
          #`$cmd`;
          unlink($file_to_process);
        }

      } else {
        logMessage(2, "INFO", "File awaiting processing: ".$file_to_process);
      }

    }
    sleep(1);
  }

  logMessage(0, "INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  $daemon_control_thread->join();
  exit(0);

} else {

  logMessage(0,"INFO", "STOPPING SCRIPT");
  Dada->nexusLogClose($log_socket);
  exit(-1);

}


sub getUnprocessedFile($) {

  (my $raw_data_dir) = @_;

  my $cmd = "find ".$raw_data_dir." -type f -name *.dada | sort";
  logMessage(2, "INFO", $cmd);
  my $find_result = `$cmd`;

  my @files = split(/\n/, $find_result);

  if ($#files >= 0) {

    $cmd = "head -c 4096 ".$files[0];
    my $raw_header = `$cmd`;

    return ($files[0], $raw_header);

  } else {
    return ("none", "");
  }

}

sub processOneFile($$) {

  (my $dir, my $cmd) = @_;

  my $result = "fail";
  my $pid = fork;

  # If for some reason the directory does not exist, create it
  if (!( -d $dir)) {
    logMessage(0, "WARN", "Output dir ".$dir." did not exist, creating it");
    `mkdir -p $dir`;
  }

  logMessage(2, "INFO", "processing in dir ".$dir);
 
  if ($pid) {

    $proc_pid = $pid;

    # Wait for the child to finish
    waitpid($pid,0);

    if ($? == 0) {
      $result = "ok";
    }
    
    # Reset the processing pid to 0
    $proc_pid = 0;

  } else {
 
    # child process. The exec command will never return 
    chdir $dir;
    logMessage(1, "INFO", $cmd);
    exec "$cmd";
  }

  logMessage(2, "INFO", "Result of processing was \"".$result."\"");
  return $result;

}


sub daemon_control_thread() {

  logMessage(2, "INFO", "control_thread: starting");

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;

  while ((!$quit_daemon) && (!(-f $daemon_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  logMessage(2, "INFO", "control_thread: unlinking PID file");
  unlink($pidfile);

  # If the daemon quit file exists, we MUST kill the dada_header process if 
  # it exists. If the processor is still processing, then we would wait for
  # it to finish.

  # It is quit possible that the processing command will be running, and 
  # in this case, the processing thread itself will exit.
  if ($proc_pid) {
    my $cmd = "kill -KILL ".$proc_pid;
    logMessage(1, "INFO", "control_thread: running ".$cmd);
    my $result = `$cmd`;
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
      $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "bg mngr", $message);
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

  if ($proc_pid) {
    `kill -KILL $proc_pid`;
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


