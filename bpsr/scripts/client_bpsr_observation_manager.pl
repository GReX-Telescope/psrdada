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


use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Bpsr;            # DADA Module for configuration options
use strict;          # strict mode (like -Wall)
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;


#
# Constants
#
use constant DEBUG_LEVEL        => 1;
use constant DADA_HEADER_BINARY => "dada_header -k deda";
use constant RSYNC_OPTS         => "-ag --rsh=/usr/bin/rsh";
use constant PIDFILE            => "bpsr_observation_manager.pid";
use constant LOGFILE            => "bpsr_observation_manager.log";


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
my $prev_header = "";
my $quit = 0;
my $daemon_control_thread = "";
my $allow_background_processing_thread = "";
my $proc_thread = "";
my $quit = 0;

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
  print "Could not open a connection to the nexus SYS log: $log_socket\n";
}

logMessage(1,"INFO", "STARTING SCRIPT");

if (! -f $daemon_quit_file ) {


  # This thread will monitor for our daemon quit file
  $daemon_control_thread = threads->new(\&daemon_control_thread, "dada_header");

  # Main Loop
  
  while ((!$quit) && (! -f $daemon_quit_file) ) {

    # Run the processing thread once
    ($quit, $prev_header) = processing_thread($prev_header);

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
# Processes a single observation
#
# Creates the required directories for the output data on the server, and generates the obs.start file
#
sub processing_thread($) {

  (my $prev_header) = @_;

  my $bindir = Dada->getCurrentBinaryVersion();
  my $dada_header_cmd = $bindir."/".DADA_HEADER_BINARY;
  my $processing_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $utc_start = "";
  my $acc_len = "";
  my $obs_offset = "";
  my $proc_cmd = "";
  my $proc_cmd_file = "";
  my $proj_id = "";
  my $beam = "";

  my %h = ();
  my @lines = ();
  my $line = "";
  my $key = "";
  my $val = "";
  my $raw_header = "";
  my $time_str;
  my $cmd = "";

  chdir $processing_dir;

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
        $h{$key} = $val;
      }
    }

    my $header_ok = 1;

    if (length($h{"UTC_START"}) < 5) {
      logMessage(0, "ERROR", "UTC_START was malformed or non existent");
      $header_ok = 0;
    }
    if (length($h{"OBS_OFFSET"}) < 1) {
      logMessage(0, "ERROR", "Error: OBS_OFFSET was malformed or non existent");
      $header_ok = 0;
    }
    if (length($h{"PROC_FILE"}) < 1) {
      logMessage(0, "ERROR", "PROC_FILE was malformed or non existent");
      $header_ok = 0;
    }

    # command line that will be run
    my $proc_cmd = "";

    # Test for a repeated header
    if ($raw_header eq $prev_header) {
      logMessage(0, "ERROR", "DADA header repeated, likely cause failed PROC_CMD, jettesioning xfer");
      $proc_cmd = "dada_dbnull -s -k ".lc($cfg{"PROCESSING_DATA_BLOCK"});

    # Or if malformed
    } elsif (! $header_ok) {
      logMessage(0, "ERROR", "DADA header malformed, jettesioning xfer");
      $proc_cmd = "dada_dbnull -s -k ".lc($cfg{"PROCESSING_DATA_BLOCK"});

    } else {

      my $obs_start_file = createLocalDirectories($h{"UTC_START"}, $h{"BEAM"}, $h{"PID"}, $raw_header);

      createRemoteDirectories($h{"UTC_START"}, $h{"BEAM"}, $h{"PID"});

      copyUTC_STARTfile($h{"UTC_START"}, $h{"BEAM"}, $obs_start_file);

      # So that the background manager knows we are processing
      $currently_processing = 1;

      $time_str = Dada->getCurrentDadaTime();

      $processing_dir .= "/".$h{"UTC_START"}."/".$h{"BEAM"};

      # Add the dada header file to the proc_cmd
      my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"PROC_FILE"};
      my %proc_cmd_hash = Dada->readCFGFile($proc_cmd_file);
      $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

      logMessage(2, "INFO", "Initial PROC_CMD: ".$proc_cmd);

      # Normal processing via the_decimator
      if ($proc_cmd =~ m/the_decimator/) {
        $proc_cmd .= " -o ".$h{"UTC_START"};
        $proc_cmd .= " ".$cfg{"PROCESSING_DB_KEY"};
      }

      # Special case for folding beam01 only
      if ($proc_cmd =~ m/dspsr/) {
        $proc_cmd .= " ".$cfg{"PROCESSING_DB_KEY"};
      }

      logMessage(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
    }

    logMessage(1, "INFO", "START ".$proc_cmd);
    logMessage(2, "INFO", "Changing dir to $processing_dir");
    chdir $processing_dir;

    $cmd = $bindir."/".$proc_cmd;
    $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl";

    logMessage(2, "INFO", "cmd = $cmd");

    my $return_value = system($cmd);
   
    if ($return_value != 0) {
      logMessage(0, "ERROR", $proc_cmd." failed: ".$?." ".$return_value);
    }

    $time_str = Dada->getCurrentDadaTime();

    # So that the background manager knows we have stopped processing
    $currently_processing = 0;

    chdir "../../";
    logMessage(1, "INFO", "END ".$proc_cmd);;

    return (0, $raw_header);

  } else {

    logMessage(2, "WARN", "dada_header_cmd failed - probably no data block");
    sleep 1;
    return (0, "");

  }
}


sub daemon_control_thread($) {

  (my $cmd_to_kill) = @_;

  logMessage(2, "INFO", "control_thread: starting");

  my $pidfile = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});

  while ((!(-f $daemon_quit_file)) && (!$quit_daemon)) {
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
  my $cmd = "killall -KILL ".$cmd_to_kill;
  logMessage(1, "INFO", "control_thread: running ".$cmd);
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
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "obs mngr", $message);
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

#
# Create the remote directories required for this observation
#

sub createRemoteDirectories($$$) {

  my ($utc_start, $beam, $proj_id) = @_;
  
  my $localhost = Dada->getHostMachineName();
  my $remote_archive_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $remote_results_dir = $cfg{"SERVER_RESULTS_NFS_MNT"};
  my $cmd = "";
  my $dir = "";

  # Wait for remote results directory to be created...

  # Ensure each directory is automounted
  if (!( -d  $remote_archive_dir)) {
    `ls $remote_archive_dir >& /dev/null`;
  }
  if (!( -d  $remote_results_dir)) {
    `ls $remote_results_dir >& /dev/null`;
  }

  # Create the nfs soft link to the local archives directory 
  chdir $remote_archive_dir."/".$utc_start;
  $cmd = "ln -s /nfs/".$localhost."/bpsr/archives/".$utc_start."/".$beam." .";
  logMessage(2, "INFO", $cmd);
  system($cmd);
  
  # Create the remote nfs directory
  $cmd = "mkdir -p ".$remote_results_dir."/".$utc_start."/".$beam;
  logMessage(2, "INFO", $cmd);
  system($cmd);

  # Adjust permission on remote results directory
  $dir = $remote_results_dir."/".$utc_start;
  $cmd = "chgrp -R ".$proj_id." ".$dir;
  logMessage(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    logMessage(0, "WARN", "Failed to chgrp remote results dir \"".
               $dir."/".$utc_start."\" to \"".$proj_id."\"");
  }

  $cmd = "chmod -R g+s ".$dir;
  logMessage(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    logMessage(0, "WARN", "Failed to chmod remote results dir \"".
               $dir."\" to \"".$proj_id."\"");
  }

}

  
#
# Create the local directories required for this observation
#
sub createLocalDirectories($$$$) {

  (my $utc_start, my $beam, my $proj_id, my $raw_header) = @_;

  my $local_archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $dir = "";

  # Create local archive directory
  $dir = $local_archive_dir."/".$utc_start."/".$beam;
  logMessage(2, "INFO", "Creating local output dir \"".$dir."\"");
  `mkdir -p $dir`;
  if ($? != 0) {
    logMessage(0,"ERROR", "Could not create local archive dir \"".
               $dir."\"");
  }

  $dir .= "/aux";
  `mkdir -p $dir`;
  if ($? != 0) {
    logMessage(0, "WARN", "Failed to create local aux dir \"".$dir."\"");
  }

  # Set GID on local archive dir
  $dir = $local_archive_dir."/".$utc_start;
  `chgrp -R $proj_id $dir`;
  if ($? != 0) {
    logMessage(0, "WARN", "Failed to chgrp local archive dir \"".
               $dir."\" to group \"".$proj_id."\"");
  }
 
  # Set group sticky bit on local archive dir
  `chmod -R g+s $dir`;
  if ($? != 0) {
    logMessage(0, "WARN", "Failed to set sticky bit on local archive dir \"".
               $dir."\"");
  }

  
  # Create an obs.start file in the processing dir:
  logMessage(2, "INFO", "Creating obs.start");
  $dir = $local_archive_dir."/".$utc_start."/".$beam;
  my $obsstart_file = $dir."/obs.start";
  open(FH,">".$obsstart_file.".tmp");
  print FH $raw_header;
  close FH;
  rename($obsstart_file.".tmp",$obsstart_file);

  return $obsstart_file;

}


#
# Copies the UTC_START file via NFS to the server's results and archive directories:
#

sub copyUTC_STARTfile($$$) {

  my ($utc_start, $beam, $obs_start_file) = @_;

  # nfs mounts
  #my $archive_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $results_dir = $cfg{"SERVER_RESULTS_NFS_MNT"};
  my $cmd = ""; 
  my $result = "";
  my $response = "";

  # Ensure each directory is automounted
  #if (!( -d  $archive_dir)) {
  #  `ls $archive_dir >& /dev/null`;
  #}
  if (!( -d  $results_dir)) {
    `ls $results_dir >& /dev/null`;
  }

  # Create the full nfs destinations
  #$archive_dir .= "/".$utc_start."/".$beam;
  $results_dir .= "/".$utc_start."/".$beam;

  $cmd = "cp ".$obs_start_file." ".$results_dir;
  logMessage(2, "INFO", "NFS copy \"".$cmd."\"");
  ($result, $response) = Dada->mySystem($cmd,0);
  if ($result ne "ok") {
                                                                                                                                                                     
    logMessage(0, "ERROR", "Failed to nfs copy \"".$obs_start_file."\" to \"".$results_dir."\": response: \"".$response."\"");
    logMessage(0, "ERROR", "Command was: \"".$cmd."\"");
    if (-f $obs_start_file) {
      logMessage(0, "ERROR", "File existed locally");
    } else {
      logMessage(0, "ERROR", "File did not exist locally");
    }
  }

  #$cmd = "cp ".$obs_start_file." ".$archive_dir;
  #logMessage(2, "INFO", "NFS copy \"".$cmd."\"");
  #($result, $response) = Dada->mySystem($cmd,0);
  #if ($result ne "ok") {
  #  logMessage(0, "ERROR", "Failed to nfs copy \"".$obs_start_file."\" to \"".$archive_dir."\": response: \"".$response."\"");
  #  logMessage(0, "ERROR", "Command was: \"".$cmd."\"");
  #  if (-f $obs_start_file) {
  #    logMessage(0, "ERROR", "File existed locally");
  #  } else {
  #    logMessage(0, "ERROR", "File did not exist locally");
  #  }
  #}

  # Since the header has been transferred, unlink it
  #unlink($obs_start_file);
  #if ($? != 0) {
  #  logMessage(0, "WARN", "Failed to unlink \"".$obs_start_file."\" file \"".$response."\"");
  #}
  logMessage(2, "INFO", "Server directories perpared");

}

