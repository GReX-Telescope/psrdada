#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use Dada;
use Caspsr;
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;
use File::Basename;

#
# Function Prototypes
#
sub main();

#
# Declare Global Variables
# 
our $user;
our $dl : shared;
our $daemon_name : shared;
our $dada_header_cmd;
our $client_logger;
our %cfg : shared;
our $quit_daemon : shared;
our $log_host;
our $log_port;
our $log_sock;
our $dada_header_pid : shared;    # PID of the dada_header command for killing
our $processor_pid : shared;      # PID of the processor (dspsr) for killing


#
# Initialize Global variables
#
%cfg = Caspsr::getConfig();
$user = "caspsr";
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$dada_header_cmd = "dada_header -k ".lc($cfg{"PROCESSING_DATA_BLOCK"});
$client_logger = "client_caspsr_src_logger.pl";
$quit_daemon = 0;
$log_host = 0;
$log_port = 0;
$log_sock = 0;
$dada_header_pid = 0;
$processor_pid = 0;


my $result = 0;
$result = main();

exit($result);


###############################################################################
#
# package functions
# 

sub main() {

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file      = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";

  $log_host          = $cfg{"SERVER_HOST"};
  $log_port          = $cfg{"SERVER_SYS_LOG_PORT"};

  my $control_thread = 0;
  my $prev_utc_start = "";
  my $prev_obs_offset = "";
  my $utc_start = "";
  my $obs_offset = "";
  my $obs_end = 0;
  my $quit = 0;
  my $result = "";
  my $response = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR "ERROR failed to start: ".$response."\n";
    return 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  # open a connection to the nexus logging port
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  msg(0,"INFO", "STARTING SCRIPT");

  # set umask so that 
  #  files : -rw-r-----
  #   dirs : drwxr-x---
  umask 0027;

  # start the control thread
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # Main Loop
  while(!$quit_daemon) {

    # Run the processing thread once
    ($quit_daemon, $utc_start, $obs_offset, $obs_end) = processingThread($prev_utc_start, $prev_obs_offset);

    if ($utc_start eq "invalid") {

      if (!$quit_daemon) {
        msg(0, "ERROR", "processingThread return an invalid obs/header");
        sleep(1);
      }

    # } elsif (($obs_end) && ($utc_start eq $prev_utc_start)) {
    #   msg(0, "ERROR", "main: obs_end and UTC_START repeated"); 

    } else {
      msg(2, "INFO", "processingThread was successful");
    }

    $prev_utc_start = $utc_start;
    $prev_obs_offset = $obs_offset;

  }

  msg(2, "INFO", "main: joining threads");
  $control_thread->join();
  msg(2, "INFO", "main: controlThread joined");

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

  return 0;
}

###############################################################################
#
# Process an observation
#
sub processingThread($$) {

  my ($prev_utc_start, $prev_obs_offset) = @_;

  my $localhost = Dada::getHostMachineName();
  my $bindir = Dada::getCurrentBinaryVersion();
  my $processing_dir = $cfg{"CLIENT_RESULTS_DIR"};

  my $copy_obs_start_thread = 0;
  my $utc_start = "invalid";
  my $obs_offset = "invalid";

  my $raw_header = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my %h = ();
  my $end_of_obs = 1;
  my $obs_xfer = 0;
  my $header_valid = 1;

  # Get the next filled header on the data block. Note that this may very
  # well hang for a long time - until the next header is written...
  $cmd =  $bindir."/".$dada_header_cmd;
  msg(2, "INFO", "Running ".$cmd);
  $raw_header = `$cmd 2>&1`;
  msg(2, "INFO", $cmd." returned");

  # since the only way to currently stop this daemon is to send a kill
  # signal to dada_header_cmd, we should check the return value
  if ($? == 0) {

    my $proc_cmd = "";

    ($result, $response) = Caspsr::processHeader($raw_header, $cfg{"CONFIG_DIR"}); 

    %h = Dada::headerToHash($raw_header);

    if ($result ne "ok") {
      msg(0, "ERROR", $response);
      msg(0, "ERROR", "DADA header malformed, jettesioning xfer");  
      $proc_cmd = "dada_dbnull -s -k ".lc($cfg{"PROCESSING_DATA_BLOCK"});

    } else {

      msg(2, "INFO", "DADA header looks correct");
      $utc_start = $h{"UTC_START"};
      $obs_offset = $h{"OBS_OFFSET"};

      $proc_cmd = $response;
      $header_valid = 1;

      # check for the OBS_XFER 
      if (defined($h{"OBS_XFER"})) {
        $obs_xfer = $h{"OBS_XFER"};
        if ($obs_xfer eq "-1") {
          $end_of_obs = 1;
        } else {
          $end_of_obs = 0;
        }
      
      } else {
        $end_of_obs = 1;
        $obs_xfer = 0;
      }

      msg(2, "INFO", "new header: UTC_START=".$utc_start.", FREQ=".$h{"FREQ"}.
                        ", OBS_OFFSET=".$obs_offset.", PROC_FILE=".$h{"PROC_FILE"}.
                        ", OBS_XFER=".$obs_xfer." END_OF_OBS=".$end_of_obs);

      # special case for and END of XFER
      # if (($obs_xfer eq "-1") && ($h{"PROC_FILE"} =~ m/dbdisk/)) {
      if ($obs_xfer eq "-1") {

        msg(1, "INFO", "Ignoring final [extra] transfer");
        msg(1, "INFO", "Ignoring: UTC_START=".$utc_start.", OBS_OFFSET=".$obs_offset.
                       ", OBS_XFER=".$obs_xfer." END_OF_OBS=".$end_of_obs);

        $proc_cmd = "dada_dbnull -s -k ".lc($cfg{"RECEIVING_DATA_BLOCK"});

      } else {

        # if we are an XFER
        if ($utc_start eq $prev_utc_start) {

          if ((!$end_of_obs) && ($obs_offset eq $prev_obs_offset)) {
            msg(0, "WARN", "The UTC_START and OBS_OFFSET has been repeated obs_xfer=".$obs_xfer.", end_of_obs=".$end_of_obs);
            $proc_cmd = "dada_dbnull -s -k ".lc($cfg{"RECEIVING_DATA_BLOCK"}); 
            msg(1, "INFO", "Ignoring repeat transfer");
          }  else {
            msg(0, "INFO", "Continuing Obs: UTC_START=".$utc_start." XFER=".$obs_xfer." EOB=".$end_of_obs." OFFSET=".$obs_offset);
          }

        # this is a new observation, setup the directories
        } else {

          msg(0, "INFO", "New Observation: UTC_START=".$utc_start." XFER=".$obs_xfer." EOB=".$end_of_obs);

          # create the local directory and UTC_START file
          msg(2, "INFO", "processingThread: createLocalDirectory()");
          ($result, $response) = createLocalDirectory($utc_start, $h{"PID"}, $raw_header);
          msg(2, "INFO", "processingThread: ".$result." ".$response);

          # if we are the first PWC, send the obs.start to the server in background thread 
          msg(2, "INFO", "processingThread: copyObsStartThread(".$utc_start.")");
          $copy_obs_start_thread = threads->new(\&copyObsStartThread, $utc_start);

        }

        $processing_dir .= "/".$utc_start;

        if ($proc_cmd =~ m/dspsr/) {

          # add the ARCHIVE_MOD command line option
          if (exists($cfg{"ARCHIVE_MOD"})) {
            my $archive_mod = $cfg{"ARCHIVE_MOD"};
            if ($proc_cmd =~ m/-L \d+/) {
              $proc_cmd =~ s/-L \d+/-L $archive_mod/;
            } else {
              $proc_cmd .= " -L ".$archive_mod;
            }
          }
          $proc_cmd .= " ".$cfg{"PROCESSING_DB_KEY"};
        }
      }
    }

    if (length($proc_cmd) > 60) {
      msg(1, "INFO", substr($proc_cmd, 0, 60)."...");
    } else {
      msg(1, "INFO", $proc_cmd);
    }

    chdir $processing_dir;

    $cmd = $bindir."/".$proc_cmd;
    $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger;

    my $returnVal = system($cmd);

    if ($returnVal != 0) {
      msg(0, "WARN", "Processing command failed: ".$?." ".$returnVal);
    }

    msg(2, "INFO", "END ".substr($proc_cmd, 0, 60)."...");

    # if we copied the obs.start file, join the thread
    if ($copy_obs_start_thread) {
      msg(2, "INFO", "processingThread: joining copyObsStartThread");
      $copy_obs_start_thread->join();
      msg(2, "INFO", "processingThread: copyObsStartThread joined");
    }

    if (($end_of_obs) || ($proc_cmd =~ /dspsr/)) {
      touchPwcFinished($h{"UTC_START"});
    }

    return (0, $utc_start, $obs_offset, $end_of_obs);

  } else {

    if (!$quit_daemon) {
      msg(0, "WARN", "dada_header_cmd failed");
      sleep 1;
    }
    return (1, $utc_start, $obs_offset, $end_of_obs);
  }
}
  

###############################################################################
#
# Create the local directory required for this observation
#
sub createLocalDirectory($$$) {

  my ($utc_start, $proj_id, $raw_header) = @_;

  msg(2, "INFO", "createLocalDirectory(".$utc_start.", ".$proj_id.", raw_header)");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $file = "";

  my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start;
  my $results_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start;

  # Create the archive and results dirs
  $cmd = "mkdir -p ".$archive_dir." ".$results_dir;
  msg(2, "INFO", "createLocalDirectory: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirectory: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0,"ERROR", "Could not create local dirs: ".$response);
    return ("fail", "could not create local dirs: ".$response);
  }

  # Set group sticky bit on local archive dir
  $cmd = "chmod g+s ".$archive_dir." ".$results_dir;
  msg(2, "INFO", "createLocalDirectory: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirectory: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "chmod g+s failed on ".$archive_dir." ".$results_dir);
  }

  # Set GID on the directory
  $cmd = "chgrp -R ".$proj_id." ".$archive_dir." ".$results_dir;
  msg(2, "INFO", "createLocalDirectory: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirectory: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "chgrp to ".$proj_id." failed on ".$archive_dir." ".$results_dir);
  }

  # create an obs.start file in the archive dir
  $file = $results_dir."/obs.start";
  open(FH,">".$file.".tmp");
  print FH $raw_header;
  close FH;
  rename($file.".tmp",$file);

  return ("ok", $file);

}

###############################################################################
#
# Copies the obs.start file via NFS to the server's results directory.
#
sub copyObsStartThread($) {

  my ($utc_start) = @_;

  my $localhost = Dada::getHostMachineName();
  my $local_file = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start."/obs.start";
  my $remote_file = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$localhost."_obs.start";
  my $cmd = "";
  my $result = "";
  my $response = "";

  if (! -f $local_file) {
    msg(0, "ERROR", "copyObsStartThread: obs.start file [".$local_file."] did not exist");
    return ("fail", "obs.start file did not exist");
  }

  $cmd = "scp -B ".$local_file." dada\@srv0:".$remote_file;
  msg(2, "INFO", "copyObsStartThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd, 0);
  msg(2, "INFO", "copyObsStartThread: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "copyObsStartThread: scp [".$cmd."] failed: ".$response);
    return ("fail", "scp failed: ".$response);
  }

  $cmd = "cp ".$local_file." ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start."/";
  msg(2, "INFO", "copyObsStartThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd, 0);
  msg(2, "INFO", "copyObsStartThread: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "copyObsStartThread: cp [".$cmd."] failed: ".$response);
    return ("fail", "cp failed: ".$response);
  }


  return ("ok", "");

}

###############################################################################
#
# copy the pwc.finished file to the server
#
sub touchPwcFinished($) {
  
  my ($utc_start) = @_;
  
  my $localhost = Dada::getHostMachineName();
  my $fname = $localhost."_pwc.finished";
  my $cmd = "";
  my $result = "";
  my $response = "";

  if ( -f $fname) {
    return ("ok", "");

  } else {

    # touch the local file
    $cmd = "touch ".$fname;
    msg(2, "INFO", "touchPwcFinished: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd, 0);
    msg(2, "INFO", "touchPwcFinished: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "touchPwcFinished: ".$cmd." failed: ".$response);
      return ("fail", "local touch failed: ".$response);
    }
  
    $cmd = "ssh -o BatchMode=yes -l dada srv0 'touch ".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$fname."'";
    msg(2, "INFO", "touchPwcFinished: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd, 0);
    msg(2, "INFO", "touchPwcFinished: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "touchPwcFinished: ".$cmd." failed: ".$response);
      return ("fail", "remote touch failed: ".$response);
    }
  
    return ("ok", "");
  }
}

###############################################################################
#
# Control thread to handle quit requests
#
sub controlThread($$) {

  my ($quit_file, $pid_file) = @_;

  msg(2, "INFO", "controlThread: starting (".$quit_file.", ".$pid_file.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  # poll for the existence of the quit_file or the global quit variable
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  # Kill the dada_header command
  $cmd = "ps aux | grep -v grep | grep ".$user." | grep '".$dada_header_cmd."' | awk '{print \$2}'";
  msg(2, "INFO", " controlThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", " controlThread: ".$result." ".$response);
  $response =~ s/\n/ /;
  if (($result eq "ok") && ($response ne "")) {
    $response =~ s/\n/ /;
    $cmd = "kill -KILL ".$response;
    msg(1, "INFO", "controlThread: Killing dada_header: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "controlThread: ".$result." ".$response);
  }

  # Kill all running dspsr commands
  $cmd = "ps aux | grep -v grep | grep ".$user." | grep dspsr | awk '{print \$2}'";
  msg(2, "INFO", "controlThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  $response =~ s/\n/ /;
  msg(2, "INFO", "controlThread: ".$result." ".$response);
  if (($result eq "ok") && ($response ne "")) {
    $cmd = "killall -KILL ".$response;
    msg(1, "INFO", "Killing all dspsr ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "controlThread: ".$result." ".$response);
  }

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  } 

  msg(2, "INFO", "controlThread: exiting");
}

###############################################################################
#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "proc mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


###############################################################################
#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # Tell threads to try and quit
  $quit_daemon = 1;
  sleep(3);
  
  if ($log_sock) {
    close($log_sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

###############################################################################
#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}


###############################################################################
#
# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # check required gloabl parameters
  if ( ($user eq "") || ($dada_header_cmd eq "") || ($client_logger eq "")) {
    return ("fail", "Error: a package variable missing [user, dada_header_cmd, client_logger]");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}

