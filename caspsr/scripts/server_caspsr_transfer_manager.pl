#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Transfer manager for CASPSR instrument, uses rsync + ssh tunnel for transfer
#
 

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use File::Basename;
use Time::Local;
use Dada;
use Caspsr;

#
# Function declarations
#
sub good($);
sub setStatus($);
sub markState($$$);
sub getObsToSend($);
sub transferObs($);
sub checkTransferredAndOld();

#
# Global variables
#
our $dl = 1;
our $daemon_name = Dada::daemonBaseName($0);
our %cfg = Caspsr::getConfig();
our $pids = "P140 P361 P456 P630 P794";
our $rate_mbits = 40;
our $quit_daemon : shared = 0;
our $curr_obs : shared = "none";
our $warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
our $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

our $swin_user = "pulsar";
our $swin_host = "shrek214-gb";
our $swin_repos = "/export/shrek214a/caspsr";

#
# Constants
#
use constant SSH_OPTS      => "-x -o BatchMode=yes";


#
# Main
#
my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

my $result = "";
my $response = "";
my $control_thread = 0;
my $tunnel_thread = 0;
my $obs = "";
my $i = 0;
my $cmd = "";
my $xfer_failure = 0;
my $files = "";
my $counter = 0;

# sanity check on whether the module is good to go
($result, $response) = good($quit_file);
if ($result ne "ok") {
  print STDERR $response."\n";
  exit 1;
}

# install signal handlers
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# become a daemon
Dada::daemonize($log_file, $pid_file);

Dada::logMsg(0, $dl ,"STARTING SCRIPT");

sleep(10);

# start the control thread
Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
$control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

setStatus("Starting script");

Dada::logMsg(1, $dl, "Starting Transfer Manager for ".$pids);

chdir $cfg{"SERVER_ARCHIVE_DIR"};

Dada::logMsg(2, $dl, "main: checkTransferredAndOld()");
($result, $response) = checkTransferredAndOld();
Dada::logMsg(2, $dl, "main: checkTransferredAndOld() ".$result." ".$response);
if ($result ne "ok"){
  Dada::logMsgWarn($warn,"checkTransferredAndOld failed");
  $quit_daemon = 1;
}

while (!$quit_daemon) {

  # Fine an observation to send based on the CWD
  Dada::logMsg(2, $dl, "main: getObsToSend()");
  ($result, $response) = getObsToSend($pids);
  Dada::logMsg(2, $dl, "main: getObsToSend(): ".$obs);

  if ($result ne "ok") {
    Dada::logMsg(2, $dl, "main: getObsToSend failed: ".$response);
    $quit_daemon = 1;
    next;
  }

  $obs = $response;

  if ($obs eq "none") {

    Dada::logMsg(2, $dl, "main: no observations to send. sleep 60");

  } else {

    $xfer_failure = 0;
    $files = "";

    setStatus("Transferring ".$obs);

    $curr_obs  = $obs;

    Dada::logMsg(2, $dl, "main: transferObs(".$obs.")");
    ($result, $response) = transferObs($obs);
    Dada::logMsg(2, $dl, "main: transferObs() ".$result." ".$response);

    if ($result eq "ok") {

      # mark this observation as transferred
      Dada::logMsg(2, $dl, "main: markState(".$obs.", obs.finished, obs.transferred)");
      ($result, $response) = markState($obs, "obs.finished", "obs.transferred");
      Dada::logMsg(2, $dl, "main: markState() ".$result." ".$response); 
      Dada::logMsg(1, $dl, $obs." transferring -> transferred");

      setStatus($obs." xfer success");

    } else {

      if (!$quit_daemon) 
      {
        Dada::logMsgWarn($warn, "main: failed to transfer ".$obs);
        Dada::logMsg(2, $dl, "main: markState(".$obs.", obs.finished, obs.transfer_error)");
        ($result, $response) = markState($obs, "obs.finished", "obs.transfer_error");
        Dada::logMsg(2, $dl, "main: markState() ".$result." ".$response);
        Dada::logMsg(1, $dl, $obs." transferring -> transfer error");

        setStatus($obs." xfer failure");
      }
    }

    $curr_obs = "none";
  }

  # If we did not transfer, sleep 60
  if ($obs eq "none") {

    setStatus("Waiting for obs");
    Dada::logMsg(2, $dl, "Sleeping 60 seconds");

    Dada::logMsg(2, $dl, "main: checkTransferredAndOld()");
    checkTransferredAndOld();
    Dada::logMsg(2, $dl, "main: checkTransferredAndOld() ".$result." ".$response);
    if ($result ne "ok"){
      Dada::logMsgWarn($warn, "checkTransferredAndOld failed");
      $quit_daemon = 1;
    }
  
    $counter = 12;
    while ((!$quit_daemon) && ($counter > 0)) {
      sleep(5);
      $counter--;
    }
  }
}

# rejoin threads
Dada::logMsg(1, $dl, "main: joining threads");
$control_thread->join();

setStatus("Script stopped");

Dada::logMsg(0, $dl, "STOPPING SCRIPT");

                                                                              
exit 0;


###############################################################################
#
# Functions
#

#
# Deletes srv0's copy of the 8 second archives if observation is > 1month old
#
sub checkTransferredAndOld() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $o = "";
  my $obs_pid = "";

  Dada::logMsg(2, $dl, "checkTransferredAndOld()");

  # Find all observations marked as obs.transferred is at least 30 days old
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -maxdepth 2 -name 'obs.transferred' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkTransferredAndOld: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkTransferredAndOld: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkTransferredAndOld: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);
  my $curr_time = time;

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {
    $o = $observations[$i];

    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsgWarn($warn, "checkTransferredAndOld: Required file missing ".$o."/obs.info");
      next;
    }

    my @t = split(/-|:/,$o);
    my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

    Dada::logMsg(2, $dl, "checkTransferredAndOld: testing ".$o." curr=".$curr_time.", unix=".$unixtime);
    # if UTC_START is less than 30 days old, dont delete it
    if ($unixtime + (30*24*60*60) > $curr_time) {
      Dada::logMsg(2, $dl, "checkTransferredAndOld: Skipping ".$o.", less than 30 days old");
      next;
    }

    $cmd = "rm -f ".$o."/*/*.ar";
    Dada::logMsg(2, $dl, "checkTransferredAndOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkTransferredAndOld: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkTransferredAndOld: rm command failed: ".$response);
      return ("fail", "find command failed: ".$response);
    }

    # mark this observation deleted on each of the PWC clients
    my $user = "caspsr";
    my $host = "";
    my $rval = 0;
    my $j = 0;

    for ($j=0; $j<$cfg{"NUM_PWC"}; $j++) {

      $host = $cfg{"PWC_".$j};

      # check that a directory for this obs existed
      $cmd = "ls -1 ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$o;
      Dada::logMsg(2, $dl, "checkTransferredAndOld: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "checkTransferredAndOld: remoteSshCommand() ".$result." ".$rval." ".$response);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkTransferredAndOld: ssh failed ".$response);
      } else {
        if ($rval != 0) {
          Dada::logMsg(0, $dl, "checkTransferredAndOld: no archive dir for ".$o." on ".$host);
        } 
        else
        {
          # touch the obs.deleted control file
          $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$o."/obs.deleted";
          Dada::logMsg(2, $dl, "checkTransferredAndOld: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
          ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
          Dada::logMsg(2, $dl, "checkTransferredAndOld: remoteSshCommand() ".$result." ".$rval." ".$response);
          if ($result ne "ok") {
            Dada::logMsgWarn($warn, "checkTransferredAndOld: ssh failed ".$response);
          } else {
            if ($rval != 0) {
              Dada::logMsgWarn($warn, "checkTransferredAndOld: remote obs.deleted touch failed: ".$response);
            } 
          }
        }
      }
    }

    Dada::logMsg(2, $dl, "checkTransferredAndOld: markState(".$o.", obs.transferred, obs.deleted)");
    ($result, $response) = markState($o, "obs.transferred", "obs.deleted");
    Dada::logMsg(2, $dl, "checkTransferredAndOld: markState() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkTransferredAndOld: markState(".$o.", obs.transferred, obs.deleted) failed: ".$response);
    } else {
      Dada::logMsg(1, $dl, $o.": transferred -> deleted");
    }

  }
  return ("ok", "");
}


#
# Tag an observation as $to, removing the existing $from
#
sub markState($$$) {

  my ($o, $from, $to) = @_;

  Dada::logMsg(2, $dl, "markState(".$o.", ".$from.", ".$to.")");

  if ($from eq "") {
    Dada::logMsgWarn($warn, "markState: \$from not specified");
    return ("fail", "old state not specified");
  }
  if ($to eq "") {
    Dada::logMsgWarn($warn, "markState: \$to not specified");
    return ("fail", "new state not specified");
  }

  my @dirs = ();
  push @dirs, $cfg{"SERVER_RESULTS_DIR"};
  push @dirs, $cfg{"SERVER_ARCHIVE_DIR"};

  my $from_path = "";
  my $to_path = "";

  for ($i=0; $i<=$#dirs; $i++) {

    $from_path = $dirs[$i]."/".$o."/".$from;
    $to_path = $dirs[$i]."/".$o."/".$to;

    if (! -f $from_path) {
      Dada::logMsgWarn($warn, "markState: \$from file did not exist");
      return ("fail", "\$from file did not exist");
    }
    if ( -f $to_path) {
      Dada::logMsgWarn($warn, "markState: \$to file already existed");
      return ("fail", "\$to file already existed");
    }
  }

  my $cmd = "";
  my $result = "";
  my $response = "";

  for ($i=0; $i<=$#dirs; $i++) {

    $from_path = $dirs[$i]."/".$o."/".$from;
    $to_path = $dirs[$i]."/".$o."/".$to;

    $cmd = "rm -f ".$from_path;
    Dada::logMsg(2, $dl, "markState: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "markState: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
    }

    $cmd = "touch ".$to_path;
    Dada::logMsg(2, $dl, "markState: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "markState: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
    }
  }

  return ("ok", "");
}


#
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
#
sub getObsToSend($) {

  (my $pids) = @_;

  Dada::logMsg(2, $dl, "getObsToSend(".$pids.")");

  my $obs_to_send = "none";

  my $archives_dir = $cfg{"SERVER_ARCHIVE_DIR"};

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @obs_finished = ();
  my $i = 0;
  my $j = 0;
  my $obs_pid = "";
  my $proc_file = "";
  my $obs = "";

  # Look for all observations marked obs.finished in SERVER_NFS_RESULTS_DIR
  $cmd = "find ".$archives_dir." -maxdepth 2 -name obs.finished ".
         "-printf \"%h\\n\" | sort | awk -F/ '{print \$NF}'";

  Dada::logMsg(2, $dl, "getObsToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getObsToSend: ".$result.":".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "getObsToSend: ".$cmd." failed: ".$response);
    return ("ok", "none");
  }

  # If there is nothing to transfer, simply return
  @obs_finished = split(/\n/, $response);
  if ($#obs_finished == -1) {
    return ("ok", "none");
  }

  Dada::logMsg(2, $dl, "getObsToSend: found ".($#obs_finished+1)." observations marked obs.finished");

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && ($obs_to_send eq "none")); $i++) {

    $obs = $obs_finished[$i];
    Dada::logMsg(2, $dl, "getObsToSend: checking ".$obs);

    # skip if no obs.info exists
    if (!( -f $archives_dir."/".$obs."/obs.info")) {
      Dada::logMsg(0, $dl, "Required file missing: ".$obs."/obs.info");
      next;
    }

    # Find the PID 
    $cmd = "grep ^PID ".$archives_dir."/".$obs."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getObsToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsToSend: ".$result.":".$response);
    if (($result ne "ok") || ($response eq "")) {
      Dada::logMsgWarn($warn, "getObsToSend: failed to parse PID from obs.info [".$obs."] ".$response);
      next;
    }
    $obs_pid = $response;

    # If the PID doesn't match, skip it
    if ($obs_pid =~ m/$pids/) {
      Dada::logMsg(2, $dl, "getObsToSend: skipping ".$obs." PID mismatch [".$obs_pid." not in ".$pids."]");
      next;
    }

    # we have an acceptable obs to send
    $obs_to_send = $obs;

  }

  Dada::logMsg(2, $dl, "getObsToSend: returning ".$obs_to_send);
  return ("ok", $obs_to_send);
}

#
# Write the current status into the CONTROL_DIR/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada::logMsg(2, $dl, "setStatus(".$message.")");

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $dir = $cfg{"SERVER_CONTROL_DIR"};
  my $file = "xfer.state";

  $cmd = "rm -f ".$dir."/".$file;
  Dada::logMsg(2, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setStatus: ".$result." ".$response);

  $cmd = "echo '".$message."' > ".$dir."/".$file;
  Dada::logMsg(2, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setStatus: ".$result." ".$response);

  return ("ok", "");
    
}

#
# Transfer the specified obs 
#
sub transferObs($) {

  my ($obs) = @_;

  Dada::logMsg(2, $dl, "transferObs(".$obs.")");

  my $localhost = Dada::getHostMachineName();

  my $source = "";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $cmd = "";
  my $dirs = "";
  my $files = "";

  if ($quit_daemon) {
    Dada::logMsg(1, $dl, "transferObs: quitting before transfer begun");
    return ("fail", "quit flag raised before transfer begun");
  }

  Dada::logMsg(1, $dl, $obs." finished -> transferring");
  Dada::logMsg(2, $dl, "Transferring ".$obs." to ".$swin_user."@".$swin_host.":".$swin_repos);

  # get the source name from the obs.info file
  $cmd = "grep ^SOURCE ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$obs."/obs.info | awk '{print \$2}'";
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "transferObs: ".$result.":".$response);
  if (($result ne "ok") || ($response eq "")) {
    Dada::logMsgWarn($warn, "transferObs: failed to parse SOURCE from obs.info [".$obs."] ".$response);
    return ("fail", "couldn't parse SOURCE from obs.info");
  }
  $source = $response;

  # create the source directory on the remote host
  $cmd = "mkdir -p ".$swin_repos."/".$source."/".$obs;
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand(".$swin_user.", ".$swin_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($swin_user, $swin_host, $cmd);
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: ssh failed for ".$cmd." on ".$swin_user."@".$swin_host.": ".$response);
    return ("fail", "ssh failure to ".$swin_user."@".$swin_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "transferObs: failed to create SOURCE dir on remote archive");
      return ("fail", "couldn't create source dir on ".$swin_user."@".$swin_host);
    }
  }

  # transfer obs.start, fres and tres sums to repos
  $files = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.start ".
           $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/*_f.tot ".
           $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/*_t.tot";

  $cmd = "rsync -a ".$files." ".$swin_user."@".$swin_host.":".$swin_repos."/".$source."/".$obs."/";
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "transferObs: ".$result.":".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: rsync failed: ".$response);
    return ("fail", "transfer failed");
  }

  # transfer archives via rsync
  $files = $cfg{"SERVER_ARCHIVE_DIR"}."/".$obs."/*/2*.ar ";

  Dada::logMsg(2, $dl, "transferObs: Transferring archives...");
  $cmd = "rsync -a ".$files." ".$swin_user."@".$swin_host.":".$swin_repos."/".$source."/".$obs."/";
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "transferObs: ".$result.":".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: rsync failed: ".$response);
    return ("fail", "transfer failed");
  }

  # adjust file permissions on the transferred observation
  $cmd = "chmod -R a-w ".$swin_repos."/".$source."/".$obs;
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand(".$swin_user.", ".$swin_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($swin_user, $swin_host, $cmd);
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: ssh failed for ".$cmd." on ".$swin_user."@".$swin_host.": ".$response);
    return ("fail", "ssh failure to ".$swin_user."@".$swin_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "transferObs: failed to remove write perms on remote archive");
      return ("fail", "couldn't remove write perms on ".$swin_user."@".$swin_host);
    }
  }

  return ("ok", "");

} 


#
# check dir for connectivity and space
#
sub checkDestination($) {

  my ($d) = @_;

  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  $cmd = "df -B 1048576 -P ".$d." | tail -n 1 | awk '{print \$4}'";

  Dada::logMsg(2, $dl, "checkDestinaion: remoteSshCommand(".$swin_user.", ".$swin_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($swin_user, $swin_host, $cmd);
  Dada::logMsg(2, $dl, "checkDestinaion: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkDestinaion: ssh failed for ".$cmd." on ".$swin_user."@".$swin_host.": ".$response);
    return ("fail", "ssh failure to ".$swin_user."@".$swin_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "checkDestinaion: remote check failed: ".$response);
      return ("fail", "couldn't check disk space");
    }
  }

  # check there is 100 * 1GB free
  if (int($response) < (100*1024)) {
    return ("fail", "less than 100 GB remaining on ".$swin_user."@".$swin_host.":".$swin_repos);
  }

  return ("ok", "");

}

sub controlThread($$) {

  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  # try to cleanly stop things
  if (!(($curr_obs eq "none") || ($curr_obs eq ""))) {
    Dada::logMsg(1, $dl, "controlThread: interrupting obs ".$curr_obs);
    sleep(2);

  } else {
    Dada::logMsg(1, $dl, "controlThread: not interrupting due to no current observation");
  }

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  return 0;
}
  

#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

} 


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

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  # check the PID is a valid group for the specified instrument
  my @groups = ();
  my @pees = split(/ /,$pids);
  my $i = 0;
  my $j = 0;
  my $valid_pid = 0;
  @groups = Dada::getProjectGroups($cfg{"INSTRUMENT"});

  for ($i=0; $i<=$#pees; $i++) {
    for ($j=0; $j<=$#groups; $j++) {
      if ($groups[$j] eq $pees[$i]) {
        $valid_pid++;
      }
    }  
  }
  if ($valid_pid != $#pees+1) {
    return ("fail", "Error: specified PIDs [".$pids."] were not all valid groups for ".$cfg{"INSTRUMENT"});
  }

  my $result = "";
  my $response = "";

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }


  # Check connectivity to swin repos
  my $rval = 0;
  $cmd = "uptime";
  ($result, $rval, $response) = Dada::remoteSshCommand($swin_user, $swin_host, $cmd);
  if ($result ne "ok") {
    return ("fail", "ssh failure to ".$swin_user."@".$swin_host.": ".$response);
  } else {
    if ($rval != 0) {
      return ("fail", "ssh remote command failure: ".$response);
    }
  }

  return ("ok", "");
}
