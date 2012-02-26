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
# Constants
#
use constant BANDWIDTH => "32768";  # KB/s


#
# Function declarations
#
sub good($);
sub setStatus($);
sub markState($$$);
sub getObsToSend();
sub transferObs($$$);
sub checkTransferred();
sub checkDeleted();

#
# Global variables
#
our $dl = 1;
our $daemon_name = Dada::daemonBaseName($0);
our %cfg = Caspsr::getConfig();
our $rate_mbits = 40;
our $quit_daemon : shared = 0;
our $curr_obs : shared = "none";
our $warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
our $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

our $r_user = "caspsr";
our $r_host = "raid0";
our $r_path = "/lfs/raid0/caspsr/finished";
our $r_module = "caspsr_upload";


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
my $source = "";
my $pid = "";
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

# start the control thread
Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
$control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

setStatus("Starting script");

Dada::logMsg(1, $dl, "Starting Transfer Manager");

chdir $cfg{"SERVER_ARCHIVE_DIR"};

Dada::logMsg(2, $dl, "main: checkTransferred()");
($result, $response) = checkTransferred();
Dada::logMsg(2, $dl, "main: checkTransferred() ".$result." ".$response);
if ($result ne "ok"){
  Dada::logMsgWarn($warn,"checkTransferred failed");
  $quit_daemon = 1;
}

Dada::logMsg(2, $dl, "main: checkDeleted()");
($result, $response) = checkDeleted();
Dada::logMsg(2, $dl, "main: checkDeleted() ".$result." ".$response);
if ($result ne "ok"){
  Dada::logMsgWarn($warn,"checkDeleted failed");
  $quit_daemon = 1;
}

while (!$quit_daemon) {

  # Fine an observation to send based on the CWD
  Dada::logMsg(2, $dl, "main: getObsToSend()");
  ($result, $pid, $source, $obs) = getObsToSend();
  Dada::logMsg(2, $dl, "main: getObsToSend(): ".$obs);

  if ($result ne "ok") {
    Dada::logMsg(2, $dl, "main: getObsToSend failed");
    $quit_daemon = 1;
    next;
  }

  if ($obs eq "none") {

    Dada::logMsg(2, $dl, "main: no observations to send. sleep 60");

  } else {

    $xfer_failure = 0;
    $files = "";

    setStatus("Transferring ".$obs);

    $curr_obs  = $obs;

    Dada::logMsg(2, $dl, "main: transferObs(".$pid.", ".$source.", ".$obs.")");
    ($result, $response) = transferObs($pid, $source, $obs);
    Dada::logMsg(2, $dl, "main: transferObs() ".$result." ".$response);

    if ($result eq "ok") {

      # mark this observation as transferred
      Dada::logMsg(2, $dl, "main: markState(".$obs.", obs.finished, obs.transferred)");
      ($result, $response) = markState($obs, "obs.finished", "obs.transferred");
      Dada::logMsg(2, $dl, "main: markState() ".$result." ".$response); 
      Dada::logMsg(1, $dl, $pid."/".$source."/".$obs." transferring -> transferred ".$response);

      setStatus($obs." xfer success");

    } else {

      if (!$quit_daemon) 
      {
        Dada::logMsgWarn($warn, "main: failed to transfer ".$obs);
        Dada::logMsg(2, $dl, "main: markState(".$obs.", obs.finished, obs.transfer_error)");
        ($result, $response) = markState($obs, "obs.finished", "obs.transfer_error");
        Dada::logMsg(2, $dl, "main: markState() ".$result." ".$response);
        Dada::logMsg(1, $dl, $pid."/".$source."/".$obs." transferring -> transfer error");

        setStatus($obs." xfer failure");
      }
    }

    $curr_obs = "none";
  }

  # If we did not transfer, sleep 60
  if ($obs eq "none") {

    setStatus("Waiting for obs");
    Dada::logMsg(2, $dl, "Sleeping 60 seconds");

    Dada::logMsg(2, $dl, "main: checkTransferred()");
    checkTransferred();
    Dada::logMsg(2, $dl, "main: checkTransferred() ".$result." ".$response);
    if ($result ne "ok"){
      Dada::logMsgWarn($warn, "checkTransferred failed");
      $quit_daemon = 1;
    }

    Dada::logMsg(2, $dl, "main: checkDeleted()");
    ($result, $response) = checkDeleted();
    Dada::logMsg(2, $dl, "main: checkDeleted() ".$result." ".$response);
    if ($result ne "ok"){
      Dada::logMsgWarn($warn,"checkDeleted failed");
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
# Deletes srv0's copy of the 8 second archives if observation has been 
# transferred and UTC_START is > 1month old
#
sub checkTransferred() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $o = "";
  my $obs_pid = "";

  Dada::logMsg(2, $dl, "checkTransferred()");

  # Find all observations marked as obs.transferred is at least 30 days old
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -maxdepth 2 -name 'obs.transferred' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkTransferred: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkTransferred: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkTransferred: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);
  my $curr_time = time;

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {
    $o = $observations[$i];

    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsgWarn($warn, "checkTransferred: Required file missing ".$o."/obs.info");
      next;
    }

    my @t = split(/-|:/,$o);
    my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

    Dada::logMsg(2, $dl, "checkTransferred: testing ".$o." curr=".$curr_time.", unix=".$unixtime);
    # if UTC_START is less than 30 days old, dont delete it
    if ($unixtime + (30*24*60*60) > $curr_time) {
      Dada::logMsg(2, $dl, "checkTransferred: Skipping ".$o.", less than 30 days old");
      next;
    }

    $cmd = "rm -f ".$o."/*/*.ar";
    Dada::logMsg(2, $dl, "checkTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkTransferred: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkTransferred: rm command failed: ".$response);
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
      Dada::logMsg(2, $dl, "checkTransferred: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "checkTransferred: remoteSshCommand() ".$result." ".$rval." ".$response);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkTransferred: ssh failed ".$response);
      } else {
        if ($rval != 0) {
          Dada::logMsg(0, $dl, "checkTransferred: no archive dir for ".$o." on ".$host);
        } 
        else
        {
          # touch the obs.deleted control file
          $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$o."/obs.deleted";
          Dada::logMsg(2, $dl, "checkTransferred: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
          ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
          Dada::logMsg(2, $dl, "checkTransferred: remoteSshCommand() ".$result." ".$rval." ".$response);
          if ($result ne "ok") {
            Dada::logMsgWarn($warn, "checkTransferred: ssh failed ".$response);
          } else {
            if ($rval != 0) {
              Dada::logMsgWarn($warn, "checkTransferred: remote obs.deleted touch failed: ".$response);
            } 
          }
        }
      }
    }

    Dada::logMsg(2, $dl, "checkTransferred: markState(".$o.", obs.transferred, obs.deleted)");
    ($result, $response) = markState($o, "obs.transferred", "obs.deleted");
    Dada::logMsg(2, $dl, "checkTransferred: markState() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkTransferred: markState(".$o.", obs.transferred, obs.deleted) failed: ".$response);
    } else {
      Dada::logMsg(1, $dl, $o.": transferred -> deleted");
    }

  }
  return ("ok", "");
}

#
# checks all observations marked as deleted, and move them to the old_results 
# dir if UTC_START > 6 months
#
sub checkDeleted()
{

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0; 
  my $o = "";
  my $obs_pid = "";
    
  Dada::logMsg(2, $dl, "checkDeleted()");
    
  # Find all observations marked as obs.deleted and at least 6 months old
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -maxdepth 2 -name 'obs.deleted' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
      
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkDeleted: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }     
      
  chomp $response;
  my @observations = split(/\n/,$response);
  my $curr_time = time;
        
  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {
    $o = $observations[$i];

    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsgWarn($warn, "checkDeleted: Required file missing ".$o."/obs.info");
      next;
    }       
          
    my @t = split(/-|:/,$o);
    my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);
            
    Dada::logMsg(2, $dl, "checkDeleted: testing ".$o." curr=".$curr_time.", unix=".$unixtime);
    # if UTC_START is less than 182 days old, dont delete it
    if ($unixtime + (182*24*60*60) > $curr_time) {
      Dada::logMsg(2, $dl, "checkDeleted: Skipping ".$o.", less than 182 days old");
      next;
    }
    
    # update the paths in obs.info for the TRES_AR and FRES_AR
    my $obs_info_file = $cfg{"SERVER_RESULTS_DIR"}."/".$o."/obs.info";
    Dada::logMsg(2, $dl, "checkDeleted: reading ".$obs_info_file);
    open FH,"<$obs_info_file" or return ("fail", "could open obs.info for reading");
    my @lines = <FH>;
    close FH;

    my $line;
    my $int_length = 0;
    my $snr = 0;

    Dada::logMsg(2, $dl, "checkDeleted: writing adjusted ".$obs_info_file);
    open FH,">$obs_info_file" or return ("fail", "could not open obs.info for writing");
    foreach $line (@lines) {

      if ($line =~ m/^INT/) {
        $int_length = 1;
      }
      if ($line =~ m/^SNR/) {
        $snr = 1;
      }
     
      # strip old redundant lines
      if (!(($line =~ m/OBS.START/) || ($line =~ m/TRES_AR/) || ($line =~ m/FRES_AR/)))
      {
        print FH $line;
      }
    }

    # If we didn't find the integration length, then get it from the TRES file
    if (!$int_length) 
    {
      $int_length = "NA";

      $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$o." -name '*_t.tot' | tail -n 1";
      Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
      if ($result eq "ok") 
      {
        $cmd = "vap -c length -n ".$response." | awk '{print \$2}'";
        Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
        if ($result eq "ok")  
        {
          $int_length = $response;
        }
      }
      Dada::logMsg(2, $dl, "checkDeleted: setting INT to ".$int_length);
      print FH "INT                 ".$int_length."\n";
    }

    # If we didn't find the SNR, then get it from the FRES file
    if (!$snr) 
    {
      $snr = "NA";

      $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$o." -name '*_f.tot' | tail -n 1";
      Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
      if ($result eq "ok") 
      {
        $cmd = "psrstat -q -j FTp -c snr ".$response." | awk -F= '{print \$2}'";
        Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
        if ($result eq "ok")  
        {
          $snr = sprintf("%5.1f",$response);
        }
      }
      Dada::logMsg(2, $dl, "checkDeleted: setting SNR to ".$snr);
      print FH "SNR                 ".$snr."\n";
    }
    close FH;

    # now change the state of the observation from deleted to old
    Dada::logMsg(2, $dl, "checkDeleted: markState(".$o.", obs.deleted, obs.old)");
    ($result, $response) = markState($o, "obs.deleted", "obs.old");
    Dada::logMsg(2, $dl, "checkDeleted: markState() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDeleted: markState(".$o.", obs.deleted, obs.old) failed: ".$response);
    } else {
      Dada::logMsg(1, $dl, $o.": deleted -> old");
    }

    # move it to old_results dir
    $cmd = "mv ".$cfg{"SERVER_RESULTS_DIR"}."/".$o." ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/";
    Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsgWarn($warn, "checkDeleted: mv command failed: ".$response);
      return ("fail", "mv command failed");
    }

    # delete the directory in the archives dir
    $cmd = "rm -rf ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o;
    Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDeleted: ".$cmd." failed: ".$response);
      return ("fail", "could not remove archive dir");
    }

    # delete all timer and .tot archives
    $cmd = "find ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/".$o." -name '*.ar' -delete -o -name '*.tot' -delete";
    Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);

    # delete any files in subdirs 
    $cmd = "find ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/".$o." -mindepth 2 -delete";
    Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);

    # delete any subdirs
    $cmd = "find ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/".$o." -mindepth 1 -type d -delete";
    Dada::logMsg(2, $dl, "checkDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDeleted: ".$result." ".$response);

    #Dada::logMsg(2, $dl, "checkDeleted: sleep(1)");
    #sleep(1);

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
sub getObsToSend() {

  Dada::logMsg(2, $dl, "getObsToSend()");

  my $obs_to_send = "none";

  my $archives_dir = $cfg{"SERVER_ARCHIVE_DIR"};

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @obs_finished = ();
  my $i = 0;
  my $j = 0;
  my $obs_pid = "";
  my $source = "";
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
    return ("ok", "none", "none", "none");
  }

  # If there is nothing to transfer, simply return
  @obs_finished = split(/\n/, $response);
  if ($#obs_finished == -1) {
    return ("ok", "none", "none", "none");
  }

  Dada::logMsg(2, $dl, "getObsToSend: found ".($#obs_finished+1)." observations marked obs.finished");

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && ($obs_to_send eq "none")); $i++) 
  {
    $obs = $obs_finished[$i];
    Dada::logMsg(2, $dl, "getObsToSend: checking ".$obs);

    # skip if no obs.info exists
    if (!( -f $archives_dir."/".$obs."/obs.info")) {
      Dada::logMsg(0, $dl, "Required file missing: ".$obs."/obs.info");
      next;
    }

    # get the SOURCE
    $cmd = "grep ^SOURCE ".$archives_dir."/".$obs."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getObsToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsToSend: ".$result.":".$response);
    if (($result ne "ok") || ($response eq "")) {
      Dada::logMsgWarn($warn, "getObsToSend: failed to parse SOURCE from obs.info [".$obs."] ".$response);
      next;
    }
    $source = $response;

    # get the PID 
    $cmd = "grep ^PID ".$archives_dir."/".$obs."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getObsToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsToSend: ".$result.":".$response);
    if (($result ne "ok") || ($response eq "")) {
      Dada::logMsgWarn($warn, "getObsToSend: failed to parse PID from obs.info [".$obs."] ".$response);
      next;
    }
    $pid = $response;

    # we have an acceptable obs to send
    $obs_to_send = $obs;

  }

  Dada::logMsg(2, $dl, "getObsToSend: returning ".$pid." ".$source." ".$obs_to_send);
  return ("ok", $pid, $source, $obs_to_send);
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
sub transferObs($$$) {

  my ($pid, $source, $obs) = @_;

  Dada::logMsg(2, $dl, "transferObs(".$pid.", ".$source.", ".$obs.")");

  my $localhost = Dada::getHostMachineName();

  my $rsync_options = "-a --no-g --chmod=go-ws --bwlimit=".BANDWIDTH.
                      " --password-file=/home/dada/.ssh/raid_rsync_pw";

  my $result = "";
  my $response = "";
  my $rval = 0;
  my $cmd = "";
  my $dirs = "";
  my $files = "";
  my @output_lines = ();
  my @bits = ();
  my $i = 0;
  my $data_rate = "";

  if ($quit_daemon) {
    Dada::logMsg(1, $dl, "transferObs: quitting before transfer begun");
    return ("fail", "quit flag raised before transfer begun");
  }

  Dada::logMsg(1, $dl, $pid."/".$source."/".$obs." finished -> transferring");
  Dada::logMsg(2, $dl, "Transferring ".$obs." to ".$r_user."@".$r_host.":".$r_path);

  # create the required directories on the remote host
  $cmd = "mkdir -m 0755 -p ".$r_path."/".$pid."; ".
         "mkdir -m 0755 -p ".$r_path."/".$pid."/".$source."; ".
         "mkdir -m 0755 -p ".$r_path."/".$pid."/".$source."/".$obs;
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: ssh failed for ".$cmd." on ".$r_user."@".$r_host.": ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "transferObs: failed to create SOURCE dir on remote archive");
      return ("fail", "couldn't create source dir on ".$r_user."@".$r_host);
    }
  }

  # transfer obs.start, fres and tres sums to repos
  $files = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.start ".
           $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/*_f.tot ".
           $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/*_t.tot";
  $cmd = "rsync ".$files." ".$r_user."@".$r_host."::".$r_module."/".$pid."/".$source."/".$obs."/ ".$rsync_options;
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
  $cmd = "rsync --stats ".$files." ".$r_user."@".$r_host."::".$r_module."/".$pid."/".$source."/".$obs."/ ".$rsync_options;
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "transferObs: ".$result.":".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: rsync failed: ".$response);
    return ("fail", "transfer failed");
  } else {
    @output_lines = split(/\n/, $response);
    for ($i=0; $i<=$#output_lines; $i++)
    {
      if ($output_lines[$i] =~ m/bytes\/sec/)
      {
        @bits = split(/[\s]+/, $output_lines[$i]);
        $data_rate = sprintf("%3.0f", ($bits[1] / 1048576))." MB @ ".sprintf("%3.0f", ($bits[6] / 1048576))." MB/s";
      }
    }
  }

  # touch remote obs.transferred
  $cmd = "touch ".$r_path."/".$pid."/".$source."/".$obs."/obs.transferred";
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "transferObs: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "transferObs: ssh failed for ".$cmd." on ".$r_user."@".$r_host.": ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "transferObs: failed to touch remote ".$r_path."/".$pid."/".$source."/".$obs."/obs.transferred");
      return ("fail", "couldn't touch remote obs.transferred");
    }
  }

  return ("ok", $data_rate);
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

  Dada::logMsg(2, $dl, "checkDestinaion: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "checkDestinaion: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkDestinaion: ssh failed for ".$cmd." on ".$r_user."@".$r_host.": ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  } else {
    if ($rval != 0) {
      Dada::logMsg(0, $dl, "checkDestinaion: remote check failed: ".$response);
      return ("fail", "couldn't check disk space");
    }
  }

  # check there is 100 * 1GB free
  if (int($response) < (100*1024)) {
    return ("fail", "less than 100 GB remaining on ".$r_user."@".$r_host.":".$r_path);
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

  my $result = "";
  my $response = "";

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }


  # Check connectivity to remote repos
  my $rval = 0;
  $cmd = "uptime";
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  if ($result ne "ok") {
    return ("fail", "ssh failure to ".$r_user."@".$r_host.": ".$response);
  } else {
    if ($rval != 0) {
      return ("fail", "ssh remote command failure: ".$response);
    }
  }

  return ("ok", "");
}
