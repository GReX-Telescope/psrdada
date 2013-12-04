#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2009 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# Transfers APSR observations to Swinburne raid disks for archival
#

#
# Constants
#
use constant SSH_OPTS       => "-x -o BatchMode=yes";
use constant BANDWIDTH      => "60000";           # KB/s
use constant DATA_DIR       => "/lfs/raid0/apsr";
use constant META_DIR       => "/lfs/data0/apsr";
use constant REQUIRED_HOST  => "raid0";
use constant REQUIRED_USER  => "apsr";

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Apsr;

#
# function prototypes
#
sub good($);
sub getObsToSend($);
sub getDest();
sub transferObs($$$$$$$);
sub moveObs($$$$$);

#
# global variable definitions
#
our $dl;
our $daemon_name;
our %cfg;
our $last_dest;
our $quit_daemon : shared;
our $warn;
our $error;
our $transfer_kill : shared;
#
# initialize globals
#
$dl = 1; 
$daemon_name = Dada::daemonBaseName(basename($0));
%cfg = Apsr::getConfig();
$last_dest = 0;
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$transfer_kill = "";

{
  $warn  = META_DIR."/logs/".$daemon_name.".warn";
  $error = META_DIR."/logs/".$daemon_name.".error";

  my $log_file    = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file    = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file   = META_DIR."/control/".$daemon_name.".quit";

  my $src_path    = DATA_DIR."/swin/send";
  my $dst_path    = DATA_DIR."/archived";
  my $err_path    = DATA_DIR."/swin/fail";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $control_thread = 0;

  my $pid = "";
  my $src= "";
  my $obs = "";
  my $r_user = "";
  my $r_host = "";
  my $r_path = "";

  my $counter = 0;
  my $sleeping = 0;

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

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  Dada::logMsg(1, $dl, "Starting APSR Transfer Manager");

  chdir $src_path;

  while (!$quit_daemon)
  {
    # find and observation to send
    Dada::logMsg(2, $dl, "main: getObsToSend(".$src_path.")");
    ($pid, $src, $obs) = getObsToSend($src_path);
    Dada::logMsg(2, $dl, "main: getObsToSend(): ".$pid." ".$src." ".$obs);

    if ($obs ne "none") 
    {
      # find a suitable destination disk that has enough space
      Dada::logMsg(2, $dl, "main: getDest()");
      ($result, $r_user, $r_host, $r_path) = getDest();
      Dada::logMsg(2, $dl, "main: getDest() ".$result. " ".$r_user."@".$r_host.":".$r_path);

      if ($result ne "ok") 
      {
        if (!$sleeping)
        {
          Dada::logMsg(1, $dl, "Waiting for destination to become available");
          $sleeping = 1;
        }
      }
      else 
      {
        $sleeping = 0;
        Dada::logMsg(2, $dl, "main: transferObs() ".$pid."/".$src."/".$obs." to ".$r_user."@".$r_host."/".$r_path);
        ($result, $response) = transferObs($src_path, $pid, $src, $obs, $r_user, $r_host, $r_path);
        Dada::logMsg(2, $dl, "main: transferObs() ".$result." ".$response);

        if ($result ne "ok")
        {
          # If we have been asked to quit during the transferObs, then failure is expected
          if ($quit_daemon) {
            Dada::logMsg(2, $dl, "main: asked to quit");
          } 
          # this observation could not be transferred for some other reason
          else
          {
            Dada::logMsgWarn($warn, "main: transferObs failed: ".$response);
            Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." send -> fail");
            Dada::logMsg(2, $dl, "main: moveObs(".$src_path.", ".$err_path.", ".$pid.", ".$src.", ".$obs.")");
            ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
            Dada::logMsg(2, $dl, "main: moveObs ".$result." ".$response);
          }
        } 
        else
        {
          Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." send -> archived ".$response);
          Dada::logMsg(2, $dl, "main: moveObs(".$src_path.", ".$dst_path.", ".$pid.", ".$src.", ".$obs.")");
          ($result, $response) = moveObs($src_path, $dst_path, $pid, $src, $obs);
          Dada::logMsg(2, $dl, "main: moveObs ".$result." ".$response);
        }
      }
    }
    else
    {
      if (!$sleeping)
      {
        Dada::logMsg(1, $dl, "Waiting for observation to transfer");
        $sleeping = 1;
      }
    }

    if ($sleeping)
    {
      # If we did not transfer, sleep 60
      Dada::logMsg(2, $dl, "Sleeping 60 seconds");
      $counter = 12;
      while ((!$quit_daemon) && ($counter > 0)) {
        sleep(5);
        $counter--;
      }
    }
  }


  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  # rejoin threads
  $control_thread->join();
                                                                                
  exit 0;
}


###############################################################################
#
# Functions
#

#
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
#
sub getObsToSend($) 
{
  (my $dir) = @_;

  Dada::logMsg(3, $dl, "getObsToSend(".$dir.")");

  my $pid = "none";
  my $src = "none";
  my $obs = "none";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @lines = ();
  my $line = "";
  my @bits = ();
  my $i = 0;
  my $found_obs = 0;

  $cmd = "find ".$dir." -mindepth 3 -maxdepth 3 -type d | sort";
  Dada::logMsg(3, $dl, "getObsToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "getObsToSend: find failed: ".$response);
    return ($pid, $src, $obs);
  }

  @lines = split(/\n/, $response);
  
  for ($i=0; (!$found_obs && $i<=$#lines); $i++)
  {
    $line = $lines[$i];

    @bits = split(/\//, $line);
    if ($#bits < 2)
    { 
      Dada::logMsgWarn($warn, "getObsToSend: not enough components in path");
      next;
    }

    $pid = $bits[$#bits-2];
    $src = $bits[$#bits-1];
    $obs = $bits[$#bits];
    $found_obs = 1;
  }

  Dada::logMsg(3, $dl, "getObsToSend: returning ".$pid." ".$src." ".$obs);
  return ($pid, $src, $obs);
}

#
# Transfers the specified observation to the specified destination
#
sub transferObs($$$$$$$) 
{
  my ($s_path, $pid, $src, $obs, $user, $host, $path) = @_;

  my $cmd = "";
  my $xfer_result = "ok";
  my $xfer_response = "ok";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $rsync_options = "-a --stats --no-g --chmod=go-ws --exclude 'band.finished' --exclude 'band.transferred' ".
                      "--bwlimit ".BANDWIDTH." --password-file=/home/apsr/.ssh/shrek_rsync_pw";

  # create the remote destination direectory
  $cmd = "mkdir -m 0755 -p ".$path."/".$pid."/".$src;
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "transferObs: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "transferObs: ssh for ".$user."@".$host." failed: ".$response);
    return ("fail", "ssh failed: ".$response);
  }
  if ($rval != 0)
  {
    Dada::logMsgWarn($warn, "transferObs: failed to create ".$pid."/".$src." directory: ".$response);
    return ("fail", "could not create remote dir");
  }

  # determine the remoe moulde name fro the preconfigured server
  my ($junk, $prefix, $r_module, $suffix) = split(/\//, $path, 4);

  $transfer_kill = "pkill -f '^rsync ".$s_path."/".$src."/".$pid."/".$obs."' -u apsr";

  # rsync the observation
  $cmd = "rsync ".$s_path."/".$pid."/".$src."/".$obs." ".$user."@".$host."::".$r_module."/apsr/".$pid."/".$src."/ ".$rsync_options;
  Dada::logMsg(2, $dl, "transferObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "transferObs: ".$result." ".$response);

  $transfer_kill = "";

  if ($result eq "ok")
  {
    # determine the data rate
    my @output_lines = split(/\n/, $response);
    my $mbytes_per_sec = 0;
    my $mbytes = 0;
    my $seconds = 0;
    my $i = 0;
    for ($i=0; $i<=$#output_lines; $i++)
    {
      if ($output_lines[$i] =~ m/bytes\/sec/)
      {
        my @bits = split(/[\s]+/, $output_lines[$i]);
        $mbytes_per_sec = $bits[6] / 1048576;
        $mbytes = $bits[1] / 1048576;
        $seconds = $mbytes / $mbytes_per_sec;

      }
    }
    $xfer_response = sprintf("%2.0f", $mbytes)." MB in ".sprintf("%2.0f",$seconds).
                     "s, ".sprintf("%2.0f", $mbytes_per_sec)." MB/s";

    Dada::logMsg(2, $dl, $pid."/".$src."/".$obs." ".$response);
  }
  else 
  {
    if ($quit_daemon)
    {
      Dada::logMsg(1, $dl, "transferObs: rsync interrupted");
      $xfer_response = "rsync interrupted for quit";
    }
    else
    {
      Dada::logMsg(0, $dl, "transferObs: rsync failed for ".$pid."/".$src."/".$obs.": ".$response);
      $xfer_response = "rsync failure";
    }
    $xfer_result = "fail";

    # Delete the partially transferred observation
    Dada::logMsg(1, $dl, "transferObs: rsync failed, deleting partial transfer at ".$path."/".$pid."/".$src."/".$obs);
    $cmd = "rm -rf ".$path."/".$pid."/".$src."/".$obs;
    Dada::logMsg(2, $dl, "transferObs: remoteSsh(".$user.", ".$host.", ".$cmd.")"); 
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(2, $dl, "transferObs: ".$result." ".$rval." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "transferObs: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
      return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
    }
    if ($rval != 0) 
    {
      Dada::logMsgWarn($warn, "transferObs: failed to delete partial transfer at: ".$path."/".$pid."/".$src."/".$obs);
      return ("fail", "failed to delete partial transfer for: ".$path."/".$pid."/".$src."/".$obs);
    }

  }

  return ($xfer_result, $xfer_response);
}

#
# check SWIN_DIRs to find an acceptable receiver for the observation
#
sub getDest() 
{

  Dada::logMsg(3, $dl, "getDest()");

  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  my $i=0;
  my $user = "";
  my $host = "";
  my $path = "";

  my $r_user = "none";
  my $r_host = "none";
  my $r_path = "none";

  for ($i=0; ($r_host eq "none" && $i<$cfg{"NUM_SWIN_DIRS"}); $i++)
  {
    ($user, $host, $path) = split(/:/,$cfg{"SWIN_DIR_".$last_dest});
    $last_dest++;
    if ($last_dest >= $cfg{"NUM_SWIN_DIRS"})
    {
      $last_dest = 0;
    }

    # test how much space is remaining on this disk
    $cmd = "df -B 1048576 -P ".$path." | tail -n 1 | awk '{print \$4}'";
    Dada::logMsg(3, $dl, "getDest: ".$user."@".$host.":".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "getDest: ".$result." ".$rval." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "getDest: ssh to ".$user."@".$host." failed: ".$response);
      next;
    }

    # check there is 50 * 1GB free
    if (int($response) < (50*1024)) {
      Dada::logMsg(1, $dl, "getDest: less than 50GB remaining on ".$user."@".$host.":".$path);
      next;
    }

    Dada::logMsg(2, $dl, "getDest: found ".$user."@".$host.":".$path);
    return ("ok", $user, $host, $path);
  }
  return ("fail", $user, $host, $path);
}

#
# Move an observation from from to to
#
sub moveObs($$$$$) {

  my ($from, $to, $pid, $src, $obs) = @_;

  Dada::logMsg(3, $dl ,"moveObs(".$from.", ".$to.", ".$pid.", ".$src.", ".$obs.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  # check that the required PID / SOURCE dir exists in the destination
  if ( ! -d ($to."/".$pid."/".$src ) ) 
  {
    $cmd = "mkdir -p ".$to."/".$pid."/".$src;
    Dada::logMsg(2, $dl, "moveObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "moveObs: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "moveObs: failed to create dir ".$to."/".$pid."/".$src.": ".$response);
      return ("fail", "could not create dest dir");
    } 
  }

  # move the observation to to
  $cmd = "mv ".$from."/".$pid."/".$src."/".$obs." ".$to."/".$pid."/".$src."/";
  Dada::logMsg(2, $dl, "moveObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "moveObs: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "moveObs: failed to move ".$src."/".$pid."/".$obs." to ".$to.": ".$response);
    return ("fail", "could not move observation");
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

  my $result = "";
  my $response = "";

  if ($transfer_kill ne "")
  {
    Dada::logMsg(1, $dl ,"controlThread: ".$transfer_kill);
    ($result, $response) = Dada::mySystem($transfer_kill);
    Dada::logMsg(1, $dl ,"controlThread: ".$result. " ".$response);
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
sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
} 


# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;

  my $result = "";
  my $response = "";

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # this script can *only* be run on the caspsr-raid0 server
  my $host = Dada::getHostMachineName();
  if ($host ne REQUIRED_HOST) {
    return ("fail", "Error: this script can only be run on ".REQUIRED_HOST);
  }

  my $curr_user = `whoami`;
  chomp $curr_user;
  if ($curr_user ne REQUIRED_USER) {
    return ("fail", "Error: this script can only be run as ".REQUIRED_USER);
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}
