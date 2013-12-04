#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2011 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# Transfers BPSR observations to Swinburne raid disks for archival
#

#
# Constants
#
#use constant BANDWIDTH        => "76800";           # = 75 MB/s 
#use constant BANDWIDTH        => "53248";           # = 52 MB/s 
use constant BANDWIDTH        => "90000";           # = 52 MB/s 
use constant DATA_DIR         => "/lfs/raid0/bpsr";
use constant META_DIR         => "/lfs/data0/bpsr";
use constant REQUIRED_HOST    => "raid0";
use constant REQUIRED_USER    => "bpsr";

use constant BPSR_USER        => "dada";
use constant BPSR_HOST        => "hipsr-srv0.atnf.csiro.au";
use constant BPSR_PATH        => "/data/bpsr";

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Bpsr;

#
# function prototypes
#
sub good($);
sub getBeamToSend($);
sub getDest($$$$);
sub transferBeam($$$$$$$);
sub moveBeam($$$$$);

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
%cfg = Bpsr::getConfig();
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
  my $dst_path    = DATA_DIR."/swin/sent";
  # my $otp_path    = DATA_DIR."/parkes/on_tape";   # for beams already marked on.tape.parkes
  my $err_path    = DATA_DIR."/swin/fail";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $control_thread = 0;

  my $pid = "";
  my $obs = "";
  my $beam = "";
  my $r_user = "";
  my $r_host = "";
  my $r_path = "";

  my $counter = 0;
  my $sleeping = 0;

  my $rval = 0;

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

  Dada::logMsg(1, $dl, "Starting BPSR Transfer Manager");

  chdir $src_path;

  while (!$quit_daemon)
  {
    # find and observation to send
    Dada::logMsg(2, $dl, "main: getBeamToSend(".$src_path.")");
    ($pid, $obs, $beam) = getBeamToSend($src_path);
    Dada::logMsg(2, $dl, "main: getBeamToSend(): ".$pid." ".$obs." ".$beam);

    if ($obs ne "none") 
    {
      # find a suitable destination disk that has enough space and is not marked as [READ|WRIT]ING
      Dada::logMsg(2, $dl, "main: getDest()");
      ($result, $r_user, $r_host, $r_path) = getDest($src_path, $pid, $obs, $beam);
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
        Dada::logMsg(2, $dl, $pid."/".$obs."/".$beam." -> ".$r_host);
        Dada::logMsg(2, $dl, "main: transferBeam() ".$pid."/".$obs."/".$beam." to ".$r_user."@".$r_host.":".$r_path);
        ($result, $response) = transferBeam($src_path, $pid, $obs, $beam, $r_user, $r_host, $r_path);
        Dada::logMsg(2, $dl, "main: transferBeam() ".$result." ".$response);

        if ($result ne "ok")
        {
          # If we have been asked to quit during the transferBeam, then failure is expected
          if ($quit_daemon) {
            Dada::logMsg(2, $dl, "main: asked to quit");
          } 
          # this observation could not be transferred for some other reason
          else
          {
            Dada::logMsgWarn($warn, "main: transferBeam failed: ".$response);
            Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." -> ".$r_host." + swin/fail");
            Dada::logMsg(2, $dl, "main: moveBeam(".$src_path.", ".$err_path.", ".$pid.", ".$obs.", ".$beam.")");
            ($result, $response) = moveBeam($src_path, $err_path, $pid, $obs, $beam);
            Dada::logMsg(2, $dl, "main: moveBeam ".$result." ".$response);
          }
        } 
        else
        {
          Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." -> ".$r_host." + swin/sent [".$response."]");
          Dada::logMsg(2, $dl, "main: moveBeam(".$src_path.", ".$dst_path.", ".$pid.", ".$obs.", ".$beam.")");
          ($result, $response) = moveBeam($src_path, $dst_path, $pid, $obs, $beam);
          Dada::logMsg(2, $dl, "main: moveBeam ".$result." ".$response);


          #Dada::logMsg(2, $dl, "main: checking for existence of ".$src_path."/".$pid."/".$obs."/".$beam."/on.tape.parkes");
          # if this beam as already been marked as on tape at parkes
          #if ( -f $src_path."/".$pid."/".$obs."/".$beam."/on.tape.parkes")
          #{
          #  Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." -> parkes/on_tape [".$response."]");
          #  Dada::logMsg(2, $dl, "main: moveBeam(".$src_path.", ".$otp_path.", ".$pid.", ".$obs.", ".$beam.")");
          #  ($result, $response) = moveBeam($src_path, $otp_path, $pid, $obs, $beam);
          #  Dada::logMsg(2, $dl, "main: moveBeam ".$result." ".$response);
          #}
          #else
          #{
          #  Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." -> parkes/archive [".$response."]");
          #  Dada::logMsg(2, $dl, "main: moveBeam(".$src_path.", ".$dst_path.", ".$pid.", ".$obs.", ".$beam.")");
          #  ($result, $response) = moveBeam($src_path, $dst_path, $pid, $obs, $beam);
          #  Dada::logMsg(2, $dl, "main: moveBeam ".$result." ".$response);

          #  # then touch the local xfer.complete so that the parkes tape archiver knows it can go [if not fail]
          #  $cmd = "touch ".$dst_path."/".$pid."/".$obs."/".$beam."/xfer.complete";
          #  Dada::logMsg(2, $dl, "main: ".$cmd);
          #  ($result, $response) = Dada::mySystem($cmd);
          #  Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          #  if ($result ne "ok")
          #  {
          #    Dada::logMsgWarn($warn, "main: failed to touch ".$dst_path."/".$pid."/".$obs."/".$beam."/xfer.complete: ".$response);
          #  }

          #  # touch remote sent.to.parkes file on bpsr server
          #  $cmd = "touch ".BPSR_PATH."/results/".$obs."/".$beam."/sent.to.parkes";
          #  Dada::logMsg(2, $dl, "main: ".BPSR_USER."@".BPSR_HOST." ".$cmd);
          #  ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
          #  Dada::logMsg(3, $dl, "main: ".$result." ".$rval." ".$response);
          #  if ($result ne "ok")
          #  {
          #    Dada::logMsgWarn($warn, "main: ssh failed to ".BPSR_HOST.": ".$response);
          #  }
          #  if ($rval != 0)
          #  {
          #    Dada::logMsgWarn($warn, "main: could not touch ".$obs."/".$beam."/sent.to.parkes. on ".BPSR_HOST.": ".$response);
          #  }
          #}

          # touch remote sent.to.swin file on apsr machines
          $cmd = "touch ".BPSR_PATH."/results/".$obs."/".$beam."/sent.to.swin";
          Dada::logMsg(2, $dl, "main: ".BPSR_USER."@".BPSR_HOST." ".$cmd);
          ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$rval." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "main: ssh failed to ".BPSR_HOST.": ".$response);
          }
          if ($rval != 0)
          {
            $response =~ s/`//;
            $response =~ s/'//;
            Dada::logMsgWarn($warn, "main: could not touch ".$obs."/".$beam."/sent.to.swin on ".BPSR_HOST.": ".$response);
          }
        }

        # check if this observations directory is now empty
        $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 1 -maxdepth 1 -type l | wc -l";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if (($result eq "ok") && ($response eq "0")) 
        {
          $cmd = "rmdir ".$src_path."/".$pid."/".$obs;
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        }
      }
    }
    else
    {
      if (!$sleeping)
      {
        Dada::logMsg(1, $dl, "Waiting for obs to transfer");
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
# Find an obs/beam to send, search chronologically.
#
sub getBeamToSend($) 
{
  (my $dir) = @_;

  Dada::logMsg(3, $dl, "getBeamToSend(".$dir.")");

  my $pid = "none";
  my $obs = "none";
  my $beam = "none";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $source = "";
  my @lines = ();
  my $line = "";
  my @bits = ();
  my $i = 0;
  my $found_obs = 0;

  $cmd = "find ".$dir." -mindepth 3 -maxdepth 3 -type l | sort";
  Dada::logMsg(3, $dl, "getBeamToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getBeamToSend: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "getBeamToSend: find failed: ".$response);
    return ($pid, $obs, $beam);
  }

  @lines = split(/\n/, $response);
  
  for ($i=0; (!$found_obs && $i<=$#lines); $i++)
  {
    $line = $lines[$i];

    @bits = split(/\//, $line);
    if ($#bits < 2)
    { 
      Dada::logMsgWarn($warn, "getBeamToSend: not enough components in path");
      next;
    }

    $pid  = $bits[$#bits-2];
    $obs  = $bits[$#bits-1];
    $beam = $bits[$#bits];
    $found_obs = 1;
  }

  Dada::logMsg(3, $dl, "getBeamToSend: returning ".$pid." ".$obs." ".$beam);
  return ($pid, $obs, $beam);
}

#
# Transfers the specified observation to the specified destination
#
sub transferBeam($$$$$$$) 
{
  my ($s_path, $pid, $obs, $beam, $user, $host, $path) = @_;

  my $cmd = "";
  my $xfer_result = "ok";
  my $xfer_response = "ok";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $rsync_options = "-a --stats --no-g --no-l -L --chmod=go-ws --exclude 'beam.finished' --exclude 'beam.transferred'".
                      " --exclude 'sent.to.swin' --exclude 'sent.to.parkes' --exclude 'on.tape.swin'".
                      " --exclude 'on.tape.parkes' --exclude 'xfer.complete' --bwlimit=".BANDWIDTH.
                      " --password-file=/home/bpsr/.ssh/shrek_rsync_pw";

  # create the WRITING file
  #$cmd = "touch ".$path."/../WRITING";
  #Dada::logMsg(2, $dl, "transferBeam: ".$cmd);
  #($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  #Dada::logMsg(3, $dl, "transferBeam: ".$result." ".$response);
  #if ($result ne "ok") 
  #{
  #  Dada::logMsgWarn($warn, "transferBeam: ssh for ".$user."@".$host." failed: ".$response);
  #  return ("fail", "ssh failed: ".$response);
  #}
  #if ($rval != 0) 
  #{
  #  Dada::logMsgWarn($warn, "transferBeam: failed to touch remote WRITING file");
  #  return ("fail", "could not touch remote WRITING file");
  #}

  # create the archive/PID/OBS directory
  $cmd = "mkdir -m 0755 -p ".$path."/archive/".$pid."/".$obs;
  Dada::logMsg(2, $dl, "transferBeam: ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "transferBeam: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "transferBeam: ssh for ".$user."@".$host." failed: ".$response);
    return ("fail", "ssh failed: ".$response);
  } 
  if ($rval != 0)
  {
    Dada::logMsgWarn($warn, "transferBeam: failed to create ".$pid."/".$obs." directory: ".$response);
    return ("fail", "could not create remote dir");
  } 

  # set the global transfer_kill pkill pattern required to kill this command
  $transfer_kill = "pkill -f '^rsync ".$s_path."/".$pid."/".$obs."/".$beam."'";

  # determine the remote module name for the preconfigured rsync server
  my ($junk, $prefix, $r_module, $suffix) = split(/\//, $path, 4);
  $cmd = "rsync ".$s_path."/".$pid."/".$obs."/".$beam." ".$user."@".$host."::".$r_module."/bpsr/archive/".$pid."/".$obs."/ ".$rsync_options;
  Dada::logMsg(2, $dl, "transferBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "transferBeam: ".$result." ".$response);

  # unset the global transfer_kill
  $transfer_kill = "";

  if (($result eq "ok") && ($rval == 0))
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
        #$mbytes = $bits[4] / 1048576;
        $mbytes = $bits[1] / 1048576;
        $seconds = $mbytes / $mbytes_per_sec;
      }
    }
    $xfer_response = sprintf("%2.0f", $mbytes)." MB in ".sprintf("%2.0f",$seconds)."s, ".sprintf("%2.0f", $mbytes_per_sec)." MB/s";

    Dada::logMsg(2, $dl, $pid."/".$obs."/".$beam." ".$response);

    # touch the remote xfer.completed file so that the swin archiver knows it can archive the beam
    $cmd = "touch ".$path."/archive/".$pid."/".$obs."/".$beam."/xfer.complete";
    Dada::logMsg(2, $dl, "transferBeam: ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "transferBeam: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "transferBeam: ssh for ".$user."@".$host." failed: ".$response);
      return ("fail", "ssh failed: ".$response);
    }
    if ($rval != 0)
    {
      Dada::logMsgWarn($warn, "transferBeam: failed to touch xfer.complete in ".$path."/archive/".$pid."/".$obs."/".$beam.": ".$response);
      return ("fail", "could not touch remote xfer.complete");
    }
  }
  else
  {
    if ($quit_daemon)
    {
      Dada::logMsg(1, $dl, "transferBeam: rsync interrupted");
      $xfer_response = "rsync interrupted for quit";
    }
    else
    {
      $xfer_response = "rsync failure";
      Dada::logMsgWarn($warn, "transferBeam: rsync failed: ".$response);
    }
    $xfer_result = "fail";

    # Delete the partially transferred observation
    Dada::logMsg(1, $dl, "transferBeam: rsync failed, deleting partial transfer at ".$path."/archive/".$pid."/".$obs."/".$beam);
    $cmd = "rm -rf ".$path."/archive/".$pid."/".$obs."/".$beam;
    Dada::logMsg(2, $dl, "transferBeam: remoteSsh(".$user.", ".$host.", ".$cmd.")"); 
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(2, $dl, "transferBeam: ".$result." ".$rval." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "transferBeam: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
    }
    if ($rval != 0) 
    {
      Dada::logMsgWarn($warn, "transferBeam: failed to delete partial transfer at: ".$path."/archive/".$pid."/".$obs."/".$beam);
    }
  }

  # remove the WRITING file
  #$cmd = "rm -f ".$path."/../WRITING";
  #Dada::logMsg(2, $dl, "transferBeam: ".$cmd);
  #($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  #Dada::logMsg(3, $dl, "transferBeam: ".$result." ".$response);
  #if ($result ne "ok")
  #{
  #  Dada::logMsgWarn($warn, "transferBeam: ssh for ".$user."@".$host." failed: ".$response);
  #  return ("fail", "ssh failed: ".$response);
  #}
  #if ($rval != 0)
  #{
  #  Dada::logMsgWarn($warn, "transferBeam: failed to remove remote WRITING file");
  #  return ("fail", "could not remove remote WRITING file");
  #}

  return ($xfer_result, $xfer_response);
}

#
# check SWIN_DIRs to find an acceptable receiver for the observation
#
sub getDest($$$$) 
{

  my ($src_path, $pid, $obs, $beam) = @_;

  Dada::logMsg(3, $dl, "getDest(".$src_path.", ".$pid.", ".$obs.", ".$beam.")");

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

  my $beam_size = 0;
  my $junk = "";

  # check how big the beam is [MB]
  $cmd = "du -sL -B 1048576 ".$src_path."/".$pid."/".$obs."/".$beam." | awk '{print \$1}'";
  Dada::logMsg(3, $dl, "getDest: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getDest: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "getDest: ".$cmd." failed: ".$response);
    return ("fail", $user, $host, $path);
  }

  $beam_size = $response;
  Dada::logMsg(2, $dl, "getDest: beam_size=".$beam_size." MB");
  
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

    # check there is 100 * 1GB free
    if (int($response) < (100*1024)) {
      Dada::logMsg(2, $dl, "getDest: less than 100 GB remaining on ".$user."@".$host.":".$path);
      next;
    }

    if (int($beam_size) > int($response)) {
      Dada::logMsg(2, $dl, "getDest: beam_size [".$beam_size."] > space on ".$user."@".$host.":".$path." [".$response."]");
      next;
    }

    return ("ok", $user, $host, $path);

    # check if this is being used for [READ|WRIT]ING
    # $cmd = "ls ".$path."/../????ING";
    # Dada::logMsg(3, $dl, "getDest: ".$user."@".$host.":".$cmd);
    # ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    # Dada::logMsg(3, $dl, "getDest: ".$result." ".$rval." ".$response);
    # if ($result ne "ok") {
    #   Dada::logMsgWarn($warn, "getDest: ssh to ".$user."@".$host." failed: ".$response);
    #   next;
    # }

    # if (($response =~ m/No such file or directory/) || ($response =~ m/ls: No match/))
    # {
    #  Dada::logMsg(3, $dl, "getDest: no control files in ".$path."/../");
    #  Dada::logMsg(2, $dl, "getDest: found ".$user."@".$host.":".$path);
    #  return ("ok", $user, $host, $path);
    #}
    #else
    #{
    #  Dada::logMsg(2, $dl, "getDest: control file existed in ".$path."/../, skipping");
    #}
  }
  return ("fail", $user, $host, $path);
}

#
# Move an observation from from to to
#
sub moveBeam($$$$$) 
{
  my ($from, $to, $pid, $obs, $beam) = @_;

  Dada::logMsg(3, $dl ,"moveBeam(".$from.", ".$to.", ".$pid.", ".$obs.", ".$beam.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  # check that the required PID / SOURCE dir exists in the destination
  Dada::logMsg(2, $dl, "moveBeam: createDir(".$to."/".$pid."/".$obs.", 0755)");
  ($result, $response) = Dada::createDir($to."/".$pid."/".$obs, 0755);
  Dada::logMsg(3, $dl, "moveBeam: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "moveBeam: failed to create dir ".$to."/".$pid."/".$obs.": ".$response);
    return ("fail", "could not create dest dir");
  }

  # move soft link for the beam from from to to
  $cmd = "mv ".$from."/".$pid."/".$obs."/".$beam." ".$to."/".$pid."/".$obs."/";
  Dada::logMsg(2, $dl, "moveBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "moveBeam: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "moveBeam: failed to move ".$pid."/".$obs."/".$beam." to ".$to."/".$pid."/".$obs.": ".$response);
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

  # kill the remote transfer
  if ($transfer_kill ne "")
  {
    Dada::logMsg(1, $dl ,"controlThread: ".$transfer_kill);
    ($result, $response) = Dada::mySystem($transfer_kill);
    Dada::logMsg(1, $dl ,"controlThread: ".$result." ".$response);
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

#
# Test to ensure all module variables are set before main
#
sub good($) 
{

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
