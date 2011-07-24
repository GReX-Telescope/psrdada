#!/usr/bin/env perl
##############################################################################
#  
#     Copyright (C) 2009 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Transfers OBS/BANDs to the CASPSR RAID array for the archival pipeline
#
 
use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use File::Basename;
use Time::Local;
use Apsr;

#
# function prototypes
#
sub good($);
sub setStatus($);
sub checkFullyTransferred();
sub checkFullyDeleted();
sub checkFullyTransferred();
sub getBandToSend();
sub transferBand($$$);
sub markState($$$$);
sub checkAllBands($);


#
# global variable definitions
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;
our $r_user;
our $r_host;
our $r_path;
our $r_module;

#
# initialize package globals
#
$dl = 1; 
$daemon_name = Dada::daemonBaseName($0);
%cfg = Apsr::getConfig();
$r_user = "apsr";
$r_host = "raid0";
$r_path = "/lfs/raid0/apsr/finished";
$r_module = "apsr_upload";
$warn = ""; 
$error = ""; 
$quit_daemon = 0;

#
# Constants
#
use constant BWLIMIT  => "52768";

###############################################################################
#
# Main 
# 

{
  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $control_thread = 0;

  my $result = "";
  my $response = "";
  my $pid  = "";
  my $obs = "";
  my $band = "";
  my $i = 0;
  my $cmd = "";
  my $xfer_failure = 0;
  my $files = "";
  my $counter = 0;

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  Dada::logMsg(0, $dl ,"STARTING SCRIPT [bandwidth=".(BWLIMIT/1024)." MB/s]");

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  setStatus("Starting script");

  chdir $cfg{"SERVER_ARCHIVE_NFS_MNT"};

  # On startup do a check for fully transferred or fully archived obvserations
  Dada::logMsg(1, $dl, "Checking for fully transferred observations");
  ($result, $response) = checkFullyTransferred();
  Dada::logMsg(2, $dl, "main: checkFullyTransferred(): ".$result.":".$response);

  Dada::logMsg(1, $dl, "Checking for fully deleted observations");
  ($result, $response) = checkFullyDeleted();
  Dada::logMsg(2, $dl, "main: checkFullyDeleted(): ".$result.":".$response);

  while (!$quit_daemon) 
  {
    # Fine an observation to send based on the CWD
    Dada::logMsg(2, $dl, "main: getBandToSend");
    ($pid, $obs, $band) = getBandToSend();
    Dada::logMsg(2, $dl, "main: getBandToSend: ".$pid." ".$obs." ".$band);

    if ($obs eq "none") 
    {
      Dada::logMsg(2, $dl, "main: no observations to send. sleep 60");

      # On startup do a check for fully transferred or fully archived obvserations
      Dada::logMsg(2, $dl, "Checking for fully transferred observations");
      ($result, $response) = checkFullyTransferred();
      Dada::logMsg(2, $dl, "main: checkFullyTransferred(): ".$result.":".$response);

      Dada::logMsg(2, $dl, "Checking for fully deleted observations");
      ($result, $response) = checkFullyDeleted();
      Dada::logMsg(2, $dl, "main: checkFullyDeleted(): ".$result.":".$response)
    } 
    else
    {
      $xfer_failure = 0;
      $files = "";

      setStatus($obs."/".$band." &rarr; ".$r_host);

      ($result, $response) = transferBand($pid, $obs, $band);

      # If we have been asked to quit
      if ($quit_daemon) 
      {
        Dada::logMsg(2, $dl, "main: asked to quit");
      } 
      else 
      {
        # Check if all bands have been trasnferred successfully, if so, mark 
        # the observation as obs.transferred
        Dada::logMsg(2, $dl, "main: checkAllBands(".$obs.")");
        ($result, $response) = checkAllBands($obs);
        Dada::logMsg(2, $dl, "main: checkAllBands: ".$result." ".$response);

        if ($result ne "ok") 
        {
          Dada::logMsgWarn($warn, "main: checkAllBands failed: ".$response);
        } 
        else
        {
          if ($response ne "all bands sent") 
          {
            Dada::logMsg(2, $dl, "main: ".$obs." not fully transferred: ".$response);
          }
          else
          {
            Dada::logMsg(0, $dl, $obs." transferred");
            markState($obs, "obs.finished", "obs.transferred", "");
            setStatus($obs." xfer success");
          }
        }
      }
    }

    # If we did not transfer, sleep 60
    if ($obs eq "none") 
    {
      setStatus("Waiting for obs");
      Dada::logMsg(2, $dl, "Sleeping 60 seconds");
    
      $counter = 12;
      while ((!$quit_daemon) && ($counter > 0))
      {
        sleep(5);
        $counter--;
      }
    }
  }

  setStatus("Script stopped");

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
# Check the obs to see if all the bands have been sent to the dest
#
sub checkAllBands($) {

  my ($obs) = @_;

  my $cmd = "";
  my $find_result = "";
  my $band = "";
  my @bands = ();
  my $all_sent = 1;
  my $result = "";
  my $response = "";
  my $nbands = 0;
  my $nbands_mounted = 0;
  my $obs_pid = "";
  my @links = ();
  my $link = "";
  my $all_online = 0;
    
  # Determine the number of NFS links in the archives dir
  $cmd = "find ".$obs." -mindepth 1 -maxdepth 1 -type l | wc -l";
  Dada::logMsg(3, $dl, "checkAllBands: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkAllBands: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkAllBands: find command failed: ".$response);
    return ("fail", "find command failed");
  } 
  $nbands = $response;
  Dada::logMsg(3, $dl, "checkAllBands: Total number of bands ".$nbands);

  # Now find the number of mounted NFS links
  $cmd = "find -L ".$obs." -mindepth 1 -maxdepth 1 -type d -printf '\%f\\n' | sort";
  Dada::logMsg(3, $dl, "checkAllBands: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkAllBands: ".$result." ".$response);
  if ($result ne "ok") { 
    Dada::logMsgWarn($warn, "checkAllBands: find command failed: ".$response);
     return ("fail", "find command failed");
  } 
  @bands = split(/\n/, $response);
  $nbands_mounted = $#bands + 1;
  Dada::logMsg(3, $dl, "checkAllBands: Total number of mounted bands: ".$nbands_mounted);
  
  # If a machine is not online, they cannot all be verified
  if ($nbands != $nbands_mounted) {

    # This may be because the directory has been deleted/lost or it may
    # be because the machine is offline

    $cmd = "find ".$obs." -mindepth 1 -maxdepth 1 -type l -printf '\%l\n' | awk -F/ '{ print \$1\"/\"\$2\"/\"\$3\"/\"\$4\"/\"\$5 }'";
    Dada::logMsg(3, $dl, "checkAllBands: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkAllBands: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkAllBands: find command failed: ".$response);
      return ("fail", "find command failed");
    }

    @links = split(/\n/, $response);
    $all_online = 1;
    foreach $link (@links) {
      $cmd = "ls -ld ".$link;
      Dada::logMsg(3, $dl, "checkAllBands: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "checkAllBands: ".$result." ".$response);
      
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkAllBands: NFS dir ".$link." was offline");
        $all_online = 0;
      } 
    }

    if (!$all_online) {
      return ("ok", "all bands not mounted");
    } else {
      Dada::logMsgWarn($warn, "checkAllBands: ".($nbands - $nbands_mounted)." band(s) from ".$obs." were missing!");
    }
 
  }

  $all_sent = 1;
    
  # skip if no obs.info exists
  if (!( -f $obs."/obs.info")) {
    Dada::logMsgWarn($warn, "checkAllBands: Required file missing ".$obs."/obs.info");
    return ("fail", $obs."/obs.info did not exist");
  }

  # get the PID
  $cmd = "grep ^PID ".$obs."/obs.info | awk '{print \$2}'";
  Dada::logMsg(3, $dl, "checkAllBands: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkAllBands: ".$result." ".$response);
  if ($result ne "ok") {
    return ("fail", "could not determine PID");
  }
  $obs_pid = $response;

  my $i=0;
  for ($i=0; (($i<=$#bands) && ($all_sent)); $i++) {
    $band = $bands[$i];
    if (! -f $obs."/".$band."/band.transferred") 
    {
      $all_sent = 0;
      Dada::logMsg(2, $dl, "checkAllBands: ".$obs."/".$band."/band.transferred did not exist");
    }
  }

  if ($all_sent) {
    return ("ok", "all bands sent");
  } else { 
    return ("ok", "all bands not sent");
  }

}




# Adjust the directory structure to match the required storage format
sub changeSourceDirStructure($$$$$) {

  my ($u, $h, $d, $obs, $band) = @_;
  
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  # determine if this is a multi fold observation
  $cmd = "server_apsr_archive_finalizer.csh ".$d." ".$obs." ".$band;
  Dada::logMsg(2, $dl, "changeSourceDirStructure: [".$h."] ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($u, $h, $cmd);
  Dada::logMsg(2, $dl, "changeSourceDirStructure: [".$h."] ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "changeSourceDirStructure: ssh failed ".$response);
    return ("fail", "");
  } else {
    if ($rval != 0) {
      Dada::logMsgWarn($warn, "changeSourceDirStructure: script failed: ".$response);
      return ("fail", "");
    }
  }

  return ("ok", "");  

}



#
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
#
sub getBandToSend() 
{
  Dada::logMsg(3, $dl, "getBandToSend()");

  my $obs  = "none";
  my $band = "none";
  my $pid  = "none";
  my $nfs_archives = $cfg{"SERVER_ARCHIVE_NFS_MNT"};

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @obs_finished = ();
  my $i = 0;
  my $j = 0;
  my @bands = ();
  my $o = "";
  my $b = "";
  my $p = "";

  # look for all observations marked obs.finished in SERVER_NFS_RESULTS_DIR
  $cmd = "find ".$nfs_archives." -maxdepth 2 -name obs.finished ".
         "-printf \"%h\\n\" | sort | awk -F/ '{print \$NF}'";

  Dada::logMsg(2, $dl, "getBandToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getBandToSend: ".$result.":".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "getBandToSend: ".$cmd." failed: ".$response);
    return ($pid, $obs, $band);
  }

  # If there is nothing to transfer, simply return
  @obs_finished = split(/\n/, $response);
  if ($#obs_finished == -1) 
  {
    return ($pid, $obs, $band);
  }

  Dada::logMsg(2, $dl, "getBandToSend: found ".($#obs_finished+1)." observations marked obs.finished");

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && ($obs eq "none") && (!$quit_daemon)); $i++) 
  {
    $o = $obs_finished[$i];
    Dada::logMsg(2, $dl, "getBandToSend: checking ".$o);

    # skip if no obs.info exists
    if (!( -f $nfs_archives."/".$o."/obs.info")) 
    {
      Dada::logMsg(0, $dl, "Required file missing: ".$o."/obs.info");
      next;
    }

    # Find the PID 
    $cmd = "grep ^PID ".$nfs_archives."/".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getBandToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getBandToSend: ".$result." ".$response);
    if (($result ne "ok") || ($response eq "")) 
    {
      Dada::logMsgWarn($warn, "getBandToSend: failed to parse PID from obs.info [".$o."] ".$response);
      next;
    }
    $p = $response;

    
    # Get the sorted list of band nfs links for this obs
    # This command will timeout on missing NFS links (6 s), but wont print ones that are missing
    $cmd = "find -L ".$nfs_archives."/".$o." -mindepth 1 -maxdepth 1 -type d -printf \"%f\\n\" | sort";
    Dada::logMsg(3, $dl, "getBandToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getBandToSend: ".$result.":".$response);

    if ($result ne "ok") 
    {
      Dada::logMsgWarn($warn, "getBandToSend: failed to get list of NFS Band links ".$response);
      next;
    }

    @bands = split(/\n/, $response);
    Dada::logMsg(3, $dl, "getBandToSend: found ".($#bands+1)." bands in obs ".$o);

    # see if we can find a band that matches
    for ($j=0; (($j<=$#bands) && ($obs eq "none")); $j++) 
    {
      $b = $bands[$j];
      Dada::logMsg(3, $dl, "getBandToSend: checking ".$o."/".$b);

      if (-f $nfs_archives."/".$o."/".$b."/band.transferred")
      {
        Dada::logMsg(2, $dl, $o."/".$b."/band.transferred existed, skipping");
        next;
      }

      if (! -f $nfs_archives."/".$o."/".$b."/obs.start")
      {
        Dada::logMsgWarn($warn, $o."/".$b."/obs.start did not exist");
        next;
      }

      if (! -f $nfs_archives."/".$o."/".$b."/band.finished")
      {
        Dada::logMsgWarn($warn, $o."/".$b."/band.finished did not exist");
        next;
      }

      # if we have the required files 
      Dada::logMsg(2, $dl, "getBandToSend: found ".$o."/".$b);
      $obs = $o;
      $band = $b;
      $pid = $p;
    }

    if ($obs eq "none") 
    {
      # Check if all bands have been trasnferred successfully, if so, mark 
      # the observation as obs.transferred
      Dada::logMsg(2, $dl, "getBandToSend: checkAllBands(".$o.")");
      ($result, $response) = checkAllBands($o);
      Dada::logMsg(2, $dl, "getBandToSend: checkAllBands: ".$result." ".$response);

      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "getBandToSend: checkAllBands failed: ".$response);
      }
      else
      {
        if ($response ne "all bands sent")
        {
          Dada::logMsg(2, $dl, "getBandToSend: ".$o." not fully transferred: ".$response);
        }
        else
        {
          Dada::logMsg(0, $dl, $o." transferred");
          markState($o, "obs.finished", "obs.transferred", "");
          setStatus($o." xfer success");
        }
      }
    }
  }

  Dada::logMsg(2, $dl, "getBandToSend: returning ".$pid.", ".$obs.", ".$band);
  return ($pid, $obs, $band);
}

#
# Write the current status into the /nfs/control/apsr/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada::logMsg(2, $dl, "setStatus(".$message.")");

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $dir = "/nfs/control/apsr";
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
# Transfer the specified obs/band on port to hosts
#
sub transferBand($$$) {

  my ($p, $o, $b) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $a_dir = $cfg{"CLIENT_ARCHIVE_DIR"};

  my $rsync_options = "-a --stats --password-file=/home/apsr/.ssh/raid0_rsync_pw --no-g --chmod=go-ws ".
                      "--exclude 'band.finished' --exclude 'sent.to.*' --bwlimit=".BWLIMIT;

  my $l_dir = "";
  my $send_host = "";
  my $transfer_result = 1;
  my @sources = ();
  my $i = 0;
  my $j = 0;
  my $s = "";
  my @output_lines = ();
  my @bits = ();
  my $data_rate = "";

  Dada::logMsg(2, $dl, "Transferring ".$o."/".$b." to ".$r_user."@".$r_host.":".$r_path);

  # Find the host on which the band directory is located
  $cmd = "find ".$o."/".$b." -maxdepth 1 -printf \"\%l\" | awk -F/ '{print \$3}'";
  Dada::logMsg(2, $dl, "transferBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "transferBand: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($error, "could not determine host on which ".$o."/".$b." resides");
    return ("fail", "could not determine host on which ".$o."/".$b." resides");
  }
  # host to send from
  $send_host = $response;

  # determine the SOURCE[s] for this band
  $cmd = "find -L ".$o."/".$b." -mindepth 1 -maxdepth 1 -type d -printf '\%f\n' | sort";
  Dada::logMsg(2, $dl, "transferBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "transferBand: ".$result." ".$response);
  if (($result ne "ok") || ($response eq ""))
  {
    # not a multifold pulsar, so extract source from obs.start
    $cmd = "grep ^SOURCE ".$o."/".$b."/obs.start | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "transferBand: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "transferBand: ".$result." ".$response);
    if (($result ne "ok") || ($response eq ""))
    {
      Dada::logMsgWarn($warn, "transferBand: failed to determine SOURCE for ".$o.": ".$response);
      markState($o."/".$b, "band.finished", "band.bad", "could not determine SOURCE");
      next;
    }
    push @sources, $response;
    Dada::logMsg(2, $dl, "transferBand: found 1 source [".$response."]"); 
  }
  else
  {
    @sources = split(/\n/, $response);
    Dada::logMsg(2, $dl, "transferBand: found ".($#sources+1)." sources");
  }


  # create the PID directory on the destination
  $cmd = "mkdir -p -m 0755 ".$r_path."/".$p;
  Dada::logMsg(2, $dl, "transferBand: ".$r_user."@".$r_host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "transferBand: ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "transferBand: ssh failed: ".$response);
  }
  if ($rval != 0)
  {
    Dada::logMsgWarn($warn, "transferBand: mkdir failed: ".$response);
  }

  # transfer each source
  for ($i=0; $i<=$#sources; $i++)
  {
    $s = $sources[$i];

    # create the observation directory on the destination
    $cmd = "mkdir -p -m 0755 ".$r_path."/".$p."/".$s."; ".
           "mkdir -p -m 0755 ".$r_path."/".$p."/".$s."/".$o."; ".
           "mkdir -p -m 0755 ".$r_path."/".$p."/".$s."/".$o."/".$b;
    Dada::logMsg(2, $dl, "transferBand: ".$r_user."@".$r_host.":".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
    Dada::logMsg(2, $dl, "transferBand: ".$result." ".$rval." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "transferBand: ssh failed: ".$response);
    }
    if ($rval != 0)
    {
      Dada::logMsgWarn($warn, "transferBand: mkdir failed: ".$response);
    }

    Dada::logMsg(1, $dl, $o."/".$b." transferring");

    # If this is not a multi-fold obs, there will be no source subdirectory
    if ($#sources == 0)
    {
      $cmd = "rsync ".$a_dir."/".$o."/".$b."/ ".
                      $r_user."@".$r_host."::".$r_module."/".$p."/".$s."/".$o."/".$b."/ ".$rsync_options;
    }
    else
    {
      $cmd = "rsync ".$a_dir."/".$o."/".$b."/".$s."/ ".
                      $a_dir."/".$o."/".$b."/obs.start ".
                      $r_user."@".$r_host."::".$r_module."/".$p."/".$s."/".$o."/".$b."/ ".$rsync_options;
    }

    Dada::logMsg(2, $dl, "transferBand: ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand("apsr", $send_host, $cmd);
    Dada::logMsg(2, $dl, "transferBand: ".$result." ".$rval." ".$response);

    if (($result eq "ok") && ($rval == 0))
    {
      # determine the data rate
      @output_lines = split(/\n/, $response);
      $j = 0;
      for ($j=0; $j<=$#output_lines; $j++)
      {
        if ($output_lines[$j] =~ m/bytes\/sec/)
        {
          @bits = split(/[\s]+/, $output_lines[$j]);
          $data_rate = sprintf("%3.0f", ($bits[1] / 1048576))." MB @ ".sprintf("%3.0f", ($bits[6] / 1048576))." MB/s";
        }
      }
      Dada::logMsg(2, $dl, $s."/".$o."/".$b." finished -> transferred ".$data_rate);
      markRemoteFile($r_user, $r_host, $r_path."/".$p."/".$s."/".$o."/".$b."/band.transferred");
    }
    else
    {
      $transfer_result = 0;
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "transferBand: ssh failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsg(0, $dl, "transferBand: rsync failed: ".$response);
      }
      Dada::logMsgWarn($warn, "failed to transfer ".$o."/".$b);
    }
  }
  if ($transfer_result eq 1)
  {
    markState($o."/".$b, "band.finished", "band.transferred", "");
    Dada::logMsg(1, $dl, $o."/".$b." finished -> transferred ".$data_rate);
    return ("ok", "");
  }
  else
  {
    markState($o."/".$b, "band.finished", "band.bad", "transfer failed: ".$response);
    return ("fail", "");
  }
} 


#
# check the remote destination exists and has sufficient disk space
#
sub checkDest() 
{

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";

  # check the directory exists / is mounted on the remote host
  $cmd = "ls -1d ".$r_path;
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "checkDest: ssh failed: ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  }
  if ($rval != 0)
  {
    Dada::logMsg(0, $dl, "checkDest: ".$cmd." failed: ".$response);
    return ("fail", "remote dir ".$r_path." did not exist");
  }

  # check for disk space on this disk
  $cmd = "df ".$r_path." -P | tail -n 1";
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "checkDest: ssh failed: ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  }
  if ($rval != 0)
  {
    Dada::logMsg(0, $dl, "checkDest: ".$cmd." failed: ".$response);
    return ("fail", "df command failed on ".$r_user."@".$r_host);
  }

  Dada::logMsg(2, $dl, "checkDest: ".$response);

  if ($response =~ m/No such file or directory/)
  {
    Dada::logMsgWarn($error, $r_path." was not a valid directory on ".$r_host);
    return ("fail", "invalid r_path");
  }
  else
  {
    my ($location, $total, $used, $avail, $junk) = split(/ +/, $response);

    my $percent_free = $avail / $total;
    my $stop_percent = 0.05;

    Dada::logMsg(2, $dl, "checkDest: used=".$used.", avail=".$avail.", total=".$total." percent_free=".$percent_free." stop_percent=".$stop_percent);

    if ($percent_free < $stop_percent)
    {
      Dada::logMsg(2, $dl, "checkDest: ".$r_path." is over ".((1.00-$stop_percent)*100)." percent full");
      return ("fail", $r_path." is over ".((1.00-$stop_percent)*100)." percent full");
    }
    else
    {
      Dada::logMsg(2, $dl, "chekcDest: ".$r_path." is ".sprintf("%0.2f",($percent_free*100))." percent free");

      # Need more than 10 Gig
      if ($avail < 10000)
      {
        Dada::logMsgWarn($warn, $r_host.":".$r_path." has less than 10 GB left");
        return ("fail", "less than 100 GB remaining");
      }
      else
      {
        return  ("ok", "");
      }
    }
  }
}


#
# change the state of the specified band
#
sub markState($$$$)
{
  my ($dir, $from, $to, $message) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  if (! -f $dir."/".$from)
  {
    Dada::logMsgWarn($warn, "markState: from state [".$from."] did not exist");
  }
  else
  {
    $cmd = "rm -f ".$dir."/".$from;
    Dada::logMsg(3, $dl, "markState: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "markState: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
    } 
  }
  if ($message ne "")
  {
    $cmd = "echo '".$message."' > ".$to;
  }
  else
  {
    $cmd = "touch ".$dir."/".$to;
  }
  Dada::logMsg(3, $dl, "markState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "markState: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
    $result = "fail";
  }
  return $result;
}



sub markRemoteFile($$$) 
{
  my ($user, $host, $file) = @_;
  Dada::logMsg(2, $dl, "markRemoteFile(".$user.", ".$host.", ".$file.")");

  my $cmd = "touch ".$file;
  my $result = "";
  my $rval = 0;
  my $response = "";
  
  Dada::logMsg(2, $dl, "markRemoteFile: remoteSshCommand(".$user.", ".$host.", ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(2, $dl, "markRemoteFile: ".$result." ".$rval." ".$response);

  if (($result eq "ok") && ($rval eq 0))
  {
    return ("ok", "");
  } 
  else
  {
    if ($result ne "ok") 
    {
      Dada::logMsg(0, $dl, "markRemoteFile: ssh failed: ".$response);
    }
    if ($rval ne 0)
    {
      Dada::logMsg(0, $dl, "markRemoteFile: touch failed: ".$response);
    }
    Dada::logMsgWarn($warn, "could not touch ".$host.":".$file);
    return ("fail", "");
  }
}


#
# Looks for fully transferred observations to see if they have been deleted yet
#
sub checkFullyTransferred() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $n_bands = 0;
  my $n_deleted = 0;
  my $n_swin = 0;
  my $n_parkes = 0;
  my $o = "";
  my $obs_pid = "";

  Dada::logMsg(2, $dl, "checkFullyTransferred()");

  # Find all observations marked as obs.transferred at least 14 days ago...
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}." -maxdepth 2 -name 'obs.transferred' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyTransferred: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {
    $o = $observations[$i];

    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsgWarn($warn, "checkFullyTransferred: Required file missing ".$o."/obs.info");
      next;
    }
  
    # get the PID 
    $cmd = "grep ^PID ".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "checkFullyTransferred: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "could not determine PID");
    }
    $obs_pid = $response;

    # find out how many band directories we have
    $cmd = "ls -1d ".$o."/*/ | wc -l";
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);
  
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyTransferred: could not determine number of band directories: ".$response);
      next;
    }
    chomp $response;
    $n_bands = $response;

    # find out how many band.deleted files we have
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 2 -name 'band.deleted' | wc -l";
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyTransferred: could not count band.deleted files: ".$response);
      next;
    }
    chomp $response;
    $n_deleted = $response;

    Dada::logMsg(2, $dl, "checkFullyTransferred: n_bands=".$n_bands.", n_deleted=".$n_deleted);

    if ($n_deleted == $n_bands) {
      $cmd = "touch ".$o."/obs.deleted";
      Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkFullyTransferred: could not touch ".$o."/obs.deleted: ".$response);
      }

      if (-f $o."/obs.transferred") {
        Dada::logMsg(2, $dl, "checkFullyTransferred: removing ".$o."/obs.transferred");
        unlink $o."/obs.transferred";
      }
      Dada::logMsg(1, $dl, $o.": transferred -> deleted");
    }
  }
  
  return ("ok", "");
}

#
# Looks for observations that have been marked obs.deleted and moves them
# to /nfs/old_archives/apsr and /nfs/old_results/apsr if they are deleted
# and > 1 month old
#
sub checkFullyDeleted() {
  
  my $i = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";
  
  Dada::logMsg(2, $dl, "checkFullyDeleted()");
  
  # Find all observations marked as obs.deleted and > 30*24 hours since being modified 
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."  -maxdepth 2 -name 'obs.deleted' -mtime +30 -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";

  Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
  
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyDeleted: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }
  
  chomp $response; 
  my @observations = split(/\n/,$response);
  my @sources = ();
  my $n_band_tres = 0;
  my $req_band_tres = 0;
  my $n_bands = 0;
  my $o = "";
  my $curr_time = time;
  
  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {

    $o = $observations[$i];

    my @t = split(/-|:/,$o);
    my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

    Dada::logMsg(2, $dl, "checkFullyDeleted: testing ".$o." curr=".$curr_time.", unix=".$unixtime);
    # if UTC_START is less than 30 days old, dont delete it
    if ($unixtime + (30*24*60*60) > $curr_time) {
      Dada::logMsg(2, $dl, "checkFullyDeleted: Skipping ".$o.", less than 30 days old");
      next;
    }

    # find out how many band directories we have
    $cmd = "ls -1d ".$o."/*/ | wc -l";
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyDeleted : could not determine number of band directories: ".$response);
      next;
    }
    chomp $response;
    $n_bands = $response;
    Dada::logMsg(2, $dl, "checkFullyDeleted: n_bands=".$n_bands);

    # find out how many sources in this observation (i.e. multifold)
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 2 -type d -printf '\%f\\n' | sort | uniq";
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyDeleted : could not determine number of sources: ".$response);
      next;
    }
    @sources = ();
    if ($response ne "") {
      @sources = split(/\n/, $response);
      $req_band_tres = ($#sources + 1) * $n_bands;
    } else {
      $req_band_tres = $n_bands;
    }
    Dada::logMsg(2, $dl, "checkFullyDeleted: expected number of band.tres files =".$req_band_tres);

    # find out how many band.tres files we have
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 3 -name 'band.tres' | wc -l";
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyDeleted: could not count band.tres files: ".$response);
      next;
    }
    chomp $response;
    $n_band_tres = $response;
    Dada::logMsg(2, $dl, "checkFullyDeleted: n_band_tres=".$n_band_tres);

    # If we have all the required band.tres files, regenerated the band summed archives
    if ($n_band_tres != $req_band_tres) {
      Dada::logMsgWarn($warn, "checkFullyDeleted: num of band.tres mismatch for ".$o.": n_band_tres [".$n_band_tres."]  != req_band_tres [".$req_band_tres."]");
    } 

    $cmd = "/home/dada/psrdada/apsr/scripts/server_apsr_band_processor.pl ".$o;
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyDeleted: server_apsr_band_processor.pl ".$o." failed: ".$response);
      $cmd = "mv ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o."/obs.deleted ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o."/obs.failed";
      Dada::logMsg(1, $dl, "checkFullyDeleted: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $dl, "checkFullyDeleted: ".$result." ".$response);
      return ("fail", "Failed to process ".$o);
    }

    #$cmd = "mv ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o." /nfs/old_archives/apsr/".$o;
    #Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    #($result, $response) = Dada::mySystem($cmd);
    #Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);

    #if ($result ne "ok") {
    #  return ("fail", "failed to move ".$o." to old_archives");
    #}

    #$result = "ok";
    #$response = "";
    #$cmd = "mv ".$cfg{"SERVER_RESULTS_NFS_MNT"}."/".$o." /nfs/old_results/apsr/".$o;
    #Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    #($result, $response) = Dada::mySystem($cmd);
    #Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);

    #if ($result ne "ok") {
    #  return ("fail", "failed to move ".$o." to old_results");
    #}

    Dada::logMsg(1, $dl, $o.": deleted -> old");
  
  }
  return ("ok", "");
}

sub controlThread($$) {

  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) 
  {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) 
  {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  }
  else
  {
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
sub good($) 
{
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

  my ($result, $response) = checkDest();
  if ($result ne "ok") {
    return ("fail", "Error: ".$response);
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}
