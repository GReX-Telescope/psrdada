#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2016 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# Transfers UTMOST observations to Swinburne for archival
#

#
# Constants
#
use constant BANDWIDTH      => "10240"; # KB/s
use constant DATA_DIR       => "/data/mopsr";
use constant META_DIR       => "/data/mopsr";
use constant REQUIRED_HOST  => "mpsr-srv0";
use constant REQUIRED_USER  => "dada";

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Mopsr;

#
# function prototypes
#
sub good($);
sub getObsToSend();
sub getDest();
sub transferTB($$&);
sub transferFB($$$);
sub transferFB_allcand_only($);
sub markState($$$);

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
our $user = "pulsar";
our $host = "farnarkle1.hpc.swin.edu.au";
our $path = "/fred/oz002/utmost";
our $meta_path = "/home/pulsar/utmost_timing_aux";

our $results_dir = DATA_DIR."/results";
our $archives_dir = DATA_DIR."/archives";

#
# initialize globals
#
$dl = 1;
$daemon_name = Dada::daemonBaseName(basename($0));
%cfg = Mopsr::getConfig();
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

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $control_thread = 0;

  my $obs = "";
  my @srcs = ();
  my $freq = "";

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

  Dada::logMsg(1, $dl, "Starting UTMOST Transfer Manager");

  chdir DATA_DIR;

  while (!$quit_daemon)
  {
    # find and observation to send
    Dada::logMsg(2, $dl, "main: getObsToSend()");
    ($obs, $freq, @srcs) = getObsToSend();
    Dada::logMsg(2, $dl, "main: getObsToSend(): ".$obs." ".$freq." @srcs");

    if ($obs ne "none") 
    {
      # find a suitable destination disk that has enough space
      Dada::logMsg(2, $dl, "main: getDest()");
      ($result) = getDest();
      Dada::logMsg(2, $dl, "main: getDest() ".$result);

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

        # transfer the observation
        Dada::logMsg(2, $dl, "main: transferTB() ".$obs." to ".$user."@".$host.":/".$path);
        ($result, $response) = transferTB($obs, $freq, \@srcs);
        Dada::logMsg(2, $dl, "main: transferTB() ".$result." ".$response);

        if ($result ne "ok")
        {
          # If we have been asked to quit during the transferTB, then failure is expected
          if ($quit_daemon) {
            Dada::logMsg(2, $dl, "main: asked to quit");
          } 
          # this observation could not be transferred for some other reason
          else
          {
            Dada::logMsgWarn($warn, "main: transferTB failed: ".$response);
            Dada::logMsg(1, $dl, @srcs."/".$obs." finished -> failed");
            Dada::logMsg(2, $dl, "checkTransferred: markState(".$obs.", obs.finished, obs.transfer_error)");
            ($result, $response) = markState($obs, "obs.finished", "obs.transfer_error");
            Dada::logMsg(2, $dl, "checkTransferred: markState() ".$result." ".$response);
          }
        }
        else
        {
          Dada::logMsg(1, $dl, @srcs."/".$obs." finished -> transferred ".$response);
          Dada::logMsg(2, $dl, "checkTransferred: markState(".$obs.", obs.finished, obs.transferred)");
          ($result, $response) = markState($obs, "obs.finished", "obs.transferred");
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          # update status in the DB:
          $cmd = "asteria_utc.py --config-dir /home/dada/linux_64/share/ -U ".$obs;
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);

          # Add the utc / src to list of desired transfers
          my $transfer_list = $meta_path."/list";
          $cmd = "cp ".$transfer_list." ".$transfer_list.".tmp; echo ".$obs." @srcs >> ".$transfer_list.".tmp; mv ".$transfer_list.".tmp ".$transfer_list;
          Dada::logMsg(2, $dl, "transferTB: ".$cmd);
          ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
          Dada::logMsg(3, $dl, "transferTB: ".$result." ".$response);

          # Request a batch processing job
          $cmd = $path."/soft/bin/submit_timing_job.sh";
          Dada::logMsg(2, $dl, "transferTB: ".$cmd);
          ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
          Dada::logMsg(3, $dl, "transferTB: ".$result." ".$response);
        }
=begin comment
        # no longer transfer central FB
        else
        {
          # transfer central FB
          Dada::logMsg(2, $dl, "main: transferFB() ".$obs." to ".$user."@".$host."/".$path);
          ($result, $response) = transferFB($obs, @srcs, $freq);
          Dada::logMsg(2, $dl, "main: transferFB() ".$result." ".$response);

          if ($result ne "ok")
          {
            # If we have been asked to quit during the transferTB, then failure is expected
            if ($quit_daemon) {
              Dada::logMsg(2, $dl, "main: asked to quit");
            }
            # this observation could not be transferred for some other reason
            else
            {
              Dada::logMsgWarn($warn, "main: transferFB failed: ".$response);
              Dada::logMsg(1, $dl, @srcs."/".$obs." finished -> failed");
              Dada::logMsg(2, $dl, "checkTransferred: markState(".$obs.", obs.finished, obs.transfer_error)");
              ($result, $response) = markState($obs, "obs.finished", "obs.transfer_error");
              Dada::logMsg(2, $dl, "checkTransferred: markState() ".$result." ".$response);
            }
          }
          else
          {
            Dada::logMsg(1, $dl, @srcs."/".$obs." finished -> transferred ".$response);
            Dada::logMsg(2, $dl, "checkTransferred: markState(".$obs.", obs.finished, obs.transferred)");
            ($result, $response) = markState($obs, "obs.finished", "obs.transferred");
            # update status in the DB:
            $cmd = "asteria_utc.py --config-dir /home/dada/linux_64/share/ -U ".$obs;
            Dada::logMsg(3, $dl, "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          }
        }
=end comment

=cut

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
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
# Return the UTC of an observation, its centre frequency and a list of tied array beams
#
sub getObsToSend() 
{
  Dada::logMsg(3, $dl, "getObsToSend()");

  my $freq = "none"; # assuming the same freq for all tied beams
  my @srcs = ();
  my $obs = "none";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @utcs = ();
  my $utc = "";
  my @bits = ();
  my $i = 0;
  my $found_obs = 0;

  # find the oldest observation (based on UTC_START) to send
  $cmd = "find ".$results_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.finished'  -printf '\%h\n' | awk -F/ '{print \$(NF)}' | sort -n";
  Dada::logMsg(2, $dl, "getObsToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "getObsToSend: find failed: ".$response);
    return ($obs, $freq, @srcs);
  }

  @utcs = split(/\n/, $response);
  
  for ($i=0; (!$quit_daemon && !$found_obs && $i<=$#utcs ); $i++)
  {
    $utc = $utcs[$i];

    # get the boresight source name
    $cmd = "grep ^SOURCE ".$results_dir."/".$utc."/obs.info | awk '{print \$2}'";
    Dada::logMsg(2, $dl, "getObsToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
    if ($result ne "ok" || $response eq "")
    {
      Dada::logMsg(1, $dl, "getObsToSend: could not determine SOURCE in obs.info");
      next;
    }
    my $boresight_source = $response;

    # if the boresight source is a pulsar and we have filterbank data...
    if ( ($boresight_source =~ m/^J/) && (-d ( $archives_dir."/FB")) )
    {
      # look for obs.finished.#.# files
      $cmd = "find ".$archives_dir."/".$utc."/FB -mindepth 1 -maxdepth 1 -type f -name 'obs.finished.*.*' -printf '%f\n'";
      Dada::logMsg(2, $dl, "getObsToSend: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "getObsToSend: failed to count obs.finished.*.* files");
        next;
      }
      if ($response eq "")
      {
        Dada::logMsg(1, $dl, "getObsToSend: no FB obs.finished.*.* files found yet");
        next;
      }

      my @files = split(/\n/, $response);
      my $file = $files[0];
      my @file_parts = split (/\./, $file);
      my $nfiles_expected = $file_parts[$#file_parts];
      my $nfiles_actual = $#file_parts + 1;
      Dada::logMsg(1, $dl, "getObsToSend: ".$nfiles_actual." of ".$nfiles_expected." found");

      if ($nfiles_actual < $nfiles_expected)
      {
        Dada::logMsg(1, $dl, "getObsToSend: still waiting for FB files to be transferred");
        next;
      }
      $found_obs = 1;
      $obs = $utc;
      push @srcs, "FB";
    }

    if ($found_obs == 0)
    {
      # Tied Beams should have a Jname as a subdirectory
      $cmd = "find ".$results_dir."/".$utc." -mindepth 1 -maxdepth 1 -type d -printf '%f\n'";
      Dada::logMsg(3, $dl, "getObsToSend: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);

      if (($result ne "ok") || ($response eq ""))
      {
        Dada::logMsg(2, $dl, "getObsToSend: could not find and subdirs of ".$utc);
        next;
      }

      my @sources = split(/\n/, $response);
      my $j;
      for ($j=0; ($j<=$#sources); $j++)
      {
        if (-f $results_dir."/".$utc."/".$sources[$j]."/obs.header")
        {
          if ($sources[$j] =~ m/^J/)
          {
            $obs = $utc;
            # $src = $sources[$j];
            push @srcs, $sources[$j];
          
            $cmd = "grep ^FREQ ".$results_dir."/".$obs."/".$sources[$j]."/obs.header | awk '{print \$2}'";
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
            if ($result ne "ok")
            {
              Dada::logMsgWarn($warn, "getObsToSend: could not extrat FREQ from ".$obs."/TB/obs.header");
              next;
            }
            $freq = $response;
            $found_obs = 1;
          }
        }
      }

      # this observation cannot be transferred, change from finished to completed 
      if (!$found_obs)
      {
        $cmd = "mv ".$results_dir."/".$utc."/obs.finished ".$results_dir."/".$utc."/obs.completed";
        Dada::logMsg(2, $dl, "getObsToSend: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "getObsToSend: ".$result." ".$response);
        Dada::logMsg(1, $dl, $utc." finished -> completed");
        # update status in the DB:
        $cmd = "asteria_utc.py --config-dir /home/dada/linux_64/share/ -U ".$utc;
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
      }
    }
  }

  Dada::logMsg(2, $dl, "getObsToSend: returning ".$obs." ".$freq." @srcs");
  return ($obs, $freq, @srcs);
}

#
# Transfers the specified observation to the specified destination
#
sub transferTB($$&) 
{
  my ($obs, $freq, @tmp) = @_;
  my @srcs = @{$tmp[0]};
  transferFB_allcand_only($obs);
  Dada::logMsg(2, $dl, "transferTB S: @srcs");

  my $cmd = "";
  my $xfer_result = "ok";
  my $xfer_response = "ok";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $rsync_options = "-az --stats --no-g --chmod=go-ws --bwlimit ".BANDWIDTH.
                      " --exclude 'obs.finished' --inplace ";
  # loop through tied array beams
  foreach (@srcs) {
    my $src = $_;
    Dada::logMsg(2, $dl, "transferTB S: ".$src);

    # create the remote destination direectories
    $cmd = "mkdir -m 0755 -p ".$path."/TB/".$src."; ".
    "mkdir -m 0755 -p ".$path."/TB/".$src."/".$obs."; ".
    "mkdir -m 0755 -p ".$path."/TB/".$src."/".$obs."/".$freq;

    Dada::logMsg(2, $dl, "transferTB: ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "transferTB: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "transferTB: ssh for ".$user."@".$host." failed: ".$response);
      return ("fail", "ssh failed: ".$response);
    }
    if ($rval != 0)
    {
      Dada::logMsgWarn($warn, "transferTB: failed to create ".$src."/".$obs." directory: ".$response);
      return ("fail", "could not create remote dir");
    }

    $transfer_kill = "pkill -f '^rsync ".$archives_dir."/".$obs."'";

    # rsync the results dir 
    $cmd = "rsync ".$results_dir."/".$obs."/".$src."/*.tot ".$results_dir."/".$obs."/".$src."/obs.header ".$results_dir."/".$obs."/molonglo_modules.txt ".$user."@".$host.":".$path."/TB/".$src."/".$obs."/".$freq."/ ".$rsync_options;
    Dada::logMsg(2, $dl, "transferTB: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "transferTB: ".$result." ".$response);

    # rsync the archives dir
    $cmd = "rsync ".$archives_dir."/".$obs."/".$src."/* ".$user."@".$host.":".$path."/TB/".$src."/".$obs."/".$freq."/ ".$rsync_options;
    Dada::logMsg(2, $dl, "transferTB: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "transferTB: ".$result." ".$response);

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

      Dada::logMsg(2, $dl, $src."/".$obs." ".$response);

      $cmd = "touch ".$path."/TB/".$src."/".$obs."/".$freq."/obs.transferred";
      Dada::logMsg(2, $dl, "transferTB: remoteSsh(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferTB: ".$result." ".$rval." ".$response);
      if (   $result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferTB: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "transferTB: failed to touch obs.transferred on: ".$path."/TB/".$src."/".$obs);
        return ("fail", "failed to touch obs.transferred on remote obs");
      }

      $cmd = "chmod -R a-w ".$path."/TB/".$src."/".$obs;
      Dada::logMsg(2, $dl, "transferTB: remoteSsh(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferTB: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferTB: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "transferTB: failed to remove write permissions on: ".$path."/TB/".$src."/".$obs);
        return ("fail", "failed to remove write premissions on remote obs");
      }
    }
    else 
    {
      if ($quit_daemon)
      {
        Dada::logMsg(1, $dl, "transferTB: rsync interrupted");
        $xfer_response = "rsync interrupted for quit";
      }
      else
      {
        Dada::logMsg(0, $dl, "transferTB: rsync failed for ".$src."/".$obs.": ".$response);
        $xfer_response = "rsync failure";
      }
      $xfer_result = "fail";

      # Delete the partially transferred observation
      Dada::logMsg(1, $dl, "transferTB: rsync failed, deleting partial transfer at ".$path."/TB/".$src."/".$obs);
      $cmd = "rm -rf ".$path."/TB/".$src."/".$obs;
      Dada::logMsg(2, $dl, "transferTB: remoteSsh(".$user.", ".$host.", ".$cmd.")"); 
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferTB: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferTB: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0) 
      {
        Dada::logMsgWarn($warn, "transferTB: failed to delete partial transfer at: ".$path."/TB/".$src."/".$obs);
        return ("fail", "failed to delete partial transfer for: ".$path."/TB/".$src."/".$obs);
      }
    }
  }

  return ($xfer_result, $xfer_response);
}

sub transferFB_allcand_only($)
{
  my ($obs) = @_;

  my $cmd = "";
  my $xfer_result = "ok";
  my $xfer_response = "ok";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $rsync_options = "-a --stats --no-g --chmod=go-ws --bwlimit ".BANDWIDTH.
                      " --exclude 'obs.finished' ";

  # create the remote destination direectories
  my $remote_path = $path."/FB/all_candidates/".$obs."/";
  $cmd = "mkdir -m 0755 -p ".$remote_path;
       
  Dada::logMsg(2, $dl, "transferFB_allcand_only: ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "transferFB_allcand_only: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "transferFB_allcand_only: ssh for ".$user."@".$host." failed: ".$response);
    return ("fail", "ssh failed: ".$response);
  }
  if ($rval != 0)
  {
    Dada::logMsgWarn($warn, "transferFB_allcand_only: failed to create ".$remote_path." directory: ".$response);
    return ("fail", "could not create remote dir");
  }

  $transfer_kill = "pkill -f '^rsync ".$results_dir."/".$obs."'";

  # find the all_candidates.dat
  $cmd = "ls -1d ".$results_dir."/".$obs."/FB/all_candidates.dat | awk -F/ '{print \$NF}'";
  Dada::logMsg(2, $dl, "transferFB_allcand_only: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "transferFB_allcand_only: ".$result." ".$response);

  if (($result eq "ok") && ($response ne ""))
  {
    # rsync the all_candidates.dat
    $cmd = "rsync ".$results_dir."/".$obs."/FB/all_candidates.dat ".$user."@".$host.":".$remote_path." ".$rsync_options;

    Dada::logMsg(2, $dl, "transferFB_allcand_only: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "transferFB_allcand_only: ".$result." ".$response);

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

      Dada::logMsg(2, $dl, $remote_path." ".$response);

      # remote write permissions on transferred observation
      $cmd = "chmod -R a-w ".$remote_path;
      Dada::logMsg(2, $dl, "transferFB_allcand_only: remoteSsh(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferFB_allcand_only: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferFB_allcand_only: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "transferFB_allcand_only: failed to remove write permissions on: ".$remote_path.": ".$response);
        return ("fail", "failed to remove write premissions on remote obs");
      }
    }
    else 
    {
      if ($quit_daemon)
      {
        Dada::logMsg(1, $dl, "transferFB_allcand_only: rsync interrupted");
        $xfer_response = "rsync interrupted for quit";
      }
      else
      {
        Dada::logMsg(0, $dl, "transferFB_allcand_only: rsync failed for ".$obs.": ".$response);
        $xfer_response = "rsync failure";
      }
      $xfer_result = "fail";

      # Delete the partially transferred observation
      Dada::logMsg(1, $dl, "transferFB_allcand_only: rsync failed, deleting partial transfer at ".$remote_path);
      $cmd = "rm -rf ".$remote_path;
      Dada::logMsg(2, $dl, "transferFB_allcand_only: remoteSsh(".$user.", ".$host.", ".$cmd.")"); 
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferFB_allcand_only: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferFB_allcand_only: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0) 
      {
        Dada::logMsgWarn($warn, "transferFB_allcand_only: failed to delete partial transfer at: ".$remote_path);
        return ("fail", "failed to delete partial transfer for: ".$remote_path);
      }
    }
  } else {
    Dada::logMsg(1, $dl, "transferFB_allcand_only: couldn't find all_candidates.dat for ".$obs);
  }

  return ($xfer_result, $xfer_response);
}

#
# Transfers the specified observation to the specified destination
#
sub transferFB($$$) 
{
  my ($obs, $src, $freq) = @_;

  my $cmd = "";
  my $xfer_result = "ok";
  my $xfer_response = "ok";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $rsync_options = "-a --stats --no-g --chmod=go-ws --bwlimit ".BANDWIDTH.
                      " --exclude 'obs.finished' ";

  # create the remote destination direectories
  $cmd = "mkdir -m 0755 -p ".$path."/FB/".$src."; ".
         "mkdir -m 0755 -p ".$path."/FB/".$src."/".$obs."; ".
         "mkdir -m 0755 -p ".$path."/FB/".$src."/".$obs."/".$freq;
       
  Dada::logMsg(2, $dl, "transferFB: ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "transferFB: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "transferFB: ssh for ".$user."@".$host." failed: ".$response);
    return ("fail", "ssh failed: ".$response);
  }
  if ($rval != 0)
  {
    Dada::logMsgWarn($warn, "transferFB: failed to create ".$src."/".$obs." directory: ".$response);
    return ("fail", "could not create remote dir");
  }

  $transfer_kill = "pkill -f '^rsync ".$archives_dir."/".$obs."'";

  # find the "central" fan beam
  $cmd = "ls -1d ".$archives_dir."/".$obs."/FB/BEAM_??? | awk -F/ '{print \$NF}'";
  Dada::logMsg(2, $dl, "transferFB: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "transferFB: ".$result." ".$response);

  if (($result eq "ok") && ($response ne ""))
  {
    my @beams = split(/\n/, $response);
    my $nbeams = $#beams;
    my $ibeam = ($nbeams / 2 ) + 1;
    my $centre_beam = $beams[$ibeam];

    # rsync the archives dir 
    $cmd = "rsync ".$archives_dir."/".$obs."/FB/".$centre_beam." ".$user."@".$host.":".$path."/FB/".$src."/".$obs."/".$freq."/ ".$rsync_options;
    Dada::logMsg(2, $dl, "transferFB: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "transferFB: ".$result." ".$response);

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

      Dada::logMsg(2, $dl, $src."/".$obs." ".$response);

      # remote write permissions on transferred observation
      $cmd = "chmod -R a-w ".$path."/FB/".$src."/".$obs."/".$freq."/".$centre_beam;
      Dada::logMsg(2, $dl, "transferFB: remoteSsh(".$user.", ".$host.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferFB: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferFB: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "transferFB: failed to remove write permissions on: ".$path."/FB/".$src."/".$obs.": ".$response);
        return ("fail", "failed to remove write premissions on remote obs");
      }
    }
    else 
    {
      if ($quit_daemon)
      {
        Dada::logMsg(1, $dl, "transferFB: rsync interrupted");
        $xfer_response = "rsync interrupted for quit";
      }
      else
      {
        Dada::logMsg(0, $dl, "transferFB: rsync failed for ".$src."/".$obs.": ".$response);
        $xfer_response = "rsync failure";
      }
      $xfer_result = "fail";

      # Delete the partially transferred observation
      Dada::logMsg(1, $dl, "transferFB: rsync failed, deleting partial transfer at ".$path."/FB/".$src."/".$obs);
      $cmd = "rm -rf ".$path."/FB/".$src."/".$obs;
      Dada::logMsg(2, $dl, "transferFB: remoteSsh(".$user.", ".$host.", ".$cmd.")"); 
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "transferFB: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "transferFB: ssh ".$user."@".$host." for ".$cmd." failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed: ".$response);
      }
      if ($rval != 0) 
      {
        Dada::logMsgWarn($warn, "transferFB: failed to delete partial transfer at: ".$path."/FB/".$src."/".$obs);
        return ("fail", "failed to delete partial transfer for: ".$path."/FB/".$src."/".$obs);
      }
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

  # test how much space is remaining on this disk
  $cmd = "df -B 1048576 -P ".$path;
  Dada::logMsg(3, $dl, "getDest: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd, "", "tail -n 1 | awk '{print \$4}'");
  Dada::logMsg(3, $dl, "getDest: ".$result." ".$rval." ".$response);

  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "getDest: ssh to ".$user."@".$host." failed: ".$response);
    return ("fail");
  }

  # check there is 1 TB free
  if (int($response) < (1024*1024)) 
  {
    Dada::logMsg(1, $dl, "getDest: less than 1TB remaining on ".$user."@".$host.":".$path);
    return ("fail");
  }

  Dada::logMsg(2, $dl, "getDest: found ".$user."@".$host.":".$path);
  return ("ok");
}

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

  my $from_path = $results_dir."/".$o."/".$from;
  my $to_path = $results_dir."/".$o."/".$to;

  if (! -f $from_path) {
    Dada::logMsgWarn($warn, "markState: \$from file did not exist");
    return ("fail", "\$from file did not exist");
  }
  if ( -f $to_path) {
    Dada::logMsgWarn($warn, "markState: \$to file already existed");
    return ("fail", "\$to file already existed");
  }

  my $cmd = "rm -f ".$from_path;
  Dada::logMsg(2, $dl, "markState: ".$cmd);
  my ($result, $response) = Dada::mySystem($cmd);
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
