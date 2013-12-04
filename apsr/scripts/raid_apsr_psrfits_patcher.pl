#!/usr/bin/env perl

###############################################################################
#
# apsr_psrfits_patcher.pl
#
# Fill in missing header parameters for a psrfits file
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use Apsr;
use Dada;
use File::Basename;
use threads;
use threads::shared;


#
# Constants
#
use constant PROCESSING_HOST  => "raid0";
use constant DATA_DIR         => "/lfs/raid0/apsr";
use constant META_DIR         => "/lfs/data0/apsr";


#
# Function Prototypes
#
sub good($);
sub controlThread($$);

#
# Globals
#
our $dl;
our $quit_daemon : shared;
our $daemon_name;
our $src_path;
our $dst_path;
our $err_path;
our $warn : shared;
our $error : shared;

#
# Initialize globals
#
$dl = 1; 
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName(basename($0));
$src_path = DATA_DIR."/psrfits/unpatched";
$dst_path = DATA_DIR."/atnf/send";
$err_path = DATA_DIR."/psrfits/fail_patch";

$warn     = META_DIR."/logs/".$daemon_name.".warn";
$error    = META_DIR."/logs/".$daemon_name.".error";

#
# Main
#
{
  my $log_file = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file = META_DIR."/control/".$daemon_name.".quit";

  my $control_thread = 0;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $d = 0;
  my $i = 0;
  my $j = 0;
  my $k = 0;
  my @bits = ();

  my $obs = "";
  my $pid = "";
  my $src = "";
  
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

  Dada::logMsg(2, $dl, "Searching for unprocessed observations...");

  while (!$quit_daemon)
  {
    # get the oldest PSRFITS file to process
    $cmd = "find ".$src_path." -mindepth 3 -maxdepth 3 -type f -name '*.?f' -printf '\%f\n' | sort | head -n 1";
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$response);
    if (($result ne "ok") || ($response eq ""))
    {
      Dada::logMsg(2, $dl ,"main: no observations to process");
      Dada::logMsg(2, $dl, "Sleeping 60 seconds");
      my $counter = 60;
      while ((!$quit_daemon) && ($counter > 0)) 
      {
        sleep(1);
        $counter--;
      }
      next;
    }

    $obs = $response;

    # get the PID, and SOURCE for this observation
    $cmd = "ls -1d ".$src_path."/*/*/".$obs;
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$response);

    @bits = split(/\//, $response);
    if ($#bits < 2) 
    {
      Dada::logMsg(0, $dl ,"main: not enough components in path for ".$obs);
      $quit_daemon = 1;
      next;
    }

    $pid = $bits[$#bits-2];
    $src = $bits[$#bits-1];

    Dada::logMsg(2, $dl ,"main: processing ".$pid."/".$src."/".$obs);
    ($result, $response) = patchObservation($pid, $src, $obs);
    Dada::logMsg(2, $dl ,"main: ".$result." ".$response);

    if ($result eq "ok")
    {
      Dada::logMsg(0, $dl , $pid."/".$src."/".$obs." psrfits/unpatched -> atnf/send");
      ($result, $response) = moveObs($src_path, $dst_path, $pid, $src, $obs);
    }
    else
    { 
      Dada::logMsg(0, $dl , $pid."/".$src."/".$obs." psrfits/unpatched -> psrfits/fail_patch [".$response."]");
      ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
    }
  }


  $control_thread->join();
  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  exit 0;
}

###############################################################################
#
# process the observation creating the fits file from the timer archives
#
sub patchObservation($$$) 
{
  # observation tag [PID/SOURCE/PSRFITS]
  my ($p, $s, $u) = @_;

  my %h = ();
  my %dfb3 = ();
  my %current = ();
  my $result = "";
  my $response = "";
  my @keys = ();
    

  # These are considered constants
  $h{"site"}         = "PARKES";
  $h{"be:nrcvr"}     = "2";
  $h{"be:phase"}     = "1";  # From DSB header parameter?
  $h{"be:tcycle"}    = "10";
  $h{"ext:trk_mode"} = "TRACK";
  $h{"sub:nsblk"}    = "1";

  # There must be extracted from a sister DFB3 archive
  ($result, $response, %dfb3) = getDFB3PSRFITSHeader($p, $s, $u);
  if ($result ne "ok")
  {
    Dada::logMsg(2, $dl, "patchObservation: getDFB3PSRFITSHeader failed: ".$response);
    return ("fail", $response);
  }
  @keys = keys (%dfb3);
  if ($#keys != 10)
  {
    Dada::logMsg(0, $dl, "patchObservation: did not extract 11 required keys from DFB3 archive");
    return ("fail", "could not extract required keys from PDFB3 archive");
  }
  $h{"rm"}           = $dfb3{"rm"};
  $h{"obs:observer"} = "'".$dfb3{"obs:observer"}."'";
  $h{"obs:observer"} =~ s/,/ /g;
  $h{"obs:observer"} =~ s/;//g;
  $h{"itrf:ant_x"}   = $dfb3{"itrf:ant_x"};
  $h{"itrf:ant_y"}   = $dfb3{"itrf:ant_y"};
  $h{"itrf:ant_z"}   = $dfb3{"itrf:ant_z"};
  $h{"rcvr:hand"}    = $dfb3{"rcvr:hand"};
  $h{"rcvr:sa"}      = $dfb3{"rcvr:sa"};
  $h{"rcvr:rph"}     = $dfb3{"rcvr:rph"};
  $h{"ext:bpa"}      = $dfb3{"ext:bpa"};
  $h{"ext:bmaj"}     = $dfb3{"ext:bmaj"};
  $h{"ext:bmin"}     = $dfb3{"ext:bmin"};

  # These already exist in the current archive under different param name
  %current = getCurrentPSRFITSHeader($p, $s, $u);
  $h{"ext:obsfreq"}  = $current{"freq"};
  $h{"ext:obsbw"}    = $current{"bw"};
  $h{"ext:obsnchan"} = $current{"nchan"};
  $h{"ext:stp_crd1"} = $current{"ext:stt_crd1"};
  $h{"ext:stp_crd2"} = $current{"ext:stt_crd2"};
  $h{"ext:stt_date"} = "20".substr($u, 1, 2)."-".substr($u, 3, 2)."-".substr($u, 5, 2);
  $h{"ext:stt_time"} = substr($u, 8, 2).":".substr($u, 10, 2).":".substr($u, 12, 2);

  my $key = "";
  my $value = "";
  my @keys = sort keys (%h);
  my $i=0;
  my $key = $keys[0];
  my $val = $h{$key};
  my $cmd = "psredit -m -a itrf -c ".$key."=".$val;

  for ($i=1; $i<=$#keys; $i++)
  {
    $key = $keys[$i];
    $val = $h{$key};
    if ($val eq "nan")
    {
      $val = 0;
    }
    $cmd .= ",".$key."=".$val;
  }
  $cmd .= " ".$src_path."/".$p."/".$s."/".$u;

  Dada::logMsg(2, $dl, "patchObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "patchObservation: ".$result." ".$response); 
  if (($result ne "ok") || ($response =~ m/ailed/) || ($response =~ m/rror/))
  {
    Dada::logMsg(0, $dl, "patchObservation: failed to apply psredit command: ".$response);
    return ("fail", "psredit failed to patch file");
  }   
  return ("ok", "patched"); 
}

sub getCurrentPSRFITSHeader($$$)
{

  # observation tag [PID/SOURCE/PSRFITS file]
  my ($p, $s, $u) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my %header = ();
  my @lines = ();
  my @parts = ();
  my $i = 0;

  $cmd = "psredit ".$src_path."/".$p."/".$s."/".$u;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    return %header; 
  }

  @lines = split(/\n/, $response);
  for ($i=0; $i<=$#lines; $i++)
  {
    @parts = split (/\s\s+/, $lines[$i]);
    if ($#parts >= 1) {
      $header{$parts[0]} = $parts[2];
    } else {
      $header{$parts[0]} = "";
    }
  }
  return %header;
}

sub getDFB3PSRFITSHeader($$$)
{
  # observation tag [PID/SOURCE/PSRFITS file]
  my ($p, $s, $u) = @_;

  # try to find a file that is similar at lagavulin:/nfs/PKCCC3_1
  my $parkes_user = "pulsar";
  my $parkes_host = "lagavulin.atnf.csiro.au";
  my $parkes_p1 = "/nfs/PKCCC3_1";
  my $parkes_p2 = "/nfs/PKCCC3_2";
  my $epping_user = "pulsar";
  my $epping_host = "herschel.atnf.csiro.au";
  #my $epping_path = "/u/kho018/Projects/fix_apsr_data";
  my $epping_path = "/pulsar/archive21/dfb3_file_listing";
  my $max_time_diff = 60; # seconds difference between APSR and DFB3 start times
  my %header = ();

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  my $remote_user = "";
  my $remote_host = "";
  my $remote_file_list = "";

  # The strings that will be used to search for the matching files on the remote disks
  my $fs_string = "";
  my $grep_string = "";
  ($fs_string, $grep_string) = getDiffTime($u, $max_time_diff);
  Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: fs_string=".$fs_string." grep_string=".$grep_string);

  # Search the parkes disks for a matching file
  $cmd = "ls -1 ".$parkes_p1."/".$fs_string." ".$parkes_p2."/".$fs_string;
  Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$parkes_user."@".$parkes_host.": ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($parkes_user, $parkes_host, $cmd);
  Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "getDFB3PSRFITSHeader: ssh ".$parkes_user."@".$parkes_host." failed: ".$response);
    return ("fail", "could not ssh to ".$parkes_user."@".$parkes_host, %header);
  }

  # If the ls command failed or didn't find any archives, try the same command at epping
  if (($rval != 0) || ($response eq "") || ($response =~ m/No such file or directory/))
  {
    #$cmd = "grep -h \"".$grep_string."\" dfb4_files_09_sorted.lis dfb5_files_sorted.lis dfb3.lis.2012-01-23-13_52_47 | awk '{print \$1}'";
    $cmd = "grep -h \"".$grep_string."\" latest.lis | awk '{print \$1}'";
    Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$epping_user."@".$epping_host.":".$epping_path.";".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($epping_user, $epping_host, $cmd, $epping_path);
    Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$result." ".$rval." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "getDFB3PSRFITSHeader: ssh ".$epping_user."@".$epping_host." failed: ".$response);
      return ("fail", "could not ssh to ".$epping_user."@".$epping_host, %header);
    }
    if (($rval != 0) || ($response eq ""))
    {
      Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: could not find matching DFB3 archive with \"".$grep_string."\"");
      return ("fail", "no archive at epping", %header);
    }
    else
    {
      $remote_host = $epping_host;
      $remote_user = $epping_user;
      $remote_file_list = $response;
    }
  }
  else
  {
    $remote_host = $parkes_host;
    $remote_user = $parkes_user;
    $remote_file_list = $response;
  }
 
  my $file_unix = psrfitsUnixTime($u);
  my $file = "";
  my $diff = 0;
  my $time = 0;
  my $min_time = 0;
  my $sister_file = "";
  my $i = 0;

  my @matching_files =  split(/\n/, $remote_file_list);
  $min_time = $max_time_diff;
  for ($i=0; $i<=$#matching_files; $i++)
  { 
    $file = basename($matching_files[$i]);
    Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: testing ".$file);
    $time = psrfitsUnixTime($file);
    $diff = abs ($file_unix - $time);
    if ($diff < $min_time)
    {
      Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: testing ".$file." is new closest with diff=".$diff);
      $sister_file = $matching_files[$i];
      $min_time = $diff;
    }
  }

  if ($sister_file eq "")
  {
    return ("fail", "no archive within ".$max_time_diff." s");
  }

  $cmd = "psredit ".$sister_file." | grep -E '^rm |itrf:ant_x|itrf:ant_y|itrf:ant_z|rcvr:hand|rcvr:sa|rcvr:rph|ext:bpa|ext:bmaj|ext:bmin|obs:observer'";
  if ($remote_host eq $parkes_host)
  {
    $cmd = "source /nfs/psr1/.login; ".$cmd;
  }

  Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$remote_user."@".$remote_host.": ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($remote_user, $remote_host, $cmd);
  Dada::logMsg(2, $dl, "getDFB3PSRFITSHeader: ".$result." ".$rval." ".$response);

  if (($result eq "ok") && ($rval == 0))
  {
    my @lines = split(/\n/, $response);
    for ($i=0; $i<=$#lines; $i++)
    { 
      my @parts = split (/\s\s+/, $lines[$i]);
      if ($#parts >= 1) {
        $header{$parts[0]} = $parts[2];
      } else {
        $header{$parts[0]} = "";
      }
    }
  }
  else
  {
    Dada::logMsg(0, $dl, "getDFB3PSRFITSHeader: ".$cmd);
    Dada::logMsg(0, $dl, "getDFB3PSRFITSHeader: psredit header extraction failed: ".$response);
    return ("fail", "could not extract header parameters from ".$sister_file, %header);
  }

  return ("ok", "extracted from ".$remote_user."@".$remote_host.":".$sister_file, %header);
}

#
# Get the time string to be used for searching for matching files
#
sub getDiffTime($$)
{
  my ($file, $max) = @_;

  my $file_unix = psrfitsUnixTime($file);

  my $min_unix = $file_unix - $max;
  my $max_unix = $file_unix + $max;

  my $min_time = psrfitsPrintTime($min_unix);
  my $max_time = psrfitsPrintTime($max_unix);

  my $min_c = "";
  my $max_c = "";
  my $i=0;

  my $file_sys_string = "s";
  my $grep_string = "s";

  for ($i=0; $i<length($min_time); $i++)
  {
    $min_c = substr($min_time,$i,1);
    $max_c = substr($max_time,$i,1);

    if ($min_c eq $max_c)
    {
      $file_sys_string .= $min_c;
      $grep_string .= $min_c;
    }
    else
    {
      $file_sys_string .= "[".$min_c."-".$max_c."]*.?f";
      $grep_string .= "[".$min_c."-".$max_c."][_0-9]*\.[cr]f";
      $i = length($min_time);
    }
  }

  return ($file_sys_string, $grep_string);
}

#
# Determine unix time from PSRFITS filename
#
sub psrfitsUnixTime($)
{
  my ($file) = @_;

  my $year = "20".substr($file,1,2);
  my $month = substr($file,3,2);
  my $day = substr($file,5,2);
  my $hour = substr($file,8,2);
  my $min = substr($file,10,2);
  my $sec = substr($file,12,2);

  my $dada_time = $year."-".$month."-".$day."-".$hour.":".$min.":".$sec;
  my $unix_time = Dada::getUnixTimeUTC($dada_time); 
  return $unix_time;
}

sub psrfitsPrintTime($)
{
  my ($time) = @_;
  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = gmtime ($time);

  $year += 1900;
  $mon++;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);

  return substr($year,2,2).$mon.$mday."_".$hour.$min.$sec;
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
    $cmd = "mkdir -m 0755 -p ".$to."/".$pid."/".$src;
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

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsg(0, $dl, "controlThread: PID file did not exist on script exit");
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
  if ($sigName ne "INT") 
  {
    print STDERR $daemon_name." : Exiting\n";
    exit 1;
  }
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

  # this script can *only* be run on the configured server
  if (index(PROCESSING_HOST, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".PROCESSING_HOST.
                    ", not ".Dada::getHostMachineName());
  }

  return ("ok", "");
}
