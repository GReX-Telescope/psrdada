#!/usr/bin/env perl

###############################################################################
#
# raid_caspsr_psrfits_converter.pl
#
# Process CASPSR files converting archives to the atnf requested format
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use Caspsr;
use Dada;
use File::Basename;
use threads;
use threads::shared;


#
# Constants
#
use constant DATA_DIR       => "/lfs/raid0/caspsr";
use constant META_DIR       => "/lfs/data0/caspsr";
use constant REQUIRED_HOST  => "raid0";
use constant REQUIRED_USER  => "caspsr";
use constant TSCRUNCH_SECONDS => 32;

#
# Function Prototypes
#
sub good($);
sub processOne($$$);
sub processObservation($$$);
sub moveObs($$$$$);

#
# Globals
#
our $dl;
our $quit_daemon : shared;
our $daemon_name;
our $src_path;
our $dst_path;
our $err_path;
our $tmp_path;
our $fin_path;
our $warn : shared;
our $error : shared;

#
# initialize package globals
#
$dl = 1; 
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName(basename($0));
$src_path = DATA_DIR."/swin/sent";
$dst_path = DATA_DIR."/atnf/send";
$tmp_path = DATA_DIR."/psrfits/temp";
$err_path = DATA_DIR."/psrfits/fail";
$fin_path = DATA_DIR."/archived";

$warn     = META_DIR."/logs/".$daemon_name.".warn";
$error    = META_DIR."/logs/".$daemon_name.".error";

#
# check the command line arguments
#
my $manual_obs = "none";

if ($#ARGV==0) 
{
  # should be of the form PID / SOURCE / UTC_START
  ($manual_obs) = @ARGV;
}

if ($manual_obs ne "none") {

  my @bits = split(/\//, $manual_obs);
  if ($#bits != 2)
  {
    Dada::logMsg(0, $dl ,"arguments must be PID/SOURCE/UTC_START");
    exit(0);
  }
  my $pid = $bits[0];
  my $src = $bits[1];
  my $obs = $bits[2];
  processOne($pid, $src, $obs);

} else {

  processLoop();

}

exit(0);


###############################################################################
#
# process just one observation
#
sub processOne($$$)
{
  my ($pid, $src, $obs) = @_;

  Dada::logMsg(2, $dl, "processOne(".$pid.", ".$src.", ".$obs.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  # check that the specified observation is available to be processed
  if ( ! -d $src_path."/".$pid."/".$src."/".$obs ) 
  {
    Dada::logMsg(0, $dl ,"processOne: observation did not exist");
    return 1;
  }

  Dada::logMsg(2, $dl ,"processOne: processObservation(".$pid.", ".$src.", ".$obs.")");
  ($result, $response) = processObservation($pid, $src, $obs);
  Dada::logMsg(2, $dl ,"processOne: processObvservation() ".$result." ".$response);

  if ($result ne "ok") 
  {
    Dada::logMsg(1, $dl, "Failed to process ".$pid."/".$src."/".$obs);
    Dada::logMsg(1, $dl ,"processOne: moveObs(".$src_path.", ".$err_path.", ".$pid.", ".$src.", ".$obs.")");
    ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
    Dada::logMsg(1, $dl, "processOne: moveObs() ".$result." ".$response);

    # ensure that the temp directory is cleaned...
    $cmd = "rm -rf ".$tmp_path."/".$pid;
    Dada::logMsg(2, $dl, "processLoop: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processLoop: ".$response);

    return 1;
  }
  else
  {
    Dada::logMsg(1, $dl, "Processed ".$pid."/".$src."/".$obs);
    Dada::logMsg(1, $dl ,"processOne: moveObs(".$src_path.", ".$fin_path.", ".$pid.", ".$src.", ".$obs.")");
    ($result, $response) = moveObs($src_path, $fin_path, $pid, $src, $obs);
    Dada::logMsg(1, $dl, "processOne: moveObs() ".$result." ".$response);
    return 0;
  }

}

###############################################################################
#
# process CASPSR observations in the standard loop
#
sub processLoop()
{
  
  my $control_thread = 0;

  my $log_file = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file = META_DIR."/control/".$daemon_name.".quit";

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
    return 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl ,"STARTING SCRIPT");

  # start the control thread
  Dada::logMsg(2, $dl, "processLoop: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);


  while (!$quit_daemon)
  {
    Dada::logMsg(2, $dl, "Searching for unprocessed observations...");

    # get a listing of all observations in the src_path
    $cmd = "find ".$src_path." -mindepth 3 -maxdepth 3 -type d -printf '\%f\n' | sort | head -n 1";
    Dada::logMsg(2, $dl, "processLoop: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processLoop: ".$response);
    if (($result ne "ok") || ($response eq ""))
    {
      Dada::logMsg(2, $dl ,"processLoop: no observations to process");
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
    Dada::logMsg(2, $dl, "processLoop: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processLoop: ".$response);

    @bits = split(/\//, $response);
    if ($#bits < 2) 
    {
      Dada::logMsg(0, $dl ,"processLoop: not enough components in path");
      $quit_daemon = 1;
      next;
    }

    $pid = $bits[$#bits-2];
    $src = $bits[$#bits-1];

    Dada::logMsg(2, $dl ,"processLoop: processing ".$pid."/".$src."/".$obs);

    Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." converting...");
    ($result, $response) = processObservation($pid, $src, $obs);
    if ($result ne "ok")
    {
      Dada::logMsg(1, $dl, "Failed to process ".$pid."/".$src."/".$obs.": ".$response);
      Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." swin/sent -> psrfits/fail");
      ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);

      # ensure that the temp directory is cleaned...
      $cmd = "rm -rf ".$tmp_path."/".$pid;
      Dada::logMsg(2, $dl, "processLoop: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "processLoop: ".$response);
      $quit_daemon = 1;
      next;
    }
    else
    {
      Dada::logMsg(2, $dl, "Processed ".$src_path."/".$pid."/".$src."/".$obs);
      Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." swin/sent -> atnf/send");
     ($result, $response) = moveObs($src_path, $fin_path, $pid, $src, $obs);
    }
  }

  $quit_daemon = 1;

  $control_thread->join();
  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;
}

###############################################################################
#
# process the observation creating the fits file from the timer archives
#
sub processObservation($$$) {

  # observation tag [PID/SOURCE/UTC_START]
  my ($p, $s, $u) = @_;

  my $os = "";    # obs.start file to parse
  my $m = "";     # MODE parameter
  my @files = ();
  my %file_counts= (); # number of archives in each band
  my $ext = "";   # PSRFITS archive extension [rf|cf]

  my $i = 0;
  my $j = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";

  # get and obs.start file to parse some header parameters from
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u." -mindepth 1 -maxdepth 1 -type f -name 'obs.start'";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response eq "")) {
    Dada::logMsg(0, $dl ,"processObservation: find obs.start [".$cmd."] failed: ".$response);
    return ("fail", "could not find and obs.start file");
  }
  $os = $response;

  # get the MODE parameter
  $cmd = "grep ^MODE ".$os." | awk '{print \$2}'";
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: could not extract MODE from obs.start: ".$response);
    return ("fail", "could not extract MODE from obs.start");
  }
  $m = $response;

  # error check on the MODE
  if ($m eq "PSR") {
    if ($s =~ m/_R$/) {
      Dada::logMsg(0, $dl ,"processObservation: MODE was PSR and SOURCE had _R suffix");
      return ("fail", "MODE[".$m."] and SOURCE[".$s."] were conflicting");
    } 
    $ext = "rf";
  } else {
    if (($s =~ m/_R$/) || ($s =~ m/^Cal/) || ($s =~ m/HYDRA/)) {
      $ext = "cf";
    } else {
      Dada::logMsg(0, $dl ,"processObservation: MODE was not PSR and SOURCE didn't have _R suffix or Cal or HYDRA prefix");
      return ("fail", "MODE[".$m."] and SOURCE[".$s."] were conflicting");
    }
  }

  Dada::logMsg(2, $dl, "SOURCE=".$s.", UTC_START=".$u.", PID=".$p.", MODE=".$m);

  # adjust the observation filename to match the ATNF requirements
  my @parts = split(/[:\-\.]/, $u);
  my $year= $parts[0];
  my $ye = substr($year, 2, 2);
  my $mo = $parts[1];
  my $da = $parts[2];
  my $ho = $parts[3];
  my $mi = $parts[4];
  my $se = $parts[5];
  my $atnf_filename = "p".$ye.$mo.$da."_".$ho.$mi.$se.".".$ext;
  Dada::logMsg(2, $dl, "processObservation: filename ".$u.".".$ext." -> ".$atnf_filename);

  # check that an archive for this PID / SOURCE / UTC_START does not already exist
  if ( -f $dst_path."/".$p."/".$s."/".$atnf_filename ) 
  {
    Dada::logMsg(0, $dl ,"processObservation: destination fits file already existed [".$dst_path."/".$p."/".$s."/".$atnf_filename."]");
    return ("fail", "fits file already existed");
  }

  if ( -d $tmp_path."/".$p."/".$s."/".$u ) {
    Dada::logMsg(0, $dl ,"processObservation: temporary processing directory for already existed [".$tmp_path."/".$p."/".$s."/".$u."]");
    return ("fail", "temporary procesing directory already existed");
  }

  # create a directory in the tmp_path for the temporary files produced by this processing
  $cmd = "mkdir -m 0755 -p ".$tmp_path."/".$p."/".$s."/".$u;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: could not create temp dir [".$tmp_path."/".$p."/".$s."/".$u."] for band reduction: ".$response);
    return ("fail", "could not create a temp dir for band reduction");
  }

  # create a sorted list of archives for psradd to use
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u." -mindepth 1 -maxdepth 1 -type f -name '2*.ar' | sort > ".$tmp_path."/".$p."/".$s."/".$u."/archives.list";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not create list of archives");
  }

  # Tscrunch the band files down to 32 second subints
  $cmd = "psradd -O ".$tmp_path."/".$p."/".$s."/".$u." -D ".TSCRUNCH_SECONDS." -M ".$tmp_path."/".$p."/".$s."/".$u."/archives.list";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response =~ m/error/)) {
    unlink ($tmp_path."/".$p."/".$s."/".$u."/archives.list");
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not Tscrunch observation");
  }

  # cleanup the meta file
  if ( -f $tmp_path."/".$p."/".$s."/".$u."/archives.list") {
    unlink ($tmp_path."/".$p."/".$s."/".$u."/archives.list");
  }

  # add the Tsrunched files into the single archive with multiple subints
  $cmd = "psradd -o ".$tmp_path."/".$p."/".$s."/".$u."/obs.it ".$tmp_path."/".$p."/".$s."/".$u."/2*.it";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not psr the Tsrunch archives together");
  }

  # check that the file now does exist
  if (! -f $tmp_path."/".$p."/".$s."/".$u."/obs.it") {
    Dada::logMsg(0, $dl ,"processObservation: obs.it file was not produced [".$tmp_path."/".$p."/".$s."/".$u."/obs.it]");
    return ("fail", "obs.it file was not produced: ".$response);
  }

  # convert the timer archive to a psrfits archive
  $cmd = "psrconv ".$tmp_path."/".$p."/".$s."/".$u."/obs.it";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response =~ m/Error/)) {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not convert the timer archive to psrfits");
  }

  # fix up the PSRFITS header as best we can
  Dada::logMsg(2, $dl, "processObservation: fixPSRFITSHeader(".$p.", ".$s.", ".$u.", obs.".$ext.")");
  ($result, $response) = fixPSRFITSHeader($p, $s, $u, "obs.".$ext);
  Dada::logMsg(3, $dl, "processObservation: fixPSRFITSHeader() ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: fixPSRFITSHeader failed");
    return ("fail", "could not patch PSRFITS header");
  }

  # clean up the .it files created during the process
  $cmd = "find ".$tmp_path."/".$p."/".$s."/".$u." -mindepth 1 -name '*.it' -delete";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not clean up intermediate .it files");
  }

  # if the PID dir does not yet exist, create it
  if ( ! -d $dst_path."/".$p) {
    $cmd = "mkdir -m 0755 -p ".$dst_path."/".$p;
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processObservation: ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
      return ("fail", "could create directory ".$dst_path."/".$p.": ".$response);
    }
  }

  # if the PID / SOURCE dir does not yet exist, create it
  if ( ! -d $dst_path."/".$p."/".$s) {
    $cmd = "mkdir -m 0755 -p ".$dst_path."/".$p."/".$s;
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processObservation: ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
      return ("fail", "could create directory ".$dst_path."/".$p."/".$s.": ".$response);
    }
  }

  # move the final fits archive to its storage location/name
  $cmd = "mv -f ".$tmp_path."/".$p."/".$s."/".$u."/obs.".$ext." ".$dst_path."/".$p."/".$s."/".$atnf_filename;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not movie final fits file to final location: ".$response);
  } 

  # chmod the archive so that it isn't accidently deleted
  $cmd = "chmod a-w ".$dst_path."/".$p."/".$s."/".$atnf_filename;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not remove write permissions on the final archive");
  }

  # delete the intermediary directories
  $cmd = "rmdir ".$tmp_path."/".$p."/".$s."/".$u;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not delete temp OBS dir [".$tmp_path."/".$p."/".$s."/".$u."]");
  }

  $cmd = "rmdir ".$tmp_path."/".$p."/".$s;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "processObservation: ".$cmd." failed: ".$response);
    return  ("fail", "could not delete temp SOURCE dir [".$tmp_path."/".$p."/".$s."]");
  }

  $cmd = "rmdir ".$tmp_path."/".$p;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not delete temp PID dir [".$tmp_path."/".$p."]");
  }

  return ($result, $response);

}

#
# Fix the PSRFITS header with constant CASPSR values
#
sub fixPSRFITSHeader($$$$)
{
  my ($p, $s, $u, $f) = @_;

  my $result = "";
  my $response = "";

  my %current = ();
  my %h = ();

  $h{"obs:projid"}   = $p;

  # These are considered constants
  $h{"site"}         = "PARKES";
  $h{"be:nrcvr"}     = "2";
  $h{"be:phase"}     = "+1";  # From DSB header parameter?
  $h{"be:tcycle"}    = "8";
  $h{"be:dcc"}       = "0";
  $h{"ext:trk_mode"} = "TRACK";
  $h{"sub:nsblk"}    = "1";

  # These already exist in the current archive under different param name
  %current = getCurrentPSRFITSHeader($p, $s, $u, $f);
  my @current_keys = keys %current;
  if ($#current_keys == -1)
  {
    Dada::logMsg(0, $dl, "fixPSRFITSHeader: failed to extract header params from current psrfits header");
    return ("fail", "could not extract psrfits header params");
  }
  $h{"ext:obsfreq"}  = $current{"freq"};
  $h{"ext:obsbw"}    = $current{"bw"};
  $h{"ext:obsnchan"} = $current{"nchan"};
  $h{"ext:stp_crd1"} = $current{"ext:stt_crd1"};
  $h{"ext:stp_crd2"} = $current{"ext:stt_crd2"};
  $h{"ext:stt_date"} = substr($u, 0, 10);
  $h{"ext:stt_time"} = substr($u, 11, 8);

  my $key = "";
  my $value = "";
  my @keys = sort keys (%h);
  my $i=0; 
  my $key = $keys[0];
  my $val = $h{$key};
  my $cmd = "psredit -m -c ".$key."=".$val;
  
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
  $cmd .= " ".$tmp_path."/".$p."/".$s."/".$u."/".$f;
  
  Dada::logMsg(2, $dl, "fixPSRFITSHeader: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "fixPSRFITSHeader: ".$result." ".$response);
  if (($result ne "ok") || ($response =~ m/ailed/) || ($response =~ m/rror/))
  {
    Dada::logMsg(0, $dl, "fixPSRFITSHeader: failed to apply psredit command: ".$response);
    return ("fail", "psredit failed to patch file");
  }   
  return ("ok", "patched");

}

sub getCurrentPSRFITSHeader($$$$)
{
  
  # observation tag [PID/SOURCE/UTC_START/file]
  my ($p, $s, $u, $f) = @_;
  
  my $cmd = "";
  my $result = "";
  my $response = "";
  my %header = ();
  my @lines = ();
  my @parts = ();
  my $i = 0;
  
  $cmd = "psredit ".$tmp_path."/".$p."/".$s."/".$u."/".$f;
  Dada::logMsg(2, $dl, "getCurrentPSRFITSHeader: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getCurrentPSRFITSHeader: ".$result." ".$response);
  if ($result ne "ok")
  { 
    Dada::logMsgWarn($warn, "getCurrentPSRFITSHeader: failed to extract header params");
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


#
# Move an observation from from to to
#
sub moveObs($$$$$) {

  my ($from, $to, $pid, $src, $obs) = @_;
  
  Dada::logMsg(3, $dl ,"moveObs(".$from.", ".$to.", ".$pid.", ".$src.", ".$obs.")");
  
  my $cmd = "";
  my $result = "";
  my $response = "";

  # check that the to/PID dir exists
  if ( ! -d ($to."/".$pid) )
  {
    $cmd = "mkdir -m 0755 -p ".$to."/".$pid;
    Dada::logMsg(2, $dl, "moveObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "moveObs: ".$result." ".$response);
    if ($result ne "ok") 
    {
      Dada::logMsgWarn($warn, "moveObs: failed to create dir ".$to."/".$pid.": ".$response);
      return ("fail", "could not create dest PID dir");
    }
  }
  
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
      return ("fail", "could not create dest PID/SOURCE dir");
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

  Dada::logMsg(1, $dl ,"controlThread: asking processing to quit...");
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
    print STDERR $daemon_name." : exiting\n";
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

  my $host = Dada::getHostMachineName();
  if ($host ne REQUIRED_HOST) {
    return ("fail", "Error: this script can only be run on ".REQUIRED_HOST);
  }

  my $curr_user = `whoami`;
  chomp $curr_user;
  if ($curr_user ne REQUIRED_USER) {
    return ("fail", "Error: this script can only be run as user ".REQUIRED_USER);
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}


END { }

1;  # return value from file
