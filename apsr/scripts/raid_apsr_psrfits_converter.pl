#!/usr/bin/env perl

###############################################################################
#
# atnf_apsr_pipeline.pl
#
# Process APSR files converting archives to the atnf requested format
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
use constant DATA_DIR       => "/lfs/raid0/apsr";
use constant META_DIR       => "/lfs/data0/apsr";
use constant REQUIRED_HOST  => "raid0";
use constant REQUIRED_USER  => "apsr";
use constant TSCRUNCH_SECONDS => 30;



#
# Function Prototypes
#
sub good($);
sub processOne($$$);
sub processObservation($$$);
sub processBand($$$$\@);
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
$src_path = DATA_DIR."/ready";
$dst_path = DATA_DIR."/psrfits/unpatched";
$tmp_path = DATA_DIR."/psrfits/temp";
$err_path = DATA_DIR."/psrfits/fail_convert";
$fin_path = DATA_DIR."/swin/send";

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
# process APSR observations in the standard loop
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
  my $waiting = 0;
  
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
      if (!$waiting)
      {
        Dada::logMsg(0, $dl, "Waiting for new observations");
        $waiting = 1;
      } 
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

    $waiting = 0;
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

    # test that we have 16 band directories
    $cmd = "find ".$src_path."/".$pid."/".$src."/".$obs." -mindepth 1 -maxdepth 1 -type d | wc -l";
    Dada::logMsg(2, $dl, "processLoop: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processLoop: ".$response);
    if ($result ne "ok") 
    {
      Dada::logMsg(0, $dl ,"processLoop: test 16 bands [".$cmd."] failed: ".$response);
      Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/fail_convert");
      ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
      next;
    }

    # should really be based on the NUM_PWC in the obs.start
    # if ($response ne "16")
    # {
    #  Dada::logMsg(1, $dl ,"processLoop: ignoring ".$pid."/".$src."/".$obs." only ".$response." bands");
    #  Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/fail_convert");
    #  ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
    #  next;
    #}

    # test that we have an obs.start file that can be processed
    $cmd = "find ".$src_path."/".$pid."/".$src."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'obs.start'";
    Dada::logMsg(2, $dl, "processLoop: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processLoop: ".$response);
    if ($result ne "ok") 
    {
      Dada::logMsg(0, $dl ,"processLoop: ".$cmd." failed: ".$response);
      Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/fail_convert");
      ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
      next;
    }
    if ($response eq "") 
    {
      Dada::logMsg(1, $dl ,"processLoop: ignoring ".$pid."/".$src."/".$obs." no obs.start files");
      Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/fail_convert");
      ($result, $response) = moveObs($src_path, $err_path, $pid, $src, $obs);
      next;
    }

    if (!$quit_daemon) 
    {
      Dada::logMsg(2, $dl, $pid."/".$src."/".$obs." converting...");
      ($result, $response) = processObservation($pid, $src, $obs);
      if ($result ne "ok")
      {
        Dada::logMsg(1, $dl, "Failed to process ".$pid."/".$src."/".$obs.": ".$response);
        Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/fail_convert");
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
        Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." ready -> psrfits/unpatched");
        ($result, $response) = moveObs($src_path, $fin_path, $pid, $src, $obs);
      }
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
  my $c = "";     # CONFIG parameter
  my $m = "";     # MODE parameter
  my $b = "";     # BAND being processed
  my @bands = (); # bands of the observation
  my @files = ();
  my %file_counts= (); # number of archives in each band
  my $ext = "";   # PSRFITS archive extension [rf|cf]

  my $i = 0;
  my $j = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";

  # get and obs.start file to parse some header parameters from
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u." -mindepth 2 -maxdepth 2 -type f -name 'obs.start' | head -n 1";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response eq "")) {
    Dada::logMsg(0, $dl ,"processObservation: find obs.start [".$cmd."] failed: ".$response);
    return ("fail", "could not find and obs.start file");
  }

  $os = $response;

  # get the CONFIG parameter
  $cmd = "grep ^CONFIG ".$os." | awk '{print \$2}'";
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: could not extract CONFIG from obs.start: ".$response);
    return ("fail", "could not extract CONFIG from obs.start");
  }
  $c = $response;

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

  Dada::logMsg(2, $dl, "SOURCE=".$s.", UTC_START=".$u.", PID=".$p.", CONFIG=".$c.", MODE=".$m);

  # adjust the observation filename to match the ATNF requirements
  my @parts = split(/[:\-\.]/, $u);
  my $year= $parts[0];
  my $ye = substr($year, 2, 2);
  my $mo = $parts[1];
  my $da = $parts[2];
  my $ho = $parts[3];
  my $mi = $parts[4];
  my $se = $parts[5];
  my $atnf_filename = "b".$ye.$mo.$da."_".$ho.$mi.$se.".".$ext;
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

  # determine the number of bands
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u." -mindepth 1 -maxdepth 1 -type d -printf '\%f\\n' | sort";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: find num bands [".$cmd."] failed: ".$response);
    return ("fail", "could not find the number of band subdirectories");
  }
  @bands = split(/\n/, $response);

  # if n_band != 16, then ignore for now
  # if (($#bands + 1) != 16) {
  #  Dada::logMsg(0, $dl ,"processObservation: number of bands [".($#bands + 1)."] was not 16");
  #  return ("fail", "observation only had ".($#bands + 1)." bands, required 16");
  #}

  # ensure that at least 1 archive and obs.start file exists for each band
  # foreach band Tscrunch down the tsrunch time and produce a full intergration for the band in 1 file
  for ($i=0; $i<=$#bands; $i++) 
  {
    $b = $bands[$i];
    $cmd = "find ".$src_path."/".$p."/".$s."/".$u."/".$b." -mindepth 1 -maxdepth 1 -type f -name '2*.ar' -printf '\%f\\n'| sort";
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processObservation: ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
      return ("fail", "could not find number of archives in band ".$b);
    }

    @files = split(/\n/, $response);

    for ($j=0; $j<=$#files; $j++) 
    {
      if ($files[$j] =~ m/^(\d\d\d\d)\-(\d\d)\-(\d\d)\-(\d\d):(\d\d):(\d0)/) 
      {
        if  (exists($file_counts{$files[$j]})) 
        {
          $file_counts{$files[$j]}++;
        }
        else 
        {
          $file_counts{$files[$j]} = 1;
        }
      } 
      else
      {
        Dada::logMsg(0, $dl ,"processObservation: archive ".$b."/".$files[$j]." not mod 10");
        return ("ok", $b." contained archives not mod 10 s");
      }
    }

    $cmd = "find ".$src_path."/".$p."/".$s."/".$u."/".$b." -mindepth 1 -maxdepth 1 -type f -name 'obs.start'";
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processObservation: ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
      return ("ok", "could not find obs.start file in band ".$b);
    }
    if ($response eq "") {
      Dada::logMsg(0, $dl ,"processObservation: band ".$b." didn't have obs.start");
      return ("ok", "no obs.start existed in band ".$b);
    }
  }

  @files = ();
  @files = sort keys %file_counts;
  my $npatched = 0;
  for ($i=0; $i<=$#files; $i++) {
    if ($file_counts{$files[$i]} < 16) {
      
      #delete ($file_counts{$files[$i]});
      Dada::logMsg(2, $dl ,"processObservation: patchMissingArchives(".$src_path.", ".$p.", ".$s.", ".$u.", ".$files[$i].")");
      ($result, $response) = patchMissingArchives($src_path, $p, $s, $u, $files[$i]);
      Dada::logMsg(3, $dl ,"processObservation: patchMissingArchives() ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl ,"processObservation: failed to patch missing archives for ".$files[$i]);
        return ("ok", "failed to patch missing archives");
      }
      $npatched += $response;
    }
  }

  if ($npatched > 0)
  {
    Dada::logMsg(0, $dl ,"had to patch ".$npatched." band archives");
  }

  @files = sort keys %file_counts;

  Dada::logMsg(2, $dl, "processObservation: processing ".($#files + 1)." files");

  # create a directory in the tmp_path for the temporary files produced by this processing
  $cmd = "mkdir -m 0755 -p ".$tmp_path."/".$p."/".$s."/".$u;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: could not create temp dir [".$tmp_path."/".$p."/".$s."/".$u."] for band reduction: ".$response);
    return ("fail", "could not create a temp dir for band reduction");
  }

  # attempt to reduce randmon file system failures, but not threadding the band reduction
  my $no_threads = 1;

  if ($no_threads)
  {
    for ($i=0; $i<=$#bands; $i++) {
      $b = $bands[$i];
      Dada::logMsg(2, $dl, "processObservation: processBand(".$p.", ".$s.", ".$u.", ".$b.")");
      ($result, $response) = processBand($p, $s, $u, $b, @files);
      if ($result ne "ok") {
        Dada::logMsg(0, $dl ,"processObservation: processBand() failed: ".$response);
        return ("fail", "processing of bands failed");
      }
    } 
  } else {
  
    my @threads = ();

    # foreach band Tscrunch down the tsrunch time and produce a full intergration for the band in 1 file
    for ($i=0; $i<=$#bands; $i++) {
      $b = $bands[$i];
      Dada::logMsg(2, $dl, "processObservation: processBand(".$p.", ".$s.", ".$u.", ".$b.")");
      @threads[$i] = threads->new(\&processBand, $p, $s, $u, $b, \@files);
    } 

    # join all the band threads 
    my $threads_ok = 1;
    for ($i=0; $i<=$#bands; $i++) {
      $b = $bands[$i];
      ($result, $response) =  @threads[$i]->join; 
      Dada::logMsg(2, $dl, "processObservation: processBand(".$p.", ".$s.", ".$u.", ".$b."): ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsg(0, $dl ,"processObservation: processBand(".$p.", ".$s.", ".$u.", ".$b.") failed: ".$response);
        $threads_ok = 0;
      }
    }
    if (!$threads_ok) {
      return ("fail", "processing of bands failed");
    }
  }

  # add the 16 bands together
  $cmd = "psradd -R -o ".$tmp_path."/".$p."/".$s."/".$u."/obs.it ".$tmp_path."/".$p."/".$s."/".$u."/*/band.it";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not psradd the bands together");
  }

  # convert the timer archive to a psrfits archive
  $cmd = "psrconv ".$tmp_path."/".$p."/".$s."/".$u."/obs.it";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response =~ m/Error/)) {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not convert the timer archive to psrfits");
  }

  # add missing headers to the psrfits file
  $cmd = "psredit -m -c be:config=".$c.",obs:projid=".$p." ".$tmp_path."/".$p."/".$s."/".$u."/obs.".$ext;
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if (($result ne "ok") || ($response =~ m/error/) || ($response =~ m/Error/)) {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not add missing header params to psrfits file");
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
  $cmd = "rmdir ".$tmp_path."/".$p."/".$s."/".$u."/*";
  Dada::logMsg(2, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processObservation: ".$cmd." failed: ".$response);
    return ("fail", "could not delete temp BAND dirs [".$tmp_path."/".$p."/".$s."/".$u."/*]");
  }

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
# Create zeroed archives for any band that does not contain the required archive
#
sub patchMissingArchives($$$$$) {

  my ($d, $p, $s, $u, $ar) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $good_archive = "";
  my @bands = ();
  my $b = "";
  my $dir = "";
  my $tmp_ar = "";
  my $out_ar = "";
  my $output = "";
  my $i = 0;

  my $npatched = 0;

  # find 1 good archive
  $cmd = "find ".$d."/".$p."/".$s."/".$u." -name ".$ar." | head -n 1 | awk -F/ '{print \$(NF-1)\"/\"\$NF}'";
  Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);
  if ($result ne "ok") {
    return ("fail", "could not find good archive to use");
  }
  
  $good_archive = $response;

  # determine the number of bands
  $cmd = "find ".$d."/".$p."/".$s."/".$u." -mindepth 1 -maxdepth 1 -type d -printf '\%f\\n' | sort";
  Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchMissingArchives: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"patchMissingArchives: find num bands [".$cmd."] failed: ".$response);
    return ("fail", "could not find the number of band subdirectories");
  }
  @bands = split(/\n/, $response);

  $cmd = "chmod u+w ".$d."/".$p."/".$s."/".$u."/*";
  Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"patchMissingArchives: could not chmod APSR source dir: ".$response);
    return ("fail", "could not chmod APSR dir");
  } 

  my $pam_errors = 0;
  # check all the bands
  for ($i=0; (($i<=$#bands) && (!$pam_errors)); $i++) {

    $b = $bands[$i];
    $dir = $d."/".$p."/".$s."/".$u."/".$b;
  
    # if this band was missing the archive...
    if (! -f $dir."/".$ar) {

      Dada::logMsg(2, $dl, "patchMissingArchives: archive ".$dir."/".$ar." was not present");
      $npatched++;
  
      $tmp_ar = $d."/".$p."/".$s."/".$u."/".$good_archive;
      $tmp_ar =~ s/\.ar$/\.zeroed/;
      $out_ar = $dir."/".$ar;

      # create a copy of archives[0] with .lowres -> .tmp
      $cmd = "pam -o ".$b." -e zeroed ".$d."/".$p."/".$s."/".$u."/".$good_archive;
      Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsg(0, $dl ,"patchMissingArchives: pam failed: ".$response);
        $pam_errors = 1;
        next;
      }

      # move the zeroed archive to the band subdir
      $cmd = "mv -f ".$tmp_ar." ".$out_ar;
      Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsg(0, $dl ,"patchMissingArchives: mv failed: ".$response);
        $pam_errors = 1;
        next;
      }

      # set the weights in the zeroed archive to 0
      $cmd = "paz -w 0 -m ".$out_ar;
      Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsg(0, $dl ,"patchMissingArchives: paz failed: ".$response);
        $pam_errors = 1;
        next;
      }
    }
  }

  $cmd = "chmod u-w ".$d."/".$p."/".$s."/".$u."/*";
  Dada::logMsg(2, $dl, "patchMissingArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchMissingArchives: ".$result." ".$response);

  if ($pam_errors) {
    return ("fail", "pam/mv/paz failed");
  } else {
    return ("ok", $npatched);
  }

}

sub processBand($$$$\@) {

  my ($p, $s, $u, $b, $ref) = @_;

  my @archives = @$ref;
  my $cmd = "";
  my $result = "";
  my $response = ""; 
  my $i = 0;

  # create a temporary dir for the band files
  $cmd = "mkdir -m 0755 -p ".$tmp_path."/".$p."/".$s."/".$u."/".$b;
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processBand: could not create temp dir for band reduction: ".$response);
    return ("fail", "could not create a temp dir for band reduction");
  }

  # create a meta file for the archives to be processed
  open FH, ">".$tmp_path."/".$p."/".$s."/".$u."/".$b."/band.archives";
  for ($i=0; $i<=$#archives; $i++)
  {
    print FH $src_path."/".$p."/".$s."/".$u."/".$b."/".$archives[$i]."\n";
  }
  close FH;

  sleep (1);
 
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u."/".$b." -mindepth 1 -maxdepth 1 -type f -name '*.ar'"; 
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." failed: ".$response);
    sleep(5);
  }

  # check the number of *.ar files that exist
  $cmd = "find ".$src_path."/".$p."/".$s."/".$u."/".$b." -mindepth 1 -maxdepth 1 -type f -name '*.ar' | wc -l";
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." failed: ".$response);
    return ("fail", "could not count .ar archives in band ".$b);
  }
  if ($response eq "0") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." returned 0 .ar archives");
    return ("fail", "found 0 archives in band ".$b);
  }

  # Tscrunch the band files down to 
  # chdir $tmp_path."/".$p."/".$s."/".$u."/".$b;
  $cmd = "psradd -O ".$tmp_path."/".$p."/".$s."/".$u."/".$b." -D ".TSCRUNCH_SECONDS." -M ".$tmp_path."/".$p."/".$s."/".$u."/".$b."/band.archives";
  # $cmd = "psradd -O ./ -D ".TSCRUNCH_SECONDS." -M ./band.archives";
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if (($result ne "ok") || ($response =~ m/error/)) {
    unlink ($tmp_path."/".$p."/".$s."/".$u."/".$b."/band.archives");
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." failed: ".$response);
    return ("fail", "could not Tscrunch band ".$b);
  }

  # cleanup the meta file
  if ( -f $tmp_path."/".$p."/".$s."/".$u."/".$b."/band.archives") {
    unlink ($tmp_path."/".$p."/".$s."/".$u."/".$b."/band.archives");
  }

  sleep (1);

  # check the number of it files produced
  $cmd = "find ".$tmp_path."/".$p."/".$s."/".$u."/".$b." -mindepth 1 -maxdepth 1 -type f -name '2*.it' | wc -l";
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." failed: ".$response);
    return ("fail", "could not count .it archives in ".$b);
  }
  if ($response eq "0") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." returned 0 .it archives");
    return ("fail", "found 0 .it archives in ".$b);
  }

  sleep (1);

  # add the Tsrunched files into the single archive with multiple subints
  $cmd = "psradd -o ".$tmp_path."/".$p."/".$s."/".$u."/".$b."/band.it ".$tmp_path."/".$p."/".$s."/".$u."/".$b."/2*.it";
  # $cmd = "psradd -o ./band.it ./2*.it";
  Dada::logMsg(2, $dl, "processBand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processBand: ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl ,"processBand: ".$cmd." failed: ".$response);
    return ("fail", "could not psr the Tsrunch archives together for ".$b);
  }

  # check that the file now does exist
  if (! -f $tmp_path."/".$p."/".$s."/".$u."/".$b."/band.it") {
    Dada::logMsg(0, $dl ,"processBand: band.it file was not produced [".$tmp_path."/".$p."/".$s."/".$u."/".$b."/band.it]");
    return ("fail", "band.it file was not produced: ".$response);
  }

  return ("ok", "band ".$b." processed");
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
