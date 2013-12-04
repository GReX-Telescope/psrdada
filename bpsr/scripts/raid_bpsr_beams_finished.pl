#!/usr/bin/env perl

###############################################################################
#
# Waits for all beams to be transferred to RAID disk before:
#   1. Patching in TCS log file to psrxml and .fil files
#   2. Run SVD alg to produce zap mask
#   3. Inserting observation into archival pipeline
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;

#
# Constants
#
use constant DATA_DIR         => "/lfs/raid0/bpsr";
use constant META_DIR         => "/lfs/data0/bpsr";
use constant REQUIRED_HOST    => "raid0";
use constant REQUIRED_USER    => "bpsr";
use constant SWIN_PROJECTS    => "P630 P786 P813 P848 P789 P855 PX022";
use constant PRKS_PROJECTS    => "P630 P682 P743 P786";

use constant TCS_USER         => "pksobs";
use constant TCS_HOST         => "joffrey.atnf.csiro.au";
use constant TCS_PATH         => "/home/pksobs/tcs/logs";

use constant BPSR_USER        => "dada";
use constant BPSR_HOST        => "hipsr-srv0.atnf.csiro.au";
use constant BPSR_PATH        => "/data/bpsr";

#
# Function prototypes
#
sub touchBeamsXferComplete($$$);
sub patchTCSLogs($$);
sub copyPsrxmlToDb($$);
sub fixSigprocFiles($$);
sub createZapMask($$);
sub tarAuxFiles($$);
sub createLinks($$$$$);
sub controlThread($$);
sub good($);

#
# Global variable declarations
#
our $dl : shared;
our $daemon_name : shared;
our $quit_daemon : shared;
our $warn : shared;
our $error : shared;

#
# Global initialization
#
$dl = 1;
$daemon_name = Dada::daemonBaseName(basename($0));
$quit_daemon = 0;

# Autoflush STDOUT
$| = 1;

# Main
{
  my $log_file       = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file       = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file      = META_DIR."/control/".$daemon_name.".quit";

  my $perm_path      = DATA_DIR."/perm";              # permanent dir for obs
  my $src_path       = DATA_DIR."/upload";            # dir to which files will be copied

  my $swin_path      = DATA_DIR."/swin/send";         # dir for swin transfer
  my $prks_path      = DATA_DIR."/parkes/archive";    # dir for parkes tape archival
  my $archived_path  = DATA_DIR."/archived";          # dir for final resting place

  my $dst_path       = "";
  my $dst            = "";

  $warn              = META_DIR."/logs/".$daemon_name.".warn";
  $error             = META_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;

  my $line = "";
  my $obs = "";
  my $src = "";
  my $pid = "";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @finished = ();
  my @bits = ();

  my $i = 0;
  my $obs_start_file = "";
  my $n_beam = 0;
  my $n_transferred = 0;

  my $rval = 0;

  my $b = "";
  my @beams = ();

  # quick sanity check
  ($result, $response) = good($quit_file);
  if ($result ne "ok") 
  {
    print STDERR $response."\n";
    exit 1;
  }

  # install signal handles
  $SIG{INT}  = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Auto flush output
  $| = 1;

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # main Loop
  while ( !$quit_daemon ) 
  {
    @finished = ();

    # look for all observations in the src_path
    $cmd = "find ".$src_path." -mindepth 2 -maxdepth 2 -type d -printf '\%h/\%f\n' | sort";
    Dada::logMsg(3, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: ".$cmd." failed: ".$response);
    }
    else
    {
      @finished = split(/\n/, $response);
      Dada::logMsg(2, $dl, "main: found ".($#finished+1)." finished observations");

      for ($i=0; (!$quit_daemon && $i<=$#finished); $i++)
      {
        $line = $finished[$i]; 
        @bits = split(/\//, $line);
        if ($#bits < 1)
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }

        $pid = $bits[$#bits-1];
        $obs = $bits[$#bits];

        Dada::logMsg(2, $dl, "main: processing ".$pid."/".$obs);
        Dada::logMsg(3, $dl, "main: pid=".$pid." obs=".$obs);

        # try to find and obs.start file
        $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'obs.start' | head -n 1";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          Dada::logMsg(2, $dl, "main: could not find and obs.start file");
          next;
        }
        $obs_start_file = $response;

        # ensure this observation exists on hipsr-srv0
        $cmd = "ls -1d ".BPSR_PATH."/results/".$obs;
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "ssh failure to ".BPSR_USER."@".BPSR_HOST.": ".$response);
          sleep(5);
          next;
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "directory did not exist ".BPSR_HOST.":".BPSR_PATH."/results/".$obs);
          next;
        }

        # determine the number of beams
        $cmd = "find ".BPSR_PATH."/results/".$obs." -mindepth 1 -maxdepth 1 -type d -name '??' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "ssh failure to ".BPSR_USER."@".BPSR_HOST.": ".$response);
          sleep(5);
          next;
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "could not count beam dirs in ".BPSR_PATH."/results/".$obs);
          next;
        }
        $n_beam = $response;
        if (($n_beam < 1) || ($n_beam > 13))
        {
          Dada::logMsgWarn($warn, "bad value for beam dirs in ".BPSR_PATH."/results/".$obs);
          next;
        }
      
        # check if all beams have been transferred
        $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'beam.transferred' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not count number of beam.transferred files for ".$obs);
          $n_transferred = 0;
          next;
        }
        $n_transferred = $response;

        if ($n_transferred != $n_beam)
        {
          Dada::logMsg(2, $dl, "main: only ".$n_transferred." of ".$n_beam." beams found for ".$obs);
          next;
        }

        # check for existence of .fil files, or .ar files to indicate successful observation
        $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -name '*.fil' -o -name '*.ar' -o -name 'aux' | wc -l";
        #$cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -name '*.fil' -o -name '*.ar' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not count number of *.fil, *.ar or aux files/dirs for ".$obs);
          next;
        }
        if ($response == 0)
        {
          Dada::logMsg(2, $dl, "main: only found ".$response." *.fil, *ar or aux files/dirs for ".$obs);
          next;
        }

        # patch in TCS logs
        Dada::logMsg(2, $dl, "main: patchTCSLogs(".$src_path."/".$pid.", ".$obs.")");
        ($result, $response) = patchTCSLogs($src_path."/".$pid, $obs);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok") 
        {
          Dada::logMsgWarn($warn, "could not patch the TCS log file: ".$response);
        }
        elsif ($response eq "no psrxml files existed")
        {
          Dada::logMsg(2, $dl, "main: skipping copyPsrxmlToDb");
        }
        else
        {
          # update the PSRXML database
          Dada::logMsg(2, $dl, "main: copyPsrxmlToDb(".$src_path."/".$pid.", ".$obs.")");
          ($result, $response) = copyPsrxmlToDb($src_path."/".$pid, $obs);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "could not copy PSRXML files to DB: ".$response);
          }

          # patch the Sigproc file header
          Dada::logMsg(2, $dl, "main: fixSigprocFiles(".$src_path."/".$pid.", ".$obs.")");
          ($result, $response) = fixSigprocFiles($src_path."/".$pid, $obs);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "could not copy sigproc files : ".$response);
          }

        }

        # perform RFI zapping with
        Dada::logMsg(2, $dl, "main: createZapMask(".$src_path."/".$pid.", ".$obs.")");
        ($result, $response) = createZapMask($src_path."/".$pid, $obs);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not create zap mask: ".$response);
        }

        # tar up the aux files
        Dada::logMsg(2, $dl, "main: tarAuxFiles(".$src_path."/".$pid.", ".$obs.")");
        ($result, $response) = tarAuxFiles($src_path."/".$pid, $obs);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not tar aux files: ".$response);
        }

        # ensure perm dir PID dir exists
        ($result, $response) = Dada::createDir($perm_path."/".$pid, 0755);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not create dst dir [".$perm_path."/".$pid."] for ".$obs);
          $quit_daemon = 1;
          next;
        }

        # remove any existing beam.* flags
        $cmd = "rm -f ".$src_path."/".$pid."/".$obs."/??/beam.*";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);

        # move obs + beams to perm dir
        $cmd = "mv ".$src_path."/".$pid."/".$obs." ".$perm_path."/".$pid."/";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "failed to move obseravation to dst_dir [".$perm_path."/".$pid."] for ".$obs);
          $quit_daemon = 1;
          next;
        }

        $dst = "";

        # select the soft link destination based on the project ID
        if (SWIN_PROJECTS =~ m/$pid/)
        {
          $dst = "swin/send";
          ($result, $response) = createLinks($perm_path, $swin_path, $dst, $pid, $obs);
        }

        if (PRKS_PROJECTS =~ m/$pid/)
        {
          if ($dst ne "")
          {
            $dst .= " + ";
          }
          $dst .= "parkes/archive";
          ($result, $response) = createLinks($perm_path, $prks_path, $dst, $pid, $obs);

          # if we are going direct to the parkes archival path, touch xfer.complete
          Dada::logMsg(2, $dl, "main: touchBeamsXferComplete(".$prks_path.", ".$pid.", ".$obs.")");
          ($result, $response) = touchBeamsXferComplete($prks_path, $pid, $obs);
          Dada::logMsg(2, $dl, "main: touchBeamsXferComplete ".$result." ".$response);
        }

        # if we did not try to send to swin or archive at parkes, just plop in archived dir
        if ($dst eq "")
        {
          ($result, $response) = createLinks($perm_path, $archived_path, "archived", $pid, $obs);
          $dst = "archived";
        }

        Dada::logMsg(1, $dl, $pid."/".$obs." finished -> ".$dst);

        # touch the relevant file in the SERVER_RESULTS_DIR
        $cmd = "touch ".BPSR_PATH."/results/".$obs."/obs.transferred";
        if ($dst eq "archived")
        {
          $cmd .= " ".BPSR_PATH."/results/".$obs."/obs.archived";
        } 

        Dada::logMsg(2, $dl, "main: ".BPSR_USER."@".BPSR_HOST.":".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
        if (($result ne "ok") || ($rval != 0))
        {
          Dada::logMsg(0, $dl, "main: ".$cmd." failed : ".$response);
        }

        sleep(1);
      }
    }

    my $counter = 30;
    Dada::logMsg(2, $dl, "main: sleeping ".($counter)." seconds");
    while ((!$quit_daemon) && ($counter > 0)) 
    {
      sleep(1);
      $counter--;
    }
  }

  Dada::logMsg(2, $dl, "main: joining threads");
  $control_thread->join();
  Dada::logMsg(2, $dl, "main: control_thread joined");

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
}

exit 0;

###############################################################################
#
# Functions
#

#
# create a set of soft links from src_path to dst_path following the convention
# of 
#     <path> / <utc_start> / <beamlink>
#
sub createLinks($$$$$)
{
  my ($src_path, $dst_path, $dst, $pid, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $b = "";
  my @beams = ();

  # create dir for soft links in dst path
  Dada::logMsg(2, $dl, "createLinks: createDir(".$dst_path."/".$pid."/".$obs.", 0755)");
  ($result, $response) = Dada::createDir($dst_path."/".$pid."/".$obs, 0755);
  Dada::logMsg(3, $dl, "createLinks: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "could not create dst dir [".$dst_path."/".$pid."/".$obs."] for ".$obs);
    return ("fail", "could not create dir ".$dst_path."/".$pid."/".$obs." ".$response);
  }

  # get beam listing from src/perm path
  $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 1 -maxdepth 1 -type d -name '??' -printf '\%f\n'";
  Dada::logMsg(2, $dl, "createLinks: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createLinks: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "createLinks: ".$cmd." failed: ".$response);
    return ("fail", "could not get beam listing");
  }

  @beams = split(/\n/, $response);
  foreach $b ( @beams)
  {
    $cmd = "ln -s ".$src_path."/".$pid."/".$obs."/".$b." ".$dst_path."/".$pid."/".$obs."/".$b;
    Dada::logMsg(2, $dl, "createLinks: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createLinks: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "createLinks: ".$cmd." failed: ".$response);
      return ("fail", "could not create soft link for ".$pid."/".$obs."/".$b);
    }
  }

  return ("ok", "")
}

#
# touch xfer.complete in all the beam subdirectories
#
sub touchBeamsXferComplete($$$)
{
  my ($path, $pid, $obs) = @_;
  
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $rval = 0;

  # get the list of beams
  $cmd = "find ".$path."/".$pid."/".$obs." -mindepth 1 -maxdepth 1 -type l -name '??' -printf '\%f\n'";
  Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".$result." ".$response);
  if (($result ne "ok") || ($response eq ""))
  {
    Dada::logMsgWarn($warn, "touchBeamsXferComplete: could get beam list: ".$response);
    return ("fail", "could not get beam list");
  }

  my @beams = split(/\n/, $response);
  my $i = 0;
  my $beam = "";
  for ($i=0; $i<=$#beams; $i++)
  {
    $beam = $beams[$i];

    $cmd = "touch ".$path."/".$pid."/".$obs."/".$beam."/xfer.complete";
    Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".$result." ".$response);
    if ($result ne "ok")
    {      
      Dada::logMsgWarn($warn, "touchBeamsXferComplete: ".$cmd." failed: "..$response);
    }

    # touch remote sent.to.swin file on hipsr machines
    $cmd = "touch ".BPSR_PATH."/results/".$obs."/".$beam."/sent.to.parkes";
    Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".BPSR_USER."@".BPSR_HOST." ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
    Dada::logMsg(3, $dl, "touchBeamsXferComplete: ".$result." ".$rval." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "touchBeamsXferComplete: ssh failed to ".BPSR_HOST.": ".$response);
    }
    if ($rval != 0)
    {
      Dada::logMsg(0, $dl, "touchBeamsXferComplete: could not touch ".$obs."/".$beam."/sent.to.parkes on ".BPSR_HOST.": ".$response);
    }
  }
  return ("ok", "");
}

#
# get TCS log file and patch psrxml files
#
sub patchTCSLogs($$)
{
  my ($path, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  # patch TCS log file
  my $tcs_log_file = $obs."_bpsr.log";
  Dada::logMsg(2, $dl, "patchTCSLogs: getting TCS log file: ".$tcs_log_file);
  $cmd = "scp -p -o BatchMode=yes ".TCS_USER."@".TCS_HOST.":".TCS_PATH."/".$tcs_log_file." ".$path."/".$obs."/";
  Dada::logMsg(2, $dl, "patchTCSLogs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "patchTCSLogs: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "failed to copy TCS logfile for ".$obs.": ".$response);
    return ("fail", "could not copy TCS log file");
  }

  # copy the TCS log file to the BPSR results directory on the BPSR_HOST
  $cmd = "scp -p -o BatchMode=yes ".$path."/".$obs."/".$obs."_bpsr.log ".BPSR_USER."@".BPSR_HOST.":".BPSR_PATH."/results/".$obs."/";
  Dada::logMsg(2, $dl, "patchTCSLogs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchTCSLogs: ".$result." ".$response);
  if ($result ne "ok")
  { 
    Dada::logMsg(0, $dl, "patchTCSLogs: ".$cmd." failed : ".$response);
    return ("fail", "could not scp bpsr log file to ".BPSR_USER."@".BPSR_HOST.":".BPSR_PATH."/results/".$obs."/");
  }

  # only merge TCS log files if a psrxml file exists for each beam
  $cmd = "find ".$path."/".$obs." -mindepth 2 -maxdepth 2 -name '*.psrxml' | wc -l";
  Dada::logMsg(2, $dl, "patchTCSLogs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchTCSLogs: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "failed to count psrxml files for ".$obs.": ".$response);
    return ("fail", "could not count psrxml files");
  }

  if ($response == 0)
  {
    Dada::logMsg(2, $dl, "patchTCSLogs: skipping merge_tcs_logs.csh as no psrxml files existed");

    # delete local copy of tcs logfile
    $cmd = "rm -f ".$path."/".$obs."/".$obs."_bpsr.log";
    Dada::logMsg(2, $dl, "patchTCSLogs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "patchTCSLogs: ".$result." ".$response);

    return ("ok", "no psrxml files existed");
  }

  # merge TCS log file with obs.start files
  $cmd = "merge_tcs_logs.csh ".$path."/".$obs." ".$tcs_log_file;
  Dada::logMsg(2, $dl, "patchTCSLogs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "patchTCSLogs: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "could not merge the TCS log file, msg was: ".$response);
    return ("fail", "marge_tcs_logs failed");
  }

  return ("ok", "");
}

#
# Copy the 13 psrxml files to the header_files directory
#
sub copyPsrxmlToDb($$) 
{
  my ($path, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $day = substr $obs, 0, 10;
  my $db_dump_dir = BPSR_PATH."/header_files/";

  # ensure remote dir exists
  $cmd = "mkdir -p ".$db_dump_dir;
  ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
  if (($result ne "ok") || ($rval != 0))
  {
    Dada::logMsg(0, $dl, "copyPsrxmlToDb: ".$cmd." failed : ".$response);
    return ("fail", "could not ensure db_dump_dir existed");
  }

  # see if this obs' psrxml data has been submitted
  $cmd = "find ".$db_dump_dir." -mindepth 2 -name '".$obs."_??.psrxml' | wc -l";
  ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
  if (($result eq "ok") && ($rval == 0))
  {
    my $remote_beam_count = $response;
    if ($remote_beam_count > 0)
    {
      Dada::logMsg(1, $dl, "Skipping psrxml copy for ".$obs);
      return ("ok", $obs." already been submitted");
    }
  }

  # get the local beam list
  $cmd = "find ".$path."/".$obs." -maxdepth 1 -type d -name '??' -printf '\%f\\n' | sort";
  Dada::logMsg(2, $dl, "copyPsrxmlToDb: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "copyPsrxmlToDb: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "copyPsrxmlToDb: ".$cmd." failed : ".$response);
    return ("fail", "could not count local beams");
  }
  my @beam_list = split(/\n/, $response);

  # copy each psrxml file to the DB on hipsr-srv0
  foreach $b ( @beam_list)
  {
    my $local = $path."/".$obs."/".$b."/".$obs.".psrxml";
    my $remote = $db_dump_dir."/".$obs."_".$b.".psrxml";
    if ( -f $local)
    {
      $cmd = "scp -p -o BatchMode=yes ".$local." ".BPSR_USER."@".BPSR_HOST.":".$remote;
      Dada::logMsg(2, $dl, "copyPsrxmlToDb: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "copyPsrxmlToDb: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "copyPsrxmlToDb: ".$cmd." failed : ".$response);
        return ("fail", "could not scp beam ".$b." psrxml file");
      }
    }

  }

  # copy the TCS log file to the BPSR results directory on the BPSR_HOST
  $cmd = "scp -p -o BatchMode=yes ".$path."/".$obs."/".$obs."_bpsr.log ".BPSR_USER."@".BPSR_HOST.":".BPSR_PATH."/results/".$obs."/";
  Dada::logMsg(2, $dl, "copyPsrxmlToDb: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "copyPsrxmlToDb: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "copyPsrxmlToDb: ".$cmd." failed : ".$response);
    return ("fail", "could not scp bpsr log file to ".BPSR_USER."@".BPSR_HOST.":".BPSR_PATH."/results/".$obs."/");
  }

  # delete the local copy of the log file
  $cmd = "rm -f ".$path."/".$obs."/".$obs."_bpsr.log";
  Dada::logMsg(2, $dl, "copyPsrxmlToDb: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "copyPsrxmlToDb: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "copyPsrxmlToDb: ".$cmd." failed : ".$response);
    return ("fail", "could not delete bpsr TCS log file");
  }

  return ("ok", ($#beam_list+1));
}

#
# fix headers of any sigproc files
#
sub fixSigprocFiles($$) 
{
  my ($path, $obs) = @_;

  my $fil_edit = "fil_edit";

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $fn_result = "ok";
  my $fil_file = "";
  my $psrxml_file = "";

  $cmd = "find ".$path."/".$obs." -maxdepth 1 -type d -name '??' -printf '\%f\\n' | sort";
  Dada::logMsg(2, $dl, "fixSigprocFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "fixSigprocFiles: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "fixSigprocFiles: ".$cmd." failed : ".$response);
    return ("fail", "could not count local beams");
  }
  my @beam_list = split(/\n/, $response);

  my $n_beams = ($#beam_list + 1);

  foreach $b ( @beam_list)
  {
    $fil_file    = $path."/".$obs."/".$b."/".$obs.".fil";
    $psrxml_file = $path."/".$obs."/".$b."/".$obs.".psrxml";

    if ((-f $fil_file) && (-f $psrxml_file))
    {
      #$cmd = "fil_edit --beam ".$b." --nbeams ".$n_beams." ".$fil_file;
      $cmd = "fix_fil_header.csh ".$psrxml_file." ".$fil_file." ".$fil_edit." ".$n_beams;
      Dada::logMsg(2, $dl, "fixSigprocFiles: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "fixSigprocFiles: ".$result." ".$response);
      if ($result ne "ok") 
      {
        Dada::logMsgWarn($warn, "fixSigprocFiles: fix_fil_header failed: ".$response);
        $fn_result = "fail";
        $response = "fix_fil_header.csh failed: ".$response;
      }
    }
    else
    {
      Dada::logMsg(2, $dl, "fixSigprocFiles: fil/psrxml file missing for ".$obs."/".$b);
    }
  }

  return ($fn_result, $response);
}

#
# create a zap mask 
#
sub createZapMask($$) 
{
  my ($path, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $file = "";
  my $ts = "";

  my %ts_list = ();

  $cmd = "find ".$path."/".$obs." -mindepth 3 -maxdepth 3 -type f -name '*.ts?' -printf '\%f\\n' | sort -n";
  Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "createZapMask: ".$cmd." failed : ".$response);
    return ("fail", "could not generate ts list for ".$obs);
  }
  my @file_list = split(/\n/, $response);
  if ($#file_list == -1)
  {
    Dada::logMsg(2, $dl, "createZapMask: no .ts? files found [".$response."]");
    return ("ok", "found no .ts? files");
  }
  foreach $file ( @file_list) 
  {
    $ts = substr($file,0,-4);
    if (!exists($ts_list{$ts}))
    {
      $ts_list{$ts} = 0;
    }
    $ts_list{$ts}++;
  }

  my @pol0 = ();
  my @pol1 = ();
  my $pol0_files = "";
  my $pol1_files = "";
  my $pol = "";
  my @ts_keys = sort keys %ts_list;
  foreach $ts ( @ts_keys) 
  {
    Dada::logMsg(2, $dl, "createZapMask: processing ts=".$ts);
    @pol0 = ();
    @pol1 = ();
    $pol0_files = "";
    $pol1_files = "";

    # pol0 list
    $cmd = "find ".$path."/".$obs." -mindepth 3 -maxdepth 3 -type f -name '".$ts.".ts0'";
    Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
    @pol0 = split(/\n/, $response);

    # pol1 list
    $cmd = "find ".$path."/".$obs." -mindepth 3 -maxdepth 3 -type f -name '".$ts.".ts1'";
    Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
    @pol1 = split(/\n/, $response);

    if ($#pol0 != $#pol1)
    {
      Dada::logMsg(0, $dl, "createZapMask: skipping ".$ts."ts? pol count mismatch [".($#pol0+1)." != ".($#pol1+1)."]");
      next;
    }

    # if we dont have at least 10 beams, skip SVD
    if ($#pol0 < 10)
    {
      Dada::logMsg(0, $dl, "createZapMask: skipping ".$ts."ts? not enough beams [".($#pol0+1)."]");
      next;
    }

    foreach $pol ( @pol0 ) { $pol0_files .= " ".$pol; }   
    foreach $pol ( @pol1 ) { $pol1_files .= " ".$pol; }

    my $num_beam = ($#pol0 + 1);
    $cmd = "dgesvd_aux_p -m ".$num_beam." -n ".$num_beam." -o ".$path."/".$obs."/rfi.mask ".$pol0_files." ".$pol1_files;

    Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "processMonFiles: dgesvd_aux_p failed: ".$response);
      Dada::logMsg(0, $dl, "pol0_files = ".$pol0_files);
      Dada::logMsg(0, $dl, "pol1_files = ".$pol1_files);
      Dada::logMsg(0, $dl, "cmd = ".$cmd);
    }

    if ( -f $path."/".$obs."/rfi.log")
    {
      $cmd = "echo ".$ts." >> ".$path."/".$obs."/rfi.log";
    }
    else
    {
     $cmd = "echo ".$ts." > ".$path."/".$obs."/rfi.log";
    }

    Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);

  }

  # gzip the rfi.mask
  $cmd = "gzip ".$path."/".$obs."/rfi.mask";
  Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "createZapMask: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "createZapMask: could not gzip rfi.mask for ".$obs.": ".$response);
     return ("fail", "could not gzip rfi.mask for ".$obs);
  }

  # get the local beam list
  $cmd = "find ".$path."/".$obs." -maxdepth 1 -type d -name '??' -printf '\%f\\n'";
  Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "createZapMask: ".$cmd." failed : ".$response);
    return ("fail", "could not count local beams");
  }
  my @beam_list = split(/\n/, $response);

  foreach $b ( @beam_list )
  {
    $cmd = "cp ".$path."/".$obs."/rfi.mask.gz ".$path."/".$obs."/rfi.log ".$path."/".$obs."/".$b."/";
    Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "createZapMask: ".$cmd." failed : ".$response);
      return ("fail", "could not copy rfi mask and log to beam subdir");
    }
  }

  $cmd = "rm -f ".$path."/".$obs."/rfi.mask.gz ".$path."/".$obs."/rfi.log";
  Dada::logMsg(2, $dl, "createZapMask: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createZapMask: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "createZapMask: ".$cmd." failed : ".$response);
    return ("fail", "could not remove rfi mask and log");
  }

  return ("ok", "");
}

#
# For each beam in the obs, tar up all files in the aux dir and
#
sub tarAuxFiles($$)
{
  my ($path, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  # get the local beam list
  $cmd = "find ".$path."/".$obs." -maxdepth 1 -type d -name '??' -printf '\%f\\n'";
  Dada::logMsg(2, $dl, "tarAuxFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tarAuxFiles: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "tarAuxFiles: ".$cmd." failed : ".$response);
    return ("fail", "could not count local beams");
  }
  my @beam_list = split(/\n/, $response);

  foreach $b ( @beam_list )
  {
    # check there are some files worth tarring up
    $cmd = "find ".$path."/".$obs."/".$b."/aux -type f -regex '.*[ts|bp|bps|cand].' | wc -l";
    Dada::logMsg(2, $dl, "tarAuxFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "tarAuxFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "tarAuxFiles: ".$cmd." failed : ".$response);
      return ("fail", "could not count aux files for ".$obs."/".$b);
    }
    if ($response > 0)
    {
      # tar all the files up
      $cmd = "tar -C ".$path."/".$obs."/".$b." -cf ".$path."/".$obs."/".$b."/aux.tar aux";
      Dada::logMsg(2, $dl, "tarAuxFiles: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "tarAuxFiles: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "tarAuxFiles: ".$cmd." failed : ".$response);
        return ("fail", "could not create aux.tar for ".$b);
      }
    }
    else
    {
      Dada::logMsg(0, $dl, "tarAuxFiles: found 0 aux files for ".$obs."/".$b);
    }

    $cmd = "rm -rf ".$path."/".$obs."/".$b."/aux";
    Dada::logMsg(2, $dl, "tarAuxFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "tarAuxFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "tarAuxFiles: ".$cmd." failed : ".$response);
    }
  }

  Dada::logMsg(2, $dl, "tarAuxFiles: processed all beams");
  return ("ok", "");

}

#
# control thread to ask daemon to quit
#
sub controlThread($$) 
{
  my ($quit_file, $pid_file) = @_;
  Dada::logMsg(2, $dl, "controlThread: starting");

  my $cmd = "";
  my $regex = "";
  my $result = "";
  my $response = "";

  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl, "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(2, $dl, "controlThread: exiting");

  return 0;
}

#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # tell threads to try and quit
  if (($sigName ne "INT") || ($quit_daemon))
  {
    $quit_daemon = 1;
    sleep(3);
  
    print STDERR $daemon_name." : Exiting\n";
    exit 1;
  }
  $quit_daemon = 1;
}

#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
}

#
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
