#!/usr/bin/env perl 

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Process mon files associated with the obseravation producing an RFI mask
# that can be of assistance during later processing
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;

use File::Basename;
use threads;
use threads::shared;
use Bpsr;

#
# Function Prototypes
#
sub good($);
sub getOldestProcessingObs();
sub processMonFiles($);
sub finalizeRFIFiles($);

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Global Variable Declarations
#
our %cfg = Bpsr::getConfig();
our $dl = 1;
our $daemon_name = Dada::daemonBaseName($0);
our $quit_daemon : shared = 0;
our $error = "";
our $warn  = "";


#
# Signal Handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

#
# Local variables
#
my $control_thread = 0;
my $pid_file = "";
my $quit_file = "";
my $log_file = "";
my $cmd = "";
my $result = "";
my $response = "";
my $o = "";
my $sleep_time = 0;
my $processing_file = "";
my $finished_file = "";
my $curr_obs = "";

# Autoflush output
$| = 1;

#
# Main
#
{

  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";

  $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $result." ".$response."\n";
    exit(1);
  }

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  chdir $cfg{"SERVER_RESULTS_DIR"};

  while (!$quit_daemon)
  {

    # get the oldest observation that is still marked as obs.processing
    Dada::logMsg(3, $dl, "main: getOldestProcessingObs()");
    ($result, $response) = getOldestProcessingObs();
    Dada::logMsg(3, $dl, "main: getOldestProcessingObs() ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "getOldestProcessingObs() failed: ".$response);
      $sleep_time = 10;
    }
    else 
    { 

      # if we have an observation to consider
      if ($response ne "none")
      {
        $sleep_time = 2;
        $o = $response;

        $processing_file = $cfg{"SERVER_RESULTS_DIR"}."/".$o."/rfi.processing";
        $finished_file = $cfg{"SERVER_RESULTS_DIR"}."/".$o."/rfi.finished";

        # if no rfi.processing and no rfi.finished, touch rfi.processing
        if ((! -f $processing_file) && (! -f $finished_file))
        {
          $cmd = "touch ".$processing_file;
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok") {
            Dada::logMsgWarn($error, "main: could not create ".$processing_file.": ".$response);
            $quit_daemon = 1;
            next;
          }
          if ($o ne $curr_obs) 
          {
            Dada::logMsg(1, $dl, $o." new -> processing");
            $curr_obs = $o;
          }
        }
        
        if (! -f $finished_file) 
        {
          # process any mon files that exist, if no mon files appearing in 
          # this observation after 60 seconds, then move rfi.processing to
          # rfi.done so that the results manager knows this bservation can 
          # be marked as finished
          Dada::logMsg(3, $dl, "main: processMonFiles(".$o.")");
          ($result, $response) = processMonFiles($o);
          Dada::logMsg(2, $dl, "main: processMonFiles() ".$result." ".$response);
          if ($result ne "ok") {
            Dada::logMsgWarn($error, "main: processMonFiles(".$o.") failed: ".$response);
            $quit_daemon = 1;
            next;
          }

          # if not mon files appeared for 60 seconds, move rfi.processing to rfi.done
          if ($response eq "finalize") {

            Dada::logMsg(2, $dl, "main: finalizeRFIFiles(".$o.")");
            ($result, $response) = finalizeRFIFiles($o);
            Dada::logMsg(2, $dl, "main: finalizeRFIFiles() ".$result." ".$response);
            if ($result ne "ok") {
              Dada::logMsgWarn($error, "main: could not finalize RFI files");
              $quit_daemon = 1;
              next;
            }
          }
        } 
      }
      else
      {
        $sleep_time = 10;
      }
    }

    # sleep for the approriate amount of time
    while (!$quit_daemon && $sleep_time > 0)
    {
      sleep(1);
      $sleep_time--;
    }
  }

  Dada::logMsg(2, $dl, "main: joining threads");

  # Rejoin our daemon control thread
  $control_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

}

exit 0;



###############################################################################
#
# Functions
#

#
# Find the oldest observation that is still marked as obs.processing
#
sub getOldestProcessingObs()
{
  Dada::logMsg(3, $dl, "getOldestProcessingObs()");

  my $cmd = "";
  my $result = "";
  my $response = "";
  
  $cmd = "find . -mindepth 2 -maxdepth 2 -name 'obs.processing' -printf '\%T@ \%h\n' | sort -rn | tail -n 1 | awk -F/ '{print \$2}'";
  Dada::logMsg(3, $dl, "getOldestProcessingObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getOldestProcessingObs: ".$result." ".$response);
  if ($result  ne "ok")
  {
    Dada::logMsgWarn($error, "getOldestProcessingObs: find obs.processing failed: ".$response);
    return ("fail", "find failed");
  } 

  if ($response eq "")
  {
    Dada::logMsg(2, $dl, "getOldestProcessingObs: could not find matching obs");
    return ("ok", "none");
  }
  else
  {
    Dada::logMsg(2, $dl, "getOldestProcessingObs: found obs ".$response);
    return ("ok", $response);
  }
}

#
# process any mon files that exist for the specified observation
# return the time since the last mon file was processed
#
sub processMonFiles($)
{

  my ($o) = @_;

  Dada::logMsg(3, $dl, "processMonFiles(".$o.")");

  my $rfi_log = $o."/rfi.log";
  my $next_mon_utc = "";
  my $num_pwc = 0;
  my $num_mon_files = 0;
  my $num_files_found = 0;
  my $pol0_files = "";
  my $pol1_files = "";
  my $mon_age = -1;
  my @bits = ();

  my $cmd = "";
  my $result = "";
  my $response = "";

  # check if this has been marked as completed
  if ( -f $o."/rfi.finished") {
    Dada::logMsg(2, $dl, "processMonFiles: already completed");
    return ("ok", 0);
  }
  
  # determine the number of PWCS
  $cmd = "grep NUM_PWC ".$o."/obs.info | awk '{print \$2}'";
  Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "processMonFiles: could not determine NUM_PWC from obs.info: ".$response);
    return ("fail", "could not determin NUM_PWC");
  } 
  $num_pwc = $response;
  Dada::logMsg(2, $dl, "processMonFiles: [".$o."] NUM_PWC=".$num_pwc);

  if (($num_pwc < 1) || ($response > 13))
  {
    Dada::logMsgWarn($error, "processMonFiles: NUM_PWC [".$num_pwc."] invalid");
    return ("fail", "NUM_PWC was invalid [".$num_pwc."]");
  }

  $num_mon_files = 2 * $num_pwc;

  # determine the utc of the next set of mon files to be processed
  if ( -f $rfi_log ) 
  {
    $cmd = "tail -n 1 ".$rfi_log;
    Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "processMonFiles: could not extract last file from ".$rfi_log);
      return ("fail", "could not extract last processed mon file");
    }

    Dada::logMsg(3, $dl, "processMonFiles: addToTime(".$response.", 10)");
    $next_mon_utc = Dada::addToTime($response, 10);
    Dada::logMsg(3, $dl, "processMonFiles: addToTime() ".$next_mon_utc);
  }
  else
  {
    $next_mon_utc = $o;
  }
  Dada::logMsg(2, $dl, "processMonFiles: [".$o."] next_mon_utc=".$next_mon_utc);

  # count the specified mon files  
  $cmd = "find -L ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o." -mindepth 3 -maxdepth 3 -perm -a+r -name '".$next_mon_utc.".ts?' -printf '\%f\n'";
  Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processMonFiles: find: ".$result." ".$response);
  if ($result ne "ok") 
  {
    $response =~ s/\n/ /g;
    Dada::logMsg(0, $dl, "processMonFiles:  find cmd to count mon files failed: ".$response);
    return ("ok", 0);
  }

  if ($response =~ m/No such file or directory/)
  {
    Dada::logMsg(1, $dl, "processMonFiles: find encountered minor error");
    $num_files_found = 0;
  }
  else
  {
    @bits = split(/\n/, $response);
    $num_files_found = $#bits + 1;
  }

  Dada::logMsg(2, $dl, "processMonFiles: num_files_found=".$num_files_found);

  if ($num_files_found> 0)
  {

    # get the pol0 list
    $cmd = "find -L ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o." -mindepth 3 -maxdepth 3 ".
           "-perm -a+r -name '".$next_mon_utc.".ts0' -printf '\%h/\%f '";
    Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    if (($result ne "ok") || ($response =~ m/No such file or directory/))
    {
      Dada::logMsgWarn($error, "processMonFiles: could not create list of mon files for pol0");
    }
    else 
    {
      $pol0_files = $response;
    }

    # get the pol1 list
    $cmd = "find -L ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o." -mindepth 3 -maxdepth 3 ".
           "-perm -a+r -name '".$next_mon_utc.".ts1' -printf '\%h/\%f '";
    Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    if (($result ne "ok") || ($response =~ m/No such file or directory/))
    {
      Dada::logMsgWarn($error, "processMonFiles: could not create list of mon files for pol1");
    }
    else
    {
      $pol1_files = $response;
    }
  }
   
  # we have all the mon files and can now process them
  if (($num_files_found == $num_mon_files) && ($pol0_files ne "") && ($pol1_files ne ""))
  {

    Dada::logMsg(1, $dl, $o." processing ".$next_mon_utc.".ts?");

    # perform the RFI test
    $cmd = "dgesvd_aux_p -m ".$num_pwc." -n ".$num_pwc." -o ".$o."/rfi.mask ".$pol0_files." ".$pol1_files ;
    Dada::logMsg(2, $dl, "processMonFiles: [".$o."] ".substr($cmd,0,60)."...");
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "processMonFiles: dgesvd_aux_p failed: ".$response);
      Dada::logMsg(0, $dl, "pol0_files = ".$pol0_files);
      Dada::logMsg(0, $dl, "pol1_files = ".$pol1_files);
      Dada::logMsg(0, $dl, "cmd = ".$cmd);
    }

    # append this mon_utc to the list
    if ( -f $rfi_log ) 
    {
      $cmd = "echo ".$next_mon_utc." >> ".$rfi_log;
    } 
    else
    {
       $cmd = "echo ".$next_mon_utc." > ".$rfi_log;
    }
    Dada::logMsg(2, $dl, "processMonFiles: [".$o."] ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "processMonFiles: could not add ".$next_mon_utc." to ".$rfi_log);
      return ("fail", "could not add mon UTC [".$next_mon_utc."] to rfi.log");
    }

    return ("ok", 0);

  }
  elsif ($num_files_found > $num_mon_files) 
  {
    Dada::logMsgWarn($error, "processMonFiles: found ".$num_files_found." mon files, expecting ".$num_mon_files);
    return ("fail", "found too many mon files [".$num_files_found."] for ".$o.", expecting ".$num_mon_files);
  }
  # if we dont have enough mon files, check the age of the rfi.log file
  else
  {
    Dada::logMsg(2, $dl, "processMonFiles: not enough mon files: found=".$num_files_found.", num_required=".$num_mon_files);
    my $file_to_check = "rfi.processing";
    if ( -f $rfi_log ) 
    {
      $file_to_check = "rfi.log";
    }

    my $time_curr = time;
    $cmd = "find ".$o." -mindepth 1 -maxdepth 1 -type f -name '".$file_to_check."' -printf '\%T\@'";
    Dada::logMsg(3, $dl, "processMonFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processMonFiles: ".$result." ".$response);
    my $time_file = $response;

    my $age = ($time_curr - $time_file);

    Dada::logMsg(2, $dl, "processMonFiles: [".$o."] age=".$age);
    if ($age > 60) {
      return ("ok", "finalize");
    } else {
      return ("ok", $age);
    }
  }
}

#
# Finalizes the RFI files by gzipping the rfi.mask then copying that mask
# to each of the beams archives directories for tape archive
#
sub finalizeRFIFiles($) 
{

  my ($o) = @_;

  Dada::logMsg(2, $dl, "finalizeRFIFiles(".$o.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my @beams = ();

  # gzip the rfi.mask file

  if ( -f $o."/rfi.mask" ) 
  {

    $cmd = "gzip ".$o."/rfi.mask";
    Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "finalizeRFIFiles: could not gzip rfi.mask for ".$o.": ".$response);
      return ("fail", "could not gzip rfi.mask for ".$o);
    }

    # get the list of beam subdirs
    $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o." -name '??' -type l -printf '\%f\n' | sort";
    Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "finalizeRFIFiles: could not find beam dirs for  ".$o.": ".$response);
      return ("fail", "could not get list of beam dirs for ".$o);
    }
    @beams = split(/\n/, $response);

    # copy the zipped rfi.mask and rfi.log to each beam directory
    for ($i=0; $i<=$#beams; $i++)
    {
      $cmd = "cp ".$o."/rfi.mask.gz ".$o."/rfi.log ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o."/".$beams[$i]."/";
      Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($error, "finalizeRFIFiles: could not copy RFI files to beam ".$beams[$i]." for ".$o.": ".$response);
        return ("fail", "could not copy files to beam ".$beams[$i]." for ".$o);
      }
    }
  }

  # move the processing to finished
  $cmd = "mv ".$o."/rfi.processing ".$o."/rfi.finished";
  Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "finalizeRFIFiles: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "finalizeRFIFiles: could not rename rfi.processing to rfi.finished for ".$o.": ".$response);
    return ("fail", "could not rename processing to finished");
  }

  Dada::logMsg(1, $dl, $o." processing -> finished");

  return ("ok", "");
}

#
# Test the configuration to ensure this daemon can run
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

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}


#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  #print STDERR basename($0)." : Exiting: ".Dada::getCurrentDadaTime(0)."\n";
  #exit(1);

}

#
# Control thread to handle quit requests
#
sub controlThread($$) 
{
  Dada::logMsg(2, $dl, "controlThread: starting");

  my ($quit_file, $pid_file) = @_;
  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }
  
  Dada::logMsg(2, $dl ,"controlThread: exiting");

} 
