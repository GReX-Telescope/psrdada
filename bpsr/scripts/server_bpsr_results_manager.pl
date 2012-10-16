#!/usr/bin/env perl 

#
# Author:   Andrew Jameson
# Created:  6 Dec, 2007
# Modified: 9 Jan, 2008
#
# This daemons runs continuously produces feedback plots of the
# current observation

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;               # strict mode (like -Wall)
use File::Basename;
use threads;
use threads::shared;
use Bpsr;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DL           => 1;


#
# Global Variable Declarations
#
our %cfg = Bpsr::getConfig();
our $quit_daemon : shared = 0;
our $daemon_name : shared = Dada::daemonBaseName($0);
our $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
our $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";

#
# Signal Handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) 
{
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}

#
# Main Loop
#
{

  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $obs_results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $control_thread = 0;
  my $coincidencer_thread = 0;

  my $cmd;
  my $dir;
  my @subdirs;

  my $i;
  my $j;

  my $result = "";
  my $response = "";
  my $counter = 0;

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # Redirect standard output and error
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, DL, "STARTING SCRIPT");

  chdir $cfg{"SERVER_RESULTS_DIR"};

  # Start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  # start the coincidencer thread 
  # $coincidencer_thread = threads->new(\&coincidencerThread);

  my $curr_processing = "";
  
  while (!$quit_daemon)
  {
    $dir = "";
    @subdirs = ();

    # TODO check that directories are correctly sorted by UTC_START time
    Dada::logMsg(2, DL, "Main While Loop, looking for data in ".$obs_results_dir);

    # get the list of all the current observations (should only be 1)
    $cmd = "find ".$obs_results_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.processing' ".
           "-printf '\%h\\n' | awk  -F/ '{print \$NF}' | sort";
    Dada::logMsg(2, DL, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "main: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "main: find command failed: ".$response);

    } elsif ($response eq "") {
      Dada::logMsg(2, DL, "main: nothing to process");

      Dada::logMsg(2, DL, "main: checkDeletedObs()");
      ($result, $response) = checkDeletedObs();
      Dada::logMsg(2, DL, "main: checkDeletedObs() ".$result ." ".$response);

    } else {

      @subdirs = split(/\n/, $response);
      Dada::logMsg(2, DL, "main: found ".($#subdirs+1)." obs.processing files");

      my $h=0;
      my $age = 0;
      my $n_png_files = 0;
      my $n_cand_files = 0;

      # For each observation
      for ($h=0; (($h<=$#subdirs) && (!$quit_daemon)); $h++) 
      {
        $dir = $subdirs[$h];

        Dada::logMsg(2, DL, "main: testing ".$subdirs[$h]);

        # If this observation has not been finished, archived or transferred
        if (!( (-f $dir."/obs.archived") || (-f $dir."/obs.finished") || (-f $dir."/obs.deleted") ||
               (-f $dir."/obs.transferred") ) ) {

          # determine the age of the observation and number of files that can be processed in it
          Dada::logMsg(2, DL, "main: getObsAge(".$dir.")");
          ($age, $n_png_files, $n_cand_files) = getObsAge($dir); 
          Dada::logMsg(2, DL, "main: getObsAge() ".$age." ".$n_png_files." ".$n_cand_files);

          # If nothing has arrived in 120 seconds, mark it as failed
          if ($age < -120) 
          {
            markObsState($dir, "processing", "failed");
          } 
          # If the obs age is over five minutes, then we consider it finished
          elsif ($age > 60) 
          {

            Dada::logMsg(1, DL, "Finalising observation: ".$dir);

            # remove any old files in the archive dir
            cleanUpObs($obs_results_dir."/".$dir);

            # we need to patch the TCS logs into the file!
            # ($result, $response) = patchInTcsLogs($obs_archive_dir, $dir);
            # if ($result ne "ok") {
            #   Dada::logMsgWarn($warn, "patchInTcsLogs failed for ".$dir.": ".$response);
            #   markObsState($dir, "processing", "failed");
            #   next;
            # }

            # If the TCS log file didn't exist, we cannot do the rest of these steps
            # if ($response eq "failed to get TCS log file") {
            #   markObsState($dir, "processing", "finished");
            #   next;
            # }
        
            # copy psrxml files to special dir for mike to deal with
            # ($result, $response) = copyPsrxmlToDb($obs_archive_dir, $dir);
            # if ($result ne "ok") {
            #   Dada::logMsgWarn($warn, "copyPsrxmlToDb failed for ".$dir.": ".$response);
            #   markObsState($dir, "processing", "failed");
            #   $quit_daemon = 1;
            #   next;
            # }

            # only observations run with the_decimator produce .fil and .psrxml files, so 
            # check the reponse from copyPsrxmlToDb to find out if we have psrxml files
            # if ($response > 0) {

              # fix the .fil headers from the psrxml file
            #   ($result, $response) = fixFilHeaders($obs_archive_dir, $dir);
            #   if ($result ne "ok") {
            #     Dada::logMsgWarn($warn, "fixFilHeaders failed for ".$dir.": ".$response);
            #     markObsState($dir, "processing", "failed");
            #     $quit_daemon = 1;
            #     next;
            #   }
            # }

            # Otherwise all the post processing parts are ok and we can mark this obs
            # as finished, and thereby ready for xfer
            markObsState($dir, "processing", "finished");

          } 
          # Else this is an active observation, try to process the .pol
          # files that may exist in each beam
          else 
          {
            if ($n_png_files > 0) 
            {
              if ($dir eq $curr_processing) {
                Dada::logMsg(2, DL, "Received ".$n_png_files." png files for ".$dir);
              } else {
                $curr_processing = $dir;
                Dada::logMsg(1, DL, "Receiving png files for ".$dir);
              }

              # Now delete all old png files
              removeAllOldPngs($dir);
            }

            if ($n_cand_files > 0)
            {
              if ($dir eq $curr_processing) {
                Dada::logMsg(2, DL, "Received ".$n_cand_files." cand files for ".$dir);
              } else {
                $curr_processing = $dir;
                Dada::logMsg(1, DL, "Receiving cand files for ".$dir);
              }

              ($result, $response) = processCandidates($dir);
              removeOldPngs($dir, "cands", "1024x768");
              removeOldPngs($dir, "dm_vs_time", "700x240");
            }
          }
        }
      } 
    }

    # If we have been asked to exit, dont sleep
    $counter = 5;
    while ((!$quit_daemon) && ($counter > 0)) {
    sleep(1);
    $counter--;
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  # $coincidencer_thread->join();
                                                                                
  Dada::logMsg(0, DL, "STOPPING SCRIPT");

  exit(0);
}

###############################################################################
#
# Functions
#

sub processCandidatesNew($)
{
  (my $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $cands_record = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/all_candidates.dat";
  my @files = ();
  my $file = "";
  my $n_processed = 0;

  # for the given obs, a list of the candiate files to be parsed beams 
  $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$obs." -mindepth 1 -maxdepth 1 -type f -name '2*_all.cand' | sort";
  Dada::logMsg(2, DL, "processCandidates: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

  if ($result ne "ok")
  {
    Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    Dada::logMsgWarn($warn, "could not find candidate files");
    return ("fail", "could not get candiate list");
  }

  # count the number of candidate files to each epoch
  @files = split(/\n/, $response);
  foreach $file ( @files )
  {
    # now append the .cands files to the accumulated total for this observation
    if (-f $cands_record)
    {
      $cmd = "cat ".$file." >> ".$cands_record;
    }
    else
    {
      $cmd = "cp ".$file." ".$cands_record;
    }
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
      Dada::logMsgWarn($warn, "could not append output to record");
      return ("fail", "could not append output to record");
    }

    # delete the intermin candidates file
    unlink $file;

    $n_processed++;
  }

  if ($n_processed > 0)
  {
    # generate 1024x768 large plot and candidate XML list
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -cands_file ".$obs."/all_candidates.dat -snr_cut 6.5 -filter_cut 11 -cand_list_xml > ".$obs."/cand_list.xml";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    }

    my $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_1024x768.tmp.png ".$obs."/".$curr_time.".cands_1024x768.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    }

    # generate 700x 240 small plot
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -cands_file ".$obs."/all_candidates.dat -snr_cut 7.0 -filter_cut 11 -just_time_dm -resolution 700x240";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    }

    $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_700x240.tmp.png ".$obs."/".$curr_time.".dm_vs_time_700x240.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    }
  }
}

sub processCandidates($)
{
  (my $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $beam = "";
  my $file = "";
  my @files = ();
  my %cands = ();
  my @bits = ();

  my $file_prefix = "";
  my $junk = "";
  my $cands_record = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/all_candidates.dat";
  my $output_file = "";
  my $n_processed = 0;

  # for the given obs, build a list of the candiate beams 
  $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$obs." -mindepth 2 -maxdepth 2 -type f -name '2*_??.cand'";
  Dada::logMsg(2, DL, "processCandidates: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
  
  if ($result ne "ok")
  {
    Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    Dada::logMsgWarn($warn, "could not find candidate files");
    return ("fail", "could not get candiate list");
  }
 
  # count the number of candidate files to each epoch
  @files = split(/\n/, $response);
  foreach $file ( @files )
  {
    @bits = split(/\//, $file);
    
    $beam = $bits[$#bits - 1];
    $file = $bits[$#bits - 0];
    ($file_prefix, $junk) = split(/_/, $file, 2); 

    if (! exists($cands{$file_prefix}))
    {
      $cands{$file_prefix} = 0;
    }
    $cands{$file_prefix} += 1;
  }

  # now do some plotting
  chdir $cfg{"SERVER_RESULTS_DIR"}."/".$obs;

  @files = sort keys %cands;
  foreach $file ( @files )
  {
    # TODO better logic 
    if (($cands{$file} == $cfg{"NUM_PWC"}) || ($#files > 0))
    {
      $cmd = "coincidencer ./??/".$file."_??.cand";
      Dada::logMsg(2, DL, "processCandidates: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "coincidencer failed to process candidates");
        return ("fail", "could not process candidates");
      }

      # remove beam candidate files
      $cmd = "rm -f ./??/".$file."_??.cand";
      Dada::logMsg(2, DL, "processCandidates: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "could not removed processed candidates");
        return ("fail", "could not remove processed candidates");
      }
      
      $output_file = $file."_all.cand";

      # now append the .cands files to the accumulated total for this observation
      if (-f $cands_record)
      {
        $cmd = "cat ".$output_file." >> ".$cands_record;
      }
      else
      {
        $cmd = "cp ".$output_file." ".$cands_record;
      }
      Dada::logMsg(2, DL, "processCandidates: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "could not append output to record");
        return ("fail", "could not append output to record");
      }

      # delete the intermin candidates file
      unlink $output_file;

      $n_processed++;
    }
    else
    {
      Dada::logMsg(2, DL, "processCandidates: skipping ".$obs."/??/".$file." as only ".$cands{$file}." candidates present");
    }
  }
 
  if ($n_processed > 0)
  {
    # generate 1024x768 large plot and candidate XML list
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -snr_cut 6.5 -filter_cut 11 -cand_list_xml > cand_list.xml";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    my $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_1024x768.tmp.png ".$curr_time.".cands_1024x768.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    # generate 700x 240 small plot
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -snr_cut 7.0 -filter_cut 11 -just_time_dm -resolution 700x240";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_700x240.tmp.png ".$curr_time.".dm_vs_time_700x240.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

  }

  chdir $cfg{"SERVER_RESULTS_DIR"};
}

# Handle INT AND TERM signals
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada::getCurrentDadaTime(0)."\n";
  exit(1);

}
                                                                                
sub controlThread($) 
{
  (my $pid_file) = @_;
  Dada::logMsg(2, DL, "controlThread: starting");

  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  my $result = "";
  my $response = "";

  Dada::logMsg(0, DL, "controlThread: killProcess(^coincidencer)");
  ($result, $response) = Dada::killProcess("^coincidencer");
  Dada::logMsg(0, DL, "controlThread: ".$result." ".$response);

  

  if (-f $pid_file)
  {
    Dada::logMsg(2, DL, "controlThread: unlinking PID file: ".$pid_file);
    unlink($pid_file);
  }

  Dada::logMsg(2, DL, "controlThread: exiting");

}

#
# Determine the age in seconds of this observation
#
sub getObsAge($) {

  (my $o) = @_;
  Dada::logMsg(3, DL, "getObsAge(".$o.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $num_beams = 0;
  my $num_png_files = 0;
  my $num_cand_files = 0;
  my $age = 0;
  my $now = 0;

  # number of beam subdirectories
  $cmd = "find ".$o." -mindepth 1 -maxdepth 1 -type d | wc -l";
  Dada::logMsg(3, DL, "getObsAge: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

  $num_beams = $response;
  Dada::logMsg(2, DL, "getObsAge: num_beams=".$num_beams);
  
  # If the clients have created the beam directories
  if ($num_beams> 0) {

    # Get the number of png files 
    $cmd = "find ".$o." -mindepth 2 -name '*.png' -printf '\%f\\n' | wc -l";
    Dada::logMsg(3, DL, "getObsAge: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

    $num_png_files = $response;    
    Dada::logMsg(2, DL, "getObsAge: num_png_files=".$num_png_files);

    # Get the number of candidate files
    $cmd = "find ".$o." -mindepth 2 -name '2*_??.cand' -printf '\%f\\n' | wc -l";
    Dada::logMsg(3, DL, "getObsAge: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

    $num_cand_files = $response;
    Dada::logMsg(2, DL, "getObsAge: num_cand_files=".$num_cand_files);

    # If we have any png files
    if ($num_png_files > 0) {

      # Get the time of the most recently created file, in any beam
      $cmd  = "find ".$o." -mindepth 2 -name '*.png' -printf '\%T@\\n' | sort -n | tail -n 1";
      Dada::logMsg(3, DL, "getObsAge: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

      $now = time;
      $age = $now - $response;
      Dada::logMsg(2, DL, "getObsAge: newest png file was ".$age." seconds old");
    
    # we haven't received/processed any png files, use obs.start
    } else {

      $cmd  = "find ".$o." -mindepth 2 -mindepth 2 -name 'obs.start' -printf '\%T@\\n' | sort -n | tail -n 1";
      Dada::logMsg(3, DL, "getObsAge: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

      # this is a "negative age" on purpose to indicate failure of this obs
      if ($response ne "") {
        $now = time;
        $age = $response - $now;
        Dada::logMsg(2, DL, "getObsAge: newest obs.start was ".$age." seconds old");

      # if we dont even have any obs.start files >.<
      } else {

        $cmd = "find ".$o." -maxdepth 0 -type d -printf \"%T@\\n\"";
        Dada::logMsg(3, DL, "getObsAge: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

        $now = time;
        $age = $response - $now;
        Dada::logMsg(2, DL, "getObsAge: obs dir was ".$age." seconds old");

      }
    }

  # no beam directories, so use the observation dir
  } else {

    $cmd = "find ".$o." -maxdepth 0 -type d -printf \"%T@\\n\"";
    Dada::logMsg(3, DL, "getObsAge: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "getObsAge: ".$result." ".$response);

    $now = time;
    $age = $response - $now;
    Dada::logMsg(2, DL, "getObsAge: obs dir was ".$age." seconds old");

  }

  return ($age, $num_png_files, $num_cand_files);
}


#
# Cleans up an observation, removing all old files, but leaves 1 set of image files
#
sub cleanUpObs($) {

  (my $dir) = @_;

  my $file = "";
  my @files = ();
  my $cmd = "";

  Dada::logMsg(2, DL, "cleanUpObs(".$dir.")");

  # Clean up all the mon files that may have been produced
  Dada::logMsg(2, DL, "cleanUpObs: removing any ts, bp or bps files");
  my $cmd = "find ".$dir." -regex \".*[ts|bp|bps][0|1]\" -printf \"%P\n\"";
  Dada::logMsg(3, DL, "find ".$dir." -regex \".*[ts|bp|bps][0|1]\" -printf \"\%P\"");
  my $find_result = `$cmd`;

  @files = split(/\n/,$find_result);
  foreach $file (@files) {
    Dada::logMsg(3, DL, "unlinking $dir."/".$file");
    unlink($dir."/".$file);
  }

  # Clean up all the old png files, except for the final ones
  Dada::logMsg(2, DL, "cleanUpObs: removing any old png files");
  removeAllOldPngs($dir);
  removeOldPngs($dir, "cands", "1024x768");

}

#
# Remove all old images in the sub directories
#
sub removeAllOldPngs($) {

  (my $dir) = @_;

  Dada::logMsg(2, DL, "removeAllOldPngs(".$dir.")");

  my $beamdir = "";
  my @beamdirs = ();
  my $i=0;

  # Get the list of beams
  opendir(DIR, $dir);
  @beamdirs = sort grep { !/^\./ && -d $dir."/".$_ } readdir(DIR);
  closedir DIR;
                                                                                                                                            
  # Foreach beam dir, delete any old pngs
  for ($i=0; (($i<=$#beamdirs) && (!$quit_daemon)); $i++) {

    $beamdir = $dir."/".$beamdirs[$i];
    Dada::logMsg(2, DL, "removeAllOldPngs: clearing out ".$beamdir);

    removeOldPngs($beamdir, "fft", "1024x768");
    removeOldPngs($beamdir, "fft", "400x300");
    removeOldPngs($beamdir, "fft", "112x84");

    removeOldPngs($beamdir, "bp", "1024x768");
    removeOldPngs($beamdir, "bp", "400x300");
    removeOldPngs($beamdir, "bp", "112x84");

    removeOldPngs($beamdir, "bps", "1024x768");
    removeOldPngs($beamdir, "bps", "400x300");
    removeOldPngs($beamdir, "bps", "112x84");

    removeOldPngs($beamdir, "ts", "1024x768");
    removeOldPngs($beamdir, "ts", "400x300");
    removeOldPngs($beamdir, "ts", "112x84");

    removeOldPngs($beamdir, "pvf", "1024x768");
    removeOldPngs($beamdir, "pvf", "400x300");
    removeOldPngs($beamdir, "pvf", "112x84");

  }
}

sub removeOldPngs($$$) 
{

  my ($dir, $type, $res) = @_;

  # remove any existing plot files that are more than 30 seconds old
  my $cmd  = "find ".$dir." -name '*".$type."_".$res.".png' -printf \"%T@ %f\\n\" | sort -n -r";
  my $result = `$cmd`;
  my @array = split(/\n/,$result);

  my $time = 0;
  my $file = "";
  my $line = "";
  my $i = 0;

  # if there is more than one result in this category and its > 20 seconds old, delete it
  for ($i=1; $i<=$#array; $i++) {

    $line = $array[$i];
    ($time, $file) = split(/ /,$line,2);

    if (($time+30) < time)
    {
      $file = $dir."/".$file;
      Dada::logMsg(3, DL, "unlinking old png file ".$file);
      unlink($file);
    }
  }
}


#
# Get the TCS log file for this observation and patch the data
# into the obs.start file
#
sub patchInTcsLogs($$) {

  my ($obs_dir, $utcname) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $tcs_logfile = $utcname."_bpsr.log";

  # check if the log file has previously been retrieved 
  if (-f $obs_dir."/".$utcname."/".$tcs_logfile) {
    Dada::logMsg(1, DL, "Skipping TCS log patch for ".$utcname);
    return ("ok", "previously patched");
  }

  # retrieve the TCS log file from jura
  Dada::logMsg(2, DL, "Getting TCS log file: /psr1/tcs/logs/".$tcs_logfile);
  $cmd = "scp -p pulsar\@pavo.atnf.csiro.au:/psr1/tcs/logs/$tcs_logfile $obs_dir/$utcname/";
  Dada::logMsg(2, DL, "patchInTcsLogs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, DL, "patchInTcsLogs: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "failed to get TCS log file: ".$response);
    return ("ok", "failed to get TCS log file");
  }

  Dada::logMsg(2, DL, "Merging TCS log file with obs.start files");
  $cmd = "merge_tcs_logs.csh $obs_dir/$utcname $tcs_logfile";
  Dada::logMsg(2, DL, "patchInTcsLogs: ".$cmd);
  ($result,$response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "patchInTcsLogs: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($error, "Could not merge the TCS log file, msg was: ".$response);
  }

  return ($result, $response);

}

#
# Copy the 13 psrxml files to the header_files directory
#
sub copyPsrxmlToDb($$) {

  my ($obs_dir, $utc_name) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $db_dump_dir = "/nfs/control/bpsr/header_files/";

  if (! -d $db_dump_dir) {
    mkdir $db_dump_dir;
  }

  my $psrxml_file = $utc_name.".psrxml";
  my $b_psrxml_file = "";
  my $b = "";
  my $day = "";
  my $submitted = 0;
  my $n_psrxml_files = 0;
  my $i = 0;

  # Check if this psrxml file has already been submitted
  $day = substr $utc_name, 0, 10;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $b = Bpsr::getBeamForPWCHost($cfg{"PWC_".$i});
    $b_psrxml_file = $utc_name."_".$b.".psrxml";

    Dada::logMsg(2, DL, "copyPsrxmlToDb: checking database to see if file exists [$day, $utc_name]");
    if (( -f $db_dump_dir."/submitted/".$day."/".$b_psrxml_file) || (-f $db_dump_dir."/failed/".$day."/".$b_psrxml_file)) {
      $submitted = 1;
    }
  }

  if ($submitted) {
    Dada::logMsg(1, DL, "Skipping psrxml copy for ".$utc_name);
    return ("ok", $utc_name." already been submitted");
  }

  Dada::logMsg(2, DL, "Copying ".$obs_dir."/".$utc_name."/??/*.psrxml to ".$db_dump_dir);

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $b = Bpsr::getBeamForPWCHost($cfg{"PWC_".$i});
    $b_psrxml_file = $utc_name."_".$b.".psrxml";

    # Copy <archives>/<utc>/<beam>/<utc>.psrxml to header_files/<utc>_<beam>.psrxml 
    if ( -f $obs_dir."/".$utc_name."/".$b."/".$psrxml_file) {

      $cmd = "cp ".$obs_dir."/".$utc_name."/".$b."/".$psrxml_file." ".$db_dump_dir."/".$b_psrxml_file;
      Dada::logMsg(2, DL, "copyPsrxmlToDb: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "copyPsrxmlToDb: ".$cmd." failed: ".$response);
      } else {
        $n_psrxml_files++;
      }

    } else {
      Dada::logMsg(2, DL, "could not find psrxml file: ".$obs_dir."/".$utc_name."/".$b."/".$psrxml_file);
    }
  }

  if ($n_psrxml_files != 13) {
    Dada::logMsg(1, DL, "copyPsrxmlToDb: ".$n_psrxml_files." of 13 existed");
  }

  return ("ok", $n_psrxml_files);
}

# Marks an observation as finished
sub markObsState($$$) 
{

  my ($o, $old, $new) = @_;

  Dada::logMsg(2, DL, "markObsState(".$o.", ".$old.", ".$new.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  my $state_change = $old." -> ".$new;
  my $old_file = "obs.".$old;
  my $new_file = "obs.".$new;
  my $file = "";
  my $ndel = 0;

  Dada::logMsg(1, DL, $o." ".$old." -> ".$new);

  $cmd = "touch ".$results_dir."/".$o."/".$new_file;
  Dada::logMsg(2, DL, "markObsState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, DL, "markObsState: ".$result." ".$response);

  $file = $results_dir."/".$o."/".$old_file;
  if ( -f $file ) {
    $ndel = unlink ($file);
    if ($ndel != 1) {
      Dada::logMsgWarn($warn, "markObsState: could not unlink ".$file);
    }
  } else {
    Dada::logMsgWarn($warn, "markObsState: expected file missing: ".$file);
  }

  # touch a beam.finished in all the client beam directories
  # $cmd = "find ".$archives_dir."/".$o." -mindepth 1 -maxdepth 1 -type l -name '??' -printf '\%f\n'";
  # Dada::logMsg(2, DL, "markObsState: ".$cmd);
  # ($result, $response) = Dada::mySystem($cmd);
  # Dada::logMsg(2, DL, "markObsState: ".$result." ".$response);
  # if (($result ne "ok") || ($response eq ""))
  # {
  #   Dada::logMsgWarn($warn, "markObsState: could get beam list");
  # } 
  # else
  # {
  #   my @beams = split(/\n/, $response);
  #   my $i = 0;
  #   my $b = "";
  #   for ($i=0; $i<=$#beams; $i++)
  #   {
  #     $b = $beams[$i]; 
  #     $cmd = "touch ".$archives_dir."/".$o."/".$b."/beam.finished";
  #     Dada::logMsg(2, DL, "markObsState: ".$cmd);
  #    ($result, $response) = Dada::mySystem($cmd);
  #     Dada::logMsg(2, DL, "markObsState: ".$result." ".$response);
  #     if ($result ne "ok") 
  #     {
  #       Dada::logMsgWarn($warn, "markObsState: could touch ".$o."/".$b."/beam.finished");
  #     }
  #   }
  # }

}

sub coincidencerThread()
{

  Dada::logMsg(1, DL, "coincidencerThread: starting");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "coincidencer -a ".$cfg{"SERVER_HOST"}." -p ".$cfg{"SERVER_COINCIDENCER_PORT"}.
         " -n ".$cfg{"NUM_PWC"}." -v | server_bpsr_server_logger.pl -n coin";
  Dada::logMsg(1, DL, "coincidencerThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(1, DL, "coincidencerThread: ".$result." ".$response);

  Dada::logMsg(1, DL, "coincidencerThread: exiting");
}

#
# Apply the fix fil headers script
#
sub fixFilHeaders($$) {

  my ($d, $utc_start) = @_;

  my $fil_edit = $cfg{"SCRIPTS_DIR"}."/fil_edit";
  my $psrxml_file = "";
  my $fil_file = "";
  my $cmd = "";
  my $i = 0;
  my $beam = "";
  my $result = "";
  my $response = "";
  my $fn_result = "ok";

  Dada::logMsg(2, DL, "Fixing TCS/Fil headers for ".$utc_start);

  #for ($i=1; $i<=13; $i++) {
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {

    #$beam = sprintf("%02d", $i);
    $beam = Bpsr::getBeamForPWCHost($cfg{"PWC_".$i});

    $cmd = "ls ".$d."/".$utc_start."/".$beam." >& /dev/null";
    Dada::logMsg(2, DL, "fixFilHeaders: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "fixFilHeaders: ".$result." ".$response);

    $fil_file    = $d."/".$utc_start."/".$beam."/".$utc_start.".fil";
    $psrxml_file = $d."/".$utc_start."/".$beam."/".$utc_start.".psrxml";

    if ((-f $fil_file) && (-f $psrxml_file)) {

      $cmd = "fix_fil_header.csh ".$psrxml_file." ".$fil_file." ".$fil_edit." ".$cfg{"NUM_PWC"};
      Dada::logMsg(2, DL, "fixFilHeaders: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, DL, "fixFilHeaders: ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "fixFilHeaders: fix_fil_header failed: ".$response);
        $fn_result = "fail";
        $response = "fix_fil_header failed: ".$response;
      }
    } else {
      Dada::logMsgWarn($warn, "fixFilHeaders: fil/psrxml file missing for ".$utc_start."/".$beam);
      $response = "fil/psrxml file missing";
    }
  }

  return ($fn_result, $response);
}

# Looks for observations that have been marked obs.deleted and moves them
# the OLD_RESULTS_DIR if deleted && > 2 weeks old
#
sub checkDeletedObs() 
{
  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";
  my @hosts = ("hipsr0", "hipsr1", "hipsr2", "hipsr3", "hipsr4", "hipsr5", "hipsr6", "hipsr7");
  my $o = "";
  my $host = "";
  my $user = "bpsr";
  my $j = 0;
  my $i = 0;
  my @observations = ();
  my $n_moved = 0;
  my $max_moved = 5;

  Dada::logMsg(2, DL, "checkDeletedObs()");

  # Find all observations marked as obs.deleted and > 14*24 hours since being modified 
  $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}." -mindepth 2 -maxdepth 2 -name 'obs.deleted' -mtime +14 -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, DL, "checkDeletedObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "checkDeletedObs: ".$result." ".$response);

  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "checkDeletedObs: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  @observations = split(/\n/,$response);
  Dada::logMsg(2, DL, "checkDeletedObs: found ".($#observations + 1)." marked obs.deleted +14 mtime");

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) 
  {

    $o = $observations[$i];

    # check that the source directory exists
    if (! -d $cfg{"SERVER_RESULTS_DIR"}."/".$o)
    {
      Dada::logMsgWarn($warn, "checkDeletedObs: ".$o." did not exist in results dir");
      next;
    }

    for ($j=0; $j<=$#hosts; $j++)
    {
      $host = $hosts[$j];

      # get a list of any beam dirs on this host
      $cmd = "find ".$cfg{"CLIENT_ARCHIVE_DIR"}." -mindepth 2 -maxdepth 2 -type d -name '".$o."' -printf '%h/%f '";
      Dada::logMsg(2, DL, "checkDeletedObs: ".$user."@".$host.": ".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(3, DL, "checkDeletedObs: ".$result." " .$rval." ".$response);
      if (($result ne "ok") || ($rval != 0))
      {
        Dada::logMsg(2, DL, "checkDeletedObs: could not find obs to delete on ".$host." for ".$o);
        next;
      }

      # now delete the directories returned by previous command
      $cmd = "rm -rf ".$response;
      Dada::logMsg(2, DL, "checkDeletedObs: ".$user."@".$host.": ".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand("bpsr", $host, $cmd);
      $result = "ok";
      $rval = 0;
      Dada::logMsg(3, DL, "checkDeletedObs: ".$result." " .$rval." ".$response);
      if (($result ne "ok") || ($rval != 0))
      {
        Dada::logMsg(0, DL, "checkDeletedObs: could not delete ".$o." on ".$host);
        next;
      }

    }

    # move the results dir
    $cmd = "mv ".$cfg{"SERVER_RESULTS_DIR"}."/".$o." ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/";
    Dada::logMsg(2, DL, "checkDeletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    $result = "ok";
    $rval = 0;
    Dada::logMsg(3, DL, "checkDeletedObs: ".$result." ".$response);
    if ($result ne "ok") 
    {
      Dada::logMsg(1, DL, "checkDeletedObs: ".$cmd." failed: ".$response);
      next;
    }

    Dada::logMsg(1, DL, $o.": deleted -> old");

    if ($n_moved > $max_moved)
    {
      return ("ok", "");
    }
    $n_moved++ ;

  }
  return ("ok", "");
}


