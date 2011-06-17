#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use File::Basename;
use Time::Local;
use Dada;
use Caspsr;

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;

#
# Initialize global variables
#
%cfg = Caspsr::getConfig();
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$warn = ""; 
$error = ""; 
$quit_daemon = 0;


# Autoflush STDOUT
$| = 1;


# 
# Function Prototypes
#
sub main();
sub countObsStart($);
sub getArchives($);
sub getObsAge($$);
sub markObsState($$$);
sub checkClientsFinished($$);
sub processObservation($$);
sub processArchive($$$$);
sub makePlotsFromArchives($$$$$$);
sub copyLatestPlots($$$);


#
# Main
#
my $result = 0;
$result = main();

exit($result);


###############################################################################
#
# package functions
# 

sub main() {

  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $obs_results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  my $control_thread   = 0;
  my @observations = ();
  my $i = 0;
  my $o = "";
  my $t = "";
  my $n_obs_starts = 0;
  my $result = "";
  my $response = "";
  my $counter = 5;
  my $cmd = "";

  # sanity check on whether the module is good to go
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
  
  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # set the 
  umask 0027;
  my $umask_val = umask;

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  chdir $obs_results_dir;

  while (!$quit_daemon) {

    # TODO check that directories are correctly sorted by UTC_START time
    Dada::logMsg(2, $dl, "main: looking for obs.processing in ".$obs_results_dir);

    # Only get observations that are marked as procesing
    $cmd = "find ".$obs_results_dir." -maxdepth 2 -name 'obs.processing' ".
           "-printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";

    Dada::logMsg(3, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);

    if ($result eq "ok") {

      @observations = split(/\n/,$response);

      # For process all valid observations
      for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {

        $o = $observations[$i];

        Dada::logMsg(2, $dl, "main: checking obs ".$o." for archives");

        # Get the number of band subdirectories
        Dada::logMsg(3, $dl, "main: countObsStart(".$o.")");
        $n_obs_starts = countObsStart($o);
        Dada::logMsg(3, $dl, "main: countObsStart() ".$n_obs_starts);
        Dada::logMsg(2, $dl, "main: ".$o." had ".$n_obs_starts." obs.start files");

        # check how long ago the last result was received
        # negative values indicate that no result has ever been received
        # and is the age of the youngest file (but -ve)
        Dada::logMsg(3, $dl, "main: getObsAge(".$o.", ".$n_obs_starts.")");
        $t = getObsAge($o, $n_obs_starts);
        Dada::logMsg(3, $dl, "main: getObsAge() ".$t);
        Dada::logMsg(2, $dl, "main: ".$o." obs age=".$t);
  
        # If we have waited 600 seconds and all clients aren't finished
        if ($t > 600) {

          processObservation($o, $n_obs_starts);
          markObsState($o, "processing", "failed");

        # newest archive was more than 32 seconds old, finish the obs.
        } elsif ($t > 32) {

          Dada::logMsg(2, $dl, "main: checkClientsFinished(".$o.", ".$n_obs_starts.")");
          ($result, $response) = checkClientsFinished($o, $n_obs_starts);
          Dada::logMsg(2, $dl, "main: checkClientsFinished() ".$result." ".$response);

          if ($result eq "ok") {
            processObservation($o, $n_obs_starts);
            markObsState($o, "processing", "finished");
            cleanResultsDir($o);
          } 

        # we are still receiving results from this observation
        } elsif ($t >= 0) {

          processObservation($o, $n_obs_starts);

        # no archives yet received, wait
        } elsif ($t > -60) {

          # no archives have been received 60+ seconds after the
          # directories were created, something is wrong with this
          # observation, mark it as failed

        } else {
          Dada::logMsg(0, $dl, "main: processing->failed else case");
          markObsState($o, "processing", "failed");
        }

      }
    }
  
    # if no obs.processing, check again in 5 seconds
    if ($#observations == -1) {
      $counter = 5;
    } else {
      $counter = 2;
    }
    
    while ($counter && !$quit_daemon) {
      sleep(1);
      $counter--;
    }

  }

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
                                                                                
  return 0;
}



###############################################################################
#
# Returns the "age" of the observation. Return value is the age in seconds of 
# the file of type $ext in the obs dir $o. If no files exist in the dir, then
# return the age of the newest dir in negative seconds
# 
sub getObsAge($$) {

  my ($o, $num_obs_starts) = @_;
  Dada::logMsg(3, $dl, "getObsAge(".$o.", ".$num_obs_starts.")");

  my $result = "";
  my $response = "";
  my $age = 0;
  my $time = 0;
  my $cmd = "";

  # current time in "unix seconds"
  my $now = time;

  # If no obs.start files yet exist, we return the age of the obs dir
  if ($num_obs_starts== 0) {

    $cmd = "find ".$o." -maxdepth 0 -type d -printf '\%T@\\n'";
    Dada::logMsg(3, $dl, "getObsAge: ".$cmd); 
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
    $time = $response;
    $age = $time - $now;

  # We have some obs.start files, see if we have any archives
  } else {

    $cmd = "find ".$o." -type f -name '*.ar' -printf '\%T@\\n' -o -name '*.tot' -printf '\%T@\\n' | sort -n | tail -n 1";
    Dada::logMsg(3, $dl, "getObsAge: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
  
    # If we didn't found a file
    if ($response ne "") {
      # we will be returning a positive value
      $time = $response;
      $age = $now - $time;

    # No files were found, use the age of the obs.start files instead
    } else {

      # check the PROC_FILE 
      $cmd = "grep ^PROC_FILE `find ".$o." -type f -name '*_obs.start' | sort -n | tail -n 1` | awk '{print \$2}'";
      Dada::logMsg(3, $dl, "getObsAge: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);

      # If this is a single pulse 
      if (($result eq "ok") && ($response =~ m/singleF/)) {

        # only declare this as finished when the pwc.finished files are written
        $cmd = "find ".$o." -type f -name '*.pwc.finished' | wc -l";
        Dada::logMsg(3, $dl, "getObsAge: [pulse] ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "getObsAge: [pulse] ".$result." ".$response);

        # if all the pwc.finished files have been touched
        if (($result eq "ok") && ($num_obs_starts == $response)) {

          $cmd = "find ".$o." -type f -name 'pwc.*.finished' -printf '\%T@\\n' | sort -n | tail -n 1";
          Dada::logMsg(3, $dl, "getObsAge: [pulse] ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "getObsAge: [pulse] ".$result." ".$response);
      
          $time = $response;
          $age = $now - $time;

        # else, hardcode the time to -1, whilst we wait for the pwc.finished to be written
        } else {

          Dada::logMsg(2, $dl, "getObsAge: [pulse] waiting for *.pwc.finished");
          $age = -1;
        }

      # this is a normal observation
      } else {

        $cmd = "find ".$o." -type f -name '*_obs.start' -printf '\%T@\\n' | sort -n | tail -n 1";
        Dada::logMsg(3, $dl, "getObsAge: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
  
        # we will be returning a negative value
        $time = $response;
        $age = $time - $now;

      }
    }
  }

  Dada::logMsg(3, $dl, "getObsAge: time=".$time.", now=".$now.", age=".$age);
  return $age;
}

###############################################################################
#
# Marks an observation as finished
# 
sub markObsState($$$) {

  my ($o, $old, $new) = @_;

  Dada::logMsg(2, $dl, "markObsState(".$o.", ".$old.", ".$new.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $archives_dir = $cfg{"SERVER_ARCHIVE_DIR"};
  my $results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  my $state_change = $old." -> ".$new;
  my $old_file = "obs.".$old;
  my $new_file = "obs.".$new;
  my $file = "";
  my $ndel = 0;

  Dada::logMsg(1, $dl, $o." ".$old." -> ".$new);

  $cmd = "touch ".$results_dir."/".$o."/".$new_file;
  Dada::logMsg(2, $dl, "markObsState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "markObsState: ".$result." ".$response);

  $cmd = "touch ".$archives_dir."/".$o."/".$new_file;
  Dada::logMsg(2, $dl, "markObsState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "markObsState: ".$result." ".$response);

  $file = $results_dir."/".$o."/".$old_file;
  if ( -f $file ) {
    $ndel = unlink ($file);
    if ($ndel != 1) {
      Dada::logMsgWarn($warn, "markObsState: could not unlink ".$file);
    }
  } else {
    Dada::logMsgWarn($warn, "markObsState: expected file missing: ".$file);
  }

  $file = $archives_dir."/".$o."/".$old_file;
  if ( -f $file ) {
    $ndel = unlink ($file);
    if ($ndel != 1) {
      Dada::logMsgWarn($warn, "markObsState: could not unlink ".$file);
    }
  } else {
    Dada::logMsgWarn($warn, "markObsState: expected file missing: ".$file);
  }

}


###############################################################################
# 
# Clean up the results directory for the observation
#
sub cleanResultsDir($) {

  (my $o) = @_;

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$o;
  my $first_obs_start = "";
  my $source = "";
  my @sources = ();
  my $cmd = "";
  my $result = "";
  my $response = "";

  # delete the PWC subdirectories for each source
  $cmd = "rmdir ".$results_dir."/*/*/";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: rmdir ".$o."/*/*/ failed: ".$response);
    return ("fail", "Could not remove some PWC subdirectories");
  }

  # delete the SOURCE subdirectories for each source
  $cmd = "rmdir ".$results_dir."/*/";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: rmdir ".$o."/*/ failed: ".$response);
    return ("fail", "Could not remove some SOURCE  subdirectories");
  }

  # move the first [PWC]_obs.start file to obs.start
  $cmd = "find ".$results_dir." -mindepth 1 -maxdepth 1 -type f -name '*_obs.start' | sort -nr | tail -n 1";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove get first obs.start file");
  }
  $first_obs_start = $response;

  $cmd = "cp ".$first_obs_start." ".$results_dir."/obs.start";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove rename first obs.start file");
  }

  $cmd = "rm -f ".$results_dir."/*_obs.start";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove delete [PWC]_obs.start files");
  }

  $cmd = "rm -f ".$results_dir."/*_pwc.finished";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove delete [PWC]_pwc.finished files");
  }

  # get a list of the sources for this obs
  $cmd = "find ".$results_dir." -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' -printf '\%f\\n' | awk -F_ '{print \$1}'";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove get a list of the sources");
  }

  Dada::logMsg(2, $dl, "cleanResultsDir: deleting old pngs");
  @sources = split(/\n/, $response);
  foreach $source (@sources) {
    Dada::removeFiles($o, "phase_vs_flux_".$source."*_240x180.png", 0);
    Dada::removeFiles($o, "phase_vs_time_".$source."*_240x180.png", 0);
    Dada::removeFiles($o, "phase_vs_freq_".$source."*_240x180.png", 0);
    Dada::removeFiles($o, "bandpass_".$source."*_240x180.png", 0);
    Dada::removeFiles($o, "phase_vs_flux_".$source."*_200x150.png", 0);
    Dada::removeFiles($o, "phase_vs_time_".$source."*_200x150.png", 0);
    Dada::removeFiles($o, "phase_vs_freq_".$source."*_200x150.png", 0);
    Dada::removeFiles($o, "bandpass_".$source."*_200x150.png", 0);
    Dada::removeFiles($o, "phase_vs_flux_".$source."*_1024x768.png", 0);
    Dada::removeFiles($o, "phase_vs_time_".$source."*_1024x768.png", 0);
    Dada::removeFiles($o, "phase_vs_freq_".$source."*_1024x768.png", 0);
    Dada::removeFiles($o, "bandpass_".$source."*_1024x768.png", 0);
  }

}

###############################################################################
#
# Process all possible archives in the observation, combining the bands
# and plotting the most recent images. Accounts for multifold  PSRS
#
sub processObservation($$) {

  my ($o, $n_obs_starts) = @_;

  Dada::logMsg(2, $dl, "processObservation(".$o.", ".$n_obs_starts.")");

  my %unprocessed = ();
  my @unprocessed_keys = ();
  my $i = 0;
  my $k = "";
  my $fres_ar = "";
  my $tres_ar = "";
  my @fres_plot = ();
  my @tres_plot = ();
  my $source = "";
  my $archive = "";
  my $file = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $latest_n_sec_archive = "";

  # get a list of unprocessed archives, filenames
  %unprocessed = getArchives($o);

  # Sort the files into time order.
  @unprocessed_keys = sort (keys %unprocessed);

  Dada::logMsg(2, $dl, "processObservation: [".$o."] found ".($#unprocessed_keys+1)." unprocessed archives");

  # Process all the files, plot at the end
  for ($i=0;(($i<=$#unprocessed_keys) && (!$quit_daemon)); $i++) 
  {
    # unprocessed partial archive, of form SOURCE/ARCHIVE
    $k = $unprocessed_keys[$i];
    ($source, $archive) = split(/\//, $k);

    $file = $unprocessed{$k};

    Dada::logMsg(1, $dl, $o." Processing ".$source."/".$archive);

    # we can now add this archive to the total for this UTC_START/SOURCE
    Dada::logMsg(2, $dl, "processObservation: appendArchive(".$o.", ".$source.", ".$file.")");
    ($result, $fres_ar, $tres_ar) = appendArchive($o, $source, $file);
    Dada::logMsg(3, $dl, "processObservation: appendArchive() ".$result);

    # dont plot if nothing was produced
    if (($fres_ar eq "") || ($tres_ar eq "")) {
      Dada::logMsgWarn($warn, "No integrated archives produced from ".$file);

    # only add this source to the plot list once
    } elsif (!(grep $_ eq $fres_ar, @fres_plot)) {
      push @fres_plot, $fres_ar;
      push @tres_plot, $tres_ar;

    # skip as the fres/tres is already listed
    } else {
       Dada::logMsg(2, $dl, "Not adding ".$fres_ar." to plot list, duplicate");
    }
  }

  # If we produce at least one archive from appendArchive()
  for ($i=0; $i<=$#tres_plot; $i++) {

    # determine the source name for this archive
    $cmd = "vap -n -c name ".$tres_plot[$i]." | awk '{print \$2}'";
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "processObservation: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "processObservation: failed to determine source name: ".$response);
      $source = "UNKNOWN";
    } else {
      $source = $response;
      chomp $source;
    }
    $source =~ s/^[JB]//;

    # we want to plot the bandpass from just the most recent ARCHIVE_MOD second archive that has been copied
    # to the servers ARCHIVE dir.
    $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o."/".$source." -name '*.ar' | sort -n | tail -n 1";
    Dada::logMsg(2, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "processObservation: ".$result." ".$response);
    if ($result eq "ok") {
      $latest_n_sec_archive = $response;
    } else {
      $latest_n_sec_archive = "";
    }

    Dada::logMsg(2, $dl, "processObservation: plotting [".$i."] (".$o.", ".$source.", ".$fres_plot[$i].", ".$tres_plot[$i].")");
    makePlotsFromArchives($o, $source, $fres_plot[$i], $tres_plot[$i], "200x150", $latest_n_sec_archive);
    makePlotsFromArchives($o, $source, $fres_plot[$i], $tres_plot[$i], "1024x768", $latest_n_sec_archive);
  
    Dada::removeFiles($o, "phase_vs_flux_".$source."*_200x150.png", 30);
    Dada::removeFiles($o, "phase_vs_time_".$source."*_200x150.png", 30);
    Dada::removeFiles($o, "phase_vs_freq_".$source."*_200x150.png", 30);
    Dada::removeFiles($o, "bandpass_".$source."*_200x150.png", 30);
    Dada::removeFiles($o, "phase_vs_flux_".$source."*_1024x768.png", 30);
    Dada::removeFiles($o, "phase_vs_time_".$source."*_1024x768.png", 30);
    Dada::removeFiles($o, "phase_vs_freq_".$source."*_1024x768.png", 30);
    Dada::removeFiles($o, "bandpass_".$source."*_1024x768.png", 30);

    copyLatestPlots($o, $source, "200x150");

  }

}


###############################################################################
#
# Append the ARCHIVE_MOD second summed archive to the total for this observation
#
sub appendArchive($$$) {

  my ($utc_start, $source, $archive) = @_;

  Dada::logMsg(2, $dl, "appendArchive(".$utc_start.", ".$source.", ".$archive.")");

  my $total_t_sum = $archive;
  my $source_f_res = $utc_start."/".$source."_f.tot";
  my $source_t_res = $utc_start."/".$source."_t.tot";

  my $cmd = "";
  my $result = "";
  my $response = "";

  if (! -f $total_t_sum) {
    Dada::logMsg(0, $dl, "appendArchive: ".$cfg{"ARCHIVE_MOD"}." second summned archive [".$total_t_sum."] did not exist");
    return ("fail", "", "");
  } 

  # If the server's archive dir for this observation doesn't exist with the source
  if (! -d $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source) {
    $cmd = "mkdir -m 0750 -p ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
  } 
    
  # save this archive to the server's archive dir for permanent archival
  $cmd = "cp --preserve=all ./".$total_t_sum." ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source."/";
  Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  $cmd = "zap.psh -m ".$total_t_sum;
  Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  # If this is the first result for this observation
  if (!(-f $source_f_res)) {

    # "create" the source's fres archive
    $cmd = "cp ".$total_t_sum." ".$source_f_res;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Fscrunc the archive
    $cmd = "pam -F -m ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Now we have the tres archive
    $cmd = "cp ".$total_t_sum." ".$source_t_res;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
  
  # we are appending to the sources f and t res archives
  } else {

    # create the new source_f_res archive [we will rename later]
    my $temp_ar = $utc_start."/temp.ar";

    # Add the new archive to the FRES total [tsrunching it]
    $cmd = "psradd -T -o ".$temp_ar." ".$source_f_res." ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
    unlink($source_f_res);
    rename($temp_ar, $source_f_res);

    # Fscrunc the archive for adding to the TRES
    $cmd = "pam -F -m ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Add the Fscrunched archive to the TRES total 
    $cmd = "psradd -o ".$temp_ar." ".$source_t_res." ".$total_t_sum;
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    unlink($source_t_res);
    rename($temp_ar, $source_t_res);

  }

  # clean up the current archive
  unlink($total_t_sum);
  Dada::logMsg(2, $dl, "appendArchive: unlinking ".$total_t_sum);

  return ("ok", $source_f_res, $source_t_res);

}

###############################################################################
#
# Returns a hash of the archives in the results directory for the specified
# observation
#
sub getArchives($) {

  my ($utc_start) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $file = "";
  my @files = ();
  my %archives = ();
  my $source = "";
  my $host = "";
  my $archive = "";
  my $tag = "";
  
  $cmd = "find ".$utc_start." -mindepth 3 -name \"*.ar\" -printf \"\%P\\n\"";
  Dada::logMsg(3, $dl, "getArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getArchives: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "getArchives: ".$cmd." failed: ".$response);
    return %archives;
  }

  @files = sort split(/\n/,$response);

  foreach $file (@files) {

    Dada::logMsg(3, $dl, "getArchives: splitting ".$file);
    ($source, $host, $archive) = split(/\//, $file, 3);
    Dada::logMsg(3, $dl, "getArchives: ".$file." -> source=".$source." host=".$host.", archive=".$archive);
    $tag = $source."/".$archive;

    if (exists $archives{$tag}) {
      Dada::logMsgWarn($warn, "getArchives: ignoring duplicate tag for ".$file);
    } else {
      $archives{$tag} = $utc_start."/".$file;
      Dada::logMsg(2, $dl, "getArchives: adding tag ".$tag);
    }
  }

  return %archives;
}


################################################################################
#
# Look for *obs.start where the * prefix is the PWC name
# 
sub countObsStart($) {

  my ($dir) = @_;

  my $cmd = "find ".$dir." -name \"*_obs.start\" | wc -l";
  my $find_result = `$cmd`;
  chomp($find_result);
  return $find_result;

}


###############################################################################
#
# Create plots for use in the web interface
#
sub makePlotsFromArchives($$$$$$) {

  my ($dir, $source, $total_f_res, $total_t_res, $res, $ten_sec_archive) = @_;

  my $web_style_txt = $cfg{"SCRIPTS_DIR"}."/web_style.txt";
  my $args = "-g ".$res;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $bscrunch = "";
  my $bscrunch_t = "";

  # If we are plotting hi-res - include
  if ($res ne "1024x768") {
    $args .= " -s ".$web_style_txt." -c below:l=unset";
    $bscrunch = " -j 'B 256, F 256'";
    $bscrunch_t = " -j 'B 256'";
  } else {
    $bscrunch = " -j 'B 512'";
    $bscrunch_t = " -j 'B 512'";
  }

  my $bin = Dada::getCurrentBinaryVersion()."/psrplot ".$args;
  my $timestamp = Dada::getCurrentDadaTime(0);

  my $pvt  = "phase_vs_time_".$source."_".$timestamp."_".$res.".png";
  my $pvfr = "phase_vs_freq_".$source."_".$timestamp."_".$res.".png";
  my $pvfl = "phase_vs_flux_".$source."_".$timestamp."_".$res.".png";
  my $bp   = "bandpass_".$source."_".$timestamp."_".$res.".png";

  # Combine the archives from the machine into the archive to be processed
  # PHASE vs TIME
  $cmd = $bin.$bscrunch_t." -jpC -p time -jFD -D ".$dir."/pvt_tmp/png ".$total_t_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # PHASE vs FREQ
  $cmd = $bin.$bscrunch." -jpC -p freq -jTD -D ".$dir."/pvfr_tmp/png ".$total_f_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # PHASE vs TOTAL INTENSITY
  $cmd = $bin.$bscrunch." -jpC -p flux -jTF -D ".$dir."/pvfl_tmp/png ".$total_f_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # BANDPASS
  # 2011-02-07 WvS added log scale (mostly for 50cm)
  if ($res eq "1024x768") {
    #$cmd = $bin." -pb -x -lpol=0,1 -c log=1 -N2,1 -c above:c= -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
    $cmd = $bin." -J '/home/dada/linux_64/bin/zap.psh' -pb -x -lpol=0,1 -N2,1 -c above:c= -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
  } else {
    #$cmd = $bin." -pb -x -lpol=0,1 -c log=1 -O -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
    $cmd = $bin." -J '/home/dada/linux_64/bin/zap.psh' -pb -x -lpol=0,1 -O -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
  }
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);


  # wait for each file to "appear"
  my $waitMax = 5;
  while ($waitMax) {
    if ( (-f $dir."/pvfl_tmp") &&
         (-f $dir."/pvt_tmp") &&
         (-f $dir."/pvfr_tmp") &&
         (-f $dir."/bp_tmp") )
    {
      $waitMax = 0;
    } else {
      $waitMax--;
      sleep(1);
    }
  }

  # rename the plot files to their correct names
  system("mv -f ".$dir."/pvt_tmp ".$dir."/".$pvt);
  system("mv -f ".$dir."/pvfr_tmp ".$dir."/".$pvfr);
  system("mv -f ".$dir."/pvfl_tmp ".$dir."/".$pvfl);
  system("mv -f ".$dir."/bp_tmp ".$dir."/".$bp);

}


###############################################################################
#
# Checks that all the *.pwc.finished files exist in the servers results dir. 
# This ensures that the obs is actually finished, for cases such as single 
# pulse
#
sub checkClientsFinished($$) {

  my ($obs, $n_obs_starts) = @_;

  Dada::logMsg(2, $dl ,"checkClientsFinished(".$obs.", ".$n_obs_starts.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $dir = $cfg{"SERVER_RESULTS_DIR"};

  # only declare this as finished when the band.finished files are written
  $cmd = "find ".$dir."/".$obs." -type f -name '*_pwc.finished' | wc -l";
  Dada::logMsg(2, $dl, "checkClientsFinished: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "checkClientsFinished: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkClientsFinished: ".$cmd." failed: ".$response);
  } else {
    if ($response eq $n_obs_starts) {
      $result = "ok";
      $response = "";
    } else {
      $result = "fail";
      $response = "not yet finished";
    }
  }

  Dada::logMsg(2, $dl ,"checkClientsFinished() ".$result." ".$response);
  return ($result, $response); 

}

###############################################################################
#
# copy's the latest plot for the specified observation and source to the
# "latest" directory
#
sub copyLatestPlots($$$)
{
  my ($o, $s, $dim) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "ls -1 ".$o."/phase_vs_flux_".$s."_*_".$dim.".png | sort | tail -n 1";
  ($result, $response) = Dada::mySystem($cmd);
  if ($result eq "ok") {
    if ( -f $response) {
      $cmd = "cp ".$response." ".$cfg{"WEB_DIR"}."/caspsr/latest/".
             "caspsr_flux_vs_phase.png";
      ($result, $response) = Dada::mySystem($cmd);
    }
  }

  $cmd = "ls -1 ".$o."/phase_vs_freq_".$s."_*_".$dim.".png | sort | tail -n 1";
  ($result, $response) = Dada::mySystem($cmd);
  if ($result eq "ok") {
    if ( -f $response) {
      $cmd = "cp ".$response." ".$cfg{"WEB_DIR"}."/caspsr/latest/".
             "caspsr_freq_vs_phase.png";
      ($result, $response) = Dada::mySystem($cmd);
    }
  }

  $cmd = "ls -1 ".$o."/phase_vs_time_".$s."_*_".$dim.".png | sort | tail -n 1";
  ($result, $response) = Dada::mySystem($cmd);
  if ($result eq "ok") {
    if ( -f $response) {
      $cmd = "cp ".$response." ".$cfg{"WEB_DIR"}."/caspsr/latest/".
             "caspsr_time_vs_phase.png";
      ($result, $response) = Dada::mySystem($cmd);
    }
  }

  $cmd = "ls -1 ".$o."/bandpass_".$s."_*_".$dim.".png | sort | tail -n 1";
  ($result, $response) = Dada::mySystem($cmd);
  if ($result eq "ok") {
    if ( -f $response) {
      $cmd = "cp ".$response." ".$cfg{"WEB_DIR"}."/caspsr/latest/".
             "caspsr_bandpass.png";
      ($result, $response) = Dada::mySystem($cmd);
    }
  }


  $cmd = "cp ".$o."/obs.info ".$cfg{"WEB_DIR"}."/caspsr/latest/";
  ($result, $response) = Dada::mySystem($cmd);

}


###############################################################################
#
# Handle quit requests asynchronously
#
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
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, $dl ,"controlThread: exiting");

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
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
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}


END { }

1;  # return value from file
