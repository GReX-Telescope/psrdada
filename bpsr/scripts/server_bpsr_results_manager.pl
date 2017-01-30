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
use MIME::Lite;
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
our %roach = Bpsr::getROACHConfig();
our $quit_daemon : shared = 0;
our $daemon_name : shared = Dada::daemonBaseName($0);
our $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
our $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
our $last_frb = "";
our %frb_actions = ();

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
# only beam 2 - 8189
#

# default FRB detection options
$frb_actions{"default"} = { "snr_cut" => 10.0, "filter_cut" => 8.0, "egal_dm" => "true", 
                            "excise_psr" => "true", "cps_cut" => "5", "cc_email" => "" };

# custom (per project) FRB detection options
$frb_actions{"P999"}    = { "snr_cut" => 8.0, "filter_cut" => 10.0, "excise_psr" => "false", "egal_dm" => "false" };
                           # "cc_email" => "manishacaleb\@gmail.com, cflynn\@swin.edu.au, ebpetroff\@gmail.com, evan.keane\@gmail.com" };

$frb_actions{"P864"}    = { "egal_dm" => "false", "excise_psr" => "false", 
                            "cps_cut" => "50", "filter_cut" => "9",
                            "cc_email" => "manishacaleb\@gmail.com, cflynn\@swin.edu.au" };

$frb_actions{"P858"}    = { "cc_email" => "superb\@lists.pulsarastronomy.net" };
$frb_actions{"P892"}    = { "cc_email" => "superb\@lists.pulsarastronomy.net" };
$frb_actions{"PX025"}   = { "cc_email" => "superb\@lists.pulsarastronomy.net" };

$frb_actions{"P871"}    = { "cc_email" => "ebpetroff\@gmail.com, cherrywyng\@gmail.com, cmlflynn\@gmail.com, davidjohnchampion\@gmail.com, evan.keane\@gmail.com, ewan.d.barr\@gmail.com, manishacaleb\@gmail.com, matthew.bailes\@gmail.com, michael\@mpifr-bonn.mpg.de, apossenti\@gmail.com, sarahbspolaor\@gmail.com, Simon.Johnston\@atnf.csiro.au, vanstraten.willem\@gmail.com, Ben.Stappers\@manchester.ac.uk" };

$frb_actions{"P789"}    = { "cc_email" => "ewan.d.barr\@gmail.com, Simon.Johnston\@atnf.csiro.au, evan.keane\@gmail.com" };

$frb_actions{"P879"}    = { "cc_email" => "cmlflynn\@gmail.com, Ramesh.Bhat\@curtin.edu.au, michael\@mpifr-bonn.mpg.de, davidjohnchampion\@gmail.com, sarahbspolaor\@gmail.com" };

#$frb_actions{"P888"}    = { "egal_dm" => "false", "excise_psr" => "false", "snr_cut" => 8.0,
#                            "cps_cut" => "50", "filter_cut" => "9", "beam_mask" => "1", 
#                            "cc_email" => "epetroff\@swin.edu.au" };


#
# Main Loop
#
{
  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $obs_results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $control_thread = 0;

  my $cmd;
  my $dir;
  my @subdirs;
  my $curr_time;
  my $now;

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

  my $curr_processing = "";

  while (!$quit_daemon)
  {
    $curr_time = time;

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
        #if (!( (-f $dir."/obs.archived") || (-f $dir."/obs.finished") || (-f $dir."/obs.deleted") ||
        #       (-f $dir."/obs.transferred") ) ) {
        # if (!(-f $dir."/obs.deleted"))
        {

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
              #if ($dir eq $curr_processing) {
              #  Dada::logMsg(2, DL, "Received ".$n_cand_files." cand files for ".$dir);
              #} else {
              #  $curr_processing = $dir;
              #  Dada::logMsg(1, DL, "Receiving cand files for ".$dir);
              #}

              #($result, $response) = processCandidates($dir);

              chdir $cfg{"SERVER_RESULTS_DIR"};

              removeOldPngs($dir, "cands", "1024x768");
              removeOldPngs($dir, "dm_vs_time", "700x240");
            }
          }
        }
      } 
    }

    $now = time;

    # If we have been asked to exit, dont sleep
    while ((!$quit_daemon) && ($now < $curr_time + 5))
    {
      sleep(1);
      $now = time;
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  Dada::logMsg(0, DL, "STOPPING SCRIPT");

  exit(0);
}

###############################################################################
#
# Functions
#


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
  my @hosts = ("hipsr0", "hipsr1", "hipsr2", "hipsr3", "hipsr4", "hipsr5", "hipsr6", "hipsr7");
  my ($cmd, $result, $response, $rval);
  my ($o, $host, $i, $j, $pid, $nbeams, $swin_to_swin, $on_tape_swin);
  my ($obs_utc_time, $curr_utc_time);
  my $month_in_secs = 30 * 24 * 60 * 60;
  my $user = "bpsr";
  my @observations = ();
  my $n_moved = 0;
  my $max_moved = 5;
  Dada::logMsg(2, DL, "checkDeletedObs()");

  # Find all observations marked as obs.deleted and > 1 hour since being modified
  $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}." -mindepth 2 -maxdepth 2 -name 'obs.deleted' -mmin +60 -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
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

    $obs_utc_time  = Dada::getUnixTimeUTC($o);
    $curr_utc_time = Dada::getUnixTimeUTC(Dada::printTime(time, "utc"));

    if (($obs_utc_time + $month_in_secs) > $curr_utc_time)
    {
      Dada::logMsg(2, DL, "checkDeletedObs: obs is less than 1 month old: ".$o);
      next;
    }

    # check that the source directory exists
    if (! -d $cfg{"SERVER_RESULTS_DIR"}."/".$o)
    {
      Dada::logMsgWarn($warn, "checkDeletedObs: ".$o." did not exist in results dir");
      next;
    }

    # If P630 data, check tape flags exist
    $cmd = "grep ^PID ".$cfg{"SERVER_RESULTS_DIR"}."/".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(2, DL, "checkDeletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "checkDeletedObs: ".$result." ".$response);
    if (($result ne "ok") || ($response eq ""))
    {
      Dada::logMsg(2, DL, "checkDeletedObs: could not determine PID for ".$o);
      next;
    }
    $pid = $response;

    # count the number of beams
    $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$o." -mindepth 1 -maxdepth 1 -type d -name '??' | wc -l";
    Dada::logMsg(2, DL, "checkDeletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "checkDeletedObs: ".$result." ".$response);
    if (($result ne "ok") || ($response eq ""))
    {
      Dada::logMsg(2, DL, "checkDeletedObs: could not counter number of beams for ".$o);
      next;
    }
    $nbeams = $response;

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


