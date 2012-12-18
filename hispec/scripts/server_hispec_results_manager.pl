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
use Hispec;

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
our %cfg = Hispec::getConfig();
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

      # Dada::logMsg(2, DL, "main: checkDeletedObs()");
      # ($result, $response) = checkDeletedObs();
      # Dada::logMsg(2, DL, "main: checkDeletedObs() ".$result ." ".$response);

    } else {

      @subdirs = split(/\n/, $response);
      Dada::logMsg(2, DL, "main: found ".($#subdirs+1)." obs.processing files");

      my $h=0;
      my $age = 0;
      my $n_png_files = 0;

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
          ($age, $n_png_files) = getObsAge($dir); 
          Dada::logMsg(2, DL, "main: getObsAge() ".$age." ".$n_png_files);

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

  return ($age, $num_png_files);
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

    removeOldPngs($beamdir, "ac", "1024x768");
    removeOldPngs($beamdir, "ac", "400x300");
    removeOldPngs($beamdir, "ac", "112x84");

    removeOldPngs($beamdir, "cc", "1024x768");
    removeOldPngs($beamdir, "cc", "400x300");
    removeOldPngs($beamdir, "cc", "112x84");

    removeOldPngs($beamdir, "tp", "1024x768");
    removeOldPngs($beamdir, "tp", "400x300");
    removeOldPngs($beamdir, "tp", "112x84");
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
  my $user = "hispec";
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
      ($result, $rval, $response) = Dada::remoteSshCommand("hispec", $host, $cmd);
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


