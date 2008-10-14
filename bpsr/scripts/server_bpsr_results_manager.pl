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
# Constants
#
use constant DEBUG_LEVEL         => 1;
use constant IMAGE_TYPE          => ".png";
use constant PIDFILE             => "bpsr_results_manager.pid";
use constant LOGFILE             => "bpsr_results_manager.log";


#
# Global Variable Declarations
#
our %cfg = Bpsr->getBpsrConfig();
our $quit_daemon : shared = 0;


#
# Signal Handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


#
# Local Variable Declarations
#

my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

my $bindir              = Dada->getCurrentBinaryVersion();
my $obs_results_dir     = $cfg{"SERVER_RESULTS_DIR"};
my $obs_archive_dir     = $cfg{"SERVER_ARCHIVE_DIR"};
my $daemon_control_thread = 0;

my $cmd;
my $timestamp = "";
my $fname = "";

#$cmd = "rm -f *.gif";
#system($cmd);

# This will have to be determined */
my $have_new_archive = 1;
my $node;
my $nodedir;

my %unprocessed = ();
my $key;
my $value;
my $num_results = 0;
my $current_key = 0;

my $fres = "";
my $tres = "";
my $current_archive = "";
my $last_archive = "";
my $obs_dir = "";

my $dir;
my @subdirs;

my @beamdirs;
my $beamdir;

my @keys;
my %processed;
my $i;
my $j;

# Autoflush output
$| = 1;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}


# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

debugMessage(0, "STARTING SCRIPT: ".Dada->getCurrentDadaTime(0));

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

chdir $obs_results_dir;


#
# Main Loop
#
%processed = ();

while (!$quit_daemon) {

  $dir = "";
  @subdirs = ();

  # TODO check that directories are correctly sorted by UTC_START time
  debugMessage(2,"Main While Loop, looking for data in ".$obs_results_dir);

  opendir(DIR,$obs_results_dir);
  @subdirs = sort grep { !/^\./ && -d $obs_results_dir."/".$_ } readdir(DIR);
  closedir DIR;

  my $h=0;

  # For each observation
  for ($h=0; (($h<=$#subdirs) && (!$quit_daemon)); $h++) {

    @beamdirs = ();

    $dir = $obs_results_dir."/".$subdirs[$h];
    $dir = $subdirs[$h];

    # If this observation has not been finalized 
    if (! -f $dir."/obs.finalized") { 

      my $most_recent_result = getMostRecentResult($dir);
      debugMessage(2, $dir." recent result = ".$most_recent_result);

      # If the data directory is more than 5 minutes old, but no
      # data files have been produced, we consider this observation
      # erroneous and delete the output directory
      if ($most_recent_result == -1) {

        # Sanity check for archives too
        #chdir $obs_archive_dir;
        #$most_recent_result = getMostRecentResult($dir);
        #chdir $obs_results_dir;

        if ($most_recent_result == -1) {
          # deleteObservation($obs_results_dir."/".$dir);
          #debugMessage(1, "Deleted empty observation: $dir");
        }

      # If the most recent result is more han 5 minutes old, consider
      # this observation finalized
      } elsif ($most_recent_result > 5*60) {

        debugMessage(1, "Finalised observation: ".$dir);
        system("touch ".$obs_results_dir."/".$dir."/obs.finalized");
        # TODO remove all extra image files

      # Else this is an active observation, try to process the .pol
      # files that may exist in each beam
      } else {

        # Get the list of beams
        opendir(SUBDIR, $dir);
        @beamdirs = sort grep { !/^\./ && -d $dir."/".$_ } readdir(SUBDIR);
        closedir SUBDIR;

        # Foreach beam dir, check for unprocessed files
        for ($i=0; (($i<=$#beamdirs) && (!$quit_daemon)); $i++) {

          $beamdir = $dir."/".$beamdirs[$i];
          debugMessage(3, "  ".$beamdir);

          # Gets files for which both .pol0 and .pol1 exist
          %unprocessed = getUnprocessedFiles($beamdir);

          @keys = sort (keys %unprocessed);

          for ($j=0; $j<=$#keys; $j++) {
            debugMessage(2, "file = ".$keys[$j]);
            processResult($beamdir, $keys[$j]);
          }
        }
      }
    }
  }

  # If we have been asked to exit, dont sleep
  if (!$quit_daemon) {
    sleep(2);
  }

}

# Rejoin our daemon control thread
$daemon_control_thread->join();
                                                                                
debugMessage(0, "STOPPING SCRIPT: ".Dada->getCurrentDadaTime(0));
                                                                                


exit(0);

###############################################################################
#
# Functions
#


#
# For the given utc_start ($dir), and archive (file) add the archive to the 
# summed archive for the observation
#
sub processResult($$) {

  my ($dir, $file) = @_;

  debugMessage(2, "processResult(".$dir.", ".$file.")");

  chdir $dir;
  my $filetype = "";
  my $filebase = "";

  my $bindir =      Dada->getCurrentBinaryVersion();
  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};

  # Delete any old images in this directory
  my $response;

  if ($file =~ m/bp$/) {
    $filetype = "bandpass";
    $filebase = substr $file, 0, -3;

  } elsif ($file =~ m/bps$/) {
    $filetype = "bandpass_rms";
    $filebase = substr $file, 0, -4;

  } elsif ($file =~ m/ts$/) {
    $filetype = "timeseries";
    $filebase = substr $file, 0, -3;

  } else {
    $filetype = "unknown";
    $filebase = "";
  }
    
  # .bp? -> meanbandpass pol ?
  # .ts? -> time series pol ?

  if ($filetype eq "unknown") {
    # skip the file

  } else {
   
    # Create the low resolution file
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 112x84 -nobox -nolabel -g /png ";
    debugMessage(2, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      debugMessage(0, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv .N=?".$file.".png ".$file."_112x84.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) { 
        chomp $response;
        debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response); 
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_112x84.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

    # Create the mid resolution file
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 400x300 -g /png";
    debugMessage(2, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      debugMessage(0, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv .N=?".$file.".png ".$file."_400x300.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) { 
        chomp $response;
        debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response); 
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_400x300.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

    # Create the high resolution file
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 1024x768 -g /png";
    debugMessage(2, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      debugMessage(0, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv .N=?".$file.".png ".$file."_1024x768.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) {
        chomp $response;
        debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response);
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_1024x768.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          debugMessage(0, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

  }
  # Delete the data file
  unlink($file."0");
  unlink($file."1");

  debugMessage(2, "unlinking ".$file);

  chdir "../../";

  return 0;

}

#
# Counts the numbers of data files received
#
sub getUnprocessedFiles($) {

  my ($dir) = @_;

  debugMessage(3, "chdir $dir");
  chdir $dir;

  my $cmd = "find . -regex \".*[ts|bp|bps][0|1]\" -printf \"%P\n\"";
  debugMessage(3, "find . -regex \".*[ts|bp|bps][0|1]\" -printf \"\%P\"");
  my $find_result = `$cmd`;

  my %archives = ();

  my @files = split(/\n/,$find_result);
  my $file = "";

  debugMessage(2, "$dir: ");

  # Add the results to the hash
  foreach $file (@files) {
    debugMessage(2, "  $file");
    # strip suffix
    my $basename = substr $file, 0, -1;
    if (! exists ($archives{$basename})) {
      $archives{$basename} = 1;
    } else {
      $archives{$basename} += 1;
    }
  }

  # Strip basenames with only 1 polaristion
  foreach $key (keys (%archives)) {
    if ($archives{$key} == 1) {
      delete($archives{$key});
    }
  }

  chdir "../../";

  return %archives;

}


sub countObsStart($) {

  my ($dir) = @_;

  my $cmd = "find ".$dir." -name \"obs.start\" | wc -l";
  my $find_result = `$cmd`;
  chomp($find_result);
  return $find_result;

}

sub deleteArchives($$) {

  (my $dir, my $archive) = @_;

  my $cmd = "rm -f ".$dir."/*/".$archive;
  debugMessage(2, "Deleting processed archives ".$cmd);
  my $response = `$cmd`;
  if ($? != 0) {
    debugMessage(0, "rm failed: \"".$response."\"");
  }

  return 0;

}


sub debugMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    print "[".$time."] ".$message."\n";
  }
}


#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada->getCurrentDadaTime(0)."\n";
  exit(1);

}
                                                                                
sub daemonControlThread() {

  debugMessage(2, "Daemon control thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  debugMessage(2, "Unlinking PID file: ".$pidfile);
  unlink($pidfile);

  debugMessage(2, "Daemon control thread ending");

}

sub getMostRecentResult($) {

  my ($dir) = @_;

  my $age = -1;

  # Check if any directories exist yet...
  my $cmd = "find ".$dir."/* -type d | wc -l";
  my $num_dirs = `$cmd`;
  chomp($num_dirs);
  debugMessage(2, "getMostRecentResult: num beam dirs = ".$num_dirs);
  my $current_unix_time = time;

  if ($num_dirs > 0) {

    $cmd  = "find ".$dir."/*/ -regex '.*[bp|ts|bps][0|1]' -printf \"%T@\\n\" | sort | tail -n 1";

    my $unix_time_of_most_recent_result = `$cmd`;
    chomp($unix_time_of_most_recent_result);

    if ($unix_time_of_most_recent_result) {

      $age = $current_unix_time - $unix_time_of_most_recent_result;
      debugMessage(2, "getMostRecentResult: most recent mon file was ".$age." seconds old");

    } else {

      debugMessage(2, "getMostRecentResult: no mon files found in $dir");

      # Check the age of the directories - if > 5 minutes then likely a dud obs.

      $cmd = "find ".$dir."/* -type d -printf \"%T@\\n\" | sort | tail -n 1";
      $unix_time_of_most_recent_result = `$cmd`;
      chomp($unix_time_of_most_recent_result);

      if ($unix_time_of_most_recent_result) {

        $age = $current_unix_time - $unix_time_of_most_recent_result;
        debugMessage(2, "Dir $dir is ".$age." seconds old");

        # If the obs.start is more than 5 minutes old, but we have no
        # archives, then this observation is considered a dud
        if ($age > 5*60) {
          $age = -1;
        }

      } else {
        $age = -1;
      }
    }
  } else {

    $cmd = "find ".$dir." -type d -printf \"%T@\\n\"";
    my $unix_time_of_dir = `$cmd`;

    if ($unix_time_of_dir) {
      $age = $current_unix_time - $unix_time_of_dir;
      debugMessage(2, "$dir age = ".$age);
      if ($age > 5*60) {
        $age = -1;
      } else {
        $age = 0;
      }
    } else {
      $age = 0;
    }
  }

  debugMessage(2, "Observation: ".$dir.", age: ".$age." seconds");

  return $age;

}

sub deleteObservation($) {

  (my $dir) = @_;
  debugMessage(1, "Deleting observation: ".$dir);
  $cmd = "rm -rf $dir";
  `$cmd`;
  return $?;

}



