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
Dada->preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant DL           => 1;
use constant IMAGE_TYPE   => ".png";
use constant PIDFILE      => "bpsr_results_manager.pid";
use constant LOGFILE      => "bpsr_results_manager.log";
use constant QUITFILE     => "bpsr_results_manager.quit";


#
# Global Variable Declarations
#
our %cfg = Bpsr->getBpsrConfig();
our $quit_daemon : shared = 0;
our $error = $cfg{"STATUS_DIR"}."/bpsr_results_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/bpsr_results_manager.warn";


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

Dada->logMsg(0, DL, "STARTING SCRIPT: ".Dada->getCurrentDadaTime(0));

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
  Dada->logMsg(2, DL, "Main While Loop, looking for data in ".$obs_results_dir);

  opendir(DIR,$obs_results_dir);
  @subdirs = sort grep { !/^\./ && !/^stats/ && -d $obs_results_dir."/".$_ } readdir(DIR);
  closedir DIR;

  my $h=0;

  my $age = 0;
  my $nfiles = 0;
  my $adir = "";

  # For each observation
  for ($h=0; (($h<=$#subdirs) && (!$quit_daemon)); $h++) {

    @beamdirs = ();

    $dir = $obs_results_dir."/".$subdirs[$h];
    $dir = $subdirs[$h];
    $adir = $obs_archive_dir."/".$subdirs[$h];

    # If this observation has not been inished, archived or transferred
    if (!( (-f $adir."/obs.archived") || (-f $adir."/obs.finished") || (-f $adir."/obs.deleted") ||
           (-f $adir."/obs.transferred") || (!(-d $adir)) )) {

      # determine the age of the observation and number of files that can be processed in it
      Dada->logMsg(2, DL, "main: getObsInfo(".$dir.")");
      ($age, $nfiles) = getObsInfo($dir); 
      Dada->logMsg(2, DL, "main: getObsInfo() ".$age." ".$nfiles);

      # If obs_age is -1 then this is a dud observation, delete it
      if ($age == -1) {

        Dada->logMsg(1, DL, "WVS FIX: no longer deleting empty observation: $dir");
	### YIKES! plot4mon is failing and directories appear empty!
        ### deleteObservation($obs_results_dir."/".$dir);
        ### deleteObservation($obs_archive_dir."/".$dir);
        # Clear obs.processing
        if (-f $obs_results_dir."/".$dir."/obs.processing") {
          unlink $obs_results_dir."/".$dir."/obs.processing";
        }

      # If the obs age is over five minutes, then we consider it finished
      } elsif ($age > 60) {

        Dada->logMsg(1, DL, "Finalising observation: ".$dir);

        # we need to patch the TCS logs into the file!
        patchInTcsLogs($obs_archive_dir,$dir);

        # remove any old files in the archive dir
        cleanUpObs($obs_results_dir."/".$dir);

        # copy psrxml files to special dir for mike to deal with
        copyPsrxmlToDb($obs_archive_dir, $dir);

        # Clear obs.processing
        if (-f $obs_results_dir."/".$dir."/obs.processing") {
          unlink $obs_results_dir."/".$dir."/obs.processing";
        }

        # mark the observation as finalised
        system("touch ".$obs_results_dir."/".$dir."/obs.finished");
        system("touch ".$obs_archive_dir."/".$dir."/obs.finished");

      # Else this is an active observation, try to process the .pol
      # files that may exist in each beam
      } else {

        if ($nfiles > 0) {

          Dada->logMsg(1, DL, "Processing ".$nfiles." mon files for ".$dir);

          # Mark this as processing
          system("touch ".$obs_results_dir."/".$dir."/obs.processing");

          # Get the list of beams
          opendir(SUBDIR, $dir);
          @beamdirs = sort grep { !/^\./ && -d $dir."/".$_ } readdir(SUBDIR);
          closedir SUBDIR;

          # Foreach beam dir, check for unprocessed files
          for ($i=0; (($i<=$#beamdirs) && (!$quit_daemon)); $i++) {

            $beamdir = $dir."/".$beamdirs[$i];
            Dada->logMsg(3, DL, "  ".$beamdir);

            # search for unprocessed files in the beam directory 
            %unprocessed = getUnprocessedFiles($beamdir);

            @keys = sort (keys %unprocessed);

            for ($j=0; $j<=$#keys; $j++) {
              Dada->logMsg(2, DL, "main: processResult(".$beamdir.", ".$keys[$j]);
              processResult($beamdir, $keys[$j]);
            }
          }

          # Now delete all old png files
          removeAllOldPngs($dir);
        }
      }
      
    } 
  }

  # If we have been asked to exit, dont sleep
  if (!$quit_daemon) {
    sleep(5);
  }

}

# Rejoin our daemon control thread
$daemon_control_thread->join();
                                                                                
Dada->logMsg(0, DL, "STOPPING SCRIPT: ".Dada->getCurrentDadaTime(0));
                                                                                


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

  Dada->logMsg(2, DL, "processResult(".$dir.", ".$file.")");

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
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 112x84 -nobox -nolabel -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -log -mmm";
    }

    Dada->logMsg(2, DL, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      Dada->logMsg(2, DL, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv ".$file.".png ".$file."_112x84.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) { 
        chomp $response;
        Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response); 
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_112x84.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

    # Create the mid resolution file
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 400x300 -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -log -mmm";
    }

    Dada->logMsg(2, DL, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      Dada->logMsg(2, DL, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv ".$file.".png ".$file."_400x300.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) { 
        chomp $response;
        Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response); 
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_400x300.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

    # Create the high resolution file
    $cmd = $bindir."/plot4mon ".$file."0 ".$file."1 -G 1024x768 -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -log -mmm";
    }

    Dada->logMsg(2, DL, "Ploting with \"".$cmd."\"");
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      Dada->logMsg(2, DL, "Plotting cmd \"".$cmd."\" failed with message \"".$response."\"");
    } else {
      $cmd  = "mv ".$file.".png ".$file."_1024x768.png";
      $response = `$cmd 2>&1`;
      if ($? != 0) {
        chomp $response;
        Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response);
      }
      if ($filetype eq "timeseries") {
        $cmd = "mv ".$filebase.".fft.png ".$filebase.".fft_1024x768.png";
        $response = `$cmd 2>&1`;
        if ($? != 0) {
          chomp $response;
          Dada->logMsg(2, DL, "Plot file rename \"".$cmd."\" failed: ".$response);
        }
      }
    }

  }
  # Delete the data file
  Dada->logMsg(3, DL, "unlinking mon files ".$file."0 and ".$file."1");

  unlink($file."0");
  unlink($file."1");

  chdir "../../";

  return 0;

}

#
# Counts the numbers of data files received
#
sub getUnprocessedFiles($) {

  my ($dir) = @_;

  Dada->logMsg(3, DL, "chdir $dir");
  chdir $dir;

  my $cmd = "find . -regex \".*[ts|bp|bps][0|1]\" -printf \"%P\n\"";
  Dada->logMsg(3, DL, "find . -regex \".*[ts|bp|bps][0|1]\" -printf \"\%P\"");
  my $find_result = `$cmd`;

  my %archives = ();

  my @files = split(/\n/,$find_result);
  my $file = "";

  Dada->logMsg(3, DL, "$dir: ");

  # Add the results to the hash
  foreach $file (@files) {
    Dada->logMsg(3, DL, "  $file");
    # strip suffix
    my $basename = substr $file, 0, -1;
    if (! exists ($archives{$basename})) {
      $archives{$basename} = 1;
    } else {
      $archives{$basename} += 1;
    }
  }

  # Strip basenames with only 1 polaristion
  my @keys = keys (%archives);
  foreach $key (@keys) {
    if ($archives{$key} == 1) {
      delete($archives{$key});
    }
  }

  my $ts_file = "";
  my $bp_file = "";
  my $bps_file = "";

  # reverse sort the files
  my @r_sort_keys = sort {$b cmp $a} @keys;
  # only keep the mon file for each type of file

  foreach $key (@r_sort_keys) {
    # .ts files
    if ($key =~ m/.ts$/) {
      if ($ts_file eq "") {
        $ts_file = $key;
      } else {
        Dada->logMsg(3, DL, "getUnprocessedFiles: discarding ".$key."?");
        unlink $key."0";
        unlink $key."1";
        delete($archives{$key});
      }
    }

    # .bp files
    if ($key =~ m/.bp$/) {
      if ($bp_file eq "") {
        $bp_file = $key;
      } else {
        Dada->logMsg(3, DL, "getUnprocessedFiles: discarding ".$key."?");
        unlink $key."0";
        unlink $key."1";
        delete($archives{$key});
      }
    } 

    # .bps files
    if ($key =~ m/.bps$/) {
      if ($bps_file eq "") {
        $bps_file = $key;
      } else {
        Dada->logMsg(3, DL, "getUnprocessedFiles: discarding ".$key."?");
        unlink $key."0";
        unlink $key."1";
        delete($archives{$key});
      }
    }
  }

  # Strip basenames with only 1 polaristion
  my $files_to_return = "";
  @keys = keys (%archives);
  foreach $key (@keys) {
    $files_to_return .= $key." ";
  }

  chdir "../../";

  Dada->logMsg(2, DL, "getUnprocessedFiles: returning ".$files_to_return);
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
  Dada->logMsg(2, DL, "Deleting processed archives ".$cmd);
  my $response = `$cmd`;
  if ($? != 0) {
    Dada->logMsgWarn($warn, "rm failed: \"".$response."\"");
  }

  return 0;

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

  Dada->logMsg(2, DL, "Daemon control thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  Dada->logMsg(2, DL, "Unlinking PID file: ".$pidfile);
  unlink($pidfile);

  Dada->logMsg(2, DL, "Daemon control thread ending");

}


#
# Finds the age of the observation and number of files
#
sub getObsInfo($) {

  (my $dir) = @_;
  Dada->logMsg(3, DL, "getObsInfo(".$dir.")");

  my $obs_age = 0;
  my $n_files = 0;
  my $cmd = "";
  my $num_dirs = 0;

  # Current time
  my $time_curr = time;

  # Determine how many beam subdirectories exist
  $cmd = "find ".$dir."/* -type d | wc -l";
  $num_dirs = `$cmd`;
  chomp $num_dirs;
  Dada->logMsg(3, DL, "getObsInfo: num_dirs = ".$num_dirs);

  # If the clients have created the beam directories
  if ($num_dirs > 0) {

    # Get the number of mon files 
    $cmd = "find ".$dir."/*/ -regex '.*[bp|ts|bps][0|1]' -printf \"%f\\n\" | wc | awk '{print \$1}'";
    $n_files = `$cmd`;
    chomp $n_files;

    # If we have any mon files
    if ($n_files > 0) {
  
      # Get the time of the most recently created file, in any beam
      $cmd  = "find ".$dir."/*/ -regex '.*[bp|ts|bps][0|1]' -printf \"%T@\\n\" | sort | tail -n 1";
      my $time_newest_file = `$cmd`;
      chomp $time_newest_file;

      $obs_age = $time_curr - $time_newest_file;
      Dada->logMsg(3, DL, "getObsInfo: newest mon file was ".$obs_age);

      # Sometimes this reports negative!
      if ($obs_age < 0) {
        $obs_age = 0;
      }

      if (($obs_age >= 0) && ($obs_age < 300)) {
        # Normal
      } else {
        Dada->logMsgWarn($warn, "getObsInfo: had mon files, but weird age: ".$obs_age);
        
      }

    # either all processed or non yet here
    } else {

      $n_files = 0;

      # Determine the "age" of the beam subdirs
      $cmd = "find ".$dir."/* -type d -printf \"%T@\\n\" | sort | tail -n 1";
      my $time_dir= `$cmd`;
      chomp $time_dir;

      if ($time_dir) {
      
        $obs_age = $time_curr - $time_dir;

        Dada->logMsg(3, DL, "getObsInfo: newest beam dir was ".$obs_age." old");

        # If the obs.processing exists
        if (-f $dir."/obs.processing") {
      
          # At the end of a legit obs, no mon files will appear for ~60 seconds, after which
          # point the observation will be finished. Only warn about it after 70 seconds 
          if ($obs_age > 70) { 
            Dada->logMsgWarn($warn, "getObsInfo: current processing ".$dir.", but no mon files, age: ". $obs_age);
          } else {
            Dada->logMsg(2, DL, "getObsInfo: current processing ".$dir.", but no mon files, age: ". $obs_age);
          }
          if ($obs_age < 0) {
            $obs_age = 0;
          }

        } else {

          # If the obs.start is more than 5 minutes old, but we have no
          if ($obs_age > 5*60) {
            Dada->logMsgWarn($warn, "getObsInfo: beam dir age (".$obs_age.") was more than 300, and no obs.processing ");
            $obs_age = -1;
          }

        }

      } else {
        Dada->logMsgWarn($warn, "getObsInfo: could not determine age of subdirs, dud obs");
        $obs_age = -1;
      }
    }


  # no directories yet
  } else {

    $n_files = 0;

    # get the age of the current directory
    $cmd = "find ".$dir." -maxdepth 0 -type d -printf \"%T@\\n\"";
    my $time_dir = `$cmd`;
    chomp $time_dir;

    $obs_age = $time_curr - $time_dir;

    Dada->logMsg(3, DL, "getObsInfo: no beam subdirs in ".$dir." age = ".$obs_age);

    # If the directory was more than 5 minutes old, it must be erroneous 
    if ($obs_age > 5*60) {
      Dada->logMsgWarn($warn, "getObsInfo: no beam subdirs in ".$dir." and more than 300 seconds old: ".$obs_age);
      $obs_age = -1;

    # Its probably a brand new directory, and the beam dirs haven't been created yet
    } else {
      Dada->logMsg(3, DL, "getObsInfo: no beam subdirs, assuming brand new obs");
      $obs_age = 0;
    }

  }
  
  Dada->logMsg(3, DL, "getObsInfo: returning ".$obs_age.", ".$n_files);

  return ($obs_age, $n_files);

}


sub deleteObservation($) {

  (my $dir) = @_;

  if (-d $dir) {

    Dada->logMsg(1, DL, "Deleting observation: ".$dir);
    $cmd = "rm -rf $dir";
    `$cmd`;
    return $?;
  } else {
    return 1;
  }

}

#
# Cleans up an observation, removing all old files, but leaves 1 set of image files
#
sub cleanUpObs($) {

  (my $dir) = @_;

  my $file = "";
  my @files = ();
  my $cmd = "";

  Dada->logMsg(2, DL, "cleanUpObs(".$dir.")");

  # Clean up all the mon files that may have been produced
  Dada->logMsg(2, DL, "cleanUpObs: removing any ts, bp or bps files");
  my $cmd = "find ".$dir." -regex \".*[ts|bp|bps][0|1]\" -printf \"%P\n\"";
  Dada->logMsg(3, DL, "find ".$dir." -regex \".*[ts|bp|bps][0|1]\" -printf \"\%P\"");
  my $find_result = `$cmd`;

  @files = split(/\n/,$find_result);
  foreach $file (@files) {
    Dada->logMsg(3, DL, "unlinking $dir."/".$file");
    unlink($dir."/".$file);
  }

  # Clean up all the old png files, except for the final ones
  Dada->logMsg(2, DL, "cleanUpObs: removing any old png files");
  removeAllOldPngs($dir);

}




#
# Remove all old images in the sub directories
#
sub removeAllOldPngs($) {

  (my $dir) = @_;

  Dada->logMsg(2, DL, "removeAllOldPngs(".$dir.")");

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
    Dada->logMsg(2, DL, "removeAllOldPngs: clearing out ".$beamdir);

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

sub removeOldPngs($$$) {

  my ($dir, $type, $res) = @_;

  # remove any existing plot files that are more than 20 seconds old
  my $cmd  = "find ".$dir." -name '*".$type."_".$res.".png' -printf \"%T@ %f\\n\" | sort -n -r";
  my $result = `$cmd`;
  my @array = split(/\n/,$result);

  my $time = 0;
  my $file = "";
  my $line = "";

  # if there is more than one result in this category and its > 20 seconds old, delete it
  for ($i=1; $i<=$#array; $i++) {

    $line = $array[$i];
    ($time, $file) = split(/ /,$line,2);

    if (($time+30) < time)
    {
      $file = $dir."/".$file;
      Dada->logMsg(3, DL, "unlinking old png file ".$file);
      unlink($file);
    }
  }
}



sub patchInTcsLogs($$){
  my ($obs_dir, $utcname) = @_;

  my $tcs_logfile = $utcname."_bpsr.log";
  my $cmd = "scp -p pulsar\@jura.atnf.csiro.au:/psr1/tcs/logs/$tcs_logfile $obs_dir/$utcname/";
  Dada->logMsg(1, DL, "Getting TCS log file via: ".$cmd);
  my ($result,$response) = Dada->mySystem($cmd);
  if ($result!="ok"){
    Dada->logMsgWarn($error, "Could not get the TCS log file, msg was: ".$response);
    return ($result,$response);
  }
  my $cmd = "merge_tcs_logs.csh $obs_dir/$utcname $tcs_logfile";
  Dada->logMsg(1, DL, "Merging TCS log file via: ".$cmd);
  my ($result,$response) = Dada->mySystem($cmd);
  Dada->logMsg(3, DL, "Merging TCS log: ".$result." ".$response);

  if ($result!="ok"){
    Dada->logMsgWarn($error, "Could not merge the TCS log file, msg was: ".$response);
  }

  return ($result,$response);

}

sub copyPsrxmlToDb($$) {

  my ($obs_dir, $utc_name) = @_;

  my $result = "";
  my $response = "";
  my $db_dump_dir = "/nfs/control/bpsr/header_files/";

  if (! -d $db_dump_dir) {
    mkdir $db_dump_dir;
  }

  my $psrxml_file = $utc_name.".psrxml";
  my $b_psrxml_file = "";
  my $b = "";

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $b = $cfg{"BEAM_".$i};
    $b_psrxml_file = $utc_name."_".$b.".psrxml";

    # Copy <archives>/<utc>/<beam>/<utc>.psrxml to header_files/<utc>_<beam>.psrxml 
    if ( -f $obs_dir."/".$utc_name."/".$b."/".$psrxml_file) {

      $cmd = "cp ".$obs_dir."/".$utc_name."/".$b."/".$psrxml_file." ".$db_dump_dir."/".$b_psrxml_file;
      Dada->logMsg(1, DL, "copyPsrxmlToDb: ".$cmd);
      ($result, $response) = Dada->mySystem($cmd);
      Dada->logMsg(2, DL, "copyPsrxmlToDb: ".$cmd);
      if ($result ne "ok") {
        Dada->logMsgWarn($warn, "copyPsrxmlToDb: ".$cmd." failed: ".$response);
      }
    } else {
      Dada->logMsg(1, DL, "could not find psrxml file: ".$obs_dir."/".$utc_name."/".$b."/".$psrxml_file);
    }
  } 

  return ("ok", "");
}
