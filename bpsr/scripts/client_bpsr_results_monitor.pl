#!/usr/bin/env perl

###############################################################################
#
# client_bpsr_results_monitor.pl 
#
# This script transfers the output of the_decimator to the results directory
# on the server

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Bpsr;               # BPSR/DADA Module for configuration options
use strict;             # strict mode (like -Wall)
use threads;
use threads::shared;


sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $pwc_id : shared;
our $beam : shared;
our %cfg : shared;
our $log_host;
our $log_port;
our $log_sock;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$beam = 0;
%cfg = Bpsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}
$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (($pwc_id >= 0) &&  ($pwc_id < $cfg{"NUM_PWC"}))
{
  # and matches configured hostname
  if ($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName())
  {
    my %roach = Bpsr::getROACHConfig();
    # determine the relevant PWC based configuration for this script 
    $beam = $roach{"BEAM_".$pwc_id};
  }
  else
  {
    print STDERR "PWC_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "PWC_ID was not a valid integer between 0 and ".($cfg{"NUM_PWC"}-1)."\n";
  usage();
  exit(1);
}

# Sanity check to prevent multiple copies of this daemon running
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);

###############################################################################
#
# Main
#
{

  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";
  my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;

  # Autoflush STDOUT
  $| = 1;

  # sanity check
  if ( ! -d $archive_dir) 
  {
    print STDERR "Beam archive dir ".$archive_dir." did not exist\n";
    exit(-1);
  }

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) 
  {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  logMsg(0,"INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  # Change to the local archive directory
  chdir $archive_dir;

  my $processed = 0;
  my $total_processed = 0;

  my $sleep_total = 5;
  my $sleep_count = 0;
  my @all_obs = ();
  my @finished_obs = ();
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $j = 0;
  my $obs = "";
  my $finished = 0;

  # Loop until daemon control thread asks us to quit
  while (!($quit_daemon)) 
  {
      
    @all_obs = ();

    # get a list of all observations in the local disk
    $cmd = "find ".$archive_dir." -mindepth 1 -maxdepth 1 -type d -name '2*' -printf '\%f\\n' | sort";
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") 
    {
      logMsg(0, "WARN", "find list of all obs failed: ".$response);
    }
    else
    {
      $total_processed = 0;

      @all_obs = split(/\n/, $response);
      logMsg(2, "INFO", "main: found ".($#all_obs+1)." obs to consider");

      # now get a list of all observations marked beam.finished and more than 10 minutes old
      # to exclude them
      $cmd = "find ".$archive_dir." -mindepth 2 -maxdepth 2 -type f -mmin +10 ".
             "-name 'beam.*' | awk -F/ '{print \$(NF-1)}' | sort -n | uniq";
      ($result, $response) = Dada::mySystem($cmd);
      if ($result ne "ok") 
      {
        logMsg(0, "WARN", "find list of completed obs failed: ".$response);
      }
      else
      {
        @finished_obs = split(/\n/, $response);
        logMsg(2, "INFO", "main: found ".($#finished_obs+1)." obs marked finished");

        my $curr_utc_time = Dada::getUnixTimeUTC(Dada::printTime(time, "utc"));
        my $day_in_secs = 24 * 60 * 60;

        for ($i=0; ((!$quit_daemon) && ($i<=$#all_obs)); $i++) 
        {
          $obs = $all_obs[$i];
          $finished = 0;
          for ($j=0; $j<=$#finished_obs; $j++) 
          {
            if ($finished_obs[$j] eq $obs) 
            {
              $finished = 1; 
            } 
          }

          if (!$finished)
          {
            my $obs_time_unix  = Dada::getUnixTimeUTC($obs);
            if ($obs_time_unix + $day_in_secs < $curr_utc_time)
            {
              $finished = 1;

              if (! -f $archive_dir."/".$obs."/beam.finished")
              {
                logMsg(1, "INFO", $obs." processing -> finished");
                push (@finished_obs, $obs);

                $cmd = "touch ".$archive_dir."/".$obs."/beam.finished";
                logMsg(2, "INFO", "main: ".$cmd);  
                ($result, $response) = Dada::mySystem($cmd);
                logMsg(2, "INFO", "main: ".$result." ".$response);
                sleep (1);
              }
            }
          }

          if (!$finished) 
          {
            logMsg(2, "INFO", "main: processMonFiles(".$obs.")");
            $processed = processMonFiles($obs);
            logMsg(2, "INFO", "main: processMonFiles processed ".$processed." files");
            $total_processed += $processed;
            
            logMsg(2, "INFO", "main: processDspsrFiles(".$obs.")");
            $processed = processDspsrFiles($obs);
            logMsg(2, "INFO", "main: processDspsrFiles processed ".$processed." files");
            $total_processed += $processed;

            logMsg(2, "INFO", "main: processTransientCandidates(".$obs.")");
            $processed = processTransientCandidates($obs);
            logMsg(2, "INFO", "main: processTransientCandidates processed ".$processed." files");
            $total_processed += $processed;
          }
        }
      }

      logMsg(3, "INFO", "main: total_processed=".$total_processed);

      # If nothing is happening, have a snooze
      if ($total_processed < 1) 
      {
        $sleep_count = 0;
        while (!$quit_daemon && ($sleep_count < $sleep_total)) {
          sleep(1);
          $sleep_count++;
        }
      }
      sleep(1);
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

  exit (0);
}


###############################################################################################
##
## Functions
##

#
# Look for mon files produced by the_decimator, plot them and send the plots to the server
#
sub processMonFiles($) 
{
  (my $list) = @_;

  my $result = "";
  my $response = "";
  my $cmd = "";

  my %unprocessed = ();
  my @mon_files = ();
  my $mon_file = "";
  my @keys = ();
  my $key = "";
  my $plot_file = "";
  my @plot_files = ();

  # delete any files from pols [2|3] for now [WvS]
  $cmd = "find ".$list." -mindepth 1 -maxdepth 1 -type f -regex '.*[ts|bp|bps][2|3]' -delete";
  logMsg(2, "INFO", "processMonFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "processMonFiles: ".$result." ".$response);

  # get ALL unprocessed mon files (not checking aux dirs)
  $cmd = "find ".$list." -mindepth 1 -maxdepth 1 -type f -regex '.*[ts|bp|bps][0|1]' | sort";
  logMsg(2, "INFO", "processMonFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "processMonFiles: ".$result." ".$response);

  @mon_files = split(/\n/, $response);

  logMsg(2, "INFO", "processMonFiles: found ".($#mon_files+1)." files to process");

  # dont bother if no mon files exist
  if ($#mon_files == -1) {
    return 0;
  }

  # add up how many of each pol we have
  foreach $mon_file (@mon_files) {
    logMsg(3, "INFO", "processMonFiles: ".$mon_file);
    $key = substr $mon_file, 0, -1;
    if (! exists ($unprocessed{$key})) {
      $unprocessed{$key} = 1;
    } else {
      $unprocessed{$key} += 1;
    }
  }

  # remove ones with only 1 pol
  @keys = keys (%unprocessed);
  foreach $key (@keys) {

    if ($unprocessed{$key} == 1) {
      # sometimes only 1 pol is outputted. if the file is > 1 minute in age
      # then delete it too
      $cmd = "stat -c \%Y ".$key."*";
      logMsg(2, "INFO", "processMonFiles: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "processMonFiles: ".$result." ".$response);
      if ((time - $response) > 60) {
        logMsg(2, "INFO", "processMonFiles: unlinking orphaned mon file ".$key);
        $cmd = "rm -f ".$key."0 ".$key."1";
        system($cmd);
      }

      delete($unprocessed{$key});
    }
  }

  @keys = keys (%unprocessed);
  logMsg(2, "INFO", "processMonFile: processing ".(($#keys+1)*2)." mon files");

  my $utc = "";
  my $file = "";
  my $pol0_file = "";
  my $pol1_file = "";
  my $file_dir = "";
  my $remote_dir = "";

  my $send_list = "";

  foreach $key (@keys) {

    logMsg(2, "INFO", "processMonFiles: processing ".$key."[0|1]");
    ($utc, $file) = split( /\//, $key);
    $file_dir = $utc;
    $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc."/".$beam;

    # Plot the mon files and return a list of plot files that have been created
    logMsg(2, "INFO", "processMonFiles: plotMonFile($file_dir, $file)");
    @plot_files = plotMonFile($file_dir, $file);
    logMsg(2, "INFO", "processMonFiles: plotMonFile returned ".($#plot_files+1)." files");

    $send_list = ""; 
    foreach $plot_file (@plot_files) 
    { 
      $send_list .= " ".$file_dir."/".$plot_file;
    }

    logMsg(2, "INFO", "processMonFiles: sendToServer(".$send_list.", ".$remote_dir.")");
    ($result, $response) = sendToServer($send_list, $remote_dir);
    logMsg(3, "INFO", "processMonFiles: sendToServer() ".$result." ".$response);

    foreach $plot_file (@plot_files) 
    { 
      unlink($file_dir."/".$plot_file);
    }

    $pol0_file = $file."0";
    $pol1_file = $file."1";

    my $aux_dir = $file_dir."/aux";

    if (! -d $aux_dir) 
    {
      logMsg(0, "WARN", "processMonFile: Aux dir for ".$utc." did not exist, creating...");
      $cmd = "mkdir -m 0755 -p ".$aux_dir;
      logMsg(2, "INFO", "processMonFile: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "processMonFile: ".$result." ".$response);
    }

    # move both pols of the mon files to the aux dir
    $cmd  = "mv ".$file_dir."/".$pol0_file." ".$file_dir."/".$pol1_file." ".$file_dir."/aux/";
    ($result, $response) = Dada::mySystem($cmd);

    if ($result ne "ok") 
    {
      logMsg(0, "ERROR", "Could not move file [".$file."] to aux dir [".$response."]");
      unlink($file_dir."/".$file."0");
      unlink($file_dir."/".$file."1");
    } 

    logMsg(2, "INFO", "processMonFile: ".$key." processed");
  }

  # return the number of files processed
  return (($#keys + 1)*2);
}

#
# process any dspsr output files produced
#
sub processDspsrFiles($) 
{
  (my $list) = @_;

  my $cmd = "find ".$list." -mindepth 1 -maxdepth 1 -type f -name '2*\.ar' | sort";
  my $find_result = `$cmd`;

  my @lines = split(/\n/,$find_result);
  my $line = "";

  my $result = "";
  my $response = "";

  my $bindir = Dada::getCurrentBinaryVersion();

  foreach $line (@lines) {

    # $line = substr($line,2);

    logMsg(2, "INFO", "Processing dspsr file \"".$line."\"");

    my ($utc, $file) = split( /\//, $line);
    my $file_dir = $utc;

    my $sumd_archive = $file_dir."/integrated.ar";
    my $curr_archive = $file_dir."/".$file;

    if (! -f $sumd_archive) {
      $cmd = "cp ".$curr_archive." " .$sumd_archive;
    } else {
      $cmd = $bindir."/psradd -s --inplace ".$sumd_archive." ".$curr_archive;
    }    

    logMsg(2, "INFO", $cmd); 
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") 
    {
      logMsg(0, "ERROR", "processDspsrFiles: ".$cmd." failed: ".$response);
    } 
    else
    {
      logMsg(2, "INFO", "processDspsrFiles: archive added");

      my $time_str = Dada::getCurrentDadaTime();
      my $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc."/".$beam;
      my $plot_file = "";
      my @plot_files = ();
      my $send_files = "";

      $plot_file = makePhaseVsFreqPlotFromArchive($time_str, "112x84", $sumd_archive, $file_dir);
      if ($plot_file ne "none")
      {
        push @plot_files, $file_dir."/".$plot_file;
      }

      $plot_file = makePhaseVsFreqPlotFromArchive($time_str, "400x300", $sumd_archive, $file_dir);
      if ($plot_file ne "none") 
      {
        push @plot_files, $file_dir."/".$plot_file;
      }

      $plot_file = makePhaseVsFreqPlotFromArchive($time_str, "1024x768", $sumd_archive, $file_dir);
      if ($plot_file ne "none") 
      {
        push @plot_files, $file_dir."/".$plot_file;
      }

      foreach $plot_file (@plot_files)
      {
        if (-f $plot_file)
        {
          $send_files .= " ".$plot_file;
        }
      }
      logMsg(2, "INFO", "processDspsrFiles: sendToServer(".$send_files.", ".$remote_dir.")");
      ($result, $response) = sendToServer($send_files, $remote_dir);
      logMsg(3, "INFO", "processDspsrFiles: sendToServer() ".$result." ".$response);
      foreach $plot_file (@plot_files)
      {
        if (-f $plot_file)
        {
          unlink $plot_file;
        }
      }

      my $tar_file = $file_dir."/archives.tar";

      # add this file to a tar file
      if ( -f $tar_file ) 
      {
        $cmd = "tar --force-local -C ".$file_dir." -rf ".$tar_file." ".$file;
      }
      else
      {
        $cmd = "tar --force-local -C ".$file_dir."  -cf ".$tar_file." ".$file;
      }
  
      logMsg(2, "INFO", $cmd);
      ($result, $response) = Dada::mySystem($cmd);
      if ($result ne "ok")
      {
        logMsg(0, "ERROR", "processDspsrFiles: ".$cmd." failed: ".$response);
      }
    }

    unlink ($curr_archive);

  }

  return ($#lines+1);

}

#
# process any transient candidates produced by heimdall pipeline
#
sub processTransientCandidates($) 
{
  (my $list) = @_;

  my $cmd = "find ".$list." -mindepth 1 -maxdepth 1 -type f -name '2*_??.cand' | sort";
  my $find_result = `$cmd`;

  my @files = split(/\n/,$find_result);
  my $file = "";
  my $line = "";
  my $utc = "";
  my $remote_dir = "";

  my $result = "";
  my $response = "";

  foreach $line (@files) 
  {
    logMsg(2, "INFO", "processTransientCandidates: processing candidate file: ".$file);

    ($utc, $file) = split( /\//, $line);

    $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc."/".$beam;

    logMsg(2, "INFO", "processTransientCandidates: sendToServer(".$line.", ".$remote_dir.")");
    ($result, $response) = sendToServer(" ".$line, $remote_dir);
    logMsg(3, "INFO", "processTransientCandidates: sendToServer() ".$result." ".$response);
 
    my $aux_dir = $utc."/aux";
    if (! -d $aux_dir)
    {
      logMsg(0, "WARN", "processTransientCandidates: Aux dir for ".$utc." did not exist, creating...");
      $cmd = "mkdir -m 0755 -p ".$aux_dir;
      logMsg(2, "INFO", "processTransientCandidates: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "processTransientCandidates: ".$result." ".$response);
    }

    # move both pols of the mon files to the aux dir
    $cmd  = "mv ".$line." ".$aux_dir."/";
    ($result, $response) = Dada::mySystem($cmd);
  }
 
  return ($#files+1);
}

# create a phase vs freq plot from a dspsr archive
#
sub makePhaseVsFreqPlotFromArchive($$$$)
{
  my ($timestr, $res, $archive, $dir) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $add = "";
  my $args = "";
  my $plotfile = "";


  # remove axes, labels and outside spaces for lil'plots
  if ($res eq "112x84") {
    $add = " -s ".$cfg{"SCRIPTS_DIR"}."/web_style.txt -c below:l=unset";
  }

  $args = "-g ".$res.$add." -jp -p freq -jT -j\"zap median,F 256,D\"";
  $plotfile = $timestr.".pvf_".$res.".png";

  $cmd = "psrplot ".$args." -D ".$dir."/".$plotfile."/png ".$archive;

  logMsg(2, "INFO", $cmd);
  ($result, $response) = Dada::mySystem($cmd);

  if ($result ne "ok") 
  {
    logMsg(0, "ERROR", "plot cmd \"".$cmd."\" failed: ".$response);
    $plotfile = "none";
  }
  else
  {
    # sometimes the plot file takes some time before it appears on disk, wait up to 5 seconds
    my $waitMax = 3;
    while ($waitMax > 0)
    {
      if (-f $dir."/".$plotfile)
      {
        $waitMax = 0;
      }
      else
      {
        logMsg(2, "INFO", "makePhaseVsFreqPlotFromArchive: waiting for ".$dir."/".$plotfile);
        sleep(1);
        $waitMax --;
      }
    }
  }

  return $plotfile;
}

sub plotMonFile($$) 
{
  my ($file_dir, $mon_file) = @_;

  logMsg(3, "INFO", "plotMonFile(".$file_dir.", ".$mon_file.")");

  my @plot_files = ();
  my $filetype = "";
  my $filebase = "";
  my $bindir =  Dada::getCurrentBinaryVersion();
  my $binary = $bindir."/plot4mon";
  my $cmd = "";
  my $result = "";
  my $response = "";

  chdir $file_dir;

  # Delete any old images in this directory
  my $response;

  if ($mon_file =~ m/bp$/) {
    $filetype = "bandpass";
    $filebase = substr $mon_file, 0, -3;

  } elsif ($mon_file =~ m/bps$/) {
    $filetype = "bandpass_rms";
    $filebase = substr $mon_file, 0, -4;

  } elsif ($mon_file =~ m/ts$/) {
    $filetype = "timeseries";
    $filebase = substr $mon_file, 0, -3;

  } else {
    $filetype = "unknown";
    $filebase = "";
  }

  logMsg(2, "INFO", "plotMonFile: ".$filetype." ".$filebase);

  if ($filetype eq "unknown") {
    # skip the file
    logMsg(2, "INFO", "plotMonFile: unknown");

  } else {

    # Create the low resolution file
    $cmd = $binary." ".$mon_file."0 ".$mon_file."1 -G 112x84 -nobox -nolabel -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -mmm ";
    }

    logMsg(2, "INFO", "plotMonFile: ".$cmd);
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      logMsg(0, "WARN", "plotMonFile: ".$cmd." failed: ".$response);

    } else {

      ($result, $response) = renamePgplotFile($mon_file.".png", $mon_file."_112x84.png");
      if ($result eq "ok") {
        push @plot_files, $mon_file."_112x84.png"; 
      } else {
        logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
      }

      if ($filetype eq "timeseries") {

        ($result, $response) = renamePgplotFile($filebase.".fft.png", $filebase.".fft_112x84.png");
        if ($result eq "ok") {
          push @plot_files, $filebase.".fft_112x84.png";
        } else {
          logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
        }
      }
    }

    # Create the mid resolution file
    $cmd = $binary." ".$mon_file."0 ".$mon_file."1 -G 400x300 -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -mmm ";
    }

    logMsg(2, "INFO", "plotMonFile: ".$cmd);
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      logMsg(0, "WARN", "plotMonFile: ".$cmd." failed: ".$response);
    } else {

      ($result, $response) = renamePgplotFile($mon_file.".png", $mon_file."_400x300.png");
      if ($result eq "ok") {
        push @plot_files, $mon_file."_400x300.png";
      } else {
        logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
      }

      if ($filetype eq "timeseries") {
        ($result, $response) = renamePgplotFile($filebase.".fft.png", $filebase.".fft_400x300.png");
        if ($result eq "ok") {
          push @plot_files, $filebase.".fft_400x300.png";
        } else {
          logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
        }
      }
    }

    # Create the high resolution file
    $cmd = $binary." ".$mon_file."0 ".$mon_file."1 -G 1024x768 -g /png";
    if ($filetype eq "timeseries") {
      $cmd = $cmd." -mmm ";
    }

    logMsg(2, "INFO", "plotMonFile: ".$cmd);
    $response = `$cmd 2>&1`;
    if ($? != 0) {
      logMsg(0, "WARN", "plotMonFile: ".$cmd." failed: ".$response);
    } else {
      ($result, $response) = renamePgplotFile($mon_file.".png", $mon_file."_1024x768.png");
      if ($result eq "ok") {
        push @plot_files, $mon_file."_1024x768.png";
      } else {
        logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
      }
    
      if ($filetype eq "timeseries") {
        ($result, $response) = renamePgplotFile($filebase.".fft.png", $filebase.".fft_1024x768.png");
        if ($result eq "ok") {
          push @plot_files, $filebase.".fft_1024x768.png";
        } else {
          logMsg(1, "INFO", "plotMonFile: rename failed: ".$response);
        }
      }
    }
  }

  chdir $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;
  return @plot_files;

}


#
# renames a file from src to dst, but waits for the file to appear as PGPLOT
# programs seems to return before the file has appeared on disk
#
sub renamePgplotFile($$) {

  my ($src, $dst) = @_;

  my $waitMax = 5;   # seconds
  my $cmd = "";
  my $result = "";
  my $response = "";

  while ($waitMax) {
    if (-f $src) {
      $waitMax = 0;
    } else {
      $waitMax--;
      sleep(1);
    }
  }
 
  if ( -f $src ) {
     $cmd = "mv ".$src." ".$dst;
    ($result, $response) = Dada::mySystem($cmd);
  } else {
    $result = "fail";
    $response = "renamePgplotFile: ".$src." did not appear after 5 seconds";
  }

  return ($result, $response);

}


# 
# transfer the specified file to the dir on the server
#
sub sendToServer($$)
{
  (my $local_files, my $remote_dir) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  # ensure each file is prefixed with a ./ to avoid mis-interpretation of : in filename
  $local_files =~ s/ 2/ \.\/2/g;

  $cmd = "rsync -p ".$local_files." dada@".$cfg{"SERVER_HOST"}.":".$remote_dir."/";

  logMsg(2, "INFO", "sendToServer: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "sendToServer: ".$result." ".$response);

  if ($result ne "ok")
  {
    logMsg(0, "WARN", "failed to copy ".$local_files." to ".$cfg{"SERVER_HOST"}.":".$remote_dir.": ".$response);
    return ("fail", "rsync failed");
  }
  return ("ok", "");
}

sub controlThread($)
{
  (my $pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) 
  {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) {

  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "results mon", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}
 

sub sigHandle($) {
  
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

  # if we CTRL+C twice, just hard exit
  if ($quit_daemon) {
    print STDERR $daemon_name." : Recevied 2 signals, Exiting\n";
    exit 1;

  # Tell threads to try and quit
  } else {

    $quit_daemon = 1;
    if ($log_sock) {
      close($log_sock);
    }
  }
}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}
