#!/usr/bin/env perl

###############################################################################
#
# client_hispec_results_monitor.pl 
#
# This script transfers the output of the_decimator to the results directory
# on the server

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Hispec;               # HISPEC/DADA Module for configuration options
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
%cfg = Hispec::getConfig();
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
    my %roach = Hispec::getROACHConfig();
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
      @all_obs = split(/\n/, $response);
      logMsg(2, "INFO", "main: found ".($#all_obs+1)." obs to consider");

      # now get a list of all observations marked beam.finished and more than 10 minutes old
      # to exclude them
      $cmd = "find ".$archive_dir." -mindepth 2 -maxdepth 2 -type f -mmin +10 ".
             "-name 'beam.*' | awk -F/ '{print \$(NF-1)}' | sort";
      ($result, $response) = Dada::mySystem($cmd);
      if ($result ne "ok") 
      {
        logMsg(0, "WARN", "find list of completed obs failed: ".$response);
      }
      else
      {
        @finished_obs = split(/\n/, $response);
        logMsg(2, "INFO", "main: found ".($#finished_obs+1)." obs marked finished");
        $total_processed = 0;

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
            logMsg(2, "INFO", "main: processHispecFiles(".$obs.")");
            $processed = processHispecFiles($obs);
            logMsg(2, "INFO", "main: processHispecFiles processed ".$processed." files");
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
sub processHispecFiles($) 
{
  (my $list) = @_;

  my $result = "";
  my $response = "";
  my $cmd = "";

  my @hispec_files = ();
  my $hispec_file = "";
  my @keys = ();
  my $key = "";
  my $plot_file = "";
  my @plot_files = ();

  # get ALL unprocessed hispec files 
  $cmd = "find ".$list." -mindepth 1 -maxdepth 1 -type f -name '*.?c' | sort";
  logMsg(2, "INFO", "processHispecFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "processHispecFiles: ".$result." ".$response);

  @hispec_files = split(/\n/, $response);

  logMsg(2, "INFO", "processHispecFiles: found ".($#hispec_files+1)." files to process");

  # dont bother if no mon files exist
  if ($#hispec_files == -1) {
    return 0;
  }

  logMsg(2, "INFO", "processHispecFiles: processing ".($#hispec_files+1)." mon files");

  my $utc = "";
  my $file = "";
  my $file_dir = "";
  my $remote_dir = "";

  my $send_list = "";

  foreach $key (@hispec_files)
  {
    logMsg(2, "INFO", "processHispecFiles: processing ".$key);
    ($utc, $file) = split( /\//, $key);
    $file_dir = $utc;
    $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc."/".$beam;

    # Plot the mon files and return a list of plot files that have been created
    logMsg(2, "INFO", "processHispecFiles: plotHispecFile($file_dir, $file)");
    @plot_files = plotHispecFile($file_dir, $file);
    logMsg(2, "INFO", "processHispecFiles: plotHispecFile returned ".($#plot_files+1)." files");

    $send_list = ""; 
    foreach $plot_file (@plot_files) 
    { 
      $send_list .= " ".$file_dir."/".$plot_file;
    }

    logMsg(2, "INFO", "processHispecFiles: sendToServer(".$send_list.", ".$remote_dir.")");
    ($result, $response) = sendToServer($send_list, $remote_dir);
    logMsg(3, "INFO", "processHispecFiles: sendToServer() ".$result." ".$response);

    foreach $plot_file (@plot_files) 
    { 
      unlink($file_dir."/".$plot_file);
    }

    unlink($file_dir."/".$file);

    logMsg(2, "INFO", "processMonFile: ".$key." processed");
  }

  # return the number of files processed
  return ($#hispec_files+ 1);
}

sub plotHispecFile($$) 
{
  my ($file_dir, $hispec_file) = @_;

  logMsg(3, "INFO", "plotHispecFile(".$file_dir.", ".$hispec_file.")");

  my @plot_files = ();
  my $cmd = "";
  my $result = "";
  my $response = "";

  chdir $file_dir;

  # selete any old images in this directory
  my $response;

  # create the low resolution file
  $cmd = "hispec_waterfall_plot ".$hispec_file." -z -l -p -g 114x84 -D ".$hispec_file.".png/png";
  logMsg(2, "INFO", "plotHispecFile: ".$cmd);
  $response = `$cmd 2>&1`;
  if ($? != 0) 
  {
    logMsg(0, "WARN", "plotHispecFile: ".$cmd." failed: ".$response);
  }
  else
  {
    ($result, $response) = renamePgplotFile($hispec_file.".png", $hispec_file."_112x84.png");
    if ($result eq "ok") {
      push @plot_files, $hispec_file."_112x84.png"; 
    } else {
      logMsg(1, "INFO", "plotHispecFile: rename failed: ".$response);
    }

  }

  # Create the mid resolution file
  $cmd = "hispec_waterfall_plot ".$hispec_file." -z -l -g 400x300 -D ".$hispec_file.".png/png";
  logMsg(2, "INFO", "plotHispecFile: ".$cmd);
  $response = `$cmd 2>&1`;
  if ($? != 0)
  {
    logMsg(0, "WARN", "plotHispecFile: ".$cmd." failed: ".$response);
  }
  else
  {
    ($result, $response) = renamePgplotFile($hispec_file.".png", $hispec_file."_400x300.png");
    if ($result eq "ok") {
      push @plot_files, $hispec_file."_400x300.png";
    } else {
      logMsg(1, "INFO", "plotHispecFile: rename failed: ".$response);
    }
  }

  # Create the high resolution file
  $cmd = "hispec_waterfall_plot ".$hispec_file." -z -l -g 1024x768 -D ".$hispec_file.".png/png";
  logMsg(2, "INFO", "plotHispecFile: ".$cmd);
  $response = `$cmd 2>&1`;
  if ($? != 0) {
    logMsg(0, "WARN", "plotHispecFile: ".$cmd." failed: ".$response);
  } else {
    ($result, $response) = renamePgplotFile($hispec_file.".png", $hispec_file."_1024x768.png");
    if ($result eq "ok") {
      push @plot_files, $hispec_file."_1024x768.png";
    } else {
      logMsg(1, "INFO", "plotHispecFile: rename failed: ".$response);
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
  ($result, $response) = Dada::mySystem($cmd,0);
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
