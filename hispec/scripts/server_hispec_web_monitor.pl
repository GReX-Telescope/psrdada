#!/usr/bin/env perl

###############################################################################
#
# server_hispec_web_monitor.pl
#
# This script maintains vital statistics for all the client/server machines
# and stores this information in memory. Web browsers will connect to a port
# on this script for this information
#

use lib $ENV{"DADA_ROOT"}."/bin";


use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Hispec;           # Hispec Module for configuration options
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DL           => 1;

#
# Global Variables
#
our %cfg                        = Hispec::getConfig();
our %roaches                    = Hispec::getROACHConfig();
our $error                      = $cfg{"STATUS_DIR"}."/hispec_web_monitor.error";
our $warn                       = $cfg{"STATUS_DIR"}."/hispec_web_monitor.warn";
our $quit_daemon : shared       = 0;
our $daemon_name : shared       = Dada::daemonBaseName($0);
our $node_info : shared         = "";
our $curr_obs : shared          = "";
our $status_info : shared       = "";
our $image_string : shared      = "";

#
# Main
#
{

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";

  my $control_thread = 0;
  my $sock = 0;
  my $rh = "";
  my $string = "";
  my $handle = 0;

  # Sanity check for this script
  if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
    print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
    print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
    exit(1);
  }

  $sock = new IO::Socket::INET (
    LocalHost => $cfg{"SERVER_HOST"}, 
    LocalPort => $cfg{"SERVER_WEB_MONITOR_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  die "Could not create listening socket: $!\n" unless $sock;

  # Redirect standard output and error
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, DL, "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($sock);   # add the main socket to the set

  Dada::logMsg(2, DL, "Waiting for connection on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_WEB_MONITOR_PORT"});

  # Disk space on nodes [R]
  # Data block info on nodes [R]
  # Load info on nodes [R]
  my $node_info_thread = 0;

  # Current images for web display [L]
  my $image_info_thread = 0;

  # Current source information [L]
  my $curr_info_thread = 0;

  # Current status file information [L]
  my $status_info_thread = 0;

  # Launch monitoring threads
  $curr_info_thread = threads->new(\&currentInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $status_info_thread = threads->new(\&statusInfoThread);
  $node_info_thread = threads->new(\&nodeInfoThread);
  $image_info_thread = threads->new(\&imageInfoThread, $cfg{"SERVER_RESULTS_DIR"});

  # Sleep for a few seconds to allow the threads to start and collect
  # their first iteration of data
  sleep(3);

  while (!$quit_daemon)
  {
    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 2);

    foreach $rh (@$rh_set) 
    {
      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $sock)
      {
        $handle = $rh->accept();
        $handle->autoflush();
        Dada::logMsg(2, DL, "Accepting connection");

        # Add this read handle to the set
        $read_set->add($handle); 
        $handle = 0;
      }
      else
      {
        $string = Dada::getLine($rh);

        if (! defined $string) 
        {
          Dada::logMsg(2, DL, "Closing a connection");
          $read_set->remove($rh);
          close($rh);
        } 
        else
        {
          Dada::logMsg(2, DL, "<- ".$string);
          my $r = "";

          if    ($string eq "node_info")     { $r = $node_info; }
          elsif ($string eq "img_info")      { $r = $image_string; }
          elsif ($string eq "curr_obs")      { $r = $curr_obs; }
          elsif ($string eq "status_info")   { $r = $status_info; }
          else    { Dada::logMsgWarn($warn, "unexpected command: ".$string); } 

          print $rh $r."\n";
          Dada::logMsg(2, DL, "-> ".$r);

          # experimental!
          $read_set->remove($rh);
          close($rh);
        }
      }
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  # Rejoin other threads
  $curr_info_thread->join();
  $status_info_thread->join();
  $node_info_thread->join();
  $image_info_thread->join();

  close($sock);
                                                                                  
  Dada::logMsg(0, DL, "STOPPING SCRIPT");
                                                                                  
  exit(0);
}

###############################################################################
#
# Functions
#

sub controlThread($) {

  (my $pid_file) = @_;

  Dada::logMsg(2, DL, "controlThread: starting");

  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  Dada::logMsg(2, DL, "Unlinking PID file: ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(1, DL, "controlThread: exiting");

}


#
# Handle INT AND TERM signals
#
sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(5);
  print STDERR $daemon_name." : Exiting\n";
  exit(1);
}


#
# maintains the curr_info string
#
sub currentInfoThread($) 
{
  (my $results_dir) = @_;

  Dada::logMsg(1, DL, "currentInfoThread: starting");

  my $sleep_time = 10;
  my $sleep_counter = 0;

  my $cmd = "find ".$results_dir." -ignore_readdir_race -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
  my $obs = "";
  my $tmp_str = "";
  my %cfg_file = ();

  while (!$quit_daemon) 
  {
    if ($sleep_counter > 0) 
    {
      $sleep_counter--;
      sleep(1);
    } 
    else
    {
      $sleep_counter = $sleep_time;

      Dada::logMsg(3, DL, "currentInfoThread: ".$cmd);
      $obs = `$cmd`;
      chomp $obs;
      Dada::logMsg(3, DL, "currentInfoThread: ".$obs);

      if (-f $results_dir."/".$obs."/obs.info")
      {
        $tmp_str = "";
        %cfg_file = Dada::readCFGFile($results_dir."/".$obs."/obs.info"); 

        $tmp_str .= "SOURCE:::".$cfg_file{"SOURCE"}.";;;";
        $tmp_str .= "RA:::".$cfg_file{"RA"}.";;;";
        $tmp_str .= "DEC:::".$cfg_file{"DEC"}.";;;";
        $tmp_str .= "CFREQ:::".$cfg_file{"CFREQ"}.";;;";
        $tmp_str .= "BANDWIDTH:::".$cfg_file{"BANDWIDTH"}.";;;";
        $tmp_str .= "ACC_LEN:::".$cfg_file{"ACC_LEN"}.";;;";
        $tmp_str .= "NUM_PWC:::".$cfg_file{"NUM_PWC"}.";;;";
        $tmp_str .= "PID:::".$cfg_file{"PID"}.";;;";
        $tmp_str .= "UTC_START:::".$cfg_file{"UTC_START"}.";;;";
        $tmp_str .= "PROC_FILE:::".$cfg_file{"PROC_FILE"}.";;;";
        $tmp_str .= "REF_BEAM:::".$cfg_file{"REF_BEAM"}.";;;";
        $tmp_str .= "NBEAM:::".$cfg_file{"NBEAM"}.";;;";
        $tmp_str .= "INTERGRATED:::0;;;";
 
        Dada::logMsg(3, DL, $tmp_str); 

        # update the global variable 
        $curr_obs = $tmp_str;

        Dada::logMsg(2, DL, "currInfoThread: ".$curr_obs);
      }
    }
  }

  Dada::logMsg(1, DL, "currentInfoThread: exiting");
}


#
# Maintains a listing of the current images
#
sub imageInfoThread($)
{
  my ($results_dir) = @_;

  Dada::logMsg(1, DL, "imageInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;

  my $cmd = "";
  my @images = ();
  my @image_types = qw(bp ts fft dts pvf);
  my $obs = "";

  my $i = 0;
  my %ac_images = ();
  my %cc_images = ();
  my %tp_images = ();
  my @keys = ();
  my $k = "";
  my $dirs_string = ();
  my @dirs = ();
  my $no_image = "";
  my $xml = "";
  my $img_string = "";

  my $img_beam = "";
  my $img_file = "";
  my $img_time = "";
  my $img_info = "";
  my $img_type = "";
  my $img_plot = "";
  my $img_res = "";
  my $img_w = "";
  my $img_h = "";
  my %beams = ();

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $beams{$i} = $roaches{"BEAM_".$i};
  }

  chdir $results_dir;

  while (!$quit_daemon)
  {
    if ($sleep_counter > 0)
    {
      $sleep_counter--;
      sleep(1);
    }
    else
    {
      $sleep_counter = $sleep_time;
      $xml = "";

      # find the most recent directory (ls gives arg list too long)
      $cmd = "find ".$results_dir." -ignore_readdir_race -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
      $obs = `$cmd`;
      chomp $obs;

      # get the listing of beam dirs
      @dirs = ();
      $cmd = "find ".$results_dir."/".$obs." -ignore_readdir_race -mindepth 1 -maxdepth 1 -type d -name '??' -printf '\%f '";
      $dirs_string = `$cmd`;
      chomp $dirs_string;

      # get the listing of small image files
      $cmd = "find ".$results_dir."/".$obs." -ignore_readdir_race -name '*_112x84.png' | sort | awk -F/ '{print \$(NF-1), \$(NF)}'";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      %ac_images = ();
      %cc_images = ();
      %tp_images = ();

      for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
      {
        $b = $beams{$i};
        $no_image =  "../images/hispec_beam_disabled_240x180.png";
        if ($dirs_string =~ m/$b/) 
        {
          $no_image = "../../images/blankimage.gif"; 
        }
        $ac_images{$b} = $no_image;
        $cc_images{$b} = $no_image;
        $tp_images{$b} = $no_image;
      }


      for ($i=0; $i<=$#images; $i++)
      {
        ($img_beam, $img_file) = split(/ /,$images[$i], 2);
        ($img_time, $img_info, $img_type) = split(/\./, $img_file);

        ($img_plot, $img_res) = split(/_/, $img_info);
        ($img_w, $img_h) = split(/x/, $img_res);

        if ($img_plot eq "ac") { $ac_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "cc") { $cc_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "tp") { $tp_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
      }

      for  ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
      {
        $b = $beams{$i};

        $xml .= "<beam name='".$b."'>";

        $xml .= "<img type='ac' width='112' height='84'>".$ac_images{$b}."</img>";
        $xml .= "<img type='cc' width='112' height='84'>".$cc_images{$b}."</img>";
        $xml .= "<img type='tp' width='112' height='84'>".$tp_images{$b}."</img>";

        $xml .= "</beam>\n";
      }

      $image_string = $xml;

      Dada::logMsg(2, DL, "imageInfoThread: collected images");
    }
  }

  Dada::logMsg(1, DL, "imageInfoThread: exiting");
}


#
# Monitors the STATUS_DIR for warnings and errors
#
sub statusInfoThread() 
{
  my $status_dir = $cfg{"STATUS_DIR"};

  Dada::logMsg(1, DL, "statusInfoThread: starting");

  my @files = ();
  my @arr = ();
  my $sleep_time = 10;
  my $sleep_counter = 0;

  my $i = 0;
  my $statuses = "";
  my $file = "";
  my $msg = "";
  my $pwc = ""; 
  my $tag = ""; 
  my $type = "";
  my $tmp_str = "";
  my $cmd = "";

  my %pwc_ids = ();
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $pwc_ids{$cfg{"PWC_".$i}} = $i;
    $pwc_ids{sprintf("%02d",$i)} = $i;
  }

  while (!$quit_daemon) 
  {
    if ($sleep_counter > 0) 
    {
      sleep(1);
      $sleep_counter--;
    } 
    else
    {
      $sleep_counter = $sleep_time;

      $tmp_str = "";

      @files = ();

      $cmd = "ls -1 ".$status_dir;
      $statuses = `$cmd`;
      chomp($statuses);
      @files = split(/\n/, $statuses);

      # get the current warnings and errors
      for ($i=0; $i<=$#files; $i++) {
        $file = $files[$i];
        $msg = `tail -n 1 $status_dir/$file`;
        chomp $msg;

        @arr = ();
        @arr = split(/\./,$file);

        $pwc = "";
        $tag = "";
        $type = "";

        # for pwc, sys and src client errors
        if ($#arr == 2) 
        {
          $pwc = $pwc_ids{$arr[0]};
          $tag = $arr[1];
          $type = $arr[2];
        } 
        elsif ($#arr == 1)
        {
          $pwc = "server";
          $tag = $arr[0];
          $type = $arr[1];
        }
        else
        {
          Dada::logMsg(0, DL, "statusInfoThread: un-parseable status file: ".$file);
        }

        if ($type eq "warn")
        {
          $tmp_str .= "<daemon_status type='warning' pwc='".$pwc."' tag='".$tag."'>".$msg."</daemon_status>";
        }
        if ($type eq "error")
        {
          $tmp_str .= "<daemon_status type='error' pwc='".$pwc."' tag='".$tag."'>".$msg."</daemon_status>";
        }
      }

      $status_info = $tmp_str;
      Dada::logMsg(2, DL, "statusInfoThread: ".$status_info);

    }
  }

  Dada::logMsg(1, DL, "statusInfoThread: exiting");
}

#
# Maintains information with clients about DB, Disk and load
#
sub nodeInfoThread() 
{
  Dada::logMsg(1, DL, "nodeInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;
  my $port = $cfg{"CLIENT_MASTER_PORT"};

  my @machines = ();
  my @results = ();
  my @responses = ();

  my $tmp_str = "";
  my $i = 0;
  my $result = "";
  my $response = "";

  # setup the list of machines that we will poll
  my %hosts = ();
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $hosts{$cfg{"PWC_".$i}} = 1;
  }
  for ($i=0; $i<$cfg{"NUM_HELP"}; $i++) {
    $hosts{$cfg{"HELP_".$i}} = 1;
  }
  $hosts{"hipsr-srv0"}  = 1;

  @machines = keys %hosts;

  my $handle = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;
                                                                                                          
    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      @results = ();
      @responses = ();

      for ($i=0; ((!$quit_daemon) && $i<=$#machines); $i++) {

        $handle = Dada::connectToMachine($machines[$i], $port, 0);
        # ensure our file handle is valid
        if (!$handle) {
          $result = "fail";
          $response = "<status><host>".$machines[$i]."</host></status>";
        } else {
          ($result, $response) = Dada::sendTelnetCommand($handle, "get_status");
          $handle->close();
        }

        $results[$i] = $result;
        $responses[$i] = $response;
      }

      # now set the global string
      $tmp_str = "<node_statuses>";
      for ($i=0; $i<=$#machines; $i++) {
        $tmp_str .= $responses[$i];
      }
      $tmp_str .= "</node_statuses>";
      $node_info = $tmp_str;

      Dada::logMsg(2, DL, "nodeInfoThread: ".$node_info);
    }
  }  
  Dada::logMsg(1, DL, "nodeInfoThread: exiting");
}
