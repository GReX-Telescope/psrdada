#!/usr/bin/env perl

###############################################################################
#
# server_mopsr_web_monitor.pl
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
use Mopsr;           # Mopsr Module for configuration options
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
our %cfg                         = Mopsr::getConfig();
our $error                       = $cfg{"STATUS_DIR"}."/mopsr_web_monitor.error";
our $warn                        = $cfg{"STATUS_DIR"}."/mopsr_web_monitor.warn";
our $quit_daemon : shared        = 0;
our $daemon_name : shared        = Dada::daemonBaseName($0);
our $node_info : shared          = "";
our $curr_obs : shared           = "";
our $status_info : shared        = "";
our $image_string : shared       = "";
our $rx_monitor_string : shared  = "";
our $pfb_monitor_string : shared = "";
our $udp_monitor_string : shared = "";
our $mgt_lock_string : shared    = "";

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
  $SIG{PIPE} = \&sigHandle;

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

  Dada::logMsg(1, DL, "Waiting for connection on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_WEB_MONITOR_PORT"});

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

  # maintain information about the RX subsystem
  my $rx_monitor_thread = 0;

  # Monitoring thread for the PFB input
  my $pfb_monitor_thread = 0;

  # Monitoring thread for the PFB ouput
  my $udp_monitor_thread = 0;

  # Launch monitoring threads
  $curr_info_thread = threads->new(\&currentInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $status_info_thread = threads->new(\&statusInfoThread);
  $node_info_thread = threads->new(\&nodeInfoThread);
  $image_info_thread = threads->new(\&imageInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $udp_monitor_thread = threads->new(\&udpMonitorThread, $cfg{"SERVER_UDP_MONITOR_DIR"});

  # Sleep for a few seconds to allow the threads to start and collect
  # their first iteration of data
  sleep(3);

  while (!$quit_daemon)
  {
    Dada::logMsg(3, DL, "Selecting handle");
    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 2);

    Dada::logMsg(3, DL, "Select returned");
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

          if    ($string eq "node_info")        { $r = $node_info; }
          elsif ($string eq "img_info")         { $r = $image_string; }
          elsif ($string eq "curr_obs")         { $r = $curr_obs; }
          elsif ($string eq "status_info")      { $r = $status_info; }
          elsif ($string eq "rx_monitor_info")  { $r = $rx_monitor_string; }
          elsif ($string eq "udp_monitor_info") { $r = $udp_monitor_string; }
          elsif ($string eq "mgt_lock_info")    { $r = $mgt_lock_string; }
          else  { Dada::logMsgWarn($warn, "unexpected command: ".$string); } 

          print $rh $r."\n";
          Dada::logMsg(2, DL, "-> ".$r);

          # experimental!
          $read_set->remove($rh);
          close($rh);
        }
      }
    }
  }

  Dada::logMsg(2, DL, "joining control_thread\n");
  # Rejoin our daemon control thread
  $control_thread->join();
  Dada::logMsg(2, DL, "control_thread joined\n");

  # Rejoin other threads
  Dada::logMsg(2, DL, "joining curr_info_thread\n");
  $curr_info_thread->join();
  Dada::logMsg(2, DL, "joining status_info_thread\n");
  $status_info_thread->join();
  Dada::logMsg(2, DL, "joining node_info_thread\n");
  $node_info_thread->join();
  Dada::logMsg(2, DL, "joining image_info_thread\n");
  $image_info_thread->join();
  Dada::logMsg(2, DL, "image_info_thread joined\n");
  #$rx_monitor_thread->join();
  #Dada::logMsg(2, DL, "monitor_thread joined\n");
  $udp_monitor_thread->join();
  Dada::logMsg(2, DL, "udp_monitor_thread joined\n");

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
  if ($sigName ne "PIPE")
  {
    print STDERR $daemon_name." : Received SIG".$sigName."\n";

    if ($quit_daemon)
    {
      print STDERR $daemon_name." : Exiting\n";
      exit(1);
    }
    $quit_daemon = 1;
  }
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
        $tmp_str .= "FREQ:::".$cfg_file{"FREQ"}.";;;";
        $tmp_str .= "BW:::".$cfg_file{"BW"}.";;;";
        $tmp_str .= "NUM_PWC:::".$cfg_file{"NUM_PWC"}.";;;";
        $tmp_str .= "PID:::".$cfg_file{"PID"}.";;;";
        $tmp_str .= "UTC_START:::".$cfg_file{"UTC_START"}.";;;";
        $tmp_str .= "PROC_FILE:::".$cfg_file{"PROC_FILE"}.";;;";
        $tmp_str .= "NANT:::".$cfg_file{"NANT"}.";;;";
        $tmp_str .= "OBSERVER:::".$cfg_file{"OBSERVER"}.";;;";
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
# Maintains informations about monitoring images
#
sub udpMonitorThread ($)
{
  my ($monitor_dir) = @_;

  Dada::logMsg(1, DL, "udpMonitorThread: starting");

  my $img_check_time = 5;
  my $img_clean_time = 10;

  my $img_check_counter = 0;
  my $img_clean_counter = 0;

  my ($cmd, $result, $response);
  my ($time, $pfb, $input, $type, $locked, $res, $ext);
  my ($key, $xres, $yres);

  my @images = ();
  my @image_types = qw(bp ts wf hg);
  my @pfbs = ();

  my $i = 0;
  my %to_use;
  my %locked = ();
  my @keys = ();
  my ($k, $no_image, $xml, $img_file, $img_string, $mgt_lock_xml);
  
  # chdir $monitor_dir;
  while (!$quit_daemon)
  {
    $img_check_counter--;
    $img_clean_counter--;

    if ($img_check_counter > 0)
    {
      sleep(1);
    }
    else
    {
      $img_check_counter = $img_check_time;
      $xml = "";
      $mgt_lock_xml = "";

      # get the listing of image files [UTC_START.PFB_ID.INPUT_ID.PLOT_TPE.XxY.png]
      $cmd = "find ".$monitor_dir." -ignore_readdir_race -name '????-??-??-??:??:??.*.*.??.?.*x*.png' | sort -n | awk -F/ '{print \$(NF)}'";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      %to_use = ();
      %locked = ();
      @pfbs = ();

      for ($i=0; $i<=$#images; $i++)
      {
        $img_file = $images[$i];
        ($time, $pfb, $input, $type, $locked, $res, $ext) = split(/\./, $img_file);
        if (!exists($to_use{$pfb}))
        {
          $to_use{$pfb} = ();
          $locked{$pfb} = ();
          push @pfbs, $pfb;
        }
        if (!exists($to_use{$pfb}{$input}))
        {
          $to_use{$pfb}{$input} = ();
          $locked{$pfb}{$input} = ();
        }
        $to_use{$pfb}{$input}{$type.".".$res} = $img_file;
        $locked{$pfb}{$input} = ($locked eq "L") ? "true" : "false";
      }

      foreach $pfb (keys %to_use)
      {
        $xml .= "<pfb id='".$pfb."'>";
        $mgt_lock_xml .= "<pfb id='".$pfb."'>";
        foreach $input ( keys  %{$to_use{$pfb}})
        {
          $xml .= "<input id='".$input."' locked='".$locked{$pfb}{$input}."'>";
          $mgt_lock_xml .= "<input id='".$input."' locked='".$locked{$pfb}{$input}."'/>";
          foreach $key ( keys  %{$to_use{$pfb}{$input}})
          {
            ($type, $res) = split(/\./, $key);
            ($xres, $yres) = split(/x/, $res);
            $xml .= "<img type='".$type."' width='".$xres."' height='".$yres."'>".$to_use{$pfb}{$input}{$key}."</img>";
          }
          $xml .= "</input>";
        }
        $xml .= "</pfb>\n";
        $mgt_lock_xml .= "</pfb>\n";
      }

      $udp_monitor_string = $xml;
      $mgt_lock_string    = $mgt_lock_xml;
      Dada::logMsg(2, DL, "udpMonitorThread: collected images");
    }

    if ($img_clean_counter <= 0)
    {
      Dada::logMsg(2, DL, "udpMonitorThread: removeOldFiles");
      removeOldFiles($monitor_dir, 0, "png");
      $img_clean_counter = $img_clean_time;
    }
  }

  Dada::logMsg(1, DL, "udpMonitorThread: exiting");
}

#
# Maintains informations about monitoring images
#
sub rxMonitorThread ($)
{
  my ($monitor_dir) = @_;

  Dada::logMsg(1, DL, "rxMonitorThread: starting");

  my $img_check_time = 5;
  my $img_clean_time = 10;

  my $img_check_counter = 0;
  my $img_clean_counter = 0;

  my ($cmd, $result, $response);
  my ($time, $rx, $module, $type, $res, $ext);
  my ($key, $xres, $yres);

  my @bins = ();
  my ($bin_string, $bin);
  my @images = ();

  my $i = 0;
  my (%bins_to_use, %to_use);
  my @keys = ();
  my $k = "";
  my $no_image = "";
  my $xml = "";
  my $img_file = "";
  my $img_string = "";

  # chdir $monitor_dir;

  while (!$quit_daemon)
  {
    $img_check_counter--;
    $img_clean_counter--;

    if ($img_check_counter > 0)
    {
      sleep(1);
    }
    else
    {
      $img_check_counter = $img_check_time;
      $xml = "";

      # find any .bin files in the rx_monitor dir
      $cmd = "find ".$monitor_dir." -ignore_readdir_race -name '2*.???.bin' | sort -n  | awk -F/ '{print \$(NF)}'";
      $bin_string = `$cmd`;
      @bins = split(/\n/, $bin_string);

      %bins_to_use = ();
      # ensure we only process / plot the most recentfiles
      foreach $bin (@bins)
      {
        ($time, $rx, $ext) = split(/\./, $bin);
        $bins_to_use{$rx} = $bin;
      }

      # now plot each of the most recent bin files (only)
      foreach $rx ( keys %bins_to_use )
      {
        $cmd = "mopsr_rxplot -g 200x150 -p hg -p sp ".$monitor_dir."/".$bins_to_use{$rx};
        Dada::logMsg(2, DL, "rxMonitorThread: ".$cmd); 
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, DL, "rxMonitorThread: ".$result." ".$response); 
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "rxMonitor: could not plot ".$bin.": ".$response);
        }
      }

      # delete all the bin files we found with our find command
      removeOldFiles($monitor_dir, 0, "bin");

      # get the listing of image files [UTC_START].[RXID].[MODULEID].[PLOTTYPE][XRES]x[YRES].png
      $cmd = "find ".$monitor_dir." -ignore_readdir_race -name '2*.*.?.??.*x*.png' | sort -n | awk -F/ '{print \$(NF)}'";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      %to_use = ();

      for ($i=0; $i<=$#images; $i++)
      {
        $img_file = $images[$i];
        Dada::logMsg(2, DL, "rxMonitorThread: img_file=".$img_file);
        ($time, $rx, $module, $type, $res, $ext) = split(/\./, $img_file);
        if (!exists($to_use{$rx}))
        {
          $to_use{$rx} = ();
        }
        if (!exists($to_use{$rx}{$module}))
        {
          $to_use{$rx}{$module} = ();
        }
        $to_use{$rx}{$module}{$type.".".$res} = $img_file;
      }

      foreach $rx ( keys %to_use)
      {
        $xml .= "<rx id='".$rx."'>";

        foreach $module ( keys %{$to_use{$rx}})
        {
          $xml .= "<module id='".$module."'>";

          foreach $key ( keys  %{$to_use{$rx}{$module}})
          {
            ($type, $res) = split(/\./, $key);
            ($xres, $yres) = split(/x/, $res);
            $xml .= "<img type='".$type."' width='".$xres."' height='".$yres."'>rx_monitor/".$to_use{$rx}{$module}{$key}."</img>";
          }
          $xml .= "</module>";
        }
        $xml .= "</rx>\n";
      }

      $rx_monitor_string = $xml;

      Dada::logMsg(2, DL, "rxMonitorThread: collected images");
    }

    if ($img_clean_counter <= 0)
    {
      # this 
      Dada::logMsg(2, DL, "rxMonitorThread: removeOldFiles");
      removeOldFiles($monitor_dir, 0, "png");
      $img_clean_counter = $img_clean_time;
    }
  }

  Dada::logMsg(1, DL, "rxMonitorThread: exiting");
}


#
# Maintains a listing of the monitoring images
#
sub imageInfoThread($)
{
  my ($results_dir) = @_;

  Dada::logMsg(1, DL, "imageInfoThread: starting");

  my $img_check_time = 5;
  my $img_clean_time = 10;
  my $img_check_counter = 0;
  my $img_clean_counter = 0;

  my $cmd = "";
  my @images = ();
  my @image_types = qw(bp fl fr ti);
  my $obs = "";

  my $i = 0;

  my ($cmd, $result, $response, $xml);
  my ($time, $type, $res, $ext);
  my ($key, $xres, $yres, $img_file);
  my ($utc_start, $ant);

  my %to_use = ();
  my $img_string = "";
  my @ants = ();

  # chdir $results_dir;

  while (!$quit_daemon)
  {
    $img_check_counter--;
    $img_clean_counter--;

    if ($img_check_counter > 0)
    {
      sleep(1);
    }
    else
    {
      $img_check_counter = $img_check_time;

      # find the most recent observation
      #$cmd = "find ".$results_dir." -mindepth 1 -mindepth 1 -type d -name '????-??-??-??:??:??' -printf '%f\n' | sort -n | tail -n 1";
      $cmd = "ls -1d ".$results_dir."/????-??-??-??:??:?? | awk -F/ '{print \$NF}' | sort -n | tail -n 1";
      Dada::logMsg(2, DL, "imageInfoThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "imageInfoThread: ".$result." ".$response);

      if ($result eq "ok")
      {
        $xml = "";
        $utc_start = $response;

        # check if mode is PSR
        #$cmd = "grep ^MODE ".$results_dir."/".$utc_start."/obs.info | awk '{print \$2}'";
        #Dada::logMsg(2, DL, "imageInfoThread: ".$cmd);
        #($result, $response) = Dada::mySystem($cmd);
        #Dada::logMsg(3, DL, "imageInfoThread: ".$result." ".$response);
        #if (($result eq "ok") && ($response eq "PSR"))
        {
          # get the listing of image files
          $cmd = "find ".$results_dir."/".$utc_start." -ignore_readdir_race -mindepth 1 ".
                 "-maxdepth 1 -name '*.*.??.*x*.png' | sort -n | awk -F/ '{print \$(NF)-1\"\/\"\$(NF)}'";
          $img_string = `$cmd`;
          @images = split(/\n/, $img_string);

          %to_use = ();
          @ants = ();

          for ($i=0; $i<=$#images; $i++)
          {
            ($ant, $img_file) = split(/\//, $images[$i]);
            ($time, $ant, $type, $res, $ext) = split(/\./, $img_file);
            if (!exists($to_use{$ant}))
            {
              $to_use{$ant} = ();
              push @ants, $ant;
            }
            $to_use{$ant}{$type.".".$res} = $img_file;
          }

          foreach $ant ( @ants )
          {
            $xml .= "<ant name='".$ant."'>";
            foreach $key ( keys %{$to_use{$ant}} )
            {
              ($type, $res) = split(/\./, $key);
              ($xres, $yres) = split(/x/, $res);
              $xml .= "<img type='".$type."' width='".$xres."' height='".$yres."'>results/".$utc_start."/".$to_use{$ant}{$key}."</img>";
            }
            $xml .= "</ant>\n";
          }
          $image_string = $xml;

          if ($img_clean_counter <= 0)
          {
            Dada::logMsg(2, DL, "imageInfoThread: removeOldFiles");
            removeOldFiles($results_dir."/".$utc_start, 1, "png");
            $img_clean_counter = $img_clean_time;
          }
          Dada::logMsg(2, DL, "imageInfoThread: collected images ".$xml);
        }
      }
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
  $hosts{$cfg{"SERVER_HOST"}}  = 1;

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


#
# remove old pngs
#
sub removeOldFiles($$$)
{
  my ($dir, $retain_most_recent, $ext) = @_;

  my ($cmd, $img_string, $i, $now);
  my ($time, $rest, $time_unix);
  
  $cmd = "find ".$dir." -ignore_readdir_race -mindepth 1 -maxdepth 1 -name '2*.*.".$ext."' -printf '%f\n' | sort -n";
  Dada::logMsg(2, DL, "removeOldFiles: ".$cmd);

  $img_string = `$cmd`;
  my @images = split(/\n/, $img_string);
  my %to_use = ();

  for ($i=0; $i<=$#images; $i++)
  {
    ($time, $rest) = split(/\./, $images[$i], 2);
    $to_use{$rest} = $images[$i];
  }

  $now = time;

  for ($i=0; $i<=$#images; $i++)
  {
    ($time, $rest) = split(/\./, $images[$i], 2);
    $time_unix = Dada::getUnixTimeLocal($time);

    # if this is not the most recent matching type + res
    if (($to_use{$rest} ne $images[$i]) || (!$retain_most_recent))
    {
      # only delete if > 40 seconds old
      if (($time_unix + 40) < $now)
      {
        Dada::logMsg(3, DL, "removeOldFiles: deleteing ".$dir."/".$images[$i].", duplicate, age=".($now-$time_unix));
        unlink $dir."/".$images[$i];
      }
      else
      {
        Dada::logMsg(3, DL, "removeOldFiles: keeping ".$dir."/".$images[$i].", duplicate, age=".($now-$time_unix));
      }
    }
    else
    {
      Dada::logMsg(3, DL, "removeOldFiles: keeping ".$dir."/".$images[$i].", newest, age=".($now-$time_unix));
    }
  }
}
