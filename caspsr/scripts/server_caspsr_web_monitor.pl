#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use Getopt::Std;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use Dada;
use Caspsr;


Dada::preventDuplicateDaemon(basename($0));

#
# Function Prototypes
#
sub main();

#
# Global Variable Declarations
#
our %cfg;
our $dl;
our $daemon_name;
our $quit_daemon : shared;
our $server_sock;
our $server_host;
our $server_port;
our $warn;
our $error;
our $node_info : shared;
our $curr_obs : shared;
our $status_info : shared;
our $image_info : shared;
our $gain_info : shared;
our $archival_info : shared;
our $dish_image : shared;

#
# Gloabl Variable Initialization
#
%cfg = Caspsr::getConfig();
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$warn = "";
$error = "";
$server_sock = 0;
$server_host = 0;
$server_port = 0;
$quit_daemon = 0;
$node_info = "";
$curr_obs = "";
$status_info = "";
$image_info = "";
$gain_info = "";
$archival_info = "";
$dish_image = 0x00;

# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = main();

exit($result);

use constant STATUS_OK    => 0;
use constant STATUS_WARN  => 1;
use constant STATUS_ERROR => 2;


###############################################################################
#
# functions
# 

sub main() {

  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  $server_host = $cfg{"SERVER_HOST"};
  $server_port = $cfg{"SERVER_WEB_MONITOR_PORT"};

  my $control_thread = 0;
  my $node_info_thread = 0;   # Node Disk space, DB Info, Load info [R]
  my $image_info_thread = 0;  # Current images for web display [L]
  my $curr_info_thread = 0;   # Current source information [L]
  my $status_info_thread = 0; # Current status file information [L]
  my $gain_info_thread = 0;   # Current gain information [L]
  my $archival_info_thread = 0;   # Current archival information [L]
  my $dish_image_thread = 0;  # Current image of PKS dish

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $i = 0;
  my $handle = 0;
  my $rh = "";
  my $string = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  if ( -f $warn ) {
    unlink $warn;
  }
  if ( -f $error) { 
    unlink $error; 
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  Dada::logMsg(0, $dl ,"STARTING SCRIPT");
  Dada::logMsg(1, $dl, "Listening on socket: ".$server_host.":".$server_port);

  # start the control thread
  Dada::logMsg(2, $dl ,"starting controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($server_sock);   # add the main socket to the set

  Dada::logMsg(2, $dl, "Waiting for connection on: ".$server_host.":".$server_port);

  # launch the threads
  $curr_info_thread = threads->new(\&currentInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $status_info_thread = threads->new(\&statusInfoThread);
  $node_info_thread = threads->new(\&nodeInfoThread);
  $image_info_thread = threads->new(\&imageInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $archival_info_thread = threads->new(\&archivalInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  #$dish_image_thread = threads->new(\&dishImageThread);
  #$gain_info_thread = threads->new(\&gainInfoThread);

  # Sleep for a few seconds to allow the threads to start and collect
  # their first iteration of data
  sleep(3);

  while (!$quit_daemon) {

    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 2);

    foreach $rh (@$rh_set) {

      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $server_sock) { 

        $handle = $rh->accept();
        $handle->autoflush();
        Dada::logMsg(3, $dl, "Accepting connection");

        # Add this read handle to the set
        $read_set->add($handle); 
        $handle = 0;

      } else {

        $string = Dada::getLine($rh);
  
        if (! defined $string) {
          Dada::logMsg(3, $dl, "Closing a connection");
          $read_set->remove($rh);
          close($rh);

        } else {
          Dada::logMsg(2, $dl, "<- '".$string."'");
          my $r = "";

          if ($string eq "node_info") {
            $r = $node_info; 
          } elsif ($string eq "img_info") { 
            $r = $image_info; 
          } elsif ($string eq "curr_obs") { 
            $r = $curr_obs; 
          } elsif ($string eq "status_info") { 
            $r = $status_info; 
          } elsif ($string eq "machine_info") {
            $r = $status_info.$node_info;
          } elsif ($string eq "archival_info") {
            $r = $archival_info;
          } elsif ($string eq "dish_image") {
            $r = $dish_image;
          } else {
            Dada::logMsgWarn($warn, "unexpected command: ".$string);
          } 

          if ($string eq "dish_image") {
            print $rh $r;
          } else {
            print $rh $r."\r\n";

            if ($dl < 3) {
              $r = substr($r, 0, 80);
            }
            Dada::logMsg(2, $dl, "-> ".$r." ...");
          }
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
  $archival_info_thread->join();
  #$dish_image_thread->join();

  close($server_sock);
                                                                                
  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
                                                                                
  return 0;
}

###############################################################################
#
# maintains the curr_info string
#
sub currentInfoThread($) {

  (my $results_dir) = @_;

  Dada::logMsg(1, $dl, "currentInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;

  my $cmd = "";
  my $obs = "";
  my $tmp_str = "";
  my $source = "";
  my %cfg_file = ();
  my $result = "";
  my $response = "";
  my $P0 = "unknown";
  my $DM = "unknown";
  my $integrated = 0;
  my $snr = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      $sleep_counter--;
      sleep(1);
    
    } else {
      $sleep_counter = $sleep_time;

      $cmd = "find ".$results_dir." -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
      Dada::logMsg(3, $dl, "currentInfoThread: ".$cmd);
      $obs = `$cmd`;
      chomp $obs ;
      Dada::logMsg(3, $dl, "currentInfoThread: ".$obs);

      if (-f $results_dir."/".$obs."/obs.info") {

        $tmp_str = "";
        %cfg_file = Dada::readCFGFile($results_dir."/".$obs."/obs.info"); 

        # Determine the P0 for this source
        $P0 = Dada::getPeriod($cfg_file{"SOURCE"});

        # Determine the DM of this source
        $DM = Dada::getDM($cfg_file{"SOURCE"});
  
        # Determine how much data has been intergrated so far
        $integrated = 0;
        $source = $cfg_file{"SOURCE"};
        $source =~ s/^[JB]//;

        Dada::logMsg(2, $dl, "currentInfoThread: [".$results_dir."/".$obs."/".$source."_t.tot]");

        if (-f $results_dir."/".$obs."/".$source."_t.tot") {
          $cmd = "vap -c length -n ".$results_dir."/".$obs."/".$source."_t.tot | awk '{print \$2}'";
          Dada::logMsg(2, $dl, "currentInfoThread: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(2, $dl, "currentInfoThread: ".$result." ".$response);
          chomp $response;
          if (($result eq "ok") && ($response ne "")) {
            $integrated = sprintf("%5.1f",$response);
          } else {
            $integrated = 0.0;
          }
        }

        Dada::logMsg(2, $dl, "currentInfoThread: [".$results_dir."/".$obs."/".$source."_f.tot]");
        if (-f $results_dir."/".$obs."/".$source."_f.tot") {
          $cmd = "psrstat -j FTp -j'B 256' -c snr ".$results_dir."/".$obs.
                 "/".$source."_t.tot 2>&1 | grep snr= | awk -F= '{print \$2}'";
          Dada::logMsg(2, $dl, "currentInfoThread: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(2, $dl, "currentInfoThread: ".$result." ".$response);
          chomp $response;
          if (($result eq "ok") && ($response ne "")) {
            $snr = sprintf("%5.1f",$response);
          } else {
            $snr = "0.0";
          }
        }

        $tmp_str .= "SOURCE:::".$cfg_file{"SOURCE"}.";;;";
        $tmp_str .= "RA:::".$cfg_file{"RA"}.";;;";
        $tmp_str .= "DEC:::".$cfg_file{"DEC"}.";;;";
        $tmp_str .= "CFREQ:::".$cfg_file{"CFREQ"}.";;;";
        $tmp_str .= "BANDWIDTH:::".$cfg_file{"BANDWIDTH"}.";;;";
        $tmp_str .= "NUM_PWC:::".$cfg_file{"NUM_PWC"}.";;;";
        $tmp_str .= "NPOL:::".$cfg_file{"NPOL"}.";;;";
        $tmp_str .= "NBIT:::".$cfg_file{"NBIT"}.";;;";
        $tmp_str .= "PID:::".$cfg_file{"PID"}.";;;";
        $tmp_str .= "UTC_START:::".$obs.";;;";
        $tmp_str .= "P0:::".$P0.";;;";
        $tmp_str .= "DM:::".$DM.";;;";
        $tmp_str .= "INTEGRATED:::".$integrated.";;;";
        $tmp_str .= "SNR:::".$snr.";;;";
        $tmp_str .= "PROC_FILE:::".$cfg_file{"PROC_FILE"}.";;;";
 
        # update the global variable 
        $curr_obs = $tmp_str;

        Dada::logMsg(3, $dl, "currentInfoThread: ".$curr_obs);
        
      }
    }
  }

  Dada::logMsg(1, $dl, "currentInfoThread: exiting");

}


#
# Maintains a listing of the current images
#
sub imageInfoThread($) {

  my ($results_dir) = @_;

  Dada::logMsg(1, $dl, "imageInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;

  my $cmd = "";
  my $image_string = "";
  my @images = ();
  my $obs = "";
  my $tmp_str = "";
  my $src_list = "";
  my @src_array = ();

  my $i = 0;
  my $k = 0;
  my $img = "";
  my $src = "";
  my %pvfl_lo = ();
  my %pvfl_hi = ();
  my %pvfr_lo = ();
  my %pvfr_hi = ();
  my %pvt_lo = ();
  my %pvt_hi = ();
  my %bp_lo = ();
  my %bp_hi = ();
  my @keys = ();
  my %srcs = ();
  my @parts = ();

  chdir $results_dir;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      $sleep_counter--;
      sleep(1);
    } else {

      $sleep_counter = $sleep_time;

      # get the most recent image directory
      $cmd = "find . -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
      $obs = `$cmd`;
      chomp $obs;

      if ($obs ne "") {

        # get all srcs based on subdirs
        %srcs = ();
        $cmd = "find ".$obs." -mindepth 1 -maxdepth 1 -type d -printf '\%f\n' | sort";
        $src_list = `$cmd`;
        chomp $src_list;

        @src_array = ();
        @src_array = split(/\n/, $src_list);
        for ($i=0; $i<=$#src_array; $i++)
        {
          $srcs{$src_array[$i]} = 1;
        }

        # also check sources based on archives
        $cmd = "find ".$obs." -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' -printf '\%f\n' | awk -F_ '{print \$1}' | sort";
        $src_list = `$cmd`;
        chomp $src_list;

        @src_array = ();
        @src_array = split(/\n/, $src_list);
        for ($i=0; $i<=$#src_array; $i++)
        {
          $srcs{$src_array[$i]} = 1;
        }

        # get all images
        $cmd = "find ".$obs." -name '*.png' | sort";
        $image_string = `$cmd`;
        @images = split(/\n/, $image_string);

        %pvfl_lo = ();
        %pvfl_hi = ();
        %pvfr_lo = ();
        %pvfr_hi = ();
        %pvt_lo = ();
        %pvt_hi = ();
        %bp_lo = ();
        %bp_hi = ();

        for ($i=0; $i<=$#images; $i++) {
          $img = $images[$i];
          @parts = split(/_/,$img);
  
          if ($img =~ m/phase_vs/) {
            $src = $parts[3];
          } else {
            $src = $parts[1];
          }
      
          if (($img =~ m/phase_vs_flux/) && ($img =~ m/240x180/)) { $pvfl_lo{$src} = $img; }
          if (($img =~ m/phase_vs_flux/) && ($img =~ m/200x150/)) { $pvfl_lo{$src} = $img; }
          if (($img =~ m/phase_vs_flux/) && ($img =~ m/1024x768/)) { $pvfl_hi{$src} = $img; }
          if (($img =~ m/phase_vs_freq/) && ($img =~ m/240x180/)) { $pvfr_lo{$src} = $img; }
          if (($img =~ m/phase_vs_freq/) && ($img =~ m/200x150/)) { $pvfr_lo{$src} = $img; }
          if (($img =~ m/phase_vs_freq/) && ($img =~ m/1024x768/)) { $pvfr_hi{$src} = $img; }
          if (($img =~ m/phase_vs_time/) && ($img =~ m/240x180/)) { $pvt_lo{$src} = $img; }
          if (($img =~ m/phase_vs_time/) && ($img =~ m/200x150/)) { $pvt_lo{$src} = $img; }
          if (($img =~ m/phase_vs_time/) && ($img =~ m/1024x768/)) { $pvt_hi{$src} = $img; }
          if (($img =~ m/bandpass/)      && ($img =~ m/240x180/)) { $bp_lo{$src} = $img; }
          if (($img =~ m/bandpass/)      && ($img =~ m/200x150/)) { $bp_lo{$src} = $img; }
          if (($img =~ m/bandpass/)      && ($img =~ m/1024x768/)) { $bp_hi{$src} = $img; }
        }

        @keys = ();
        @keys = sort keys %srcs;
        for ($i=0; $i<=$#keys; $i++) {
          $k = $keys[$i];
          if (! defined $pvfl_lo{$k}) { $pvfl_lo{$k} = "../../../images/blankimage.gif"; }
          if (! defined $pvfl_hi{$k}) { $pvfl_hi{$k} = "../../../images/blankimage.gif"; }
          if (! defined $pvfr_lo{$k}) { $pvfr_lo{$k} = "../../../images/blankimage.gif"; }
          if (! defined $pvfr_hi{$k}) { $pvfr_hi{$k} = "../../../images/blankimage.gif"; }
          if (! defined $pvt_lo{$k})  { $pvt_lo{$k} = "../../../images/blankimage.gif"; }
          if (! defined $pvt_hi{$k})  { $pvt_hi{$k} = "../../../images/blankimage.gif"; }
          if (! defined $bp_lo{$k})   { $bp_lo{$k} = "../../../images/blankimage.gif"; }
          if (! defined $bp_hi{$k})   { $bp_hi{$k} = "../../../images/blankimage.gif"; }
        }

        # now update the global variables for each image type
        $tmp_str = "utc_start:::".$obs.";;;";
        $tmp_str .= "npsrs:::".($#keys+1).";;;";
        for ($i=0; $i<=$#keys; $i++) {
          $k = $keys[$i];
          $tmp_str .= "psr".$i.":::".$k.";;;";
          $tmp_str .= "pvfl_200x150:::".$pvfl_lo{$k}.";;;";
          $tmp_str .= "pvt_200x150:::".$pvt_lo{$k}.";;;";
          $tmp_str .= "pvfr_200x150:::".$pvfr_lo{$k}.";;;";
          $tmp_str .= "bp_200x150:::".$bp_lo{$k}.";;;";
          $tmp_str .= "pvfl_1024x768:::".$pvfl_hi{$k}.";;;";
          $tmp_str .= "pvt_1024x768:::".$pvt_hi{$k}.";;;";
          $tmp_str .= "pvfr_1024x768:::".$pvfr_hi{$k}.";;;";
          $tmp_str .= "bp_1024x768:::".$bp_hi{$k}.";;;";
        }

        $image_info = $tmp_str;
        Dada::logMsg(3, $dl, "imageInfoThread: ".$tmp_str);
      }
    }
  }

  Dada::logMsg(1, $dl, "imageInfoThread: exiting");

}


#
# Monitors the STATUS_DIR for warnings and errors
#
sub statusInfoThread() {

  my $status_dir = $cfg{"STATUS_DIR"};

  Dada::logMsg(1, $dl, "statusInfoThread: starting");

  my @server_daemons = split(/ /,$cfg{"SERVER_DAEMONS"});
  my %clients = ();
  my @files = ();
  my %warnings= ();
  my %errors= ();
  my @arr = ();
  my $sleep_time = 2;
  my $sleep_counter = 0;

  my $i = 0;
  my $host = "";
  my $statuses = "";
  my $file = "";
  my $msg = "";
  my $source = "";
  my $tmp_str = "";
  my $key = "";
  my $value = "";
  my $cmd = "";

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {

      $sleep_counter = $sleep_time;
      %warnings = ();
      %errors = ();
      @files = ();

      $cmd = "ls -1 ".$status_dir;
      $statuses = `$cmd`;
      @files = split(/\n/, $statuses);

      # get the current warnings and errors
      for ($i=0; $i<=$#files; $i++) {
        $file = $files[$i];
        $msg = `tail -n 1 $status_dir/$file`;
        chomp $msg;

        $source = "";
        @arr = ();
        @arr = split(/\./,$file);

        # for pwc, sys and src client errors
        if ($#arr == 2) {
          $source = $arr[1]."_".$arr[0];
        } elsif ($#arr == 1) {
          $source = $arr[0];
        }

        if ($file =~ m/\.warn$/) {
          $warnings{$source} = $msg;
        }
        if ($file =~ m/\.error$/) {
          $errors{$source} = $msg;
        }

      }

      # The results array is now complete, build the response
      # string
      $tmp_str = "";
      while (($key, $value) = each(%warnings)){
        $tmp_str .= $key.":::1:::".$value.";;;;;;";
      }
      while (($key, $value) = each(%errors)){
        $tmp_str .= $key.":::2:::".$value.";;;;;;";
      }
      #$tmp_str .= "\n";

      $status_info = $tmp_str;

      Dada::logMsg(2, $dl, "statusInfoThread: ".$status_info);

    }
  }

  Dada::logMsg(1, $dl, "statusInfoThread: exiting");
}

#
# Maintains information with clients about DB, Disk and load
#
sub nodeInfoThread() {

  Dada::logMsg(1, $dl, "nodeInfoThread: starting");

  my $sleep_time = 2;
  my $sleep_counter = 0;
  my $port = $cfg{"CLIENT_MASTER_PORT"};

  my @machines = ();
  my @results = ();
  my @responses = ();
  my $result = "";
  my $response = "";
  my $tmp_str = "";
  my $i = 0;

  # setup the list of machines that we will poll
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    push(@machines, $cfg{"PWC_".$i});
  }
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    push(@machines, $cfg{"DEMUX_".$i});
  }
  push(@machines, "srv0");
  @machines = Dada::array_unique(@machines);
  @machines = sort @machines;

  my $handle = 0;
  my @bits = ();
  my $db_full;
  my $db_blocks;
  my $stop_obs = 0;
  my $tcs_state_sock = 0;
  my $tcs_sock = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;
                                                                                                          
    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      @results = ();
      @responses = ();

      for ($i=0; $i<=$#machines; $i++) {

        Dada::logMsg(2, $dl, "nodeInfoThread: connecting to ".$machines[$i].":".$port); 
        $handle = Dada::connectToMachine($machines[$i], $port, 0);
        Dada::logMsg(3, $dl, "nodeInfoThread: connection: ".$handle);

        # ensure our file handle is valid
        if (!$handle) {

          $response = "0 0 0;;;0 0;;;0.00,0.00,0.00;;;0.0;;;0.0";
          # check if offline, or scripts not running
          $handle = Dada::connectToMachine($machines[$i], 22, 0);
          if (!$handle) {
            $result = "offline";
          } else {
            $result = "stopped";
            $handle->close();
          }
        } else {
          ($result, $response) = Dada::sendTelnetCommand($handle, "get_status");
          $handle->close();
        }
        if ($result ne "ok") {
          $response = "0 0 0;;;0 0;;;0.00,0.00,0.00;;;0.0;;;0.0";
        }
        Dada::logMsg(2, $dl, "nodeInfoThread: result was: ".$result." ".$response);

        $results[$i] = $result;
        $responses[$i] = $response;

        # check if any of the datablocks was over 50% full
        @bits = split(/;;;/, $response);
        ($db_blocks, $db_full) = split(/ /,$bits[1]);
        if ($machines[$i] =~ m/gpu/) {
          Dada::logMsg(2, $dl, "nodeInfoThread: ".$machines[$i]." db_blocks=".$db_blocks.", db_full=".$db_full);
          if (($db_blocks > 0) && ($db_full > ($db_blocks * 0.5))) {
            Dada::logMsg(1, $dl, $machines[$i]." datablock over 50% full: ".$db_full." / ".$db_blocks);
            $stop_obs = 1;
            Dada::logMsg(1, $dl, $machines[$i]." STOP OBS DISABLED");
            $stop_obs = 0;
          }
        }
      }

      # now set the global string
      $tmp_str = "";
      for ($i=0; $i<=$#machines; $i++) {
        $tmp_str .= $machines[$i].":::".$results[$i].":::".$responses[$i].";;;;;;";
      }
      $node_info = $tmp_str;

      Dada::logMsg(2, $dl, "nodeInfoThread: ".$node_info);

      # if we are trying to stop the current observation due to a full datablock
      if ($stop_obs) {

        $tcs_state_sock = Dada::connectToMachine($cfg{"SERVER_HOST"}, $cfg{"TCS_STATE_INFO_PORT"});
        if (!$tcs_state_sock) {
          Dada::logMsgWarn($warn, "nodeInfoThread: could not connect to TCS Interface state info port");
        } else {

          Dada::logMsg(2, $dl, "nodeInfoThread: TCS <- state");
          print $tcs_state_sock "state\n";
          $result = "ok";
          $response = Dada::getLine($tcs_state_sock);
          Dada::logMsg(2, $dl, "nodeInfoThread: TCS -> ".$result." ".$response);

          if ($result ne "ok") {
            Dada::logMsgWarn($warn, "nodeInfoThread: error response from TCS Interface state info port: ".$response); 
          } else {
            # If we are recording, issue the stop command
            
            if ($response =~ m/^Recording/) {
              $tcs_sock =  Dada::connectToMachine($cfg{"TCS_INTERFACE_HOST"}, $cfg{"TCS_INTERFACE_PORT"});
              if (!$tcs_sock) {
                Dada::logMsgWarn($warn, "nodeInfoThread: could not connect to TCS Interface");
              } else {
                Dada::logMsgWarn($warn, "Stopping Observation due to datablock > 50% full");
                Dada::logMsg(2, $dl, "nodeInfoThread: TCS <- stop");
                ($result, $response) = Dada::sendTelnetCommand($tcs_sock, "stop");
                Dada::logMsg(2, $dl, "nodeInfoThread: TCS -> ".$result." ".$response);
                if ($result ne "ok") {  
                  Dada::logMsgWarn($warn, "nodeInfoThread: error response from TCS Interface: ".$response);
                }
                $tcs_sock->close();
              }
            } elsif ($response =~ m/^Idle/) {
              $stop_obs = 0;
            } elsif ($response =~ m/^Stop/) {
              # we are Stopping or Stopped
            } else {
              Dada::logMsgWarn($warn, "nodeInfoThread: TCS in odd state: ".$response);
            }
          }
          $tcs_state_sock->close();
        }
      }
    }
  }  

  Dada::logMsg(1, $dl, "nodeInfoThread: exiting");

}

sub gainInfoThread() {

  Dada::logMsg(1, $dl, "gainInfoThread: starting");
                                                                                                                                                                          
  my $sleep_time = 4;
  my $sleep_counter = 0;
  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"SERVER_GAIN_REPORT_PORT"};
  my $handle = 0;
  my $result = "";
  my $response = ""; 

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      Dada::logMsg(2, $dl, "gainInfoThread: connecting to ".$host.":".$port);
      $handle = Dada::connectToMachine($host, $port, 0);

      # ensure our file handle is valid
      if (!$handle) {
        $result = "fail";
      } else {
        Dada::logMsg(2, $dl, "gainInfoThread: -> REPORT GAINS");
        print $handle "REPORT GAINS\r\n";
        $response = Dada::getLine($handle);
        if ($response) {
          Dada::logMsg(2, $dl, "gainInfoThread: <- ".$response);
          $result = "ok";
        } else {
          $result = "fail";
        }
        $handle->close();
      } 

      if ($result ne "ok") {
        $response = "50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50";
      } else {
        chomp $response;
      }
      Dada::logMsg(2, $dl, "gainInfoThread: ".$response);
      $gain_info = $response;
    }
  }

  Dada::logMsg(1, $dl, "gainInfoThread: exiting");

}

sub archivalInfoThread() {

  my ($results_dir) = @_;

  Dada::logMsg(1, $dl, "archivalInfoThread: starting");

  my $sleep_time = 60;
  my $sleep_counter = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";

  my %states = ();
  my @lines = ();
  my $obs = "";
  my $file = "";
  my $junk = "";
  my $state = "";
  my $i = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      $cmd = "find ".$results_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.*' | grep -v obs.info | grep -v obs.start | awk -F/ '{print \$(NF-1)\" \"\$NF}' | sort";
      Dada::logMsg(2, $dl, "archivalInfoThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);

      %states = ();
      $states{"failed"} = 0;
      $states{"finished"} = 0;
      $states{"transferred"} = 0;
      $states{"deleted"} = 0;

      if ($result eq "ok") 
      {
        @lines = split(/\n/, $response);
        for ($i=0; $i<=$#lines; $i++)
        {
          ($obs, $file) = split(/ /, $lines[$i], 2);
          ($junk, $state) = split(/\./, $file, 2);

          if (defined $states{$state}) {
            $states{$state} += 1;
          }
        }
      }

      $response = "";
      foreach $state (sort keys %states) {
        $response .= $state.":::".$states{$state}.";;;";
      }

      Dada::logMsg(2, $dl, "archivalInfoThread: ".$response);
      $archival_info = $response;
    }
  }

  Dada::logMsg(1, $dl, "archivalInfoThread: exiting");

}


sub dishImageThread() {
  
  Dada::logMsg(1, $dl, "dishImageThread: starting");

  my $sleep_time = 5;
  my $sleep_counter = 0;
  my $host = "130.155.181.48";
  my $port = "8080";
  my $sock= 0;
  my $result = "";
  my $response = "";
  my $http_header = "";
  my $size = 0;
  my $line = "";
  my @bits = ();
  my $reading_header = 1;
  my $data = 0x00;
  my $bytes_read = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {

      $sleep_counter = $sleep_time;
      Dada::logMsg(2, $dl, "dishImageThread: getting new image");
  
      if (!$sock) { 
        $sock = Dada::connectToMachine($host, $port);
      }

      if (!$sock) {
        $sock = 0;

      } else {
  
        # parameters used to talk to the parkes webcam
        print $sock "GET /images1sif\?random=14_30_10_0.11715700675075003 HTTP/1.0\n";
        print $sock "User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4) Gecko/20070515 Firefox/2.0.0.4\n";
        print $sock "Host: 130.155.181.48\n";
        print $sock "\n\n";

        # read the HTTP header
        $reading_header = 1;
        $http_header = "";
      
        while (($reading_header) && ($line = <$sock>)) {

          # this line indicates the file size
          if ($line =~ m/^Content-Length:/) {
            @bits = split(/: /, $line);
            $size = $bits[1];
            $size =~ s/\n//;
            $size =~ s/\r//;
          }
          if ($line eq "\r\n") {
            $reading_header = 0;
          }
          if ($line =~ m/^Content/) {
            $line =~ s/\n//;
            $line =~ s/\r//;
            $http_header .= $line."\n";
            Dada::logMsg(2, $dl, "dishImageThread: Content line: ".$line);
          }
        }
        $bytes_read = read ($sock, $data, $size);
        $dish_image = $http_header.$data;
        $data = 0;
        $sock->close();
        $sock = 0;
      }
    }
  }

  Dada::logMsg(1, $dl, "dishImageThread: exiting");

}


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


#
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

  $server_sock = new IO::Socket::INET (
    LocalHost => $server_host,
    LocalPort => $server_port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  if (!$server_sock) {
    return ("fail", "Could not create listening socket: ".$server_host.":".$server_port);
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}
