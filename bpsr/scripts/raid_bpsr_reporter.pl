#!/usr/bin/env perl

###############################################################################
#
# Gathers information regarding the BPSR data on the RAID server for reporting
# to interested daemons
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Bpsr;

#
# Constants
#
use constant DATA_DIR         => "/lfs/raid0/bpsr";
use constant META_DIR         => "/lfs/data0/bpsr";
use constant REQUIRED_HOST    => "raid0";
use constant REQUIRED_USER    => "bpsr";

#
# Function prototypes
#
sub getObsInArea($$);
sub controlThread($$);
sub good($);

#
# Global variable declarations
#
our $dl : shared;
our %cfg : shared;
our $daemon_name : shared;
our $quit_daemon : shared;
our $warn : shared;
our $error : shared;
our $swin_send_info : shared;
our $in_perm_info : shared;
our $beam_size_info : shared;

#
# Global initialization
#
$dl = 1;
%cfg = Bpsr::getConfig();
$daemon_name = Dada::daemonBaseName(basename($0));
$quit_daemon = 0;
$swin_send_info = "";
$in_perm_info = "";
$beam_size_info = "";

# Autoflush STDOUT
$| = 1;

# Main
{
  my $log_file      = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file      = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file     = META_DIR."/control/".$daemon_name.".quit";

  my $upload_path        = DATA_DIR."/upload";
  my $swin_send_path     = DATA_DIR."/swin/send";
  my $swin_sent_path     = DATA_DIR."/swin/sent";
  my $prks_archive_path  = DATA_DIR."/parkes/archive";
  my $prks_on_tape_path  = DATA_DIR."/parkes/on_tape";
  my $archived_path      = DATA_DIR."/archived";
  my $perm_path          = DATA_DIR."/perm";

  $warn             = META_DIR."/logs/".$daemon_name.".warn";
  $error            = META_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;
  my $reporter_thread = 0;

  my %sizes = ();
  my %pid_summary = ();
  my @upload = ();
  my @swin_send = ();
  my @perm = ();

  my @keys = ();

  my $pid = "";
  my $obs = "";
  my $beam = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $info = "";

  # quick sanity check
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    exit 1;
  }

  # install signal handles
  $SIG{INT}  = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Auto flush output
  $| = 1;

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # start a thread to handle socket requests
  $reporter_thread = threads->new(\&reporterThread);

  # main Loop
  while ( !$quit_daemon ) 
  {

    Dada::logMsg(2, $dl, "main: updating list of what's where");
    # update the list of observations in each area
    @upload = getObsInArea($upload_path, "d");
    @swin_send = getObsInArea($swin_send_path, "l");
    @perm = getObsInArea($perm_path, "d");

    Dada::logMsg(2, $dl, "main: checking observation sizes");

    # ensure the sizes hash is up to date
    $info = "";
    for ($i=0; (($i<=$#perm) && (!$quit_daemon)); $i++)
    {
      ($pid, $obs, $beam) = split ("/", $perm[$i]);
      if (!exists($sizes{$obs}))
      {
        $cmd = "du -sb ".$perm_path."/".$perm[$i]." | awk '{print \$1}'";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result eq "ok")
        {
          $sizes{$obs} = $response;
        }
        else
        {
          delete $sizes{$obs};
        }
      }
      $info .= $pid."/".$obs."/".$beam."\n";
    }
    $in_perm_info = $info;
    $info = "";

    # do same for uploaded
    for ($i=0; (($i<=$#upload) && (!$quit_daemon)); $i++)
    {
      ($pid, $obs, $beam) = split ("/", $upload[$i]);
      if ((!exists($sizes{$obs})) && (-f $upload_path."/".$upload[$i]."/beam.transferred"))
      {
        $cmd = "du -sb ".$upload_path."/".$upload[$i]." | awk '{print \$1}'";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result eq "ok")
        {
          $sizes{$obs} = $response;
        }
      }
    }

    # now update the upload + swin_send summary data for web interface
    $info = "";

    %pid_summary = ();
    for ($i=0; $i<=$#upload; $i++)
    {
      ($pid, $obs, $beam) = split ("/", $upload[$i]);
      if (exists($sizes{$obs}))
      {
        if (!exists($pid_summary{$pid}))
        {
          $pid_summary{$pid} = 0;
        }
        $pid_summary{$pid} += $sizes{$obs};
      }
    }
    for ($i=0; $i<=$#swin_send; $i++)
    {
      ($pid, $obs, $beam) = split ("/", $swin_send[$i]);
      if (exists($sizes{$obs}))
      {
        if (!exists($pid_summary{$pid}))
        {
          $pid_summary{$pid} = 0;
        }
        $pid_summary{$pid} += $sizes{$obs};
      }
    }
    @keys = keys %pid_summary;
    for ($i=0; $i<=$#keys; $i++)
    {
      $info .= $keys[$i]." ".$pid_summary{$keys[$i]}."\n";
    }
    $swin_send_info = $info;
    Dada::logMsg(2, $dl, "main: swin_send_info=".$swin_send_info);

    # size
    $info = "";
    @keys = keys %sizes;
    for ($i=0; $i<=$#keys; $i++)
    {
      $info .= $keys[$i]." ".$sizes{$keys[$i]}."\n";
    }
    $beam_size_info = $info;
    $info = "";

    my $counter = 60;
    Dada::logMsg(2, $dl, "main: sleeping ".($counter)." seconds");
    while ((!$quit_daemon) && ($counter > 0)) 
    {
      sleep(1);
      $counter--;
    }
  }

  Dada::logMsg(2, $dl, "main: joining threads");
  $control_thread->join();
  Dada::logMsg(2, $dl, "main: control_thread joined");
  $reporter_thread->join();
  Dada::logMsg(2, $dl, "main: reporter_thread joined");

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
}

exit 0;

###############################################################################
#
# Functions
#

sub getObsInArea($$)
{
  (my $path, my $type) = @_;
  Dada::logMsg(2, $dl, "getObsInArea(".$path.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @list = ();
  my @lines = ();
  my @bits = ();
  my $pid = "";
  my $obs = "";
  my $beam = "";
  my $tag = "";
  my $i = 0;

  $cmd = "find ".$path." -mindepth 3 -maxdepth 3 -type ".$type." -printf '\%h/\%f\n' | sort";
  Dada::logMsg(3, $dl, "getObsInArea: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getObsInArea: ".$result." ".$response);
  if ($result eq "ok")
  {
    @lines = split(/\n/, $response);
    for ($i=0; (!$quit_daemon && $i<=$#lines); $i++)
    {
      @bits = split(/\//, $lines[$i]);
      if ($#bits < 5) 
      {
        Dada::logMsgWarn($warn, "getObsInArea: not enough components in path [".$lines[$i]."]");
        next;     
      }
      $pid      = $bits[$#bits-2];
      $obs      = $bits[$#bits-1];
      $beam     = $bits[$#bits];
      $tag = $pid."/".$obs."/".$beam;
      push @list, $tag;
    }
  } 
  return @list;
}

# 
# reporter thread to respond to requests for data on raid server
#
sub reporterThread()
{
  Dada::logMsg(1, $dl, "reporterThread: starting");

  my $server_socket = new IO::Socket::INET (
    LocalHost => $cfg{"RAID_HOST"},
    LocalPort => $cfg{"RAID_WEB_MONITOR_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  die "Could not create listening socket: $!\n" unless $server_socket;

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($server_socket);   # add the main socket to the set
  Dada::logMsg(2, $dl, "reporterThread: waiting for connection on ".$cfg{"RAID_HOST"}.":".$cfg{"RAID_WEB_MONITOR_PORT"});

  my $handle = 0;
  my $rh = 0;
  my $input = "";
  my $r = "";

  while (!$quit_daemon)
  {
    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$rh_set) 
    {

      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $server_socket) 
      {
        $handle = $rh->accept();
        $handle->autoflush();
        Dada::logMsg(2, $dl, "reporterThread: accepting connection");

        # Add this read handle to the set
        $read_set->add($handle);
        $handle = 0;
      } 
      else
      {
        $input = Dada::getLine($rh);
        if (! defined $input)
        {
          Dada::logMsg(2, $dl, "reporterThread: closing a connection");
          $read_set->remove($rh);
          close($rh); 
          $rh = 0;
        } 
        else
        {
          Dada::logMsg(2, $dl, "<- ".$input);
          $r = "";

          if ($input eq "swin_send_info")   { $r = $swin_send_info; }
          elsif ($input eq "in_perm_info")   { $r = $in_perm_info; }
          elsif ($input eq "beam_size_info")   { $r = $beam_size_info; }
          else    { Dada::logMsgWarn($warn, "unexpected command: ".$input); }

          chomp $r;
          print $rh $r."\r\n";
          Dada::logMsg(2, $dl, "-> ".$r);
          print $rh "ok\r\n";
        }
      }
    }
  }

  if ($rh)
  {
    Dada::logMsg(2, $dl, "reporterThread: closing active connection on thread exit");
    close($rh);
  }

  Dada::logMsg(2, $dl, "reporterThread: closing server socket");
  close ($server_socket);

  Dada::logMsg(1, $dl, "reporterThread: exiting");
}

#
# control thread to ask daemon to quit
#
sub controlThread($$) 
{
  my ($quit_file, $pid_file) = @_;
  Dada::logMsg(2, $dl, "controlThread: starting");

  my $cmd = "";
  my $regex = "";
  my $result = "";
  my $response = "";

  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl, "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(2, $dl, "controlThread: exiting");

  return 0;
}

#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # tell threads to try and quit
  if (($sigName ne "INT") || ($quit_daemon))
  {
    $quit_daemon = 1;
    sleep(3);
  
    print STDERR $daemon_name." : Exiting\n";
    exit 1;
  }
  $quit_daemon = 1;
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

  my $host = Dada::getHostMachineName();
  if ($host ne REQUIRED_HOST) {
    return ("fail", "Error: this script can only be run on ".REQUIRED_HOST);
  }

  my $curr_user = `whoami`;
  chomp $curr_user;
  if ($curr_user ne REQUIRED_USER) {
    return ("fail", "Error: this script can only be run as user ".REQUIRED_USER);
  }
  
  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}
