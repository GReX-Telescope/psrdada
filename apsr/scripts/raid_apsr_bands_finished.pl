#!/usr/bin/env perl

###############################################################################
#
# Waits for all bands to be transferred to RAID disk before inserting 
# observation into archival pipeline
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;

#
# Constants
#
use constant ROOT_DIR       => "/lfs/raid0/apsr";
use constant REQUIRED_HOST  => "caspsr-raid0";
use constant REQUIRED_USER  => "apsr";


#
# Function prototypes
#
sub controlThread($$);
sub good($);

#
# Global variable declarations
#
our $dl : shared;
our $daemon_name : shared;
our $quit_daemon : shared;
our $warn : shared;
our $error : shared;

#
# Global initialization
#
$dl = 1;
$daemon_name = Dada::daemonBaseName(basename($0));
$quit_daemon = 0;

# Autoflush STDOUT
$| = 1;

# Main
{
  my $log_file  = ROOT_DIR."/logs/".$daemon_name.".log";
  my $pid_file  = ROOT_DIR."/control/".$daemon_name.".pid";
  my $quit_file = ROOT_DIR."/control/".$daemon_name.".quit";

  my $src_path  = ROOT_DIR."/finished";
  my $dst_path  = ROOT_DIR."/swin/send";

  $warn         = ROOT_DIR."/logs/".$daemon_name.".warn";
  $error        = ROOT_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;

  my $line = "";
  my $obs = "";
  my $src = "";
  my $pid = "";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @finished = ();
  my @bits = ();

  my $i = 0;
  my $obs_start_file = "";
  my $n_band = 0;
  my $n_transferred = 0;

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

  # main Loop
  while ( !$quit_daemon ) 
  {
    @finished = ();

    # look for all observations in the src_path
    $cmd = "find ".$src_path." -mindepth 3 -maxdepth 3 -type d -printf '\%h/\%f\n'";
    Dada::logMsg(3, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: ".$cmd." failed: ".$response);
    }
    else
    {
      @finished = split(/\n/, $response);
      Dada::logMsg(2, $dl, "main: found ".($#finished+1)." finished observations");

      for ($i=0; $i<=$#finished; $i++)
      {  
        $line = $finished[$i]; 
        @bits = split(/\//, $line);
        if ($#bits < 2)
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }

        $pid = $bits[$#bits-2];
        $src = $bits[$#bits-1];
        $obs = $bits[$#bits];

        Dada::logMsg(2, $dl, "main: processing ".$src."/".$obs);
        Dada::logMsg(3, $dl, "main: pid=".$pid." src=".$src." obs=".$obs);

        # try to find and obs.start file
        $cmd = "find ".$src_path."/".$pid."/".$src."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'obs.start' | head -n 1";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          Dada::logMsg(2, $dl, "main: could not find an obs.start file");
          next;
        }
        $obs_start_file = $response;

        # extract the number of bands
        $cmd = "grep NBAND ".$obs_start_file." | awk '{print \$2}'";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          Dada::logMsgWarn($warn, "could not parse NBAND from ".$obs_start_file);
          next;
        }
        $n_band = $response;
      
        # check if all 16 sub bands have been transferred
        $cmd = "find ".$src_path."/".$pid."/".$src."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'band.transferred' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not count number of band.finished files for ".$src."/".$obs);
          $n_transferred = 0;
          next;
        }
        $n_transferred = $response;

        if ($n_transferred != $n_band)
        {
          Dada::logMsg(2, $dl, "main: only ".$n_transferred." of ".$n_band." bands found for ".$src."/".$obs);
          next;
        }

        if (! -d  $dst_path."/".$pid."/".$src ) 
        {
          $cmd = "mkdir -p ".$dst_path."/".$pid."/".$src;
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "could not create dst dir for ".$src."/".$obs);
            next;
          }
        }

        # if we have reached this point move the observation to the dst_path
        $cmd = "mv ".$src_path."/".$pid."/".$src."/".$obs." ".$dst_path."/".$pid."/".$src."/".$obs;
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "failed to move obseravation to dst_dir for ".$src."/".$obs);
        }
        else
        {
          Dada::logMsg(1, $dl, $pid."/".$src."/".$obs.": finished -> swin/send");
        }
      }
    }

    my $counter = 10;
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

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
}

exit 0;

###############################################################################
#
# Functions
#

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
