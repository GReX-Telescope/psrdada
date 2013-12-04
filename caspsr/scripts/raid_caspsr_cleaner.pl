#!/usr/bin/env perl

###############################################################################
#
# Deletes archived CASPSR observations that are > 12 weeks old
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
use constant DATA_DIR         => "/lfs/raid0/caspsr";
use constant META_DIR         => "/lfs/data0/caspsr";
use constant REQUIRED_HOST    => "raid0";
use constant REQUIRED_USER    => "caspsr";

use constant CASPSR_USER      => "dada";
use constant CASPSR_HOST      => "caspsr-srv0.atnf.csiro.au";
use constant CASPSR_PATH      => "/lfs/data0/caspsr";


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
  my $log_file   = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file   = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file  = META_DIR."/control/".$daemon_name.".quit";

  my $timer_path = DATA_DIR."/archived";
  my $atnf_path  = DATA_DIR."/atnf/sent";

  $warn          = META_DIR."/logs/".$daemon_name.".warn";
  $error         = META_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;

  my $line = "";
  my $src = "";
  my $obs = "";
  my $pid = "";
  my $parent_path = "";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my @archived = ();
  my @bits = ();
  my $i = 0;
  my $j = 0;
  my $path = "";
  my $mtime = 0;
  my $curr_time = 0;
  my $delete_obs = 0;

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
    @archived = ();

    # look for all obs in the timer and atnf paths
    $cmd = "find ".$timer_path." ".$atnf_path." -mindepth 3 -maxdepth 3 -printf '\%h/\%f \%T@\n' | sort";
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: ".$cmd." failed: ".$response);
    }
    else
    {
      @archived = split(/\n/, $response);
      Dada::logMsg(2, $dl, "main: found ".($#archived+1)." archived observations");

      for ($i=0; (!$quit_daemon && $i<=$#archived); $i++)
      {
        $line = $archived[$i]; 

        # extract path and time
        @bits = split(/ /, $line, 2);
        if ($#bits != 1) 
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }
        $path  = $bits[0];
        $mtime = int($bits[1]);

        @bits = split(/\//, $path);
        if ($#bits < 3)
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }

        $parent_path = $bits[0];
        for ($j=1; $j<($#bits-2); $j++)
        {
          $parent_path .= "/".$bits[$j];
        }
        $pid = $bits[$#bits-2];
        $src = $bits[$#bits-1];
        $obs = $bits[$#bits];

        # get the current time
        $curr_time = time;
        $delete_obs = 0;

        Dada::logMsg(2, $dl, "main: testing ".$pid."/".$src."/".$obs." age=".($curr_time - $mtime)." s");

        # If 12 weeks old, delete
        if ($curr_time > ($mtime + (12*7*24*60*60)))
        {
          $delete_obs = 1;
        }

        Dada::logMsg(2, $dl, "main: processing ".$parent_path."/".$pid."/".$src.
                             "/".$obs." delete_obs=".$delete_obs);

        # if we are to delete this obs
        if ($delete_obs)
        {
          Dada::logMsg(1, $dl, $pid."/".$src."/".$obs." deleted");

          # delete the directory
          $cmd = "rm -rf ".$parent_path."/".$pid."/".$src."/".$obs;
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "main: ".$cmd." failed: ".$response);
          }

          # check if the parent_path/pid/src directory is empty
          $cmd = "find ".$parent_path."/".$pid."/".$src."/ -mindepth 1 -maxdepth 1 | wc -l";
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if (($result eq "ok") && ($response eq "0"))
          {
            $cmd = "rmdir ".$parent_path."/".$pid."/".$src;
            Dada::logMsg(2, $dl, "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          }
        }
      }
    }

    @archived = ();

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
