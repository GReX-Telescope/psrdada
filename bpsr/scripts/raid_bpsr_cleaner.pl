#!/usr/bin/env perl

###############################################################################
#
# Deletes archived BPSR P630 observations that are > 1 week old
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
use constant DATA_DIR         => "/lfs/raid0/bpsr";
use constant META_DIR         => "/lfs/data0/bpsr";
use constant REQUIRED_HOST    => "raid0";
use constant REQUIRED_USER    => "bpsr";


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

  my $src_path   = DATA_DIR."/archived";

  $warn          = META_DIR."/logs/".$daemon_name.".warn";
  $error         = META_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;

  my $line = "";
  my $obs = "";
  my $pid = "";
  my $beam = "";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @finished = ();
  my @bits = ();
  my $source = "";
  my %sources = ();
  my $obs_start_file = "";
  my $i = 0;
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
    @finished = ();

    # look for all obs/beams in the src_path
    $cmd = "find ".$src_path." -mindepth 3 -maxdepth 3 -type d -printf '\%h/\%f \%T@\n' | sort";
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: ".$cmd." failed: ".$response);
    }
    else
    {
      @finished = split(/\n/, $response);
      Dada::logMsg(2, $dl, "main: found ".($#finished+1)." archived observations");

      for ($i=0; (!$quit_daemon && $i<=$#finished); $i++)
      {
        $line = $finished[$i]; 

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

        $pid  = $bits[$#bits-2];
        $obs  = $bits[$#bits-1];
        $beam = $bits[$#bits];

        if (exists($sources{$obs}))
        {
          $source = $sources{$obs};
        }
        else
        {
          # try to find and obs.start file
          $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'obs.start' | head -n 1";
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if (($result ne "ok") || ($response eq ""))
          {
            Dada::logMsg(2, $dl, "main: could not find and obs.start file");
            next;
          }
          $obs_start_file = $response;

          # determine the source
          $cmd = "grep SOURCE ".$obs_start_file." | awk '{print \$2}'";
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if (($result ne "ok") || ($response eq ""))
          {
            Dada::logMsgWarn($warn, "could not extact SOURCE from ".$obs_start_file);
            next;
          }
          $source = $response;
          $sources{$obs} = $source;
        }

        # get the current time
        $curr_time = time;
        $delete_obs = 0;

        Dada::logMsg(2, $dl, "main: testing ".$pid."/".$obs."/".$beam." source=".$source." age=".($curr_time - $mtime)." s");

        # If P630 + Survey Pointing + > 3 days old, delete
        if (($pid eq "P630") && ($source =~ m/^G/) && ($curr_time > ($mtime + (3*24*60*60))))
        {
          $delete_obs = 1;
        }

        # If P630 + > 7 days old, delete
        if( ($pid eq "P630") && ($curr_time > ($mtime + (7*24*60*60))))
        {
          $delete_obs = 1;
        }

        # If special control file exists, delete!
        if (-f  $src_path."/".$pid."/".$obs."/".$beam."/raid.delete") 
        {
          $delete_obs = 1;
        }

        Dada::logMsg(2, $dl, "main: processing ".$pid."/".$obs."/".$beam." delete_obs=".$delete_obs);

        # if this is a P630 observation and more than 1 week old, delete it
        if ($delete_obs)
        {
          Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." deleted [".$source."]");

          $cmd = "rm -rf ".$src_path."/".$pid."/".$obs."/".$beam;
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);

          # check if the src_path/pid/obs directory is empty
          $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 1 -maxdepth 1 -type d -name '??' | wc -l";
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if (($result eq "ok") && ($response eq "0")) 
          {
            $cmd = "rmdir ".$src_path."/".$pid."/".$obs;
            Dada::logMsg(2, $dl, "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          }
        }
        #sleep(1); 
      }
    }

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
