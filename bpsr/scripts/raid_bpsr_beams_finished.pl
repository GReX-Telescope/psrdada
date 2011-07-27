#!/usr/bin/env perl

###############################################################################
#
# Waits for all beams to be transferred to RAID disk before inserting 
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
use constant ROOT_DIR         => "/lfs/raid0/bpsr";
use constant REQUIRED_HOST    => "caspsr-raid0";
use constant REQUIRED_USER    => "bpsr";
use constant SWIN_PROJECTS    => "P630";
use constant PRKS_PROJECTS    => "P630 P682 P743 P786";


#
# Function prototypes
#
sub touchBeamsXferComplete($$$);
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
  my $log_file       = ROOT_DIR."/logs/".$daemon_name.".log";
  my $pid_file       = ROOT_DIR."/control/".$daemon_name.".pid";
  my $quit_file      = ROOT_DIR."/control/".$daemon_name.".quit";

  my $src_path       = ROOT_DIR."/finished";
  my $swin_path      = ROOT_DIR."/swin/send";
  my $prks_path      = ROOT_DIR."/parkes/archive";
  my $archived_path  = ROOT_DIR."/archived";
  my $dst_path       = "";
  my $dst            = "";

  $warn              = ROOT_DIR."/logs/".$daemon_name.".warn";
  $error             = ROOT_DIR."/logs/".$daemon_name.".error";

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
  my $n_beam = 0;
  my $n_transferred = 0;

  my $apsr_user = "dada";
  my $apsr_host = "apsr-srv0.atnf.csiro.au";
  my $rval = 0;

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
    $cmd = "find ".$src_path." -mindepth 2 -maxdepth 2 -type d -printf '\%h/\%f\n' | sort";
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

      for ($i=0; (!$quit_daemon && $i<=$#finished); $i++)
      {  
        $line = $finished[$i]; 
        @bits = split(/\//, $line);
        if ($#bits < 1)
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }

        $pid = $bits[$#bits-1];
        $obs = $bits[$#bits];

        Dada::logMsg(2, $dl, "main: processing ".$pid."/".$obs);
        Dada::logMsg(3, $dl, "main: pid=".$pid." obs=".$obs);

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

        # determine the number of beams
        $cmd = "find /export/archives/bpsr/".$obs." -mindepth 1 -maxdepth 1 -type l -name '??' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($apsr_user, $apsr_host, $cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "ssh failure to ".$apsr_user."@".$apsr_host.": ".$response);
          sleep(5);
          next;
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "could not count beam dirs in /export/archives/bpsr/".$obs);
          next;
        }
        $n_beam = $response;
        if (($n_beam < 1) || ($n_beam > 13))
        {
          Dada::logMsgWarn($warn, "bad value for beam dirs in /export/archives/bpsr/".$obs);
          next;
        }
      
        # check if all 16 sub beams have been transferred
        $cmd = "find ".$src_path."/".$pid."/".$obs." -mindepth 2 -maxdepth 2 -type f -name 'beam.transferred' | wc -l";
        Dada::logMsg(3, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "could not count number of beam.transferred files for ".$obs);
          $n_transferred = 0;
          next;
        }
        $n_transferred = $response;

        if ($n_transferred != $n_beam)
        {
          Dada::logMsg(2, $dl, "main: only ".$n_transferred." of ".$n_beam." beams found for ".$obs);
          next;
        }

        # select the destination based on the project ID
        if (SWIN_PROJECTS =~ m/$pid/)
        {
          $dst_path = $swin_path;
          $dst = "swin/send";
        }
        elsif (PRKS_PROJECTS =~ m/$pid/)
        {
          $dst_path = $prks_path;
          $dst = "parkes/archive";
        }
        else
        {
          $dst_path = $archived_path;
          $dst = "archived";
        }

        Dada::logMsg(2, $dl, "main: dst_path=".$dst_path);

        if (! -d  $dst_path."/".$pid ) 
        {
          $cmd = "mkdir -m 0755 -p ".$dst_path."/".$pid;
          Dada::logMsg(3, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "could not create dst dir [".$dst_path."/".$pid."] for ".$obs);
            $quit_daemon = 1;
            next;
          }
        }

        # remove any existing sent.to.* flags or beam.* flags
        $cmd = "rm -f ".$src_path."/".$pid."/".$obs."/??/sent.to.* ".$src_path."/".$pid."/".$obs."/??/beam.*";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);

        # if we have reached this point move the observation to the dst_path
        $cmd = "mv ".$src_path."/".$pid."/".$obs." ".$dst_path."/".$pid;
        Dada::logMsg(2, $dl, "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "failed to move obseravation to dst_dir [".$dst_path."/".$pid."] for ".$obs);
        }
        else
        {
          Dada::logMsg(1, $dl, $pid."/".$obs.": finished -> ".$dst);
        }

        # if we are going direct to the parkes archival path, touch xfer.complete
        if ($dst_path eq $prks_path)
        {
          Dada::logMsg(2, $dl, "main: touchBeamsXferComplete(".$dst_path.", ".$pid.", ".$obs.")");
          ($result, $response) = touchBeamsXferComplete($dst_path, $pid, $obs);
          Dada::logMsg(2, $dl, "main: touchBeamsXferComplete ".$result." ".$response);
        }
        sleep(1);
      }
    }

    my $counter = 30;
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
# touch xfer.complete in all the beam subdirectories
#
sub touchBeamsXferComplete($$$)
{
  my ($path, $pid, $obs) = @_;
  
  my $cmd = "";
  my $result = "";
  my $response = "";

  # get the list of beams
  $cmd = "find ".$path."/".$pid."/".$obs." -mindepth 1 -maxdepth 1 -type d -name '??' -printf '\%f\n'";
  Dada::logMsg(3, $dl, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "main: ".$result." ".$response);
  if (($result ne "ok") || ($response eq ""))
  {
    Dada::logMsgWarn($warn, "touchBeamsXferComplete: could get beam list: ".$response);
    return ("fail", "could not get beam list");
  }

  my @beams = split(/\n/, $response);
  my $i = 0;
  my $beam = "";
  for ($i=0; $i<=$#beams; $i++)
  {
    $beam = $beams[$i];
    $cmd = "touch ".$path."/".$pid."/".$obs."/".$beam."/xfer.complete";
    Dada::logMsg(3, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);
    if ($result ne "ok")
    {      
      Dada::logMsgWarn($warn, "touchBeamsXferComplete: ".$cmd." failed: "..$response);
    }
  }
  return ("ok", "");
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
