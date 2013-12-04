#!/usr/bin/env perl

###############################################################################
#
# Handles observations that have been moved to either sent/swin and/or 
# parkes/on_tape. Once a beams pointings have been moved to th
# survey pointings [source= ^G*]  and archives all other pointings
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

use constant BPSR_USER        => "dada";
use constant BPSR_HOST        => "hipsr-srv0.atnf.csiro.au";
use constant BPSR_PATH        => "/data/bpsr";

use constant SWIN_PROJECTS    => "P456 P630 P786 P813 P848 P789 P855";
use constant PRKS_PROJECTS    => "P630 P682 P743";

#
# Function prototypes
#
sub checkDirEmpty($);
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
  my $log_file      = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file      = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file     = META_DIR."/control/".$daemon_name.".quit";

  my $prks_src_path = DATA_DIR."/parkes/on_tape";
  my $swin_src_path = DATA_DIR."/swin/sent";
  my $dst_path      = DATA_DIR."/archived";
  my $perm_path     = DATA_DIR."/perm";

  $warn             = META_DIR."/logs/".$daemon_name.".warn";
  $error            = META_DIR."/logs/".$daemon_name.".error";

  my $control_thread = 0;

  my $line = "";
  my $obs = "";
  my $pid = "";
  my $beam = "";

  my $src_path = "";
  my $area = "";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $rval = 0;
  my @finished = ();
  my %combined = ();
  my %to_check = ();
  my @to_check_keys = ();
  my @bits = ();

  my $i = 0;
  my $j = 0;
  my $obs_start_file = "";
  my $n_beam = 0;
  my $n_transferred = 0;

  my $curr_time = 0;
  my $path = "";
  my $mtime = 0;

  my $archive = 0;
  my $other_path = "";
  my $other_area = "";

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
    %to_check = ();
    @to_check_keys = ();

    # look for all obs/beams in the src_path
    $cmd = "find ".$swin_src_path." ".$prks_src_path." -mindepth 3 -maxdepth 3 -type l -printf '\%h/\%f\n' | sort";
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
      %combined = ();

      Dada::logMsg(2, $dl, "main: found ".($#finished+1)." on_tape observations");

      for ($i=0; (!$quit_daemon && $i<=$#finished); $i++)
      {
        $line = $finished[$i]; 
        @bits = split(/\//, $line);
        if ($#bits < 5)
        {
          Dada::logMsgWarn($warn, "main: not enough components in path");
          next;
        }
     
        $src_path = ""; 
        for ($j=1; $j<($#bits-4); $j++)
        {
          $src_path .= "/".$bits[$j];
        }
        $area     = $bits[$#bits-4]."/".$bits[$#bits-3];
        $pid      = $bits[$#bits-2];
        $obs      = $bits[$#bits-1];
        $beam     = $bits[$#bits];

        # for multi area projects, setup the "other" area
        $other_path = $swin_src_path;
        $other_area = "swin/sent";
        if ($area eq "swin/sent") 
        {
          $other_path = $prks_src_path;
          $other_area = "parkes/on_tape";
        }

        # count this obs
        if (!exists($combined{$obs."/".$beam}))
        {
          $combined{$obs."/".$beam} = 0;
        }
        $combined{$obs."/".$beam} += 1;

        Dada::logMsg(3, $dl, "main: processing ".$pid."/".$obs."/".$beam." area=".$area." count=".$combined{$obs."/".$beam});

        $archive = 1;

        # check if both sources are required and available
        if ((SWIN_PROJECTS =~ m/$pid/) && (PRKS_PROJECTS =~ m/$pid/) && ($combined{$obs."/".$beam} < 2))
        {
          $archive = 0;
        }
  
        # if we meet the criteria to archive
        if ($archive)
        {
          Dada::logMsg(2, $dl, "main: archive ".$pid."/".$obs." src_path=".$src_path." area=".$area);

          if (!exists($to_check{$pid."/".$obs}))
          {
            $to_check{$pid."/".$obs} = 0;
          }
          $to_check{$pid."/".$obs} += 1;

          # ensure dst pid/obs dir exists
          Dada::logMsg(2, $dl, "main: createDir(".$dst_path."/".$pid."/".$obs.", 0755)");
          ($result, $response) = Dada::createDir($dst_path."/".$pid."/".$obs, 0755);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "could not create dir [".$dst_path."/".$pid."/".$obs."]");
            next;
          }

          # remove any existing flags 
          $cmd = "rm -f ".$src_path."/".$pid."/".$obs."/".$beam."/xfer.complete ".
                          $src_path."/".$pid."/".$obs."/".$beam."/on.tape.parkes";
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);

          $cmd = "mv ".$src_path."/".$area."/".$pid."/".$obs."/".$beam." ".$dst_path."/".$pid."/".$obs."/";
          Dada::logMsg(2, $dl, "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "failed to move beam ".$beam." to ".$dst_path."/".$pid."/".$obs."/");
          }
          else
          {
            if ($combined{$obs."/".$beam} == 2)
            {
              Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." ".$area." + ".$other_area." -> archived");
            }
            else
            {
              Dada::logMsg(1, $dl, $pid."/".$obs."/".$beam." ".$area." -> archived");
            }
          }

          # handle the other area if it exists
          if ($combined{$obs."/".$beam} == 2)
          {
            # delete the soft links and parent dir
            $cmd = "rm -f ".$other_path."/".$pid."/".$obs."/".$beam;
            Dada::logMsg(2, $dl, "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              Dada::logMsg(0, $dl, "main: ".$cmd." failed: ".$response);
            }
          }
          sleep(1); 
        }
        else
        {
          Dada::logMsg(3, $dl, "main: skipping archival of ".$pid."/".$obs."/".$beam);
        }
      }
    }

    # now check if dirs are empty
    @to_check_keys = sort keys %to_check;
    for ($i=0; $i<=$#to_check_keys; $i++)
    {
      ($pid, $obs) = split(/\//, $to_check_keys[$i]);
    
      # check if all the expected beams have now been archived
      my $perm_beams = "";
      my $dst_beams = "";

      $cmd = "find ".$perm_path."/".$pid."/".$obs." -maxdepth 1 -type d -name '??' | wc -l";
      Dada::logMsg(2, $dl, "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "main: ".$result." ".$response);
      if ($result eq "ok")
      {
        $perm_beams = $response;
      }

      $cmd = "find ".$dst_path."/".$pid."/".$obs." -maxdepth 1 -type l -name '??' | wc -l";
      Dada::logMsg(2, $dl, "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "main: ".$result." ".$response);
      if ($result eq "ok")
      {
        $dst_beams = $response;
      }

      # if we have some beams and the number of archived beams eq total beams, touch obs.archived on hipsr server
      if (($perm_beams ne "") && ($perm_beams eq $dst_beams))
      {
        # touch the relevant file in the SERVER_RESULTS_DIR
        $cmd = "touch ".BPSR_PATH."/results/".$obs."/obs.archived";
        Dada::logMsg(2, $dl, "main: ".BPSR_USER."@".BPSR_HOST.":".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand(BPSR_USER, BPSR_HOST, $cmd);
        if (($result ne "ok") || ($rval != 0))
        {
          Dada::logMsg(0, $dl, "main: ".$cmd." failed : ".$response);
        }

        if (SWIN_PROJECTS =~ m/$pid/)
        {
          Dada::logMsg(2, $dl, "main: checkDirEmpty(".$swin_src_path."/".$pid."/".$obs.")");
          ($result, $response) = checkDirEmpty($swin_src_path."/".$pid."/".$obs);
          Dada::logMsg(3, $dl, "main: checkDirEmpty() ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, $dl, "main: checkDirEmpty failed: ".$response);
            next;
          }
        }

        if (PRKS_PROJECTS =~ m/$pid/)
        {
          Dada::logMsg(2, $dl, "main: checkDirEmpty(".$prks_src_path."/".$pid."/".$obs.")");
          ($result, $response) = checkDirEmpty($prks_src_path."/".$pid."/".$obs);
          Dada::logMsg(3, $dl, "main: checkDirEmpty() ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, $dl, "main: checkDirEmpty failed: ".$response);
            next;
          }
        }
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
# check if the directory is empty
#
sub checkDirEmpty($)
{
  (my $dir) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  if (! -d $dir)
  {
    Dada::logMsg(0, $dl, "checkDirEmpty: ".$dir." dir did not exist");
    return ("fail", "dir did not exist")
  }

  # ensure there is nothing in this dir and that 
  $cmd = "find -L ".$dir." -mindepth 1 | wc -l";
  Dada::logMsg(2, $dl, "checkDirEmpty: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkDirEmpty: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "checkDirEmpty: ".$cmd." failed: ".$response);
    return ("fail", "find command faild");
  }

  if ($response eq "0")
  {
    $cmd = "rmdir ".$dir;
    Dada::logMsg(2, $dl, "checkDirEmpty: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkDirEmpty: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "checkDirEmpty: ".$cmd." failed: ".$response);
      return ("fail", "could not delete ".$dir);
    }
  }
  else
  {
    Dada::logMsg(0, $dl, "checkDirEmpty: skipping ".$dir." as not empty");
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
