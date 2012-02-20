#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Bpsr;


#
# function prototypes
#
sub good($);
sub checkRemoteScripts($$$);
sub startRemoteScript($$$);
sub stopRemoteScripts($$$);


#
# global variable declarations
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;
our @r_scripts;


#
# global variable initialization
#
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Bpsr::getConfig();
$quit_daemon = 0;
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
@r_scripts = ("bpsr_archiver", "bpsr_beams_finished", "bpsr_swin_transferrer", "bpsr_cleaner");

{

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $control_thread = 0;
  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $r_user = "bpsr";
  my $r_host = "raid0";
  my $r_path = "/lfs/data0/bpsr/control";

  # Every 60 seconds, check for the existence of the script on the remote machine
  my $counter_freq = 60;
  my $counter = 0;
  my $premature_exit = 0;
  my $pipeline_failure = 0;

  my %results = ();
  my @keys = "";
  my $k = "";
  my $i = 0;

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    exit 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl ,"STARTING SCRIPT");

  Dada::logMsg(1, $dl, "Clearing status warn/error files");
  if (-f $warn) {
    unlink($warn); 
  }
  if ( -f $error) {
    unlink($error); 
  }

  # start the control thread
  Dada::logMsg(2, $dl ,"starting controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # Set if a daemon is running there already
  Dada::logMsg(2, $dl, "main: checkRemoteScript(".$r_user.", ".$r_host.", ".$r_path.")");
  %results = checkRemoteScripts($r_user, $r_host, $r_path);

  @keys = keys %results;
  for ($i=0; (($i<=$#keys) && (!$quit_daemon)); $i++)
  {
    $k = $keys[$i];
    if ($results{$k} eq -1)
    {
      Dada::logMsg(0, $dl, "main: ".$k." could not SSH to ".$r_user."@".$r_host);
      $quit_daemon = 1;
      next;
    }
    elsif ($results{$k} eq 0)
    {
      Dada::logMsg(2, $dl, "main: ".$k." not running");
    }
    elsif ($results{$k} eq 1)
    {
      Dada::logMsg(0, $dl, "main: ".$k." not running, but PID file exists");
      Dada::logMsgWarn($warn, $k." not running, but PID file exists");

    }
    elsif ($results{$k} eq 2)
    {
      Dada::logMsg(0, $dl, "main: ".$k." running, but PID file did not exist");
      Dada::logMsgWarn($warn, $k." running, but PID file did not exist");
    }
    elsif ($results{$k} eq 3)
    {
      Dada::logMsg(1, $dl, "main: ".$k." running, and PID exists");
    }
    else
    {
      Dada::logMsg(1, $dl, "main: ".$k." unknown state [".$results{$k}."]");
    }

    if (($results{$k} ge 0) && ($results{$k} le 1))
    {
      Dada::logMsg(2, $dl, "main: startRemoteScript(".$r_user.", ".$r_host.", ".$k.")");
      ($result, $response) = startRemoteScript($r_user, $r_host, $k);
      Dada::logMsg(2, $dl, "main: startRemoteScript() ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "main: startRemoteScript failed for ".$k.": ".$response);
        Dada::logMsgWarn($warn, "failed to start ".$k);
        $quit_daemon = 1;
      }
    }
  }

  $pipeline_failure = 0;

  # Poll for the existence of the control file
  while ((!$quit_daemon) && (!$premature_exit)) 
  {
    Dada::logMsg(3, $dl, "main: Sleeping for quit_daemon");
    sleep(1);

    if ($counter == $counter_freq) 
    {
      $counter = 0;

       # Set if a daemon is running there already
      Dada::logMsg(2, $dl, "main: checkRemoteScript(".$r_user.", ".$r_host.", ".$r_path.")");
      %results = checkRemoteScripts($r_user, $r_host, $r_path);
      @keys = keys %results;
      for ($i=0; (($i<=$#keys) && (!$quit_daemon)); $i++)
      {
        $k = $keys[$i];
        if ($results{$k} eq -1)
        {
          Dada::logMsg(1, $dl, "main: ".$k." could not SSH to ".$r_user."@".$r_host);
        }
        elsif ($results{$k} eq 0)
        {
          Dada::logMsg(0, $dl, "main: ".$k." not running");
        }
        elsif ($results{$k} eq 1)
        {
          Dada::logMsg(0, $dl, "main: ".$k." not running, but PID file exists");
          Dada::logMsgWarn($warn, $k." not running, but PID file exists");

        }
        elsif ($results{$k} eq 2)
        {
          Dada::logMsg(0, $dl, "main: ".$k." running, but PID file did not exist");
          Dada::logMsgWarn($warn, $k." running, but PID file did not exist");
        }
        elsif ($results{$k} eq 3)
        {
          Dada::logMsg(2, $dl, "main: ".$k." running, and PID exists");
        }
        else
        {
          Dada::logMsg(1, $dl, "main: ".$k." unknown state [".$results{$k}."]");
        }

        if (($results{$k} ge 0 ) && ($results{$k} le 2))
        {
          $pipeline_failure = 1;
        }
      }

      if ($pipeline_failure)
      {
        Dada::logMsgWarn($error, "remote script exited unexpectedly");
        $premature_exit = 1;
      }
    } else {
      $counter++;
    }
  }

  Dada::logMsg(2, $dl, "main: stopRemoteScripts(".$r_user.", ".$r_host.", ".$r_path.")");
  ($result, $response) = stopRemoteScripts($r_user, $r_host, $r_path);
  Dada::logMsg(2, $dl, "main: stopRemoteScripts() ".$result." ".$response);

  $quit_daemon = 1;

  # Rejoin our daemon control thread
  $control_thread->join();

  Dada::logMsg(0, $dl ,"STOPPING SCRIPT");

  exit 0;

}

#
# Checks to see if the raid pipeline scripts are running
#
sub checkRemoteScripts($$$) 
{
  my ($user, $host, $control_path) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";
  my $i = 0;
  my $d = "";
  my $daemon_pid = "";
  my $daemon_quit = "";
  my %results = ();

  for ($i=0; $i<=$#r_scripts; $i++)
  {
    $d = $r_scripts[$i];

    $cmd = "pgrep -u bpsr -f '^perl.*raid_".$d.".pl'";
  
    Dada::logMsg(2, $dl, "checkRemoteScript: ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(2, $dl, "checkRemoteScript: ".$result." ".$response);
  
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "checkRemoteScripts: ssh failed: ".$response);
      Dada::logMsgWarn($warn, "could not ssh to ".$user."@".$host);
      $results{$d} = -1;
      next;
    }
    # if the pgrep succeeds, process is running
    if ($rval eq 0)
    {
      $results{$d} = 2;
    }
    else
    {
      $results{$d} = 0;
    }

    # also check for the existence of the PID file
    $cmd = "ls -1 ".$control_path."/".$d.".pid";
     Dada::logMsg(2, $dl, "checkRemoteScript: ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(2, $dl, "checkRemoteScript: ".$result." ".$response);

    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "checkRemoteScripts: ssh failed: ".$response);
      Dada::logMsgWarn($warn, "could not ssh to ".$user."@".$host);
      next;
    }
    # if the pgrep succeeds, process is running
    if ($rval eq 0)
    {
      $results{$d}++;
    }
  }

  return %results;
}


sub startRemoteScript($$$)
{
  my ($user, $host, $script) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";  
  my $response = "";

  $cmd = "raid_".$script.".pl";
  Dada::logMsg(2, $dl, "startRemoteScript: remoteSshCommand(".$user.", ".$host.", ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(2, $dl, "startRemoteScript: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "startRemoteScript: ssh failed: ".$response);
    return ("fail", "ssh to ".$user."@".$host." failed");
  }
  if ($rval != 0)
  {
    Dada::logMsg(0, $dl, "startRemoteScript: failed to start daemon ".$script.": ".$response);
    return ("fail", "could not start ".$script);
  }
  return ("ok", "script started");
}

sub stopRemoteScripts($$$)
{
  my ($user, $host, $control_dir) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";
  my $scripts_stopped = 0;
  my $script = "";
  my $i = 0;

  # ensure all scripts are stopped
  $cmd = "touch";
  for ($i=0; $i<=$#r_scripts; $i++)
  {
    $script = $r_scripts[$i];
    $cmd .= " ".$control_dir."/".$script.".quit";
  }

  # touch the remote quit file
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(2, $dl, "stopRemoteScript: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {   
    Dada::logMsg(0, $dl, "stopRemoteScript: ssh failed: ".$response);
    return ("fail", "ssh to ".$user."@".$host." failed");
  }   
  if ($rval != 0)
  {   
    Dada::logMsg(0, $dl, "stopRemoteScript: failed to touch quit files: ".$response);
    return ("fail", "could not touch quit files");
  }   

  while (!$scripts_stopped)
  {
    $scripts_stopped = 1;
    for ($i=0; $i<=$#r_scripts; $i++)
    {
      $script = $r_scripts[$i];

      $cmd = "pgrep -u bpsr -f '^perl.*raid_".$script.".pl'";
      Dada::logMsg(2, $dl, "stopRemoteScript: ".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(2, $dl, "stopRemoteScript: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "stopRemoteScript: ssh failed: ".$response);
        return ("fail", "ssh to ".$user."@".$host." failed");
      }
      # if rval == 0, pgrep succeeded and process is running
      if ($rval eq 0)
      {
        $scripts_stopped = 0;
        next;
      }
    }
    if (!$scripts_stopped)
    {
      sleep(5);
    }
  }

  # cleanup the quit file and pid files
  $cmd = "rm -f ";
  for ($i=0; $i<=$#r_scripts; $i++)
  {
    $script = $r_scripts[$i];
    $cmd .= " ".$control_dir."/".$script.".quit ".$control_dir."/".$script.".pid";
  }
  Dada::logMsg(2, $dl, "stopRemoteScript: remoteSshCommand(".$user.", ".$host.", ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  Dada::logMsg(2, $dl, "stopRemoteScript: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {   
    Dada::logMsg(0, $dl, "stopRemoteScript: ssh failed: ".$response);
    return ("fail", "ssh to ".$user."@".$host." failed");
  }   
  return ("ok", "script stopped");

}

sub controlThread($$) 
{

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

  Dada::logMsg(1, $dl, "controlThread: exiting");

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  #sleep(3);
  #print STDERR $daemon_name." : Exiting\n";
  #exit 1;
  
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

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}
