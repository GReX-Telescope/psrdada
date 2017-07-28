#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  28 Jan 2009
# 
# This daemons deletes observations slowly from the local disk that 
# have been written to tape in all required locations
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Bpsr;            # DADA Module for configuration options
use strict;          # strict mode (like -Wall)
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use Net::hostent;
use File::Basename;
use Time::HiRes qw(usleep);

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variable Declarations
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $pwc_id : shared;
our $beam : shared;
our %cfg : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$beam = 0;
%cfg = Bpsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;

# Check command line arguments is 1
if ($#ARGV != 0)
{
  usage();
  exit(1);
}
$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (($pwc_id >= 0) &&  ($pwc_id < $cfg{"NUM_PWC"}))
{
  # and matches configured hostname
  if ($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName())
  {
    my %roach = Bpsr::getROACHConfig();
    # determine the relevant PWC based configuration for this script 
    $beam = $roach{"BEAM_".$pwc_id};
  }
  else
  {
    print STDERR "PWC_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "PWC_ID was not a valid integer between 0 and ".($cfg{"NUM_PWC"}-1)."\n";
  usage();
  exit(1);
}

# Sanity check to prevent multiple copies of this daemon running
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);


#
# Main
#
{

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  my $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  logMsg(0, "INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  my $control_thread = threads->new(\&controlThread, $pid_file);

  my $result = "";
  my $response = "";
  my $obs = "";
  my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;

  # Main Loop
  while (!$quit_daemon) 
  {
    ($result, $response, $obs) = findCompletedBeam($archive_dir);

    if (($result eq "ok") && ($obs ne "none"))
    {
      logMsg(1, "INFO", "Deleting Files ".$beam."/".$obs);
      ($result, $response) = deleteCompletedBeam($archive_dir, $obs);
      if ($result ne "ok")
      {
        logMsg(0, "WARN", "Failed to delete ".$obs."/".$beam." ".$response);
        $quit_daemon = 1;
      }
      else
      {
        sleep(1);
      }
    } 
    else
    {
      logMsg(2, "INFO", "Found no beams to delete for ".$beam);

      ($result, $response, $obs) = findDeletedBeam($archive_dir);
      if (($result eq "ok") && ($obs ne "none"))
      {
        logMsg(1, "INFO", "Deleting Dir ".$beam."/".$obs);
        ($result, $response) = deleteDeletedBeam($archive_dir, $obs);
        if ($result ne "ok")
        {
          logMsg(0, "WARN", "Failed to delete ".$obs."/".$beam." ".$response);
          $quit_daemon = 1;
        }
        else
        {
          sleep (1);
        }
      }
    }

    my $counter = 120;
    while ((!$quit_daemon) && ($counter > 0) && ($obs eq "none")) 
    {
      if ($counter == 120)
      {
        logMsg(2, "INFO", "Sleeping ".($counter)." seconds");
      }
      sleep(1);
      $counter--;
    }
  }

  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");

  Dada::nexusLogClose($log_sock);

  exit(0);
}


#
# Find an obs/beam that has the required on.tape.* flags set
#
sub findCompletedBeam($) 
{
  (my $archives_dir) = @_;

  logMsg(2, "INFO", "findCompletedBeam(".$archives_dir.")");

  my $found_obs = 0;
  my $result = "";
  my $response = ""; 
  my $obs = "none";
  my $cmd = "";
  my $i = 0;
  my $k = "";   # obs
  my $source = "";  # source
  my $pid = "";       # first letter of source
  my $file = "";

  my %deleted        = ();    # obs that have a beam.deleted
  my %transferred    = ();    # obs that have a beam.transferred

  my @array = ();

  $cmd = "find ".$archives_dir." -mindepth 2 -maxdepth 2 -type f -name beam.deleted".
         " -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  logMsg(3, "INFO", "findCompletedBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "findCompletedBeam: ".$result." ".$response);
  if ($result eq "ok") {
    @array= split(/\n/, $response);
    for ($i=0; $i<=$#array; $i++) {
      $deleted{$array[$i]} = 1;
    }
  }

  $cmd = "find ".$archives_dir." -mindepth 2 -maxdepth 2 -type f -name beam.transferred".
         " -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  logMsg(3, "INFO", "findCompletedBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "findCompletedBeam: ".$result." ".$response);
  if ($result eq "ok") {
    @array = ();
    @array= split(/\n/, $response);
    for ($i=0; $i<=$#array; $i++) {
      if (!defined($deleted{$array[$i]})) {
        $transferred{$array[$i]} = 1;
      }
    }
  }

  @array = ();

  my @keys = sort keys %transferred;
  my $k = "";
  my $space = 0;

  logMsg(2, "INFO", "findCompletedBeam: ".($#keys+1)." observations to consider");

  for ($i=0; ((!$quit_daemon) && (!$found_obs) && ($i<=$#keys)); $i++)
  {
    $k = $keys[$i];

    # check the type of the observation
    $file = $archives_dir."/".$k."/obs.start";
    logMsg(2, "INFO", "findCompletedBeam: testing for existence: ". $file);
    if (-f $file)
    {
      $cmd = "grep ^SOURCE ".$file." | awk '{print \$2}'";
      logMsg(3, "INFO", "findCompletedBeam: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(3, "INFO", "findCompletedBeam: ".$result." ".$response);

      if ($result ne "ok") {
        logMsg(0, "WARN", "findCompletedBeam could not extract SOURCE from obs.start");
        next;
      } 

      $source = $response;
      chomp $source;

      logMsg(3, "INFO", "findCompletedBeam: SOURCE=".$source);

      $cmd = "grep ^PID ".$file." | awk '{print \$2}'";
      logMsg(3, "INFO", "findCompletedBeam: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(3, "INFO", "findCompletedBeam: ".$result." ".$response);
  
      if ($result ne "ok") {
        logMsg(0, "WARN", "findCompletedBeam: could not extract PID from obs.start");
        next;
      }
      $pid = $response;

      $found_obs = 1;
      if ($found_obs) 
      {
        $cmd = "du -sh ".$archives_dir."/".$k." | awk '{print \$1}'";
        logMsg(2, "INFO", "findCompletedBeam: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(2, "INFO", "findCompletedBeam: ".$result." ".$response);
        $space = $response;
        
        logMsg(2, "INFO", "findCompletedBeam: found ".$k." PID=".$pid.", SIZE=".$space);
        $obs = $k;
      }
    }
    else
    {
      if (-f $archives_dir."/".$k."/obs.deleted")
      {
        $cmd = "touch ".$archives_dir."/".$k."/beam.deleted";
        logMsg(2, "INFO", "findCompletedBeam: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(2, "INFO", "findCompletedBeam: ".$result." ".$response);
      }
      else
      {
        logMsg(1, "INFO", "findCompletedBeam: file did not exist :".$file);
      }
    }
  } 

  $result = "ok";
  $response = "";

  logMsg(2, "INFO", "findCompletedBeam ".$result.", ".$response.", ".$obs);

  return ($result, $response, $obs);
}

#
# Find an obs/beam that has been deleted and is > 1month old
#
sub findDeletedBeam($) 
{
  (my $archives_dir) = @_;

  logMsg(2, "INFO", "findDeletedBeam(".$archives_dir.")");

  my ($cmd, $result, $response);
  my $obs = "none";

  # look for observations that were marked as deleted over 31 days ago
  $cmd = "find ".$archives_dir." -mindepth 2 -maxdepth 2 -type f -name beam.deleted -mtime +31 ".
         " -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort -n | head -n 1";
  logMsg(2, "INFO", "findCompletedBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "findCompletedBeam: ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(3, "INFO", "findCompletedBeam: ".$cmd." failed: ".$response);
    return ("fail", "find command failed");
  }
  else
  {
    if ($response eq "")
    {
      $response = "no beams to delete";
    }
    else
    {
      $obs = $response;  
      $response = "";
    }
    return ("ok", $response, $obs);
  }
}

#
# Delete the specified obs/beam 
#
sub deleteCompletedBeam($$) 
{
  my ($dir, $obs) = @_;

  my $result = "";
  my $response = "";
  my $path = $dir."/".$obs;

  logMsg(2, "INFO", "Deleting archived files in ".$path);

  my $rm_file = "/tmp/".$beam."_".$obs."_slow_rm.ls";

  if (-f $rm_file) {
    unlink $rm_file;
  }

  my $cmd = "find ".$path." -name '*.fil' -o -name '*.psrxml' -o -name '*.ar' ".
            "-o -name '*.tar' -o -name '*.png' -o -name '*.bp*' ".
            "-o -name '*.ts?' -o -name '*.cand' -o -name 'rfi.*' ".
            "-o -name '*.dada' > ".$rm_file;
  logMsg(2, "INFO", $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", $result." ".$response);

  if ($result ne "ok") {
    logMsg(0, "ERROR", "deleteCompletedBeam: find command failed: ".$response);
    return ("fail", "find command failed");
  }

  # get the list of files
  $cmd = "cat ".$rm_file;
  logMsg(2, "INFO", $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", $result." ".$response);
  if (($result eq "ok") && ($response ne ""))
  {
    chomp $response;
    my $files = $response;
    $files =~ s/\n/ /g;
    
    if ($files ne "")
    {
      $cmd = "chmod u+w ".$files;
      logMsg(2, "INFO", $cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", $result." ".$response);
      if ($result ne "ok")
      {
        logMsg(1, "WARN", "failed to chmod u+w files [".$files."]: ".$response);
      }
    }

    $cmd = "slow_rm -r 256 -M ".$rm_file;
    logMsg(2, "INFO", $cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") {
      logMsg(1, "WARN", $result." ".$response);
    }
  }

  if (-d $path."/aux")
  {
    rmdir $path."/aux";
  }

  $cmd = "touch ".$path."/beam.deleted";
  logMsg(2, "INFO", $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", $result." ".$response);

  # If the old style UTC_START/BEAM/obs.deleted existed, change it to a beam.deleted
  if (-f $path."/obs.deleted") {
    unlink $path."/obs.deleted";
  }

  if (-f $rm_file) {
    unlink $rm_file;
  }

  $result = "ok";
  $response = "";

  return ($result, $response);
}


#
# Delete the specified obs/beam 
#
sub deleteDeletedBeam($$) 
{
  my ($dir, $obs) = @_;

  my ($cmd, $result, $response);
  my $path = $dir."/".$obs;

  logMsg(2, "INFO", "Deleting archived files in ".$path);

  if (-d $path)
  {
    $cmd = "rm -rf ".$path;
    logMsg(2, "INFO", "deleteDeletedBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "deleteDeletedBeam: ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "ERROR", "deleteDeletedBeam: ".$cmd." failed: ".$response);
      return ("fail", "could not delete beam ".$obs);
    }
    return ("ok", "");
  }
  else
  {
    return ("fail", $path." did not exist");
  }
}

sub controlThread($) 
{

  (my $pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";
  
  while ((!$quit_daemon) && (! -f $host_quit_file ) && (! -f $pwc_quit_file))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) {

  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "cleaner", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub sigHandle($) {

  my $sigName = shift;
  my $time = Dada::getCurrentDadaTime();
  print STDERR "[".$time."] ".$daemon_name." : Received SIG".$sigName."\n";

  # if we CTRL+C twice, just hard exit
  if ($quit_daemon) {
    print STDERR "[".$time."] ".$daemon_name." : Recevied 2 signals, Exiting\n";
    exit 1;

  # Tell threads to try and quit
  } else {

    $quit_daemon = 1;
    #if ($log_sock) {
    #  close($log_sock);
    #}
  }
}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}

