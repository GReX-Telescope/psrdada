#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# 
# 
###############################################################################

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Mopsr;
use strict;
use threads;
use threads::shared;


sub usage() 
{
  print "Usage: ".basename($0)." BS_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $localhost : shared;
our $bs_id : shared;
our $binary : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;
our $bin_dir;
our $srv_ip;
our $bw_limit;
our $bs_tag;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bs");
$bs_id = -1;
$binary = "";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BS_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BS_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
$bin_dir = "/home/vivek/software/linux_64/bin";
$srv_ip = "172.17.228.204";
$bw_limit = int((64 * 1024) / $cfg{"NUM_BS"});

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$bs_id  = $ARGV[0];

# ensure that our bs_id is valid 
if (($bs_id >= 0) &&  ($bs_id < $cfg{"NUM_BS"}))
{
  # and matches configured hostname
  if ($cfg{"BS_".$bs_id} ne Dada::getHostMachineName())
  {
    print STDERR "BS_".$bs_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "bs_id was not a valid integer between 0 and ".($cfg{"NUM_BS"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$bs_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bs_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bs_id.".src.log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bs_id.".pid";

  # this is data stream we will be reading from

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  msg (0, "INFO", "STARTING SCRIPT");

  $bs_tag = sprintf("BS%02d", $bs_id);

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $raw_header, $cand_file, $proc_cmd);
  my (@parts, $n, $utc_start, $proc_dir);

  while (!$quit_daemon)
  {
    $cmd = "find ".$cfg{"CLIENT_RESULTS_DIR"}." -mindepth 2 -maxdepth 2 -type f -name 'obs.finished' | sort -n -r | tail -n 1";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      # get the observation UTC_START
      @parts = split (/\//, $response);
      $n = $#parts;

      $utc_start = $parts[$n-1];
      $proc_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start;
      
      msg(2, "INFO", "main: utc_start=".$utc_start);

      if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
      {
        ($result, $response) = transferSmirf($utc_start);
        if ($result eq "ok")
        {
          $cmd = "rm -rf ".$proc_dir."/*";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            msg(0, "ERROR", "main: failed to delete ".$proc_dir."/*: ".$response);
            $quit_daemon = 1;
          }

          $cmd = "touch ".$proc_dir."/obs.transferred";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            msg(0, "ERROR", "main: failed to touch ".$proc_dir."/obs.transferred: ".$response);
            $quit_daemon = 1;
          }
        }
      }
    }

    my $counter = 10;
    while (!$quit_daemon && $counter > 0)
    {
      sleep(1);
      $counter --;
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($sys_log_sock);

  exit (0);
}


sub transferSmirf ($)
{
  my ($utc_start) = @_;

  my ($cmd, $result, $response);

  my $local_file = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start."/\*";

  # have subdirs that contain folded archives 
  $cmd = "rsync ".$local_file." -a --stats --bwlimit=".$bw_limit." --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
         " upload\@".$srv_ip."::smirf/".$utc_start."/".$bs_tag;
  msg(1, "INFO", "transferSmirf: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(1, "INFO", "transferSmirf: ".$result." ".$response);

  if ($result ne "ok")
  {
    if ($quit_daemon)
    {
      msg(0, "INFO", "transfer of ".$utc_start." interrupted");
      return ("fail", "transfer interrupted");
    }
    else
    {
      msg(0, "WARN", "transfer of ".$local_file." failed: ".$response);
      return ("fail", "transfer of ".$local_file." failed");
    }
  }
  return ("ok", "");
}

#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$)
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$bs_id), $time, "sys", $type, "bs_dspsr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bs_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^rsync ".$cfg{"CLIENT_RESULTS_DIR"};
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");

}

sub sigHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

  # if we CTRL+C twice, just hard exit
  if ($quit_daemon) {
    print STDERR $daemon_name." : Recevied 2 signals, Exiting\n";
    exit 1;

  # Tell threads to try and quit
  } else {

    $quit_daemon = 1;
    if ($sys_log_sock) {
      close($sys_log_sock);
    }
  }
}

sub sigPipeHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $sys_log_sock = 0;
  if ($log_host && $sys_log_port) {
    $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  }
}

