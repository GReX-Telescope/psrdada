#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bp_sigproc.pl 
#
# write blocks fan beams into individual sigproc files
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
  print "Usage: ".basename($0)." PROC_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our %ct : shared;
our $localhost : shared;
our $proc_id : shared;
our $db_key : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bp");
%ct = Mopsr::getCornerturnConfig("bp");
$proc_id = -1;
$db_key = "dada";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$proc_id  = $ARGV[0];

# ensure that our proc_id is valid 
if (($proc_id >= 0) &&  ($proc_id < $cfg{"NUM_BP"}))
{
  # and matches configured hostname
  if ($cfg{"BP_".$proc_id} ne Dada::getHostMachineName())
  {
    print STDERR "BP_".$proc_id."[".$cfg{"BP_".$proc_id}."] did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "proc_id was not a valid integer between 0 and ".($cfg{"NUM_BP"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$proc_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".src.log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".pid";

  # this is data stream we will be reading from
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $proc_id, $cfg{"NUM_BP"}, $cfg{"INJECTED_DATA_BLOCK"});

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

  my $bp_tag = sprintf ("BP%02d", $proc_id);
  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $raw_header);
  my ($proc_dir, $ibeam, $key, $split_keys);

  # the beams that this proc_id processes
  my $start_beam = $ct{"BEAM_FIRST_RECV_".$proc_id};
  my $end_beam   = $ct{"BEAM_LAST_RECV_".$proc_id};

  # +1 due to 0-based indexing
  my $nbeam = ($end_beam - $start_beam) + 1;

  # continuously run mopsr_dbsigproc
  while (!$quit_daemon)
  {
    $cmd = "dada_header -t bpsigproc -k ".$db_key;
    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "main: ".$cmd." returned");

    if ($? != 0)
    {
      if ($quit_daemon)
      {
        msg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
        msg(0, "ERROR", "dada_header failed: ".$raw_header);
        $quit_daemon = 1;
      }
    }
    else
    {
      my %header = Dada::headerToHash($raw_header);
      msg (0, "INFO", "UTC_START=".$header{"UTC_START"}." SOURCE=".$header{"SOURCE"}." NCHAN=".$header{"NCHAN"}." NANT=".$header{"NANT"});

      # Fan Beams dont really have a "SOURCE" per se
      $proc_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$bp_tag."/".$header{"UTC_START"}."/FB";
      Dada::mkdirRecursive($proc_dir, 0755);

      # create an obs.header file for the Fan Beam
      my $file = $proc_dir."/obs.header";
      msg(0, "INFO", "main: creating ".$file);
      open (FH,">".$file.".tmp");
      my $k = "";
      foreach $k ( keys %header)
      {
        print FH Dada::headerFormat($k, $header{$k})."\n";
      }
      close FH;
      rename($file.".tmp", $file);

      # transfer this header file to the server
      msg(1, "INFO", "main: transferObsHeader(".$proc_dir.", ".$header{"UTC_START"}.", FB");
      ($result, $response) = transferObsHeader ($proc_dir, $header{"UTC_START"}, "FB");
      if ($result ne "ok")
      {
        msg(0, "WARN", "transferObsHeader failed: ".$response);
      }
      else
      {
        unlink ($proc_dir."/obs.header");
      }

      chdir $proc_dir;

      $cmd = "touch ".$proc_dir."/obs.processing";
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$result." ".$response);
      if ($result ne "ok")
      {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

      # creates a sigproc filterbank file and obs.header in BEAM_### subdirs
      $cmd = "mopsr_dbsigproc -k ".$db_key." ".($start_beam + 1)." ".$ct{"NBEAM"}." -s -z";
      msg(1, "INFO", "START ".$cmd);
      ($result, $response) = Dada::mySystemPiped ($cmd, $src_log_file, $src_log_sock, "src", sprintf("%02d",$proc_id), $daemon_name, "bp_sigproc");
      msg(1, "INFO", "END   ".$cmd);
      if ($result ne "ok")
      {
        $quit_daemon = 1;
        if ($result ne "ok")
        {
          msg(0, "ERROR", $cmd." failed: ".$response);
        }
      }

      $cmd = "mv obs.processing obs.completed";
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$result." ".$response);
      if ($result ne "ok")
      {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

      chdir $cfg{"CLIENT_RECORDING_DIR"};
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


#
# transfers the obs.header file to the server
#
sub transferObsHeader($$$)
{
  my ($local_dir, $utc_start, $source) = @_;
  my ($cmd, $result, $response);

  my $local_file = $local_dir."/obs.header";
  my $remote_file = $utc_start."/".$source."/obs.header.".sprintf("BP%02d", $proc_id);

  msg(2, "INFO", "transferObsHeader: local_file=".$local_file);
  msg(2, "INFO", "transferObsHeader: remote_file=".$remote_file);

  if (-f $local_file)
  {
    my $ntries = 0;
    while ($ntries < 3)
    {
      $cmd = "rsync -a --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
             $local_file." upload\@172.17.228.204::results/".$remote_file;

      msg(2, "INFO", "transferObsHeader: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "transferObsHeader: ".$result." ".$response);
      if ($result ne "ok")
      {
        if ($quit_daemon)
        {
          msg(0, "INFO", "transfer of ".$utc_start." interrupted");
          return ("fail", "transfer interrupted");
        }
        else
        {
          $ntries += 1;
          msg(0, "INFO", "transfer of ".$local_file." failed: ".$response);
          sleep (2);
        }
      }
      else
      {
        return ("ok", "");
      }
    }
    msg(0, "WARN", "transfer of ".$local_file." failed: ".$response);
    return ("fail", "transfer of ".$local_file." failed");

  }
  return ("ok", "file did not exist: ".$local_file);
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_sigproc", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^dada_header -t bpsigproc -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^mopsr_dbsigproc -k ".$db_key;
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

