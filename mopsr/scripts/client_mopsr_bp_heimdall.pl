#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bp_heimdall.pl 
#
# run multibeam heimdall on the reblocked, 8-bit unsigned ints
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
our $hires;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bp");
$proc_id = -1;
$db_key = "";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
$hires = 0;
if (($cfg{"CONFIG_NAME"} =~ m/320chan/) || ($cfg{"CONFIG_NAME"} =~ m/312chan/))
{
  $hires = 1;
}

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
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $proc_id, $cfg{"NUM_BP"}, $cfg{"HEIMDALL_DATA_BLOCK"});

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

  # need the cornerturn config to know which beams are where
  my %ct = Mopsr::getCornerturnConfig("bp");

  msg (0, "INFO", "STARTING SCRIPT");

  my $bp_tag = sprintf ("BP%02d", $proc_id);
  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $raw_header, $proc_cmd, $proc_dir, $scratch_dir);

  # continuously run mopsr_dbib for this PWC
  while (!$quit_daemon)
  {
    $cmd = "dada_header -k ".$db_key;
    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "main: ".$cmd." returned");

    # by default discard all incoming data
    $proc_cmd = "dada_dbnull -z -s -k ".$db_key;

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
      open FH, ">/tmp/header.heimdall.".$bp_tag;
      print FH $raw_header;
      close FH;

      my %header = Dada::headerToHash($raw_header);
      msg (0, "INFO", "UTC_START=".$header{"UTC_START"}." NCHAN=".$header{"NCHAN"}." NANT=".$header{"NANT"}." NBIT=".$header{"NBIT"});

      # heimdall produces output in the current working directory
      $proc_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$bp_tag."/".$header{"UTC_START"};
      if (!-d $proc_dir)
      {
        mkdir $proc_dir;
      }
  
      #$scratch_dir = $cfg{"CLIENT_SCRATCH_DIR"}."/".$bp_tag;
      #if (!-d $scratch_dir)
      #{ 
      #  mkdir $scratch_dir;
      #}
      #$scratch_dir = $cfg{"CLIENT_SCRATCH_DIR"}."/".$bp_tag."/".$header{"UTC_START"};
      #if (!-d $scratch_dir)
      #{ 
      #  mkdir $scratch_dir;
      #}

      # starting beam for this heimdall instance
      my $start_beam = 1 + $ct{"BEAM_FIRST_RECV_".$proc_id};
  
      if ($hires)
      {
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.25 -boxcar_max 4096 -beam ".$start_beam." -output_dir ".$proc_dir;
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.25 -boxcar_max 2048 -beam ".$start_beam." -coincidencer ".$cfg{"SERVER_HOST"}.":".$cfg{"COINCIDENCER_PORT"};
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.20 -boxcar_max 1024 -beam ".$start_beam." -coincidencer ".$cfg{"SERVER_HOST"}.":".$cfg{"COINCIDENCER_PORT"};
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.20 -boxcar_max 256 -beam ".$start_beam." -coincidencer ".$cfg{"SERVER_HOST"}.":".$cfg{"COINCIDENCER_PORT"};
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 5000 -dm_tol 1.20 -boxcar_max 256 -beam ".$start_beam." -coincidencer ".$cfg{"SERVER_HOST"}.":".$cfg{"COINCIDENCER_PORT"};
        #$proc_cmd = "dada_dbdisk -k ".$db_key." -s -D ".$scratch_dir;
      }
      else
      {
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.05 -boxcar_max 4096 -beam ".$start_beam." -output_dir ".$proc_dir;
        $proc_cmd = "heimdall -k ".$db_key." -gpu_id ".$cfg{"BP_GPU_ID_".$proc_id}." -dm 0 2000 -dm_tol 1.05 -boxcar_max 4096 -beam ".$start_beam." -coincidencer ".$cfg{"SERVER_HOST"}.":".$cfg{"COINCIDENCER_PORT"};
      }

      my ($binary, $junk) = split(/ /,$proc_cmd, 2);
      $cmd = "ls -l ".$cfg{"SCRIPTS_DIR"}."/".$binary;
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "main: ".$cmd.": ".$result." ".$response);

      if (exists($cfg{"BP_CORE_".$proc_id}))
      {
        $cmd = "numactl -C ".$cfg{"BP_CORE_".$proc_id}." ".$proc_cmd;
      }
      else
      {
        $cmd = $proc_cmd;
      }
      msg(1, "INFO", "START ".$cmd);
      ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", sprintf("%02d",$proc_id), $daemon_name, "bp_heimdall");
      msg(1, "INFO", "END   ".$cmd);
      if ($result ne "ok")
      {
        $quit_daemon = 1;
        if ($result ne "ok")
        {
          msg(0, "ERROR", $cmd." failed: ".$response);
        }
      }
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_heimdall", $msg);
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

  $cmd = "^dada_header -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^dada_dbnull -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^heimdall -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(0, "WARN", "controlThread: PID file did not exist on script exit");
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

