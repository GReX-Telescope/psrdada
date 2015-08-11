#!/usr/bin/env perl

# 
# MOPSR AQ processing script
#
#   Runs the AQ DSP pipeline on each input stream
# 
# Author:   Andrew Jameson
# 

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Mopsr;          # DADA Module for configuration options
use strict;          # strict mode (like -Wall)
use File::Basename; 
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;

sub aqdspSrcLogger($);

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
our $db_key : shared;
our %cfg : shared;
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
$pwc_id = 0;
$db_key = "";
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

#
# Local Variable Declarations
#
my $log_file = "";
my $pid_file = "";
my $control_thread = 0;
my $prev_header = "";

#
# Check command line arguments is 1
#
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
    # determine the relevant PWC based configuration for this script 
    $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
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


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd_file, $proc_cmd);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
  $pid_file     = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  msg(1,"INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  msg(2, "INFO", "main: receiving datablock key global=".$db_key);

  my $recv_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
  my $send_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"TRANSMIT_DATA_BLOCK"});

  my $curr_raw_header = "";
  my $prev_raw_header = "";
  my %header = ();

  # Main Loop
  while (!$quit_daemon) 
  {
    %header = ();

    # next header to read from the receiving data_block
    $cmd =  "dada_header -k ".$recv_db_key;
    msg(2, "INFO", "main: ".$cmd);
    $curr_raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "main: ".$cmd." returned");

    # by default, discard the observation unless config is valid
    $proc_cmd = "dada_dbnull -z -s -k <IN_DADA_KEY>";

    if ($? != 0)
    {
      if ($quit_daemon)
      {
        msg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
        msg(0, "ERROR", "dada_header failed: ".$curr_raw_header);
        $quit_daemon = 1;
      }
    }
    else
    {
      %header = Dada::headerToHash($curr_raw_header);

      if (exists($header{"AQ_PROC_FILE"}))
      {
        $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$header{"AQ_PROC_FILE"};

        msg(2, "INFO", "Full path to AQ_PROC_FILE: ".$proc_cmd_file);
        if ( ! ( -f $proc_cmd_file ) )
        {
          msg(0, "ERROR", "AQ_PROC_FILE did not exist: ".$proc_cmd_file);
        }
        else
        {
          msg(1, "INFO", "AQ_PROC_FILE=".$proc_cmd_file);
          my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
          $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
          msg(1, "INFO", "PROC_CMD=".$proc_cmd);
        }
      }

      $proc_cmd =~ s/<IN_DADA_KEY>/$recv_db_key/;

      $proc_cmd =~ s/<OUT_DADA_KEY>/$send_db_key/;

      $proc_cmd =~ s/<BAYS_FILE>/$cfg{"MOLONGLO_BAYS_FILE"}/;

      $proc_cmd =~ s/<MODULES_FILE>/$cfg{"MOLONGLO_MODULES_FILE"}/;

      $proc_cmd =~ s/<SIGNAL_PATHS_FILE>/$cfg{"MOLONGLO_SIGNAL_PATHS_FILE"}/;

      $proc_cmd =~ s/<DADA_GPU_ID>/$cfg{"PWC_GPU_ID_".$pwc_id}/;

      if (exists($header{"OBSERVING_TYPE"}))
      {
        if ($header{"OBSERVING_TYPE"} eq "TRACKING")
        {
          $proc_cmd .= " -t";
        }
        if ($header{"OBSERVING_TYPE"} eq "STATIONARY")
        {
          $proc_cmd .= " -g";
        }
      }

      if ($curr_raw_header eq $prev_raw_header)
      {
        msg(0, "ERROR", "main: header repeated, jettesioning observation");
        $proc_cmd = "dada_dbnull -k ".$recv_db_key." -s -z";
      }

      # create a local dir for the PFB_ID 
      my $local_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$header{"PFB_ID"}."/".$header{"UTC_START"};
      msg(2, "INFO", "mkdirRecursive(".$local_dir.", 0755)");
      ($result, $response) = Dada::mkdirRecursive($local_dir, 0755);
      msg(3, "INFO", $result." ".$response);
      if ($result ne "ok")
      {
        return ("fail", "Could not create local dir: ".$response);
      }

      open FH, ">".$local_dir."/obs.header";
      print FH $curr_raw_header;
      close FH;

      msg(1, "INFO", "START ".$proc_cmd);
      ($result, $response) = Dada::mySystemPiped ($proc_cmd, $src_log_file, 
                                                  $src_log_sock, "src", sprintf("%02d",$pwc_id), 
                                                  $daemon_name, "aqdsp");

      if ($result ne "ok")
      {
        msg(1, "WARN", "cmd failed: ".$response);
      }
      msg(1, "INFO", "END   ".$proc_cmd);
    }

    $prev_raw_header = $curr_raw_header;  
  }

  msg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}

sub controlThread($) 
{
  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  $cmd = "^dada_header -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^mopsr_aqdsp ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) 
  {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");
}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$pwc_id), $time, "sys", $type, "aqdsp_mgr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
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

