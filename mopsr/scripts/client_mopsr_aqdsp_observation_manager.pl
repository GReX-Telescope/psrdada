#!/usr/bin/env perl

# 
# Simple MOPSR processing script
#
#   Runs the antenna splitter on dada datablock
#   Runs the PROC_FILE on each of the output data blocks
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
our $log_port;
our $log_sock;
our $client_logger;

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
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;
$client_logger = "client_mopsr_src_logger.pl";

#
# Local Variable Declarations
#
my $log_file = "";
my $pid_file = "";
my $control_thread = 0;
my $prev_header = "";
my $quit = 0;

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
  if (($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName()) || ($cfg{"PWC_".$pwc_id} eq "localhost"))
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
  my $cmd = "";
  my $result = "";
  my $response = "";

  $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
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

  logMsg(1,"INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  logMsg(2, "INFO", "main: receiving datablock key global=".$db_key);

  my $recv_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
  my $aqdsp_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"AQDSP_DATA_BLOCK"});

  my $curr_raw_header = "";
  my $prev_raw_header = "";
  my %header = ();
  my $aqdsp_thread = 0;

  # Main Loop
  while (!$quit_daemon) 
  {
    %header = ();

    # next header to read from the receiving data_block
    $cmd =  "dada_header -k ".$recv_db_key;
    logMsg(2, "INFO", "main: ".$cmd);
    $curr_raw_header = `$cmd 2>&1`;
    logMsg(2, "INFO", "main: ".$cmd." returned");

    if ($? != 0)
    {
      if ($quit_daemon)
      {
        logMsg(0, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
        logMsg(0, "ERROR", "dada_header failed: ".$curr_raw_header);
        $quit_daemon = 1;
      }
    }
    elsif ($curr_raw_header eq $prev_raw_header)
    {
      logMsg(0, "ERROR", "main: header repeated, jettesioning observation");

      # start null thread to draing the datablock
      my $null_thread = threads->new(\&nullThread, $recv_db_key, "proc");
      $null_thread->join();
      $null_thread = 0;
    }
    else
    {
      %header = Dada::headerToHash($curr_raw_header);

      # now run the GPU AQ DSP pipeline
      $result = aqdspThread($recv_db_key, $aqdsp_db_key);
    }

    $prev_raw_header = $curr_raw_header;  

    if ($quit) {
      $quit_daemon = 1;
    }
  }

  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

  exit(0);
}

sub aqdspThread($$)
{
  (my $in_key, my $out_key) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $cmd = "mopsr_aqdsp -r -o -d ".$cfg{"PWC_GPU_ID_".$pwc_id}." -s ".$in_key." ".$out_key;

  my $full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$pwc_id." aqdsp";

  logMsg(1, "INFO", "START ".$cmd);
  logMsg(2, "INFO", "aqdspThread: ".$cmd);
  ($result, $response) = Dada::mySystem($full_cmd);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "aqdsp thread failed :".$response);
  }
  logMsg(2, "INFO", "aqdspThread: ".$result." ".$response);
  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}


#
# runs a thread to execute dada_dbnull on the specified data block key
#
sub nullThread($$)
{
  (my $db_key, my $tag) = @_;

  my $cmd = "dada_dbnull -s -k ".$db_key;
  my $result = "";
  my $response = "";

  my $full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$pwc_id." null";

  logMsg(1, "INFO", "START ".$cmd);

  logMsg(2, "INFO", "nullThread: ".$full_cmd);
  ($result, $response) = Dada::mySystem($full_cmd);
  logMsg(2, "INFO", "nullThread: ".$result." ".$response);

  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}
  
sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  # Kill the dada_header command
  $cmd = "ps aux | grep -v grep | grep ".$cfg{"USER"}." | grep 'dada_header -k ".$db_key."' | awk '{print \$2}'";

  logMsg(2, "INFO", "controlThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  $response =~ s/\n/ /g;
  logMsg(2, "INFO", "controlThread: ".$result." ".$response);

  if (($result eq "ok") && ($response ne "")) 
  {
    $cmd = "kill -KILL ".$response;
    logMsg(1, "INFO", "controlThread: Killing dada_header -k ".$db_key.": ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(2, "INFO", "controlThread: ".$result." ".$response);
  }

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) {

  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "obs mngr", $msg);
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
    if ($log_sock) {
      close($log_sock);
    }
  }
}

sub sigPipeHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}

