#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bp_send.pl 
#
# Send daemon for the Beam-Former to Beam-Processor cornerturn
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
  print "Usage: ".basename($0)." SEND_ID\n";
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
our $send_id : shared;
our $db_key : shared;
our $log_host;
our $sys_log_port;
our $sys_log_sock;
our $src_log_port;
our $src_log_sock;
our $testing;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bf");            # read the BF config
%ct = Mopsr::getCornerturnConfig("bp");   # read the BP cornerturn
$send_id = -1;
$db_key = "dada";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BF_SYS_LOG_PORT"};
$sys_log_sock = 0;
$src_log_port = $cfg{"SERVER_BF_SRC_LOG_PORT"};
$src_log_sock = 0;
$testing = 0;

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$send_id  = $ARGV[0];

# ensure that our send_id is valid 
if (($send_id >= 0) &&  ($send_id < $ct{"NSEND"}))
{
  # and matches configured hostname
  if ($ct{"SEND_".$send_id} ne Dada::getHostMachineName())
  {
    print STDERR "SEND_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "SEND_ID was not a valid integer between 0 and ".($ct{"NSEND"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$send_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$send_id.".log";
  my $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$send_id.".src.log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$send_id.".pid";

  # this is data stream we will be reading from
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $send_id, $cfg{"NUM_BF"}, $cfg{"FAN_BEAMS_DATA_BLOCK"});

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

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $raw_header, $full_cmd, $sleep_time);

  $sleep_time = 1;

  # continuously run mopsr_dbib for this BF
  while (!$quit_daemon)
  {
    $cmd = "dada_header -k ".$db_key;
    msg(1, "INFO", "main: ".$cmd);
    $raw_header = `$cmd`;
    if ($? != 0)
    {
      if (!$quit_daemon)
      {
        msg(0, "ERROR", $cmd." failed: ".$response);
        $quit_daemon = 1;
        sleep (1);
      }
      else
      {
        msg(0, "INFO", $cmd." failed, but quit_daemon==true");
      }
    }
    else
    {
      my %header = Dada::headerToHash ($raw_header);
      msg (0, "INFO", "UTC_START=".$header{"UTC_START"}." NCHAN=".$header{"NCHAN"}." NANT=".$header{"NANT"});

      open FH, ">/tmp/header.bp_send.".$send_id;
      print FH $raw_header;
      close FH;

      $cmd = "mopsr_dbib_SFT -k ".$db_key." ".$send_id." ".$cfg{"CONFIG_DIR"}."/mopsr_bp_cornerturn.cfg -s";

      msg(1, "INFO", "START ".$cmd);
      ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", sprintf("%02d",$send_id), $daemon_name, "bp_send");
      msg(1, "INFO", "END   ".$cmd." ".$result." ".$response);

      if ($result ne "ok")
      {
        if (!$quit_daemon)
        {
          msg(0, "INFO", $cmd." failed: ".$response);
          $sleep_time += 5;

          # this can occurr when not all receivers are ready
          msg(0, "INFO", "Trying again after ".$sleep_time." seconds");
          sleep ($sleep_time);
        }
        if ($sleep_time > 20)
        {
          msg(0, "ERR", "Failed to create connection to ibdb");
          $quit_daemon = 1;
        }
      }
      else
      {
        $sleep_time = 1;
      }
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$send_id), $time, "sys", $type, "bp_send", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$send_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^dada_header -k ".$db_key;
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(2, "INFO", "controlThread: killProcess() ".$result." ".$response);

  $cmd = "^mopsr_dbib_SFT -k ".$db_key;
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(2, "INFO", "controlThread: killProcess() ".$result." ".$response);

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

