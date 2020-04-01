#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2014 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_aq_diskdb.pl 
#
# load a preconfigured file into the datablock
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


sub aq_diskdbSrcLogger($);
sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $pwc_id : shared;
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
%cfg = Mopsr::getConfig();
$pwc_id = -1;
$db_key = "dada";
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_SRC_LOG_PORT"};
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

$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (($pwc_id >= 0) &&  ($pwc_id < $cfg{"NUM_PWC"}))
{
  # and matches configured hostname
  if ($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName())
  {
    $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
  }
  else
  {
    print STDERR "PWC_".$pwc_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "pwc_id was not a valid integer between 0 and ".($cfg{"NUM_PWC"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);

###############################################################################
#
# Main
#
{
  my ($cmd, $result, $response, $pid_file);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
  $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

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

  my ($pfb_id, $cmd, $result, $response, $file, $proc_cmd, $valid_data, $byte_offset);

  # ID tag for this PFB
  $pfb_id = $cfg{"PWC_PFB_ID_".$pwc_id};

  my $utc = "2018-05-28-04:19:19";
  my @byte_offsets = ( "0000140849996800" );

  $cmd = "dada_diskdb -k ".$db_key." -s";
  my $valid_data = 0;
  foreach $byte_offset ( @byte_offsets )
  {
    $file = "/data/mopsr/diskdb/".$pfb_id."/".$utc."_".$byte_offset.".000000.dada";
    if ( -f $file )
    {
      $cmd .= " -f ".$file;
      $valid_data = 1;
    }
  }

  if ($valid_data)
  {
    msg(1, "INFO", "START ".$cmd);
    ($result, $response) = Dada::mySystemPiped ($cmd, $src_log_file, $src_log_sock, "src", sprintf("%02d",$pwc_id), $daemon_name, "aqdsp");
    if ($result ne "ok")
    {
      msg(1, "WARN", "cmd failed: ".$response);
    }
    msg(1, "INFO", "END   ".$cmd);
  }
  else
  {
    msg(1, "WARN", "no .dada files existed");
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$pwc_id), $time, "sys", $type, "aqdisk", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^dada_diskdb -k ".$db_key;
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

