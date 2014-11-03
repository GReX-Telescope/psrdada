#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bf_buffers.pl
#
# Runs and maintains information on all Shared Memory buffers on this host
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
  print "Usage: ".basename($0)."\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $localhost : shared;
our $pwc_id : shared;
our $binary : shared;
our $regex : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
$pwc_id = -1;
$binary = "unknown";
$regex = "unknown";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
# TODO check this is good idea
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;

# Check command line argument
if ($#ARGV != -1)
{
  usage();
  exit(1);
}

# firstly get the hostname, to determine which channels will be received 
# on this host
$localhost = Dada::getHostMachineName());

my @channels = ();
for ($i=0; $i<$cfg{"NRECV"}; $i++)
{
  if ($cfg{"RECV_".$i} eq $localhost)
  {
    push @channels, $i;
  }
}

if ($#channels == -1)
{
  print STDERR "Could not find machine channels for ".$localhost."\n";
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
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;
  
  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  my $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
  my $udp_ip   = $cfg{"PWC_UDP_IP_".$pwc_id};
  my $udp_port = $cfg{"PWC_UDP_PORT_".$pwc_id};
  my $udp_core = $cfg{"PWC_UDP_CORE_".$pwc_id};
  my $pfb_id   = $cfg{"PWC_PFB_ID_".$pwc_id};

  my $pwc_port     = int($cfg{"PWC_PORT"});
  my $pwc_logport = int($cfg{"PWC_LOGPORT"});
  if ($cfg{"USE_BASEPORT"} eq "yes")
  {
    $pwc_port     += int($pwc_id);
    $pwc_logport += int($pwc_id);
  }

  my $pwc_state = $cfg{"PWC_STATE_".$pwc_id};
  my $mon_dir   = $cfg{"CLIENT_UDP_MONITOR_DIR"}."/".$pfb_id;

  my ($cmd, $result, $response, $full_cmd);

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) 
  {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }
  logMsg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  logMsg (2, "INFO", "mon_dir=".$mon_dir);

  # ensure the monitoring dir exists
  if (! -d $mon_dir ) 
  {
    $cmd = "mkdir -p ".$mon_dir;
    logMsg (2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg (3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg (0, "ERROR", "failed to create ".$mon_dir.": ".$response);
      $quit_daemon = 1;
    }
  }

  logMsg (2, "INFO", "pwc_state=".$pwc_state);

  # simple run 1 instance of the PWC, this should be persistent!
  if (($pwc_state eq "active") || ($pwc_state eq "passive"))
  {
    $binary = $cfg{"PWC_BINARY"};
    $regex = $binary." -m ".$pfb_id;
    $cmd = $binary." -m ".$pfb_id." -c ".$pwc_port." -k ".lc($key).
           " -l ".$pwc_logport.  " -i ".$udp_ip." -p ".$udp_port.
           " -b ".$udp_core." -M ".$mon_dir;

    # passive PWC's respond to the PWCC, but insert 0's into the data stream
    if ($pwc_state eq "passive")
    {
      $cmd .= " -0";
    }

  }
  elsif ($pwc_state eq "inactive") 
  {
    $binary = "mopsr_udpdump";
    $regex = $binary." -m ".$pfb_id;
    $cmd = "mopsr_udpdump -m ".$pfb_id." -c ".$pwc_port." -l ".$pwc_logport.
           " -M ".$mon_dir." -i ".$udp_ip." -p ".$udp_port;
  }
  else
  {
    logMsg (0, "ERROR", "unrecognized pwc_state [".$pwc_state."]");
    $quit_daemon = 1;
  }

  if (!$quit_daemon)
  {
    my $full_cmd = $cmd ." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_mopsr_src_logger.pl ".$pwc_id." pwc";

    logMsg(1, "INFO", "START ".$cmd);
    logMsg(2, "INFO", "main: ".$full_cmd);
    ($result, $response) = Dada::mySystem ($full_cmd);
    logMsg(2, "INFO", "main: ".$result." ".$response);
    logMsg(1, "INFO", "END   ".$cmd);
    if (($result ne "ok") && (!$quit_daemon))
    {
      logMsg(0, "ERROR", $cmd." failed: ".$response);
    }
  }

  # Rejoin our daemon control thread
  logMsg(2, "INFO", "joining control thread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

  exit (0);
}

#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$)
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "pwc", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  if ($regex ne "")
  { 
    Dada::logMsg(2, $dl ,"controlThread: killProcess(".$regex.", mpsr)");
    my ($result, $response) = Dada::killProcess($regex, "mpsr");
    Dada::logMsg(2, $dl ,"controlThread: killProcess() ".$result." ".$response);
  }

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

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

