#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_pwc.pl 
#
# Runs the configured PWC_BINARY for this host
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

sub pwcSrcLogger($);
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
our $localhost : shared;
our $pwc_id : shared;
our $binary : shared;
our $regex : shared;
our $log_host;
our $log_port;
our $log_sock;
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
$binary = "unknown";
$regex = "unknown";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
# TODO check this is good idea
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;

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
  if ($cfg{"PWC_".$pwc_id} ne Dada::getHostMachineName())
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

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;
  
  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
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
    $pwc_port    += int($pwc_id);
    $pwc_logport += int($pwc_id);
  }

  # here we will select the preferred modules from this list of e.g. "EG01 7 2"
  my ($cmd, $result, $response);
  my @ants = ();
  if ( -f "/home/dada/linux_64/share/preferred_modules.txt" )
  {
    $cmd = "grep ".$pfb_id." /home/dada/linux_64/share/preferred_modules.txt";
    ($result, $response) = Dada::mySystem($cmd);
    my @parts = split(/ +/, $response, 2);
    @ants = split(/ +/, $parts[1]);
  }
  else
  {
    @ants      = split (/ /, $cfg{"PWC_ANTS"});
  }

  my $pwc_state = $cfg{"PWC_STATE_".$pwc_id};
  my $mon_dir   = $cfg{"CLIENT_UDP_MONITOR_DIR"}."/".$pfb_id;
  my $start_chan = 25;
  my $end_chan   = 64;

  if (exists($cfg{"PWC_START_CHAN"}))
  {
    $start_chan = $cfg{"PWC_START_CHAN"};
  }
  if (exists($cfg{"PWC_END_CHAN"}))
  {
    $end_chan = $cfg{"PWC_END_CHAN"};
  }

  my ($cmd, $result, $response, $ant);

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) 
  {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }
  msg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  msg (2, "INFO", "mon_dir=".$mon_dir);

  # ensure the monitoring dir exists
  if (! -d $mon_dir ) 
  {
    $cmd = "mkdir -p ".$mon_dir;
    msg (2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg (3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      msg (0, "ERROR", "failed to create ".$mon_dir.": ".$response);
      $quit_daemon = 1;
    }
  }

  msg (2, "INFO", "pwc_state=".$pwc_state);

  # simple run 1 instance of the PWC, this should be persistent!
  if (($pwc_state eq "active") || ($pwc_state eq "passive"))
  {
    $binary = $cfg{"PWC_BINARY"};
    $regex = $binary." -m ".$pfb_id;
    if ($binary eq "mopsr_udpdb_dual")
    {
      $cmd = $binary." -m ".$pfb_id." -c ".$pwc_port." -k ".lc($key).
             " -l ".$pwc_logport.  " -i ".$udp_ip.":".$udp_port.
             " -i ".$udp_ip.":".($udp_port+1).
             " -b ".$udp_core." -M ".$mon_dir;
    }
    else
    {
      $cmd = "numactl -C ".$udp_core." ".$binary." -m ".$pfb_id." -c ".$pwc_port." -k ".lc($key).
             " -l ".$pwc_logport.  " -i ".$udp_ip." -p ".$udp_port.
             " -b ".$udp_core." -M ".$mon_dir;
    }

    # only select channels if using selants version
    if ($binary eq "mopsr_udpdb_selants")
    {
      $cmd .= " -C ".$start_chan.
              " -D ".$end_chan;
      foreach $ant (@ants)
      {
        $cmd .= " -a ".$ant;
      }
    }

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
    msg (0, "ERROR", "unrecognized pwc_state [".$pwc_state."]");
    $quit_daemon = 1;
  }

  if (!$quit_daemon)
  {
    msg(1, "INFO", "START ".$cmd);
    ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, 0, "pwc", sprintf("%02d",$pwc_id), $daemon_name, "pwc");
    msg(2, "INFO", "main: ".$result." ".$response);
    msg(1, "INFO", "END   ".$cmd);
    if (($result ne "ok") && (!$quit_daemon))
    {
      msg(0, "ERROR", $cmd." failed: ".$response);
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

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
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, sprintf("%02d",$pwc_id), $time, "sys", $type, "pwc", $msg);
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

  if ($regex ne "")
  { 
    msg(2, "INFO", "controlThread: killProcess(".$regex.", mpsr)");
    my ($result, $response) = Dada::killProcess($regex, "mpsr");
    msg(2, "INFO", "controlThread: killProcess() ".$result." ".$response);
  }

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

