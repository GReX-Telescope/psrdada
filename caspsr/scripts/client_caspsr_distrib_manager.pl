#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  6 May 2010
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Dada;
use Caspsr;
use File::Basename;
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;

#
# Prototypes
#
sub good($);
sub msg($$$);

#
# Global variables
#
our $dl;
our $daemon_name;
our %cfg;
our $client_logger : shared;
our @threads;
our @stats_threads;
our @results;
our @indexes;
our @udp_ports;
our @control_ports;
our $quit_daemon : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# initialize package globals
#
$dl = 1; 
$daemon_name = 0;
%cfg = Caspsr::getConfig();
$client_logger = "client_caspsr_src_logger.pl";
@threads = ();
@stats_threads = ();
@results = ();
@indexes = ();
@udp_ports = ();
@control_ports = ();
$quit_daemon = 0;
$log_host = 0;
$log_port = 0;
$log_sock = 0;


###############################################################################
#
# Main 
# 

$daemon_name = Dada::daemonBaseName($0);

my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".pid";
my $quit_file      = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";

$log_host          = $cfg{"SERVER_HOST"};
$log_port          = $cfg{"SERVER_DEMUX_LOG_PORT"};

my $control_thread = 0;
my $stats_thread = 0;
my $result = "";
my $response = "";

# sanity check on whether the module is good to go
($result, $response) = good($quit_file);
if ($result ne "ok") {
  print STDERR $response."\n";
  return 1;
}

# install signal handles
$SIG{INT}  = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# become a daemon
Dada::daemonize($log_file, $pid_file);

# Open a connection to the server_sys_monitor.pl script
$log_sock = Dada::nexusLogOpen($log_host, $log_port);
if (!$log_sock) {
  print STDERR "Could not open log port: ".$log_host.":".$log_port."\n";
}

logMsg(0, "INFO", "STARTING SCRIPT");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

# Main Loop
while (!($quit_daemon)) {

  # Run the demuxer threads for each instance of the demuxer
  msg(1, "INFO", "main: starting demuxer threads");
  my $i=0;
  for ($i=0; $i<=$#threads; $i++) {
    msg(2, "INFO", "main: demuxerThread(".$indexes[$i].", ".$udp_ports[$i].", ".$control_ports[$i].")");
    $threads[$i] = threads->new(\&demuxerThread, $indexes[$i], $udp_ports[$i], $control_ports[$i]);

    # msg(2, "INFO", "main: statsThread(".$indexes[$i].", ".$control_ports[$i].")");
    # $stats_threads[$i] = threads->new(\&statsThread, $indexes[$i], $control_ports[$i]);
  }

  msg(2, "INFO", "main: joining demuxer threads");
  for ($i=0; $i<=$#threads; $i++) {
    $results[$i] = $threads[$i]->join();
    if ($results[$i] ne "ok") {
      msg(0, "ERROR", "main: demuxerThread[".$i."] failed");
      $quit_daemon = 1;
    }
  }
  msg(1, "INFO", "main: demuxer threads ended");

  # msg(2, "INFO", "main: joining stats threads");
  # for ($i=0; $i<=$#threads; $i++) {
  #   $stats_threads[$i]->join();
  # }
  # msg(1, "INFO", "main: stats threads ended");

  sleep(1);

}

logMsg(2, "INFO", "main: joining threads");
$control_thread->join();
logMsg(2, "INFO", "main: control_thread joined");

logMsg(0, "INFO", "STOPPING SCRIPT");
Dada::nexusLogClose($log_sock);

exit(0);

###############################################################################
# 
# Run the demuxer 
#

sub demuxerThread($$$) {

  my ($index, $udp_port, $control_port) = @_;

  msg(2, "INFO", "demuxerThread(".$index.", ".$udp_port.", ".$control_port.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $receivers = "";
  my $i = 0;
  
  for ($i=0; $i<$cfg{"NUM_RECV"}; $i++) {
    $receivers .= $cfg{"RECV_".$i}." ";
  }

  $cmd  = "sudo LD_LIBRARY_PATH=/usr/local/pgplot ".
          $cfg{"DEMUX_BINARY"}." -n ".$cfg{"PKTS_PER_XFER"}.
          " -o ".$control_port." -p ".$udp_port." -q ".$cfg{"RECV_PORT"}.
          " ".$index." ".$cfg{"NUM_DEMUX"}." ".$receivers;

  $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." demux".$index;

  msg(1, "INFO", "demuxerThread: running ".$cfg{"DEMUX_BINARY"}." ".$index." listen on port ".$udp_port);

  msg(2, "INFO", "demuxerThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    msg(0, "ERROR", "demuxerThread ".$cmd." failed:  ".$response);
  }

  msg(2, "INFO", "demuxerThread(".$index.", ".$udp_port.", ".$control_port.") ending");

  return ($result);

}

###############################################################################
#
# statsThread. Handles signals to exit
#
sub statsThread($$) {

  my ($index, $control_port) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $handle = 0;
  my $localhost = Dada::getHostMachineName();

  while (!$quit_daemon) {

    msg(1, "INFO", "statsThread: connectToMachine(".$localhost.", ".$control_port.")");
    $handle = Dada::connectToMachine($localhost, $control_port);
    if ($handle) {
      msg(1, "INFO", "statsThread: ".$control_port." <- STATS");
      ($result, $response) = Dada::sendTelnetCommand($handle, "STATS");
      msg(1, "INFO", "statsThread: ".$control_port." -> ".$result." ".$response);

      $handle->close()
    }
    sleep(2);
  }
  msg(1, "INFO", "statsThread: exiting");
}



###############################################################################
#
# Control Thread. Handles signals to exit
#
sub controlThread($$) {

  my ($quit_file, $pid_file) = @_;

  msg(2, "INFO", "controlThread: starting");

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  # instruct the DEMUX_BINARIES to exit forthwith
  my $host = Dada::getHostMachineName();
  my $i = 0;
  my $handle = 0;

  for ($i=0; $i<=$#control_ports; $i++) {
    msg(1, "INFO", "controlThread: connectToMachine(".$host.", ".$control_ports[$i].")");
    $handle = Dada::connectToMachine($host, $control_ports[$i]);
    if ($handle) {
      msg(1, "INFO", "controlThread: ".$control_ports[$i]." <- QUIT");
      ($result, $response) = Dada::sendTelnetCommand($handle, "QUIT");
      msg(1, "INFO", "controlThread: ".$control_ports[$i]." -> ".$result." ".$response);
      $handle->close();
    }
  }

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");

  return 0;
}


###############################################################################
#
# Logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      #print "opening nexus log: ".$log_host.":".$log_port."\n";
      #$log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "obs mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


###############################################################################
#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # Tell threads to try and quit
  $quit_daemon = 1;
  sleep(3);
  
  if ($log_sock) {
    close($log_sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

###############################################################################
#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}

###############################################################################
#
# Test to ensure all variables are set 
#
sub good($) {

  my ($quit_file) = @_;

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # check that demuxers have been defined in the config
  if (! defined($cfg{"NUM_DEMUX"})) {
    return ("fail", "Error: NUM_DEMUX not defined in caspsr cfg file");
  }

  if ($daemon_name eq "") {
    return ("fail", "Error: a package variable missing [daemon_name]");
  }

  my $host = Dada::getHostMachineName();
  my $i = 0;
  my @bits = ();
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    if ($cfg{"DEMUX_".$i} =~ m/$host/) {
      push(@threads, 0);
      push(@indexes, $i);
      push(@udp_ports, $cfg{"DEMUX_UDP_PORT_".$i});
      push(@control_ports, $cfg{"DEMUX_CONTROL_PORT_".$i});
    }
  }

  if ($#indexes == -1) {
    return ("fail", "Error: no demuxers were defined for this host in config file");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}

###############################################################################
