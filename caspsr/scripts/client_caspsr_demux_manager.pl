#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Runs caspsr_udpNdb and 4 x caspsr_dbib's on the demuxers
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
sub runDemuxer($);
sub recveiverThread($$$);

#
# Global variables
#
our $dl;
our $daemon_name;
our %cfg;
our $client_logger : shared;
our $control_port : shared;
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
$control_port = 0;
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

my $n_recv         = $cfg{"NUM_RECV"};
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

my $i=0;
my $i_demux = 0;
my $host = Dada::getHostMachineName();
my $recv_thread = 0;
my $recv_result = "";

# determine the number of this demuxer
for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++)
{
  if ($host eq $cfg{"DEMUX_".$i}) 
  {
    $i_demux = $i;
  }
}

# Main Loop
while (!($quit_daemon)) {

  msg(2, "INFO", "main: launchReceiversThread(".$n_recv.", ".$i_demux.")");
  $recv_thread = threads->new(\&launchReceiversThread, $n_recv, $i_demux);

  # Run caspsr_udpNdb on this machine
  msg(2, "INFO", "main: runDemuxer(".$i_demux.")"); 
  ($result, $response) = runDemuxer($i_demux);
  msg(2, "INFO", "main: runDemuxer() ".$result." ".$response);

  $recv_result = $recv_thread->join();
  if ($recv_result ne "ok") {
    msg(0, "ERROR", "main: launchReceiversThread failed");
    $quit_daemon = 1;
  } 

  msg(1, "INFO", "main: launchReceiversThread ended");

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
# Run the demuxer (caspsr_udpNdb) 
#
sub runDemuxer($) 
{

  my ($i_demux) = @_;

  msg(2, "INFO", "runDemuxer(".$i_demux.")");

  $control_port = $cfg{"DEMUX_CONTROL_PORT"};

  my $cmd = "";
  my $result = "";
  my $response = "";
  
  my $binary       = $cfg{"DEMUX_BINARY"};
  my $pks_per_xfer = $cfg{"PKTS_PER_XFER"};
  my $n_demux    = $cfg{"NUM_DEMUX"};
  my $udp_port     = $cfg{"DEMUX_UDP_PORT_".$i_demux};
  my $host         = Dada::getHostMachineName();

  my $i = 0;
  my $db_keys = "";
  # get the list of datablocks to write to
  for ($i=0; $i<$cfg{"NUM_RECV"}; $i++)
  {
    $db_keys .= " ".$cfg{"RECV_DB_".$i};
  }

  $cmd  = $binary." -n ".$pks_per_xfer." -o ".$control_port.
          " -p ".$udp_port." ".$i_demux." ".$n_demux.$db_keys;

  $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$host."_udp";

  msg(1, "INFO", "runDemuxer: running ".$binary." ".$i_demux.", listening on port ".$udp_port);

  msg(2, "INFO", "runDemuxer: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    msg(0, "ERROR", "runDemuxer ".$cmd." failed:  ".$response);
  }

  my $regex = "^perl.*".$client_logger." ".$host."_udp";
  msg(2, "INFO", "receiverThread: killProcess(".$regex.")");
  my ($killresult, $killresponse) = Dada::killProcess($regex);
  msg(2, "INFO", "receiverThread: killProcess: ".$result." ".$response);
  if ($killresult ne "ok") {
    msg(1, "INFO", "receiverThread: killProcess(".$regex.") returned ".$response);
  }


  msg(2, "INFO", "runDemuxer(".$i_demux.") ending");

  return ($result, $response);

}

###############################################################################
#
# Wait for a new header to arrive on the primary datablock, and then launch
# the NUM_RECV receivers to proces data that should be arriving on each datablock
#
sub launchReceiversThread($$)
{
  
  my ($n_recv, $i_demux) = @_;

  msg(2, "INFO", "launchReceiversThread(".$n_recv.", ".$i_demux.")");

  my $first_db_key = $cfg{"RECV_DB_0"};
  my $cmd = "";
  my $raw_header = "";
  my %h = ();
  my $header_valid = 0;
  my $obs_offset = "";
  my $obs_xfer = "";
  my $i_recv = 0;

  my @recv_threads = ();
  my @recv_results = ();

  for ($i=0; $i<$n_recv; $i++)
  {
    push @recv_threads, 0;
    push @recv_results, 0;
  }

  # continuously run the receivers, they should only be processing observation
  # at a time

  while (!$quit_daemon) {

    # Get the next filled header on the data block. Note that this may very
    # well hang for a long time - until the next header is written...
    $cmd = "dada_header -k ".$first_db_key;
    msg(2, "INFO", "launchReceiversThread: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "launchReceiversThread: ".$cmd." returned");

    # since the only way to currently stop this daemon is to send a kill
    # signal to dada_header_cmd, we should check the return value
    if ($? == 0) {

      %h = Dada::headerToHash($raw_header);

      $header_valid = 1;

      if (defined($h{"OBS_OFFSET"}) && length($h{"OBS_OFFSET"}) > 0) {
        $obs_offset = $h{"OBS_OFFSET"};
      } else {
        msg(0, "ERROR", "Error: OBS_OFFSET [".$h{"OBS_OFFSET"}."] was malformed or non existent");
        $header_valid = 0;
      }

      if (defined($h{"OBS_XFER"}) && length($h{"OBS_XFER"}) > 0) {
        $obs_xfer = $h{"OBS_XFER"};
      } else {
        msg(0, "ERROR", "Error: OBS_XFER [".$h{"OBS_XFER"}."] was malformed or non existent");
        $header_valid = 0;
      }

      if ($obs_xfer ne "0") {
        msg(0, "ERROR", "Error: OBS_XFER == ".$obs_xfer.", expected 0");
        $header_valid = 0;
      }

      msg(2, "INFO", "launchReceiversThread: OBS_OFFSET=".$obs_offset.", OBS_XFER=".$obs_xfer);

      # Run the receiver threads, one for each PWC
      msg(1, "INFO", "launchReceiversThread: starting ".$n_recv." receiver threads");
      for ($i=0; $i<$n_recv; $i++)
      {
        $i_recv = $i;
        msg(2, "INFO", "launchReceiversThread: receiverThread(".$i_recv.", ".$i_demux.",  ".$header_valid.")");
        $recv_threads[$i] = threads->new(\&receiverThread, $i_recv, $i_demux, $header_valid);
      } 
      msg(2, "INFO", "launchReceiversThread: receivers running");

      # join the receiver_threads
      msg(2, "INFO", "launchReceiversThread: joining ".$n_recv." recv threads");
      for ($i=0; $i<$n_recv; $i++)
      {
        if ($recv_threads[$i] != 0)
        {
          msg(2, "INFO", "launchReceiversThread: joining recv_thread ".$i);
          $recv_results[$i] = $recv_threads[$i]->join();
          msg(2, "INFO", "launchReceiversThread: recv_thread ".$i." joined: ".$recv_results[$i]);
          if ($result ne "ok")
          {
            msg(0, "ERROR", "launchReceiversThread: recv_thread[".$i."] failed");
            $quit_daemon = 1;
          }
        }
        $recv_threads[$i] = 0;
      }

      msg(1, "INFO", "launchReceiversThread: recveiver threads ended");

    } else {
      if (!$quit_daemon) {
        msg(0, "WARN", "launchReceiversThread: ".$cmd." failed - probably no data block");
        sleep 1; 
      }
    }
  }
  return "ok";
}



###############################################################################
# 
# Run something to process on each datablock, also take note of the NUM_PWC
# vs NUM_RECV, so that dbnull can be run if a GPU node is unavailable
#
sub receiverThread($$$)
{

  my ($i_recv, $i_demux, $header_valid) = @_;

  msg(2, "INFO", "receiverThread(".$i_recv.", ".$i_demux.", ".$header_valid.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my %h = ();

  my $active_bin   = $cfg{"IB_ACTIVE_BINARY"};
  my $inactive_bin = $cfg{"IB_INACTIVE_BINARY"};
  my $ib_port      = $cfg{"DEMUX_IB_PORT_".$i_demux};
  my $chunk_size   = $cfg{"IB_CHUNK_SIZE"};
  my $n_demux      = $cfg{"NUM_DEMUX"};
  my $ib_dest      = $cfg{"RECV_".$i_recv};   # IPoIB hostname
  my $db_key       = $cfg{"RECV_DB_".$i_recv};

  # check that a PWC matches this IB_DEST
  my $is_active = 0;
  my $ip_dest = "";
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $ip_dest = $cfg{"PWC_".$i};
    if ($ib_dest =~ m/$ip_dest/) {
      $is_active = 1;
    }
  }

  if (!$header_valid) {
    $cmd = "dada_dbnull -s -k ".$db_key;
  } else {
    if ($is_active) {
      $cmd = $active_bin." -S -c ".$chunk_size." -p ".$ib_port." -k ".$db_key." ".$ib_dest." ".$i_demux." ".$n_demux;
    } else {
      msg(0, "INFO", "[".$i_recv."] Jettesioning data destined for ".$ib_dest." since not PWC configured");
      $cmd = $inactive_bin." -s -k ".$db_key;
    }
  }

  $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$host."_ib".$i_recv;

  my $short_cmd = substr($cmd, 0, 50)."...";
    
  msg(2, "INFO", "receiverThread: [".$i_recv."] START ".$short_cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "receiverThread: [".$i_recv."] END   ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "receiverThread: [".$i_recv."]".$cmd." failed:  ".$response);
  }

  my $regex = "^perl.*".$client_logger." ".$host."_ib".$i_recv;
  msg(2, "INFO", "receiverThread: killProcess(".$regex.")");
  my ($killresult, $killresponse) = Dada::killProcess($regex);
  msg(2, "INFO", "receiverThread: killProcess: ".$result." ".$response);
  if ($killresult ne "ok") {
    msg(1, "INFO", "receiverThread: killProcess(".$regex.") returned ".$response);
  }

  return $result;
}


###############################################################################
#
# Control Thread. Handles signals to exit
#
sub controlThread($$) {

  my ($quit_file, $pid_file) = @_;

  msg(2, "INFO", "controlThread: starting");

  my $cmd = "";
  my $regex = "";
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

  msg(1, "INFO", "controlThread: connectToMachine(".$host.", ".$control_port.")");
  $handle = Dada::connectToMachine($host, $control_port);
  if ($handle) {
    msg(1, "INFO", "controlThread: ".$control_port." <- QUIT");
    ($result, $response) = Dada::sendTelnetCommand($handle, "QUIT");
    msg(1, "INFO", "controlThread: ".$control_port." -> ".$result." ".$response);
    $handle->close();
  }

  $regex = "^dada_header";
  msg(2, "INFO", "controlThread: killProcess(".$regex.")");
  ($result, $response) = Dada::killProcess($regex);
  msg(2, "INFO", "controlThread: killProcess: ".$result." ".$response);
  if ($result ne "ok") {
    msg(1, "INFO", "controlThread: killProcess(".$regex.") returned ".$response);
  }
  
  $regex = "^".$cfg{"IB_ACTIVE_BINARY"};
  msg(2, "INFO", "controlThread: killProcess(".$regex.")");
  ($result, $response) = Dada::killProcess($regex);
  msg(2, "INFO", "controlThread: killProcess: ".$result." ".$response);
  if ($result ne "ok") {
    msg(1, "INFO", "controlThread: killProcess(".$regex.") returned ".$response);
  }

  $regex = "^".$cfg{"IB_INACTIVE_BINARY"};
  msg(2, "INFO", "controlThread: killProcess(".$regex.")");
  ($result, $response) = Dada::killProcess($regex);
  msg(2, "INFO", "controlThread: killProcess: ".$result." ".$response);
  if ($result ne "ok") {
    msg(1, "INFO", "controlThread: killProcess(".$regex.") returned ".$response);
  }

  # allow 1 second for the logger (attahced to these daemons) to quit
  sleep(1);

  $regex = "^perl.*".$client_logger;
  msg(2, "INFO", "controlThread: killProcess(".$regex.")");
  ($result, $response) = Dada::killProcess($regex);
  msg(2, "INFO", "controlThread: killProcess: ".$result." ".$response);
  if ($result ne "ok") {
    msg(1, "INFO", "controlThread: killProcess(".$regex.") returned ".$response);
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
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "demux mngr", $msg);
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

  # check for definition of demuxer variables
  if (! defined($cfg{"DEMUX_CONTROL_PORT"})) {
    return ("fail", "Error: DEMUX_CONTROL_PORT not defined in caspsr cfg file");
  }

  my $i = 0;
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    if (! defined($cfg{"DEMUX_UDP_PORT_".$i})) {
      return ("fail", "Error: DEMUX_UDP_PORT_".$i." not deinfed in caspsr cfg file");
    }
    if (! defined($cfg{"DEMUX_IB_PORT_".$i})) {
      return ("fail", "Error: DEMUX_IB_PORT_".$i." not deinfed in caspsr cfg file");
    }
  }
   
  # check for definition of RECV_ variables 
  for ($i=0; $i<$cfg{"NUM_RECV"}; $i++) {
    if (! defined($cfg{"RECV_".$i})) {
      return ("fail", "Error: RECV_".$i." not defined in caspsr cfg file");
    }
    if (! defined($cfg{"RECV_DB_".$i})) {
      return ("fail", "Error: RECV_DB_".$i." not defined in caspsr cfg file");
    }
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}

###############################################################################
