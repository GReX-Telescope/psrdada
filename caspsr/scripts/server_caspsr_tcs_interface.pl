#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use Time::Local;
use Switch;
use Dada;
use Caspsr;

Dada::preventDuplicateDaemon(basename($0));

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our $tcs_cfg_file;
our $tcs_spec_file;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;
our $pwcc_running : shared;
our $current_state : shared;
our $recording_start : shared;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our %tcs_cmds;
our %site_cfg;
our $pwcc_thread;
our $utc_stop_thread;
our $utc_stop_remaining : shared;
our $utc_start_unix;
our $tcs_host;
our $tcs_port;
our $tcs_sock;


#
# Global Variable Initialization
#
%cfg           = Caspsr::getConfig();
$dl            = 1;
$daemon_name   = Dada::daemonBaseName($0);
$tcs_cfg_file  = $cfg{"CONFIG_DIR"}."/caspsr_tcs.cfg";
$tcs_spec_file = $cfg{"CONFIG_DIR"}."/caspsr_tcs.spec";
$warn = "";
$error = "";
$quit_daemon = 0;
$tcs_host = "";
$tcs_port = 0;
$tcs_sock = 0;
$pwcc_running = 0;
$current_state = "";
$recording_start = 0;
$pwcc_host = "";
$pwcc_port = 0;
$client_master_port = 0;
%tcs_cmds = ();
%site_cfg = ();
$pwcc_thread = 0;
$utc_start_unix = 0;
$utc_stop_thread = 0;
$utc_stop_remaining = -1;

#
# Constants
#
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant TERMINATOR         => "\r\n";
use constant IBOB_CONTROL_IP    => "192.168.0.15";
use constant IBOB_CONTROL_PORT  => "7";

#
# Function Prototypes
#
sub main();


my $result = 0;
$result = main();

exit($result);


###############################################################################
#
# package functions
# 

sub main() {

  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $server_host =     $cfg{"SERVER_HOST"};
  my $config_dir =      $cfg{"CONFIG_DIR"};

  # Connection to TCS
  $tcs_host =        $cfg{"TCS_INTERFACE_HOST"};
  $tcs_port =        $cfg{"TCS_INTERFACE_PORT"};
  my $tcs_state_port =  $cfg{"TCS_STATE_INFO_PORT"};

  # PWCC (dada_pwc_command) 
  $pwcc_host    = $cfg{"PWCC_HOST"};
  $pwcc_port    = $cfg{"PWCC_PORT"};

  # Set some global variables
  $client_master_port     = $cfg{"CLIENT_MASTER_PORT"};

  my $handle = "";
  my $peeraddr = "";
  my $hostinfo = "";  
  my $command = "";
  my $key = "";
  my $result = "";
  my $response = "";
  my $state_thread = 0;
  my $control_thread = 0;
  my $tcs_connected = 0;
  my $rh = 0;;
  my $hostname = "";
  my $cmd = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  %tcs_cmds = ();
  %site_cfg = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);

  # set initial state
  $current_state = "Idle";

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # set umask so that 
  #  files : -rw-r-----
  #   dirs : drwxr-x---
  umask 0027;

  Dada::logMsg(0, $dl, "Programming ibob");
  $cmd = "cat /home/dada/ib_ibob_config.txt | bibob_terminal 192.168.0.15 7";
  Dada::logMsg(2, $dl, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "main: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(1, $dl, "main: ".$result." ".$response);
  }

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  foreach $key (keys (%site_cfg)) {
    Dada::logMsg(2, $dl, "site_cfg: ".$key." => ".$site_cfg{$key});
  }

  # generate the cfg file required to launch dada_pwc_command 
  ($result, $response) = generateConfigFile($tcs_cfg_file);

  # configure the ibob with the default settings for caspsr
  # ($result, $response) = configureIbob();

  # Launch a persistent dada_pwc_command with the $tcs_cfg_file
  $pwcc_thread = threads->new(\&pwccThread);

  # start the stateThread
  Dada::logMsg(2, $dl, "main: stateThread()");
  $state_thread = threads->new(\&stateThread);

  my $read_set = new IO::Select();
  $read_set->add($tcs_sock);

  Dada::logMsg(2, $dl, "main: listening for TCS connection ".$tcs_host.":".$tcs_port);

  # Main Loop,  We loop forever unless asked to quit
  while (! $quit_daemon) {

    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles) {
  
      if ($rh == $tcs_sock) {

        # Only allow 1 connection from TCS
        if ($tcs_connected) {
        
          $handle = $rh->accept();
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          $handle->close();
          $handle = 0;
          Dada::logMsgWarn($warn, "Rejecting additional connection from ".$hostinfo->name);

        } else {

          # Wait for a connection from the server on the specified port
          $handle = $rh->accept();
          $handle->autoflush(1);
          $read_set->add($handle);

          # Get information about the connecting machine
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          $hostname = $hostinfo->name;

          Dada::logMsg(1, $dl, "Accepting connection from ".$hostname);
          $tcs_connected = 1;
          $handle = 0; 

        }

      # we have received data on the current read handle
      } else {

        $command = <$rh>;

        # If we have lost the connection...
        if (! defined $command) {

          Dada::logMsg(1, $dl, "Lost TCS connection from ".$hostname);
          $read_set->remove($rh);
          $rh->close();
          $tcs_connected = 0;

        # Else we have received a command
        } else {

          # clean the line up a little
          $command =~ s/\r//;
          $command =~ s/\n//;
          $command =~ s/#(.)*$//;
          $command =~ s/ +$//;

          if ($command ne "") {
            # handle the command from TCS
            ($result, $response) = processTCSCommand($command, $rh);
          } else {
            Dada::logMsgWarn($warn, "Received empty string from TCS");
            print $rh "ok".TERMINATOR;
          }
        }
      }
    }

    if (($current_state =~ m/^Recording/) && (time > $recording_start)) {
      if ($utc_stop_remaining > 0) {
        $current_state = "Recording [".(time - $recording_start).", ".$utc_stop_remaining." remaining]";
      } else {
        $current_state = "Recording [".(time - $recording_start)." secs]";
      }
    }

    
    if ($current_state eq "Stopped") {
      Dada::logMsg(1, $dl, "main: stopping_thread finished, changing to IDLE");
      $current_state = "Idle";
    }

    if ($utc_stop_thread && ($utc_stop_remaining < 0)) {
      Dada::logMsg(1, $dl, "main: utc_stop_thread finished");
      $utc_stop_thread->join();
      $utc_stop_thread = 0;
    }

  }

  Dada::logMsg(0, $dl, "main: joining threads");

  # rejoin threads
  $control_thread->join();
  $pwcc_thread->join();
  $state_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;
}



sub processTCSCommand($$) {

  my ($cmd, $handle) = @_;

  my $result = "";
  my $response = "";
  my $key = "";
  my $val = "";
  my $lckey = "";

  ($key, $val) = split(/ +/,$cmd,2);

  $lckey = lc $key;

  Dada::logMsg(1, $dl, "TCS -> ".$cmd);
  if ($key eq "PROCFIL") {
    $key = "PROC_FILE";
  }

  switch ($lckey) {

    case "start" {

      if (($current_state eq "Recording") || ($current_state =~ m/Starting/)) {
        $result = "fail";
        Dada::logMsg(1, $dl, "TCS <- ".$result);
        print $handle $result.TERMINATOR;
        return ("fail", "received start command when in ".$current_state." state");
      }

      Dada::logMsg(2, $dl, "processTCSCommand: START");

      %tcs_cmds = fixTCSCommands(\%tcs_cmds);


      # Check the TCS commands for validity
      Dada::logMsg(2, $dl, "processTCSCommand: parseTCSCommands()");
      ($result, $response) = parseTCSCommands();
      Dada::logMsg(2, $dl, "processTCSCommand: parseTCSCommands() ".$result ." ".$response);

      # Send an immediate response to TCS so we dont get a timeout
      Dada::logMsg(1, $dl, "TCS <- ".$result);
      print $handle $result.TERMINATOR;

      if ($result ne "ok") {

        Dada::logMsgWarn($error, "processTCSCommand: parseTCSCommands failed: ".$response);
        $current_state = "TCS Config Error: ".$response;

      } else {

        if ($response eq "Wrong receiver") {
          Dada::logMsgWarn($error, "Wrong receiver specified, ignoring");
          $current_state = "Idle";

        } else {

          my $max_wait = 32;
          while (($current_state ne "Stopped") && ($current_state ne "Idle") && ($max_wait > 0)) {
            Dada::logMsg(1, $dl, "waiting for return to Idle [countdown=".$max_wait."]");
            $max_wait--;
            sleep(1);
          }
      
          $current_state = "Starting...";

          # clear the status directory
          Dada::logMsg(2, $dl, "processTCSCommand: clearStatusDir()");
          clearStatusDir();

          # Add site.config parameters to the tcs_cmds;
          Dada::logMsg(2, $dl, "processTCSCommand: addSiteConfig()");
          addSiteConfig();
  
          # check that the PWCC is actually running
          if (!$pwcc_running) {

            Dada::logMsgWarn($warn, "PWCC thread was not running, attemping to relaunch");
            $pwcc_thread->join();
            Dada::logMsg(0, $dl, "processTCSCommand: pwcc_thread was joined");
            $pwcc_thread = threads->new(\&pwccThread);
            Dada::logMsg(0, $dl, "processTCSCommand: pwcc_thread relaunched");

          }

          # Create spec file for dada_pwc_command
          Dada::logMsg(2, $dl, "processTCSCommand: generateSpecificationFile(".$tcs_spec_file.")");
          ($result, $response) = generateSpecificationFile($tcs_spec_file);
          Dada::logMsg(2, $dl, "processTCSCommand: generateSpecificationFile() ".$result." ".$response);

          # Issue the start command itself
          Dada::logMsg(2, $dl, "processTCSCommand: start(".$tcs_spec_file.")");
          ($result, $response) = start($tcs_spec_file);
          Dada::logMsg(2, $dl, "processTCSCommand: start() ".$result." ".$response.")");

          if ($result eq "fail") {

            $current_state = "Start Failed: ".$response;
  
          } else {

            # determine the unix time of the utc_start
            my @t = split(/-|:/,$response); 
            $recording_start = timegm($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

            Dada::logMsg(2, $dl, "processTCSCommand: START successful");
            $current_state = "Recording";
            Dada::logMsg(2, $dl, "processTCSCommand: STATE=Recording");

            # Tell TCS what our utc start was
            $cmd = "start_utc ".$response;
            Dada::logMsg(1, $dl, "TCS <- ".$cmd);
            print $handle $cmd.TERMINATOR;

          }
        }
      }
      %tcs_cmds = ();
    }

    case "stop" {

      Dada::logMsg(2, $dl, "Processing STOP command");

      if (($current_state =~ m/Recording/) || ($current_state eq "Error")) {

        # we ask the demuxers to stop the dataflow on a specific byte
        # that is in the future.
        my $utc_stop_unix_time = (time+5);

        if ($utc_stop_unix_time < ($utc_start_unix + 32)) {
          Dada::logMsg(1, $dl, "processTCSCommand: [stop] overriding stop time to minimum of 32 seconds");
          $utc_stop_unix_time = $utc_start_unix + 32;
        }

        my $utc_stop_time = Dada::printDadaUTCTime($utc_stop_unix_time);

        Dada::logMsg(2, $dl, "processTCSCommand: stopDemuxers(".$utc_stop_time.")");
        ($result, $response) = stopDemuxers($utc_stop_time);
        Dada::logMsg(2, $dl, "processTCSCommand: stopDistibutors() ".$result." ".$response);


        # tell the nexus to stop on the same UTC_STOP time in the future
        Dada::logMsg(2, $dl, "processTCSCommand: stopNexus(".$utc_stop_time.")");
        ($result, $response) = stopNexus($utc_stop_time);
        Dada::logMsg(2, $dl, "processTCSCommand: stopNexus() ".$result." ".$response);

        $current_state = "Stopping...";

        Dada::logMsg(1, $dl, "processTCSCommand: stopInBackground()");
        my $tmp_thr_id = threads->new(\&stopInBackground);
        $tmp_thr_id->detach();

      } elsif ($current_state eq "Idle") {

        Dada::logMsg(1, $dl, "Received additional stop command");  
        $result = "ok";

      } elsif ($current_state eq "Preparing") {
  
        Dada::logMsgWarn($warn, "Received STOP during preparing state");
        $result = "ok";

      } elsif ($current_state eq "Stopping...") {

        Dada::logMsgWarn($warn, "Received STOP whilst already Stopping");
        $result = "ok";

      } else {

        $result = "fail";
        $response = "received STOP command when in ".$current_state;

      }

    }

    case "quit" {

    } 

    # This should be a header parameter, add it to the tcs_cmds hash
    else {

      if ($key =~ m/SET_UTC_START/) {
    
        Dada::logMsgWarn($warn, "Ignoring ".$key." -> ".$val." from APSR piggyback");
        $result = "ok";
        $response = "";

      } elsif (($current_state =~ m/Starting/) || ($current_state eq "Recording")) {

        Dada::logMsgWarn($warn, "received ".$key." -> ".$val." when in ".$current_state);
        $result = "fail";
        $response = "unexpected header command when in ".$current_state;

      } else {

        $tcs_cmds{$key} = $val;
        $result = "ok";
        $response = "";
        #$current_state = "Preparing";

        if (($key eq "BANDWIDTH") && ($val eq "0.000000")) {
          $result = "fail";
          $response = "cowardly refusing to observe with bandwidth=0.0";
          $current_state = "Error";
        }

      }
    }
  }

  # If the command failed, log it
  if ($result eq "fail") {
    Dada::logMsgWarn($error, $response);
  }

  # Special "hack" case as we return "ok" to a start
  # command without waiting
  if ($handle && ($lckey ne "start")) {

    if ($result eq "fail") {
      $current_state = "Error";
      print $handle $result.TERMINATOR;
      print $handle $response.TERMINATOR;

    } else {
      print $handle $result.TERMINATOR;
      Dada::logMsg(1, $dl, "TCS <- ".$result);
    }
  }
  return ($result, $response);

}
  
#
# Runs dada_pwc_command in non daemon mode. All ouput should be logged to
# the log file specified
#
sub pwccThread() {

  Dada::logMsg(1, $dl, "pwccThread: starting");

  my $result = "";
  my $response = "";
  my $cmd = "";

  if ( -f $tcs_cfg_file ) {

    $cmd = "dada_pwc_command ".$tcs_cfg_file." >> ".$cfg{"SERVER_LOG_DIR"}."/dada_pwc_command.log";
    Dada::logMsg(2, $dl, "pwccThread: ".$cmd);
    $pwcc_running = 1;
    ($result, $response) = Dada::mySystem($cmd);
    $pwcc_running = 0;
    Dada::logMsg(2, $dl, "pwccThread: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "pwccThread: dada_pwc_command failed");
    }

  } else {
    Dada::logMsgWarn($warn, "pwccThread: tcs config file did not exist: ".$tcs_cfg_file);
  }

  Dada::logMsg(1, $dl, "pwccThread: exiting");
  return ($result);
}




#
# Opens a socket and reports the current state of the PWCC
#
sub stateThread() {

  Dada::logMsg(1, $dl, "stateThread: starting");

  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"TCS_STATE_INFO_PORT"};
  my $read_set = 0;
  my $handle = 0;
  my $line = "";
  my $rh = 0;
  my $hostname = "";
  my $hostinfo = 0;

  # open the listening socket
  Dada::logMsg(2, $dl, "stateThread: opening socket ".$host.":".$port);
  my $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );

  if (!$sock) {
    Dada::logMsgWarn($error, "stateThread: could not create socket ".$host.":".$port);
    return 1;
  }

  Dada::logMsg(2, $dl, "stateThread: listening socket opened ".$host.":".$port);

  $read_set = new IO::Select();
  $read_set->add($sock);

  while (!$quit_daemon) {

    # Get all the readable handles from the read set
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles) {

      # If we are accepting a connection
      if ($rh == $sock) {
 
        $handle = $rh->accept();
        $handle->autoflush();
        $hostinfo = gethostbyaddr($handle->peeraddr);
        $hostname = $hostinfo->name;

        Dada::logMsg(3, $dl, "stateThread: Accepting connection from ".$hostname);
        $read_set->add($handle);
        $handle = 0;

      } else {

        $line = Dada::getLine($rh);

        if (! defined $line) {
          Dada::logMsg(3, $dl, "stateThread: closing read handle");
          $read_set->remove($rh);
          close($rh);

        } else {

          Dada::logMsg(3, $dl, "stateThread: received ".$line);
          if ($line eq "state") {
            print $rh $current_state."\r\n";
            Dada::logMsg(3, $dl, "stateThread: replied ".$current_state);
          }
        }
      }
    }
  }

  Dada::logMsg(2, $dl, "stateThread: exiting");
}



sub quitPWCCommand() {
  
  my $handle = 0;
  my $result = "";
  my $response = "";
  
  Dada::logMsg(2, $dl, "quitPWCCommand()");

  if (! $pwcc_running) {
    Dada::logMsg(2, $dl, "quitPWCCommand: dada_pwc_command not running");
    return ("ok", "");

  } else {

    Dada::logMsg(2, $dl, "quitPWCCommand: connecting to dada_pwc_command: ".$pwcc_host.":".$pwcc_port);
    $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);

    if ($handle) {

      # Ignore the "welcome" message
      $response = <$handle>;

      # Send quit command
      Dada::logMsg(2, $dl, "quitPWCCommand: sending quit to dada_pwc_command");
      print $handle "quit\r\n";
      $handle->close();

      # wait 2 seconds for the nexus to quite
      my $nwait = 2;
      while (($pwcc_running) && ($nwait > 0)) {
        sleep(1);
        $nwait--;
      }

      if ($pwcc_running) {
        Dada::logMsgWarn($warn, "Had to kill dada_pwc_command");
        ($result, $response) = Dada::killProcess("dada_pwc_command ".$tcs_cfg_file);
        Dada::logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      }

      return ("ok","");

    # try to kill the process manually
    } else {
      Dada::logMsgWarn($warn, "quitPWCCommand: could not connect to dada_pwc_command");
      ($result, $response) = Dada::killProcess("dada_pwc_command ".$tcs_cfg_file);
      Dada::logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      return ($result, $response);
    }

  }
}


#
# Send the START command to the pwcc
#
sub start($) {
                                                                              
  my ($file) = @_;

  my $rVal = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $utc_start = "UNKNOWN";

  while (! $pwcc_running) {
    Dada::logMsg(0, $dl, "start: waiting for dada_pwc_command to start");
    sleep(1);
  }
  sleep(1);

  # Connect to dada_pwc_command
  my $handle = Dada::connectToMachine($pwcc_host, $pwcc_port, 5);

  if (!$handle) {
    Dada::logMsg(0, $dl, "start: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue START command"); 

  } else {

    # Ignore the "welcome" message
    $result = <$handle>;

    # Check we are in the IDLE state before continuing
    if (Dada::waitForState("idle", $handle, 5) != 0) {
      Dada::logMsg(0, $dl, "start: nexus was not in the idle state after 5 seconds"); 
      return ("fail", "nexus was not in IDLE state");
    }

    # Send CONFIG command with apsr_tcs.spec
    $cmd = "config ".$file;
    Dada::logMsg(1, $dl, "nexus <- ".$cmd);
    ($result,$response) = Dada::sendTelnetCommand($handle,$cmd);
    Dada::logMsg(1, $dl, "nexus -> ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "start: config command failed: ".$response);
      return ("fail", "CONFIG command failed on nexus: ".$response)
    }

    # Wait for the PREPARED state
    if (Dada::waitForState("prepared",$handle,10) != 0) {
      Dada::logMsg(0, $dl, "start: nexus did not enter PREPARED state 10 seconds after config command");
      return ("fail", "nexus did not enter PREPARED state");
    }
    Dada::logMsg(2, $dl, "Nexus now in PREPARED state");

    # Send start command to the nexus
    $cmd = "start";
    Dada::logMsg(1, $dl, "start: nexus <- ".$cmd);
    ($result,$response) = Dada::sendTelnetCommand($handle,$cmd);
    Dada::logMsg(1, $dl, "start: nexus -> ".$result." ".$response);

    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "start: start command failed: ".$response);
      return ("fail", "START command failed on nexus: ".$response);
    }

    # give the nexus a few seconds to prepare itself and open the infiniband
    # connections
    sleep(3);

    # Tell the demuxers to start polling the incoming data for the reset packet
    Dada::logMsg(2, $dl, "start: threadedDemuxerCommand(START)");
    ($result, $response) = threadedDemuxerCommand("START");
    Dada::logMsg(2, $dl, "start: threadedDemuxerCommand() ".$result);

    # Instruct the ibob to rearm and get the corresponding UTC_START
    $cmd = "bibob_start_observation -m ".$cfg{"ARCHIVE_MOD"}." 192.168.0.15 7";
    Dada::logMsg(1, $dl, "start: ".$cmd);
    ($result,$response) = Dada::mySystem($cmd);
    Dada::logMsg(1, $dl, "start: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "start: ".$cmd." failed: ".$response);
      return ("fail", "bibob_start failed");
    }
    $utc_start = $response;

    ($result, $response) = set_utc_start($utc_start);
    if ($result ne "ok") {
      Dada::logMsgWarn($error, "Failed to set UTC_START: ".$response);
      return ("fail", "Failed to set UTC_START");
    }

    $utc_start_unix = Dada::getUnixTimeUTC($utc_start);
    Dada::logMsg(2, $dl, "start: setting utc_start_unix=".$utc_start_unix." from ".$utc_start);

    # for the tcs simulator, if an observing length has been specified
    # in seconds, then start a thread to issue a UTC_STOP after this
    # time has elapsed
    if (exists($tcs_cmds{"LENGTH"})) {
      my $int = int($tcs_cmds{"LENGTH"});
      Dada::logMsg(1, $dl, "start: LENGTH=".$int);
      if ($int > 0) {
        $utc_stop_remaining = $int;
        Dada::logMsg(1, $dl, "start: will stop recording after ".$utc_stop_remaining." seconds");
        $utc_stop_thread = threads->new(\&utcStopCommand);
      }
    }

    Dada::logMsg(1, $dl, "start() ok ".$utc_start);
    return ("ok", $utc_start);
  }

}


#
# Sends a UTC_START command to the pwcc
#
sub set_utc_start($) {

  my ($utc_start) = @_;

  Dada::logMsg(2, $dl, "set_utc_start(".$utc_start.")");

  my $source = "";
  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $i = 0;

  # Now that we know the UTC_START, create the required results and archive 
  # directories and put the observation summary file there...

  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;
  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
  my $proj_id     = $tcs_cmds{"PID"};

  # Create the observations' archive and results dirs
  $cmd = "mkdir ".$archive_dir;
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($error, "set_utc_start: ".$cmd." failed: ".$response);
    return ("fail", "could not create archive_dir");
  }

  $cmd = "mkdir -m 0770 ".$results_dir;
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($error, "set_utc_start: ".$cmd." failed: ".$response);
    return ("fail", "could not create results_dir");
  }

  # TODO make this work for multi fold sources 
  $source = $tcs_cmds{"SOURCE"};
  $source =~ s/^[JB]//;
  
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $cmd = "mkdir -p ".$results_dir."/".$source."/".$cfg{"PWC_".$i};
    Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($error, "set_utc_start: ".$cmd." failed: ".$response);
      return ("fail", "could not create ".$cfg{"PWC_".$i}."'s results_dir");
    }
  }

  my $caspsr_groups = `groups caspsr`;
  chomp $caspsr_groups;
  if ($caspsr_groups =~ m/$proj_id/) {
    # Do nothing
  } else {
    Dada::logMsgWarn($warn, "PID ".$proj_id." invalid, using caspsr instead"); 
    $proj_id = "caspsr";
  }

  $cmd = "chgrp -R ".$proj_id." ".$archive_dir." ".$results_dir;
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not chgrp results and archive dir to ".$proj_id.": ".$response);
  }

  $cmd = "chmod -R g+sw ".$results_dir;
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not chmod results dir to g+sw: ".$response);
  }

  $cmd = "chmod -R g+s ".$archive_dir;
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not chmod results and archive dir to g+w: ".$response);
  }

  my $fname = $archive_dir."/obs.info";
  Dada::logMsg(2, $dl, "set_utc_start: creating ".$fname);
  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";
  print FH Dada::headerFormat("SOURCE",$tcs_cmds{"SOURCE"})."\n";
  print FH Dada::headerFormat("RA",$tcs_cmds{"RA"})."\n";
  print FH Dada::headerFormat("DEC",$tcs_cmds{"DEC"})."\n";
  print FH Dada::headerFormat("CFREQ",$tcs_cmds{"CFREQ"})."\n";
  print FH Dada::headerFormat("PID",$tcs_cmds{"PID"})."\n";
  print FH Dada::headerFormat("BANDWIDTH",$tcs_cmds{"BANDWIDTH"})."\n";
  print FH Dada::headerFormat("PROC_FILE",$tcs_cmds{"PROC_FILE"})."\n";
  print FH "\n";
  print FH Dada::headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  print FH Dada::headerFormat("NBIT",$tcs_cmds{"NBIT"})."\n";
  print FH Dada::headerFormat("NPOL",$tcs_cmds{"NPOL"})."\n";
  print FH Dada::headerFormat("NDIM",$tcs_cmds{"NDIM"})."\n";
  print FH Dada::headerFormat("RESOLUTION",$tcs_cmds{"RESOLUTION"})."\n";
  print FH Dada::headerFormat("CONFIG",$tcs_cmds{"CONFIG"})."\n";
  close FH;

  $cmd = "cp ".$fname." ".$results_dir."/";
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not copy obs.info to results dir: ".$response);
  }

  $cmd = "touch ".$archive_dir."/obs.processing";
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not create obs.process in archive dir: ".$response);
  }

  $cmd = "touch ".$results_dir."/obs.processing";
  Dada::logMsg(2, $dl, "set_utc_start: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could not create obs.process in results dir: ".$response);
  }

  # connect to nexus
  my $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);
  if (!$handle) {
    Dada::logMsg(0, $dl, "set_utc_start: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue SET_UTC_START command"); 
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;
  
  # Wait for the prepared state
  if (Dada::waitForState("recording", $handle, 10) != 0) {
    Dada::logMsg(0, $dl, "set_utc_start: nexus did not enter RECORDING state 10 seconds after START command");
    $handle->close();
    return ("fail", "nexus did not enter RECORDING state");
  }

  # Send UTC Start command to the nexus
  $cmd = "set_utc_start ".$utc_start;

  Dada::logMsg(1, $dl, "nexus <- ".$cmd);
  ($result,$response) = Dada::sendTelnetCommand($handle,$cmd);
  Dada::logMsg(1, $dl, "nexus -> ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($error, "set_utc_start: nexus returned ".$response." after sending ".$cmd);
    return ("fail", $cmd." failed on nexus: ".$response);
  }

  $handle->close();

  # Send UTC Start to the demuxers
  $cmd = "SET_UTC_START ".$utc_start;
  Dada::logMsg(1, $dl, "set_utc_start: demuxers <- ".$cmd);
  Dada::logMsg(2, $dl, "set_utc_start: threadedDemuxerCommand(".$cmd.")");
  ($result, $response) = threadedDemuxerCommand($cmd);
  Dada::logMsg(2, $dl, "set_utc_start: threadedDemuxerCommand() ".$result." ".$response);
  Dada::logMsg(1, $dl, "set_utc_start: demuxers -> ".$result);
  if ($result ne "ok") {
    Dada::logMsgWarn($error, "set_utc_start: threadedDemuxerCommand(".$cmd.") failed with ".$response);
    return ("fail", $cmd." failed on nexus: ".$response);
  }
  

  return ($result, $utc_start);

}

################################################################################
#
# Ask the Demuxers to Stop on the time in the future
#
sub stopDemuxers($)
{

  my ($utc_stop) = @_;

  Dada::logMsg(2, $dl, "stopDemuxers(".$utc_stop.")");

  my $result = "";
  my $response = "";
  my $cmd = "UTC_STOP ".$utc_stop;

  # stop the demuxers on the specified time
  Dada::logMsg(1, $dl, "stopDemuxers: demuxers <- ".$cmd);
  Dada::logMsg(2, $dl, "stopDemuxers: threadedDemuxerCommand(".$cmd.")");
  ($result, $response) = threadedDemuxerCommand($cmd);
  Dada::logMsg(2, $dl, "stopDemuxers: threadedDemuxerCommand() ".$result." ".$response);
  Dada::logMsg(1, $dl, "stopDemuxers: demuxers -> ".$result);

  return ($result, $response);
}


###############################################################################
#
# Ask the nexus to stop
#
sub stopNexus($) 
{

  my ($utc_stop) = @_;

  Dada::logMsg(2, $dl, "stopNexus(".$utc_stop.")");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;
  
  Dada::logMsg(2, $dl, "stopNexus: opening connection to ".$pwcc_host.":".$pwcc_port);
  $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);
  if (!$handle) {
    Dada::logMsg(0, $dl, "stopNexus: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue STOP <UTC_STOP>");
  }

   # Ignore the "welcome" message
  $ignore = <$handle>;
  
  $cmd = "stop ".$utc_stop;
  
  Dada::logMsg(1, $dl, "stopNexus: nexus <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  Dada::logMsg(1, $dl, "stopNexus: nexus -> ".$result." ".$response);

  if ($result ne "ok") { 
    Dada::logMsg(0, $dl, "stopNexus: ".$cmd." failed: ".$response);
    $response = $cmd." command failed on nexus";
  }

  $handle->close();

  return ($result, $response);

}

###############################################################################
#
# stop the observation in a background thread.
#
sub stopInBackground() {

  Dada::logMsg(2, $dl, "stopInBackground()");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;
  my $i = 0;

  Dada::logMsg(2, $dl, "stopInBackground: opening connection to ".$pwcc_host.":".$pwcc_port);
  $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);
  if (!$handle) {
    Dada::logMsgWarn($error, "Could not connect to nexus to wait for IDLE state");
    $current_state = "Error";
    Dada::logMsg(2, $dl, "stopInBackground:  exiting");
    return 1;
  }

   # Ignore the "welcome" message
  $ignore = <$handle>;

  # Check we are in the IDLE state before continuing
  Dada::logMsg(1, $dl, "stopInBackground: nexus waiting for return to idle state");
  if (Dada::waitForState("idle", $handle, 40) != 0) {
    Dada::logMsgWarn($error, "stopInBackground: nexus was not in the idle state after 40 seconds");
    $current_state = "Error";
  } else {
    Dada::logMsg(1, $dl, "stopInBackground: nexus now in idle state");
    $current_state = "Stopped";
  }

  # Close nexus connection
  $handle->close();

  Dada::logMsg(2, $dl, "stopInBackground:  exiting");
  return 0;

}

###############################################################################
#
# Send threaded commands to each of the demuxers
#
sub threadedDemuxerCommand($) {

  my ($cmd) = @_;

  my $i = 0;
  my @threads = ();
  my @results = ();
  my $host = "";
  my $port = $cfg{"DEMUX_CONTROL_PORT"};
  
  # start a thread for each demuxer 
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    $host = $cfg{"DEMUX_".$i};
    Dada::logMsg(2, $dl, "threadedDemuxerCommand: sending ".$cmd." to ".$host.":".$port);
    $threads[$i] = threads->new(\&demuxerThread, $host, $port, $cmd);
  }

  # join each thread
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    Dada::logMsg(2, $dl, "threadedDemuxerCommand: joining thread ".$i);
    $results[$i] = $threads[$i]->join();
  }

  Dada::logMsg(2, $dl, "threadedDemuxerCommand: all threads joined");

  # check the results
  my $overall_result = "ok";
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    if ($results[$i] ne "ok") {
      Dada::logMsgWarn($error, "threadedDemuxerCommand: ".$cmd." on ".
                       $cfg{"DEMUX_".$i}.":".$port." failed");
      $overall_result = "fail";
    }
  }
  return ($overall_result, ""); 

}

###############################################################################
#
# sends a command to the specified machine:port
#
sub demuxerThread($$$) {

  my ($host, $port, $cmd) = @_;
  
  my $handle = 0;
  my $result = "";
  my $response = "";

  $handle = Dada::connectToMachine($host, $port);
  if (!$handle) {
    Dada::logMsgWarn($error, "demuxerThread: could not connect to ".$host.":".$port);
    return "fail";
  }

  Dada::logMsg(2, $dl, $host.":".$port." <- ".$cmd);
  ($result,$response) = Dada::sendTelnetCommand($handle, $cmd);
  Dada::logMsg(2, $dl, $host.":".$port." -> ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($error, "demuxerThread: ".$cmd." failed: ".$response);
    return "fail";
  }

  return "ok";

}

###############################################################################
# 
# trim preceeding whitespace
# 
sub ltrim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  return $string;
}

sub getParameterFromArray($\@) {
                                                                                
  (my $parameter, my $arrayRef) = @_;
                                                                                
  my @array = @$arrayRef;

  # Generate the key/value combinations for the specification
  my @arr;
  my $line;
  my $value = "";
  my $i = 0;

  for ($i=0; $i<=$#array; $i++) {
    $line = $array[$i];

    # strip and comments
    $line =~ s/#.*//;

    if ($line =~ m/^$parameter /) {
      @arr = split(/ +/,$line);
      $value = $arr[1];
      chomp($value);
    }

  }
  return $value;
}


#
# Checks that TCS supplied us with the MINIMAL set of commands 
# necessary to run 
# 
sub parseTCSCommands() {

  my $result = "ok";
  my $response = "";

  my @cmds = qw(SOURCE RA DEC RECEIVER CFREQ PID NBIT NDIM NPOL BANDWIDTH PROC_FILE MODE RESOLUTION);
  my $cmd;

  if (exists $tcs_cmds{"MODE"}) {
    if ($tcs_cmds{"MODE"} eq "CAL") {
      push(@cmds, "CALFREQ");
    }
  }

  foreach $cmd (@cmds) {
    if (!(exists $tcs_cmds{$cmd})) {
      $result = "fail";
      $response .= " ".$cmd;
    } 
    elsif (!(defined $tcs_cmds{$cmd}))
    {
      $result = "fail";
      $response .= " ".$cmd;
    }
    elsif ($tcs_cmds{$cmd} eq "")
    {
      $result = "fail";
      $response .= " ".$cmd;
    }
    else
    {
      Dada::logMsg(2, $dl, "parseTCSCommands: found header parameter ".$cmd);
    }

  }
  if ($result eq "fail") {
    Dada::logMsg(0, $dl, "parseTCSCommands: missing header parameter(s) ".$response);
    return ("fail", "Missing Parameter(s) ".$response);
  }

  # Check that the PROC_FILE exists in the CONFIG_DIR
  if (! -f $cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"} ) {
    Dada::logMsg(0, $dl, "parseTCSCommands: PROC_FILE [".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}."] did not exist");
    return ("fail", "PROC_FILE ".$tcs_cmds{"PROC_FILE"}." did not exist");
  }

  if (!($tcs_cmds{"RECEIVER"} =~ m/MULTI/)) {
    Dada::logMsg(0, $dl, "parseTCSCommands: cannot observe with anything other than the multibeam received @ 20cm");
    return ("ok", "Wrong receiver");
  }

  # Check the the PID is valid
  my $caspsr_groups = `groups caspsr`;
  chomp $caspsr_groups;
  my $proj_id = $tcs_cmds{"PID"};
  if (!($caspsr_groups =~ m/$proj_id/)) {
    Dada::logMsg(0, $dl,  "parseTCSCommands: PID [".$proj_id."] was invalid");
    return ("fail", "PID [".$proj_id."] was an invalid CASPSR Project ID");
  }

  my $source   = $tcs_cmds{"SOURCE"};
  my $mode     = $tcs_cmds{"MODE"};
  my $proc_bin = "";

  $cmd = "grep PROC_CMD ".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}." | awk '{print \$2}'";
  Dada::logMsg(2, $dl, "parseTCSCommands: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "parseTCSCommands: ".$result." ".$response);

  if ($result ne "ok") { 
    Dada::logMsg(0, $dl,  "parseTCSCommands: could not extract the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
    return ("fail", "could not determine the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
  }

  $proc_bin = $response;
  if (!-f $cfg{"SCRIPTS_DIR"}."/".$proc_bin) {
    Dada::logMsg(0, $dl,  "parseTCSCommands: binary [".$proc_bin."] specified in PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] did not exist in the bin dir [".$cfg{"SCRIPTS_DIR"}."]");
    return ("fail", "PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] contain an invalid binary");
  }

  # Check the SOURCE, MODE make sense for a DSPSR based PROC_FILE
  if (($mode eq "PSR") && ($proc_bin =~ m/dspsr/)) {
    my $dm = Dada::getDM($source);
    if ($dm eq "NA") {
      Dada::logMsg(0, $dl,  "parseTCSCommands: SOURCE [".$source."] did not exist in CASPSR's psrcat catalogue or the tempo tzpar directory");
      return ("fail", "SOURCE [".$source."] did not exist in CASPSRs catalogue");
    }
  }

  # check for MULTI fold 
  if ($tcs_cmds{"PROC_FILE"} eq "dspsr.multi") {

    my $short_source = $source;
    $short_source =~ s/^[JB]//;
    $short_source =~ s/[a-zA-Z]*$//;

    # find the source in multi.txt
    $cmd = "grep ^".$short_source." ".$cfg{"CONFIG_DIR"}."/multi.txt";
    my $multi_string = `$cmd`;
    if ($? != 0) {
      Dada::logMsg(0, $dl,  "parseTCSCommands: SOURCE [".$short_source."] did not exist in ".$cfg{"CONFIG_DIR"}."/multi.txt");
      return ("fail", "SOURCE [".$source."] did not exist in CASPSRs multifold list");

    } else {

      chomp $multi_string;
      my @multis = split(/ +/,$multi_string);

      if (! -f $cfg{"CONFIG_DIR"}."/".$multis[2]) {
        Dada::logMsg(0, $dl,  "parseTCSCommands: Multi-source file [".$cfg{"CONFIG_DIR"}."/".$multis[2]."] did not exist");
        return ("fail", "The multifold source file [".$multis[2]."] did not exist");
      }
    }
  }

  return ("ok", "");

}


sub fixTCSCommands(\%) {

  my ($tcs_cmds_ref) = @_;

  my %cmds = %$tcs_cmds_ref;

  my %fix = ();
  $fix{"src"} = "SOURCE";
  $fix{"ra"} = "RA";
  $fix{"dec"} = "DEC";
  $fix{"band"} = "BANDWIDTH";
  $fix{"freq"} = "CFREQ";
  $fix{"procfil"} = "PROC_FILE";
  $fix{"pid"} = "PID";
  
  my %add = (); 
  $add{"MODE"} = "PSR";
  $add{"CALFREQ"} = "11.123000";
  $add{"NBIT"} = "8";
  $add{"NPOL"} = "2";
  $add{"NDIM"} = "1";
  $add{"RECEIVER"} = "MULTI";
  $add{"RESOLUTION"} = "1";

  my %new_cmds = ();

  my $key = "";

  foreach $key (keys (%cmds)) {

    if (exists $fix{$key}) {
      $new_cmds{$fix{$key}} = $cmds{$key};
    } else {
      $new_cmds{$key} = $cmds{$key};
    }
  }
  
  # kludge for 50cm observing
  # always set bandwidth to -400 irrespective of TCS
  $new_cmds{"BANDWIDTH"} = "-400";
  if (($new_cmds{"CFREQ"} > 1200) && ($new_cmds{"CFREQ"} < 1500)) {
     $new_cmds{"CFREQ"} = "1382";
  }
  if (($new_cmds{"CFREQ"} > 500) && ($new_cmds{"CFREQ"} < 800)) {
     $new_cmds{"CFREQ"} = "628";
  }

  if (($new_cmds{"SOURCE"} =~ m/_R$/) || ($new_cmds{"SOURCE"} =~ m/HYDRA/)) {
    $add{"MODE"} = "CAL";
  }

  foreach $key (keys (%add)) {
    if (!(exists $new_cmds{$key})) {
      $new_cmds{$key} = $add{$key};
    }
  }
  
  return %new_cmds;
}


#
# Addds the required keys,values to the TCS commands based
# on the hardwired constants of the DFB3. These are:
#
# 1. Always sending to 16 PWCs
# 2. Lowest freq band goes to apsr00, highest to apsr15
# 3. 16 Bands (for the moment)
#
sub addSiteConfig() {

  Dada::logMsg(2, $dl, "addSiteConfig()");

  my $key = "";
  my $bw = 0;
  my $i = 0;

  $tcs_cmds{"NUM_PWC"}     = $cfg{"NUM_PWC"};
  $tcs_cmds{"HDR_SIZE"}    = $site_cfg{"HDR_SIZE"};
  $tcs_cmds{"BW"}          = $tcs_cmds{"BANDWIDTH"};
  $tcs_cmds{"FREQ"}        = $tcs_cmds{"CFREQ"};

  # Add the site configuration to tcs_cmds
  foreach $key (keys (%site_cfg)) {
    $tcs_cmds{$key} = $site_cfg{$key};
  }

  $bw = $tcs_cmds{"BW"};
  # Determine the TSAMP based upon NDIM and BW 
  if ($bw == 0.0) {
    $tcs_cmds{"TSAMP"} = 0.0
  } else {
    $tcs_cmds{"TSAMP"} = (1.0 / abs($bw)) * ($tcs_cmds{"NDIM"} / 2);
  }

  # number of channels and bands hardcoded to 1
  $tcs_cmds{"NBAND"} = 1;
  $tcs_cmds{"NCHAN"} = 1;

  # Set the instrument
  $tcs_cmds{"INSTRUMENT"} = uc($cfg{"INSTRUMENT"});
    
}


#
# Generates the config file required for dada_pwc_command
#
sub generateConfigFile($) {

  my ($fname) = @_;

  my $string = "";
  
  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);

  print FH "# Header file created by ".$daemon_name."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  $string = Dada::headerFormat("NUM_PWC", $cfg{"NUM_PWC"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  # Port information for dada_pwc_command
  $string = Dada::headerFormat("PWC_PORT", $cfg{"PWC_PORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWC_LOGPORT", $cfg{"PWC_LOGPORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWCC_PORT", $cfg{"PWCC_PORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWCC_LOGPORT", $cfg{"PWCC_LOGPORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("LOGFILE_DIR", $cfg{"SERVER_LOG_DIR"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("HDR_SIZE", $site_cfg{"HDR_SIZE"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("COM_POLL", "10");
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  my $i=0;
  for($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $string = Dada::headerFormat("PWC_".$i, $cfg{"PWC_".$i});
    print FH $string."\n";
    Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);
  }
  close FH;

  return ("ok", "");

}

#
# Generate the specification file used in the dada_pwc_command's CONFIG command
#
sub generateSpecificationFile($) {

  my ($fname) = @_;

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Specification File created by ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  my %ignore = ();
  $ignore{"NUM_PWC"} = "yes";
  my $i=0;
  for ($i=0; $i<$tcs_cmds{"NUM_PWC"}; $i++) {
    $ignore{"PWC_".$i} = "yes";
  }

  # Print the keys
  my @sorted = sort (keys %tcs_cmds);

  my $line;
  foreach $line (@sorted) {
    if (!(exists $ignore{$line})) {
      print FH Dada::headerFormat($line, $tcs_cmds{$line})."\n";
      Dada::logMsg(2, $dl, $tcs_spec_file." ".Dada::headerFormat($line, $tcs_cmds{$line}));
    }
  }

  close FH;
  return ("ok","");
}
  

#
# delete all the files in the STATUS_DIR
#
sub clearStatusDir() {

  my $cmd = "";
  my $result = "";
  my $response = "";

  # Clear the /apsr/status files
  $cmd = "rm -f ".$cfg{"STATUS_DIR"}."/*";
  
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could clean STATUS_DIR: ".$response);
  }

}


###############################################################################
#
# Thread that issues the STOP command after the specified number of seconds have
# elapsed
#
sub utcStopCommand() {

  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl ,"utcStopCommand: stopping in ".$utc_stop_remaining." seconds");

  while ( (!$quit_daemon) && ($utc_stop_remaining > 0) && 
          (($current_state =~ m/Starting/) || ($current_state =~ m/Recording/)) ) {

    sleep 1;
    $utc_stop_remaining--;
    Dada::logMsg(2, $dl ,"utcStopCommand: ".$utc_stop_remaining." seconds remaining");

  }

  # if we have successfully timed out 
  if ( ($utc_stop_remaining == 0) && ($current_state =~ m/Recording/) ) {
    my $dud_handle = 0;
    Dada::logMsg(2, $dl ,"utcStopCommand: processTCSCommand('stop', ".$dud_handle.")");
    ($result, $response) = processTCSCommand("stop", $dud_handle);
    Dada::logMsg(2, $dl ,"utcStopCommand: processTCSCommand ".$result." ".$response);
  } else {
    Dada::logMsg(1, $dl ,"utcStopCommand: waiting loop ended prematurely");
  }

  $utc_stop_remaining = -1;

  return 0;

}


###############################################################################
#
#
#
sub controlThread($$) {

  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  Dada::logMsg(2, $dl ,"controlThread: quit detected");

  # Manually tell dada_pwc_command to quit
  quitPWCCommand();

  Dada::logMsg(2, $dl ,"controlThread: PWCC has exited");

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(2, $dl ,"controlThread: exiting");

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(5);
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

} 

#
# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;
  my $cmd = "";
  my $result = "";
  my $response = "";

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  # check IBOB connectivity
  $cmd = "ping -q -c 1 ".IBOB_CONTROL_IP;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    return ("fail", "Could not ping IBOB at ".IBOB_CONTROL_IP." - try manual reset of IBOB");
  }

  # check IBOB response to commands
  $cmd = "echo 'regread ip_ctr/reg_num_ips\nquit' | bibob_terminal ".
         IBOB_CONTROL_IP." ".IBOB_CONTROL_PORT;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    return ("fail", "Could not interact with IBOB at ".IBOB_CONTROL_IP.":".
                    IBOB_CONTROL_PORT." - try manual reset of IBOB");
  }

  # check that the response is correct
  if (!($response =~ m/Received response in 63 bytes/))
  {
    return ("fail", "IBOB did not response to a control command, try ".
            "manual reset of IBOB");
  }

  # check the Demux nodes are receiving 800MB/s each on eth2
  my $handle = Dada::connectToMachine("demux0", "8650");
  my @bits = ();
  my @rates = ();
  my $lines_parsed = 0;
  my $lines_matched = 0;
  while ($result = <$handle>) 
  {
    $lines_parsed++;
    if ($result =~ m/\"bytes_in\"/)
    { 
      $lines_matched++;
      @bits = split(/ /,$result);
      @bits = split(/"/,$bits[2]);
      push @rates, $bits[1];
    }
  }
  $handle->close();
  if ($#rates == 1)
  {
    if (($rates[0] < 800000000) || ($rates[1] < 800000000))
    {
      return ("fail", "Demuxer nodes not receiving 800MB/s each, reset IBOB");
    }
  }
  else
  {
    return ("fail", "could not extract input data rate metrics from ganglia on demux0");
  }

  $tcs_sock = new IO::Socket::INET (
    LocalHost => $tcs_host,
    LocalPort => $tcs_port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1
  );
  if (!$tcs_sock) {
    return ("fail", "Could not create listening socket: ".$tcs_host.":".$tcs_port);
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  # clear any warnings or errors associated with this daemon
  if (-f $warn)
  {
    unlink $warn;
  }
  if (-f $error)
  {
    unlink $error;
  }

  return ("ok", "");
}
