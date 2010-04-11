package Dada::server_tcs_interface;

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use Switch;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $daemon_name %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $daemon_name;
our $tcs_cfg_file;
our $tcs_spec_file;
our %cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $warn;
our $error;
our $pwcc_running : shared;
our $current_state : shared;
our $use_dfb_simulator;
our $dfb_sim_host;
our $dfb_sim_port;
our $dfb_sim_dest_port;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our %tcs_cmds;
our %site_cfg;
our $pwcc_thread;
our $tcs_host;
our $tcs_port;
our $tcs_sock;

#
# initialize package globals
#
$dl = 1; 
$daemon_name = 0;
$tcs_cfg_file = "";
$tcs_spec_file = "";
%cfg = ();

#
# initialize other variables
#
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$tcs_host = "";
$tcs_port = 0;
$tcs_sock = 0;
$pwcc_running = 0;
$current_state = "";
$use_dfb_simulator = "";
$dfb_sim_host = "";
$dfb_sim_port = 0;
$dfb_sim_dest_port = 0;
$pwcc_host = "";
$pwcc_port = 0;
$client_master_port = 0;
%tcs_cmds = ();
%site_cfg = ();
$pwcc_thread = 0;


use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant DFBSIM_DURATION    => "3600";    # Simulator runs for 1 hour
use constant TERMINATOR         => "\r";
use constant NHOST              => 16;        # This is constant re DFB3

###############################################################################
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
  $use_dfb_simulator      = $cfg{"USE_DFB_SIMULATOR"};
  $dfb_sim_host           = $cfg{"DFB_SIM_HOST"};
  $dfb_sim_port           = $cfg{"DFB_SIM_PORT"};
  $dfb_sim_dest_port      = $cfg{"DFB_SIM_DEST_PORT"};
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

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  %tcs_cmds = ();
  %site_cfg = Dada->readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);

  # set initial state
  $current_state = "Idle";

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada->daemonize($log_file, $pid_file);

  Dada->logMsg(0, $dl, "STARTING SCRIPT");

  # start the control thread
  Dada->logMsg(2, "INFO", "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  foreach $key (keys (%site_cfg)) {
    Dada->logMsg(2, $dl, "site_cfg: ".$key." => ".$site_cfg{$key});
  }

  # generate the cfg file required to launch dada_pwc_command 
  ($result, $response) = generateConfigFile($tcs_cfg_file);

  # Launch a persistent dada_pwc_command with the $tcs_cfg_file
  $pwcc_thread = threads->new(\&pwccThread);

  # start the stateThread
  Dada->logMsg(2, $dl, "main: stateThread()");
  $state_thread = threads->new(\&stateThread);

  my $read_set = new IO::Select();
  $read_set->add($tcs_sock);

  Dada->logMsg(2, $dl, "main: listening for TCS connection ".$tcs_host.":".$tcs_port);

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
          Dada->logMsgWarn($warn, "Rejecting additional connection from ".$hostinfo->name);

        } else {

          # Wait for a connection from the server on the specified port
          $handle = $rh->accept();
          $handle->autoflush(1);
          $read_set->add($handle);

          # Get information about the connecting machine
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          $hostname = $hostinfo->name;

          Dada->logMsg(1, $dl, "Accepting connection from ".$hostname);
          $tcs_connected = 1;
          $handle = 0; 

        }

      # we have received data on the current read handle
      } else {

        $command = <$rh>;

        # If we have lost the connection...
        if (! defined $command) {

          Dada->logMsg(1, $dl, "Lost TCS connection from ".$hostname);
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
            Dada->logMsgWarn($warn, "Received empty string from TCS");
            print $rh "ok".TERMINATOR;
          }
        }
      }
    }
  }

  Dada->logMsg(0, $dl, "main: joining threads");

  # rejoin threads
  $control_thread->join();
  $pwcc_thread->join();
  $state_thread->join();

  Dada->logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;
}



sub processTCSCommand($$) {

  my ($cmd, $handle) = @_;

  my $result = "";
  my $response = "";
  my $key = "";
  my $val = "";
  my $lckey = "";
  my $utc_start = "UNKNOWN";

  ($key, $val) = split(/ +/,$cmd,2);

  $lckey = lc $key;

  Dada->logMsg(1, $dl, "TCS -> ".$cmd);
  if ($key eq "PROCFIL") {
    $key = "PROC_FILE";
  }

  switch ($lckey) {

    case "bat" {
      Dada->logMsg(2, $dl, "Processing BAT command");

      if ($current_state ne "Recording") {
        Dada->logMsg(1, $dl, "TCS <- fail");
        print $handle "fail".TERMINATOR;

      } else {

        Dada->logMsg(1, $dl, "TCS <- ok");
        print $handle "ok".TERMINATOR;

        $cmd = "bat_to_utc ".$val;
        Dada->logMsg(2, $dl, "processTCSCommand: ".$cmd);
        ($result, $response) = Dada->mySystem($cmd);
        Dada->logMsg(2, $dl, "processTCSCommand: ".$result." ".$response);

        if ($result ne "ok") {
          Dada->logMsg(0, $dl, "processTCSCommand: ".$cmd." failed: ".$response);
          $response = "bat_to_utc failed: ".$response;

        } else {

          $utc_start = ltrim($response);
          ($result, $response) = set_utc_start($utc_start);

          # After the utc has been set, we can reset the tcs_cmds
          %tcs_cmds = ();
        }
      }
    }

    case "start" {

      Dada->logMsg(2, $dl, "Processing START command");

      # Check that %tcs_cmds has all the required parameters in it
      ($result, $response) = parseTCSCommands();

      # Send response to TCS
      Dada->logMsg(1, $dl, "TCS <- ".$result);
      print $handle $result.TERMINATOR;

      if ($result eq "ok") {

        # clear the status directory
        clearStatusDir();

        # Add site.config parameters to the tcs_cmds;
        addSiteConfig();

        # check that the PWCC is actually running
        if (!$pwcc_running) {

          Dada->logMsgWarn($warn, "PWCC thread was not running, attemping to relaunch");
          $pwcc_thread->join();
          Dada->logMsg(0, $dl, "pwcc_thread was joined");
          $pwcc_thread = threads->new(\&pwccThread);
          Dada->logMsg(0, $dl, "pwcc_thread relaunched");

        }


        # Create spec file for dada_pwc_command
        ($result, $response) = generateSpecificationFile($tcs_spec_file);

        # Issue the start command itself
        ($result, $response) = start($tcs_spec_file);

        if ($result eq "fail") {

          # Dada->logMsgWarn($error, "start command failed: ".$response);
          $current_state = "Failed to start";

        } else {
          Dada->logMsg(2, $dl, "Start command successful ".$response);
          $current_state = "Recording";

          # If simulating, start the DFB simulator
          if ($use_dfb_simulator) {

            ($result, $response) = startDFB();

            if ($result eq "ok") {

              $utc_start = $response;
              Dada->logMsg(1, $dl, "UTC_START = ".$utc_start);
              ($result,$response) = set_utc_start($response);
              Dada->logMsg(1, $dl, "set_utc_start: ".$result.":".$response);
            }
            %tcs_cmds = ();
          }

        }
      } else {
        $current_state = "TCS Config Error: ".$response;
      }
    }

    case "stop" {

      Dada->logMsg(2, $dl, "Processing STOP command");
      ($result, $response) = stop();

      # Stop the simulator (if not using DFB3)
      if ($use_dfb_simulator) {
        my $dfbstopthread = threads->new(\&stopDFBSimulator, $dfb_sim_host);
        $dfbstopthread->detach();
      }

      $current_state = "Idle";
      $utc_start = "";

    }

    case "set_utc_start" {
      Dada->logMsg(2, $dl, "Processing SET_UTC_START command");
      $utc_start = $val;
      ($result, $response) = set_utc_start(ltrim($utc_start), \%tcs_cmds);
      %tcs_cmds = ();
    }
  
    case "quit" {

    } 

    # This should be a header parameter, add it to the tcs_cmds hash
    else {

      $tcs_cmds{$key} = $val;
      $result = "ok";
      $response = "";
      $current_state = "Preparing";

      if (($key eq "BANDWIDTH") && ($val eq "0.000000")) {
        $result = "fail";
        $response = "cowardly refusing to observe with bandwidth=0.0";
        $current_state = "Error";
      }

    }
  }

  # If the command failed, log it
  if ($result eq "fail") {
    Dada->logMsgWarn($error, $response);
  }

  # Special "hack" case as we return "ok" to a start
  # command without waiting
  if (!(($lckey eq "start") || ($lckey eq "bat") )) {

    if ($result eq "fail") {
      $current_state = "Error";
      print $handle $result.TERMINATOR;
      print $handle $response.TERMINATOR;

      if ($use_dfb_simulator) {
        my $dfbstopthread2 = threads->new(\&stopDFBSimulator, $dfb_sim_host);
        $dfbstopthread2->detach();
      }

    } else {
      print $handle $result.TERMINATOR;
      Dada->logMsg(1, $dl, "TCS <- ".$result);
    }
  }

}
  
#
# Runs dada_pwc_command in non daemon mode. All ouput should be logged to
# the log file specified
#
sub pwccThread() {

  Dada->logMsg(1, $dl, "pwccThread: starting");

  my $result = "";
  my $response = "";
  my $cmd = "";

  if ( -f $tcs_cfg_file ) {

    $cmd = "dada_pwc_command ".$tcs_cfg_file." >> ".$cfg{"SERVER_LOG_DIR"}."/dada_pwc_command.log";
    Dada->logMsg(2, $dl, "pwccThread: ".$cmd);
    $pwcc_running = 1;
    ($result, $response) = Dada->mySystem($cmd);
    $pwcc_running = 0;
    Dada->logMsg(2, $dl, "pwccThread: ".$result." ".$response);

    if ($result ne "ok") {
      Dada->logMsgWarn($warn, "pwccThread: dada_pwc_command failed");
    }

  } else {
    Dada->logMsgWarn($warn, "pwccThread: tcs config file did not exist: ".$tcs_cfg_file);
  }

  Dada->logMsg(1, $dl, "pwccThread: exiting");
  return ($result);
}




#
# Opens a socket and reports the current state of the PWCC
#
sub stateThread() {

  Dada->logMsg(1, $dl, "stateThread: starting");

  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"TCS_STATE_INFO_PORT"};
  my $read_set = 0;
  my $handle = 0;
  my $line = "";
  my $rh = 0;
  my $hostname = "";
  my $hostinfo = 0;

  # open the listening socket
  Dada->logMsg(2, $dl, "stateThread: opening socket ".$host.":".$port);
  my $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );

  if (!$sock) {
    Dada->logMsgWarn($error, "stateThread: could not create socket ".$host.":".$port);
    return 1;
  }

  Dada->logMsg(2, $dl, "stateThread: listening socket opened ".$host.":".$port);

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

        Dada->logMsg(3, $dl, "stateThread: Accepting connection from ".$hostname);
        $read_set->add($handle);
        $handle = 0;

      } else {

        $line = Dada->getLine($rh);

        if (! defined $line) {
          Dada->logMsg(3, $dl, "stateThread: closing read handle");
          $read_set->remove($rh);
          close($rh);

        } else {

          Dada->logMsg(3, $dl, "stateThread: received ".$line);
          if ($line eq "state") {
            print $rh $current_state."\r\n";
            Dada->logMsg(3, $dl, "stateThread: replied ".$current_state);
          }
        }
      }
    }
  }

  Dada->logMsg(2, $dl, "stateThread: exiting");
}



sub quitPWCCommand() {
  
  my $handle = 0;
  my $result = "";
  my $response = "";
  
  Dada->logMsg(2, $dl, "quitPWCCommand()");

  if (! $pwcc_running) {
    Dada->logMsg(2, $dl, "quitPWCCommand: dada_pwc_command not running");
    return ("ok", "");

  } else {

    Dada->logMsg(2, $dl, "quitPWCCommand: connecting to dada_pwc_command: ".$pwcc_host.":".$pwcc_port);
    $handle = Dada->connectToMachine($pwcc_host, $pwcc_port);

    if ($handle) {

      # Ignore the "welcome" message
      $response = <$handle>;

      # Send quit command
      Dada->logMsg(2, $dl, "quitPWCCommand: sending quit to dada_pwc_command");
      print $handle "quit\r\n";
      $handle->close();

      # wait 2 seconds for the nexus to quite
      my $nwait = 2;
      while (($pwcc_running) && ($nwait > 0)) {
        sleep(1);
        $nwait--;
      }

      if ($pwcc_running) {
        Dada->logMsgWarn($warn, "Had to kill dada_pwc_command");
        ($result, $response) = Dada->killProcess("dada_pwc_command ".$tcs_cfg_file);
        Dada->logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      }

      return ("ok","");

    # try to kill the process manually
    } else {
      Dada->logMsgWarn($warn, "quitPWCCommand: could not connect to dada_pwc_command");
      ($result, $response) = Dada->killProcess("dada_pwc_command ".$tcs_cfg_file);
      Dada->logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      return ($result, $response);
    }

  }
}


#
# Send the START command to the pwcc, optionally starting a DFB simualtor
#
sub start($) {
                                                                              
  my ($file) = @_;

  my $rVal = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";

  # If we will run a separate DFB simulator
  if ($use_dfb_simulator) {

    # ARGS: host, dest port, nbit, npol, mode, duration 
    ($result, $response) = createDFBSimulator();

    # Give it half a chance to startup
    sleep(2);

  }

  while (! $pwcc_running) {
    Dada->logMsg(0, $dl, "start: waiting for dada_pwc_command to start");
    sleep(1);
  }
  sleep(1);

  # Connect to dada_pwc_command
  my $handle = Dada->connectToMachine($pwcc_host, $pwcc_port, 5);

  if (!$handle) {
    Dada->logMsg(0, $dl, "start: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue START command"); 

  } else {

    # Ignore the "welcome" message
    $result = <$handle>;

    # Check we are in the IDLE state before continuing
    if (Dada->waitForState("idle", $handle, 5) != 0) {
      Dada->logMsg(0, $dl, "start: nexus was not in the idle state after 5 seconds"); 
      return ("fail", "nexus was not in IDLE state");
    }

    # Send CONFIG command with apsr_tcs.spec
    $cmd = "config ".$file;
    Dada->logMsg(1, $dl, "nexus <- ".$cmd);
    ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
    Dada->logMsg(1, $dl, "nexus -> ".$result." ".$response);
    if ($result ne "ok") { 
      Dada->logMsg(0, $dl, "start: config command failed: ".$response);
      return ("fail", "CONFIG command failed on nexus: ".$response)
    }

    # Wait for the PREPARED state
    if (Dada->waitForState("prepared",$handle,10) != 0) {
      Dada->logMsg(0, $dl, "start: nexus did not enter PREPARED state 10 seconds after config command");
      return ("fail", "nexus did not enter PREPARED state");
    }

    Dada->logMsg(2, $dl, "Nexus now in PREPARED state");

    # Send start command 
    $cmd = "start";

    Dada->logMsg(1, $dl, "nexus <- ".$cmd);
    ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
    Dada->logMsg(1, $dl, "nexus -> ".$result." ".$response);

    if ($result ne "ok") { 
      Dada->logMsg(0, $dl, "start: start command failed: ".$response);
      return ("fail", "START command failed on nexus: ".$response);
    }

    # Wait for the prepared state
    if (Dada->waitForState("recording",$handle,10) != 0) {
      Dada->logMsg(0, $dl, "start: nexus did not enter RECORDING state 10 seconds after start command");
      return ("fail", "nexus did not enter RECORDING state");
    }

    Dada->logMsg(2, $dl, "Nexus now in RECORDING state");

    # Close nexus connection
    $handle->close();

    if ($result eq "ok") {
      $response = "START successful";
    }

    return ($result, $response);
  }

}


#
# Sends a UTC_START command to the pwcc
#
sub set_utc_start($) {

  my ($utc_start) = @_;

  Dada->logMsg(2, $dl, "set_utc_start(".$utc_start.")");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";

  # Now that we know the UTC_START, create the required results and archive 
  # directories and put the observation summary file there...

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;
  my $proj_id     = $tcs_cmds{"PID"};

  # Ensure each directory is automounted
  if (!( -d $archive_dir)) {
    `ls $archive_dir >& /dev/null`;
  }

  if (!( -d $results_dir)) {
    `ls $results_dir >& /dev/null`;
  }

  $cmd = "mkdir -p ".$results_dir;
  system($cmd);

  $cmd = "mkdir -p ".$archive_dir;
  system($cmd);

  my $apsr_groups = `groups apsr`;
  chomp $apsr_groups;
  if ($apsr_groups =~ m/$proj_id/) {
    # Do nothing
  } else {
    Dada->logMsgWarn($warn, "PID ".$proj_id." invalid, using apsr instead"); 
    $proj_id = "apsr";
  }

  $cmd = "chgrp -R ".$proj_id." ".$results_dir;
  system($cmd);

  $cmd = "chgrp -R ".$proj_id." ".$archive_dir;
  system($cmd);

  $cmd = "chmod -R g+s ".$results_dir;
  system($cmd);

  $cmd = "chmod -R g+s ".$archive_dir;
  system($cmd);

  my $fname = $results_dir."/obs.info";
  Dada->logMsg(2, $dl, "set_utc_start: creating obs.info \"".$fname."\"");

  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";
  print FH Dada->headerFormat("SOURCE",$tcs_cmds{"SOURCE"})."\n";
  print FH Dada->headerFormat("RA",$tcs_cmds{"RA"})."\n";
  print FH Dada->headerFormat("DEC",$tcs_cmds{"DEC"})."\n";
  print FH Dada->headerFormat("CFREQ",$tcs_cmds{"CFREQ"})."\n";
  print FH Dada->headerFormat("PID",$tcs_cmds{"PID"})."\n";
  print FH Dada->headerFormat("BANDWIDTH",$tcs_cmds{"BANDWIDTH"})."\n";
  print FH Dada->headerFormat("PROC_FILE",$tcs_cmds{"PROC_FILE"})."\n";
  print FH "\n";
  print FH Dada->headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  print FH Dada->headerFormat("NBIT",$tcs_cmds{"NBIT"})."\n";
  print FH Dada->headerFormat("NPOL",$tcs_cmds{"NPOL"})."\n";
  print FH Dada->headerFormat("NDIM",$tcs_cmds{"NDIM"})."\n";
  print FH Dada->headerFormat("CONFIG",$tcs_cmds{"CONFIG"})."\n";
  close FH;

  $cmd = "touch ".$results_dir."/obs.processing";
  system($cmd);

  $cmd = "touch ".$archive_dir."/obs.processing";
  system($cmd);

  $cmd = "cp ".$fname." ".$archive_dir;
  system($cmd);

  $cmd = "sudo -b chown -R apsr ".$results_dir;
  system($cmd);
                                                                              
  $cmd = "sudo -b chown -R apsr ".$archive_dir;
  system($cmd);


  # connect to nexus
  my $handle = Dada->connectToMachine($pwcc_host, $pwcc_port);

  if (!$handle) {
    Dada->logMsg(0, $dl, "set_utc_start: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue SET_UTC_START command"); 
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  # Wait for the prepared state
  if (Dada->waitForState("recording", $handle, 10) != 0) {
    Dada->logMsg(0, $dl, "set_utc_start: nexus did not enter RECORDING state 10 seconds after START command");
    $handle->close();
    return ("fail", "nexus did not enter RECORDING state");
  }

  # Send UTC Start command to the nexus
  $cmd = "set_utc_start ".$utc_start;

  Dada->logMsg(1, $dl, "nexus <- ".$cmd);
  ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
  Dada->logMsg(1, $dl, "nexus -> ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsg(0, $dl, "set_utc_start: nexus returned ".$response." after sending ".$cmd);
  }

  $handle->close();

  return ($result, $response);

}


#
# Sends the "stop" command to the Nexus
#
sub stop() {

  Dada->logMsg(2, $dl, "stop()");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $handle = 0;

  Dada->logMsg(2, $dl, "stop: connecting to: ".$pwcc_host.":".$pwcc_port);
  $handle = Dada->connectToMachine($pwcc_host, $pwcc_port);

  if (!$handle) {
    Dada->logMsg(0, $dl, "stop: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue STOP command");
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  Dada->logMsg(2, $dl, "stop: nexus <- stop");
  ($result, $response) = Dada->sendTelnetCommand($handle, "stop");
  Dada->logMsg(2, $dl, "stop: nexus -> ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsg(0, $dl, "stop: stop command failed: ".$response);
    $response = "stop command failed on nexus";
  }

  # Close nexus connection
  $handle->close();

  return ($result, $response);

}


sub ltrim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  return $string;
}

sub createDFBSimulator() {

  my $host      =       $cfg{"DFB_SIM_HOST"};
  my $dest_port = "-p ".$cfg{"DFB_SIM_DEST_PORT"};

  my $nbit      = "-b ".$tcs_cmds{"NBIT"};
  my $npol      = "-k ".$tcs_cmds{"NPOL"};
  my $ndim      = "-g ".$tcs_cmds{"NDIM"};
  my $tsamp     = "-t ".$tcs_cmds{"TSAMP"};

  # By default set the "period" of the signal to 6400 bytes;
  my $calfreq   = "-c 6400";

  if ($tcs_cmds{"MODE"} eq "CAL") {

    # if ($tcs_cmds{"NBIT"} eq "2") {
      # START Hack whilst resolution is broken
      # $calfreq    = "-c ".($tcs_cmds{"CALFREQ"}/2.0);
      # $npol       = "-k 1";
      # END  Hack whilst resolution is broken
    # } else {

      # Correct resolution changes
      $calfreq    = "-c ".$tcs_cmds{"CALFREQ"};

    # }

  } else {
    $calfreq    = "-j ".$calfreq; 
  }

  my $drate = $tcs_cmds{"NBIT"} * $tcs_cmds{"NPOL"} * $tcs_cmds{"NDIM"};
  $drate = $drate * (1.0 / $tcs_cmds{"TSAMP"});
  $drate = $drate / 8;    # bits to bytes
  $drate = $drate * 1000000;

  $drate     = "-r ".$drate;
  my $duration  = "-n ".DFBSIM_DURATION;
  my $dest      = "192.168.1.255";
  

  Dada->logMsg(2, $dl, "createDFBSimulator: $host, $dest_port, $nbit, $npol, $ndim, $tsamp, $calfreq, $duration");

  my $args = "-y $dest_port $nbit $npol $ndim $tsamp $calfreq $drate $duration $dest";

  my $result = "";
  my $response = "";

  # Launch dfb simulator on remote host
  my $dfb_cmd = "dfbsimulator -d -v -a ".$args;
  my $handle = Dada->connectToMachine($host, $client_master_port);

  if (!$handle) {
    return ("fail", "Could not connect to client_master_control.pl ".$host.":".$client_master_port);
  }

  Dada->logMsg(2, $dl, "createDFBSimulator: sending cmd ".$dfb_cmd);

  ($result, $response) = Dada->sendTelnetCommand($handle,$dfb_cmd);

  Dada->logMsg(2, $dl, "createDFBSimulator: received reply: (".$result.", ".$response.")");

  $handle->close();

  return ($result, $response);

}


#
# Starts the remote dfb simulator and gets the UTC_START of the first 
# packet sent
#
sub startDFB() {

  my $host = $cfg{"DFB_SIM_HOST"};
  my $port = $cfg{"DFB_SIM_PORT"};

  my $result = "";
  my $response = "";

  Dada->logMsg(2, $dl, "startDFB: ()");

  my $handle = Dada->connectToMachine($host, $port);
  if (!$handle) {
    return ("fail", "Could not connect to apsr_test_triwave ".$host.":".$port);
  }

  Dada->logMsg(2, $dl, "startDFB: sending command \"start\"");
 
  ($result, $response) = Dada->sendTelnetCommand($handle,"start");

  Dada->logMsg(2, $dl, "startDFB: received reply (".$result.", ".$response.")");

  $handle->close();

  return ($result, $response);

}

sub stopDFBSimulator() {

  my $host = $cfg{"DFB_SIM_HOST"};

  my $result = "";
  my $response = "";

  Dada->logMsg(1, $dl, "stopDFBSimulator: stopping simulator in 10 seconds");
  sleep(10);

  # connect to client_master_control.pl
  my $handle = Dada->connectToMachine($host, $client_master_port);

  if (!$handle) {
    return ("fail", "Could not connect to client_master_control.pl ".$host.":".$client_master_port);
  }

  # This command will kill the udp packet generator
  my $dfb_cmd = "stop_dfbs";

  ($result, $response) = Dada->sendTelnetCommand($handle,$dfb_cmd);

  Dada->logMsg(1, $dl, "stopDFBSimulator: received reply: ".$response);

  $handle->close();

  return ($result, $response);
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

  my @cmds = qw(CONFIG SOURCE RA DEC RECEIVER CFREQ PID NBIT NDIM NPOL BANDWIDTH PROC_FILE MODE);
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
  }
  if ($result eq "fail") {
    Dada->logMsg(0, $dl, "parseTCSCommands: missing header parameter(s) ".$response);
    return ("fail", "Missing Parameter(s) ".$response);
  }

  # Check that the PROC_FILE exists in the CONFIG_DIR
  if (! -f $cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"} ) {
    Dada->logMsg(0, $dl, "parseTCSCommands: PROC_FILE [".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}."] did not exist");
    return ("fail", "PROC_FILE ".$tcs_cmds{"PROC_FILE"}." did not exist");
  }

  # Check the the PID is valid
  my $apsr_groups = `groups apsr`;
  chomp $apsr_groups;
  my $proj_id = $tcs_cmds{"PID"};
  if (!($apsr_groups =~ m/$proj_id/)) {
    Dada->logMsg(0, $dl,  "parseTCSCommands: PID [".$proj_id."] was invalid");
    return ("fail", "PID [".$proj_id."] was an invalid APSR Project ID");
  }

  my $source   = $tcs_cmds{"SOURCE"};
  my $mode     = $tcs_cmds{"MODE"};
  my $proc_bin = "";

  $cmd = "grep PROC_CMD ".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}." | awk '{print \$2}'";
  Dada->logMsg(2, $dl, "parseTCSCommands: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "parseTCSCommands: ".$result." ".$response);

  if ($result ne "ok") { 
    Dada->logMsg(0, $dl,  "parseTCSCommands: could not extract the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
    return ("fail", "could not determine the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
  }

  $proc_bin = $response;
  if (!-f $cfg{"SCRIPTS_DIR"}."/".$proc_bin) {
    Dada->logMsg(0, $dl,  "parseTCSCommands: binary [".$proc_bin."] specified in PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] did not exist in the bin dir [".$cfg{"SCRIPTS_DIR"}."]");
    return ("fail", "PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] contain an invalid binary");
  }

  # Check the SOURCE, MODE make sense for a DSPSR based PROC_FILE
  if (($mode eq "PSR") && ($proc_bin =~ m/dspsr/)) {
    my $dm = Dada->getDM($source);
    if ($dm eq "NA") {
      Dada->logMsg(0, $dl,  "parseTCSCommands: SOURCE [".$source."] did not exist in APSR's psrcat catalogue or the tempo tzpar directory");
      return ("fail", "SOURCE [".$source."] did not exist in APSRs catalogue");
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
      Dada->logMsg(0, $dl,  "parseTCSCommands: SOURCE [".$short_source."] did not exist in ".$cfg{"CONFIG_DIR"}."/multi.txt");
      return ("fail", "SOURCE [".$source."] did not exist in APSRs multifold list");

    } else {

      chomp $multi_string;
      my @multis = split(/ +/,$multi_string);

      if (! -f $cfg{"CONFIG_DIR"}."/".$multis[2]) {
        Dada->logMsg(0, $dl,  "parseTCSCommands: Multi-source file [".$cfg{"CONFIG_DIR"}."/".$multis[2]."] did not exist");
        return ("fail", "The multifold source file [".$multis[2]."] did not exist");
      }
    }
  }

  return ("ok", "");

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

  Dada->logMsg(2, $dl, "addSiteConfig()");

  my $key = "";
  my $cf = 0;
  my $tbw = 0;
  my $bw = 0;
  my $i = 0;

  $tcs_cmds{"NUM_PWC"}     = NHOST;
  $tcs_cmds{"HDR_SIZE"}    = $site_cfg{"HDR_SIZE"};

  $cf  = int($tcs_cmds{"CFREQ"});        # centre frequency
  $tbw = int($tcs_cmds{"BANDWIDTH"});    # total bandwidth
  $bw  = $tbw / 16;                      # bandwidth per channel

  for ($i=0; $i<NHOST; $i++)
  {
    $tcs_cmds{"Band".$i."_BW"} = -1 * $bw;
    $tcs_cmds{"Band".$i."_FREQ"} = $cf - ($tbw/2) + ($bw/2) + ($bw*$i);
  }

  # Add the site configuration to tcs_cmds
  foreach $key (keys (%site_cfg)) {
    $tcs_cmds{$key} = $site_cfg{$key};
  }

  # Determine the TSAMP based upon NDIM and BW 
  if ($bw == 0.0) {
    $tcs_cmds{"TSAMP"} = 0.0
  } else {
    $tcs_cmds{"TSAMP"} = (1.0 / abs($bw)) * ($tcs_cmds{"NDIM"} / 2);
  }

  # Setup the number of channels per node
  if (!(exists $tcs_cmds{"NBAND"})) {
    $tcs_cmds{"NBAND"} = 16;
  }
  $tcs_cmds{"NCHAN"} = $tcs_cmds{"NBAND"} / 16;
  
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
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";

  $string = Dada->headerFormat("NUM_PWC", $cfg{"NUM_PWC"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  # Port information for dada_pwc_command
  $string = Dada->headerFormat("PWC_PORT", $cfg{"PWC_PORT"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada->headerFormat("PWC_LOGPORT", $cfg{"PWC_LOGPORT"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada->headerFormat("PWCC_PORT", $cfg{"PWCC_PORT"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada->headerFormat("PWCC_LOGPORT", $cfg{"PWCC_LOGPORT"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada->headerFormat("LOGFILE_DIR", $cfg{"SERVER_LOG_DIR"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada->headerFormat("HDR_SIZE", $site_cfg{"HDR_SIZE"});
  print FH $string."\n";
  Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);

  my $i=0;
  for($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $string = Dada->headerFormat("PWC_".$i, $cfg{"PWC_".$i});
    print FH $string."\n";
    Dada->logMsg(2, $dl, $tcs_cfg_file." ".$string);
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
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";

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
      print FH Dada->headerFormat($line, $tcs_cmds{$line})."\n";
      Dada->logMsg(2, $dl, $tcs_spec_file." ".Dada->headerFormat($line, $tcs_cmds{$line}));
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
  
  ($result, $response) = Dada->mySystem($cmd);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "Could clean STATUS_DIR: ".$response);
  }

}



sub controlThread($$) {

  Dada->logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada->logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  Dada->logMsg(2, $dl ,"controlThread: quit detected");

  # Manually tell dada_pwc_command to quit
  quitPWCCommand();

  Dada->logMsg(2, $dl ,"controlThread: PWCC has exited");

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada->logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada->logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada->logMsg(2, $dl ,"controlThread: exiting");

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

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada->getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada->getHostMachineName());
  }

  if ($cfg{"NUM_PWC"} != NHOST) {
    print STDERR "ERROR: Dada config file's NUM_PWC (".$cfg{"NUM_PWC"}.")".
                 " did not match the expected value of ".NHOST."\n";
    return 1;
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
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}




END { }

1;  # return value from file
