#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  3 Dec, 2007
# Modigied: 11 Mar, 2008
#
# DADA Primary Write Client Controller Controller (no kidding!)
#
# This daemons runs continuously listening for control messages 
# from TCS (Telescope Control System). Messages that control 
# state transition are:
#
# START         Use the header parameters previously recieved and begin the
#               observation with them
# SET_UTC_START Forward the UTC_START to the pwcc
# STOP          Stop the Observation
#
# All other parameters recived other than those above are 
# considered key=>value parameters and form the header for the
# following observation


#
# Include Modules
#

use IO::Socket;     # Standard perl socket library
use IO::Select;     
use Net::hostent;
use File::Basename;
use Dada;           # DADA Module for configuration options
use threads;        # Perl threads module
use threads::shared; 
use strict;         # strict mode (like -Wall)

#
# Constants
#

use constant DEBUG_LEVEL        => 1;         # 0 None, 1 Minimal, 2 Verbose
use constant PIDFILE            => "apsr_tcs_interface.pid";
use constant LOGFILE            => "apsr_tcs_interface.log";
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant DFBSIM_DURATION    => "3600";    # Simulator runs for 1 hour
use constant TERMINATOR         => "\r";
use constant NHOST              => 16;        # This is constant re DFB3

#
# Global Variables
#
our $current_state : shared = "Idle";
our $pwcc_running : shared  = 0;
our $quit_threads : shared  = 0;
our %cfg : shared           = Dada->getDadaConfig();
our $use_dfb_simulator      = $cfg{"USE_DFB_SIMULATOR"};
our $dfb_sim_host           = $cfg{"DFB_SIM_HOST"};
our $dfb_sim_port           = $cfg{"DFB_SIM_PORT"};
our $dfb_sim_dest_port      = $cfg{"DFB_SIM_DEST_PORT"};
our $pwcc_host              = $cfg{"PWCC_HOST"};
our $pwcc_port              = $cfg{"PWCC_PORT"};
our $client_master_port     = $cfg{"CLIENT_MASTER_PORT"};

#
# Variable Declarations
#
my $server_host =     $cfg{"SERVER_HOST"};
my $logfile =         $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile =         $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
my $config_dir =      $cfg{"CONFIG_DIR"};
my $tcs_host =        $cfg{"TCS_INTERFACE_HOST"};
my $tcs_port =        $cfg{"TCS_INTERFACE_PORT"};
my $tcs_state_port =  $cfg{"TCS_STATE_INFO_PORT"};
my $server_logdir =   $cfg{"SERVER_LOG_DIR"};

my $handle = "";
my $peeraddr = "";
my $hostinfo = "";  
my $command = "";
my @cmds = "";
my $key = "";
my $lckey = "";
my $val = "";
my $result = "";
my $response = "";
my %tcs_cmds = ();          # Hash of commands from TCS
my $failure = "";
my $pwcc_thread = 0;
my $state_thread = 0;
my $daemon_control_thread = 0;
my $rh;

my %site_cfg = Dada->readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);


# set initial state
$current_state = "Idle";

# Autoflush output
$| = 1;

# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}

# Check that the dada.cfg matches NHOST
if ($cfg{"NUM_PWC"} != NHOST) {
  print STDERR "ERROR: Dada config file's NUM_PWC (".$cfg{"NUM_PWC"}.") did not match the expected value of ".NHOST."\n";
  exit(1);
}

# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

logMessage(1, "Opening socket for control commands on ".$tcs_host.":".$tcs_port);

my $tcs_socket = new IO::Socket::INET (
  LocalHost => $tcs_host,
  LocalPort => $tcs_port,
  Proto => 'tcp',
  Listen => 1,
  Reuse => 1
);

die "Could not create socket: $!\n" unless $tcs_socket;

logMessage(0, "STARTING SCRIPT: ".Dada->getCurrentDadaTime(0));

foreach $key (keys (%site_cfg)) {
  logMessage(2, "site_cfg: ".$key." => ".$site_cfg{$key});
}

# Run dada_pwc_command with the most recent config. This will be killed
# if the CONFIG command is received

my $pwcc_logfile = $server_logdir."/dada_pwc_command.log";
my $pwcc_file = $config_dir."/tcs.cfg";
my $utc_start = "";

$pwcc_thread = threads->new(\&pwcc_thread, $pwcc_file);
logMessage(2, "dada_pwc_command thread started");

# This thread will simply report the current state of the PWCC_CONTROLLER
$state_thread = threads->new(\&state_reporter_thread, $server_host, $tcs_state_port);
logMessage(2, "state_reporter_thread started");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

my $read_set = new IO::Select();  # create handle set for reading
$read_set->add($tcs_socket);   # add the main socket to the set

# Main Loop,  We loop forever unless asked to quit
while (!$quit_threads) {

  # Get all the readable handles from the server
  my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
  logMessage(3, "select on read_set returned");
                                                                                
  foreach $rh (@$readable_handles) {
  
    if ($rh == $tcs_socket) {

      # Only allow 1 connection from TCS
      if ($handle) {
        
        $handle = $tcs_socket->accept() or die "accept $!\n";

        $peeraddr = $handle->peeraddr;
        $hostinfo = gethostbyaddr($peeraddr);

        logMessage(0, "WARN: Rejecting additional connection from ".$hostinfo->name);
        $handle->close();

      } else {

        # Wait for a connection from the server on the specified port
        $handle = $tcs_socket->accept() or die "accept $!\n";

        # Ensure commands are immediately sent/received
        $handle->autoflush(1);

        # Add this read handle to the set
        $read_set->add($handle);

        # Get information about the connecting machine
        $peeraddr = $handle->peeraddr;
        $hostinfo = gethostbyaddr($peeraddr);
        logMessage(1, "Accepting connection from ".$hostinfo->name);
      }

    } else {
     
      $command = <$rh>;

      # If we have lost the connection...
      if (! defined $command) {

        logMessage(1, "Lost TCS connection");

        $read_set->remove($rh);
        close($rh);
        $handle->close();
        $handle = 0;

      # Else we have received a command
      } else {

        $result = "";
        $response = "";

        # clean up the string
        my $cleaned_command = $command;
        $cleaned_command =~ s/\r//;
        $cleaned_command =~ s/\n//;
        $cleaned_command =~ s/#(.)*$//;
        $cleaned_command =~ s/ +$//;
        $command = $cleaned_command;

        @cmds = split(/ +/,$command,2);
        $key = $cmds[0];
        $val = $cmds[1];
        $lckey = lc $key;

        logMessage(1, "TCS -> ".$key."\t".$val);

                                                                                                                                                                              
################################################################################
#
# START command 
#
# 1. Write the config file
# 2. Write the specification file
#
#

        # START command will take all received header parameters and 
        # use them to launch dada_pwc_command and start the observation 
        #
        # WARNING!!!
        # We return an "ok" to TCS immediately so that it does not 
        # timeout whilst we start things up... 
  
        if ($lckey eq "start") {

          logMessage(2, "Processing START command"); 

          # Check that %tcs_cmds has all the required parameters in it
          ($result, $response) = parseTCSCommands(\%tcs_cmds);

          # Send response to TCS
          logMessage(1, "TCS <- ".$result);
          print $handle $result.TERMINATOR;

          if ($result ne "ok") {

            logMessage(0, "parseTCSCommands returned \"".$result.":".$response."\"");

          } else {

            # quit/kill the current daemon
            quit_pwc_command();

            # Add the extra commands/config for each PWC
            %tcs_cmds = addHostCommands(\%tcs_cmds, \%site_cfg);

            # Create the tcs.cfg file to launch dada_pwc_command
            ($result, $response) = generateConfigFile($cfg{"CONFIG_DIR"}."/tcs.cfg", \%tcs_cmds);
            logMessage(0, "generateConfigFile: ".$result.":".$response); 

            # rejoin the pwcc command thread
            $pwcc_thread->join();

            # Now that we have a successful header. Launch dada_pwc_command in
            $pwcc_thread = threads->new(\&pwcc_thread, $cfg{"CONFIG_DIR"}."/tcs.cfg");

            # Create the tcs.spec file to launch dada_pwc_command
            ($result, $response) = generateSpecificationFile($cfg{"CONFIG_DIR"}."/tcs.spec", \%tcs_cmds);
            logMessage(0, "generateSpecFile: ".$result.":".$response);
  
            # Issue the start command itself
            ($result, $response) = start($cfg{"CONFIG_DIR"}."/tcs.spec", \%tcs_cmds);
            logMessage(0, "start: ".$result.":".$response);

            if ($result eq "fail") {
              logMessage(0, "Error running start command \"".$response."\"");
            } else {
              logMessage(2, "Start command successful \"".$response."\"");
              $current_state = "Recording";

              # If simulating, start the DFB simulator
              if ($use_dfb_simulator) {

                ($result, $response) = startDFB();

                if ($result eq "ok") {
                  $utc_start = $response;

                  logMessage(1, "UTC_START = ".$utc_start);
                  ($result,$response) = set_utc_start($response, \%tcs_cmds);
                  logMessage(1, "set_utc_start: ".$result.":".$response);

                }
                %tcs_cmds = ();
              }

            }
          }

################################################################################
#
# BAT command is the Baryonic Atomic Time.
#
        } elsif ($lckey eq "bat") {

          logMessage(2, "Processing BAT command");
          logMessage(1, "Sending a fake \"ok\" to TCS");
          print $handle "ok".TERMINATOR;

          my $bindir = Dada->getCurrentBinaryVersion();
          my $utc_cmd = $bindir."/bat_to_utc ".$val;

          logMessage(2, "Running ".$utc_cmd);

          $utc_start = `$utc_cmd`;
  
          chomp $utc_start;

          ($result, $response) = set_utc_start(ltrim($utc_start), \%tcs_cmds);

          # After the utc has been set, we can reset the tcs_cmds
          %tcs_cmds = ();

        } elsif ($lckey eq "set_utc_start") {

          logMessage(2, "Processing SET_UTC_START command");

          $utc_start = $val;

          ($result, $response) = set_utc_start(ltrim($utc_start), \%tcs_cmds);

          # After the utc has been set, we can reset the tcs_cmds
          %tcs_cmds = ();


################################################################################
#
# STOP command
#
        } elsif ($lckey eq "stop")  {

          logMessage(2, "Processing STOP command"); 
          ($result, $response) = stop();

          # Stop the simulator (if not using DFB3)
          if ($use_dfb_simulator) {
            my $dfbstopthread = threads->new(\&stopDFBSimulator, $dfb_sim_host);
            $dfbstopthread->detach();
          }

          $current_state = "Idle";
          $utc_start = "";


################################################################################
#
# REC_STOP command
#
        } elsif ($lckey eq "rec_stop") {

          logMessage(2, "Processing REC_STOP command");
          ($result, $response) = rec_stop(ltrim($val));
          $current_state = "Idle";
          $utc_start = "";


################################################################################
#
# DURATION command
#
        } elsif ($lckey eq "duration") {

          logMessage(2, "Processing DURATION command");
          logMessage(0, "utc_start = ".$utc_start);
   
          my $utc_stop = Dada->addToTime($utc_start,$val);
          ($result, $response) = rec_stop(ltrim($utc_stop));


        } elsif ($lckey eq "quit_script") {

          logMessage(2, "Processing QUIT_SCRIPT command");
          $quit_threads = 1;
          $handle->close();

          quit_pwc_command();

        } else {

          logMessage(3, "Processing HEADER parameter"); 

          # TODO - PROC FILE HACK UNTIL TCS IS FIXED
          if ($key eq "PROCFIL") {
            $key = "PROC_FILE";
          }

          $tcs_cmds{$key} = $val;

          $result = "ok";
          $response = "";
          $current_state = "Preparing";

        }

        # Special "hack" case as we return "ok" to a start
        # command without waiting
        if (!(($lckey eq "start") || ($lckey eq "bat") )) {

          if ($result eq "fail") {
            $current_state = "Error";
            logMessage(0, "ERROR :".$result." ".$response);
            print $handle $result.TERMINATOR;
            print $handle $response.TERMINATOR;

            if ($use_dfb_simulator) {
              my $dfbstopthread = threads->new(\&stopDFBSimulator, $dfb_sim_host);
              $dfbstopthread->detach();
            }

          } else {
            print $handle $result.TERMINATOR;
            logMessage(1, "TCS <- ".$result);
          }
        }
      }
    }
  }
}

logMessage(2, "Joining threads");

# rejoin threads
$daemon_control_thread->join();
$pwcc_thread->join();
$state_thread->join();

logMessage(0, "STOPPING SCRIPT: ".Dada->getCurrentDadaTime(0));

exit 0;


###############################################################################
#
# Functions
#


#
# Runs dada_pwc_command in non daemon mode. All ouput should be logged to
# the log file specified
#
sub pwcc_thread($) {

  (my $fname) = @_;

  my $logfile = $cfg{"SERVER_LOG_DIR"}."/".PWCC_LOGFILE;
  my $port    = $cfg{"PWCC_PORT"};
  my $bindir = Dada->getCurrentBinaryVersion();

  my $cmd = $bindir."/dada_pwc_command -p ".$port." -c ".$fname." >> ".$logfile." 2>&1";

  logMessage(2, "pwcc_thread: running dada_pwc_command -p ".$port);

  $pwcc_running = 1;
  my $returnVal = system($cmd);
  $pwcc_running = 0;

  if ($returnVal == 0) {   
    logMessage(2, "pwcc_thread: dada_pwc_command returned ".$returnVal);
    logMessage(2, "pwcc_thread: exiting");
    return "ok";
  } else {
    logMessage(0, "pwcc_thread: dada_pwc_command returned ".$returnVal);
    logMessage(2, "pwcc_thread: exiting");
    return "fail";
  }
}


sub state_reporter_thread($$) {

  my ($host, $port) = @_;

  my $state_socket = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );

  logMessage(2, "state_reporter: created socket ".$host.":".$port);

  die "state_reporter: Could not create socket: $!\n" unless $state_socket;

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($state_socket);    # add the main socket to the set

  my $handle;
  my $result = "";
  my $rh;

  # monitor the global variable for "quitting"
  while (!$quit_threads) {

    # Get all the readable handles from the read set
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    logMessage(3, "state_reporter: select time loop");

    foreach $rh (@$readable_handles) {

      # If we are accepting a connection
      if ($rh == $state_socket) {
  
        # Wait for a connection from the server on the specified port
        $handle = $rh->accept() or die "accept $!\n";
        $handle->autoflush();
        my $hostinfo = gethostbyaddr($handle->peeraddr);
        my $hostname = $hostinfo->name;
                                                                                
        logMessage(3, "state_reporter: Accepting connection from ".$hostname);

        $read_set->add($handle);
      } else {

        # Get the request
        $result = Dada->getLine($handle);

        if (! defined $result) {
          logMessage(3,"state_reporter: closing rh");
          $read_set->remove($rh);
          close($rh);
        } else {

          logMessage(3,"state_reported: \"".$result."\"");

          if ($result eq "state") {
            print $handle $current_state."\r\n";
            logMessage(3,"state_reporter: \"".$current_state."\"");
          }
        }
      }
    }
  }

  logMessage(2, "state_reporter: exiting");

}

sub quit_pwc_command() {

  my $host = $cfg{"PWCC_HOST"};
  my $port = $cfg{"PWCC_PORT"};

  my $handle = Dada->connectToMachine($host, $port);
  my $success = 1;
                                                                                                                
  if (!$handle) {

    logMessage(0, "Could not connect to Nexus (".$host.": ".$port.")");

    # try to kill the process manually
    my $result = "";
    my $response = "";
    ($result, $response) = Dada->killProcess("dada_pwc_command");

    return ($result, $response);

  # We have connected to the 
  } else {

    # Ignore the "welcome" message
    my $result = <$handle>;

    # Send config command
    my $cmd = "quit";
    logMessage(2, "quit_pwc_command: sending: \"".$cmd."\"");
    print $handle $cmd."\r\n";
    $handle->close();

    my $nwait = 2;
    while(($pwcc_running) && ($nwait > 0)) {
      sleep(1);
      $nwait--;
    }
    if ($pwcc_running) {
       logMessage(0, "Was forced to kill dada_pwc_command");
      ($result, $response) = Dada->killProcess("dada_pwc_command");
    }

    return ("ok","");

  }
}


#
# Send the START command to the pwcc, optionally starting a DFB simualtor
#
sub start($\%) {
                                                                              
  my ($file, $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  my $rVal = 0;
  my $cmd;

  my $result;
  my $response;

  # If we will run a separate DFB simulator
  if ($use_dfb_simulator) {

    # ARGS: host, dest port, nbit, npol, mode, duration 
    ($result, $response) = createDFBSimulator(\%tcs_cmds);

    # Give it half a chance to startup
    sleep(1);

  }

  # Connect to dada_pwc_command
  my $handle = Dada->connectToMachine($pwcc_host, $pwcc_port, 5);

  if (!$handle) {
    return ("fail", "Could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);

  } else {

    # Ignore the "welcome" message
    $result = <$handle>;

    # Check we are in the IDLE state before continuing
    if (Dada->waitForState("idle", $handle, 5) != 0) {
      return ("fail", "Nexus was not in IDLE state");
    }

    # Send CONFIG command
    $cmd = "config ".$file;
    ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
    logMessage(1, "Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

    if ($result ne "ok") { 
      return ("fail", "config command failed on nexus: \"".$response."\"");
    }

    # Wait for the PREPARED state
    if (Dada->waitForState("prepared",$handle,10) != 0) {
      return ("fail", "Nexus did not enter PREPARED state after config command");
    }
    logMessage(2, "Nexus now in PREPARED state");

    # Send start command 
    $cmd = "start";

    ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
    logMessage(1,"Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

    if ($result ne "ok") { 
      logMessage(1, "start command failed: (".$result.", ".$response.")");
      return ("fail", "start command failed on nexus: \"".$response."\"");
    }

    # Wait for the prepared state
    if (Dada->waitForState("recording",$handle,10) != 0) {
      return ("fail", "Nexus did not enter RECORDING state after \"start\" command");
    }

    logMessage(2, "Nexus now in \"RECORDING\" state");

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
sub set_utc_start($\%) {

  my ($utc_start, $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  logMessage(1,"set_utc_start(".$utc_start.")");

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

  $cmd = "chgrp -R ".$proj_id." ".$results_dir;
  system($cmd);

  $cmd = "chgrp -R ".$proj_id." ".$archive_dir;
  system($cmd);

  $cmd = "chmod -R g+s ".$results_dir;
  system($cmd);

  $cmd = "chmod -R g+s ".$archive_dir;
  system($cmd);

  my $fname = $results_dir."/obs.info";
  logMessage(2,"set_utc_start: creating obs.info \"".$fname."\"");

  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";
  print FH Dada->headerFormat("SOURCE",$tcs_cmds{"SOURCE"})."\n";
  print FH Dada->headerFormat("RA",$tcs_cmds{"RA"})."\n";
  print FH Dada->headerFormat("DEC",$tcs_cmds{"DEC"})."\n";
  print FH Dada->headerFormat("CFREQ",$tcs_cmds{"CFREQ"})."\n";
  print FH Dada->headerFormat("PID",$tcs_cmds{"PID"})."\n";
  print FH Dada->headerFormat("BANDWIDTH",$tcs_cmds{"BANDWIDTH"})."\n";
  print FH "\n";
  print FH Dada->headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  print FH Dada->headerFormat("NBIT",$tcs_cmds{"NBIT"})."\n";
  print FH Dada->headerFormat("NPOL",$tcs_cmds{"NPOL"})."\n";
  print FH Dada->headerFormat("NDIM",$tcs_cmds{"NDIM"})."\n";
  close FH;

  $cmd = "cp ".$fname." ".$archive_dir;
  system($cmd);

  # connect to nexus
  my $handle = Dada->connectToMachine($cfg{"PWCC_HOST"}, $cfg{"PWCC_PORT"});

  if (!$handle) {
    logMessage(1,"connect failed");
    return ("fail", "Could not connect to Nexus ".$cfg{"PWCC_HOST"}.":".$cfg{"PWCC_PORT"});
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  # Wait for the prepared state
  if (Dada->waitForState("recording", $handle, 10) != 0) {
    return ("fail", "Nexus took more than 10 seconds to enter \"recording\" state");
  }

  # Send UTC Start command to the nexus
  $cmd = "set_utc_start ".$utc_start;

  ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
  logMessage(1,"Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

  # Close nexus connection
  $handle->close();

  return ($result, $response);

}


#
# Sends the "stop" command to the Nexus
#
sub stop() {

  my $ignore = "";
  my $result = "";
  my $response = "";

  my $handle = Dada->connectToMachine($cfg{"PWCC_HOST"}, $cfg{"PWCC_PORT"});

  if (!$handle) {
    return ("fail", "Could not connect to Nexus ".$cfg{"PWCC_HOST"}.":".$cfg{"PWCC_PORT"});
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  my $cmd = "stop";

  ($result, $response) = Dada->sendTelnetCommand($handle,$cmd);
  logMessage("Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

  # Close nexus connection
  $handle->close();

  return ($result, $response);

}


#
# Sends the "rec_stop" command to the Nexus
#
sub rec_stop($) {

  (my $utc) = @_;

  logMessage(2, "rec_stop (".$utc.")");

  my $ignore = "";
  my $result = "";
  my $response = "";

  my $handle = Dada->connectToMachine($cfg{"PWCC_HOST"}, $cfg{"PWCC_PORT"});
  if (!$handle) {
    return ("fail", "Could not connect to Nexus ".$cfg{"PWCC_HOST"}.":".$cfg{"PWCC_PORT"});
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  my $cmd = "rec_stop ".$utc;

  ($result, $response) = Dada->sendTelnetCommand($handle,$cmd);
  logMessage("Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

  # Close nexus connection
  $handle->close();

  return ($result, $response);
}


sub logMessage($$) {
  my ($level, $message) = @_;
  
  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada->getCurrentDadaTime(0)."] ".$message."\n";
  }
}

sub ltrim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  return $string;
}

sub createDFBSimulator(\%) {

  (my $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  my $host      =       $cfg{"DFB_SIM_HOST"};
  my $dest_port = "-p ".$cfg{"DFB_SIM_DEST_PORT"};

  my $nbit      = "-b ".$tcs_cmds{"NBIT"};
  my $npol      = "-k ".$tcs_cmds{"NPOL"};
  my $ndim      = "-g ".$tcs_cmds{"NDIM"};
  my $tsamp     = "-t ".$tcs_cmds{"TSAMP"};

  # By default set the "period" of the signal to 64000 bytes;
  my $calfreq   = "-c 64000";

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
  

  logMessage(2,"createDFBSimulator: $host, $dest_port, $nbit, $npol, $ndim, $tsamp, $calfreq, $duration");

  my $args = "-y $dest_port $nbit $npol $ndim $tsamp $calfreq $drate $duration $dest";

  my $result = "";
  my $response = "";

  # Launch dfb simulator on remote host
  my $dfb_cmd = "dfbsimulator -d -a ".$args;
  my $handle = Dada->connectToMachine($host, $client_master_port);

  if (!$handle) {
    return ("fail", "Could not connect to client_master_control.pl ".$host.":".$client_master_port);
  }

  logMessage(2,"createDFBSimulator: sending cmd ".$dfb_cmd);

  ($result, $response) = Dada->sendTelnetCommand($handle,$dfb_cmd);

  logMessage(2,"createDFBSimulator: received reply: (".$result.", ".$response.")");

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

  logMessage(2,"startDFB: ()");

  my $handle = Dada->connectToMachine($host, $port);
  if (!$handle) {
    return ("fail", "Could not connect to apsr_test_triwave ".$host.":".$port);
  }

  logMessage(2,"startDFB: sending command \"start\"");
 
  ($result, $response) = Dada->sendTelnetCommand($handle,"start");

  logMessage(2,"startDFB: received reply (".$result.", ".$response.")");

  $handle->close();

  return ($result, $response);

}

sub stopDFBSimulator() {

  my $host = $cfg{"DFB_SIM_HOST"};

  my $result = "";
  my $response = "";

  logMessage(1,"stopDFBSimulator: stopping simulator in 10 seconds");
  sleep(10);

  # connect to client_master_control.pl
  my $handle = Dada->connectToMachine($host, $client_master_port);

  if (!$handle) {
    return ("fail", "Could not connect to client_master_control.pl ".$host.":".$client_master_port);
  }

  # This command will kill the udp packet generator
  my $dfb_cmd = "stop_dfbs";

  ($result, $response) = Dada->sendTelnetCommand($handle,$dfb_cmd);

  logMessage(1,"stopDFBSimulator: received reply: ".$response);

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
# Handle INT AND TERM signals
#
sub sigHandle($) {
                                                                                
  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_threads = 1;
  # sleep to allow threads to quit 
  sleep(3);
  logMessage(0, "STOPPING SCRIPT: ".Dada->getCurrentDadaTime(0));
  exit(1);
                                                                                
}


#
# Polls for the "quitdaemons" file in the control dir
#
sub daemonControlThread() {

  logMessage(2, "daemon_control: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_threads)) {
    logMessage(3, "daemon_control: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_threads = 1;

  # Manually tell dada_pwc_command to quit
  quit_pwc_command();

  logMessage(2, "daemon_control: Unlinking PID file ".$pidfile);
  unlink($pidfile);

  logMessage(2, "daemon_control: exiting");

}


#
# Checks that TCS supplied us with the MINIMAL set of commands 
# necessary to run 
# 
sub parseTCSCommands(\%) {

  (my $tcs_cmds_ref) = @_;
    
  my %tcs_cmds = %$tcs_cmds_ref;

  my $result = "ok";
  my $response = "";

  my @cmds = qw(CONFIG SOURCE RA DEC RECEIVER CFREQ PID NBIT NDIM NPOL BANDWIDTH PROC_FILE MODE CALFREQ);
  my $cmd;

  foreach $cmd (@cmds) {
    if (!(exists $tcs_cmds{$cmd})) {
      $result = "fail";
      $response .= " ".$cmd;
    }
  }

  if ($result eq "fail") {
    $response = "Missing Parameter(s)".$response;
  }
  return ($result, $response);

} 

#
# Addds the required keys,values to the TCS commands based
# on the hardwired constants of the DFB3. These are:
#
# 1. Always sending to 16 PWCs
# 2. Lowest freq band goes to apsr00, highest to apsr15
# 3. 16 Bands (for the moment)
#
sub addHostCommands(\%\%) {

  my ($tcs_cmds_ref, $site_cfg_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;
  my %site_cfg = %$site_cfg_ref;

  $tcs_cmds{"NUM_PWC"}     = NHOST;
  $tcs_cmds{"PWC_PORT"}    = 56026;
  $tcs_cmds{"LOGFILE_DIR"} = $cfg{"SERVER_LOG_DIR"};
  $tcs_cmds{"HDR_SIZE"}    = $site_cfg{"HDR_SIZE"};

  # Determine the BW & FREQ for each channel
  my $cf = int($tcs_cmds{"CFREQ"});         # centre frequency
  my $tbw = int($tcs_cmds{"BANDWIDTH"});    # total bandwidth
  my $bw = -1 * ($tbw / int(NHOST));        # bandwidth per channel

  my $i=0;
  for ($i=0; $i<NHOST; $i++) {
    $tcs_cmds{"Band".$i."_BW"} = $bw;
    $tcs_cmds{"Band".$i."_FREQ"} = $cf - ($tbw/2) + ($bw/2) + ($bw*$i);
  }

  # Add the site configuration to tcs_cmds
  foreach $key (keys (%site_cfg)) {
    $tcs_cmds{$key} = $site_cfg{$key};
  }

  # Determine the TSAMP based upon NDIM and BW 
  $tcs_cmds{"TSAMP"} = (1.0 / abs($bw)) * ($tcs_cmds{"NDIM"} / 2);

  # Setup the number of channels per node
  if (!(exists $tcs_cmds{"NBAND"})) {
    $tcs_cmds{"NBAND"} = NHOST;
  }
  $tcs_cmds{"NCHAN"} = $tcs_cmds{"NBAND"} / NHOST;
  
  # Set the instrument
  $tcs_cmds{"INSTRUMENT"} = uc($cfg{"INSTRUMENT"});
    
  return %tcs_cmds;

}


#
# Generates the config file required for dada_pwc_command
#
sub generateConfigFile($\%) {

  my ($fname, $tcs_cmds_ref) = @_;
                                                                                                                                                                              
  my %tcs_cmds = %$tcs_cmds_ref;
  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);

  print FH "# Header file created by ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";
  print FH  Dada->headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  logMessage(2, "tcs.cfg: ".Dada->headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"}));
  print FH  Dada->headerFormat("PWC_PORT",$tcs_cmds{"PWC_PORT"})."\n";
  logMessage(2, "tcs.cfg: ".Dada->headerFormat("PWC_PORT",$tcs_cmds{"PWC_PORT"}));
  print FH  Dada->headerFormat("LOGFILE_DIR",$tcs_cmds{"LOGFILE_DIR"})."\n";
  logMessage(2, "tcs.cfg: ". Dada->headerFormat("LOGFILE_DIR",$tcs_cmds{"LOGFILE_DIR"}));
  print FH  Dada->headerFormat("HDR_SIZE",$tcs_cmds{"HDR_SIZE"})."\n";
  logMessage(2, "tcs.cfg: ".Dada->headerFormat("HDR_SIZE",$tcs_cmds{"HDR_SIZE"}));

  my $i=0;
  for($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    print FH Dada->headerFormat("PWC_".$i,$cfg{"PWC_".$i})."\n";
    logMessage(2, "tcs.cfg: ".Dada->headerFormat("PWC_".$i,$cfg{"PWC_".$i}));
  }
  close FH;

  return ("ok", "");

}


sub generateSpecificationFile($\%) {

  my ($fname, $tcs_cmds_ref) = @_;
                                                                                                                                                                              
  my %tcs_cmds = %$tcs_cmds_ref;

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Specification File created by ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";

  my %ignore = ();
  $ignore{"NUM_PWC"} = "yes";
  $ignore{"PWC_PORT"} = "yes";
  $ignore{"LOGFILE_DIR"} = "yes";
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
      logMessage(2, "tcs.spec: ".Dada->headerFormat($line, $tcs_cmds{$line}));
    }
  }

  close FH;
  return ("ok","");
}
  










