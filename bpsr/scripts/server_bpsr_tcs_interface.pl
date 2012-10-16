#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  3 Dec, 2007
# Modigied: 11 Mar, 2008
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use IO::Socket;     # Standard perl socket library
use IO::Select;     
use Net::hostent;
use File::Basename;
use threads;        # Perl threads module
use threads::shared; 
use Dada;           # DADA Module for configuration options
use Bpsr;           # Bpsr Module for configuration options
use strict;         # strict mode (like -Wall)

sub quitPWCCommand();


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant PIDFILE            => "bpsr_tcs_interface.pid";
use constant LOGFILE            => "bpsr_tcs_interface.log";
use constant QUITFILE           => "bpsr_tcs_interface.quit";
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant TERMINATOR         => "\r";

#
# Global variable declarations
#
our $dl;
our $daemon_name;
our %cfg : shared;
our %roach : shared;
our $current_state : shared;
our $pwcc_running : shared;
our $quit_threads : shared;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our $error;
our $warn;
our $pwcc_thread;

#
# global variable initialization
#
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Bpsr::getConfig();
%roach = Bpsr::getROACHConfig();
$current_state = "Idle";
$pwcc_running = 0;
$quit_threads = 0;
$pwcc_host = $cfg{"PWCC_HOST"};
$pwcc_port = $cfg{"PWCC_PORT"};
$client_master_port = $cfg{"CLIENT_MASTER_PORT"};
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
$pwcc_thread = 0;


#
# Main
#
{
  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  my $server_host =     $cfg{"SERVER_HOST"};
  my $config_dir =      $cfg{"CONFIG_DIR"};
  my $tcs_host =        $cfg{"TCS_INTERFACE_HOST"};
  my $tcs_port =        $cfg{"TCS_INTERFACE_PORT"};
  my $tcs_state_port =  $cfg{"TCS_STATE_INFO_PORT"};

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
  my $state_thread = 0;
  my $control_thread = 0;
  my $rh;

  my $roach = "";
  my @roaches = ();
  my $beam = "";
  my $level = "";
  my %levels = ();
  my $cmd = "";

  my %site_cfg = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);

  # set initial state
  $current_state = "Idle";

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  # Sanity check for this script
  if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) 
  {
    print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
    print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
    exit(1);
  }

  if (-f $warn) {
    unlink $warn;
  }
  if (-f $error) {
    unlink $error;
  }

  # Redirect standard output and error
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  Dada::logMsg(0, $dl, "Waiting for Multibob to be configured");
  ($result, $response) = waitForMultibobBoot();
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Failed to configure multibob server: ".$response);
  } else {
    Dada::logMsg(0, $dl, "Multibob ready");
  }

  Dada::logMsg(1, $dl, "Opening socket for control commands on ".$tcs_host.":".$tcs_port);

  my $tcs_socket = new IO::Socket::INET (
    LocalHost => $tcs_host,
    LocalPort => $tcs_port,
    Proto => 'tcp',
    Listen => 1,
    ReuseAddr => 1
  );

  die "Could not create socket: $!\n" unless $tcs_socket;

  foreach $key (keys (%site_cfg)) {
    Dada::logMsg(2, $dl, "site_cfg: ".$key." => ".$site_cfg{$key});
  }

  my $localhost = Dada::getHostMachineName();
  my $pwcc_file = $config_dir."/bpsr_tcs.cfg";
  my $utc_start = "";


  # Create the initial bpsr_tcs.cfg file to launch dada_pwc_command
  ($result, $response) = generateConfigFile($cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg", \%site_cfg);
  Dada::logMsg(1, $dl, "Generated initial config file ".$result.":".$response);

  $pwcc_thread = threads->new(\&pwcc_thread, $pwcc_file);
  Dada::logMsg(2, $dl, "dada_pwc_command thread started");

  # This thread will simply report the current state of the PWCC_CONTROLLER
  $state_thread = threads->new(\&state_reporter_thread, $server_host, $tcs_state_port);
  Dada::logMsg(2, $dl, "state_reporter_thread started");

  # Start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file, $quit_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($tcs_socket);   # add the main socket to the set

  # Main Loop,  We loop forever unless asked to quit
  while (!$quit_threads) {

    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
    Dada::logMsg(3, $dl, "select on read_set returned");
                                                                                  
    foreach $rh (@$readable_handles) {
    
      if ($rh == $tcs_socket) {

        # Only allow 1 connection from TCS
        if ($handle) {
          
          $handle = $tcs_socket->accept() or die "accept $!\n";

          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);

          Dada::logMsgWarn($warn, "Rejecting additional connection from ".$hostinfo->name);
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
          Dada::logMsg(1, $dl, "Accepting connection from ".$hostinfo->name);
        }

      } else {
       
        $command = <$rh>;

        # If we have lost the connection...
        if (! defined $command) {

          Dada::logMsg(1, $dl, "lost connection from ".$hostinfo->name);

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
          $cleaned_command =~ s/\r//;       # strip \r
          $cleaned_command =~ s/\n//;       # strip \n
          $cleaned_command =~ s/#(.)*$//;   # delete comments
          $cleaned_command =~ s/\s+$//;     # remove trailing whitespaces
          $cleaned_command =~ s/\0//g;      # remove all null characters
          $command = $cleaned_command;

          @cmds = split(/ +/,$command,2);
          $key = $cmds[0];
          $val = $cmds[1];
          $lckey = lc $key;

          Dada::logMsg(1, $dl, "TCS -> ".Dada::headerFormat($key, $val));

          #######################################################################
          # START command 
          if ($lckey eq "start") 
          {
            Dada::logMsg(2, $dl, "Processing START command"); 

            # adjust the command names
            %tcs_cmds = fixTCSCommands(\%tcs_cmds);
  
            # set levels of roaches and return hash of level values
            ($result, %levels) = setRoachLevels();

            # Set the number of active beams
            %tcs_cmds = setActiveBeamsTCS(\%tcs_cmds, \%levels);

            # Check that %tcs_cmds has all the required parameters in it
            ($result, $response) = parseTCSCommands(\%tcs_cmds);

            # Send response to TCS
            if ($result ne "ok")
            {
              Dada::logMsg(0, $dl, "TCS <- ".$result);
            }
            else
            {
              Dada::logMsg(2, $dl, "TCS <- ".$result);
            } 
            print $handle $result.TERMINATOR;

            if ($result ne "ok") {

              Dada::logMsg(0, $dl, "parseTCSCommands returned \"".$result.":".$response."\"");

            } else {

              Dada::logMsg(1, $dl, "TCS commands parsed ok");

              # quit/kill the current daemon
              quitPWCCommand();
  
              # Clear the status files
              my $cmd = "rm -f ".$cfg{"STATUS_DIR"}."/*";
              ($result, $response) = Dada::mySystem($cmd);
              if ($result ne "ok") {
                Dada::logMsgWarn($warn, "Could not delete status files: $response");
              }

              # Add the extra commands/config for each PWC
              %tcs_cmds = addHostCommands(\%tcs_cmds, \%site_cfg);

              # Create the tcs.cfg file to launch dada_pwc_command
              Dada::logMsg(2, $dl, "main: generateConfigFile(".$cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg)");
              ($result, $response) = generateConfigFile($cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg", \%tcs_cmds);
              Dada::logMsg(2, $dl, "main: generateConfigFile() ".$result." ".$response);

              # rejoin the pwcc command thread
              $pwcc_thread->join();

              # Now that we have a successful header. Launch dada_pwc_command in
              $pwcc_thread = threads->new(\&pwcc_thread, $cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg");

              # Create the bpsr_tcs.spec file to launch dada_pwc_command
              Dada::logMsg(2, $dl, "main: generateSpecificationFile(".$cfg{"CONFIG_DIR"}."/bpsr_tcs.spec)");
              ($result, $response) = generateSpecificationFile($cfg{"CONFIG_DIR"}."/bpsr_tcs.spec", \%tcs_cmds);
              Dada::logMsg(2, $dl, "main: generateSpecificationFile() ".$result." ".$response);
  
              # Issue the start command itself
              Dada::logMsg(2, $dl, "main: start(".$cfg{"CONFIG_DIR"}."/bpsr_tcs.spec)");
              ($result, $response) = start($cfg{"CONFIG_DIR"}."/bpsr_tcs.spec", \%tcs_cmds);
              if ($result ne "ok") {
                Dada::logMsgWarn($warn, "start() failed \"".$response."\"");
              } else {
                Dada::logMsg(2, $dl, "start() successful \"".$response."\"");

                $utc_start = $response;

                $current_state = "Recording";

                $cmd = "start_utc ".$utc_start;
                Dada::logMsg(1, $dl, "TCS <- ".$cmd);
                print $handle $cmd.TERMINATOR;

                %tcs_cmds = ();
              }
            }

          #######################################################################
          # STOP command
          }
          elsif ($lckey eq "stop")
          {
            Dada::logMsg(2, $dl, "Processing STOP command");

            if (($current_state eq "Stopping") || ($current_state eq "Idle"))
            {
              # ignore this
              Dada::logMsg(1, $dl, "main: ignoring STOP command since state is ".$current_state);
            }
            else
            {

              # issue stop command to nexus
              Dada::logMsg(2, $dl, "main: stopNexus()");
              ($result, $response) = stopNexus();
              Dada::logMsg(2, $dl, "main: stopNexus ".$result." ".$response);

              if ($result ne "ok")
              {
                Dada::logMsgWarn($error, "Stop command failed");
                $current_state = "Error";
              }
              else
              {
                # wait for the nexus to return to the Idle state
                $current_state = "Stopping";
                my $stop_thread = threads->new(\&stopInBackground);
                $stop_thread->detach();
              }
            }

          #######################################################################
          # REC_STOP command
          } elsif ($lckey eq "rec_stop") {

            Dada::logMsg(2, $dl, "Processing REC_STOP command");
            ($result, $response) = rec_stop(ltrim($val));
            $current_state = "Idle";
            $utc_start = "";

          } elsif ($lckey eq "quit_script") {

            Dada::logMsg(2, $dl, "Processing QUIT_SCRIPT command");
            $quit_threads = 1;
            $handle->close();

            quitPWCCommand();

          } else {

            Dada::logMsg(3, $dl, "Processing HEADER parameter"); 

            # TODO - PROC FILE HACK UNTIL TCS IS FIXED
            if ($key eq "PROCFIL") {
              $key = "PROC_FILE";
            }

            $tcs_cmds{$key} = $val;

            $result = "ok";
            $response = "";
            $current_state = "Preparing";

          }

          if ($result eq "fail") 
          {
            $current_state = "Error";
            Dada::logMsgWarn($error, $result." ".$response);
            print $handle $result.TERMINATOR;
            print $handle $response.TERMINATOR;
          } else {

            # we have already replied to the start command      
            if ($lckey eq "start") {

            } else {

              print $handle $result.TERMINATOR;
              Dada::logMsg(1, $dl, "TCS <- ".$result);
            }
          }
        }
      }
    }
  }

  Dada::logMsg(2, $dl, "Joining threads");

  # rejoin threads
  $control_thread->join();
  $pwcc_thread->join();
  $state_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
}
exit 0;


###############################################################################
#
# Functions
#


#
# Runs dada_pwc_command in non daemon mode. All ouput should be logged to
# the log file specified
#
sub pwcc_thread($) 
{
  (my $config_file) = @_;
  Dada::logMsg(2, $dl, "pwcc_thread(".$config_file.")");

  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".PWCC_LOGFILE;
  my $cmd = "dada_pwc_command ".$config_file." >> ".$log_file." 2>&1";
  my $rval = 0;

  Dada::logMsg(2, $dl, "pwcc_thread: ".$cmd);

  # set global 
  $pwcc_running = 1;
  $rval = system($cmd);
  $pwcc_running = 0;

  if ($rval == 0)
  {
    Dada::logMsg(2, $dl, "pwcc_thread: dada_pwc_command returned ".$rval);
    Dada::logMsg(2, $dl, "pwcc_thread: exiting");
    return "ok";
  }
  else
  {
    Dada::logMsgWarn($warn, "pwcc_thread: dada_pwc_command returned ".$rval);
    Dada::logMsg(2, $dl, "pwcc_thread: exiting");
    return "fail";
  }
}


sub state_reporter_thread($$) 
{
  my ($host, $port) = @_;

  my $state_socket = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );

  Dada::logMsg(1, $dl, "state_reporter: created socket ".$host.":".$port);

  die "state_reporter: Could not create socket: $!\n" unless $state_socket;

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($state_socket);    # add the main socket to the set

  my $handle;
  my $result = "";
  my $rh;

  while (!$quit_threads)
  {
    # Get all the readable handles from the read set
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles)
    {
      # if we are accepting a connection
      if ($rh == $state_socket) 
      {
        $handle = $rh->accept() or die "accept $!\n";
        $handle->autoflush();
        my $hostinfo = gethostbyaddr($handle->peeraddr);
        my $hostname = $hostinfo->name;
        Dada::logMsg(3, $dl, "state_reporter: accepting connection from ".$hostname);
        $read_set->add($handle);
      } 
      else
      {
        # get the request
        $result = Dada::getLine($rh);

        if (! defined $result) 
        {
          Dada::logMsg(3, $dl, "state_reporter: closing rh");
          $read_set->remove($rh);
          close($rh);
        } 
        else
        {
          Dada::logMsg(3, $dl, "state_reporter: <- ".$result);

          if ($result eq "state")
          {
            print $rh $current_state."\r\n";
            Dada::logMsg(3, $dl, "state_reporter: -> ".$current_state);
          }

          if ($result eq "num_pwcs")
          {
            print $rh $cfg{"NUM_PWC"}."\r\n";
            Dada::logMsg(3, $dl, "state_reporter: -> ".$cfg{"NUM_PWC"});
          }
        }
      }
    }
  }

  Dada::logMsg(2, $dl, "state_reporter: exiting");
}

sub quitPWCCommand() 
{
  my $host = $cfg{"PWCC_HOST"};
  my $port = $cfg{"PWCC_PORT"};
  my $handle = 0;
  my $success = 1;
  my $cmd = "";
  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl, "quitPWCCommand: connecting to ".$host.":".$port);
  $handle = Dada::connectToMachine($host, $port);
  if ($handle) 
  {
    # ignore the "welcome" message
    $response = <$handle>;

    $cmd = "quit";
    Dada::logMsg(2, $dl, "quitPWCCommand: nexus <- ".$cmd);
    print $handle $cmd."\r\n";
    $handle->close();
    $handle = 0;
  
    # allow 2 seconds for nexus to exit
    my $nwait = 2;
    while(($pwcc_running) && ($nwait > 0))
    {
      sleep(1);
      $nwait--;
    }
  }
  else
  {
    Dada::logMsgWarn($warn, "could not connect to nexus to call quit");
  }

  # if the nexus is still running, kill it
  if ($pwcc_running) 
  {
    Dada::logMsgWarn($warn, "Was forced to kill dada_pwc_command");
    Dada::logMsg(0, $dl, "quitPWCCommand: killProcess(dada_pwc_command.*bpsr_tcs.cfg)");
    ($result, $response) = Dada::killProcess("dada_pwc_command.*bpsr_tcs.cfg");
    Dada::logMsg(0, $dl, "quitPWCCommand: ".$result." ".$response);
  }
  else
  {
    $result = "ok";
    $response = "nexus exited";
  }

  return ($result, $response);
}


#
# Send the START command to the pwcc
#
sub start($\%)
{
  my ($file, $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;
  my $rVal = 0;
  my $cmd;
  my $result;
  my $response;
  my $localhost = Dada::getHostMachineName();

  Dada::logMsg(2, $dl, "connecting to roach manager");
  my $roach_mngr = Dada::connectToMachine($localhost, $cfg{"IBOB_MANAGER_PORT"});
  if (!$roach_mngr)
  {
    Dada::logMsg(0, $dl, "Could not connect to roach manager");
    return ("fail", "utc unknown");
  }
  else
  {
    # ignore welcome message
    $response = <$roach_mngr>;

    # ensure roach_mngr is in active state
    Dada::logMsg(2, $dl, "waiting for active state");
    ($result, $response) = Bpsr::waitForMultibobState("active", $roach_mngr, 2);
    Dada::logMsg(2, $dl, "waiting for active state: ".$result." ".$response);
    if ($result ne "ok") 
    {
      Dada::logMsgWarn($error, "roach_mngr was not in the active state: ".$response);
      return ("fail", "utc_unknown");
    }

    # start flow of 10GbE packets
    $cmd = "start_tx";
    Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
    ($result,$response) = Dada::sendTelnetCommand($roach_mngr,$cmd);
    Dada::logMsg(1, $dl, "roach_mngr -> ".$result);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "roach_mngr-> ".$response);
    } else {
      Dada::logMsg(2, $dl, "roach_mngr-> ".$response);
    }

    # connect to dada_pwc_command
    Dada::logMsg(2, $dl, "Connecting to PWCC ".$pwcc_host.":".$pwcc_port);
    my $handle = Dada::connectToMachine($pwcc_host, $pwcc_port, 5);

    if (!$handle) 
    {
      return ("fail", "Could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    }
    else
    {
      # ignore the "welcome" message
      $result = <$handle>;

      # check we are in the IDLE state before continuing
      if (Dada::waitForState("idle", $handle, 5) != 0) 
      {
        return ("fail", "Nexus was not in IDLE state");
      }
      Dada::logMsg(2, $dl, "PWCC in IDLE state");

      # send CONFIG command to PWCC
      $cmd = "config ".$file;
      Dada::logMsg(1, $dl, "PWCC <- ".$cmd);
      ($result,$response) = Dada::sendTelnetCommand($handle,$cmd);
      Dada::logMsg(1, $dl, "PWCC -> ".$result." ".$response);
      if ($result ne "ok")
      {
        return ("fail", "config command failed on nexus: \"".$response."\"");
      }

      # wait for the PREPARED state
      Dada::logMsg(1, $dl, "PWCC waiting for prepared");
      if (Dada::waitForState("prepared",$handle,5) != 0)
      {
        return ("fail", "Nexus did not enter PREPARED state after config command");
      }
      Dada::logMsg(2, $dl, "Nexus in PREPARED state");

      # send start command
      $cmd = "start";
      Dada::logMsg(1, $dl, "PWCC <- ".$cmd);
      ($result,$response) = Dada::sendTelnetCommand($handle,$cmd);
      Dada::logMsg(1, $dl, "PWCC -> ".$result." ".$response);
      if ($result ne "ok")
      {
        return ("fail", "start command failed on nexus: ".$response);
      } 

      # Run the rearming script on ibob manager
      $cmd = "arm";
      Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
      ($result,$response) = Dada::sendTelnetCommand($roach_mngr,$cmd);
      Dada::logMsg(1, $dl, "roach_mngr -> ".$result);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, $response);
      }
      my @lines = split(/\n/, $response);
      my $utc_start = ltrim(@lines[1]);
      chomp $utc_start;
      Dada::logMsg(1, $dl, "UTC_START ".$utc_start);


      # Setup the server output directories before telling the clients to begin
      Dada::logMsg(2, $dl, "start: set_utc_start(".$utc_start.")");
      ($result, $response) = set_utc_start($utc_start, \%tcs_cmds);
      Dada::logMsg(2, $dl, "start: set_utc_start() ".$result." ".$response);

      # Now we should have a UTC_START!
      $cmd = "set_utc_start ".$utc_start;
      Dada::logMsg(1, $dl, "PWCC <- ".$cmd);
      ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
      Dada::logMsg(1, $dl, "PWCC -> ".$result);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, $response);
      }

      # Wait for the prepared state
      if (Dada::waitForState("recording",$handle,30) != 0) {
        return ("fail", "Nexus did not enter RECORDING state after clock command");
      }

      Dada::logMsg(2, $dl, "Nexus now in RECORDING state");

      # Close nexus connection
      $handle->close();

      # Close the roach connection
      $cmd = "exit";
      Dada::logMsg(2, $dl, "roach_mngr <- ".$cmd);
      print $roach_mngr $cmd."\r\n";

      sleep(1);

      $roach_mngr->close();

      return ($result, $utc_start);

    }
  }
}


#
# Sends a UTC_START command to the pwcc
#
sub set_utc_start($\%) {

  my ($utc_start, $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  Dada::logMsg(2, $dl, "set_utc_start(".$utc_start.")");

  my $ignore = "";
  my $result = "ok";
  my $response = "";
  my $cmd = "";

  # Now that we know the UTC_START, create the required results and archive
  # directories and put the observation summary file there...

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
  my $proj_id     = $tcs_cmds{"PID"};

  Dada::logMsg(2, $dl, "Setting up results dir: ".$results_dir);

  $cmd = "mkdir -m 0755 -p ".$results_dir;
  my ($resu, $resp) = Dada::mySystem($cmd,0);
  if ($resu != "ok") {
    Dada::logMsgWarn($warn, "Failed to create the server results directory (".$results_dir.") \"".$resp."\"");
  }

  my $fname = $results_dir."/obs.info";
  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);

  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";
  print FH Dada::headerFormat("SOURCE",$tcs_cmds{"SOURCE"})."\n";
  print FH Dada::headerFormat("RA",$tcs_cmds{"RA"})."\n";
  print FH Dada::headerFormat("DEC",$tcs_cmds{"DEC"})."\n";
  print FH Dada::headerFormat("FA",$tcs_cmds{"FA"})."\n";
  print FH Dada::headerFormat("CFREQ",$tcs_cmds{"CFREQ"})."\n";
  print FH Dada::headerFormat("PID",$tcs_cmds{"PID"})."\n";
  print FH Dada::headerFormat("BANDWIDTH",$tcs_cmds{"BANDWIDTH"})."\n";
  print FH Dada::headerFormat("ACC_LEN",$tcs_cmds{"ACC_LEN"})."\n";
  print FH Dada::headerFormat("UTC_START",$utc_start)."\n";
  print FH "\n";
  print FH Dada::headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  print FH Dada::headerFormat("NBIT",$tcs_cmds{"NBIT"})."\n";
  print FH Dada::headerFormat("NPOL",$tcs_cmds{"NPOL"})."\n";
  print FH Dada::headerFormat("NDIM",$tcs_cmds{"NDIM"})."\n";
  print FH Dada::headerFormat("NCHAN",$tcs_cmds{"NCHAN"})."\n";
  print FH Dada::headerFormat("PROC_FILE",$tcs_cmds{"PROC_FILE"})."\n";
  close FH;

  # Create the obs.processing files
  $cmd = "touch ".$results_dir."/obs.processing";
  system($cmd);
  
  return ($result, $response);
}


#
# Connect to the nexus and issue the stop command
#
sub stopNexus()
{
  Dada::logMsg(2, $dl, "stopNexus()");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;

  my $host = $cfg{"PWCC_HOST"};
  my $port = $cfg{"PWCC_PORT"};

  Dada::logMsg(2, $dl, "stopNexus: opening connection to ".$host.":".$port);
  $handle = Dada::connectToMachine($host, $port);
  if (!$handle) {
    Dada::logMsg(0, $dl, "stopNexus: could not connect to dada_pwc_command ".$host.":".$port);
    return ("fail", "could not connect to nexus to issue STOP");
  }

   # Ignore the "welcome" message
  $ignore = <$handle>;

  $cmd = "stop";

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


#
# wait for BPSR to stop in the background
#
sub stopInBackground() 
{
  Dada::logMsg(2, $dl, "stopThread()");

  my $host = $cfg{"PWCC_HOST"};
  my $port = $cfg{"PWCC_PORT"};

  my $ignore = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $handle = 0;
  
  my $wait_inc = 5;
  my $wait_max = 20;
  my $wait = 0;
  my $nexus_state = 0;

  Dada::logMsg(2, $dl, "stopThread: opening connection to ".$host.":".$port);
  $handle = Dada::connectToMachine($host, $port);
  if (!$handle) 
  {
    Dada::logMsgWarn($error, "Could not connect to nexus to wait for IDLE");

    Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
    ($result, $response) = stopRoachTX();
    Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

    return;
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  # wait for the IDLE state
  while (($current_state eq "Stopping") && ($wait <= $wait_max))
  {

    Dada::logMsg(2, $dl, "stopThread: waitForState idle");
    $nexus_state = Dada::waitForState("idle", $handle, $wait_inc);
    Dada::logMsg(2, $dl, "stopThread: waitForState ".$nexus_state);

    $wait += $wait_inc;

    if ($nexus_state == 0)
    {
      Dada::logMsg(2, $dl, "stopThread: STATE now Idle");
      $current_state = "Idle";
    }
    elsif ($nexus_state == -1)
    {
      Dada::logMsg(1, $dl, "stopThread: waited ".$wait." of ".$wait_max.
                          " for idle state");
    }
    elsif ($nexus_state == -2)
    {
      Dada::logMsg(1, $dl, "stopThread: nexus in soft error state, attemping reset");

      $cmd = "reset";
      Dada::logMsg(1, $dl, "stopThread: PWCC <- ".$cmd);
      ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
      Dada::logMsg(1, $dl, "stopThread: PWCC -> ".$result.":".$response);

    }
    elsif ($nexus_state < -2) 
    {
      Dada::logMsg(1, $dl, "stopThread: nexus in error state [".$nexus_state."], attempting to reset");
      $handle->close();
      $handle = 0;
      Dada::logMsg(1, $dl, "stopThread: resetNexus(".$nexus_state.")");
      ($result, $response) = resetNexus($nexus_state);
      Dada::logMsg(1, $dl, "stopThread: resetNexus() ".$result." ".$response);

      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "stopThread: resetNexus failed: ".$response);
      }

      # if all goes well, the state should now be reset to Idle

      # reconnect to the nexus for next loop iteration
      Dada::logMsg(2, $dl, "stopThread: opening connection to ".$host.":".$port);
      $handle = Dada::connectToMachine($host, $port);
      if (!$handle)
      {
        Dada::logMsgWarn($error, "Could not connect to nexus to wait for IDLE");

        Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
        ($result, $response) = stopRoachTX();
        Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

        return;
      }

      # Ignore the "welcome" message
      $ignore = <$handle>;
    }
  }

  if ($handle) {
    $handle->close;
  }

  Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
  ($result, $response) = stopRoachTX();
  Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

  return;
}


###############################################################################
#
# connect to roach manager and tell it to stop transmitting 10GbE packets
#
sub stopRoachTX()
{
  my $cmd = "";
  my $result = "";
  my $response = "";

  my $host = Dada::getHostMachineName();
  my $port = $cfg{"IBOB_MANAGER_PORT"};

  my $roach_mngr = Dada::connectToMachine($host, $port);
  if (!$roach_mngr)
  {
    Dada::logMsg(1, $dl, "Could not connect to roach manager at ".$host.":".$port);
    return ("fail", "could not connect to roach manager");
  }

  # ignore welcome message
  $response = <$roach_mngr>;

  # issue stop command to stop flow of 10GbE packets
  $cmd = "stop_tx";
  Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($roach_mngr, $cmd);
  Dada::logMsg(2, $dl, "roach_mngr -> ".$result." ".$response);
  Dada::logMsg(1, $dl, "roach_mngr -> ".$result);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "roach_mngr -> ".$response);
  }

  $cmd = "exit";
  Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
  print $roach_mngr $cmd."\r\n";

  sleep(1);

  $roach_mngr->close();
  return ("ok" ,"");
}


#
# reset the nexus based on the Dada::waitForError status
#
sub resetNexus($)
{
  (my $nexus_state) = @_;
  Dada::logMsg(1, $dl, "resetNexus(".$nexus_state.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;

  my $pwcs_string = "";
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $pwcs_string .= " ".$cfg{"PWC_".$i};
  }

  # Tell the PWCC to quit
  Dada::logMsg(1, $dl, "PWCC <- quit");
  quitPWCCommand();
  $pwcc_thread->join();

  $cmd = "stop_pwcs";
  Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." <- ".$cmd);
    ($result, $response) = Bpsr::clientCommand($cmd, $cfg{"PWC_".$i});
    Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." -> ".$result.":".$response);
  }

  # if error state is fatal
  if ($nexus_state < -3) 
  {
    $cmd = "destroy_dbs";
    Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
    for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) 
    {
      Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." <- ".$cmd);
      ($result, $response) = Bpsr::clientCommand($cmd, $cfg{"PWC_".$i});
      Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." -> ".$result.":".$response);
    }

    $cmd = "init_dbs";
    Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
    for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) 
    {
      Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." <- ".$cmd);
      ($result, $response) = Bpsr::clientCommand($cmd, $cfg{"PWC_".$i});
      Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." -> ".$result.":".$response);
    }
  }

  $cmd = "start_pwcs";
  Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) 
  {
    Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." <- ".$cmd);
    ($result, $response) = Bpsr::clientCommand($cmd, $cfg{"PWC_".$i});
    Dada::logMsg(2, $dl, $cfg{"PWC_".$i}." -> ".$result.":".$response);
  }

  # Relaunch PWCC Thread with the current/previous config
  Dada::logMsg(1, $dl, "PWCC <- ".$cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg");
  $pwcc_thread = threads->new(\&pwcc_thread, $cfg{"CONFIG_DIR"}."/bpsr_tcs.cfg");

  if ($nexus_state == -4) {
    return ("ok", "FATAL ERROR occurred. PWCC, PWC's and DB restarted");
  } elsif ($nexus_state == -3) {
    return ("ok", "HARD ERROR occurred. PWCC and PWC's restarted");
  } else {
    return ("fail", "UNKOWN ERROR occurred");
  }

}


#
# Sends the "rec_stop" command to the Nexus
#
sub rec_stop($) {

  (my $utc) = @_;

  Dada::logMsg(2, $dl, "rec_stop (".$utc.")");

  my $ignore = "";
  my $result = "";
  my $response = "";

  my $handle = Dada::connectToMachine($cfg{"PWCC_HOST"}, $cfg{"PWCC_PORT"});
  if (!$handle) {
    return ("fail", "Could not connect to Nexus ".$cfg{"PWCC_HOST"}.":".$cfg{"PWCC_PORT"});
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  my $cmd = "rec_stop ".$utc;

  ($result, $response) = Dada::sendTelnetCommand($handle,$cmd);
  logMessage("Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

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
  sleep(5);
  Dada::logMsgWarn($warn, "STOPPING SCRIPT");
  exit(1);

}


#
# Thread to wait for quit signal
#
sub controlThread() 
{
  Dada::logMsg(1, $dl, "controlThread: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  # Poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_threads)) 
  {
    Dada::logMsg(3, $dl, "controlThread: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_threads = 1;

  # Manually tell dada_pwc_command to quit
  quitPWCCommand();

  Dada::logMsg(2, $dl, "controlThread: Unlinking PID file ".$pidfile);
  unlink($pidfile);

  Dada::logMsg(1, $dl, "controlThread: exiting");

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

  my @cmds = qw(ACC_LEN SOURCE RA DEC FA BW RECEIVER CFREQ PID PROC_FILE MODE RESOLUTION REF_BEAM NBEAM OBSERVER);
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


sub fixTCSCommands(\%) 
{

  my ($tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  my %fix = ();
  $fix{"src"}      = "SOURCE";
  $fix{"ra"}       = "RA";
  $fix{"dec"}      = "DEC";
  $fix{"band"}     = "BW";
  $fix{"freq"}     = "FREQ";
  $fix{"receiver"} = "RECEIVER";
  $fix{"acclen"}   = "ACC_LEN";
  $fix{"procfil"}  = "PROC_FILE";
  $fix{"nbit"}     = "NDECI_BIT";
  $fix{"tconst"}   = "T_CONST";
  $fix{"chanav"}   = "CHAN_AV";
  $fix{"nprod"}    = "N_PROD";
  $fix{"ftmax"}    = "FT_MAX";
  $fix{"pid"}      = "PID";
  $fix{"observer"} = "OBSERVER";
  $fix{"nbeam"}    = "NBEAM";
  $fix{"refbeam"}  = "REF_BEAM";
  $fix{"obsval"}   = "OBS_VAL";
  $fix{"obsunit"}  = "OBS_UNIT";
  $fix{"tscrunch"} = "TSCRUNCH";
  $fix{"fscrunch"} = "FSCRUNCH";

  my %add = ();
  $add{"STATE"}      = "PPQQ";
  $add{"MODE"}       = "PSR";
  $add{"FA"}         = "23";
  $add{"NCHAN"}      = "1024";
  $add{"NBIT"}       = "8";
  $add{"NPOL"}       = "2";
  $add{"NDIM"}       = "1";
  $add{"CFREQ"}      = "1382";
  $add{"RECEIVER"}   = "MULTI";
  $add{"RESOLUTION"} = "2048";
  $add{"BANDWIDTH"}  = "-400";

  my %new_cmds = ();
  my $key = "";

  foreach $key (keys (%tcs_cmds)) {

    if (exists $fix{$key}) {
      $new_cmds{$fix{$key}} = $tcs_cmds{$key};
    } else {
      $new_cmds{$key} = $tcs_cmds{$key};
    }
  }

  foreach $key (keys (%add)) {
    if (!(exists $new_cmds{$key})) {
      $new_cmds{$key} = $add{$key};
    }
  }

  if (!exists($new_cmds{"ACC_LEN"}) || ($new_cmds{"ACC_LEN"} eq "0") || ($new_cmds{"ACC_LEN"} == 0)) {
    $new_cmds{"ACC_LEN"} = 25;
  }

  if (!exists($new_cmds{"NDECI_BIT"}) || ($new_cmds{"NDECI_BIT"} eq "0") || ($new_cmds{"NDECI_BIT"} == 0)) {
    $new_cmds{"NDECI_BIT"} = 2;
  }

  if (!exists($new_cmds{"TSCRUNCH"}) || ($new_cmds{"TSCRUNCH"} eq "0") || ($new_cmds{"TSCRUNCH"} == 0)) {
    $new_cmds{"TSCRUNCH"} = 1;
  }

  if (!exists($new_cmds{"FSCRUNCH"}) || ($new_cmds{"FSCRUNCH"} eq "0") || ($new_cmds{"FSCRUNCH"} == 0)) {
    $new_cmds{"FSCRUNCH"} = 1;
  }

  $new_cmds{"PROC_FILE"} = uc($new_cmds{"PROC_FILE"});

  my $proj_id = $new_cmds{"PID"};
  if (!($proj_id =~ m/P\d\d\d/))
  {
    Dada::logMsgWarn($warn, "PID ".$proj_id." invalid, using P000 instead");
    $new_cmds{"PID"} = "P000";
  }

  return %new_cmds;
}

sub setActiveBeamsTCS(\%\%)
{
  my ($tcs_cmds_ref, $levels_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;
  my %levels = %$levels_ref;
  
  # defaults
  my $ref_beam = "01";
  my $n_beam = 13;
  my $i = 0;

  if ( exists($tcs_cmds{"REF_BEAM"}) && ($tcs_cmds{"REF_BEAM"} > 0) )
  {
    $ref_beam = sprintf("%02d", $tcs_cmds{"REF_BEAM"});
  }
  if ( exists($tcs_cmds{"NBEAM"}) && ($tcs_cmds{"NBEAM"} > 0) )
  {
    $n_beam = $tcs_cmds{"NBEAM"};
  }

  # initally, set all beams to on
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $tcs_cmds{"Band".$i."_BEAM_ACTIVE"} = "on";
  }

  # set the beam levels
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    if (exists($levels{$roach{"BEAM_".$i}}))
    {
      $tcs_cmds{"Band".$i."_BEAM_LEVEL"} = $levels{$roach{"BEAM_".$i}};
    }
  }

  # for single beam observing, disable all pwcs who do not match ref_beam
  if ($n_beam eq "1")
  {
    for ($i=0; $i<$roach{"NUM_ROACH"}; $i++)
    {
      if ($ref_beam ne $roach{"BEAM_".$i})
      {
        $tcs_cmds{"Band".$i."_BEAM_ACTIVE"} = "off";
      }
    }
  }

  return %tcs_cmds;
}

sub setActiveBeamsLocal(\%)
{
  my ($tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;

  my $beams_file = Dada::getDADA_ROOT()."/share/bpsr_active_beams.cfg";
  if ( ! -f $beams_file)
  {
    Dada::logMsg(2, $dl, "setActiveBeams: ".$beams_file." did not exist");
    return %tcs_cmds;
  }

  my %beams = Dada::readCFGFileIntoHash($beams_file, 0);

  # always expect 13 beams for BPSR, check file matches this
  my $i = 0;
  my $beam = "";
  my $config_valid = 1;
  for ($i=1; $i<=13; $i++)
  {
    $beam = sprintf("%02d", $i);
    if (!((exists($beams{"BEAM_".$beam})) && 
         (($beams{"BEAM_".$beam} eq "on") || ($beams{"BEAM_".$beam} eq "off") )))
    {
      Dada::logMsg(1, $dl, "setActiveBeams: problem with BEAM_".$beam." in beam file");
      $config_valid = 0;
    }
  }

  if (!$config_valid)
  {
    Dada::logMsgWarn($warn, "Active Beam file was malformed: ".$beams_file);
    return %tcs_cmds;
  }

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $beam = $roach{"BEAM_".$i};
    Dada::logMsg(2, $dl, "setActiveBeams: setting tcs{BAND".$i."_BEAM_ACTIVE} = ".$beams{"BEAM_".$beam});
    $tcs_cmds{"Band".$i."_BEAM_ACTIVE"} = $beams{"BEAM_".$beam};
  }

  return %tcs_cmds;
}


#
# Addds the required keys,values to the TCS commands based
# on the hardwired constants. These are:
#
# 1. Always sending to 16 PWCs
# 2. Lowest freq band goes to apsr00, highest to apsr15
# 3. 16 Bands (for the moment)
#
sub addHostCommands(\%\%) 
{

  my ($tcs_cmds_ref, $site_cfg_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;
  my %site_cfg = %$site_cfg_ref;
  my $key = "";

  $tcs_cmds{"NUM_PWC"}     = $cfg{"NUM_PWC"};
  $tcs_cmds{"HDR_SIZE"}    = $site_cfg{"HDR_SIZE"};

  # Determine the BW & FREQ for each channel
  my $cf = int($tcs_cmds{"CFREQ"});         # centre frequency
  my $bw = int($tcs_cmds{"BW"});            # bandwidth per beam

  my $i=0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $tcs_cmds{"Band".$i."_BW"} = $bw;
    $tcs_cmds{"Band".$i."_BEAM"} = $roach{"BEAM_".$i};
  }

  # Add the site configuration to tcs_cmds
  foreach $key (keys (%site_cfg)) {
    $tcs_cmds{$key} = $site_cfg{$key};
  }

  # Determine the TSAMP based upon NDIM and BW 
  $tcs_cmds{"TSAMP"} = calcTsampFromAccLen($tcs_cmds{"ACC_LEN"});

  # Set the instrument
  $tcs_cmds{"INSTRUMENT"} = uc($cfg{"INSTRUMENT"});
    
  return %tcs_cmds;

}

sub setRoachLevels()
{
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $localhost = Dada::getHostMachineName();
  my %levels = ();

  Dada::logMsg(1, $dl, "connecting to roach manager");
  my $roach_mngr = Dada::connectToMachine($localhost, $cfg{"IBOB_MANAGER_PORT"});
  if (!$roach_mngr)
  {
    Dada::logMsg(0, $dl, "Could not connect to roach manager");
    return ("fail", %levels);
  }

  # ignore welcome message
  $response = <$roach_mngr>;

  # ensure roach_mngr is in active state
  Dada::logMsg(1, $dl, "waiting for active state");
  ($result, $response) = Bpsr::waitForMultibobState("active", $roach_mngr, 2);
  Dada::logMsg(1, $dl, "waiting for active state: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "roach_mngr was not in the active state: ".$response);
    $roach_mngr->close();
    return ("fail", %levels);
  }

  # set the levels on the ROACH boards
  $cmd = "levels";
  Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($roach_mngr, $cmd);
  Dada::logMsg(1, $dl, "roach_mngr -> ".$result);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "roach_mngr -> ".$response);
  }
  else
  {
    Dada::logMsg(1, $dl, "roach_mngr -> ".$response);
    # use this information to determine if the LNAs for each roach are on (for heimdall)

    my @roaches = split(/ /, $response);
    my $beam = "";
    my $level = "";
    my $roach = "";
    foreach $roach (@roaches)
    {
      ($beam, $level) = split(/:/, $roach);
      $levels{$beam} = $level;
      Dada::logMsg(2, $dl, "setRoachLevels: setting levels{".$beam."}=".$level);
    }
  }

  # close the roach connection
  $cmd = "exit";
  Dada::logMsg(1, $dl, "roach_mngr <- ".$cmd);
  print $roach_mngr $cmd."\r\n";

  $roach_mngr->close();

  return ("ok", %levels);
}


#
# Generates the config file required for dada_pwc_command
#
sub generateConfigFile($\%) {

  my ($fname, $tcs_cmds_ref) = @_;

  my %tcs_cmds = %$tcs_cmds_ref;
  my $string = "";
  my $active_pwcs = $cfg{"NUM_PWC"};

  # special case for observing with primary beam [01] only
  if (exists($tcs_cmds{"NUM_BEAMS"}) && ($tcs_cmds{"NUM_BEAMS"} eq "1")) {
    $active_pwcs = 1;  
  }

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Header file created by ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  $string = Dada::headerFormat("NUM_PWC", $active_pwcs);
  print FH $string."\n";
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  $string = Dada::headerFormat("PWC_PORT",$cfg{"PWC_PORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);
 
  $string = Dada::headerFormat("PWC_LOGPORT",$cfg{"PWC_LOGPORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  $string = Dada::headerFormat("PWCC_PORT",$cfg{"PWCC_PORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  $string = Dada::headerFormat("PWCC_LOGPORT",$cfg{"PWCC_LOGPORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  $string = Dada::headerFormat("LOGFILE_DIR",$cfg{"SERVER_LOG_DIR"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);
 
  $string = Dada::headerFormat("HDR_SIZE",$tcs_cmds{"HDR_SIZE"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  $string = Dada::headerFormat("USE_BASEPORT", 1);
  print FH $string."\n";
  Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);

  my $i=0;
  for($i=0; $i<$active_pwcs; $i++) {
    $string = Dada::headerFormat("PWC_".$i,$cfg{"PWC_".$i});
    print FH $string."\n";
    Dada::logMsg(2, $dl, "bpsr_tcs.cfg: ".$string);
  }

  close FH;

  return ("ok", "");

}


sub generateSpecificationFile($\%) {

  my ($fname, $tcs_cmds_ref) = @_;
                                                                                                                                                                              
  my %tcs_cmds = %$tcs_cmds_ref;

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
      Dada::logMsg(2, $dl, "bpsr_tcs.spec: ".Dada::headerFormat($line, $tcs_cmds{$line}));
    }
  }

  close FH;
  return ("ok","");
}
  

sub calcTsampFromAccLen($) {
      
  (my $acclen) = @_;

  my $clock = 400; # Mhz
  my $nchannels = 1024;

  my $spectra_per_millisec = $clock / ($nchannels * $acclen);
  my $tsamp = 1 / $spectra_per_millisec;

  return $tsamp;

}

sub waitForMultibobBoot() 
{

  my $host = Dada::getHostMachineName();
  my $port = $cfg{"IBOB_MANAGER_PORT"};

  my $result;
  my $response;

  sleep(5);

  my $handle = Dada::connectToMachine($host, $port, 10);

  if (!$handle) {
    return ("fail", "Could not connect to multibob_server ".$host.":".$port);
  }

  # ignore welcome message
  $response = <$handle>;

  ($result, $response) = Bpsr::waitForMultibobState("active", $handle, 60);

  close($handle);

  return ($result, $response);

}
