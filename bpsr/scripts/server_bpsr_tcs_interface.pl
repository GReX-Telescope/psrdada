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

use constant DEBUG_LEVEL        => 1;     # 0 None, 1 Minimal, 2 Verbose
use constant PIDFILE            => "bpsr_tcs_interface.pid";
use constant LOGFILE            => "bpsr_tcs_interface.log";
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant DFBSIM_DURATION    => "3600";    # Simulator runs for 1 hour
use constant TERMINATOR         => "\r";

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
our %obs_config : shared    = ();

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
my %config = ();            # Hash of the 
my @specification = ();
my %all_config = ();
my $failure = "";
my $pwcc_thread = 0;
my $state_thread = 0;
my $daemon_control_thread = 0;
my $rh;

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


# Redirect standard output and error
Dada->daemonize($logfile, $pidfile);

logMessage("STARTING SCRIPT: ".Dada->getCurrentDadaTime(0));

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
        logMessage(1, "TCS handle = ".$handle);

      # Else we have received a command
      } else {

        $result = "";
        $response = "";

        # clean up the string
        my $cleaned_command = $command;
        $cleaned_command =~ s/\r//;
        $cleaned_command =~ s/\n//;
        $cleaned_command =~ s/ +$//;
        $command = $cleaned_command;

        @cmds = split(/ +/,$command,2);
        $key = $cmds[0];
        $val = $cmds[1];
        $lckey = lc $key;

        logMessage(1, "TCS -> ".$key."\t".$val);

        # START command will take all received header parameters and 
        # use them to launch dada_pwc_command and start the observation 
        #
        # WARNING!!!
        # We return an "ok" to TCS immediately so that it does not 
        # timeout whilst we start things up... This only ocurrs for 
        # the start command 
  
        if ($lckey eq "start") {

          logMessage(2, "Processing START command"); 
      
          logMessage(1, "Sending a fake \"ok\" to TCS"); 
          print $handle "ok".TERMINATOR;

          my $nwait = 5;
          while ((!$pwcc_running) && ($nwait > 0)) {
            logMessage(1, "Waiting for dada_pwc_command to start");
            sleep(1);
            $nwait--;
          }

          if (!$pwcc_running) {
            logMessage(0, "Error: dada_pwc_command is not running");
            $result = "fail";
            $response = "Error: dada_pwc_command is not running";

          } else {

            logMessage(1, "Waiting for dada_pwc_command AGAIN");
            sleep(1);

            my $key = "";
            my $value = "";

            # Write the received specificaiton to file
            ($result, $response) = writeSpecificationFile(\@specification);

            logMessage(2, "writeSpecificationFile returned (".$result.", ".$response.")");

            if ($result eq "ok") {

              # Tell the pwcc_command to start with the current specification? 
              ($result, $response) = start($response);

              if ($result eq "fail") {
                logMessage(0, "Error running start command \"".$response."\"");
              } else {
                logMessage(2, "Start command successful \"".$response."\"");
                %config = ();
                @specification = ();
                $current_state = "Recording";
              }

              # If simulating, start the DFB simulator
              if ($use_dfb_simulator) {

                ($result, $response) = startDFB();

                if ($result eq "ok") {
                  $utc_start = $response;
                  logMessage(1, "UTC_START = ".$utc_start);
                  ($result,$response) = set_utc_start($response);
                  %all_config = ();
                }
              }
            }
          }

        #
        # configure dada_pwc_command with the specification configuration file
        #
        } elsif ($lckey eq "config") {

          #logMessage(1, "Ignoring CONFIG command");

          logMessage(1, "Processing CONFIG command");

          # quit/kill the current daemon
          quit_pwc_command();

          # rejoin the thread
          $result = $pwcc_thread->join();

          logMessage(2, "dada_pwc_command thread joined");

          my $config_file = $config_dir."/".$val;

          if (!(-f $config_file)) {

            $result = "fail";
            $response = "CONFIG file ".$val." did not exist";

            # setup a "fake" config file
            #$config_file = $config_dir."/"."testing/16_2bit.cfg";
            #logMessage(0, "HARDCODING config file to: ".$config_file);
          } else {

            # parse the config file, checking for syntax
            ($result, $response) = parseConfigFile($config_file);

            if ($result eq "ok") {
      
              # write dada_pwc_command's config file (tcs.cfg)
              ($result, $response, $pwcc_thread) = writeConfigFile($config_file, $pwcc_file);

            }

            # Add the CONFIG parameter to the header so we know how the DFB3 
            # was configured
            push(@specification,$command);

          }


        # BAT command is the Baryonic Atomic Time. Needs to be converted to
        # UTC by Willem's fancy program

        } elsif ($lckey eq "bat") {

          logMessage(2, "Processing BAT command");
          logMessage(1, "Sending a fake \"ok\" to TCS");
          print $handle "ok".TERMINATOR;

          my $bindir = Dada->getCurrentBinaryVersion();
          my $utc_cmd = $bindir."/bat_to_utc ".$val;

          logMessage(2, "Running ".$utc_cmd);

          $utc_start = `$utc_cmd`;
  
          chomp $utc_start;

          ($result, $response) = set_utc_start(ltrim($utc_start));

          # After the utc has been set, we can reset the obs_config
          %obs_config = ();

        } elsif ($lckey eq "set_utc_start") {

          logMessage(2, "Processing SET_UTC_START command");

          $utc_start = $val;

          ($result, $response) = set_utc_start(ltrim($utc_start));

          # After the utc has been set, we can reset the obs_config
          %obs_config = ();

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

        } elsif ($lckey eq "rec_stop") {

          logMessage(2, "Processing REC_STOP command");
          ($result, $response) = rec_stop(ltrim($val));
          $current_state = "Idle";
          $utc_start = "";
 
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

        } elsif ($lckey eq "bandwidth") {

          # quietly ignore this parameter until it is fixed
          $result = "ok";

        } else {

          logMessage(3, "Processing HEADER parameter"); 
          push(@specification,$command);
          $result = "ok";
          $response = "";
          $current_state = "Preparing";

        }

        # Special "hack" case as we return "ok" to a start
        # command without waiting
        if (!(($lckey eq "start") || ($lckey eq "bat"))) {

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


###############################################################################
#
# Parses the specified "cfg" file to ensure that the minimal set of header 
# parameters are present. A hash of the values in the config file on success
#

sub parseConfigFile($) {

  (my $fname) = @_;

  if (!(-f $fname)) {
    return ("fail", "Config file $fname did not exist");
  }

  my $missing_params = "";
  my $malformed_params = "";
  my $result = "";
  my $response = "";
  my $i=0;

  # get a hash of the cfg file
  my %config = Dada->readCFGFileIntoHash($fname, 0);

  my @params_to_add = ();

  # These parameters MUST be declared in the DFB3 CONFIG file
  my @required_params = qw(NBAND BANDWIDTH);

  if (!(exists $config{"LOGFILE_DIR"})) {
    if (!(exists($config{"INSTRUMENT"}))) {
      push(@params_to_add,Dada->headerFormat("LOGFILE_DIR","logs/apsr"));
    } else {
      push(@params_to_add,Dada->headerFormat("LOGFILE_DIR","logs/".lc($config{"INSTRUMENT"})));
    }
  }

  # first check for NHOST
  if (!(exists $config{"NHOST"})) {
     $missing_params .= "NHOST ";
  } else {

    # If defined we need to ensure an "ADDRESS" and "BANDS" exist for each host
    for ($i=0;$i<$config{"NHOST"};$i++) {
      push(@required_params, "HOST".$i."_ADDRESS");
      push(@required_params, "HOST".$i."_BANDS");
    }
  }

  # next check for NCHANNEL
  if (!(exists $config{"NCHANNEL"})) {
     $missing_params .= "NCHANNEL ";
  } else {

    # If defined we need to ensure an "CHANi_FREQ" is defined for each channel
    for ($i=0;$i<$config{"NCHANNEL"};$i++) {
      push(@required_params, "CHAN".$i."_FREQ");
    }

  }

  my $param;

  # Now check that all the required parameters are defined
  foreach $param (@required_params) {
    if (!(exists $config{$param})) {
      $missing_params .= $param." ";
    }
    if (length($config{$param}) < 1) {
      $malformed_params .= $param." ";
    }
  }

  # Check that the number of bands sum correctly
  if (($missing_params ne "") || ($malformed_params ne "")) {

    my $total_bands = 0;

    for ($i=0; $i<$config{"NHOST"}; $i++) {
      my @num_bands = split(/,/, $config{"HOST".$i."_BANDS"});
      $total_bands += ($#num_bands + 1);
    }

    if ($total_bands != int($config{"NCHANNEL"})) {
      $malformed_params .= "NCHANNELS did not sum correctly with HOSTi_BANDS";
    }
  }

  # If we are missing any required parameters, return fail message
  if (($missing_params ne "") || ($malformed_params ne "")) {
    $result = "fail";
    if ($missing_params ne "") {
      $response .= "Missing header parameters \"".$missing_params."\"";
    } else {
      if ($malformed_params ne "") {
        $response .= " Malformed header parameters \"".$malformed_params."\"";
      }
    }
    logMessage(2,"parseConfigFile: $response");

  } else {

    logMessage(2,"parseConfigFile: config file valid");
    $result = "ok";

  }

  return ($result, $response);

}



###############################################################################
#
# Creates the .cfg file used to initialize dada_pwc_command. This file should contain:
#
# NUM_PWC, PWC_PORT, LOGFILE_DIR, HDR_SIZE, PWC_i (for i=0 -> NUM_PWC)
#

sub writeConfigFile($$) {

  my ($fname, $pwcc_file) = @_;

  logMessage(2,"writeConfigFile: input file ".$fname);

  my $FH = "";
  my %config = ();       # A hash containing key -> value pairs
  my $i = 0;

  # Ensure the obs_config is empty
  %obs_config = ();

  # Read the config file specified
  %config = Dada->readCFGFileIntoHash($fname,0);

  # Setup a few default parameters if they do not exist
  if (!(exists $config{"LOGFILE_DIR"})) {
    if (!(exists $config{"INSTRUMENT"})) {
      $config{"LOGFILE_DIR"} = "logs/apsr";
    } else {
      $config{"LOGFILE_DIR"} = "logs/".lc($config{"INSTRUMENT"});
    }
  }
  if (!(exists $config{"PWC_PORT"})) {
    $config{"PWC_PORT"} = "56026";
  }
  if (!(exists($config{"HDR_SIZE"}))) {
    $config{"HDR_SIZE"} = "4096";
  }

  # Make some modifications to the names/values
  my $NUM_PWC = int($config{"NHOST"});
  my $NCHANNEL_per_node = int($config{"NCHANNEL"}) / $NUM_PWC;
  my $BW_per_node = $config{"BANDWIDTH"} / $NUM_PWC;

  logMessage(2, "writeConfigFile: BANDWITH=".$config{"BANDWIDTH"}.", NUM_PWC=".$NUM_PWC.", BW_per_node=".$BW_per_node);

  # Make modifications to some parameters based on NUM_PWCs
  $obs_config{"NUM_PWC"} = $config{"NHOST"};
  $obs_config{"NCHANNEL"} = $NCHANNEL_per_node;
  $obs_config{"NBAND"} = $config{"NBAND"};
  $obs_config{"HDR_SIZE"} = $config{"HDR_SIZE"};
  $obs_config{"LOGFILE_DIR"} = $config{"LOGFILE_DIR"};
  $obs_config{"BANDWIDTH"} = $config{"BANDWIDTH"};
  $obs_config{"INSTRUMENT"} = $config{"INSTRUMENT"};

  my @bands = ();
  my $j=0;

  for ($i=0; $i<$NUM_PWC; $i++) {

     # hostname of receiving node
    $obs_config{"PWC_".$i} = $config{"HOST".$i."_ADDRESS"};

    # get the number of channels for this PWC
    @bands = split(/,/, $config{"HOST".$i."_BANDS"});

    for ($j=0;$j<=$#bands;$j++) {
      $obs_config{"Band".$i."_FREQ"} = $config{"CHAN".$bands[$j]."_FREQ"};
      $obs_config{"Band".$i."_BW"} = "$BW_per_node";
    }
  }

  logMessage(2,"writeConfigFile: creating pwcc header file: ".$pwcc_file);

  # else we got everything we need! woot
  open FH, ">$pwcc_file" or return ("fail","Could not create writeable header file: ".$pwcc_file);

  print FH "# Header file created by ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";
  print FH  Dada->headerFormat("NUM_PWC",$obs_config{"NUM_PWC"})."\n";
  print FH  Dada->headerFormat("PWC_PORT",$obs_config{"PWC_PORT"})."\n";
  print FH  Dada->headerFormat("LOGFILE_DIR",$obs_config{"LOGFILE_DIR"})."\n";
  print FH  Dada->headerFormat("HDR_SIZE",$obs_config{"HDR_SIZE"})."\n";
  for($i=0; $i<$obs_config{"NUM_PWC"};$i++) {
    print FH Dada->headerFormat("PWC_".$i,$obs_config{"PWC_".$i})."\n";
  }
  close FH;

  # Now that we have a successful header. Launch dada_pwc_command in 
  # a separate thread.
  my $thread_handle = threads->new(\&pwcc_thread, $pwcc_file);

  logMessage(2,"writeConfigFile: return \"ok\"");

  return ("ok", "", $thread_handle);

}

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
# create the specification file. We will have received the config file already
# and the specifcation parameters from TCS
#

sub writeSpecificationFile(\@) {

  (my $specificationRef) = @_;

  my @specification = @$specificationRef;

  # get the site specific configuration parameters
  my %site_config = Dada->readCFGFile($cfg{"CONFIG_DIR"}."/site.cfg");

  # file to be sent to dada_pwc_command
  my $fname = $cfg{"CONFIG_DIR"}."/tcs.spec";

  # minimal set of header parameters 
  my @required_header_params = qw(SOURCE RA DEC PID PROC_FILE RECEIVER CFREQ NBIT NDIM NPOL);

  my $param = "";
  my $missing_params = "";
  my $malformed_params = "";
  my $i = 0;

  my $key = "";
  my $value = "";
  
  # print out what we got 
  while ( ($key, $value) = each(%obs_config) ) {
    logMessage(2,"\$obs_config{".$key."} = ".$value);
  }
  foreach $value (@specification) {
    logMessage(2,"\@specification[i] = ".$value);
  }

  # Add the site.cfg specification parameters to the specification array
  foreach $key (keys (%site_config)) {
    push(@specification,Dada->headerFormat($key, $site_config{$key}));
  }

  # Generate the key/value combinations for the specification
  my @arr;
  my %spec_params = ();
  for ($i=0;$i<=$#specification;$i++) {
    $value = $specification[$i];
    # get rid of comments
    $value =~ s/#.*//;
    # skip blank lines
    if (length($value) > 0) {
      # strip comments
      @arr = split(/ +/,$value);
      if ((length(@arr[0]) > 0) && (length(@arr[1]) > 0)) {
        $spec_params{$arr[0]} = $arr[1];
        $obs_config{$arr[0]} = $arr[1];
      }
    }
  }

  foreach $value (@specification) {
    logMessage(2,"full spec:  ".$value);
  }

  foreach $key (keys (%spec_params)) {
    logMessage(2,"\$spec_param{".$key."} = ".$spec_params{$key});
  }


  # Check for the minimal parameters required
  foreach $param (@required_header_params) {
    if (!(exists $spec_params{$param})) {

      # have a default PROC_FILE  
      if ($param eq "PROC_FILE") {

        logMessage(0, "Manually setting PROC_FILE To dbdisk.scratch");
        push(@specification,Dada->headerFormat("PROC_FILE","dbdisk.scratch"));
        $spec_params{$param} = "dbdisk.scratch";
        $obs_config{$param} = "dbdisk.scratch";

      } else {
        $missing_params .= $param." ";
      }
    }
    if (length($spec_params{$param}) < 1) {
      $malformed_params .= $param." ";
    }
  }
  
  if (($missing_params ne "") || ($malformed_params ne "")) {
    my $response = "";
    if ($missing_params ne "") {
      $response .= "Missing header parameters \"".$missing_params."\"";
    } else {
      if ($malformed_params ne "") {
        $response .= " Malformed header parameters \"".$malformed_params."\"";
      }
    }
    logMessage(2,"Missing params \"".$response."\"");
    return ("fail", $response);
  }

  # Add the Tsamp HEADER parameter now that we know NDIM exists
  my $NUM_PWC = int($obs_config{"NUM_PWC"});
  my $BW_per_node = $obs_config{"BANDWIDTH"} / $NUM_PWC;
  my $TSAMP_per_node = 1.0 / $BW_per_node;
                                                                                                                                                                                            
  # if we are sampling complex values (NDIM=2) then our sampling rate is
  # half that of just real values, although we still require 2 "values"
  # for the complex sampling
  if (int($obs_config{"NDIM"}) == 1) {
    $TSAMP_per_node *= 0.5;
  }

  logMessage(2,"BANDWIDTH: ".$obs_config{"BANDWIDTH"}.", BW: ".$BW_per_node.", TSAMP: ".$TSAMP_per_node);
                                                                                                                                                                                            
  # Make modifications to some parameters based on NUM_PWCs
  $obs_config{"TSAMP"} = $TSAMP_per_node;

  logMessage(2, "adding: TSAMP => ".$obs_config{"TSAMP"});
  push(@specification,Dada->headerFormat("TSAMP", $obs_config{"TSAMP"}));

  logMessage(2,"Checking config parameters");

  # Now update the @specification array with values from 
  # the config array...
  my @chk_params = qw(NBAND NCHANNEL TSAMP INSTRUMENT);
  foreach $key (keys %obs_config) {
    # TODO update dspsr so that it recognizes CHANi_BW and CHANi_FREQ
    if ($key =~ m/^Band(\d+)_/) {
      push(@chk_params,$key);
    }
  }

  logMessage(2,"Checking centre frequency");

  my $matched = 0;
  my $centre_freq = "";

  # get the centre frequency
  for ($i=0; $i<$#specification; $i++) {
    @arr = split(/ +/,$specification[$i], 2);
    logMessage(2, "Specification:\t".$arr[0]."\t".$arr[1]);
    if ($arr[0] eq "CFREQ") {
      $centre_freq = $arr[1];
      # splice(@specification,$i,1);
      # logMessage(2, "removing CFREQ header param");
    }
    if (@arr[0] eq "FREQ") {
      splice(@specification,$i,1);
      logMessage(2, "removing FREQ header param");
    }

  }


  logMessage(2, "Centre Frequency = ".$centre_freq);  

  foreach $value (@chk_params) {

    #logMessage(1,"Added ".$value." ,".$config{$value});

    # Make the band FREQs absolute
    #TODO update dspsr so that it recognizes CHANi_BW and CHANi_FREQ    
    #if ($value =~ m/^Band(\d+)_CHAN(\d+)_FREQ/) 
    if ($value =~ m/^Band(\d+)_FREQ/) {
      logMessage(2,"Changing ".$value." from ".$obs_config{$value}." to ".($obs_config{$value} + $centre_freq));
      $obs_config{$value} += $centre_freq;
    }

    push(@specification,Dada->headerFormat($value,$obs_config{$value}));
  }

  if (-f $fname) {
    unlink($fname);
  }

  open FH,">$fname" or return ("fail","Could not create writeable header file: ".$fname);
  logMessage(1, "Writing specification to ".$fname);

  foreach $value (@specification) {
    print FH $value."\n";
  }
  close FH;

  return ("ok", $fname);
}


#
# Send the START command to the pwcc, optionally starting a DFB simualtor
#
sub start($) {
                                                                              
  my ($file) = @_;

  my $rVal = 0;
  my $cmd;

  my $result;
  my $response;

  # If we will run a separate DFB simulator
  if ($use_dfb_simulator) {

    # ARGS: host, dest port, nbit, npol, mode, duration 
    ($result, $response) = createDFBSimulator();

    # Give it half a chance to startup
    sleep(1);

  }

  # Connect to dada_pwc_command
  my $handle = Dada->connectToMachine($pwcc_host, $pwcc_port);

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
sub set_utc_start($) {

  (my $utc_start) = @_;

  logMessage(1,"set_utc_start(".$utc_start.")");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";

  # Now that we know the UTC_START, create the required results and archive 
  # directories and put the observation summary file there...

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;
  my $proj_id     = $obs_config{"PID"};

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
  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
                                                                                                                                                                              
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada->getCurrentDadaTime()."\n\n";
  print FH Dada->headerFormat("SOURCE",$obs_config{"SOURCE"})."\n";
  print FH Dada->headerFormat("RA",$obs_config{"RA"})."\n";
  print FH Dada->headerFormat("DEC",$obs_config{"DEC"})."\n";
  print FH Dada->headerFormat("CFREQ",$obs_config{"CFREQ"})."\n";
  print FH Dada->headerFormat("PID",$obs_config{"PID"})."\n";
  print FH Dada->headerFormat("BANDWIDTH",$obs_config{"BANDWIDTH"})."\n";
  print FH "\n";
  print FH Dada->headerFormat("NUM_PWC",$obs_config{"NUM_PWC"})."\n";
  print FH Dada->headerFormat("NBIT",$obs_config{"NBIT"})."\n";
  print FH Dada->headerFormat("NPOL",$obs_config{"NPOL"})."\n";
  print FH Dada->headerFormat("NDIM",$obs_config{"NDIM"})."\n";
  close FH;

  $cmd = "cp ".$fname." ".$archive_dir;
  system($cmd);

  # connect to nexus
  my $handle = Dada->connectToMachine($cfg{"PWCC_HOST"}, $cfg{"PWCC_PORT"});

  if (!$handle) {
    return ("fail", "Could not connect to Nexus ".$cfg{"PWCC_HOST"}.":".$cfg{"PWCC_PORT"});
  }

  # Ignore the "welcome" message
  $ignore = <$handle>;

  # Wait for the prepared state
  if (Dada->waitForState("recording",$handle,10) != 0) {
    return ("fail", "Nexus took more than 10 seconds to enter \"recording\" state");
  }

  # Send UTC Start command to the nexus
  $cmd = "set_utc_start ".$utc_start;

  ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
  logMessage(1,"Sent \"".$cmd."\", Received \"".$result." ".$response."\"");

  # Close nexus connection
  $handle->close();

  return ($result,$response);

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

sub createDFBSimulator() {

  my $host      =       $cfg{"DFB_SIM_HOST"};
  my $dest_port = "-p ".$cfg{"DFB_SIM_DEST_PORT"};

  my $nbit      = "-b ".$obs_config{"NBIT"};
  my $npol      = "-k ".$obs_config{"NPOL"};
  my $ndim      = "-g ".$obs_config{"NDIM"};
  my $tsamp     = "-t ".$obs_config{"TSAMP"};

  # By default set the "period" of the signal to 64000 bytes;
  my $calfreq   = "-c 64000";

  if ($obs_config{"MODE"} eq "CAL") {

    # if ($obs_config{"NBIT"} eq "2") {
      # START Hack whilst resolution is broken
      # $calfreq    = "-c ".($obs_config{"CALFREQ"}/2.0);
      # $npol       = "-k 1";
      # END  Hack whilst resolution is broken
    # } else {

      # Correct resolution changes
      $calfreq    = "-c ".$obs_config{"CALFREQ"};

    # }

  } else {
    $calfreq    = "-j ".$calfreq; 
  }

  my $drate = $obs_config{"NBIT"} * $obs_config{"NPOL"} * $obs_config{"NDIM"};
  $drate = $drate * (1.0 / $obs_config{"TSAMP"});
  $drate = $drate / 8;    # bits to bytes
  $drate = $drate * 1000000;

  $drate     = "-r ".$drate;
  my $duration  = "-n ".DFBSIM_DURATION;
  my $dest      = "192.168.1.255";
  

  logMessage(2,"createDFBSimulator: $host, $dest_port, $nbit, $npol, $ndim, $tsamp, $calfreq, $duration");

  my $args = "$dest_port $nbit $npol $ndim $tsamp $calfreq $drate $duration $dest";

  my $result = "";
  my $response = "";

  # Launch dfb simulator on remote host
  my $dfb_cmd = "dfbsimulator -d -a -y ".$args;
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



