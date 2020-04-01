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
use strict;         # strict mode (like -Wall)
use IO::Socket;     # Standard perl socket library
use IO::Select;     
use Net::hostent;
use File::Basename;
use threads;        # Perl threads module
use threads::shared; 
use XML::Simple qw(:strict);
use Data::Dumper;
use Dada;           # DADA Module for configuration options
use Mopsr;          # Mopsr Module for configuration options

sub quitPWCCommand();

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant PIDFILE            => "mopsr_tmc_interface.pid";
use constant LOGFILE            => "mopsr_tmc_interface.log";
use constant QUITFILE           => "mopsr_tmc_interface.quit";
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant TERMINATOR         => "\r";

use constant FURBY_DATABASE	    => "/home/dada/furby_database";

# We must always begin on a 3 second boundary since the pkt rearm UTC
use constant PKTS_PER_3_SECONDs => 390625;
#use constant PKTS_PER_3_SECONDs => 292969;

#
# Global variable declarations
#
our $dl;
our $daemon_name;
our %cfg : shared;
our %bf_cfg : shared;
our %site_cfg : shared;
our $current_state : shared;
our $current_config;
our $pwcc_running : shared;
our $quit_threads : shared;
our $n_ant : shared;
our $spec_generated : shared;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our $error;
our $warn;
our $pwcc_thread;
our $utc_stop : shared;
our $tobs_secs : shared;
our $utc_start : shared;
our $pkt_utc_start : shared;
our $hires;
our $packets_per_period : shared;
our $period_length: shared;
our $frb_detector_started : shared;

#
# global variable initialization
#
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
%bf_cfg = Mopsr::getConfig("bf");
%site_cfg = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);
$current_state = "Idle";
$current_config = $cfg{"CONFIG_NAME"};
$pwcc_running = 0;
$quit_threads = 0;
$n_ant  = "N/A";
$spec_generated = 0;
$pwcc_host = $cfg{"PWCC_HOST"};
$pwcc_port = $cfg{"PWCC_PORT"};
$client_master_port = $cfg{"CLIENT_MASTER_PORT"};
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
$pwcc_thread = 0;
$tobs_secs = -1;
$utc_stop = "";
$utc_start = "UNKNOWN";
$pkt_utc_start = "UNKNOWN";
$frb_detector_started = 0;
if (($cfg{"CONFIG_NAME"} =~ m/320chan/) || ($cfg{"CONFIG_NAME"} =~ m/312chan/))
{
  $hires = 1;
  $packets_per_period = 390625;
  $period_length = 4;
}
else
{
  $hires = 0;
  $packets_per_period = 390625;
  $period_length = 3;
}


#
# Main
#
{
  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  my $server_host =     $cfg{"SERVER_HOST"};
  my $config_dir =      $cfg{"CONFIG_DIR"};
  my $tmc_host =        $cfg{"TMC_INTERFACE_HOST"};
  my $tmc_port =        $cfg{"TMC_INTERFACE_PORT"};
  my $tmc_state_port =  $cfg{"TMC_STATE_INFO_PORT"};

  my $tmc_cmd;

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
  my $failure = "";
  my $state_thread = 0;
  my $control_thread = 0;
  my $rh;

  my $ant = "";
  my $cmd = "";
  my $xml = "";

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

  # read current UTC_START from the configuration file
  ($result, $pkt_utc_start) = getPktUtcStart();
  if ($result ne "ok")
  {
    print STDERR "ERROR: could not read packet UTC_START\n";   
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

  Dada::logMsg(1, $dl, "Opening socket for control commands on ".$tmc_host.":".$tmc_port);

  my $tmc_socket = new IO::Socket::INET (
    LocalHost => $tmc_host,
    LocalPort => $tmc_port,
    Proto => 'tcp',
    Listen => 1,
    ReuseAddr => 1
  );

  die "Could not create socket: $!\n" unless $tmc_socket;

  foreach $key (keys (%site_cfg)) {
    Dada::logMsg(2, $dl, "site_cfg: ".$key." => ".$site_cfg{$key});
  }

  my $localhost = Dada::getHostMachineName();
  my $pwcc_file = $config_dir."/mopsr_tmc.cfg";

  # Create the mopsr_tmc.cfg file to launch dada_pwc_command
  ($result, $response) = genConfigFile($cfg{"CONFIG_DIR"}."/mopsr_tmc.cfg");
  Dada::logMsg(1, $dl, "Generated initial config file ".$result.":".$response);

  $pwcc_thread = threads->new(\&pwcc_thread, $pwcc_file);
  Dada::logMsg(2, $dl, "dada_pwc_command thread started");

  # This thread will simply report the current state of the PWCC_CONTROLLER
  $state_thread = threads->new(\&state_reporter_thread, $server_host, $tmc_state_port);
  Dada::logMsg(2, $dl, "state_reporter_thread started");

  # Start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file, $quit_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($tmc_socket);   # add the main socket to the set

  # Main Loop,  We loop forever unless asked to quit
  while (!$quit_threads)
  {
    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
    Dada::logMsg(3, $dl, "select on read_set returned");
                                                                                  
    foreach $rh (@$readable_handles)
    {
      if ($rh == $tmc_socket)
      {
        # Only allow 1 connection from TMC
        if ($handle)
        {
          $handle = $tmc_socket->accept() or die "accept $!\n";

          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          my $host = "localhost";
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsgWarn($warn, "Rejecting additional connection from ".$host);
          $handle->close();
        }
        else
        {
          # Wait for a connection from the server on the specified port
          $handle = $tmc_socket->accept() or die "accept $!\n";

          # Ensure commands are immediately sent/received
          $handle->autoflush(1);

          # Add this read handle to the set
          $read_set->add($handle);

          # Get information about the connecting machine
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          my $host = "localhost";
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsg(2, $dl, "Accepting connection from ".$host);
        }
      }
      else
      {
        $command = <$rh>;
    
        # If we have lost the connection...
        if (! defined $command)
        {
          my $host = "localhost";
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsg(2, $dl, "lost connection from ".$host);

          $read_set->remove($rh);
          close($rh);
          $handle->close();
          $handle = 0;

        # Else we have received a command
        }
        else
        {
          ($result, $response, $xml) = parseXMLCommand($command);
          if ($result ne "ok")
          {
            Dada::logMsg(1, $dl, "failed to parse XML command: ".$response);
          }
          print $handle $xml.TERMINATOR;
        }
      }
    }

    # if the global UTC_STOP is specified, check if this time is within 10 seconds
    if ($utc_stop ne "")
    {
      # get the current unix time :)
      my $curr_time_unix = time;

      my $curr_utc = Dada::printTime($curr_time_unix, "utc");

      # convert UTC_STOP to unix time
      my $stop_time_unix = Dada::getUnixTimeUTC($utc_stop);

      my $remaining = $stop_time_unix - $curr_time_unix;
      if (($remaining % 60 == 0) || ($remaining <= 10))
      {
        Dada::logMsg (1, $dl, "UTC_STOP=".$utc_stop." UTC=".$curr_utc.", stopping in ".$remaining." s");
      }

      if ($curr_time_unix + 3 > $stop_time_unix)
      {
        Dada::logMsg (1, $dl, "performing UTC_STOP");

        $command = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
        $command .= "<mpsr_tmc_message>\n";
        $command .= "<command>stop</command>\n";
        $command .= "<utc_date>".$utc_stop."</utc_date>\n";
        $command .= "</mpsr_tmc_message>\n";

        ($result, $response, $xml) = parseXMLCommand($command);
        if ($result ne "ok")
        {
          Dada::logMsg(1, $dl, "failed to parse XML command: ".$response);
        }
        else
        {
          Dada::logMsg(1, $dl, "Stop command worked!\n"); 
        }
        $utc_stop = "";
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

sub parseXMLCommand($)
{
  my ($xml_in) = @_;

  my $xml_out;
  my $xml;
  my $cmd = "";
  my $result = "ok";
  my $response = "";
  my $command = "";
  my $utc_date = "";

  Dada::logMsg(2, $dl, "parseXMLCommand: eval XML");
  Dada::logMsg(3, $dl, "parseXMLCommand: RAW=".$xml_in);
  eval {
    $xml = XMLin ($xml_in, ForceArray => 0, KeyAttr => 0, SuppressEmpty => 1, NoAttr => 1);
  };
   
  # If the XML parsing failed 
  if ($@)
  {
    $result = "fail";
    $response = "failed to parse xml";
  }
  else
  {
    # check that a command was specified
    my $has_command = eval { exists $xml->{'command'} };
    if (!$has_command)
    {
      $result = "fail";
      $response = "No command was specified";
    }
    else
    {
      $command = $xml->{'command'};
      $utc_date = "";
      if (eval { exists $xml->{'utc_date'} } )
      {
        $utc_date = $xml->{'utc_date'};
      }

      $xml_out = "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml_out .= "<mpsr_tmc_message>";

      if ($command eq "prepare")
      {
        if (($current_state eq "Idle") || ($current_state eq "Prepared"))
        {

          Dada::logMsg(1, $dl, "received prepare command");
          # check all the meta-data has been supplied
          my %required = ('west_arm_parameters' => ['tracking', 'ns_tilt', 'md_angle'],
            'east_arm_parameters' => ['tracking', 'ns_tilt', 'md_angle'],
            'signal_parameters' => ['bandwidth', 'centre_frequency', 'nant', 'nchan', 'ndim', 'npol', 'nbit'],
            'pfb_parameters' => ['oversampling_ratio', 'sampling_time', 'channel_bandwidth', 'dual_sideband', 'resolution'],
            'observation_parameters' => ['observer', 'tobs'],
            'boresight_parameters' => [ 'project_id', 'name', 'ra', 'dec', 'rfi_mitigation', 'antenna_weights', 'delay_tracking', 'processing_file' ] );

          my ($set, $param);
          foreach $set (keys %required) 
          {
            foreach $param (@{$required{$set}})
            {
              Dada::logMsg(2, $dl, "parseXMLCommand: checking [".$set."][".$param."]");
              if (! eval { exists $xml->{$set}{$param} } )
              {
                $result = "fail";
                $response .= "MISSING PARAM: ".$set."/".$param." ";
              }
              else
              {
                if ($xml->{$set}{$param} eq "")
                {
                  $result = "fail";
                  $response .= "EMPTY PARAM: ".$set."/".$param." ";
                }
              }
            }
          }

   
          # correlation mode and fan/tied modes are mutually exclusive due to performance constraints
          my ($i, $tb_key);
          my @tb_keys = ();
          for ($i=0; $i<$bf_cfg{"NUM_TIED_BEAMS"}; $i++)
          {
            push (@tb_keys, "tied_beam_".$i."_parameters");
          }

          if ((eval { exists $xml->{'correlation_parameters'} }) && (eval { exists $xml->{'fan_beams_parameters'} }))
          {
            $result = "fail";
            $response .= "Cannot support Correlation && Fan Beams";
          }

          if ((eval { exists $xml->{'correlation_parameters'} }) && (eval { exists $xml->{'mod_beams_parameters'} }))
          {
            $result = "fail";
            $response .= "Cannot support Correlation && Module Beams";
          }

          if ((eval { exists $xml->{'correlation_parameters'} }) && (eval { exists $xml->{'furbies'} }))
          {
            if ((eval { exists $xml->{'furbies'}{'num_furbies'} }) && ($xml->{'furbies'}{'num_furbies'} > 0))
            {
              $result = "fail";
              $response .= "Cannot add furbies in Correlation mode";
            }
          }

          if ((eval { exists $xml->{'furbies'} }) && (eval { exists $xml->{'mod_beams_parameters'} }))
          {
            if ((eval { exists $xml->{'furbies'}{'num_furbies'} }) && ($xml->{'furbies'}{'num_furbies'} > 0))
            {
              $result = "fail";
              $response .= "Cannot add furbies in Module beam mode";
            }
          }

          if ((eval { exists $xml->{'fan_beams_parameters'} }) && (eval { exists $xml->{'mod_beams_parameters'} }))
          {
            $result = "fail";
            $response .= "Cannot support Fan Beams && Module Beams";
          }

          foreach $tb_key (@tb_keys)
          {
            if ((eval { exists $xml->{'correlation_parameters'} }) && (eval { exists $xml->{$tb_key} }))
            {
              $result = "fail";
              $response .= "Cannot support Correlation && Tied Beams";
            }
          }

	  #VG: Adding furby params here in optionals
          my %opt_req = ('correlation_parameters' => ['mode', 'project_id', 'type', 'processing_file', 'dump_time'],
                         'fan_beams_parameters' => ['mode', 'project_id', 'nbeams', 'beam_spacing'],
                         'mod_beams_parameters' => ['mode', 'project_id'], 
                         'furbies' => ['num_furbies']);

          foreach $tb_key (@tb_keys)
          {
            $opt_req{$tb_key} = ['mode', 'project_id', 'name', 'ra', 'dec', 'processing_file'];
          }

          foreach $set (keys %opt_req)
          {
            if (eval { exists $xml->{$set}})
            {
              foreach $param (@{$opt_req{$set}})
              {
                Dada::logMsg(2, $dl, "parseXMLCommand: checking [".$set."][".$param."]");
                if (! eval { exists $xml->{$set}{$param} } )
                {
                  $result = "fail";
                  $response .= "MISSING PARAM: ".$set."/".$param." ";
                }
                else
                {
                  if ($xml->{$set}{$param} eq "")
                  {
                    $result = "fail";
                    $response .= "EMPTY PARAM: ".$set."/".$param." ";
                  }
                }
              }
            }
          }

          if ($result ne "fail")
          {
            # generate specification file
            Dada::logMsg(2, $dl, "main: genSpecFile(".$cfg{"CONFIG_DIR"}."/mopsr_tmc.spec)");
            ($result, $response) = genSpecFile ($cfg{"CONFIG_DIR"}."/mopsr_tmc.spec", $xml);
            Dada::logMsg(2, $dl, "main: genSpecFile() ".$result." ".$response);
           
            # ensure that we cannot start without a fresh specification file
            $spec_generated = 1;
            $current_state = "Prepared";
            if ($result eq "ok")
            {
              $response =  "parsed correctly";
            }
          }
        }
        else
        {
          $result = "fail";
          $response = "Received stop command whilst ".$current_state;
        }

        $xml_out .=   "<reply>".$result."</reply>";
        $xml_out .=   "<response>".$response."</response>";
      }
      elsif ($command eq "start")
      {
        Dada::logMsg(1, $dl, "received start command");
        if (!$spec_generated)
        {
          $result = "fail";
          $response = "did not receive prepare command with valid meta-data";
        }
        else
        {
          $current_state = "Starting";
          Dada::logMsg(2, $dl, "parseXMLCommand: start(".$cfg{"CONFIG_DIR"}."/mopsr_tmc.spec)");
          ($result, $response) = start($cfg{"CONFIG_DIR"}."/mopsr_tmc.spec");
          Dada::logMsg(2, $dl, "parseXMLCommand: start() ".$result." ".$response);
          $current_state = "Recording";

        }
        $spec_generated = 0;
        $xml_out .=   "<reply>".$result."</reply>";
        $xml_out .=   "<response>".$response."</response>";
      }
      elsif ($command eq "stop")
      {
        Dada::logMsg(1, $dl, "received stop command");
        if (($current_state eq "Stopping") || ($current_state eq "Idle"))
        {
          $result = "fail";
          $response = "Received stop command whilst ".$current_state;
        }
        elsif ($current_state eq "Prepared")
        {
          $result = "ok";
          $response = "Returning from Prepared to Idle";
          $current_state = "Idle";
          $spec_generated = 0;
        }
        else
        {
          $current_state = "Stopping";
          
          # disable any previous pending auto-stop commands
          $utc_stop = "";

          # issue stop command to nexus
          Dada::logMsg(2, $dl, "parseXMLCommand: stopNexus(".$utc_date.")");
          ($result, $response) = stopNexus($utc_date);
          Dada::logMsg(2, $dl, "parseXMLCommand: stopNexus ".$result." ".$response);

          if ($result ne "ok")
          {
            Dada::logMsgWarn($error, "Stop command failed");
            $current_state = "ERROR: ".$response;
          }
          else
          {
            Dada::logMsg(2, $dl, "parseXMLCommand: stopInBackground()");
            # stopInBackground();
            Dada::logMsg(2, $dl, "parseXMLCommand: stopInBackground completed");
            $current_state = "Idle";
          }
          $spec_generated = 0;
        }
        $xml_out .= "<reply>".$result."</reply>";
        $xml_out .= "<response>".$response."</response>";
      }
      elsif ($command eq "query")
      {
        $xml_out .= "<reply>".$result."</reply>";
        $xml_out .= "<response>";
        $xml_out .=   "<mpsr_status>".$current_state."</mpsr_status>";
        $xml_out .=   "<mpsr_configuration>".$current_config."</mpsr_configuration>";
        $xml_out .=   "<snr_status>";
        $xml_out .=     "<source>";
        $xml_out .=       "<name epoch='J2000'>J0437-4715</name>";
        $xml_out .=       "<snr>1.23</snr>";
        $xml_out .=     "</source>";
        $xml_out .=   "</snr_status>";
        $xml_out .=   "<snr_status>";
        $xml_out .=     "<source>";
        $xml_out .=       "<name epoch='J2000'>J0835-4510</name>";
        $xml_out .=       "<snr>4.23</snr>";
        $xml_out .=     "</source>";
        $xml_out .=   "</snr_status>";
        $xml_out .= "</response>";
      }
      else
      {
        $result = "fail";
        $response = "unrecognized command [".$command."]";
        $xml_out .=   "<reply>".$result."</reply>";
        $xml_out .=   "<response>".$response."</response>";
      }
    }
    if ($dl > 1)
    {
      print Dumper($xml);
    }
  }

  $xml_out .= "</mpsr_tmc_message>";

  return ($result, $response, $xml_out);
}

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
    Listen => 8,
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
        my $host = "localhost";
        if (defined $hostinfo)
        {
          $host = $hostinfo->name;
          Dada::logMsg(3, $dl, "state_reporter: accepting connection from ".$host);
        }
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

          if ($result eq "num_ant")
          {
            print $rh $n_ant."\r\n";
            Dada::logMsg(3, $dl, "state_reporter: -> ".$n_ant);
          }

          if ($result eq "utc_start")
          {
            print $rh $utc_start."\r\n";
            Dada::logMsg(3, $dl, "state_reporter: -> ".$utc_start);
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
    Dada::logMsg(0, $dl, "quitPWCCommand: killProcess(dada_pwc_command.*mopsr_tmc.cfg)");
    ($result, $response) = Dada::killProcess("dada_pwc_command.*mopsr_tmc.cfg");
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
sub start($)
{
  my ($file) = @_;

  my $rVal = 0;
  my ($cmd, $result, $response);
  my $localhost = Dada::getHostMachineName();
  my $start_packet = 1;

  open FH, ">>$file" or return ("fail", "Could not create writeable file: ".$file);

  my %h = Dada::readCFGFileIntoHash($file, 0);
  Dada::logMsg(1, $dl, "start: MODE='".$h{"MODE"}."' PKT_UTC_START='".$pkt_utc_start."'");
  if (!($pkt_utc_start =~ m/UNKNOWN/))
  {
    # get the unix time for the PKT UTC_START and the current time
    my $pkt_start_unix  = Dada::getUnixTimeUTC($pkt_utc_start);
    my $curr_time_unix  = time; 

    Dada::logMsg(2, $dl, "start: pkt_start_unix=".$pkt_start_unix);
    Dada::logMsg(2, $dl, "start: curr_time_unix=".$curr_time_unix);
   
    # plan to start in 5 secs, modulo 3s since pkt_start_unix
    my $obs_start_unix = $curr_time_unix + 5;
    Dada::logMsg(2, $dl, "start: obs_start_unix=".$obs_start_unix);
    my $remainder = ($obs_start_unix - $pkt_start_unix) % $period_length;
    Dada::logMsg(2, $dl, "start: remainder=".$remainder);
    if ($remainder > 0)
    {
      $obs_start_unix += ($period_length - $remainder);
    }
    Dada::logMsg(2, $dl, "start: obs_start_unix=".$obs_start_unix);

    $utc_start = Dada::printTime ($obs_start_unix, "utc");
    Dada::logMsg(1, $dl, "UTC_START=".$utc_start);

    # determine packet offset for this start time
    my $offset = $obs_start_unix - $pkt_start_unix;
    Dada::logMsg(2, $dl, "start: packet offset=".$offset);

    # the start packet necessary for the obs_start_unix
    $start_packet = 1 + (($offset / $period_length) * $packets_per_period);
    Dada::logMsg(2, $dl, "start: start_packet=".$start_packet);

    # write the start packet to the spec file
    print FH Dada::headerFormat("UTC_START", $utc_start)."\n";

    open FH2, ">".$cfg{"CONFIG_DIR"}."/last_utc.start";
    print FH2 $utc_start."\n";
    close FH2;
  }

  print FH Dada::headerFormat("PKT_START", $start_packet)."\n";
  close FH;

  # clear the status files
  $cmd = "rm -f ".$cfg{"STATUS_DIR"}."/*";
  Dada::logMsg(2, $dl, "start: ".$cmd);
  my ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "start: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "Could not delete status files: $response");
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
    Dada::logMsg(2, $dl, "PWCC waiting for prepared");
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

    if ($start_packet == 1)
    {
      # Run the rearming script on ibob manager
      $cmd = "/home/swin/pfb_rearm";
      my $user = "swin";
      my $host = "skamp2";
      my $rval = 0;

      Dada::logMsg(1, $dl, "start: ".$user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(1, $dl, "start: ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        logMsg(0, "WARN", "start: ssh failure to ".$user."@".$host.": ".$response);
        return ("fail", "could not ssh to ".$host." to re-arm");
      }
      elsif ($rval != 0)
      {
        logMsg(0, "WARN", "start: ".$user."@".$host.":".$cmd." failed:  ".$response);
        return ("fail", "could re-arm command failed");
      }
      else
      {
        $utc_start = $response;
        logMsg(0, "INFO", "start: UTC_START=".$utc_start);

        # since this is a CORR_CAL observation, record the start time
        setPktUtcStart ($utc_start);
      }
    }
      
    # if a TOBS was specified in the command from TMC
    if ($tobs_secs > 0)
    {
      $utc_stop = Dada::addToTime($utc_start, $tobs_secs);
      logMsg(0, "INFO", "TOBS=".$tobs_secs." UTC_STOP=".$utc_stop);
    }
    else
    {
      logMsg(0, "INFO", "TOBS=".$tobs_secs." No UTC_STOP specified");
    }

    # Setup the server output directories before telling the clients to begin
    Dada::logMsg(1, $dl, "start: createLocalDirs (".$utc_start.")");
    ($result, $response) = createLocalDirs ($utc_start);
    Dada::logMsg(2, $dl, "start: createLocalDirs() ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, $response);
    }

    # Wait for the recording state
    Dada::logMsg(2, $dl, "start: waiting for nexus to enter RECORDING state");
    if (Dada::waitForState("recording", $handle, 30) != 0) {
      return ("fail", "nexus did not enter RECORDING state after start command");
    }
    Dada::logMsg(2, $dl, "start: nexus now in RECORDING state");

    # If we are doing a re-arm we now have a UTC_START
    if ($start_packet == 1)
    {
      $cmd = "set_utc_start ".$utc_start;
      Dada::logMsg(1, $dl, "PWCC <- ".$cmd);
      ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
      Dada::logMsg(1, $dl, "PWCC -> ".$result);
      if ($result ne "ok") 
      {
        Dada::logMsgWarn($warn, $response);
      }
    }

    # Close nexus connection
    $handle->close();

    if ($h{"FB_ENABLED"} eq "true")
    {
      # start the FRB detector
      startFRBDetector ($file, $utc_start);
      $frb_detector_started = 1;
    }

    return ($result, $utc_start);
  }
}

sub startFRBDetector ($$)
{
  my ($spec_file, $utc_start) = @_;
  my %h = Dada::readCFGFileIntoHash($spec_file, 0);
  my $xml;

  my $observing_type = "TRACKING";
  if ($h{"DELAY_TRACKING"} eq "false")
  {
    #if ($h{"MD_ANGLE"} == 0)
    #{
      $observing_type = "STATIONARY";
    #}
    #else
    #{
    #  $observing_type = "TRANSITING";
    #}
  }

  $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
  $xml .= "<frb_detector_message>";
  $xml .=   "<cmd>start</cmd>";
  $xml .=   "<source>FB</source>";
  $xml .=   "<utc_start>".$utc_start."</utc_start>";
  $xml .=   "<ra>".$h{"RA"}."</ra>";
  $xml .=   "<dec>".$h{"DEC"}."</dec>";
  $xml .=   "<md_angle>".$h{"MD_ANGLE"}."</md_angle>";
  $xml .=   "<ns_tilt>".$h{"NS_TILT"}."</ns_tilt>";
  $xml .=   "<pid>".$h{"PID"}."</pid>";
  $xml .=   "<mode>PSR</mode>";
  $xml .=   "<config>".$h{"CONFIG"}."</config>";
  $xml .=   "<observing_type>".$observing_type."</observing_type>";
  $xml .=   "<nchan>".$h{"NCHAN"}."</nchan>";
  $xml .=   "<nbit>".$h{"NBIT"}."</nbit>";
  $xml .=   "<tsamp>".$h{"TSAMP"}."</tsamp>";
  $xml .=   "<pulsar_flagging>true</pulsar_flagging>";
  $xml .=   "<beam_spacing units='degrees'>".$h{"FB_BEAM_SPACING"}."</beam_spacing>";
  $xml .= "</frb_detector_message>";

  Dada::logMsg(2, $dl, "startFRB: ".$xml);
  
  Dada::logMsg(1, $dl, "FRBD <- start");
  my ($result, $response) = sendFRBDetectorMessage($xml);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, $response);
  }
  Dada::logMsg(1, $dl, "FRBD -> ".$result." ".$response);
}

sub stopFRBDetector ($)
{
  my ($utc_stop) = @_;
  my $xml;

  $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
  $xml .= "<frb_detector_message>";
  $xml .=   "<cmd>stop</cmd>";
  $xml .=   "<utc_stop>".$utc_stop."</utc_stop>";
  $xml .= "</frb_detector_message>";

  Dada::logMsg(1, $dl, "FRBD <- stop");
  my ($result, $response) = sendFRBDetectorMessage($xml);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, $response);
  }
  Dada::logMsg(1, $dl, "FRBD -> ".$result." ".$response);
}

sub sendFRBDetectorMessage ($)
{
  my ($xml) = @_;

  # open a socket to the server_mopsr_frb_detector
  my $frb_host = $cfg{"SERVER_HOST"};
  my $frb_port = $cfg{"FRB_DETECTOR_BASEPORT"};

  my $handle = Dada::connectToMachine($frb_host, $frb_port, 1);
  if (!$handle)
  {
    return ("fail", "could not connect to FRB Detector on ".$frb_host.":".$frb_port);
  }
  

  print $handle $xml."\r\n";
  my $response = Dada::getLine($handle);
  #my @lines = <$handle>;
  #my $response = "";
  #my $line;
  #foreach $line (@lines)
  #{
  #  $response .= $line;
  #}
    
  close ($handle);

  eval {
    $xml = XMLin ($response, ForceArray => 0, KeyAttr => 0, SuppressEmpty => 1, NoAttr => 1);
  };

  # If the XML parsing failed 
  if ($@)
  {
    return ("fail", "failed to parse XML from FRB detector");
  }
  else
  {
    # check if reply is ok | fail
    if (! eval { exists $xml->{'reply'} } )
    {
      return ("fail", "Malformed XML reply from FRB detector");
    }
    else
    {
      if ($xml->{'reply'} ne "ok")
      {
         return ("fail", "FRB detector returned fail");
      }
    }
  }
  return ("ok", "")
}

sub createLocalDirs($)
{
  my ($utc_start) = @_;

  Dada::logMsg(1, $dl, "createLocalDirs: reading ".$cfg{"CONFIG_DIR"}."/mopsr_tmc.spec");

  my %spec = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/mopsr_tmc.spec", 0);
  my ($cmd, $result, $response); 

  my $rdir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
  my $adir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;

  # Now that we know the UTC_START, create the required results and archive
  # directories and put the observation summary file there...
  Dada::logMsg(1, $dl, "createLocalDirs: creating local dirs");
  Dada::mkdirRecursive($adir, 0755);
  Dada::mkdirRecursive($rdir, 0755);

  my $fname = $rdir."/obs.info";
  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  print FH Dada::headerFormat("SOURCE",    $spec{"SOURCE"})."\n";
  print FH Dada::headerFormat("RA",        $spec{"RA"})."\n";
  print FH Dada::headerFormat("DEC",       $spec{"DEC"})."\n";
  print FH Dada::headerFormat("MD_ANGLE",  $spec{"MD_ANGLE"})."\n";
  print FH Dada::headerFormat("NS_TILT",   $spec{"NS_TILT"})."\n";
  print FH Dada::headerFormat("FREQ",      $spec{"FREQ"})."\n";
  print FH Dada::headerFormat("PID",       $spec{"PID"})."\n";
  print FH Dada::headerFormat("BW",        $spec{"BW"})."\n";
  print FH Dada::headerFormat("MODE",      $spec{"MODE"})."\n";
  print FH Dada::headerFormat("CONFIG",    $spec{"CONFIG"})."\n";
  #print FH Dada::headerFormat("OBSERVING_TYPE", $spec{"OBSERVING_TYPE"})."\n";
  print FH Dada::headerFormat("UTC_START", $utc_start)."\n";
  print FH "\n";
  print FH Dada::headerFormat("NUM_PWC",   $cfg{"NUM_PWC"})."\n";
  print FH Dada::headerFormat("NCHAN",     $spec{"NCHAN"})."\n";
  print FH Dada::headerFormat("NBIT",      $spec{"NBIT"})."\n";
  print FH Dada::headerFormat("NPOL",      $spec{"NPOL"})."\n";
  print FH Dada::headerFormat("NDIM",      $spec{"NDIM"})."\n";
  print FH Dada::headerFormat("NANT",      int($spec{"NANT"}) * int($cfg{"NUM_PWC"}))."\n";
  print FH Dada::headerFormat("AQ_PROC_FILE", $spec{"AQ_PROC_FILE"})."\n";
  print FH Dada::headerFormat("OBSERVER",  $spec{"OBSERVER"})."\n";
  print FH "\n";
  print FH Dada::headerFormat("TRACKING",  $spec{"TRACKING"})."\n";
  print FH Dada::headerFormat("DELAY_TRACKING", $spec{"DELAY_TRACKING"})."\n";
  print FH Dada::headerFormat("ANTENNA_WEIGHTS", $spec{"ANTENNA_WEIGHTS"})."\n";
  print FH Dada::headerFormat("RFI_MITIGATION", $spec{"RFI_MITIGATION"})."\n";
  print FH "\n";
  print FH Dada::headerFormat("FB_ENABLED", $spec{"FB_ENABLED"})."\n";
  print FH Dada::headerFormat("MB_ENABLED", $spec{"MB_ENABLED"})."\n";
  print FH Dada::headerFormat("CORR_ENABLED", $spec{"CORR_ENABLED"})."\n";
  if ($spec{"CORR_ENABLED"} eq "true")
  {
    print FH Dada::headerFormat("CORR_SOURCE", $spec{"SOURCE"})."\n";
    Dada::mkdirRecursive($rdir."/".$spec{"SOURCE"}, 0755);
    Dada::mkdirRecursive($adir."/".$spec{"SOURCE"}, 0755);
  }
  if (($spec{"FB_ENABLED"} eq "true") || ($spec{"MB_ENABLED"} eq "true"))
  {
    Dada::mkdirRecursive($rdir."/FB", 0755);
    Dada::mkdirRecursive($adir."/FB", 0755);
    
    if ( ($spec{"INJECTED_FURBYS"} + 0)  > 0)
    {
	Dada::mkdirRecursive($adir."/Furbys", 0755);
        print FH Dada::headerFormat("INJECTED_FURBYS", $spec{"INJECTED_FURBYS"})."\n";
    }
  }

  my $i;
  for ($i=0; $i<$bf_cfg{"NUM_TIED_BEAMS"}; $i++)
  {
    my $prefix = "TB".$i;
    print FH Dada::headerFormat($prefix."_ENABLED", $spec{$prefix."_ENABLED"})."\n";
    if ($spec{$prefix."_ENABLED"} eq "true")
    {
      print FH Dada::headerFormat($prefix."_SOURCE", $spec{$prefix."_SOURCE"})."\n";
      Dada::mkdirRecursive($rdir."/".$spec{$prefix."_SOURCE"}, 0755);
      Dada::mkdirRecursive($adir."/".$spec{$prefix."_SOURCE"}, 0755);
    }
  }

  close FH;

  $cmd = "cp ".$rdir."/obs.info ".$adir."/";
  Dada::logMsg(2, $dl, "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "could not copy obs.info to archives dir");
  }

  # copy the molonglo_modules.txt file into the observation directory
  my $mo_file = $cfg{"CONFIG_DIR"}."/molonglo_modules.txt";
  if (-f $mo_file)
  {
    $cmd = "cp ".$mo_file." ".$rdir."/";
    Dada::logMsg(2, $dl, "createLocalDirs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "createLocalDirs: ".$result." ".$response);
    if ($result ne "ok")
    {
      return ("fail", "could not copy molonglo_modules.txt to results dir: ".$response);
    }
  }

  if ( ($spec{"INJECTED_FURBYS"} + 0) > 0 )		#Adding zero to interpret the string as a number
  {
    my $f_id;
    my @f_ids = split(",", $spec{"FURBY_IDS"} );
    my $furby_file;
    foreach $f_id (@f_ids)
    {
        $furby_file = FURBY_DATABASE."/furby_".$f_id;
        if ( -f $furby_file )
        {
          $cmd = "cp ".$furby_file." ".$adir."/Furbys/";
          Dada::logMsg(2, $dl, "createLocalDirs: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada:logMsg(3, $dl, "createLocalDirs: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, $dl, "createLocalDirs: Could not copy ".$furby_file);
          }
        }
        else
        {
          Dada::logMsg(0, $dl, "createLocalDirs: ".$furby_file." does not exist");
          return("fail", "Requested furby id (".$furby_file.") does not exist");
        }
    }
  }

  # always dump this now!
  #if ($spec{"MODE"} =~ m/CORR/)
  {
    my $tracking_flag = 1;
    if ($spec{"OBSERVING_TYPE"} ne "TRACKING")
    {
      $tracking_flag = 0;
    }
    Dada::logMsg(2, $dl, "createLocalDirs: dumpAntennaMapping($utc_start, $tracking_flag);");
    ($result, $response) = dumpAntennaMapping($utc_start, $tracking_flag);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "createLocalDirs: could not dump antenna mapping: ".$response);
    }
  }

  # Create the obs.processing files
  $cmd = "touch ".$rdir."/obs.processing";
  Dada::logMsg(2, $dl, "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "could not touch obs.processing in results dir");
  }

  return ("ok", "");
}


#
# Connect to the nexus and issue the stop command
#
sub stopNexus($)
{
  my ($utc) = @_;

  Dada::logMsg(2, $dl, "stopNexus(".$utc.")");

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

  my $utc_stop = "";

   # Ignore the "welcome" message
  $ignore = <$handle>;

  if ($utc eq "")
  {
    Dada::logMsg(1, $dl, "stopNexus: no UTC_STOP specified");

    # get the current unix time 
    my $curr_time_unix = time;

    $utc_stop = Dada::printTime(($curr_time_unix + 5), "utc");
    Dada::logMsg(1, $dl, "stopNexus: generated: ".$utc_stop." (now + 5)");
  }
  else
  {
    $utc_stop = $utc;
  }
  $cmd = "stop ".$utc_stop;

  Dada::logMsg(1, $dl, "stopNexus: nexus <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  Dada::logMsg(1, $dl, "stopNexus: nexus -> ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "stopNexus: ".$cmd." failed: ".$response);
    $response = $cmd." command failed on nexus";
  }

  $handle->close();

  if ($frb_detector_started)
  {
    # instruct the FRB detector the current observation is over
    stopFRBDetector ($utc_stop);
    $frb_detector_started = 0;
  }

  # insert the UTC_STOP into the obs.info fil

  my $unixtime_start = Dada::getUnixTimeUTC($utc_start);
  my $unixtime_stop = Dada::getUnixTimeUTC($utc_stop);
  my $integrated = $unixtime_stop - $unixtime_start;

  Dada::logMsg(1, $dl, "stopNexus: start=".$utc_start." stop=".$utc_stop." int=".$integrated);
  my $fname = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/obs.info";
  open FH, ">>$fname" or return ("fail","Could not append writeable file: ".$fname);
  print FH "UTC_STOP ".$utc_stop."\n";
  print FH "INT ".$integrated."\n";
  close FH;

  return ($result, $response);
}


#
# wait for stop in the background
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

    #Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
    #($result, $response) = stopRoachTX();
    #Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

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

        #Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
        #($result, $response) = stopRoachTX();
        #Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

        return;
      }

      # Ignore the "welcome" message
      $ignore = <$handle>;
    }
  }

  if ($handle) {
    $handle->close;
  }

  Dada::logMsg(2, $dl, "stopThread: exiting");

  #Dada::logMsg(2, $dl, "stopThread: stopRoachTX()");
  #($result, $response) = stopRoachTX();
  #Dada::logMsg(2, $dl, "stopThread: stopRoachTX ".$result." ".$response);

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

  my $pwc;
  my @pwcs = ();
  my $pwcs_string = "";

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    push @pwcs, $cfg{"PWC_".$i};
    $pwcs_string .= " ".$cfg{"PWC_".$i};
  }

  # Tell the PWCC to quit
  Dada::logMsg(1, $dl, "PWCC <- quit");
  quitPWCCommand();
  $pwcc_thread->join();

  $cmd = "stop_pwcs";
  Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
  foreach $pwc (@pwcs)
  {
    Dada::logMsg(2, $dl, $pwc." <- ".$cmd);
    ($result, $response) = Mopsr::clientCommand($cmd, $pwc);
    Dada::logMsg(2, $dl, $pwc." -> ".$result.":".$response);
  }

  # if error state is fatal
  if ($nexus_state < -3) 
  {
    $cmd = "destroy_dbs";
    Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
    foreach $pwc (@pwcs)
    {
      Dada::logMsg(2, $dl, $pwc." <- ".$cmd);
      ($result, $response) = Mopsr::clientCommand($cmd, $pwc);
      Dada::logMsg(2, $dl, $pwc." -> ".$result.":".$response);
    }

    $cmd = "init_dbs";
    Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
    foreach $pwc (@pwcs)
    {
      Dada::logMsg(2, $dl, $pwc." <- ".$cmd);
      ($result, $response) = Mopsr::clientCommand($cmd, $pwc);
      Dada::logMsg(2, $dl, $pwc." -> ".$result.":".$response);
    }
  }

  $cmd = "start_pwcs";
  Dada::logMsg(1, $dl, "PWCs <- ".$cmd);
  foreach $pwc (@pwcs)
  {
    Dada::logMsg(2, $dl, $pwc." <- ".$cmd);
    ($result, $response) = Mopsr::clientCommand($cmd, $pwc);
    Dada::logMsg(2, $dl, $pwc." -> ".$result.":".$response);
  }

  # Relaunch PWCC Thread with the current/previous config
  Dada::logMsg(1, $dl, "PWCC <- ".$cfg{"CONFIG_DIR"}."/mopsr_tmc.cfg");
  $pwcc_thread = threads->new(\&pwcc_thread, $cfg{"CONFIG_DIR"}."/mopsr_tmc.cfg");

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
sub sigHandle($)
{
  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  if ($quit_threads)
  {
    print STDERR basename($0)." : exiting\n";
    exit(1);
  }
  $quit_threads = 1;
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
# Generates the config file required for dada_pwc_command
#
sub genConfigFile($)
{
  my ($fname) = @_;

  my $string = "";

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Header file created by ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  $string = Dada::headerFormat("PWC_PORT",$cfg{"PWC_PORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);
 
  $string = Dada::headerFormat("PWC_LOGPORT",$cfg{"PWC_LOGPORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  $string = Dada::headerFormat("PWCC_PORT",$cfg{"PWCC_PORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  $string = Dada::headerFormat("PWCC_LOGPORT",$cfg{"PWCC_LOGPORT"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  $string = Dada::headerFormat("LOGFILE_DIR",$cfg{"SERVER_LOG_DIR"});
  print FH $string."\n";    
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);
 
  $string = Dada::headerFormat("HDR_SIZE", $site_cfg{"HDR_SIZE"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  $string = Dada::headerFormat("USE_BASEPORT", 1);
  print FH $string."\n";
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  my $i=0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) 
  {
    $string = Dada::headerFormat("PWC_".$i, $cfg{"PWC_".$i});
    print FH $string."\n";
    Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);
  }

  $string = Dada::headerFormat("NUM_PWC", $cfg{"NUM_PWC"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, "mopsr_tmc.cfg: ".$string);

  close FH;

  return ("ok", "");

}


sub genSpecFile($\%) 
{
  my ($fname, $xml) = @_;

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Specification File created by ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  my @specs = ();
  my ($prefix, $line);

  push @specs, Dada::headerFormat("HDR_VERSION", $site_cfg{"HDR_VERSION"});
  push @specs, Dada::headerFormat("HDR_SIZE",    $site_cfg{"HDR_SIZE"});
  push @specs, Dada::headerFormat("CONFIG",      $cfg{"CONFIG_NAME"});

  push @specs, Dada::headerFormat("TRACKING", $xml->{'east_arm_parameters'}{'tracking'});
  push @specs, Dada::headerFormat("MD_ANGLE", $xml->{'east_arm_parameters'}{'md_angle'});
  push @specs, Dada::headerFormat("NS_TILT", $xml->{'east_arm_parameters'}{'ns_tilt'});

  # signal parameters
  push @specs, Dada::headerFormat("BW", $xml->{'signal_parameters'}{'bandwidth'});
  push @specs, Dada::headerFormat("FREQ", $xml->{'signal_parameters'}{'centre_frequency'});
  push @specs, Dada::headerFormat("NANT", $xml->{'signal_parameters'}{'nant'});
  push @specs, Dada::headerFormat("NCHAN", $xml->{'signal_parameters'}{'nchan'});
  push @specs, Dada::headerFormat("NDIM", $xml->{'signal_parameters'}{'ndim'});
  push @specs, Dada::headerFormat("NPOL", $xml->{'signal_parameters'}{'npol'});
  push @specs, Dada::headerFormat("NBIT", $xml->{'signal_parameters'}{'nbit'});

  # pfb parameters
  push @specs, Dada::headerFormat("TSAMP", $xml->{'pfb_parameters'}{'sampling_time'});
  push @specs, Dada::headerFormat("OSAMP_RATIO", $xml->{'pfb_parameters'}{'oversampling_ratio'});
  push @specs, Dada::headerFormat("DSB", $xml->{'pfb_parameters'}{'dual_sideband'});
  push @specs, Dada::headerFormat("RESOLUTION", $xml->{'pfb_parameters'}{'resolution'});

  # observation parameters
  push @specs, Dada::headerFormat("OBSERVER", $xml->{'observation_parameters'}{'observer'});
  # TOBS is not always specified
  $tobs_secs = "-1";
  if (eval { exists $xml->{'observation_parameters'}{'tobs'} })
  {
    $tobs_secs =  $xml->{'observation_parameters'}{'tobs'};
  }
  push @specs, Dada::headerFormat("TOBS", $tobs_secs);

  # ensure name has no whitespace
  $xml->{'boresight_parameters'}{'name'} =~ s/\s/_/g;

  # source parameters
  push @specs, Dada::headerFormat("PID", $xml->{'boresight_parameters'}{'project_id'});
  push @specs, Dada::headerFormat("SOURCE", $xml->{'boresight_parameters'}{'name'});
  push @specs, Dada::headerFormat("RA", $xml->{'boresight_parameters'}{'ra'});
  push @specs, Dada::headerFormat("DEC", $xml->{'boresight_parameters'}{'dec'});
  push @specs, Dada::headerFormat("RFI_MITIGATION", $xml->{'boresight_parameters'}{'rfi_mitigation'});
  push @specs, Dada::headerFormat("ANTENNA_WEIGHTS", $xml->{'boresight_parameters'}{'antenna_weights'});
  push @specs, Dada::headerFormat("DELAY_TRACKING", $xml->{'boresight_parameters'}{'delay_tracking'});
  push @specs, Dada::headerFormat("AQ_PROC_FILE", $xml->{'boresight_parameters'}{'processing_file'});

  # correlation parameters
  $prefix = "CORR";
  if (eval { exists $xml->{'correlation_parameters'} })
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "true");
    push @specs, Dada::headerFormat($prefix."_PID", $xml->{'correlation_parameters'}{'project_id'});
    push @specs, Dada::headerFormat($prefix."_MODE", $xml->{'correlation_parameters'}{'mode'});
    push @specs, Dada::headerFormat($prefix."_DUMP_TIME", $xml->{'correlation_parameters'}{'dump_time'});
    push @specs, Dada::headerFormat($prefix."_PROC_FILE", $xml->{'correlation_parameters'}{'processing_file'});
    push @specs, Dada::headerFormat("MODE", $xml->{'correlation_parameters'}{'mode'});
  }
  else
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "false");
  }

  # tied array beam parameters
  my $i;
  for ($i=0; $i<$bf_cfg{"NUM_TIED_BEAMS"}; $i++)
  { 
    my $key = "tied_beam_".$i."_parameters";
    $prefix = "TB".$i;
    Dada::logMsg(1, $dl, "testing for ".$key);
    if (eval { exists $xml->{$key} })
    {
      $xml->{$key}{'name'} =~ s/\s/_/g;
      push @specs, Dada::headerFormat($prefix."_ENABLED", "true");
      push @specs, Dada::headerFormat($prefix."_PID", $xml->{$key}{'project_id'});
      push @specs, Dada::headerFormat($prefix."_MODE", $xml->{$key}{'mode'});
      push @specs, Dada::headerFormat($prefix."_PROC_FILE", $xml->{$key}{'processing_file'});
      push @specs, Dada::headerFormat($prefix."_SOURCE", $xml->{$key}{'name'});
      push @specs, Dada::headerFormat($prefix."_RA", $xml->{$key}{'ra'});
      push @specs, Dada::headerFormat($prefix."_DEC", $xml->{$key}{'dec'});
      if (eval { exists $xml->{$key}{'dm'} })
      {
        push @specs, Dada::headerFormat($prefix."_DM", $xml->{$key}{'dm'});
      }
      else
      {
        push @specs, Dada::headerFormat($prefix."_DM", -1);
      }
      push @specs, Dada::headerFormat("MODE", $xml->{$key}{'mode'});
    }
    else
    {
      push @specs, Dada::headerFormat($prefix."_ENABLED", "false");
    }
  }

  # fan beams parameters
  my $prefix = "FB";
  if (eval { exists $xml->{'fan_beams_parameters'} } )
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "true");
    push @specs, Dada::headerFormat($prefix."_PID", $xml->{'fan_beams_parameters'}{'project_id'});
    push @specs, Dada::headerFormat($prefix."_MODE", $xml->{'fan_beams_parameters'}{'mode'});
    push @specs, Dada::headerFormat($prefix."_BEAM_SPACING", $xml->{'fan_beams_parameters'}{'beam_spacing'});
    push @specs, Dada::headerFormat("MODE", $xml->{'fan_beams_parameters'}{'mode'});
  }
  else
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "false");
  }

#VG: Adding furby parameters
  # furby parameters
  my $prefix = "FURBY";
  if (eval { exists $xml->{'furbies'} } )
  {
    #ensure that none of the furby params have leading white spaces
    $xml->{'furbies'}{'furby_ids'} =~ s/^\s+//;
    $xml->{'furbies'}{'furby_beam'} =~ s/^\s+//;
    $xml->{'furbies'}{'furby_tstamps'} =~ s/^\s+//;
    
    #ensure that none of the furby params have trailing white spaces
    $xml->{'furbies'}{'furby_ids'} =~ s/\s+$//;
    $xml->{'furbies'}{'furby_beam'} =~ s/\s+$//;
    $xml->{'furbies'}{'furby_tstamps'} =~ s/\s+$//;
    
    #replacing any white spaces in the middle with comma ","
    $xml->{'furbies'}{'furby_ids'} =~ s/\s/,/g;
    $xml->{'furbies'}{'furby_beam'} =~ s/\s/,/g;
    $xml->{'furbies'}{'furby_tstamps'} =~ s/\s/,/g;
    push @specs, Dada::headerFormat("INJECTED_FURBYS", $xml->{'furbies'}{'num_furbies'});
    if ($xml->{'furbies'}{'num_furbies'} > 0)
    {
      push @specs, Dada::headerFormat($prefix."_IDS", $xml->{'furbies'}{'furby_ids'});
      push @specs, Dada::headerFormat($prefix."_BEAMS", $xml->{'furbies'}{'furby_beams'});
      push @specs, Dada::headerFormat($prefix."_TSTAMPS", $xml->{'furbies'}{'furby_tstamps'});
    }
  }
  else
  {
    #Nothing to do in else, even putting 0 is not necessary. Putting it anyway
    push @specs, Dada::headerFormat("INJECTED_FURBYS", "0");
  }

  # mod beams parameters
  my $prefix = "MB";
  if (eval { exists $xml->{'mod_beams_parameters'} } )
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "true");
    push @specs, Dada::headerFormat($prefix."_PID", $xml->{'mod_beams_parameters'}{'project_id'});
    push @specs, Dada::headerFormat("MODE", $xml->{'mod_beams_parameters'}{'mode'});
  }
  else
  {
    push @specs, Dada::headerFormat($prefix."_ENABLED", "false");
  }

  # hardwired / site config
  push @specs, Dada::headerFormat("OBS_OFFSET", "0");
  push @specs, Dada::headerFormat("TELESCOPE", $site_cfg{"TELESCOPE"});
  push @specs, Dada::headerFormat("RECEIVER", $site_cfg{"RECEIVER"});
  push @specs, Dada::headerFormat("INSTRUMENT", "MOPSR");
  push @specs, Dada::headerFormat("FILE_SIZE", $site_cfg{"FILE_SIZE"});

  # TODO remove the default for this an add it to the parent XML as mandatory
  my $ut1_offset = 0;
  if (eval { exists $xml->{'observation_parameters'}{'ut1_offset'} } ) 
  {
    $ut1_offset = $xml->{'observation_parameters'}{'ut1_offset'};
  }
  else
  {
    my $ut1_cfg_file = $cfg{"CONFIG_DIR"}."/ut1_offset.cfg";
    my %ut1_cfg =  Dada::readCFGFileIntoHash($ut1_cfg_file, 0);
    $ut1_offset = $ut1_cfg{"OFFSET"};
  }

  push @specs, Dada::headerFormat("UT1_OFFSET", $ut1_offset);

  # now add PWC specific command
  my $i = 0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    push @specs, Dada::headerFormat("Band".$i."_PFB_ID", $cfg{"PWC_PFB_ID_".$i});
  }

  # add the best module rankings to the spec
  my ($result, $best_mods) = getBestModules();
  if ($result ne "ok")
  {
    push @specs, Dada::headerFormat("RANKED_MODULES", "0,1,2,3,4,5,6,7");
  }
  else
  {
    push @specs, Dada::headerFormat("RANKED_MODULES", $best_mods);
  }

  # add custom parameters if they exist
  if (eval { exists $xml->{'custom_parameters'} } )
  {
    my $key;
    foreach $key (keys %{$xml->{'custom_parameters'}})
    {
      my $dada_key = uc($key);
      Dada::logMsg(1, $dl, "custom_parameter: ".$key."=".$xml->{'custom_parameters'}{$key}." Dada key=".$dada_key);
      my $spec;
      my $matched = 0;
      foreach $spec (@specs)
      {
        if ($spec =~ m/$dada_key/)
        {
          $matched = 1;
        }
      }
      if ($matched)
      {
        Dada::logMsg(1, $dl, "Ignoring custom param since it already existed");
      }
      else
      {
        push @specs, Dada::headerFormat ($dada_key, $xml->{'custom_parameters'}{$key});
      }
    }
  }

  # write the file
  foreach $line (@specs) 
  {
    Dada::logMsg(1, $dl, "MObS -> ".$line);
    Dada::logMsg(2, $dl, "mopsr_tmc.spec: ".$line);
    print FH $line."\n";
  }

  close FH;
  return ("ok","");
}

#
# some custom sorting routines
#
sub intsort
{
  if ((int $a) < (int $b))
  {
    return -1;
  }
  elsif ((int $a) > (int $b))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

sub modsort
{
  my $mod_a = $a;
  my $mod_b = $b;

  $mod_a =~ s/-B/-0/;
  $mod_a =~ s/-G/-1/;
  $mod_a =~ s/-Y/-2/;
  $mod_a =~ s/-R/-3/;
  $mod_b =~ s/-B/-0/;
  $mod_b =~ s/-G/-1/;
  $mod_b =~ s/-Y/-2/;
  $mod_b =~ s/-R/-3/;

  return $mod_a cmp $mod_b;
}

#
# Get the reference module and preferred modules for this configuration
#
sub getBestModules()
{
  my $rm_file = $cfg{"CONFIG_DIR"}."/mopsr_ranked_modules.txt";

  my ($result, $rm, $irm, $imod);
  my @rms = ();
  my @mods;
  my $pm_count = 0;
  my $pm_max = 8;
  if ($hires == 1)
  {
    $pm_max = 12;
  }
  my $best_modules = "";

  # read all the ranked modules in, 1 line to array element
  open(FHR,"<".$rm_file) or return ("fail", "could not open ranked modules file for reading");
  my @rms = <FHR>;
  close (FHR);

  # get the module ordering of the current modules
  ($result, @mods) = getOrderedModules();

  # find which ranked modules are current and record them
  for ($irm=0; $irm<=$#rms; $irm++)
  {
    $rm = $rms[$irm];
    chomp $rm;
    for ($imod=0; $imod<=$#mods; $imod++)
    {
      if (($rm eq $mods[$imod]) && ($pm_count < $pm_max))
      {
        if ($best_modules eq "")
        {
          $best_modules = $imod;
        }
        else
        {
          $best_modules .= ",".$imod;
        }
        $pm_count++;
      }
    }
  }
  return ("ok", $best_modules);
}


#
# return an ordered list of the modules for this configuration
#
sub getOrderedModules()
{
  my $sp_file = $cfg{"CONFIG_DIR"}."/mopsr_signal_paths.txt";
  my $mo_file = $cfg{"CONFIG_DIR"}."/molonglo_modules.txt";
  my $pm_file = $cfg{"CONFIG_DIR"}."/preferred_modules.txt";

  my %sp = Dada::readCFGFileIntoHash($sp_file, 1);
  my %mo = Dada::readCFGFileIntoHash($mo_file, 1);
  my %aq_cfg = Mopsr::getConfig("aq");

  my @sp_keys_sorted = sort modsort keys %sp;

  # now generate the listing of antennas the correct ordering
  my ($i, $send_id, $first_ant, $last_ant, $pfb_id, $imod, $rx);
  my ($pfb, $pfb_input, $bay_id);
  my %pfb_mods = ();
  my @mods = ();
  for ($i=0; $i<$aq_cfg{"NUM_PWC"}; $i++)
  {
    logMsg(2, $dl, "getOrderedModules: i=".$i);
    # if this PWC is an active or passive
    if ($aq_cfg{"PWC_STATE_".$i} ne "inactive")
    {
      $send_id = $aq_cfg{"PWC_SEND_ID_".$i};

      # this is the mapping in RAM for the input to the calibration code
      $first_ant = $cfg{"ANT_FIRST_SEND_".$send_id};
      $last_ant  = $cfg{"ANT_LAST_SEND_".$send_id};

      # now find the physics antennnas for this PFB
      $pfb_id  = $aq_cfg{"PWC_PFB_ID_".$i};

      if ( -f $pm_file )
      {
        my %pm = Dada::readCFGFileIntoHash($pm_file, 1);
        my @pfb_mods = split(/ +/, $pm{$pfb_id});
        $imod = $first_ant;
        foreach $rx ( @sp_keys_sorted )
        {
          ($pfb, $pfb_input) = split(/ /, $sp{$rx});
          logMsg(3, $dl, "getOrderedModules: pfb=".$pfb." pfb_input=".$pfb_input);
          if ($pfb eq $pfb_id) 
          {
            my $pfb_mod;
            foreach $pfb_mod (@pfb_mods)
            {
              if ($pfb_input eq $pfb_mod)
              {
                if (($imod >= $first_ant) && ($imod <= $last_ant))
                {  
                  $mods[$imod] = $rx;
                  $imod++;
                }
                else
                {
                  return ("fail", "failed to identify modules correctly");
                }
              }
            }
          }
        }
      }
      else
      {
        logMsg(3, $dl, "getOrderedModules: pfb_id=".$pfb_id." ants=".$first_ant." -> ".$last_ant);

        my @pfb_mods = split(/ +/, $aq_cfg{"PWC_ANTS"});
        %pfb_mods = ();
        my $j = 0;
        foreach $rx ( @sp_keys_sorted )
        {
          ($pfb, $pfb_input) = split(/ /, $sp{$rx});
          logMsg(3, $dl, "getOrderedModules: rx=".$rx." pfb=".$pfb." pfb_input=".$pfb_input);

          # If this RX board matches the PFB board we are currently processing
          if ($pfb eq $pfb_id)
          {
            # determine which index this RX board will have in the antenna ordering
            for ($j=0; $j<=$#pfb_mods; $j++)
            {
              if ($pfb_input eq $pfb_mods[$j])
              {
                $mods[($first_ant + $j)] = $rx;
                logMsg(3, $dl, "getOrderedModules: mods[".($first_ant + $j)."]=".$rx);
              }
            }
          }
        }
      }
    }
  }
  return ("ok", @mods);
}


#
# Dumps the antenna mapping for this observation
#
sub dumpAntennaMapping($$)
{
  my ($obs, $tracking) = @_;

  my $mo_file = $cfg{"CONFIG_DIR"}."/molonglo_modules.txt";
  my $ba_file = $cfg{"CONFIG_DIR"}."/molonglo_bays.txt";

  my $antenna_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.antenna";
  my $baselines_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.baselines";
  my $refant_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.refant";
  my $priant_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.priant";

  my %mo = Dada::readCFGFileIntoHash($mo_file, 1);
  my %ba = Dada::readCFGFileIntoHash($ba_file, 1);
  my %aq_cfg = Mopsr::getConfig("aq");

  my ($cmd, $result, $best_mods);
  my @mods;

  # get the ordered list of modules
  ($result, @mods) = getOrderedModules();
  if ($result ne "ok")
  {
    return ("fail", "could not get ordered list of modules");
  }
  my $i;

  # get the list of top ranked modules that exist in the configuration
  ($result, $best_mods) = getBestModules();
  if ($result ne "ok")
  {
    return ("fail", "could not get list of best modules");
  }
  Dada::logMsg(2, $dl, "dumpAntennaMapping: best_mods=".$best_mods);

  open(FHA,">".$antenna_file) or return ("fail", "could not open antenna file for writing");
  open(FHB,">".$baselines_file) or return ("fail", "could not open baselines file for writing");
  open(FHC,">".$refant_file) or return ("fail", "could not open reference antenna file for writing");
  open(FHD,">".$priant_file) or return ("fail", "could not open primary antenna file for writing");

  # ants should contain a listing of the antenna orderings
  my ($mod_id, $bay_id, $dist, $delay, $scale, $imod, $jmod);

  my @best_list = split(/,/, $best_mods);

  # write the best module name to file
  print FHC $mods[$best_list[0]]."\n";
  close FHC;

  # write the best module names to file
  foreach $imod (@best_list)
  {
    print FHD $mods[$imod]."\n";
  }
  close FHD;

  for ($imod=0; $imod<=$#mods; $imod++)
  {
    $mod_id = $mods[$imod];
    $bay_id = substr($mod_id,0,3);
    if ($tracking)
    {
      $dist = $ba{$bay_id};
    }
    else
    {
      ($dist, $delay, $scale) = split(/ /,$mo{$mod_id},3);
    }

    Dada::logMsg(2, $dl, "imod=".$imod." ".$mod_id.": dist=".$dist." delay=".$delay." scale=".$scale);
    print FHA $mod_id." ".$dist." ".$delay."\n";

    for ($jmod=$imod+1; $jmod<=$#mods; $jmod++)
    {
      Dada::logMsg(2, $dl, $mods[$imod]." ".$mods[$jmod]);
      print FHB $mods[$imod]." ".$mods[$jmod]."\n";
    }
  }

  close(FHA);
  close(FHB);

  my ($cmd, $result, $response);
  $cmd = "cp ".$antenna_file." ".$baselines_file." ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$obs."/";
  Dada::logMsg(2, $dl, "dumpAntennaMapping: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "dumpAntennaMapping: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "could not copy antenna files to archive dir");
  }

  return ("ok", "");
}


sub getPktUtcStart ()
{
  my $fname = $cfg{"CONFIG_DIR"}."/mopsr.pkt_utc_start";
  open FH, "<$fname" or return ("fail","Could not read file: ".$fname);
  my @lines = <FH>;
  close FH;
 
  if ($#lines != 0)
  {
    return ("fail", "expected 1, but found ".($#lines+1)." in ".$fname);
  }

  my $line = $lines[0];
  chomp $line;

  return ("ok", $line);
}

sub setPktUtcStart ($)
{
  (my $pkt_utc) = @_;

  # update global variable for UTC_START
  $pkt_utc_start = $pkt_utc;

  my $fname = $cfg{"CONFIG_DIR"}."/mopsr.pkt_utc_start";
  open FH, ">$fname" or return ("fail","Could not create writeable file: ".$fname);
  print FH $pkt_utc."\n";
  close FH;

  return ("ok", "");
}

sub checkEphemeris($)
{

}
