#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# 


use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Bpsr;            # DADA Module for configuration options
use strict;          # strict mode (like -Wall)
use File::Basename; 
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variable Declarations
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $pwc_id : shared;
our $beam : shared;
our %cfg : shared;
our %roach : shared;
our $log_host;
our $log_port;
our $log_sock;
our $proc_cmd_kill : shared;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$beam = "";
%cfg = Bpsr::getConfig();
%roach = Bpsr::getROACHConfig();
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;
$proc_cmd_kill = "";

#
# Local Variable Declarations
#
my $log_file = "";
my $pid_file = "";
my $control_thread = 0;
my $prev_header = "";
my $quit = 0;

#
# Check command line arguments is 1
#
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
  if ($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName())
  {
    # determine the relevant PWC based configuration for this script 
    $beam = $roach{"BEAM_".$pwc_id};
  }
  else
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


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd, $junk);

  $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  logMsg(1,"INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  # for receipt of UDP data
	my $recv_db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});

  # for Pscrunched data to be search ed by heimdall
	my $trans_db_key   = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"TRANSIENT_DATA_BLOCK"});

	my $curr_raw_header = "";
	my $prev_raw_header = "";
	my %h = ();

  # Main Loop
  while (!$quit_daemon) 
  {
		%h = ();

		# next header to read from the receiving data_block
    $cmd = "dada_header -k ".$recv_db_key;
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $curr_raw_header) = Dada::mySystem ($cmd);
    logMsg(3, "INFO", "main: ".$curr_raw_header);

    if ($result ne "ok")
		{
      if ($quit_daemon)
      {
			  logMsg(2, "INFO", "main: dada_header failed, but quit_daemon true");
      }
      else
      {
			  logMsg(1, "WARN", "dada_header failed, and quit_daemon != true");
        $quit_daemon = 1;
      }
		}
		elsif ($curr_raw_header eq $prev_raw_header)
		{
			logMsg(0, "ERROR", "main: header repeated, jettesioning observation");
      ($result, $response) = jettisonObservation($recv_db_key);
      if ($result ne "ok")
      {
        logMsg (0, "WARN", "main: failed to jettison observation");
        $quit_daemon = 1;
      }
		}
		else
		{
      %h = Dada::headerToHash ($curr_raw_header);

      my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"PROC_FILE"};
      logMsg (2, "INFO", "Full path to PROC_FILE: ".$proc_cmd_file);

      my %proc_cmd_hash = Dada::readCFGFile ($proc_cmd_file);
      my $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
      my $trans_db_key_run = "";
      my $lna_on = (int($h{"BEAM_LEVEL"}) > 262140) ? 0 : 1;

      ($proc_cmd_kill, $junk) = split(/ /, $proc_cmd, 2);

      if ($h{"PID"} eq "P999")
      {
        $lna_on = 1;
      }

      # we only use the transient key if proc_cmd is the decimator, this beam
      # is enabled and the LNA is active
      if (($proc_cmd =~ m/the_decimator/) && ($h{"BEAM_ACTIVE"} eq "on") && ($lna_on))
      {
        $trans_db_key_run = $trans_db_key;
      }

      # Dont ever run transient detection pipeline if the CAL is on
      if ($h{"MODE"} eq "CAL")
      {
        $trans_db_key_run = "";
      }

      # Dont ever tun the transient detection pipeline if the observation length is < 30s
      if (exists($h{"OBS_VAL"}) && (int($h{"OBS_VAL"}) < 30))
      {
        $trans_db_key_run = "";
      }
      
      # create the required directories
      my $obs_start_file = createLocalDir($h{"UTC_START"}, $curr_raw_header);

      # create a remote directory necessary for archives etc
      ($result, $response) = createRemoteDir ($h{"UTC_START"}, $obs_start_file);

      # process the primary component of the observation
      ($result, $response) = processObservation ($proc_cmd,
                                                 $recv_db_key, $trans_db_key_run, 
                                                 $obs_start_file, $curr_raw_header);

      logMsg(2, "INFO", "main: all threads launched, waiting for processing Thread to finish");
		}

		$prev_raw_header = $curr_raw_header;	

    if ($quit)
    {
      $quit_daemon = 1;
    }
  }


  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

  exit(0);
}

#
# Processes a single observation
#
# Creates the required directories for the output data on the server, and generates the obs.start file
#
sub processObservation($$$$$) 
{
  my ($proc_cmd, $in_key, $out_key, $obs_start_file, $raw_header) = @_;

	my %h = Dada::headerToHash ($raw_header);

  my $processing_dir = "";
  my $utc_start = "";
  my $acc_len = "";
  my $obs_offset = "";

  my @lines = ();
  my $line = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  my $header_ok = 1;

  if (length($h{"UTC_START"}) < 5) 
	{
    logMsg(0, "ERROR", "UTC_START was malformed or non existent");
    $header_ok = 0;
  }
  if (length($h{"OBS_OFFSET"}) < 1) 
	{
    logMsg(0, "ERROR", "Error: OBS_OFFSET was malformed or non existent");
    $header_ok = 0;
  }
  if (length($h{"PROC_FILE"}) < 1)
	{
    logMsg(0, "ERROR", "PROC_FILE was malformed or non existent");
    $header_ok = 0;
  }

  # command line that will be run
  if (exists($h{"BEAM_ACTIVE"}))
  {
    logMsg(2, "INFO", "h{BEAM_ACTIVE} ".$h{"BEAM_ACTIVE"});
  }
  else
  {
    logMsg(2, "INFO", "h{BEAM_ACTIVE} not set");
  }

	# if malformed
	if (! $header_ok) 
	{
		logMsg(0, "ERROR", "DADA header malformed, jettesioning xfer");
		$proc_cmd = "dada_dbnull -k ".$in_key." -s -z -q";
	} 
	elsif ($beam ne $h{"BEAM"})
	{
		logMsg(0, "ERROR", "Beam mismatch between header[".$h{"BEAM"}."] and config[".$beam."]");
		$proc_cmd = "dada_dbnull -k ".$in_key." -s -z -q";
	}
  else
  { 
    $processing_dir =  $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};

    # never do anything if this beam is disabled
    if ($h{"BEAM_ACTIVE"} eq "off")
    {
      logMsg(2, "INFO", "Ignoring observation as header[BEAM_ACTIVE] == off");
      $proc_cmd = "dada_dbnull -k ".$in_key." -s -z -q";
    }

    logMsg(2, "INFO", "Initial PROC_CMD: ".$proc_cmd);

    # replace <DADA_INFO> tags with the matching input .info file
    if ($proc_cmd =~ m/<DADA_INFO>/)
    {
      my $tmp_info_file =  "/tmp/bpsr_".$in_key.".info";
      # ensure a file exists with the write processing key
      if (! -f $tmp_info_file)
      {
        open FH, ">".$tmp_info_file;
        print FH "DADA INFO:\n";
        print FH "key ".$in_key."\n";
        close FH;
      }
      $proc_cmd =~ s/<DADA_INFO>/$tmp_info_file/;
    }

    # replace <DADA_KEY> tags with the matching input key
    $proc_cmd =~ s/<DADA_KEY>/$in_key/;

    # replace <DADA_RAW_DATA> tag with processing dir
    $proc_cmd =~ s/<DADA_DATA_PATH>/$processing_dir/;

    # replace DADA_UTC_START with actual UTC_START
    $proc_cmd =~ s/<DADA_UTC_START>/$h{"UTC_START"}/;

    # replace BPSR_NDECI_BIT with number of bits to be decimated too
    $proc_cmd =~ s/<BPSR_NDECI_BIT>/$h{"NDECI_BIT"}/;

    # replace BPSR_NDADA_UTC_START with actual UTC_START
    $proc_cmd =~ s/<BPSR_TSCRUNCH>/$h{"TSCRUNCH"}/;

    # replace BPSR_NDADA_UTC_START with actual UTC_START
    $proc_cmd =~ s/<BPSR_FSCRUNCH>/$h{"FSCRUNCH"}/;

    # special cases for the decimator
    if ($out_key ne "")
    {
      $proc_cmd .= " -k ".$out_key;
    }

    logMsg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
  }

  logMsg(1, "INFO", "START [proc] ".$proc_cmd);
  logMsg(2, "INFO", "Changing dir to $processing_dir");

  $cmd = "cd ".$processing_dir."; ".$proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." proc";

  logMsg(2, "INFO", "cmd = $cmd");

  my $return_value = system($cmd);
   
  if ($return_value != 0) {
    logMsg(0, "ERROR", $proc_cmd." failed: ".$?." ".$return_value);
  }

  logMsg(1, "INFO", "END   [proc] ".$proc_cmd);

  if (($processing_dir ne "") && (-d $processing_dir))
  {
    $cmd = "touch ".$processing_dir."/beam.finished";
    logMsg(2, "INFO", "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "processObservation: ".$result." ".$response);
  }

	return ("ok");
}

sub jettisonObservation($)
{
  my ($key) = @_;

  my ($cmd, $proc_cmd);

  $proc_cmd = "dada_dbnull -k ".$key." -s -z -q";

  $cmd = $proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." proc";

  logMsg(1, "INFO", "START [proc] ".$proc_cmd);
  my $return_value = system($cmd);

  if ($return_value != 0) 
  {
    logMsg(0, "ERROR", $proc_cmd." failed: ".$?." ".$return_value);
    return ("fail", "");
  }

  logMsg(1, "INFO", "END   [proc] ".$proc_cmd);
  return ("ok", "");
}


#
# Thread to create remote NFS links on the server
#
sub createRemoteDir($$) 
{
  my ($utc_start, $obs_start_file) = @_;

  logMsg(2, "INFO", "createRemoteDir(".$utc_start.", ".$obs_start_file.")");

  my $user = "dada";
  my $host = $cfg{"SERVER_HOST"};
  my $cmd = "mkdir -m 0755 -p ".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$beam;

  my $result = "";
  my $response = "";
  my $rval = 0;

  my $attempts_left = 5;

  my $use_nfs = 0;

  while ($attempts_left > 0)
  {
    if ($use_nfs)
    {
      logMsg(2, "INFO", "createRemoteDir: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "createRemoteDir: ".$result." ".$response);
    }
    else
    {
      logMsg(2, "INFO", "createRemoteDir: ".$user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      logMsg(2, "INFO", "createRemoteDir: ".$result." ".$rval." ".$response);
    }

    if (($result eq "ok") && ($rval == 0))
    {
      logMsg(2, "INFO", "createRemoteDir: remote directory created");

      # now copy obs.start file to remote directory
      if ($use_nfs)
      {
        $cmd = "cp ".$obs_start_file." ".$user."@".$host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$beam."/";
      }
      else
      {
        $cmd = "scp ".$obs_start_file." ".$user."@".$host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$beam."/";
      }
      logMsg(2, "INFO", "createRemoteDir: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "createRemoteDir: ".$result." ".$response);
      if ($result ne "ok") 
      {
        logMsg(0, "INFO", "createRemoteDir: ".$cmd." failed: ".$response);
        logMsg(0, "WARN", "could not copy obs.start file to server");
        return ("fail", "could not copy obs.start file");
      }
      else
      {
        return ("ok", "");
      }
      
    }
    else
    {
      if ($result ne "ok")
      {
        logMsg(0, "INFO", "createRemoteDir: ssh failed ".$user."@".$host.": ".$response);
        logMsg(0, "WARN", "could not ssh to server");
      }
      else
      {
        logMsg(0, "INFO", "createRemoteDir: ".$cmd." failed: ".$response);
        logMsg(0, "WARN", "could not create dir on server");
      }
      $attempts_left--;
      sleep(1);
    }
  }

  return ("fail", "could not create remote directory");
}


sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";
	my $recv_db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});

  my ($cmd, $result, $response, $process, $user);

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  # we have dada_headers listening on the receiving, event and dump data blocks - kill them, 
  # then allow a short time for those threads to exit

  $process = "^dada_header -k ".$recv_db_key;
  $user = "bpsr";

  logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
  ($result, $response) = Dada::killProcess($process, $user, $$);
  logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  sleep (1);

  $process = $proc_cmd_kill;
  logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
  ($result, $response) = Dada::killProcess($process, $user, $$);
  logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  logMsg(2, "INFO", "controlThread: checking if PID file [".$pid_file."] exists");
  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

}




#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) {

  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "proc", $msg);
    }
    print STDERR "[".$time."] ".$msg."\n";
  }
}



sub sigHandle($) {

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

sub sigChildHandle ()
{
  my $stiff;
  while (($stiff = waitpid(-1, &WNOHANG)) > 0) 
  {
    # do something with $stiff if you want
  }

  # install *after* calling waitpid
  $SIG{CHLD} = \&igPipeHandle;
}

#
# Create the local directories required for this observation
#
sub createLocalDir ($$) 
{
  (my $utc_start, my $raw_header) = @_;

  logMsg(2, "INFO", "createLocalDir(".$utc_start.")");

  my $base = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;
  my $dir = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  # create local archive directory with group sticky bit set
  $dir = $base."/".$utc_start;
  logMsg(2, "INFO", "createLocalDir: creating ".$dir);
  $cmd = "mkdir -m 2755 -p ".$dir;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    logMsg(0, "WARN", "Could not create ".$dir.": ".$response);
  }

  # create the aux subsubdir
  $dir = $base."/".$utc_start."/aux";
  logMsg(2, "INFO", "createLocalDir: creating ".$dir);
  $cmd = "mkdir -m 2755 -p ".$dir;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    logMsg(0, "WARN", "Could not create ".$dir.": ".$response);
  }

  # create an obs.start file in the processing dir:
  logMsg(2, "INFO", "createLocalDir: creating obs.start");
  my $file = $base."/".$utc_start."/obs.start";
  open(FH,">".$file.".tmp");
  print FH $raw_header;
  close FH;
  rename($file.".tmp", $file);

  logMsg(2, "INFO", "createLocalDir: returning ".$file);
  return $file;
}
