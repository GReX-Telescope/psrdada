#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# 


use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use Mopsr;          # DADA Module for configuration options
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
our $db_key : shared;
our %cfg : shared;
our $log_host;
our $log_port;
our $log_sock;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$beam = "";
$db_key = "";
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;

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
    $beam = "01";
    $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});
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
  my $cmd = "";
  my $result = "";
  my $response = "";

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

	my $recv_db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});

	my $curr_raw_header = "";
	my $prev_raw_header = "";
	my %header = ();

	my $processing_thread = 0;

  # Main Loop
  while (!$quit_daemon) 
  {
		%header = ();

		# next header to read from the receiving data_block
    $cmd =  "dada_header -k ".$recv_db_key;
    logMsg(2, "INFO", "main: ".$cmd);
    $curr_raw_header = `$cmd 2>&1`;
    logMsg(2, "INFO", "main: ".$cmd." returned");

		if ($? != 0)
		{
			logMsg(0, "INFO", "dada_header failed, but quit_daemon true");
		}
		elsif ($curr_raw_header eq $prev_raw_header)
		{
			logMsg(0, "ERROR", "main: header repeated, jettesioning observation");

      # start null thread to draing the datablock
      my $null_thread = threads->new(\&nullThread, $recv_db_key, "proc");
      $null_thread->join();
      $null_thread = 0;
		}
		else
		{
      %header = Dada::headerToHash($curr_raw_header);

      # determine processing command
      my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$header{"PROC_FILE"};
      logMsg(2, "INFO", "Full path to PROC_FILE: ".$proc_cmd_file);
      
      my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
      my $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
      my $obs_start_file = createLocalDirs($header{"UTC_START"}, $curr_raw_header);

      # process the incoming data
      logMsg(2, "INFO", "main: processObs(".$recv_db_key.", ".$obs_start_file." , [curr_raw_header])");
      $result = processObs($recv_db_key, $obs_start_file, $curr_raw_header);
		}

		$prev_raw_header = $curr_raw_header;	

    if ($quit) {
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
sub processObs($$$)
{
  (my $db_key, my $obs_start_file, my $raw_header) = @_;

	my %h = Dada::headerToHash($raw_header);
  my $processing_dir = "";
  my $utc_start = "";
  my $obs_offset = "";
  my $proc_cmd = "";
  my $proc_cmd_file = "";
  my $remote_dirs_thread = 0;

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
	my $proc_cmd = "";

	# if malformed
	if (! $header_ok) 
	{
		logMsg(0, "ERROR", "DADA header malformed, jettesioning xfer");
		$proc_cmd = "dada_dbnull -s -k ".$db_key;
	} 
	elsif ($beam ne $h{"BEAM"})
	{
		logMsg(0, "ERROR", "Beam mismatch between header[".$h{"BEAM"}."] and config[".$beam."]");
		$proc_cmd = "dada_dbnull -s -k ".$db_key;
	}
  else
  { 
    # launch thread to create output directories on the server, 
    # as this can take a long time due to load
    $remote_dirs_thread = threads->new(\&remoteDirsThread, $h{"UTC_START"}, $obs_start_file);

    $processing_dir =  $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};

    # Add the dada header file to the proc_cmd
    my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"PROC_FILE"};

    logMsg(2, "INFO", "Full path to PROC_FILE: ".$proc_cmd_file);

    my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
    $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

    logMsg(2, "INFO", "Initial PROC_CMD: ".$proc_cmd);

    # replace <DADA_INFO> tags with the matching input .info file
    if ($proc_cmd =~ m/<DADA_INFO>/)
    {
      my $tmp_info_file =  "/tmp/mopsr_".$db_key.".info";
      # ensure a file exists with the write processing key
      if (! -f $tmp_info_file)
      {
        open FH, ">".$tmp_info_file;
        print FH "DADA INFO:\n";
        print FH "key ".$db_key."\n";
        close FH;
      }
      $proc_cmd =~ s/<DADA_INFO>/$tmp_info_file/;
    }

    # replace <DADA_KEY> tags with the matching input key
    $proc_cmd =~ s/<DADA_KEY>/$db_key/;

    # replace <DADA_RAW_DATA> tag with processing dir
    $proc_cmd =~ s/<DADA_DATA_PATH>/$processing_dir/;

    # replace DADA_UTC_START with actual UTC_START
    $proc_cmd =~ s/<DADA_UTC_START>/$h{"UTC_START"}/;

    # replace DADA_GPU_ID with actual GPU_ID 
    $proc_cmd =~ s/<DADA_GPU_ID>/$cfg{"PWC_GPU_ID_".$pwc_id}/;

    logMsg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
  }

  logMsg(1, "INFO", "START ".$proc_cmd);
  logMsg(2, "INFO", "Changing dir to $processing_dir");

  $cmd = "cd ".$processing_dir."; ".$proc_cmd." ";

  logMsg(2, "INFO", "cmd = $cmd");

  my $return_value = system($cmd);
   
  if ($return_value != 0) {
    logMsg(0, "ERROR", $proc_cmd." failed: ".$?." ".$return_value);
  }

  logMsg(1, "INFO", "END   ".$proc_cmd);

  if (($processing_dir ne "") && (-d $processing_dir))
  {
    $cmd = "touch ".$processing_dir."/beam.finished";
    logMsg(2, "INFO", "processingThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "processingThread: ".$result." ".$response);
  }

	if ($remote_dirs_thread)
	{
		logMsg(2, "INFO", "processingThread: joining remote_dirs_thread");
		$remote_dirs_thread->join();
	}

	return ("ok");
}

#
# runs a thread to execute dada_dbnull on the specified data block key
#
sub nullThread($$)
{
  (my $db_key, my $tag) = @_;

  my $cmd = "dada_dbnull -s -k ".$db_key;
  my $result = "";
  my $response = "";

  my $full_cmd = $cmd;

  logMsg(1, "INFO", "START ".$cmd);

  logMsg(2, "INFO", "nullThread: ".$full_cmd);
  ($result, $response) = Dada::mySystem($full_cmd);
  logMsg(2, "INFO", "nullThread: ".$result." ".$response);

  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}
  
#
# Thread to create remote NFS links on the server
#
sub remoteDirsThread($$)
{
  my ($utc_start, $obs_start_file) = @_;

  logMsg(2, "INFO", "remoteDirsThread(".$utc_start.", ".$obs_start_file.")");

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
      logMsg(2, "INFO", "remoteDirsThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "remoteDirsThread: ".$result." ".$response);
    }
    else
    {
      logMsg(2, "INFO", "remoteDirsThread: ".$user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      logMsg(2, "INFO", "remoteDirsThread: ".$result." ".$rval." ".$response);
    }

    if (($result eq "ok") && ($rval == 0))
    {
      logMsg(2, "INFO", "remoteDirsThread: remote directory created");

      # now copy obs.start file to remote directory
      if ($use_nfs)
      {
        $cmd = "cp ".$obs_start_file." ".$user."@".$host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$beam."/";
      }
      else
      {
        $cmd = "scp ".$obs_start_file." ".$user."@".$host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$beam."/";
      }
      logMsg(2, "INFO", "remoteDirsThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "remoteDirsThread: ".$result." ".$response);
      if ($result ne "ok") 
      {
        logMsg(0, "INFO", "remoteDirsThread: ".$cmd." failed: ".$response);
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

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  # Kill the dada_header command
  $cmd = "ps aux | grep -v grep | grep mopsr | grep 'dada_header -k ".$db_key."' | awk '{print \$2}'";

  logMsg(2, "INFO", "controlThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  $response =~ s/\n/ /;
  logMsg(2, "INFO", "controlThread: ".$result." ".$response);

  if (($result eq "ok") && ($response ne "")) {
    $cmd = "kill -KILL ".$response;
    logMsg(1, "INFO", "controlThread: Killing dada_header -k ".$db_key.": ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(2, "INFO", "controlThread: ".$result." ".$response);
  }

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
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "obs mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
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

#
# Create the local directories required for this observation
#
sub createLocalDirs($$)
{
  (my $utc_start, my $raw_header) = @_;

  logMsg(2, "INFO", "createLocalDirs(".$utc_start.")");

  my $base = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;
  my $dir = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  # create local archive directory with group sticky bit set
  $dir = $base."/".$utc_start;
  logMsg(2, "INFO", "createLocalDirs: creating ".$dir);
  $cmd = "mkdir -m 2755 -p ".$dir;
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    logMsg(0, "WARN", "Could not create ".$dir.": ".$response);
  }

  # create an obs.start file in the processing dir:
  logMsg(2, "INFO", "createLocalDirs: creating obs.start");
  my $file = $base."/".$utc_start."/obs.start";
  open(FH,">".$file.".tmp");
  print FH $raw_header;
  close FH;
  rename($file.".tmp", $file);

  logMsg(2, "INFO", "createLocalDirs: returning ".$file);
  return $file;
}
