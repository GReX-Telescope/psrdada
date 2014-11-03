#!/usr/bin/env perl

# 
# Simple MOPSR processing script
#
#   Runs the antenna splitter on dada datablock
#   Runs the PROC_FILE on each of the output data blocks
# 
# Author:   Andrew Jameson
# 

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use strict;          # strict mode (like -Wall)
use warnings;

use forks;
use forks::shared;

use Mopsr;          # DADA Module for configuration options
use File::Basename; 
#use threads;         # standard perl threads
#use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;
use LockFile::Simple;

#
# Function Prototypes
#
sub logMsg($$$);

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
our $db_key : shared;
our %cfg : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;
our $split_dbs;
our $override : shared;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$db_key = "";
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
$split_dbs = 1;
$override = 0;

#
# Local Variable Declarations
#
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
  if (($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName()) || ($cfg{"PWC_".$pwc_id} eq "localhost"))
  {
    # determine the relevant PWC based configuration for this script 
    $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
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

# Also check that we are an ACTIVE or PASSIVE PWC
if (($cfg{"PWC_STATE_".$pwc_id} ne "active") && ($cfg{"PWC_STATE_".$pwc_id} ne "passive"))
{
  print STDOUT "Config file specified PWC_STATE_".$pwc_id."=".$cfg{"PWC_STATE_".$pwc_id}.", not starting\n";
  exit(0);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
# Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);


#
# Main
#
{
  my $cmd = "";
  my $result = "";
  my $response = "";

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  # Dada::daemonize($sys_log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  logMsg(1,"INFO", "STARTING SCRIPT");

  # special case for EG01
  if ($cfg{"PWC_PFB_ID_".$pwc_id} eq "EG02") 
  {
    $override = 1;
  }
  $override = 0;

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  logMsg(2, "INFO", "main: receiving datablock key global=".$db_key);

	my $recv_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
	#my $proc_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"AQDSP_DATA_BLOCK"});
	#my $dump_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"DUMP_DATA_BLOCK"});

  # start a thread to processed delayed events
  #my $events_thread  = threads->new(\&eventsThread, $recv_db_key, $dump_db_key);
  my $events_thread  = 0;

  # start up a thread to dump any valid events to the local disk
  #my $dump_thread    = threads->new(\&dumperThread, $dump_db_key);
  my $dump_thread    = 0;

  my %proc_threads = ();
  my @proc_db_keys = ();
  my @proc_db_ids = split(/ /,$cfg{"PROCESSING_DATA_BLOCK"});
  my $proc_db_id;
  
	my $curr_raw_header = "";
	my $prev_raw_header = "";
	my %header = ();
	my $splitter_thread = 0;
	my $transpose_thread = 0;
  my $nant = 0;

  $cmd = "mkdir -p /tmp/tempo2/mpsr/";
  logMsg(1, "INFO", "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "main: ".$cmd." returned");

  # Main Loop
  while (!$quit_daemon) 
  {
		%header = ();

    $cmd = "rm -f /tmp/tempo2/mpsr/.lock /tmp/tempo2/mpsr/mpsr.lock";
    logMsg(1, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(2, "INFO", "main: ".$cmd." returned");

		# next header to read from the receiving data_block
    $cmd =  "dada_header -k ".$recv_db_key;
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $curr_raw_header) = Dada::mySystem($cmd);
    logMsg(2, "INFO", "main: ".$cmd." returned");

    $cmd = "ls -1d ".$cfg{"CONFIG_DIR"};
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "main: ".$cmd." config_dir=".$response);
    if ($response ne $cfg{"CONFIG_DIR"})
    {
      logMsg(0, "ERROR", "NFS automount for ".$cfg{"CONFIG_DIR"}." failed: ".$response);
      $quit_daemon = 1;
    }

		if ($? != 0)
		{
      if ($quit_daemon)
      {
			  logMsg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
			  logMsg(0, "ERROR", "dada_header failed: ".$curr_raw_header);
        $quit_daemon = 1;
      }
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

      # setup the results directory for this observation
      my $obs_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$cfg{"PWC_PFB_ID_".$pwc_id}."/".$header{"UTC_START"};
      logMsg(2, "INFO", "main: mkdirRecursive(".$obs_dir." 0755)");
      ($result, $response) = Dada::mkdirRecursive($obs_dir, 0755);
      logMsg(3, "INFO", "main: mkdirRecursive ".$result." ".$response);
      if ($result ne "ok")
      {
        logMsg(0, "ERROR", "could not create results dir [".$obs_dir."] ".$response);
        logMsg(0, "ERROR", "jettesoning observation");
        # start null thread to draing the datablock
        my $null_thread = threads->new(\&nullThread, $recv_db_key, "proc");
        $null_thread->join();
        $null_thread = 0;
      }

      @proc_db_keys = ();
      if (($header{"PROC_FILE"} eq "mopsr.dbib") || 
          ($header{"PROC_FILE"} eq "mopsr.dbdisk"))
      {
        $split_dbs = 0;
        push @proc_db_keys, $recv_db_key;
      }
      else
      {
        if ($override)
        {
          $header{"PROC_FILE"} = "mopsr.dbib";
          $split_dbs = 0;
          push @proc_db_keys, $recv_db_key;
        }
        else
        {
          foreach $proc_db_id ( @proc_db_ids )
          {
            push @proc_db_keys, Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $proc_db_id);
          }
        }
      }

      # $split_dbs  = $header{"SEPARATE_ANTENNA"};

      logMsg(2, "INFO", "main: mkdir ".$obs_dir);
      mkdir $obs_dir, 0755;

      my $tempo2_master = 0;

      if (!$override)
      {
        # try to lock the file
        my $lock_handle = trylock ("/tmp/tempo2/mpsr/mpsr");

        # if we managed to lock the directory, then we delete the files
        if ($lock_handle)
        {
          $cmd = "rm -f /tmp/tempo2/mpsr/.lock /tmp/tempo2/mpsr/pulsar.par /tmp/tempo2/mpsr/t2pred.dat";
          $tempo2_master = 1;
          logMsg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          logMsg(2, "INFO", "main: ".$result." ".$response);

          sleep(1);

          logMsg(2, "INFO", "main: lock_handle->release()");
          $lock_handle->release();
          logMsg(2, "INFO", "main: lock_handle->release returned");
        }
        # otherwise we wait for these files to be deleted
        else
        {
          my $waiting = 5;
          while ($waiting > 0)
          {
            if ((-f "/tmp/tempo2/mpsr/pulsar.par") || (-f "/tmp/tempo2/mpsr/t2pred.dat"))
            {
              $waiting--;
              sleep(1);
            }
            else
            {
              $waiting = 0;
            }
          }
        }
      }

      # number of antenna encoded in input datablock
      $nant = $cfg{"NANT"};

      # if we are processing antenna independently
      if ($split_dbs)
      {
        if ($nant != ($#proc_db_keys + 1))
        {
          logMsg(0, "ERROR", "main: NANT [".$nant."]  != number of processing datablock keys [".($#proc_db_keys+1)."]");
        }

        # also run a aqdsp thread to do the transpose
        #$transpose_thread = threads->new(\&transposeThread, $recv_db_key, $proc_db_key);

        # now run the mopsr_dbsplitter to separate the antenna into the processing data blocks
        $splitter_thread = threads->new(\&splitterThread, $recv_db_key, \@proc_db_keys);

      }

      my $proc_db_key = "";
      # for each Antenna, spawn a processing thread
      my $first = 1;
      foreach $proc_db_key ( @proc_db_keys)
      {
        if ($first)
        {
          $proc_threads{$proc_db_key} = threads->new(\&processAntennaThread, $proc_db_key, $tempo2_master);
        }
        else
        {
          $proc_threads{$proc_db_key} = threads->new(\&processAntennaThread, $proc_db_key, 0);
        }
        $first = 0;
      }
      
      if ($split_dbs)
      {
        # now join the threads we launched
        logMsg(2, "INFO", "main: joining splitterThread");
        $result = $splitter_thread->join();
        logMsg(2, "INFO", "main: splitterThread: ".$result);
      }

      foreach $proc_db_key ( @proc_db_keys)
      {
        logMsg(2, "INFO", "main: joining processAntennaThread[".$proc_db_key."]");
        $result = $proc_threads{$proc_db_key}->join();
        logMsg(2, "INFO", "main: processAntennaThread[".$proc_db_key."]: ".$result);
      }
		}

		$prev_raw_header = $curr_raw_header;	

    if ($quit) {
      $quit_daemon = 1;
    }
  }

  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  if ($events_thread)
  {
    logMsg(1, "INFO", "main: joining eventsThread");
    $events_thread->join();
  }

  if ($dump_thread)
  {
    logMsg(2, "INFO", "main: joining dumperThread");
    $dump_thread->join();
  }

  # hard clean the lockfile
  $cmd = "rm -f /tmp/tempo2/mpsr/mpsr.lock /tmp/tempo2/mpsr/.lock";
  ($result, $response) = Dada::mySystem ($cmd);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "main [".$cmd."] failed:".$response);
  }

  logMsg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}

sub splitterThread($\@)
{
  (my $in_key, my $out_keys_ref) = @_;

  my @out_keys = @$out_keys_ref;
  my $out_key;

  my ($cmd, $result, $response); 

  $cmd = "mopsr_dbsplitdb -z -s ".$in_key;
  foreach $out_key (@out_keys)
  {
    $cmd .= " ".$out_key;
  }

  logMsg(1, "INFO", "START ".$cmd);
  logMsg(2, "INFO", "splitterThread: ".$cmd);
  ($result, $response ) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", $pwc_id, $daemon_name, "split");
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "splitter thread failed :".$response);
  }
  logMsg(2, "INFO", "splitterThread: ".$result." ".$response);
  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}

sub transposeThread($$)
{
  (my $in_key, my $out_key) = @_;

  my ($cmd, $result, $response, $gpu_id);

  $gpu_id = $cfg{"PWC_GPU_ID_".$pwc_id};

  $cmd = "mopsr_aqdsp -s ".$in_key." ".$out_key." -r -o -d ".$gpu_id." ".
          $cfg{"MOLONGLO_MODULES_FILE"}." ".
          $cfg{"MOLONGLO_SIGNAL_PATHS_FILE"};


  logMsg(1, "INFO", "START ".$cmd);
  logMsg(2, "INFO", "transposeThread: ".$cmd);
  ($result, $response ) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", $pwc_id, $daemon_name, "trans");
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "transpose thread failed :".$response);
  }
  logMsg(2, "INFO", "transposeThread: ".$result." ".$response);
  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}

#
# Processes a single antenna from a single observation
#
# Creates the required directories for the output data on the server, and generates the obs.header file
#
sub processAntennaThread($$)
{
  my ($db_key, $gen) = @_;

  my ($cmd, $result, $response, $ant_raw_header);

  $cmd =  "dada_header -k ".$db_key;
  logMsg(1, "INFO", "processAntennaThread: ".$cmd." gen=".$gen);
  ($result, $ant_raw_header) = Dada::mySystem($cmd);
  logMsg(1, "INFO", "processAntennaThread[".$db_key."] dada_header returned");
  my %h = Dada::headerToHash($ant_raw_header);

  if (!$split_dbs)
  {
    $h{"ANT_ID"} = "00";
  }

  if ($override)
  {
    $h{"PROC_FILE"} = "mopsr.dbib";
  }

  # create the local directory for this observation / antenna
  if ((!exists($h{"ANT_ID"})) || (createLocalDirs(\%h) < 0))
  {
    logMsg(0, "ERROR", "processAntennaThread: could not create local dir for key ".$db_key);
    $result = nullThread ($db_key, "proc");
    return ($result, "jettesioned observation");
  }

  logMsg(1, "INFO", "processAntennaThread: local dirs created");

  my $obs_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$cfg{"PWC_PFB_ID_".$pwc_id}."/".$h{"UTC_START"};
  my $obs_header = $obs_dir."/".$h{"ANT_ID"}."/obs.header";

  my $processing_dir = "";
  my $utc_start = "";
  my $obs_offset = "";
  my $proc_cmd = "";
  my $proc_cmd_file = "";
  my $remote_dirs_thread = 0;

  my @lines = ();
  my $line = "";

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

	# if malformed
	if (!$header_ok) 
	{
		logMsg(0, "ERROR", "DADA header malformed, jettesioning xfer");
		$proc_cmd = "dada_dbnull -s -k ".$db_key;
	}
  elsif ((!$gen) && (int($h{"ANT_ID"}) > 3))
  {
    logMsg(0, "INFO", "Deliberately disable ant: ".$h{"ANT_ID"});
    $proc_cmd = "dada_dbnull -s -k ".$db_key." -z";
  }
  else
  {
    # launch thread to create output directories on the server, 
    # as this can take a long time due to load
    $remote_dirs_thread = threads->new(\&remoteDirsThread, $h{"UTC_START"}, $cfg{"PWC_PFB_ID_".$pwc_id}, $h{"ANT_ID"}, $obs_header);

    $processing_dir = $obs_dir."/".$h{"ANT_ID"};

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

    # replace DADA_ANT_ID with actual ANT_ID
    $proc_cmd =~ s/<DADA_ANT_ID>/$h{"ANT_ID"}/;

    # replace DADA_ANT_ID with actual ANT_ID
    $proc_cmd =~ s/<DADA_PFB_ID>/$h{"PFB_ID"}/;

    # replace DADA_ANT_ID with actual ANT_ID
    my $mpsr_ib_port = 40000 + int($pwc_id);
    $proc_cmd =~ s/<MPSR_IB_PWC_PORT>/$mpsr_ib_port/;

    # replace DADA_GPU_ID with actual GPU_ID 
    $proc_cmd =~ s/<DADA_GPU_ID>/$cfg{"PWC_GPU_ID_".$pwc_id}/;

    logMsg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
  }

  if ($proc_cmd =~ m/dspsr/) 
  {
    if (!$gen)
    {
      my $exists = 0;
      # wait for the predictor
      while (!$exists)
      {
        if ((-f "/tmp/tempo2/mpsr/pulsar.par") && (-f "/tmp/tempo2/mpsr/t2pred.dat"))
        {
          $exists = 1;
          logMsg(2, "INFO", "/tmp/tempo2/mpsr/pulsar.par && /tmp/tempo2/mpsr/t2pred.dat both exist now");
        }
        else
        {
          logMsg(2, "INFO", "waiting for /tmp/tempo2/mpsr/pulsar.par && /tmp/tempo2/mpsr/t2pred.dat");
        }
        sleep(1); 
      }
      $proc_cmd .= " -E /tmp/tempo2/mpsr/pulsar.par -P /tmp/tempo2/mpsr/t2pred.dat";
      sleep(1);
    }
  }

  logMsg(1, "INFO", "START ".$proc_cmd);
  logMsg(2, "INFO", "Changing dir to $processing_dir");

  chdir $processing_dir;

  ($result, $response) = Dada::mySystemPiped ($proc_cmd, $src_log_file, $src_log_sock, "src", $pwc_id, $daemon_name, "proc");
 
  if ($result ne "ok") {
    logMsg(0, "ERROR", $proc_cmd." failed: ".$response);
  }

  logMsg(1, "INFO", "END   ".$proc_cmd);

  if (($processing_dir ne "") && (-d $processing_dir))
  {
    $cmd = "touch ".$processing_dir."/ant.finished";
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

  #my $full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$pwc_id." null";

  logMsg(1, "INFO", "START ".$cmd);
  ($result, $response ) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", $pwc_id, $daemon_name, "null");
  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}


#
# Persistent thread that continuously runs the dada_dbevent asynchoronously a datablock
#
sub eventsThread($$)
{
  my ($in_key, $out_key) = @_;

  my ($cmd, $full_cmd, $result, $response, $obs_header);

  # port on which to listen for event dumping requests
  my $event_port = (int($cfg{"CLIENT_EVENT_BASEPORT"}) + int($pwc_id));

  while (!$quit_daemon)
  {
    $cmd = "dada_header -k ".$in_key;
    logMsg(1, "INFO", "eventsThread: ".$cmd);
    ($result, $obs_header) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "eventsThread: ".$cmd." returned");
    if ($result ne "ok")
    {
      if ($quit_daemon)
      {
        logMsg(2, "INFO", "eventsThread: dada_header -k ".$in_key." failed, and quit_daemon true");
        return ("ok");
      }
      logMsg(1, "WARN", "eventsThread: dada_header -k ".$in_key." failed, but quit_daemon not true");
      sleep (1);
    }
    else
    {
      logMsg(3, "INFO", "eventsThread: HEADER=[".$obs_header."]");
      $cmd = "dada_dbevent ".$in_key." ".$out_key." -p ".$event_port." -b 90 -t 96";
      $cmd = "dada_dbnull -k ".$in_key." -s";

      my %h = Dada::headerToHash($obs_header);
      logMsg(2, "INFO", "      [evnt] ".$h{"UTC_START"});
      logMsg(1, "INFO", "START [evnt] ".$cmd);
      ($result, $response ) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", $pwc_id, $daemon_name, "event");
      logMsg(1, "INFO", "END   [evnt] ".$cmd);
      if ($result ne "ok")
      {
        logMsg(1, "WARN", "eventsThread: ".$cmd." failed ".$response);
      }
    }
  }
}


#
# Thread to listen on a specific data block and just write any data
# directly to disk. For use with dada_dbevent
#  
sub dumperThread($)
{
  my ($key) = @_;

  my ($cmd, $full_cmd, $result, $response, $dump_header);

  my $pfb = $cfg{"PWC_PFB_ID_".$pwc_id};
  my $can_dump = 1;
  my $dump_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$pfb;

  if (! -d $dump_dir)
  {
    $cmd = "mkdir -m 0755 ".$dump_dir;
    logMsg(2, "INFO", "dumperThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "dumperThread: ".$result." ".$response);
    if ($result ne "ok")
    {
      $can_dump = 0;
    }
  }

  $cmd = "touch ".$dump_dir."/test.touch";
  logMsg(2, "INFO", "dumperThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "dumperThread: ".$result." ".$response);
  if ($result ne "ok")
  {
    $can_dump = 0;
  }
  else
  {
    logMsg(2, "INFO", "dumperThread: unlink ".$dump_dir."/test.touch");
    unlink $dump_dir."/test.touch";
  }

  while (!$quit_daemon)
  {
    $cmd = "dada_header -k ".$key;
    logMsg(2, "INFO", "dumperThread: ".$cmd);
    ($result, $dump_header) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "dumperThread: ".$cmd." returned");
    if (($result ne "ok") || ($dump_header eq ""))
    {
      if ($quit_daemon)
      {
        logMsg(2, "INFO", "dumperThread: dada_header -k ".$key." failed, and quit_daemon true");
        return ("ok");
      }
      logMsg(1, "WARN", "dumperThread: dada_header -k ".$key." failed, but quit_daemon not true");
      sleep (1);
    }
    else
    {
      logMsg(3, "INFO", "dumperThread: HEADER=[".$dump_header."]");
      if ($can_dump)
      {
        $cmd = "dada_dbdisk -k ".$key." -D ".$dump_dir." -s";
      }
      else
      {
        $cmd = "dada_dbnull -q -z -k ".$key." -s";
      }
      #$full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$pwc_id." dump";
      logMsg(1, "INFO", "START [dump] ".$cmd);
      #($result, $response) = Dada::mySystem($full_cmd);
      logMsg(1, "INFO", "END   [dump] ".$result." ".$response);
    }
  }
  return ("ok");
}


#
# Thread to create remote NFS links on the server
#
sub remoteDirsThread($$$$)
{
  my ($utc_start, $pfb_id, $ant_id, $obs_header) = @_;

  logMsg(2, "INFO", "remoteDirsThread(".$utc_start.", ".$pfb_id.", ".$ant_id.", ".$obs_header.")");

  my $user = $cfg{"USER"};
  my $host = $cfg{"SERVER_HOST"};
  my $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$pfb_id."_".$ant_id;
  my $cmd = "mkdir -m 2755 -p ".$remote_dir;

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

      # now copy obs.header file to remote directory
      if ($use_nfs)
      {
        $cmd = "cp ".$obs_header." ".$remote_dir."/";
      }
      else
      {
        $cmd = "scp ".$obs_header." ".$user."@".$host.":".$remote_dir."/";
      }
      logMsg(2, "INFO", "remoteDirsThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "remoteDirsThread: ".$result." ".$response);
      if ($result ne "ok") 
      {
        logMsg(0, "INFO", "remoteDirsThread: ".$cmd." failed: ".$response);
        logMsg(0, "WARN", "could not copy obs.header file to server");
        return ("fail", "could not copy obs.header file");
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

  my $user = "mpsr";
  my $pwc_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECV_DATA_BLOCK"});
  my $process = "^dada_header -k ".$pwc_key;

  logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
  ($result, $response) = Dada::killProcess($process, $user);
  logMsg(3, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  my $dump_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"DUMP_DATA_BLOCK"});
  $process = "^dada_header -k ".$dump_key;
  logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
  ($result, $response) = Dada::killProcess($process, $user);
  logMsg(3, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  my @processes_to_kill = ();
  push @processes_to_kill, "^dada_dbevent ".$pwc_key." ".$dump_key;
  push @processes_to_kill, "^dada_dbnull";

  foreach $process ( @processes_to_kill)
  {
    logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    logMsg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  logMsg(1, "INFO", "controlThread: exiting");
}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) 
{
  my ($level, $type, $msg) = @_;
  if ($level <= $dl) 
  {
    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock))
    {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock)
    {
      Dada::nexusLogMessage($sys_log_sock, $pwc_id, $time, "sys", $type, "obs mngr", $msg);
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
    if ($sys_log_sock) {
      close($sys_log_sock);
    }
    if ($src_log_sock) {
      close($src_log_sock);
    }
  }
}

#sub sigPipeHandle($) 
#{
#  my $sigName = shift;
#  print STDERR $daemon_name." : Received SIG".$sigName."\n";
#  $sys_log_sock = 0;
#  if ($log_host && $sys_log_port) {
#    $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
#  }
#}

#
# Create the local directories required for this observation
#
sub createLocalDirs(\%)
{
  my ($h_ref) = @_;

  logMsg(2, "INFO", "createLocalDirs()");
 
  my %h = %$h_ref;
  my $utc_start = $h{"UTC_START"};
  my $ant_id    = $h{"ANT_ID"};
  my $pfb_id    = $cfg{"PWC_PFB_ID_".$pwc_id};
  my $ant_dir   = $cfg{"CLIENT_RESULTS_DIR"}."/".$pfb_id."/".$utc_start."/".$ant_id;

  my ($cmd, $result, $response);

  logMsg(2, "INFO", "createLocalDirs: mkdirRecursive(".$ant_dir.", 0755)");
  ($result, $response) = Dada::mkdirRecursive($ant_dir, 0755);
  logMsg(3, "INFO", "createLocalDirs: ".$result." ".$response);

  # create an obs.header file in the processing dir:
  logMsg(2, "INFO", "createLocalDirs: creating obs.header");
  my $file = $ant_dir."/obs.header";
  open(FH,">".$file.".tmp");
  my $k = "";
  foreach $k ( keys %h) 
  {
    print FH Dada::headerFormat($k, $h{$k})."\n";
  }
  close FH;
  rename($file.".tmp", $file);

  return 0;
}
