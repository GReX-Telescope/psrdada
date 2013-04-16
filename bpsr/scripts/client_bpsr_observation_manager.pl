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
our $db_key : shared;
our %cfg : shared;
our %roach : shared;
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
%cfg = Bpsr::getConfig();
%roach = Bpsr::getROACHConfig();
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
    $beam = $roach{"BEAM_".$pwc_id};
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
	my $trans_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"TRANSIENT_DATA_BLOCK"});

	my $curr_raw_header = "";
	my $prev_raw_header = "";
	my %header = ();

	my $processing_thread = 0;
	my $auxiliary_thread = 0;
	my $transient_thread = 0;

  my $out_db_key_used = "";

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

      # start 2 null threads as the datablock is configured for 2 readers
      my $null_thread1 = threads->new(\&nullThread, $recv_db_key, "proc");
      my $null_thread2 = threads->new(\&nullThread, $recv_db_key, "deci");

      $null_thread1->join();
      $null_thread2->join();
		}
		else
		{
      %header = Dada::headerToHash($curr_raw_header);

      # determine if we run the transient pipeline
      $trans_db_key = "";
      if (-f $cfg{"CONFIG_DIR"}."/run_heimdall")
      {
        $trans_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"TRANSIENT_DATA_BLOCK"});
      }

      # check the BEAM_LEVEL (average scaling coefficient) that server_bpsr_tcs_interface set, 
      # if > 40000, then its likely that the LNA is off, an so we wont run heimdall
      # check whether beam is active or not
      if (exists($header{"BEAM_LEVEL"}) && (int($header{"BEAM_LEVEL"}) > 40000) && ($header{"BEAM_ACTIVE"} eq "off"))
      {
        logMsg(0, "WARN", "LNA for Beam ".$beam." appears to be off");
        # disable transient pipeline as this beam's LNA may not even be on
        $trans_db_key = "";
      }

      # determine processing command
      my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$header{"PROC_FILE"};
      logMsg(2, "INFO", "Full path to PROC_FILE: ".$proc_cmd_file);
      
      my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
      my $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

      my $obs_start_file = createLocalDirs($header{"UTC_START"}, $curr_raw_header);

      # if the proc_cmd matches the decimator, then we can perform monitoring and
      # transient analysis without running dbcopydb and digiplot
      if (($proc_cmd =~ m/the_decimator/) && ($header{"BEAM_ACTIVE"} ne "off"))
      {
        $processing_thread = threads->new(\&processingThread, $recv_db_key, $trans_db_key, $obs_start_file, $curr_raw_header);
    
        $auxiliary_thread = threads->new(\&nullThread, $recv_db_key, "deci");
      }
      else
      {
        # start a thread to duplicate the datablock stream
        # logMsg(2, "INFO", "main: starting dbcopydbThread");
        # $dbcopydb_thread = threads->new(\&dbcopydbThread, $in_db_key, $proc_db_key, $aux_db_key);

        # start the processing thread primary DB
        logMsg(2, "INFO", "main: starting processingThread");
        $processing_thread = threads->new(\&processingThread, $recv_db_key, "", $obs_start_file, $curr_raw_header);

        # start the plot/mon thread on aux DB
        logMsg(2, "INFO", "main: starting auxiliaryThread");
        $auxiliary_thread = threads->new(\&auxiliaryThread, $recv_db_key, $trans_db_key, \%header);
      }

      if ($trans_db_key ne "")
      {
        $transient_thread = threads->new(\&transientThread, $trans_db_key, \%header);
      }

			# we now wait for threads to finish, just join the threads

      logMsg(2, "INFO", "main: joining processingThread");
			($result, $response) = $processing_thread->join();
			if ($result ne "ok")
			{
				logMsg(0, "ERROR", "main: processingThread failed: ".$response);
        $quit = 1;
			}

      if ($auxiliary_thread)
      {
        logMsg(2, "INFO", "main: joining auxiliaryThread");
        ($result, $response) = $auxiliary_thread->join();
        if ($result ne "ok")
        {
          logMsg(0, "ERROR", "main: auxiliaryThread failed: ".$response);
          $quit = 1;
        }
        $auxiliary_thread = 0;
      }

      if ($transient_thread)
      {
        ($result, $response) = $transient_thread->join();
        if ($result ne "ok")
        {
          logMsg(0, "ERROR", "main: transientThread failed: ".$response);
          $quit = 1;
        }
        $transient_thread = 0;
      }

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
# runs a thread to duplicate the datablock stream from in_key to out_key1 and out_key2
#
sub dbcopydbThread($$$) 
{
  (my $in_key, my $out_key1, my $out_key2) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $full_cmd = "";

  $cmd = "dada_dbcopydb -z -s ".$in_key." ".$out_key1." ".$out_key2;
  logMsg(2, "INFO", "dbcopydbThread: ".$cmd);

  logMsg(1, "INFO", "START ".$cmd);
  $full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." copy";
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(1, "INFO", "END   ".$cmd);
  logMsg(2, "INFO", "dbcopydbThread: ".$result." ".$response);
	if ($result ne "ok")
	{
		logMsg(0, "INFO", "dbcopydbThread: ".$cmd." failed: ".$response);
		if ($quit_daemon)
		{
			logMsg(0, "INFO", "dbcopydbThread: quit_daemon true, probably ok");
		}
		else
		{
			logMsg(0, "ERROR", "dada_dbcopydb failed, restart may be required");	
		}
		return "fail";
	}
	else
	{
		return "ok";
	}
}


#
# Processes a single observation
#
# Creates the required directories for the output data on the server, and generates the obs.start file
#
sub processingThread($$$$) 
{
  (my $in_key, my $out_key, my $obs_start_file, my $raw_header) = @_;

	my %h = Dada::headerToHash($raw_header);
  my $processing_dir = "";
  my $utc_start = "";
  my $acc_len = "";
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

  if (exists($h{"BEAM_ACTIVE"}))
  {
    logMsg(2, "INFO", "h{BEAM_ACTIVE} ".$h{"BEAM_ACTIVE"});
  }
  else
  {
    logMsg(2, "INFO", "h{BEAM_ACTIVE} not set");
  }

  # move to parent loop
  # my $obs_start_file = createLocalDirs($h{"UTC_START"}, $raw_header);

	# if malformed
	if (! $header_ok) 
	{
		logMsg(0, "ERROR", "DADA header malformed, jettesioning xfer");
		$proc_cmd = "dada_dbnull -s -k ".$in_key;
	} 
	elsif ($beam ne $h{"BEAM"})
	{
		logMsg(0, "ERROR", "Beam mismatch between header[".$h{"BEAM"}."] and config[".$beam."]");
		$proc_cmd = "dada_dbnull -s -k ".$in_key;
	}
  else
  { 

    # launch thread to create remote NFS dirs, as this can take a long time due to NFS load
    $remote_dirs_thread = threads->new(\&remoteDirsThread, $h{"UTC_START"}, $obs_start_file);

    $processing_dir =  $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};

    if ($h{"BEAM_ACTIVE"} eq "off")
    {
      logMsg(0, "INFO", "Ignoring observation as header[BEAM_ACTIVE] == off");
      $proc_cmd = "dada_dbnull -s -k ".$in_key;
    }
    else
    {
      # Add the dada header file to the proc_cmd
      my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"PROC_FILE"};

      logMsg(2, "INFO", "Full path to PROC_FILE: ".$proc_cmd_file);

      my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
      $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
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
    if (($proc_cmd =~ m/the_decimator/) && ($out_key ne ""))
    {
      $proc_cmd .= " -k ".$out_key;
    }

    logMsg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
  }

  logMsg(1, "INFO", "START ".$proc_cmd);
  logMsg(2, "INFO", "Changing dir to $processing_dir");

  $cmd = "cd ".$processing_dir."; ".$proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." proc";

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
# runs a thread to execute auxiliary processing pipeline, which runs
# the decimator, without producing psrxml or sigproc files
#
sub auxiliaryThread($$$) 
{
  (my $in_key, my $out_key, my $header_ref) = @_;

  my %h = %$header_ref;

	my $cmd = "";
	my $result = "";
	my $response = "";
  my $gpu_id = $cfg{"PWC_GPU_ID_".$pwc_id};
  my $processing_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};
  my $dada_info_file = "/tmp/bpsr_".$in_key.".info";

  # always set digitizer nbit to 8 and 10 second blocks
  $cmd = "the_decimator -n -b 8 -B 10 -c -o ".$h{"UTC_START"};
  
  # if we have an output key, enable this
  if ($out_key ne "")
  {
    $cmd .= " -k ".$out_key;
  }

  # ensure a file exists with the write processing key
  if (! -f $dada_info_file)
  {
    open FH, ">".$dada_info_file;
    print FH "DADA INFO:\n";
    print FH "key ".$in_key."\n";
    close FH;
  }
  $cmd .= " ".$dada_info_file;

	logMsg(1, "INFO", "START ".$cmd);

  my $full_cmd = "cd ".$processing_dir."; ".$cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." deci";

	logMsg(2, "INFO", "auxiliaryThread: ".$full_cmd);
	($result, $response) = Dada::mySystem($full_cmd);
	logMsg(2, "INFO", "auxiliaryThread: ".$result." ".$response);

	logMsg(1, "INFO", "END   ".$cmd);

	return "ok";
}

#
# runs a thread to execute dada_dbnull on the specified data block key
#
sub nullThread($$)
{
  (my $in_key, my $tag) = @_;

  my $cmd = "dada_dbnull -s -k ".$in_key;
  my $result = "";
  my $response = "";

  my $full_cmd = $cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." ",$tag;

  logMsg(1, "INFO", "START ".$cmd);

  logMsg(2, "INFO", "nullThread: ".$full_cmd);
  ($result, $response) = Dada::mySystem($full_cmd);
  logMsg(2, "INFO", "nullThread: ".$result." ".$response);

  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}
  
#
# runs a thread to execute transient processing pipeline
#
sub transientThread($$) 
{
  (my $key, my $header_ref) = @_;

  my %h = %$header_ref;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $gpu_id = $cfg{"PWC_GPU_ID_".$pwc_id};
  my $processing_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};

  # added 4096 to give an extra 2^12th bin for junk events
  #$cmd = "heimdall -k ".$key." -gpu_id ".$gpu_id." -beam ".$beam." -output_dir ".$processing_dir;
  $cmd = "heimdall -dm 0 1500 -boxcar_max 4096 -min_tscrunch_width 8 -k ".$key." -gpu_id ".$gpu_id.
         " -zap_chans 0 150 -zap_chans 335 338 -zap_chans 181 183 -max_giant_rate 20000 -beam ".$beam." -output_dir ".$processing_dir;
  logMsg(1, "INFO", "START ".$cmd);

  my $full_cmd = "cd ".$processing_dir."; ".$cmd."  2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$pwc_id." tran";

  logMsg(2, "INFO", "transientThread: ".$full_cmd);
  ($result, $response) = Dada::mySystem($full_cmd);
  logMsg(2, "INFO", "transientThread: ".$result." ".$response);

  logMsg(1, "INFO", "END   ".$cmd);

  return "ok";
}




#
# Thread to create remote NFS links on the server
#
sub remoteDirsThread($$) {

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



#
# Fold
# 
sub foldBeam($) 
{

  my ($raw_header) = @_;

  if ($raw_header eq "") 
  {
    logMsg(0, "WARN", "foldBeam: no header supplied");
    return;
  }

  my %h = Dada::headerToHash($raw_header);
  my $proc_cmd = "";
  my $cmd = "";
  my $rval = 0;
  my $result = "";
  my $response = "";
  my $fil_file = "";
  my $par_file = "";
  my $ar_file = "";
  my $work_dir = "";

  my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"PROC_FILE"};
  my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
  $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
  logMsg(2, "INFO", "foldBeam: Initial PROC_CMD: ".$proc_cmd);

  # If we 2bit file and source starts with J and Beam01
  if (!(($proc_cmd =~ m/the_decimator/) && ($h{"SOURCE"} =~ m/^J/) && ($beam eq "01")))
  {
    logMsg(2, "INFO", "foldBeam: didn't meet folding criteria");
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
    return;
  }

  logMsg(1, "INFO", "foldBeam: UTC_START=".$h{"UTC_START"}.", SOURCE=".$h{"SOURCE"});

  # check that the SIGPROC filterbank file exists
  $fil_file = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/".$h{"UTC_START"}.".fil";
  if (! -f $fil_file) 
  {
    logMsg(1, "INFO", "foldBeam: fil file: ".$fil_file." did not exist");
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
  }

  # ensure we have the latest copy of the HTRU database
  $cmd = "scp -o BatchMode=yes dada\@jura.atnf.csiro.au:/psr1/cvshome/pulsar/soft_atnf/search/hitrun/database/htru_interim.db ~/";
  logMsg(2, "INFO", "foldBeam: ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($cfg{"USER"}, $cfg{"SERVER_HOST"}, $cmd);
  logMsg(3, "INFO", "foldBeam: ".$result." ".$rval." ".$response);

  if (($result ne "ok") || ($rval != 0))
  {
    logMsg(1, "INFO", "foldBeam: failed to update HTRU psrcat db");
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
    return 0;
  }
    
  # create a temporary par file 
  $par_file = "/tmp/".$h{"SOURCE"}.".par";
  logMsg(1, "INFO", "foldBeam: setting par_file to: ".$par_file);
  if (-f $par_file ) 
  {
    unlink ($par_file);
  }
  $cmd = "psrcat -merge ~/htru_interim.db -e ".$h{"SOURCE"}." > ".$par_file;
  logMsg(1, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  # since psrcat does not produce set a return value on success/failure i
  # to find a psr, check the parfile for "not in catalogue"
  $cmd = "grep 'not in catalogue' ".$par_file;
  logMsg(2, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  # if we found this string, then the ephemeris will be invlaid
  if ($result eq "ok")
  {
    logMsg(1, "INFO", "foldBeam: failed to extract ephemeris from ~/htru_interim.db");
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
    return 0;
  }

  $work_dir = "/tmp/".$h{"UTC_START"};

  # working dir to dump dspsr output
  $cmd = "mkdir ".$work_dir;
  logMsg(2, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "INFO", "foldBeam: failed to create ".$work_dir.": ".$response);
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
    return 0;
  }

  $ar_file = $work_dir."/".$h{"UTC_START"}.".ar";

  # run dspsr with the temp par file (CHECK WORKING DIR)
  $cmd = "cd ".$work_dir."; dspsr -q -t 3 -U 1 -E ".$par_file." -L 10 ".$fil_file." -e aa";
  logMsg(1, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  if ($result ne "ok") 
  {
    logMsg(0, "WARN", "foldBeam: dspsr failed: ".$response);
    $cmd = "rm -rf ".$work_dir;
    ($result, $response) = Dada::mySystem($cmd);
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);

    return 0;
  }

  # psradd the output archives together
  $cmd = "psradd -o -T ".$ar_file." ".$work_dir."/*.aa";
  logMsg(1, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  if ($result ne "ok")
  {
    logMsg(0, "WARN", "foldBeam: psradd failed: ".$response);
    $cmd = "rm -rf ".$work_dir;
    ($result, $response) = Dada::mySystem($cmd);
    $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
    logMsg(2, "INFO", "foldBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "foldBeam: ".$result." ".$response);
    return 0;
  }

  # copy the archive to results dir
  $cmd = "scp ".$ar_file." dada@".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_RESULTS_DIR"}."/".$h{"UTC_START"}."/".$beam."/";
  logMsg(1, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  if ($result eq "ok") {
    logMsg(1, "INFO", "foldBeam: SUCCESS!!");
  } else {
    logMsg(0, "WARN", "foldBeam: failed to copy dspsrd file: ".$response);
  }

  $cmd = "rm -rf ".$work_dir;
  logMsg(2, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(2, "INFO", "foldBeam: ".$result." ".$response);

  $cmd = "touch ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"}."/beam.finished";
  logMsg(2, "INFO", "foldBeam: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "foldBeam: ".$result." ".$response);

  return 0;

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
  $cmd = "ps aux | grep -v grep | grep bpsr | grep 'dada_header -k ".$db_key."' | awk '{print \$2}'";

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

sub sigPipeHandle($) {

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
sub createLocalDirs($$) {

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

  # create the aux subsubdir
  $dir = $base."/".$utc_start."/aux";
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
