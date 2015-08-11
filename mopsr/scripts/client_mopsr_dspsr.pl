#!/usr/bin/env perl

# 
# Simple MOPSR processing script
#
#   Runs dspsr on the individiaul data blocks
# 
# Author:   Andrew Jameson
# 

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use strict;
use warnings;

use Mopsr;          # DADA Module for configuration options
use File::Basename; 
use threads;
use threads::shared;
use IPC::Open3;
use IO::Select;
use LockFile::Simple qw(trylock lock unlock);
use Time::HiRes qw(usleep);
use Symbol 'gensym'; 

#
# Function Prototypes
#
sub logMsg($$$);
sub multiFork($\@);
sub multiForkLog($\@);
sub prepareObservation($$);
sub createLocalDirs(\%);
sub genTempo2Polyco($$);


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
our @db_keys : shared;
our %cfg : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
@db_keys = ();
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

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
    my $db_id;
    my @db_ids = split(/ /,$cfg{"PROCESSING_DATA_BLOCK"});
    foreach $db_id ( @db_ids )
    {
      push @db_keys, Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $db_id);
    }
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
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd, $curr_raw_header, $prev_raw_header);
  my ($i, $index, $line, $rval, $db_key);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Auto flush output
  $| = 1;

  $cmd = "mkdir -p /tmp/tempo2/mpsr";
  logMsg(0, "INFO", "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(0, "INFO", "main: ".$result." ".$response);

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

  # this thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

	$curr_raw_header = "";
	$prev_raw_header = "";
	my %header;

  my (@cmds, @results, @responses, $results_ref, $responses_ref);
  my ($obs_dir);
  my $err = gensym;

  my $nkeys = $#db_keys + 1;

  # Main Loop
  while (!$quit_daemon) 
  {
		%header = ();

    # run dada_header on all the input data blocks
    for ($i=0; $i<=$#db_keys; $i++)
    {
      $cmds[$i] = "dada_header -k ".$db_keys[$i];
    }
    ($results_ref, $responses_ref) = multiFork ($nkeys, @cmds);
    @results = @$results_ref;
    @responses = @$responses_ref;

    if ($quit_daemon)
    {
      logMsg(0, "INFO", "main: quit daemon true after dada_header returns, breaking");
      next;
    }

    for ($i=0; $i<$nkeys; $i++)
    {
      if ($results[$i] ne "ok") 
      {
        logMsg(0, "WARN", "dada header failed: ".$responses[$i]);
        $quit_daemon = 1;
      }
    }

    if ($quit_daemon)
    {
      logMsg(0, "INFO", "main: a dada_header instance failed, breaking");
      next;
    }

    # ensure that our primary configuration directory (NFS) is mounted
    $cmd = "ls -1d ".$cfg{"CONFIG_DIR"};
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "main: ".$cmd." config_dir=".$response);
    if ($response ne $cfg{"CONFIG_DIR"})
    {
      logMsg(0, "ERROR", "NFS automount for ".$cfg{"CONFIG_DIR"}." failed: ".$response);
      $quit_daemon = 1;
    }

    my $need_polyco = 1;
    # get the first header
    my %h = Dada::headerToHash($responses[0]);
    if (($h{"AQ_PROC_FILE"} eq "mopsr.dspsr.gpu.1fold") || ($h{"AQ_PROC_FILE"} eq "mopsr.dspsr.gpu.10fold"))
    {
      $need_polyco = 0;
    }

    if ($need_polyco)
    {
      logMsg(1, "INFO", "generating polycol once for all antenna");
      # generate the dspsr polycol necessary for all threads on this host
      logMsg(2, "INFO", "main: genTempo2Polyco(".$db_keys[0].", ".$responses[0].")");
      ($result, $response) = genTempo2Polyco($db_keys[0], $responses[0]);
      logMsg(3, "INFO", "main: genTempo2Polyco() ".$result." ".$response);
      if ($result ne "ok")
      { 
        logMsg(0, "ERROR", "main: genTempo2Polyco failed: ".$response);
      }
    }

    # perform all the necessary setup for observations and return the 
    # required command for execution on the datablocks
    logMsg(1, "INFO", "preparing observation");
    for ($i=0; $i<$nkeys; $i++)
    {
      ($result, $cmds[$i]) = prepareObservation($db_keys[$i], $responses[$i]);
      if ($result ne "ok")
      {
        logMsg(0, "ERROR", "main: failed to prepareObservation for key ".$i." [".$db_keys[$i]."]");
        $quit_daemon = 1;
      }
    }

    if (!$quit_daemon)
    {
      if ($need_polyco)
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
          usleep(10000);
        }
      }

      ($results_ref) = multiForkLog ($nkeys, @cmds);
      @results = @$results_ref;
    }
  }

  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}


sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my ($cmd, $result, $response);

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) 
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my $user = "mpsr";
  my @processes_to_kill = ();
  my ($i, $process);
  for ($i=0; $i<=$#db_keys; $i++)
  {
    push @processes_to_kill, "^dada_header -k ".$db_keys[$i];
  }

  foreach $process ( @processes_to_kill)
  {
    logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  @processes_to_kill = ();
  for ($i=0; $i<=$#db_keys; $i++)
  {
    push @processes_to_kill, "^dspsr /tmp/mopsr_".$db_keys[$i].".info";
  }

  foreach $process ( @processes_to_kill)
  {
    logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  logMsg(1, "INFO", "controlThread: checking for PID file");
  if ( -f $pid_file) 
  {
    logMsg(1, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  logMsg(1, "INFO", "controlThread: exiting");
}

sub prepareObservation($$)
{
  my ($key, $header) = @_;

  my %h = Dada::headerToHash($header);

  # default command to return is jettision
  my $proc_cmd = "dada_dbnull -k ".$key." -s -z";

  # default processing directory
  my $obs_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$cfg{"PWC_PFB_ID_".$pwc_id}."/".$h{"UTC_START"};

  my $proc_dir = $obs_dir."/".$h{"ANT_ID"};

  # create the local directories required
  if (!exists($h{"ANT_ID"}))
  {
    logMsg(0, "INFO", "prepareObservation: ANT_ID did not exist in header");
    return ("fail", $obs_dir, $proc_cmd);
  }

  if (createLocalDirs(%h) < 0)
  {
    logMsg(0, "INFO", "prepareObservation: failed to create local dir");
    return ("fail", $obs_dir, $proc_cmd);
  }

  logMsg(2, "INFO", "prepareObservation: local dirs created");

  my $obs_header = $obs_dir."/".$h{"ANT_ID"}."/obs.header";

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
  if (length($h{"AQ_PROC_FILE"}) < 1)
  {
    logMsg(0, "ERROR", "AQ_PROC_FILE was malformed or non existent");
    $header_ok = 0;
  }

  # if malformed
  if (!$header_ok)
  {
    logMsg(0, "ERROR", "DADA header malformed, jettesioning xfer");
    $proc_cmd = "dada_dbnull -k ".$key." -s -z";
  }
  # only process inputs 0-7
  #elsif ((int($h{"ANT_ID"}) < 0) || (int($h{"ANT_ID"} > 7)))
  #{
  #  logMsg(0, "INFO", "Deliberately disable ant: ".$h{"ANT_ID"});
  #  $proc_cmd = "dada_dbnull -k ".$key." -s -z -q";
  #}
  else
  {
    # Add the dada header file to the proc_cmd
    my $proc_cmd_file = $cfg{"CONFIG_DIR"}."/".$h{"AQ_PROC_FILE"};

    logMsg(2, "INFO", "Full path to AQ_PROC_FILE: ".$proc_cmd_file);

    my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
    $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

    logMsg(2, "INFO", "Initial PROC_CMD: ".$proc_cmd);

    # replace <DADA_INFO> tags with the matching input .info file
    if ($proc_cmd =~ m/<DADA_INFO>/)
    {
      my $tmp_info_file =  "/tmp/mopsr_".$key.".info";
      # ensure a file exists with the write processing key
      if (! -f $tmp_info_file)
      {
        open FH, ">".$tmp_info_file;
        print FH "DADA INFO:\n";
        print FH "key ".$key."\n";
        close FH;
      }
      $proc_cmd =~ s/<DADA_INFO>/$tmp_info_file/;
    }

    # replace <DADA_KEY> tags with the matching input key
    $proc_cmd =~ s/<DADA_KEY>/$key/;

    # replace <DADA_RAW_DATA> tag with processing dir
    $proc_cmd =~ s/<DADA_DATA_PATH>/$proc_dir/;

    # replace DADA_UTC_START with actual UTC_START
    $proc_cmd =~ s/<DADA_UTC_START>/$h{"UTC_START"}/;

    # replace DADA_ANT_ID with actual ANT_ID
    $proc_cmd =~ s/<DADA_ANT_ID>/$h{"ANT_ID"}/;

    # replace DADA_ANT_ID with actual ANT_ID
    $proc_cmd =~ s/<DADA_PFB_ID>/$h{"PFB_ID"}/;

    # replace DADA_ANT_ID with actual ANT_ID
    my $mpsr_ib_port = 40000 + int($pwc_id);
    $proc_cmd =~ s/<MPSR_IB_PWC_PORT>/$mpsr_ib_port/;

    my $gpu_id = $cfg{"PWC_GPU_ID_".$pwc_id};

    # replace DADA_GPU_ID with actual GPU_ID
    $proc_cmd =~ s/<DADA_GPU_ID>/$gpu_id/;

    if (($proc_cmd =~ m/dspsr/) && (!($proc_cmd =~ m/ -c /)))
    {
      $proc_cmd .= " -E /tmp/tempo2/mpsr/pulsar.par -P /tmp/tempo2/mpsr/t2pred.dat";

      if (int($h{"ANT_ID"}) % 2 == 0)
      {
        $proc_cmd .= " -cpu 4";
      }
      else
      {
        $proc_cmd .= " -cpu 5";
      }
    }

    # processing must ocurring in the specified dir
    $proc_cmd = "cd ".$proc_dir."; ".$proc_cmd;

    logMsg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);
  }
  return ("ok", $proc_cmd);
}

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
      Dada::nexusLogMessage($sys_log_sock, $pwc_id, $time, "sys", $type, "dspsr mngr", $msg);
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

sub sigPipeHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $sys_log_sock = 0;
  if ($log_host && $sys_log_port) {
    $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  }
}

sub multiFork($\@)
{
  my ($ncmds, $cmds_ref) = @_;

  my @cmds = @$cmds_ref;

  my @pids = ();
  my @ins = ();
  my @outs = ();
  my @errs = ();

  my ($i, $cmd, $cmd_in, $cmd_out, $cmd_err, $index, $line, $rval, $pid);

  my @out_bufs = ();
  my @err_bufs = ();
  my @results = ();
  my @responses = ();

  my $err = gensym;
  my $sel = IO::Select->new();

  # first we need to run dada_header on the input data blocks 
  for ($i=0; $i<$ncmds; $i++)
  {
    $cmd = $cmds[$i];
    push (@results, "fail");
    push (@responses, "");
  
    my ($cmd_in, $cmd_out, $cmd_err);

    logMsg(2, "INFO", "multiFork: open3 for ".$cmd);
    logMsg(0, "INFO", "START ".$cmd);
    #eval {
      $pid = open3 ($cmd_in, $cmd_out, $cmd_err, $cmd);
    #};
    #die "open3: $@\n" if $@;

    push (@pids, $pid);
    push (@ins, $cmd_in);

    $sel->add ($cmd_out);
    push (@outs, $cmd_out);
    push (@out_bufs, "");
   
    if ($cmd_err)
    { 
      $sel->add ($cmd_err);
      push (@outs, $cmd_err);
      push (@out_bufs, "");
    }

    # close STDIN
    close ($cmd_in);
  }

  # now that all are launched, we need to read back the data
  # from STDOUT and STDERR
  while (my @ready = $sel->can_read())
  {
    foreach my $handle (@ready)
    {
      $index = -1;
      for ($i=0; $i<$ncmds; $i++)
      {
        if ($handle == $outs[$i])
        {
          $index = $i;
        }
      }
      if ($index eq -1)
      {
        logMsg(0, "ERROR", "could not match handle to I/O FD");
        sleep (1);
        $index = 0;
      }

      # read from the handle in 1024 byte chunks
      my $bytes_read = sysread ($handle, my $buf='', 1024);
      if ($bytes_read == -1)
      {
        #warn("Error reading from child's STDOUT: $!\n");
        $sel->remove($handle);
        next;
      }
      if ($bytes_read == 0)
      {
        # print("Child's STDOUT closed\n");
        $sel->remove($handle);
        next;
      }
      else
      {
        my @lines = split(/\n/, $buf);
        my $nlines = $#lines + 1;
        if (!($buf =~ m/\n$/))
        {
          $nlines -= 1;
        }
        for ($i=0; $i<$nlines; $i++)
        {
          $line = $lines[$i];
          # print "index=$index lines[$i]=$line\n";

          if ($out_bufs[$index] ne "")
          {
            $line = $out_bufs[$index].$line;
            $out_bufs[$index] = "";
          }

          $responses[$index] .= $line."\n";
        }

        if (!($buf =~ m/\n$/))
        {
          $out_bufs[$index] = $lines[$nlines];
        }
      }
    }
  }

  for ($i=0; $i<$ncmds; $i++)
  {
    waitpid ($pids[$i], 0);
    $rval = $? >> 8;
    $results[$i] = ($rval == 0) ? "ok" : "fail";
    logMsg(0, "INFO", "END   ".$cmds[$i]);
    chomp ($responses[$i]);
  }

  return (\@results, \@responses);
}


sub multiForkLog($\@)
{
  my ($ncmds, $cmds_ref) = @_;

  my @cmds = @$cmds_ref;

  my @pids = ();
  my @ins = ();
  my @outs = ();
  my @errs = ();

  my ($i, $cmd, $cmd_in, $cmd_out, $cmd_err, $index, $line, $rval, $pid);

  my @out_bufs = ();
  my @err_bufs = ();
  my @results = ();

  my $err = gensym;
  my $sel = IO::Select->new();

  # first we need to run dada_header on the input data blocks 
  for ($i=0; $i<$ncmds; $i++)
  {
    $cmd = $cmds[$i];
    push (@results, "fail");
  
    my ($cmd_in, $cmd_out, $cmd_err);
    logMsg(2, "INFO", "multiFork: open3 for ".$cmd);

    logMsg(0, "INFO", "START ".$cmd);
    #eval {
      $pid = open3 ($cmd_in, $cmd_out, $cmd_err, $cmd);
    #};
    #die "open3: $@\n" if $@;

    push (@pids, $pid);
    push (@ins, $cmd_in);

    $sel->add ($cmd_out);
    push (@outs, $cmd_out);
    push (@out_bufs, "");
   
    if ($cmd_err)
    { 
      $sel->add ($cmd_err);
      push (@outs, $cmd_err);
      push (@out_bufs, "");
    }

    # close STDIN
    close ($cmd_in);
  }

  # now that all are launched, we need to read back the data
  # from STDOUT and STDERR
  while (my @ready = $sel->can_read())
  {
    foreach my $handle (@ready)
    {
      $index = -1;
      for ($i=0; $i<$ncmds; $i++)
      {
        if ($handle == $outs[$i])
        {
          $index = $i;
        }
      }
      if ($index eq -1)
      {
        logMsg(0, "ERROR", "could not match handle to I/O FD");
        sleep (1);
        $index = 0;
      }

      # read from the handle in 1024 byte chunks
      my $bytes_read = sysread ($handle, my $buf='', 1024);
      if ($bytes_read == -1)
      {
        warn("Error reading from child's STDOUT: $!\n");
        $sel->remove($handle);
        next;
      }
      if ($bytes_read == 0)
      {
        #print("Child's STDOUT closed\n");
        $sel->remove($handle);
        next;
      }
      else
      {
        my @lines = split(/\n/, $buf);
        my $nlines = $#lines + 1;
        if (!($buf =~ m/\n$/))
        {
          $nlines -= 1;
        }
        for ($i=0; $i<$nlines; $i++)
        {
          $line = $lines[$i];
          #print "index=$index lines[$i]=$line\n";

          if ($out_bufs[$index] ne "")
          {
            $line = $out_bufs[$index].$line;
            $out_bufs[$index] = "";
          }

          Dada::nexusPipeLog ($line, $src_log_sock, $src_log_file, "src", $pwc_id, $daemon_name, "proc");
        }

        if (!($buf =~ m/\n$/))
        {
          $out_bufs[$index] = $lines[$nlines];
        }
      }
    }
  }

  for ($i=0; $i<$ncmds; $i++)
  {
    waitpid ($pids[$i], 0);
    $rval = $? >> 8;
    logMsg(0, "INFO", "END   ".$cmds[$i]);
    $results[$i] = ($rval == 0) ? "ok" : "fail";
  }

  return (\@results);
}

#
# generate a new set of t2 predictors to be used with dspsr
#
sub genTempo2Polyco($$)
{
  my ($key, $raw_header) = @_;
  my ($cmd, $result, $response);

  $cmd = "mkdir -p /tmp/tempo2/mpsr";
  logMsg(2, "INFO", "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  logMsg(3, "INFO", "main: ".$result." ".$response);

  {
    my $lock_handle = trylock ("/tmp/tempo2/mpsr/mpsr");
    if ($lock_handle)
    {
      $cmd = "rm -f /tmp/tempo2/mpsr/.lock /tmp/tempo2/mpsr/pulsar.par ".
             "/tmp/tempo2/mpsr/t2pred.dat";
      logMsg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "main: ".$result." ".$response);
    
      sleep(1);
    
      logMsg(2, "INFO", "main: lock_handle->release()");
      $lock_handle->release();
      logMsg(2, "INFO", "main: lock_handle->release returned");

      # now create the new ephemeris and t2predictor
      my %h = Dada::headerToHash($raw_header);
      my $source = $h{"SOURCE"};
      
      my $tmp_ephem_file = "/tmp/tempo2/mpsr/pulsar.eph";
      $cmd = "psrcat -all -e ".$h{"SOURCE"}." > ".$tmp_ephem_file;
      logMsg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "main: ".$result." ".$response);

      my $tmp_view_file =  "/tmp/mopsr_".$key.".viewer";

      # ensure a file exists with the write processing key
      if (! -f $tmp_view_file)
      {
        open FH, ">".$tmp_view_file;
        print FH "DADA INFO:\n";
        print FH "key ".$key."\n";
        print FH "viewer\n";
        close FH;
      }

      $cmd = "t2pred ".$tmp_ephem_file." ".$tmp_view_file;
      logMsg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(2, "INFO", "main: ".$result." ".$response);
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
          usleep(100000);
        }
        else
        {
          $waiting = 0;
        }
      }
    }
  }
  return ("ok", "");
}
