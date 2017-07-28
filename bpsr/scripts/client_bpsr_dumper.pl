#!/usr/bin/env perl

#
# Monitor dump datablock to write dump events to disk
#

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
$beam = "";
%cfg = Bpsr::getConfig();
%roach = Bpsr::getROACHConfig();
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
  my ($cmd, $result, $response, $proc_cmd, $full_cmd);
  my (%h);

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

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  logMsg(1,"INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  # a short buffer for writing events to disk
	my $dump_db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"DUMP_DATA_BLOCK"});

  my $can_dump = 1;
  my $can_backup_dump = 1;

  my $dump_dir = "";
  my $backup_dump_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$beam;

	my $curr_raw_header = "";
	my $prev_raw_header = "";

  # create the backup dump diretory if it does not exist
  if (! -d $backup_dump_dir)
  {
    logMsg(2, "INFO", "main: Dada::createDir(".$backup_dump_dir.", 0755)");
    ($result, $response) = Dada::createDir($backup_dump_dir, "0755");
    logMsg(2, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      $can_backup_dump = 0;
    }

    $cmd = "touch ".$backup_dump_dir."/test.touch";
    logMsg(2, "INFO", "dumperThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    logMsg(3, "INFO", "dumperThread: ".$result." ".$response);
    if ($result ne "ok")
    {
      $can_backup_dump = 0;
    }
    else
    {
      logMsg(2, "INFO", "dumperThread: unlink ".$backup_dump_dir."/test.touch");
      unlink $backup_dump_dir."/test.touch";
    }
  }


  # Main Loop
  while (!$quit_daemon) 
  {

		# next header to read from the receiving data_block
    $cmd = "dada_header -t dumper -k ".$dump_db_key;
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
    else
    {
      # get the header so we can extract the UTC_START
      %h = Dada::headerToHash ($curr_raw_header);

      # create dump directory (the default observation directory) if it doesn't
      # already exist
      $can_dump = 1;
      $dump_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$h{"UTC_START"};
      if (! -d $dump_dir)
      {
        logMsg(2, "INFO", "main: Dada::createDir(".$dump_dir.", 0755)");
        ($result, $response) = Dada::createDir($dump_dir, "0755");
        logMsg(2, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          $can_dump = 0;
        }
      }

      # check that it is possible to create a file in this directory
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

      # determine the processing command
      if ($can_dump)
      {
        $cmd = "dada_dbdisk -k ".$dump_db_key." -D ".$dump_dir." -s";
      }
      elsif ($can_backup_dump)
      {
        $cmd = "dada_dbdisk -k ".$dump_db_key." -D ".$backup_dump_dir." -s";
      }
      else
      {
        $cmd = "dada_dbnull -k ".$dump_db_key." -s  -z";
      }

      logMsg(1, "INFO", "START [dump] ".$cmd);
      ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock,
                                                 "src", sprintf("%02d",$pwc_id),
                                                 $daemon_name, "dump");
      logMsg(1, "INFO", "END   [dump] ".$cmd);
      if ($result ne "ok")
      {
        logMsg(1, "WARN", "main: ".$cmd." failed ".$response);
      }

      # if we dumped successfully, make a backup copy on the local disk
      if ($can_dump && $can_backup_dump)
      {
        $cmd = "cp ".$dump_dir."/*.dada ".$backup_dump_dir."/";
        logMsg(2, "INFO", "dumperThread: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(3, "INFO", "dumperThread: ".$result." ".$response);
      }
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
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}

sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";
	my $dump_db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"DUMP_DATA_BLOCK"});

  my ($cmd, $result, $response, $process, $user);
  my @processes_to_kill = ();

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  # we have dada_headers listening on the receiving, event and dump data blocks - kill them, 
  # then allow a short time for those threads to exit
  $process = "^dada_header -t dumper -k ".$dump_db_key;
  $user = "bpsr";

  logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
  ($result, $response) = Dada::killProcess($process, $user, $$);
  logMsg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  sleep (1);

  # now try to kill any other binary processes that might be running
  push @processes_to_kill, "^dada_dbdisk -k ".$dump_db_key;
  push @processes_to_kill, "^dada_dbnull -k ".$dump_db_key;

  foreach $process ( @processes_to_kill)
  {
    logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
    ($result, $response) = Dada::killProcess($process, $user, $$);
    logMsg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
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
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, $pwc_id, $time, "sys", $type, "dump", $msg);
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
    if ($sys_log_sock) {
      close($sys_log_sock);
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
