#!/usr/bin/env perl

#
# Pscrunch the input data block
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
  print "Usage: ".basename($0)." SVD_ID\n";
  print "  SVD_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variable Declarations
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $svd_id : shared;
our $beam : shared;
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
$svd_id = 0;
$beam = "";
%cfg = Bpsr::getConfig("svd");
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
$svd_id  = $ARGV[0];

# ensure that our svd_id is valid 
if (($svd_id >= 0) &&  ($svd_id < $cfg{"NUM_SVD"}))
{
  # and matches configured hostname
  if ($cfg{"SVD_".$svd_id} eq Dada::getHostMachineName())
  {
    # GOOD
  }
  else
  {
    print STDERR "SVD_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "SVD_ID was not a valid integer between 0 and ".($cfg{"NUM_SVD"}-1)."\n";
  usage();
  exit(1);
}


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$svd_id);


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd, $proc_dir, $full_cmd);

  $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$svd_id.".log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$svd_id.".pid";

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

  my $db_key  = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $svd_id, $cfg{"NUM_SVD"}, $cfg{"CLEAN_DATA_BLOCK"});
	my $curr_raw_header = "";
	my $prev_raw_header = "";
  my %h = ();

  # Main Loop
  while (!$quit_daemon) 
  {
		# next header to read from the transient data_block
    $cmd = "dada_header -k ".$db_key;
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
      if ($curr_raw_header eq $prev_raw_header)
		  {  
			  logMsg(0, "ERROR", "main: header repeated, jettesioning observation");
        $proc_cmd = "dada_dbnull -k ".$db_key." -s ";
      } 
		  else
		  {
        %h = Dada::headerToHash ($curr_raw_header);

        my $tobs = 0;
        if (exists($h{"OBS_VAL"}))
        {
          my $tobs = $h{"OBS_VAL"};
          if (exists($h{"OBS_UNIT"}))
          {
            if ($h{"OBS_UNIT"} eq "MINUTES")
            {
              $tobs = $tobs * 60;
            }
          }
        }

        if ($h{"MODE"} eq "CAL")
        {
          logMsg(0, "INFO", "main: MODE=CAL, ignoring observation");
          $proc_cmd = "dada_dbnull -k ".$db_key." -s";
        }
        elsif ($tobs < 30)
        {
          logMsg(0, "INFO", "main: OBS_VAL < 30s, ignoring observation");
          $proc_cmd = "dada_dbnull -k ".$db_key." -s";
        }
        else
        {
          my $ct_file = $cfg{"CONFIG_DIR"}."/bpsr_cornerturn.cfg";
          $proc_cmd = "bpsr_dbnic -k ".$db_key." ".$svd_id." ".$ct_file;
        }
      }

      # setup the full processing command
      $full_cmd = $proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/client_bpsr_src_logger.pl ".$svd_id." clean_send";

      logMsg(1, "INFO", "START [clean_send] ".$proc_cmd);
      ($result, $response) = Dada::mySystem($full_cmd);
      logMsg(1, "INFO", "END   [clean_clean] ".$proc_cmd);
      if ($result ne "ok")
      {
        logMsg(1, "WARN", "main: ".$proc_cmd." failed ".$response);
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
  Dada::nexusLogClose($log_sock);

  exit(0);
}

sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $svd_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$svd_id.".quit";
	my $db_key    = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $svd_id, $cfg{"NUM_SVD"}, $cfg{"CLEAN_DATA_BLOCK"});

  my ($cmd, $result, $response, $process, $user);
  my @processes_to_kill = ();

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $svd_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  $process = "^dada_header -k ".$db_key;
  $user = "bpsr";

  logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
  ($result, $response) = Dada::killProcess($process, $user, $$);
  logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  sleep (1);

  # now try to kill any other binary processes that might be running
  push @processes_to_kill, "^bpsr_dbnic -k ".$db_key;
  push @processes_to_kill, "^dada_dbnull -k ".$db_key;

  foreach $process ( @processes_to_kill)
  {
    logMsg(1, "INFO", "controlThread: killProcess(".$process.", ".$user.", ".$$.")");
    ($result, $response) = Dada::killProcess($process, $user, $$);
    logMsg(1, "INFO", "controlThread: killProcess ".$result." ".$response);
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
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $svd_id, $time, "sys", $type, "scrunch", $msg);
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
