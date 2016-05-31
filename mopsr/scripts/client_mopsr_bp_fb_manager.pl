#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bp_filterbank_manager.pl 
#
# transfer filterbank files from clients to the server
#
###############################################################################

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Mopsr;
use strict;
use threads;
use threads::shared;


sub usage() 
{
  print "Usage: ".basename($0)." PROC_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our %ct : shared;
our $localhost : shared;
our $proc_id : shared;
our $db_key : shared;
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
%cfg = Mopsr::getConfig("bp");
%ct = Mopsr::getCornerturnConfig("bp");
$proc_id = -1;
$db_key = "dada";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$proc_id  = $ARGV[0];

# ensure that our proc_id is valid 
if (($proc_id >= 0) &&  ($proc_id < $cfg{"NUM_BP"}))
{
  # and matches configured hostname
  if ($cfg{"BP_".$proc_id} ne Dada::getHostMachineName())
  {
    print STDERR "BP_".$proc_id."[".$cfg{"BP_".$proc_id}."] did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "proc_id was not a valid integer between 0 and ".($cfg{"NUM_BP"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$proc_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".src.log";
  my $pid_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".pid";

  # this is data stream we will be reading from
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $proc_id, $cfg{"NUM_BP"}, $cfg{"PROCESSING_DATA_BLOCK"});

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  msg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $utc_start);
  my @parts = ();

  chdir $cfg{"CLIENT_RECORDING_DIR"};

  # look for filterbank files to transfer to the server via rsync
  while (!$quit_daemon)
  {
    $cmd = "find . -maxdepth 2 -type f -name 'obs.finished' | sort -nr | tail -n 1";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      # get the observation UTC_START
      @parts = split (/\//, $response);

      if ($#parts == 2)
      {
        $utc_start = $parts[1];
        msg(2, "INFO", "main: found finished observation: ".$utc_start);

        # get the list of beams in this observation
        $cmd = "find ".$utc_start." -maxdepth 1 -type d -name 'BEAM_???' -printf '%f\n' | sort -n";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
    
        if (($result ne "ok") || ($response eq ""))
        {
          msg(0, "WARN", "no beams found in ".$utc_start);

          $cmd = "mv ".$utc_start."/obs.finished ".$utc_start."/obs.failed";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
        }
        else
        {
          $cmd = "rsync -a --stats --bwlimit=8192 --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
                 "./".$utc_start."/BEAM_??? upload\@172.17.228.204::archives/".$utc_start."/";
          #$cmd = "rsync -a --stats --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
          #       "./".$utc_start."/BEAM_??? upload\@192.168.5.10::archives/".$utc_start."/";

          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            if ($quit_daemon)
            {
              msg(0, "INFO", "transfer of ".$utc_start." interrupted");
            }
            else
            {
              msg(0, "ERROR", "transfer of ".$utc_start." failed: ".$response);

              $cmd = "mv ".$utc_start."/obs.finished ".$utc_start."/obs.failed";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
            }
          }
          else
          {
            # determine the data rate
            my @output_lines = split(/\n/, $response);
            my $mbytes_per_sec = 0;
            my $j = 0;
            for ($j=0; $j<=$#output_lines; $j++)
            {
              if ($output_lines[$j] =~ m/bytes\/sec/)
              {
                my @bits = split(/[\s]+/, $output_lines[$j]);
                $mbytes_per_sec = $bits[6] / 1048576;
              }
            }
            my $data_rate = sprintf("%5.2f", $mbytes_per_sec)." MB/s";

            msg(1, "INFO", $utc_start." finished -> transferred [".$data_rate."]");

            $cmd = "mv ".$utc_start."/obs.finished ".$utc_start."/obs.transferred";
            msg(2, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(3, "INFO", "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              msg(0, "ERROR", $cmd." failed: ".$response);
              $quit_daemon = 1;
            }

            my $user = "dada";
            my $host = "172.17.228.204";
            my $rval;
    
            $cmd = "touch ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/obs.finished.".$proc_id.".".$cfg{"NUM_BP"};
            msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
            ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
            msg(2, "INFO", "main: ".$result." ".$rval." ".$response);

            $cmd = "rm -rf ".$utc_start."/BEAM_???";
            msg(2, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(3, "INFO", "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              msg(0, "ERROR", $cmd." failed: ".$response);
              $quit_daemon = 1;
            }
          }
        } 
      }
    }
      
    my $counter = 10;
    while (!$quit_daemon && $counter > 0)
    {
      sleep(1);
      $counter --;
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($sys_log_sock);

  exit (0);
}

#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$)
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_fb_mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^rsync -a --stats";
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO", "controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");

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

