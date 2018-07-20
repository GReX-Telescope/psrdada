#!/usr/bin/env perl

# 
# MOPSR AQ Dump Transfer Script
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
our %cfg : shared;
our $log_host;
our $sys_log_port;
our $sys_log_sock;
our $sys_log_file;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$sys_log_sock = 0;
$sys_log_file = "";


#
# Local Variable Declarations
#
my $log_file = "";
my $pid_file = "";
my $control_thread = 0;
my $prev_header = "";

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
  if ($cfg{"PWC_".$pwc_id} ne Dada::getHostMachineName())
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
  my ($cmd, $result, $response, $rval, $utc_start, $i);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $pid_file     = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

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
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  msg(1,"INFO", "STARTING SCRIPT");

  # This thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

  my $aq_dir = $cfg{"CLIENT_DUMP_DIR"}."/".$cfg{"PWC_PFB_ID_".$pwc_id};
  my $user = "mpsr";
  my $host = "mpsr-bf08";

  # Main Loop
  while (!$quit_daemon) 
  {
    $cmd = "find ".$aq_dir." -mindepth 1 -maxdepth 1 -type f -name '*.dada' -mmin +2 | awk -F/ '{print \$(NF)}'";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      my @files = split(/\n/, $response);

      for ($i=0; (!$quit_daemon && $i<=$#files); $i++)
      {
        my $file = $files[$i];

        my ($utc_start, $rest) = split(/_/, $file, 2);
        msg(2, "INFO", "main: utc_start=".$utc_start." rest=".$rest);

        $cmd = "mkdir -p /data/mopsr/voltage_dumps/".$utc_start."/".$cfg{"PWC_PFB_ID_".$pwc_id};
          
        msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
        msg(2, "INFO", "main: ".$result." ".$rval." ".$response);

        if (($result eq "ok") && ($rval == 0))
        { 
          $cmd = "rsync -a --stats --bwlimit=1024 --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
                 $aq_dir."/".$file." upload\@mpsr-bf08::voltages/".$utc_start."/".$cfg{"PWC_PFB_ID_".$pwc_id}."/";

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
              $cmd = "mv ".$aq_dir."/".$utc_start."/aqdsp.finished ".$aq_dir."/".$utc_start."/aqdsp.transfer_failed";
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

            msg(1, "INFO", $file." finished -> transferred [".$data_rate."]");

            $cmd = "rm -f ".$aq_dir."/".$file;
            msg(2, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(3, "INFO", "main: ".$result." ".$response);
          }
        }
        else
        {
          msg(1, "WARN", "failed to create remote directory");
        }
      }
    }

    my $to_sleep = 30;
    while (!$quit_daemon && $to_sleep > 0)
    {
      $to_sleep--;
      sleep(1);
    }
  }

  msg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);

  exit(0);
}

sub controlThread($) 
{
  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  $cmd = "^rsync -a --stats";
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO", "controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) 
  {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");
}



#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$) 
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl) {

    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$pwc_id), $time, "sys", $type, "dump_xfer", $msg);
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

