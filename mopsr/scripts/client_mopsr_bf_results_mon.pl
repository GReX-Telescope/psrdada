#!/usr/bin/env perl

###############################################################################
#
# client_mopsr_results_monitor.pl 
#

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
  print "Usage: ".basename($0)." CHAN_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $chan_id : shared;
our $chan_tag : shared;
our $log_host;
our $log_port;
our $log_sock;
our $nchan_out;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
$chan_id = -1;
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_BF_SYS_LOG_PORT"};
$log_sock = 0;
$nchan_out = 32;


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$chan_id  = $ARGV[0];
$chan_tag = "CH".sprintf("%02d",$chan_id);

# ensure that our chan_id is valid 
if (($chan_id >= 0) &&  ($chan_id < $cfg{"NCHAN"}))
{
  # and matches configured hostname
  if ($cfg{"RECV_".$chan_id} ne Dada::getHostMachineName())
  {
    print STDERR "RECV_".$chan_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "chan_id was not a valid integer between 0 and ".($cfg{"NCHAN"}-1)."\n";
  usage();
  exit(1);
}


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$chan_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$chan_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$chan_id.".pid";
  my $use_nfs = 1;

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) 
  {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }
  logMsg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my $client_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$chan_tag;

  my $last_obs = "";
  my $last_plot = 0;
  my $should_plot_cc = "";
  my $should_plot_ac = "";

  my ($cmd, $result, $response, $obs, $range, $file, $file_list);
  my $sleep_total = 5;
  my $sleep_count;

  my @ac_files;
  my @cc_files;
  my %to_send;
  my $ac_file;
  my $cc_file;

  my ($ref_ant, $ref_ant_number);

  if (! -d $client_dir)
  {
    $cmd = "mkdir -p ".$client_dir;
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      logMsg(0, "WARN", "main: ".$cmd." failed: ".$response);
    }
  }

  while (!$quit_daemon)
  {
    ($result, $response) = sumCorrFiles($client_dir, "ac");
    if ($result ne "ok")
    {
      logMsg(0, "WARN", "failed to sum ac files: ".$response);
    }

    ($result, $response) = sumCorrFiles($client_dir, "cc");
    if ($result ne "ok")
    {
      logMsg(0, "WARN", "failed to sum cc files: ".$response);
    }

    $sleep_count = 0;
    while (!$quit_daemon && ($sleep_count < $sleep_total))
    {
      sleep(1);
      $sleep_count++;
    }
  }

  # Rejoin our daemon control thread
  logMsg(2, "INFO", "joining control thread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

  exit (0);
}


sub sumCorrFiles($$)
{
  my ($dir, $ext) = @_;

  my $sum = $ext.".sum";
  my $first_time = 0;
  my $last_sum = "";
  my ($cmd, $result, $rval, $response, $obs, $range, $file, $file_fscr, $nant);
  my @files;

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"};
  my $server_host = $cfg{"SERVER_HOST"};
  my $server_user = $cfg{"USER"};

  # look for any corr files
  $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name '*.".$ext."' -printf '%f\n'| sort -n";

  logMsg(2, "INFO", "sumCorrFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    logMsg(0, "WARN", "find list of dump files failed: ".$response);
    return ("fail", "");
  }
  elsif ($response eq "")
  {
    logMsg(2, "INFO", "sumCorrFiles: no .".$ext." files found");
  }
  else
  {
    @files = split(/\n/, $response);
    logMsg(2, "INFO", "main: found ".($#files+1)." .".$ext." files");
    if ($#files >= 0)
    {
      foreach $file (@files)
      {
        # determine UTC for this file
        ($obs, $range) = split(/_/, $file, 2);

        # determine NANT
        $cmd = "grep ^NANT ".$dir."/".$obs."/obs.header | awk '{print \$2}'";
        logMsg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(3, "INFO", "sumCorrFiles: ".$result." ".$response);
        if ($result ne "ok")
        {
          return ("fail", "could not determine NANT from obs.header");
        }
        $nant = $response;

        # create a scrunched copy of this file
        $file_fscr = $file.".fscr";
        if ($ext eq "ac")
        {
          $cmd = "mopsr_corr_fscr -a -F ".$nchan_out." ".$nant." ".$dir."/".$obs."/".$file." ".$dir."/".$obs."/".$file_fscr;
        }
        else
        {
          $cmd = "mopsr_corr_fscr -F ".$nchan_out." ".$nant." ".$dir."/".$obs."/".$file." ".$dir."/".$obs."/".$file_fscr;
        }
        logMsg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(3, "INFO", "sumCorrFiles: ".$result." ".$response);

        if (!(-f $dir."/".$obs."/".$sum))
        {
          $first_time = 1;
          $cmd = "rsync -a ".$dir."/".$obs."/obs\.* ".
          $server_user."\@".$server_host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$obs."/".$chan_tag."/";
          logMsg(2, "INFO", "sumCorrFiles: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          logMsg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            logMsg(0, "WARN", $cmd." failed: ".$response);
            next;
          }
        }

        # copy the new new Fscrunch archive to the server
        $cmd = "rsync -a ".$dir."/".$obs."/".$file_fscr." ".
               $server_user."\@".$server_host.":".$cfg{"SERVER_RESULTS_DIR"}."/".$obs."/".$chan_tag."/".$file;
        logMsg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          logMsg(0, "WARN", $cmd." failed: ".$response);
          next;
        }

        $cmd = "rm -f  ".$dir."/".$obs."/".$file." ".$dir."/".$obs."/".$file_fscr;
        logMsg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        logMsg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          logMsg(0, "WARN", "summCorrFiles: ".$cmd." failed: ".$response);
        }
      }
    }
  }
  return ("ok", "");
}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$)
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $chan_id, $time, "sys", $type, "results mon", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$chan_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  logMsg(2, "INFO", "controlThread: exiting");

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

