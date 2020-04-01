#!/usr/bin/env perl

###############################################################################
#
# client_mopsr_bf_tbs_results.pl 
#
# process output of tied-array beams on the BFs

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
  print "Usage: ".basename($0)." BF_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $bf_id : shared;
our $bf_tag : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bf");
$bf_id = -1;
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_BF_SYS_LOG_PORT"};
$log_sock = 0;

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$bf_id  = $ARGV[0];
$bf_tag = "BF".sprintf("%02d",$bf_id);

# ensure that our chan_id is valid 
if (($bf_id >= 0) &&  ($bf_id < $cfg{"NUM_BF"}))
{
  # and matches configured hostname
  if ($cfg{"BF_".$bf_id} ne Dada::getHostMachineName())
  {
    print STDERR "BF_".$bf_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "bf_id was not a valid integer between 0 and ".($cfg{"NUM_BF"}-1)."\n";
  usage();
  exit(1);
}


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$bf_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bf_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bf_id.".pid";
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
  msg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my $results_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bf_tag;

  my ($cmd, $result, $response, $obs, $file, $found_something);
  my ($line, @lines, $utc_start, $source, $i, @paths, $n, $last_obs);

  my $sleep_total = 5;
  my $sleep_count;
  my %to_send;
  my $last_obs = "";

  if (! -d $results_dir)
  {
    $cmd = "mkdir -p ".$results_dir;
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      msg(0, "WARN", "main: ".$cmd." failed: ".$response);
    }
  }

  while (!$quit_daemon)
  {
    # get list of archives to process in the results dir
    $cmd = "find ".$results_dir." -mindepth 3 -maxdepth 3 ".
           "-name '????-??-??-??:??:??.ar' ".
           "-o -name '????-??-??-??:??:??.sf' ".
           "-o -name 'pulse_*.ar' ".
           "| sort -n";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    my $found_something = 0;

    if ($result ne "ok")
    {
      msg(2, "WARN", "find command ".$cmd." failed ".$response);
    }
    elsif ($response eq "")
    {
      msg(2, "INFO", "main: did not find any archives to process");
    }
    else
    {
      @lines = split(/\n/,$response);
      for ($i=0; (($i<=$#lines) && (!$quit_daemon)); $i++)
      {
        $line = $lines[$i];
        $found_something = 1;

        # since archives will be of the form . / UTC_START / SOURCE / FILE, grab last 3
        @paths = split(/\//, $line);
        $n = $#paths;

        $utc_start = $paths[$n-2];
        $source    = $paths[$n-1];
        $file      = $paths[$n];

        if (!($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/))
        {
          msg(0, "WARN", "UTC_START [".$utc_start."] did not match expected form ????-??-??-??:??:??");
          sleep(1);
          next;
        }

        if ($utc_start ne $last_obs)
        {
          msg(0, "INFO", "processing archives from ".$utc_start);
          $last_obs = $utc_start;
        }

        msg(2, "INFO", "main: processArchive(".$bf_tag.", ".$utc_start.", ".$source.", ".$file.")");
        ($result, $response) = processArchive($bf_tag, $utc_start, $source, $file);
        msg(3, "INFO", "main: processArchive ".$result." ".$response);
      }
    }

    # If we didn't find any archives, sleep.
    $sleep_count = 0;
    while (!$quit_daemon && ($sleep_count < $sleep_total))
    {
      sleep(1);
      $sleep_count++;
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

  exit (0);
}


###############################################################################
#
# Process an archive, sending it to the server
#
sub processArchive($$$$)
{
  my ($bf_tag, $utc_start, $source, $file) = @_;

  msg(2, "INFO", "processArchive(".$bf_tag.", ".$utc_start.", ".$file.")");

  my $server = $cfg{"SERVER_HOST"};
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $local_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bf_tag."/".$utc_start."/".$source;
  my $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$source."/".$bf_tag;
  my $local_file = "";
  my $remote_file = "";
  my $i = 0;

  # check for the obs.header file in the directory
  if (-f $local_dir."/obs.header")
  {
    # create the remote directory
    ($result, $response) = createRemoteDir ($remote_dir);
    if ($result ne "ok")
    {
      msg(0, "WARN", "failed to create remote dir [".$remote_dir."] ".$response);
    }

    # copy the obs.header to the remote directory
    $local_file = $local_dir."/obs.header";
    $remote_file = $remote_dir."/obs.header";
    msg(2, "INFO", "processArchive: ".$local_file." -> ".$remote_file);

    msg(2, "INFO", "processArchive: sendToServer(".$local_file.", dada, ".$server.", ".$remote_file.")");
    ($result, $response) = sendToServer($local_file, "dada", $server, $remote_file);
    msg(2, "INFO", "processArchive: sendToServer() ".$result." ".$response);

    unlink ($local_file);
  }

  $local_file = $local_dir."/".$file;
  $remote_file = $remote_dir."/".$file;
  msg(2, "INFO", "processArchive: ".$local_file." -> ".$remote_file);

  msg(2, "INFO", "processArchive: sendToServer(".$local_file.", dada, ".$server.", ".$remote_file.")");
  ($result, $response) = sendToServer($local_file, "dada", $server, $remote_file);
  msg(2, "INFO", "processArchive: sendToServer() ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "WARN", "processArchive: sendToServer() failed: ".$response);
  }

  # unlink the file for now, since we have full res fil on server
  unlink ($local_file);

  return ("ok", "");
}

###############################################################################
#
# create remote directory
#
sub createRemoteDir($)
{
  my ($remote_dir) = @_;

  my $user = $cfg{"USER"};
  my $host = $cfg{"SERVER_HOST"};
  my $cmd = "mkdir -m 2755 -p ".$remote_dir;
  my ($result, $rval, $response);

  msg(2, "INFO", "createRemoteDir: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  msg(2, "INFO", "createRemoteDir: ".$result." ".$rval." ".$response);

  if (($result eq "ok") && ($rval == 0))
  {
    msg(2, "INFO", "createRemoteDir: remote directory created");
    return ("ok", "");
  }
  else
  {
    return ("fail", $response);
  }
}

###############################################################################
#
# Copy the file to the server
#
sub sendToServer($$$$) 
{
  my ($file, $user, $server, $dest_file) = @_;

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $use_cp = 0;

  if ($use_cp)
  {
    # ensure its automounted
    $cmd = "ls -1d ".$cfg{"SERVER_NFS_RESULTS_DIR"};
    msg(2, "INFO", "sendToServer: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "sendToServer: ".$result." ".$response);

    $cmd = "cp ".$file." ".$dest_file;
  }
  else
  {
    $cmd = "rsync -a ".$file." ".$user."@".$server.":".$dest_file;
  }

  msg(2, "INFO", "sendToServer: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "sendToServer: ".$result." ".$response);

  return ($result, $response);
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
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, sprintf("%02d",$bf_id), $time, "sys", $type, "bp_arch_mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bf_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

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

