#!/usr/bin/env perl

###############################################################################
#
# This script transfers data from a directory on the pwc, to a directory on the
# nexus machine

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;        
use Mopsr;
use strict;
use warnings;
use IO::Socket;
use File::Basename;
use threads;
use threads::shared;
use Dada;

#
# Function prototypes
#
sub usage();
sub msg($$$);
sub processArchive($$$$);


#
# Global Variables
# 
our $dl : shared;
our $daemon_name : shared;
our %cfg : shared;
our $quit_daemon : shared;
our $localhost : shared;
our $pwc_id : shared;
our $log_host;
our $log_port;
our $log_sock;


#
# Initialize module variables
#
%cfg = Mopsr::getConfig();
$dl = 1;
$localhost = Dada::getHostMachineName();
$pwc_id = -1;
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;
$daemon_name = Dada::daemonBaseName($0);
$quit_daemon = 0;

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
  if (!(($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName()) || ($cfg{"PWC_".$pwc_id} eq "localhost")))
  {
    print STDERR "PWC_ID did not match configured hostname [".$cfg{"PWC_".$pwc_id}."]\n";
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

# Autoflush STDOUT
$| = 1;

{

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";
  my $pfb_id         = $cfg{"PWC_PFB_ID_".$pwc_id};
  my $results_dir    = $cfg{"CLIENT_RESULTS_DIR"}."/".$pfb_id;

  $log_host = $cfg{"SERVER_HOST"};
  $log_port = $cfg{"SERVER_SYS_LOG_PORT"};

  my $control_thread = 0;
  my @paths = ();
  my @lines = ();
  my $line;
  my $found_something;
  my $i=0;
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $sleep_counter = 0;
  my $archive = "";
  my $n = 0;
  my ($utc_start, $ant_id, $file);

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Open a connection to the nexus logging port
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  msg(0,"INFO", "STARTING SCRIPT");

  msg(2, "INFO", "main: mkdirRecursive(".$results_dir.", 0755)");
  ($result, $response) = Dada::mkdirRecursive($results_dir, 0755);
  msg(2, "INFO", "main: mkdirRecursive: ".$result." ".$response);

  # start the control thread
  msg(2, "INFO", "starting controlThread(".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $pid_file);

  # Loop until daemon control thread asks us to quit
  while (!$quit_daemon) 
  {
    # get list of archives to process in the results dir
    $cmd = "find ".$results_dir." -name '20*.ar' | sort -n";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    $found_something = 0;

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

        # since archives will be of the form . / UTC_START / ANT_ID / FILE, grab last 3
        @paths = split(/\//, $line);
        $n = $#paths;
      
        $utc_start = $paths[$n-2];
        $ant_id    = $paths[$n-1];
        $file      = $paths[$n];

        msg(2, "INFO", "main: processArchive(".$pfb_id.", ".$utc_start.", ".$ant_id.", ".$file.")");
        ($result, $response) = processArchive($pfb_id, $utc_start, $ant_id, $file);
        msg(3, "INFO", "main: processArchive ".$result." ".$response);
      }
    }

    # If we didn't find any archives, sleep.
    $sleep_counter = 5;
    while ((!$quit_daemon) && (!$found_something) && ($sleep_counter)) {
      sleep(1);
      $sleep_counter--;
    }

  }

  # Rejoin our daemon control thread
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  Dada::nexusLogClose($log_sock);

}

exit (0);

###############################################################################
#
# package functions
# 

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
}

###############################################################################
#
# Process an archive, sending it to the server
#
sub processArchive($$$$)
{
  my ($pfb_id, $utc_start, $ant_id, $file) = @_;

  msg(2, "INFO", "processArchive(".$pfb_id.", ".$utc_start.", ".$ant_id.", ".$file.")");

  my $server = $cfg{"SERVER_HOST"};
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $local_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$pfb_id."/".$utc_start."/".$ant_id;
  my $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$pfb_id."_".$ant_id;
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

  $local_file = $cfg{"CLIENT_RESULTS_DIR"}."/".$pfb_id."/".$utc_start."/".$ant_id."/".$file;
  $remote_file = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$pfb_id."_".$ant_id."/".$file;
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

  logMsg(2, "INFO", "createRemoteDir: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  logMsg(2, "INFO", "createRemoteDir: ".$result." ".$rval." ".$response);
  
  if (($result eq "ok") && ($rval == 0))
  {
    logMsg(2, "INFO", "createRemoteDir: remote directory created");
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


###############################################################################
#
# monitor for quit requests
#

sub controlThread($) 
{
  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my $cmd = "";
  my $result = "";
  my $response = "";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) {
    sleep(1);
  }

  $quit_daemon = 1;

  msg(2, "INFO", "controlThread: unlinking PID file");
  if (-f $pid_file)
  {
    unlink($pid_file);
  }
  
  return 0;
} 


###############################################################################
#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$)
{
  my ($level, $type, $msg) = @_;
  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock )
    {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock)
    {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $type, "arch mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


###############################################################################
#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # Tell threads to try and quit
  $quit_daemon = 1;
  sleep(3);
  
  if ($log_sock) {
    close($log_sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}


###############################################################################
#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}

END { }

1;  # return value from file
