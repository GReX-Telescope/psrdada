#!/usr/bin/env perl

###############################################################################
#
# This script transfers data from a directory on the pwc, to a directory on the
# nexus machine

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;        
use Caspsr;
use strict;
use warnings;
use IO::Socket;
use File::Basename;
use threads;
use threads::shared;
use Dada;

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Function Prototypes
#
sub good();
sub processArchive($$);
sub msg($$$);


#
# Global Variables
# 
our $dl : shared;
our $daemon_name : shared;
our %cfg : shared;
our $hostname : shared;
our $quit_daemon : shared;
our $pwc_id : shared;
our $log_host;
our $log_port;
our $log_sock;


#
# Initialize module variables
#
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Caspsr::getConfig();
$hostname = Dada::getHostMachineName();
$quit_daemon = 0;
$pwc_id = $ARGV[0];
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;

# ensure that our pwc_id is valid 
if (!Dada::checkPWCID($pwc_id, %cfg))
{
  usage();
  exit(1);
}

# Autoflush STDOUT
$| = 1;

{

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";
  my $archive_dir    = $cfg{"CLIENT_ARCHIVE_DIR"};   # hi res archive storage
  my $results_dir    = $cfg{"CLIENT_RESULTS_DIR"};   # hi res archive output

  $log_host = $cfg{"SERVER_HOST"};
  $log_port = $cfg{"SERVER_SYS_LOG_PORT"};

  my $control_thread = 0;
  my @lines = ();
  my $line;
  my $found_something;
  my $i=0;
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $sleep_counter = 0;

  # sanity check on whether the module is good to go
  ($result, $response) = good();
  if ($result ne "ok") {
    print STDERR $response."\n";
    exit 1;
  }

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

  msg(0, "INFO", "STARTING SCRIPT");

  # start the control thread
  msg(2, "INFO", "starting controlThread(".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $pid_file);

  # Change to the dspsr output directory
  chdir $results_dir;

  # Loop until daemon control thread asks us to quit
  while (!($quit_daemon)) {

    @lines = ();
    $found_something = 0;

    # Look for dspsr archives (both normal and pulse_ archives)
    $cmd = "find . -name \"*.ar\" | sort";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if ($result ne "ok") {
      msg(2, "WARN", "find command ".$cmd." failed ".$response);

    } else {

      @lines = split(/\n/,$response);
      for ($i=0; (($i<=$#lines) && (!$quit_daemon)); $i++) {

        $line = $lines[$i];
        $found_something = 1;

        # strip the leading ./ if it exists
        $line =~ s/^.\///;

        if (!($line =~ /pulse_/)) {
          msg(1, "INFO", "Processing archive ".$line);
        }

        ($result, $response) = processArchive($line, $archive_dir);
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

  exit 0;

}

###############################################################################
#
# Process an archive, sending it to srv0 via rsync
#
sub processArchive($$) {

  my ($path, $archive_dir) = @_;

  msg(2, "INFO", "processArchive(".$path.", ".$archive_dir.")");

  my $localhost = Dada::getHostMachineName();
  my $server = $cfg{"SERVER_HOST"};
  my $server_base_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $dir = "";
  my $file = "";
  my $source = "";
  my $local_file = "";
  my $remote_file = "";
  my $i = 0;
  
  # get the directory the file resides in. This may be 2 subdirs deep
  my @parts = split(/\//, $path);

  for ($i=0; $i<$#parts; $i++) {
    # skip the leading ./
    if ($parts[$i] ne ".") {
      $dir .= $parts[$i]."/";
    }
  }

  # remove the trailing slash
  $dir =~ s/\/$//;
  $file = $parts[$#parts];
  $local_file = $dir."/".$file;

  $cmd = "vap -n -c name ".$local_file." | awk '{print \$2}'";
  Dada::logMsg(2, $dl, "processArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "processArchive: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "processArchive: failed to determine source name: ".$response);
    $source = "UNKNOWN";
  } else {
    $source = $response;
    chomp $source;
  }
  $source =~ s/^[JB]//;
  $remote_file = $server_base_dir."/".$dir."/".$source."/".$localhost."/".$file;

  msg(2, "INFO", "processArchive: ".$local_file." -> ".$remote_file);

  # We never decimate pulse archives, or transfer them to the server
  if ($file =~ m/^pulse_/) {

    # do nothing

  } else {

    msg(2, "INFO", "processArchive: rsyncCopy(".$local_file.", dada, ".$server.", ".$remote_file.")");
    ($result, $response) = rsyncCopy($local_file, "dada", $server, $remote_file);
    msg(2, "INFO", "processArchive: rsyncCopy() ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "procesArchive: nfsCopy() failed: ".$response);
    }

    msg(2, "INFO", "processArchive: moving to archive ".$local_file);
    $cmd = "mv ".$local_file." ".$archive_dir."/".$dir."/";
    msg(2, "INFO", "processArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "processArchive: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "processArchive: failed to move ".$local_file." to archive: ".$response);
      return ("ok", "");
    }

  }

  return ("ok", "");
}


###############################################################################
#
# Copy the file to the server via rsync
#
sub rsyncCopy($$$$) 
{
  my ($file, $user, $server, $dest_file) = @_;

  my $result = "";
  my $response = "";
  my $cmd = "";

  $cmd = "rsync -a ./".$file." ".$user."@".$server.":".$dest_file;
  msg(2, "INFO", "rsyncCopy: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "rsyncCopy: ".$result." ".$response);

  return ($result, $response);

}


###############################################################################
#
# monitor for quit requests
#
sub controlThread($) {

  msg(2, "INFO", "controlThread: starting");

  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread(".$pid_file.")");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";
  
  # poll for the existence of the control file
  while ((!$quit_daemon) && (! -f $host_quit_file) && (! -f $pwc_quit_file)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  msg(2, "INFO", "controlThread: unlinking PID file");
  if (-f $pid_file) {
    unlink($pid_file);
  }
  
  return 0;
} 


###############################################################################
#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $class, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $hostname, $time, "sys", $class, "arch mngr", $msg);
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


###############################################################################
#
# Test to ensure all module variables are set before main
#
sub good() {

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";
  
  # check the quit file does not exist on startup
  if ((-f $host_quit_file) || (-f $pwc_quit_file)) {
    return ("fail", "Error: quit file existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  if ( $daemon_name eq "") {
    return ("fail", "Error: a package variable missing [daemon_name]");
  }

  if (! -d $cfg{"CLIENT_ARCHIVE_DIR"} ) {
    return("fail", "Error: archive dir ".$cfg{"CLIENT_ARCHIVE_DIR"}." did not exist");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}



END { }

1;  # return value from file
