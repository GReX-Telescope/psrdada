#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Disk Cleaner for CASPSR instrument, deletes archives from PWC machines after
# the observation is marked deleted and observation is 6 months old
#

use strict;
use warnings;
use File::Basename;
use Time::Local;
use threads;
use threads::shared;
use Net::hostent;
use Time::Local;
use Dada;
use Caspsr;

#
# Global variables
#
our $dl : shared = 1;
our $daemon_name : shared = Dada::daemonBaseName($0);
our %cfg : shared = Caspsr::getConfig();
our $quit_daemon : shared = 0;
our $log_host = 0;
our $log_port = 0;
our $log_sock = 0;

#
# Function declarations
#
sub good($);
sub msg($$$);
sub findCompletedObs();
sub deleteCompletedObs($);


###############################################################################
#
# Main
# 

my $log_file      = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
my $pid_file      = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".pid";
my $quit_file     = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";

$log_host         = $cfg{"SERVER_HOST"};
$log_port         = $cfg{"SERVER_SYS_LOG_PORT"};

my $control_thread = 0;
my $result = "";
my $response = "";
my $cmd = "";
my $obs = "";

# sanity check on whether the module is good to go
($result, $response) = good($quit_file);
if ($result ne "ok") {
  print STDERR $response."\n";
  return 1;
}

# install signal handles
$SIG{INT}  = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# become a daemon
Dada::daemonize($log_file, $pid_file);

# Open a connection to the server_sys_monitor.pl script
$log_sock = Dada::nexusLogOpen($log_host, $log_port);
if (!$log_sock) {
  print STDERR "Could not open log port: ".$log_host.":".$log_port."\n";
}

msg(0, "INFO", "STARTING SCRIPT");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

# Main Loop
while (!($quit_daemon)) 
{
  msg(2, "INFO", "main: findCompletedObs()");
  ($result, $response, $obs) = findCompletedObs();
  msg(2, "INFO", "main: findCompletedObs() ".$result." ".$response." ".$obs);

  if (($result eq "ok") && ($obs ne "none")) 
  {
    msg(2, "INFO", "main: deleteCompletedObs(".$obs.")");
    ($result, $response) = deleteCompletedObs($obs);
    msg(2, "INFO", "main: deleteCompletedObs() ".$result." ".$response);

    if ($result ne "ok") 
    {
      msg(0, "ERROR", "Failed to delete ".$obs.": ".$response);
      $quit_daemon = 1;
    } 
    else 
    {
      msg(1, "INFO", $obs.": transferred -> deleted");
    }
  } 

  my $counter = 120;
  msg(2, "INFO", "main: sleeping ".($counter*5)." seconds");
  while ((!$quit_daemon) && ($counter > 0) && ($obs eq "none")) 
  {
    sleep(1);
    $counter--;
  }

}

$control_thread->join();

msg(0, "INFO", "STOPPING SCRIPT");
Dada::nexusLogClose($log_sock);

exit 0;


#
# Find an obs that has been marked as deleted and greater than 
# 6 months old
#
sub findCompletedObs() 
{
  msg(2, "INFO", "findCompletedObs()");

  my $archives_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $found_obs = 0;
  my $result = "";
  my $response = "";
  my $obs = "none";
  my $cmd = "";
  my $i = 0;
  my $o = "";
  
  my @deleted = ();    # obs that have been marked obs.deleted (by transfer_manager)

  $cmd = "find ".$archives_dir." -mindepth 2 -maxdepth 2 -name 'obs.deleted' ".
         "-printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  msg(2, "INFO", "findCompletedObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "findCompletedObs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "findCompletedObs: find failed: ".$response);
    return ("fail", "find obs.deleted failed", "none");
  }

  @deleted = split(/\n/, $response);

  msg(2, "INFO", "findCompletedObs: ".($#deleted+1)." observations to consider");

  my $curr_time = time;

  for ($i=0; ((!$quit_daemon) && (!$found_obs) && ($i<=$#deleted)); $i++) {

    $o = $deleted[$i];

    # check the UTC_START time to ensure this obs is > 6 months old
    my @t = split(/-|:/,$o);
    my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

    Dada::logMsg(2, $dl, "findCompletedObs: testing ".$o." curr=".$curr_time.", unix=".$unixtime);
    # if UTC_START is less than 6 months [182 days] old, dont delete it
    if ($unixtime + (182*24*60*60) > $curr_time) {
      Dada::logMsg(2, $dl, "findCompletedObs: Skipping ".$o.", less than 6 months old");
      next;
    }

    $obs = $o;
    $found_obs = 1;

  }

  $result = "ok";
  $response = "";

  msg(2, "INFO", "findCompletedObs ".$result.", ".$response.", ".$obs);

  return ($result, $response, $obs);
}

#
# Delete the specified obs
#
sub deleteCompletedObs($) 
{
  my ($obs) = @_;

  msg(2, "INFO", "deleteCompletedObs(".$obs.")");

  my $archives_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $results_dir  = $cfg{"CLIENT_RESULTS_DIR"};
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $path = $archives_dir."/".$obs;

  msg(2, "INFO", "Deleting archived files in ".$path);

  if (-f $path."/slow_rm.ls") {
    $cmd = "rm -f ".$path."/slow_rm.ls";
    msg(2, "INFO", "deleteCompletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
      return ("fail", "failed to delete ".$path."/slow_rm.ls");
    }
  }

  $cmd = "find ".$path." -name '*.ar' -o -name 'obs.start' -o -name 'obs.deleted' > ".$path."/slow_rm.ls";
  msg(2, "INFO", "deleteCompletedObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "deleteCompletedObs: find command failed: ".$response);
    return ("fail", "find command failed");
  }

  $cmd = "slow_rm -r 256 -M ".$path."/slow_rm.ls";

  msg(2, "INFO", $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
    return ("fail", "failed to delete files for ".$obs);
  }

  if (-f $path."/slow_rm.ls") 
  {
    $cmd = "rm -f ".$path."/slow_rm.ls";
    msg(2, "INFO", "deleteCompletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
      return ("fail", "failed to delete ".$path."/slow_rm.ls");
    }
  }

  # delete the archives dir
  $cmd = "rmdir ".$path;
  msg(2, "INFO", "deleteCompletedObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
    return ("fail", "failed to delete ".$path);
  }

  $path = $results_dir."/".$obs;

  if (-d $path)
  {
    # delete the results dir
    $cmd = "rm -f ".$path."/*";
    msg(2, "INFO", "deleteCompletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
      return ("fail", "failed to delete results_dir files");
    }

    $cmd = "rmdir ".$path;
    msg(2, "INFO", "deleteCompletedObs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "deleteCompletedObs: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "deleteCompletedObs: ".$cmd." failed: ".$response);
      return ("fail", "failed to delete ".$path);
    }
  } 
  else
  {
    msg(0, "WARN", "deleteCompletedObs: ".$path." did not exist");
  }
    
  $result = "ok";
  $response = "";

  return ($result, $response);
}

sub controlThread($$) {
  
  my ($quit_file, $pid_file) = @_;
  
  msg(2, "INFO", "controlThread : starting");

  my $cmd = "";
  my $result = "";
  my $response = "";
  
  while ((!$quit_daemon) && (!(-f $quit_file))) {
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


#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "cleaner", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


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

#
# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  if ($daemon_name eq "") {
    return ("fail", "Error: a package variable missing [daemon_name]");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
    
} 
