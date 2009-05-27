#!/usr/bin/env perl

###############################################################################
#
# server_bpsr_multibob_manager.pl
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Bpsr;           # BPSR Module 
use strict;         # strict mode (like -Wall)
use threads;
use threads::shared;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada->preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DEBUG_LEVEL  => 1;
use constant DL           => 1;
use constant PIDFILE      => "bpsr_multibob_manager.pid";
use constant LOGFILE      => "bpsr_multibob_manager.log";
use constant QUITFILE     => "bpsr_multibob_manager.quit";
use constant MULTIBOB_BIN => "multibob_server";


#
# Global Variables
#
our %cfg   = Bpsr->getBpsrConfig();
our $error = $cfg{"STATUS_DIR"}."/bpsr_multibob_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/bpsr_multibob_manager.warn";
our $quit_daemon : shared  = 0;


# Autoflush output
$| = 1;


# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


#
# Local Varaibles
#
my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

my $daemon_control_thread = 0;
my $multibob_thread = 0;
my $multibob_plot_thread = 0;

my $port  = $cfg{"IBOB_MANAGER_PORT"};
my $npwc  = $cfg{"NUM_PWC"};

my $i;
my $result;
my $response;
my $command = "config";

#
# Main
#

Dada->daemonize($logfile, $pidfile);

Dada->logMsg(0, DL, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

# Start the multibob_server thread
$multibob_thread = threads->new(\&multibobThread);

# Start the multibob plotting thread
$multibob_plot_thread = threads->new(\&multibobPlotThread);

# Wait for threads to return
while (!$quit_daemon) {
  sleep(2);
}

# rejoin threads
$daemon_control_thread->join();
$multibob_thread->join();
$multibob_plot_thread->join();

Dada->logMsg(0, DL, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#


#
# Runs the multibob_server on localhost. If the server fails, then 
# try to relaunch it
#
sub multibobThread() {

  my $runtime_dir  = $cfg{"SERVER_STATS_DIR"};
  my $port         = $cfg{"IBOB_MANAGER_PORT"};
  my $multibob_bin = MULTIBOB_BIN;

  my $rval  = 0;
  my $cmd   = "";

  checkMultibobServer($port, $multibob_bin);

  $cmd =  "cd ".$runtime_dir."; echo '' | ".$multibob_bin." -n ".$npwc." -p ".$port." 2>&1";
  $cmd .= " | server_bpsr_server_logger.pl";

  my $multibob_config_thread = 0;

  while (!$quit_daemon) {

    $multibob_config_thread = threads->new(\&configureMultibobServerWrapper);
    $multibob_config_thread->detach();

    # This command should "hang" until the multibob_server command has terminated
    Dada->logMsg(1, DL, "multiBobThread: ".$cmd);
    system($cmd);
    Dada->logMsg(1, DL, "multiBobThread: cmd returned");

    if (!$quit_daemon) {
      Dada->logMsgWarn($warn, "multibob_server exited unexpectedly, re-launching");
    }

    sleep(1);

  }

}

#
# Configure the mulibob_server
#
sub configureMultibobServerWrapper() 
{

  Dada->logMsg(1, DL, "configureMultibobServer: configuring multibob_server");

  my ($result, $response) = Bpsr->configureMultibobServer();

  if ($result ne "ok") {
    Dada->logMsgWarn($error, "configureMultibobServer: failed ".$response);

  } else {
    Dada->logMsg(1, DL, "configureMultibobServer: done");
  }
}

# 
# Monitors the /nfs/results/bpsr/stats directory creating the PD Bandpass plots
# as requried
#
sub multibobPlotThread()
{

  Dada->logMsg(1, DL, "multibobPlotThread: thread starting");

  my $bindir    = Dada->getCurrentBinaryVersion();
  my $stats_dir = $cfg{"SERVER_STATS_DIR"};
  my @bramfiles = ();
  my $bramfile  = "";
  my $plot_cmd  = "";
  my $cmd       = "";
  my $result    = "";
  my $response  = "";

  my $i=0;
  my $j=0;

  while (!$quit_daemon) {
   
    Dada->logMsg(2, DL, "multibobPlotThread: looking for bramdump files in ".$stats_dir);
 
    # look for plot files
    opendir(DIR,$stats_dir);
    @bramfiles = sort grep { !/^\./ && /\.bramdump$/ } readdir(DIR);
    closedir DIR;

    if ($#bramfiles == -1) {
      Dada->logMsg(2, DL, "multibobPlotThread: no files, sleeping");
    }

    # plot any existing bramplot files
    for ($i=0; $i<=$#bramfiles; $i++) 
    {
      $bramfile = $stats_dir."/".$bramfiles[$i];

      $plot_cmd = $bindir."/bpsr_bramplot ".$bramfile;

      Dada->logMsg(2, $plot_cmd);

      ($result, $response) = Dada->mySystem($plot_cmd);

      if ($result ne "ok") {
        Dada->logMsgWarn($warn, "plot of ".$bramfile." failed ".$response);
      } else {
        Dada->logMsg(2, DL, "bpsr_bramplot ".$bramfile.": ".$response);
      }
      unlink($bramfile);
    }

    sleep(1);

    my $ibob = "";

    for ($i=0; $i < $cfg{"NUM_PWC"}; $i++ ) {
      $ibob = $cfg{"IBOB_DEST_".$i};
      removeOldPngs($stats_dir, $ibob, "1024x768");
      removeOldPngs($stats_dir, $ibob, "400x300");
      removeOldPngs($stats_dir, $ibob, "112x84");
    }

    sleep(1);

  }

  Dada->logMsg(2, DL, "multibobPlotThread: thread exiting");

}


#
# Polls for the "quitdaemons" file in the control dir
#
sub daemonControlThread() {

  Dada->logMsg(1, DL, "daemon_control: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  # poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    Dada->logMsg(3, DL, "daemon_control: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # signal threads to exit
  $quit_daemon = 1;

  my $localhost = Dada->getHostMachineName();

  # set the global variable to quit the daemon
  my $handle = Dada->connectToMachine($localhost, $cfg{"IBOB_MANAGER_PORT"},1);
  if (!$handle) {
    Dada->logMsgWarn($warn, "daemon_control: could not connect to ".MULTIBOB_BIN);

  } else {

    # ignore welcome message
    $response = <$handle>;
    Dada->logMsg(0, DL, "daemon_control: multibob_server <- quit");
    print $handle "quit\r\n";
    close($handle);
  }

  Dada->logMsg(2, DL, "daemon_control: Unlinking PID file ".$pidfile);
  unlink($pidfile);

  Dada->logMsg(2, DL, "daemon_control: exiting");

}

#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;

  Dada->logMsgWarn($warn, basename($0).": Received SIG".$sigName);

  $quit_daemon = 1;
  sleep(5);
 
  Dada->logMsgWarn($warn, basename($0).": Exiting");

  exit(1);

}

sub sigIgnore($) {
                                                                                                        
  my $sigName = shift;
  Dada->logMsgWarn($warn, basename($0)."  Received SIG".$sigName);
  Dada->logMsgWarn($warn, basename($0)." Ignoring");

}


sub removeOldPngs($$$) {

  my ($stats_dir, $ibob, $res) = @_;

  Dada->logMsg(2, DL, "removeOldPngs(".$stats_dir.", ".$ibob.", ".$res.")");

  # remove any existing plot files that are more than 10 seconds old
  my $cmd  = "find ".$stats_dir." -name '*".$ibob."_".$res.".png' -printf \"%T@ %f\\n\" | sort -n -r";
  my $result = `$cmd`;
  my @array = split(/\n/,$result);

  my $time = 0;
  my $file = "";
  my $line = "";


  # if there is more than one result in this category and its > 10 seconds old, delete it
  for ($i=1; $i<=$#array; $i++) {

    $line = $array[$i];
    ($time, $file) = split(/ /,$line,2);

    if (($time+20) < time)
    {
      $file = $stats_dir."/".$file;
      Dada->logMsg(2, DL, "removeOldPngs: unlink ".$file);
      unlink($file);
    }
  }

  Dada->logMsg(2, DL, "removeOldPngs: exiting");
}

sub checkMultibobServer($$) {

  my ($port, $process_name) = @_;

  Dada->logMsg(1, DL, "checkMultibobServer(".$port.", ".$process_name.")");

  my $localhost = Dada->getHostMachineName();

  # Check if the binary is running
  my $cmd = "ps aux | grep ".$command." | grep -v grep > /dev/null";
  Dada->logMsg(1, DL, "checkMultibobServer: ".$cmd);
  my $rval = system($cmd);
                                                                                                          
  if ($rval == 0) {
                                                                                                          
    Dada->logMsgWarn($warn, "checkMultibobServer: a multibob_server was running");
                                                                                                          
    # set the global variable to quit the daemon
    my $handle = Dada->connectToMachine($localhost, $port, 1);
    if (!$handle) {
      Dada->logMsgWarn($warn, "checkMultibobServer: could not connect to ".MULTIBOB_BIN);

    } else {

      # ignore welcome message
      $response = <$handle>;
                                                                                                          
      Dada->logMsg(0, DL, "checkMultibobServer: multibob <- close");
      ($result, $response) = Dada->sendTelnetCommand($handle, "close");
      Dada->logMsg(0, DL, "checkMultibobServer: multibob -> ".$result.":".$response);
                                                                                                          
      Dada->logMsg(0, DL, "checkMultibobServer: multibob_server <- quit");
      print $handle "quit\r\n";
      close($handle);
    }

    sleep(1);

    # try again to ensure it exited
    Dada->logMsg(1, DL, "checkMultibobServer: ".$cmd);
    $rval = system($cmd);

    if ($rval == 0) {

      Dada->logMsgWarn($warn, "checkMultibobServer: multibob_server, refused to exit gracefully, killing...");
      $cmd = "killall -KILL multibob_server";
      $rval = system($cmd);

    }
  }
                                                                                                          
  sleep(1);
}
