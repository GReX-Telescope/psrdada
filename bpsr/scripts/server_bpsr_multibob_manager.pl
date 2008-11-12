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
# Constants
#
use constant DEBUG_LEVEL  => 1;
use constant PIDFILE      => "bpsr_multibob_manager.pid";
use constant LOGFILE      => "bpsr_multibob_manager.log";
use constant MULTIBOB_BIN => "multibob_server";


#
# Global Variables
#
our %cfg = Bpsr->getBpsrConfig();      # Bpsr.cfg
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
my $multibob_ssh_thread = 0;
my $multibob_plot_thread = 0;

my $uname = $cfg{"IBOB_GATEWAY_USERNAME"};
my $host  = $cfg{"IBOB_GATEWAY"};
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

logMessage(0, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

# Start the ibob SSH thread
$multibob_ssh_thread = threads->new(\&multiBobSshThread);

# Start the multibob plotting thread
$multibob_plot_thread = threads->new(\&multiBobPlotThread);

# Wait for threads to return
while (!$quit_daemon) {
  sleep(2);
}

# rejoin threads
$daemon_control_thread->join();
$multibob_ssh_thread->join();
$multibob_plot_thread->join();

logMessage(0, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#

sub multiBobSshThread() 
{

  my $runtime_dir = "/lfs/data0/bpsr/stats";
  my $multibob_bin = MULTIBOB_BIN;
  my $uname = $cfg{"IBOB_GATEWAY_USERNAME"};
  my $host  = $cfg{"IBOB_GATEWAY"};
  my $port  = $cfg{"IBOB_MANAGER_PORT"};
  my $rval  = 0;
  my $cmd   = "";

  checkExistingMultibobServer($uname, $host, $port, $multibob_bin);

  $cmd =  "ssh -n -l ".$uname." ".$host." \"cd ".$runtime_dir."; ".
          $multibob_bin." -n ".$npwc." -p ".$port."\" 2>&1";

  my $multibob_config_thread = 0;

  while (!$quit_daemon) {

    $multibob_config_thread = threads->new(\&configureMultibobServerWrapper);
    $multibob_config_thread->detach();

    # This command should "hang" until the multibob_server command has terminated
    logMessage(1, "multiBobSshThread: ".$cmd);
    system($cmd);
    logMessage(1, "multiBobSshThread: ssh cmd returned");

    if (!$quit_daemon) {
      logMessage(0, "ERROR: multibob_server returned unexpectedly");
    }

    sleep(1);

  }
}

sub configureMultibobServerWrapper() 
{
  logMessage(1, "configureMultibobServer: configuring");
  my ($result, $response) = Bpsr->configureMultibobServer();
  if ($result ne "ok") {
    logMessage(0, "ERROR: configureMultibobServer: failed ".$response);
  } else {
    logMessage(1, "configureMultibobServer: done");
  }
}

# 
# Monitors the /nfs/results/bpsr/stats directory creating the PD Bandpass plots
# as requried
#
sub multiBobPlotThread()
{

  logMessage(2, "multiBobPlotThread: thread starting");

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
   
    logMessage(2, "multiBobPlotThread: looking for bramdump files in ".$stats_dir);
 
    # look for plot files
    opendir(DIR,$stats_dir);
    @bramfiles = sort grep { !/^\./ && /\.bramdump$/ } readdir(DIR);
    closedir DIR;

    if ($#bramfiles == -1) {
      logMessage(2, "multiBobPlotThread: no files, sleeping");
    }

    # plot any existing bramplot files
    for ($i=0; $i<=$#bramfiles; $i++) 
    {
      $bramfile = $stats_dir."/".$bramfiles[$i];

      $plot_cmd = $bindir."/bpsr_bramplot ".$bramfile;

      logMessage(2, $plot_cmd);

      ($result, $response) = Dada->mySystem($plot_cmd);

      if ($result ne "ok") {
        logMessage(0, "plot of ".$bramfile." failed ".$response);
      } else {
        logMessage(2, "bpsr_bramplot ".$bramfile.": ".$response);
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

  logMessage(2, "multiBobPlotThread: thread exiting");

}


#
# Polls for the "quitdaemons" file in the control dir
#
sub daemonControlThread() {

  logMessage(2, "daemon_control: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;

  my $daemon_quit_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  # poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    logMessage(3, "daemon_control: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # signal threads to exit
  $quit_daemon = 1;

  # set the global variable to quit the daemon
  my $handle = Dada->connectToMachine($cfg{"IBOB_GATEWAY"}, $cfg{"IBOB_MANAGER_PORT"},1);
  if (!$handle) {
    logMessage(0, "daemon_control: could not connect to ".MULTIBOB_BIN);
  } else {

    # ignore welcome message
    $response = <$handle>;
    logMessage(0, "daemon_control: multibob_server <- quit");
    print $handle "quit\r\n";
    close($handle);
  }

  logMessage(2, "daemon_control: Unlinking PID file ".$pidfile);
  unlink($pidfile);

  logMessage(2, "daemon_control: exiting");

}

#
# Logs a message to the Nexus
#
sub logMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    print "[".$time."] ".$message."\n";
  }
}

#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada->getCurrentDadaTime(0)."\n";
  exit(1);

}

sub sigIgnore($) {
                                                                                                        
  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  print STDERR basename($0)." Ignoring\n";
                                                                                                        
}


sub removeOldPngs($$$) {

  my ($stats_dir, $ibob, $res) = @_;

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

    if (($time+10) < time)
    {
      $file = $stats_dir."/".$file;
      logMessage(2, "unlinking old png file ".$file);
      unlink($file);
    }
  }
}

sub checkExistingMultibobServer($$$$) {

  my ($uname, $host, $port, $command) = @_;

  # Check if the binary is running
  my $cmd = "ssh -l ".$uname." ".$host." \"ps aux | grep ".$command." | grep -v grep > /dev/null\"";
  logMessage(1, $cmd);
  my $rval = system($cmd);

  if ($rval == 0) {

    logMessage(0, "checkExistingMultibobServer: a multibob_server was running on ".$host);

    # set the global variable to quit the daemon
    my $handle = Dada->connectToMachine($host, $port, 1);
    if (!$handle) {
      logMessage(0, "checkExistingMultibobServer: could not connect to ".MULTIBOB_BIN);
    } else {
      $response = <$handle>;

      ($result, $response) = Dada->sendTelnetCommand($handle, "close");
      logMessage(0, "checkExistingMultibobServer: close -> multibob -> ".$result.":".$response);

      logMessage(0, "checkExistingMultibobServer: multibob_server <- quit");
      print $handle "quit\r\n";
      close($handle);
    }

    sleep(1);

    # try again to ensure it exited
    logMessage(1, $cmd);
    $rval = system($cmd);

    if ($rval == 0) {

      logMessage(0, "checkExistingMultibobServer: multibob_server still running on ".$host." refused to exit gracefully, killing...");
      $cmd = "ssh -l ".$uname." ".$host." \"killall -KILL multibob_server\"";
      $rval = system($cmd);

    }
  }

  sleep(1);

}

