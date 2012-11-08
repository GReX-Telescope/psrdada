#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use Dada;
use Apsr;

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}


#
# Function prototypes
#
sub createLocalDirs($$$$);
sub remoteDirsThread($$$$);
sub jettison($);
sub process($$$);
sub touchBandFinished($$);
sub msg($$$);


#
# Global variable declarations
#
our $dl : shared;
our %cfg : shared;
our $pwc_id : shared;
our $user : shared;
our $daemon_name : shared;
our $gain_controller;
our $client_logger;
our $quit_daemon : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Global initialization
#
$dl = 1;
%cfg = Apsr::getConfig();
$pwc_id = 0;
$user = "apsr";
$daemon_name = Dada::daemonBaseName($0);
$gain_controller = "client_apsr_gain_controller.pl";
$client_logger = "client_apsr_src_logger.pl";
$quit_daemon = 0;
$log_host = 0;
$log_port = 0;
$log_sock = 0;

# Autoflush STDOUT
$| = 1;

# get the PWC ID
if ($#ARGV != 0)
{
  usage();
  exit(1);
}
$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (!Dada::checkPWCID($pwc_id, %cfg))
{
  usage();
  exit(1);
}

# set the DB key for this PWC
my $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});

# Main
{
  my $cmd = "";
  my $result = "";
  my $response = "";

  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") 
  {
    print STDERR "Duplicate script running\n";
    exit 1;
  }

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  $log_host          = $cfg{"SERVER_HOST"};
  $log_port          = $cfg{"SERVER_SYS_LOG_PORT"};

  my $control_thread = 0;
  my $calibrator_thread = 0;
  my $remote_dirs_thread = 0;
  my $gain_control_thread = 0;

  my %h = ();
  my $prev_header = "";
  my $raw_header = "";
  my $utc_start = "";
  my $band = "";
  my $source = "";
  my $pid = "";
  my $mode = "";
  my $proc_cmd = "";
  my $proc_dir = "";
  my $obs_start_file = "";

  # install signal handles
  $SIG{INT}  = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could not open log port: ".$log_host.":".$log_port."\n";
  }

  msg(0, "INFO", "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  # main Loop
  while ( !$quit_daemon ) 
  {

    # wait for the next header to be written to the datablock
    $cmd = "dada_header -k ".$db_key;
    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", $cmd." returned");

    # if the header has been written successfully (and not just killed)
    if ($? == 0) 
    {

      msg(2, "INFO", "main: Apsr::processHeader()\n");
      # Check if the header conforms to APSR specifications
      ($result, $response) = Apsr::processHeader($raw_header, $cfg{"CONFIG_DIR"});
      msg(2, "INFO", "main: Apsr::processHeader() ".$result." ".$response);

      # if the header is bad or a repeat, jettison it
      if ($result ne "ok") 
      {
        msg(1, "WARN", "main: malformed header: ".$response.", jettisoning");
        $result = jettison($db_key);
      }
      elsif ($raw_header eq $prev_header)
      {
        msg(1, "WARN", "main: repeated header, jettisoning");
        $result = jettison($db_key);
      }
      else
      {
        %h = Dada::headerToHash($raw_header);

        $utc_start = $h{"UTC_START"};
        $band = $h{"FREQ"};
        $source = $h{"SOURCE"};
        $pid = $h{"PID"};
        $mode = $h{"MODE"};
        $proc_cmd = $response;
        $proc_dir = $utc_start."/".$band;

        # setup the local directories
        $obs_start_file = createLocalDirs($utc_start, $band, $pid, $raw_header);

        # launch thread to create remote directories
        $remote_dirs_thread = threads->new(\&remoteDirsThread, $utc_start, $band, $pid, $obs_start_file);

        # launch the gain/level controller thread
        msg(1, "INFO", "main: starting level setting thread");
        $gain_control_thread  = threads->new(\&gainControlThread, $db_key);

        # process observation with the specified processing command, in the specified directory
        ($result) = process($proc_dir, $proc_cmd, $db_key);

        # stop the gain/level controller thread
        $cmd = "killall digimon";
        ($result, $response) = Dada::mySystem($cmd);
        msg(2, "INFO", "main: joining gain controller thread");
        $gain_control_thread->join();
        msg(2, "INFO", "main: gain controller thread joined");
        $gain_control_thread = 0;

        # join remote dirs thread
        msg(2, "INFO", "main: joining remoteDirsThread");
        $remote_dirs_thread->join();
        msg(2, "INFO", "main: remoteDirsThread joined");
 
        # touch the band.finished file in the 
        touchBandFinished($utc_start, $band);

        # Join a previous calibrator thread if it existed
        if ($calibrator_thread) {
          $calibrator_thread->join();
          $calibrator_thread = 0;
        }

        if (($mode =~ m/CAL/) || ($mode =~ m/HYDRA/))
        {
          msg(2, "INFO", "main: calibratorThread(".$utc_start.", ".$band.", ".$source.")");
          $calibrator_thread = threads->new(\&calibratorThread, $utc_start, $band, $source);
        }
      }
    } 
    # dada_header may have been killed
    else
    {
      if ( !$quit_daemon ) {
        msg(0, "WARN", "dada_header failed");
        sleep 1;
      }
    }

    $prev_header = $raw_header;
  }

  msg(2, "INFO", "main: joining threads");
  $control_thread->join();
  msg(2, "INFO", "main: control_thread joined");

  if ($calibrator_thread) {
    $calibrator_thread->join();
    msg(2, "INFO", "main: calibrator_thread joined");
  }

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

}

exit 0;

###############################################################################
#
# Functions
#

#
# process the observation 
#
sub process($$$)
{
  my ($proc_dir, $proc_cmd, $db_key) = @_;

  msg(2, "INFO", "process(".$proc_dir.", ".$proc_cmd.", ".$db_key.")");

  my $cmd = "";
  my $result = "";

  # replace <DADA_INFO> tags with the matching input .info file
  if ($proc_cmd =~ m/<DADA_INFO>/)
  {
    my $tmp_info_file =  "/tmp/bpsr_".$db_key.".info";
    # ensure a file exists with the write processing key
    if (! -f $tmp_info_file)
    {
      open FH, ">".$tmp_info_file;
      print FH "DADA INFO:\n";
      print FH "key ".$db_key."\n";
      close FH;
    }
    $proc_cmd =~ s/<DADA_INFO>/$tmp_info_file/;
  }

  # replace <DADA_KEY> tags with the matching input key
  $proc_cmd =~ s/<DADA_KEY>/$db_key/;

  # replace <DADA_RAW_DATA> tag with processing dir
  $proc_cmd =~ s/<DADA_DATA_PATH>/$proc_dir/;

  # output the result to the client logger
  $cmd = $proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ".$pwc_id." proc";

  msg(2, "INFO", "process: ".$cmd);
  chdir $cfg{"CLIENT_RESULTS_DIR"}."/".$proc_dir;

  msg(1, "INFO", "START ".$proc_cmd);
  my $rval = system($cmd);
  msg(1, "INFO", "END ".$proc_cmd);

  if ($rval == 0) 
  {
    $result = "ok";
  }
  else
  {
    msg(0, "WARN", "process: cmd failed: ".$?." ".$rval);
    $result = "fail";
  }

  return $result;

}

#
# jettison the observation by running dada_dbnull on the datablock
#
sub jettison($)
{
  my ($db_key) = @_;

  msg(0, "INFO", "jettison(".$db_key.")");

  my $cmd = "";
  my $result = "ok";

  $cmd = "dada_dbnull -s -k ".$db_key." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger;;

  msg(1, "INFO", "jettison: START ".$cmd);
  my $rval = system($cmd);
  msg(1, "INFO", "jettison: END ".$cmd);

  if ($rval != 0)
  {
    $result = "fail";
  }

  return $result;

}



sub controlThread($) 
{
  my ($pid_file) = @_;
  msg(2, "INFO", "controlThread: starting");

  my $cmd = "";
  my $regex = "";
  my $result = "";
  my $response = "";

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  while ((!$quit_daemon) && (! -f $host_quit_file) && (! -f $pwc_quit_file)) {
    sleep(1);
  }

  $quit_daemon = 1;

  # Kill the dada_header command
  $regex = "^dada_header";
  msg(2, "INFO", "controlThread: killProcess(".$regex.", ".$user.")");
  ($result, $response) = Dada::killProcess($regex, $user);
  msg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(1, "WARN", "controlThread: killProcess for ".$regex." failed: ".$response);
  }

  $regex = "^dspsr";
  msg(2, "INFO", "controlThread: killProcess(".$regex.", ".$user.")");
  ($result, $response) = Dada::killProcess($regex, $user);
  msg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(1, "WARN", "controlThread: killProcess for ".$regex." failed: ".$response);
  }

  $regex = "^dada_dbnull";
  msg(2, "INFO", "controlThread: killProcess(".$regex.", ".$user.")");
  ($result, $response) = Dada::killProcess($regex, $user);
  msg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(1, "WARN","controlThread: killProcess for ".$regex." failed: ".$response);
  }

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");

  return 0;
}

# 
# this thread waits for the archives from the calibrator to have been 
# transferred to the ARCHIVES_DIR, then copies the Tscrunched psradded
# archives to the appropriate calibrator database dir
#
sub calibratorThread($$$) {

  my ($utc_start, $band, $source) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $archives_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $tres_archive = $archives_dir."/".$utc_start."/".$band."/band.tres";
  my $fres_archive = $archives_dir."/".$utc_start."/".$band."/band.fres";

  # The tres archive *should* have been produced by the archive_manager
  if (! -f $tres_archive) {
    return ("fail", "tres archive did not exist: ".$tres_archive);
  }

  my $max_wait = 50;  # seconds
  my $done = 0;
  my $n_archives = 0;
  my $n_added = 0;

  msg(1, "INFO", "calibratorThread: Waiting 10 seconds for dspsr to finish");

  while ((!$quit_daemon) && ($max_wait > 0) && (!$done)) {
 
    sleep(10); 

    $n_archives = 0;
    $n_added = 0;

    # find out how many archives exist the tres archive
    $cmd = "psredit -q -c nsubint ".$tres_archive." | awk -F= '{print \$2}'";
    msg(2, "INFO", "calibratorThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "calibratorThread: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "calibratorThread: ".$cmd." failed: ".$response);
      $done = 2;
    } else {
      $n_added = $response;
    }

    # find out how many archive exist on disk
    $cmd = "find ".$archives_dir."/".$utc_start."/".$band." -name '*.ar' | wc -l";
    msg(2, "INFO", "calibratorThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "calibratorThread: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "calibratorThread: ".$cmd." failed: ".$response);
      $done = 2;
    } else {
      $n_archives = $response;
    }

    # If we have a match
    if (($n_added == $n_archives) && ($n_added != 0)) {
      $done = 1;
    }
  }

  # if we exited the while loop because we had an tres archive, add it to the DB
  if ($done == 1) {

    my $dir = $cfg{"CLIENT_CALIBRATOR_DIR"}."/".$source."/".$utc_start;

    $cmd = "mkdir -p ".$dir;
    msg(2, "INFO", "calibratorThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "calibratorThread: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "calibratorThread: ".$cmd." failed: ".$response);
    }

    $cmd = "cp ".$fres_archive." ".$dir."/".$band.".ar";
    msg(2, "INFO", "calibratorThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "calibratorThread: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "calibratorThread: ".$cmd." failed: ".$response);
    }

    $cmd = "pac -w -p ".$cfg{"CLIENT_CALIBRATOR_DIR"}." -u ar -u fits";
    msg(2, "INFO", "calibratorThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "calibratorThread: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "calibratorThread: ".$cmd." failed: ".$response);
    }
  }

  msg(1, "INFO", "calibratorThread: exiting");

  return 0;
}


#
# Thread to create the remote directories and copy the obs_start file
# 
sub remoteDirsThread($$$$) 
{
  my ($utc_start, $band, $proj_id, $obs_start_file) = @_;

  msg(2, "INFO", "remoteDirsThread(".$utc_start.", ".$band.", ".$proj_id.", ".$obs_start_file.")");

  my $remote_archive_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $remote_results_dir = $cfg{"SERVER_RESULTS_NFS_MNT"};
  my $localhost = Dada::getHostMachineName();
  my $cmd = "";
  my $dir = "";
  my $user_groups = "";

  # Ensure each directory is automounted
  if (!( -d  $remote_archive_dir)) {
    `ls $remote_archive_dir >& /dev/null`;
  }
  if (!( -d  $remote_results_dir)) {
    `ls $remote_results_dir >& /dev/null`;
  }

  # If the remote archives dir did not yet exist for some strange reason
  if (! -d $remote_archive_dir."/".$utc_start ) {
    $cmd = "mkdir -p ".$remote_archive_dir."/".$utc_start;
    msg(2, "INFO", $cmd);
    system($cmd);
  }

  # Create the nfs soft link to the local archives directory 
  $cmd = "ln -s /nfs/".$localhost."/".$user."/archives/".$utc_start."/".$band.
          " ".$remote_archive_dir."/".$utc_start."/".$band;
  msg(2, "INFO", $cmd);
  system($cmd);

  $cmd = "mkdir -p ".$remote_results_dir."/".$utc_start."/".$band;
  msg(2, "INFO", $cmd);
  system($cmd);

  # Check whether the user is a member of the specified group
  $user_groups = `groups $user`;
  chomp $user_groups;

  if ($user_groups =~ m/$proj_id/) {
    msg(2, "INFO", "Chmodding to ".$proj_id);
  } else {
    msg(0, "WARN", "Project ".$proj_id." was not an ".$user." group. Using '".$user."' instead'");
    $proj_id = $user;
  }

  # Adjust permission on remote archive directory
  $dir = $remote_archive_dir."/".$utc_start."/".$band;
  $cmd = "chgrp ".$proj_id." ".$dir;
  msg(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    msg(0, "WARN", "Failed to chgrp remote archive dir to ".$proj_id);
  }

  $cmd = "chmod g+s ".$dir;
  msg(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    msg(0, "WARN", "Failed to chmod remote archive dir to ".$proj_id);
  }

  # Adjust permission on remote results directory
  $dir = $remote_results_dir."/".$utc_start."/".$band;
  $cmd = "chgrp ".$proj_id." ".$dir;
  msg(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    msg(0, "WARN", "Failed to chgrp remote results dir to ".$proj_id);
  }

  $cmd = "chmod g+s ".$dir;
  msg(2, "INFO", $cmd);
  `$cmd`;
  if ($? != 0) {
    msg(0, "WARN", "Failed to chmod remote results dir to ".$proj_id);
  }

  copyObsStart($utc_start, $band, $obs_start_file);

}

  
#
# Create the local directories required for this observation
#
sub createLocalDirs($$$$) {

  (my $utc_start, my $centre_freq, my $proj_id, my $raw_header) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $file = "";

  my $results_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start."/".$centre_freq;
  my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start."/".$centre_freq;

  # Create the results dir
  msg(2, "INFO", "Creating local results dir: ".$results_dir);
  $cmd = "mkdir -p ".$results_dir;
  msg(2, "INFO", "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0,"ERROR", "Could not create local results dir: ".$response);
  }

  # Create the archive dir
  msg(2, "INFO", "Creating local results dir: ".$archive_dir);
  $cmd = "mkdir -p ".$archive_dir;
  msg(2, "INFO", "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0,"ERROR", "Could not create local archive dir: ".$response);
  }

  # create an obs.start file in the archive dir
  $file = $archive_dir."/obs.start";
  open(FH,">".$file.".tmp");
  print FH $raw_header;
  close FH;
  rename($file.".tmp",$file);

  # set the GID on the UTC_START dirs
  $results_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start;
  $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start;

  # Set GID on these dirs
  $cmd = "chgrp -R $proj_id $results_dir $archive_dir";
  msg(2, "INFO", "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "chgrp to ".$proj_id." failed on $results_dir $archive_dir");
  }

  # Set group sticky bit on local archive dir
  $cmd = "chmod -R g+s $results_dir $archive_dir";
  msg(2, "INFO", "createLocalDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "chmod g+s failed on $results_dir $archive_dir");
  }

  return $file;

}



#
# Copies the UTC_START file via NFS to the server's results directory
# utcstart file resides in clients archive dir
#
sub copyObsStart($$$) {

  my ($utc_start, $centre_freq, $obs_start) = @_;

  my $dir = "";
  my $cmd = ""; 
  my $result = "";
  my $response = "";

  # Ensure each directory is automounted
  $cmd = "ls ".$cfg{"SERVER_RESULTS_NFS_MNT"}." >& /dev/null";
  ($result, $response) = Dada::mySystem($cmd);

  # Create the full nfs destinations
  $dir = $cfg{"SERVER_RESULTS_NFS_MNT"}."/".$utc_start."/".$centre_freq;

  $cmd = "cp ".$obs_start." ".$dir."/";
  msg(2, "INFO", "NFS copy \"".$cmd."\"");
  ($result, $response) = Dada::mySystem($cmd,0);

  if ($result ne "ok") {
    msg(0, "ERROR", "NFS copy failed: ".$obs_start." to ".$dir.", response: ".$response);
    msg(0, "ERROR", "NFS cmd: ".$cmd);
    if (-f $obs_start) {
      msg(0, "ERROR", "File existed locally");
    } else {
      msg(0, "ERROR", "File did not exist locally");
    }
  }

  msg(2, "INFO", "Server directories perpared");

}

#
# Run the level setting cmd with the specified number of channels
#
sub gainControlThread($) 
{
  msg(2, "INFO", "gainControlThread: starting");
  (my $key) = @_;
  
  # ensure that a .viewer file exists for this datablock{
  my $tmp_viewer_file =  "/tmp/bpsr_".$db_key.".viewer";
  if (! -f $tmp_viewer_file)
  {
    open FH, ">".$tmp_viewer_file;
    print FH "DADA INFO:\n";
    print FH "key ".$key."\n";
    print FH "viewer\n";
    close FH;
  }

  my $nchan = 1;
  my $cmd = "digimon ".$tmp_viewer_file." | ".$cfg{"SCRIPTS_DIR"}."/".$gain_controller." ".$pwc_id." ".$nchan;

  msg(2, "INFO", "gainControlThread: ".$cmd);

  my $returnVal = system($cmd);
  if ($returnVal != 0) {
    msg(0, "WARN", "gainControlThread: digimon failed: ".$cmd." ".$returnVal);
  }

  msg(2, "INFO", "gainControlThread: exiting");
}


#
# If there are no rawdata files, touch band.finished
#
sub touchBandFinished($$) {

  my ($utc_start, $centre_freq) = @_;
  msg(2, "INFO", "touchBandFinished(".$utc_start.", ".$centre_freq.")");

  my $dir = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  # Ensure the results directory is mounted
  $cmd = "ls ".$cfg{"SERVER_RESULTS_NFS_MNT"}." >& /dev/null";
  ($result, $response) = Dada::mySystem($cmd);
  
  # Create the full nfs destinations
  $dir = $cfg{"SERVER_RESULTS_NFS_MNT"}."/".$utc_start."/".$centre_freq;
  
  $cmd = "touch ".$dir."/band.finished";
  msg(2, "INFO", "touchBandFinished: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd ,0);
  msg(2, "INFO", "touchBandFinished: ".$result." ".$response);
  
  # Touch a local band.finished for the transfer manager to use
  $dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start."/".$centre_freq;
  $cmd = "touch ".$dir."/band.finished";
  msg(2, "INFO", "touchBandFinished: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd ,0);
  msg(2, "INFO", "touchBandFinished: ".$result." ".$response);

}


#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $class, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      print "opening nexus log: ".$log_host.":".$log_port."\n";
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $class, "obs mngr", $msg);
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
