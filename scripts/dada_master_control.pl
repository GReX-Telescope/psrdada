#!/usr/bin/env perl

###############################################################################
#
# dada_master_control.pl
#
# This script runs on boot and opens a socket to provide key information to the
# various daemons. The commands it will respond to are:
# 
# load_info             : return the current load of the machines
# disk_info path        : return information on a disk path
# db_info key [key2]    : return datablock information for the specified keys
# kill_pname name       : kill the named process
# kill_pid pid          : kill the specified pid
# init_db key nbuf size : 
# dest_db key           : destroys the shared memory defined by key

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;         # strict mode (like -Wall)
use POSIX qw(setsid);
use IO::Socket;     # Standard perl socket library
use Net::hostent;   # To determine the hostname of the server
use Switch;
use Dada;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant DEBUG_LEVEL      => 1;
use constant LOGFILE          => "dada_master_control.log";
use constant DADA_MASTER_PORT => 44444;


#
# Global Variables
#
our $quit_daemon = 0;

#
# Local Variables
#
my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".PIDFILE;

my $host = Dada::getHostMachineName();
my $port = DADA_MASTER_PORT;
my $control_thread = 0;

# Turn the script into a daemon
Dada::daemonize($logfile, $pidfile);

$control_thread = control_thread = threads->new(\&controlThread, $pid_file);

my $server = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
);

if (!$server) {
  print STDERR "Could not create a socket at ".$host.":".$port.": $!\n";
  exit(1);
}

my $read_set = 0;
my $rh = 0;
my $handle = 0;
my $comamand = "";

$read_set = new IO::Select();  # create handle set for reading
$read_set->add($server);       # add the main socket to the set

# Main loop
while (!$quit_daemon) {

  # Get all the readable handles from the server
  my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

  foreach $rh (@$readable_handles) {

    # If we are accepting a new connection
    if ($rh == $server) {

      # Wait for a connection from the server on the specified port
      $handle = $tcs_socket->accept() or die "accept $!\n";

      # Ensure commands are immediately sent/received
      $handle->autoflush(1);

      # Add this read handle to the set
      $read_set->add($handle);

      # Get information about the connecting machine
      $peeraddr = $handle->peeraddr;
      $hostinfo = gethostbyaddr($peeraddr);
      logMsg(1, "Accepting connection from ".$hostinfo->name);

    } else {

      $command = <$rh>;

      if (! defined $command) {

        logMsg(1, "Lost connection");

        $read_set->remove($rh);
        close($rh);
        $handle->close();
        $handle = 0;

      } else {

        # clean up the string
        $command =~ s/\r//;
        $command =~ s/\n//;
        $command =~ s/#(.)*$//;
        $command =~ s/ +$//;

        logMsg(1, "<- ".$command);

        $response = handleCommand($command);

        print $handler $response."\r\n";
        logMsg(1, "-> ".$response;

        $handle->close();

      }
    }
  }
}

$control_thread->join();

logMessage(0, "STOPPING SCRIPT");

exit 0;


###############################################################################
#
# Functions
#


sub handleCommand($) {

  (my $command) = @_;

  my @cmds= ();
  my $result = "ok";
  my $response = "";
  my $cmd = "";
  my $args = "";

  logMessage(2, "handleCommand: command=".$command);
  ($cmd, $args) = split(/ /,$command, 2);
  logMessage(2, "handleCommand: cmd=".$cmd.", args=".$args);

  # Handle the command
  switch ($cmd) {

    case "load_info" {
      ($result,$response) = Dada::getLoadInfo();
    }

    case "disk_info" {
      if ($args eq "") {
        $result = "fail";
        $response = "no argument specified";
      } else {
        ($result,$response) = Dada::getDiskInfo($args);
      } 
    }

    case "db_info" {
      if ($args eq "") {
        $result = "fail";
        $response = "no argument specified";
      } else {
        my @keys = split(/ /,$args);
        for ($i=0; $i<=$#keys; $i++) {
          ($sub_result,$sub_response) = Dada::getDBInfo($keys[$i]);
          $response .= $sub_response;
          if ($result eq "ok") {
            $result = $sub_result;
          }
        }
      }
    }

    case "init_db" {
      if ($args eq "") {
        $result = "fail";
        $response = "no argument specified";
      } else {

        my ($key, $nbufs, $bufsz) = split(/ /, $args);
        if (($key eq "") or ($nbufs eq "") or ($bufsz eq "")) {
          $result = "fail";
          $response = "arguments missing";
        } else {
          $script = "sudo ~/linux64/bin/dada_db -l -k ".$key." -n ".$nbufs." -n ".$bufzs;
          ($result, $response) = Dada::mySystem($script);
        }
      }
    }

    case "destroy_db" {
      if ($args eq "") {
        $result = "fail";
        $response = "no argument specified";
      } else {
        $script = "sudo ~/linux64/bin/dada_db -d -k ".$key;
        ($result, $response) = Dada::mySystem($script);
      }
    }

    case "kill_pname" {
      if ($args eq "") {
        $result = "fail";
        $response = "no argument specified";
      } else {
        $script = "killall -TERM ".$args;
        ($result, $response) = Dada::mySystem($script);
      }


    }







  # request for load info
  if ($cmds[0] eq "load_info") {
    ($result,$response) = Dada::getLoadInfo();
  }

  


  elsif ($commands[0] eq "stop_pwcs") {
    $cmd = "killall ".$cfg{"PWC_BINARY"};
    ($result,$response) = mysystem($cmd, 0);  
  }

  elsif ($commands[0] eq "stop_pwc") {
    $cmd = "killall -KILL ".$commands[1];
    ($result,$response) = mysystem($cmd, 0);  
  }

  elsif ($commands[0] eq "stop_dfbs") {
    $cmd = "killall -KILL ".$cfg{"DFB_SIM_BINARY"};
    ($result,$response) = mysystem($cmd, 0);
  }

  elsif ($commands[0] eq "stop_srcs") {
    $cmd = "killall dada_dbdisk dspsr dada_dbnic";
    ($result,$response) = mysystem($cmd, 0);  
  }

  elsif ($commands[0] eq "kill_process") {
    ($result,$response) = Dada::killProcess($commands[1]);
  }

  elsif ($commands[0] eq "start_bin") {
    $cmd = $current_binary_dir."/".$commands[1];
    ($result,$response) = mysystem($cmd, 0);  
  }

  elsif ($commands[0] eq "start_pwcs") {
    $cmd = $current_binary_dir."/".$cfg{"PWC_BINARY"}." -d -k dada -p ".$cfg{"CLIENT_UDPDB_PORT"}." -c ".$cfg{"PWC_PORT"}." -l ".$cfg{"PWC_LOGPORT"};
    ($result,$response) = mysystem($cmd, 0);
  }
  
  elsif ($commands[0] eq "set_bin_dir") {
    ($result, $response) = Dada::setBinaryDir($commands[1]);
  }
                                                                                                                                                      
  elsif ($commands[0] eq "get_bin_dir") {
    ($result, $response) = Dada::getCurrentBinaryVersion();
  }
                                                                                                                                                      
  elsif ($commands[0] eq "get_bin_dirs") {
    ($result, $response) = Dada::getAvailableBinaryVersions();
  }
                                                                                                                                                      
  elsif ($commands[0] eq "destroy_db") {

    my $temp_result = "";
    my $temp_response = "";

    $result = "ok";
    $response = "";

    my @datablocks = split(/ /,$cfg{"DATA_BLOCKS"});
    my $db;

    foreach $db (@datablocks) {

      if ( (defined $cfg{$db."_BLOCK_BUFSZ"}) && (defined $cfg{$db."_BLOCK_NBUFS"}) ) {
    
        # Create the dada data block
        $cmd = "sudo ".$current_binary_dir."/dada_db -d -k ".lc($db);
        ($temp_result,$temp_response) = mysystem($cmd, 0);
   
      } else {
  
        $temp_result = "fail";
        $temp_response = "config file configuration error";
 
      }

      if ($temp_result eq "fail") {
        $result = "fail";
      }
      $response = $response.$temp_response;

    }

  }

  elsif ($commands[0] eq "init_db") {

    my $temp_result = "";
    my $temp_response = "";

    $result = "ok";
    $response = "";

    my @datablocks = split(/ /,$cfg{"DATA_BLOCKS"});
    my $db;

    foreach $db (@datablocks) {

      if ( (defined $cfg{$db."_BLOCK_BUFSZ"}) && (defined $cfg{$db."_BLOCK_NBUFS"}) ) {

        # Create the dada data block
        $cmd = "sudo ".$current_binary_dir."/dada_db -l -k ".lc($db)." -b ".$cfg{$db."_BLOCK_BUFSZ"}." -n ".$cfg{$db."_BLOCK_NBUFS"};
        ($temp_result,$temp_response) = mysystem($cmd, 0);

      } else {

        $temp_result = "fail";
        $temp_response = "config file configuration error";

      }        

      if ($temp_result eq "fail") {
        $result = "fail";
      }
      $response = $response.$temp_response;

    }

  }
                                                                                                                                                      
  elsif ($commands[0] eq "stop_daemons") {
    ($result,$response) = stopDaemons();
  }

  elsif ($commands[0] eq "daemon_info") {
    ($result,$response) = getDaemonInfo();
  }

  elsif ($commands[0] eq "start_daemons") {
    chdir($cfg{"SCRIPTS_DIR"});

    $result = "ok";
    $response = "ok";
 
    my $daemon_result = "";
    my $daemon_response = "";

    my $cmd = "";
    my $daemon;

    foreach $daemon (@daemons) {

      $cmd = "./client_".$daemon.".pl";
      ($daemon_result, $daemon_response) = mysystem($cmd, 0);

      if ($daemon_result eq "fail") {
        $result = "fail";
        $response .= $daemon_response;
      }
    }
  }

  elsif ($commands[0] eq "start_helper_daemons") {

    chdir($cfg{"SCRIPTS_DIR"});

    $result = "ok";
    $response = "ok";

    my $daemon_result = "";
    my $daemon_response = "";

    my $cmd = "";
    my $daemon;

    foreach $daemon (@helper_daemons) {

      $cmd = "./client_".$daemon.".pl";
      ($daemon_result, $daemon_response) = mysystem($cmd, 0);

      if ($daemon_result eq "fail") {
        $result = "fail";
        $response .= $daemon_response;
      }
    }
  }

  elsif ($commands[0] eq "dfbsimulator") {
    $cmd = $current_binary_dir."/".$cfg{"DFB_SIM_BINARY"}." ".$commands[1];
    ($result,$response) = mysystem($cmd, 0);
  }

  elsif ($commands[0] eq "system") {
    $cmd = $commands[1];
    ($result,$response) = mysystem($cmd, 0);  
  }

  elsif ($commands[0] eq "get_disk_info") {
    ($result,$response) = Dada::getDiskInfo($cfg{"CLIENT_RECORDING_DIR"});
  }

  elsif ($commands[0] eq "get_db_info") {
    ($result,$response) = Dada::getDBInfo(lc($cfg{"PROCESSING_DATA_BLOCK"}));
  }

  elsif ($commands[0] eq "get_alldb_info") {
    ($result,$response) = Dada::getAllDBInfo($cfg{"DATA_BLOCKS"});
  }

  elsif ($commands[0] eq "get_db_xfer_info") {
     ($result,$response) = Dada::getXferInfo();
  }

  elsif ($commands[0] eq "get_load_info") {
    ($result,$response) = Dada::getLoadInfo();
  }

  elsif ($commands[0] eq "set_udp_buffersize") {
    $cmd = "sudo /sbin/sysctl -w net.core.wmem_max=67108864";
    ($result,$response) = mysystem($cmd, 0);
    if ($result eq "ok") {
      $cmd = "sudo /sbin/sysctl -w net.core.rmem_max=67108864";
      ($result,$response) = mysystem($cmd, 0);
    }
  }

  elsif ($commands[0] eq "get_all_status") {
    my $subresult = "";
    my $subresponse = "";

    ($result,$subresponse) = Dada::getDiskInfo($cfg{"CLIENT_RECORDING_DIR"});
    $response = "DISK_INFO:".$subresponse."\n";
    
    ($subresult,$subresponse) = Dada::getALLDBInfo($cfg{"DATA_BLOCKS"});
    $response .= "DB_INFO:".$subresponse."\n";
    if ($subresult eq "fail") {
      $result = "fail";
    }

    ($subresult,$subresponse) = Dada::getLoadInfo();
    $response .= "LOAD_INFO:".$subresponse;
    if ($subresult eq "fail") {
      $result = "fail";
    }

  }

  elsif($commands[0] eq "get_status") {
    my $subresult = "";
    my $subresponse = "";
                                                                                                                                                               
    ($result,$subresponse) = Dada::getRawDisk($cfg{"CLIENT_RECORDING_DIR"});
    $response = $subresponse.";;;";

    ($subresult,$subresponse) = Dada::getAllDBInfo(lc($cfg{"PROCESSING_DATA_BLOCK"}));
    $response .= $subresponse.";;;";
    if ($subresult eq "fail") {
      $result = "fail";
    }

    ($subresult,$subresponse) = Dada::getLoadInfo();
    $response .= $subresponse.";;;";
    if ($subresult eq "fail") {
      $result = "fail";
    }

    ($subresult,$subresponse) = Dada::getUnprocessedFiles($cfg{"CLIENT_RECORDING_DIR"});
    $response .= $subresponse;
    if ($subresult eq "fail") {
      $result = "fail";
    }

  }

  elsif ($commands[0] eq "stop_master_script") {
    $quit = 1;
    $result = "ok";
    $response = "";
  }

  elsif ($commands[0] eq "help") {
    $response .= "Available Commands:\n";
    $response .= "clean_scratch      delete all data in ".$cfg{"CLIENT_SCRATCH_DIR"}."\n";
    $response .= "clean_archives     delete all temporary archives in ".$cfg{"CLIENT_ARCHIVES_DIR"}."\n";
    $response .= "clean_rawdata      delete all raw data files in ".$cfg{"CLIENT_RECORDING_DIR"}."\n";
    $response .= "clean_logs         delete all logfiles in ".$cfg{"CLIENT_LOGS_DIR"}."\n";
    $response .= "start_pwcs         runs ".$cfg{"PWC_BINARY"}." -d -p ".$cfg{"CLIENT_UDPDB_PORT"}."\n";
    $response .= "stop_pwcs          send a kill signal to all PWCS\n";
    $response .= "stop_pwc process   send a kill signal to the named process\n";
    $response .= "stop_srcs          send a kill signal to all SRCS [dada_dbdisk dspsr dada_dbnic]\n";
    $response .= "stop_dfbs          kill the dfb simulator [".$cfg{"DFB_SIM_BINARY"}."]\n";
    $response .= "start_bin cmdline  runs the \"cmdline\" binary file from the current bin dir\n";
    $response .= "set_bin_dir dir    sets the current binary directory to \"dir\"\n";
    $response .= "get_bin_dir        returns the current binary directory\n";
    $response .= "get_bin_dirs       returns a list of all valid binary directories\n";
    $response .= "destroy_db         attempts to destroy the data block\n";
    $response .= "init_db            attempts to create a data block with ".$cfg{"DATA_BLOCK_NBUFS"}." nbufs of ".$cfg{"DATA_BLOCK_BUFSZ"}."bytes each\n";
    $response .= "stop_daemons       send a kill signal to the various agent daemons\n";
    $response .= "start_daemons      starts the various agent daemons\n";
    $response .= "system cmdline     runs the command \"cmdline\" in a tcsh shell\n";
    $response .= "get_disk_info      returns capacity information about the recording disk\n";
    $response .= "get_db_info        returns capacity information about the data block\n";
    $response .= "get_alldb_info     returns num_blocks and num_full for each
data block\n";
    $response .= "get_db_xfer_info   returns information about current xfers in the data block\n";
    $response .= "get_load_info      returns capacity information about the data block\n";
    $response .= "get_all_status     returns disk, data block and load information\n";
    $response .= "set_udp_buffersize sets the kernels udp buffer size to 67108864 bytes\n";
    $response .= "help               print this help listing";
    $response .= "stop_master_script exit this daemon!";
    $result = "ok";
  }

  else {
    $result = "fail";
    $response = "Unrecognized command ".$commands[0];
  } 

  return ($result,$response);

}

sub mysystem($$) {

  (my $cmd, my $background=0) = @_;

  my $rVal = 0;
  my $result = "ok";
  my $response = "";
  my $realcmd = $cmd." 2>&1";

  if ($background) { $realcmd .= " &"; }

  if (RUN_SYS_COMMANDS eq "true") {

    logMessage(2, "About to run ".$realcmd);
    $response = `$realcmd`;
    $rVal = $?;
    $/ = "\n";
    chomp $rString;
  } else {
    logMessage(0, "CMD = ".$realcmd);
  }

  # If the command failed
  if ($rVal != 0) {
    $result = "fail";
  }
  logMessage(2, "response = $response");

  return ($result,$response);
                                                                                                                          
}

sub stopDaemons() {

  my $allStopped = "false";
  my $daemon_control_file = Dada::getDaemonControlFile($cfg{"CLIENT_CONTROL_DIR"});
  my $threshold = 20; # seconds
  my $daemon = "";
  my $allStopped = "false";
  my $result = "";
  my $response = "";

  `touch $daemon_control_file`;

  while (($allStopped eq "false") && ($threshold > 0)) {

    $allStopped = "true";
    foreach $daemon (@daemons) {
      my $cmd = "ps auxwww | grep \"perl ./client_".$daemon.".pl\" | grep -v grep";
      `$cmd`;
      if ($? == 0) {
        logMessage(1, "daemon ".$daemon." is still running");
        $allStopped = "false";
        if ($threshold < 10) {
          ($result, $response) = Dada::killProcess($daemon);
        }
      }
      
    }
    $threshold--;
    sleep(1);
  }

  my $message = "";
  if (unlink($daemon_control_file) != 1) {
    $message = "Could not unlink the daemon control file \"".$daemon_control_file."\"";
    logMessage(0, "Error: ".$message);
    return ("fail", $message);
  }
  # If we had to resort to a "kill", send an warning message back
  if (($threshold > 0) && ($threshold < 10)) {
    $message = "Daemons did not exit cleanly within ".$threshold." seconds, a KILL signal was used and they exited";
    logMessage(0, "Error: ".$message);
    return ("fail", $message);
  } 
  if ($threshold <= 0) {
    $message = "Daemons did not exit cleanly after ".$threshold." seconds, a KILL signal was used and they exited";
    logMessage(0, "Error: ".$message);
    return ("fail", $message);
  }

  return ("ok","Daemons exited correctly");

}


sub setupDirectories() {

  if (($cfg{"CLIENT_SCRATCH_DIR"}) && (! -d $cfg{"CLIENT_SCRATCH_DIR"})) {
    `mkdir -p $cfg{"CLIENT_SCRATCH_DIR"}`;
  }
  if (($cfg{"CLIENT_CONTROL_DIR"}) && (! -d $cfg{"CLIENT_CONTROL_DIR"})) {
    `mkdir -p $cfg{"CLIENT_CONTROL_DIR"}`;
  }
  if (($cfg{"CLIENT_RECORDING_DIR"}) && (! -d $cfg{"CLIENT_RECORDING_DIR"})) {
    `mkdir -p $cfg{"CLIENT_RECORDING_DIR"}`;
  }
  if (($cfg{"CLIENT_LOG_DIR"}) && (! -d $cfg{"CLIENT_LOG_DIR"})) {
    `mkdir -p $cfg{"CLIENT_LOG_DIR"}`;
  }
  if (($cfg{"CLIENT_ARCHIVE_DIR"}) && (! -d $cfg{"CLIENT_ARCHIVE_DIR"})) {
    `mkdir -p $cfg{"CLIENT_ARCHIVE_DIR"}`;
  }

}

sub logMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    print "[".Dada::getCurrentDadaTime()."] ".$message."\n";
  }
}



sub getDaemonInfo() {

  my $control_dir = $cfg{"CLIENT_CONTROL_DIR"}; 
  my $daemon;
  my $perl_daemon;
  my $daemon_pid_file;
  my $cmd;
  my %array = ();

  foreach $daemon (@daemons) {
    $perl_daemon = "client_".$daemon.".pl";
    $daemon_pid_file = $daemon.".pid";

    # Check to see if the process is running
    $cmd = "ps aux | grep ".$perl_daemon." | grep -v grep > /dev/null";
    logMessage(2, $cmd);
    `$cmd`;
    if ($? == 0) {
      $array{$daemon} = 1;
    } else {
      $array{$daemon} = 0;
    }

    # check to see if the PID file exists
    if (-f $control_dir."/".$daemon_pid_file) {
      $array{$daemon}++;
    }
  }

  # Add the PWC as a daemon
  $cmd = "ps aux | grep ".$cfg{"PWC_BINARY"}." | grep -v grep > /dev/null";
  logMessage(2, $cmd);
  `$cmd`;
  if ($? == 0) {
    $array{$cfg{"PWC_BINARY"}} = 2;
  } else {
    $array{$cfg{"PWC_BINARY"}} = 0;
  }

  my $i=0;
  my $result = "ok";
  my $response = "";

  my @keys = sort (keys %array);
  for ($i=0; $i<=$#keys; $i++) {
    if ($array{$keys[$i]} != 2) {
      $result = "fail";
    }
    $response .= $keys[$i]." ".$array{$keys[$i]}.",";
  }


  return ($result, $response);

}
                                                                                                   


