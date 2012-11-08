#!/usr/bin/env perl

###############################################################################
#
# This script transfers data from a directory on the pwc, to a directory on the
# nexus machine

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use IO::Socket;
use File::Basename;
use threads;
use threads::shared;
use Apsr;
use Dada;

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Function Prototypes
#
sub msg($$$);
sub decimateArchive($$);
sub processArchive($$);
sub processBasebandFile($$);

#
# Global Variables
# 
our $dl : shared;
our $daemon_name : shared;
our %cfg : shared;
our $quit_daemon : shared;
our $pwc_id : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Initialize Globals
#
$dl = 1;
%cfg = Apsr::getConfig();
$daemon_name = Dada::daemonBaseName($0);
$quit_daemon = 0;
$log_host = "";
$log_port = 0;
$log_sock = 0;
$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (!Dada::checkPWCID($pwc_id, %cfg))
{
  usage();
  exit(1);
}

# Autoflush STDOUT
$| = 1;

# Main
{

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";
  my $archive_dir    = $cfg{"CLIENT_ARCHIVE_DIR"};   # hi res archive storage
  my $results_dir    = $cfg{"CLIENT_RESULTS_DIR"};   # dspsr output directory

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


  # sanity check on whether the module is good to go
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok")
  {
    print STDERR "Duplicate script running\n";
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

  msg(0,"INFO", "STARTING SCRIPT");

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
    msg(2, "INFO", "main: ".$result." ".$response);

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
          msg(1, "INFO", $line);
        }

        ($result, $response) = processArchive($line, $archive_dir);
      }
    }

    # Look for 2 minute old .dada files that may have been produced in the results dir via baseband mode (i.e. P427)
    $cmd = "find . -name \"*.dada\" -cmin +1 | sort";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "main: ".$result." ".$response);

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
          msg(1, "INFO", "Processing baseband file ".$line);
        }

        ($result, $response) = processBasebandFile($line, $archive_dir);
      }
    }

    # If we didn't find any archives, sleep.
    if (!($found_something)) {
      sleep(5);
    }

  }

  # Rejoin our daemon control thread
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  Dada::nexusLogClose($log_sock);

  exit 0;

}


#
# Decimate the archive with the psh_script
#
sub decimateArchive($$)
{
  my ($file, $psh_script) = @_;

  msg(2, "INFO", "decimateArchive(".$file.", ".$psh_script.")");
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $bindir = Dada::getCurrentBinaryVersion();

  msg(2, "INFO", "Decimating archive ".$file);

  # if there are more than 16 channels
  $cmd = $bindir."/".$psh_script." ".$file;

  msg(2, "INFO", "decimateArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "decimateArchive ".$result." ".$response);

  if ($result ne "ok") {
    msg(0, "WARN", $psh_script." failed to decimate ".$file.": ".$response);
    return ("fail", "decimation failed ".$response);
  } else {
    $file =~ s/.ar$/.lowres/;
    return ("ok", $file);
  }
}


#
# Process an archive. Decimate it, send the decimated archive to srv via
# NFS then move the high res archive to the storage area
#
sub processArchive($$)
{
  my ($path, $archive_dir) = @_;

  msg(2, "INFO", "processArchive(".$path.", ".$archive_dir.")");

  my $result = "";
  my $response = "";
  my $dir = "";
  my $file = "";
  my $file_decimated = "";
  my $i = 0;
  my $nfs_results =  $cfg{"SERVER_RESULTS_NFS_MNT"};
  my $cmd = "";

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
  msg(2, "INFO", "processArchive: ".$dir."/".$file);

  # We never decimate pulse archives, or transfer them to the server
  if ($file =~ m/^pulse_/) {

    # do nothing

  } else {
    $file_decimated = decimateArchive($dir."/".$file, "lowres.psh");

    if (-f $file_decimated) {
      msg(2, "INFO", "processArchive: nfsCopy(".$file_decimated.", ".$nfs_results.", ".$dir);
      ($result, $response) = nfsCopy($file_decimated, $nfs_results, $dir);
      msg(2, "INFO", "processArchive: nfsCopy() ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "WARN", "procesArchive: nfsCopy() failed: ".$response);
      }

      msg(2, "INFO", "processArchive: unlinking ".$file_decimated);
      unlink($file_decimated);
    } else {
      msg(0, "WARN", "procesArchive: decimateArchive failed: ".$response);
    }

  }

  msg(2, "INFO", "processArchive: moving ".$file." to archive");

  if (! -d ($archive_dir."/".$dir)) {
    msg(2, "INFO", "processArchive: creating archive dir ".$archive_dir."/".$dir);

    $cmd = "mkdir -p ".$archive_dir."/".$dir;
    msg(2, "INFO", "processArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "processArchive: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "WARN", "failed to create archive dir: ".$response);
    }
  }
  
  $cmd = "mv ".$dir."/".$file." ".$archive_dir."/".$dir."/";
  msg(2, "INFO", "processArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "processArchive: ".$result." ".$response);
  
  if ($result ne "ok") {
    msg(0, "WARN", "failed to move ".$file." to archive dir ".$response);
  }
  
  # Add the archive to the accumulated totals
  if (!($file =~ m/^pulse_/)) {
    my $fres_archive = $archive_dir."/".$dir."/band.fres";
    my $tres_archive = $archive_dir."/".$dir."/band.tres";

    # If this is the first file, simply copy it
    if (! -f ($fres_archive)) {

      # The T scrunched (fres) archive
      $cmd = "cp ".$archive_dir."/".$dir."/".$file." ".$fres_archive;
      msg(2, "INFO", "processArchive: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "processArchive: ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

      # The F scrunched (tres) archive
      $cmd = "pam -F -e tres ".$fres_archive;
      msg(2, "INFO", "processArchive: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "processArchive: ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

    # Otherwise we are appending to the archive
    } else {

      # Add the archive to the T scrunched total
      $cmd = "psradd -T --inplace ".$fres_archive." ".$archive_dir."/".$dir."/".$file;
      msg(2, "INFO", "processArchive: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "processArchive: ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

      # Add the archive to the F scrunched total
      $cmd = "psradd -jF --inplace ".$tres_archive." ".$archive_dir."/".$dir."/".$file;
      msg(2, "INFO", "processArchive: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "processArchive: ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "WARN", $cmd." failed: ".$response);
      }

    }
  }
  
  return ("ok", "");
}

#
# Process an archive. Decimate it, send the decimated archive to srv via
# NFS then move the high res archive to the storage area
#   
sub processBasebandFile($$)
{
  my ($path, $archive_dir) = @_;
    
  msg(1, "INFO", "processBasebandFile(".$path.", ".$archive_dir.")");

  my $result = "";
  my $response = "";
  my $dir = ""; 
  my $file = "";
  my $file_decimated = "";
  my $i = 0;
  my $cmd = "";

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
  msg(1, "INFO", "processBasebandFile: ".$dir."/".$file);

   $cmd = "mv ".$dir."/".$file." ".$archive_dir."/".$dir."/";
  msg(1, "INFO", "processBasebandFile: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "processBasebandFile: ".$result." ".$response);

  return ($result, $response); 

}




#
# Copy a the file in dir to nfsdir
#
sub nfsCopy($$$)
{
  (my $file, my $nfsdir, my $dir) = @_;
  
  my $result = "";
  my $response = "";
  
  # If the nfs dir isn't automounted, ensure it is
  if (! -d $nfsdir) {
    `ls $nfsdir >& /dev/null`;
  }
  
  # this can ocurr when using multifold
  if (! -d $nfsdir."/".$dir) {
     msg(0, "INFO", "nfsCopy: ".$nfsdir."/".$dir." did not exist, creating");
    `mkdir -p $nfsdir/$dir`;
  }
     
  my $tmp_file = $file.".tmp";
  
  my $cmd = "cp ".$file." ".$nfsdir."/".$tmp_file;
  msg(2, "INFO", "NFS copy \"".$cmd."\"");
  ($result, $response) = Dada::mySystem($cmd,0);

  if ($result ne "ok") {
    return ("fail", "Command was \"".$cmd."\" and response was \"".$response."\"");
  } else {
    $cmd = "mv ".$nfsdir."/".$tmp_file." ".$nfsdir."/".$file;
    msg(2, "INFO", $cmd);
    ($result, $response) = Dada::mySystem($cmd,0);
    if ($result ne "ok") {
      return ("fail", "Command was \"".$cmd."\" and response was \"".$response."\"");
    } else {
      return ("ok", "");
    }
  }
}



sub controlThread($)
{
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

  if (-f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  }
  
  msg(2, "INFO", "controlThread: exiting");
  return 0;
} 


#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$)
{
  my ($level, $class, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, "sys", $class, "arch mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($)
{
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
sub sigPipeHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  if ($log_sock) {
    $log_sock->close();
  }
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}
