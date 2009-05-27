#!/usr/bin/env perl

###############################################################################
#
# server_apsr_transfer_manager.pl
#
# Transfers APSR observations from the APSR machines to Swinburne.
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Required Modules
#
use strict;
use Apsr;
use File::Basename;
use threads;
use threads::shared;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada->preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant DL            => 1;
use constant PIDFILE       => "apsr_transfer_manager.pid";
use constant LOGFILE       => "apsr_transfer_manager.log";
use constant QUITFILE      => "apsr_transfer_manager.quit";
use constant DATA_RATE     => 400;    # Mbits / sec
use constant TCP_WINDOW    => 1700;   # About right for Parkes-Swin transfers
use constant VSIB_MIN_PORT => 24301;
use constant VSIB_MAX_PORT => 24326;

#
# Global Variables
#
our %cfg   = Apsr->getApsrConfig();   # Apsr.cfg
our $error = $cfg{"STATUS_DIR"}."/apsr_transfer_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/apsr_transfer_manager.warn";
our $quit_daemon : shared  = 0;

# For interrupting and killing in the signal handlers
our $user    : shared = "";      # swinburne user
our $host    : shared = "";      # swinburne host
our $dir     : shared = "";      # swinburne dir
our $curr_obs    : shared = "none";  # current obs being transferred 


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
my $obs = "";
my $result;
my $response;
my $i=0;

#
# Destination disks at swinburne
#
my @dests = ();

for ($i=0; $i<$cfg{"NUM_SWIN_DIRS"}; $i++) {
  push (@dests, $cfg{"SWIN_DIR_".$i});
}

#
# Main
#

# Dada->daemonize($logfile, $pidfile);
Dada->logMsg(0, DL, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

setStatus("Starting script");

my $sent_to_swin = 0;
my $dest_i = 0;

chdir $cfg{"SERVER_ARCHIVE_NFS_MNT"};

# Ensure no vsib processes running locally or remotely
# TODO this currently will kill BPSR transfers and vice versa
interrupt_all(\@dests);

#
# Main loop
#

# We cycle through ports to prevent "Address already in use" problems
my $vsib_port = VSIB_MIN_PORT;

my $cmd = "";
my $xfer_failure = 0;
my $files = "";
my $counter = 0;

Dada->logMsg(1, DL, "Starting Main");

while (!$quit_daemon) {

  # Fine an observation to send based on the CWD
  Dada->logMsg(2, DL, "main: getObsToSend()");
  ($obs) = getObsToSend($cfg{"SERVER_ARCHIVE_NFS_MNT"});
  Dada->logMsg(2, DL, "main: getObsToSend(): ".$obs);

  if ($obs eq "none") {

    Dada->logMsg(2, DL, "main: no observations to send. sleep 60");
    
  } else {

    # Find a place to write to
    ($user, $host, $dir) = getDestination(\@dests, $dest_i);

    if ($host eq "none") {
      Dada->logMsg(2, DL, "no destinations available");
      setStatus("WARN: no dest available");
      $obs = "none";

    } else {
  
      Dada->logMsg(2, DL, "Destination: ".$user."@".$host.":".$dir);

      # If we have something to send
      if ($quit_daemon) {
        # Quit from the loop

      } elsif (($obs ne "none") && ($obs =~ /^2/)) {

        $xfer_failure = 0;
        $files = "";

        setStatus($obs." &rarr; ".$host);

        # Transfer the files
        $curr_obs = $obs;

        ($vsib_port, $xfer_failure) = transferObs($obs, $vsib_port, $user, $host, $dir);

        # If we have been asked to quit
        if ($quit_daemon) {

          Dada->logMsg(2, DL, "main: asked to quit");

        } else {

          # Test to see if this beam was sent correctly
          if (!$xfer_failure) {

            ($result, $response) = checkTransfer($user, $host, $dir, $obs);

            if ($result eq "ok") {

              Dada->logMsg(0, DL, "Successfully transferred: ".$obs);

              # Mark this as obs.transferred, removing obs.finished
              $cmd = "touch ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$obs."/obs.transferred";
              Dada->logMsg(2, DL, "main: ".$cmd);
              ($result, $response) = Dada->mySystem($cmd);
              Dada->logMsg(2, DL, "main: ".$result." ".$response);
              if ($result ne "ok") {
                Dada->logMsgWarn($warn, "could not touch ".$obs."/obs.transferred");
              }

              unlink ($cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$obs."/obs.finished");

              # change the remote directory structure to SOURCE / UTC_START
              Dada->logMsg(2, DL, "main: changeSourceDirStructure($user, $host, $dir, $obs)");
              ($result, $response) = changeSourceDirStructure($user, $host, $dir, $obs);
              Dada->logMsg(2, DL, "main: changeSourceDirStructure: ".$result." ".$response);

            } else {
              $xfer_failure = 1;
              Dada->logMsgWarn($warn, "Transfer failed: ".$obs." ".$response);
            }
          }
        

          # If we have a failed transfer, mark the /nfs/archives as failed
          if ($xfer_failure) {
            setStatus("WARN: ".$obs." xfer failed");
          } else {
            setStatus($obs." xfer success");
          }
          $curr_obs = "none";
        }
      }

    } 
  }

  # If we did not transfer, sleep 60
  if ($obs eq "none") {

    setStatus("Waiting for obs");
    Dada->logMsg(2, DL, "Sleeping 60 seconds");

    $counter = 12;
    while ((!$quit_daemon) && ($counter > 0)) {
      sleep(5);
      $counter--;
    }
  }
}

# rejoin threads
$daemon_control_thread->join();

setStatus("Script stopped");
Dada->logMsg(0, DL, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#


#
# Ensure the files sent matches the local
#
sub checkTransfer($$$$) {

  my ($user, $host, $dir, $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";
  my $local_list = "local";
  my $remote_list = "remote";

  # Get the listing of the files (assumes cwd is SERVER_ARCHIVES_NFS_MNT)
  $cmd = "find ".$obs." -mindepth 1 -not -name 'obs.finished' -printf '\%h/\%f \%T@\n' | sort";

  Dada->logMsg(2, DL, "checkRemoteArchive: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, DL, "checkRemoteArchive: ".$result." ".$response);
  if ($result  ne "ok") {
    Dada->logMsgWarn($warn, "checkRemoteArchive: local ls failed ".$response);
  } else {
    $local_list = $response;
  }

  $cmd = "find ".$obs." -mindepth 1 -printf '\%h/\%f \%T@\\n' | sort";
  # Get the remote listing of the files
  Dada->logMsg(2, DL, "checkRemoteArchive: [".$host."] ".$cmd);
  ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd, $dir);  
  Dada->logMsg(3, DL, "checkRemoteArchive: [".$host."] ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "checkRemoteArchive: ssh failed ".$response);
  } else {
    if ($rval != 0) {
      Dada->logMsgWarn($warn, "checkRemoteArchive: getting remote list failed: ".$response); 
    } else {
      $remote_list = $response;
    }
  }

  $cmd = "rm -f WRITING";
  ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd, $dir);
  if ($result  ne "ok") {
    Dada->logMsgWarn($warn, "checkRemoteArchive: ssh failure: ".$response);
  } else {
    if ($rval != 0) {
      Dada->logMsgWarn($warn, "checkRemoteArchive: could not remove ".$user."@".$host.":".$dir."/WRITING");
    }
  }

  if ($local_list eq $remote_list) {
    return ("ok","");

  } else {
    Dada->logMsgWarn($warn, "checkRemoteArchive: local files did not match remote ones after transfer");
    Dada->logMsgWarn($warn, "local=".$local_list);
    Dada->logMsgWarn($warn, "remote=".$remote_list);
    return ("fail", "archive mismatch");
  }
}


# Adjust the directory structure to match the required storage format
sub changeSourceDirStructure($$$$) {

  my ($u, $h, $d, $obs) = @_;
  
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  # determine if this is a multi fold observation
  $cmd = "server_apsr_archive_finalizer.csh ".$d." ".$obs;
  Dada->logMsg(2, DL, "changeSourceDirStructure: [".$host."] ".$cmd);
  ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd);
  Dada->logMsg(2, DL, "changeSourceDirStructure: [".$host."] ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "changeSourceDirStructure: ssh failed ".$response);
    return ("fail", "");
  } else {
    if ($rval != 0) {
      Dada->logMsgWarn($warn, "changeSourceDirStructure: script failed: ".$response);
      return ("fail", "");
    }
  }

  return ("ok", "");  

}


#
# Find a destination to send from the list
#
sub getDestination(\@$) {

  (my $disks_ref, my $startdisk) = @_;

  Dada->logMsg(2, DL, "getDestination()");

  my @disks = @$disks_ref;
  my @disk_components = ();

  my $c=0;
  my $i=0;
  my $disk = "";
  my $user = "";
  my $host = "";
  my $path = "";
  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $space = "";

  # If the array as 0 size, return none
  if ($#disks == -1) {
    return ("none", "none", "none");
  }

  for ($c=0; $c<=$#disks; $c++) {
    $i = ($c + $startdisk)%($#disks+1);

    $disk = $disks[$i];

    $user = "";
    $host = "";
    $path = "";

    @disk_components = split(":",$disk,3);

    if ($#disk_components == 2) {

      $user = $disk_components[0];
      $host = $disk_components[1];
      $path = $disk_components[2];
      $space = 0;

      Dada->logMsg(2, DL, "getDestination: user=".$user.", host=".$host.", path=".$path);

      $cmd = "ls ".$path;
      Dada->logMsg(2, DL, "getDestination: [".$host."] ".$cmd);
      ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd);
      Dada->logMsg(2, DL, "getDestination: [".$host."] ".$result." ".$response);
      if ($result ne "ok") {
        Dada->logMsgWarn($warn, "getDestination: ".$user."@".$host." ".$cmd." failed: ".$response);
        next;
      }

      # check if this is being used for reading
      $cmd = "ls ".$path."/READING";
      Dada->logMsg(2, DL, "getDestination: [".$host."] ".$cmd);
      ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd);
        Dada->logMsg(2, DL, "getDestination: [".$host."] ".$result." ".$response);
      if ($result ne "ok") {
        Dada->logMsgWarn($warn, "getDestination: ".$user."@".$host." ".$cmd." failed: ".$response);
        next;
      }

      # If the READING file did not exist
      if ($response =~ m/No such file or directory/) {

        $cmd = "df ".$path." -P | tail -n 1";
        Dada->logMsg(2, DL, "getDestination: [".$host."] ".$cmd);
        ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd);
        Dada->logMsg(2, DL, "getDestination: [".$host."] ".$result." ".$response);
        if ($result ne "ok") {
          Dada->logMsgWarn($warn, "getDestination: ".$user."@".$host." ".$cmd." failed: ".$response);
          next;
        } 

      # The READING file DID exist, skip this disk
      } else {
        Dada->logMsg(0, DL, "Skipping ".$user."@".$host.":".$path.", currently READING");
        next;

      }

      # If we have got this far, then we have an available disk, check space remaining
      # response contains the df command output
      if ($response =~ m/No such file or directory/) {
        Dada->logMsgWarn($error, "getDestination:" .$user."@".$host.":".$path." was not a valid directory");
        next;

      } 

      my ($location, $total, $used, $avail, $junk) = split(/ +/,$response);
      Dada->logMsg(2, DL, "getDestination: used=$used, avail=$avail, total=$total");

      if (($avail / $total) < 0.05) {
        Dada->logMsgWarn($warn, "getDestination: ".$host.":".$path." is over 95% full");
        next;

      } 

      # Need more than 10 Gig
      if ($avail < 10000) {
        Dada->logMsgWarn($warn, "getDestination: ".$host.":".$path." has less than 10 GB left");
        next;

      }

      Dada->logMsg(2, DL, "getDestination: found : ".$user."@".$host.":".$path);
      return ($user,$host,$path);
    }
  }

  Dada->logMsg(2, DL, "getDestination: no holding area found");
  return ("none", "none", "none");

}


#
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
#
sub getObsToSend($) {

  my ($staging_dir) = @_;

  Dada->logMsg(2, DL, "getObsToSend(".$staging_dir.")");

  my $obs_to_send = "none";
  my $nfs_archives = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $cmd = "";
  my $result = "";
  my $response = "";
  my @obs_finished = ();
  my $i = 0;
  my $pid = "";

  # Ensure the NFS directory is mounted
  $cmd = "ls -ld ".$nfs_archives;
  Dada->logMsg(2, DL, "getObsToSend: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, DL, "getObsToSend: ".$result.":".$response);

  # Look for all observations marked obs.finished in SERVER_NFS_RESULTS_DIR
  $cmd = "find ".$nfs_archives." -maxdepth 2 -name obs.finished ".
         "-printf \"%h\\n\" | sort | awk -F/ '{print \$NF}'";

  Dada->logMsg(2, DL, "getObsToSend: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, DL, "getObsToSend: ".$result.":".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "getObsToSend: ".$cmd." failed: ".$response);
    return ("none");
  }

  # If there is nothing to transfer, simply return
  @obs_finished = split(/\n/, $response);
  if ($#obs_finished == -1) {
    return ("none");
  }

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && ($obs_to_send eq "none")); $i++) {

    $obs = $obs_finished[$i];
    Dada->logMsg(2, DL, "getObsToSend: checking ".$obs);

    # skip if no obs.info exists
    if (!( -f $nfs_archives."/".$obs."/obs.info")) {
      Dada->logMsg(0, DL, "Required file missing: ".$obs."/obs.info");
      next;
    }

    # Find the PID 
    $cmd = "grep ^PID ".$nfs_archives."/".$obs."/obs.info | awk '{print \$2}'";
    Dada->logMsg(2, DL, "getObsToSend: ".$cmd);
    ($result, $response) = Dada->mySystem($cmd);
    Dada->logMsg(2, DL, "getObsToSend: ".$result.":".$response);
    if ($result ne "ok") {
      Dada->logMsgWarn($warn, "getObsToSend: failed to parse PID from obs.info: ".$response);
      next;
    }
    $pid = $response;

    # We have a valid obs to transfer, this will break the loop
    if (($pid eq "P140") || ($pid eq "P456") || ($pid eq "P361")) {
      $obs_to_send = $obs;
      Dada->logMsg(1, DL, "Found obs to send: ".$obs);

    # If this is NOT something we are interested in, move it to the storage area
    } else {

      $cmd = "mv ".$obs." ".$cfg{"SERVER_ARCHIVE_STORAGE"}."/";
      Dada->logMsg(2, DL, "getObsToSend: moving non swin observation".$cmd);
      # ($result, $response) = Dada->mySystem($cmd);
      # Dada->logMsg(2, DL, "getObsToSend: ".$result.":".$response);
      next;
    }
  }

  Dada->logMsg(2, DL, "getObsToSend: returning ".$obs_to_send);

  return ($obs_to_send);

}

#
# Create the receiving directory on the remote machine
# and then run vsib_recv
#
sub run_vsib_recv($$$$$) {

  my ($u, $h, $d, $obs_dir, $port) = @_;

  my $cmd = "";
  my $pipe = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  # create the writing file
  $cmd = "touch ".$d."/WRITING";
  Dada->logMsg(2, DL, "run_vsib_recv [".$host."]: ".$cmd);
  ($result, $response) = Dada->remoteSshCommand($u, $h, $cmd);
  Dada->logMsg(2, DL, "run_vsib_recv [".$host."]: ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "run_vsib_recv [".$host."]: could not touch remote WRITING file");
  }

  # sleeping to avoid immediately consecutive ssh commands
  sleep(1);

  # run vsib_recv
  $cmd = "vsib_recv -q -1 -w ".TCP_WINDOW." -p ".$port." | tar -x";
  $pipe = "server_apsr_logger.pl -n vsib_recv_".$host;
  Dada->logMsg(2, DL, "run_vsib_recv [".$host."]: ".$cmd." | ".$pipe);
  ($result, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d, $pipe);
  Dada->logMsg(1, DL, "run_vsib_recv [".$host."]: ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($error, "run_vsib_recv[".$host."]: ssh failure ".
                             $response);
    return ("fail");
  }

  if ($rval != 0) {
    Dada->logMsgWarn($warn, "vsib_recv returned a non zero exit value");
    return ("fail");
  } else {
    Dada->logMsg(2, DL, "vsib_recv succeeded");
    return ("ok");
  }

}

#
# Runs vsib_proxy on the localhost, it will connect to s_host upon receiving 
# an incoming connection
#
sub run_vsib_proxy($$) {

  my ($host, $port) = @_;

  Dada->logMsg(2, DL, "run_vsib_proxy()");

  my $vsib_args = "-1 -q -w ".TCP_WINDOW." -p ".$port." -H ".$host;
  my $cmd = "vsib_proxy ".$vsib_args." 2>&1 | server_bpsr_server_logger.pl -n vsib_proxy";
  Dada->logMsg(2, DL, $cmd);

  # Run the command on the localhost
  system($cmd);

  # Check return value for "success"
  if ($? != 0) {

    Dada->logMsgWarn($error, "run_vsib_proxy: command failed");
    return ("fail");

  # On sucesss, return ok
  } else {

    Dada->logMsg(2, DL, "run_vsib_proxy: succeeded");
    return ("ok");

  }

}


sub interrupt_all(\@) {

  my ($s_ref) = @_;

  Dada->logMsg(2, DL, "interrupt_all()");

  my @s = @$s_ref;
  my $cmd = "";
  my $i=0;

  # kill all potential senders
  interrupt_vsib_send();

  my $disk = "";
  my $user = "";
  my $host = "";
  my $path = "";
  my @disk_components = ();

  # for each swinburne disk
  for ($i=0; $i<=$#s; $i++) {

    $disk = $s[$i];
    $user = "";
    $host = "";
    $path = "";
    @disk_components = split(":",$disk,3);

    Dada->logMsg(2, DL, "interrupt_all: ".$disk);

    if ($#disk_components == 2) {
      $user = $disk_components[0];
      $host = $disk_components[1];
      $path = $disk_components[2];
      interrupt_vsib_recv($user, $host, $path, "none");
    }
  }

}

#
# ssh to host specified and kill all vsib_send's
#
sub interrupt_vsib_send() {

  Dada->logMsg(2, DL, "interrupt_vsib_send()");
 
  my $cmd = "killall vsib_send";
  Dada->logMsg(1, DL, "interrupt_vsib_send: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, DL, "interrupt_vsib_send: ".$result." ".$response);

  return ($result, $response);
}

#
# ssh to host specified and kill all vsib_recv's, cleanup the obs
#
sub interrupt_vsib_recv($$$$) {

  my ($user, $host, $dir, $obs) = @_;

  Dada->logMsg(2, DL, "interrupt_vsib_recv(".$user.", ".$host.", ".$dir.", ".$obs.")");

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  if ($host ne "none") {

    # kill vsib_recv
    $cmd = "killall vsib_recv";
    Dada->logMsg(2, DL, "interrupt_vsib_recv: ".$cmd);
    ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd);
    if ($result  ne "ok") {
      Dada->logMsgWarn($warn, "interrupt_vsib_recv: ssh failure ".$response);
    } else {
      if ($rval != 0) {
        Dada->logMsgWarn($warn, "interrupt_vsib_recv: failed to kill vsib_recv: ".$response);
      }
    }

    # delete the WRITING flag
    $cmd = "rm -f WRITING";
    Dada->logMsg(2, DL, "interrupt_vsib_recv: ".$cmd);
    ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd, $dir);
    if ($result  ne "ok") {
      Dada->logMsgWarn($warn, "interrupt_vsib_recv: ssh failure ".$response);
    } else {
      if ($rval != 0) {
        Dada->logMsgWarn($warn, "interrupt_vsib_recv: failed to remove WRITING flag: ".$response);
      } 
    }

    # optionally delete the transfer specied, this is a beam
    if (!(($obs eq "none") || ($obs eq ""))) {
      $cmd = "rm -rf ".$obs;
      Dada->logMsg(2, DL, "interrupt_vsib_recv: ".$cmd);
      ($result, $rval, $response) = Dada->remoteSshCommand($user, $host, $cmd, $dir);
      if ($result  ne "ok") {
        Dada->logMsgWarn($warn, "interrupt_vsib_recv: ssh failure ".$response);
      } else {
        if ($rval != 0) {
          Dada->logMsgWarn($warn, "interrupt_vsib_recv: failed to remove WRITING flag: ".$response);
        } 
      }
    }
  }
}

#
# run vsib_send  with tar -c obs feeding stdin to port to the recv_host
#
sub run_vsib_send($$$$) {

  my ($dir, $obs, $recv_host, $port) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  if ($recv_host eq "none") {
    return ("fail");
  }

  $cmd = "tar --exclude obs.finished -c ".$obs." | vsib_send -q -w ".TCP_WINDOW." -p ".$port." -H ".$recv_host." -z ".
            DATA_RATE." 2>&1 | server_apsr_logger.pl -n vsib_send"; 

  Dada->logMsg(2, DL, "run_vsib_send: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(1, DL, "run_vsib_send: ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "run_vsib_send: vsib_send failed ".$response);
    return("fail");
  } else {
    return("ok");
  }
}


#
# Since vsib_recv requires sub directories to be created before 
# the files arrive, get the list of subdirs
#
sub getObsDirs($) {

  my ($obs) = @_; 
  Dada->logMsg(2, DL, "getObsDirs(".$obs.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "find ".$obs." -type d -print \"\%f \"";
  Dada->logMsg(2, DL, "getObsDirs: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, DL, "getObsDirs: ".$result.":".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "getObsDirs: find failed: ".$response);
    return ("fail", "find failed on subdirs");
  }
  if ($response eq "") {
    return ("fail", "found no subdirs");
  } 
  return ("ok", $response);

}


#
# Returns the files to be sent for this observation, should
# also handle multi-fold mode.
#
sub getObsFiles($) {

  my ($obs) = @_;

  Dada->logMsg(2, DL, "getObsFiles(".$obs.")");

  my $dir = $obs;
  my @files = ();
  my $file_list = "";
  my $cmd = "";
  my $list = "";

  # obs.info *.ar
  $cmd = "find ".$dir." -name '*.ar' -o -name 'obs.info'";
  Dada->logMsg(2, DL, "getObsFiles: ".$cmd);

  # produce a \n separated list (for writing to file)
  $list = `$cmd`;
  if ($? == 0) {
    return ("ok", $list);
  } else {
    return ("fail", "find command failed");
  }
}


#
# Polls for the "quitdaemons" file in the control dir
#
sub daemonControlThread() {

  Dada->logMsg(2, DL, "daemon_control: thread starting");

  my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $daemon_quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  if (-f $daemon_quit_file) {
    print STDERR "daemon_control: quit file existed on startup, exiting\n";
    exit(1);
  }

  # poll for the existence of the control file
  while ((!-f $daemon_quit_file) && (!$quit_daemon)) {
    Dada->logMsg(3, DL, "daemon_control: Polling for ".$daemon_quit_file);
    sleep(1);
  }

  # signal threads to exit
  $quit_daemon = 1;

  # try to cleanly stop things
  if (!(($curr_obs eq "none") || ($curr_obs eq ""))) {
    Dada->logMsg(1, DL, "daemon_control: interrupting obs ".$curr_obs);
    interrupt_vsib_send();
    interrupt_vsib_recv($user, $host, $dir, $curr_obs);
    sleep(2);

  } else {
    Dada->logMsg(1, DL, "daemon_control: not interrupting due to no current observation");
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
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada->getCurrentDadaTime(0)."\n";

}


#
# Write the current status into the /nfs/control/apsr/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada->logMsg(2, DL, "setStatus(".$message.")");

  my $dir = "/nfs/control/apsr";
  my $file = "xfer.state";

  my $cmd = "rm -f ".$dir."/".$file;

  Dada->logMsg(2, DL, "setStatus: ".$cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, DL, "setStatus: ".$result." ".$response);

  $cmd = "echo '".$message."' > ".$dir."/".$file;
  Dada->logMsg(2, DL, "setStatus: ".$cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, DL, "setStatus: ".$result." ".$response);

  return ("ok", "");
    
}

#
# Transfer the specified obs/beam on port to hosts
#
sub transferObs($$$$$) {

  my ($obs, $port, $u, $h, $d) = @_;

  my $result = "";
  my $response = "";
  my $rval = 0;
  my $cmd = "";
  my $dirs = "";

  # Thread ID's
  my $send_thread = 0;
  my $recv_thread = 0;

  my $xfer_failure = 0;

  Dada->logMsg(1, DL, "Transferring ".$obs." to ".$u."@".$h.":".$d);

  # Get the directories that must be created
  # $dirs = getObsDirs($obs);

  # Get the files that must be transferred
  # $files = getObsFiles($obs);
  
  Dada->logMsg(2, DL, "run_vsib_recv(".$u.", ".$h.", ".$d.", ".$obs.", ".$port.")");
  $recv_thread = threads->new(\&run_vsib_recv, $u, $h, $d, $obs, $port);

  # sleep a small amount to allow startup of the receiver
  sleep(4);

  # Wait for the recv threads to be "ready"
  my $receiver_ready = 0;
  my $recv_wait = 10;
  while ((!$receiver_ready) && ($recv_wait > 0) && (!$quit_daemon)) {

    $receiver_ready = 1;

    my $ps_cmd = "ps aux | grep vsib_recv | grep ".$u." | grep -v grep";
    ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $ps_cmd);
    if ($result ne "ok") {
      Dada->logMsgWarn($warn, "ssh cmd failed: ".$response);
      $receiver_ready = 0;
    } else {
      if ($rval != 0) {
        $receiver_ready = 0;
      }
    } 

    if (!$receiver_ready) {
      Dada->logMsg(1, DL, "Waiting 2 seconds for receiver [".$h."] to be ready");
      sleep(2);
      $recv_wait--;
    }
  }

  if ($quit_daemon) {
    Dada->logMsgWarn($warn, "daemon asked to quit whilst waiting for vsib_recv");
    $xfer_failure = 1;

  } elsif (!$receiver_ready) {
    Dada->logMsgWarn($error, "vsib_recv was not ready after 20 seconds of waiting");
    $xfer_failure = 1;

  } else {

  }

  if ($xfer_failure) {
    # Ensure the receiving thread is not hung waiting for an xfer
    interrupt_vsib_recv($u, $h, $d, $curr_obs);

  } else {

    sleep(2);

    # start the sender
    Dada->logMsg(2, DL, "run_vsib_send(".$cfg{"SERVER_ARCHIVE_NFS_MNT"}.", ".$obs.", ".$h.", ".$port.")");
    $send_thread = threads->new(\&run_vsib_send, $cfg{"SERVER_ARCHIVE_NFS_MNT"}, $obs, $h, $port);

    # Transfer runs now
    Dada->logMsg(1, DL, "Transfer now running: ".$obs);

    # join the sending thread
    Dada->logMsg(2, DL, "joining send_thread");
    $result = $send_thread->join();
    Dada->logMsg(2, DL, "send_thread: ".$result);
    if ($result ne "ok") {
      $xfer_failure = 1;
      Dada->logMsgWarn($warn, "send_thread failed for ".$obs);
    }
  }

  # join the recv thread, if it exists
  Dada->logMsg(2, DL, "joining recv_thread");
  $result = $recv_thread->join();
  Dada->logMsg(2, DL, "recv_thread: ".$result);
  if ($result ne "ok") {
    $xfer_failure = 1;
    Dada->logMsgWarn($warn, "recv_thread failed for ".$obs);
  }

  if ($xfer_failure) {
    Dada->logMsgWarn($warn, "Transfer error for obs: ".$obs);
  } else {
    Dada->logMsg(1, DL, "Transfer complete: ".$obs);
  }

  # Increment the vsib port to prevent SO_REUSEADDR based issues
  $port++;
  if ($port == VSIB_MAX_PORT) {
    $port = VSIB_MIN_PORT;
  }

  return ($port, $xfer_failure);

} 

