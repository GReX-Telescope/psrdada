###############################################################################
#  
#     Copyright (C) 2009 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Generalized transfer manager for APSR and CAPSR instruments
#
 
package Dada::server_transfer_manager;

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use File::Basename;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $daemon_name $server_logger $pid $dest $dest_id $optdir $rate %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $daemon_name;
our %cfg;
our $server_logger;
our $pid;
our $dest;
our $dest_id;
our $optdir;
our $rate;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $curr_obs : shared;
our $curr_band : shared;
our $instrument;
our $warn;
our $error;

our $send_user : shared;
our $send_host : shared;
our $dest_user : shared;
our $dest_host : shared;
our $dest_dir  : shared;


#
# initialize package globals
#
$dl = 1; 
$daemon_name = 0;
%cfg = ();
$server_logger = "";
$pid = "";
$dest = "";
$dest_id = 0;
$optdir = "";
$rate = 0;

#
# initialize other variables
#
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$curr_obs = "none";
$curr_band = "none";
$instrument = "";
$send_user = "none";
$send_host = "none";
$dest_user = "";
$dest_host = "";
$dest_dir = "";

#
# Constants
#
use constant TCP_WINDOW    => 1700;   # About right for Parkes-Swin transfers
use constant VSIB_MIN_PORT => 25301;
use constant VSIB_MAX_PORT => 25326;
use constant SSH_OPTS      => "-x -o BatchMode=yes";


###############################################################################
#
# package functions
# 

sub main() {

  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
  $instrument = $cfg{"INSTRUMENT"};

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $result = "";
  my $response = "";
  my $control_thread = 0;
  my $obs = "";
  my $band = "";
  my $i = 0;
  my $vsib_port = 0;
  my $cmd = "";
  my $xfer_failure = 0;
  my $files = "";
  my $counter = 0;

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }
  # user/host/dir should now be set and tested

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;


  # become a daemon
  # Dada->daemonize($log_file, $pid_file);
  
  Dada->logMsg(0, $dl ,"STARTING SCRIPT");

  # start the control thread
  Dada->logMsg(2, "INFO", "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  setStatus("Starting script");

  # We cycle through ports to prevent "Address already in use" problems
  $vsib_port = VSIB_MIN_PORT;

  Dada->logMsg(1, $dl, "Starting Main");

  chdir $cfg{"SERVER_ARCHIVE_NFS_MNT"};

  while (!$quit_daemon) {

    # Fine an observation to send based on the CWD
    Dada->logMsg(2, $dl, "main: getBandToSend()");
    ($obs, $band) = getBandToSend($cfg{"SERVER_ARCHIVE_NFS_MNT"});
    Dada->logMsg(2, $dl, "main: getBandToSend(): ".$obs."/".$band);


    if ($obs eq "none") {
      Dada->logMsg(2, $dl, "main: no observations to send. sleep 60");

    } else {

      # Find a place to write to
      ($result, $response) = checkDestination($dest_user, $dest_host, $dest_dir);

      if ($result ne "ok") {
        Dada->logMsgWarn($warn, "destination not available: ".$response);
        setStatus("WARN: ".$response);
        $obs = "none";
        $band = "none";

      } else {

        Dada->logMsg(2, $dl, "Destination: ".$dest_user."@".$dest_host.":".$dest_dir);

        # If we have something to send
        if ($quit_daemon) {
          # Quit from the loop

        } elsif (($obs ne "none") && ($obs =~ /^2/)) {

          $xfer_failure = 0;
          $files = "";

          setStatus($obs."/".$band." &rarr; ".$dest_host);

          $curr_obs  = $obs;
          $curr_band = $band;

          ($vsib_port, $xfer_failure) = transferBand($obs, $band, $vsib_port);

          # If we have been asked to quit
          if ($quit_daemon) {

            Dada->logMsg(2, $dl, "main: asked to quit");

          } else {

            # Test to see if this band was sent correctly
            if (!$xfer_failure) {

              ($result, $response) = checkTransfer($dest_user, $dest_host, $dest_dir, $obs, $band);

              if ($result eq "ok") {

                if ($pid ne "P427") {

                  # change the remote directory structure to SOURCE / UTC_START
                  Dada->logMsg(2, $dl, "main: changeSourceDirStructure(".$dest_user.", ".$dest_host.", ".$dest_dir.", ".$obs.", ".$band.")");
                  ($result, $response) = changeSourceDirStructure($dest_user, $dest_host, $dest_dir, $obs, $band);
                  Dada->logMsg(2, $dl, "main: changeSourceDirStructure: ".$result." ".$response);

                } else {

                  # touch a remote xfer.complete for P427 
                  Dada->logMsg(2, $dl, "main: markRemoteFile(xfer.complete, ".$dest_user.", ".$dest_host.", ".$dest_dir.", ".$obs."/".$band.")");
                  ($result, $response) = markRemoteFile("xfer.complete", $dest_user, $dest_host, $dest_dir, $obs."/".$band);
                  Dada->logMsg(2, $dl, "main: markRemoteFile: ".$result." ".$response);

                }

                markState($obs, $band, "", "sent.to.".$dest);
                # Check if all bands have been trasnferred successfully, if so, mark 
                # the observation as sent.to.dest
                Dada->logMsg(2, $dl, "main: checkAllBands(".$obs.")");
                ($result, $response) = checkAllBands($obs);
                Dada->logMsg(2, $dl, "main: checkAllBands: ".$result." ".$response);

                if ($result ne "ok") {
                  Dada->logMsgWarn($warn, "main: checkAllBands failed: ".$response);

                } else {

                  if ($response ne "all bands sent") {
                    Dada->logMsg(2, $dl, "Obs ".$obs." not fully transferred: ".$response);

                  } else {
                    Dada->logMsg(0, $dl, "transferred to ".$dest.": ".$obs);
                    markState($obs, "", "obs.finished", "obs.transferred");
                    setStatus($obs." xfer success");

                  }
                }

              } else {
                $xfer_failure = 1;
                Dada->logMsgWarn($warn, "Transfer failed: ".$obs."/".$band." ".$response);

              }

            }

            # If we have a failed transfer, mark the /nfs/archives as failed
            if ($xfer_failure) {
              setStatus("WARN: ".$obs." xfer failed");
            } else {
              setStatus($obs." xfer success");
            }

            $curr_obs = "none";
            $curr_band = "none";
            
          }
        }
      }
    }

    Dada->logMsg(1, $dl, "EARLY QUIT ENABLED FOR P427 TRANSFERS");
    $quit_daemon = 1;

    # If we did not transfer, sleep 60
    if ($obs eq "none") {
  
      setStatus("Waiting for obs");
      Dada->logMsg(2, $dl, "Sleeping 60 seconds");
    
      $counter = 12;
      while ((!$quit_daemon) && ($counter > 0)) {
        sleep(5);
        $counter--;
      }
    }
  }

  setStatus("Script stopped");

  Dada->logMsg(0, $dl, "STOPPING SCRIPT");

  # rejoin threads
  $control_thread->join();
                                                                                
  return 0;
}



###############################################################################
#
# Functions
#

sub markState($$$$) {

  my ($o, $b, $from, $to) = @_;

  Dada->logMsg(2, $dl, "markState(".$o.", ".$b.", ".$from.", ".$to.")");

  my $path = $o;
  my $cmd = "";
  my $result = "";
  my $response = "";

  if ($b ne "") {
    $path .= "/".$b;
  }

  if ($from ne "") {

    Dada->logMsg(1, $dl, $path.": ".$from." -> ".$to);
    $cmd = "rm -f ".$path."/".$from;
    Dada->logMsg(2, $dl, "markState: ".$cmd);
    ($result, $response) = Dada->mySystem($cmd);
    Dada->logMsg(2, $dl, "markState: ".$result." ".$response);
    if ($result ne "ok") {
      Dada->logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
    }

  } else {
    Dada->logMsg(1, $dl, $path." -> ".$to);

  }

  $cmd = "touch ".$path."/".$to;
  Dada->logMsg(2, $dl, "markState: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "markState: ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "markState: ".$cmd." failed: ".$response);
  }

}


#
# Ensure the files sent matches the local
#
sub checkTransfer($$$$$) {

  my ($u, $h, $d, $obs, $band) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";
  my $local_list = "local";
  my $remote_list = "remote";
  my $send_host = "";
  my $send_dir = $cfg{"CLIENT_ARCHIVE_DIR"};

  # Find the host on which the band directory is located
  $cmd = "find ".$obs."/".$band." -maxdepth 1 -printf \"\%l\" | awk -F/ '{print \$3}'";
  Dada->logMsg(2, $dl, "checkTransfer: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "checkTransfer: ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($error, "could not determine host on which ".$obs."/".$band." resides");
    return ("fail", "could not determine host on which ".$obs."/".$band." resides");
  }

  # host for vsib_send
  $send_host = $response;

  # Get the listing of the files (assumes cwd is SERVER_ARCHIVES_NFS_MNT)
  $cmd = "find -L ".$obs."/".$band." -mindepth 1 -not -name 'obs.finished' -printf '\%h/\%f \%T@\\n' | sort | uniq";

  Dada->logMsg(2, $dl, "checkTransfer [".$send_host."]: (".$instrument.", ".$send_host.", ".$cmd.", ".$send_dir.")");
  ($result, $rval, $response) = Dada->remoteSshCommand($instrument, $send_host, $cmd, $send_dir);  
  Dada->logMsg(3, $dl, "checkTransfer [".$send_host."]: ".$result." ".$response);
  if ($result  ne "ok") {
    Dada->logMsgWarn($warn, "checkTransfer: local find failed ".$response);
  } else {
    $local_list = $response;
  }

  $cmd = "find ".$obs."/".$band." -mindepth 1 -printf '\%h/\%f \%T@\\n' | sort | uniq";
  # Get the remote listing of the files
  Dada->logMsg(2, $dl, "checkTransfer: [".$dest_host."] (".$u.", ".$h.", ".$cmd.", ".$d.")");
  ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d);  
  Dada->logMsg(3, $dl, "checkTransfer: [".$dest_host."] ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "checkTransfer: ssh failed ".$response);
  } else {
    if ($rval != 0) {
      Dada->logMsgWarn($warn, "checkTransfer: getting remote list failed: ".$response); 
    } else {
      $remote_list = $response;
    }
  }

  $cmd = "rm -f ../WRITING";
  ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d);
  if ($result  ne "ok") {
    Dada->logMsgWarn($warn, "checkRemoteArchive: ssh failure: ".$response);
  } else {
    if ($rval != 0) {
      Dada->logMsgWarn($warn, "checkRemoteArchive: could not remove ".$u."@".$h.":".$d."/../WRITING");
    }
  }

  if ($local_list eq $remote_list) {
    return ("ok","");

  } else {
    Dada->logMsgWarn($warn, "checkRemoteArchive: local files did not match remote ones after transfer");
    #Dada->logMsgWarn($warn, "local=".$local_list);
    #Dada->logMsgWarn($warn, "remote=".$remote_list);
    return ("fail", "archive mismatch");
  }
}

#
# Check the obs to see if all the bands have been sent to the dest
#
sub checkAllBands($) {

  my ($obs) = @_;

  my $cmd = "";
  my $find_result = "";
  my $beam = "";
  my @beams = ();
  my $all_sent = 1;
  my $result = "";
  my $response = "";
  my $nbeams = 0;
  my $nbeams_mounted = 0;
  my $obs_pid = "";
    
  # Determine the number of NFS links in the archives dir
  $cmd = "find ".$obs." -mindepth 1 -maxdepth 1 -type l | wc -l";
  Dada->logMsg(2, $dl, "checkAllBands: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "checkAllBands: ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "checkAllBands: find command failed: ".$response);
    return ("fail", "find command failed");
  } 
  $nbeams = $response;
  Dada->logMsg(2, $dl, "checkAllBands: Total number of beams ".$nbeams);
    
  # Now find the number of mounted NFS links
  $cmd = "find -L ".$obs." -mindepth 1 -maxdepth 1 -type d -printf '\%f\\n' | sort";
  Dada->logMsg(2, $dl, "checkAllBands: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  if ($result ne "ok") { 
    Dada->logMsgWarn($warn, "checkAllBands: find command failed: ".$response);
     return ("fail", "find command failed");
  } 
  @beams = split(/\n/, $response);
  $nbeams_mounted = $#beams + 1;
  Dada->logMsg(2, $dl, "checkAllBands: Total number of mounted beams: ".$nbeams_mounted);
  
  # If a machine is not online, they cannot all be verified
  if ($nbeams != $nbeams_mounted) {
    return ("ok", "all beams not mounted");
  
  } else {
    $all_sent = 1;
    
    # skip if no obs.info exists
    if (!( -f $obs."/obs.info")) {
      Dada->logMsgWarn($warn, "checkAllBands: Required file missing ".$obs."/obs.info");
      return ("fail", $obs."/obs.info did not exist");
    }

    # get the PID
    $cmd = "grep ^PID ".$obs."/obs.info | awk '{print \$2}'";
    Dada->logMsg(2, $dl, "checkAllBands: ".$cmd);
    ($result, $response) = Dada->mySystem($cmd);
    Dada->logMsg(2, $dl, "checkAllBands: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "could not determine PID");
    }
    $obs_pid = $response;

    my $i=0;
    for ($i=0; (($i<=$#beams) && ($all_sent)); $i++) {
      $beam = $beams[$i];
      if (! -f $obs."/".$beam."/sent.to.".$dest) {
        $all_sent = 0;
        Dada->logMsg(2, $dl, "checkAllBands: ".$obs."/".$beam."/sent.to.".$dest." did not exist");
      }
    }

    if ($all_sent) {
      return ("ok", "all bands sent");
    } else { 
      return ("ok", "all bands not sent");
    }
  }

}




# Adjust the directory structure to match the required storage format
sub changeSourceDirStructure($$$$$) {

  my ($u, $h, $d, $obs, $band) = @_;
  
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  # determine if this is a multi fold observation
  $cmd = "server_apsr_archive_finalizer.csh ".$d." ".$obs." ".$band;
  Dada->logMsg(2, $dl, "changeSourceDirStructure: [".$h."] ".$cmd);
  ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd);
  Dada->logMsg(2, $dl, "changeSourceDirStructure: [".$h."] ".$result." ".$response);

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
# Find an observation to send, search chronologically. Look for observations that have 
# an obs.finished in them
#
sub getBandToSend() {

  Dada->logMsg(2, $dl, "getBandToSend()");

  my $obs_to_send = "none";
  my $band_to_send = "none";
  my $nfs_archives = $cfg{"SERVER_ARCHIVE_NFS_MNT"};

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @obs_finished = ();
  my $i = 0;
  my $j = 0;
  my $obs_pid = "";
  my $obs = "";
  my $band = "";
  my @bands = ();

  # Ensure the NFS directory is mounted
  $cmd = "ls -ld ".$nfs_archives;
  Dada->logMsg(2, $dl, "getBandToSend: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, $dl, "getBandToSend: ".$result.":".$response);

  # Look for all observations marked obs.finished in SERVER_NFS_RESULTS_DIR
  $cmd = "find ".$nfs_archives." -maxdepth 2 -name obs.finished ".
         "-printf \"%h\\n\" | sort | awk -F/ '{print \$NF}'";

  Dada->logMsg(2, $dl, "getBandToSend: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(3, $dl, "getBandToSend: ".$result.":".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "getBandToSend: ".$cmd." failed: ".$response);
    return ("none", "none");
  }

  # If there is nothing to transfer, simply return
  @obs_finished = split(/\n/, $response);
  if ($#obs_finished == -1) {
    return ("none", "none");
  }

  Dada->logMsg(2, $dl, "getBandToSend: found ".($#obs_finished+1)." observations marked obs.finised");

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && ($obs_to_send eq "none")); $i++) {

    $obs = $obs_finished[$i];
    Dada->logMsg(2, $dl, "getBandToSend: checking ".$obs);

    # skip if no obs.info exists
    if (!( -f $nfs_archives."/".$obs."/obs.info")) {
      Dada->logMsg(0, $dl, "Required file missing: ".$obs."/obs.info");
      next;
    }

    # Find the PID 
    $cmd = "grep ^PID ".$nfs_archives."/".$obs."/obs.info | awk '{print \$2}'";
    Dada->logMsg(3, $dl, "getBandToSend: ".$cmd);
    ($result, $response) = Dada->mySystem($cmd);
    Dada->logMsg(3, $dl, "getBandToSend: ".$result.":".$response);
    if (($result ne "ok") || ($response eq "")) {
      Dada->logMsgWarn($warn, "getBandToSend: failed to parse PID from obs.info [".$obs."] ".$response);
      next;
    }
    $obs_pid = $response;

    # If the PID doesn't match, skip it
    if ($obs_pid ne $pid) {
      Dada->logMsg(2, $dl, "getBandToSend: skipping ".$obs." PID mismatch [".$obs_pid." ne ".$pid."]");
      next;
    }

    # Get the sorted list of band nfs links for this obs
    # This command will timeout on missing NFS links (6 s), but wont print ones that are missing
    $cmd = "find -L ".$nfs_archives."/".$obs." -mindepth 1 -maxdepth 1 -type d -printf \"%f\\n\" | sort";
    Dada->logMsg(3, $dl, "getBandToSend: ".$cmd);
    ($result, $response) = Dada->mySystem($cmd);
    Dada->logMsg(3, $dl, "getBandToSend: ".$result.":".$response);

    if ($result ne "ok") {
      Dada->logMsgWarn($warn, "getBandToSend: failed to get list of NFS Band links ".$response);
      next;
    }

    @bands = split(/\n/, $response);
    Dada->logMsg(3, $dl, "getBandToSend: found ".($#bands+1)." bands in obs ".$obs);

    # See if we can find a band that matches
    for ($j=0; (($j<=$#bands) && ($obs_to_send eq "none")); $j++) {

      $band = $bands[$j];
      Dada->logMsg(3, $dl, "getBandToSend: checking ".$obs."/".$band);

      # If the remote NFS mount exists
      if (-f $nfs_archives."/".$obs."/".$band."/obs.start") {

        if (! -f $nfs_archives."/".$obs."/".$band."/sent.to.".$dest ) {
          Dada->logMsg(2, $dl, "getBandToSend: found ".$obs."/".$band);
          $obs_to_send = $obs;
          $band_to_send = $band;
        } else {
          Dada->logMsg(3, $dl, "getBandToSend: ".$obs."/".$band."/sent.to.".$dest." existed");
        }
      } else {
        Dada->logMsgWarn($warn, $obs."/".$band."/obs.start did not exist, or dir was not mounted");
      }
    }
  }

  Dada->logMsg(2, $dl, "getBandToSend: returning ".$obs_to_send.", ".$band_to_send);
  return ($obs_to_send, $band_to_send);
}

#
# Create the receiving directory on the remote machine
# and then run vsib_recv
#
sub run_vsib_recv($$$$) {

  my ($u, $h, $d, $port) = @_;

  my $cmd = "";
  my $ssh_cmd = "";
  my $pipe = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  # create the writing file
  $cmd = "touch ".$d."/../WRITING";
  Dada->logMsg(2, $dl, "run_vsib_recv [".$h."]: ".$cmd);
  ($result, $response) = Dada->remoteSshCommand($u, $h, $cmd);
  Dada->logMsg(2, $dl, "run_vsib_recv [".$h."]: ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "run_vsib_recv [".$h."]: could not touch remote WRITING file");
  }

  # sleeping to avoid immediately consecutive ssh commands
  sleep(1);

  # run vsib_recv 'manually' for the output piping to server_logger
  $cmd = "vsib_recv -q -1 -w ".TCP_WINDOW." -p ".$port." | tar -x";
  $ssh_cmd =  "ssh ".SSH_OPTS." -l ".$u." ".$h." \"cd ".$d."; ".$cmd."\" 2>&1 | ".$server_logger." -n vsib_recv";
  Dada->logMsg(2, $dl, "run_vsib_recv [".$h."]: ".$ssh_cmd);
  system($ssh_cmd);

  if ($? == 0) {
    $result = "ok";
  } elsif ($? == -1) {
    $result = "fail";
    Dada->logMsgWarn($error, "run_vsib_recv: failed to execute: $!");
  } elsif ($? & 127) {
    $result = "fail";
    Dada->logMsgWarn($error, "run_vsib_recv: child died with signal ".($? & 127));
  } else {
    $result = "fail";
    Dada->logMsg(1, $dl, "run_vsib_recv: child exited with value ".($? >> 8));
  }

  return $result;

  #Dada->logMsg(2, $dl, "run_vsib_recv remoteSshCommand(".$u.", ".$h.", ".$cmd.", ".$d.", ".$pipe.")");
  #($result, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d, $pipe);
  #Dada->logMsg(2, $dl, "run_vsib_recv [".$h."]: ".$result." ".$response);
  #if ($result ne "ok") {
  #  Dada->logMsgWarn($error, "run_vsib_recv [".$h."]: ssh failure ".
  #                           $response);
  #  return ("fail");
  #}

  #if ($rval != 0) {
  #  Dada->logMsgWarn($warn, "vsib_recv returned a non zero exit value");
  #  return ("fail");
  #} else {
  #  Dada->logMsg(2, $dl, "vsib_recv succeeded");
  #  return ("ok");
  #}

}

#
# Runs vsib_proxy on the localhost, it will connect to s_host upon receiving 
# an incoming connection
#
sub run_vsib_proxy($$) {

  my ($host, $port) = @_;

  Dada->logMsg(2, $dl, "run_vsib_proxy()");

  my $vsib_args = "-1 -q -w ".TCP_WINDOW." -p ".$port." -H ".$host;
  my $cmd = "vsib_proxy ".$vsib_args." 2>&1 | ".$server_logger." -n vsib_proxy";
  Dada->logMsg(2, $dl, $cmd);

  # Run the command on the localhost
  system($cmd);

  # Check return value for "success"
  if ($? != 0) {

    Dada->logMsgWarn($error, "run_vsib_proxy: command failed");
    return ("fail");

  # On sucesss, return ok
  } else {

    Dada->logMsg(2, $dl, "run_vsib_proxy: succeeded");
    return ("ok");

  }

}


sub interrupt_all() {

  Dada->logMsg(2, $dl, "interrupt_all()");

  # kill all potential senders
  Dada->logMsg(2, $dl, "interrupt_all: interrupt_vsib_send()");
  interrupt_vsib_send($send_user, $send_host);

  Dada->logMsg(2, $dl, "interrupt_all: interrupt_vsib_proxy()");
  interrupt_vsib_proxy();

  Dada->logMsg(2, $dl, "interrupt_all: interrupt_vsib_recv()");
  interrupt_vsib_recv($dest_user, $dest_host, $dest_dir, "none", "none");

}

#
# ssh to host specified and kill all vsib_send's
#
sub interrupt_vsib_send($$) {

  my ($u, $h) = @_;
 
  Dada->logMsg(2, $dl, "interrupt_vsib_send(".$u.", ".$h.")");
  
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $rval = 0;

  if (($send_user eq "none") || ($send_host eq "none")) {

    $result = "ok";
    $response = "vsib_send not running";

  } else {

    $cmd = "killall vsib_send";
    Dada->logMsg(2, $dl, "interrupt_vsib_send: ".$cmd);
    ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd);
    Dada->logMsg(2, $dl, "interrupt_vsib_send: ".$result." ".$rval." ".$response);

    if ($result  ne "ok") {
      Dada->logMsgWarn($warn, "interrupt_vsib_send: ssh failure ".$response);
    } else {
      if ($rval != 0) {
        Dada->logMsgWarn($warn, "interrupt_vsib_send: failed to kill vsib_send: ".$response);
      }
    }
  }

  return ($result, $response);
}


#
# ssh to host specified and kill all vsib_recv's, cleanup the obs
#
sub interrupt_vsib_recv($$$$$) {

  my ($u, $h, $d, $obs, $band) = @_;

  Dada->logMsg(2, $dl, "interrupt_vsib_recv(".$u.", ".$h.", ".$d.", ".$obs.", ".$band.")");

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  if ($h ne "none") {

    # kill vsib_recv
    $cmd = "killall vsib_recv";
    Dada->logMsg(2, $dl, "interrupt_vsib_recv: ".$cmd);
    ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd);
    if ($result  ne "ok") {
      Dada->logMsgWarn($warn, "interrupt_vsib_recv: ssh failure ".$response);
    } else {
      if ($rval != 0) {
        Dada->logMsgWarn($warn, "interrupt_vsib_recv: failed to kill vsib_recv: ".$response);
      }
    }

    # delete the WRITING flag
    $cmd = "rm -f ../WRITING";
    Dada->logMsg(2, $dl, "interrupt_vsib_recv: ".$cmd);
    ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d);
    if ($result  ne "ok") {
      Dada->logMsgWarn($warn, "interrupt_vsib_recv: ssh failure ".$response);
    } else {
      if ($rval != 0) {
        Dada->logMsgWarn($warn, "interrupt_vsib_recv: failed to remove WRITING flag: ".$response);
      } 
    }

    # optionally delete the transfer specied, this is a beam
    if (!(($obs eq "none") || ($obs eq ""))) {
      $cmd = "rm -rf ".$obs."/".$band;
      Dada->logMsg(2, $dl, "interrupt_vsib_recv: ".$cmd);
      ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd, $d);
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

sub interrupt_vsib_proxy() {

  Dada->logMsg(2, $dl, "interrupt_vsib_proxy()");
 
  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "killall vsib_proxy";
  Dada->logMsg(1, $dl, "interrupt_vsib_proxy: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "interrupt_vsib_proxy: ".$result." ".$response);

  return ($result, $response);
}

#
# run vsib_send  with tar -c obs feeding stdin to port to the recv_host
#
sub run_vsib_send($$$$$) {

  my ($obs, $band, $s_host, $r_host, $port) = @_;

  Dada->logMsg(2, $dl, "run_vsib_send(".$obs.", ".$band.", ".$s_host.
                       ", ".$r_host.", ".$port.")");

  my $cmd = "";
  my $ssh_cmd = "";
  my $pipe = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $s_dir = $cfg{"CLIENT_ARCHIVE_DIR"};

  $cmd .= "tar --exclude obs.finished -c ".$obs." | ";
  $cmd .= "vsib_send -q -w ".TCP_WINDOW." -p ".$port." -H ".$r_host." -z ".$rate;

  $send_user = $instrument;
  $send_host = $s_host;

  $ssh_cmd =  "ssh ".SSH_OPTS." -l ".$instrument." ".$s_host." \"cd ".$s_dir."; ".$cmd."\" 2>&1 | ".$server_logger." -n vsib_send";
  Dada->logMsg(2, $dl, "run_vsib_send [".$s_host."]: ".$ssh_cmd);
  system($ssh_cmd);

  if ($? == 0) {
    $result = "ok";
  } elsif ($? == -1) {
    $result = "fail";
    Dada->logMsgWarn($error, "run_vsib_send: failed to execute: $!");
  } elsif ($? & 127) {
    $result = "fail";
    Dada->logMsgWarn($error, "run_vsib_send: child died with signal ".($? & 127));
  } else {
    $result = "fail";
    Dada->logMsg(1, $dl, "run_vsib_send: child exited with value ".($? >> 8));
  }

  #Dada->logMsg(2, $dl, "run_vsib_send: remoteSshCommand(".$instrument.", ".$s_host.", ".$cmd.", ".$s_dir.", ".$pipe.")");
  #($result, $rval, $response) = Dada->remoteSshCommand($instrument, $s_host, $cmd, $s_dir, $pipe);
  #Dada->logMsg(2, $dl, "run_vsib_send: remoteSshCommand: ".$result." ".$rval." ".$response);

  $send_user = "none";
  $send_host=  "none";

  return $result;

  #if ($result ne "ok") {
  #  Dada->logMsgWarn($warn, "run_vsib_send: vsib_send failed ".$response);
  #  return("fail");
  #} else {
  #  return("ok");
  #}
}


#
# Write the current status into the /nfs/control/apsr/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada->logMsg(2, $dl, "setStatus(".$message.")");

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $dir = "/nfs/control/apsr";
  my $file = "xfer.state";

  $cmd = "rm -f ".$dir."/".$file;
  Dada->logMsg(2, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "setStatus: ".$result." ".$response);

  $cmd = "echo '".$message."' > ".$dir."/".$file;
  Dada->logMsg(2, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "setStatus: ".$result." ".$response);

  return ("ok", "");
    
}

#
# Transfer the specified obs/beam on port to hosts
#
sub transferBand($$$) {

  my ($obs, $band, $port) = @_;

  my $localhost = Dada->getHostMachineName();
  my $path      = $obs."/".$band;
  my $result = "";
  my $response = "";
  my $rval = 0;
  my $cmd = "";
  my $dirs = "";

  # Thread ID's
  my $send_thread = 0;
  my $proxy_thread = 0;
  my $recv_thread = 0;

  my $send_host = "";
  my $xfer_failure = 0;

  my $u = $dest_user;
  my $h = $dest_host;
  my $d = $dest_dir;

  Dada->logMsg(1, $dl, "Transferring ".$path." to ".$u."@".$h.":".$d);

  # Find the host on which the band directory is located
  $cmd = "find ".$path." -maxdepth 1 -printf \"\%l\" | awk -F/ '{print \$3}'";
  Dada->logMsg(2, $dl, "transferBand: ".$cmd);
  ($result, $response) = Dada->mySystem($cmd);
  Dada->logMsg(2, $dl, "transferBand: ".$result." ".$response);
  if ($result ne "ok") {
    Dada->logMsgWarn($error, "could not determine host on which ".$path." resides");
    return ("fail", "could not determine host on which ".$path." resides");
  }

  # host for vsib_send
  $send_host = $response;

  Dada->logMsg(2, $dl, "run_vsib_recv(".$u.", ".$h.", ".$d.", ".$port.")");
  $recv_thread = threads->new(\&run_vsib_recv, $u, $h, $d, $port);

  # If we are sending to swin, need to start a proxy on the localhost
  if ($dest eq "swin") {
    $proxy_thread = threads->new(\&run_vsib_proxy, $h, $port);
  }

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
      Dada->logMsg(1, $dl, "Waiting 2 seconds for receiver [".$h."] to be ready");
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
    interrupt_vsib_recv($u, $h, $d, $curr_obs, $curr_band);

  } else {

    sleep(2);

    # start the sender
    if ($proxy_thread) {
      Dada->logMsg(2, $dl, "run_vsib_send(".$obs.", ".$band.", ".$send_host.", ".$localhost.", ".$port.")");
      $send_thread = threads->new(\&run_vsib_send, $obs, $band, $send_host, $localhost, $port);
    } else {
      Dada->logMsg(2, $dl, "run_vsib_send(".$obs.", ".$band.", ".$send_host.", ".$h.", ".$port.")");
      $send_thread = threads->new(\&run_vsib_send, $obs, $band, $send_host, $h, $port);
    }

    # Transfer runs now
    Dada->logMsg(1, $dl, "Transfer now running: ".$path);

    # join the sending thread
    Dada->logMsg(2, $dl, "joining send_thread");
    $result = $send_thread->join();
    Dada->logMsg(2, $dl, "send_thread: ".$result);
    if ($result ne "ok") {
      $xfer_failure = 1;
      Dada->logMsgWarn($warn, "send_thread failed for ".$obs);
    }
  }

  # join the proxy thread, if it exists
  if ($proxy_thread) {
    $result = $proxy_thread->join();
    $proxy_thread = 0;
    Dada->logMsg(2, $dl, "proxy_thread: ".$result);
    if ($result ne "ok") {
      $xfer_failure = 1;
      Dada->logMsgWarn($warn, "proxy_thread failed for ".$path);
    }
  }

  # join the recv thread, if it exists
  Dada->logMsg(2, $dl, "joining recv_thread");
  $result = $recv_thread->join();
  $recv_thread = 0;
  Dada->logMsg(2, $dl, "recv_thread: ".$result);
  if ($result ne "ok") {
    $xfer_failure = 1;
    Dada->logMsgWarn($warn, "recv_thread failed for ".$path);
  }

  if ($xfer_failure) {
    Dada->logMsgWarn($warn, "Transfer error for obs: ".$path);
  } else {
    Dada->logMsg(1, $dl, "Transfer complete: ".$path);
  }

  # Increment the vsib port to prevent SO_REUSEADDR based issues
  $port++;
  if ($port == VSIB_MAX_PORT) {
    $port = VSIB_MIN_PORT;
  }

  return ($port, $xfer_failure);

} 


#
# check user@host:dir for connectivity and space
#
sub checkDestination($$$) {

  my ($u, $h, $d) = @_;

  my $result = "";
  my $rval = 0;
  my $response = "";
  my $cmd = "";

  $cmd = "df -B 1048576 -P ".$d." | tail -n 1 | awk '{print \$4}'";

  # test how much space is remaining on this disk
  ($result, $rval, $response) = Dada->remoteSshCommand($u, $h, $cmd);

  if ($result ne "ok") {
    return ("fail", "could not ssh to ".$u."@".$h.": ".$response);
  }

  # check there is 260 * 1GB free
  if (int($response) < (260*1024)) {
    return ("fail", "less than 260 GB remaining on ".$u."@".$h.":".$d);
  }

  # check if this is being used for reading
  $cmd = "ssh ".SSH_OPTS." -l ".$u." ".$h." \"ls ".$d."/../READING\" 2>&1";
  $result = `$cmd`;
  chomp $result;
  if ($result =~ m/No such file or directory/) {
    Dada->logMsg(1, $dl, "checkDestination: no READING file existed in ".$d."/../READING");
  } else {
    Dada->logMsg(1, $dl, "checkDestination: READING file existed in ".$d."/../READING");
    return ("fail", "Disk is being used for READING");
  }

  return ("ok", "");

}

sub markRemoteFile($$$$$) {

  my ($file, $user, $host, $dir, $subdir) = @_;
  Dada->logMsg(2, $dl, "markRemoteCompleted(".$file.", ".$user.", ".$host.", ".$dir.", ".$subdir.")");

  my $cmd = "cd ".$dir."; touch ".$subdir."/".$file;
  my $ssh_cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"".$cmd."\"";

  Dada->logMsg(2, $dl, "markRemoteCompleted: ".$cmd);
  my ($result, $response) = Dada->mySystem($ssh_cmd);
  Dada->logMsg(2, $dl, "markRemoteCompleted: ".$result." ".$response);

  if ($result ne "ok") {
    Dada->logMsgWarn($warn, "could not mark ".$host.":".$dir."/".$subdir." as completed");
  }

  return ($result, $response);
}


sub controlThread($$) {

  Dada->logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada->logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  # try to cleanly stop things
  if (!(($curr_obs eq "none") || ($curr_obs eq ""))) {
    Dada->logMsg(1, $dl, "controlThread: interrupting obs ".$curr_obs."/".$curr_band);
    interrupt_vsib_send($send_user, $send_host);
    interrupt_vsib_proxy();
    interrupt_vsib_recv($dest_user, $dest_host, $dest_dir, $curr_obs, $curr_band);
    sleep(2);

  } else {
    Dada->logMsg(1, $dl, "controlThread: not interrupting due to no current observation");
  }

  if ( -f $pid_file) {
    Dada->logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada->logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

} 


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

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada->getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada->getHostMachineName());
  }

  # ensure required 'external' variables have been set
  if (($server_logger eq "") || ($pid eq "") || ($dest eq "") || ($rate eq "")) {
    return ("fail", "Error: not all package globals were set");
  }

  # check the PID is a valid group for the specified instrument
  my @groups = ();
  my $i = 0;
  my $valid_pid = 0;
  @groups = Dada::getProjectGroups($instrument);
  for ($i=0; $i<=$#groups; $i++) {
    if ($groups[$i] eq $pid) {
      $valid_pid = 1;
    }  
  }
  if (!$valid_pid) {
    return ("fail", "Error: specified PID [".$pid."] was not a group for ".$instrument);
  }

  # check the dest is swin/parkes
  if (($dest ne "swin") && ($dest ne "parkes")) {
    return ("fail", "Error: configured dest [".$dest."] was not swin or parkes");
  }

  # check that only 1 destination has been configured
  if ($cfg{"NUM_".uc($dest)."_DIRS"} < 1) {
    return ("fail", "Error: ".$cfg{"NUM_".uc($dest)."_DIRS"}." were configured. Only 1 is permitted");
  }

  # get the user/host/dir for the specified destination
  ($dest_user, $dest_host, $dest_dir) = split(/:/,$cfg{uc($dest)."_DIR_".$dest_id});

  # if an optional dir has been configured on the command line
  if ($optdir ne "") {
    $dest_dir = $optdir;
  }

  my ($result, $response) = checkDestination($dest_user, $dest_host, $dest_dir);

  if ($result ne "ok") {
    return ("fail", "Error: ".$response);
  }

  return ("ok", "");
}


END { }

1;  # return value from file
