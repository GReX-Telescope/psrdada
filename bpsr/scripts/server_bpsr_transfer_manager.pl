#!/usr/bin/env perl

###############################################################################
#
# server_bpsr_transfer_manager.pl
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;  # Standard perl socket library
use IO::Select;  # Allows select polling on a socket
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Net::hostent;
use File::Basename;
use Bpsr;     # BPSR Module 
use strict;   # strict mode (like -Wall)
use threads;
use threads::shared;


#
# Constants
#
use constant DEBUG_LEVEL  => 1;
use constant PIDFILE   => "bpsr_transfer_manager.pid";
use constant LOGFILE   => "bpsr_transfer_manager.log";
use constant DATA_RATE => 38;
use constant TCP_WINDOW   => 1700;
use constant VSIB_PORT => 41000;
use constant SSH_OPTS  => "-x -o BatchMode=yes";


#
# Global Variables
#
our %cfg = Bpsr->getBpsrConfig();   # Bpsr.cfg
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

#
# Holding areas for later tape archival
#
my @swin_disks = ();
my @parkes_disks = ();
my @swin_fold_disks = ();
my @parkes_fold_disks = ();
my $i=0;

for ($i=0; $i<$cfg{"NUM_SWIN_DIRS"}; $i++) {
  push (@swin_disks, $cfg{"SWIN_DIR_".$i});
}

for ($i=0; $i<$cfg{"NUM_PARKES_DIRS"}; $i++) {
  push (@parkes_disks, $cfg{"PARKES_DIR_".$i});
}

for ($i=0; $i<$cfg{"NUM_SWIN_FOLD_DIRS"}; $i++) {
  push (@swin_fold_disks, $cfg{"SWIN_FOLD_DIR_".$i});
}

for ($i=0; $i<$cfg{"NUM_PARKES_FOLD_DIRS"}; $i++) {
  push (@parkes_fold_disks, $cfg{"PARKES_FOLD_DIR_".$i});
}


my ($s_user, $s_host, $s_dir);
my ($p_user, $p_host, $p_dir);

my $obs;
my $result;
my $response;

#
# Main
#

Dada->daemonize($logfile, $pidfile);

logMessage(0, "STARTING SCRIPT");

# Start the daemon control thread
$daemon_control_thread = threads->new(\&daemonControlThread);

my $swin_from_host = "150.229.108.237";
my @swin_recv_threads = ();
my @swin_recv_results = ();
my @swin_recv_responses = ();

my $parkes_from_host = "srv0";
my @parkes_recv_threads = ();
my @parkes_recv_results = ();
my @parkes_recv_responses = ();

my @send_threads = ();
my @send_results = ();
my @send_responses = ();

my $sent_to_swin = 0;
my $sent_to_parkes = 0;
my $swin_start_disk=0;
my $parkes_start_disk=0;

chdir $cfg{"SERVER_ARCHIVE_NFS_MNT"};

# At startup we should check that nothing is already happening by killing any
# processes that might be hanging round here or on the client.
killDisks(\@parkes_disks);
killLocal();


while (!$quit_daemon) {

# See if we can write to either place...
# This is important otherwise we can get stuck in a loop waiting for one place.
  ($s_user, $s_host, $s_dir) = findHoldingArea(\@swin_disks,$swin_start_disk);
 # $s_host="none"; # MJK TEST
    ($p_user, $p_host, $p_dir) = findHoldingArea(\@parkes_disks,$parkes_start_disk);
  logMessage(3, "Suitible hosts: $s_host, $p_host");

# Look for and observation (archives dir) that is ready to send.
  ($obs, $sent_to_swin, $sent_to_parkes) = getObsToSend($s_host ne "none", $p_host ne "none");
  if ($obs eq "none"){
#Try again, not requiring a swin send
    ($obs, $sent_to_swin, $sent_to_parkes) = getObsToSend(0, $p_host ne "none");
  }
  if ($obs eq "none"){
#Try again, not requiring a parkes send
    ($obs, $sent_to_swin, $sent_to_parkes) = getObsToSend($s_host ne "none",0);
  }

  logMessage(3, "Trying to send: $obs s:$sent_to_swin p:$sent_to_parkes");

  if ($obs ne "none") {

# If we have an observation to send, determine the swin and parkes
# holding areas to use
    logMessage(3, "Trying to send: $obs s:$sent_to_swin p:$sent_to_parkes");   

# Test to see if observation is not a survey pointing
    my $is_fold = checkIfFold($obs);

      $sent_to_swin=-1; #WVS disable sending to Swinburne
      if ($sent_to_swin == 0) {
        if ($is_fold){
#change this if disk 0 fails
          ($s_user, $s_host, $s_dir) = findHoldingArea(\@swin_fold_disks,0);
        } else { 
          ($s_user, $s_host, $s_dir) = findHoldingArea(\@swin_disks,$swin_start_disk);
          $swin_start_disk = ($swin_start_disk + 1) % ($#swin_disks+1);
        }
      } else {
        ($s_user, $s_host, $s_dir) = ("none", "none", "none");
      } 
    if ($sent_to_parkes == 0) {
      logMessage(3, "Trying to send to Parkes: ".$obs);
      if ($is_fold){
        ($p_user, $p_host, $p_dir) = findHoldingArea(\@parkes_fold_disks,0);
      } else {
        ($p_user, $p_host, $p_dir) = findHoldingArea(\@parkes_disks,$parkes_start_disk);
        $parkes_start_disk = ($parkes_start_disk + 1) % ($#parkes_disks+1);
      }
    } else {
      ($p_user, $p_host, $p_dir) =  ("none", "none", "none");
    }

# If at least one destination is available
    if (($p_host ne "none") || ($s_host ne "none")) {

      my %files = getFiles($obs);
      my @keys = keys (%files);
      my $key = "";

      my @threads = ();
      my @results = ();
      my @responses = ();

      logMessage(1, "Processing Observation: ".$obs);

# Launch vsib_sends on the server (running in server mode)
      logMessage(1, "vsib_send ".$obs." to ".$s_host.", ".$p_host);
      for ($i=0; $i<=$#keys; $i++) {
        $key = $keys[$i];
        logMessage(2, "run_vsib_send(".$s_host.", ".$p_host.", ".$obs.", ".$i.")");
        @send_threads[$i] = threads->new(\&run_vsib_send, $s_host, $p_host, $files{$key}, $i);
      }

# Launch vib_recv clients on parkes and swin hosts
      sleep(1);

      logMessage(1, "vsib_recv ".$obs." on ".$s_host.", ".$p_host);
      for ($i=0; $i<=$#keys; $i++) {
        $key = $keys[$i];

        if ($s_host ne "none") {
          logMessage(2, "run_vsib_recv(".$s_user.", ".$s_host.", ".$s_dir.", ".$swin_from_host.", ".$i.")");
          @swin_recv_threads[$i] = threads->new(\&run_vsib_recv, $s_user, $s_host, $s_dir, $swin_from_host, $key, $i);
        }

        if ($p_host ne "none") {
          logMessage(2, "run_vsib_recv(".$p_user.", ".$p_host.", ".$p_dir.", ".$parkes_from_host.", ".$i.")");
          @parkes_recv_threads[$i] = threads->new(\&run_vsib_recv, $p_user, $p_host, $p_dir, $parkes_from_host, $key, $i);
        }
    
        # sleep a small amount so that the ssh server doesn't get overwhelmed
        Time::HiRes::usleep(50000);
      }

      # Threads are all running now. join them all
      logMessage(1, "Threads launched, waiting for completion");

      for ($i=0; $i<=$#keys; $i++) {
        $key = $keys[$i];

        ($send_results[$i], $send_responses[$i]) = $send_threads[$i]->join;
        logMessage(2, "run_vsib_send: ".$send_results[$i].":".$send_responses[$i]);

        if ($s_host ne "none") {
          ($swin_recv_results[$i], $swin_recv_responses[$i]) = $swin_recv_threads[$i]->join;
          logMessage(2, "run_vsib_recv: ".$swin_recv_results[$i].":".$swin_recv_responses[$i]);
        }

        if ($p_host ne "none") {
          ($parkes_recv_results[$i], $parkes_recv_responses[$i]) = $parkes_recv_threads[$i]->join;
          logMessage(2, "run_vsib_recv: ".$parkes_recv_results[$i].":".$parkes_recv_responses[$i]);
        }
      }

      logMessage(1, "Threads joined, checking transfer");

      # Now test the results to ensure the data was transferred correctly
      if ($p_host ne "none") {

        ($result, $response) = checkRemoteArchive($p_user, $p_host, $p_dir, $obs);
        if ($result eq "ok") {
          logMessage(0, "Archiving to parkes successful");
          markArchiveCompleted("sent.to.parkes", $obs); 
        } else {
          logMessage(0, "Archiving to parkes failed: ".$response);
          markArchiveCompleted("error.to.parkes", $obs);
        }
      }

      if ($s_host ne "none") {
        ($result, $response) = checkRemoteArchive($s_user, $s_host, $s_dir, $obs);
        if ($result eq "ok") {
          logMessage(0, "Archiving to swin successful");
          markArchiveCompleted("sent.to.swin", $obs); 
        } else {
          logMessage(0, "Archiving to swin failed: ".$response);
          markArchiveCompleted("error.to.swin", $obs);
        }
      }

      # Chmod the remote directories to 777 to allow later moving to the on_tape dir
      

    } else {

      logMessage(0, "Obs ".$obs." is ready to send, but swin & parkes unavailable to receive");
    }

  }

  logMessage(2, "Sleeping 5 seconds");
  sleep(5);

}

# rejoin threads
$daemon_control_thread->join();

logMessage(0, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#


#
# Ensure the files sent matches the local
#
sub checkRemoteArchive($$$$) {

  my ($user, $host, $dir, $obs) = @_;


#find -L -maxdepth 2 \( ! -iname 'error.to*' ! -iname 'sent.to.*' ! -iname 'aux' ! -iname 'obs.*' -type f -printf '%p %s\n' \) | sort
  my $cmd = "find -L ".$obs." -maxdepth 2 \\(  ! -name '*.log' ! -iname 'error.to.*' ! -iname 'sent.to.*' ! -iname 'obs.finalized' ! -iname 'obs.info' ! -iname 'aux' -type f -printf '\%p \%s\\n' \\) | sort";

  logMessage(2, $cmd);
  my $local_list = `$cmd`;
  if ($? != 0) {
    logMessage(0, "find command failed: ".$local_list);
  }

  $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"cd ".$dir."; find -L ".$obs." \\( ! -iname 'error.to.*' ! -iname 'sent.to.*' -type f -printf '\%p \%s\\n' \\)\" | sort";
  logMessage(1, $cmd);
  my $remote_list = `$cmd`;

  if ($? != 0) {
    logMessage(0, "ssh command failed: ".$remote_list);
  }

  $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"cd ".$dir."; rm -f WRITING\"";
  logMessage(1, $cmd);
  system($cmd);

  if ($local_list eq $remote_list) {
    # Ok, so put a xfer.complete in the target dir
    # We have to chmod 777 it so that parkes can write the 'sent.to.tape' file.
    $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"cd ".$dir."; chmod 777 $dir; chmod 777 $obs; chmod 777 $obs/*/ ; touch $obs/xfer.complete\"";
    logMessage(2, $cmd);
    my ($result, $response) = Dada->mySystem($cmd);

    return ($result,$response);
  } else {

    logMessage(0, "ARCHIVE MISMATCH: local: ".$local_list);
    logMessage(0, "ARCHIVE MISMATCH: remote: ".$remote_list);

    return ("fail", "archive mismatch");
  }

}

sub markArchiveCompleted($$) {

  my ($file, $obs) = @_;

  my $cmd = "touch ".$obs."/".$file;
  logMessage(2, "markArchiveCompleted: ".$cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  logMessage(2, "markArchiveCompleted: ".$result." ".$response);

  return ($result, $response);
}


sub findHoldingArea(\@$) {

  (my $disks_ref, my $startdisk) = @_;

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

  for ($c=0; $c<=$#disks; $c++) {
    $i = ($c + $startdisk)%($#disks+1);

    $disk = $disks[$i];
    logMessage(2, "Evaluating ".$path);

    $user = "";
    $host = "";
    $path = "";

    @disk_components = split(":",$disk,3);

    if ($#disk_components == 2) {

      $user = $disk_components[0];
      $host = $disk_components[1];
      $path = $disk_components[2];


# check for disk space on this disk
      $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"ls ".$path."\" 2>&1";
      logMessage(2, $cmd);
      $result = `$cmd`;

      if ($? != 0) {
        chomp $result;
        logMessage(0, "ssh cmd '".$cmd."' failed: ".$result);
        $result = "";
      } else {

# check if this is being used for reading
        $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"ls ".$path."/READING\" 2>&1";
        $result = `$cmd`;
        chomp $result;
        if ($result =~ m/No such file or directory/) {
#There is no READING file.
          $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"df ".$path." -P\" | tail -n 1";
          logMessage(2, $cmd);
          $result = `$cmd`;
          if ($? != 0) {
            logMessage(0, "df command ".$cmd." failed: ".$result);
            $result = "";
          } 
        } else {
# we are writing to the disk
          logMessage(0,"Skipping disk $host:$path as the disk is being used for reading");
          $result="";
        }
      }
    } else {

      logMessage(0, "disk line syntax error ".$disk);
      $result = "";

    }

    if ($result ne "") {

      chomp($result);

      logMessage(2, "df_result  = $result");

      if ($result =~ m/No such file or directory/) {

        logMessage(0, "Error: ".$user." ".$host." ".$path." was not a valid directory");
        $result = "";

      } else {

        my ($location, $total, $used, $avail, $junk) = split(" ",$result);
        logMessage(2, "used = $used, avail = $avail, total = $total");

        if (($avail / $total) < 0.1) {

          logMessage(0, "Warning: ".$host.":".$path." is over 90% full");

        } else {

# Need more than 10 Gig
          if ($avail < 10000) {

            logMessage(0, "Warning: ".$host.":".$path." has less than 10 GB left");

          } else {
            logMessage(1, "Holding area: ".$user." ".$host." ".$path);
            return ($user,$host,$path); 
          }
        }
      }
    }
  }

  return ("none", "none", "none");

}


#
# Search the archives directory for an observation that is ready to send to
# Swinburne
#
sub getObsToSend($$) {

  my ($accept_swin, $accept_parkes) = @_;

  logMessage(2, "getObsToSend()");

  my $archives_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};

# produce a "unixtime filename" list of all the obs.finalized
  my $cmd = "find ".$archives_dir." -name \"obs.finalized\" -printf \"%T@ %f\\n\" | sort -r";

  my $subdir = "";
  my @subdirs = ();

# get all sub directories only
  opendir(DIR,$archives_dir);
  @subdirs = sort grep { !/^\./ && -d $archives_dir."/".$_ } readdir(DIR);
  closedir DIR;

  my $obs_finalized = 0;
  my $sent_to_swin = 0;
  my $sent_to_parkes = 0;
  my $fil_count = 0;
  my $tar_count = 0;
  my $xml_count = 0;
  my $int_archive_count = 0;
  my $obs_start_count = 0;

  my $dir = "";
  my $candidate = "none";
  my $parkes_error_count = 0;
  my $swin_error_count=0;
  my $expected_count=0;

# For each observation
  my $i=0;
  for ($i=0; (($i<=$#subdirs) && (!$quit_daemon)); $i++) {

    $expected_count = $cfg{"NUM_PWC"};
    $obs_finalized = 0;
    $sent_to_swin = 0;
    $sent_to_parkes = 0;

    $obs = $subdirs[$i];
    $dir = $archives_dir."/".$obs;

# ensure that all the NFS beam directories are mounted
    $cmd = "ls ".$dir."/* >& /dev/null";
    logMessage(3, $cmd);
    ($result, $response) = Dada->mySystem($cmd);
    logMessage(3, $result.":".$response);

    if (-f $dir."/obs.finalized") {
      $obs_finalized = 1;
    }
    if (-f $dir."/sent.to.swin") {
      $sent_to_swin = 1;
    }
    if (-f $dir."/sent.to.parkes") {
      $sent_to_parkes = 1;
    }
    if (-f $dir."/error.to.swin") {
      $sent_to_swin = -1;
    }
    if (-f $dir."/error.to.parkes") {
      $sent_to_parkes = -1;
    }

    $cmd = "find -L ".$dir." -maxdepth 2 -name 'integrated.ar' | wc | awk '{print \$1}'";
    $int_archive_count = `$cmd`;
    chomp $int_archive_count;

    $cmd = "find -L ".$dir." -maxdepth 2 -name 'aux.tar' | wc | awk '{print \$1}'";
    $tar_count = `$cmd`;
    chomp $tar_count;

    $cmd = "find -L ".$dir." -maxdepth 2 -name '*.fil' | wc | awk '{print \$1}'";
    $fil_count = `$cmd`;
    chomp $fil_count;

    $cmd = "find -L ".$dir." -maxdepth 2 -name 'obs.start' | wc | awk '{print \$1}'";
    $obs_start_count = `$cmd`;
    chomp $obs_start_count;

    $cmd = "find -L ".$dir." -maxdepth 2 -name '*.psrxml' | wc | awk '{print \$1}'";
    $xml_count = `$cmd`;
    chomp $xml_count;

    logMessage(3, "dir ".$dir.", obs ".$obs.",  finalized ".$obs_finalized.", swin ".$sent_to_swin."/".$accept_swin.", parkes ".$sent_to_parkes."/".$accept_parkes." [$tar_count,$fil_count,$obs_start_count,$xml_count] /".$cfg{"NUM_PWC"});

    if ( $int_archive_count > 0) {
      logMessage(3, "Skipping $obs, was a test pulsar obs");
      next;
    }

    if ( not ($obs_finalized)){
      logMessage(3,"Skipping $obs, observation not 'finalized'");
      next;
    }
    if ( not (((($sent_to_swin == 0 && $accept_swin) || not $accept_swin)) && ((($sent_to_parkes == 0 && $accept_parkes) || not $accept_parkes)))){
      logMessage(3,"Skipping $obs, observation already sent to servers we have avaliable.");
      next;
    }
    if (not (($sent_to_swin==0)||($sent_to_parkes==0))){
      logMessage(3,"Skipping $obs, observation already sent to all servers.");
      next;
    }

    if (not ($fil_count == $expected_count)){
      logMessage(3,"Skipping $obs, number of *.fil files is $fil_count, expected $expected_count");
      next;
    }
    if (not ($tar_count == $expected_count)){
      logMessage(3,"Skipping $obs, number of aux.tar files is $tar_count, expected $expected_count");
      next;
    }
    if (not ($xml_count == $expected_count)){
      logMessage(3,"Skipping $obs, number of *.psrxml files is $xml_count, expected $expected_count");
      next;
    }
    if (not ($obs_start_count == $expected_count)){
      next;
    }

# If all the above tests pass. this is a valid candidate for 'XFER'
    $candidate=$obs;
# break out of the loop.
    last;
  }
  if($swin_error_count){
    logMessage(0,"Warning: there were $swin_error_count files skipped to Swin as they failed to transfer sucessfuly");
  }
  if($parkes_error_count){
    logMessage(0,"Warning: there were $parkes_error_count files skipped to Parkes as they failed to transfer sucessfuly");
  }
  logMessage(2, "getObsToSend: ".$candidate.", ".$sent_to_swin." ",$sent_to_parkes);

  return ($candidate, $sent_to_swin, $sent_to_parkes);

}


sub run_vsib_recv($$$$$$) {

  my ($s_user, $s_host, $s_dir, $f_host, $obs_dir, $index) = @_;

  my $cmd = "ssh ".SSH_OPTS." -l ".$s_user." ".$s_host." \"cd ".$s_dir."; touch WRITING\"";
  logMessage(2, $cmd);
  system($cmd);


  my $remote_cmd = "vsib_recv -w ".TCP_WINDOW." -p ".(VSIB_PORT + $index)." -H ".$f_host." >>& transfer.log".$index;
  my $cmd =  "ssh ".SSH_OPTS." -l ".$s_user." ".$s_host." \"cd ".$s_dir."; mkdir -p ".$obs_dir."; ".$remote_cmd."\"";

  logMessage(2, $cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  logMessage(3, "vsib_recv: ".$result." ".$response);

  if ($result ne "ok") {
    logMessage(0, "vsib_recv returned a non zero exit value");
  }

  return ($result, $response);

}


sub run_vsib_send($$$$) {

  my ($s_host, $p_host, $files, $index) = @_;

  if (($p_host eq "none") && ($s_host eq "none")) {
    return ("fail", "no output requested");
  }

  my $bindir = Dada->getCurrentBinaryVersion();

  my $vsi_hosts = "\"".$p_host." ".$s_host."\"";

  if ($p_host eq "none") {
    $vsi_hosts = $s_host;
  } elsif ($s_host eq "none") {
    $vsi_hosts = $p_host;
  } else {
# normal
  }


  my $cmd = "vsib_send -s -q -w ".TCP_WINDOW." -p ".(VSIB_PORT+$index)." -H ".$vsi_hosts.
    " -z ".DATA_RATE." ".$files;

  logMessage(2, $cmd);

  $cmd .= " 2>&1";

  my $result = "fail";
  my $response = "";

  system($cmd);

  if ($? != 0) {
    logMessage(0, "vsib_send returned a non-zero exit value");
  } else {
    $result = "ok";
    $response = "";
  }

  return ($result, $response);

}

sub getFiles($) {

  (my $obs) = @_;

  my $archives_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $dir = $archives_dir."/".$obs;

  my $subdir = "";
  my @subdirs = ();
  my %files = ();
  my $fil_file = "";
  my $tar_file = "";
  my $xml_file = "";
  my $obs_start_file = "";

# get all sub directories only
  opendir(DIR,$dir);
  @subdirs = sort grep { !/^\./ && -d $dir."/".$_ } readdir(DIR);
  closedir DIR;

  logMessage(3, "Getting files in $dir");

# For each sub dir
  my $i=0;
  for ($i=0; $i<=$#subdirs; $i++) {
    $subdir = $subdirs[$i];

    my $cmd = "find -L ".$obs."/".$subdir." -name \"*.fil\" -printf \"%f\"";
    logMessage(3, $cmd);
    $fil_file = `$cmd`;
    if ($? != 0) {
      logMessage(0, "getFiles: no .fil file  in ".$obs."/".$subdir);
      $fil_file = "";
    }

    $tar_file = $obs."/".$subdir."/aux.tar";
    if (! -f $tar_file) {
      logMessage(0, "getFiles: no .tar file in ".$obs."/".$subdir);
      $tar_file = "";
    } else {
      $tar_file = "aux.tar";
    }

    $cmd = "find -L ".$obs."/".$subdir." -name \"*.psrxml\" -printf \"%f\"";
    logMessage(3, $cmd);
    $xml_file = `$cmd`;
    if ($? != 0) {
      logMessage(0, "getFiles: no .psrxml file  in ".$obs."/".$subdir);
      $fil_file = "";
    }

    $obs_start_file = $obs."/".$subdir."/obs.start";
    if (! -f $obs_start_file) {
      logMessage(0, "getFiles: no obs.start file in ".$obs."/".$subdir);
      $obs_start_file = "";
    } else {
      $obs_start_file = "obs.start";
    }

    if (($tar_file ne "") && ($fil_file ne "") && ($xml_file ne "") && ($obs_start_file ne "")) {
      $files{$obs."/".$subdir} = $obs."/".$subdir."/".$fil_file." ".$obs."/".$subdir."/".$tar_file." ".$obs."/".$subdir."/".$xml_file." ".$obs."/".$subdir."/".$obs_start_file;
    }

  }
  return %files;

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
        killDisks(\@parkes_disks);
        killLocal();

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

}

sub killAndCleanExisting($$$) {

  my ($s_user, $s_host, $s_dir) = @_;

  my $cmd =  "ssh ".SSH_OPTS." -l ".$s_user." ".$s_host." \"server_bpsr_transfer_killer.sh ".$s_dir."\"";

  logMessage(2, $cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  logMessage(3, "server_bpsr_transfer_killer.sh: ".$result." ".$response);

  if ($result ne "ok") {
    logMessage(0, "server_bpsr_transfer_killer returned a non zero exit value");
  }

  return ($result, $response);

}

sub killLocal(){
  my $cmd = "server_bpsr_transfer_kill_server.sh";
  logMessage(2, $cmd);
  my ($result, $response) = Dada->mySystem($cmd);
  logMessage(3, "server_bpsr_transfer_kill_server.sh: ".$result." ".$response);

  if ($result ne "ok") {
    logMessage(0, "server_bpsr_transfer_kill_server returned a non zero exit value");
  }

  return ($result, $response);
}

sub killDisks(\@) {

  (my $disks_ref) = @_;

  my @disks = @$disks_ref;
  my @disk_components = ();

  my $i=0;
  my $disk = "";
  my $user = "";
  my $host = "";
  my $path = "";
  my $cmd = "";
  my $result = "";

  for ($i=0; $i<=$#disks; $i++) {

    my $disk = $disks[$i];
    logMessage(2, "Evaluating ".$path);

    $user = "";
    $host = "";
    $path = "";

    @disk_components = split(":",$disk,3);

    if ($#disk_components == 2) {

      $user = $disk_components[0];
      $host = $disk_components[1];
      $path = $disk_components[2];
      ($result,$response) = killAndCleanExisting($user,$host,$path);

    }
  }
  return;
}


#
# This checks if the data is a real survey pointing or a pulsar
#
sub checkIfFold($){
  my $obs = shift;
  my $is_fold = 0;
#$obs/01/\*.fil
  my $cmd="sed -e 's:^.*<source_name_centre_beam> *\(.\)[^ ]* *</source_name_centre_beam>.*\$:\1:p' -e '/.*/d' $obs/01/*.psrxml";
  my ($result, $response) = Dada->mySystem($cmd);
  
  if ($result==0 && $response!=""){
    if ($response eq 'G'){
      $is_fold = 0;
    } else {
      $is_fold = 1;
    }
  }
  
  logMessage(2, "checkIfFold(): For $obs is_fold=".$is_fold);

  return $is_fold;
}
