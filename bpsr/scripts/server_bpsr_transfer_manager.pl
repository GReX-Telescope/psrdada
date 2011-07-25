#!/usr/bin/env perl
###############################################################################
#
# server_bpsr_transfer_manager.pl
#
# Transfers obsevation/beam to the swinburne and parkes holding areas for 
# subsequent tape archival
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Required Modules
#
use strict;
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use File::Basename;
use threads;
use threads::shared;
use Thread::Queue;
use Bpsr;

sub usage() {
  print "Usage: ".basename($0)." PID\n";
  print "   PID   Project ID. The bpsr user must be a member of this unix group\n";
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant BANDWIDTH     => "76800";  # 75 MB/s total bandwidth
use constant PIDFILE       => "bpsr_transfer_manager.pid";
use constant LOGFILE       => "bpsr_transfer_manager.log";
use constant QUITFILE      => "bpsr_transfer_manager.quit";

#
# Global Variables
#
our $dl = 1;
our %cfg = Bpsr::getConfig();   # Bpsr.cfg
our $error = $cfg{"STATUS_DIR"}."/bpsr_transfer_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/bpsr_transfer_manager.warn";
our $quit_daemon : shared  = 0;
our $pid : shared = "";
our $r_user = "bpsr";
our $r_host = "raid0";
our $r_path = "/lfs/raid0/bpsr/finished";
our $r_module = "bpsr_upload";
our $num_p_threads = 2;


# clear the error and warning files if they exist
if ( -f $warn ) {
  unlink ($warn);
}
if ( -f $error) {
  unlink ($error);
}

# Autoflush output
$| = 1;

# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

if ($#ARGV != 0) {
  usage();
  exit(1);
}

$pid  = $ARGV[0];

#
# Local Varaibles
#
my $logfile = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pidfile = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
my $control_thread = 0;
my $pid_report_thread = 0;
my $obs = "";
my $beam = "";
my $result;
my $response;
my $i=0;
my @bits = ();

#
# CASPSR Raid staging area
#
my @p_disks = ();
for ($i=0; $i<$cfg{"NUM_PARKES_DIRS"}; $i++) {
  push (@p_disks, $cfg{"PARKES_DIR_".$i});
}

#
# Main
#
Dada::daemonize($logfile, $pidfile);

Dada::logMsg(0, $dl, "STARTING SCRIPT PID=".$pid." BANDWDITH=".sprintf("%2.0f", (BANDWIDTH / 1024))." MB/s");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread);

# Start the PID reporting thread
$pid_report_thread = threads->new(\&pidReportThread, $pid);

setStatus("Starting script");

my $p_i = 0;

# Ensure that destination directories exist for this project
($result, $response) = checkDest();
if ($result ne "ok") {
  $quit_daemon = 1;
  sleep(3);
  exit(1);
}

chdir $cfg{"SERVER_ARCHIVE_NFS_MNT"};

Dada::logMsg(1, $dl, "main: checking for fully finished observations");
($result, $response) = checkFullyFinished();
Dada::logMsg(2, $dl, "main: checkFullyFinished(): ".$result.":".$response);

Dada::logMsg(1, $dl, "main: checking for fully transferred observations");
($result, $response) = checkFullyTransferred();
Dada::logMsg(2, $dl, "main: checkFullyTransferred(): ".$result.":".$response);

Dada::logMsg(1, $dl, "main: checking for fully archived observations");
($result, $response) = checkFullyArchived();
Dada::logMsg(2, $dl, "main: checkFullyArchived(): ".$result.":".$response);

Dada::logMsg(1, $dl, "main: checking for fully deleted observations");
($result, $response) = checkFullyDeleted();
Dada::logMsg(2, $dl, "main: checkFullyDeleted(): ".$result.":".$response);

my $p_in = new Thread::Queue;
my $p_out = new Thread::Queue;
my $p_count = 0;


# start the Transfer threads
my @p_threads = ();

for ($i=0; $i<$num_p_threads; $i++)
{
  push (@p_threads, 0);
  $p_threads[$i] = threads->new(\&transferThread, $p_in, $p_out, $i);
}

###############################################################################
#
# Main loop
#

Dada::logMsg(1, $dl, "starting main loop");

my $host = "";
my $obs = "";
my $beam = "";
my $input_string = "";
my $output_string = "";

my @nodes_in_use = ();
my @tmp_array = ();
my @obs_to_check = ();

my $xfer_count = 0;
my $p_sleep = 0;

while ((!$quit_daemon) || ($p_count > 0))
{

  Dada::logMsg(2, $dl, "main: quit_daemon=".$quit_daemon.", p_count=".$p_count);
  
  # Look for observations to send to staging area
  if ((!$quit_daemon) && ($p_count < $num_p_threads) && ($p_sleep <= 0))
  {

    # Check that the destination is available
    ($result, $response) = checkDest();

    if ($result eq "ok")
    {
      $p_i = ($p_i + 1) % ($#p_disks+1);
      # find and obs/beam to send to parkes, mark beam as obs.transferring...
      Dada::logMsg(2, $dl, "main: getBeamToSend()");
      ($result, $host, $obs, $beam) = getBeamToSend(\@nodes_in_use);
      Dada::logMsg(2, $dl, "main: getBeamToSend() ".$result." ".$host." ".$obs." ".$beam);

      if (($result eq "ok") && ($host ne "none") && ($obs ne "none") && ($beam ne "none"))
      {
        $input_string = $host." ".$obs." ".$beam;
        Dada::logMsg(2, $dl, "main: enqueue parkes [".$input_string."]");
        $p_in->enqueue($input_string);
        push (@nodes_in_use, $host);
        $p_count++;
        $p_sleep = 2;
      }
      else
      {
        $p_sleep = 60;
      }
    }
  }
  if ($p_sleep > 0)
  {
    $p_sleep--;
  }

  @obs_to_check = ();
  
  # if a beam has finished transfer, remove it from the queue
  if ($p_out->pending())
  {
    Dada::logMsg(2, $dl, "main: dequeuing transfer");
    $output_string = $p_out->dequeue();
    @bits = ();
    @bits = split(/ /,$output_string);
    if ($#bits != 3)
    {
      Dada::logMsgWarn($warn, "main: dequeue string did not have 4 parts [".$output_string."]");
      $quit_daemon = 1;
    }
    $result = $bits[0];
    $host = $bits[1];
    $obs = $bits[2];
    $beam = $bits[3];

    Dada::logMsg(2, $dl, "main: dequeued parkes xfer ".$obs."/".$beam." from ".$host. "[".$result."]");

    # decrement the parkes transfer count
    $p_count--;

    # remove this host from the nodes in use array
    @tmp_array = ();
    for ($i=0; $i<$#nodes_in_use; $i++)
    {
      if ($nodes_in_use[$i] != $host)
      {
        push (@tmp_array, $nodes_in_use[$i]);
      }
    }
    @nodes_in_use = ();
    for ($i=0; $i<$#tmp_array; $i++)
    {
      push (@nodes_in_use, $tmp_array[$i]);
    }
    $xfer_count++;

    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: xfer for ".$obs."/".$beam." failed");
      $quit_daemon = 1;
    } 
    else
    {
      push (@obs_to_check, $obs);
    }
  }

  # for any recent transferred obs/beams, check if all the 
  # required xfers have been done
  for ($i=0; $i<=$#obs_to_check; $i++)
  {
    $obs = $obs_to_check[$i];

    if ( ! -f $obs."/obs.transferred") 
    {

      Dada::logMsg(2, $dl, "main: checking ".$obs." finished -> transferred");

      # Check if all beams have been transferred successfully, if so, mark 
      # the observation as sent.to.dest
      Dada::logMsg(2, $dl, "main: checkAllBeams(".$obs.")");
      ($result, $response) = checkAllBeams($obs);
      Dada::logMsg(2, $dl, "main: checkAllBeams: ".$result." ".$response);

      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "main: checkAllBeams failed: ".$response);
      }
      else
      {
        if ($response ne "all beams sent") 
        {
          Dada::logMsg(2, $dl, "Obs ".$obs." not fully transferred: ".$response);
        } 
        else
        {
          Dada::logMsg(0, $dl, $obs." finished -> transferred");
          touchLocalFile($obs, "obs.transferred");
          unlink ($obs."/obs.finished");
        }
      }
    }
  }

  if ((($#obs_to_check >= 0) && ($xfer_count % 13 == 0)) || ($p_sleep == 59))
  {
    Dada::logMsg(2, $dl, "main: observations otc=".$#obs_to_check." xfer_count=".$xfer_count." mod=".($xfer_count % 13)." p_s=".$p_sleep);

    Dada::logMsg(2, $dl, "main: checking for fully finished observations");
    ($result, $response) = checkFullyFinished();
    Dada::logMsg(2, $dl, "main: checkFullyFinished(): ".$result.":".$response);

    Dada::logMsg(2, $dl, "main: checking for fully transferred observations");
    ($result, $response) = checkFullyTransferred();
    Dada::logMsg(2, $dl, "main: checkFullyTransferred(): ".$result.":".$response);

    Dada::logMsg(2, $dl, "main: checking for fully archived observations");
    ($result, $response) = checkFullyArchived();
    Dada::logMsg(2, $dl, "main: checkFullyArchived(): ".$result.":".$response);

    Dada::logMsg(2, $dl, "main: checking for fully deleted observations");
    ($result, $response) = checkFullyDeleted();
    Dada::logMsg(2, $dl, "main: checkFullyDeleted(): ".$result.":".$response);
  }

  setStatus($p_count." threads transferring");

  Dada::logMsg(2, $dl, "main: sleep(1)"); 
  sleep(1);

}

# rejoin threads
Dada::logMsg(1, $dl, "main: joining controlThread");
$control_thread->join();

# rejoin threads
Dada::logMsg(1, $dl, "main: joining pid_report_thread");
$pid_report_thread->join();

for ($i=0; $i<$num_p_threads; $i++)
{
  if ($p_threads[$i] != 0)
  {
    Dada::logMsg(1, $dl, "main: joining p_thread[".$i."]");
    $p_threads[$i]->join();
  }
}

setStatus("Script stopped");
Dada::logMsg(0, $dl, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#

#
#  Generic Transfer Thread, dequeues a transfer request from the specified 
#  queue and transfer the beam to the decoded destination, cheking for XFER
#  success on the way
#
sub transferThread($$$)
{
  my ($in, $out, $tid) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $quit_wait = 10;
  my @bits = ();
  my $transfer_result = 0;
  my $rval = 0;
  my $l_dir = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $bwlimit = BANDWIDTH / $num_p_threads;
  my $rsync_options = "-a --stats --password-file=/home/bpsr/.ssh/raid0_rsync_pw --no-g --chmod=go-ws ".
                      "--exclude 'aux' --exclude 'beam.finished' --exclude 'beam.transferring' --bwlimit=".$bwlimit;

  my $h = "";
  my $o = "";
  my $b = "";

  my @output_lines = ();
  my $data_rate = "";
  my $i = 0;

  Dada::logMsg(2, $dl, "xfer[".$tid."] starting");

  while (!$quit_daemon) 
  {
    Dada::logMsg(2, $dl, "xfer[".$tid."] while(!quit_daemon)");

    while ($in->pending) 
    {
      Dada::logMsg(2, $dl, "xfer[".$tid."] while(!in->pending)");

      if ($quit_daemon) 
      {
        if ($quit_wait > 0) 
        {
          Dada::logMsg(2, $dl, "xfer[".$tid."]: quit_daemon=1 while in->pending=true, waiting...");
          $quit_wait--;
        } 
        else 
        {
          Dada::logMsg(0, $dl, "xfer[".$tid."]: quit_daemon=1 while in->pending=true, quitting!");
          return 0;
        }
      }

      # try to dequeue a input transfer string from the queue
      Dada::logMsg(2, $dl, "xfer[".$tid."] calling in->dequeue");
      $input_string = $in->dequeue_nb();
      Dada::logMsg(2, $dl, "xfer[".$tid."] in->dequeue returns");
  
      # if we missed the dequeue, then return
      if ($input_string eq undef) 
      {
        Dada::logMsg(2, $dl, "xfer[".$tid."] missed the dequeue, polling again");
        next;
      }

      Dada::logMsg(2, $dl, "xfer[".$tid."] dequeued input string '".$input_string."'");

      # decode the transfer parameters
      @bits = ();
      @bits = split(/ /,$input_string);

      # ensure we decoded 3 elements
      if ($#bits != 2)
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] could not decode ".$input_string." into 3 params");
      }

      $h = $bits[0];
      $o = $bits[1];
      $b = $bits[2];

      # create the observation directory on the destination
      $cmd = "mkdir -m 0755 -p ".$r_path."/".$pid."/".$o;
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$r_host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] mkdir failed: ".$response); 
      }

      Dada::logMsg(1, $dl, "xfer[".$tid."] ".$o."/".$b." transferring");

      $cmd = "rsync ".$l_dir."/".$o."/".$b." ".$r_user."@".$r_host."::".$r_module."/".$pid."/".$o."/ ".$rsync_options;

      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $h, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);

      if (($result eq "ok") && ($rval == 0))
      {
        $transfer_result = 1; 
        # determine the data rate
        @output_lines = split(/\n/, $response);
        $i = 0;
        for ($i=0; $i<=$#output_lines; $i++)
        {
          if ($output_lines[$i] =~ m/bytes\/sec/)
          {
            @bits = split(/[\s]+/, $output_lines[$i]);
            $data_rate = sprintf("%3.0f", ($bits[6] / 1048576))." MB/s";
          }
        }


        # update the local file flags
        touchLocalFile($o."/".$b, "sent.to.parkes");
        touchLocalFile($o."/".$b, "sent.to.swin");
        touchLocalFile($o."/".$b, "beam.transferred");
  
        # update the remote file flags
        touchRemoteFile($r_user, $r_host, $r_path."/".$pid."/".$o."/".$b, "beam.transferred");

        Dada::logMsg(1, $dl, "xfer[".$tid."] ".$o."/".$b." finished -> transferred ".$data_rate);
      }
      else
      {
        $transfer_result = 0; 
        if ($result ne "ok")
        {
          Dada::logMsg(0, $dl, "xfer[".$tid."] ssh failed: ".$response);
        }
        if ($rval != 0)
        {
          Dada::logMsg(0, $dl, "xfer[".$tid."] rsync failed: ".$response); 
        }
      }

      if ($transfer_result)
      {
        $output_string = "ok ".$h." ".$o." ".$b;
      }
      else
      {
        $output_string = "fail ".$h." ".$o." ".$b;

        Dada::logMsgWarn($warn, "xfer[".$tid."] failed to transfer ".$o."/".$b);

        $cmd = "mv ".$o."/".$b."/obs.start ".$o."/".$b."/obs.bad";
        Dada::logMsg(1, $dl, "xfer[".$tid."]: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(1, $dl, "xfer[".$tid."]: ".$result." ".$response);

        touchLocalFile($o."/".$b, "error.to.parkes");
        touchLocalFile($o."/".$b, "error.to.swin");

        Dada::logMsg(1, $dl, "xfer[".$tid."] ".$o."/".$b." finished -> error");
      }

      Dada::logMsg(2, $dl, "xfer[".$tid."]: dequeueing ".$o."/".$b);

      if ( -f $o."/".$b."/beam.transferring" )
      {
        Dada::logMsg(2, $dl, "xfer[".$tid."]: unlinking ".$o."/".$b."/beam.transferring");
        unlink ($o."/".$b."/beam.transferring");
      }
      else 
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] ".$o."/".$b."/beam.transferring missing");
      }
      
      $out->enqueue($output_string);
    }

    sleep(1);
  }

  Dada::logMsg(2, $dl, "xfer[".$tid."]: exiting");
}

sub touchLocalFile($$) 
{
  my ($dir, $file) = @_;

  my $cmd = "sudo -u bpsr touch ".$dir."/".$file;
  Dada::logMsg(2, $dl, "touchLocalFile: ".$cmd);
  my ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "touchLocalFile: ".$result." ".$response);

  return ($result, $response);
}

sub touchRemoteFile($$$$)
{
  my ($u, $h, $d, $f) = @_;

  my $cmd = "touch ".$d."/".$f;
  Dada::logMsg(2, $dl, "touchRemoteFile: ".$u."@".$h.": ".$cmd);
  my ($result, $rval, $response) = Dada::remoteSshCommand($u, $h, $cmd);
  Dada::logMsg(2, $dl, "touchRemoteFile: ".$result." ".$rval." ".$response);
  if ($result ne "ok") 
  {
    return ("fail", "ssh to ".$u."@".$h." failed: ".$response);
  }
  if ($rval != 0)
  {
    return ("fail", "touch ".$d."/".$f." failed: ".$response);
  }
  return ("ok", "");
  
}

sub checkDest() 
{

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";

  # check the directory exists / is mounted on the remote host
  $cmd = "ls -1d ".$r_path;
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "checkDest: ssh failed: ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  }
  if ($rval != 0)
  {
    Dada::logMsg(0, $dl, "checkDest: ".$cmd." failed: ".$response);
    return ("fail", "remote dir ".$r_path." did not exist");
  }

  # check for disk space on this disk
  $cmd = "df ".$r_path." -P | tail -n 1";
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand(".$r_user.", ".$r_host.", ".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
  Dada::logMsg(2, $dl, "checkDest: remoteSshCommand() ".$result." ".$rval." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "checkDest: ssh failed: ".$response);
    return ("fail", "ssh failure to ".$r_user."@".$r_host);
  }
  if ($rval != 0)
  {
    Dada::logMsg(0, $dl, "checkDest: ".$cmd." failed: ".$response);
    return ("fail", "df command failed on ".$r_user."@".$r_host);
  }

  Dada::logMsg(2, $dl, "checkDest: ".$response);

  if ($response =~ m/No such file or directory/) 
  {
    Dada::logMsgWarn($error, $r_path." was not a valid directory on ".$r_host);
    return ("fail", "invalid r_path");
  } 
  else 
  {
    my ($location, $total, $used, $avail, $junk) = split(/ +/, $response);

    my $percent_free = $avail / $total;
    my $stop_percent = 0.05;

    Dada::logMsg(2, $dl, "checkDest: used=".$used.", avail=".$avail.", total=".$total." percent_free=".$percent_free." stop_percent=".$stop_percent);

    if ($percent_free < $stop_percent) 
    {
      Dada::logMsg(2, $dl, "checkDest: ".$r_path." is over ".((1.00-$stop_percent)*100)." percent full");
      return ("fail", $r_path." is over ".((1.00-$stop_percent)*100)." percent full");
    } 
    else
    {
      Dada::logMsg(2, $dl, "chekcDest: ".$r_path." is only ".($percent_free*100)." percent full");

      # Need more than 10 Gig
      if ($avail < 10000) 
      {
        Dada::logMsgWarn($warn, $r_host.":".$r_path." has less than 10 GB left");
        return ("fail", "less than 100 GB remaining");
      } 
      else 
      {
        return  ("ok", "");
      }
    }
  }
}


#
# Find a beam to send. Looks for observations that have an obs.finished in them
# but beams not marked as obs.transferring
# 
sub getBeamToSend(\@) 
{

  (my $array_ref) = @_;
  my @nodes_in_use = @$array_ref;
  
  my $cmd = "";
  my $result = "";
  my $response = "";

  my $o = "";
  my $b = "";
  my $obs_pid = "";
  my $path = "";
  
  my $archives_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $host = "none";
  my $obs = "none";
  my $beam = "none";

  my @obs_finished = ();
  my @beams = ();

  my $time = 0;

  # control flags
  my $have_obs = 0;
  my $survey_obs = 0;
  my $sent_to_file = 0;
  
  my $host_available = 0;
  my $i = 0;
  my $j = 0;
  my $k = 0;
  
  # find all observations that have been finished
  $cmd = "find ".$archives_dir." -maxdepth 2 -name \"obs.finished\"".
         " -printf \"%h %T@\\n\" | sort | awk -F/ '{print \$NF}'";
  Dada::logMsg(2, $dl, "getBeamToSend: ".$cmd);
  $response = `$cmd`;
  chomp $response;
  @obs_finished = split(/\n/, $response);

  if ($#obs_finished == -1) {
    $result = "ok";
    return ($result, $host, $obs, $beam);
  }

  # Ensure NFS mounts exist for subsequent parsing
  $cmd = "ls /nfs/apsr??/ >& /dev/null";
  Dada::logMsg(2, $dl, "getBeamToSend: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "getBeamToSend: ".$result.":".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Not all NFS directories were mounted");
  }

  Dada::logMsg(2, $dl, "Found (".$#obs_finished.") observations with obs.finished");

  # Go through the list of finished observations, looking for something to send
  for ($i=0; (($i<=$#obs_finished) && (!$have_obs)); $i++) {

    Dada::logMsg(3, $dl, "getBeamToSend: checking ".$obs_finished[$i]);
    ($o, $time) = split(/ /,$obs_finished[$i]);
    Dada::logMsg(3, $dl, "getBeamToSend: time: ".$time.", obs:".$obs);

    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsg(0, $dl, "Required file missing: ".$o."/obs.info");
      next;
    }

    # get the PID
    $cmd = "grep ^PID ".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getBeamToSend: ".$cmd);
    $obs_pid = `$cmd`;
    chomp $obs_pid;
    Dada::logMsg(3, $dl, "getBeamToSend: PID = ".$obs_pid);
    if ($obs_pid ne $pid) {
      Dada::logMsg(3, $dl, "getBeamToSend: skipping ".$o." as its PID [".$obs_pid."] was not ".$pid);
      next;
    }

    # Get the sorted list of beam nfs links for this obs
    # This command will timeout on missing NFS links (6 s), but wont print ones that are missing
    $cmd = "find -L ".$o." -mindepth 1 -maxdepth 1 -type d -printf \"%f\\n\" | sort";
    Dada::logMsg(3, $dl, "getBeamToSend: ".$cmd);
    $response = `$cmd`;
    chomp $response;
    @beams = split(/\n/, $response);
    Dada::logMsg(3, $dl, "getBeamToSend: found ".($#beams+1)." beams in obs ".$o);

    # See if we can find a beam that matches
    for ($j=0; (($j<=$#beams) && (!$have_obs)); $j++) {

      $b = $beams[$j];
      Dada::logMsg(3, $dl, "getBeamToSend: checking ".$o."/".$b);

      # check if this beam is currently marked as beam.transferring
      if ( -f $o."/".$b."/beam.transferring") 
      {
        Dada::logMsg(2, $dl, "getBeamToSend: skipping ".$o."/".$b.", it is marked beam.transferring");
        next;
      }

      if ( ! -f $o."/".$b."/beam.finished" )
      {
        Dada::logMsg(2, $dl, "getBeamToSend: skipping ".$o."/".$b.", it is NOT beam.finished");
        next;
      }

      if ((-f $o."/".$b."/sent.to.swin") && (-f $o."/".$b."/sent.to.parkes"))
      {
        Dada::logMsg(2, $dl, "getBeamToSend: skipping ".$o."/".$b.", it is marked sent.to.swin and sent.to.parkes");
        next;
      }
    
      # find the host on which this beam resides
      $host = "none";
      $path = $archives_dir."/".$o."/".$b;
      $cmd = "find ".$path." -maxdepth 1 -printf \"\%l\" | awk -F/ '{print \$3}'";
      Dada::logMsg(2, $dl, "getBeamToSend: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $dl, "getBeamToSend: ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "could not determine host on which ".$path." resides");
        next;
      }
      $host = $response;

      # check that this host is not in the nodes_in_use array
      $host_available = 1;
      for ($k=0; $k<$#nodes_in_use; $k++)
      {
        if ($host eq $nodes_in_use[$k])
        {
          $host_available = 0;
        }
      }
      if (!$host_available)
      {
        Dada::logMsg(2, $dl, "getBeamToSend: skipping ".$o."/".$b." as its host [".$host."] is in use");
        next;
      }
      
      # If the remote NFS mount exists
      if (-f $o."/".$b."/obs.start") 
      {
        $have_obs = 1;
        $obs = $o;
        $beam = $b;
        Dada::logMsg(2, $dl, "getBeamToSend: found beam ".$obs."/".$beam);
      } 
      else 
      {
        Dada::logMsgWarn($warn, $o."/".$b."/obs.start did not exist, or dir was not mounted");
        Dada::logMsg(3, $dl, $o."/".$b."/obs.start did not exist, or dir was not mounted");
      }
    }
  }

  Dada::logMsg(2, $dl, "have_obs: ".$have_obs);

  if ($have_obs) 
  {
    Dada::logMsg(2, $dl, "getBeamToSend: to_send obs=".$obs.", beam=".$beam." PID=".$obs_pid);

    # touch a beam.transferring file in the beam directory
    $cmd = "touch ".$archives_dir."/".$obs."/".$beam."/beam.transferring";
    Dada::logMsg(2, $dl, "getBeamToSend: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "getBeamToSend: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "getBeamToSend: failed to touch beam.transferring in ".$obs."/".$beam);
    }
  }

  return ("ok", $host, $obs, $beam);
    
}



#
# Polls for the "quitdaemons" file in the control dir
#
sub controlThread() {

  Dada::logMsg(2, $dl, "controlThread: starting");

  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  if (-f $quit_file) {
    print STDERR "controlThread: quit file existed on startup, exiting\n";
    exit(1);
  }

  # poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    Dada::logMsg(3, $dl, "controlThread: Polling for ".$quit_file);
    sleep(1);
  }

  Dada::logMsg(1, $dl, "controlThread: quit signal detected");

  # signal threads to exit
  $quit_daemon = 1;

  Dada::logMsg(2, $dl, "controlThread: Unlinking PID file ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(2, $dl, "controlThread: exiting");

}

#
# Reports the PID the script was launched with on a socket
#
sub pidReportThread($) {

  (my $daemon_pid) = @_;

  Dada::logMsg(1, $dl, "pidReportThread: thread starting");

  my $sock = 0;
  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"SERVER_XFER_PID_PORT"};
  my $handle = 0;
  my $string = "";
  my $rh = 0;

  $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1
  );

  if (!$sock) {
    Dada::logMsgWarn($warn, "Could not create PID reporting socket [".$host.":".$port."]: ".$!);

  } else {

    my $read_set = new IO::Select();
    $read_set->add($sock);

    while (!$quit_daemon) {

      # Get all the readable handles from the server
      my ($rh_set) = IO::Select->select($read_set, undef, undef, 1);

      foreach $rh (@$rh_set) {
  
        if ($rh == $sock) { 
    
          $handle = $rh->accept();
          $handle->autoflush();
          Dada::logMsg(2, $dl, "pidReportThread: Accepting connection");

          # Add this read handle to the set
          $read_set->add($handle);
          $handle = 0;

        } else {

          $string = Dada::getLine($rh);

          if (! defined $string) {
            Dada::logMsg(2, $dl, "pidReportThread: Closing a connection");
            $read_set->remove($rh);
            close($rh);
  
          } else {

            Dada::logMsg(2, $dl, "pidReportThread: <- ".$string);

            if ($string eq "get_pid") {
              print $rh $daemon_pid."\r\n";
              Dada::logMsg(2, $dl, "pidReportThread: -> ".$daemon_pid);
            } else {
              Dada::logMsgWarn($warn, "pidReportThread: received unexpected string: ".$string);
            }
          }
        }
      }
    }
  }

  Dada::logMsg(1, $dl, "pidReportThread: thread exiting");

  return 0;
}
#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada::getCurrentDadaTime(0)."\n";

}


#
# Write the current status into the /nfs/control/bpsr/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada::logMsg(2, $dl, "setStatus(".$message.")");

  my $dir = "/nfs/control/bpsr/";
  my $file = "xfer.state";
  my $cmd = "rm -f ".$dir."/".$file;
  my $result = "";
  my $response = "";


  Dada::logMsg(3, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "setStatus: ".$result." ".$response);

  $cmd = "echo '".$message."' > ".$dir."/".$file;
  Dada::logMsg(3, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "setStatus: ".$result." ".$response);

  return ("ok", "");
    
}

#
# checks all the beam directories in obs to see if they have
# been sent
#
sub checkAllBeams($) {

  my ($obs) = @_;

  Dada::logMsg(3, $dl, "checkAllBeams(".$obs.")");

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
  Dada::logMsg(3, $dl, "checkAllBeams: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkAllBeams: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkAllBeams: find command failed: ".$response);
    return ("fail", "find command failed");
  }
  $nbeams = $response;
  Dada::logMsg(3, $dl, "checkAllBeams: Total number of beams ".$nbeams);

  # Now find the number of mounted NFS links
  $cmd = "find -L ".$obs." -mindepth 1 -maxdepth 1 -type d -printf '\%f\\n' | sort";
  Dada::logMsg(3, $dl, "checkAllBeams: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkAllBeams: find command failed: ".$response);
     return ("fail", "find command failed");
  }
  @beams = split(/\n/, $response);
  $nbeams_mounted = $#beams + 1;
  Dada::logMsg(3, $dl, "checkAllBeams: Total number of mounted beams: ".$nbeams_mounted);

  # If a machine is not online, they cannot all be verified
  if ($nbeams != $nbeams_mounted) {
    return ("ok", "all beams not mounted");

  } else {
    $all_sent = 1;

    # skip if no obs.info exists
    if (!( -f $obs."/obs.info")) {
      Dada::logMsgWarn($warn, "checkAllBeams: Required file missing ".$obs."/obs.info");
      return ("fail", $obs."/obs.info did not exist");
    }

    # get the PID
    $cmd = "grep ^PID ".$obs."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "checkAllBeams: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkAllBeams: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "could not determine PID");
    }
    $obs_pid = $response;

    # check that the beam has been marked as transferred (to RAID disk)
    for ($i=0; (($i<=$#beams) && ($all_sent)); $i++) 
    {
      $beam = $beams[$i];
      if (!( -f $obs."/".$beam."/beam.transferred"))
      {
        $all_sent = 0;
        Dada::logMsg(2, $dl, "checkAllBeams: ".$obs."/".$beam."/beam.transferred did not exist");
      }
    }

    if ($all_sent) {
      Dada::logMsg(2, $dl, "checkAllBeams: all beams sent");
      return ("ok", "all beams sent");
    } else {
      return ("ok", "all beams not sent");
    }
  }

}


#
# Looks for observations that have been marked obs.deleted and moves them
# to /nfs/old_archives/bpsr and /nfs/old_results/bpsr if they are deleted
# and > 1 month old
#
sub checkFullyDeleted() {

  my $cmd = "";
  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl, "checkFullyDeleted()");

  # Find all observations marked as obs.deleted and > 14*24 hours since being modified 
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -mindepth 2 -maxdepth 2 -name 'obs.deleted' -mtime +14 -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  #$cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -mindepth 2 -maxdepth 2 -name 'obs.deleted' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyDeleted: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);
  my $s = "";
  my $n_beams = 0;
  my $n_deleted = 0;
  my $o = "";
  my @remote_hosts = ();
  my $host = "";
  my $user = "";
  my $rval = "";
  my $j = 0;
  my $i = 0;

  Dada::logMsg(2, $dl, "checkFullyDeleted: found ".($#observations + 1)." marked obs.deleted +14 mtime");

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {

    $o = $observations[$i];

    # check that the source directories exist
    if (! -d $cfg{"SERVER_ARCHIVE_DIR"}."/".$o) {
      return ("fail", $cfg{"SERVER_ARCHIVE_DIR"}."/".$o." did not exist");
    }
    if (! -d $cfg{"SERVER_RESULTS_DIR"}."/".$o) {
      return ("fail", $cfg{"SERVER_RESULTS_DIR"}."/".$o." did not exist");
    }

    $result = "ok";
    $response = "";

    # get a list of the host machines that have an archive dir on them
    $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o." -mindepth 1 -maxdepth 1 -type l -printf '\%l\n' | awk -F/ '{print \$3}' | sort";
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "failed to find remote hosts for ".$o);
    }
    @remote_hosts = split(/\n/, $response);

    # copy the archives dir, converting NFS links to remote nodes to directories
    $cmd = "cp -rL ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o." /nfs/old_archives/bpsr/";
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "checkFullyDeleted: ".$cmd." failed: ".$response);
      return ("fail", "failed to copy ".$o." to old_archives");
    }

    # delete the client archive dirs for each remote host
    $cmd = "rm -rf ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$o;
    $user = "bpsr";

    for ($j=0; $j<=$#remote_hosts; $j++) {

      $host = $remote_hosts[$j];

      Dada::logMsg(3, $dl, "checkFullyDeleted: ".$user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(3, $dl, "checkFullyDeleted: [".$host."] ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkFullyDeleted: ssh failed: ".$response);
        return ("fail", "ssh ".$user."@".$host." failed: ".$response);
      } else {
        if ($rval != 0) {
          Dada::logMsgWarn($warn, "checkFullyDeleted: could not delete remote dir [".$o."]");
          return ("fail", $cmd." failed: : ".$response);
        }
      }
    }

    # delete the archives directory
    $cmd = "rm -rf ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o;
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "checkFullyDeleted: ".$cmd." failed: ".$response);
      return ("fail", "failed deleted archives directory for ".$o);
    }

    $result = "ok";
    $response = "";

    # move the results dir
    $cmd = "mv ".$cfg{"SERVER_RESULTS_NFS_MNT"}."/".$o." /nfs/old_results/bpsr/".$o;
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(1, $dl, "checkFullyDeleted: ".$cmd." failed: ".$response);
      return ("fail", "failed to move ".$o." to old_results");
    }

    Dada::logMsg(1, $dl, $o.": deleted -> old");

  }
  return ("ok", "");
}

#
# Looks for fully archived observations, and marks obs.deleted if they
#
sub checkFullyArchived() {

  my $cmd = "";
  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl, "checkFullyArchived()");

  # Find all observations marked as obs.transferred to 
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}." -mindepth 2 -maxdepth 2 -name 'obs.archived' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkFullyArchived: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyArchived: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyArchived: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);
  my $s = "";
  my $n_beams = 0;
  my $n_deleted = 0;
  my $o = "";

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {
    $o = $observations[$i];

    if (-f $o."/obs.deleted") {
      Dada::logMsg(2, $dl, "checkFullyArchived: removing old ".$o."/obs.archived");
      unlink $o."/obs.archived";
      next;
    }

    # find out how many beam directories we have
    $cmd = "find  ".$o." -mindepth 1 -maxdepth 1 -type l | wc -l";
    Dada::logMsg(3, $dl, "checkFullyArchived: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyArchived: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyArchived: could not determine number of beam directories: ".$response);
      next;
    }
    chomp $response;
    $n_beams = $response;

    # find out how many beam.deleted files we have
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 2 -name 'beam.deleted' | wc -l";
    Dada::logMsg(3, $dl, "checkFullyArchived: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyArchived: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyArchived: could not count beam.deleted files: ".$response);
      next;
    }
    chomp $response;
    $n_deleted = $response;

    if ($n_deleted == $n_beams) {
      $cmd = "touch ".$o."/obs.deleted";
      Dada::logMsg(1, $dl, $o.": archived -> deleted");
      Dada::logMsg(2, $dl, "checkFullyArchived: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "checkFullyArchived: ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkFullyArchived: could not touch ".$o."/obs.deleted: ".$response);
      }

      if (-f $o."/obs.archived") {
        Dada::logMsg(2, $dl, "checkFullyArchived: removing ".$o."/obs.archived");
        unlink $o."/obs.archived";
      }

    }
  }
  return ("ok", "");
}

#
# Looks for finished observations to see if they have been transferred yet
#
sub checkFullyFinished() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $o = "";

  Dada::logMsg(2, $dl, "checkFullyFinished()");

  # Find all observations marked as obs.transferred to 
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."  -mindepth 2 -maxdepth 2 -name 'obs.finished' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkFullyFinished: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyFinished: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyFinished: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;

  my @observations = split(/\n/,$response);

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) 
  {
    $o = $observations[$i];

    Dada::logMsg(2, $dl, "checkFullyFinished: checking ".$o." finished -> transferred");

    # Check if all beams have been transferred successfully, if so, mark 
    # the observation as sent.to.dest
    Dada::logMsg(3, $dl, "checkFullyFinished: checkAllBeams(".$o.")");
    ($result, $response) = checkAllBeams($o);
    Dada::logMsg(3, $dl, "checkFullyFinished: checkAllBeams: ".$result." ".$response);

    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "checkFullyFinished: checkAllBeams failed: ".$response);
    }
    else
    {
      if ($response ne "all beams sent")
      {
        Dada::logMsg(2, $dl, "Obs ".$o." not fully transferred: ".$response);
      }
      else
      {
        Dada::logMsg(0, $dl, $o." finished -> transferred");
        touchLocalFile($o, "obs.transferred");
        unlink ($o."/obs.finished");
      }
    }
  }
}



#
# Looks for fully transferred observations to see if they have been archived yet
#
sub checkFullyTransferred() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $i = 0;
  my $n_beams = 0;
  my $n_swin = 0;
  my $n_parkes = 0;
  my $o = "";
  my $obs_pid = "";

  Dada::logMsg(2, $dl, "checkFullyTransferred()");
 
  # Find all observations marked as obs.transferred to 
  $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."  -mindepth 2 -maxdepth 2 -name 'obs.transferred' -printf '\%h\\n' | awk -F/ '{print \$NF}' | sort";
  Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkFullyTransferred: find command failed: ".$response);
    return ("fail", "find command failed: ".$response);
  }

  chomp $response;
  my @observations = split(/\n/,$response);

  for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++) {

    $o = $observations[$i];
  
    # skip if no obs.info exists
    if (!( -f $o."/obs.info")) {
      Dada::logMsgWarn($warn, "checkFullyTransferred: Required file missing ".$o."/obs.info");
      next;
    }

    # get the PID 
    $cmd = "grep ^PID ".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "checkFullyTransferred: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "could not determine PID");
    }
    $obs_pid = $response;
  
    Dada::logMsg(3, $dl, "checkFullyTransferred: getObsDestinations(".$obs_pid.", ".$cfg{$obs_pid."_DEST"}.")");
    my ($want_swin, $want_parkes) = Bpsr::getObsDestinations($obs_pid, $cfg{$obs_pid."_DEST"});
    Dada::logMsg(2, $dl, "checkFullyTransferred: getObsDestinations want swin:".$want_swin." parkes:".$want_parkes);

    # find out how many beam directories we have
    $cmd = "find ".$o." -mindepth 1 -maxdepth 1 -type l | wc -l";
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyTransferred: could not determine number of beam directories: ".$response);
      next;
    }
    chomp $response;
    $n_beams = $response;

    $n_swin = 0;
    $n_parkes = 0;

    # num on.tape.swin
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 2 -name 'on.tape.swin' | wc -l";
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);
    
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyTransferred: could not count on.tape.swin files: ".$response);
      next;
    }
    chomp $response;
    $n_swin = $response;

    # num on.tape.parkes
    $cmd = "find -L ".$o." -mindepth 2 -maxdepth 2 -name 'on.tape.parkes' | wc -l";
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyTransferred: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkFullyTransferred: could not count on.tape.parkes files: ".$response);
      next;
    }
    chomp $response;
    $n_parkes = $response;

    # now the conditions
    if ( ($want_swin && ($n_swin < $n_beams)) || 
         ($want_parkes && ($n_parkes < $n_beams)) ){

      Dada::logMsg(2, $dl, "checkFullyTransferred: ".$o." [".$obs_pid."] swin[".$n_swin."/".$n_beams."] ".
                          "parkes[".$n_parkes."/".$n_beams."]");

    } else {

      if (-f $o."/obs.archived") {
        # do nothing

      } else {
        $cmd = "touch ".$o."/obs.archived";
        Dada::logMsg(1, $dl, $o.": transferred -> archived");
        Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, $dl, "checkFullyTransferred: ".$result." ".$response);

      }

      if (-f $o."/obs.transferred") {
        $cmd = "rm -f ".$o."/obs.transferred";
        Dada::logMsg(2, $dl, "checkFullyTransferred: ".$o." has been fully archived, deleting obs.transferred");
        Dada::logMsg(2, $dl, "checkFullyTransferred: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, $dl, "checkFullyTransferred: ".$result." ".$response);
      }

    }
  }
 
  return ("ok", ""); 
}
