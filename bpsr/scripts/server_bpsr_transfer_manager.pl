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
use constant PIDFILE       => "bpsr_transfer_manager.pid";
use constant LOGFILE       => "bpsr_transfer_manager.log";
use constant QUITFILE      => "bpsr_transfer_manager.quit";
use constant SSH_OPTS      => "-x -o BatchMode=yes";

#
# Global Variables
#
our $dl = 1;
our %cfg = Bpsr::getConfig();   # Bpsr.cfg
our $error = $cfg{"STATUS_DIR"}."/bpsr_transfer_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/bpsr_transfer_manager.warn";
our $quit_daemon : shared  = 0;
our $pid : shared = "";


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
# Holding areas for later tape archival
#
my @s_disks = ();
my @p_disks = ();

for ($i=0; $i<$cfg{"NUM_SWIN_DIRS"}; $i++) {
  push (@s_disks, $cfg{"SWIN_DIR_".$i});
}

for ($i=0; $i<$cfg{"NUM_PARKES_DIRS"}; $i++) {
  push (@p_disks, $cfg{"PARKES_DIR_".$i});
}

#
# Main
#
Dada::daemonize($logfile, $pidfile);

Dada::logMsg(0, $dl, "STARTING SCRIPT");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread);

# Start the PID reporting thread
$pid_report_thread = threads->new(\&pidReportThread, $pid);

setStatus("Starting script");

my $s_i = 0;
my $p_i = 0;

# Ensure that destination directories exist for this project
($result, $response) = checkDestinations(\@s_disks);
if ($result ne "ok") {
  $quit_daemon = 1;
  sleep(3);
  exit(1);
}

# Ensure that destination directories exist for this project
($result, $response) = checkDestinations(\@p_disks);
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
Dada::logMsg(1, $dl, "main: checkFullyDeleted(): ".$result.":".$response);

my $s_in = new Thread::Queue;
my $s_out = new Thread::Queue;
my $s_count = 0;

my $p_in = new Thread::Queue;
my $p_out = new Thread::Queue;
my $p_count = 0;


# start the Transfer threads
my $num_s_threads = $cfg{"NUM_SWIN_DIRS"} - 1;
my $num_p_threads = $cfg{"NUM_PARKES_DIRS"} - 1;
my @s_threads = ();
my @p_threads = ();

$num_s_threads = 2;
$num_p_threads = 2;

for ($i=0; $i<$num_s_threads; $i++)
{
  push (@s_threads, 0);
  $s_threads[$i] = threads->new(\&transferThread, $s_in, $s_out, "swin", $i);
}
for ($i=0; $i<$num_p_threads; $i++)
{
  push (@p_threads, 0);
  $p_threads[$i] = threads->new(\&transferThread, $p_in, $p_out, "parkes", $i);
}

###############################################################################
#
# Main loop
#

Dada::logMsg(1, $dl, "starting main loop");

my $host = "";
my $obs = "";
my $beam = "";
my $is_fold = "";
my $input_string = "";
my $output_string = "";

my @nodes_in_use = ();
my @tmp_array = ();
my @obs_to_check = ();

my $r_user = "";
my $r_host = "";
my $r_dir = "";
my $xfer_count = 0;

my $s_sleep = 0;
my $p_sleep = 0;

while ((!$quit_daemon) || ($s_count > 0) || ($p_count > 0))
{

  Dada::logMsg(2, $dl, "main: quit_daemon=".$quit_daemon.", s_count=".$s_count.", p_count=".$p_count);
  
  # if observations are not queued for swin
  #   find swin obs and enqueue it
  if ((!$quit_daemon ) && ($s_count < $num_s_threads) && ($s_sleep <= 0))
  {
    Dada::logMsg(2, $dl, "main: findHoldingArea(swin)");
    ($r_user, $r_host, $r_dir) = findHoldingArea(\@s_disks, $s_i);
    Dada::logMsg(2, $dl, "main: findHoldingArea(swin) ".$r_user." ".$r_host." ".$r_dir);

    if (($r_user ne "none") && ($r_host ne "none") && ($r_dir ne "none"))
    {
      $s_i = ($s_i + 1) % ($#s_disks+1);
      # find and obs/beam to send to swin, mark beam as obs.transferring...
      Dada::logMsg(2, $dl, "main: getBeamToSend(swin)");
      ($result, $host, $obs, $beam, $is_fold) = getBeamToSend("swin", \@nodes_in_use);
      Dada::logMsg(2, $dl, "main: getBeamToSend(swin) ".$result." ".$host." ".$obs." ".$beam." ".$is_fold.")");

      if (($result eq "ok") && ($host ne "none") && ($obs ne "none") && ($beam ne "none"))
      {

        if ($is_fold) 
        {
          Dada::logMsg(2, $dl, "main: ".$obs."/".$beam." is a folded observation");
          $r_dir .= "/".$pid."/pulsars";
        } 
        else 
        {
          Dada::logMsg(2, $dl, "main: ".$obs."/".$beam." is a survey observation");
          $r_dir .= "/".$pid."/staging_area";
        }

        $input_string = $host." ".$obs." ".$beam." ".$r_user." ".$r_host." ".$r_dir;
        Dada::logMsg(2, $dl, "main: enqueue swin [".$input_string."]");
        $s_in->enqueue($input_string);
        push (@nodes_in_use, $host);
        $s_count++;
        $s_sleep = 2;
      }
      else
      {
        Dada::logMsg(2, $dl, "main: no beams for swin, removing ".$r_host.":".$r_dir."/WRITING flag");
        my $cmd = "ssh ".SSH_OPTS." -l ".$r_user." ".$r_host." \"rm -f ".$r_dir."/../WRITING\"";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        system($cmd);
        if ($? != 0) {
          Dada::logMsgWarn($warn, "main: could not remove remote WRITING file");
        }
        $s_sleep = 60;
      }

    }
  }
  if ($s_sleep > 0)
  {
    $s_sleep--;
  }

  #
  # Look for observations to send to Parkes
  #
  if ((!$quit_daemon) && ($p_count < $num_p_threads) && ($p_sleep <= 0))
  {
    Dada::logMsg(2, $dl, "main: findHoldingArea(parkes)");
    ($r_user, $r_host, $r_dir) = findHoldingArea(\@p_disks, $p_i);
    Dada::logMsg(2, $dl, "main: findHoldingArea(parkes) ".$r_user." ".$r_host." ".$r_dir);

    if (($r_user ne "none") && ($r_host ne "none") && ($r_dir ne "none"))
    {
      $p_i = ($p_i + 1) % ($#p_disks+1);
      # find and obs/beam to send to parkes, mark beam as obs.transferring...
      Dada::logMsg(2, $dl, "main: getBeamToSend(parkes)");
      ($result, $host, $obs, $beam, $is_fold) = getBeamToSend("parkes", \@nodes_in_use);
      Dada::logMsg(2, $dl, "main: getBeamToSend(parkes) ".$result." ".$host." ".$obs." ".$beam." ".$is_fold.")");

      if (($result eq "ok") && ($host ne "none") && ($obs ne "none") && ($beam ne "none"))
      {
        if ($is_fold) 
        {
          Dada::logMsg(2, $dl, "main: ".$obs."/".$beam." is a folded observation");
          $r_dir .= "/".$pid."/pulsars";
        } 
        else
        {
          Dada::logMsg(2, $dl, "main: ".$obs."/".$beam." is a survey observation");
          $r_dir .= "/".$pid."/staging_area";
        }

        $input_string = $host." ".$obs." ".$beam." ".$r_user." ".$r_host." ".$r_dir;
        Dada::logMsg(1, $dl, "main: enqueue parkes [".$input_string."]");
        $p_in->enqueue($input_string);
        push (@nodes_in_use, $host);
        $p_count++;
        $p_sleep = 2;
      }
      else
      {
        Dada::logMsg(2, $dl, "main: no beams for parkes, removing ".$r_host.":".$r_dir."/WRITING flag");
        my $cmd = "ssh ".SSH_OPTS." -l ".$r_user." ".$r_host." \"rm -f ".$r_dir."/../WRITING\"";
        Dada::logMsg(2, $dl, "main: ".$cmd);
        system($cmd);
        if ($? != 0) {
          Dada::logMsgWarn($warn, "checkTransferredFiles: could not remove remote WRITING file");
        }
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
  if ($s_out->pending()) 
  {
    Dada::logMsg(2, $dl, "main: dequeuing swin transfer");
    $output_string = $s_out->dequeue();
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

    Dada::logMsg(2, $dl, "main: dequeued swin xfer ".$obs."/".$beam." from ".$host);

    # decremeting the current swin tranfer count
    $s_count--;
    
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
      Dada::logMsgWarn($warn, "main: swin xfer for ".$obs."/".$beam." failed");
      $quit_daemon = 1;
    }
    else
    {
      push (@obs_to_check, $obs);
    }
  }

  # if a beam has finished transfer, remove it from the queue
  if ($p_out->pending())
  {
    Dada::logMsg(2, $dl, "main: dequeuing parkes transfer");
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

    Dada::logMsg(2, $dl, "main: dequeued parkes xfer ".$obs."/".$beam." from ".$host);

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
      Dada::logMsgWarn($warn, "main: parkes xfer for ".$obs."/".$beam." failed");
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
          recordTransferResult("obs.transferred", $obs);
          unlink ($obs."/obs.finished");
        }
      }
    }
  }

  if ((($#obs_to_check >= 0) && ($xfer_count % 26 == 0)) || ($s_sleep == 59) || ($p_sleep == 59))
  {
    Dada::logMsg(2, $dl, "main: observations otc=".$#obs_to_check." xfer_count=".$xfer_count." mod=".($xfer_count % 26)." s_s=".$s_sleep." p_s=".$p_sleep);

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

  setStatus(($s_count + $p_count)." threads transferring");

  Dada::logMsg(2, $dl, "main: sleep(1)"); 
  sleep(1);

}

# rejoin threads
Dada::logMsg(1, $dl, "main: joining controlThread");
$control_thread->join();

# rejoin threads
Dada::logMsg(1, $dl, "main: joining pid_report_thread");
$pid_report_thread->join();

for ($i=0; $i<$num_s_threads; $i++)
{
  if ($s_threads[$i] != 0)
  {
    Dada::logMsg(1, $dl, "main: joining s_thread[".$i."]");
    $s_threads[$i]->join();
  }
}
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
sub transferThread($$$$)
{
  my ($in, $out, $dest, $tid) = @_;

  my $have_connection = 1;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $log_message = 0;
  my $quit_wait = 10;
  my @bits = ();
  my $transfer_result = 0;
  my $file_list = "";
  my $local_file_list = "";
  my $pathed_local_file_list = "";
  my $path = "";
  my $rval = 0;
  my $tag = "";
  if ($dest eq "swin")
  {
    $tag = "swin".$tid;
  }
  if ($dest eq "parkes")
  {
    $tag = "prks".$tid;
  }

  Dada::logMsg(2, $dl, "xfer[".$tag."] starting");

  while (!$quit_daemon) 
  {
    Dada::logMsg(2, $dl, "xfer[".$tag."] while(!quit_daemon)");

    while ($in->pending) 
    {
      Dada::logMsg(2, $dl, "xfer[".$tag."] while(!in->pending)");

      if ($quit_daemon) 
      {
        if ($quit_wait > 0) 
        {
          Dada::logMsg(2, $dl, "xfer[".$tag."]: quit_daemon=1 while in->pending=true, waiting...");
          $quit_wait--;
        } 
        else 
        {
          Dada::logMsg(0, $dl, "xfer[".$tag."]: quit_daemon=1 while in->pending=true, quitting!");
          return 0;
        }
      }

      # try to dequeue a input transfer string from the queue
      Dada::logMsg(2, $dl, "xfer[".$tag."] calling in->dequeue");
      $input_string = $in->dequeue_nb();
      Dada::logMsg(2, $dl, "xfer[".$tag."] in->dequeue returns");
  
      # if we missed the dequeue, then return
      if ($input_string eq undef) 
      {
        Dada::logMsg(2, $dl, "xfer[".$tag."] missed the dequeue, polling again");
        next;
      }

      Dada::logMsg(2, $dl, "xfer[".$tag."] dequeued input string '".$input_string."'");

      # decode the transfer parameters
      @bits = ();
      @bits = split(/ /,$input_string);

      # ensure we decoded 4 elements
      if ($#bits != 5)
      {
        Dada::logMsgWarn($warn, "xfer[".$tag."] could not decode ".$input_string." into 6 params");
      }

      $host = $bits[0];
      $obs  = $bits[1];
      $beam = $bits[2];
      $r_user = $bits[3];
      $r_host = $bits[4];
      $r_dir = $bits[5];


      Dada::logMsg(2, $dl, "xfer[".$tag."] getBeamFiles(".$obs.", ".$beam.", 1)");
      $file_list = getBeamFiles($obs, $beam, 1);
      Dada::logMsg(2, $dl, "xfer[".$tag."] getBeamFiles() ".$file_list);

      Dada::logMsg(2, $dl, "xfer[".$tag."] getBeamFiles(".$obs.", ".$beam.", 0)");
      $local_file_list = getBeamFiles($obs, $beam, 0);
      Dada::logMsg(2, $dl, "xfer[".$tag."] getBeamFiles() ".$local_file_list);


      # create the destination directory
      $cmd = "mkdir -p ".$r_dir."/".$obs."/".$beam;
      Dada::logMsg(2, $dl, "xfer[".$tag."] ".$r_user."@".$r_host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tag."] ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "xfer[".$tag."] ssh failed: ".$response);
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "xfer[".$tag."] mkdir failed: ".$response); 
      }

      Dada::logMsg(1, $dl, "xfer[".$tag."] ".$obs."/".$beam.": transferring");

      # rsync the specified obs to the specified dir 
      # if parkes, we have direct ssh access from the host to the r_host
      if ($dest eq "parkes") 
      {
        $pathed_local_file_list = $local_file_list;
        $pathed_local_file_list =~ s/ 2/ \.\/2/g;
        $pathed_local_file_list =~ s/^2/\.\/2/g;
        $cmd = "rsync -av --bwlimit=30720 ".$pathed_local_file_list." ".$r_user."@".$r_host.":".$r_dir."/".$obs."/".$beam."/";
        Dada::logMsg(2, $dl, "xfer[".$tag."] ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $host, $cmd, "/lfs/data0/bpsr/archives");
        Dada::logMsg(2, $dl, "xfer[".$tag."] ".$result." ".$rval." ".$response);
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "xfer[".$tag."] ssh failed: ".$response);
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "xfer[".$tag."] rsync failed: ".$response); 
        }
      }
      else  # if swin, everything goes through srv0
      {
        # rsync the specified obs to the specified dir 
        $cmd = "rsync -av --sockopts=SO_SNDBUF=2000000,SO_RCVBUF=2000000 ".$file_list." ".$r_user."@".$r_host.":".$r_dir."/".$obs."/".$beam."/";
        Dada::logMsg(2, $dl, "xfer[".$tag."] ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, $dl, "xfer[".$tag."] ".$result." ".$response);
      }

      # determine the data rate
      @bits = split(/\n/, $response);
      for ($i=0; $i<=$#bits; $i++)
      {
        if ($bits[$i] =~ m/bytes\/sec/)
        {
          Dada::logMsg(1, $dl, "xfer[".$tag."] ".$obs."/".$beam.": ".$bits[$i]);
        }
      }
  
      $transfer_result = 0;
      $path = $obs."/".$beam;

      if ($result eq "ok") 
      {
        Dada::logMsg(2, $dl, "xfer[".$tag."] (".$r_user.", ".$r_host.", ".$r_dir.", ".$local_file_list.")");
        ($result, $response) = checkTransferredFiles($r_user, $r_host, $r_dir, $local_file_list);
        Dada::logMsg(2, $dl, "xfer[".$tag."] checkTransferredFiles() ".$result." ".$response);

        if ($result eq "ok") 
        {
          Dada::logMsg(2, $dl, "Successfully transferred to ".$dest.": ".$path);
          recordTransferResult("sent.to.".$dest, $path);
          markRemoteCompleted("xfer.complete", $r_user, $r_host, $r_dir, $path);
          $transfer_result = 1; 
        }
        else 
        {
          Dada::logMsgWarn($warn, "xfer[".$tag."] checkTransferFiles failed");
        }
      } 
      else
      {
        Dada::logMsgWarn($warn, "xfer[".$tag."] rsync failed: ".$response);
      }

      if ($transfer_result)
      {
        $output_string = "ok ".$host." ".$obs." ".$beam;
        Dada::logMsg(1, $dl, "xfer[".$tag."] ".$obs."/".$beam.": finished -> transferred");
      }
      else
      {
        $output_string = "fail ".$host." ".$obs." ".$beam;

        Dada::logMsgWarn($warn, "xfer[".$tag."] failed to transfer ".$path." to ".$dest.": ".$response);

        $cmd = "mv ".$path."/obs.start ".$path."/obs.bad";
        Dada::logMsg(1, $dl, "xfer[".$tag."]: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(1, $dl, "xfer[".$tag."]: ".$result." ".$response);

        recordTransferResult("error.to.".$dest, $path);
        Dada::logMsg(1, $dl, $obs."/".$beam.": finished -> error");

      }

      Dada::logMsg(2, $dl, "xfer[".$tag."]: dequeueing ".$obs."/".$beam);

      if ( -f $obs."/".$beam."/beam.transferring" )
      {
        Dada::logMsg(2, $dl, "xfer[".$tag."]: unlinking ".$obs."/".$beam."/beam.transferring");
        unlink ($obs."/".$beam."/beam.transferring");
      }
      else 
      {
        Dada::logMsgWarn($warn, "xfer[".$tag."] ".$obs."/".$beam."/beam.transferring missing");
      }
      
      $out->enqueue($output_string);
    }

    sleep(1);
  }

  Dada::logMsg(2, $dl, "xfer[".$tag."]: exiting");
}


#
# Ensure the files sent matches the local
#
sub checkTransferredFiles($$$$) {

  my ($user, $host, $dir, $files) = @_;

  Dada::logMsg(2, $dl, "checkTransferredFiles(".$user.", ".$host.", ".$dir.", ".$files.")");

  my $cmd = "ls -l ".$files." | awk '{print \$5\" \"\$9}' | sort";

  Dada::logMsg(3, $dl, "checkTransferredFiles: ".$cmd);
  my $local_list = `$cmd`;
  if ($? != 0) {
    Dada::logMsgWarn($warn, "checkTransferredFiles: local find failed ".$local_list);
  }

  $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"cd ".$dir."; ls -l ".$files."\" | awk '{print \$5\" \"\$9}' | sort";
  
  Dada::logMsg(3, $dl, "checkTransferredFiles: ".$cmd);
  my $remote_list = `$cmd`;

  if ($? != 0) {
    Dada::logMsgWarn($warn, "checkTransferredFiles: remote find failed ".$remote_list);
  }

  # Regardless of transfer success, remove the WRITIING flag
  $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"cd ".$dir."; rm -f ../../../WRITING\"";
  Dada::logMsg(3, $dl, "checkTransferredFiles: ".$cmd);
  system($cmd);
  if ($? != 0) {
    Dada::logMsgWarn($warn, "checkTransferredFiles: could not remove remote WRITING file");
  }

  if (($local_list eq $remote_list) && ($local_list ne "")) 
  {
    Dada::logMsg(2, $dl, "checkTransferredFiles: returning ok");
    return ("ok","");
  } 
  else 
  {
    $local_list =~ s/\n/ /g;
    $remote_list =~ s/\n/ /g;
    Dada::logMsgWarn($warn, "ARCHIVE MISMATCH: local: ".$local_list);
    Dada::logMsgWarn($warn, "ARCHIVE MISMATCH: remote: ".$remote_list);
    return ("fail", "archive mismatch");
  }

}


sub recordTransferResult($$) {

  my ($file, $dir) = @_;

  my $cmd = "sudo -u bpsr touch ".$dir."/".$file;
  Dada::logMsg(2, $dl, "recordTransferResult: ".$cmd);
  my ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "recordTransferResult: ".$result." ".$response);

  return ($result, $response);
}

sub markRemoteCompleted($$$$$) {

  my ($file, $user, $host, $dir, $obs) = @_;
  Dada::logMsg(2, $dl, "markRemoteCompleted(".$file.", ".$user.", ".$host.", ".$dir.", ".$obs.")");

  my $cmd = "cd ".$dir."; touch ".$obs."/".$file;
  my $ssh_cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"".$cmd."\"";

  Dada::logMsg(2, $dl, "markRemoteCompleted: ".$cmd);
  my ($result, $response) = Dada::mySystem($ssh_cmd);
  Dada::logMsg(2, $dl, "markRemoteCompleted: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "could not mark ".$host.":".$dir."/".$obs." as completed");
  }

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
  my $reading = 0;
  my $writing = 0;

  # If the array as 0 size, return none
  if ($#disks == -1) {
    return ("none", "none", "none");
  }

  for ($c=0; $c<=$#disks; $c++) {
    $i = ($c + $startdisk)%($#disks+1);

    $disk = $disks[$i];
    Dada::logMsg(2, $dl, "Evaluating ".$disk);

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
      Dada::logMsg(2, $dl, $cmd);
      $result = `$cmd`;

      if ($? != 0) {
        chomp $result;
        Dada::logMsgWarn($warn, "ssh cmd '".$cmd."' failed: ".$result);
        $result = "";
      } else {

        $reading = 1;
        $writing = 1;

        # check if this is being used for reading
        $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"ls ".$path."/../READING\" 2>&1";
        $result = `$cmd`;
        chomp $result;
        if ($result =~ m/No such file or directory/) {
          $reading = 0;
        }

        # check if this is being used for writing 
        $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"ls ".$path."/../WRITING\" 2>&1";
        $result = `$cmd`;
        chomp $result;
        if ($result =~ m/No such file or directory/) {
          $writing = 0;
        }
      
        if ((!$writing) && (!$reading))
        {
          # There is no READING file.
          $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"df ".$path." -P\" | tail -n 1";
          Dada::logMsg(2, $dl, $cmd);
          $result = `$cmd`;
          if ($? != 0) {
            Dada::logMsgWarn($warn, "df command ".$cmd." failed: ".$result);
            $result = "";
          } 
        } else {
          # we are writing to the disk
          Dada::logMsg(2, $dl, "Skipping disk $host:$path it was being used for reading[".$reading."] writing[".$writing."]");
          $result="";
        }
      }
    } else {

      Dada::logMsgWarn($warn, "disk line syntax error ".$disk);
      $result = "";

    }

    if ($result ne "") {

      chomp($result);

      Dada::logMsg(2, $dl, "df_result  = $result");

      if ($result =~ m/No such file or directory/) {

        Dada::logMsgWarn($error, $user." ".$host." ".$path." was not a valid directory");
        $result = "";

      } else {

        my ($location, $total, $used, $avail, $junk) = split(" ",$result);

        my $percent_free = $avail / $total;
        my $stop_percent = 0.05;
        if ($host =~ m/apsr/) {
          $stop_percent = 0.25;
        }

        Dada::logMsg(2, $dl, $host.":".$path." used=".$used.", avail=".$avail.", total=".$total." percent_free=".$percent_free." stop_percent=".$stop_percent);

        if ($percent_free < $stop_percent) {

          Dada::logMsg(2, $dl, $host.":".$path." is over ".((1.00-$stop_percent)*100)." percent full");

        } else {

          Dada::logMsg(2, $dl, $host.":".$path." is only ".($percent_free*100)." percent full");

          # Need more than 10 Gig
          if ($avail < 10000) {

            Dada::logMsgWarn($warn, $host.":".$path." has less than 10 GB left");

          } else {
            Dada::logMsg(2, $dl, "Holding area: ".$user." ".$host." ".$path.": touching WRITING flag");
            $cmd = "ssh ".SSH_OPTS." -l ".$user." ".$host." \"touch ".$path."/../WRITING\"";
            Dada::logMsg(2, $dl, $cmd);
            $result = `$cmd`;
            return ($user,$host,$path);
          }
        }
      }
    }
  }

  return ("none", "none", "none");

}


#
# Find a beam to send. Looks for observations that have an obs.finished in them
# but beams not marked as obs.transferring
# 
sub getBeamToSend($\@) 
{

  (my $dest, my $array_ref) = @_;
    
  my @nodes_in_use = @$array_ref;
  
  Dada::logMsg(2, $dl, "getBeamToSend(".$dest.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $o = "";
  my $b = "";
  my $obs_pid = "";
  my $source = "";
  my $path = "";
  
  my $archives_dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"};
  my $host = "none";
  my $obs = "none";
  my $beam = "none";
  my $is_fold = 0;

  my @obs_finished = ();
  my @beams = ();

  my $time = 0;

  # control flags
  my $have_obs = 0;
  my $survey_obs = 0;
  my $twobit_file = 0;
  my $sent_to_file = 0;
  
  my $host_available = 0;
  my $want_swin = 0;
  my $want_parkes = 0;
  my $want_flag = 0;
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
    return ($result, $host, $obs, $beam, $is_fold);
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

    # get the SOURCE 
    $cmd = "grep SOURCE ".$o."/obs.info | awk '{print \$2}'";
    Dada::logMsg(3, $dl, "getBeamToSend: ".$cmd);
    $source = `$cmd`;
    chomp $source;
    Dada::logMsg(3, $dl, "getBeamToSend: SOURCE = ".$source);

    # determine if this is a survey pointing [i.e. source begins with G]
    if ($source =~ m/^G/) {
      $survey_obs = 1;
      $is_fold = 0;
    } else {
      $survey_obs = 0;
      $is_fold = 1;
    }

    # determine the required destinations based on PID
    ($want_swin, $want_parkes) = Bpsr::getObsDestinations($obs_pid, $cfg{$obs_pid."_DEST"});
    Dada::logMsg(2, $dl, "getBeamToSend: ".$o." [".$source."] want[".$want_swin.",".$want_parkes."] fold[".$is_fold."]");
    if (($dest eq "swin") && ($want_swin)) 
    {
      $want_flag = 1;
    }
    elsif (($dest eq "parkes") && ($want_parkes))
    {
      $want_flag = 1;
    }
    else
    {
      $want_flag = 0;
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

      if ( -f $o."/".$b."/sent.to.".$dest )
      {
        Dada::logMsg(2, $dl, "getBeamToSend: skipping ".$o."/".$b.", it is marked sent.to.".$dest);
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
      if (-f $o."/".$b."/obs.start") {

        # see what flags we have
        $twobit_file = (-f $o."/".$b."/".$o.".fil") ? 1 : 0;
        $sent_to_file = (-f $o."/".$b."/sent.to.".$dest) ? 1 : 0;
        Dada::logMsg(3, $dl, "getBeamToSend: ".$obs."/".$beam." fil[".$twobit_file."] ".
                            $dest."[".$want_flag.",".$sent_to_file."]");

        if (($want_flag and !$sent_to_file))
        {
          $have_obs = 1;
          $obs = $o;
          $beam = $b;
          Dada::logMsg(2, $dl, "getBeamToSend: found beam for ".$dest.": ".$obs."/".$beam);
        }

      } else {
        Dada::logMsgWarn($warn, $o."/".$b."/obs.start did not exist, or dir was not mounted");
        Dada::logMsg(3, $dl, $o."/".$b."/obs.start did not exist, or dir was not mounted");
      }
    }
  }

  Dada::logMsg(2, $dl, "have_obs: ".$have_obs);

  if ($have_obs) {
    Dada::logMsg(2, $dl, "getBeamToSend: to_send obs=".$obs.", beam=".$beam);
    Dada::logMsg(3, $dl, "getBeamToSend: to_send PID=".$obs_pid.", SOUCE=".$source);
    Dada::logMsg(3, $dl, "getBeamToSend: to_send survey_obs=".$survey_obs.", is_fold=".$is_fold);

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

  return ("ok", $host, $obs, $beam, $is_fold);
    
}


#
# Returns the files to be sent for this beam, full path include NFS links
#
sub getBeamFiles($$$) {

  my ($obs, $beam, $full_path) = @_;

  Dada::logMsg(2, $dl, "getBeamFiles(".$obs.", ".$beam.", ".$full_path.")");

  my $dir = "";
  if ($full_path)
  {
    $dir = $cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$obs."/".$beam;
  }
  else
  {
    $dir = $obs."/".$beam;
  }
  my @files = ();
  my $file_list = "";
  my $cmd = "";
  my $list = "";

  # aux.tar, integrated.ar, $obs.psrxml, $obs.fil, obs.txt and obs.start
  $cmd = "find -L ".$dir." -maxdepth 1 -name '*ar' -o -name '".$obs.".*' -o -name  'obs.*' -o -name 'rfi.*'";
  Dada::logMsg(3, $dl, "getBeamFiles: ".$cmd);
  $list = `$cmd`;
  chomp $list;
  @files = split(/\n/,$list);

  my $i=0;
  for ($i=0; $i<=$#files; $i++) {
    $file_list .= $files[$i]." ";
  }
  
  Dada::logMsg(3, $dl, "getBeamFiles: ".$file_list);

  return $file_list;

}



#
# Polls for the "quitdaemons" file in the control dir
#
sub controlThread() {

  Dada::logMsg(2, $dl, "controlThread: starting");

  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

  Dada::logMsg(1, $dl, "controlThread: quit_file = ".$quit_file);

  if (-f $quit_file) {
    print STDERR "controlThread: quit file existed on startup, exiting\n";
    exit(1);
  }

  # poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    Dada::logMsg(3, $dl, "controlThread: Polling for ".$quit_file);
    sleep(1);
  }

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
    Reuse => 1,
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

    Dada::logMsg(3, $dl, "checkAllBeams: getObsDestinations(".$obs_pid.", ".$cfg{$obs_pid."_DEST"}.")");
    my ($want_swin, $want_parkes) = Bpsr::getObsDestinations($obs_pid, $cfg{$obs_pid."_DEST"});
    Dada::logMsg(3, $dl, "checkAllBeams: getObsDestinations want swin:".$want_swin." parkes:".$want_parkes);
 
    for ($i=0; (($i<=$#beams) && ($all_sent)); $i++) {
      $beam = $beams[$i];

      if ($want_swin && (! -f $obs."/".$beam."/sent.to.swin")) {
        $all_sent = 0;
        Dada::logMsg(2, $dl, "checkAllBeams: ".$obs."/".$beam."/sent.to.swin did not exist");
      }

      if ($want_parkes && (! -f $obs."/".$beam."/sent.to.parkes")) {
        Dada::logMsg(2, $dl, "checkAllBeams: ".$obs."/".$beam."/sent.to.parkes did not exist");
        $all_sent = 0;
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
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "checkFullyDeleted: ".$result." ".$response);
    if ($result ne "ok") {
      return ("fail", "failed to find remote hosts for ".$o);
    }
    @remote_hosts = split(/\n/, $response);

    # copy the archives dir, converting NFS links to remote nodes to directories
    $cmd = "cp -rL ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o." /nfs/old_archives/bpsr/";
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
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

      Dada::logMsg(2, $dl, "checkFullyDeleted: ".$user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
      Dada::logMsg(3, $dl, "checkFullyDeleted: [".$host."] ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "checkFullyDeleted: ssh failed: ".$response);
        return ("fail", "ssh ".$user."@".$host." failed: ".$response);
      } else {
        if ($rval != 0) {
          Dada::logMsgWarn($warn, "checkFullyDeleted: remote dir did not exist: ".$response);
          return ("fail", $cmd." failed: : ".$response);
        }
      }
    }

    # delete the archives directory
    $cmd = "rm -rf ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}."/".$o;
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
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
    Dada::logMsg(2, $dl, "checkFullyDeleted: ".$cmd);
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
    Dada::logMsg(2, $dl, "checkFullyFinished: checkAllBeams(".$o.")");
    ($result, $response) = checkAllBeams($o);
    Dada::logMsg(2, $dl, "checkFullyFinished: checkAllBeams: ".$result." ".$response);

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
        recordTransferResult("obs.transferred", $o);
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

sub checkDestinations(\@) {

  my ($ref) = @_;

  Dada::logMsg(2, $dl, "checkDestinations()");

  my @dests = @$ref;
  my $i = 0;
  my $user = "";
  my $host = "";
  my $path = "";
  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  # for each destinatinos
  for ($i=0; $i<=$#dests; $i++) {

    ($user, $host, $path) = split(":",$dests[$i],3);
    Dada::logMsg(2, $dl, "checkDestinations: ".$user."@".$host.":".$path);

    $cmd = "ls ".$path;
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDestinations: ssh failed: ".$response);
      return ("fail", "ssh ".$user."@".$host." failed: ".$response);

    } else {
      if ($rval != 0) {
        Dada::logMsgWarn($warn, "checkDestinations: remote dir did not exist: ".$response);
        return ("fail", "remote dir did not exist: ".$response);
      }
    }

    $cmd = "ls ".$path."/".$pid."/staging_area > /dev/null";
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDestinations: ssh failed: ".$response);
      return ("fail", "ssh ".$user."@".$host." failed: ".$response);

    } else {
      if ($rval != 0) {
        Dada::logMsgWarn($warn, "checkDestinations: remote dir did not exist: ".$response);
        $cmd = "mkdir -p ".$path."/".$pid."/staging_area";
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
      }
    }

    $cmd = "ls ".$path."/".$pid."/on_tape > /dev/null";
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDestinations: ssh failed: ".$response);
      return ("fail", "ssh ".$user."@".$host." failed: ".$response);
    } else {
      if ($rval != 0) {
        Dada::logMsgWarn($warn, "checkDestinations: remote dir did not exist: ".$response);
        $cmd = "mkdir -p ".$path."/".$pid."/on_tape";
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
      }
    }

    $cmd = "ls ".$path."/".$pid."/pulsars > /dev/null";
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$cmd);
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "checkDestinations: ssh failed: ".$response);
      return ("fail", "ssh ".$user."@".$host." failed: ".$response);
    } else {
      if ($rval != 0) {
        Dada::logMsgWarn($warn, "checkDestinations: remote dir did not exist: ".$response);
        $cmd = "mkdir -p ".$path."/".$pid."/pulsars";
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
        Dada::logMsg(1, $dl, "checkDestinations: [".$host."] ".$result." ".$response);
      }
    }
    
    Dada::logMsg(2, $dl, "checkDestinations: ".$user."@".$host.":".$path." ok");
  }

  return ("ok", "");
}


