#!/usr/bin/env perl
###############################################################################
#
# server_apsr_transfer_manager.pl
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
use Apsr;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant BANDWIDTH     => "102400";  # 100 MB/s total bandwidth

#
# Global Variables
#
our $dl = 1;
our %cfg = Apsr::getConfig();   # Apsr.cfg
our $error = "";
our $warn  = "";
our $quit_daemon : shared  = 0;
our $r_user = "apsr";
our $r_host = "raid0";
our $r_path = "/lfs/raid0/apsr/P778";
our $r_module = "apsr_P778_upload";
our $num_p_threads = 5;
our $daemon_name = Dada::daemonBaseName($0);

# Autoflush output
$| = 1;

# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

#
# Local Varaibles
#
my $log_file  = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
my $pid_file  = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

my $control_thread = 0;
my $result = "";
my $response = "";
my $i=0;
my @bits = ();

$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";

# clear the error and warning files if they exist
if ( -f $warn ) {
  unlink ($warn);
}
if ( -f $error) {
  unlink ($error);
}

#
# Main
#
Dada::daemonize($log_file, $pid_file);

Dada::logMsg(0, $dl, "STARTING SCRIPT PID=778 BANDWDITH=".sprintf("%2.0f", (BANDWIDTH / 1024))." MB/s");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread, $pid_file, $quit_file);

setStatus("Starting script");

my $p_i = 0;

# Ensure that destination directories exist for this project
($result, $response) = checkDest();
if ($result ne "ok") {
  $quit_daemon = 1;
  sleep(3);
  exit(1);
}

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
my $cmd = "";
my $result = "";
my $response = "";
my $host = "";
my $file = "";
my $host_i = 0;
my $n_hosts = $cfg{"NUM_PWC"};
my @hosts = ();
my %in_use = ();
my %have_files = ();

for ($i=0; $i<$n_hosts; $i++)
{
  push @hosts, $cfg{"PWC_".$i};
  $in_use{$cfg{"PWC_".$i}} = 0;
  $have_files{$cfg{"PWC_".$i}} = 0;
}

my $p_sleep = 0;
my $input_string = "";
my $output_string = "";
my $host_in_use = 0;
my $in_use_string = "";
my $have_files_string = "";
my $have = 0;


while ((!$quit_daemon) || ($p_count > 0))
{

  $in_use_string = "";
  for ($i=0; $i<=$#hosts; $i++) {
    if ($in_use{$hosts[$i]}) {
      $in_use_string .= $hosts[$i]." ";
    }
  }

  # check if any files exist for the hosts
  $have_files_string = "";
  $have = 0;
  $cmd = "ls -1d /nfs/apsr??/apsr/P778_raw/*.dada 2>&1 | awk -F/ '{print \$3}' | sort | uniq";
  Dada::logMsg(2, $dl, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "main: ".$result." ".$response);
  $response =~ s/\n/ /g;
  for ($i=0; $i<$n_hosts; $i++)
  {
    $host = $hosts[$i];
    if (($result eq "ok") && ($response ne "apsr??") && ($response =~ m/$host/))
    {
      $have_files{$host} = 1;
      $have_files_string .= $host." ";

      # if this host is also not currently in use then we have a file to process
      if (!$in_use{$hosts[$i]})
      {
        $have = 1;
      }
    }
    else
    {
      $have_files{$host} = 0;
    } 
  }
  
  Dada::logMsg(2, $dl, "main: quit=".$quit_daemon.", p_count=".$p_count."/".$num_p_threads.", p_sleep=".$p_sleep." have_files=".$have_files_string." in_use=".$in_use_string);

  # if we have an inactive thread 
  if (($have) && (!$quit_daemon) && ($p_count <= $num_p_threads) && ($p_sleep <= 0))
  {

    # find next available host that is not in use or there are no files for the host...
    Dada::logMsg(2, $dl, "main: checking if ".$hosts[$host_i]." [".$host_i."] is not in use [".$in_use{$hosts[$host_i]}."] and have_files[".$have_files_string."]");
    while (($in_use{$hosts[$host_i]}) || (!$have_files{$hosts[$host_i]}))
    {
      $host_i = ($host_i + 1) % $n_hosts;
      Dada::logMsg(2, $dl, "      checking if ".$hosts[$host_i]." [".$host_i."] is in use (".$in_use{$hosts[$host_i]}.")");
    }
    $host = $hosts[$host_i];
    
    $input_string = $host;
    Dada::logMsg(2, $dl, "main: enqueue parkes [".$input_string."]");
    $p_in->enqueue($input_string);
    $p_count++;
    $in_use{$host} = 1;

    # incremenet the current host index
    $host_i++;
    if ($host_i >= $n_hosts)
    {
      $host_i = 0;
    }
  }
  else
  {
    if (($p_count == 0) && ($p_sleep <= 0))
    {
      $p_sleep = 6;
    }
  }
  if ($p_sleep > 0)
  {
    $p_sleep--;
  }
  
  Dada::logMsg(2, $dl, "main: sleep(1)"); 
  sleep(10);

  # if a beam has finished transfer, remove it from the queue

  while ($p_out->pending())
  {
    Dada::logMsg(2, $dl, "main: dequeuing transfer");
    $output_string = $p_out->dequeue();
    @bits = ();
    @bits = split(/ /,$output_string, 3);
    if ($#bits != 2)
    {
      Dada::logMsgWarn($warn, "main: dequeue string did not have 3 parts [".$output_string."]");
      $quit_daemon = 1;
    }
    $result = $bits[0];
    $host = $bits[1];
    $file = $bits[2];

    Dada::logMsg(2, $dl, "main: dequeued xfer ".$host.":".$file." [".$result."]");

    # decrement the parkes transfer count
    $p_count--;

    # remove this host from the nodes in use array
    $in_use{$host} = 0;

    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "main: xfer for ".$host." failed");
      $quit_daemon = 1;
    } 
  }

  setStatus($p_count." threads transferring");

}

# rejoin threads
Dada::logMsg(1, $dl, "main: joining controlThread");
$control_thread->join();

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
#  queue and runs the file transfer script on the specified host
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
  my $l_dir = "/lfs/data0/apsr/P778_raw";
  my $bwlimit = sprintf("\%d", (BANDWIDTH / $num_p_threads));
  my $rsync_options = "-a --stats --password-file=/home/apsr/.ssh/raid0_rsync_pw --no-g --chmod=go-ws ".
                      "--bwlimit=".$bwlimit;

  my $host = "";
  my $file = "";
  my $freq = "";
  my $utc_start = "";

  my @output_lines = ();
  my $data_rate = "";
  my $i = 0;
  my $junk = "";

  Dada::logMsg(2, $dl, "xfer[".$tid."] starting");

  while (!$quit_daemon) 
  {
    Dada::logMsg(3, $dl, "xfer[".$tid."] while(!quit_daemon)");

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
      $host = $input_string;

      # find a file to send on the specified host that was last modified 2 minutes ago
      $cmd = "find ".$l_dir." -type f -name '*.dada' -mmin +2 | sort -n | head -n 1";
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
      
      # we found something valid
      if (($response ne "") && ($response =~ m/dada/))
      {

      }
      else
      {
        if ($result ne "ok")
        {
          Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
          $out->enqueue("fail ".$host." ssh failed");
          next;
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "xfer[".$tid."] find failed: ".$response." [".$host."]"); 
          #$out->enqueue("fail ".$host." find failed");
          #next;
        }
        if ($response eq "")
        {
          $out->enqueue("ok ".$host." no files to transfer");
          next;
        }
      }
      $file = $response;

      # determine the centre frequency of the sub-band from the file
      $cmd = "dada_edit -c FREQ ".$file;
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
        $out->enqueue("fail ".$host." ssh failed");
        next;
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] dada_edit failed: ".$response);
        $out->enqueue("fail ".$host." dada_edit failed");
        next;
      }
      $freq = $response;

      $utc_start = basename($file);
      ($utc_start, $junk) = split(/_/, $utc_start);

      # create the utc_start subdir
      $cmd = "mkdir -m 0755 -p ".$r_path."/".$utc_start;
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$r_host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
        $out->enqueue("fail ".$host." ssh failed");
        next;
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] mkdir failed: ".$response);
        $out->enqueue("fail ".$host." mkdir failed");
        next;
      }

      # create the band subdir
      $cmd = "mkdir -m 0755 -p ".$r_path."/".$utc_start."/".$freq;
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$r_host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
        $out->enqueue("fail ".$host." ssh failed");
        next;
      }
      if ($rval != 0)
      {
        Dada::logMsgWarn($warn, "xfer[".$tid."] mkdir failed: ".$response);
        $out->enqueue("fail ".$host." mkdir failed");
        next;
      }

      Dada::logMsg(1, $dl, "xfer[".$tid."] ".$host."/".basename($file)." finished -> transferring");

      $cmd = "rsync ".$file." ".$r_user."@".$r_host."::".$r_module."/".$utc_start."/".$freq."/ ".$rsync_options;

      Dada::logMsg(2, $dl, "xfer[".$tid."] ".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $host, $cmd);
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

        # delete the local file
        $cmd = "rm -f ".$file;
        Dada::logMsg(2, $dl, "xfer[".$tid."] ".$r_user."@".$host.":".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $host, $cmd);
        Dada::logMsg(2, $dl, "xfer[".$tid."] ".$result." ".$rval." ".$response);
        if ($result ne "ok") 
        {
          Dada::logMsgWarn($warn, "xfer[".$tid."] ssh failed: ".$response);
          $out->enqueue("fail ".$host." ssh failed");
          next;
        }
        if ($rval != 0)
        {
          Dada::logMsgWarn($warn, "xfer[".$tid."] rm -f failed: ".$response);
          $out->enqueue("fail ".$host." mkdir failed");
          next;
        }

        Dada::logMsg(1, $dl, "xfer[".$tid."] ".$host."/".basename($file)." transferring -> transferred ".$data_rate);
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
        $output_string = "ok ".$host." ".$file;
      }
      else
      {
        $output_string = "fail ".$host." ".$file;
        Dada::logMsgWarn($warn, "xfer[".$tid."] failed to transfer ".$host.":".basename($file));
      }

      Dada::logMsg(2, $dl, "xfer[".$tid."]: dequeueing ".$host.":".basename($file));

      $out->enqueue($output_string);
    }

    sleep(1);
  }

  Dada::logMsg(2, $dl, "xfer[".$tid."]: exiting");
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
# Polls for the "quitdaemons" file in the control dir
#
sub controlThread($$) {

  my ($pid_file, $quit_file) = @_;

  Dada::logMsg(2, $dl, "controlThread: starting");

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
# Write the current status into the /nfs/control/apsr/xfer.state file
#
sub setStatus($) {

  (my $message) = @_;

  Dada::logMsg(3, $dl, "setStatus(".$message.")");

  my $dir = "/nfs/control/apsr/";
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
