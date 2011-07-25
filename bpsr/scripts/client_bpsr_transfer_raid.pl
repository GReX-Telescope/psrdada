#!/usr/bin/env perl

###############################################################################
#
# Transfers BPSR observations to CASPSR Raid array for archival
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Bpsr;

#
# Constants
#

#
# Function prototypes
#
sub markBeamState($$$$$);
sub controlThread($$);
sub good($);
sub msg($$$);

#
# Global variable declarations
#
our $dl : shared;
our %cfg : shared;
our $user : shared;
our $daemon_name : shared;
our $bwlimit;
our $quit_daemon : shared;
our $log_host;
our $log_port;
our $log_sock;

#
# Global initialization
#
$dl = 1;
%cfg = Bpsr::getConfig();
$user = "bpsr";
$bwlimit = int(70000 / int($cfg{"NUM_PWC"}));  # 70 MB/s / NUM_PWC (nomially 6.1 MB/s)
$daemon_name = Dada::daemonBaseName($0);
$quit_daemon = 0;
$log_host = 0;
$log_port = 0;
$log_sock = 0;

# Autoflush STDOUT
$| = 1;

# Main
{

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file      = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";

  $log_host          = $cfg{"SERVER_HOST"};
  $log_port          = $cfg{"SERVER_SYS_LOG_PORT"};

  my $r_user         = "bpsr";
  my $r_host         = "raid0";
  my $r_path         = "/lfs/raid0/bpsr/finished";
  my $r_module       = "bpsr_upload";
  my $a_dir          = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $r_dir          = "";

  my $control_thread = 0;

  my $o = "";
  my $b = "";
  my $pid = "";
  my @sources = ();
  my @finished = ();

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $l_dir = "";
  my $rval = 0;

  my $i = 0;
  my $j = 0;

  my $rsync_options = "-a --password-file=/home/bpsr/.ssh/raid0_rsync_pw --stats --no-g --chmod=go-ws ".
                      "--exclude 'aux' --exclude 'beam.finished' --bwlimit=".$bwlimit;

  # quick sanity check
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    exit 1;
  }

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
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # main Loop
  while ( !$quit_daemon ) 
  {
    @finished = ();

    # look for observations marked beam.finished
    $cmd = "find ".$a_dir." -mindepth 3 -maxdepth 3 -type f -name 'beam.finished' -printf '\%h\n' | awk -F/ '{print \$(NF-1)\"\/\"\$(NF)}' | sort";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(1, "WARN", "main: ".$cmd." failed: ".$response);
    }
    else
    {
      # a list of UTC_STARTS that are marked as finished
      @finished = split(/\n/, $response);
      msg(2, "INFO", "main: found ".($#finished+1)." finished observations");

      for ($i=0; (!$quit_daemon &&  $i<=$#finished); $i++)
      {
        ($o, $b) = split(/\//, $finished[$i]);
        msg(2, "INFO", "main: processing ".$o."/".$b);

        # get the PID
        $cmd = "grep PID ".$a_dir."/".$o."/".$b."/obs.start | awk '{print \$2}'";
        msg(3, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          msg(1, "WARN", "main: failed to determine PID for ".$o.": ".$response);
          markBeamState($o, $b, "finished", "bad", "could not determine PID");
          next;
        }
        $pid = $response;

        # check the PID matches an BPSR pid
        my $bpsr_groups = `groups bpsr`;
        chomp $bpsr_groups;
        if (!($bpsr_groups =~ m/$pid/)) 
        {
          msg(1, "WARN", "main: PID [".$pid."] was not a valid BPSR group for ".$o);
          markBeamState($o, $b, "finished", "bad", "PID [".$pid."] was not an BPSR group");
          next;
        }

        # ensure the PID / OBS directory exists on RAID server
        $r_dir = $r_path."/".$pid."/".$o;
        $cmd = "mkdir -m 0755 -p ".$r_dir;
        msg(2, "INFO", "main: ".$cmd);
        ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd, $r_path);
        msg(3, "INFO", "main: ".$result." ".$response);
        if (($result ne "ok") || ($rval != 0))
        {
          if ($rval != 0) {
            msg(0, "WARN", "main: remote cmd [".$cmd."] failed: ".$response);
          } else {
            msg(0, "WARN", "main: ssh failed ".$response);
          }
          # this is a fairly serious error - exit
          $quit_daemon = 1;
          next;
        }

        # transfer archives via rsync 
        $cmd = "rsync ".$a_dir."/".$o."/".$b." ".$r_host."::".$r_module."/".$pid."/".$o."/ ".$rsync_options;
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          # clean up remote dir
          $cmd = "rm -rf ".$r_dir."/".$b;
          msg(1, "INFO", "main: remoteSsh(".$r_user.", ".$r_host.", ".$cmd.")");
          ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
          msg(1, "INFO", "main: ".$result." ".$response);

          if ($quit_daemon)
          {
            msg(0, "INFO", "main: rsync failed but due to daemon quitting");
          }
          else
          {
            $quit_daemon = 1;
            msg(0, "WARN", "main: rsync [".$cmd."] failed: ".$response);
            msg(1, "INFO", $o.": finished -> bad");
            markBeamState($o, $b, "finished", "bad", "rsync failure: ".$response);
            next;
          }
        }
        else 
        {

          # determine the data rate
          my @output_lines = split(/\n/, $response);
          my $mbytes_per_sec = 0;
          my $j = 0;
          for ($j=0; $j<=$#output_lines; $j++)
          {
            if ($output_lines[$j] =~ m/bytes\/sec/)
            {
              my @bits = split(/[\s]+/, $output_lines[$j]);
              $mbytes_per_sec = $bits[6] / 1048576;
            }
          }
          my $data_rate = sprintf("%5.2f", $mbytes_per_sec)." MB/s";

          $r_dir = $r_path."/".$pid."/".$o;
          $cmd = "touch ".$r_dir."/".$b."/beam.transferred";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd, $r_path);
          msg(3, "INFO", "main: ".$result." ".$response);
          if (($result ne "ok") || ($rval != 0))
          {
            if ($rval != 0) {
              msg(0, "WARN", "main: remote cmd [".$cmd."] failed: ".$response);
            } else {
              msg(0, "WARN", "main: ssh failed ".$response);
            }
          }
          # if we transferred everything ok, touch the sent.to.swin and sent.to.parkes files
          else 
          {
            $cmd = "touch ".$a_dir."/".$o."/".$b."/sent.to.swin ".$a_dir."/".$o."/".$b."/sent.to.parkes";
            msg(2, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(3, "INFO", "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              msg(0, "WARN", "main: could not touch sent.to flags: ".$response);
              markBeamState($o, $b, "finished", "bad", "could not touch sent.to flags");
              msg(1, "INFO", $o.": finished -> failed");
            }
            else 
            {
              markBeamState($o, $b, "finished", "transferred", "");
              msg(1, "INFO", $o.": finished -> transferred ".$data_rate);
            }
          }
        }
      }
    }

    my $counter = 10;
    msg(2, "INFO", "main: sleeping ".($counter)." seconds");
    while ((!$quit_daemon) && ($counter > 0) && ($#finished == -1)) 
    {
      sleep(1);
      $counter--;
    }
  }

  msg(2, "INFO", "main: joining threads");
  $control_thread->join();
  msg(2, "INFO", "main: control_thread joined");

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

}

exit 0;

###############################################################################
#
# Functions
#

#
# change the state of the specified beam
#
sub markBeamState($$$$$)
{
  my ($o, $b, $from, $to, $message) = @_;

  my $dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$o."/".$b;
  my $cmd = "";
  my $result = "";
  my $response = "";
  
  if (! -f $dir."/beam.".$from) 
  {
    msg(0, "WARN", "markBeamState: from state [".$from."] did not exist");
    $cmd = "rm -f ".$dir."/beam.*";
  }
  else
  {
    $cmd = "rm -f ".$dir."/beam.".$from;
  }
  msg(3, "INFO", "markBeamState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "markBeamState: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "WARN", "markBeamState: ".$cmd." failed: ".$response);
  }

  if ($message ne "")
  {
    $cmd = "echo '".$message."' > ".$dir."/beam.".$to;
  } 
  else
  {
    $cmd = "touch ".$dir."/beam.".$to;
  }
  msg(3, "INFO", "markBeamState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "markBeamState: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "WARN", "markBeamState: ".$cmd." failed: ".$response);
    $result = "fail";
  }
  return $result;
}

#
# control thread to ask daemon to quit
#
sub controlThread($$) 
{
  my ($quit_file, $pid_file) = @_;
  msg(2, "INFO", "controlThread: starting");

  my $cmd = "";
  my $regex = "";
  my $result = "";
  my $response = "";

  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  $quit_daemon = 1;

  $regex = "^rsync";
  msg(2, "INFO", "controlThread: killProcess(".$regex.", ".$user.")");
  ($result, $response) = Dada::killProcess($regex, $user);
  msg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(1, "WARN", "controlThread: killProcess for ".$regex." failed: ".$response);
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
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      print "opening nexus log: ".$log_host.":".$log_port."\n";
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "xfer", $msg);
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
  
  # tell threads to try and quit
  if (($sigName ne "INT") || ($quit_daemon))
  {
    $quit_daemon = 1;
    sleep(3);
  
    if ($log_sock) {
      close($log_sock);
    } 
  
    print STDERR $daemon_name." : Exiting\n";
    exit 1;
  }
  $quit_daemon = 1;
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

#
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

  if ( ($daemon_name eq "") || ($user eq "") ) {
    return ("fail", "Error: a package variable missing [daemon_name, user]");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}
