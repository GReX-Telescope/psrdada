#!/usr/bin/env perl

###############################################################################
#
# Transfers APSR observations to CASPSR Raid array for archival
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Dada;
use Apsr;

#
# Constants
#

#
# Function prototypes
#
sub markBandState($$$$);
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
%cfg = Apsr::getConfig();
$user = "apsr";
$bwlimit = int(50000 / int($cfg{"NUM_PWC"}));  # 50 MB/s / NUM_PWC (normally 3.1 MB/s)
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

  my $r_user         = "apsr";
  my $r_host         = "raid0";
  my $r_path         = "/lfs/raid0/apsr/finished";
  my $r_module       = "apsr_upload";
  my $a_dir          = $cfg{"CLIENT_ARCHIVE_DIR"};
  my $r_dir          = "";

  my $control_thread = 0; 
  my $rsync_options = "-a --password-file=/home/apsr/.ssh/raid0_rsync_pw --stats --no-g --chmod=go-ws ".
                      "--exclude 'band.finished' --bwlimit=".$bwlimit;

  my $o = "";
  my $b = "";
  my $s = "";
  my $pid = "";
  my @sources = ();
  my @finished = ();

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $l_dir = "";
  my $transfer_ok = 0;
  my $rval = 0;

  my $i = 0;
  my $j = 0;
  my $k = 0;

  my @output_lines = ();
  my $mbytes_per_sec = 0;
  my $mbytes = 0;
  my $seconds = 0;
  my @bits = ();
  my $xfer_rate = "";

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

  msg(0, "INFO", "STARTING SCRIPT BANDWIDTH=".sprintf("%2.1f",($bwlimit/1024))." MB/s");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # main Loop
  while ( !$quit_daemon ) 
  {
    @finished = ();

    # look for observations marked band.finished
    $cmd = "find ".$a_dir." -mindepth 3 -maxdepth 3 -type f ".
           "-name 'band.finished' -printf '\%h\n' | awk -F/ '{print \$(NF-1)\"\/\"\$(NF)}' | sort";
    msg(3, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(1, "WARN", "main: ".$cmd." failed: ".$response);
    }
    else
    {
      @finished = split(/\n/, $response);
      msg(2, "INFO", "main: found ".($#finished+1)." finished observations");

      for ($i=0; (!$quit_daemon &&  $i<=$#finished); $i++)
      {
        ($o, $b) = split(/\//, $finished[$i]);

        msg(2, "INFO", "main: processing ".$o."/".$b);
        @sources = ();

        # get the SOURCE[s] of the observation
        $cmd = "find ".$a_dir."/".$o."/".$b." -mindepth 1 -maxdepth 1 -type d -printf '\%f\n'";
        msg(3, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq "")) 
        {
          # not a multifold pulsar, so extract source from obs.start
          $cmd = "grep SOURCE ".$a_dir."/".$o."/".$b."/obs.start | awk '{print \$2}'";
          msg(3, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if (($result ne "ok") || ($response eq ""))
          {
            msg(1, "WARN", "main: failed to determine SOURCE for ".$o.": ".$response);
            markBandState($o."/".$b, "finished", "bad", "could not determine SOURCE");
            next;
          }
          push @sources, $response;
        }
        else
        {
          @sources = split(/\n/, $response);
        }

        # get the PID
        $cmd = "grep PID ".$a_dir."/".$o."/".$b."/obs.start | awk '{print \$2}'";
        msg(3, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          msg(1, "WARN", "main: failed to determine PID for ".$o.": ".$response);
          markBandState($o."/".$b, "finished", "bad", "could not determine PID");
          next;
        }
        $pid = $response;

        # check the PID matches an APSR pid
        my $apsr_groups = `groups apsr`;
        chomp $apsr_groups;
        if (!($apsr_groups =~ m/$pid/)) 
        {
          msg(1, "WARN", "main: PID [".$pid."] was not a valid APSR group for ".$o);
          markBandState($o."/".$b, "finished", "bad", "PID [".$pid."] was not an APSR group");
          next;
        }

        msg(2, "INFO", "main: found ".($#sources+1)." sources for ".$o);

        # copy the archives from each source to the caspsr RAID server
        $transfer_ok = 1;
        for ($j=0; (!$quit_daemon && $j<=$#sources); $j++)
        {
          $s = $sources[$j];
          if ($s eq "") 
          {
            msg(1, "WARN", "main: SOURCE was empty for ".$o);
            markBandState($o."/".$b, "finished", "bad", "SOURCE was empty");
            next;
          }

          # ensure the PID / SOURCE directory exists on RAID server
          $r_dir = $r_path."/".$pid."/".$s."/".$o;
          $cmd = "mkdir -p ".$r_dir;
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
            $transfer_ok = 0;
            $quit_daemon = 1;
            next;
          }

          # If this is not a multi-fold obs, there will be no source subdirectory
          if ($#sources == 0)
          { 
            $l_dir = $a_dir."/".$o."/".$b;
          }
          else
          {
            $l_dir = $a_dir."/".$o."/".$b."/obs.start ".$a_dir."/".$o."/".$b."/".$s;
          }

          # transfer archives via rsync 
          #$cmd = "rsync -a --no-g --chmod=go-ws --exclude 'band.finished' --bwlimit=".$bwlimit." ".$l_dir."/ ".$r_user."@".$r_host.":".$r_dir."/".$b;
          $cmd = "rsync ".$l_dir."/ ".$r_user."@".$r_host."::".$r_module."/".$pid."/".$s."/".$o."/".$b." ".$rsync_options;
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            $transfer_ok = 0;
            if ($quit_daemon)
            {
              msg(0, "INFO", "main: rsync failed but due to daemon quitting, deleting partially transferred band");

              # clean up remote dir
              $cmd = "rm -rf ".$r_dir."/".$b;
              msg(1, "INFO", "main: remoteSsh(".$r_user.", ".$r_host.", ".$cmd.")");
              ($result, $rval, $response) = Dada::remoteSshCommand($r_user, $r_host, $cmd);
              msg(1, "INFO", "main: ".$result." ".$response);

            }
            else
            {
              msg(0, "WARN", "main: rsync [".$cmd."] failed: ".$response);
              # this is a fairly serious error - exit
              $quit_daemon = 1;
              next;
            }
          }
          else 
          {
          
            # determine the data rate
            @output_lines = split(/\n/, $response);
            $mbytes_per_sec = 0;
            $mbytes = 0;
            $seconds = 0;
            for ($k=0; $k<=$#output_lines; $k++)
            {
              if ($output_lines[$k] =~ m/bytes\/sec/)
              {
                @bits = split(/[\s]+/, $output_lines[$k]);
                $mbytes_per_sec = $bits[6] / 1048576;
                $mbytes = $bits[1] / 1048576;
                $seconds = $mbytes / $mbytes_per_sec;
              }
            }
            $xfer_rate = sprintf("%2.1f", $mbytes)." MB in ".sprintf("%2.1f",$seconds)."s, ".sprintf("%2.1f", $mbytes_per_sec)." MB/s";

            $r_dir = $r_path."/".$pid."/".$s."/".$o;
            $cmd = "touch ".$r_dir."/".$b."/band.transferred";
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
              $transfer_ok = 0;
              next;
            }
          }
        }
        if ($transfer_ok)
        {
          markBandState($o."/".$b, "finished", "transferred", "");
          msg(1, "INFO", $pid."/".$s."/".$o." finished -> transferred ".$xfer_rate);
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
# change the state of the specified band
#
sub markBandState($$$$)
{
  my ($o, $from, $to, $message) = @_;

  my $dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$o;
  my $cmd = "";
  my $result = "";
  my $response = "";
  
  if (! -f $dir."/band.".$from) 
  {
    msg(0, "WARN", "markBandState: from state [".$from."] did not exist");
    $cmd = "rm -f ".$dir."/band.*";
  }
  else
  {
    $cmd = "rm -f ".$dir."/band.".$from;
  }
  msg(3, "INFO", "markBandState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "markBandState: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "WARN", "markBandState: ".$cmd." failed: ".$response);
  }

  if ($message ne "")
  {
    $cmd = "echo '".$message."' > band.".$to;
  } 
  else
  {
    $cmd = "touch ".$dir."/band.".$to;
  }
  msg(3, "INFO", "markBandState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "markBandState: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "WARN", "markBandState: ".$cmd." failed: ".$response);
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
    #msg(1, "WARN", "controlThread: PID file did not exist on script exit");
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
      Dada::nexusLogMessage($log_sock, $time, "sys", $type, "xfer mngr", $msg);
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
