#!/usr/bin/env perl

###############################################################################
#
# client_mopsr_bf_corr_results.pl 
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;
use Getopt::Std;
use File::Basename;
use Mopsr;
use strict;
use threads;
use threads::shared;


sub usage() 
{
  print "Usage: ".basename($0)." BF_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $bf_id : shared;
our $bf_tag : shared;
our $log_host;
our $log_port;
our $log_sock;
our $nchan_in;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bf");
$bf_id = -1;
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_BF_SYS_LOG_PORT"};
$log_sock = 0;


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$bf_id  = $ARGV[0];
$bf_tag = "BF".sprintf("%02d",$bf_id);
$nchan_in = ($cfg{"RECV_CHAN_LAST_".$bf_id} - $cfg{"RECV_CHAN_FIRST_".$bf_id}) + 1;

# ensure that our bf_id is valid 
if (($bf_id >= 0) &&  ($bf_id < $cfg{"NUM_BF"}))
{
  # and matches configured hostname
  if ($cfg{"BF_".$bf_id} ne Dada::getHostMachineName())
  {
    print STDERR "BF_".$bf_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "bf_id was not a valid integer between 0 and ".($cfg{"NUM_BF"}-1)."\n";
  usage();
  exit(1);
}


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$bf_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bf_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bf_id.".pid";
  my $use_nfs = 1;

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  # open a connection to the server_sys_monitor.pl script
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) 
  {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }
  msg (0, "INFO", "STARTING SCRIPT");

  my $control_thread = threads->new(\&controlThread, $pid_file);

  my $client_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bf_tag;

  my $last_obs = "";
  my $last_plot = 0;
  my $should_plot_cc = "";
  my $should_plot_ac = "";

  my ($cmd, $result, $response, $obs, $range, $file, $file_list);
  my $sleep_total = 5;
  my $sleep_count;

  my @ac_files;
  my @cc_files;
  my %to_send;
  my $ac_file;
  my $cc_file;

  my ($ref_ant, $ref_ant_number);

  if (! -d $client_dir)
  {
    $cmd = "mkdir -p ".$client_dir;
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      msg(0, "WARN", "main: ".$cmd." failed: ".$response);
    }
  }

  while (!$quit_daemon)
  {
    ($result, $response) = procXGPUFiles($client_dir);
    if ($result ne "ok")
    {
      msg(0, "WARN", "failed to process XGPU files: ".$response);
    }

    ($result, $response) = sumCorrFiles($client_dir, "ac");
    if ($result ne "ok")
    {
      msg(0, "WARN", "failed to sum ac files: ".$response);
    }

    ($result, $response) = sumCorrFiles($client_dir, "cc");
    if ($result ne "ok")
    {
      msg(0, "WARN", "failed to sum cc files: ".$response);
    }

    $sleep_count = 0;
    while (!$quit_daemon && ($sleep_count < $sleep_total))
    {
      sleep(1);
      $sleep_count++;
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($log_sock);

  exit (0);
}


sub sumCorrFiles($$)
{
  my ($dir, $ext) = @_;

  my $sum = $ext.".sum";
  my $last_sum = "";
  my ($cmd, $result, $rval, $response, $obs, $range, $file, $file_fscr, $nant);
  my ($line, @lines, @paths, $n, $utc_start, $source);
  my ($local_dir, $remote_dir);
  my @files;

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"};
  my $server_host = $cfg{"SERVER_HOST"};
  my $server_user = $cfg{"USER"};

  # look for any corr files
  $cmd = "find ".$dir." -mindepth 3 -maxdepth 3 -type f -name '*.".$ext."' | sort -n";

  msg(2, "INFO", "sumCorrFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    msg(0, "WARN", "find list of dump files failed: ".$response);
    return ("fail", "");
  }
  elsif ($response eq "")
  {
    msg(2, "INFO", "sumCorrFiles: no .".$ext." files found");
  }
  else
  {
    @lines = split(/\n/, $response);
    msg(2, "INFO", "sumCorrFiles: found ".($#lines+1)." .".$ext." files");
    if ($#lines >= 0)
    {
      foreach $line (@lines)
      {
        # since archives will be of the form . / UTC_START / SOURCE / FILE, grab last 3
        @paths = split(/\//, $line);
        $n = $#paths;

        $utc_start = $paths[$n-2];
        $source    = $paths[$n-1];
        $file      = $paths[$n];

        # determine UTC for this file
        ($obs, $range) = split(/_/, $file, 2);

        $local_dir = $dir."/".$obs."/".$source;
        $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/".$source."/".$bf_tag;

        # determine the number of antenna

        $cmd = "grep ^NANT ".$local_dir."/obs.header | awk '{print \$2}'";
        msg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "sumCorrFiles: ".$result." ".$response);
        if ($result ne "ok")
        {
          return ("fail", "could not determine NANT from obs.header");
        }
        $nant = $response;

        my $nchan_files = 0;
        my @stat = stat $local_dir."/".$file;
        my $filesize = $stat[7];
        # check if the number of input channels matches the expected value
        if ($ext eq "ac")
        {
          $nchan_files = $filesize / ($nant * 4);
        }
        else
        {
          my $nbaselines = ($nant * ($nant - 1)) / 2;
          $nchan_files = $filesize / ($nbaselines * 8);
        }

        my $nchan_out = 0;
        # XGPU mode, no scrunching...
        if ($nchan_files == $nchan_in)
        {
          $nchan_out = $nchan_in;
        }
        # regular FX correlation mode
        elsif ($nchan_files == ($nchan_in * $cfg{"CORR_F_CHANNELISATION"}))
        {
          $nchan_out = ($nchan_in * $cfg{"CORR_F_CHANNELISATION"}) / $cfg{"CORR_F_CHAN_SCRUNCH"};
        }
        else
        {
          return ("fail", "could not determine correlation Fscrunch factor");
        }

        # create a scrunched copy of this file
        $file_fscr = $file.".fscr";
        if ($ext eq "ac")
        {
          $cmd = "mopsr_corr_fscr -a -F ".$nchan_out." ".$nant." ".$local_dir."/".$file." ".$local_dir."/".$file_fscr;
        }
        else
        {
          $cmd = "mopsr_corr_fscr -F ".$nchan_out." ".$nant." ".$local_dir."/".$file." ".$local_dir."/".$file_fscr;
        }
        msg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "sumCorrFiles: ".$result." ".$response);

        if (!(-f $local_dir."/header.copied"))
        {
          ($result, $response) = createRemoteDir ($remote_dir);
          if ($result ne "ok")
          {
            return ("fail", "could not create remote directory");
          }

          $cmd = "rsync -a ".$local_dir."/obs\.* ".
          $server_user."\@".$server_host.":".$remote_dir."/";
          msg(2, "INFO", "sumCorrFiles: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            msg(0, "WARN", $cmd." failed: ".$response);
            next;
          }
          $cmd = "touch ".$local_dir."/header.copied";
          msg(2, "INFO", "sumCorrFiles: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "sumCorrFiles: ".$result." ".$response);
        }

        # copy the new Fscrunch file to the server
        $cmd = "rsync -a ".$local_dir."/".$file_fscr." ".
               $server_user."\@".$server_host.":".$remote_dir."/".$file;
        msg(2, "INFO", "sumCorrFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          msg(0, "WARN", $cmd." failed: ".$response);
          next;
        }

        $cmd = "rm -f  ".$local_dir."/".$file." ".$local_dir."/".$file_fscr;
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          msg(0, "WARN", "sumCorrFiles: ".$cmd." failed: ".$response);
        }
      }
    }
  }
  return ("ok", "");
}

sub procXGPUFiles($)
{
  my ($dir) = @_;

  my $last_sum = "";
  my ($cmd, $result, $rval, $response, $obs, $range, $file, $ac_file, $cc_file, $nant);
  my ($line, @lines, @paths, $n, $source, $utc_start, $source);
  my ($local_dir, $remote_dir);

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"};
  my $server_host = $cfg{"SERVER_HOST"};
  my $server_user = $cfg{"USER"};

  # look for any xc files
  $cmd = "find ".$dir." -mindepth 3 -maxdepth 3 -type f -name '*.xc' | sort -n";

  msg(2, "INFO", "procXGPUFiles: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    msg(0, "WARN", "find list of xc files failed: ".$response);
    return ("fail", "");
  }
  elsif ($response eq "")
  {
    msg(2, "INFO", "procXGPUFiles: no xc files found");
  }
  else
  {
    @lines = split(/\n/, $response);
    msg(2, "INFO", "procXGPUFiles: found ".($#lines+1)." .xc files");
    if ($#lines >= 0)
    {
      foreach $line (@lines)
      {
        # since archives will be of the form . / UTC_START / SOURCE / FILE, grab last 3
        @paths = split(/\//, $line);
        $n = $#paths;

        $utc_start = $paths[$n-2];
        $source    = $paths[$n-1];
        $file      = $paths[$n];

        # determine UTC for this file
        ($obs, $range) = split(/_/, $file, 2);

        $local_dir = $dir."/".$obs."/".$source;
        $remote_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/".$source."/".$bf_tag;

        #  convert the XC file into ac and cc files
        $cmd = "mopsr_xgpu_convert ".$local_dir."/".$file;
        msg(1, "INFO", "procXGPUFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
        if  ($result ne "ok")
        {
          return  ("fail", "could not convert xGPU dump file");
        }

        $ac_file = $file;
        $cc_file = $file;

        $ac_file =~ s/\.xc$/.ac/;
        $cc_file =~ s/\.xc$/.cc/;

        # determine NANT
        $cmd = "grep ^NANT ".$local_dir."/obs.header | awk '{print \$2}'";
        msg(2, "INFO", "procXGPUFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
        if ($result ne "ok")
        {
          return ("fail", "could not determine NANT from obs.header");
        }
        $nant = $response;

        if (! -f $local_dir."/header.copied")
        {
          ($result, $response) = createRemoteDir ($remote_dir);
          if ($result ne "ok")
          {
            return ("fail", "could not create remote directory");
          }

          $cmd = "rsync -a ".$local_dir."/obs\.* ".
          $server_user."\@".$server_host.":".$remote_dir."/";
          msg(2, "INFO", "procXGPUFiles: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
          if ($result ne "ok")
          {
            msg(0, "WARN", $cmd." failed: ".$response);
            next;
          }
          $cmd = "touch ".$local_dir."/header.copied";
          msg(2, "INFO", "procXGPUFiles: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
        }

        # copy the new new Fscrunch archive to the server
        $cmd = "rsync -a ".$local_dir."/".$ac_file." ".$local_dir."/".$cc_file." ".
               $server_user."\@".$server_host.":".$remote_dir."/";
        msg(2, "INFO", "procXGPUFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
        if ($result ne "ok")
        {
          msg(0, "WARN", $cmd." failed: ".$response);
          next;
        }

        $cmd = "rm -f  ".$local_dir."/".$ac_file." ".$local_dir."/".$cc_file." ".$local_dir."/".$file;
        msg(2, "INFO", "procXGPUFiles: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "procXGPUFiles: ".$result." ".$response);
        if ($result ne "ok")
        {
          msg(0, "WARN", "procXGPUFiles: ".$cmd." failed: ".$response);
        }
      }
    }
  }
  return ("ok", "");
}

sub createRemoteDir($)
{
  my ($remote_dir) = @_;

  my $user = $cfg{"USER"};
  my $host = $cfg{"SERVER_HOST"};
  my $cmd = "mkdir -m 2755 -p ".$remote_dir;
  my ($result, $rval, $response);

  msg(2, "INFO", "createRemoteDir: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  msg(2, "INFO", "createRemoteDir: ".$result." ".$rval." ".$response);

  if (($result eq "ok") && ($rval == 0))
  {
    msg(2, "INFO", "createRemoteDir: remote directory created");
    return ("ok", "");
  }
  else
  {
    return ("fail", $response);
  }
}

#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$)
{
  my ($level, $type, $msg) = @_;

  if ($level <= $dl)
  {
    my $time = Dada::getCurrentDadaTime();
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, sprintf("%02d",$bf_id), $time, "sys", $type, "corr_results", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bf_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");

}

sub sigHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

  # if we CTRL+C twice, just hard exit
  if ($quit_daemon) {
    print STDERR $daemon_name." : Recevied 2 signals, Exiting\n";
    exit 1;

  # Tell threads to try and quit
  } else {

    $quit_daemon = 1;
    if ($log_sock) {
      close($log_sock);
    }
  }
}

sub sigPipeHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}

