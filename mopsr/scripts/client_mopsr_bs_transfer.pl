#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bs_transfer.pl 
#
# transfer filterbank files from bs clients to mpsr-bf08
#
###############################################################################

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
  print "Usage: ".basename($0)." BS_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our %bp_cfg : shared;
our %smirf_cfg : shared;
our $localhost : shared;
our $proc_id : shared;
our $db_key : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;
our $bw_limit;
our $srv_ip;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bs");
%smirf_cfg = Dada::readCFGFileIntoHash("/home/vivek/SMIRF/config/smirf.cfg", 0);
$proc_id = -1;
$db_key = "dada";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BS_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BS_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

# For transport via 1GbE
$bw_limit = int((64 * 1024) / $cfg{"NUM_BS"});
$srv_ip = "mpsr-bf08";

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$proc_id  = $ARGV[0];

# ensure that our proc_id is valid 
if (($proc_id >= 0) &&  ($proc_id < $cfg{"NUM_BS"}))
{
  # and matches configured hostname
  if ($cfg{"BS_".$proc_id} ne Dada::getHostMachineName())
  {
    print STDERR "BS_".$proc_id."[".$cfg{"BS_".$proc_id}."] did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "proc_id was not a valid integer between 0 and ".($cfg{"NUM_BS"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$proc_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".src.log";
  my $pid_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".pid";

  # Autoflush STDOUT
  $| = 1;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  msg (0, "INFO", "STARTING SCRIPT");

  my $bs_tag = sprintf ("BS%02d", $proc_id);
  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $utc_start, $source, $n);
  my @parts = ();

  my $proc_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag;

  # look for filterbank files to transfer to the server via rsync
  while (!$quit_daemon)
  {
    $cmd = "find ".$proc_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.finished' | sort -n -r | tail -n 1";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      # get the observation UTC_START
      @parts = split (/\//, $response);
      $n = $#parts;

      $utc_start = $parts[$n-1];

      msg(2, "INFO", "main: utc_start=".$utc_start);
      
      if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
      {
        my $local_dir = $proc_dir."/".$utc_start;
        msg (2, "INFO", "main: found finished observation: ".$local_dir);

        my $source_name_file = $proc_dir."/".$utc_start."/".$utc_start.".".$bs_tag.".sourcename";
        $cmd = "grep ^SOURCE ".$source_name_file." | awk '{print \$2}'";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        my $smirf_source = $response;

        my $remote_dir = $smirf_cfg{"SURVEY_DIR"}."/results/".$smirf_source."/".$utc_start;
        msg(2, "INFO", "main: createRemoteDir(".$remote_dir.")");
        ($result, $response) = createRemoteDir($remote_dir);
        msg(3, "INFO", "main: ".$result." ".$response);
        if ($result ne "ok")
        {
          msg(0, "WARN", "failed to create remote directory: ".$response);
          markObs($proc_dir."/".$utc_start, "obs.finished", "obs.failed");
        }
        else
        {
          ($result, $response) = processCars($proc_dir, $utc_start);
          if ($result ne "ok")
          {
            msg(2, "INFO", "main: failed to process CARS for ".$utc_start.": ".$response);
            markObs($proc_dir."/".$utc_start, "obs.finished", "obs.failed");
          }
          else
          {
            # transfer beams 
            ($result, $response) = transferBS ($local_dir, $utc_start, $smirf_source);
            if ($result ne "ok")
            {
              msg(0, "ERROR", "transfer of ".$utc_start." failed: ".$response);
              markObs($proc_dir."/".$utc_start, "obs.finished", "obs.failed");
            }
            else
            {
              msg(0, "INFO", "transfer of ".$utc_start." completed");
              markObs($proc_dir."/".$utc_start, "obs.finished", "obs.transferred");

              my ($rval);
              my $user = $smirf_cfg{"SURVEY_USER"};
              my $host = $smirf_cfg{"SURVEY_HOST"};

              $cmd = "touch ".$smirf_cfg{"SURVEY_DIR"}."/results/".$smirf_source."/".$utc_start."/obs.transferred.".$bs_tag;
              msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
              ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
              msg(2, "INFO", "main: ".$result." ".$rval." ".$response);
              if (($result ne "ok") || ($rval != 0))
              {       
                msg(0, "WARN", "could not touch obs.transferred.".$bs_tag.": ".$response);
              }
            }
          }
        } # create remote dir ok
      } # utc_start match regex
    } # obs.finished file exists
        
    my $counter = 10;
    while (!$quit_daemon && $counter > 0)
    {
      sleep(1);
      $counter --;
    }
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($sys_log_sock);

  exit (0);
}

sub markObs ($$$)
{
  my ($dir, $old_file, $new_file) = @_;

  my ($cmd, $result, $response);

  $cmd = "mv ".$dir."/".$old_file." ".$dir."/".$new_file;
  msg(2, "INFO", "markObs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "markObs: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "could not mark obs ".$new_file);
  }
  else
  {
    return ("ok", "");
  }
}

sub createRemoteDir($)
{
  my ($remote_dir) = @_;

  my $user = $smirf_cfg{"SURVEY_USER"};
  my $host = $smirf_cfg{"SURVEY_HOST"};
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

sub processCars($$)
{
  my ($proc_dir, $utc_start) = @_;

  my $local_dir = $proc_dir."/".$utc_start;
  my ($cmd, $result, $response, $line);
  my $bs_tag = sprintf ("BS%02d", $proc_id);

  # homogenise the directory structure thanks to DSPSR's single candidate director
  # naming convention
  $cmd = "find ".$local_dir."/cars -mindepth 3 -maxdepth 3 -type f -name '*.car' | sort";
  msg(2, "INFO", "processCars ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "processCars: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "find of .car files failed");
  }
  elsif ($response eq "")
  {
    msg(1, "INFO", "no candidate archives found for ".$utc_start);
    return ("ok", "no CARs found");
  }
  else
  {
    my @lines = split(/\n/, $response);
    my ($line);

    foreach $line ( @lines )
    {
      my @cars = split(/\//, $line);
      my $m = $#cars;
      my $car = $cars[$m];
      my $parent_dir = $cars[$m-1];
      my $smirf_pointing_name = $cars[$m-2];

      my @cands = split(/\_/, $parent_dir, 3);
      my $cand_num = $cands[2];
      my @radecs = split(/\./, $smirf_pointing_name, 2);
      my $radec = $radecs[0];

      my $new_filename = $local_dir."/cars/".$radec."_".$bs_tag."_".$cand_num.".car";

      $cmd = "mv ".$line." ".$new_filename;
      msg(2, "INFO", "processCars: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "processCars: ".$result." ".$response);
      if ($result ne "ok")
      {
        return ("fail", "failed to rename ".$line.": ".$response);
      }
    }

    $cmd = "ls -1 ".$local_dir."/cars/*/obs.headers | sort -n";
    msg(2, "INFO", "processCars: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "processCars: ".$result." ".$response);
    if (($result eq "ok") && ($response ne ""))
    {
      my @headers = split(/\n/, $response);
      $cmd = "touch ".$local_dir."/cars/obs.headers.".$bs_tag;
      my $header;
      foreach $header ( @headers )
      {
        $cmd = "echo #========".$header."=========== >> ".$local_dir."/cars/obs.headers.".$bs_tag;
        msg(2, "INFO", "processCars: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "processCars: ".$result." ".$response);
        if ($result ne "ok")
        { 
          return ("fail", "could not append obs.headers");
        }

        $cmd = "cat ".$header." >> ".$local_dir."/cars/obs.headers.".$bs_tag;
        msg(2, "INFO", "processCars: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "processCars: ".$result." ".$response);
        if ($result ne "ok")
        {
          return ("fail", "could not append obs.headers");
        }
      }
    }

    $cmd = "rm -f ".$local_dir."/cars/*/obs.headers";
    msg(2, "INFO", "processCars: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "processCars: ".$result." ".$response);
    if ($result ne "ok")
    {
      return ("fail", "could not delete cars/*/obs.headers");
    }

    $cmd = "rmdir ".$local_dir."/cars/*/*/";
    msg(2, "INFO", "processCars: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "processCars: ".$result." ".$response);
    if ($result ne "ok")
    {
      return ("fail", "could not rmdir cars/*/*/");
    }

    $cmd = "rm -rf ".$local_dir."/cars/*/";
    msg(2, "INFO", "processCars: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "processCars: ".$result." ".$response);
    if ($result ne "ok")
    {
      return ("fail", "could not rmdir cars/*/");
    }

  }
  return ("ok", "");
}

sub transferBS ($$$)
{
  my ($local_dir, $utc_start, $smirf_source) = @_;
  my ($cmd, $result, $response);

  $cmd = "rsync -a ".$local_dir." upload\@".$srv_ip."::smirf/results/".$smirf_source."/ ".
         "--exclude 'obs.headers' --exclude 'obs.stitched' --exclude 'obs.folded' --exclude 'obs.finished' ".
         "--stats --bwlimit=".$bw_limit." --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd";
  msg(2, "INFO", "transferBS: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "transferBS: ".$result." ".$response);
  if ($result ne "ok")
  { 
    if ($quit_daemon)
    { 
      msg(0, "INFO", "transfer of ".$utc_start." interrupted");
      return ("ok", "rsync of beams interrupted"); 
    }
    else
    { 
      msg(0, "ERROR", "transfer of ".$utc_start." failed: ".$response);
      return ("fail", "rsync of beams failed"); 
    }
  }

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
              
  msg(1, "INFO", $utc_start." finished -> transferred [".$data_rate."]");

  return ("ok", "");
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
    if (!($sys_log_sock)) {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock) {
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bs_transfer", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  my $bs_tag = sprintf ("BS%02d", $proc_id);
  $cmd = "^rsync -a ".$cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag;
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO", "controlThread: killProcess() ".$result." ".$response);

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
    if ($sys_log_sock) {
      close($sys_log_sock);
    }
  }
}

sub sigPipeHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $sys_log_sock = 0;
  if ($log_host && $sys_log_port) {
    $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  }
}

