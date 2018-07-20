#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bp_filterbank_manager.pl 
#
# transfer filterbank files from clients to the server
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
  print "Usage: ".basename($0)." PROC_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our %smirf_cfg : shared;
our %ct : shared;
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
our $smirf_bw_limit;
our $srv_ip;
our $smirf_ip;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bp");
%smirf_cfg = Dada::readCFGFileIntoHash("/home/vivek/SMIRF/config/smirf.cfg", 0);
%ct = Mopsr::getCornerturnConfig("bp");
$proc_id = -1;
$db_key = "dada";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
# For transport via Infiniband
#$bw_limit = (128 * 1024) / $cfg{"NUM_BP"};
#$srv_ip = "192.168.5.10";
# For transport via 1GbE
$bw_limit = int((32 * 1024) / $cfg{"NUM_BP"});
$smirf_bw_limit = int((640 * 1024) / $cfg{"NUM_BP"});
$srv_ip = "172.17.228.204";
$smirf_ip = "192.168.5.120";

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$proc_id  = $ARGV[0];

# ensure that our proc_id is valid 
if (($proc_id >= 0) &&  ($proc_id < $cfg{"NUM_BP"}))
{
  # and matches configured hostname
  if ($cfg{"BP_".$proc_id} ne Dada::getHostMachineName())
  {
    print STDERR "BP_".$proc_id."[".$cfg{"BP_".$proc_id}."] did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "proc_id was not a valid integer between 0 and ".($cfg{"NUM_BP"}-1)."\n";
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

  # this is data stream we will be reading from
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $proc_id, $cfg{"NUM_BP"}, $cfg{"PROCESSING_DATA_BLOCK"});

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

  my $bp_tag = sprintf ("BP%02d", $proc_id);
  my $control_thread = threads->new(\&controlThread, $pid_file);

  my ($cmd, $result, $response, $utc_start, $source, $n);
  my @parts = ();

  my $proc_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$bp_tag;

  # look for filterbank files to transfer to the server via rsync
  while (!$quit_daemon)
  {
    $cmd = "find ".$proc_dir." -mindepth 3 -maxdepth 3 -type f -name 'obs.finished' | sort -n -r | tail -n 1";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      # get the observation UTC_START
      @parts = split (/\//, $response);
      $n = $#parts;

      $utc_start = $parts[$n-2];
      $source = $parts[$n-1];

      # also extract the PID if possible
      $cmd = "grep ^PID ".

      msg(2, "INFO", "main: utc_start=".$utc_start." source=".$source);
      
      if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
      {
        my $local_dir = $proc_dir."/".$utc_start."/".$source;
        msg (2, "INFO", "main: found finished observation: ".$local_dir);

        # also extract the PID if possible
        $cmd = "grep ^PID ".$local_dir."/BEAM_???/obs.header | awk '{print \$2}' | tail -n 1";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
        my $pid = $response;

        # get the list of beams in this observation
        $cmd = "find ".$proc_dir."/".$utc_start."/".$source." -maxdepth 1 -type d -name 'BEAM_???' -printf '%f\n' | sort -n";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
    
        if (($result ne "ok") || ($response eq ""))
        {
          msg(0, "WARN", "no beams found in ".$local_dir);

          $cmd = "mv ".$local_dir."/obs.finished ".$local_dir."/obs.failed";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
        }
        else
        {
          my $remote_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source;
          msg(2, "INFO", "main: createRemoteDir(".$remote_dir.")");
          ($result, $response) = createRemoteDir($remote_dir);
          msg(3, "INFO", "main: ".$result." ".$response);
          if ($result ne "ok")
          {
            msg(0, "WARN", "failed to create remote directory: ".$response);
            sleep (1);
          }
          else
          {
            # transfer the RFI lists from the FRB detector
            ($result, $response) = transferRFIList ($local_dir, $utc_start, $source);
            if ($result ne "ok")
            {
              msg(0, "WARN", "failed to transfer rfi list: ".$response);
              $cmd = "mv ".$local_dir."/obs.finished ".$local_dir."/obs.failed";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
              next;
            }
            elsif ($response ne "")
            {
              msg(1, "INFO", "transferRFIList: ".$response);
            }
            else
            {
              msg(2, "INFO", "transferRFIList: transferred successfully");
            }

            # if a SMIRF observation 
            if ($pid eq "P001")
            {
              # also extract the PID if possible
              $cmd = "grep ^SOURCE ".$local_dir."/BEAM_???/obs.header | awk '{print \$2}' | tail -n 1";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
              my $smirf_source = $response;

              ($result, $response) = transferSmirfBeams ($local_dir, $utc_start, $smirf_source);
              if ($result ne "ok")
              {
                msg(1, "WARN", "SMIRF transfer for ".$utc_start." failed: ".$response);
              }
            }

            # transfer beams 
            ($result, $response) = transferBeams ($local_dir, $utc_start, $source);
            if ($result ne "ok")
            {
              msg(0, "ERROR", "transfer of ".$utc_start." failed: ".$response);
              $cmd = "mv ".$local_dir."/obs.finished ".$local_dir."/obs.failed";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
            }
            elsif ($response ne "")
            {
              msg(1, "INFO", "transferBeams: ".$response." skipping");
            }
            else
            {
              $cmd = "mv ".$local_dir."/obs.finished ".$local_dir."/obs.transferred";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
              if ($result ne "ok")
              {
                msg(0, "ERROR", $cmd." failed: ".$response);
                $quit_daemon = 1;
              }

              my $user = "dada";
              my $host = "172.17.228.204";
              my $rval;
      
              $cmd = "touch ".$remote_dir."/obs.finished.".$proc_id.".".$cfg{"NUM_BP"};
              msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
              ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
              msg(2, "INFO", "main: ".$result." ".$rval." ".$response);

              $cmd = "rm -rf ".$local_dir."/BEAM_???";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
              if ($result ne "ok")
              {
                msg(0, "ERROR", $cmd." failed: ".$response);
                $quit_daemon = 1;
              }
            } # transfer beams ok
          } # create remote dir ok
        } # beams existed 
      } # utc_start match regex
    } # obs.finished file exists
        
    my $counter = 10;
    while (!$quit_daemon && $counter > 0)
    {
      sleep(1);
      $counter --;
    }
    msg(2, "INFO", "main: cleanTransferredDirs(".$proc_dir.")");
    ($result, $response) = cleanTransferredDirs($proc_dir);
    msg(2, "INFO", "main: cleanTransferredDirs: ".$result." ".$response);
    msg(2, "INFO", "main: quit_daemon=".$quit_daemon);
  }

  # Rejoin our daemon control thread
  msg(2, "INFO", "joining control thread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");

  # Close the nexus logging connection
  Dada::nexusLogClose($sys_log_sock);

  exit (0);
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

sub transferRFIList($$$)
{
  my ($local_dir, $utc_start, $source) = @_;

  my ($cmd, $result, $response);

  my $rfi_file = "candidates.list.".sprintf("BP%02d", $proc_id);
  my $local_file = "";

  if (-f $local_dir."/".$rfi_file)
  {
    $local_file = $local_dir."/".$rfi_file;
  }
  elsif (-f $local_dir."/../".$rfi_file)
  {
    $local_file = $local_dir."/../".$rfi_file;
  }
  else
  {
    return ("ok", "Candidates file did not exist for ".$utc_start."/".$source);
  }

  $cmd = "rsync -a --stats --bwlimit=".$bw_limit." --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
         $local_file." upload\@172.17.228.204::results/".$utc_start."/".$source."/";
  msg(2, "INFO", "transferRFIList: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "transferRFIList: ".$result." ".$response);
  if ($result ne "ok")
  {
    if ($quit_daemon)
    {
      msg(0, "INFO", "transfer of ".$utc_start." interrupted");
      return ("fail", "transfer interrupted");
    }
    else
    {
      msg(0, "WARN", "transfer of ".$local_file." failed: ".$response);
      return ("fail", "transfer of ".$local_file." failed");
    }
  }

  unlink $local_file;
  return ("ok", "");
}

sub transferBeams ($$$)
{
  my ($local_dir, $utc_start, $source) = @_;
  my ($cmd, $result, $response);

  $cmd = "rsync -a --stats --bwlimit=".$bw_limit." --no-g --chmod=go-ws --password-file=/home/mpsr/.ssh/rsync_passwd ".
         $local_dir."/BEAM_??? upload\@".$srv_ip."::archives/".$utc_start."/".$source."/";
  msg(2, "INFO", "transferBeams: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "transferBeams: ".$result." ".$response);
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
              
  msg(1, "INFO", $utc_start."/".$source." finished -> transferred [".$data_rate."]");

  return ("ok", "");
}

sub transferSmirfBeams ($$$)
{
  my ($local_dir, $utc_start, $source) = @_;
  my ($cmd, $result, $response, $rval);

  my $user = $smirf_cfg{"SURVEY_USER"};
  my $host = $smirf_cfg{"SURVEY_HOST"};

  $cmd = "mkdir -p -m 0755 ".$smirf_cfg{"SURVEY_DIR"}."/data/".$source."/".$utc_start."/FB";
  msg(2, "INFO", "transferSmirfBeams: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  msg(2, "INFO", "transferSmirfBeams: ".$result." ".$rval." ".$response);

  $cmd = "rsync -a --stats --bwlimit=".$smirf_bw_limit." --no-g --chmod=go-ws ".
         "--password-file=/home/mpsr/.ssh/rsync_passwd ".
         $local_dir."/BEAM_??? upload\@".$smirf_ip."::smirf/data/".$source."/".$utc_start."/FB/";
  msg(2, "INFO", "transferSmirfBeams: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "transferSmirfBeams: ".$result." ".$response);
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

  msg(1, "INFO", $utc_start."/".$source." finished -> transferred [".$data_rate."]");

  return ("ok", "");
}


sub cleanTransferredDirs ($)
{
  my ($proc_dir) = @_;
  my ($cmd, $result, $response, $obs);

  # look for observations marked obs.transferred that are > 1day old
  $cmd = "find ".$proc_dir." -mindepth 3 -maxdepth 3 -type f -name 'obs.transferred' -mtime +1";
  msg(2, "INFO", "cleanTransferredDirs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "main: ".$result." ".$response);

  if (($result eq "ok") && ($response ne ""))
  {
    my @list = split(/\n/, $response);
    $response = "";
    foreach $obs (@list)
    {
      my @parts = split (/\//, $obs);
      my $utc_start = $parts[$#parts-2];
      msg(1, "INFO", "cleaning ".$utc_start);

      $cmd = "rm -rf ".$proc_dir."/".$utc_start;
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$result." ".$response);
    }
    return ("ok", "deleted ". ($#list+1)." observations");
  }
  return ("ok", "no observations to delete");
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_fb_mngr", $msg);
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

  $cmd = "^rsync -a --stats";
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

