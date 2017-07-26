#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bs_distrib_beams.pl 
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
%bp_cfg = Mopsr::getConfig("bp");
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
$bw_limit = int((640 * 1024) / $cfg{"NUM_BS"});
$srv_ip = "192.168.5.120";

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

# find the BP sub directories to monitor
my $localhost = $cfg{"BS_".$proc_id};
my @bp_dirs = ();
my $i;
for ($i=0; $i<$bp_cfg{"NUM_BP"}; $i++)
{
  if ($bp_cfg{"BP_".$i} eq $localhost)
  {
    push @bp_dirs, sprintf("BP%02d", $i);
  }
}
my $bp_dir_list = join(" ", @bp_dirs);

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

  my ($cmd, $result, $response, $utc_start, $source, $n, $line);
  my ($bp, $transfer_list_file, $bp_dir);
  my @parts = ();
  my @lines = ();

  chdir $bp_cfg{"CLIENT_RECORDING_DIR"};

  msg(2, "INFO", "main: bp_dir_list=[".$bp_dir_list."]");

  # look for rawdata directories with obs.completed in them
  while (!$quit_daemon)
  {
    if ($bp_dir_list ne "")
    {
      $cmd = "find ".$bp_dir_list." -mindepth 3 -maxdepth 3 -type f -name 'obs.completed' | sort -n";
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$result." ".$response);

      if (($result eq "ok") && ($response ne ""))
      {
        @lines = split (/\n/, $response);
        my %utcs = ();

        foreach $line ( @lines )
        {
          # get the observation UTC_START
          @parts = split (/\//, $line);
          $n = $#parts;
          $utc_start = $parts[$n-2];
          $bp = $parts[$n-3];
          if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
          {
            # check if this observation has already been distributed
            if (!(exists $utcs{$utc_start}))
            {
              $utcs{$utc_start} = $bp;
            }
            else
            {
              $utcs{$utc_start} .= " ".$bp;
            }
          }
        }

        foreach $utc_start (sort keys %utcs)
        {
          msg(2, "INFO", "main: utc_start=".$utc_start." bps=".$utcs{$utc_start});
        
          # check for the SMIRF transfer list
          my $smirf_base_dir = $smirf_cfg{"SMIRF_BASE"}."/".$bs_tag."/".$utc_start;
          $transfer_list_file = $smirf_base_dir."/".$utc_start.".".$bs_tag.".rsync";

          if (-f $transfer_list_file )
          {
            msg (2, "INFO", "main: ".$transfer_list_file." existed");

            # extract the beams that must be transferred from the BP node...
            open FH, "<".$transfer_list_file;
            my @beams = <FH>;
            close FH;
            my $beam_list = "";
            my $beam;
            foreach $beam ( @beams)
            {
              chomp $beam;
              $beam_list .= " ".$beam;
            }

            msg (2, "INFO", "main: utc_start=".$utc_start." beam_list=[".$beam_list."]");

            # create remote directories of the form: /data/mopsr/rawdata/<bp_id>/<utc_start>/FB
            my $remote_dirs = "";
            foreach $bp_dir ( @bp_dirs )
            {
              $remote_dirs .= " ".$bp_cfg{"CLIENT_RECORDING_DIR"}."/".$bp_dir."/".$utc_start."/FB";
            }
            msg(2, "INFO", "main: createRemoteDirs(".$remote_dirs.")");
            ($result, $response) = createRemoteDirs($remote_dirs);
            msg(3, "INFO", "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              msg(0, "WARN", "failed to create remote directory: ".$response);
              sleep (1);
            }
            else
            {
              if ($beam_list ne "")
              {
                # transfer beams 
                ($result, $response) = transferBS ($utc_start, $beam_list);
                if ($result ne "ok")
                {
                  msg(0, "ERROR", "transfer of beams for ".$utc_start." failed: ".$response);
                  $cmd = "mv ".$transfer_list_file." ".$smirf_base_dir."/".$utc_start.".".$bs_tag.".rsync_failed";
                  msg(2, "INFO", "main: ".$cmd);
                  ($result, $response) = Dada::mySystem($cmd);
                  msg(3, "INFO", "main: ".$result." ".$response);
                }
                else
                {
                  if ($response eq "")
                  {
                    # touch obs.completed in the remote FB dir
                    $cmd = "touch";
                    foreach $bp_dir ( @bp_dirs )
                    {
                      $cmd .= " ".$bp_cfg{"CLIENT_RECORDING_DIR"}."/".$bp_dir."/".$utc_start."/FB/obs.completed";
                    }
                    my $user = "mpsr";
                    my $host = $smirf_cfg{"SURVEY_HOST"};
                    my $rval = 0;
                    msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
                    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
                    msg(2, "INFO", "main: ".$result." ".$rval." ".$response);
                    if ($result ne "ok")
                    {
                      msg(0, "ERROR", $cmd." failed: ".$response);
                      $quit_daemon = 1;
                    }

                    # mark this observation as completed by changing the name of the rsync files
                    $cmd = "mv ".$transfer_list_file." ".$smirf_base_dir."/".$utc_start.".".$bs_tag.".rsynced";
                    msg(2, "INFO", "main: ".$cmd);
                    ($result, $response) = Dada::mySystem($cmd);
                    msg(3, "INFO", "main: ".$result." ".$response);
                    if ($result ne "ok")
                    {
                      msg(0, "ERROR", $cmd." failed: ".$response);
                      $quit_daemon = 1;
                    }
                  } # rsync ok and not interrupted
                } # rsync ok 
              } # if beams to transfer
              else
              {
                # touch obs.completed in the remote FB dir
                $cmd = "touch";
                foreach $bp_dir ( @bp_dirs )
                {
                  $cmd .= " ".$bp_cfg{"CLIENT_RECORDING_DIR"}."/".$bp_dir."/".$utc_start."/FB/obs.completed";
                }
                my $user = "mpsr";
                my $host = $smirf_cfg{"SURVEY_HOST"};
                my $rval = 0;
                msg(2, "INFO", "main: ".$user."@".$host.":".$cmd);
                ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
                msg(2, "INFO", "main: ".$result." ".$rval." ".$response);
                if ($result ne "ok")
                {
                  msg(0, "ERROR", $cmd." failed: ".$response);
                  $quit_daemon = 1;
                }

                $cmd = "mv ".$transfer_list_file." ".$smirf_base_dir."/".$utc_start.".".$bs_tag.".rsync.nobeams";
                msg(2, "INFO", "main: ".$cmd);
                ($result, $response) = Dada::mySystem($cmd);
                msg(3, "INFO", "main: ".$result." ".$response);
                if ($result ne "ok")
                { 
                  msg(0, "ERROR", $cmd." failed: ".$response);
                  $quit_daemon = 1;
                }
              }
            } # create remote dir ok
          } # if rsync file exists
          else
          {
            msg(2, "INFO", "main: ".$transfer_list_file." did not exist");
          }
        } # foreach utc_start
      } # any obs.completed files exist
    }
          
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

sub createRemoteDirs($)
{
  my ($remote_dirs) = @_;

  my $user = "mpsr";
  my $host = $smirf_cfg{"SURVEY_HOST"};
  my $cmd = "mkdir -m 2755 -p ".$remote_dirs;
  my ($result, $rval, $response);

  msg(2, "INFO", "createRemoteDirs: ".$user."@".$host.":".$cmd);
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
  msg(2, "INFO", "createRemoteDirs: ".$result." ".$rval." ".$response);

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

sub transferBS ($$)
{
  my ($utc_start, $local_dirs) = @_;

  my ($cmd, $result, $response);

  # local dir is bp_cfg{CLIENT_RECORDING_DIR}
  # local_dirs are list of BP??/<UTC_START>/BEAM_???
  $cmd = "rsync -aR ".$local_dirs." upload\@".$srv_ip."::rawdata/ ".
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
              
  msg(1, "INFO", $utc_start." completed -> distributed [".$data_rate."]");

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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bs_distrib", $msg);
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

  $cmd = "^rsync -aR ".$bp_cfg{"CLIENT_RECORDING_DIR"};
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

