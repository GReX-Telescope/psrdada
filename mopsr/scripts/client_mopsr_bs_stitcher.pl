#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# 
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
our %smirf_cfg : shared;
our $localhost : shared;
our $bs_id : shared;
our $db_key : shared;
our $binary : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;
our $bin_dir;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bs");
%smirf_cfg = Dada::readCFGFileIntoHash("/home/vivek/SMIRF/config/smirf.cfg", 0);
$bs_id = -1;
$db_key = "";
$binary = "";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BS_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BS_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
$bin_dir = $smirf_cfg{"SMIRF_BIN_DIR"};


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$bs_id  = $ARGV[0];

# ensure that our bs_id is valid 
if (($bs_id >= 0) &&  ($bs_id < $cfg{"NUM_BS"}))
{
  # and matches configured hostname
  if ($cfg{"BS_".$bs_id} ne Dada::getHostMachineName())
  {
    print STDERR "BS_".$bs_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "bs_id was not a valid integer between 0 and ".($cfg{"NUM_BS"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$bs_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bs_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$bs_id.".src.log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bs_id.".pid";

  # this is data stream we will be reading from

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

  my $control_thread = threads->new(\&controlThread, $pid_file);
  my $bs_tag = sprintf("BS%02d", $bs_id);

  # this is data stream we will be reading from
  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $bs_id, $cfg{"NUM_BS"}, $cfg{"PROCESSING_DATA_BLOCK"});

  my ($cmd, $result, $response, $raw_header, $cand_file, $proc_cmd);
  my (@parts, $n, $utc_start);

  my $client_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag;

  if (!(-d $client_dir))
  {
    msg(1, "INFO", "main: creating ".$client_dir);
    ($result, $response) = Dada::mkdirRecursive($client_dir, 0755);
    if ($result ne "ok")
    {
      msg(1, "ERROR", "failed to create ".$client_dir.": ".$response);
      $quit_daemon = 1;
    } 
  }

  while (!$quit_daemon)
  {
    $cmd = "find ".$client_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.processing' | sort -n";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      my @observations = split(/\n/, $response);
      my $observation;
      foreach $observation ( @observations )
      {
        # get the observation UTC_START
        @parts = split (/\//, $observation);
        $n = $#parts;

        $utc_start = $parts[$n-1];

        msg(2, "INFO", "main: utc_start=".$utc_start);

        if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
        {
          my $proc_dir = $client_dir."/".$utc_start;

          if (-f $proc_dir."/obs.stitched")
          {
            msg(2, "INFO", "obs.processing and obs.stitched existed for ".$utc_start.", ignoring");
          }
          elsif (-f $proc_dir."/".$utc_start.".shortlisted.".$bs_tag)
          {
            $bin_dir = $smirf_cfg{"SMIRF_BIN_DIR"};
            $binary = "SMIRFsoup -k ".$db_key." -T -i ".$utc_start." -b ".$bs_id ;

            my $cpu_core = $cfg{"BS_STITCHER_CORE_".$bs_id};
            $cmd = "numactl -C ".$cpu_core." ".$bin_dir."/".$binary;

            msg(1, "INFO", "START ".$cmd);
            ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock, "src", sprintf("%02d",$bs_id), $daemon_name, "bs_stitcher");
            msg(1, "INFO", "END   ".$cmd);

            $binary = "";
            if ($result ne "ok")
            {
              if (!$quit_daemon)
              {
                msg(0, "ERROR", $cmd." failed: ".$response);
                $cmd = "mv ".$proc_dir."/obs.processing ".$proc_dir."/obs.failed";
                msg(2, "INFO", "main: ".$cmd);
                ($result, $response) = Dada::mySystem($cmd);
                msg(3, "INFO", "main: ".$result." ".$response);
              }
              else
              {
                msg(0, "INFO", $cmd." interrupted due to quit request");
              }
            }
            else
            {
              $cmd = "touch ".$proc_dir."/obs.stitched";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
        
              $cmd = "find ".$proc_dir."/ -name 'candidates_*.".$bs_tag."' | wc -l";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
              if (($result eq "ok") && ($response eq "0"))
              {
                msg(1, "INFO", "No stiching was performed, touching obs.peasouped");

                # since we have no candidates, client_mopsr_bs_dspsr will not mark this
                # observation as processing -> finished, so lets do it here
                my $dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag."/".$utc_start;
                $cmd = "mv ".$dir."/obs.processing ".$dir."/obs.finished";
                msg(2, "INFO", "main: ".$cmd);
                ($result, $response) = Dada::mySystem($cmd);
                msg(3, "INFO", "main: ".$result." ".$response);
                if ($result ne "ok")
                {
                  msg(1, "ERROR", "Failed to move obs.processing to obs.finished");
                }

                # but we need to tell the local instances of BP that the peasouping are finished
                $cmd = "ls -1 ".$cfg{"CLIENT_RECORDING_DIR"}."/BP??/".$utc_start."/FB/obs.completed";
                msg(2, "INFO", "main: ".$cmd);
                ($result, $response) = Dada::mySystem($cmd);
                msg(3, "INFO", "main: ".$result." ".$response);

                if (($result eq "ok") && ($response ne ""))
                {
                  my @lines = split(/\n/, $response);
                  my ($line, $peasouped_file);
                  foreach $line ( @lines )
                  {
                    $peasouped_file = $line;
                    $peasouped_file =~ s/completed/peasouped/;
                    msg(1, "INFO", "creating ".$peasouped_file);
                    $cmd = "touch ".$peasouped_file;
                    msg(2, "INFO", "main: ".$cmd);
                    ($result, $response) = Dada::mySystem($cmd);
                    msg(3, "INFO", "main: ".$result." ".$response);
                  }
                }
              } # if not stitching was performed
              else
              {
                msg (2, "INFO", "main: stitching resulted in ".$response." candidates");
              }
            } # stitcher succeeded
          }
          else
          {
            msg(2, "INFO", $proc_dir."/".$utc_start.".shortlisted.".$bs_tag." did not yet exist");
          }
        }
        else
        {
          msg(0, "WARN", "UTC_START [".$utc_start."] did not match expected pattern");
        }
      } # foreach observation
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$bs_id), $time, "sys", $type, "bs_sticher", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$bs_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  if ($binary ne "")
  {
    $cmd = "^".$binary;
    msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
    ($result, $response) = Dada::killProcess($cmd, "mpsr");
    msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);
  }

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

