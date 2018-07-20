#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bs_mux_recv.pl 
#
# process a single channel of TS data
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
$bin_dir = "/home/vivek/software/linux_64/bin";


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

  my ($cmd, $result, $response, $raw_header, $cand_file, $proc_cmd, $final);
  my $prev_utc_start = "none";
  my $utc_start = "none";

  my $client_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag;

  while (!$quit_daemon)
  {
    $cmd = "dada_header -t bs_dspsr -k ".$db_key;
    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "main: ".$cmd." returned");

    if ($? != 0)
    {
      if ($quit_daemon)
      { 
        msg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      { 
        msg(0, "ERROR", "dada_header failed: ".$raw_header);
        $quit_daemon = 1;
      }
    }
    else
    {
      my %header = Dada::headerToHash($raw_header);
      msg (2, "INFO", "UTC_START=".$header{"UTC_START"}." NCHAN=".$header{"NCHAN"}." CAND_FILE=".$header{"CAND_FILE"});
      $utc_start = $header{"UTC_START"};
      $final = $header{"FINAL_STITCH"}; # "true" or "false"
  
      $cand_file = $header{"CAND_FILE"};
      my $proc_dir = $client_dir."/".$header{"UTC_START"}."/".$header{"FOLD_OUT"};

      my $tmp_info_file =  "/tmp/mopsr_".$db_key.".info";
      if (! -f $cand_file)
      {
        msg (0, "ERROR", "CAND_FILE [".$cand_file."] did not exist");
        $proc_cmd = "dada_dbnull -k ".$db_key." -z -s";
        $binary = "dada_dbnull -k ".$db_key;
      }
      else
      {
        $cmd = "grep -v SOURCE ".$cand_file." | awk '{print \$1}'";
        msg(1, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(1, "INFO", "main: ".$result." ".$response);
        if (($result eq "ok") && ($response ne ""))  
        {
          my @sources = split(/\n/, $response);
          if ($#sources == 0)
          {
            $proc_dir .= "/".$sources[0];    

            $cmd = "mkdir -p ".$proc_dir;
            msg(1, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(1, "INFO", "main: ".$result." ".$response);
            if ($result ne "ok")
            {
              msg(1, "ERROR", "failed to create ".$proc_dir.": ".$response);
            }
          }
        }

        my $cpu_core = $cfg{"BS_DSPSR_CORE_".$bs_id};
        #$proc_cmd = "numactl -C ".$cpu_core." ".$bin_dir."/dspsr -e car -L 60 -Lmin 10 -A -w ".$cand_file." ".$tmp_info_file;
        $proc_cmd = "export T2USER=".$cfg{"USER"}.$bs_tag."; ".$bin_dir."/dspsr -e car -L 10 -Lmin 9.5 -A -b 128 -w ".$cand_file." ".$tmp_info_file;
        $binary =  "dspsr -e car";
      }
  
      # create a local directory for the output from this channel
      ($result, $response) = createLocalDirs(\%header);
      if ($result ne "ok")
      {
        msg (0, "ERROR", "failed to create local dir: ".$response);
        $proc_cmd = "dada_dbnull -k ".$db_key." -z -s";
        $binary = "dada_dbnull -k ".$db_key;
      }

      # ensure a file exists with the write processing key
      if (! -f $tmp_info_file)
      {
        open FH, ">".$tmp_info_file;
        print FH "DADA INFO:\n";
        print FH "key ".$db_key."\n";
        close FH;
      }

      $cmd = "cd ".$proc_dir."; ".$proc_cmd;
      msg(1, "INFO", "START ".$proc_cmd);
      ($result, $response) = Dada::mySystemPiped($cmd, $src_log_file, $src_log_sock,
                                                 "src", sprintf("%02d",$bs_id), $daemon_name, "bs_dspsr");
      msg(1, "INFO", "END   ".$proc_cmd);
      $binary = "";
      if ($result ne "ok")
      {
        $quit_daemon = 1;
        if ($result ne "ok")
        {
          msg(0, "ERROR", $cmd." failed: ".$response);
        }
      }
      
      # If this is the final stitch of this observation
      if ($final eq "true")
      {
        $cmd = "touch ".$client_dir."/".$header{"UTC_START"}."/obs.folded";
        msg(1, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);

        # since we are likely to do the pdmping on the BF08 server, we are done
        $cmd = "mv ".$client_dir."/".$header{"UTC_START"}."/obs.processing ".$client_dir."/".$header{"UTC_START"}."/obs.finished";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);

        # but we need to tell the local instances of BP that the peasouping is finished
        $cmd = "ls -1 ".$cfg{"CLIENT_RECORDING_DIR"}."/BP??/".$header{"UTC_START"}."/FB/obs.completed";
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

        # if this instance is BF08, manually clean the hires filterbanks
        if (sprintf("%02d", $bs_id) == $smirf_cfg{"EDGE_BS"})
        {
          msg(1, "INFO", "Deleting HIRES filterbanks on BS Edge node");
          $cmd = "rm -rf ".$cfg{"CLIENT_RECORDING_DIR"}."/BP??/".$header{"UTC_START"};
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
        }
      }
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
# Create the local directories required for this observation
#
sub createLocalDirs(\%)
{
  my ($h_ref) = @_;

  msg(2, "INFO", "createLocalDirs()");

  my %h = %$h_ref;
  my $utc_start = $h{"UTC_START"};
  my $bs_tag    = sprintf("BS%02d", $bs_id);
  my $dir       = $cfg{"CLIENT_RESULTS_DIR"}."/".$bs_tag."/".$utc_start."/".$h{"FOLD_OUT"};

  my ($cmd, $result, $response);
  if (! -d $dir) 
  {
    msg(2, "INFO", "createLocalDirs: mkdirRecursive(".$dir.", 0755)");
    ($result, $response) = Dada::mkdirRecursive($dir, 0755);
    msg(3, "INFO", "createLocalDirs: ".$result." ".$response);
    if ($result ne "ok")
    {
      return ("fail", "Could not create local dir: ".$response);
    }
  }

  # create an obs.headers file in the processing dir:
  msg(2, "INFO", "createLocalDirs: creating obs.header");
  my $file = $dir."/obs.headers";
  if (-f $file)
  {
    open(FH,">>".$file);
    print FH "# ============================\n";
  }
  else
  {
    open(FH,">".$file);
  }
  my $k = "";
  foreach $k ( keys %h)
  {
    print FH Dada::headerFormat($k, $h{$k})."\n";
  }
  close FH;

  return ("ok", $dir);
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$bs_id), $time, "sys", $type, "bs_dspsr", $msg);
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

  $cmd = "^dada_header -t bs_dspsr -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^dada_dbnull -k ".$db_key;
  msg(2, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  if ($binary ne "")
  {
    $cmd = "^".$binary;
    msg(1, "INFO" ,"controlThread: killProcess(".$cmd.", mpsr)");
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

