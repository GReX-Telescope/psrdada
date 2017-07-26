#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bfdsp.pl 
#
# Convert baseband antenna data in SFT format to detected beam data in SFT
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
  print "Usage: ".basename($0)." CHAN_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our %ct : shared;
our $localhost : shared;
our $chan_id : shared;
our $in_db_key : shared;
our $fb_db_key : shared;
our @tb_db_keys : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bf");
%ct = Mopsr::getCornerturnConfig("bp");   # read the BP cornerturn
$chan_id = -1;
$in_db_key = "";
@tb_db_keys = ();
$fb_db_key = "";
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BF_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BF_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$chan_id  = $ARGV[0];

# ensure that our chan_id is valid 
if (($chan_id >= 0) &&  ($chan_id < $cfg{"NCHAN"}))
{
  # and matches configured hostname
  if ($cfg{"RECV_".$chan_id} ne Dada::getHostMachineName())
  {
    print STDERR "RECV_".$chan_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "chan_id was not a valid integer between 0 and ".($cfg{"NCHAN"}-1)."\n";
  usage();
  exit(1);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$chan_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$chan_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$chan_id.".src.log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$chan_id.".pid";

  # this is data stream we will be reading from
  $in_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $chan_id, $cfg{"NUM_BF"}, $cfg{"RECEIVING_DATA_BLOCK"});
  $fb_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $chan_id, $cfg{"NUM_BF"}, $cfg{"FAN_BEAMS_DATA_BLOCK"});
  my $i;
  for ($i=0; $i<$cfg{"NUM_TIED_BEAMS"}; $i++)
  {
    push (@tb_db_keys, Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $chan_id, $cfg{"NUM_BF"}, $cfg{"TIED_BEAM_".$i."_DATA_BLOCK"}));
  }

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

  my ($cmd, $result, $response, $raw_header, $full_cmd, $proc_cmd_file);
  my ($proc_cmd, $proc_dir, $bf_dir, $tracking, $prev_utc_start);

  $bf_dir  = "BF".sprintf("%02d", $chan_id);

  $prev_utc_start = "";

  # continuously run mopsr_dbib for this PWC
  while (!$quit_daemon)
  {
    $cmd = "dada_header -t bfdsp -k ".$in_db_key;
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
      msg(0, "INFO", "UTC_START=".$header{"UTC_START"}." NCHAN=".$header{"NCHAN"}." NANT=".$header{"NANT"});

      my $valid_obs= 0;
      if (($header{"FB_ENABLED"} eq "true") || ($header{"MB_ENABLED"} eq "true"))
      {
        $valid_obs = 1;
      }
      for ($i=0; $i<$cfg{"NUM_TIED_BEAMS"}; $i++)
      {
        if ($header{"TB".$i."_ENABLED"} eq "true")
        {
          $valid_obs = 1;
        }
      }

      $proc_cmd = "dada_dbnull -k ".$in_db_key." -z -s";

      if ($valid_obs)
      {
        # create the local directories to save some information 
        # about this step in the pipeline for this observation
        msg(1, "INFO", "main: createLocalDirs()");
        ($result, $response) = createLocalDirs (\%header);
        msg(1, "INFO", "main: ".$result." ".$response);

        if ($result ne "ok")
        {
          msg(0, "ERROR", "failed to create local directories");
        }
        elsif ($prev_utc_start eq $header{"UTC_START"})
        {
          msg(0, "ERROR", "UTC_START repeated, jettesioning observation");
        }
        else
        {
          if ($cfg{"BF_STATE_".$chan_id} eq "active")
          {
            $proc_cmd = "mopsr_bfdsp ".$in_db_key." ". $cfg{"MOLONGLO_BAYS_FILE"}." ".
                        $cfg{"MOLONGLO_MODULES_FILE"}." -d " .$cfg{"BF_GPU_ID_".$chan_id}." -s ";

            if ($header{"FB_ENABLED"} eq "true")
            {
              $proc_cmd .= " -f ".$fb_db_key." -b ".$ct{"NBEAM"};
            }

            for ($i=0; $i<$cfg{"NUM_TIED_BEAMS"}; $i++)
            {
              if ($header{"TB".$i."_ENABLED"} eq "true")
              {
                $proc_cmd .= " -t ".$tb_db_keys[$i];
              }
            }

            if ($header{"MB_ENABLED"} eq "true")
            {
              $proc_cmd .= " -m ".$fb_db_key." -b ".$header{"NANT"};
            }

            $proc_cmd .= " -c ".$cfg{"BF_CORE_".$chan_id};
          }
          else
          {
            msg(0, "INFO", "BF_STATE_".$chan_id." == ".$cfg{"BF_STATE_".$chan_id});
          }
        }
      }

      $prev_utc_start = $header{"UTC_START"};

      my ($binary, $junk) = split(/ /,$proc_cmd, 2);
      $cmd = "ls -l ".$cfg{"SCRIPTS_DIR"}."/".$binary;
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "main: ".$cmd.": ".$result." ".$response);

      $cmd = $proc_cmd;
      if ($proc_dir ne "")
      {
        $cmd = "cd ".$proc_dir."; ".$proc_cmd;
      }

      msg(1, "INFO", "START ".$cmd);
      ($result, $response) = Dada::mySystemPiped ($cmd, $src_log_file, $src_log_sock, 
                                                 "src", sprintf("%02d",$chan_id), $daemon_name, "proc");
      msg(1, "INFO", "END   ".$cmd);
      if ($result ne "ok")
      {
        $quit_daemon = 1;
        if ($result ne "ok")
        {
          msg(0, "ERROR", $cmd." failed: ".$response);
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
  my $bf_dir  = "BF".sprintf("%02d", $chan_id);
  my $dir   = $cfg{"CLIENT_RESULTS_DIR"}."/".$bf_dir."/".$utc_start;

  my ($cmd, $result, $response);

  msg(2, "INFO", "createLocalDirs: mkdirRecursive(".$dir.", 0755)");
  ($result, $response) = Dada::mkdirRecursive($dir, 0755);
  msg(3, "INFO", "createLocalDirs: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "ERROR", "Could not create local dir: ".$response);
    return ("fail", "Could not create local dir: ".$response);
  }

  # create an obs.header file in the processing dir:
  msg(1, "INFO", "createLocalDirs: creating obs.header.bfdsp");
  my $file = $dir."/obs.header.bfdsp";
  open(FH,">".$file.".tmp");
  my $k = "";
  foreach $k ( keys %h)
  {
    print FH Dada::headerFormat($k, $h{$k})."\n";
  }
  close FH;
  rename($file.".tmp", $file);

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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$chan_id), $time, "sys", $type, "bfdsp", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$chan_id.".quit";

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file)))
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my ($cmd, $result, $response);

  $cmd = "^dada_header -t bfdsp -k ".$in_db_key;
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO", "controlThread: killProcess() ".$result." ".$response);

  $cmd = "^dada_dbnull -k ".$in_db_key;
  msg(2, "INFO", "controlThread: killProcess(".$cmd.", mpsr)");
  ($result, $response) = Dada::killProcess($cmd, "mpsr");
  msg(3, "INFO" ,"controlThread: killProcess() ".$result." ".$response);

  if (-f $pid_file) {
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

