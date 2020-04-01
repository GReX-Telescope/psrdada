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
our %ct : shared;
our $localhost : shared;
our $proc_id : shared;
our $bp_tag : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;
our $hires;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig("bp");
%ct = Mopsr::getCornerturnConfig("bp");
$proc_id = -1;
$localhost = Dada::getHostMachineName(); 
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";
if (($cfg{"CONFIG_NAME"} =~ m/320chan/) || ($cfg{"CONFIG_NAME"} =~ m/312chan/))
{
  $hires = 1;
}
else
{
  $hires = 0;
}

# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$proc_id  = $ARGV[0];
$bp_tag = sprintf("BP%02d", $proc_id);

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

  my ($cmd, $result, $response, $utc_start, $source, $n, $beam, $hires_nchan);
  my @parts = ();

  my $proc_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$bp_tag;

  if (! -d $proc_dir)
  {
    Dada::mkdirRecursive($proc_dir, 0755);
  }

  # change to the parent dir 
  chdir $cfg{"CLIENT_RECORDING_DIR"};

  # look for filterbank files to transfer to the server via rsync
  while (!$quit_daemon)
  {
    $cmd = "find ".$proc_dir." -mindepth 3 -maxdepth 3 -type f -mmin +2 -name 'obs.completed' -printf '%h\n' | awk -F/ '{print \$(NF-1)\"/\"\$(NF)}' | sort -n";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (($result eq "ok") && ($response ne ""))
    {
      my @observations = split(/\n/, $response);
      my $obs;
      foreach $obs ( @observations )
      {
        # get the observation UTC_START
        @parts = split (/\//, $obs);
        $utc_start = $parts[0];
        $source = $parts[1];

        msg(2, "INFO", "main: utc_start=".$utc_start." source=".$source);
      
        if ($utc_start =~ m/\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d/)
        {
          my $local_dir = $proc_dir."/".$utc_start."/".$source;
          msg (2, "INFO", "main: found completed observation: ".$local_dir);

          # also extract the PID if possible
          $cmd = "grep ^PID ".$local_dir."/BEAM_???/obs.header | awk '{print \$2}' | tail -n 1";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(2, "INFO", "main: ".$result." ".$response);

          # if this observation belons to the Galactic Plane Survey, wait for obs.peasouped
          my $ready_to_decimate = 1;
          if (($result eq "ok") && ($response eq "P001"))
          {
            if (!(-f $local_dir."/obs.peasouped"))
            {
              msg(2, "INFO", "main: waiting for ".$bp_tag."/".$utc_start."/".$source."/obs.peasouped");
              $ready_to_decimate = 0;
            }
          }

          if ($ready_to_decimate)
          {
            # get the list of beams in this observation
            $cmd = "find ".$bp_tag."/".$utc_start."/".$source." -maxdepth 1 -type d -name 'BEAM_???' -printf '%f\n' | sort -n";
            msg(2, "INFO", "main: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            msg(3, "INFO", "main: ".$result." ".$response);
            if (($result ne "ok") || ($response eq ""))
            {
              msg(0, "WARN", "no beams found in ".$local_dir);

              $cmd = "mv ".$local_dir."/obs.completed ".$local_dir."/obs.failed";
              msg(2, "INFO", "main: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              msg(3, "INFO", "main: ".$result." ".$response);
            }
            else
            {
              my $all_converted = 1;
              my $any_failed = 0;
              if ($hires)
              {
                my @beams = split (/\n/, $response);
                my $ibeam;
                for ($ibeam=0; ((!$quit_daemon) && ($ibeam<=$#beams)); $ibeam++)
                {
                  $beam = $beams[$ibeam];

                  msg(2, "INFO", "main: convertHiresFilterbank (".$utc_start.", ".$source.", ".$beam.")");
                  ($result, $response) = convertHiresFilterbank ($utc_start, $source, $beam);
                  msg(2, "INFO", "main: convertHiresFilterbank: ".$result." ".$response);

                  if ($result ne "ok") 
                  {
                    $all_converted = 0;
                    
                    if ($quit_daemon)
                    {
                      msg(1, "INFO", "main: conversion of ".$bp_tag."/".$utc_start."/".$source."/".$beam." interrupted");
                    }
                    else
                    { 
                      $any_failed = 1;
                      msg(0, "WARN", "failed to convert ".$bp_tag."/".$utc_start."/".$source."/".$beam.": ".$response);
                    }
                  }
                }

                if ($quit_daemon)
                {
                  $all_converted = 0;
                  msg(1, "INFO", "main: conversion of ".$bp_tag."/".$utc_start."/".$source." interrupted");
                }
              }

              if ($any_failed)
              {
                msg(1, "INFO", $bp_tag."/".$utc_start."/".$source." completed -> failed");
                $cmd = "mv ".$local_dir."/obs.completed ".$local_dir."/obs.failed";
                msg(2, "INFO", "main: ".$cmd);
                ($result, $response) = Dada::mySystem($cmd);
                msg(3, "INFO", "main: ".$result." ".$response);
              }
              else
              {
                if ($all_converted)
                {
                  msg(1, "INFO", $bp_tag."/".$utc_start."/".$source." completed -> finished");

                  $cmd = "mv ".$local_dir."/obs.completed ".$local_dir."/obs.finished";
                  msg(2, "INFO", "main: ".$cmd);
                  ($result, $response) = Dada::mySystem($cmd);
                  msg(3, "INFO", "main: ".$result." ".$response);
                  if ($result ne "ok")
                  {
                    msg(0, "ERROR", $cmd." failed: ".$response);
                    $quit_daemon = 1;
                  }        
                } # all beams converted
                else
                {
                  msg(1, "INFO", $bp_tag."/".$utc_start."/".$source." interuppted")
                } # did not convert some beams due to interrupt
              } 
            } # beams existed 
          } # read to decimate
        } # utc_start match regex
      } # foreach observation
    } # find obs.completed worked
        
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

sub convertHiresFilterbank ($$$)
{
  my ($utc_start, $source, $beam) = @_;

  my ($cmd, $result, $response);
  my ($prefix, $nchan, $tsamp);

  my $hires_fb_file = $bp_tag."/".$utc_start."/".$source."/".$beam."/".$utc_start.".fil";
  my $lowres_fb_file = $bp_tag."/".$utc_start."/".$source."/".$beam."/".$utc_start.".fil.lowres";

  if (-f $lowres_fb_file) 
  {
    unlink ($lowres_fb_file);
  }

  if ( -f $hires_fb_file )
  {
    my $fb_nchan_hires = $ct{"NCHAN_COARSE"};
    my $fb_nchan_lowres = $fb_nchan_hires / 8;

    # check that is is actually a hires file!
    $cmd = "header ".$hires_fb_file." | grep 'Number of channels'";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(0, "WARN", $response);
      return ("fail", "could not determined number of channels in hires file");
    }
    ($prefix, $nchan) = split (/: /, $response);
    if ($nchan eq $fb_nchan_lowres)
    {
      return ("ok", "filterbank file already converted to lowres");
    }
    if ($nchan ne $fb_nchan_hires)
    {
      return ("fail", "hires file had ".$nchan." channels, expected ".$fb_nchan_hires);
    }

    $cmd = "header ".$hires_fb_file." | grep 'Sample time'";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);
    if ($result ne "ok")
    { 
      msg(0, "WARN", $response);
      return ("fail", "could not determined number of channels in hires file");
    }
    ($prefix, $tsamp) = split (/: /, $response);
    if (($tsamp - 327.68) > 1)
    {
      return ("fail", "hires file had ".$tsamp." us sampling time, expected 327.68 ");
    }

    $cmd = "digifil ".$hires_fb_file." -Q -B 10 -c -f 8 -t 2 -b 8 -o ".$lowres_fb_file;
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "main: ".$result." ".$response);

    if (-f "digifil.scaling") 
    {
      unlink ("digifil.scaling");
    }
    if ($result ne "ok")
    {
      if ($quit_daemon)
      {
        msg(1, "INFO", "convertHiresFilterbank: interrupted digifil conversion");
        return ("fail", "interrupted digifil conversion");
      }
      else
      {
        msg(1, "INFO", "convertHiresFilterbank: failed digifil conversion: ".$response);
        return ("fail", "digifil failed: ".$response);
      }
    }

    unlink ($hires_fb_file);
    rename ($lowres_fb_file, $hires_fb_file);

    return ("ok", "");
  }

  return ("fail", "hires file ".$hires_fb_file." did not exist");
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_fb_deci", $msg);
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

  $cmd = "^digifil ".$bp_tag;
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

