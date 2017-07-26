#!/usr/bin/env perl

###############################################################################
#
# client_mopsr_results_monitor.pl 
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
  print "Usage: ".basename($0)." PWC_ID\n";
}

#
# Global Variables
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our %cfg : shared;
our $pwc_id : shared;
our $log_host;
our $log_port;
our $log_sock;


#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
$pwc_id = -1;
$log_host = $cfg{"SERVER_HOST"};
$log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$log_sock = 0;


# Check command line argument
if ($#ARGV != 0)
{
  usage();
  exit(1);
}

$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (($pwc_id >= 0) &&  ($pwc_id < $cfg{"NUM_PWC"}))
{
  # and matches configured hostname
  if (!(($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName()) || ($cfg{"PWC_".$pwc_id} eq "localhost")))
  {
    print STDERR "PWC_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "PWC_ID was not a valid integer between 0 and ".($cfg{"NUM_PWC"}-1)."\n";
  usage();
  exit(1);
}


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);

###############################################################################
#
# Main
#
{
  # Register signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  my $pid_file =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";
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

  my $client_mon_dir = $cfg{"CLIENT_UDP_MONITOR_DIR"}."/". $cfg{"PWC_PFB_ID_".$pwc_id};
  my $server_mon_dir;
  if ($use_nfs)
  {
    $server_mon_dir = $cfg{"SERVER_UDP_MONITOR_NFS"};
  }
  else
  {
    $server_mon_dir = $cfg{"SERVER_UDP_MONITOR_DIR"};
  }
  
  my $server_host    = $cfg{"SERVER_HOST"};
  my $server_user    = $cfg{"USER"};

  my ($cmd, $result, $response, $key, $img, $file, $time, $pfb, $input, $type, $res, $ext, $file_list);
  my $sleep_total = 2;
  my $sleep_count;

  my @mon_images;
  my @dump_files;
  my @stats_files;
  my %to_send;
  my $dump_file;

  my $schan = 0;
  my $echan = ($cfg{"PWC_END_CHAN"} - $cfg{"PWC_START_CHAN"});

  if (! -d $client_mon_dir)
  {
    $cmd = "mkdir -p ".$client_mon_dir;
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      msg(0, "WARN", "main: ".$cmd." failed: ".$response);
    }
  }

  while (!$quit_daemon)
  {
    # look for any dump files
    $cmd = "find ".$client_mon_dir." -mindepth 1 -maxdepth 1 -type f -name '*.dump' | sort";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      msg(0, "WARN", "find list of dump files failed: ".$response);
      @dump_files = ();
    }
    else
    {
      @dump_files = split(/\n/, $response);
      msg(2, "INFO", "main: found ".($#dump_files+1)." monitoring images");
      if ($#dump_files >= 0)
      {
        foreach $dump_file (@dump_files)
        {
          $cmd = "mopsr_dumpplot -p -g 160x120 -c ".$schan." -d ".$echan." ".$dump_file;
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);

          # rename the dump file as .dumped
          $cmd = "mv ".$dump_file." ".$dump_file."ed";
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
        }
      }
    }

    # look for any monitoring output files
    $cmd = "find ".$client_mon_dir." -mindepth 1 -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sort";
    msg(2, "INFO", "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);

    if ($result ne "ok")
    {
      msg(0, "WARN", "find list of all obs failed: ".$response);
      @mon_images = ();
    }
    else
    {
      @mon_images = split(/\n/, $response);
      $file_list = "";

      msg(2, "INFO", "main: found ".($#mon_images+1)." monitoring images");
      if ($#mon_images >= 0)
      {
        # ensure directory is automounted
        if ($use_nfs)
        {
          $cmd = "ls -1d ".$server_mon_dir;
          msg(2, "INFO", "main: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          msg(3, "INFO", "main: ".$result." ".$response);
        }

        %to_send = ();
        # in case there are more than one of each class
        foreach $img (@mon_images)
        {
          ($time, $pfb, $input, $type, $res, $ext) = split(/\./, $img);
          $to_send{$pfb.".".$input.".".$type.".".$res} = $img;
          msg(2, "INFO", "main: to_send{".$pfb.".".$input.".".$type.".".$res."}=".$img);
        }

        $file_list = "";
        foreach $key (keys %to_send)
        {
          $file_list .= $client_mon_dir."/".$to_send{$key}." ";
        }

        # get any stats files
        @stats_files = ();
        $cmd = "find ".$client_mon_dir." -mindepth 1 -maxdepth 1 -type f -name '*.stats' -printf '%f\n' | sort";
        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        if (($result eq "ok") && ($response ne ""))
        {
          @stats_files = split (/\n/, $response);
          foreach $file (@stats_files)
          {
            $file_list .= $client_mon_dir."/".$file;
          }
        }
        msg(2, "INFO", "main: file_list=".$file_list);

        if ($use_nfs)
        {
          $cmd = "cp ".$file_list." ".$server_mon_dir."/";
        }
        else
        {
          $cmd = "rsync -a ".$file_list." ".$server_user."\@".$server_host.":".$server_mon_dir."/";
        }

        msg(2, "INFO", "main: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        msg(3, "INFO", "main: ".$result." ".$response);
          
        foreach $img (@mon_images)
        {
          unlink $client_mon_dir."/".$img;
        }
        foreach $file (@stats_files)
        {
          unlink $client_mon_dir."/".$file;
        }
      }

      # delete any dumped files that are more than 60s old
      $cmd = "find ".$client_mon_dir." -mindepth 1 -maxdepth 1 -type f -mmin +1 -name '*.dumped' -delete";
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$result." ".$response);
      if ($result ne "ok")
      {
        msg(0, "WARN", "main: ".$cmd." failed: ".$response);
      }
    }

    if ($#mon_images < 0)
    { 
      $sleep_count = 0;
      while (!$quit_daemon && ($sleep_count < $sleep_total))
      {
        sleep(1);
        $sleep_count++;
      }
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
      Dada::nexusLogMessage($log_sock, sprintf("%02d",$pwc_id), $time, "sys", $type, "results_mon", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}

sub controlThread($)
{
  (my $pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

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

