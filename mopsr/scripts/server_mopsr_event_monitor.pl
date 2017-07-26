#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  16 Apr, 2014
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use strict;         # strict mode (like -Wall)
use IO::Socket;     # Standard perl socket library
use IO::Select;     
use Net::hostent;
use File::Basename;
use threads;        # Perl threads module
use threads::shared; 
use Dada;           # DADA Module for configuration options
use Mopsr;           # Mopsr Module for configuration options


#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#

#
# Global variable declarations
#
our $dl;
our $daemon_name;
our %cfg : shared;
our $quit_threads : shared;
our $error;
our $warn;
our $pwcc_thread;

#
# global variable initialization
#
$dl = 2;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
$quit_threads = 0;
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";


#
# Main
#
{
  my $log_file  = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file  = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  my $listen_host = "129.78.128.156";
  my $listen_port = "23456";

  my $tmc_host = $cfg{"TMC_INTERFACE_HOST"};
  my $tmc_port = $cfg{"TMC_STATE_INFO_PORT"};

  my ($cmd, $result, $response, $n_events, $prks_utc_start, $utc_start, $i);
  my ($handle, $host, $peeraddr, $hostinfo, $command);
  my ($tmc_sock, $delay_time, $mpsr_event_utc, $event_utc);
  my ($smearing_time, $filter_time, $total_time);
  my ($start_time, $end_time, $start_utc, $end_utc);
  my ($port, $sock);

  my ($rh, $line);
  my @bits = ();
  my @events = ();

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT}  = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  if (-f $warn) {
    unlink $warn;
  }
  if (-f $error) {
    unlink $error;
  }

  # Redirect standard output and error
  # Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  Dada::logMsg(1, $dl, "Opening socket for control commands on ".$listen_host.":".$listen_port);

  my $listen_socket = new IO::Socket::INET (
    LocalHost => $listen_host,
    LocalPort => $listen_port,
    Proto => 'tcp',
    Listen => 1,
    ReuseAddr => 1
  );

  die "Could not create socket: $!\n" unless $listen_socket;

  my $localhost = Dada::getHostMachineName();

  # start the daemon control thread
  my $control_thread = threads->new(\&controlThread, $pid_file, $quit_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($listen_socket);   # add the main socket to the set

  # Main Loop,  We loop forever unless asked to quit
  while (!$quit_threads)
  {
    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
    Dada::logMsg(3, $dl, "select on read_set returned");
                                                                                  
    foreach $rh (@$readable_handles)
    {
      if ($rh == $listen_socket)
      {
        $host = "localhost";
        # Only allow 1 connection from Parkes Event Script
        if ($handle)
        {
          my $handle2 = $listen_socket->accept() or die "accept $!\n";

          $peeraddr = $handle2->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsgWarn($warn, "Rejecting additional connection from ".$host);
          $handle2->close();
        }
        else
        {
          # Wait for a connection from the server on the specified port
          $handle = $listen_socket->accept() or die "accept $!\n";

          # Ensure commands are immediately sent/received
          $handle->autoflush(1);

          # Add this read handle to the set
          $read_set->add($handle);

          # Get information about the connecting machine
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsg(1, $dl, "Accepting connection from ".$host);
        }
      }
      else
      {

        $line = <$rh>;
        $host = "localhost";

        # If we have lost the connection...
        if (! defined $line)
        {
          if (defined $hostinfo)
          {
            $host = $hostinfo->name;
          }
          Dada::logMsg(1, $dl, "lost connection from ".$host);

          $read_set->remove($rh);
          close($rh);
          $handle = 0;

        # Else we have received a legit line 
        }
        else
        {
          $/ = "\n";
          chomp $line;
          $/ = "\r";
          chomp $line;
          $/ = "\n";

          @bits = split(/ /, $line);
          if ($bits[0] ne "N_EVENTS")
          {
            Dada::logMsgWarn($warn, "Unexpected socket command: [".$line."] ignoring rest of message");
            print $rh "fail\r\n";
          }
          else
          {
            $n_events = $bits[1];
            Dada::logMsg(1, $dl, "N_EVENTS=".$n_events);

            $prks_utc_start = Dada::getLine($rh);
            Dada::logMsg(1, $dl, "PRKS_UTC_START=".$prks_utc_start);

            @events = ();
            for ($i=0; $i<$n_events; $i++)
            {
              $line = Dada::getLine($rh);
              Dada::logMsg(1, $dl, "EVENT_".$i."=".$line);
              push (@events, $line);
            }

            Dada::logMsg(1, $dl, "Events message from PRKS received");

            # now get the local UTC_START from the TMC interface
            $tmc_sock = Dada::connectToMachine($tmc_host, $tmc_port);
            if ($tmc_sock)
            {
              print $tmc_sock "utc_start\r\n";
              $response = Dada::getLine($tmc_sock);
              close ($tmc_sock);

              if (($response ne "") && ($response ne "UNKNOWN"))
              {
                $utc_start = $response;
                Dada::logMsg(1, $dl, "MPSR_UTC_START=".$utc_start);
          
                # build the local DBEVENT message
                my $event_message  = "N_EVENTS ".($n_events)."\n".
                                     $utc_start."\n";
                for ($i=0; $i<$n_events; $i++)
                {
                  my ($prks_event_utc, $snr, $dm, $filter) = split(/ /, $events[$i]);
                  Dada::logMsg(1, $dl, "EVENT_".$i." PRKS_UTC=".$prks_event_utc." SNR=".$snr." DM=".$dm." FILTER=".$filter);

                  # parkes band is 1182->1582 MHz
                  # molonglo band is 810-850 MHz
                  # bw = 732 MHz
                  # cfreq = 850 + 732/2 = 1216 MHz
                  $cmd = "dmsmear -f 1216 -b 732 -n 1024 -d ".$dm." -q";
                  Dada::logMsg(2, $dl, "main: cmd=".$cmd);
                  ($result, $response) = Dada::mySystem($cmd);
                  Dada::logMsg(3, $dl, "main: ".$result." ".$response);

                  $response =~ s/^ +//;
                  $delay_time = $response;

                  Dada::logMsg(1, $dl, "EVENT_".$i.": delay_time=".$delay_time);
                  $mpsr_event_utc = Dada::addToTimeFractional($prks_event_utc, $delay_time);
                  Dada::logMsg(1, $dl, "EVENT_".$i.": PRKS=".$prks_event_utc." DM=".$dm." MPSR=".$mpsr_event_utc." delay=".$delay_time);

                  # now determine the smearing across the MPSR band
                  $cmd = "dmsmear -f 830 -b 40 -n 1024 -d ".$dm." -q";
                  Dada::logMsg(2, $dl, "main: cmd=".$cmd);
                  ($result, $response) = Dada::mySystem($cmd);
                  Dada::logMsg(3, $dl, "main: ".$result." ".$response);

                  $response =~ s/ +$//;
                  $smearing_time = $response;

                  $filter_time   = (2 ** $filter) * 0.000064;

                  $total_time = $filter_time + $smearing_time;

                  # be generous and allow lots of time before and after the event
                  $start_time = -2 * $total_time;
                  $end_time   = 4 * $total_time;

                  # determine abosulte start time in UTC
                  $start_utc = Dada::addToTimeFractional($mpsr_event_utc, $start_time);

                  # determine abosulte end time in UTC
                  $end_utc = Dada::addToTimeFractional($mpsr_event_utc, $end_time);

                  Dada::logMsg(1, $dl, "EVENT_".$i.": MPSR START=".$start_utc." END=".$end_utc);

                  $event_message .= $start_utc." ".$end_utc." ".$dm." ".$snr." ".$filter_time." 1\n";
                }

                for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
                {
                  $host = $cfg{"PWC_".$i};
                  $port = int($cfg{"CLIENT_AQ_EVENT_BASEPORT"}) + int($i);
                  if ($cfg{"PWC_STATE_".$i} ne "inactive")
                  {
                    Dada::logMsg(1, $dl, "main: opening connection for FRB dump to ".$host.":".$port);
                    $sock = Dada::connectToMachine($host, $port, 1);
                    if ($sock)
                    {
                      Dada::logMsg(1, $dl, "main: connection to ".$host.":".$port." established");
                      print $sock $event_message;
                      close ($sock);
                      $sock = 0;
                    }
                    else
                    {
                      Dada::logMsg(0, $dl, "main: connection to ".$host.":".$port." failed");
                    }
                  }
                  else
                  {
                    Dada::logMsg(2, $dl, "main: skipping ".$host.":".$port." since marked inactive PWC");
                  }
                }
              }
              else
              {
                Dada::logMsgWarn($error, "Could not get UTC_START from TMC_INTERFACE: ".$response);
              }
            }
            Dada::logMsg(2, $dl, "main: messages passed to dada_dbevent");

            Dada::logMsg(2, $dl, "main: reply ok to hipsr-srv0");
            print $rh "ok\r\n";
          }
        }
      }
    }
  }
  Dada::logMsg(2, $dl, "Joining threads");

  # rejoin threads
  $control_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  exit 0;
}


#
# Handle INT AND TERM signals
#
sub sigHandle($)
{
  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  if ($quit_threads)
  {
    print STDERR basename($0)." : exiting\n";
    exit(1);
  }
  $quit_threads = 1;
}


#
# Thread to wait for quit signal
#
sub controlThread($$) 
{
  Dada::logMsg(1, $dl, "controlThread: thread starting");

  my ($pid_file, $quit_file) = @_;

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_threads)) 
  {
    Dada::logMsg(3, $dl, "controlThread: Polling for ".$quit_file);
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_threads = 1;

  Dada::logMsg(2, $dl, "controlThread: Unlinking PID file ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(1, $dl, "controlThread: exiting");

}

