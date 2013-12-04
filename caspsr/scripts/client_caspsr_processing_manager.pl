#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use Dada;
use Caspsr;
use threads;         # standard perl threads
use threads::shared; # standard perl threads
use IO::Socket;      # Standard perl socket library
use IO::Select;      # Allows select polling on a socket
use Net::hostent;
use File::Basename;

sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Function Prototypes
#
sub good();
sub msg($$$);
sub process($$);
sub decimationThread($);
sub basebandThread();
sub auxiliaryThread();
sub auxiliaryControlThread();

#
# Declare Global Variables
# 
our $user;
our $dl : shared;
our $daemon_name : shared;
our $client_logger;
our %cfg : shared;
our $pwc_id : shared;
our $hostname : shared;
our $quit_daemon : shared;
our $log_host;
our $log_port;
our $log_sock;
our $recv_db : shared;            # DB where data is being received over IB
our $proc_db : shared;            # DB where data is being processed 
our $baseband_db : shared;        # DB where baseband data is read 
our $baseband_port : shared;      # port for baseband recorder
our $auxiliary_command : shared;  # Command from auxiliaryControlThread 
our $auxiliary_state : shared;    # state of the auxiliaryThread 
our $primary_state : shared;    # state of the auxiliaryThread 

#
# Initialize Global variables
#
$user = "caspsr";
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$client_logger = "client_caspsr_src_logger.pl";
%cfg = Caspsr::getConfig();
$pwc_id = $ARGV[0];
$hostname = Dada::getHostMachineName();
$quit_daemon = 0;
$log_host = 0;
$log_port = 0;
$log_sock = 0;
$recv_db = "";
$proc_db = "";
$baseband_db = "";
$baseband_port = "";
$auxiliary_command = "disable";
$auxiliary_state = "inactive";
$primary_state = "inactive";

# ensure that our pwc_id is valid 
if (!Dada::checkPWCID($pwc_id, %cfg))
{
  usage();
  exit(1);
}

# Autoflush STDOUT
$| = 1;

{
  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file       = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  $log_host          = $cfg{"SERVER_HOST"};
  $log_port          = $cfg{"SERVER_SYS_LOG_PORT"};

  my $control_thread = 0;
  my $decimation_thread = 0;
  my $auxiliary_thread = 0;
  my $auxiliary_control_thread = 0;
  my $baseband_thread = 0;
  my $prev_utc_start = "";
  my $prev_obs_offset = "";
  my $utc_start = "";
  my $obs_offset = "";
  my $obs_end = 0;
  my $quit = 0;
  my $result = "";
  my $response = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good();
  if ($result ne "ok") {
    print STDERR "ERROR failed to start: ".$response."\n";
    exit 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  # open a connection to the nexus logging port
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  msg(0,"INFO", "STARTING SCRIPT");

  # start the control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  my $raw_header = "";
  my $cmd = ""; 
  my %h = ();
  my $tdec = 1;

  my $aj_test = 0;

  $recv_db     = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"RECEIVING_DATA_BLOCK"});
  $proc_db     = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"PROCESSING_DATA_BLOCK"});
  $baseband_db = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"BASEBAND_DATA_BLOCK"});

  # launch a thread to handle recording of event driven baseband data
  $baseband_thread = threads->new(\&basebandThread);

  # launch a thread to handle auxiliary processing of processing data block
  $auxiliary_control_thread = threads->new(\&auxiliaryControlThread);
  
  # main Loop
  while (!$quit_daemon) 
  {
    # wait for the next header on the receiving data_block
    $cmd =  "dada_header -k ".$recv_db;
    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", $cmd." returned");
 
    # check the return value, since dada_header can fail if killed
    if ($? == 0) 
    {
      %h = ();
      %h = Dada::headerToHash($raw_header);
      
      $tdec = 1;
      # check if this transfer requires decimation and is 2 or 6
      if (defined($h{"TDEC"}) && (($h{"TDEC"} == 2)))
      {
        $tdec = $h{"TDEC"};
      }

      msg(2, "INFO", "main: decimationThread(".$tdec.")");
      $decimation_thread = threads->new(\&decimationThread, $tdec);

      sleep(1);

      # launch an auxiliary thread to process the auxiliary stream of just this observation
      # this thread should return after encountering the OBS_XFER == -1 (1 byte) xfer
      $auxiliary_thread = threads->new(\&auxiliaryThread);

      # continue to run the processing thread until this observation is complete
      $obs_end = 0;
      while (!$obs_end && !$quit_daemon)
      {

        msg(2, "INFO", "main: process(".$prev_utc_start.", ".$prev_obs_offset.")");
        ($quit_daemon, $utc_start, $obs_offset, $obs_end) = process($prev_utc_start, $prev_obs_offset);
        msg(2, "INFO", "main: process quit_daemon=".$quit_daemon." utc_start=". 
                       $utc_start." obs_offset=".$obs_offset." obs_end=".$obs_end);

        if ($utc_start eq "invalid") 
        {
          if (!$quit_daemon) 
          {
            msg(0, "ERROR", "processing return an invalid obs/header");
            sleep(1);
          } 
        }
        else 
        {
          msg(2, "INFO", "process was successful");
        }

        $prev_utc_start = $utc_start;
        $prev_obs_offset = $obs_offset;
      }

      if ($decimation_thread)
      {
        msg(2, "INFO", "main: joining decimation_thread");
        $result = $decimation_thread->join();
        msg(2, "INFO", "main: decimation_thread joined");
        $decimation_thread = 0;
        if ($result ne "ok") 
        {
          msg(0, "WARN", "main: decimation_thread failed");
        }
      }

      if ($auxiliary_thread)
      {
        msg(2, "INFO", "main: joining auxiliary_thread");
        $result = $auxiliary_thread->join();
        msg(2, "INFO", "main: auxiliary_thread joined");
        $auxiliary_thread = 0;
        if ($result ne "ok")
        {
          msg(0, "WARN", "main: auxiliary_thread failed");
        }
      }
    }
    else
    {
      if ($quit_daemon)
      {
        msg(2, "INFO", "dada_header -k ".$recv_db." failed, but quit_daemon true");
      }
      else
      {
        msg(0, "WARN", "dada_header -k ".$recv_db." failed: ".$response);
        sleep 1;
      }
    }
  }

  msg(2, "INFO", "main: joining threads");
  $control_thread->join();
  msg(2, "INFO", "main: controlThread joined");

  $baseband_thread->join();
  msg(2, "INFO", "main: basebandThread joined");

  $auxiliary_control_thread->join();
  msg(2, "INFO", "main: auxiliaryControlThread joined");

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($log_sock);

  exit 0;
}

###############################################################################
#
# Run the CASPSR decimation to resmaple the input data [if required]
#
sub decimationThread($) 
{
  my ($tdec) = @_;

  msg(2, "INFO", "decimationThread(".$tdec.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "caspsr_dbdecidb -S -z -t ".$tdec." ".$recv_db." ".$proc_db.
         " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." deci";

  msg(2, "INFO", "decimationThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(3, "INFO", "decimationThread: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "decimationThread: ".$cmd." failed:  ".$response);
  }

  return $result;
}

###############################################################################
#
# Baseband recording thread, is always active and waits on the aux datablock
# fada for any observations to process
#
sub basebandThread()
{
  my $cmd = "";
  my $result = "";
  my $response = "";
  my %h = ();
  my $obs_offset = "";
  my $obs_xfer = "";

  # determine the port number for the remote connection
  my $localhost = Dada::getHostMachineName();
  my $i = 0;
  my $port = 0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    if ($cfg{"PWC_".$i} eq $localhost)
    {
      $port = $cfg{"BASEBAND_RAID_PORT_".$i};
    }
  }

  if ($port eq 0)
  {
    msg(0, "ERROR", "basebandThread: could not determine port number");
    $quit_daemon = 1;
  }

  msg(2, "INFO", "basebandThread: running on port ".$port);

  while (!$quit_daemon)
  {
    # Get the next header from the baseband data block
    $cmd = "dada_header -k ".$baseband_db;
    msg(2, "INFO", "basebandThread: running ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(2, "INFO", "basebandThread: ".$cmd." returned");

    # check that this returned ok
    if ($result eq "ok")
    {
      %h = Dada::headerToHash($response);
      $obs_xfer = $h{"OBS_XFER"};
      $obs_offset = $h{"OBS_OFFSET"};
      msg(1, "INFO", "basebandThread: processing XFER=".$obs_xfer." OFFSET=".$obs_offset);

      # if this is the final "1 byte" transfer, just ignoring it
      if ($obs_xfer eq "-1")
      {
        $cmd = "dada_dbnull -s -k ".$baseband_db;
      }
      # transfer just 1 XFER to raid0-ib
      else
      {
        $cmd = "dada_dbib -s -c 16384 -k ".$baseband_db." -p ".$port." raid0-ib";
      }
      $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." dbib";

      msg(1, "INFO", "basebandThread: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(2, "INFO", "basebandThread: ".$result." ".$response);
      if ($result ne "ok") {
        msg(0, "ERROR", "basebandThread: ".$cmd." failed:  ".$response);
      }
    }
    else
    {
      if (!$quit_daemon) 
      {
        msg(0, "WARN", "basebandThread: ".$cmd." failed");
        sleep 1;
      }
    }
  }

  msg(2, "INFO", "basebandThread: exiting");

  return 0;
}


###############################################################################
#
# control the processing of the auxiliary data stream the processing data block
#
# only process 1 observation [multiple XFERs], then return
#
sub auxiliaryThread()
{
  msg(2, "INFO", "auxiliaryThread: starting");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my %h = ();
  my $obs_offset = "";
  my $obs_xfer = "";
  my $end_thread = 0;

  # we can run 3 programs on the auxiliary stream of the processing data block:
  #   ignore:         dada_dbnull -z -s -k proc_db  
  #   full baseband:  dada_dbib -s -c 16384 -k proc_db -p [port]
  #   event baseband: dada_dbevent proc_db base_db

  while ((!$quit_daemon) && (!$end_thread))
  {
    # get the next header from the processing data block
    $cmd = "dada_header -k ".$proc_db;
    msg(2, "INFO", "auxiliaryThread: running ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    msg(3, "INFO", "auxiliaryThread: ".$cmd." returned");

    # check that this returned ok
    if ($result eq "ok")
    {
      %h = Dada::headerToHash($response);
      $obs_xfer = $h{"OBS_XFER"};
      $obs_offset = $h{"OBS_OFFSET"};
      msg(2, "INFO", "auxiliaryThread: processing XFER=".$obs_xfer." OFFSET=".$obs_offset);

      if ($obs_xfer eq "-1")
      {
        $cmd = "dada_dbnull -s -k ".$proc_db;
        $auxiliary_state = "inactive";
        $end_thread = 1;
      }
      elsif ($auxiliary_command eq "baseband")
      {
        $cmd = "dada_dbib -s -c 16384 -k ".$proc_db." -p ".$baseband_port." raid0-ib";
        $auxiliary_state = "baseband";
      } 
      elsif ($auxiliary_command eq "events")
      {
        $cmd = "dada_dbevent -s -k ".$proc_db." ".$baseband_db;
        $auxiliary_state = "events";
      } 
      else
      {
        $cmd = "dada_dbnull -q -s -k ".$proc_db;
        $auxiliary_state = "inactive";
      }

      $cmd .= " 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." aux";

      msg(2, "INFO", "auxiliaryThread: running ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "auxiliaryThread: ".$cmd." returned");

    }
    else
    {
      if (!$quit_daemon)
      {
        msg(1, "WARN", "auxiliaryThread: dada_header failed: ".$response);
        return "fail";
      } 
      else
      {
        msg(2, "INFO", "auxiliaryThread: dada_header failed, but quit_daemon true: ".$response);
      }
    }
  }

  msg(2, "INFO", "auxiliaryThread: exiting");
  return "ok";
}

###############################################################################
#
# listen on port for auxiliary thread control commands
#
sub auxiliaryControlThread()
{
  msg(2, "INFO", "auxiliaryControlThread: starting");
  my $cmd = "";
  my $result = "";
  my $response = "";

  my $localhost = Dada::getHostMachineName();
  my $aux_port = $cfg{"CLIENT_DECIDB_PORT"};

  my $aux_sock = new IO::Socket::INET (
    LocalHost => $localhost,
    LocalPort => $aux_port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1
  );

  if (!$aux_sock) 
  {
    msg(1, "INFO",  "could not create listening socket: ".$localhost.":".$aux_port);
    return -1;
  }

  my $read_set = new IO::Select();
  $read_set->add($aux_sock);

  my $rh = 0;
  my $peeraddr = 0;
  my $handle = 0;
  my $hostinfo = 0;
  my $hostname = "";
  my $command = "";

  while (! $quit_daemon)
  {
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
    foreach $rh (@$readable_handles)
    {
      if ($rh == $aux_sock)
      {
        $handle = $rh->accept();
        $handle->autoflush(1);
        $read_set->add($handle);

        $peeraddr = $handle->peeraddr;
        $hostinfo = gethostbyaddr($peeraddr);
        $hostname = $hostinfo->name;

        msg(2, "INFO", "accepting connection from ".$hostname);
      }
      else
      {
        $command = <$rh>;
        if (! defined $command)
        {
          msg(2, "INFO", "client disconneted");
          $read_set->remove($rh);
          $rh->close();
        }
        else
        {

          # clean the line up a little
          $command =~ s/\r//;
          $command =~ s/\n//;
          $command =~ s/#(.)*$//;
          $command =~ s/\s+$//;
          $command =~ s/\0//g;      # remove all null characters


          if ($command eq "baseband")
          {
            msg(1, "INFO", "enabling full baseband recording");
            $auxiliary_command = "baseband";
            msg(2, "INFO", "-> auxiliary Enabling full baseband recording...");
            print $rh "auxiliary Enabling full baseband recording...\n";

            print $rh "ok\r\n";
          }
          elsif ($command eq "events")
          {
            msg(1, "INFO", "enabling baseband event recording");
            $auxiliary_command = "events";
            msg(2, "INFO", "-> auxiliary Enabling baseband event recording...");
            print $rh "auxiliary Enabling baseband event recording...\r\n";

            print $rh "ok\r\n";
          }
          elsif ($command eq "disable")
          {
            msg(1, "INFO", "disabling all recording");
            $auxiliary_command = "disable";

            msg(2, "INFO", "-> auxiliary Disabling all baseband recording...");
            print $rh "auxiliary Disabling all baseband recording...\r\n";

            print $rh "ok\r\n";
          }
          elsif ($command eq "state")
          {
            msg(2, "INFO", "<- ".$command);

            msg(2, "INFO", "-> primary ".$primary_state." fold");
            print $rh "primary ".$primary_state." fold\r\n";
            msg(2, "INFO", "-> auxiliary ".$auxiliary_state." ".$auxiliary_command);
            print $rh "auxiliary ".$auxiliary_state." ".$auxiliary_command."\r\n";

            print $rh "ok\r\n";
          }
          elsif ($command eq "")
          {
            msg(1, "INFO", "empty command");
            print $rh "fail\r\n";
          }
          else
          {
            msg(1, "INFO", "unrecognized event [".$command."], disabling all recording");
            $auxiliary_command = "disable";
            print $rh "unrecognised event [".$command."]\r\n";
            print $rh "fail\r\n";
          }
        }
      }
    }
  }
  msg(2, "INFO", "auxiliaryControlThread: exiting");
}

###############################################################################
#
# Process an observation
#
sub process($$) 
{

  my ($prev_utc_start, $prev_obs_offset) = @_;

  my $localhost = Dada::getHostMachineName();
  my $bindir = Dada::getCurrentBinaryVersion();
  my $processing_dir = $cfg{"CLIENT_RESULTS_DIR"};

  my $copy_obs_start_thread = 0;
  my $utc_start = "invalid";
  my $obs_offset = "invalid";

  my $raw_header = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my %h = ();
  my $obs_xfer = 0;
  my $obs_end = 1;
  my $proc_cmd = "";


  # Get the next header from the processing data block
  $cmd = "dada_header -k ".$proc_db;
  msg(2, "INFO", "process: running ".$cmd);
  $raw_header = `$cmd 2>&1`;
  msg(2, "INFO", "process: ".$cmd." returned");

  # check that this returned ok
  if ($? == 0) 
  {
    msg(2, "INFO", "process: headerToHash()");
    %h = Dada::headerToHash($raw_header);
    $utc_start = $h{"UTC_START"};
    $obs_offset = $h{"OBS_OFFSET"};

    msg(2, "INFO", "process: processHeader()");
    ($result, $response) = Caspsr::processHeader($raw_header, $cfg{"CONFIG_DIR"}); 
    msg(2, "INFO", "process: processHeader() ".$result." ".$response);

    if ($result ne "ok") 
    {
      msg(0, "INFO", "DADA header malformed, jettesioning xfer");  
      msg(0, "ERROR", $response);
      $proc_cmd = "dada_dbnull -S -k ".$proc_db;
    }
    else
    {
      msg(2, "INFO", "process: DADA header correct");

      $proc_cmd = $response;

      # check for the OBS_XFER 
      if (defined($h{"OBS_XFER"})) {
        $obs_xfer = $h{"OBS_XFER"};
        $obs_end = ($obs_xfer eq "-1") ? 1 : 0;
      }

      msg(2, "INFO", "process: UTC_START=".$utc_start.", FREQ=".$h{"FREQ"}.
                     ", OBS_OFFSET=".$obs_offset.", PROC_FILE=".$h{"PROC_FILE"}.
                     ", OBS_XFER=".$obs_xfer." OBS_END=".$obs_end);

      # special case for and END of XFER
      if ($obs_xfer eq "-1") 
      {
        msg(1, "INFO", "Ignoring final [extra] transfer");
        msg(1, "INFO", "Ignoring: UTC_START=".$utc_start.", OBS_OFFSET=".$obs_offset.
                       ", OBS_XFER=".$obs_xfer." OBS_END=".$obs_end);
        $proc_cmd = "dada_dbnull -s -k ".$proc_db;
      } 
      else 
      {
        # if we are a regular XFER
        if ($utc_start eq $prev_utc_start) 
        {
          # dspsr should process an entire observation, so this is an error state
          if ($proc_cmd =~ m/dspsr/) 
          {
            msg(0, "INFO", "UTC_START repeated, jettesoning transfer");
            $proc_cmd = "dada_dbnull -S -k ".$proc_db;
          }
          # the same header has been repeated, also error state
          elsif ($obs_offset eq $prev_obs_offset)
          {
            msg(0, "INFO", "UTC_START and OBS_OFFSET repeated, obs_xfer=".$obs_xfer);
            $proc_cmd = "dada_dbnull -S -k ".$proc_db;
            msg(1, "INFO", "Ignoring repeat transfer");
          } 
          # otherwise the proc command will handle this continuing xfer
          else 
          {
            msg(0, "INFO", "Continuing Obs: UTC_START=".$utc_start.  " XFER=".
                           $obs_xfer." OFFSET=".$obs_offset);
          }
        }
        # this is a new observation, setup the directories
        else
        {
          msg(0, "INFO", "New Observation: UTC_START=".$utc_start." XFER=".$obs_xfer." OBS_OFFSET=".$obs_offset);

          # create the local directory and UTC_START file
          msg(2, "INFO", "process: createLocalDirectory()");
          ($result, $response) = createLocalDirectory($utc_start, $raw_header);
          msg(2, "INFO", "process: ".$result." ".$response);

          # send the obs.start to the server in background thread 
          msg(2, "INFO", "process: copyObsStartThread(".$utc_start.")");
          $copy_obs_start_thread = threads->new(\&copyObsStartThread, $utc_start);
        }

        $processing_dir .= "/".$utc_start;

        # replace <DADA_INFO> tags with the matching input .info file
        if ($proc_cmd =~ m/<DADA_INFO>/)
        {
          my $tmp_info_file =  "/tmp/caspsr_".$proc_db.".info";
          # ensure a file exists with the write processing key
          if (! -f $tmp_info_file)
          {
            open FH, ">".$tmp_info_file;
            print FH "DADA INFO:\n";
            print FH "key ".$proc_db."\n";
            close FH;
          }
          $proc_cmd =~ s/<DADA_INFO>/$tmp_info_file/;
        }

        # replace <DADA_KEY> tags with the matching input key
        $proc_cmd =~ s/<DADA_KEY>/$proc_db/;

        if ($proc_cmd =~ m/dspsr/) 
        {
          # add the ARCHIVE_MOD command line option
          if (exists($cfg{"ARCHIVE_MOD"})) 
          {
            my $archive_mod = $cfg{"ARCHIVE_MOD"};
            if ($proc_cmd =~ m/-L \d+/) {
              $proc_cmd =~ s/-L \d+/-L $archive_mod/;
            } else {
              $proc_cmd .= " -L ".$archive_mod;
            }
          }

          if ($proc_cmd =~ m/-tdec 2/)
          {
            msg(0, "INFO", "process: removing -tdec 2 from proc_cmd");
            $proc_cmd =~ s/-tdec 2//;
          }
        }
      }
    }

    my $binary = "";
    my $rest = "";
    ($binary, $rest) = split(/ /,$proc_cmd,2);

    chdir $processing_dir;

    # run the command

    $cmd = $proc_cmd." 2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." proc; exit \${PIPESTATUS[0]}";

    msg(2, "INFO", "process: ".$cmd);

    $primary_state = "fold";

    msg(1, "INFO", "START: ".$proc_cmd);
    system($cmd);
    my $rval = $? >> 8;
    msg(1, "INFO", "END  : ".$proc_cmd." returned ".$rval);

    $primary_state = "inactive";

    if ((!$quit_daemon) && ($rval != 0))
    {
      msg(0, "WARN", $binary." failed, should restart backend!");
      # $cmd = "dada_dbscrubber -k ".$proc_db;
      # msg(0, "INFO", "process: ".$cmd);
      # ($result, $response) = Dada::mySystem($cmd);
      # msg(0, "INFO", "process: ".$result." ".$response);
    } 
    else 
    {
      if (($binary =~ m/dspsr/) || ($binary =~ m/caspsr_dbnum/)) {
        $obs_end = 1;
      }
    }

    # if we copied the obs.start file, join the thread
    if ($copy_obs_start_thread) 
    {
      msg(2, "INFO", "process: joining copyObsStartThread");
      $copy_obs_start_thread->join();
      msg(2, "INFO", "process: copyObsStartThread joined");
    }

    if (($obs_end) || ($binary =~ m/dspsr/))
    {
      touchPwcFinished($h{"UTC_START"});
    }

    return (0, $utc_start, $obs_offset, $obs_end);

  } else {

    if (!$quit_daemon) {
      msg(0, "WARN", "dada_header -k ".$proc_db." failed");
      sleep 1;
    }
    return (1,$utc_start, $obs_offset, $obs_end);
  }
}
  

###############################################################################
#
# Create the local directory required for this observation
#
sub createLocalDirectory($$) {

  my ($utc_start, $raw_header) = @_;

  msg(2, "INFO", "createLocalDirectory(".$utc_start.", raw_header)");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $file = "";

  my $archive_dir = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start;
  my $results_dir = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start;

  # Create the archive and results dirs
  $cmd = "mkdir -p ".$archive_dir." ".$results_dir;
  msg(2, "INFO", "createLocalDirectory: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirectory: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0,"ERROR", "Could not create local dirs: ".$response);
    return ("fail", "could not create local dirs: ".$response);
  }

  # Set group sticky bit on local archive dir
  $cmd = "chmod g+s ".$archive_dir." ".$results_dir;
  msg(2, "INFO", "createLocalDirectory: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  msg(2, "INFO", "createLocalDirectory: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "WARN", "chmod g+s failed on ".$archive_dir." ".$results_dir);
  }

  # create an obs.start file in the archive dir
  $file = $results_dir."/obs.start";
  open(FH,">".$file.".tmp");
  print FH $raw_header;
  close FH;
  rename($file.".tmp",$file);

  return ("ok", $file);

}

###############################################################################
#
# Copies the obs.start file via NFS to the server's results directory.
#
sub copyObsStartThread($) {

  my ($utc_start) = @_;

  my $localhost = Dada::getHostMachineName();
  my $local_file = $cfg{"CLIENT_RESULTS_DIR"}."/".$utc_start."/obs.start";
  my $remote_file = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$localhost."_obs.start";
  my $cmd = "";
  my $result = "";
  my $response = "";

  if (! -f $local_file) {
    msg(0, "ERROR", "copyObsStartThread: obs.start file [".$local_file."] did not exist");
    return ("fail", "obs.start file did not exist");
  }

  # wait at least 10 seconds to be sure the directories are created on the server
  msg(2, "INFO", "copyObsStartThread: sleep(10)");
  sleep(10);

  $cmd = "scp -B ".$local_file." dada\@srv0:".$remote_file;
  msg(2, "INFO", "copyObsStartThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd, 0);
  msg(2, "INFO", "copyObsStartThread: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "copyObsStartThread: scp [".$cmd."] failed: ".$response);
    return ("fail", "scp failed: ".$response);
  }

  $cmd = "cp ".$local_file." ".$cfg{"CLIENT_ARCHIVE_DIR"}."/".$utc_start."/";
  msg(2, "INFO", "copyObsStartThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd, 0);
  msg(2, "INFO", "copyObsStartThread: ".$result." ".$response);
  if ($result ne "ok") {
    msg(0, "ERROR", "copyObsStartThread: cp [".$cmd."] failed: ".$response);
    return ("fail", "cp failed: ".$response);
  }


  return ("ok", "");

}

###############################################################################
#
# copy the pwc.finished file to the server
#
sub touchPwcFinished($) {
  
  my ($utc_start) = @_;
  
  my $localhost = Dada::getHostMachineName();
  my $fname = $localhost."_pwc.finished";
  my $cmd = "";
  my $result = "";
  my $response = "";

  if ( -f $fname) {
    return ("ok", "");

  } else {

    # touch the local file
    $cmd = "touch ".$fname;
    msg(2, "INFO", "touchPwcFinished: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd, 0);
    msg(2, "INFO", "touchPwcFinished: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "touchPwcFinished: ".$cmd." failed: ".$response);
      return ("fail", "local touch failed: ".$response);
    }
  
    $cmd = "ssh -o BatchMode=yes -l dada srv0 'touch ".$cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/".$fname."'";
    msg(2, "INFO", "touchPwcFinished: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd, 0);
    msg(2, "INFO", "touchPwcFinished: ".$result." ".$response);
    if ($result ne "ok") {
      msg(0, "ERROR", "touchPwcFinished: ".$cmd." failed: ".$response);
      return ("fail", "remote touch failed: ".$response);
    }
  
    return ("ok", "");
  }
}

###############################################################################
#
# Control thread to handle quit requests
#
sub controlThread($) {

  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread: starting (".$pid_file.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  # poll for the existence of the control file
  while ((!$quit_daemon) && (! -f $host_quit_file) && (! -f $pwc_quit_file)) {
    sleep(1);
  }

  $quit_daemon = 1;

  my @tokill = ("dada_header", "dspsr", "caspsr_dbdecidb", "dada_dbnull");
  my $i = 0;
  my $regex = "";
  
  for ($i=0; $i<=$#tokill; $i++) 
  {
    $regex = "^".$tokill[$i];
    msg(2, "INFO", "controlThread: killProcess(".$regex.", ".$user.")");
    ($result, $response) = Dada::killProcess($regex, $user);
    msg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(1, "WARN", "controlThread: killProcess for ".$regex." failed: ".$response);
    }
  }

  if ( -f $pid_file) {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(1, "WARN", "controlThread: PID file did not exist on script exit");
  } 

  msg(2, "INFO", "controlThread: exiting");
}

###############################################################################
#
# logs a message to the nexus logger and prints to stdout
#
sub msg($$$) {

  my ($level, $type, $msg) = @_;
  if ($level <= $dl) {
    my $time = Dada::getCurrentDadaTime();
    if (! $log_sock ) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }
    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $hostname, $time, "sys", $type, "proc mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


###############################################################################
#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # Tell threads to try and quit
  $quit_daemon = 1;
  sleep(3);
  
  if ($log_sock) {
    close($log_sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

###############################################################################
#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}


###############################################################################
#
# Test to ensure all module variables are set before main
#
sub good() {

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  =  $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  # check the quit file does not exist on startup
  if ((-f $host_quit_file) || (-f $pwc_quit_file)) {
    return ("fail", "Error: quit file existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # check required gloabl parameters
  if ( ($user eq "") || ($client_logger eq "")) {
    return ("fail", "Error: a package variable missing [user, client_logger]");
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  $baseband_port = "";
  my $localhost = Dada::getHostMachineName();
  my $i=0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    if ($cfg{"PWC_".$i} eq $localhost)
    {
      $baseband_port = $cfg{"BASEBAND_RAID_PORT_".$i};
    }
  }

  if ($baseband_port eq "")
  {
    return ("fail", "Error: could not determine baseband port number");
  }

  return ("ok", "");

}

