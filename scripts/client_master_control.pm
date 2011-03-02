package Dada::client_master_control;

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use Thread::Queue;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );     # eg: TAG => [ qw!name1 name2! ],

  # your exported package globals go here,
  # as well as any optionally exported functions
  @EXPORT_OK   = qw($dl $daemon_name $pwc_add %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $daemon_name;
our $pwc_add;
our @daemons;
our @binaries;
our @helper_daemons;
our @dbs;
our $daemon_prefix;
our $control_dir;
our $host : shared;
our %cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $port;
our $sock;
our $raw_disk_result : shared;
our $raw_disk_response : shared;
our $unproc_files_result : shared;
our $unproc_files_response : shared;
our $daemons_result : shared;
our $daemons_response : shared;
our $daemons_response_xml : shared;
our $db_result : shared;
our $db_response : shared;
our $db_status : shared;
our $load_result : shared;
our $load_response : shared;
our $temp_result : shared;
our $temp_response : shared;

#
# initialize package globals
#
$dl = 1; 
$daemon_name = 0;
$pwc_add = "";
@daemons = ();
@binaries = ();
@dbs = ();
@helper_daemons = ();
$daemon_prefix = "";
$control_dir = "";
$host = "";
%cfg = ();

#
# initialize other variables
#
$quit_daemon = 0;
$port = 0;
$sock = 0;
$raw_disk_result = "ok";
$raw_disk_response= "0.0";
$unproc_files_result = "ok";
$unproc_files_response = "0.0";
$daemons_result = "ok";
$daemons_response = "";
$daemons_response_xml = "";
$db_result = "na";
$db_response = "0 0 ";
$db_status = "0 0 ";
$load_result = "ok";
$load_response = "0.00,0.00,0.00";
$temp_result = "ok";
$temp_response = "0";


###############################################################################
#
# package functions
# 

sub main() {

  $port = $cfg{"CLIENT_MASTER_PORT"};
  $sock = 0;

  my $result = "";
  my $response = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good();
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  my $log_file       = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";;
  my $pid_file       = $control_dir."/".$daemon_name.".pid";
  my $quit_file      = $control_dir."/".$daemon_name.".quit";
  my $archive_dir    = $cfg{"CLIENT_ARCHIVE_DIR"};   # hi res archive storage
  my $results_dir    = $cfg{"CLIENT_RESULTS_DIR"};   # dspsr output directory

  my $i=0;
  for ($i=0; $i<=$#daemons; $i++) {
    Dada::logMsg(2, $dl, "main: daemon[".$i."] = ".$daemons[$i]);
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # start the daemons that maintain information on timely operations
  my $disk_thread_id = 0;
  my $daemons_thread_id = 0;
  my $db_thread_id = 0;
  my $load_thread_id = 0;
  my $temp_thread_id = 0;
  my $control_thread_id = 0;

  $disk_thread_id = threads->new(\&diskThread);
  $daemons_thread_id = threads->new(\&daemonsThread);
  $db_thread_id = threads->new(\&dbThread);
  $load_thread_id = threads->new(\&loadThread);
  $temp_thread_id = threads->new(\&tempThread);
  $control_thread_id = threads->new(\&controlThread, $quit_file, $pid_file);
 
  my $read_set = new IO::Select();
  my $rh = 0;
  my $handle = 0;

  # create 2 queues, one for incoming commands, and one to release the 
  # file handle from scope when the command is completed
  my $in = new Thread::Queue;
  my $out = new Thread::Queue;

  # hash to hold file_handles -> sockets whilst they are being processed
  my %handles = ();

  # number of worker threads to handle multiple connections
  my $n_threads = 3;
  my @tids = ();
  my $tid = 0;
  my $file_handle = 0;

  for ($i=0; $i<$n_threads; $i++) {
    $tid = threads->new(\&commandThread, $in, $out, $i);
    push @tids, $tid;
  }

  $read_set->add($sock);

  my @handles = ();

  while (!$quit_daemon) {

   # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles) {

      # start a thread to handle this connection
      if ($rh == $sock) {

        $handle = $rh->accept();

        # convert the socket to a filehandle and add this to the queue
        $file_handle = fileno($handle);
        Dada::logMsg(3, $dl, "main: enqueuing socket as file handle [".$file_handle."]");
        $in->enqueue($file_handle);

        # now save the file_handle and socket in hash so it does not go out of scope
        # and for closing later
        $handles{$file_handle} = $handle;

        $handle = 0;

      } else {
        Dada::logMsg(0, $dl, "main: received connection not on socket");
      }
    }

    # if one of the commandThreads has finished with a socket, close that socket
    # and remove it from the handles hash
    if ($out->pending()) {
      # get the file handle

      $file_handle = $out->dequeue();
      Dada::logMsg(3, $dl, "main: removing handle [".$file_handle."]");

      # get the socket object from the hash and close it
      $handle = $handles{$file_handle};
      $handle->close();

      # remove the socket from the hash
      delete $handles{$file_handle};
      $handle = 0;
      $file_handle = 0;
    }

  }

  Dada::logMsg(0, $dl, "Joining threads");

  for ($i=0; $i<$n_threads; $i++) {
    $tids[$i]->join();
  }
  $disk_thread_id->join();
  $daemons_thread_id->join();
  $db_thread_id->join();
  $load_thread_id->join();
  $temp_thread_id->join();
  $control_thread_id->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
  close($sock);

  return 0;

}

sub commandThread($$$) {
 
  (my $in, my $out, my $tid) = @_;

  Dada::logMsg(1, $dl, "commandThread[".$tid."] starting");

  my $fh = 0;
  my $sock;

  my $have_connection = 1;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $log_message = 0;
  my $quit_wait = 10;

  while (!$quit_daemon) {

    while ($in->pending) {

      if ($quit_daemon) {

        if ($quit_wait > 0) {
          Dada::logMsg(2, $dl, "commandThread[".$tid."]: quit_daemon=1 while in->pending=true, waiting...");
          $quit_wait--;
        } else {
          Dada::logMsg(0, $dl, "commandThread[".$tid."]: quit_daemon=1 while in->pending=true, quitting!");
          return 0;
        } 
      }

      # dequeue a file handle from the queue
      $fh = $in->dequeue();

      Dada::logMsg(3, $dl, "commandThread[".$tid."] dequeued socket as [".$fh."]");

      # reopen this file handle as a socket
      open $sock, "+<&=".$fh or warn $! and die;

      $sock->autoflush(1);

      $have_connection = 1;

      Dada::logMsg(3, $dl, "commandThread[".$tid."]: accepting connection");

      while ($have_connection) {

        if ($quit_daemon) {
          Dada::logMsg(1, $dl, "commandThread[".$tid."]: quit_daemon=1 while have_connection=true");
        }

        # read something from the socket
        $cmd = <$sock>;

        if (! defined $cmd) {
          Dada::logMsg(3, $dl, "commandThread[".$tid."]: lost connection");
          $sock->close();
          $have_connection = 0;

        } else {

          # clean the command
          $cmd =~ s/\r//;
          $cmd =~ s/\n//;
          $cmd =~ s/ +$//;

          $log_message = 1;
          if ((($cmd =~ m/get_status/) || ($cmd =~ m/xml/)) && ($dl < 2)) {
            $log_message = 0;
          }

          if ($log_message) {
            Dada::logMsg(1, $dl, "[".$tid."] <- ".$cmd);
          }

          ($result, $response) = handleCommand($cmd); 

          if ($log_message) {
            Dada::logMsg(1, $dl, "[".$tid."] -> ".$result." ".$response);
          }

          # if it was an XML request, do not append result
          if ($cmd =~ m/xml/) {
            print $sock $response."\n";

          # else use the telnet style formatting
          } else {

            if ($result ne "ok") {
              Dada::logMsg(0, $dl, "[".$tid."] cmd=".$cmd.", result=".$result.", response=".$response);
            } 
            if ($response ne "") {
              print $sock $response."\r\n";
            }
            print $sock $result."\r\n";
          }
        }
      }
      Dada::logMsg(3, $dl, "commandThread[".$tid."]: closing connection");
      $out->enqueue($fh);
    }
    sleep (1);
  }

  Dada::logMsg(1, $dl, "commandThread[".$tid."]: thread exiting");

  return 0;

}



sub handleCommand($) {

  (my $string) = @_;

  Dada::logMsg(3, $dl, "handleCommand: string= '".$string."'");

  my @cmds = split(/ /,$string, 2);
  my $key = $cmds[0];

  my $current_binary_dir = Dada::getCurrentBinaryVersion();
  my $cmd = "";
  my $result = "ok";
  my $response = "";

  if ($key eq "clean_scratch") {
    $result = "fail";
    $response = "command deprecated/disbaled";
  }

  elsif ($key eq "clean_archives") {
    $result = "fail";
    $response = "command deprecated/disbaled";
  }

  elsif ($key eq "clean_rawdata") {
    $result = "fail";
    $response = "command deprecated/disbaled";
  }

  elsif ($key eq "clean_logs") {
    $result = "fail";
    $response = "command deprecated/disbaled";
  }

  elsif ($key eq "stop_iface") {
    $cmd = "sudo /sbin/ifdown ".$cfg{"PWC_DEVICE"};
    Dada::logMsg(1, $dl, $cmd);
    ($result, $response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "stop_pwcs") {
    $cmd = "killall ".$cfg{"PWC_BINARY"};
    ($result,$response) = Dada::mySystem($cmd);  
    if (defined($cfg{"PWC_DEVICE"})) {
      $cmd = "sudo /sbin/ifdown ".$cfg{"PWC_DEVICE"};
      Dada::logMsg(1, $dl, $cmd);
      ($result,$response) = Dada::mySystem($cmd);
    }
  }

  elsif ($key eq "stop_pwc") {
    $cmd = "sudo killall -KILL ".$cmds[1];
    ($result,$response) = Dada::mySystem($cmd);  
    if (defined($cfg{"PWC_DEVICE"})) {
      $cmd = "sudo /sbin/ifdown ".$cfg{"PWC_DEVICE"};
      Dada::logMsg(1, $dl, $cmd);
      ($result,$response) = Dada::mySystem($cmd);
    }
  }

  elsif ($key eq "stop_dfbs") {
    $cmd = "killall -KILL ".$cfg{"DFB_SIM_BINARY"};
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "stop_srcs") {
    $cmd = "killall dada_dbdisk dspsr dada_dbnic";
    ($result,$response) = Dada::mySystem($cmd);  
  }

  elsif ($key eq "kill_process") {
    ($result,$response) = Dada::killProcess($cmds[1]);
  }

  elsif ($key eq  "start_bin") {
    $cmd = $current_binary_dir."/".$cmds[1];
    ($result,$response) = Dada::mySystem($cmd);  
  }

  elsif ($key eq "start_iface") {
    $cmd = "sudo /sbin/ifup ".$cfg{"PWC_DEVICE"};
    Dada::logMsg(1, $dl, $cmd);
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "start_pwcs") {
    if (defined($cfg{"PWC_DEVICE"})) {
      $cmd = "sudo /sbin/ifup ".$cfg{"PWC_DEVICE"};
      Dada::logMsg(1, $dl, $cmd);
      ($result,$response) = Dada::mySystem($cmd);
    }
    $cmd = $current_binary_dir."/".$cfg{"PWC_BINARY"}." ".
          " -k ".lc($cfg{"RECEIVING_DATA_BLOCK"}).
          " -c ".$cfg{"PWC_PORT"}.
          " -l ".$cfg{"PWC_LOGPORT"};
  
    # add any instrument specific options here
    $cmd .= $pwc_add;
    
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "set_bin_dir") {
    ($result, $response) = Dada::setBinaryDir($cmds[1]);
  }

  elsif ($key eq "get_bin_dir") {
    ($result, $response) = Dada::getCurrentBinaryVersion();
  }

  elsif ($key eq "get_bin_dirs") {
    ($result, $response) = Dada::getAvailableBinaryVersions();
  }

  elsif ($key eq "destroy_db") {

    my $db_key = $cmds[1];

    $cmd = $current_binary_dir."/dada_db -d -k ".$db_key;
    Dada::logMsg(2, $dl, $cmd);
    ($result, $response) = Dada::mySystem($cmd);

  }

  elsif ($key eq "destroy_dbs") {

    my @dbs = split(/ /,$cfg{"DATA_BLOCKS"});
    my $db = "";
    my $tmp_result = "";
    my $tmp_response = "";

    foreach $db (@dbs) {
      
      if ( (defined $cfg{$db."_BLOCK_BUFSZ"}) && (defined $cfg{$db."_BLOCK_NBUFS"}) ) {
        $cmd = $current_binary_dir."/dada_db -d -k ".lc($db);
        ($tmp_result, $tmp_response) = Dada::mySystem($cmd);

      } else {
        $tmp_result = "fail";
        $tmp_response = "config file configuration error";

      }
     
      if ($tmp_result eq "fail") {
        $result = "fail";
      }

      $response = $response.$tmp_response;        
    }

  }

  elsif ($key eq "init_db") {

    my @args = split(/ /, $cmds[1]);
    my $db_key = $args[0];
    my $db_nbufs = $args[1];
    my $db_bufsz = $args[2];

    $cmd = $current_binary_dir."/dada_db -k ".$db_key.
           " -b ".$db_bufsz." -n ".$db_nbufs;  
    Dada::logMsg(2, $dl, $cmd);
    ($result, $response) = Dada::mySystem($cmd);
  
  } elsif ($key eq "init_dbs") {
 
    my @dbs = split(/ /,$cfg{"DATA_BLOCKS"});
    my $db;
    my $tmp_result = "";
    my $tmp_response = "";

    foreach $db (@dbs) {

      if ( (defined $cfg{$db."_BLOCK_BUFSZ"}) && (defined $cfg{$db."_BLOCK_NBUFS"}) ) {
        $cmd = $current_binary_dir."/dada_db -k ".lc($db).
               " -b ".$cfg{$db."_BLOCK_BUFSZ"}." -n ".$cfg{$db."_BLOCK_NBUFS"};
        ($tmp_result,$tmp_response) = Dada::mySystem($cmd);

      } else {
        $tmp_result = "fail";
        $tmp_response = "config file configuration error";

      }        

      if ($tmp_result eq "fail") {
        $result = "fail";
      }
      $response = $response.$tmp_response;
    }

  }

  elsif ($key eq "stop_daemon") {
    if ($cmds[1] =~ m/_master_control/) {
      Dada::logMsg(0, $dl, "stopping master control [".$cmds[1]."]");
      $quit_daemon = 1;
    } else {
      if ($cmds[1] eq "pwcs") {
        $cmd = "killall ".$cfg{"PWC_BINARY"};
        ($result,$response) = Dada::mySystem($cmd);
        if (defined($cfg{"PWC_DEVICE"})) {
          $cmd = "sudo /sbin/ifdown ".$cfg{"PWC_DEVICE"};
          Dada::logMsg(1, $dl, $cmd);
          ($result,$response) = Dada::mySystem($cmd);
        }
      } else {
        if ($host =~ m/srv0/)
        {
          ($result,$response) = stopDaemon($cmds[1], 300);
        }
        else
        {
          ($result,$response) = stopDaemon($cmds[1], 30);
        }
      }
    }
  }

  elsif ($key eq "stop_daemons") {
    ($result,$response) = stopDaemons(\@daemons);
  }

  elsif ($key eq "stop_helper_daemons") {
    ($result,$response) = stopDaemons(\@helper_daemons);
  }

  elsif ($key eq "daemon_info") {
    $result = $daemons_result;
    $response = $daemons_response;
    #($result, $response) = getDaemonInfo(\@daemons);
  }

  elsif ($key eq "daemon_info_xml") {
    $response = $daemons_response_xml;
  }

  elsif ($key eq "start_daemon") {

    if ($cmds[1] eq "pwcs") {
      if (defined($cfg{"PWC_DEVICE"})) {
        $cmd = "sudo /sbin/ifup ".$cfg{"PWC_DEVICE"};
        Dada::logMsg(1, $dl, $cmd);
        ($result,$response) = Dada::mySystem($cmd);
      }
      $cmd = $current_binary_dir."/".$cfg{"PWC_BINARY"}." ".
             " -k ".lc($cfg{"RECEIVING_DATA_BLOCK"}).
             " -c ".$cfg{"PWC_PORT"}.
             " -l ".$cfg{"PWC_LOGPORT"};

      # add any instrument specific options here
      $cmd .= $pwc_add;
    } else {

      # if there are space separate arguements
      if ($cmds[1] =~ m/ /)
      {
        my @args = split(/ /, $cmds[1], 2);
        $cmd = $daemon_prefix."_".$args[0].".pl ".$args[1];
      }
      else
      {
        $cmd = $daemon_prefix."_".$cmds[1].".pl";
      }
    }
    ($result, $response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "start_daemons") {
    ($result, $response) = startDaemons(\@daemons);
  }

  elsif ($key eq "start_helper_daemons") {
    ($result, $response) = startDaemons(\@helper_daemons);
  }

  elsif ($key eq "dfbsimulator") {
    $cmd = $current_binary_dir."/".$cfg{"DFB_SIM_BINARY"}." ".$cmds[1];
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "system") {
    ($result,$response) = Dada::mySystem($cmds[1], 0);  
  }

  elsif ($key eq "get_disk_info") {
    ($result,$response) = Dada::getDiskInfo($cfg{"CLIENT_RECORDING_DIR"});
  }

  elsif ($key eq "get_db_info") {
    ($result,$response) = Dada::getDBInfo(lc($cfg{"PROCESSING_DATA_BLOCK"}));
  }

  elsif ($key eq "get_alldb_info") {
    ($result,$response) = Dada::getAllDBInfo($cfg{"DATA_BLOCKS"});
  }

  elsif ($key eq "db_info") {
    ($result,$response) = Dada::getDBInfo(lc($cmds[1]));
  }

  elsif ($key eq "get_db_xfer_info") {
   ($result,$response) = Dada::getXferInfo();
  }

  elsif ($key eq "get_load_info") {
    ($result,$response) = Dada::getLoadInfo();
  }

  elsif ($key eq "set_udp_buffersize") {
    $cmd = "sudo /sbin/sysctl -w net.core.wmem_max=67108864";
    ($result,$response) = Dada::mySystem($cmd);

    if ($result eq "ok") {
      $cmd = "sudo /sbin/sysctl -w net.core.rmem_max=67108864";
      ($result,$response) = Dada::mySystem($cmd);
    }
  }

  elsif ($key eq "get_all_status") {
    my $subresult = "";
    my $subresponse = "";

    ($result,$subresponse) = Dada::getDiskInfo($cfg{"CLIENT_RECORDING_DIR"});
    $response = "DISK_INFO:".$subresponse."\n";
  
    ($subresult,$subresponse) = Dada::getAllDBInfo($cfg{"DATA_BLOCKS"});
    $response .= "DB_INFO:".$subresponse."\n";
    if ($subresult eq "fail") {
      $result = "fail";
    }

    ($subresult,$subresponse) = Dada::getLoadInfo();
    $response .= "LOAD_INFO:".$subresponse;
    if ($subresult eq "fail") {
      $result = "fail";
    }
  }

  elsif ($key eq "get_status") {

    my $subresult = "";
    my $subresponse = "";

    $response = $raw_disk_response.";;;";
    $response .= $db_status.";;;";
    $response .= $load_response.";;;";
    $response .= $unproc_files_response.";;;";
    $response .= $temp_response;

    if (($raw_disk_result ne "ok") || (($db_result ne "ok") && ($db_result ne "na"))|| 
        ($load_result ne "ok") || ($unproc_files_result ne "ok") ||
        ($temp_result ne "ok")) {
      $result = "fail";
    }
  }

  elsif ($key eq "stop_master_script") {
    $quit_daemon = 1;
  }

  else {
    $result = "fail";
    $response = "Unrecognized command ".$string;
  } 

  Dada::logMsg(3, $dl, "handleCommand() ".$result." ".$response);

  return ($result,$response);

}

#   
# stops the specified client daemon, optionally kills it after a period
#
sub stopDaemon($;$) {
    
  (my $daemon, my $timeout=10) = @_;
    
  my $pid_file  = $control_dir."/".$daemon.".pid";
  my $quit_file = $control_dir."/".$daemon.".quit";
  my $script    = $daemon_prefix."_".$daemon.".pl";
  my $ps_cmd    = "ps auxwww | grep '".$script."' | grep perl | grep -v grep";
  my $cmd = "";
    
  system("touch ".$quit_file);
    
  my $counter = $timeout;
  my $running = 1;

  while ($running && ($counter > 0)) {
    `$ps_cmd`;
    if ($? == 0) {
      Dada::logMsg(0, $dl, "daemon ".$daemon." still running");
    } else {
      $running = 0;
    }
    $counter--;
    sleep(1);
  }

  my $result = "ok";
  my $response = "daemon exited";
  
  # If the daemon is still running after the timeout, kill it
  if ($running) {
    ($result, $response) = Dada::killProcess("^perl.*".$script);
    $response = "daemon had to be killed";
    $result = "fail";
  }
  
  if (unlink($quit_file) != 1) {
    $response = "Could not unlink the quit command file ".$quit_file;
    $result = "fail";
    Dada::logMsg(0, $dl, "Error: ".$response);
  } 

  if (-f $pid_file) {
    if (unlink($pid_file) != 1) {
      $result = "fail";
      $response .= ", could not unlink pid file: ".$pid_file;
    }
  }
    
  return ($result, $response);
}


sub stopDaemons(\@) {

  my ($ref) = @_;
  my @ds = @$ref;

  my $threshold = 20;
  my $all_stopped = 0;
  my $quit_file = "";
  my $d = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $script = "";

  # Touch the quit files for each daemon
  foreach $d (@ds) {
    $quit_file = $control_dir."/".$d.".quit";
    $cmd = "touch ".$quit_file;
    system($cmd);
  }

  while ((!$all_stopped) && ($threshold > 0)) {

    $all_stopped = 1;
    foreach $d (@ds) {
      $script = $daemon_prefix."_".$d.".pl";
      $cmd = "ps aux | grep ".$cfg{"USER"}." | grep '".$script."' | grep perl | grep -v grep";
      `$cmd`;
      if ($? == 0) {
        Dada::logMsg(1, $dl, $d." is still running");
        $all_stopped = 0;
        if ($threshold < 10) {
          ($result, $response) = Dada::killProcess("^perl.*".$script);
        }
      }
    }
    $threshold--;
    sleep(1);
  }

  # Clean up the quit files
  foreach $d (@ds) {
    $quit_file = $control_dir."/".$d.".quit";
    unlink($quit_file);
  }

  # If we had to resort to a "kill", send an warning message back
  if (($threshold > 0) && ($threshold < 10)) {
    $result = "ok";
    $response = "KILL signal required to terminate some daemons";

  } elsif ($threshold <= 0) {
    $result = "fail";
    $response = "KILL signal did not terminate all daemons";

  } else {
    $result = "ok";
    $response = "Daemons exited correctly";
  }

  return ($result, $response);

}


sub startDaemons(\@) {

  my ($ref) = @_;
  my @ds = @$ref;

  my $d = ""; 
  my $cmd = "";
  my $result = "ok";
  my $response = "";
  my $daemon_result = "";
  my $daemon_response = "";

  foreach $d (@ds) {
    $cmd = $daemon_prefix."_".$d.".pl";
    Dada::logMsg(2, $dl, "Starting daemon: ".$cmd);
    ($daemon_result, $daemon_response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "Result: ".$daemon_result.":".$daemon_response);

     if ($daemon_result eq "fail") {
       $result = "fail";
       $response .= $daemon_response;
    }
  }

  return ($result, $response);

}

sub getDaemonInfo(\@) {

  my ($ref) = @_;
  my @ds = @$ref;

  my $d = "";
  my $cmd;
  my %array = ();
  my $i = 0;

  foreach $d (@ds) {

    # Check to see if the process is running
    $cmd = "ps aux | grep ".$daemon_prefix."_".$d.".pl | grep -v grep > /dev/null";
    Dada::logMsg(2, $dl, "getDaemonInfo: ".$cmd);
    `$cmd`;
    if ($? == 0) {
      $array{$d} = 1;
    } else {
      $array{$d} = 0;
    }

    # check to see if the PID file exists
    if (-f $control_dir."/".$d.".pid") {
      $array{$d}++;
    }
  }

  # If custom binaries should be running on this host, check them
  for ($i=0; $i<=$#binaries; $i++) {
    $b = $binaries[$i];
    $cmd = "pgrep ".$b." > /dev/null";
    Dada::logMsg(2, $dl, $cmd);
    `$cmd`;
    if ($? == 0) {
      $array{$b} = 2;
    } else {
      $array{$b} = 0;
    }
  }

  my $result = "ok";
  my $response = "";

  my @keys = sort (keys %array);
  for ($i=0; $i<=$#keys; $i++) {
    if ($array{$keys[$i]} != 2) {
      $result = "fail";
    }
    $response .= $keys[$i]." ".$array{$keys[$i]}.",";
  }

  return ($result, $response);

}


###############################################################################
#
# monitor the disk capacity and unprocessed files
#
sub diskThread() {

  my $result = "";
  my $response = "";
  my $sleep_time = 5;
  my $sleep_counter = 0;

  Dada::logMsg(1, $dl, "diskThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      ($result,$response) = Dada::getRawDisk($cfg{"CLIENT_RECORDING_DIR"});
      Dada::logMsg(3, $dl, "diskThread getRawDisk(".$cfg{"CLIENT_RECORDING_DIR"}.") = ".$result." ".$response);
      $raw_disk_response = $response;

      ($result,$response) = Dada::getUnprocessedFiles($cfg{"CLIENT_RECORDING_DIR"});
      $unproc_files_response = $response;

      Dada::logMsg(2, $dl, "diskThread: raw_disk=".$raw_disk_response.", unproc_files=".$unproc_files_response);
    }
  }

  Dada::logMsg(1, $dl, "diskThread: exiting");

}

###############################################################################
#
# monitor the client daemons
#
sub daemonsThread() {

  my $result = "";
  my $response = "";
  my $sleep_time = 5;
  my $sleep_counter = 0;
  my @ds = ();
  my @keys = ();
  my $i = 0;
  my %running = ();
  my $d = "";
  my $cmd = "";
  my $xml = "";

  Dada::logMsg(1, $dl, "daemonsThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      %running = ();

      foreach $d (@daemons) {

        $cmd = "ps aux | grep ".$daemon_prefix."_".$d.".pl | grep perl | grep -v grep > /dev/null";
        Dada::logMsg(3, $dl, "daemonsThread: ".$cmd);
        `$cmd`;
        if ($? == 0) {
          $running{$d} = 1;
        } else {
          $running{$d} = 0;
        }

        # check to see if the PID file exists
        if (-f $control_dir."/".$d.".pid") {
          $running{$d}++;
        }
      }

      # If custom binaries should be running on this host, check them e.g. PWC_BINARY
      for ($i=0; $i<=$#binaries; $i++) {
        $b = $binaries[$i];
        $cmd = "pgrep ".$b." > /dev/null";
        Dada::logMsg(2, $dl, $cmd);
        `$cmd`;
        if ($? == 0) {
          $running{$b} = 2;
        } else {
          $running{$b} = 0;
        }
      }

      # check if helper daemons are running on this host
      foreach $d (@helper_daemons) 
      {
         $cmd = "ps aux | grep ".$daemon_prefix."_".$d.".pl | grep perl | grep -v grep > /dev/null";
        Dada::logMsg(3, $dl, "daemonsThread: ".$cmd);
        `$cmd`;
        if ($? == 0) {
          $running{$d} = 1;
        } else {
          $running{$d} = 0;
        }

        # check to see if the PID file exists
        if (-f $control_dir."/".$d.".pid") {
          $running{$d}++;
        }
      }

      $result = "ok";
      $response = "";

      #$xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml  = "<daemon_info>";
      $xml .=   "<host>".$host."</host>";
      $xml .=   "<".$daemon_name.">2</".$daemon_name.">";

      # parse the results into the response strings
      @keys = sort (keys %running);
      for ($i=0; $i<=$#keys; $i++) {
        if ($running{$keys[$i]} != 2) {
          $result = "fail";
        }
        $response .= $keys[$i]." ".$running{$keys[$i]}.",";
        if ($keys[$i] eq $cfg{"PWC_BINARY"}) {
          $xml .= "<pwcs>".$running{$keys[$i]}."</pwcs>";
        }
        else {
          $xml .= "<".$keys[$i].">".$running{$keys[$i]}."</".$keys[$i].">";
        }
      }
    
      # if this client has a datablock
      if ($db_result ne "na") {

        @keys = split(/\n/, $db_response);
        for ($i=0; $i<=$#keys; $i++) 
        {
          my @bits = split(/ /, $keys[$i]);
          if ($bits[1] eq "ok") {
            $xml .= "<buffer_".$bits[0].">2</buffer_".$bits[0].">";
          } else {
            $xml .= "<buffer_".$bits[0].">0</buffer_".$bits[0].">";
          }
        }
      }

      $xml .=  "</daemon_info>";
  
      $daemons_result = $result;
      $daemons_response = $response;
      $daemons_response_xml = $xml;

      Dada::logMsg(2, $dl, "daemonsThread: ".$daemons_result." ".$daemons_response);

    }
  }

  Dada::logMsg(1, $dl, "daemonsThread: exiting");

}

###############################################################################
#
# monitor the data blocks (PWC only)
#
sub dbThread() {

  my $result = "";
  my $response = "";
  my $total_result = "";
  my $total_response = "";
  my $sleep_time = 2;
  my $sleep_counter = 0;
  my $i = 0; 
  my $blocks_total = 0;
  my $blocks_full = 0;
  my @bits = ();

  Dada::logMsg(1, $dl, "dbThread: starting [".$sleep_time." polling]");
 
  for ($i=0; $i<=$#dbs; $i++) {
    $dbs[$i] = lc($dbs[$i]);
  }

  if ($#dbs > -1) {

    while (!$quit_daemon) {

      # sleep 
      if ($sleep_counter > 0) {
        sleep(1);
        $sleep_counter--;

      # The time has come to check the status warnings
      } else {

        $sleep_counter = $sleep_time;

        $total_result = "ok";
        $total_response = "";

        $blocks_total = 0;
        $blocks_full = 0;

        for ($i=0; $i<=$#dbs; $i++) {

          Dada::logMsg(3, $dl, "dbThread: getAllDBInfo(".$dbs[$i].")");
          ($result,$response) = Dada::getAllDBInfo($dbs[$i]);
          Dada::logMsg(3, $dl, "dbThread: ".$result." ".$response);
          if ($i < $#dbs) {
            $total_response .= $dbs[$i]." ".$result." ".$response."\n";
          } else {
            $total_response .= $dbs[$i]." ".$result." ".$response;
          }
          if ($result ne "ok") {
            $total_result = "fail";
          } else {
            @bits = split(/ /, $response);
            $blocks_total += $bits[0];
            $blocks_full += $bits[1];
            Dada::logMsg(3, $dl, "dbThread: blocks_total=".$blocks_total.", blocks_full=".$blocks_full.", response=".$response);
          }
        }
        $db_result = $total_result;
        $db_response = $total_response;
        $db_status = $blocks_total." ".$blocks_full;
        Dada::logMsg(2, $dl, "dbThread: ".$db_result." ".$db_response);
      }
    }
  } 

  Dada::logMsg(1, $dl, "dbThread: exiting");

}

###############################################################################
#
# monitor the load
#
sub loadThread() {

  my $result = "";
  my $response = "";
  my $sleep_time = 2;
  my $sleep_counter = 0;

  Dada::logMsg(1, $dl, "loadThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      ($result,$response) = Dada::getLoadInfo();
      $load_result = $result;
      $load_response = $response;

      Dada::logMsg(2, $dl, "loadThread: ".$load_result." ".$load_response);

    }
  }

  Dada::logMsg(1, $dl, "loadThread: exiting");

}

###############################################################################
#
# monitor the system tempreature
#
sub tempThread() {

  my $result = "";
  my $response = "";
  my $sleep_time = 5;
  my $sleep_counter = 0;

  Dada::logMsg(1, $dl, "tempThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      ($result,$response) = Dada::getTempInfo();
      $temp_result = $result;
      $temp_response = $response;

      Dada::logMsg(2, $dl, "tempThread: ".$temp_result." ".$temp_response);

    }
  }

  Dada::logMsg(1, $dl, "tempThread: exiting");

}


###############################################################################
#
# handle file based Quit command
#
sub controlThread($$) {

  my ($quit_file, $pid_file) = @_;
  Dada::logMsg(1, $dl, "controlThread: starting");
  
  my $cmd = "";
  my $result = "";
  my $response = "";
  
  while ((!$quit_daemon) && (!(-f $quit_file))) {
    sleep(1);
  } 
  
  $quit_daemon = 1;
  
  if ( -f $pid_file) {
    Dada::logMsg(2, $dl, "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsg(1, $dl, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, $dl, "controlThread: exiting");

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
  
  if ($sock) {
    close($sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}


#
# Test to ensure all module variables are set before main
#
sub good() {

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  if ( $daemon_name eq "") {
    return ("fail", "Error: a package variable missing [daemon_name]");
  }

  # open a socket for listening connections
  $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  if (!$sock) {
    return ("fail", "Could not create listening socket: ".$host.":".$port);
  }
  Dada::logMsg(1, $dl, "Opened socket on ".$host.":".$port);

  return ("ok", "");

}

END { }

1;  # return value from file
