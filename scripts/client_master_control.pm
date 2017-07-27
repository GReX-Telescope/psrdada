package Dada::client_master_control;

use lib $ENV{"DADA_ROOT"}."/bin";

$ENV{"THREADS_IP_MASK"} = 1;

use strict;
use warnings;
use threads;
use threads::shared;
use IO::Socket;
use IO::Select;
use Net::hostent;
use File::Basename;
use Thread::Queue;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main &computeDBSize);
  %EXPORT_TAGS = ( );     # eg: TAG => [ qw!name1 name2! ],

  # your exported package globals go here,
  # as well as any optionally exported functions
  @EXPORT_OK   = qw($dl $daemon_name %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $daemon_name;
our %pwcs;
our $num_pwc;
our $primary_db;
our $daemon_prefix;
our $control_dir;
our $log_dir;
our $host : shared;
our $user : shared;
our %cfg;
our %site_cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $kill_daemon : shared;
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
our $db_response_xml : shared;
our $load_result : shared;
our $load_response : shared;
our $temp_result : shared;
our $temp_response : shared;

#
# initialize package globals
#
$dl = 1; 
$daemon_name = 0;
%pwcs = ();
$num_pwc = 0;
$primary_db = "";
$daemon_prefix = "";
$control_dir = "";
$log_dir = "";
$host = "";
$user = "";
%cfg = ();
%site_cfg = ();

#
# initialize other variables
#
$quit_daemon = 0;
$kill_daemon = 0;
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
$db_response_xml = "";
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

  my $cmd = "";
  my $result = "";
  my $response = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good();
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  %site_cfg       = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);
  my $log_file       = $log_dir."/".$daemon_name.".log";;
  my $pid_file       = $control_dir."/".$daemon_name.".pid";
  my $quit_file      = $control_dir."/".$daemon_name.".quit";
  my $archive_dir    = $cfg{"CLIENT_ARCHIVE_DIR"};   # hi res archive storage
  my $results_dir    = $cfg{"CLIENT_RESULTS_DIR"};   # dspsr output directory

  # determine type of each daemon
  my ($i, $d, $pwc);
  foreach $pwc ( keys %pwcs )
  {
    foreach $d ( @{ $pwcs{$pwc}{"daemons"} } )
    {
      Dada::logMsg(2, $dl, "main: pwc=".$pwc." daemon=".$d);
    }
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
  my $n_threads = 6;
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

  # launch a thread that will kill this daemon after 10 seconds
  $kill_daemon = 1;
  my $kill_thread = threads->new(\&killThread, 10);

  for ($i=0; $i<$n_threads; $i++) {
    $tids[$i]->join();
    Dada::logMsg(0, $dl, "worker thread ".$i." of ".$n_threads." joined");
  }
  $disk_thread_id->join();
  Dada::logMsg(0, $dl, "disk_thread joined");
  $daemons_thread_id->join();
  Dada::logMsg(0, $dl, "daemons_thread joined");
  $db_thread_id->join();
  Dada::logMsg(0, $dl, "db_thread joined");
  $load_thread_id->join();
  Dada::logMsg(0, $dl, "load_thread joined");
  $temp_thread_id->join();
  Dada::logMsg(0, $dl, "temp_thread joined");
  $control_thread_id->join();
  Dada::logMsg(0, $dl, "control_thread joined");

  # If we have come this far, cancel the killThread
  $kill_daemon = 0;
  $kill_thread->join();

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



sub handleCommand($) 
{
  (my $string) = @_;

  Dada::logMsg(2, $dl, "handleCommand: string= '".$string."'");

  # each command consists of a key and optional pwc and args
  my $key = "";
  my $pwc = -1;
  my $args = "";

  my $part = "";
  my @parts = split(/&/, $string);
  my $k = "";
  my $v = "";

  # if the command is a simple 1 word command, accept it
  if ((!($string =~ m/=/)) && (!($string =~ m/&/)))
  {
    $key = $string;
  } 
  else 
  {
    foreach $part (@parts)
    {
      ($k, $v) = split(/=/, $part, 2);
      if ($k eq "cmd") {
        $key = $v;
      }
      if ($k eq "pwc") {
        $pwc = int($v);
      }
      if ($k eq "args") {
        $args = $v;
      }
    }
  }
  
  if ($key eq "") 
  {
    return ("fail", "no command specified");
  }
  
  Dada::logMsg(2, $dl, "handleCommand: key=".$key." pwc=".$pwc." args=".$args);

  my $cmd = "";
  my $result = "ok";
  my $response = "";

  if ($key eq "stop_dfbs") 
  {
    $cmd = "killall -KILL ".$cfg{"DFB_SIM_BINARY"};
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "kill_process") 
  {
    ($result,$response) = Dada::killProcess($args);
  }

  elsif ($key eq  "start_bin") 
  {
    ($result,$response) = Dada::mySystem($args);  
  }

  elsif ($key eq "destroy_db") 
  {
    my @db_ids = ($args); 
    ($result, $response) = destroyDBs($pwc, \@db_ids);
  }

  elsif ($key eq "destroy_dbs") 
  {
    my @db_ids = split(/\s+/, $cfg{"DATA_BLOCK_IDS"});
    ($result, $response) = destroyDBs($pwc, \@db_ids);
  }

  elsif ($key eq "init_db")
  {
    my @db_ids = ($args);
    ($result, $response) = initDBs($pwc, \@db_ids);
  }

  # initialize all data blocks for all pwcs
  elsif ($key eq "init_dbs") 
  {
    my @db_ids = split(/\s+/, $cfg{"DATA_BLOCK_IDS"});
    ($result, $response) = initDBs($pwc, \@db_ids);
  }

  elsif ($key eq "stop_daemon") 
  {
    if ($args  =~ m/_master_control/) 
    {
      Dada::logMsg(0, $dl, "stopping master control [".$args."]");
      $quit_daemon = 1;
    }
    else
    {
      # dont kill persistent server daemons ever, allow them to stop on their own
      if (defined ($cfg{"SERVER_DAEMONS_PERSIST"}) && ($cfg{"SERVER_DAEMONS_PERSIST"} =~ m/$args/)) {
        Dada::logMsg(1, $dl, "special case for persistent server daemon");
        ($result,$response) = stopDaemon($pwc, $args, 5, 0);
      } else {
        ($result,$response) = stopDaemon($pwc, $args, 30);
      }
    }
  }

  elsif ($key eq "stop_daemons") 
  {
    Dada::logMsg(1, $dl, "handleCommand: stopDaemons(".$pwc.")");
    ($result,$response) = stopDaemons($pwc);
  }

  elsif ($key eq "stop_helper_daemons") {
    ($result,$response) = stopDaemons($pwc);
  }

  elsif ($key eq "daemon_info") 
  {
    $result = $daemons_result;
    $response = $daemons_response;
  }

  elsif ($key eq "daemon_info_xml") {
    $response = $daemons_response_xml;
  }

  elsif ($key eq "start_daemon")
  {
    if (length($args) < 1)
    {
      $result = "fail";
      $response = "argument required";
    }
    else
    {
      my $custom_args = "";
      my @custom_daemons = ();

      # if there are space separate arguements
      if ($args =~ m/ /)
      {
        my $d = "";
        ($d, $custom_args) = split(/ /, $args, 2);
        $args = $d;
      }

      Dada::logMsg(2, $dl, "handleCommand: startDamemons(".$pwc.", ".$args.", ".$custom_args.")");
      ($result, $response) = startDaemons($pwc, $args, $custom_args);
      Dada::logMsg(3, $dl, "handleCommand() ".$result." ".$response);
    }
  }

  elsif ($key eq "start_daemons") 
  {
    Dada::logMsg(2, $dl, "handleCommand: startDamemons(".$pwc.", daemons, )");
    ($result, $response) = startDaemons($pwc, "daemons", "");
  }

  elsif ($key eq "start_helper_daemons") 
  {
    Dada::logMsg(2, $dl, "handleCommand: startDamemons(".$pwc.", helper_daemons, )");
    ($result, $response) = startDaemons($pwc, "helper_daemons", "");
  }

  elsif ($key eq "dfbsimulator") 
  {
    $cmd = $cfg{"DFB_SIM_BINARY"}." ".$args;
    ($result,$response) = Dada::mySystem($cmd);
  }

  elsif ($key eq "system") 
  {
    ($result,$response) = Dada::mySystem($args);  
  }

  elsif ($key eq "get_disk_info") 
  {
    ($result,$response) = Dada::getDiskInfo($log_dir);
  }

  elsif ($key eq "get_db_info") 
  {
    my $db_keys = getDBList($cfg{"PROCESSING_DATA_BLOCK"});
    ($result,$response) = Dada::getDBInfo($db_keys);
  }

  elsif ($key eq "get_alldb_info") 
  {
    my $db_keys = getDBList($cfg{"DATA_BLOCK_IDS"});
    ($result,$response) = Dada::getAllDBInfo($db_keys);
  }

  elsif ($key eq "db_info") 
  {
    my $db_keys = getDBList($args);
    ($result,$response) = Dada::getDBInfo($db_keys);
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

  elsif ($key eq "get_all_status") 
  {
    my $subresult = "";
    my $subresponse = "";

    ($result,$subresponse) = Dada::getDiskInfo($log_dir);
    $response = "DISK_INFO:".$subresponse."\n";
  
    my $db_keys = getDBList($cfg{"DATA_BLOCK_IDS"});
    ($subresult,$subresponse) = Dada::getAllDBInfo($db_keys);
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
  elsif ($key eq "get_status") 
  {
    my $xml = "";
    $xml = "<node_status host='".$host."'>";
    $xml .= $raw_disk_response;
    $xml .= $db_response_xml;
    $xml .= $load_response;
    $xml .= $temp_response;
    $xml .= "</node_status>";

    if (($raw_disk_result ne "ok") || (($db_result ne "ok") && ($db_result ne "na"))|| 
        ($load_result ne "ok") || ($unproc_files_result ne "ok") ||
        ($temp_result ne "ok")) {
      $result = "fail";
    }
    $response = $xml;
  }

  elsif ($key eq "stop_master_script") 
  {
    $quit_daemon = 1;
  }

  else {
    $result = "fail";
    $response = "Unrecognized command ".$string;
  } 

  Dada::logMsg(3, $dl, "handleCommand() ".$result." ".$response);

  return ($result,$response);

}

###############################################################################
#
# return a string of datablock keys based on the DB IDs
#
sub getDBList($) 
{
  my ($db_id_string) = @_;

  my $db_id = "";
  my @db_ids = split(/\s+/, $db_id_string);
  my $db_keys = "";
  my $db_prefix = $cfg{"DATA_BLOCK_PREFIX"};
  my $key = "";
  my $pwc = "";

  foreach $pwc (keys %pwcs)
  {
    foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } )
    {
      # $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc, $num_pwc, $db_id);
      $key = $pwcs{$pwc}{"dbs"}{$db_id}{"key"};
      if ($db_keys == "")
      {
        $db_keys = $key;
      } 
      else
      {
        $db_keys .= " ".$key;
      }
    }
  }
  return $db_keys;
}


###############################################################################
#
# init the list of specified DBs for all PWCs running on this host
#
sub initDBs($\@) 
{
  (my $pwc_to_init, my $ref) = @_;
  my @db_ids = @$ref;

  my $pwc = "";
  my $db_id = "";
  my $id = 0;
  my $valid_db_id = 0;

  my $result = "ok";
  my $response = "";

  foreach $pwc (keys %pwcs)
  {
    if (($pwc_to_init >= 0) && ($pwc_to_init != $pwc)) 
    {
      Dada::logMsg(2, $dl, "initDBs: skipping ".$pwc." as not in list");
      next;
    }

    # get the list of configure DB's for this PWC
    foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } )
    {
      $valid_db_id = 0; 
      foreach $id ( @db_ids) 
      {
        if ($db_id == $id)
        {
          $valid_db_id = 1;
        }
      }
  
      if ($valid_db_id)
      {
        my $key   = $pwcs{$pwc}{"dbs"}{$db_id}{"key"};
        my $bufsz = $pwcs{$pwc}{"dbs"}{$db_id}{"bufsz"};
        my $nbufs = $pwcs{$pwc}{"dbs"}{$db_id}{"nbufs"};
        my $nread = $pwcs{$pwc}{"dbs"}{$db_id}{"nread"};
        my $page  = $pwcs{$pwc}{"dbs"}{$db_id}{"page"};
        my $numa  = $pwcs{$pwc}{"dbs"}{$db_id}{"numa"};

        # check if the DB already exists
        my $cmd = "ipcs | grep 0x0000".$key." | wc -l";
        Dada::logMsg(1, $dl, "initDBs: pwc=".$pwc." cmd=".$cmd);
        my ($tmp_result,$tmp_response) = Dada::mySystem($cmd);
        Dada::logMsg(1, $dl, "initDBs: ".$tmp_result." ".$tmp_response);
        if (($tmp_result eq "ok") && ($tmp_response eq "0"))
        {
          $cmd = "dada_db -a ".$site_cfg{"HDR_SIZE"}." -k ".lc($key)." -b ".$bufsz." -n ".$nbufs." -r ".$nread." -l";
          if ($page eq "true")
          {
            $cmd .= " -p";
          }

          if ($numa ne "-1")
          {
            $cmd .= " -c ".$cfg{"BLOCK_NUMA_".$db_id};
          }

          Dada::logMsg(1, $dl, "initDBs: pwc=".$pwc." cmd=".$cmd);
          my ($tmp_result,$tmp_response) = Dada::mySystem($cmd);
          Dada::logMsg(2, $dl, "initDBs: ".$tmp_result." ".$tmp_response);

          if ($tmp_result eq "fail") {
            $result = "fail";
          }
          $response = $response.$tmp_response."<BR>";
        }
        else
        {
          $result = "fail";
          $response = $response." DB ".$key." already existed\n";   
        }
      }
    }
  }

  if ($response eq "") 
  {
    $result = "ok";
    $response = "No matching DBs found";
  }
  return ($result, $response);
}

################################################################################
#
# compute the size of data block based on the parameters of the signal
#
sub computeDBSize($$$$$$$)
{
  my ($nsamp, $nchan, $npol, $ndim, $nbit, $nant, $nbeam) = @_;

  print "nsamp=".$nsamp." nchan=".$nchan." npol=".$npol." ndim=".$ndim." nant=".$nant." nbeam=".$nbeam." nbit=".$nbit."\n";

  my $size = ($nsamp * $nchan * $npol * $ndim * $nant * $nbeam * $nbit) / 8;

  return $size;
}


###############################################################################
#
# destroy the list of specified DBs for all PWCs running on this host
#
sub destroyDBs($\@) 
{
  (my $pwc_to_destroy, my $ref) = @_;
  my @db_ids = @$ref;

  my $pwc = "";
  my $db_id = "";
  my $id = 0;
  my $valid_db_id = 0;

  my $result = "ok";
  my $response = "";

  foreach $pwc (keys %pwcs) 
  {
    if (($pwc_to_destroy >= 0) && ($pwc_to_destroy != $pwc)) {
      Dada::logMsg(2, $dl, "destroyDBs: skipping ".$pwc." as not in list");
      next;
    }
    foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } ) 
    {
      $valid_db_id = 0;
      foreach $id (@db_ids)
      {
        # only can have DB ids between 0 and 7
        if ($db_id == $id)
        {
          $valid_db_id = 1;
        }
      }

      if ($valid_db_id) 
      {
        my $key = $pwcs{$pwc}{"dbs"}{$db_id}{"key"};
        my $cmd = "dada_db -k ".lc($key)." -d";
        Dada::logMsg(2, $dl, "destroyDBs: pwc=".$pwc." cmd=".$cmd);
        my ($tmp_result,$tmp_response) = Dada::mySystem($cmd);
        if ($tmp_result eq "fail") {
          $result = "fail";
        }
        $response = $response.$tmp_response."<BR>";
      }
    }
  } 
  if ($response eq "") 
  {
    $result = "ok";
    $response = "No matching DBs found";
  }
  return ($result, $response);
}



###############################################################################
#   
# stops the specified client daemon, optionally kills it after a period
#
sub stopDaemon($$;$$) 
{
  (my $pwc_to_stop, my $daemon, my $timeout=10, my $kill_after_timeout=1) = @_;
  
  # by default affect all scripts on this host  
  my $pid_file  = $control_dir."/".$daemon.".pid";
  my $quit_file = $control_dir."/".$daemon.".quit";

  my $script = "";
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $pgrep = "";

  # determine the type of daemon (perl, python or unknown)
  ($result, $response) = getDaemonName($daemon);
  if ($result ne "ok")
  {
    return ("fail", $response);
  }
  $script = $response;

  # determine the pgrep command to run for this script
  if ($script =~ m/.pl$/)
  {
    $pgrep = "^perl.*".$script;
  }
  elsif ($script =~ m/.py$/)
  {
    $pgrep = "^.*python.*".$script;
  }

  # if we have specified a PWC to affect, append it as the first command line arguement
  if ($pwc_to_stop >= 0)
  {
    $script    .= " ".$pwc_to_stop;
    $pid_file  = $control_dir."/".$daemon."_".$pwc_to_stop.".pid";
    $quit_file = $control_dir."/".$daemon."_".$pwc_to_stop.".quit";
  }

  system("touch ".$quit_file);
    
  my $counter = $timeout;
  my $running = 1;
  my $ps_cmd = "pgrep -u ".$user." -l -f '".$pgrep."' | grep -v 'sh -c'";

  while ($running && ($counter > 0)) 
  {
    Dada::logMsg(1, $dl, "stopDaemon: ".$ps_cmd);
    ($result, $response) = Dada::myShell($ps_cmd);
    Dada::logMsg(1, $dl, "stopDaemon: ".$result." ".$response);
    if ($result eq "ok")
    {
      Dada::logMsg(0, $dl, "daemon ".$daemon." still running: ".$response);
      sleep(1);
    } 
    else 
    {
      $running = 0;
    }
    $counter--;
  }

  $result = "ok";
  $response = "daemon exited";

  # if the daemon is not running, but the PID file exists, delete it
  if (!$running && -f $pid_file) 
  {
    if (unlink($pid_file) != 1) 
    {
      $result = "fail";
      $response .= ", could not unlink pid file: ".$pid_file;
    }
  } 
  
  # If the daemon is still running after the timeout, kill it
  if ($running && $kill_after_timeout) 
  {
    ($result, $response) = Dada::killProcess($pgrep, $user);
    $response = "daemon had to be killed";
    $result = "fail";
  }
  
  if (unlink($quit_file) != 1)
  {
    $response = "Could not unlink the quit command file ".$quit_file;
    $result = "fail";
    Dada::logMsg(0, $dl, "Error: ".$response);
  } 

  if (-f $pid_file && $kill_after_timeout) 
  {
    if (unlink($pid_file) != 1)
    {
      $result = "fail";
      $response .= ", could not unlink pid file: ".$pid_file;
    }
  }
    
  return ($result, $response);
}

sub stopDaemons($) {

  (my $pwc_to_stop) = @_;

  my $threshold = 20;
  my $all_stopped = 0;

  my $d_add = "";
  my $p_add = "";
  my ($quit_file, $d, $cmd, $result, $response, $script, $pgrep, $pwc);
  my @pwcs_to_stop = ();

  if ($pwc_to_stop >= 0) 
  {
    $d_add = "_".$pwc_to_stop;
    $p_add = " ".$pwc_to_stop;
    push @pwcs_to_stop, $pwc_to_stop;
  }
  else
  {
    foreach $pwc (keys %pwcs)
    {
      push @pwcs_to_stop, $pwc;
    }
  }

  # give asynchronous message to each daemon to quit
  foreach $pwc (@pwcs_to_stop)
  {
    foreach $d ( @{ $pwcs{$pwc}{"daemons"} } )
    {
      $quit_file = $control_dir."/".$d.$d_add.".quit";
      $cmd = "touch ".$quit_file;
      system($cmd);
    }
  }

  # daemon can be perl or python
  while ((!$all_stopped) && ($threshold > 0)) 
  {
    $all_stopped = 1;

    foreach $pwc (@pwcs_to_stop)
    {
      foreach $d ( @{ $pwcs{$pwc}{"daemons"} } )
      {
        # determine the type of daemon (perl, python or unknown)
        ($result, $response) = getDaemonName($d);
        if ($result ne "ok")
        {
          return ("fail", $response);
        }
        $script = $response;

        # determine the pgrep command to run for this script
        if ($script =~ m/.pl$/)
        {
          $pgrep = "^perl.*".$script;
        }
        elsif ($script =~ m/.py$/)
        {
          $pgrep = "^.*python.*".$script;
        }
        else
        {
          return ("fail", "could not identify suffix for ".$d)
        }

        $cmd = "pgrep -u ".$user." -l -f '".$pgrep.$p_add."' | grep -v 'sh -c'";
        ($result, $response) = Dada::myShell($cmd);
        if ($result eq "ok")
        {
          Dada::logMsg(1, $dl, $d.$d_add." is still running");
          $all_stopped = 0;
          if ($threshold < 10) 
          {
            ($result, $response) = Dada::killProcess($pgrep, $user);
          }
        }
      }
    }

    if (!$all_stopped)
    {
      $threshold--;
      sleep(1);
    }
  }

  # clean up the quit files
  foreach $pwc (@pwcs_to_stop)
  {
    foreach $d ( @{ $pwcs{$pwc}{"daemons"} } )
    {
      $quit_file = $control_dir."/".$d.$d_add.".quit";
      if (-f $quit_file)
      {
        unlink($quit_file);
      }
    }
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

sub startDaemons($$$) {

  (my $pwc_to_start, my $type, my $args) = @_;

  my $d = ""; 
  my $cmd = "";
  my $pwc = "";
  my $result = "ok";
  my $response = "";
  my $daemon_result = "";
  my $daemon_response = "";

  foreach $pwc (keys %pwcs)
  {
    if (($pwc_to_start >= 0) && ($pwc_to_start != $pwc)) 
    {
       Dada::logMsg(2, $dl, "startDaemons: skipping ".$pwc." as not in list");
       next;
    }
    # if this is a preconfigured daemon group
    if (exists( $pwcs{$pwc}{$type}))
    {
      foreach $d ( @{ $pwcs{$pwc}{$type} } )
      {
        # determine the type of daemon (perl, python or unknown)
        ($daemon_result, $daemon_response) = getDaemonName($d);
        if ($daemon_result eq "ok")
        {
          $cmd = $daemon_response." ".$pwc." ";
          Dada::logMsg(2, $dl, "startDaemons: ".$cmd);
          ($daemon_result, $daemon_response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "startDaemons: ".$daemon_result.":".$daemon_response);
        }
        else
        {
          $result = "fail";
          $response .= "could not find matching daemon";
        }
      }
    }
    else
    {
      foreach $d ( @{ $pwcs{$pwc}{"daemons"}} )
      {
        if ($d eq $type)
        {
          ($daemon_result, $daemon_response) = getDaemonName($type);
          if ($daemon_result eq "ok")
          {
            $cmd = $daemon_response." ".$pwc." ".$args;
            Dada::logMsg(1, $dl, "startDaemons: ".$cmd);
            ($daemon_result, $daemon_response) = Dada::mySystem($cmd);
            Dada::logMsg(1, $dl, "startDaemons: ".$daemon_result.":".$daemon_response);
          }
          else
          {
            $result = "fail";
            $response .= "could not find matching daemon";
          }
        }
      }
    }
  }

  return ($result, $response);

}

sub getDaemonName($)
{
  (my $d) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "ls -1 ".$cfg{"SCRIPTS_DIR"}."/".$daemon_prefix."_".$d.".p?";
  Dada::logMsg(2, $dl, "getDaemonName: ".$cmd);
  ($result, $response) = Dada::myShell($cmd);
  Dada::logMsg(3, $dl, "getDaemonName: ".$result." ".$response);
  if ($result eq "ok")
  {
    return ("ok", basename($response));
  }
  else
  {
    Dada::logMsg(1, $dl, "getDaemonName: could not identify script for ".$d);
    return ("fail", "could not identify script for ".$d);
  }

#  my @extensions = ("pl", "py");
#  my $ext;

#  foreach $ext ( @extensions)
#  {
    # determine the type of daemon (perl, or python)
#    $cmd = "ls -1 ".$cfg{"SCRIPTS_DIR"}."/".$daemon_prefix."_".$d.".".$ext;
#    Dada::logMsg(2, $dl, "getDaemonName: ".$cmd);
#    ($result, $response) = Dada::mySystem($cmd);
#    Dada::logMsg(3, $dl, "getDaemonName: ".$result." ".$response);
#    Dada::logMsg(1, $dl, "getDaemonName: ".$result." ".$response);
#    if ($result eq "ok")
#    {
#      return ("ok", basename($response));
#    }
#  }
#
#  return ("fail", "could not identify script for ".$d);
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
  my $xml = "";

  my $used = "";
  my $available = "";
  my $size = "";

  Dada::logMsg(1, $dl, "diskThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) 
  {
    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      ($result,$response) = Dada::getRawDisk($log_dir);
      Dada::logMsg(3, $dl, "diskThread getRawDisk(".$log_dir.") = ".$result." ".$response);
      if ($result eq "ok") 
      {
        ($size, $used, $available) = split(/ /,$response);
        $xml  = "<disk path='".$cfg{"CLIENT_RECORDING_DIR"}."' units='MB' size='".$size."' used='".$used."'>".$available."</disk>";
      }

      $raw_disk_response = $xml;

      Dada::logMsg(2, $dl, "diskThread: raw_disk=".$raw_disk_response);
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
  my $sleep_time = 2;
  my $sleep_counter = 0;
  my @keys = ();
  my $i = 0;
  my %running = ();
  my $pwc = "";
  my %pwc_running = ();
  my $d = "";
  my $cmd = "";
  my $xml = "";
  my $k = "";
  my $tag = "";
  my %pids = ();
  my $handle = 0;
  my $db_id = 0;
  my $db_key = "";
  my %pgreps = ();

  # determine pgrep command for each daemon
  foreach $pwc (keys %pwcs)
  {
    foreach $d ( @{ $pwcs{$pwc}{"daemons"} } )
    {
      # determine the type of daemon (perl or python)

      ($result, $response) = getDaemonName($d);
      Dada::logMsg(2, $dl, "daemonsThread: ".$d." -> ".$response." [".$result."]");
      if ($result eq "ok")
      {
        if ($response =~ m/.pl$/)
        {
          $pgreps{$d} = "^perl.*".$response;
          Dada::logMsg(2, $dl, "daemonsThread: pgreps{".$d."}=[".$pgreps{$d}."]");
        }
        elsif ($response=~ m/.py$/)
        {
          $pgreps{$d} = "^.*python.*".$response;
          Dada::logMsg(2, $dl, "daemonsThread: pgreps{".$d."}=[".$pgreps{$d}."]");
        }
        else
        {
          Dada::logMsg(1, $dl, "daemonsThread: could not match ".$response." to .py\$ or .pl\$");
        }
      }
      else
      {
        Dada::logMsg(1, $dl, "daemonsThread: getDaemonName failed: ".$response);
      }
    }
  }

  Dada::logMsg(1, $dl, "daemonsThread: starting [".$sleep_time." polling]");

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      %pwc_running = ();
      %running = ();
      %pids = ();

      foreach $pwc (keys %pwcs)
      {
        foreach $d ( @{ $pwcs{$pwc}{"daemons"}} )
        {
          if (exists($pgreps{$d}))
          {
            $cmd = "pgrep -u ".$user." -f -l '".$pgreps{$d}."'";
            # for clients, all daemons must be launched with a PWC_ID argument
            if ($pwc ne "server") 
            {
              $cmd = "pgrep -u ".$user." -f -l '".$pgreps{$d}." ".$pwc."'";
            }
            $cmd .= " | grep -v 'sh -c'";
            Dada::logMsg(2, $dl, "daemonsThread [daemon]: ".$cmd);
            ($result, $response) = Dada::myShell($cmd);
            Dada::logMsg(3, $dl, "daemonsThread [daemon]: ".$result." ".$response);
            if (($result eq "ok") && ($response ne "")) 
            {
              $pwc_running{$pwc}{$d} = 1;
            } 
            else 
            {
              Dada::logMsg(3, $dl, "daemonsThread [daemon] \$pwc_running{".$pwc."}{".$d."} = 0 result=".$result." response=".$response);
              $pwc_running{$pwc}{$d} = 0;
            }

            # check to see if the PID file exists
            if (($pwc ne "server") && (-f $control_dir."/".$d."_".$pwc.".pid"))
            {
              $pwc_running{$pwc}{$d} += 1;
            }
            if (($pwc eq "server") && (-f $control_dir."/".$d.".pid"))
            {
              $pwc_running{$pwc}{$d} += 1;
            }
          }
          else
          {
             Dada::logMsg(1, $dl, "daemonsThread [daemon]: [".$d."] did not exist in \$pgreps{}");
            $pwc_running{$pwc}{$d} = 0;
          }
        }
      }

      # check if helper daemons are running on this host
      foreach $pwc (keys %pwcs)
      {
        foreach $d ( @{$pwcs{$pwc}{"helper_daemons"}} )
        {
          $cmd = "pgrep -f '^perl.*".$daemon_prefix."_".$d.".pl ".$pwc."'";
          Dada::logMsg(3, $dl, "daemonsThread [helper]: ".$cmd);
          ($result, $response) = Dada::myShell($cmd);
          if (($result eq "ok") && ($response ne "")) {
            $pwc_running{$pwc}{$d} = 1;
          } else {
            $pwc_running{$pwc}{$d} = 0;
          }

          # check to see if the PID file exists
          if (-f $control_dir."/".$d.".pid") {
            $pwc_running{$pwc}{$d} += 1;
          }
        } 
      }

      $result = "ok";
      $response = "";

      $xml = "";

      # there is only ever 1 host and master control script
      $xml .=   "<host>".$host."</host>";
      $xml .=   "<".$daemon_name.">2</".$daemon_name.">";

      # for each host there can be multiple PWCs
      foreach $pwc (keys %pwcs)
      {
        if ($pwc ne "server") 
        {
          $xml .=   "<pwc id='".$pwc."' state='".$pwcs{$pwc}{"state"}."'>";
        }

        # add daemons and pwcs to the XML output
        foreach $k ( keys %{ $pwc_running{$pwc} } )
        {
          $tag = $k;
          if (defined($pids{$k})) 
          {
            $xml .= "<".$tag." pid='".$pids{$tag}."'>".$pwc_running{$pwc}{$k}."</".$tag.">";
          }
          else 
          {
            $xml .= "<".$tag.">".$pwc_running{$pwc}{$k}."</".$tag.">";
          }
        }

        # add datablocks to the XML output
        if ($db_result ne "na")
        {
          $db_id = "";
          foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } )
          {
            $db_key = lc($pwcs{$pwc}{"dbs"}{$db_id}{"key"});
            if ($db_response =~ m/$db_key:ok/)
            {
              $xml .= "<buffer_".$db_id.">2</buffer_".$db_id.">";
            }
            else
            {
              $xml .= "<buffer_".$db_id.">0</buffer_".$db_id.">";
            }
          }
        } 

        if ($pwc ne "server")
        {
          $xml .=   "</pwc>";
        }
      }

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
sub dbThread() 
{
  my $result = "";
  my $response = "";
  my $total_result = "";
  my $total_response = "";
  my $sleep_time = 5;
  my $sleep_counter = 0;
  my $i = 0; 
  my $blocks_total = 0;
  my $blocks_full = 0;
  my @bits = ();
  my @dbs_to_report = ();

  my $nblocks = "";
  my $nfull = "";

  Dada::logMsg(1, $dl, "dbThread: starting [".$sleep_time." polling]");
  my $pwc = "";
  my $db_id = "";
  my $key = "";
  my $xml = "";

  if ($primary_db ne "") 
  {
    push @dbs_to_report, lc($primary_db);
  }
  else 
  { 
    foreach $pwc (keys %pwcs) {
      foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } )
      {
        my $key = $pwcs{$pwc}{"dbs"}{$db_id}{"key"};
        push @dbs_to_report, lc($key);
      }
    }  
  }

  for ($i=0; $i<=$#dbs_to_report; $i++) 
  {
    Dada::logMsg(1, $dl, "dbThread: reporting on DB[".$i."] ".$dbs_to_report[$i]);
  }

  if ($#dbs_to_report > -1)
  {
    while (!$quit_daemon)
    {
      if ($sleep_counter > 0)
      {
        sleep(1);
        $sleep_counter--;
      }
      else
      {
        $sleep_counter = $sleep_time;

        $xml = "";
        $result = "";
        $response = "";

        foreach $pwc (keys %pwcs)
        {
          foreach $db_id ( keys %{ $pwcs{$pwc}{"dbs"} } )
          {
            $key = $pwcs{$pwc}{"dbs"}{$db_id}{"key"};
            ($result, $nblocks, $nfull) = Dada::getDBStatus($key);
            $response .= $key.":".$result." ";
            $xml .= "<datablock pwc_id='".$pwc."' db_id='".$db_id."' key='".$key."' size='".$nblocks."'>".$nfull."</datablock>";
            if (($nblocks > 0) && (($nblocks - $nfull) <= 1))
            {
              Dada::logMsg(1, $dl, "dbThread: pwc_id=".$pwc." db_id=".$db_id." key=".$key." full_blocks=".$nfull." of ".$nblocks);
            }
          }
        }

        # update global with current information
        if ($response =~ m/fail/) {
          $db_result = "fail";
        } else {
          $db_result = "ok";
        }
        $db_response     = $response;
        $db_response_xml = $xml;

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

  my $cmd  = "";
  my $result = "";
  my $response = "";
  my $sleep_time = 2;
  my $sleep_counter = 0;

  my $ncore = 1;
  my $xml = "";
  my $load1 = "";
  my $load5 = "";
  my $load15 = "";

  Dada::logMsg(1, $dl, "loadThread: starting [".$sleep_time." polling]");

  # get the number of cores / processors in this host
  $cmd = "cat /proc/cpuinfo | grep processor | wc -l";
  ($result, $response) = Dada::mySystem($cmd);
  if ($result eq "ok") {
    $ncore = $response;
  } else {
    Dada::logMsg(0, $dl, "loadThread: could not determine number of processors, assuming 1");
  }

  while (!$quit_daemon) {

    # sleep 
    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;

    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      ($result,$response) = Dada::getLoadInfo();
      ($load1, $load5, $load15) = split(/,/, $response); 
      
      $xml = "<load ncore='".$ncore."'>".$load1."</load>";

      $load_result = $result;
      $load_response = $xml;

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
  my $xml = "";

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
      
      $xml = "<temperature units='Celcius'>";

      if ($response eq "")
      {
        $xml .= "0";
      }
      else
      {
        $xml .= $response
      }
      $xml .= "</temperature>";
      
      $temp_result = $result;
      $temp_response = $xml;

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

  # launch a thread that will kludgily kill this script after 5 seconds
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
# causes daemon to exit regardless after specified time, unless it is ready to
# exit cleanly.
#
sub killThread($) 
{
  my ($seconds_to_wait) = @_;

  while (($seconds_to_wait > 0) && ($kill_daemon))
  {
    Dada::logMsg(2, $dl, "killThread: sleeping, ".$seconds_to_wait." seconds remaining");
    if ($seconds_to_wait <= 5) {
      Dada::logMsg(1, $dl, "killThread: sleeping, ".$seconds_to_wait." seconds remaining");
    }
  
    sleep(1);
    $seconds_to_wait--;
  }

  if ($kill_daemon) {
    Dada::logMsg(1, $dl, "killThread: HARD exit");
    exit -1;
  } else {
    return 0;
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

  Dada::logMsg(0, $dl, "Listening on ".$host.":".$port);
  # open a socket for listening connections
  $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 3,
    ReuseAddr => 1
  );
  if (!$sock) {
    return ("fail", "Could not create listening socket: ".$host.":".$port);
  }
  Dada::logMsg(2, $dl, "Opened socket on ".$host.":".$port);

  return ("ok", "");

}

END { }

1;  # return value from file
