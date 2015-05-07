package Dada::server_sys_monitor;

###############################################################################
#
# monitors all multilog messages from the various client daemons. It also 
# writes warning and error messages to the STATUS_DIR for display in the 
# web interface
#
###############################################################################

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
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
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $log_host $log_port $daemon_name $master_log_node_prefix $master_log_prefix %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $log_host;
our $log_port;
our $daemon_name;
our $master_log_node_prefix;
our $master_log_prefix;
our %cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $log_sock;
our $log_lock : shared;
our $warn;
our $error;

#
# initialize package globals
#
$dl = 1; 
$log_host = "";
$log_port = 0;
$log_sock = 0;
$log_lock = 0;
$daemon_name = 0;
$master_log_node_prefix= "";
$master_log_prefix= "";
%cfg = ();

#
# initialize other variables
#
$warn = ""; 
$error = ""; 
$quit_daemon = 0;

###############################################################################
#
# package functions
# 

sub main() {

  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file  = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file  = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $control_thread = 0;
  my $logging_thread = 0;

  my $read_set = 0;
  my $rh = 0;
  my $handle = 0;
  my $hostname = "";
  my $hostinfo = 0;
  my $host = "";
  my $domain = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }
  $cmd = "rm -f ".$cfg{"STATUS_DIR"}."/*.src.* ".$cfg{"STATUS_DIR"}."/*.sys.*";
  ($result, $response) = Dada::mySystem($cmd);

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  # add a delimiting period to the end of the node prefix
  if ($master_log_node_prefix ne "")
  {
    $master_log_node_prefix .= "."; 
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;
  
  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # create a thread queue to handle the incoming log messages for the
  # logging thread to handle
  my $in = new Thread::Queue;

  # start the control thread
  Dada::logMsg(2, $dl, "main: starting controlThread");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # start the logging thread
  Dada::logMsg(2, $dl, "starting loggingThread");
  $logging_thread= threads->new(\&loggingThread, $in);

  # create a read set for handle connections and data 
  $read_set = new IO::Select();
  $read_set->add($log_sock);

  Dada::logMsg(1, $dl, "waiting for connections on ".$log_host.":".$log_port);

  my $read_something = 0;
  my $poll_handle = 0;
  my $line = "";

  while (!$quit_daemon) {

    # Get all the readable handles from the log_sock 
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles) {

      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $log_sock) {

        $handle = $rh->accept();

        $handle->autoflush(1);

        # get the hostname for this connection
        $hostinfo = gethostbyaddr($handle->peeraddr);
        if (defined $hostinfo)
        {
          $hostname = $hostinfo->name;
          ($host, $domain) = split(/\./,$hostname,2);
        }
        else
        {
          $host = "localhost";
        }

        Dada::logMsg(2, $dl, "main [".$host."] accepting connection");

        $read_set->add($handle);
        $handle = 0;

      }
      else
      {
        # get the hostname for this connection
        $hostinfo = gethostbyaddr($rh->peeraddr);
        if (defined $hostinfo)
        {
          $hostname = $hostinfo->name;
          ($host, $domain) = split(/\./,$hostname,2);
        }
        else
        {
          $host = "localhost";
        }
        Dada::logMsg(3, $dl, "main [".$host."] processing message");

         # set the input record seperator to \r\n
        $/ = "\r\n";

        # make the socket non blocking
        $rh->blocking(0);

        $read_something = 0;
        $poll_handle = 1;

        # read as much data as we can from the socket
        while (!$quit_daemon && $poll_handle)
        {
          $line = $rh->getline;

          # if nothing at the socket 
          if ((!defined $line) || ($line eq ""))
          {
            # stop polling
            $poll_handle = 0;

            # if we haven't read anything, the socket is shutting down
            if (!$read_something)
            {
              Dada::logMsg(2, $dl, "main [".$host."] lost connection");
              $read_set->remove($rh);
              $rh->close();
            }
          }
          else
          {
            $read_something = 1;
            $line =~ s/\r\n$//;
            Dada::logMsg(2, $dl, "main [".$host."] <- ".$line);
            $in->enqueue($line);
          }
        }
      }
    }
  }

    
  # join the control and logging threads
  Dada::logMsg(2, $dl, "main: joining control_thread");
  $control_thread->join();

  Dada::logMsg(2, $dl, "main: joining logging_thread");
  $logging_thread->join();


  my @active_handles = $read_set->handles;
  foreach $rh (@active_handles )
  {
    Dada::logMsg(2, $dl, "main: closing socket: ".$rh);
    $rh->close();
  }

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;

}


###############################################################################
#
# Dequeues messages from the log queue and write them to the relevant logfiles
#
sub loggingThread($) 
{
  (my $in) = @_;

  Dada::logMsg(2, $dl, "loggingThread: starting");

  my $statusfile_dir = $cfg{"STATUS_DIR"};
  my $logfile_dir    = $cfg{"SERVER_LOG_DIR"};
  
  my $line = "";
  my $status_file = "";
  my $pwc_log_file = "";
  my $combined_log_file = "";

  my @bits = ();
  my $src = "";
  my $time = "";
  my $type = "";
  my $class = "";
  my $program = "";

  while (!$quit_daemon) 
  {

    if ($in->pending)
    {
      $message = $in->dequeue();

      # extract the message parameters
      @bits = split(/\|/, $message, 6);

      if ($#bits == 5) 
      {
#       src       source of message (hostname or PWC ID) 
#       time      timestamp of message
#       type      type of message (pwc, sys, src) 
#       class     class of message (INFO, WARN, ERROR) 
#       program   script or binary that generated message (e.g. obs mngr)
#       message   message itself

        $src     = $bits[0];
        $time    = $bits[1];
        $type    = $bits[2];
        $class   = $bits[3];
        $program = $bits[4];
        $message = $bits[5];

        $pwc_log_file = $logfile_dir."/".$master_log_node_prefix.$src.".".$type.".log";
        $combined_log_file = $logfile_dir."/".$master_log_prefix.".".$type.".log";

        if (($class eq "WARN") || ($class eq "ERROR")) {
          $status_file = $statusfile_dir."/".$master_log_node_prefix.$src.".".$type.".".lc($class);
        } else {
          $status_file = "";
        }

        if ($class eq "INFO") {
          $line = "[".$time."] ".$program.": ".$message;
        } else {
          $line = "[".$time."] ".$program.": ".$class.": ".$message;
        }
        Dada::logMsg(3, $dl, "loggingThread: ".$src." ".$line);

        # log message to the PWC specific log file
        if (-f $pwc_log_file) {
          open(FH,">>".$pwc_log_file);
        } else {
          open(FH,">".$pwc_log_file);
        }
        print FH $line."\n";
        close FH;

        # log the message to the combined log file
        if (-f $combined_log_file) {
          open(FH,">>".$combined_log_file);
        } else {
          open(FH,">".$combined_log_file);
        }
        print FH $src." ".$line."\n";
        close FH;

        # if the file is a warning or error, we create a warn/error file too
        if ($status_file ne "") {
          if (-f $status_file) {
            open(FH,">>".$status_file);
          } else {
            open(FH,">".$status_file);
          }
          print FH $program.": ".$line."\n";
          close FH;
        }   
      }
      else
      {
        Dada::logMsg(1, $dl, "loggingThread: ignoring message [".$message."]");
      }
    } 
    else
    {
      Dada::logMsg(3, $dl, "loggingThread: no messages pending");
      sleep(1);
    }
  }

  Dada::logMsg(2, $dl, "loggingThread: exiting");
  return 0;
}

###############################################################################
#
# listens for quit commands
#
sub controlThread($$) {

  Dada::logMsg(1, $dl, "controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl, "controlThread quit_file=".$quit_file);
  Dada::logMsg(2, $dl, "controlThread pid_file=".$pid_file);

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl, "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, $dl, "controlThread: exiting");
  return 0;
}
  


###############################################################################
#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
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

} 

###############################################################################
#
# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  if ($master_log_prefix eq "") {
    return ("fail", "master_log_prefix was not set");
  } 

  my $n_listen = $cfg{"NUM_PWC"} * 10;

  $log_sock = new IO::Socket::INET (
    LocalHost => $log_host,
    LocalPort => $log_port,
    Proto => 'tcp',
    Listen => $n_listen,
    ReuseAddr => 1
  );
  if (!$log_sock) {
    return ("fail", "Could not create listening socket: ".$log_host.":".$log_port);
  }

  return ("ok", "");

}




END { }

1;  # return value from file
