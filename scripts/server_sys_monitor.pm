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
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $log_host $log_port $daemon_name $master_log_prefix %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $log_host;
our $log_port;
our $daemon_name;
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
  my $read_set = 0;
  my $rh = 0;
  my $handle = 0;
  my $hostname = "";
  my $hostinfo = 0;
  my $host = "";
  my $domain = "";
  my $result = "";
  my $response = "";
  my $tid = 0;
  my @threads = ();

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;
  
  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  # start the control thread
  Dada::logMsg(2, $dl, "starting controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # create a read set for handle connections and data 
  $read_set = new IO::Select();
  $read_set->add($log_sock);

  Dada::logMsg(2, $dl, "Waiting for connection on ".$log_host.":".$log_port);

  while (!$quit_daemon) {

    # Get all the readable handles from the log_sock 
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 2);

    foreach $rh (@$readable_handles) {

      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $log_sock) {

        $handle = $rh->accept();
        $handle->autoflush(1);
        $hostinfo = gethostbyaddr($handle->peeraddr);
        $hostname = $hostinfo->name;
        ($host, $domain) = split(/\./,$hostname,2);
        Dada::logMsg(2, $dl, "Accepting connection from ". $hostname);

        # start a thread to handle this connection
        $tid = threads->new(\&logThread, $handle, $host);
        $tid->detach();
        #push @threads, $tid;
        $handle = 0;

      } else {
        Dada::logMsg(0, $dl, "main: received connection not on lock_sock");

      }
    }

  

  }

  # Rejoin our daemon control thread
  $control_thread->join();

  #my $i = 0;
  #for ($i=0; $i<=$#threads; $i++) {
  #  $threads[$i]->join();
  #}

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  close($log_sock);

  return 0;

}


###############################################################################
# 
#
#
sub logThread($$) {

  my ($handle, $host) = @_;

  Dada::logMsg(2, $dl, "logThread(".$handle.", ".$host.")");

  my $poll_handle = 1;
  my $read_something = 0;
  my $line = "";
  my $quit_this_thread = 0;
  my $read_set = 0;
  my $rh = 0;

  # set the input record seperator to \r\n
  $/ = "\r\n";

  # make the socket non blocking
  $handle->blocking(0);

  # create a read set for handle connections and data 
  $read_set = new IO::Select();
  $read_set->add($handle);

  while (!$quit_daemon && $handle) {

    # Get all the readable handles from the log_sock 
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    $read_something = 0;
    $poll_handle = 1;

    foreach $rh (@$readable_handles) {

      if ($rh && ($rh == $handle)) {

        # since the handle is non blocking, we poll read it until we get 
        # nothing back, then drop this loop and return to selecting on 
        # the socket
        while (!$quit_daemon && $poll_handle) {

          $line = $handle->getline;
         
          # if there was nothing at the socket
          if ((!defined $line) || ($line eq "")) {

            # stop polling the socket
            $poll_handle = 0;

            # if we haven't read anything, the socket is shutting down
            if (!$read_something) {
              Dada::logMsg(2, $dl, "logThread: lost connection from ".$host);
              $read_set->remove($rh);
              close($rh);
              $rh = 0;
              $handle = 0;
            }
  
          } else {

            $read_something = 1;        

            # strip a trailing \r\n if it exists
            $line =~ s/\r\n$//;

            # the log_lock is explicitly unlocked when it goes out of scope
            lock($log_lock);
            Dada::logMsg(2, $dl, "logThread [".$host."] received: ".$line);
            my $result = logMessage($host, $line);
            if ($result ne "ok") {
              Dada::logMsg(0, $dl, "logThread [".$host."] misformed: ".$line);
            }
          
          }
        }
        Dada::logMsg(3, $dl, "logThread [".$handle."] log_loop ended");

      } else {
        Dada::logMsg(0, $dl, "logThread: received data on wrong handle");
      }
    }
  }

  # need to clean up memory used
  Dada::logMsg(2, $dl, "logThread: exiting");

}


###############################################################################
#
# logs a message to the desginated log file, NOT the scripts' log
#
sub logMessage($$) {

  (my $machine, my $string) = @_;

  my $statusfile_dir = $cfg{"STATUS_DIR"};
  my $logfile_dir    = $cfg{"SERVER_LOG_DIR"};
  my $status_file = "";
  my $host_log_file = "";
  my $combined_log_file = "";
  my $time = "";
  my $tag = "";
  my $lvl = "";
  my $src = "";
  my $msg = "";

  # determine the source machine
  my @array = split(/\|/,$string,5);
  if ($#array == 4) {

    ($time, $tag, $lvl, $src, $msg) = split(/\|/,$string,5);

    $host_log_file = $logfile_dir."/".$machine.".".$tag.".log";
    $combined_log_file = $logfile_dir."/".$master_log_prefix.".".$tag.".log";

    if ($lvl eq "WARN") {
      $status_file = $statusfile_dir."/".$machine.".".$tag.".warn";
    } 
    if ($lvl eq "ERROR") {
      $status_file = $statusfile_dir."/".$machine.".".$tag.".error";
    }

    # Log the message to the hosts' log file
    if ($host_log_file ne "") {
      if (-f $host_log_file) {
        open(FH,">>".$host_log_file);
      } else {
        open(FH,">".$host_log_file);
      }

      if ($lvl eq "INFO") {
        print FH "[".$time."] ".$src.": ".$msg."\n";
      } else {
        print FH "[".$time."] ".$src.": ".$lvl.": ".$msg."\n";
      }
      close FH;
    }

    # Log the message to the combined log file
    if (-f $combined_log_file) {
      open(FH,">>".$combined_log_file);
    } else {
      open(FH,">".$combined_log_file);
    }


    if (($lvl eq "WARN") || ($lvl eq "ERROR")) {
      print FH $machine." [".$time."] ".$lvl." ".$src.": ".$msg."\n";  
    } else  {
      print FH $machine." [".$time."] ".$src.": ".$msg."\n";
    }
    
    close FH;

    # If the file is a warning or error, we create a warn/error file too
    if ($status_file ne "") {
      if (-f $status_file) {
        open(FH,">>".$status_file);
      } else {
        open(FH,">".$status_file);
      }
      print FH $src.": ".$msg."\n";
      close FH;
    }
  } else {
    return "fail";
  }
}

###############################################################################
#
# listens for quit commands
#
sub controlThread($$) {

  Dada::logMsg(1, $dl, "controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl, "controlThread(".$quit_file.", ".$pid_file.")");

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

  $log_sock = new IO::Socket::INET (
    LocalHost => $log_host,
    LocalPort => $log_port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  if (!$log_sock) {
    return ("fail", "Could not create listening socket: ".$log_host.":".$log_port);
  }

  if ($master_log_prefix eq "") {
    return ("fail", "master_log_prefix was not set");
  } 

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");

}




END { }

1;  # return value from file
