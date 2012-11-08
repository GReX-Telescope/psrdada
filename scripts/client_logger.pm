package Dada::client_logger;

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($log_host $log_port $log_sock $dl $daemon_name $type $dameon %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $log_host;
our $log_port;
our $log_sock;
our $dl;
our $pwc_id;
our $daemon_name;
our %cfg;
our $type;
our $daemon;

#
# non-exported package globals go here
#

#
# initialize package globals
#
$log_host = 0;
$log_port = 0;
$log_sock = 0;
$dl = 1; 
$pwc_id = 0;
$daemon_name = 0;
%cfg = ();
$type = 0;
$daemon = 0;

#
# initialize other variables
#


###############################################################################
#
# package functions
# 

sub main() {

  msg(2, "INFO", "client_logger::".$daemon_name);

  my $line     = "";
  my $time     = "";
  my $class     = "";

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # Open a connection to the nexus logging port
  $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  if (!$log_sock) {
    print STDERR "Could open log port: ".$log_host.":".$log_port."\n";
  }

  msg(2, "INFO", "client_logger: begin reading from STDIN");

  while (defined($line = <STDIN>)) {

    chomp $line;

    $class = "INFO";

    # Try to parse the line if it has the standard format
    if ($line =~ m/^\[(\d\d\d\d)\-(\d\d)\-(\d\d)\-(\d\d):(\d\d):(\d\d)\]/) {
      $time = substr($line,1,19);
      $line = substr($line,22);

      if ($line =~ m/^WARN: /) {
        $class = "WARN";
        $line = substr($line,6);
      }

      if ($line =~ m/^ERROR: /) {
        $class = "ERROR";
        $line = substr($line,7);
      }
  
    } else {
      $time = Dada::getCurrentDadaTime();
    }

    # Always log these messages
    msg(0, $class, $line, $time);

  }
  msg(2, "INFO", "client_logger: finished reading from STDIN");
  
  return  0;
}

#
# Logs a message to the Nexus
#
sub msg($$$;$) {

  (my $level, my $class, my $message, my $time="") = @_;

  my $log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name.".log";

  if ($level <= $dl) {

    if ((!defined($time)) || ($time eq "")) {
      $time = Dada::getCurrentDadaTime();
    }

    # open the socket if it is closed
    if (!($log_sock)) {
      $log_sock = Dada::nexusLogOpen($log_host, $log_port);
    }

    if ($log_sock) {
      Dada::nexusLogMessage($log_sock, $pwc_id, $time, $type, $class, $daemon, $message);
    }

    open FH, ">>".$log_file;
    print FH "[".$time."] ".$message."\n";
    close FH; 
  }
}



#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  
  # Tell threads to try and quit
 
  if ($log_sock) {
    close($log_sock);
  } 
  
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

#
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  if ($log_sock) {
    $log_sock->close();
  }
  $log_sock = 0;
  if ($log_host && $log_port) {
    $log_sock = Dada::nexusLogOpen($log_host, $log_port);
  }

}




END { }

1;  # return value from file
