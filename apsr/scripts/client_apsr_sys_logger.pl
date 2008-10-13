#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  1 Feb 2008
# Modified: 1 Feb 2008
# 

#
# Include Modules
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;         # strict mode (like -Wall)
use File::Basename;
use Getopt::Std;
use Apsr;           # APSR/DADAModule for configuration options


#
# Constants
#
use constant DEBUG_LEVEL   => 1;
use constant LOGFILE       => "apsr_sys_logger.log";

#
# Global Variable Declarations
#
our %cfg : shared = Apsr->getApsrConfig();      # Apsr.cfg in a hash
our $log_socket;
our $log_fh;

#
# Local Variable Declarations
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
my $type = "INFO";
my $line = "";

#
# Register Signal handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# Auto flush output
$| = 1;

my %opts;
getopts('e', \%opts);

if ($opts{e}) {
  $type = "ERROR";
}


# Open a connection to the nexus logging facility
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
if (!$log_socket) {
  print "Could not open a connection to the nexus SRC log: $log_socket\n";
}

open $log_fh, ">>".$logfile;

my $timestamp = "";

while (defined($line = <STDIN>)) {

  chomp $line;

  # Try to determine if there is a timestamp there already 
  if ($line =~ m/^\[(\d\d\d\d)\-(\d\d)\-(\d\d)\-(\d\d):(\d\d):(\d\d)\]/) {
    $timestamp = substr($line,1,19);   
    $line = substr($line,22);
    $type = "INFO";

    if ($line =~ m/^WARN: /) {
      $type = "WARN";
      $line = substr($line,6);
    }

    if ($line =~ m/^ERROR: /) {
      $type = "ERROR";
      $line = substr($line,7);
    }

  } else {

    $timestamp = Dada->getCurrentDadaTime();
    $type = "INFO";
  } 

  logMessage(0,$timestamp,$type,$line);

}

close $log_fh;

exit 0;


#
# Logs a message to the Nexus
#
sub logMessage($$$$) {
  (my $level, my $time, my $type, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    if (!($log_socket)) {
      $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "obs mngr", $message);
    }
    print $log_fh "[".$time."] ".$message."\n";
  }
}

sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";

  if ($log_socket) {
    close($log_socket);
  }

  print STDERR basename($0)." : Exiting\n";

  exit 1;

}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $log_socket = 0;
  $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SRC_LOG_PORT"});

}


