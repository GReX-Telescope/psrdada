#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use Net::hostent;
use Caspsr;         # CASPSR/DADA Module for configuration options
use threads;
use threads::shared;
use strict;         # strict mode (like -Wall)


#
# Constants
#


#
# Global Variables
#
our %cfg : shared;
our $dl : shared;


#
# Initialize Global Variables
# 
$dl = 1;
%cfg = Caspsr::getConfig();


#
# Local Variables
#

my $cmd = "";
my $arg = "";
my $host_offset = 1;

my $i;
my $j;
my @machines = ();
my @threads = ();
my @results = ();
my @responses = ();
my $failure = "false";
my $comm_string = "";

# get the command
$cmd = $ARGV[0];


# some commands may have arguments before the machine list
if (($cmd eq "start_daemon") || ($cmd eq "stop_daemon") || ($cmd eq "db_info")) {
  $arg = $ARGV[1];
  $host_offset = 2;
}

# get the hosts from the arguements
for ($i=$host_offset; $i<=$#ARGV; $i++) {
  push(@machines, $ARGV[$i]);
}

for ($i=0; $i<=$#machines; $i++) {

  # special case for starting the master control script
  if (($cmd eq "start_daemon") && ($arg eq "caspsr_master_control")) {

    $comm_string = "ssh -x -l caspsr ".$machines[$i]." \"client_caspsr_master_control.pl\"";
    @threads[$i] = threads->new(\&sshCmdThread, $comm_string);

  } else {

    if ($arg ne "") {
      $comm_string = $cmd." ".$arg;
    } else {
      $comm_string = $cmd;
    }

    @threads[$i] = threads->new(\&commThread, $machines[$i], $cfg{"CLIENT_MASTER_PORT"}, $comm_string);
  }

  if ($? != 0) {
    @results[$i] = "dnf"; 
    @responses[$i] = "dnf"; 
  }
}

for($i=0;$i<=$#machines;$i++) {
  if ($results[$i] ne "dnf") {  
    (@results[$i],@responses[$i]) = @threads[$i]->join;
  }
}

for($i=0;$i<=$#machines;$i++) {
  print $machines[$i].":".$results[$i].":".$responses[$i]."\n";
  if (($results[$i] eq "fail") || ($results[$i] eq "dnf")) {
    $failure = "true";
  }
}

if ($failure eq "true") {
  exit 0;
} else {
  exit 0;
}


sub sshCmdThread($) {

  (my $command) = @_;

  my $result = "fail";
  my $response = "Failure Message";

  $response = `$command`;
  if ($? == 0) {
    $result = "ok";
  }
  return ($result, $response);
  
}

sub commThread($$$) {

  my ($host, $port, $cmd) = @_;

  logMsg(2, $dl, "[".$host."] commThread(".$host.", ".$port.", ".$cmd.")");

  my $handle = 0;
  my $result = "fail";
  my $response = "Failure Message";

  logMsg(2, $dl, "[".$host."] connectToMachine(".$host.", ".$port.", 1)");
  $handle = Dada::connectToMachine($host, $port, 1);
  # ensure our file handle is valid

  if (!$handle) { 
    logMsg(2, $dl, "[".$host."] could not connect to ".$host.":".$port);
    return ("fail","Could not connect to machine ".$host.":".$port);
  }
  logMsg(2, $dl, "[".$host."] connection opened");

  logMsg(2, $dl, "[".$host."] sendTelnetCommand(".$handle.", ".$cmd.")");
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  logMsg(2, $dl, "[".$host."] sendTelnetCommand() ".$result." ".$response);

  $handle->close();
  logMsg(2, $dl, "[".$host."] connection closed");

  logMsg(2, $dl, "[".$host."] returning ".$result." ".$response);
  return ($result, $response);

}

###############################################################################
#
# log a message based on debug level
#
sub logMsg($$$) {

  my ($lvl, $dlvl, $message) = @_;
  if ($lvl <= $dlvl) {
    my $time = Dada::getCurrentDadaTime();
    print STDOUT "[".$time."] ".$message."\n";
  }
}

