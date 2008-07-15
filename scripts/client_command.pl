#!/usr/bin/env perl

use IO::Socket;     # Standard perl socket library
use Net::hostent;
use Dada;           # DADA Module for configuration options
use threads;
use threads::shared;
use strict;         # strict mode (like -Wall)


#
# Constants
#

use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared = Dada->getDadaConfig();      # dada.cfg in a hash


my $i;
my $j;
my $commString = @ARGV[0];
my @threads;
my @results;
my @responses;
my $failure = "false";

my $logdir     = $cfg{"CLIENT_LOG_DIR"};
my $controldir = $cfg{"CLIENT_CONTROL_DIR"};
my $archivedir = $cfg{"CLIENT_ARCHIVE_DIR"};
my $rawdatadir = $cfg{"CLIENT_RECORDING_DIR"};
my $scratchdir = $cfg{"CLIENT_SCRATCH_DIR"};

my @machines = ();

# Machines to run this command on
for ($i=1;$i<=$#ARGV;$i++) {
  push(@machines, $ARGV[$i]);
}

for ($i=1;$i<=$#ARGV;$i++) {

  if ($commString eq "start_master_script") {
    
    my $result;
    my $response;

    my $string = "ssh -x ".@ARGV[$i]." \"cd ".$cfg{"SCRIPTS_DIR"}."; ./client_master_control.pl\"";
    @threads[$i] = threads->new(\&sshCmdThread, $string);

  } else {
    @threads[$i] = threads->new(\&commThread, $commString, @ARGV[$i]);
  }

  if ($? != 0) {
    @results[$i] = "dnf"; 
    @responses[$i] = "dnf"; 
  }
}

for($i=1;$i<=$#ARGV;$i++) {
  if ($results[$i] ne "dnf") {  
    (@results[$i],@responses[$i]) = @threads[$i]->join;
  }

}

for($i=1;$i<=$#ARGV;$i++) {
  print $ARGV[$i].":".$results[$i].":".$responses[$i]."\n";
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

sub commThread($$) {

  (my $command, my $machine) = @_;

  my $result = "fail";
  my $response = "Failure Message";
 
  my $handle = Dada->connectToMachine($machine, $cfg{"CLIENT_MASTER_PORT"}, 0);
  # ensure our file handle is valid
  if (!$handle) { 
    return ("fail","Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"}); 
  }

  ($result, $response) = Dada->sendTelnetCommand($handle,$command);

  $handle->close();

  return ($result, $response);

}

