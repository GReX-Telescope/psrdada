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

use Dada;           # DADA Module for configuration options
use strict;         # strict mode (like -Wall)
use File::Basename;


#
# Constants
#
use constant DEBUG_LEVEL   => 2;
use constant MIN_GAIN      => 1;
use constant MAX_GAIN      => 65535;
use constant DEFAULT_GAIN  => 2000;


#
# Global Variable Declarations
#
our %cfg = Dada->getDadaConfig();      # dada.cfg in a hash
our $socket;
our $log_fh;

#
# Local Variable Declarations
#
my $channel_base;
my $dfb_response;
my $line = "";
my $current_gain = 0;
my $new_gain = 0;
my $gain_step = 0;
my $cmd = "";

#
# Register Signal handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
$SIG{PIPE} = \&sigPipeHandle;

# Auto flush output
$| = 1;

# Get command line
if ($#ARGV!=2) {
  usage();
  exit 1;
}

my ($chan, $pol, $gain) = @ARGV;

# Open a connection to the nexus logging facility
$socket = Dada->connectToMachine($cfg{"SERVER_HOST"},$cfg{"SERVER_GAIN_CONTROL_PORT"}, 1);

if (!$socket) {

  print "Could not open a connection to server_dfb3_gain_controller.pl : $socket\n";
  close $log_fh;
  exit 1;

} else {
  my $i=0;
 
  if ($chan eq "all") {

    for ($i=0; $i<$cfg{"NUM_PWC"};$i++) {
     print "GAIN: [$i,0] = ".set_dfb3_gain($socket, $i, $pol, $gain)."\n";
    }

  } else {
    print "GAIN: [$i,0] = ".set_dfb3_gain($socket, $chan, $pol, $gain)."\n";
  }

  logMessage(0, "Closing socket");

  close($socket);

}

exit 0;


#
# Logs a message to the Nexus
#
sub logMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();

    print "[".$time."] ".$message."\n";
  }
}

sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";

  if ($socket) {
    close($socket);
  }

  print STDERR basename($0)." : Exiting\n";

  exit 1;

}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $socket = 0;
  $socket = Dada->connectToMachine($cfg{"SERVER_HOST"},$cfg{"SERVER_GAIN_CONTROL_PORT"}, 1);

}

sub usage() {

  print "Usage: ".$0." chan pol gain\n";
  print "   chan   Channel to affect (all for all channels\n";
  print "   pol    Polarisation value to set [0,1]\n";
  print "   gain   Gain value to set\n";
  print "\n";

}


sub set_dfb3_gain($$$$) {

  my ($socket, $chan, $pol, $val) = @_;
                                                                                                                                           
  my $cmd = "";
  my $dfb_response = "";

  if ($pol eq "all") {

    $cmd = "APSRGAINHACK ".$chan." 0 ".$val;
    logMessage(2, "srv0 <- ".$cmd);
    print $socket $cmd."\r\n";
    $dfb_response = Dada->getLine($socket);
    logMessage(2, "srv0 -> ".$dfb_response);

    $cmd = "APSRGAINHACK ".$chan." 1 ".$val;
    logMessage(2, "srv0 <- ".$cmd);
    print $socket $cmd."\r\n";
    $dfb_response = Dada->getLine($socket);
    logMessage(2, "srv0 -> ".$dfb_response);

  } else {

    $cmd = "APSRGAINHACK ".$chan." ".$pol." ".$val;
    logMessage(2, "srv0 <- ".$cmd);
    print $socket $cmd."\r\n";
    $dfb_response = Dada->getLine($socket);
    logMessage(2, "srv0 -> ".$dfb_response);

  }

  if ($dfb_response ne "OK") {
    logMessage(0, "DFB3 returned an error: \"".$dfb_response);
  }

  return $dfb_response;
}

sub get_dfb3_gain($$$) {

  (my $socket, my $chan, my $pol) = @_;

  my $cmd = "APSRGAIN ".$chan." ".$pol;

  logMessage(2, "srv0 <- ".$cmd);

  print $socket $cmd."\r\n";

  # The DFB3 should return a string along the lines of
  # OK <chan> <pol> <value>

  my $dfb_response = Dada->getLine($socket);

  logMessage(2, "srv0 -> ".$dfb_response);

  my ($result, $dfb_chan, $dfb_pol, $dfb_gain) = split(/ /,$dfb_response);

  if ($result ne "OK") {

    logMessage(0, "DFB3 returned an error: \"".$dfb_response);
    return ("fail", 0);

  }

  if ($dfb_chan ne $chan) {
    logMessage(0, "DFB3 returned an channel mismatch: requested ".$chan.", received ".$dfb_chan);
    return ("fail", 0);
  }

  if ($dfb_pol ne $pol) {
    logMessage(0, "DFB3 returned an pol mismatch: requested ".$pol.", received ".$dfb_pol);
    return ("fail", 0);
  }

  return ("ok", $dfb_gain);
                                                                                                                                                                                                
}

