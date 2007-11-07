#!/usr/bin/env perl

use IO::Socket;     # Standard perl socket library
use Net::hostent;
use Dada;           # DADA Module for configuration options
use threads;
use Switch;
use List::Util 'shuffle';

use strict;         # strict mode (like -Wall)

my @DFB_MACHINES = Dada->getDFBMachines();
my @PWC_MACHINES = Dada->getPWCMachines();
my $rVal;
my $val;

my $rString;
my $result;
my $response;
my $cmd;

#
# States
#
use constant IDLE          => 1;
use constant PREPARED      => 2;
use constant CLOCKING      => 3;
use constant CLOCKING_UTC  => 4;
use constant RECORDING     => 5;
use constant RECORDING_UTC => 6;

#
# Commands
#
use constant HEADER         => 1;
use constant CLOCK          => 2;
use constant START          => 3;
use constant SET_UTC_START  => 4;
use constant REC_START      => 5;
use constant REC_STOP       => 6;
use constant START_DFB      => 7;
use constant SLEEP          => 8;

use constant NSTATES         => 8;
use constant MAX_REC_TOGGLES => 2;
use constant DEBUG_LEVEL     => 0;

use constant CLOCKING_SLEEP_TIME      => 5;
use constant CLOCKING_UTC_SLEEP_TIME  => 5;
use constant RECORDING_SLEEP_TIME     => 5;
use constant RECORDING_UTC_SLEEP_TIME => 5;
use constant CLOCK_TOGGLE_TIME        => 2;

my @state_trans;
@state_trans = ( 
[2,0,0,0,0,0,0,0],
[0,3,5,0,0,0,0,0],
[0,0,0,4,0,0,1,3],
[0,0,0,0,6,0,1,4],
[0,0,0,6,0,0,1,5],
[0,0,0,0,0,4,1,6] );

my $i=1;
my $j=0;
my $ref;

if (DEBUG_LEVEL >= 2) {
  print "State array:\n";
  for ($i=0; $i<=$#state_trans; $i++) {
    $ref = $state_trans[$i];
    for ($j=0;$j<8;$j++) {
      print $ref->[$j].", ";
    }
    print "\n";
  }
}

my @all_paths = getRecursiveTransitions(1,0,0,@state_trans);

my $longest_test = 0;

for ($i=0; $i<=$#all_paths; $i++) {
  if (DEBUG_LEVEL >= 2) {
    print "$i = ".@all_paths[$i]."\n";
  }
  if ((length @all_paths[$i]) > $longest_test) {
    $longest_test = length @all_paths[$i];
  }
}

my @indexes = (0 .. $#all_paths);
if (DEBUG_LEVEL >= 2) {
  print "@indexes\n";
}

my @shuffled_indexes = shuffle(@indexes);

if (DEBUG_LEVEL >= 2) {
  print "@shuffled_indexes\n";
}

my $test = "";
if ($#ARGV >= 0) {
  my $argnum;
  foreach $argnum (0 .. $#ARGV) {
    if ($argnum == $#ARGV) {
      $test=$test.$ARGV[$argnum];
    } else {
      $test=$test.$ARGV[$argnum]." ";
    }
  }
  runTest($test);
} else {
  my $pass = "true";
  my $test_length = 0;
  for ($i=0; (($i<=$#shuffled_indexes) && ($pass eq "true")); $i++) {
    print STDERR "Test ".$i." ".@all_paths[@shuffled_indexes[$i]].":";
    if (DEBUG_LEVEL >= 1) { print STDERR "\n"; }
    $test_length = length @all_paths[@shuffled_indexes[$i]];
    for ($j=0;$j<($longest_test - $test_length);$j++) {
      print STDERR " ";
    }
    $pass = runTest(@all_paths[@shuffled_indexes[$i]]);
    print STDERR "pass\n";
  }
}
exit 0;

#
# functions
#


sub runTest($) {

  (my $state_sequence) = @_;

  my @state_array = split(', ',$state_sequence);
  my $pass = "true";
  my $result = "fail";
  my $reponse = "";


  # Connect to pwc command
  my $handle = Dada->connectToMachine(Dada->NEXUS_MACHINE,Dada->NEXUS_CONTROL_PORT);
  if (!$handle) {
    $pass = "Error connecting to Nexus: ".Dada->NEXUS_MACHINE.":".Dada->NEXUS_CONTROL_PORT."\n";
  } else {

    # ignore the "welcome" message
    $response = <$handle>;

    my $i;
    for ($i=0; (($i<$#state_array) && ($pass eq "true")); $i++) {
      if (DEBUG_LEVEL >= 2) {
        print "Testing ".@state_array[$i]." => ".@state_array[$i+1]."\n";
      }
      ($result, $response) = handleTransition(@state_array[$i],@state_array[$i+1], $handle);
      if ($result ne "ok") {
        $pass = $reponse;
      }
    }
  }
  return $pass;
}


sub getLegalTransitions($$) {

  (my $state, my @transitions) = @_;

  # since array is indexed from 0, not 1
  my $ref = $transitions[($state-1)];

  my $i=0;
  my $j=0;
  my @legal_trans;

  for ($i=0; $i<NSTATES; $i++) {
    if ($ref->[$i] > 0) {
      print $state." => ".$ref->[$i]." legal\n";
      @legal_trans[$j] = $ref->[$i];
      $j++;
    } 
  }
  return @legal_trans;
}

sub getRecursiveTransitions($$$$) {

  (my $state, my $state_prev, my $nrectoggles, my @all) = @_;

  #print "getRecursiveTransitions(".$state.")\n";
  my $index = $state - 1;

  # pointer to all legal transitions from state
  my $ref = $all[$index];

  my $i=0;
  my $j=0;
  my $k=0;
  my @legal_trans;
  my @arr;

  for ($i=0; $i<NSTATES; $i++) {
    #print "$state.$i: (comparing $state == ".$ref->[$i].") ";
    # if there is a legal, transition for this state
    if (($ref->[$i] > 1) && (!(($ref->[$i] == $state) && ($state == $state_prev))) && ($nrectoggles < MAX_REC_TOGGLES)) {
      #print $state." -> ".$ref->[$i]."\n";
      if (($state == CLOCKING_UTC) && ($ref->[$i] == RECORDING_UTC)) {
        $nrectoggles++;
      }
      # get and array of "strings" of all possible sequences
      my @arr = getRecursiveTransitions($ref->[$i], $state, $nrectoggles, @all);
      for ($k=0; $k<=$#arr; $k++) {
        @legal_trans[$j] = $state.", ".@arr[$k];
        $j++;
      }
    
    } elsif ($ref->[$i] == 1) {
      #print "end state\n";
      # This is the "end state", hence no comma at the end
      @legal_trans[$j] = "$state, 1";
      $j++;
    } else {
      #print "no match\n";
    }
  }
  return @legal_trans;
}

sub handleTransition($$$) {

  (my $curr_state, my $new_state, my $handle) = @_;

  if (DEBUG_LEVEL >= 2) { print "Transition ".$curr_state." => ".$new_state."\n"; }

  my $trans = $curr_state."->".$new_state;

  my $nexus_cmd = "";
  my $pwc_cmd = "";
  my $result = "fail";
  my $response = "";

  switch ($trans) {

    # IDLE (config)
    case "1->2" {

      my $min_pl = 500;         # bytes
      my $max_pl = 8092;        # bytes
      my $packet_header = 14;   # bytes;
      my $random_pl= int(rand($max_pl-($min_pl+$packet_header)))+$min_pl;

      ($result,$response) = Dada->prepareDFB(64,$random_pl);
      $nexus_cmd = "config ".Dada->CONFIG_FILE;
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);

    }

    # PREPARED (stop)
    case "2->1" {
      if (DEBUG_LEVEL >= 1) { print "stop dfb\n"; }
      ($result,$response) = Dada->stopDFB();
      $nexus_cmd = "stop";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # PREPARED (clock)
    case "2->3" {
      $nexus_cmd = "clock";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # PREPARED (start)
    case "2->5" {
      $nexus_cmd = "start";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # CLOCKING (stop)
    case "3->1" {
      ($result,$response) = Dada->stopDFB();
      if (DEBUG_LEVEL >= 1) { print "stop dfb\n"; }
      $nexus_cmd = "stop";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # CLOCKING (sleep)
    case "3->3" {
      if (DEBUG_LEVEL >= 1) { print "sleep ".CLOCKING_SLEEP_TIME."\n"; }
      sleep CLOCKING_SLEEP_TIME;
      $result = "ok";
    }

    # CLOCKING (dfb,setutc)
    case "3->4" {
      if (DEBUG_LEVEL >= 1) { print "start dfb\n"; }
      $nexus_cmd = "set_utc_start ".Dada->startDFB();
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # CLOCKING_UTC (stop)
    case "4->1" {
      if (DEBUG_LEVEL >= 1) { print "stop dfb\n"; }
      ($result,$response) = Dada->stopDFB();
      $nexus_cmd = "stop";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # CLOCKING_UTC (sleep)
    case "4->4" {
      if (DEBUG_LEVEL >= 1) { print "sleep ".CLOCKING_UTC_SLEEP_TIME."\n"; }
      sleep CLOCKING_UTC_SLEEP_TIME;
      $result = "ok";
    }

    # CLOCKING_UTC (rec_start)
    case "4->6" {
      $nexus_cmd = "rec_start ".Dada->getCurrentDadaTime(2);
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # RECORDING (stop)
    case "5->1" {
      if (DEBUG_LEVEL >= 1) { print "stop dfb\n"; }
      ($result,$response) = Dada->stopDFB();
      $nexus_cmd = "stop";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # RECORDING (sleep)
    case "5->5" {
      if (DEBUG_LEVEL >= 1) { print "sleep ".RECORDING_SLEEP_TIME."\n"; }
      sleep RECORDING_SLEEP_TIME;
      $result = "ok";
    }

    # RECORDING (dfb,setutc)
    case "5->6" {
      if (DEBUG_LEVEL >= 1) { print "start dfb\n"; }
      $nexus_cmd = "set_utc_start ".Dada->startDFB();
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # RECORDING (stop)
    case "6->1" {
      if (DEBUG_LEVEL >= 1) { print "stop dfb\n"; }
      ($result,$response) = Dada->stopDFB();
      $nexus_cmd = "stop";
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # RECORDING (rec_stop)
    case "6->4" {
      $nexus_cmd = "rec_stop ".Dada->getCurrentDadaTime(2);
      ($result,$response) = Dada->sendTelnetCommand($handle,$nexus_cmd);
    }

    # RECORDING (sleep)
    case "6->6" {
      if (DEBUG_LEVEL >= 1) { print "sleep ".RECORDING_UTC_SLEEP_TIME."\n"; }
      sleep RECORDING_UTC_SLEEP_TIME;
      $result = "ok";
    }

    else {
      print "Unhandled case: '$trans'\n";
    }
  }

  if ((length $nexus_cmd) > 0) {
    print $nexus_cmd."\n";
  }

  if (DEBUG_LEVEL >= 1) {
    print "sent:     '$nexus_cmd'\n";
    print "result:   '$result'\n";
    print "response: '$response'\n";
  }

  my $stateString = getStateString($new_state);

  if (Dada->waitForState($stateString, $handle, 500) < 0) {
    $result = "fail";
    $response = "ERROR: Could not change from $curr_state to $new_state successfully";
  }

  return ($result, $response);
}


sub getStateString($) {

  (my $id) = @_;

  my $state= "unknown state";

  if ($id eq "1") { $state = "idle"; }
  if ($id eq "2") { $state = "prepared"; }
  if ($id eq "3") { $state = "clocking"; }
  if ($id eq "4") { $state = "clocking"; }
  if ($id eq "5") { $state = "recording" }
  if ($id eq "6") { $state = "recording"; }

  return $state;
}



