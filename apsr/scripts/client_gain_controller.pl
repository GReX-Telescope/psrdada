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
use constant DEBUG_LEVEL   => 1;
use constant LOGFILE       => "gain_controller.log";
use constant MIN_GAIN      => 1;
use constant MAX_GAIN      => 65535;


#
# Global Variable Declarations
#
our %cfg = Dada->getDadaConfig();      # dada.cfg in a hash
our $log_socket;
our $socket;
our $log_fh;

#
# Local Variable Declarations
#
my $logfile = $cfg{"CLIENT_LOG_DIR"}."/".LOGFILE;
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
if ($#ARGV!=0) {
  usage();
  exit 1;
}

(my $nchan) = @ARGV;

# 
# <STDIN> command is GAIN <chan> <pol> <dim> where:
#
#   <chan> varies from 0 to $chan
#   <pol>  is 0 or 1     
#   <dim   is always 0
#
#  Therefore we always have a max on 2*chan gain values to maintain 

my @pol0_gains = ();
my @pol1_gains = ();

my $i;
for ($i=0; $i < $nchan; $i++) {
  @pol0_gains[$i] = 0;
  @pol1_gains[$i] = 0;
}

# Open a connection to the nexus logging facility
$log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
if (!$log_socket) {
  print "Could not open a connection to the nexus SYS log: $log_socket\n";
}

open $log_fh,">>$logfile" or die "Could not open logfile \"".$logfile."\" for appending\n";

# Open a connection to the nexus logging facility
$socket = Dada->connectToMachine($cfg{"SERVER_HOST"},$cfg{"SERVER_GAIN_CONTROL_PORT"}, 1);

if (!$socket) {
  print "Could not open a connection to server_dfb3_gain_controller.pl : $socket\n";
  close $log_fh;
  exit 1;

} else {
  
  $cmd = "CHANNEL_BASE";
  logMessage(1, "INFO", "srv0 <- ".$cmd);
  print $socket $cmd."\r\n";
  $channel_base = Dada->getLine($socket);
  logMessage(1, "INFO", "srv0 -> ".$channel_base);


  logMessage(2, "INFO", "Asking for initial gains");
  my $abs_chan = 0;
  my $result = "";

  # Get the initial gain setting for every channel
  for ($i=0; $i < $nchan; $i++) {

    $abs_chan = ($nchan * $channel_base) + $i;

    ($result, @pol0_gains[$i]) = get_dfb3_gain($socket, $abs_chan, 0);
    ($result, @pol1_gains[$i]) = get_dfb3_gain($socket, $abs_chan, 1);

  }

  logMessage(2, "INFO", "Received initial gains");

  my $last_val = 0;

  while (defined($line = <STDIN>)) {

    chomp $line;
    logMessage(2, "INFO", "STDIN : ".$line);

    # Check that we got a gain string from STDIN
    if ($line =~ /^GAIN (\d) (0|1) (0|1) (\d|\.)+$/) {

      my ($ignore, $chan, $pol, $dim, $requested_val) = split(/ /,$line);

      $abs_chan = ($nchan * $channel_base) + $chan;

      if (($requested_val > $last_val + 5) || ($requested_val < $last_val -5)) {
        print $socket "BIG CHANGE: pol".$pol." ".$last_val." => ".$requested_val."\r\n";
      }
      $last_val = $requested_val;

      if ($pol eq 0) {
        $current_gain = $pol0_gains[$chan];
      } else {
        $current_gain = $pol1_gains[$chan];
      }

      logMessage(2, "INFO", "Current Gain = ".$current_gain);
      logMessage(2, "INFO", "CHAN=".$chan.", ABS_CHAN=".$abs_chan.", POL=".$pol.", DIM=".$dim.", VAL=".$requested_val);

      $new_gain = int($current_gain * $requested_val);
      logMessage(2, "INFO", "New Gain = ".$new_gain);

      if ($new_gain > MAX_GAIN) {
        $new_gain = MAX_GAIN;
      }

      if ($new_gain < MIN_GAIN) {
        $new_gain = MIN_GAIN;
      }

      # Only forward message if gain is different and we have both pols
      if ($new_gain != $current_gain) {

        if ($new_gain >= MAX_GAIN) {
          logMessage(0, "WARN", "Gain is set to Max ".MAX_GAIN);
        }

        if ($new_gain <= MIN_GAIN) {
          logMessage(0, "WARN", "Gain is set to Min ".MIN_GAIN);
        }

        logMessage(2, "INFO", $line." % changes gain from ".$current_gain." to ".$new_gain);

        $cmd = "APSRGAIN ".$abs_chan." ".$pol." ".$new_gain;
        logMessage(1, "INFO", "srv0 <- ".$cmd);

        $result = set_dfb_gain($socket, $abs_chan, $pol, $current_gain, $new_gain);  

        logMessage(1, "INFO", "srv0 -> ".$result);

        if ($pol eq 0) {
          @pol0_gains[$chan] = $result;
        } else {
          @pol1_gains[$chan] = $result;
        }
    
      }
    
    } elsif ($line =~ /^LEVEL (\d) (0|1) (0|1) (-|\d|\.)+$/) {

      # Ignore LEVEL commands - only relevant to CPSR/2

    } else {

      logMessage(0, "WARN", "STDIN: \"$line\" was not a sensible gain input");

    }
  }
}

close $log_fh;
close $log_socket;

exit 0;


#
# Logs a message to the Nexus
#
sub logMessage($$$) {
  my ($level, $type, $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    my $time = Dada->getCurrentDadaTime();
    if (!($log_socket)) {
      $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});
    }
    if ($log_socket) {
      Dada->nexusLogMessage($log_socket, $time, "sys", $type, "gain ctrl", $message);
    }
    $| = 1;
    print $log_fh "[".$time."] ".$message."\n";
  }
}

sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";

  if ($socket) {
    close($socket);
  }
  if ($log_socket) {
    close($log_socket);
  }

  print STDERR basename($0)." : Exiting\n";

  exit 1;

}

sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  close($socket);
  close($log_socket);
  $socket = 0;
  $log_socket = 0;
  $socket = Dada->connectToMachine($cfg{"SERVER_HOST"},$cfg{"SERVER_GAIN_CONTROL_PORT"}, 1);
  $log_socket = Dada->nexusLogOpen($cfg{"SERVER_HOST"},$cfg{"SERVER_SYS_LOG_PORT"});

}

sub usage() {
  
  print "Usage: ".$0." nchan\n";
  print "   chan   Total number of channels on this node\n";
  print "\n";
 
}


sub get_dfb3_gain($$$) {

  (my $socket, my $chan, my $pol) = @_;

  my $cmd = "APSRGAIN ".$chan." ".$pol;

  logMessage(2, "INFO", "srv0 <- ".$cmd);

  print $socket $cmd."\r\n";

  # The DFB3 should return a string along the lines of
  # OK <chan> <pol> <value>

  my $dfb_response = Dada->getLine($socket);

  logMessage(2, "INFO", "srv0 -> ".$dfb_response);

  my ($result, $dfb_chan, $dfb_pol, $dfb_gain) = split(/ /,$dfb_response);

  if ($result ne "OK") {

    logMessage(0, "WARN", "DFB3 returned an error: \"".$dfb_response."\"");
    return ("fail", 0);

  }

  if ($dfb_chan ne $chan) {
    logMessage(0, "WARN", "DFB3 returned an channel mismatch: requested ".$chan.", received ".$dfb_chan);
    return ("fail", 0);
  }

  if ($dfb_pol ne $pol) {
    logMessage(0, "WARN", "DFB3 returned an pol mismatch: requested ".$pol.", received ".$dfb_pol);
    return ("fail", 0);
  }

  return ("ok", $dfb_gain);

}

sub set_dfb_gain($$$$$) {

  my ($socket, $chan, $pol, $curr_val, $val) = @_;

  my $cmd = "APSRGAIN ".$chan." ".$pol." ".$val;

  logMessage(2, "INFO", "srv0 <- ".$cmd);
  
  print $socket $cmd."\r\n";
 
  # The DFB3 should return a string along the lines of
  # OK 

  my $dfb_response = Dada->getLine($socket);

  logMessage(2, "INFO", "srv0 -> ".$dfb_response);

  if ($dfb_response ne "OK") {

    logMessage(0, "WARN", "DFB3 returned an error: \"".$dfb_response);
    return $curr_val;

  } else {

    return $val;
  }

}

                                                                                                                                                                                 


