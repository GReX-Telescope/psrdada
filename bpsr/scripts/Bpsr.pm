package Bpsr;

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Math::BigInt;
use Math::BigFloat;
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);
use Dada;

require Exporter;
require AutoLoader;

@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &getBpsrConfig
  &getBpsrConfigFile
  &waitForMultibobState
  &getMultibobState
  &set_ibob_levels
  &start_ibob
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant DEBUG_LEVEL  => 0;


sub getBpsrConfig() {
  my $config_file = getBpsrCFGFile();
  my %config = Dada->readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub getBpsrCFGFile() {
  return $DADA_ROOT."/share/bpsr.cfg";
}

sub waitForMultibobState($$$$) {

  (my $module, my $stateString, my $handle, my $Twait) = @_;

  my $pwc;
  my @pwcs;
  my $myready = "no";
  my $counter = $Twait;
  my $i=0;

  if (DEBUG_LEVEL >= 1) {
    print $stateString." ".$Twait."\n";
  }
  while (($myready eq "no") && ($counter > 0)) {


    if ($counter == $Twait) {
      ;
    } elsif ($counter == ($Twait-1)) {
      if (DEBUG_LEVEL >= 1) { print STDERR "Waiting for ibobs to become  $stateString."; }
    } else {
      if (DEBUG_LEVEL >= 1) { print STDERR "."; }
    }

    $myready = "yes";

    (@pwcs) = getMultibobState("Bpsr", $handle);

    for ($i=0; $i<=$#pwcs;$i++) {
      $pwc = @pwcs[$i];
      if ($pwc ne $stateString) {
        if (DEBUG_LEVEL >= 1) {
          print "Waiting for IBOB".$i." to transition to ".$stateString."\n";
        }
        $myready = "no";
      }
    }

    sleep 1;
    $counter--;
  }

  if (($counter+1) != $Twait) {
    if (DEBUG_LEVEL >= 0) { print STDERR "\n"; }
  }

  if ($myready eq "yes") {
    return ("ok", "");
  } else {
    return ("fail", "");
  }

}

sub getMultibobState($$) {

  (my $module, my $handle) = @_;
  my $result = "fail";
  my $response = "";

  ($result, $response) = Dada->sendTelnetCommand($handle,"state");

  if ($result eq "ok") {
    #Parse the $response;
    my @array = split('\n',$response);
    my $line;
    my @temp_array;

    my @pwcs;
    foreach $line (@array) {

      if (index($line,"> ") == 0) {
        $line = substr($line,2);
      }

      if (index($line,"IBOB") == 0) {
        @temp_array = split(" ",$line);
        push (@pwcs, $temp_array[2]);
      }
    }

    return (@pwcs);
  } else {
    return 0;
  }

}


__END__
