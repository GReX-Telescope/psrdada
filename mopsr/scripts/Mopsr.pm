package Mopsr;

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
  &clientCommand
  &getObsDestinations
  &getConfig
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant DEBUG_LEVEL  => 0;



sub logMessage($$) 
{
  my ($level, $msg) = @_;
  if ($level <= 2)
  {
     print "[".Dada::getCurrentDadaTime(0)."] ".$msg."\n";
  }
}

sub clientCommand($$)
{
  my ($command, $machine) = @_;

  my %cfg = Mopsr::getConfig();
  my $result = "fail";
  my $response = "Failure Message";

  my $handle = Dada::connectToMachine($machine, $cfg{"CLIENT_MASTER_PORT"}, 0);
  # ensure our file handle is valid
  if (!$handle) {
    return ("fail","Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"});
  }

  ($result, $response) = Dada::sendTelnetCommand($handle,$command);

  $handle->close();

  return ($result, $response);

}

# Return the destinations that an obs with the specified PID should be sent to
sub getObsDestinations($$) {
  
  my ($obs_pid, $dests) = @_;
  
  my $want_swin = 0;
  my $want_parkes = 0;
  
  if ($dests =~ m/swin/) {
    $want_swin = 1;
  }
  if ($dests =~ m/parkes/) {
    $want_parkes = 1;
  }

  return ($want_swin, $want_parkes);

}

sub getConfig(;$) 
{
  (my $sub_type) = @_;
  if ($sub_type eq "")
  {
    $sub_type = "aq";
  }

  my $config_file = $DADA_ROOT."/share/mopsr.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);

  my $ct_config_file = $DADA_ROOT."/share/mopsr_cornerturn.cfg";
  my %ct_config = Dada::readCFGFileIntoHash($ct_config_file, 0);

  my %combined = (%config, %ct_config);

  my $sub_config_file = $DADA_ROOT."/share/mopsr_".$sub_type.".cfg";
  my %sub_config = Dada::readCFGFileIntoHash($sub_config_file, 0);
  %combined = (%combined, %sub_config);

  return %combined;
}

__END__
