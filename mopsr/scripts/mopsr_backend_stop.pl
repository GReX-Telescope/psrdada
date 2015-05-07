#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Graceful shutdown of the MOPSR backend base on the configuration
#

use lib $ENV{"DADA_ROOT"}."/bin";

use threads;
use threads::shared;
use strict;
use warnings;
use Getopt::Std;
use Mopsr;

#
# Function Prototypes
#
sub usage();
sub debugMessage($$);

#
# Constants
#
use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared    = Mopsr::getConfig();

#
# Local Variables
#
my $cmd = "";
my $result = "";
my $response = "";

my %opts;
getopts('h', \%opts);

if ($opts{h}) 
{
  usage();
  exit(0);
}
my %aq_cfg = Mopsr::getConfig("aq");
my %bf_cfg = Mopsr::getConfig("bf");
my %bp_cfg = Mopsr::getConfig("bp");


# socket connections to each of the master control scripts
my $num_aq = $aq_cfg{"NUM_PWC"};
my $num_bf = $bf_cfg{"NUM_BF"};
#my $num_bp = $bp_cfg{"NUM_BP"};
my $num_bp = 0;

my @aqs = ();
my @bfs = ();
my @bps = ();

my ($i, $port, $host);

$port = $aq_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_aq; $i++)
{
  $host = $aq_cfg{"PWC_".$i};
  $aqs[$i] = Dada::connectToMachine($host, $port);
  if (!$aqs[$i])
  {
    debugMessage(0, "ERROR: could not connect to AQ[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$port = $bf_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_bf; $i++)
{
  $host = $bf_cfg{"BF_".$i};
  $bfs[$i] = Dada::connectToMachine($host, $port);
  if (!$bfs[$i])
  {
    debugMessage(0, "ERROR: could not connect to BF[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$port = $bp_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_bp; $i++)
{
  $host = $bp_cfg{"PWC_".$i};
  $bps[$i] = Dada::connectToMachine($host, $port);
  if (!$bps[$i])
  {
    debugMessage(0, "ERROR: could not connect to BP[".$i."] ".$host.":".$port);
    exit (1);
  }
}


debugMessage(0, "Performing graceful shutdown");

# stop server TMC interface

# stop AQ daemons

# stop BF daemons

# stop BP daemons

# close socket connections
for ($i=0; $i<$num_aq; $i++)
{
  if ($aqs[$i])
  {
    $aqs[$i]->close();
  }
}

for ($i=0; $i<$num_bf; $i++)
{
  if ($bfs[$i])
  {
    $bfs[$i]->close();
  }
}
for ($i=0; $i<$num_bp; $i++)
{
  if ($bps[$i])
  {
    $bps[$i]->close();
  }
}

exit 0;

###############################################################################
#
# Functions
#

sub usage() {
  print "Usage: ".$0." [options]\n";
  print "   -h     print help text\n";
}

sub debugMessage($$) 
{
  my ($level, $message) = @_;

  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada::getCurrentDadaTime(0)."] ".$message."\n";
  }
}

