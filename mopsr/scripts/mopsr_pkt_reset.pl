#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Performs a hard stop of the BPSR Instrument
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

if ($opts{h}) {
  usage();
  exit(0);
}

debugMessage(0, "Resetting MOPSR Packet Counter");

$cmd = "echo 'UNKNOWN' > ".$cfg{"CONFIG_DIR"}."/mopsr.pkt_utc_start";
debugMessage(0, $cmd);
($result, $response) = Dada::mySystem($cmd);

exit 0;

###############################################################################
#
# Functions
#

sub usage() {
  print "Usage: ".$0." [options]\n";
  print "   -h     print help text\n";
}

sub debugMessage($$) {
  my ($level, $message) = @_;

  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada::getCurrentDadaTime(0)."] ".$message."\n";
  }
}
