#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2013 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# client_mopsr_bf_tb0.pl 
#
# run BF processing engine on single TB
# 
###############################################################################

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use Mopsr;
use client_mopsr_bf_tb;

# 
# Check command line argument
#
if ($#ARGV != 0)
{
  Mopsr::client_mopsr_bf_tb->usage();
  exit(1);
}

Dada::preventDuplicateDaemon(basename($0)." ".$ARGV[0]);

my $bf_id = $ARGV[0];
my $tb_id = 1;

# ensure that our bf_id is valid 
if (($bf_id >= 0) &&  ($bf_id < $Mopsr::client_mopsr_bf_tb::cfg{"NUM_BF"}))
{
  # and matches configured hostname
  if ($Mopsr::client_mopsr_bf_tb::cfg{"BF_".$bf_id} ne Dada::getHostMachineName())
  {
    print STDERR "BF_".$bf_id." did not match configured hostname [".Dada::getHostMachineName()."]\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "bf_id was not a valid integer between 0 and ".($Mopsr::client_mopsr_bf_tb::cfg{"NUM_BF"}-1)."\n";
  usage();
  exit(1);
}

if ($tb_id >= $Mopsr::client_mopsr_bf_tb::cfg{"NUM_TIED_BEAMS"})
{
  printf STDERR "tb_id was >= NUM_TIED_BEAMS\n";
  exit(1);
}


#
# Initialize module variables
#
$Mopsr::client_mopsr_bf_tb::dl = 1;
$Mopsr::client_mopsr_bf_tb::daemon_name = Dada::daemonBaseName($0);
$Mopsr::client_mopsr_bf_tb::bf_id = $bf_id;
$Mopsr::client_mopsr_bf_tb::tb_id = $tb_id;

# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = Mopsr::client_mopsr_bf_tb->main();

exit($result);

