#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Mopsr;
use Dada::server_general_dir_archiver qw(%cfg);

#
# Global Variable Declarations
#
%cfg = Mopsr::getConfig();

sub usage() {
  print STDERR "Usage: ".$0."\n";
}

#
# Initialize module variables
#
$Dada::server_general_dir_archiver::dl = 1;
$Dada::server_general_dir_archiver::daemon_name = Dada::daemonBaseName($0);
$Dada::server_general_dir_archiver::robot = 0;
$Dada::server_general_dir_archiver::drive_id = 0;
$Dada::server_general_dir_archiver::type = "swin";
$Dada::server_general_dir_archiver::pid = "SMIRF";
$Dada::server_general_dir_archiver::required_host = "mpsr-bf08.obs.molonglo.local";

# Autoflush STDOUT
$| = 1;

if ($#ARGV != -1) {
  usage();
  exit(1);
}

my $result = 0;

$result = Dada::server_general_dir_archiver->main();

exit($result);

