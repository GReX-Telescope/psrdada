#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Mopsr;
use Dada::client_logger qw(%cfg);


#
# Function prototypes
#
sub usage();

#
# Global Variable Declarations
#
%cfg = Mopsr::getConfig();


#
# Initialize module variables
#
$Dada::client_logger::dl = 1;
$Dada::client_logger::log_host = $cfg{"SERVER_HOST"};
$Dada::client_logger::log_port = $cfg{"SERVER_BF_SRC_LOG_PORT"};
$Dada::client_logger::log_sock = 0;
$Dada::client_logger::daemon_name = Dada::daemonBaseName($0);
$Dada::client_logger::type = "src";
$Dada::client_logger::daemon = "proc";

# Autoflush STDOUT
# $| = 1;

# Autoflush STDOUT and STDERR
my $ofh = select STDOUT;
$| = 1;
select STDERR;
$| = 1;
select $ofh;

{
  # parse command line
  if ($#ARGV != 1) 
  {
    usage();
    exit(1);
  }

  $Dada::client_logger::pwc_id = $ARGV[0];
  $Dada::client_logger::daemon = $ARGV[1];

  my $result = 0;
  $result = Dada::client_logger->main();

  exit($result);
}

sub usage()
{
  print STDERR "Usage: ".Dada::daemonBaseName($0)." chan_id tag\n";
  print STDERR "   chan_id   integer of the RECV / Channel\n";
  print STDERR "   daemon    short name of program generating output\n";
}


