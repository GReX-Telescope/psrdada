#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Caspsr;
use Dada::client_logger qw(%cfg);


#
# Global Variable Declarations
#
%cfg = Caspsr::getConfig();


#
# Initialize module variables
#
$Dada::client_logger::dl = 1;
$Dada::client_logger::log_host = $cfg{"SERVER_HOST"};
$Dada::client_logger::log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$Dada::client_logger::log_sock = 0;
$Dada::client_logger::daemon_name = Dada::daemonBaseName($0);
$Dada::client_logger::tag = "src";
$Dada::client_logger::daemon = "proc";

# Get command line
if ($#ARGV==0) {
  (my $new_daemon) = @ARGV;    
  $Dada::client_logger::daemon = $new_daemon;
  if ($new_daemon =~ m/demux/) {
    $Dada::client_logger::log_port = $cfg{"SERVER_DEMUX_LOG_PORT"};
  }
}

# Autoflush STDOUT and STDERR
my $ofh = select STDOUT;
$| = 1;
select STDERR;
$| = 1;
select $ofh;

my $result = 0;
$result = Dada::client_logger->main();

exit($result);
