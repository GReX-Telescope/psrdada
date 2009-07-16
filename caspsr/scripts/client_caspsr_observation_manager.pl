#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;
use Caspsr;
use Dada::client_observation_manager qw(%cfg);


#
# Global Variables
# 
%cfg = Caspsr->getConfig();

#
# Initialize module variables
#
$Dada::client_observation_manager::dl = 2;
$Dada::client_observation_manager::log_host = $cfg{"SERVER_HOST"};
$Dada::client_observation_manager::log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$Dada::client_observation_manager::log_sock = 0;
$Dada::client_observation_manager::daemon_name = Dada->daemonBaseName($0);
$Dada::client_observation_manager::dada_header_cmd = "dada_header -k deda";
$Dada::client_observation_manager::gain_controller = "client_caspsr_gain_controller.pl";
$Dada::client_observation_manager::client_logger = "client_caspsr_sys_logger.pl";


# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = Dada::client_observation_manager->main();

exit($result);

