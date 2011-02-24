#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Bpsr;
use Dada::client_master_control qw(%cfg);

#
# Function prototypes
#
sub setupClientType();


#
# Global Variables
# 
%cfg = Bpsr::getConfig();

#
# Initialize module variables
#
$Dada::client_master_control::dl = 1;
$Dada::client_master_control::daemon_name = Dada::daemonBaseName($0);
$Dada::client_master_control::pwc_add = " -d ";

my $client_setup = setupClientType();
if (! $client_setup) 
{
  print "ERROR: Failed to setup client type based on bpsr.cfg\n";
  exit(1);
}

# Autoflush STDOUT, making it "hot"
select STDOUT;
$| = 1;

my $result = 0;
$result = Dada::client_master_control->main();

exit($result);


###############################################################################
#
# Funtions


# 
# Determine what type of host we are, and if we have any datablocks
# to monitor
#
sub setupClientType() 
{

  $Dada::client_master_control::host = Dada::getHostMachineName();
  my $i = 0;
  my $found = 0;

  # if running on the server machine
  if ($cfg{"SERVER_ALIASES"} =~ m/$Dada::client_master_control::host/) {
    $Dada::client_master_control::host = "srv0";
    @Dada::client_master_control::daemons = split(/ /,$cfg{"SERVER_DAEMONS"});
    @Dada::client_master_control::helper_daemons = split(/ /,$cfg{"SERVER_DAEMONS_PERSIST"});
    @Dada::client_master_control::dbs = ();
    $Dada::client_master_control::daemon_prefix = "server";
    $Dada::client_master_control::control_dir = $cfg{"SERVER_CONTROL_DIR"};
    $found = 1;
  }
  else
  {
    $Dada::client_master_control::daemon_prefix = "client";
    $Dada::client_master_control::control_dir = $cfg{"CLIENT_CONTROL_DIR"};

    # see if we are a PWC
    for ($i=0; (($i<$cfg{"NUM_PWC"}) && (!$found)); $i++) {
      if ($Dada::client_master_control::host =~ m/$cfg{"PWC_".$i}/) {
        @Dada::client_master_control::daemons = split(/ /,$cfg{"CLIENT_DAEMONS"});
        @Dada::client_master_control::dbs = split(/ /,$cfg{"DATA_BLOCKS"});
        @Dada::client_master_control::binaries = ();
        push @Dada::client_master_control::binaries, $cfg{"PWC_BINARY"};
        $found = 1;
      }
    }
    
    # If we matched a PWC
    if ($found) 
    {
      $Dada::client_master_control::pwc_add = " -d -i ".$cfg{"PWC_IFACE"};
    }
    # check if we are a helper
    else
    {
      for ($i=0; (($i<$cfg{"NUM_HELP"}) && (!$found)); $i++) {
        if ($Dada::client_master_control::host =~ m/$cfg{"HELP_".$i}/) {
          @Dada::client_master_control::daemons = ();
          @Dada::client_master_control::dbs = ();
          @Dada::client_master_control::binaries = ();
          $found = 1;
        }
      }
    }
  }

  return $found;

}

