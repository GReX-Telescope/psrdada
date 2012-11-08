#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Apsr;
use Dada::client_master_control qw(%cfg);

#
# Function prototypes
#
sub setupClientType();


#
# Global Variables
# 
%cfg = Apsr::getConfig();

#
# Initialize module variables
#
$Dada::client_master_control::dl = 1;
$Dada::client_master_control::daemon_name = Dada::daemonBaseName($0);
$Dada::client_master_control::pwc_add = " -d ";

my $client_setup = setupClientType();
if (! $client_setup) 
{
  print "ERROR: Failed to setup client type based on apsr.cfg\n";
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

  $Dada::client_master_control::primary_db = "";
  @Dada::client_master_control::dbs = ();
  @Dada::client_master_control::pwcs = ();

  # if running on the server machine
  if ($cfg{"SERVER_ALIASES"} =~ m/$Dada::client_master_control::host/) {
    Dada::logMsg(2, $Dada::client_master_control::dl, "matched server alias");
    $Dada::client_master_control::host = "srv0";
    $Dada::client_master_control::user = "dada";
    @Dada::client_master_control::daemons = split(/ /,$cfg{"SERVER_DAEMONS"});
    $Dada::client_master_control::daemon_prefix = "server";
    $Dada::client_master_control::control_dir = $cfg{"SERVER_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"SERVER_LOG_DIR"};
    push (@Dada::client_master_control::pwcs, "server");
    $found = 1;
  }
  else
  {
    $Dada::client_master_control::daemon_prefix = "client";
    $Dada::client_master_control::control_dir = $cfg{"CLIENT_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"CLIENT_LOG_DIR"};
    $Dada::client_master_control::user = $cfg{"USER"};

    # see if we are a PWC
    for ($i=0; (($i<$cfg{"NUM_PWC"}) && (!$found)); $i++) 
    {
      if ($Dada::client_master_control::host =~ m/$cfg{"PWC_".$i}/)
      {
        Dada::logMsg(2, $Dada::client_master_control::dl, "matched pwc");

        # add to list of PWCs on this host
        push (@Dada::client_master_control::pwcs, $i);

        # determine data blocks on this host
        my @ids = split(/ /,$cfg{"DATA_BLOCK_IDS"});
        my $key = "";
        my $id = 0;
        foreach $id (@ids)
        {
          $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $i, $id);
          # check nbufs and bufsz
          if ((!defined($cfg{"BLOCK_BUFSZ_".$id})) || (!defined($cfg{"BLOCK_NBUFS_".$id}))) {
            return 0;
          }
          $Dada::client_master_control::dbs{$i}{$id} = $key;
        }

        @Dada::client_master_control::daemons = split(/ /,$cfg{"CLIENT_DAEMONS"});
        @Dada::client_master_control::binaries = ();
        push @Dada::client_master_control::binaries, $cfg{"PWC_BINARY"};
        $found = 1;
      }
    }
    
    # If we matched a PWC
    if ($found) 
    {
      #$Dada::client_master_control::pwc_add = " -D ".$cfg{"CLIENT_LOG_DIR"}."/apsr_udpdb.log -p ".$cfg{"CLIENT_UDPDB_PORT"};
    }
    # check if we are a DFB simulator
    else
    {
      for ($i=0; (($i<$cfg{"NUM_DFB"}) && (!$found)); $i++) {
        if ($Dada::client_master_control::host =~ m/$cfg{"DFB_".$i}/) {
          Dada::logMsg(2, $Dada::client_master_control::dl, "matched dfb simulator");
          @Dada::client_master_control::daemons = ();
          @Dada::client_master_control::binaries = ();
          $found = 1;
        }
      }
    }
  }

  return $found;

}

