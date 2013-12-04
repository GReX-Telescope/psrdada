#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Caspsr;
use Dada::client_master_control qw(%cfg);

#
# Function prototypes
#
sub setupClientType();


#
# Global Variables
# 
%cfg = Caspsr::getConfig();

#
# Initialize module variables
#
$Dada::client_master_control::dl = 1;
$Dada::client_master_control::daemon_name = Dada::daemonBaseName($0);
$Dada::client_master_control::pwc_add = " -d ";

my $client_setup = setupClientType();
if (! $client_setup) 
{
  print "ERROR: Failed to setup client type based on caspsr.cfg\n";
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
    $Dada::client_master_control::host = "srv0";
    $Dada::client_master_control::user = "dada";
    @Dada::client_master_control::daemons = split(/ /,$cfg{"SERVER_DAEMONS"});
    $Dada::client_master_control::daemon_prefix = "server";
    $Dada::client_master_control::control_dir = $cfg{"SERVER_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"SERVER_LOG_DIR"};
    push (@Dada::client_master_control::pwcs, "server");
    $found = 1;
  }
  # check for PWC or DEMUX
  else
  {
    $Dada::client_master_control::daemon_prefix = "client";
    $Dada::client_master_control::control_dir = $cfg{"CLIENT_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"CLIENT_LOG_DIR"};
    $Dada::client_master_control::user = $cfg{"USER"};

    my $index = -1;
    # see if we are a PWC
    for ($i=0; (($i<$cfg{"NUM_PWC"}) && (!$found)); $i++) 
    {
      if ($Dada::client_master_control::host =~ m/$cfg{"PWC_".$i}/)
      {
        $found = 1;
        $index = $i;
        Dada::logMsg(2, $Dada::client_master_control::dl, "matched pwc");

        @Dada::client_master_control::daemons = split(/ /,$cfg{"CLIENT_DAEMONS"});
        @Dada::client_master_control::binaries = ();
        push @Dada::client_master_control::binaries, $cfg{"PWC_BINARY"};

        # add to list of PWCs on this host
        push (@Dada::client_master_control::pwcs, $i);

        $Dada::client_master_control::pwc_add = " -d ";
        my $add_string = "";

        # Raw IB mode
        if (defined $cfg{"IB_CHUNK_SIZE"}) {
          $add_string = " -p ".$cfg{"DEMUX_IB_PORT_0"}." -C ".$cfg{"IB_CHUNK_SIZE"}." ".$cfg{"NUM_DEMUX"};
        } else {
          $add_string = " -p ".$cfg{"CLIENT_UDPDB_PORT"}." ".$i." ".$cfg{"NUM_RECV"}." ".($cfg{"NUM_DEMUX"} * $cfg{"PKTS_PER_XFER"})." 0";
        }

        if ($add_string ne "") {
          $Dada::client_master_control::pwc_add .= " ".$add_string;
        }
      }
    }

    for ($i=0; (($i<$cfg{"NUM_DEMUX"}) && (!$found)); $i++) 
    {
      if ($Dada::client_master_control::host =~ m/$cfg{"DEMUX_".$i}/)
      {
        $found = 1;
        $index = $i;

        @Dada::client_master_control::daemons = split(/ /,$cfg{"DEMUX_DAEMONS"});
        @Dada::client_master_control::binaries = ();
        push (@Dada::client_master_control::pwcs, $i);

        # rewrite configuration datablocks
        $cfg{"DATA_BLOCK_PREFIX"} = $cfg{"DEMUX_BLOCK_PREFIX"};
        $cfg{"DATA_BLOCK_IDS"}    = $cfg{"DEMUX_BLOCK_IDS"};
        my @dbs = split(/ /,$cfg{"DATA_BLOCK_IDS"});
        my $j = 0;
        for ($j=0; $j<=$#dbs; $j++)
        {
          $cfg{"BLOCK_NBUFS_".$j} = $cfg{"DEMUX_BLOCK_NBUFS_".$j};
          $cfg{"BLOCK_BUFSZ_".$j} = $cfg{"DEMUX_BLOCK_BUFSZ_".$j};
          $cfg{"BLOCK_NREAD_".$j} = $cfg{"DEMUX_BLOCK_NREAD_".$j};
        }
      }
    }

    # if we are a PWC or DEMUX, setup relevant datablocks
    if ($found)
    {
      # determine data blocks on this host
      my @ids = split(/ /,$cfg{"DATA_BLOCK_IDS"});
      my $key = "";
      my $id = 0;
      foreach $id (@ids)
      {
        $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $index, $id);
        # check nbufs and bufsz
        if ((!defined($cfg{"BLOCK_BUFSZ_".$id})) || (!defined($cfg{"BLOCK_NBUFS_".$id}))) {
          return 0;
        }
        $Dada::client_master_control::dbs{$index}{$id} = $key;
      }
    }
  }
  # see if we are a RAID server 
  if (!$found && (($Dada::client_master_control::host eq "raid0") ||
                   $Dada::client_master_control::host eq "caspsr-raid0"))
  {
    $found = 1;
    $Dada::client_master_control::daemon_prefix = "raid";
  }

  return $found;
}

