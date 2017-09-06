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

  my %roach = Bpsr::getROACHConfig();
  my $host = Dada::getHostMachineName();
  my $i = 0;
  my $found = 0;

  %Dada::client_master_control::pwcs = ();

  # if running on the server machine
  if ($cfg{"SERVER_ALIASES"} =~ m/$host/)
  {
    $Dada::client_master_control::host = $host;
    $Dada::client_master_control::user = "dada";
    $Dada::client_master_control::daemon_prefix = "server";
    $Dada::client_master_control::pwcs{"server"} = ();
    $Dada::client_master_control::pwcs{"server"}{"daemons"} = [split(/ /,$cfg{"SERVER_DAEMONS"})];
    $Dada::client_master_control::pwcs{"server"}{"dbs"} = ();
    $Dada::client_master_control::pwcs{"server"}{"state"} = "server";
    $Dada::client_master_control::num_pwc = 0;
    $Dada::client_master_control::control_dir = $cfg{"SERVER_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"SERVER_LOG_DIR"};
    push (@Dada::client_master_control::pwcs, "server");
    $found = 1;
  }
  else
  {
    $Dada::client_master_control::host = $host;
    $Dada::client_master_control::daemon_prefix = "client";
    $Dada::client_master_control::control_dir = $cfg{"CLIENT_CONTROL_DIR"};
    $Dada::client_master_control::log_dir = $cfg{"CLIENT_LOG_DIR"};
    $Dada::client_master_control::user = $cfg{"USER"};
    $Dada::client_master_control::num_pwc = $cfg{"NUM_PWC"};

		# Ensure the required directories exist
		my $cmd = "";
		my $dir = "";
		my $result = "";
		my $response = "";
		my @dirs = ($cfg{"CLIENT_LOCAL_DIR"}, $cfg{"CLIENT_CONTROL_DIR"}, $cfg{"CLIENT_LOG_DIR"}, 
							  $cfg{"CLIENT_ARCHIVE_DIR"}, $cfg{"CLIENT_RECORDING_DIR"}, $cfg{"CLIENT_SCRATCH_DIR"});


    for ($i=0; ($i<$cfg{"NUM_PWC"}); $i++)
    {
      if ($host =~ m/$cfg{"PWC_".$i}/)
      {
        my $beam = $roach{"BEAM_".$i};
        print $host." ".$i." matched ".$beam."\n";
        push @dirs, $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam;
      }
    }

		foreach $dir ( @dirs )
		{
			if (! -d $dir)
			{
				$cmd = "mkdir -m 0755 ".$dir;
				($result, $response) = Dada::mySystem($cmd);
				if ($result ne "ok")
				{
					print STDERR "Could not create ".$dir.": ".$response."\n";
					return 0;
				}
			}
		}

    # see how many PWCs are running on this host
    for ($i=0; ($i<$cfg{"NUM_PWC"}); $i++) 
    {
      if ($host =~ m/$cfg{"PWC_".$i}/) 
      {
        # add to list of PWCs on this host
        $Dada::client_master_control::pwcs{$i} = ();

        # hash for the data blocks for this PWC
        $Dada::client_master_control::pwcs{$i}{"dbs"} = ();
        $Dada::client_master_control::pwcs{$i}{"daemons"} = ();
        $Dada::client_master_control::pwcs{$i}{"state"} = "active";

        # determine data blocks for this PWC
        my @ids = split(/ /,$cfg{"DATA_BLOCK_IDS"});
        my $key = "";
        my $id = 0;
        my ($nbufs, $bufsz);
        foreach $id (@ids) 
        {
          $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $i, $cfg{"NUM_PWC"}, $id);
          if (!defined($cfg{"BLOCK_NBUFS_".$id}))
          {
            return 0;
          }
          $nbufs = $cfg{"BLOCK_NBUFS_".$id};

          # if a BUFSZ is specfied use that
          if (defined($cfg{"BLOCK_NBUFS_".$id}))
          {
            $bufsz = $cfg{"BLOCK_BUFSZ_".$id};
          }
          else
          {
            if (!defined($cfg{"BLOCK_NSAMP_".$id}))
            { 
              return 0;
            }
            # compute bufsz from params
            my $nsamp = $cfg{"BLOCK_NSAMP_".$id};
            my $nchan = $cfg{"NCHAN"};
            my $nbit = 8;
            my $npol = 1;
            my $ndim = 2;
            my $nant = ($cfg{"ANT_LAST_SEND_".$i} - $cfg{"ANT_FIRST_SEND_".$i}) + 1;
            my $nbeam = 1;
            $bufsz = Dada::client_master_control::computeDBSize($nsamp, $nchan, $npol, $ndim, $nbit, $nant, $nbeam);
          } 

          # the number of reader for this data block
          my $nread = 1;
          if (defined $cfg{"BLOCK_NREAD_".$id})
          {
            $nread = $cfg{"BLOCK_NREAD_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"nread"} = $nread;

          # if this data block is to be paged into RAM
          my $page = "false";
          if (defined $cfg{"BLOCK_PAGE_".$id})
          {
            $page = $cfg{"BLOCK_PAGE_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"page"} = $page;

          # numa node for this datablock
          my $numa = "-1";
          if (defined $cfg{"BLOCK_NUMA_".$id})
          {
            $numa = $cfg{"BLOCK_NUMA_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"numa"} = $numa;


          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"key"} = $key;
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"nbufs"} = $cfg{"BLOCK_NBUFS_".$id};
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"bufsz"} = $bufsz;
        }

        $Dada::client_master_control::pwcs{$i}{"daemons"} = [split(/ /,$cfg{"CLIENT_DAEMONS"})];
        $found = 1;
      }
    }
  }

  return $found;
}

