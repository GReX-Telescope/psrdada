#!/usr/bin/env perl

###############################################################################
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Mopsr;
use Dada::client_master_control qw(%cfg);

#
# Function prototypes
#
sub setupClientType();
sub usage();

#
# Global Variables
# 
%cfg = Mopsr::getConfig("bp");

#
# Initialize module variables
#
$Dada::client_master_control::dl = 1;
$Dada::client_master_control::daemon_name = Dada::daemonBaseName($0);

my $client_setup = setupClientType();
if (! $client_setup) 
{
  print "ERROR: Failed to setup client type based on mopsr_bp.cfg\n";
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

sub usage()
{
  print STDERR "usage: $0\n";
}


# 
# Determine what type of host we are, and if we have any datablocks
# to monitor
#
sub setupClientType() 
{
  my $host = Dada::getHostMachineName();
  my $i = 0;
  my $found = 0;

  my %bp_ct =Mopsr::getCornerturnConfig("bp");   # read the BP cornerturn

  %Dada::client_master_control::pwcs = ();


  $Dada::client_master_control::host = $host;
  $Dada::client_master_control::daemon_prefix = "client";

  $Dada::client_master_control::control_dir = $cfg{"CLIENT_CONTROL_DIR"};
  $Dada::client_master_control::log_dir     = $cfg{"CLIENT_LOG_DIR"};
  $Dada::client_master_control::user        = "mpsr";
  $Dada::client_master_control::num_pwc     = $cfg{"NUM_BP"};

  # Ensure the required directories exist
  my $cmd = "";
  my $dir = "";
  my $result = "";
  my $response = "";
  my @dirs = ($cfg{"CLIENT_LOCAL_DIR"}, $cfg{"CLIENT_CONTROL_DIR"}, $cfg{"CLIENT_LOG_DIR"});

  for ($i=0; ($i<$cfg{"NUM_BP"}); $i++)
  {
    if ($host =~ m/$cfg{"BP_".$i}/)
    {
      push @dirs, $cfg{"CLIENT_ARCHIVE_DIR"};
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

  # see how many BP's are running on this host
  for ($i=0; ($i<$cfg{"NUM_BP"}); $i++) 
  {
    if ($host =~ m/$cfg{"BP_".$i}/) 
    {
      # add to list of PWCs on this host
      $Dada::client_master_control::pwcs{$i} = ();

      my $state    = $cfg{"BP_STATE_".$i};

      # hash for the data blocks for this PWC
      $Dada::client_master_control::pwcs{$i}{"dbs"} = ();
      $Dada::client_master_control::pwcs{$i}{"daemons"} = ();
      $Dada::client_master_control::pwcs{$i}{"state"} = $state;

      # add to list of PWCs on this host
      push (@Dada::client_master_control::pwcs, $i);

      # only active or passive clients will have DBs
      if (($state eq "active") || ($state eq "passive"))
      {
        # determine data blocks for this PWC
        my @ids = split(/ /,$cfg{"DATA_BLOCK_IDS"});
        my $key = "";
        my $id = 0;
        foreach $id (@ids) 
        {
          $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $i, $cfg{"NUM_BP"}, $id);

          # check nbufs and bufsz
          if (!defined($cfg{"BLOCK_NBUFS_".$id}))
          {
            return 0;
          }
          my $bufsz;
          if (defined($cfg{"BLOCK_BUFSZ_".$id}))
          {
            $bufsz =  $cfg{"BLOCK_BUFSZ_".$id};
          }
          else
          {
            if (!defined($cfg{"BLOCK_NSAMP_".$id}))
            {
              return 0;
            }

            # compute bufsz from params
            my $nsamp = $cfg{"BLOCK_NSAMP_".$id};
            my $nchan = ($bp_ct{"NCHAN_COARSE"} * $bp_ct{"NCHAN_FINE"});
            my $nbeam = ($bp_ct{"BEAM_LAST_RECV_".$i} - $bp_ct{"BEAM_FIRST_RECV_".$i}) + 1;
            my $nbit = 8;
            my $npol = 1;
            my $ndim = 1;
            my $nant = 1;

            $bufsz = Dada::client_master_control::computeDBSize($nsamp, $nchan, $npol, $ndim, $nbit, $nant, $nbeam);  
            print "bufsz=".$bufsz."\n";
          }

          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"key"} = $key;
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"nbufs"} = $cfg{"BLOCK_NBUFS_".$id};
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"bufsz"} = $bufsz;

          my $nread = 1;
          if (defined $cfg{"BLOCK_NREAD_".$id})
          {
            $nread = $cfg{"BLOCK_NREAD_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"nread"} = $nread;

          my $page = "false";
          if (defined $cfg{"BLOCK_PAGE_".$id})
          {
            $page = $cfg{"BLOCK_PAGE_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"page"} = $page;

          my $numa = "-1";
          if (defined $cfg{"BLOCK_NUMA_".$id})
          {
            $numa = $cfg{"BLOCK_NUMA_".$id};
          }
          $Dada::client_master_control::pwcs{$i}{"dbs"}{$id}{"numa"} = $numa;
        } 

        $Dada::client_master_control::pwcs{$i}{"daemons"} = [split(/ /,$cfg{"CLIENT_DAEMONS"})];
      }
      elsif ($state eq "inactive")
      {
        $Dada::client_master_control::pwcs{$i}{"daemons"} = [split(/ /,$cfg{"CLIENT_DAEMONS_INACTIVE"})];
      }
      else
      {
      
      }
      $found = 1;
    }
  }

  return $found;
}
