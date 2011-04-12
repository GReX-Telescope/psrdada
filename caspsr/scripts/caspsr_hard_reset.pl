#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Performs a hard stop of the CASPSR Instrument
#

use lib $ENV{"DADA_ROOT"}."/bin";

use threads;
use threads::shared;
use strict;
use warnings;
use Getopt::Std;
use Caspsr;

#
# Function Prototypes
#
sub usage();
sub debugMessage($$);
sub killServerDaemons($);
sub killClientDaemons($$);
sub destroyClientSpecial($);
sub destroyDemuxerSpecial($);

#
# Constants
#
use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared    = Caspsr::getConfig();
our @server_daemons  = split(/ /,$cfg{"SERVER_DAEMONS"});
our @client_daemons  = split(/ /,$cfg{"CLIENT_DAEMONS"});
our @demux_daemons = split(/ /,$cfg{"DEMUX_DAEMONS"});


#
# Local Variables
#
my $cmd = "";
my $result = "";
my $response = "";
my @clients = ();
my @demuxers = ();
my @all_clients = ();
my @servers = ();
my @all = ();
my $i=0;
my $j=0;
my $start=1;
my $stop=1;

my %opts;
getopts('h', \%opts);

if ($opts{h}) {
  usage();
  exit(0);
}

debugMessage(0, "Hard Stopping CASPSR");

push @server_daemons, "caspsr_master_control";
push @client_daemons, "caspsr_master_control";
push @demux_daemons, "caspsr_master_control";

# Setup directories should they not exist
my $control_dir = $cfg{"SERVER_CONTROL_DIR"};

if (! -d $control_dir) {
  system("mkdir -p ".$control_dir);
}

# Generate hosts lists
for ($i=0; $i < $cfg{"NUM_PWC"}; $i++) {
  push(@clients, $cfg{"PWC_".$i});
  push(@all_clients, $cfg{"PWC_".$i});
  push(@all, $cfg{"PWC_".$i});
}
for ($i=0; $i < $cfg{"NUM_DEMUX"}; $i++) {
  for ($j=0; $j<=$#demuxers; $j++) {
  }
  push(@demuxers, $cfg{"DEMUX_".$i});
  push(@all_clients, $cfg{"DEMUX_".$i});
  push(@all, $cfg{"DEMUX_".$i});
}
for ($i=0; $i < $cfg{"NUM_SRV"}; $i++) {
  push(@servers, $cfg{"SRV_".$i});
  push(@all, $cfg{"SRV_".$i});
}


# ensure we dont have duplicates [from DEMUXS especially]
@clients = Dada::array_unique(@clients);
@all_clients = Dada::array_unique(@all_clients);
@all = Dada::array_unique(@all);
@demuxers = Dada::array_unique(@demuxers);

# stop all scripts running on the server
debugMessage(0, "Killing Server Daemons");
($result, $response) = killServerDaemons(\@server_daemons);

# stop all scripts running on the demuxers
debugMessage(0, "Killing Demuxer Daemons");
($result, $response) = killClientDaemons(\@demuxers, \@demux_daemons);

# stop all scripts running on the gpus
debugMessage(0, "Killing GPU Daemons");
($result, $response) = killClientDaemons(\@clients, \@client_daemons);

# kill DEMUX_BINARY (caspsr_udpNnic)
debugMessage(0, "Destroying Demuxer SHM");
($result, $response) = destroyDemuxerSpecial(\@demuxers);

# kill PWC_BINARY (caspsr_udpdn) and shared memory segments on gpus
debugMessage(0, "Destroying GPU SHM");
($result, $response) = destroyClientSpecial(\@clients);


# Clear the web interface status directory
my $dir = $cfg{"STATUS_DIR"};
if (-d $dir) {

  $cmd = "rm -f ".$dir."/*.error";
  debugMessage(0, "clearing .error from status_dir");
  ($result, $response) = Dada::mySystem($cmd);

  $cmd = "rm -f ".$dir."/*.warn";
  debugMessage(0, "clearing .warn from status_dir");
  ($result, $response) = Dada::mySystem($cmd);
}

exit 0;

###############################################################################
#
# Functions
#


sub usage() {
  print "Usage: ".$0." [options]\n";
  print "   -h     print help text\n";
}

sub debugMessage($$) {
  my ($level, $message) = @_;

  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada::getCurrentDadaTime(0)."] ".$message."\n";
  }
}

#
# Kill all specified daemons
#
sub killServerDaemons($) 
{

  my ($daemonsRef) = @_;

  my $prefix = "server";
  my @daemons = @$daemonsRef;
  my $d = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $control_dir = $cfg{"SERVER_CONTROL_DIR"};

  for ($i=0; $i<=$#daemons; $i++) 
  {
    if ($daemons[$i] eq "caspsr_master_control")
    {
      $d = "client_".$daemons[$i].".pl";
    }
    else
    {
      $d = $prefix."_".$daemons[$i].".pl";
    }

    # get the pid of the daemon
    $cmd = "ps auxwww | grep perl | grep '".$d."' | grep -v grep | awk '{print \$2}'";
    debugMessage(2, "killServerDaemons: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    debugMessage(2, "killServerDaemons: ".$result." ".$response);

    if ($result ne "ok")
    {
      debugMessage(1, "killServerDaemons: could not determine PID for ".$d);
    }
    else
    {
      debugMessage(2, "killServerDaemons: killProcess(".$d.")");
      ($result, $response) = Dada::killProcess($d);
      debugMessage(2, "killServerDaemons: ".$result." ".$response);
    }

    if ( -f $control_dir."/".$daemons[$i].".quit" )
    {
      debugMessage(2, "killServerDaemons: unlinking ".$control_dir."/".$daemons[$i].".quit");
      unlink ($control_dir."/".$daemons[$i].".quit");
    }

    if ( -f $control_dir."/".$daemons[$i].".pid" )
    {
      debugMessage(2, "killServerDaemons: unlinking ".$control_dir."/".$daemons[$i].".pid");
      unlink ($control_dir."/".$daemons[$i].".pid");
    }
  } 

  return ("ok", "all stopped");

}

#
# Kill all specified daemons on remote hosts
#
sub killClientDaemons($$) 
{

  my ($hostsRef, $daemonsRef) = @_;

  my $prefix = "client";
  my @hosts = @$hostsRef;
  my @daemons = @$daemonsRef;
  my $control_dir = $cfg{"CLIENT_CONTROL_DIR"};

  my $u = "caspsr";
  my $d = "";
  my $h = "";
  my $result = "";
  my $response = "";
  my $rval = "";
  my $cmd = "";

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];

    for ($j=0; $j<=$#daemons; $j++)
    {
      $d = $prefix."_".$daemons[$j].".pl";

      # get the pid of the daemon
      $cmd = "ps auxwww | grep perl | grep '".$d."' | grep -v grep | awk '{print \$2}'";
      debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($u, $h, $cmd);
      debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

      if ($result ne "ok")
      {
        debugMessage(1, "killClientDaemons: could not determine PID for ".$d);
      }
      else
      {
        $cmd = "kill -KILL ".$response;
        debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
        ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
        debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);
      }

      $cmd = "rm -f ".$control_dir."/".$daemons[$j].".quit ".$control_dir."/".$daemons[$j].".pid";
      debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
      ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
      debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    }
  }

  return ("ok", "all stopped");

}

#
# Destroys all shared memory arrays on the client machines
#
sub destroyClientSpecial($)
{
  my ($hostsRef) = @_;

  my @hosts = @$hostsRef;
    
  my $u = "caspsr";
  my $h = "";
  my $result = "";
  my $response = "";
  my $rval = "";
  my $cmd = "";

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];

    $cmd = "sudo killall ".$cfg{"PWC_BINARY"};
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "sudo killall dspsr";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "sudo killall caspsr_dbdecidb";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "sudo /home/dada/linux_64/bin/dada_db -d";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

  }

  return ("ok", "all nuked");
}

#
# Destroys udpNnic on client machines
# 
sub destroyDemuxerSpecial($)
{ 
  my ($hostsRef) = @_;

  my @hosts = @$hostsRef;
   
  my $u = "caspsr";
  my $h = "";
  my $result = "";
  my $response = "";
  my $rval = "";
  my $cmd = "";

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];
  
    $cmd = "sudo killall ".$cfg{"DEMUX_BINARY"};
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);
  
    $cmd = "sudo killall ".$cfg{"IB_ACTIVE_BINARY"};
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);
  
    $cmd = "sudo killall ".$cfg{"IB_INACTIVE_BINARY"};
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);
  
  } 

  return ("ok", "all nuked");
}  
