#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2010 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Performs a hard stop of the BPSR Instrument
#

use lib $ENV{"DADA_ROOT"}."/bin";

use threads;
use threads::shared;
use strict;
use warnings;
use Getopt::Std;
use Bpsr;

#
# Function Prototypes
#
sub usage();
sub debugMessage($$);
sub killServerDaemons($);
sub killClientDaemons($$);
sub destroyClientSpecial($);

#
# Constants
#
use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared    = Bpsr::getConfig();
our @server_daemons  = split(/ /,$cfg{"SERVER_DAEMONS"});
our @client_daemons  = split(/ /,$cfg{"CLIENT_DAEMONS"});


#
# Local Variables
#
my $cmd = "";
my $result = "";
my $response = "";
my @clients = ();
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

debugMessage(0, "Hard Stopping BPSR");

push @server_daemons, "bpsr_master_control";
push @client_daemons, "bpsr_master_control";

# Generate hosts lists
for ($i=0; $i < $cfg{"NUM_PWC"}; $i++) {
  push(@clients, $cfg{"PWC_".$i});
  push(@all, $cfg{"PWC_".$i});
}
for ($i=0; $i < $cfg{"NUM_SRV"}; $i++) {
  push(@servers, $cfg{"SRV_".$i});
  push(@all, $cfg{"SRV_".$i});
}

# ensure we dont have duplicates
@clients = Dada::array_unique(@clients);
@all = Dada::array_unique(@all);


# stop all scripts running on the clients 
debugMessage(0, "Killing Client Daemons");
($result, $response) = killClientDaemons(\@clients, \@client_daemons);

# kill PWC_BINARY (bpsr_udpdb) and shared memory segments on clients
debugMessage(0, "Destroying Client SHM");
($result, $response) = destroyClientSpecial(\@clients);

# stop all scripts running on the server
debugMessage(0, "Killing Server Daemons");
($result, $response) = killServerDaemons(\@server_daemons);

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
  my $control_dir;

  for ($i=0; $i<=$#daemons; $i++)
  {
    if ($daemons[$i] eq "bpsr_master_control")
    {  
      $control_dir = $cfg{"CLIENT_CONTROL_DIR"};
    }
    else
    {
      $control_dir = $cfg{"SERVER_CONTROL_DIR"};
    }
    $cmd = "touch ".$control_dir."/".$daemons[$i].".quit";
    debugMessage(2, "killServerDaemons: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    debugMessage(2, "killServerDaemons: ".$result." ".$response);
  }

  sleep(5);

  for ($i=0; $i<=$#daemons; $i++) 
  {
    if ($daemons[$i] eq "bpsr_master_control")
    {
      $d = "client_".$daemons[$i].".pl";
      $control_dir = $cfg{"CLIENT_CONTROL_DIR"};
    }
    else
    {
      $d = $prefix."_".$daemons[$i].".pl";
      $control_dir = $cfg{"SERVER_CONTROL_DIR"};
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
      debugMessage(2, "killServerDaemons: killProcess(^perl.*".$d.")");
      ($result, $response) = Dada::killProcess("^perl.*".$d);
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

  my $u = "bpsr";
  my $d = "";
  my $h = "";
  my $result = "";
  my $response = "";
  my $rval = "";
  my $cmd = "";

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];
    $cmd = "touch";

    for ($j=0; $j<=$#daemons; $j++)
    {
      $cmd .= " ".$control_dir."/".$daemons[$j].".quit";
    }
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $rval, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);
  }

  sleep(10);

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];

    for ($j=0; $j<=$#daemons; $j++)
    {
      $d = $prefix."_".$daemons[$j].".pl";

      # get the pid of the daemon
      $cmd = "pgrep -u ".$u." -f '^perl.*".$d."'";
      debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
      ($result, $rval, $response) = Dada::remoteSshCommand($u, $h, $cmd);
      debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

      if ($result ne "ok")
      {
        debugMessage(1, "killClientDaemons: could not determine PID for ".$d);
      }
      elsif ($response eq "") 
      {
        debugMessage(2, "killClientDaemons: ".$d." not running");
      }
      else
      {
        $cmd = "pkill -KILL -u ".$u." -f '^perl.*".$d."'";
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
    
  my $u = "bpsr";
  my $h = "";
  my $result = "";
  my $response = "";
  my $rval = "";
  my $cmd = "";
  my $i = 0;
  my $j = 0;

  for ($i=0; $i<=$#hosts; $i++)
  {
    $h = $hosts[$i];

    $cmd = "killall -KILL ".$cfg{"PWC_BINARY"};
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "killall -KILL dspsr";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "killall -KILL dada_dbnull";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "killall -KILL dada_dbevent";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    $cmd = "echo destroy_dbs";

    for ($j=0; ($j<$cfg{"NUM_PWC"}); $j++)
    {
      if ($h =~ m/$cfg{"PWC_".$j}/)
      {
        # determine data blocks for this PWC
        my @ids = split(/ /,$cfg{"DATA_BLOCK_IDS"});
        my $key = "";
        my $id = 0;
        foreach $id (@ids)
        {
          $key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $j, $id);
          # check nbufs and bufsz
          
          $cmd .= "; dada_db -d -k ".$key;
        }
      }
    }

    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);

    # and just to be safe :)
    $cmd = "ipcrme";
    debugMessage(2, "killClientDaemons: remoteSshCommand(".$u.", ".$h.", ".$cmd.")");
    ($result, $response) = Dada::remoteSshCommand($u, $h, $cmd);
    debugMessage(2, "killClientDaemons: ".$result." ".$rval." ".$response);


  }

  return ("ok", "all nuked");
}
