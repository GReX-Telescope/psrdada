#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Correct Startup of the MOPSR backend based on the configuration
#

use lib $ENV{"DADA_ROOT"}."/bin";

use threads;
use threads::shared;
use strict;
use Getopt::Std;
use Mopsr;

#
# Function Prototypes
#
sub usage();
sub debugMessage($$);
sub tc($$);
#sub threadedTelnetCommand($\@);

#
# Constants
#
use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared    = Mopsr::getConfig();

#
# Local Variables
#
my $cmd = "";
my $result = "";
my $response = "";

my %opts;
getopts('h', \%opts);

if ($opts{h}) 
{
  usage();
  exit(0);
}
my %aq_cfg = Mopsr::getConfig("aq");
my %bf_cfg = Mopsr::getConfig("bf");
my %bp_cfg = Mopsr::getConfig("bp");

# socket connections to each of the master control scripts
my $num_aq = 0;
my $num_bf = 0;
my $num_bp = 0;

my %aq_hosts_counts = ();
my %bf_hosts_counts = ();
my %bp_hosts_counts = ();
my @bp_hosts = ();
my @aq_hosts = ();
my @bf_hosts = ();
my @bps = ();
my @aqs = ();
my @bfs = ();
my $srv;

my ($i, $port, $host);

# determine number of AQ nodes
for ($i=0; $i<$aq_cfg{"NUM_PWC"}; $i++)
{
  $host = $aq_cfg{"PWC_".$i};
  if (!defined ($aq_hosts_counts{$host}))
  {
    $aq_hosts_counts{$host} = 0;
  }
  $aq_hosts_counts{$host} += 1;
  debugMessage(2, "aq_hosts_counts{".$host."}=". $aq_hosts_counts{$host});
}
@aq_hosts = sort keys %aq_hosts_counts;
$num_aq = $#aq_hosts + 1;

$port = $bf_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$bf_cfg{"NUM_BF"}; $i++)
{
  $host = $bf_cfg{"BF_".$i};
  if (!defined ($bf_hosts_counts{$host}))
  {
    $bf_hosts_counts{$host} = 0;
  }
  $bf_hosts_counts{$host} += 1;
}
@bf_hosts = sort keys %bf_hosts_counts;
$num_bf = $#bf_hosts + 1;


$port = $bp_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$bp_cfg{"NUM_BP"}; $i++)
{
  $host = $bp_cfg{"BP_".$i};
  if (!defined ($bp_hosts_counts{$host}))
  {
    $bp_hosts_counts{$host} = 0;
  }
  $bp_hosts_counts{$host} += 1;
}
@bp_hosts = sort keys %bp_hosts_counts;
$num_bp = $#bp_hosts + 1;

$cmd = "ssh dada\@mpsr-srv0 client_mopsr_master_control.pl";
($result, $response) = Dada::mySystem($cmd);


for ($i=0; $i<$num_aq; $i++)
{
  $host = $aq_hosts[$i];
  debugMessage(1, "starting AQ master control on ".$host);
  $cmd = "ssh mpsr@".$host." client_mopsr_master_control.pl";
  ($result, $response) = Dada::mySystem($cmd);
}
for ($i=0; $i<$num_bf; $i++)
{
  $host = $bf_hosts[$i];
  debugMessage(1, "starting BF master control on ".$host);
  $cmd = "ssh mpsr@".$host." client_mopsr_bf_master_control.pl";
  ($result, $response) = Dada::mySystem($cmd);
}
for ($i=0; $i<$num_bp; $i++)
{
  $host = $bp_hosts[$i];
  debugMessage(1, "starting BP master control on ".$host);
  $cmd = "ssh mpsr@".$host." client_mopsr_bp_master_control.pl";
  ($result, $response) = Dada::mySystem($cmd);
}

$port = $aq_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_aq; $i++)
{
  $host = $aq_hosts[$i];
  debugMessage(1, "connecting to AQ[".$i."] ".$host.":".$port);
  $aqs[$i] = Dada::connectToMachine($host, $port);
  if (!$aqs[$i])
  {
    debugMessage(0, "ERROR: could not connect to AQ[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$port = $bf_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_bf; $i++)
{
  $host = $bf_hosts[$i];
  debugMessage(1, "connecting to BF[".$i."] ".$host.":".$port);
  $bfs[$i] = Dada::connectToMachine($host, $port);
  if (!$bfs[$i])
  {
    debugMessage(0, "ERROR: could not connect to BF[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$port = $bp_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$num_bp; $i++)
{
  $host = $bp_hosts[$i];
  debugMessage(1, "connecting to BP[".$i."] ".$host.":".$port);
  $bps[$i] = Dada::connectToMachine($host, $port);
  if (!$bps[$i])
  {
    debugMessage(0, "ERROR: could not connect to BP[".$i."] ".$host.":".$port);
    exit (1);
  }
}


$host = $cfg{"SERVER_HOST"};
$port = $cfg{"CLIENT_MASTER_PORT"};
debugMessage(0, "connecting to ".$host.":".$port);
$srv = Dada::connectToMachine($host, $port);
if (!$srv)
{
  debugMessage(0, "ERROR: could not connect to SRV ".$host.":".$port);
  exit (1);
}

debugMessage(0, "Performing startup");

# init datablocks
debugMessage(1, "Creating AQ datablocks");
threadedTelnetCommand("cmd=init_dbs", \@aqs);

debugMessage(1, "Creating BF datablocks");
threadedTelnetCommand("cmd=init_dbs", \@bfs);

debugMessage(1, "Creating BP datablocks");
threadedTelnetCommand("cmd=init_dbs", \@bps);

# next we start all server daemons (EXCEPT TMC Interface)
my @srv_daemons = split(/ /, $aq_cfg{"SERVER_DAEMONS"});
my @srv_daemons_custom = ();
my $d;
foreach $d ( @srv_daemons )
{
  if ($d ne "mopsr_tmc_interface")
  {
    push (@srv_daemons_custom, $d);
  }
}

foreach $d ( @srv_daemons_custom )
{
  debugMessage(1, "Starting ".$d." on server");
  ($result, $response) = Dada::sendTelnetCommand($srv, "cmd=start_daemon&args=".$d);
}

sleep (1);

debugMessage(1, "Starting AQ daemons");
threadedTelnetCommand("cmd=start_daemons", \@aqs);

debugMessage(1, "Starting BF daemons");
threadedTelnetCommand("cmd=start_daemons", \@bfs);

debugMessage(1, "Starting BP daemons");
threadedTelnetCommand("cmd=start_daemons", \@bps);

$d = "mopsr_tmc_interface";
debugMessage(1, "Starting ".$d." on server");
($result, $response) = Dada::sendTelnetCommand($srv, "cmd=start_daemon&args=".$d);

debugMessage(1, "Closing Sockets");

# close socket connections
for ($i=0; $i<$num_aq; $i++)
{
  if ($aqs[$i])
  {
    $aqs[$i]->close();
  }
}

for ($i=0; $i<$num_bf; $i++)
{
  if ($bfs[$i])
  {
    $bfs[$i]->close();
  }
}
for ($i=0; $i<$num_bp; $i++)
{
  if ($bps[$i])
  {
    $bps[$i]->close();
  }
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

sub debugMessage($$) 
{
  my ($level, $message) = @_;

  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada::getCurrentDadaTime(0)."] ".$message."\n";
  }
}

sub tc($$)
{
  my ($sock, $cmd) = @_;
  my ($result, $response);
  ($result, $response) = Dada::sendTelnetCommand ($sock, $cmd);
}

sub threadedTelnetCommand($\@)
{
  my ($cmd, $socks_ref) = @_;

  my @socks = @$socks_ref;

  my @threads = ();
  my $i;
  for ($i=0; $i<=$#socks; $i++)
  {
    $threads[$i]= threads->new(\&tc, $socks[$i], $cmd);
  }
  for ($i=0; $i<=$#socks; $i++)
  {
    $threads[$i]->join();
  }
}

