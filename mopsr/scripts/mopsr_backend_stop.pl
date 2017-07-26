#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
# 
# Graceful shutdown of the MOPSR backend based on the configuration
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
my %bs_cfg = Mopsr::getConfig("bs");

# socket connections to each of the master control scripts
my $num_aq = 0;
my $num_bf = 0;
my $num_bp = 0;
my $num_bs = 0;

my %aq_hosts_counts = ();
my %bf_hosts_counts = ();
my %bp_hosts_counts = ();
my %bs_hosts_counts = ();
my @aq_hosts = ();
my @bf_hosts = ();
my @bp_hosts = ();
my @bs_hosts = ();
my @aqs = ();
my @bfs = ();
my @bps = ();
my @bss = ();
my $srv;

my ($i, $port, $host);

# determine number of AQ nodes
$port = $aq_cfg{"CLIENT_MASTER_PORT"};
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
debugMessage(1, "Connecting to ".$num_aq." AQ master control daemons");
for ($i=0; $i<$num_aq; $i++)
{
  $host = $aq_hosts[$i];
  debugMessage(2, "connecting to AQ[".$i."] ".$host.":".$port);
  $aqs[$i] = Dada::connectToMachine($host, $port);
  if (!$aqs[$i])
  {
    debugMessage(0, "ERROR: could not connect to AQ[".$i."] ".$host.":".$port);
    exit (1);
  }
}

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
debugMessage(1, "Connecting to ".$num_bf." BF master control daemons");
for ($i=0; $i<$num_bf; $i++)
{
  $host = $bf_hosts[$i];
  debugMessage(2, "connecting to BF[".$i."] ".$host.":".$port);
  $bfs[$i] = Dada::connectToMachine($host, $port);
  if (!$bfs[$i])
  {
    debugMessage(0, "ERROR: could not connect to BF[".$i."] ".$host.":".$port);
    exit (1);
  }
}

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
debugMessage(1, "Connecting to ".$num_bp." BP master control daemons");
for ($i=0; $i<$num_bp; $i++)
{
  $host = $bp_hosts[$i];
  debugMessage(2, "connecting to BP[".$i."] ".$host.":".$port);
  $bps[$i] = Dada::connectToMachine($host, $port);
  if (!$bps[$i])
  {
    debugMessage(0, "ERROR: could not connect to BP[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$port = $bs_cfg{"CLIENT_MASTER_PORT"};
for ($i=0; $i<$bs_cfg{"NUM_BS"}; $i++)
{
  $host = $bs_cfg{"BS_".$i};
  if (!defined ($bs_hosts_counts{$host}))
  { 
    $bs_hosts_counts{$host} = 0;
  }
  $bs_hosts_counts{$host} += 1;
}
@bs_hosts = sort keys %bs_hosts_counts;
$num_bs = $#bs_hosts + 1;
debugMessage(1, "Connecting to ".$num_bs." BS master control daemons");
for ($i=0; $i<$num_bs; $i++)
{
  $host = $bs_hosts[$i];
  debugMessage(2, "connecting to BS[".$i."] ".$host.":".$port);
  $bss[$i] = Dada::connectToMachine($host, $port);
  if (!$bss[$i])
  {
    debugMessage(0, "ERROR: could not connect to BS[".$i."] ".$host.":".$port);
    exit (1);
  }
}

$host = $cfg{"SERVER_HOST"};
$port = $cfg{"CLIENT_MASTER_PORT"};
debugMessage(1, "Connecting to SRV master control daemon");
debugMessage(2, "connecting to ".$host.":".$port);
$srv = Dada::connectToMachine($host, $port);
if (!$srv)
{
  debugMessage(0, "ERROR: could not connect to SRV ".$host.":".$port);
  exit (1);
}

debugMessage(1, "Stopping TMC interface on server");
($result, $response) = Dada::sendTelnetCommand($srv, "cmd=stop_daemon&args=mopsr_tmc_interface");

sleep(2);

# stop AQ daemons
debugMessage(1, "Stopping AQ daemons");
threadedTelnetCommand("cmd=stop_daemons", \@aqs);

debugMessage(1, "Stopping BF daemons");
threadedTelnetCommand("cmd=stop_daemons", \@bfs);

debugMessage(1, "Stopping BP daemons");
threadedTelnetCommand("cmd=stop_daemons", \@bps);

debugMessage(1, "Stopping BS daemons");
threadedTelnetCommand("cmd=stop_daemons", \@bss);

sleep(2);

debugMessage(1, "Stopping SRV daemons");
($result, $response) = Dada::sendTelnetCommand($srv, "cmd=stop_daemons");

# destroy datablocks
debugMessage(1, "Destroying AQ datablocks");
threadedTelnetCommand("cmd=destroy_dbs", \@aqs);

debugMessage(1, "Destroying BF datablocks");
threadedTelnetCommand("cmd=destroy_dbs", \@bfs);

debugMessage(1, "Destroying BP datablocks");
threadedTelnetCommand("cmd=destroy_dbs", \@bps);

debugMessage(1, "Destroying BS datablocks");
threadedTelnetCommand("cmd=destroy_dbs", \@bss);

sleep(2);

debugMessage(1, "Stopping SRV master control");
($result, $response) = Dada::sendTelnetCommand($srv, "cmd=stop_daemon&args=mopsr_master_control");

debugMessage(1, "Stopping AQ master control");
threadedTelnetCommand("cmd=stop_daemon&args=mopsr_master_control", \@aqs);

debugMessage(1, "Stopping BF master control");
threadedTelnetCommand("cmd=stop_daemon&args=mopsr_master_control", \@bfs);

debugMessage(1, "Stopping BP master control");
threadedTelnetCommand("cmd=stop_daemon&args=mopsr_master_control", \@bps);

debugMessage(1, "Stopping BS master control");
threadedTelnetCommand("cmd=stop_daemon&args=mopsr_master_control", \@bss);


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
for ($i=0; $i<$num_bs; $i++)
{
  if ($bss[$i])
  {
    $bss[$i]->close();
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

