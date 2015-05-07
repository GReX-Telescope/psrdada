#!/usr/bin/env perl

#
# Author:   Andrew Jameson
# Created:  3 Dec, 2007
# Modigied: 11 Mar, 2008
#

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use strict;         # strict mode (like -Wall)
use IO::Socket;     # Standard perl socket library
use IO::Select;     
use Net::hostent;
use File::Basename;
use threads;        # Perl threads module
use threads::shared; 
use XML::Simple qw(:strict);
use Data::Dumper;
use Dada;           # DADA Module for configuration options
use Mopsr;          # Mopsr Module for configuration options

sub quitPWCCommand();

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));


#
# Constants
#
use constant PIDFILE            => "mopsr_tmc_interface.pid";
use constant LOGFILE            => "mopsr_tmc_interface.log";
use constant QUITFILE           => "mopsr_tmc_interface.quit";
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant TERMINATOR         => "\r";
# We must always begin on a 3 second boundary since the pkt rearm UTC
use constant PKTS_PER_3_SECONDs => 390625;

#
# Global variable declarations
#
our $dl;
our $daemon_name;
our %cfg : shared;
our %site_cfg : shared;
our $current_state : shared;
our $pwcc_running : shared;
our $quit_threads : shared;
our $n_ant : shared;
our $spec_generated : shared;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our $error;
our $warn;
our $pwcc_thread;
our $utc_stop : shared;
our $tobs_secs : shared;

#
# global variable initialization
#
$dl = 2;
$daemon_name = Dada::daemonBaseName($0);
%cfg = Mopsr::getConfig();
%site_cfg = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);
$current_state = "Idle";
$pwcc_running = 0;
$quit_threads = 0;
$n_ant  = "N/A";
$spec_generated = 0;
$pwcc_host = $cfg{"PWCC_HOST"};
$pwcc_port = $cfg{"PWCC_PORT"};
$client_master_port = $cfg{"CLIENT_MASTER_PORT"};
$warn = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
$error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
$pwcc_thread = 0;
$tobs_secs = -1;
$utc_stop = "";


#
# Main
#
{
  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  my $server_host =     $cfg{"SERVER_HOST"};
  my $config_dir =      $cfg{"CONFIG_DIR"};
  my $tmc_host =        $cfg{"TMC_INTERFACE_HOST"};
  my $tmc_port =        $cfg{"TMC_INTERFACE_PORT"};
  my $tmc_state_port =  $cfg{"TMC_STATE_INFO_PORT"};

  my $tmc_cmd;

  my $handle = "";
  my $peeraddr = "";
  my $hostinfo = "";  
  my $command = "";
  my @cmds = "";
  my $key = "";
  my $lckey = "";
  my $val = "";
  my $result = "";
  my $response = "";
  my $failure = "";
  my $state_thread = 0;
  my $control_thread = 0;
  my $rh;

  my $ant = "";
  my $cmd = "";
  my $xml = "";

  # set initial state
  $current_state = "Idle";

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  # Sanity check for this script
  #if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) 
  #{
  #  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  #  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  #  exit(1);
  #}

  if (-f $warn) {
    unlink $warn;
  }
  if (-f $error) {
    unlink $error;
  }

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  my $obs = $ARGV[0];
  my $tracking = 0;

  ($result, $response) = dumpAntennaMapping($obs, $tracking);
  Dada::logMsg(0, $dl, $result." ".$response);

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");
}
exit 0;


###############################################################################
#
# Functions
#

#
# some custom sorting routines
#
sub intsort
{
  if ((int $a) < (int $b))
  {
    return -1;
  }
  elsif ((int $a) > (int $b))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

sub modsort
{
  my $mod_a = $a;
  my $mod_b = $b;

  $mod_a =~ s/-B/-0/;
  $mod_a =~ s/-G/-1/;
  $mod_a =~ s/-Y/-2/;
  $mod_a =~ s/-R/-3/;
  $mod_b =~ s/-B/-0/;
  $mod_b =~ s/-G/-1/;
  $mod_b =~ s/-Y/-2/;
  $mod_b =~ s/-R/-3/;

  return $mod_a cmp $mod_b;
}


#
# Dumps the antenna mapping for this observation
#
sub dumpAntennaMapping($$)
{
  my ($obs, $tracking) = @_;

  my $ct_file = $cfg{"CONFIG_DIR"}."/mopsr_cornerturn.cfg";
  my $sp_file = $cfg{"CONFIG_DIR"}."/mopsr_signal_paths.txt";
  my $mo_file = $cfg{"CONFIG_DIR"}."/molonglo_modules.txt";
  my $ba_file = $cfg{"CONFIG_DIR"}."/molonglo_bays.txt";
  my $pm_file = $cfg{"CONFIG_DIR"}."/preferred_modules.txt";

  my $antenna_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.antenna";
  my $baselines_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.baselines";
  my $refant_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.refant";

  my %ct = Dada::readCFGFileIntoHash($ct_file, 0);
  my %sp = Dada::readCFGFileIntoHash($sp_file, 1);
  my %mo = Dada::readCFGFileIntoHash($mo_file, 1);
  my %ba = Dada::readCFGFileIntoHash($ba_file, 1);
  my %aq_cfg = Mopsr::getConfig("aq");

  my @sp_keys_sorted = sort modsort keys %sp;

  # now generate the listing of antennas the correct ordering
  my ($i, $send_id, $first_ant, $last_ant, $pfb_id, $imod, $rx);
  my ($pfb, $pfb_input, $bay_id);
  my %pfb_mods = ();
  my @mods = ();
  logMsg(2, $dl, "dumpAntennaMapping: aq_cfg{NUM_PWC}=".$aq_cfg{"NUM_PWC"});
  for ($i=0; $i<$aq_cfg{"NUM_PWC"}; $i++)
  {
    logMsg(2, $dl, "dumpAntennaMapping: i=".$i);
    # if this PWC is an active or passive
    if ($aq_cfg{"PWC_STATE_".$i} ne "inactive")
    {
      $send_id = $aq_cfg{"PWC_SEND_ID_".$i};

      # this is the mapping in RAM for the input to the calibration code
      $first_ant = $cfg{"ANT_FIRST_SEND_".$send_id};
      $last_ant  = $cfg{"ANT_LAST_SEND_".$send_id};

      # now find the physics antennnas for this PFB
      $pfb_id  = $aq_cfg{"PWC_PFB_ID_".$i};

      if ( -f $pm_file )
      {
        my %pm = Dada::readCFGFileIntoHash($pm_file, 1);
        my @pfb_mods = split(/ +/, $pm{$pfb_id});
        my $pfb_mod;
        $imod = $first_ant;

        # for each of the specified PFB inputs
        foreach $pfb_mod (@pfb_mods)
        {
          # find the corresponding RX
          foreach $rx ( @sp_keys_sorted )
          {
            ($pfb, $pfb_input) = split(/ /, $sp{$rx});
            if (($pfb eq $pfb_id) && ($pfb_input eq $pfb_mod))
            {
              $mods[$imod] = $rx;
              $imod++;
            }
          }
        }
        if ($imod != $last_ant + 1)
        {
          return ("fail", "failed to identify modules correctly");
        }
      }
      else
      {
        logMsg(3, $dl, "dumpAntennaMapping: pfb_id=".$pfb_id." ants=".$first_ant." -> ".$last_ant);

        my @pfb_mods = split(/ +/, $aq_cfg{"PWC_ANTS"});
        $imod = $first_ant;
        %pfb_mods = ();
        foreach $rx ( @sp_keys_sorted )
        {
          ($pfb, $pfb_input) = split(/ /, $sp{$rx});
          logMsg(3, $dl, "dumpAntennaMapping: pfb=".$pfb." pfb_input=".$pfb_input);
          if ($pfb eq $pfb_id)
          {
            my $pfb_mod;
            foreach $pfb_mod (@pfb_mods)
            {
              if ($pfb_input eq $pfb_mod)
              {
                if (($imod >= $first_ant) && ($imod <= $last_ant))
                {
                  $mods[$imod] = $rx;
                  $imod++;
                }
                else
                {
                  return ("fail", "failed to identify modules correctly");
                }
              }
            }
          }
        }
      }
    }
  }
  open(FHA,">".$antenna_file) or return ("fail", "could not open antenna file for writing");
  open(FHB,">".$baselines_file) or return ("fail", "could not open baselines file for writing");
  open(FHC,">".$refant_file) or return ("fail", "could not open reference antenna file for writing");

  # ants should contain a listing of the antenna orderings
  my ($mod_id, $dist, $delay, $scale, $jmod);
  my $ref_mod = -1;

  # determine the reference module/antenna
  for ($imod=0; $imod<=$#mods; $imod++)
  {
    if ($ref_mod == -1)
    {
      if ($mods[$imod] =~ m/E01/)
      {
        print FHC $mods[$imod]."\n";
        $ref_mod = $imod;
      }
    }
  }
  if ($ref_mod == -1)
  {
    print FHC $mods[0]."\n";
    $ref_mod = 0;
  }
  close FHC;

  for ($imod=0; $imod<=$#mods; $imod++)
  {
    $mod_id = $mods[$imod];
    $bay_id = substr($mod_id,0,3);
    if ($tracking)
    {
      $dist = $ba{$bay_id};
    }
    else
    {
      ($dist, $delay, $scale) = split(/ /,$mo{$mod_id},3);
    }

    Dada::logMsg(2, $dl, "imod=".$imod." ".$mod_id.": dist=".$dist." delay=".$delay." scale=".$scale);
    print FHA $mod_id." ".$dist." ".$delay."\n";

    for ($jmod=$imod+1; $jmod<=$#mods; $jmod++)
    {
      Dada::logMsg(2, $dl, $mods[$imod]." ".$mods[$jmod]);
      print FHB $mods[$imod]." ".$mods[$jmod]."\n";
    }
  }

  close(FHA);
  close(FHB);

  my ($cmd, $result, $response);
  $cmd = "cp ".$antenna_file." ".$baselines_file." ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$obs."/";
  Dada::logMsg(2, $dl, "dumpAntennaMapping: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "dumpAntennaMapping: ".$result." ".$response);
  if ($result ne "ok")
  {
    return ("fail", "could not copy antenna files to archive dir");
  }

  return ("ok", "");
}
