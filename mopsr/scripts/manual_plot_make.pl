#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use File::Basename;
use Time::Local;
use Time::HiRes qw(usleep);
use Dada;
use Mopsr;
#use MakePlotsFromArchives;

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our %cfg;
our %bf_cfg;
our %bp_cfg;
our %bp_ct;
our $quit_daemon : shared;
our $warn;
our $error;
our $coarse_nchan;
our $hires;

#
# Initialize global variables
#
%cfg = Mopsr::getConfig();
%bf_cfg = Mopsr::getConfig("bf");
%bp_cfg = Mopsr::getConfig("bp");
%bp_ct = Mopsr::getCornerturnConfig("bp");
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$coarse_nchan = 32;
if (($cfg{"CONFIG_NAME"} =~ m/320chan/) || ($cfg{"CONFIG_NAME"} =~ m/312chan/))
{
  $hires = 1;
}
else
{
  $hires = 0;
}

# Autoflush STDOUT
$| = 1;


# 
# Function Prototypes
#
sub main();

#
# Main
#
my $result = 0;
$result = main();

exit($result);


sub main() 
{
  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  my $num_args = $#ARGV + 1;
  if ($num_args != 2) {
    print "\nUsage: manual_plot_make.pl UTC PSR\n";
    exit;
  }

# (2) we got two command line args, so assume they are the
# first name and last name
  my $UTC =$ARGV[0];
  my $PSR =$ARGV[1];

  my $obs_results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  chdir $obs_results_dir;
  my $fres_ar = $UTC."/".$PSR."/".$PSR."_f.tot";
  my $tres_ar = $UTC."/".$PSR."/".$PSR."_t.tot";

  Mopsr::makePlotsFromArchives($UTC, $fres_ar, $tres_ar, "120x90", "", $PSR, 0, %cfg);
  Mopsr::makePlotsFromArchives($UTC, $fres_ar, $tres_ar, "1024x768", "", $PSR, 0, %cfg);
  #MakePlotsFromArchives::makePlotsFromArchives($UTC, $fres_ar, $tres_ar, "120x90", "", $PSR, %cfg);
  #MakePlotsFromArchives::makePlotsFromArchives($UTC, $fres_ar, $tres_ar, "1024x768", "", $PSR, %cfg);

  return 0;
}

#makePlotsFromArchives is copied from results_mgr and all references to ten_sec_archive are removed (bandpass plot)



