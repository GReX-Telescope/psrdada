#!/usr/bin/env perl 

#
# Author:   Andrew Jameson
# Created:  6 Dec, 2016
#
# This daemons runs continuously produces feedback plots of the
# current observation

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;               # strict mode (like -Wall)
use File::Basename;
use threads;
use threads::shared;
use MIME::Lite;
use Bpsr;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DL           => 1;


#
# Global Variable Declarations
#
our %cfg = Bpsr::getConfig();
our %roach = Bpsr::getROACHConfig();
our $quit_daemon : shared = 0;
our $daemon_name : shared = Dada::daemonBaseName($0);
our $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";
our $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
our $last_frb = "";
our %frb_actions = ();

#
# Signal Handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) 
{
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}

#
# only beam 2 - 8189
#

# default FRB detection options
$frb_actions{"default"} = { "snr_cut" => 10.0, "filter_cut" => 8.0, "egal_dm" => "true", 
                            "excise_psr" => "true", "cps_cut" => "5", "cc_email" => "" };

# custom (per project) FRB detection options
$frb_actions{"P999"}    = { "snr_cut" => 45.0, "filter_cut" => 9.0, "snr_cut" => 9.0, "excise_psr" => "true", "egal_dm" => "true", "cps_cut" => "70" }; 
                            
$frb_actions{"P456"}     = { "cc_email" => "stefan.oslowski\@gmail.com, ryanmshannon\@gmail.com" };

$frb_actions{"P864"}    = { "egal_dm" => "false", "excise_psr" => "false", 
                            "cps_cut" => "50", "filter_cut" => "9",
                            "cc_email" => "manishacaleb\@gmail.com, cflynn\@swin.edu.au" };

$frb_actions{"P858"}    = { "cc_email" => "superb\@lists.pulsarastronomy.net" };
$frb_actions{"P892"}    = { "cc_email" => "superb\@lists.pulsarastronomy.net" };
$frb_actions{"PX025"}   = { "cc_email" => "superb\@lists.pulsarastronomy.net" };

$frb_actions{"P871"}    = { "cc_email" => "ebpetroff\@gmail.com, cherrywyng\@gmail.com, cmlflynn\@gmail.com, davidjohnchampion\@gmail.com, evan.keane\@gmail.com, ewan.d.barr\@gmail.com, manishacaleb\@gmail.com, matthew.bailes\@gmail.com, michael\@mpifr-bonn.mpg.de, apossenti\@gmail.com, sarahbspolaor\@gmail.com, Simon.Johnston\@atnf.csiro.au, vanstraten.willem\@gmail.com, Ben.Stappers\@manchester.ac.uk" };

$frb_actions{"P789"}    = { "cc_email" => "ewan.d.barr\@gmail.com, Simon.Johnston\@atnf.csiro.au, evan.keane\@gmail.com" };

$frb_actions{"P879"}    = { "cc_email" => "cmlflynn\@gmail.com, Ramesh.Bhat\@curtin.edu.au, michael\@mpifr-bonn.mpg.de, davidjohnchampion\@gmail.com, sarahbspolaor\@gmail.com" };

$frb_actions{"P938"}    = { "cc_email" => "v.vikram.ravi\@gmail.com, ryanmshannon\@gmail.com" };

#$frb_actions{"P888"}    = { "egal_dm" => "false", "excise_psr" => "false", "snr_cut" => 8.0,
#                            "cps_cut" => "50", "filter_cut" => "9", "beam_mask" => "1", 
#                            "cc_email" => "epetroff\@swin.edu.au" };


#
# Main Loop
#
{
  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $obs_results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $control_thread = 0;
  my $coincidencer_thread = 0;

  my $cmd;
  my $dir;
  my @subdirs;
  my $curr_time;
  my $now;

  my $i;
  my $j;

  my $result = "";
  my $response = "";
  my $counter = 0;

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # Redirect standard output and error
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, DL, "STARTING SCRIPT");

  chdir $cfg{"SERVER_RESULTS_DIR"};

  # Start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  # start the coincidencer thread 
  # $coincidencer_thread = threads->new(\&coincidencerThread);

  my $curr_processing = "";

  while (!$quit_daemon)
  {
    $curr_time = time;

    $dir = "";
    @subdirs = ();

    # TODO check that directories are correctly sorted by UTC_START time
    Dada::logMsg(2, DL, "Main While Loop, looking for data in ".$obs_results_dir);

    # get the list of all the current observations (should only be 1)
    $cmd = "find ".$obs_results_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.processing' ".
           "-printf '\%h\\n' | awk  -F/ '{print \$NF}' | sort";
    Dada::logMsg(2, DL, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "main: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "main: find command failed: ".$response);

    } elsif ($response eq "") {
      Dada::logMsg(2, DL, "main: nothing to process");

    } else {

      @subdirs = split(/\n/, $response);
      Dada::logMsg(2, DL, "main: found ".($#subdirs+1)." obs.processing files");

      my $h=0;
      my $age = 0;
      my $n_png_files = 0;
      my $n_cand_files = 0;

      # For each observation
      for ($h=0; (($h<=$#subdirs) && (!$quit_daemon)); $h++) 
      {
        $dir = $subdirs[$h];

        Dada::logMsg(2, DL, "main: testing ".$dir);
        
        if ($coincidencer_thread)
        {
          ($result, $response) = processCandidates($dir);
        }
        else
        {
          ($result, $response) = processCandidatesOld($dir);
        }

        chdir $cfg{"SERVER_RESULTS_DIR"};
      } 
    }

    $now = time;

    # If we have been asked to exit, dont sleep
    while ((!$quit_daemon) && ($now < $curr_time + 1))
    {
      sleep(1);
      $now = time;
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  if ($coincidencer_thread)
  {
    $coincidencer_thread->join();
  }
                                                                                
  Dada::logMsg(0, DL, "STOPPING SCRIPT");

  exit(0);
}

###############################################################################
#
# Functions
#
#
# process coincidenced candidates
#
sub processCandidates($)
{
  (my $obs) = @_;

  my ($cmd, $result, $response, $i);

  my $obs_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$obs;
  my $cands_record = $obs_dir."/all_candidates.dat";
  my @files = ();
  my $file = "";
  my $n_processed = 0;

  # for the given obs, a list of the candiate files to be parsed beams 
  $cmd = "find ".$obs_dir." -mindepth 1 -maxdepth 1 ".
         "-type f -name '2???-??-??-??:??:??_all.cand' -printf '%f\n' | sort -n";
  Dada::logMsg(2, DL, "processCandidates: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

  if ($result ne "ok")
  {
    Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
    Dada::logMsgWarn($warn, "could not find candidate files");
    return ("fail", "could not get candiate list");
  }

  # count the number of candidate files to each epoch
  @files = split(/\n/, $response);

  # update the beam mask from the bpsr_active_beams configuration file
  my $bpsr_active_beams_file = $cfg{"CONFIG_DIR"}."/bpsr_active_beams.cfg";
  my $beam_mask = 0;
  if (-f $bpsr_active_beams_file)
  {
    my %beam_config = Dada::readCFGFileIntoHash($bpsr_active_beams_file, 0);
    for ($i=1; $i<=13; $i++)
    {
      my $key = sprintf ("BEAM_%02d_pt", $i);
      if ($beam_config{$key} eq "on")
      {
        $beam_mask = $beam_mask | (1 << ($i-1));
      }
    }
  }
  else
  {
    # enable all beams
    $beam_mask = 8191;
  }

  # firstly, search for any FRBs 
  foreach $file ( @files )
  {
    # try to detect FRBS in just this single candidates file (not the biggun')
    Dada::logMsg(2, DL, "processCandidates: detectFRBs($obs, $file, $beam_mask, $last_frb)");
    ($result, $response) = detectFRBs($obs, $file, $beam_mask, $last_frb);
    Dada::logMsg(3, DL, "processCandidates: $result $response");
    if ($result eq "ok")
    {
      $last_frb = $response;
    }
  }

  # now append the candidate files to the accumulated total for this observation
  foreach $file ( @files )
  {
    if (-f $cands_record)
    {
      $cmd = "cat ".$obs_dir."/".$file." | sort -k 2 -n >> ".$cands_record;
    }
    else
    {
      $cmd = "cat ".$obs_dir."/".$file." | sort -k 2 -n > ".$cands_record;
    }

    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
      Dada::logMsgWarn($warn, "could not append output to record");
      return ("fail", "could not append output to record");
    }
    unlink ($obs_dir."/".$file);
    $n_processed++;
  }

  if ($n_processed > 0)
  {
    # get the number of linues in candidates files
    $cmd = "wc -l ".$cands_record;
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidates: ".$cmd." failed: ".$response);
      chdir $cfg{"SERVER_RESULTS_DIR"};
      return;
    }

    # check if more than 10000 rows
    my $nrows = $response;
    my $rows_args = "";
    if ($nrows > 10000)
    {
      $rows_args = " -skip_rows ".($nrows - 10000);
    }

    chdir $obs_dir;

    # generate 1024x768 large plot and candidate XML list
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -snr_cut 6.5 -filter_cut 11 -cand_list_xml ".$rows_args." > cand_list.xml";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    my $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_1024x768.tmp.png ".$curr_time.".cands_1024x768.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    # generate 700x 240 small plot
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -beam_mask ".$beam_mask." -snr_cut 7.0 -filter_cut 11 -just_time_dm -resolution 700x240".$rows_args;
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);

    $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_700x240.tmp.png ".$curr_time.".dm_vs_time_700x240.png";
    Dada::logMsg(2, DL, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidates: ".$result." ".$response);
  }

  chdir $cfg{"SERVER_RESULTS_DIR"};
}

#
# process uncoincidenced candidates
#
sub processCandidatesOld($)
{
  (my $obs) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $beam = "";
  my $file = "";
  my @files = ();
  my %cands = ();
  my @bits = ();

  my $file_prefix = "";
  my $junk = "";
  my $cands_record = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/all_candidates.dat";
  my $output_file = "";
  my $n_processed = 0;
  my ($i, $frb_event_string, $sock);

  # for the given obs, build a list of the candiate beams 
  $cmd = "find ".$cfg{"SERVER_RESULTS_DIR"}."/".$obs." -mindepth 2 -maxdepth 2 -type f -name '2*_??.cand'";
  Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
  
  if ($result ne "ok")
  {
    Dada::logMsg(0, DL, "processCandidatesOld: ".$cmd." failed: ".$response);
    Dada::logMsgWarn($warn, "could not find candidate files");
    return ("fail", "could not get candiate list");
  }
 
  # count the number of candidate files to each epoch
  @files = split(/\n/, $response);
  foreach $file ( @files )
  {
    @bits = split(/\//, $file);
    
    $beam = $bits[$#bits - 1];
    $file = $bits[$#bits - 0];
    ($file_prefix, $junk) = split(/_/, $file, 2); 

    if (! exists($cands{$file_prefix}))
    {
      $cands{$file_prefix} = 0;
    }
    $cands{$file_prefix} += 1;
  }

  # now do some plotting
  chdir $cfg{"SERVER_RESULTS_DIR"}."/".$obs;

  my $nbeams = `ls -1d ?? | wc -l`;

  @files = sort keys %cands;
  my $file_count = $#files + 1;
  foreach $file ( @files )
  {
    # process this file if the number of files == NUMPWC or if have more than 10 sets of files
    if (($cands{$file} == $nbeams) || ($file_count > 10))
    {
      $cmd = "coincidencer ./??/".$file."_??.cand";
      Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd." count=".$cands{$file}." nsets=".$file_count);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidatesOld: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "coincidencer failed to process candidates");
        return ("fail", "could not process candidates");
      }

      # remove beam candidate files
      $cmd = "rm -f ./??/".$file."_??.cand";
      Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidatesOld: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "could not removed processed candidates");
        return ("fail", "could not remove processed candidates");
      }
      
      $output_file = $file."_all.cand";

      # now append the .cands files to the accumulated total for this observation
      if (-f $cands_record)
      {
        $cmd = "cat ".$output_file." >> ".$cands_record;
      }
      else
      {
        $cmd = "cp ".$output_file." ".$cands_record;
      }
      Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, DL, "processCandidatesOld: ".$cmd." failed: ".$response);
        Dada::logMsgWarn($warn, "could not append output to record");
        return ("fail", "could not append output to record");
      }

      $n_processed++;
    }
    else
    {
      Dada::logMsg(2, DL, "processCandidatesOld: skipping ".$obs."/??/".$file." as only ".$cands{$file}." candidates present");
    }
  }

  # update the beam mask from the bpsr_active_beams configuration file
  my $bpsr_active_beams_file = $cfg{"CONFIG_DIR"}."/bpsr_active_beams.cfg";
  my $beam_mask = 0;
  if (-f $bpsr_active_beams_file)
  {
    my %beam_config = Dada::readCFGFileIntoHash($bpsr_active_beams_file, 0);
    for ($i=1; $i<=13; $i++)
    {
      my $key = sprintf ("BEAM_%02d_pt", $i);
      if ($beam_config{$key} eq "on")
      {
        $beam_mask = $beam_mask | (1 << ($i-1));
      }
    }
  }
  else
  {
    # enable all beams
    $beam_mask = 8191;
  }

  # if the loop processed at least 1 file, update the plots
  if ($n_processed > 0)
  {
    # get the number of linues in candidates files
    $cmd = "wc -l all_candidates.dat";
    Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "processCandidatesOld: ".$cmd." failed: ".$response);
      chdir $cfg{"SERVER_RESULTS_DIR"};
      return;
    }

    # check if more than 50000 rows
    my $nrows = $response;
    my $rows_args = "";
    if ($nrows > 50000)
    {
      $rows_args = " -skip_rows ".($nrows - 50000);
    }

    # generate 1024x768 large plot and candidate XML list
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -snr_cut 6.5 -filter_cut 11 -cand_list_xml ".$rows_args." > cand_list.xml";
    Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);

    my $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_1024x768.tmp.png ".$curr_time.".cands_1024x768.png";
    Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);

    # generate 700x 240 small plot
    $cmd = $cfg{"SCRIPTS_DIR"}."/trans_gen_overview.py -beam_mask ".$beam_mask." -snr_cut 7.0 -filter_cut 11 -just_time_dm -resolution 700x240".$rows_args;
    Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);

    $curr_time = Dada::getCurrentDadaTime();
    $cmd = "mv overview_700x240.tmp.png ".$curr_time.".dm_vs_time_700x240.png";
    Dada::logMsg(2, DL, "processCandidatesOld: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "processCandidatesOld: ".$result." ".$response);
  }

  # now search for FRBs 
  foreach $file ( @files )
  {
    $output_file = $file."_all.cand"; 
    if ( -f $output_file)
    {
      # try to detect FRBS in just this single candidates file (not the biggun')
      Dada::logMsg(2, DL, "processCandidatesOld: detectFRBs($obs, $output_file, $beam_mask, $last_frb)");
      ($result, $response) = detectFRBs($obs, $output_file, $beam_mask, $last_frb);
      Dada::logMsg(3, DL, "processCandidatesOld: $result $response");
      if ($result eq "ok")
      {
        $last_frb = $response;
      }
    
      unlink ($output_file);
    }
  }

  chdir $cfg{"SERVER_RESULTS_DIR"};
}


#
# detect FRBs from the candidate file in the current dir
#
sub detectFRBs($$$$)
{
  my ($obs, $cands_file, $default_beam_mask, $last_frb) = @_;

  my ($cmd, $result, $response);
  my ($source, $proc_file, $pid, $gl, $gb, $galactic_dm);

  my $to_email = 'ajameson@swin.edu.au';
  my $cc_email = "";
  my $obs_info = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.info";
  my $beam_info = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/beam_info.xml";

  # get the SOURCE and PID
  $cmd = "grep ^SOURCE ".$obs_info." | awk '{print \$2}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $source) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$source);

  # get the PROC_FILE
  $cmd = "grep ^PROC_FILE ".$obs_info." | awk '{print \$2}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $proc_file) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$source);

  # Get the PID
  $cmd = "grep ^PID ".$obs_info." | awk '{print \$2}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $pid) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$pid);

  # determine gl and gb
  $gl = "";
  $gb = "";
  if (-f $beam_info)
  {
    $cmd = "grep \"beam_info beam='01'\" ".$beam_info;
    Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);

    my @parts = split(/ /, $response);

    if ($#parts == 5)
    {
      $gl = substr($parts[4], 4, (length($parts[4]) - 5));
      $gb = substr($parts[5], 4, (length($parts[5]) - 6));
    }
  }
   
  # if we could not get the gl/gb above, calculate it from RA/DEC
  if (($gl eq "") || ($gb eq ""))
  {
    my ($ra, $dec, $ra_deg, $dec_deg);
    $cmd = "grep ^RA ".$obs_info." | awk '{print \$2}'";
    Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
    ($result, $ra) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$ra);

    $cmd = "grep ^DEC ".$obs_info." | awk '{print \$2}'";
    Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
    ($result, $dec) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$dec);

    ($result, $ra_deg) = Dada::convertHHMMSSToDegrees($ra);
    ($result, $dec_deg) = Dada::convertDDMMSSToDegrees($dec);

    Dada::logMsg(2, DL, "detectFRBs: RA ".$ra." -> ".$ra_deg);
    Dada::logMsg(2, DL, "detectFRBs: DEC ".$dec." -> ".$dec_deg);

    $cmd = "eq2gal.py ".$ra_deg." ".$dec_deg;
    Dada::logMsg(3, DL, "detectFRBs: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
    if ($result eq "ok")
    {
      ($gl, $gb) = split(/ /, $response, 2);
    } 
    else
    {
      Dada::logMsg(1, DL, "detectFRBs: couldn't calculate Gl, Gb from RA, DEC: ".$response);
      return ("ok", "");
    }
  }

  Dada::logMsg(2, DL, "detectFRBs: gl=".$gl." gb=".$gb);

  # determine the DM for this GL and GB
  $cmd = "cd \$HOME/opt/NE2001/runtime; ./NE2001 ".$gl." ".$gb." 100 -1 | grep ModelDM | awk '{print \$1}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $galactic_dm) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$galactic_dm);

  $cmd = "frb_detector.py -cands_file ".$cfg{"SERVER_RESULTS_DIR"}."/".$obs."/".$cands_file." ";

  my $snr_cut    = $frb_actions{"default"}{"snr_cut"};
  my $filter_cut = $frb_actions{"default"}{"filter_cut"};
  my $egal_dm    = $frb_actions{"default"}{"egal_dm"};
  my $psr_cut    = $frb_actions{"default"}{"excise_psr"};
  my $cps_cut    = $frb_actions{"default"}{"cps_cut"};
  my $cc_email   = $frb_actions{"default"}{"cc_email"};
  my $beam_mask  = $default_beam_mask;

  if (exists($frb_actions{$pid}))
  {
    if ( exists ($frb_actions{$pid}{"snr_cut"}))    { $snr_cut   = $frb_actions{$pid}{"snr_cut"}; }
    if ( exists ($frb_actions{$pid}{"filter_cut"})) { $filter_cut   = $frb_actions{$pid}{"filter_cut"}; }
    if ( exists ($frb_actions{$pid}{"egal_dm"}))    { $egal_dm   = $frb_actions{$pid}{"egal_dm"}; }
    if ( exists ($frb_actions{$pid}{"excise_psr"})) { $psr_cut   = $frb_actions{$pid}{"excise_psr"}; }
    if ( exists ($frb_actions{$pid}{"cps_cut"}))    { $cps_cut   = $frb_actions{$pid}{"cps_cut"}; }
    if ( exists ($frb_actions{$pid}{"cc_email"}))   { $cc_email  = $frb_actions{$pid}{"cc_email"}; }
    if ( exists ($frb_actions{$pid}{"beam_mask"}))  { $beam_mask = $frb_actions{$pid}{"beam_mask"}; }
  }

  # flag for turning off the DM check
  if (($egal_dm eq "false") && ($proc_file ne "THEDSPSR"))
  {
    $galactic_dm = 0.1;
  }

  if (($pid eq "P858") && 
      (($source eq "J1819-1458") || ($source eq "J1644-4559") || ($source eq "J1752-2806")))
  {
    $galactic_dm = "0.1";
    $snr_cut = "30.0";
    $filter_cut = "8";
    $cps_cut = "50.0";
    $psr_cut = "false";
    $cc_email = "evan.keane\@gmail.com,manishacaleb\@gmail.com,v.vikram.ravi\@gmail.com";

    if ($source eq "J1644-4559")
    {
      $snr_cut = "20";
      $filter_cut = "7"
    }

    if ($source eq "J1819-1458")
    {
      $snr_cut = "10";
    }

    if ($source eq "J1752-2806")
    {
      $snr_cut = "50";
      $filter_cut= "9";
    }
  }

  if (($pid eq "P888") && ($source eq "1955-28"))
  {
    #$snr_cut = "20";
  }

  if ($proc_file eq "THEDSPSR")
  {
    $psr_cut = "true";
  }

  if ($beam_mask ne "")
  {
    $cmd .= " -beam_mask ".$beam_mask;
  }

  Dada::logMsg(1, DL, "detecting FRBs on ".$cands_file." with ".$beam_mask);
  
  $cmd .= " -snr_cut ".$snr_cut." -filter_cut ".$filter_cut." -max_cands_per_sec ".$cps_cut." ".$galactic_dm;

  # we want the events sorted in order of time ideally...
  $cmd .= " | sort -n";

  # check if any FRB's in candidates file
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);

  if (($result eq "ok") && ($response ne ""))
  {
    my @frbs = split(/\n/, $response);

    Dada::logMsg(2, DL, "detectFRBs: FRB Detection for ".$obs." [".($#frbs+1)." events]");
    {
      $last_frb = $obs;
      my $subject = "Possible FRB in ".$obs;
      my $frb_name = "FRB".substr($obs, 2,2).substr($obs,5,2).substr($obs,8,2);

      # dont cc for multiple FRBs in a single observation (RFI)
      if ($#frbs > 10)
      {
        $cc_email = '';
      }

      my $max_snr = 0;
      my $frb;
      foreach $frb (@frbs)
      {
        my ($snr, $time, $sample, $dm, $filter, $prim_beam) = split(/\t/, $frb);
        if ($snr > $max_snr)
        {
          $max_snr = $snr;
        }
      }

      # don't email people unless its a brightish one
      if ($max_snr < 10)
      {
        if ($pid eq "P892")
        {
          $cc_email = "ebpetroff\@gmail.com, shivanibhandari58\@gmail.com, evan.keane\@gmail.com, cmlflynn\@gmail.com";
        }
        elsif ($pid eq "P999")
        {
          $cc_email = ""; 
        }
        else
        {
          $cc_email = "";
        }
      }

      my $msg = MIME::Lite->new(
        From    => 'BPSR FRB Detector <jam192@hipsr-srv0.atnf.csiro.au>',
        To      => $to_email,
        Cc      => $cc_email,
        Subject => 'New Detection: '.$frb_name,
        Type    => 'multipart/mixed',
        Data    => "Here's the PNG file you wanted"
      );

      # generate HTML part of email
      my $html = "<body>";

      $html .= "<table cellpadding=2 cellspacing=2>\n";

      $html .= "<tr><th style='text-align: left;'>UTC START</th><td>".$obs."</td></tr>\n";
      $html .= "<tr><th style='text-align: left;'>Source</th><td>".$source."</td></tr>\n";
      $html .= "<tr><th style='text-align: left; padding-right:5px;'>PID</th><td>".$pid."</td></tr>\n";
      $html .= "<tr><th style='text-align: left;'>NE2001 DM</th><td>".$galactic_dm."</td></tr>\n";

      $html .= "</table>\n";

      # contact all the dbevent processes and dump the raw data
      my ($host, $port, $sock, $frb, $beam);
      my ($time_secs, $time_subsecs, $filter_time, $smearing_time, $total_time, $start_time, $end_time);
      my ($start_utc, $end_utc, $frb_event_string, $mpsr_event_string, $event_utc);

      my $src_html = "<h3>Beams Positions &amp; Known Sources</h3>";

      # setup a hash of psrs in beams
      my %psrs_in_beams = ();

      # firstly check if there is a known PSR in this position source in the 
      if (-f "beaminfo.xml")
      {
        $src_html .= "<table width='100%' border=0 cellpadding=2px cellspacing=2px>\n";
        $src_html .= "<tr>";
        $src_html .= "<th style='text-align: left;'>Beam</th>";
        $src_html .= "<th style='text-align: left;'>RA</th>";
        $src_html .= "<th style='text-align: left;'>DEC</th>";
        $src_html .= "<th style='text-align: left;'>Gl</th>";
        $src_html .= "<th style='text-align: left;'>Gb</th>";
        $src_html .= "<th style='text-align: left;'>PSR</th>";
        $src_html .= "</tr>\n";

        open FH, "<beaminfo.xml";
        my @lines = <FH>;
        close FH;

        my ($line, $ra, $dec, $gl, $gb, $psrs, $name, $dm);
        my $beam = "00";
        foreach $line (@lines)
        {
          chomp $line;
          if ($line =~ m/beam_info beam/)
          {
            $gl = "--";
            $gb = "--";
            $psrs = "";
            my @bits = split(/ /,$line);
            $beam = substr($bits[1], 6, 2);
            $ra = substr($bits[2],5,length($bits[2])-6);
            $dec = substr($bits[3],6,length($bits[3])-7);
            if ($#bits > 3)
            {
              $gl = substr($bits[4],4,length($bits[4])-6);
              $gb = substr($bits[5],4,length($bits[5])-6);
            }
            $psrs_in_beams{$beam} = ();
          } 
          elsif ($line =~ m/beam_info/)  
          {
            $src_html .= "<tr>";
            $src_html .=   "<td>".$beam."</td>";
            $src_html .=   "<td>".$ra."</td>";
            $src_html .=   "<td>".$dec."</td>";
            $src_html .=   "<td>".$gl."</td>";
            $src_html .=   "<td>".$gb."</td>";
            $src_html .=   "<td>".$psrs."</td>";
            $src_html .= "</tr>\n";
          }
          elsif ($line =~ m/psr name/)
          {
            my @bits = split(/ /,$line);
            $name = substr($bits[1], 6, length($bits[1])-7);
            $dm = substr($bits[2], 4, length($bits[2])-5);
            if ($psrs ne "")
            {
              $psrs .= ", ";
            }
            $psrs .= $name." [DM=".$dm."]\n";
            $psrs_in_beams{$beam}{$dm} = $name;
          }
          else
          {
            Dada::logMsg(3, DL, "detectFRBs: ignoring");
          }
        }
        $src_html .= "</table>";
      }
      else
      {
        $src_html .= "<p>No beaminfo.xml existed for this observation</p>\n";
      }

      $html .= "<hr/>\n";

      #
      # FRB Table
      #
      $html .= "<h3>FRB Detections</h3>";
      $html .= "<table width='100%' border=0 cellpadding=2px cellspacing=2px>\n";
      $html .= "<tr>";
      $html .= "<th style='text-align: left;'>SNR</th>";
      $html .= "<th style='text-align: left;'>Time</th>";
      $html .= "<th style='text-align: left;'>DM</th>";
      $html .= "<th style='text-align: left;'>Length</th>";
      $html .= "<th style='text-align: left;'>Beam</th>";
      $html .= "<th style='text-align: left;'>Known Source(s)</th>";
      $html .= "</tr>\n";

      my $num_frbs_legit = 0;

      # NB this requires a prefix, see below
      $frb_event_string = "";
      $mpsr_event_string = "";

      # a record of good vs bad
      my %frbs_legit = ();

      foreach $frb (@frbs)
      {
        Dada::logMsg(2, DL, "detectFRBs: frb=".$frb);
        my ($snr, $time, $sample, $dm, $filter, $prim_beam) = split(/\t/, $frb);
        my ($delta_dm, $psr_dm, $related_psrs, $padded_prim_beam, $legit);

        $related_psrs = "";
        $delta_dm = $dm * 0.05;
        $padded_prim_beam = sprintf("%02d", $prim_beam);
        $legit = 1;

        # check if there is a known source nearby this DM for this beam
        foreach $psr_dm ( keys %{ $psrs_in_beams{$padded_prim_beam} } )
        {
          # always include any pulsars in the beam
          $related_psrs .= $psrs_in_beams{$padded_prim_beam}{$psr_dm}." [DM=".$psr_dm."]<br/>";

          Dada::logMsg(2, DL, "detectFRBs: FRB dm=".$dm." testing psr_dm=".$psr_dm);
          # if we are excising FRB's with close PSRs
          # the window of DM that we consider the source is a match for the FRB dm
          if (($psr_cut eq "true") && ($dm < ($psr_dm + $delta_dm)))
          {
            Dada::logMsg(1, DL, "detectFRBs: ignoring event since dm=".$dm." too close to ".$psrs_in_beams{$padded_prim_beam}{$psr_dm}." [DM=".$psr_dm."]");
            $legit = 0;
          }
        }

        if (($pid eq "P858") && 
            (($source eq "J1819-1458") || ($source eq "J1644-4559") || ($source eq "J1752-2806")))
        {
          $legit = 1;
          $cc_email = "evan.keane\@gmail.com";
        }

        $frbs_legit{$frb} = $legit;

        if ($legit)
        {
          # we have a least 1 FRB in this list
          $num_frbs_legit++;

          $filter_time   = (2 ** $filter) * 0.000064;

          $html .= "<tr>";
          $html .= "<td>".$snr."</td>";
          $html .= "<td>".sprintf("%5.2f",$time)."</td>";
          $html .= "<td>".$dm."</td>";
          $html .= "<td>".($filter_time * 1000)."</td>";
          $html .= "<td>".$padded_prim_beam."</td>";
          $html .= "<td>".$related_psrs."</td>";
          $html .= "</tr>\n";

          $cmd = "dmsmear -f 1382 -b 400 -n 1024 -d ".$dm." -q";
          Dada::logMsg(2, DL, "detectFRBs: cmd=".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
          $response =~ s/ +$//;
          $smearing_time = $response;

          $total_time = $filter_time + $smearing_time;

          # be generous and allow 2 times the smearing time before and after the event
          $start_time = $time - (1 * $total_time);
          $end_time   = $time + (2 * $total_time);

          Dada::logMsg(2, DL, "detectFRBs: FRB filter_time=".$filter_time." smearing_time=".$smearing_time." total_time=".$total_time);
          Dada::logMsg(2, DL, "detectFRBs: FRB start_time=".$start_time." end_time=".$end_time);

          # determine abosulte start time in UTC
          $start_utc = Dada::addToTimeFractional($obs, $start_time);

          # determine abosulte end time in UTC
          $end_utc = Dada::addToTimeFractional($obs, $end_time);

          # the event time at top of band in UTC
          $event_utc = Dada::addToTimeFractional($obs, $time);

          $frb_event_string .= $start_utc." ".$end_utc." ".$dm." ".$snr." ".$prim_beam."\n";
          $mpsr_event_string .= $event_utc." ".$snr." ".$dm." ".$filter."\n";
        }
      }
      $html .= "  </table>";

      $html .= "<hr/>";

      $html .= $src_html;

      # now only do something if we have at least 1 legitimate FRB, but more than 3 is RFI
      if ($num_frbs_legit > 0) 
      {
        Dada::logMsg(1, DL, "detectFRBs: EMAIL FRB Detection for ".$obs.", ".$num_frbs_legit." events");

        if ($num_frbs_legit < 4)
        {
          # build the socket command line
          my $sock_string = "N_EVENTS ".($num_frbs_legit)."\n".
                            $obs."\n".
                            $frb_event_string;

          my $mpsr_sock_string = "N_EVENTS ".($num_frbs_legit)."\n".
                                 $obs."\n".
                                 $mpsr_event_string;

          print $sock_string;

          my $event_sock = 1;
          if ($event_sock)
          {
            my $i = 0;
            for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
            {
              $host = $cfg{"PWC_".$i};
              $port = int($cfg{"CLIENT_EVENT_BASEPORT"}) + int($i);
              Dada::logMsg(2, DL, "detectFRBs: opening connection for FRB dump to ".$host.":".$port);
              $sock = Dada::connectToMachine($host, $port);
              if ($sock)
              {
                Dada::logMsg(2, DL, "detectFRBs: connection to ".$host.":".$port." established");

                print $sock $sock_string;
                close ($sock);
                $sock = 0;
              }
              else
              {
                Dada::logMsg(0, DL, "detectFRBs: connection to ".$host.":".$port." failed");
              }
            }

            if ($pid eq "P999")
            {
              my $mpsr_host = "utmost.usyd.edu.au";
              my $mpsr_port = "23456";

              my $mpsr_sock = Dada::connectToMachine($mpsr_host, $mpsr_port, 1);
              if ($mpsr_sock)
              {
                Dada::logMsg(0, DL, "detectFRBs: connected to ".$mpsr_host.":".$mpsr_port);
                Dada::logMsg(0, DL, "detectFRBs: sending event information");

                # remove trailing \n since sendTelnetCommand will add \r\n

                chomp $mpsr_sock_string;
                ($result, $response) = Dada::sendTelnetCommand ($mpsr_sock, $mpsr_sock_string);
                if ($result ne "ok")
                {
                  Dada::logMsg(0, DL, "detectFRBs: failed to send FRB event information to MPSR: ".$response);
                  $html .= "<P>NOTE: ERROR: could not send FRB event information to Molonglo for FRB dump</p>";
                } 
                else
                {
                  $html .= "<P>NOTE: FRB dump command issued to Molonglo</p>";
                  Dada::logMsg(0, DL, "detectFRBs: sent FRB event information to MPSR");
                }
                close ($mpsr_sock);
              }
              else
              {
                $html .= "<P>NOTE: ERROR: Could not connect to Molonglo for FRB dump</p>";
                Dada::logMsg(0, DL, "detectFRBs: could not connect to ".$mpsr_host.":".$mpsr_port);
              }
              $mpsr_sock = 0;
            }
          }
        }

        if ($num_frbs_legit > 5)
        {
          $html .= "<p>NOTE: ".($num_frbs_legit)." FRBs were detected. This execeeded the ".
                   " limit of 5, so none of them were plotted</p>";
        }

        $html .= "<hr/>";
        $html .= "<h3>Plots</h3>";
        $html .= "</body>";

        ### Add the html message part:
        $msg->attach(
          Type     => 'text/html',
          Data     => $html,
        );

        $cmd = "find -maxdepth 1 -type f -name '2*.cands_1024x768.png' | sort -n | tail -n 1";
        Dada::logMsg(2, DL, "detectFRBs: cmd=".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
        if ($result eq "ok")
        {
          ### Attach a part... the make the message a multipart automatically:
          $msg->attach(
            Type        => 'image/png',
            Id          => 'heimdall_overview',
            Path        => $cfg{"SERVER_RESULTS_DIR"}.'/'.$obs.'/'.$response,
            Filename    => $frb_name.'.png',
            Disposition => 'attachment'
          );
        }

        # if we have more than 5 FRB events, dont plot them all
        if ($num_frbs_legit <= 5)
        {
          foreach $frb (@frbs)
          {
            if ($frbs_legit{$frb})
            {
              Dada::logMsg(2, DL, "detectFRBs: FRB=".$frb);
              my ($snr, $time, $sample, $dm, $filter, $prim_beam) = split(/\t/, $frb);

              # determine host for the beam
              my $i=0;
              $host = "";
              my $beam = sprintf("%02d", $prim_beam);
              for ($i=0; (($host eq "") && ($i<$cfg{"NUM_PWC"})); $i++)
              {
                Dada::logMsg(3, DL, "detectFRBs: testing if ".$roach{"BEAM_".$i}." eq ".$beam);
                if ($roach{"BEAM_".$i} eq $beam)
                {
                  $host = $cfg{"PWC_".$i};
                }
              }
    
              if ($host eq "")
              {
                next;
              }

              my $fil_file = $cfg{"CLIENT_ARCHIVE_DIR"}."/".$beam."/".$obs."/".$obs.".fil";
              my $plot_cmd = "trans_freqplus_plot.py ".$fil_file." ".$sample." ".$dm." ".$filter." ".$snr;
              my $local_img = $obs."_".$beam."_".$sample.".png";

              # create a freq_plus file
              $cmd = "ssh bpsr@".$host." '".$plot_cmd."' > /tmp/".$local_img;
              Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);

              # get the first 5 chars / bytes of the file 
              $cmd = "head -c 5 /tmp/".$local_img;
              Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);

              # if we did manage to find the filterbank file
              if (($result eq "ok") && ($response ne "ERROR"))
              {
                $msg->attach(
                  Type        => 'image/png',
                  Path        => '/tmp/'.$local_img,
                  Filename    => $local_img,
                  Disposition => 'attachment'
                );
              }
            }
          }
        }

        $msg->send;

        $cmd = "rm -f /tmp/*_??_*.png";
        Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
      }
    }
  }
  else
  {
    Dada::logMsg(2, DL, "detectFRBs: NO FRB Detection for ".$obs);
    $last_frb = "";
  }

  return ("ok", $last_frb);
}


# Handle INT AND TERM signals
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(3);
  print STDERR basename($0)." : Exiting: ".Dada::getCurrentDadaTime(0)."\n";
  exit(1);

}
                                                                                
sub controlThread($) 
{
  (my $pid_file) = @_;
  Dada::logMsg(2, DL, "controlThread: starting");

  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  my $result = "";
  my $response = "";

  #Dada::logMsg(0, DL, "controlThread: killProcess(^coincidencer)");
  #($result, $response) = Dada::killProcess("^coincidencer");
  #Dada::logMsg(0, DL, "controlThread: ".$result." ".$response);
  
  if (-f $pid_file)
  {
    Dada::logMsg(2, DL, "controlThread: unlinking PID file: ".$pid_file);
    unlink($pid_file);
  }

  Dada::logMsg(2, DL, "controlThread: exiting");
}

sub coincidencerThread()
{
  Dada::logMsg(1, DL, "coincidencerThread: starting");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "coincidencer -a ".$cfg{"SERVER_HOST"}." -p ".$cfg{"SERVER_COINCIDENCER_PORT"}.
         " -n ".$cfg{"NUM_PWC"}." | server_bpsr_server_logger.pl -n coin";
  Dada::logMsg(1, DL, "coincidencerThread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(1, DL, "coincidencerThread: ".$result." ".$response);

  Dada::logMsg(1, DL, "coincidencerThread: exiting");
}
