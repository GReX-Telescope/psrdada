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

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;
our $coarse_nchan;

#
# Initialize global variables
#
%cfg = Mopsr::getConfig();
$dl = 1;
$daemon_name = Dada::daemonBaseName($0);
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$coarse_nchan = 32;


# Autoflush STDOUT
$| = 1;


# 
# Function Prototypes
#
sub main();
sub countHeaders($);
sub getArchives($);
sub getObsAge($$);
sub markObsState($$$);
sub checkClientsFinished($$);
sub processObservation($$);
sub processArchive($$$$);
sub makePlotsFromArchives($$$$$$);
sub removeOldPngs($);
sub getObsMode($);

#
# Main
#
my $result = 0;
$result = main();

exit($result);


###############################################################################
#
# package functions
# 

sub main() 
{
  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $obs_results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  my $control_thread   = 0;
  my @observations = ();
  my ($i, $obs_mode);
  my $o = "";
  my $t = "";
  my $n_obs_headers = 0;
  my $result = "";
  my $response = "";
  my $counter = 5;
  my $cmd = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  ## set the 
  umask 022;
  my $umask_val = umask;

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  chdir $obs_results_dir;

  while (!$quit_daemon)
  {

    # TODO check that directories are correctly sorted by UTC_START time
    Dada::logMsg(2, $dl, "main: looking for obs.processing in ".$obs_results_dir);

    # Only get observations that are marked as procesing
    $cmd = "find ".$obs_results_dir." -mindepth 2 -maxdepth 2 -name 'obs.processing' ".
           "-printf '\%h\\n' | awk -F/ '{print \$NF}' | sort -n";

    Dada::logMsg(3, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);

    if ($result eq "ok") 
    {
      @observations = split(/\n/,$response);

      # For process all valid observations
      for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++)
      {
        $o = $observations[$i];
        $obs_mode = getObsMode($o);

        # Get the number of band subdirectories
        Dada::logMsg(3, $dl, "main: countHeaders(".$o.")");
        $n_obs_headers = countHeaders($o);
        Dada::logMsg(3, $dl, "main: countHeaders() ".$n_obs_headers);
        Dada::logMsg(2, $dl, "main: ".$o." had ".$n_obs_headers." obs.header files");

        # check how long ago the last result was received
        # negative values indicate that no result has ever been received
        # and is the age of the youngest file (but -ve)
        Dada::logMsg(2, $dl, "main: getObsAge(".$o.", ".$n_obs_headers.")");
        $t = getObsAge($o, $n_obs_headers);
        Dada::logMsg(3, $dl, "main: getObsAge() ".$t);
        Dada::logMsg(2, $dl, "main: ".$o." obs age=".$t);
  
        if ($obs_mode eq "PSR")
        {
          Dada::logMsg(2, $dl, "main: checking obs ".$o." for archives");
          # if newest archive was more than 32 seconds old, finish the obs.
          if ($t > 32)
          {
            processObservation($o, $n_obs_headers);
            markObsState($o, "processing", "finished");
            cleanResultsDir($o);
          } 

          # we are still receiving results from this observation
          elsif ($t >= 0) 
          {
            processObservation($o, $n_obs_headers);
          }

          # no archives yet received, wait
          elsif ($t > -60)
          {
            # no archives have been received 60+ seconds after the
            # directories were created, something is wrong with this
            # observation, mark it as failed

          } else {
            Dada::logMsg(0, $dl, "main: processing->failed else case");
            markObsState($o, "processing", "failed");
          }
        }

        if (($obs_mode eq "CORR") || ($obs_mode eq "CORR_CAL"))
        {
          Dada::logMsg(2, $dl, "CORR obs: ".$o." t=".$t);
          # we have not received a correlation dump for at least 32s
          if ($t > 64)
          {
            Dada::logMsg(2, $dl, "CORR marking as finished");
            markObsState($o, "processing", "finished");
          }

          #  we are still receiving correlation dumps from this observation
          elsif ($t >= 0)
          {
            Dada::logMsg(2, $dl, "CORR processing dumps");
            processCorrObservation($o, $n_obs_headers);
          }
          elsif ($t > -80)
          {
            Dada::logMsg(2, $dl, "main: still waiting for dumps from a CORR observatino");
          }
          else
          {
            # no correlation dumps have been received for at least 80 seconds,
            # this is a failed observation
            Dada::logMsgWarn($warn, "No correlator dumps received for 80 seconds");
            markObsState($o, "processing", "failed");
          }

        }
      }
    }

    # if no obs.processing, check again in 5 seconds
    if ($#observations == -1) {
      $counter = 5;
    } else {
      $counter = 2;
    }
   
    while ($counter && !$quit_daemon) {
      sleep(1);
      $counter--;
    }

  }
  Dada::logMsg(2, $dl, "main: joining controlThread");
  $control_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

                                                                                
  return 0;
}

sub getObsMode($)
{
  (my $obs) = @_;

  my $obs_info = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.info";
  my $mode = "UNKNOWN";

  if (-f $obs_info)
  {
    Dada::logMsg(2, $dl, "main: obs_info=".$obs_info);
    my $cmd = "grep MODE ".$obs_info." | awk '{print \$2}'";
    my ($result, $response) = Dada::mySystem($cmd);
    if ($result eq "ok")
    {
      $mode = $response;
    }
  }

  return $mode;
}


###############################################################################
#
# Returns the "age" of the observation. Return value is the age in seconds of 
# the file of type $ext in the obs dir $o. If no files exist in the dir, then
# return the age of the newest dir in negative seconds
# 
sub getObsAge($$)
{
  my ($o, $num_obs_headers) = @_;
  Dada::logMsg(3, $dl, "getObsAge(".$o.", ".$num_obs_headers.")");

  my $result = "";
  my $response = "";
  my $age = 0;
  my $time = 0;
  my $cmd = "";

  # current time in "unix seconds"
  my $now = time;

  # If no obs.header files yet exist, we return the age of the obs dir
  if ($num_obs_headers== 0)
  {
    $cmd = "find ".$o." -maxdepth 0 -type d -printf '\%T@\\n'";
    Dada::logMsg(3, $dl, "getObsAge: ".$cmd); 
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
    $time = $response;
    $age = $time - $now;

  # We have some obs.header files, see if we have any archives
  } 
  else
  {
    $cmd = "find ".$o." -type f -name '*.ar' -printf '\%T@\\n' ".
           "-o -name '*.tot' -printf '\%T@\\n' ".
           "-o -name '*.cc' -printf '\%T@\\n' ".
           "-o -name '*.sum' -printf '\%T@\\n' ".
           "| sort -n | tail -n 1";

    Dada::logMsg(3, $dl, "getObsAge: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
  
    # If we found a file
    if ($response ne "")
    {
      # we will be returning a positive value
      $time = $response;
      $age = $now - $time;
    }

    # No files were found, use the age of the obs.header files instead
    else
    {
      $cmd = "find ".$o." -type f -name 'obs.header' -printf '\%T@\\n' | sort -n | tail -n 1";
      Dada::logMsg(3, $dl, "getObsAge: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "getObsAge: ".$result." ".$response);
  
      # we will be returning a negative value
      $time = $response;
      $age = $time - $now;
    }
  }

  Dada::logMsg(3, $dl, "getObsAge: time=".$time.", now=".$now.", age=".$age);
  return $age;
}

###############################################################################
#
# Marks an observation as finished
# 
sub markObsState($$$) {

  my ($o, $old, $new) = @_;

  Dada::logMsg(2, $dl, "markObsState(".$o.", ".$old.", ".$new.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $archives_dir = $cfg{"SERVER_ARCHIVE_DIR"};
  my $results_dir  = $cfg{"SERVER_RESULTS_DIR"};
  my $state_change = $old." -> ".$new;
  my $old_file = "obs.".$old;
  my $new_file = "obs.".$new;
  my $file = "";
  my $ndel = 0;

  Dada::logMsg(1, $dl, $o." ".$old." -> ".$new);

  $cmd = "touch ".$results_dir."/".$o."/".$new_file;
  Dada::logMsg(2, $dl, "markObsState: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "markObsState: ".$result." ".$response);

  $file = $results_dir."/".$o."/".$old_file;
  if ( -f $file ) {
    $ndel = unlink ($file);
    if ($ndel != 1) {
      Dada::logMsgWarn($warn, "markObsState: could not unlink ".$file);
    }
  } else {
    Dada::logMsgWarn($warn, "markObsState: expected file missing: ".$file);
  }
}


###############################################################################
# 
# Clean up the results directory for the observation
#
sub cleanResultsDir($) {

  (my $o) = @_;

  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$o;
  my ($ant, $source, $first_obs_header);
  my ($cmd, $result, $response);
  my @sources = ();
  my @ants = ();

  $cmd = "rm -f ".$results_dir."/*/*_pwc.finished";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok"){
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove delete [PWC]_pwc.finished files");
  }

  # get a list of the ants for this obs
  $cmd = "find ".$results_dir." -mindepth 1 -maxdepth 1 -type d -printf '%f\n'";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not get a list of ants");
  }
  @ants = split(/\n/, $response);

  # get a list of the sources for this obs
  $cmd = "find ".$results_dir." -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' -printf '\%f\\n' | awk -F_ '{print \$1}'";
  Dada::logMsg(2, $dl, "cleanResultsDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "cleanResultsDir: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "cleanResultsDir: ".$cmd." failed: ".$response);
    return ("fail", "Could not remove get a list of the sources");
  }
  @sources = split(/\n/, $response);

  Dada::logMsg(2, $dl, "cleanResultsDir: deleting old pngs");
  removeOldPngs($o);
}

sub processCorrObservation($$)
{
  my ($o, $n_obs_headers) = @_;
  Dada::logMsg(2, $dl, "processCorrObservation(".$o.", ".$n_obs_headers.")");

  my $i = 0;
  my $k = "";
  my ($archive_dir, $nchan, $summed_file, $first_time);
  my ($cmd, $result, $response, $ichan, $file_list);
  my ($file, $junk, $chan_dir, $tsum, $plus, $ifile);
  my ($key);
  my @chans = ();
  my @archives = ();
  my @dumps = ();
  my @files = ();
  my %unprocessed = ();

  # ensure the archives dir exists
  $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$o;

  if (! -d $archive_dir)
  {
    $cmd = "mkdir -m 0755 ".$archive_dir;
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processCorrObservation: ".$cmd." failed: ".$response);
      return ("fail", "could not create dir");
    }
  }

  # move into the observation directory
  chdir $o;

  # get a list of all thechannels
  $cmd = "find . -mindepth 1 -maxdepth 1 -type d -name 'CH*' -printf '%f\n' | sort -n";
  Dada::logMsg(3, $dl, "processCorrObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "processCorrObservation: ".$cmd." failed: ".$response);
    return;
  }

  @chans = split(/\n/, $response);
  $nchan = $#chans + 1;

  # get a list of the unprocessed correlator dumps
  $cmd = "find . -mindepth 2 -maxdepth 2 -type f -name '*.?c' -printf '%f\n' | sort -n";
  Dada::logMsg(3, $dl, "processCorrObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "processCorrObservation: ".$cmd." failed: ".$response);
    return;
  }
  @files = split(/\n/, $response);

  foreach $file (@files) 
  {
    if (! exists($unprocessed{$file}))
    {
      $unprocessed{$file} = 0;
    }
    $unprocessed{$file} += 1;
  }

  @files = sort keys %unprocessed;
  my $n_appended = 0;
  for ($ifile=0; $ifile<=$#files; $ifile++)
  {
    $file = $files[$ifile];
    $summed_file = 0;

    if ($unprocessed{$file} == $nchan) 
    {
      Dada::logMsg(2, $dl, "processCorrObservation: appendCorrFile(".$file.")");
      ($result, $response) = appendCorrFile($file);
      if ($result ne "ok")
      {
        Dada::logMsgWarn($warn, "processCorrObservation: failed to Fappend file [".$file."]: ".$response);
        return ("fail", "could not Fappend file");;
      }
      $n_appended++;
      $summed_file = $response;

      $tsum = "ac.sum";
      $plus = "-a";

      if ($file =~ m/\.cc/)
      {
        $tsum = "cc.sum";
        $plus = "-c";
      }

      # now add this file to the Tscrunched total
      $first_time = 0;
      $cmd = "mopsr_corr_sum ".$plus." ".$tsum." ".$plus." ".$summed_file." ".$tsum;

      if (!( -f $tsum))
      {
        $first_time = 1;
        $cmd = "cp ".$summed_file." ".$tsum;
      }

      Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);

      # finally moved this file to the archives directory
      $cmd = "mv ".$summed_file." ".$archive_dir."/";
      Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);

      # create an FSUM'ed obs.header for this observation
      if (($first_time) && ($tsum eq "cc.sum"))
      {
        $cmd = "find . -name 'obs.header' | head -n 1";
        Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          Dada::logMsg(0, $dl, "processCorrObservation: ".$cmd." failed: ".$response);
          return ("fail", "could not find a obs.header file");
        }
        my %h = Dada::readCFGFileIntoHash($response, 0);

        # now determine the centre frequnecy
        $cmd = "grep ^FREQ */obs.header | awk '{print \$2}' | sort -n";
        Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
        if (($result ne "ok") || ($response eq ""))
        {
          Dada::logMsg(0, $dl, "processCorrObservation: ".$cmd." failed: ".$response);
          return ("fail", "could not extract coarse channel freqs");
        }

        my @freqs = split(/\n/, $response);
        my $freq_lo = $freqs[0];
        my $freq_hi = $freqs[$#freqs];
        my $new_freq = $freq_lo + (($freq_hi - $freq_lo) / 2);

        my $new_bw = $nchan * $h{"BW"};

        # overwrite the old values with new ones
        $h{"FREQ"}  = $new_freq;
        $h{"BW"}    = $new_bw;
        $h{"NCHAN"} = $nchan;
        $h{"ORDER"} = "SF";

        my $fname = "obs.header";
        open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
        print FH "# Specification File created by ".$0."\n";
        print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

        foreach $key ( keys %h )
        {
          # ignore some irrelvant keys
          if (!($key =~ m/ANT_ID_/))
          {
            print FH Dada::headerFormat($key, $h{$key})."\n";
          }
        }
        close FH;

        $cmd = "cp obs.header ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o."/";
        Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
      }
    }
  }

  if (($n_appended > 0) && (-f "cc.sum"))
  {
    my $tsamp = 1.28;
    my $band_tsamp = $tsamp / $nchan;
    my $band_nchan = $nchan * $coarse_nchan;

    $cmd = "mopsr_solve_delays ".$band_nchan." cc.sum obs.antenna -t ".$band_tsamp." -p -a 0 -r /home/dada/phonecall";
    Dada::logMsg(2, $dl, "processCorrObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processCorrObservation: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processCorrObservation: ".$cmd." failed: ".$response);
      return ("fail", "failed to solve for delays");
    }
  }

  chdir $cfg{"SERVER_RESULTS_DIR"};

}


sub appendCorrFile($)
{
  my ($file) = @_;

  my ($cmd, $result, $response, $chan_file, $plus);
  my @chan_files;

  $cmd = "find . -name '".$file."' | sort -n";
  Dada::logMsg(2, $dl, "appendCorrFile: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendCorrFile: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "appendCorrFile: ".$cmd." failed: ".$response);
    return ("fail", "could not get file list");
  }

  @chan_files = split(/\n/, $response);
  $cmd = "mopsr_corr_fsum ".$coarse_nchan." ";

  $plus = "-a";
  if ($file =~ m/\.cc$/)
  {
    $plus = "-c";
  }
  
  foreach $chan_file (@chan_files)
  {
    $cmd .= " ".$plus." ".$chan_file;
  }

  $cmd .= " ".$file;

  Dada::logMsg(2, $dl, "appendCorrFile: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendCorrFile: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "appendCorrFile: ".$cmd." failed: ".$response);
    return ("fail", "could not get file list");
  }

  foreach $chan_file (@chan_files)
  {
    unlink $chan_file;
  }


  return ("ok", $file);
}


##############################################################################
#
# Process all possible archives in the observation, combining the bands
# and plotting the most recent images. Accounts for multifold  PSRS
#
sub processObservation($$) 
{
  my ($o, $n_obs_headers) = @_;
  Dada::logMsg(2, $dl, "processObservation(".$o.", ".$n_obs_headers.")");

  my $i = 0;
  my $k = "";
  my ($fres_ar, $tres_ar, $latest_archive);
  my ($source, $archive, $file, $ant);
  my ($cmd, $result, $response, $iant);
  my @ants = ();
  my @archives = ();

  # get the source for this observation
  $cmd = "grep ^SOURCE ".$o."/obs.info | awk '{print \$2}'";
  Dada::logMsg(3, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "processObservation:  ".$cmd." failed: ".$response);
    return;
  }
  $source = $response;

  # get a list of all the antenna
  $cmd = "find ".$o." -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n";
  Dada::logMsg(3, $dl, "processObservation: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processObservation: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsgWarn($warn, "processObservation: ".$cmd." failed: ".$response);
    return;
  }

  @ants = split(/\n/, $response);

  for ($iant=0; (($iant<=$#ants) && (!$quit_daemon)); $iant++)
  {
    $ant = $ants[$iant];

    # get a list of the unprocessed archives for this antenna
    $cmd = "find ".$o."/".$ant." -mindepth 1 -maxdepth 1 -type f -name '2*.ar' -printf '%f\n' | sort -n";
    Dada::logMsg(3, $dl, "processObservation: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processObservation: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processObservation: ".$cmd." failed: ".$response);
      return;
    }
    
    Dada::logMsg(2, $dl, "processObservation: appendArchives(".$o.", ".$source.", ".$ant.")");
    ($result, $fres_ar, $tres_ar) = appendArchives($o, $ant, $source);
    Dada::logMsg(3, $dl, "processObservation: appendArchives() ".$result);

    if (($result eq "ok") && ($fres_ar ne "") && ($tres_ar ne ""))
    {
      $cmd = "find ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$o."/".$ant."/ -name '2*.ar' | sort -n | tail -n 1";
      Dada::logMsg(2, $dl, "processObservation: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $dl, "processObservation: ".$result." ".$response);
      if ($result eq "ok") {
        $latest_archive = $response;
      } else {
        $latest_archive = "";
      }

      Dada::logMsg(2, $dl, "processObservation: plotting [".$i."] (".$o.", ".$ant.", ".$source.", ".$fres_ar.", ".$tres_ar.")");
      makePlotsFromArchives($o, $ant, $fres_ar, $tres_ar, "120x90", $latest_archive);
      makePlotsFromArchives($o, $ant, $fres_ar, $tres_ar, "1024x768", $latest_archive);
    }
  }
  removeOldPngs($o);
}




###############################################################################
#
# Append the ARCHIVE_MOD second summed archive to the total for this observation
#
sub appendArchive($$$$) {

  my ($utc_start, $ant, $source, $archive) = @_;

  Dada::logMsg(2, $dl, "appendArchive(".$utc_start.", ".$ant.", ".$source.", ".$archive.")");

  my $total_t_sum = $utc_start."/".$ant."/".$archive;
  my $source_f_res = $utc_start."/".$ant."/".$source."_f.tot";
  my $source_t_res = $utc_start."/".$ant."/".$source."_t.tot";
  my $sixty_four_log = $utc_start."/".$ant."/snr_60.log";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $new_pm_text = "";
  my @powers = ();
  my $power = "";

  my $nchan = (int($cfg{"PWC_END_CHAN"}) - int($cfg{"PWC_START_CHAN"})) + 1;
  Dada::logMsg(1, $dl, "appendArchive: nchan=".$nchan." END=".$cfg{"PWC_END_CHAN"}." START=".$cfg{"PWC_START_CHAN"});
  my $psh;
  if ($nchan <= 8)
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon8.psh";
  }
  elsif ($nchan == 20)
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon20.psh";
  }
  else
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon.psh";
  }

  if (! -f $total_t_sum) 
  {
    Dada::logMsg(0, $dl, "appendArchive: archive [".$total_t_sum."] did not exist");
    return ("fail", "", "");
  } 

  # If the server's archive dir for this observation doesn't exist with the source
  if (! -d $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source) {
    $cmd = "mkdir -m 0755 -p ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$ant;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
  } 
    
  # save this archive to the server's archive dir for permanent archival
  $cmd = "cp --preserve=all ./".$total_t_sum." ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$ant."/";
  Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  $new_pm_text = "";
  # The total power monitor needs first line as int:freq
  if ( ! -f $utc_start."/".$ant."/power_monitor.log" ) 
  {
    $cmd = "psrstat -J ".$psh." -Q -c 'int:freq' ./".$total_t_sum." | awk '{print \$2}' | awk -F, '{ printf (\"UTC\"); for(i=1;i<=NF;i++) printf (\",\%4.0f\",\$i); printf(\"\\n\") }'";
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result eq "ok")
    {
      $new_pm_text = $response."\n";
    }
  }

  $cmd = "psrstat -J ".$psh." -Q -q -l chan=0- -c all:sum ./".$total_t_sum." | awk '{printf(\"%6.3f\\n\",\$1)}'";
  Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
  if ($result eq "ok")
  {
    @powers = split(/\n/,$response);

    $archive =~ s/\.ar$//;
    my $archive_time_unix = Dada::getUnixTimeUTC($archive);
    my $utc_time_unix = Dada::getUnixTimeUTC($utc_start);
    my $offset = $archive_time_unix - $utc_time_unix;

    $new_pm_text .= $offset;
    foreach $power ( @powers )
    {
      $new_pm_text .= ",".$power;
    }
    $new_pm_text .= "\n";

    open FH, ">>$utc_start/$ant/power_monitor.log";
    print FH $new_pm_text;
    close FH;
  }

  # If this is the first result for this observation
  if (!(-f $source_f_res)) 
  {
    # "create" the source's fres archive
    $cmd = "cp ".$total_t_sum." ".$source_f_res;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Fscrunc the archive
    $cmd = "pam -F -m ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Now we have the tres archive
    $cmd = "cp ".$total_t_sum." ".$source_t_res;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
  
  # we are appending to the sources f and t res archives
  } else {

    # Add the new archive to the FRES total [tsrunching it]
    $cmd = "psradd -T -o ".$source_f_res." ".$source_f_res." ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Fscrunc the archive for adding to the TRES
    $cmd = "pam -F -m ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # Add the Fscrunched archive to the TRES total 
    $cmd = "psradd -o ".$source_t_res." ".$source_t_res." ".$total_t_sum;
    Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    Dada::logMsg(2, $dl, "appendArchive: done");
  }

  # If this source looks like a CAL
  if (0)
  {
    if (( $source =~ m/_R$/ ) || ( $source =~ m/^HYDRA/ ))
    {
      Dada::logMsg(1, $dl, "appendArchive: skipping SNR histogram as ".$source." is a CAL");
    }
    else
    {
      $cmd = "psredit -c length ".$source_t_res." -Q | awk '{print \$2}'";
      Dada::logMsg(2, $dl, "appendArchive: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "appendArchive: ".$result." ".$response);

      if (($result eq "ok") && (int($response) > 60))
      {
        # get the snr and length of last 60 seconds
        $cmd = "last_60s_snr.psh ".$source_t_res;
        Dada::logMsg(1, $dl, "appendArchive: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(1, $dl, "appendArchive: ".$result." ".$response);

        my $sixty_four_nsubint = "";
        my $sixty_four_length = "";
        my $sixty_four_snr = "";

        my @lines = ();
        my $line = "";
        my @bits = ();

        if ($result ne "ok")
        {
           Dada::logMsg(0, $dl, "appendArchive: ".$cmd." failed: ".$response);
        }
        else
        {
          @lines = split(/\n/, $response);
          foreach $line (@lines)
          {
            Dada::logMsg(3, $dl, "appendArchive: line=".$line);
            @bits = split(/=/, $line);
            if ($bits[0] =~ /nsubint/) { $sixty_four_nsubint = $bits[1]; }
            if ($bits[0] =~ /length/)  { $sixty_four_length  = $bits[1]; }
            if ($bits[0] =~ /snr/)     { $sixty_four_snr     = $bits[1]; }
          }
        }

        if ($sixty_four_nsubint eq "6")
        {
          # update this observations log
          open FH, ">>".$sixty_four_log;
          Dada::logMsg(2, $dl, "appendArchive: writing snr=".$sixty_four_snr." length=". $sixty_four_length);
          print FH "snr=".$sixty_four_snr." length=". $sixty_four_length."\n";
          close FH;
        }
      }
    }
  }

  # clean up the current archive
  unlink($total_t_sum);
  Dada::logMsg(2, $dl, "appendArchive: unlinking ".$total_t_sum);

  return ("ok", $source_f_res, $source_t_res);

}

sub appendArchives($$$)
{
  my ($utc_start, $ant, $source) = @_;

  my ($cmd, $result, $response, $archive, $new_pm_text);
  my (@powers, $power, @archives, $psh);

  my $nchan = (int($cfg{"PWC_END_CHAN"}) - int($cfg{"PWC_START_CHAN"})) + 1;
  if ($nchan <= 8)
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon8.psh";
  }
  elsif ($nchan == 20)
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon20.psh";
  }
  else
  {
    $psh = $cfg{"SCRIPTS_DIR"}."/power_mon.psh";
  }
  Dada::logMsg(3, $dl, "appendArchives: nchan=".$nchan." psh=".$psh);


  # get a list of the unprocessed archives for this antenna
  $cmd = "find ".$utc_start."/".$ant." -mindepth 1 -maxdepth 1 -type f -name '2*.ar' -printf '%f\n' | sort -n";
  Dada::logMsg(3, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "appendArchives: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  my @files = split(/\n/, $response);
  if ($#files == -1)
  {
    Dada::logMsg(2, $dl, "appendArchives: no archives found for ".$utc_start."/".$ant);
    return ("ok", "", "");
  }

  Dada::logMsg(1, $dl, "processing ".$utc_start."/".$ant." ".($#files+1)." files");

  # If the server's archive dir for this observation doesn't exist with the source
  if (! -d $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$source) {
    $cmd = "mkdir -m 0755 -p ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$ant;
    Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "appendArchives: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }
  }

  my $archive_list = "";
  my $file;
  foreach $file ( @files)
  {
    $archive =  $utc_start."/".$ant."/".$file;

    $archive_list .= $archive." ";

    # save this archive to the server's archive dir for permanent archival
    $cmd = "cp --preserve=all ./".$archive." ".$cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start."/".$ant."/";
    Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "appendArchives: ".$cmd." failed: ".$response);
      return ("fail", "", "");
    }

    # first create the power-monitor log file for plotting
    $new_pm_text = "";
    # The total power monitor needs first line as int:freq
    if ( ! -f $utc_start."/".$ant."/power_monitor.log" )
    {
      $cmd = "psrstat -J ".$psh." -Q ".
             "-c 'int:freq' ./".$archive." | awk '{print \$2}' ".
             "| awk -F, '{ printf (\"UTC\"); for(i=1;i<=NF;i++) ".
             "printf (\",\%4.0f\",\$i); printf(\"\\n\") }'";
      Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
      if ($result eq "ok")
      {
        $new_pm_text = $response."\n";
      }
    }

    $cmd = "psrstat -J ".$psh." -Q -q -l chan=0- -c all:sum ./".$archive." | awk '{printf(\"%6.3f\\n\",\$1)}'";
    Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);

    @powers = ();
    if ($result eq "ok")
    {
      @powers = split(/\n/,$response);
      my $file_time = $file;

      $file_time =~ s/\.ar$//;
      my $archive_time_unix = Dada::getUnixTimeUTC($file_time);
      my $utc_time_unix = Dada::getUnixTimeUTC($utc_start);
      my $offset = $archive_time_unix - $utc_time_unix;

      $new_pm_text .= $offset;
      foreach $power ( @powers )
      {
        $new_pm_text .= ",".$power;
      }
      $new_pm_text .= "\n";

      open FH, ">>$utc_start/$ant/power_monitor.log";
      print FH $new_pm_text;
      close FH;
    }
  }

  my $f_res = $utc_start."/".$ant."/".$source."_f.tot";
  my $t_res = $utc_start."/".$ant."/".$source."_t.tot";

  my $f_res_interim = $utc_start."/".$ant."/".$source."_f.int";
  my $t_res_interim = $utc_start."/".$ant."/".$source."_t.int";

  if (-f $f_res_interim) 
  {
    unlink ($f_res_interim);
  }

  if ($#files > 0)
  {
    $cmd = "psradd -o ".$f_res_interim." -T ".$archive_list;
  }
  else
  {
    $cmd = "cp ".$archive_list." ".$f_res_interim;
  }
  Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);

  # if the f_res file does not yet exist
  if (! (-f $f_res)) 
  {
    $cmd = "mv ".$f_res_interim." ".$f_res;
  }
  else
  {
    $cmd = "psradd -T -o ".$f_res." ".$f_res." ".$f_res_interim;
  }
  Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsg(0, $dl, "appendArchives: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  if (-f $t_res_interim)
  {
    unlink ($t_res_interim);
  }

  if ($#files > 0)
  {
    $cmd = "psradd -o ".$t_res_interim." ".$archive_list;
  }
  else
  {
    $cmd = "cp ".$archive_list." ".$t_res_interim;
  }
  Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);

  $cmd = "pam -F -m ".$t_res_interim;
  Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "appendArchives: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  if (! -f $t_res)
  {
    $cmd = "mv ".$t_res_interim." ".$t_res;
  }
  else
  {
    $cmd = "psradd -o ".$t_res." ".$t_res." ".$t_res_interim;
  }
  Dada::logMsg(2, $dl, "appendArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "appendArchives: ".$result." ".$response);
  if ($result ne "ok") 
  {
    Dada::logMsg(0, $dl, "appendArchives: ".$cmd." failed: ".$response);
    return ("fail", "", "");
  }

  foreach $file (@files)
  {
    $archive =  $utc_start."/".$ant."/".$file;
    if (-f $archive)
    {
      Dada::logMsg(2, $dl, "appendArchives: unlinking ".$archive);
      unlink ($archive);
    }
  }

  if (-f $f_res_interim)
  {
    unlink ($f_res_interim);
  }
  if (-f $t_res_interim)
  {
    unlink ($t_res_interim);
  }

  return ("ok", $f_res, $t_res);
}




###############################################################################
#
# Returns a hash of the archives in the results directory for the specified
# observation
#
sub getArchives($) {

  my ($utc_start) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $file = "";
  my @files = ();
  my %archives = ();
  my $source = "";
  my $host = "";
  my $archive = "";
  my $tag = "";
  
  $cmd = "find ".$utc_start." -mindepth 3 -name \"*.ar\" -printf \"\%P\\n\"";
  Dada::logMsg(3, $dl, "getArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getArchives: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "getArchives: ".$cmd." failed: ".$response);
    return %archives;
  }

  @files = sort split(/\n/,$response);

  foreach $file (@files) {

    Dada::logMsg(3, $dl, "getArchives: splitting ".$file);
    ($source, $host, $archive) = split(/\//, $file, 3);
    Dada::logMsg(3, $dl, "getArchives: ".$file." -> source=".$source." host=".$host.", archive=".$archive);
    $tag = $source."/".$archive;

    if (exists $archives{$tag}) {
      Dada::logMsgWarn($warn, "getArchives: ignoring duplicate tag for ".$file);
    } else {
      $archives{$tag} = $utc_start."/".$file;
      Dada::logMsg(2, $dl, "getArchives: adding tag ".$tag);
    }
  }

  return %archives;
}


################################################################################
#
# Look for *obs.header where the * prefix is the PWC name
# 
sub countHeaders($) {

  my ($dir) = @_;

  my $cmd = "find ".$dir." -name 'obs.header' | wc -l";
  my $find_result = `$cmd`;
  chomp($find_result);
  return $find_result;

}


###############################################################################
#
# Create plots for use in the web interface
#
sub makePlotsFromArchives($$$$$$) 
{
  my ($dir, $ant, $total_f_res, $total_t_res, $res, $ten_sec_archive) = @_;

  my $web_style_txt = $cfg{"SCRIPTS_DIR"}."/web_style.txt";
  my $args = "-g ".$res." -c above:c='".$dir."/".$ant."'";
  my $pm_args = "-g ".$res." -m ".$ant;
  my ($cmd, $result, $response);
  my ($bscrunch, $bscrunch_t);

  my $nchan = (int($cfg{"PWC_END_CHAN"}) - int($cfg{"PWC_START_CHAN"})) + 1;
  if ($nchan == 20)
  {
    $nchan = 4;
  }
  if ($nchan == 40)
  {
    $nchan = 8;
  }

  # If we are plotting hi-res - include
  if ($res ne "1024x768") {
    $args .= " -s ".$web_style_txt." -c below:l=unset";
    $bscrunch = "";
    $bscrunch_t = "";
    #$bscrunch = " -j 'B 256'";
    #$bscrunch_t = " -j 'B 256'";
    $pm_args .= " -p";
  } else {
    $bscrunch = "";
    $bscrunch_t = "";
  }

  my $bin = Dada::getCurrentBinaryVersion()."/psrplot ".$args;
  my $timestamp = Dada::getCurrentDadaTime(0);

  my $ti = $timestamp.".".$ant.".ti.".$res.".png";
  my $fr = $timestamp.".".$ant.".fr.".$res.".png";
  my $fl = $timestamp.".".$ant.".fl.".$res.".png";
  my $bp = $timestamp.".".$ant.".bp.".$res.".png";
  my $pm = $timestamp.".".$ant.".pm.".$res.".png";

  # Combine the archives from the machine into the archive to be processed
  # PHASE vs TIME
  $cmd = $bin.$bscrunch_t." -p time -jFD -D ".$dir."/pvt_tmp/png ".$total_t_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # PHASE vs FREQ
  $cmd = $bin.$bscrunch." -p freq -jTD -D ".$dir."/pvfr_tmp/png ".$total_f_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # PHASE vs TOTAL INTENSITY
  $cmd = $bin.$bscrunch." -p flux -jTFD -D ".$dir."/pvfl_tmp/png ".$total_f_res;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # BANDPASS
  $cmd = $bin." -pb -x -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # POWER MONITOR
  $cmd = "mopsr_pmplot -c ".$nchan." ".$pm_args." -D ".$dir."/pm_tmp/png ".$dir."/".$ant."/power_monitor.log";
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

  # wait for each file to "appear"
  my $waitMax = 5;
  while ($waitMax) {
    if ( (-f $dir."/pvfl_tmp") &&
         (-f $dir."/pvt_tmp") &&
         (-f $dir."/pvfr_tmp") &&
         (-f $dir."/bp_tmp") &&
         (-f $dir."/pm_tmp") )
    {
      $waitMax = 0;
    } else {
      $waitMax--;
      usleep(500000);
    }
  }

  # rename the plot files to their correct names
  system("mv -f ".$dir."/pvt_tmp ".$dir."/".$ti);
  system("mv -f ".$dir."/pvfr_tmp ".$dir."/".$fr);
  system("mv -f ".$dir."/pvfl_tmp ".$dir."/".$fl);
  system("mv -f ".$dir."/bp_tmp ".$dir."/".$bp);
  system("mv -f ".$dir."/pm_tmp ".$dir."/".$pm);
  Dada::logMsg(2, $dl, "makePlotsFromArchives: plots renamed");
}


###############################################################################
#
# Checks that all the *.pwc.finished files exist in the servers results dir. 
# This ensures that the obs is actually finished, for cases such as single 
# pulse
#
sub checkClientsFinished($$) {

  my ($obs, $n_obs_headers) = @_;

  Dada::logMsg(2, $dl ,"checkClientsFinished(".$obs.", ".$n_obs_headers.")");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $dir = $cfg{"SERVER_RESULTS_DIR"};

  # only declare this as finished when the band.finished files are written
  $cmd = "find ".$dir."/".$obs." -type f -name '*_pwc.finished' | wc -l";
  Dada::logMsg(2, $dl, "checkClientsFinished: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "checkClientsFinished: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "checkClientsFinished: ".$cmd." failed: ".$response);
  } else {
    if ($response eq $n_obs_headers) {
      $result = "ok";
      $response = "";
    } else {
      $result = "fail";
      $response = "not yet finished";
    }
  }

  Dada::logMsg(2, $dl ,"checkClientsFinished() ".$result." ".$response);
  return ($result, $response); 

}

#
# remove old pngs
#
sub removeOldPngs($)
{
  my ($dir) = @_;

  my ($cmd, $img_string, $i, $now);
  my ($time, $ant, $type, $res, $ext, $time_unix);

  $cmd = "find ".$dir." -ignore_readdir_race -mindepth 1 -maxdepth 1 -name '2*.*.??.*x*.png' -printf '%f\n' | sort -n";
  Dada::logMsg(2, $dl, "removeOldPngs: ".$cmd);

  $img_string = `$cmd`;
  my @images = split(/\n/, $img_string);
  my %to_use = ();

  for ($i=0; $i<=$#images; $i++)
  {
    ($time, $ant, $type, $res, $ext) = split(/\./, $images[$i]);
    if (!exists($to_use{$ant}))
    {
      $to_use{$ant} = ();
    }
    $to_use{$ant}{$type.".".$res} = $images[$i];
  }

  $now = time;

  for ($i=0; $i<=$#images; $i++)
  {
    ($time, $ant, $type, $res, $ext) = split(/\./, $images[$i]);
    $time_unix = Dada::getUnixTimeLocal($time);

    # if this is not the most recent matching type + res
    if ($to_use{$ant}{$type.".".$res} ne $images[$i])
    {
      # only delete if > 30 seconds old
      if (($time_unix + 30) < $now)
      {
        Dada::logMsg(3, $dl, "removeOldPngs: deleteing ".$dir."/".$images[$i].", duplicate, age=".($now-$time_unix));
        unlink $dir."/".$images[$i];
      }
      else
      {
        Dada::logMsg(3, $dl, "removeOldPngs: keeping ".$dir."/".$images[$i].", duplicate, age=".($now-$time_unix));
      }
    }
    else
    {
      Dada::logMsg(3, $dl, "removeOldPngs: keeping ".$dir."/".$images[$i].", newest, age=".($now-$time_unix));
    }
  }
}


###############################################################################
#
# Handle quit requests asynchronously
#
sub controlThread($$) {

  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, $dl ,"controlThread: exiting");

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  if ($quit_daemon)
  {
    print STDERR $daemon_name." : Exiting\n";
    exit 1;
  }
  $quit_daemon = 1;
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($)
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
} 


# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  # Ensure more than one copy of this daemon is not running
  my ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}


END { }

1;  # return value from file
