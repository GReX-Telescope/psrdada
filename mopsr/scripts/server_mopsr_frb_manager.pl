#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use File::Basename;
use Time::Local;
use Time::HiRes qw(usleep);
use XML::Simple qw(:strict);
use MIME::Lite;
use Math::Trig ':pi';
use Dada;
use Mopsr;

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our %cfg;
our %bp_cfg;
our %bp_ct;
our $quit_daemon : shared;
our $warn;
our $error;
our $coarse_nchan;

#
# Initialize global variables
#
%cfg = Mopsr::getConfig();
%bp_cfg = Mopsr::getConfig("bp");
%bp_ct = Mopsr::getCornerturnConfig("bp");
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
sub processCandidates($);
sub processArchive($$$);
sub makePlotsFromArchives($$$$$$);
sub removeOldPngs($);
sub getObsInfo($);

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
  my ($control_thread, $coincidencer_thread, $dump_thread);
  my @observations = ();
  my ($i, $obs_mode, $obs_config);
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

  my $testing = 0;
  if ($testing)
  {
    my %c = ();
    $c{"start_utc"}   = "2017-06-27-21:41:02.6";
    $c{"utc"}         = "2017-06-27-21:41:02.7";
    $c{"end_utc"}     = "2017-06-27-21:41:02.8";
    $c{"dm"}          = 175;
    $c{"width"}       = 0.00065536;
    $c{"snr"}         = 47;
    $c{"sample"}      = 99377;
    $c{"filter"}      = 5;
    $c{"probability"} = 0.1235;

    $c{"utc_start"}   = "2017-08-27-16:05:48";
    $c{"beam"}        = 257;

    %c = supplementCand(\%c);

    # test for extra galacitcity
    $c{"extra_galactic"} = 0;
    if (($c{"galactic_dm"} ne "None") && ($c{"dm"} ne "None") && ($c{"dm"} > $c{"galactic_dm"}))
    {
      $c{"extra_galactic"} = 1;
    }

    my $to_email = "ajameson\@swin.edu.au";
    my $cc_email = "";
    my $did_dump = "false";
    Dada::logMsg(1, $dl, "main: threads->new(\&generateEmail");
    my $email_thread = threads->new(\&generateEmail, $to_email, $cc_email, $did_dump, \%c);
    Dada::logMsg(1, $dl, "main: email_thread->join()");
    $email_thread->join();
    Dada::logMsg(1, $dl, "main: email_thread joined");
    $quit_daemon = 1;
    $coincidencer_thread = 0;
    $dump_thread = 0;
  }
  else
  {
    $coincidencer_thread = threads->new(\&coincidencerThread);
    $dump_thread = threads->new(\&dumperThread);
  }

  while (!$quit_daemon)
  {
    # TODO check that directories are correctly sorted by UTC_START time
    Dada::logMsg(2, $dl, "main: looking for obs.processing in ".$obs_results_dir);

    # Only get observations that are marked as procesing
    $cmd = "find ".$obs_results_dir." -mindepth 2 -maxdepth 2 -name 'obs.processing' ".
           "-printf '\%h\\n' | awk -F/ '{print \$NF}' | sort -n";

    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "main: ".$result." ".$response);

    if ($result eq "ok") 
    {
      @observations = split(/\n/,$response);

      # For process all valid observations
      for ($i=0; (($i<=$#observations) && (!$quit_daemon)); $i++)
      {
        $o = $observations[$i];

        $obs_mode = getObsInfo($o);

        Dada::logMsg(2, $dl, "main: assessing observation=".$o." obs_mode=".$obs_mode);

        if ($obs_mode eq "FB")
        {
          Dada::logMsg(2, $dl, "main: processCandidates(".$o.")");
          ($result, $response) = processCandidates($o);
          Dada::logMsg(2, $dl, "main: processCandidates ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsgWarn($warn, "processCandidates(".$o.") failed: ".$response);       
          }
        }
      }
    }

    # if no obs.processing, check again in 5 seconds
    if ($#observations == -1) {
      $counter = 5;
    } else {
      $counter = 1;
    }
   
    while ($counter && !$quit_daemon) {
      sleep(1);
      $counter--;
    }
  }

  Dada::logMsg(2, $dl, "main: joining controlThread");
  $control_thread->join();

  if ($coincidencer_thread)
  {
    Dada::logMsg(2, $dl, "main: joining coincidencerThread");
    $coincidencer_thread->join();
  }

  if ($dump_thread)
  {
    Dada::logMsg(2, $dl, "main: joining dumpThread");
    $dump_thread->join();
  }

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

                                                                                
  return 0;
}

sub getObsInfo($)
{
  (my $obs) = @_;

  my $obs_info = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/obs.info";
  my $mode = "UNKNOWN";

  if (-f $obs_info)
  {
    Dada::logMsg(2, $dl, "main: obs_info=".$obs_info);
    my $cmd = "grep FB_ENABLED ".$obs_info." | awk '{print \$2}'";
    my ($result, $response) = Dada::mySystem($cmd);
    if (($result eq "ok") && ($response eq "true"))
    {
      $mode = "FB";
    }
    else
    {
      Dada::logMsg(2, $dl, "main: obs_info=".$obs_info);
      my $cmd = "grep MB_ENABLED ".$obs_info." | awk '{print \$2}'";
      my ($result, $response) = Dada::mySystem($cmd);
      if (($result eq "ok") && ($response eq "true"))
      {
        $mode = "FB";
      }
    }
  }

  return ($mode);
}

sub processCandidates($)
{
  my ($utc_start) = @_;

  my ($cmd, $result, $response);
  my ($file, @files, $timestamp, $suffix);
  my ($line, @lines);

  # for finding coincidenced candidate files
  $cmd = "find ".$utc_start." -ignore_readdir_race -mindepth 1 -maxdepth 1 ".
         "-type f -name '2???-??-??-??:??:??_all.cand' -printf '%f\n' | sort -n";

  Dada::logMsg(2, $dl, "processCandidates: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "processCandidates: ".$result." ".$response);
  if ($result ne "ok") 
  {
    return ("fail", "find command failed");
  }

  @files = split(/\n/, $response);

  my $n_processed = 0;

  my $cands_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start."/FB";

  if (!(-d $cands_dir))
  {
    ($result, $response) = Dada::mkdirRecursive ($cands_dir, 0755);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processCandidates: mkdirRecursive(".$cands_dir.") failed: ".$response);
      return ("fail", "mkdirRecursive failed");
    }
  }

  my $snr_cut = 9;
  foreach $file (@files)
  {
    ($timestamp, $suffix) = split(/_/, $file, 2);

    open FH,"<".$utc_start."/".$timestamp."_all.cand" or return -1;
    @lines = <FH>;
    close FH;

    my $num_candidates = $#lines + 1;

    Dada::logMsg (2, $dl, "processCandidates: utc_start=".$utc_start." timestamp=".$timestamp." num_candidates=".$num_candidates);

    # this is where a transient search detection program would go
    if ($num_candidates > 0)
    {
      # transmit the candidates to the FRB detection server
      my $xml =  "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml .= "<frb_detector_message>";
      $xml .= "<cmd>candidates</cmd>";
      $xml .= "<utc_start>".$utc_start."</utc_start>";
      my $event_time = "";
      my $ncands = 0;
      my $ncands_over9 = 0;
      foreach $line ( @lines)
      {
        chomp $line;
        my ($snr, $samp_idx, $sample_time, $filter, $dm_trial, $dm, $members, $begin, $end, $nbeams, $primary_beam, $max_snr, $beam) =  split(/\s+/, $line);

        if ($beam ne "1")
        {
          $ncands += 1;
          if (int($snr) >= $snr_cut)
          {
            $xml .= "<cand>".$line."</cand>";
            $ncands_over9 += 1;
          }
        }
        if ($event_time eq "" || $event_time eq "nan")
        {
          $event_time = $sample_time;
        }
      } 
      $xml .= "</frb_detector_message>";
      
      # get the unix time for the UTC_START
      my $utc_start_unix = Dada::getUnixTimeUTC($utc_start);

      # get the current unix time :)
      my $curr_time_unix = time;

      my $delta_start = ($curr_time_unix - $utc_start_unix);

      Dada::logMsg(2, $dl, "processCandidates: ncands=".$ncands." delta_start=".$delta_start." event_time=".$event_time." offset ".(int($delta_start) - int($event_time))."s");
      Dada::logMsg(2, $dl, "processCandidates: ncands>=".$snr_cut."=".$ncands_over9." offset=".(int($delta_start) - int($event_time))."s");

      # open a socket to the server_mopsr_frb_detector
      my $frb_host = $cfg{"SERVER_HOST"};
      my $frb_port = $cfg{"FRB_DETECTOR_BASEPORT"};

      if ($ncands_over9 > 0)
      {
        Dada::logMsg(1, $dl, "processCandidates: connecting to ".$frb_host.":".$frb_port);
        my $handle = Dada::connectToMachine($frb_host, $frb_port, 1);
        if ($handle)
        {
          Dada::logMsg(2, $dl, "processCandidates: sending ".$ncands_over9." cands");
          print $handle $xml."\r\n"; 
          Dada::logMsg(3, $dl, "processCandidates: getting reply");
          my $response = Dada::getLine($handle);
          Dada::logMsg(3, $dl, "processCandidates: closing socket");
          close ($handle);

          eval {
            $xml = XMLin ($response, ForceArray => 0, KeyAttr => 0, SuppressEmpty => 1, NoAttr => 1);
          };

          # If the XML parsing failed 
          if ($@)
          {
            Dada::logMsgWarn($warn, "failed to parse XML reply from FRB detector");
          }
          else
          {
            my $reply = "unknown";
            # check if reply is ok | fail
            if (! eval { exists $xml->{'reply'} } )
            {
              Dada::logMsgWarn($warn, "Malformed XML reply from FRB detector");
            }
            else
            {
              $reply =  $xml->{'reply'};
            }
            Dada::logMsg(1, $dl, "processCandidates: FRB detector replied: ".$reply);
          }
        }
      }
    }

    Dada::logMsg(2, $dl, "processCandidates: appending ".$utc_start."/".$timestamp."_all.cand to all_candidates.dat");

    # add this timestamp to the accumulated total
    if ( -f $cands_dir."/all_candidates.dat")
    {
      $cmd = "cat ".$utc_start."/".$timestamp."_all.cand >> ".$cands_dir."/all_candidates.dat";
    }
    else
    {
      $cmd = "cat ".$utc_start."/".$timestamp."_all.cand > ".$cands_dir."/all_candidates.dat";
    }
    Dada::logMsg(2, $dl, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processCandidates: ".$cmd." failed");
      return ("fail", "cat command failed");
    }

    $n_processed++;

    $cmd = "rm -f ".$utc_start."/".$timestamp."*.cand*";
    Dada::logMsg(2, $dl, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "processCandidates: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($warn, "processCandidates: ".$cmd." failed");
      return ("fail", "cat command failed");
    }
  }

  Dada::logMsg(2, $dl, "processCandidates: processed all .cand files for ".$utc_start);

  if (( -f $cands_dir."/all_candidates.dat") && ($n_processed > 0))
  {

    # determine end of observation
    $cmd = "tail -n 1000 ".$cands_dir."/all_candidates.dat | sort -k 3 | tail -n 1 | awk '{print \$3}'";
    Dada::logMsg(2, $dl, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processCandidates: ".$result." ".$response);
    if (($result ne "ok") && ($response ne ""))
    {
      Dada::logMsgWarn($warn, "processCandidates: ".$cmd." failed");
      return ("fail", "could not determine observation end");
    }

    my $obs_start = 0;
    my $obs_end = $response;

    my $img_idx = 0;

    if (($obs_end - $obs_start) > 1800)
    {
      $img_idx = int($obs_end / 1800);
      $obs_start = $img_idx * 1800;
    }

    $obs_end = $obs_start + 1800;

    # create the plot
    $cmd = "mopsr_plot_cands -f ".$cands_dir."/all_candidates.dat ".
           "-time1 ".$obs_start." -time2 ".$obs_end." -nbeams ".$bp_ct{"NBEAM"}.
           " -max_beam ".$bp_ct{"NBEAM"}." -snr_cut 7 -scale 6 -nbeams_cut 10 ".
           "-dev ".$utc_start."/".$timestamp.".FB.".sprintf("%02d", $img_idx).
           ".850x680.png/png";
    Dada::logMsg(2, $dl, "processCandidates: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "processCandidates: ".$result." ".$response);
    if (($result ne "ok") && (0))
    {
      Dada::logMsgWarn($warn, "processCandidates: ".$cmd." failed");
      return ("fail", "plot command failed");
    }
    Dada::logMsg(1, $dl, "processCandidates: updated candidate plots for ".$utc_start);
  }

  Dada::logMsg(2, $dl, "processCandidates: completed");
  return ("ok", "");
}

###############################################################################
#
# Runs the coincidencer which persists through multiple observations
#
sub coincidencerThread()
{
  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"COINCIDENCER_PORT"};
  my $nbeam = $bp_ct{"NBEAM"};
  my $log = $cfg{"SERVER_LOG_DIR"}."/mopsr_coincidencer.log";
  #my $cmd = "coincidencer -a ".$host." -p ".$port." -n ".$nbeam." | awk '{print strftime(\"%Y-%m-%d-%H:%M:%S\")\" \"$0 }' >> ".$log;
  my $cmd = "coincidencer -a ".$host." -p ".$port." -n ".$nbeam." -v >> ".$log;

  Dada::logMsg(1, $dl, "coincidencerThread: ".$cmd);
  my ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "coincidencerThread: ".$result." ".$response);
  if ($result ne "ok")
  {
    if (!$quit_daemon)
    {
      Dada::logMsgWarn($warn, "coincidencerThread: coincidencer failed: ".$response);
    }
    return ("fail");
  }
  return ("ok");
}

sub dumperThread ()
{
  my ($result, $response);

  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"FRB_DETECTOR_DUMPPORT"};
  my $min_dump_separation = 10;   # seconds
  my $min_email_separation = 10;  # seconds

  my $socket = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    ReuseAddr => 1
  );
  die "Could not create socket: $!\n" unless $socket;

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($socket);          # add the main socket to the set

  my ($handle, $rh, $peeraddr, $hostinfo, $last_dump_unix, $last_email_unix);

  Dada::logMsg(1, $dl, "dumperThread: listening for connections on ".$host.":".$port);

  # time of previous dump command 
  $last_dump_unix = 0;
  $last_email_unix = 0;

  # Main Loop,  We loop forever unless asked to quit
  while (!$quit_daemon)
  {
    # Get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
    Dada::logMsg(3, $dl, "select on read_set returned");

    my $obs_results_dir  = $cfg{"SERVER_RESULTS_DIR"};
    my $obs_archives_dir = $cfg{"SERVER_ARCHIVE_DIR"};

    foreach $rh (@$readable_handles)
    {
      if ($rh == $socket)
      {
        # Wait for a connection from the server on the specified port
        $handle = $socket->accept() or die "accept $!\n";

        # Ensure commands are immediately sent/received
        $handle->autoflush(1);

        # Add this read handle to the set
        $read_set->add($handle);

        # Get information about the connecting machine
        $peeraddr = $handle->peeraddr;
        $hostinfo = gethostbyaddr($peeraddr);
        my $host = "None";
        if (defined $hostinfo)
        { 
          $host = $hostinfo->name;
        }
        Dada::logMsg(2, $dl, "Accepting connection from ".$host);
      }
      else
      {
        my $command = <$rh>;

        # If we have lost the connection...
        if (! defined $command)
        {
          my $host = "localhost";
          if (defined $hostinfo)
          { 
            $host = $hostinfo->name;
          }
          Dada::logMsg(2, $dl, "Lost connection from ".$host);

          $read_set->remove($rh);
          close($rh);
          #$handle->close();
          #$handle = 0;
        }
        else
        {
          Dada::logMsg(2, $dl, "Command [".$command."]");

          my $xml = "";
          eval {
            $xml = XMLin ($command, ForceArray => 0, KeyAttr => 0, SuppressEmpty => 1, NoAttr => 1);
          };

          # If the XML parsing failed 
          if ($@)
          {
            Dada::logMsgWarn($warn, "failed to parse XML dump command from FRB detector");
          }
          else
          {
            # check if cmd exists
            if (eval { exists $xml->{'cmd'} })
            {
              if ($xml->{'cmd'} eq "dump")
              {
                my %c = ();
                $c{"start_utc"}   = $xml->{'cand_start_utc'};
                $c{"utc"}         = $xml->{'cand_utc'};
                $c{"end_utc"}     = $xml->{'cand_end_utc'};
                $c{"dm"}          = $xml->{'cand_dm'};
                $c{"width"}       = $xml->{'cand_width'};
                $c{"snr"}         = $xml->{'cand_snr'};
                $c{"sample"}      = $xml->{'cand_sample'};
                $c{"filter"}      = $xml->{'cand_filter'};
                $c{"probability"} = $xml->{'probability'};

                $c{"utc_start"} = $xml->{'utc_start'};
                $c{"beam"}      = $xml->{'beam_number'};

                # the unix time for this dump
                my $event_unix = Dada::getUnixTimeUTC ($c{"utc"});

                # supplement additional information for the event
                %c = supplementCand(\%c);

                my $obs_info_file = $obs_results_dir."/".$c{"utc_start"}."/obs.info";
                my $obs_header_file = $obs_results_dir."/".$c{"utc_start"}."/FB/obs.header.BP00";
                # Check if Furby
                my $cmd = "grep INJECTED_FURBYS ".$obs_info_file." | awk '{print \$2}'";
                my ($result, $response) = Dada::mySystem($cmd);

                my (@furby_ids, @furby_beams, @furby_tstamps);
                my $is_furby = 0;
                if (($result eq "ok") && ($response) )
                {
                  my $nfurbies = int($response);
                  my %obs_header = Dada::readCFGFileIntoHash ($obs_header_file, 0);

                  # get FURBY_IDS
                  if (exists $obs_header{"FURBY_IDS"})
                  {
                    @furby_ids = split(/,/, $obs_header{"FURBY_IDS"});
                  }
                  else
                  {
                    Dada::logMsg(0, $dl, "main: FURBY_IDS doesn't exist in file: ".$obs_header_file);
                  }

                  # get FURBY_BEAMS
                  if (exists $obs_header{"FURBY_BEAMS"})
                  {
                    @furby_beams = split(/,/, $obs_header{"FURBY_BEAMS"});
                  }
                  else
                  {
                    Dada::logMsg(0, $dl, "main: FURBY_BEAMS doesn't exist in file: ".$obs_header_file);
                  }

                  # get furby_tstamps
                  if (exists $obs_header{"FURBY_TSTAMPS"})
                  {
                    @furby_tstamps = split(/,/, $obs_header{"FURBY_TSTAMPS"});
                  }
                  else
                  {
                    Dada::logMsg(0, $dl, "main: FURBY_TSTAMPS doesn't exist in file: ".$obs_header_file);
                  }

                  # furby tsamp is relative to centre of the molonglo band. Heimdall's 
                  # tsamp is relative to the top of the band:
                  my $centre_tstamp = int((8.3*0.015*$c{"dm"}*(0.8435)**(-3)*1000)/327.68) + $c{"sample"};

                  # Check if any of the furbys coincides with FRB trigger

                  my $i;
                  for ($i=0; $i<$nfurbies; $i++)
                  {
                    my $furby_id = $furby_ids[$i];
                    my $furby_beam = $furby_beams[$i];
                    my $furby_tstamp = $furby_tstamps[$i]/0.00032768;
                    if (($furby_beam == $c{"beam"}) && 
                      ($furby_tstamp <= ($centre_tstamp + 1500)) && ($furby_tstamp >= ($centre_tstamp - 1500)) && 
                      (!($is_furby))) 
                    {
                      # Furby found
                      Dada::logMsg(0, $dl, "main: Found a furby - id: ".$furby_id.", beam: ".$furby_beam.", tstamp: ".$furby_tstamp);

                      $is_furby = 1;
                      # log it
                      my $furby_log_file = $obs_archives_dir."/".$c{"utc_start"}."/Furbys/furbys.log";

                      # Write header if file doesn't exist
                      if (!( -f $furby_log_file))
                      {
                        open FH, ">$furby_log_file" or DADA::logMsg(0, $dl, 
                          "main: couldn't open ".$furby_log_file." for writing");
                        print FH "#furby_id tstamp beam boxcar dm snr proba\n";
                      }
                      else
                      {
                        open FH, ">>$furby_log_file" or DADA::logMsg(0, $dl, 
                          "main: couldn't open ".$furby_log_file." for appending");
                      }
                      print FH $furby_id." ".$c{"sample"}." ".$c{"beam"}." ".$c{"filter"}.
                        " ".$c{"dm"}." ".$c{"snr"}." ".$c{"probability"}."\n";
                      close FH;
                    }
                  }
                                    

                }
                else
                {
                  Dada:logMsg(0, $dl, "Could not read INJECTED_FURBYS from ".$obs_info_file);
                }

 
                # test for extra galacitcity
                # TODO Change this value back to 0 after geting a J1644 dump
                $c{"extra_galactic"} = 0;
                if (($c{"galactic_dm"} ne "None") && ($c{"dm"} ne "None") && ($c{"dm"} > $c{"galactic_dm"}))
                {
                  $c{"extra_galactic"} = 1; 
                }

                my $did_dump = "false";
                if ($c{"extra_galactic"} && ($event_unix > $last_dump_unix + $min_dump_separation) && !($is_furby))
                {
                  $last_dump_unix = $event_unix;

                  my $event_message = $c{"start_utc"}." ".$c{"end_utc"}." ".$c{"dm"}." ".
                                      $c{"snr"}." ".$c{"width"}." ".$c{"beam"};

                  Dada::logMsg(0, $dl, "FRB: message=".$event_message);
                  my $dumping_enabled = 1;
                  if ($dumping_enabled)
                  {
                    $did_dump = "true";
                    my $i;
                    for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
                    {
                      my $host = $cfg{"PWC_".$i};
                      my $port = int($cfg{"CLIENT_AQ_EVENT_BASEPORT"}) + int($i);
                      if ($cfg{"PWC_STATE_".$i} ne "inactive")
                      {
                        Dada::logMsg(1, $dl, "main: opening connection for FRB dump to ".$host.":".$port);
                        my $sock = Dada::connectToMachine($host, $port, 1);
                        if ($sock)
                        {
                          Dada::logMsg(1, $dl, "main: connection to ".$host.":".$port." established");
                          print $sock "N_EVENTS 1\n";
                          print $sock $c{"utc_start"}."\n";
                          print $sock $event_message."\n";
                          close ($sock);
                          $sock = 0;
                        }
                        else
                        {
                          Dada::logMsg(0, $dl, "main: connection to ".$host.":".$port." failed");
                        }
                      }
                      else
                      {
                         Dada::logMsg(2, $dl, "main: skipping ".$host.":".$port." since marked inactive PWC");
                      }
                    }
                  }
                }

                if (!($is_furby))
                {
                  if (($event_unix > $last_email_unix + $min_email_separation))
                  {
                    Dada::logMsg(0, $dl, "FRB: UTC_START=".$c{"utc_start"}." BEAM=".$c{"beam"});
                    Dada::logMsg(0, $dl, "FRB: DUMP ".$c{"start_utc"}." to ".$c{"end_utc"});
                    Dada::logMsg(0, $dl, "FRB: EVENT DM=".$c{"dm"}." WIDTH=".($c{"width"}*1000)."ms ".
                                         "SNR=".$c{"snr"}." PROB=".$c{"probability"});
                    $last_email_unix = $event_unix;
                    my $to_email = "ajameson\@swin.edu.au";
                    my $cc_email = "adeller\@swin.edu.au,".
                                   "bateman.tim\@gmail.com,".
                                   "Timothy.Bateman\@sydney.edu.au,".
                                   "cflynn\@swin.edu.au,".
                                   "fjankowsk\@gmail.com,".
                                   "kaplant\@ucsc.edu,".
                                   "adityapartha3112\@gmail.com,".
                                   "mbailes\@swin.edu.au,".
                                   "stefanoslowski\@swin.edu.au,".
                                   "manishacaleb\@gmail.com,".
                                   "shivanibhandari58\@gmail.com,".
                                   "v.vikram.ravi\@gmail.com,".
                                   "cday\@swin.edu.au,".
                                   "vivekgupta\@swin.edu.au,".
                                   "vivekvenkris\@gmail.com,".
                                   "wfarah\@swin.edu.au";
                    my $email_thread = threads->new(\&generateEmail, $to_email, $cc_email, $did_dump, \%c);
                    $email_thread->detach();
                  }
                  else
                  {
                    Dada::logMsg(0, $dl, "FRB: IGNORE EVENT DM=".$c{"dm"}." WIDTH=".($c{"width"}*1000)."ms ".
                                         "SNR=".$c{"snr"}." BEAM=".$c{"beam"}." PROB=".$c{"probability"});
                  }
                }
              }
              else
              {
                Dada::logMsg(0, $dl, "main: unexpected cmd [".$xml->{"cmd"}."]");
              }  
            }
            else
            {
              Dada::logMsg(0, $dl, "main: no cmd tag in XML message");
            }
          }
        }
      }
    }
  }
}

sub supplementCand(\%)
{
  my ($cand_ref) = @_;
  my %h = %$cand_ref;

  my ($cmd, $result, $response);

  my ($yyyy, $mm, $dd, $rest) = split("-",$h{"utc"}, 4);
  my ($hours, $mins, $secs) = split(":", $rest);

  $h{"frb_name"}  = "FRB ".$yyyy."-".$mm."-".$dd."-".$hours.":".$mins.":".$secs;
  $h{"cand_name"} = "CAND ".$yyyy."-".$mm."-".$dd."-".$hours.":".$mins.":".$secs;
  $h{"frb_prefix"}  = "FRB".$yyyy.$mm.$dd;
  $h{"cand_prefix"} = "CAN".$yyyy.$mm.$dd;

  $h{"delta_md_deg"} = 0;
  $h{"nbeam"} = "None";
  $h{"beam_spacing"} = "None";

  my $obs_header_file = $cfg{"SERVER_RESULTS_DIR"}."/".$h{"utc_start"}."/FB/obs.header";
  if (-f $obs_header_file)
  {
    my %obs_header = Dada::readCFGFileIntoHash ($obs_header_file, 0);

    $h{"nbeam"} = $obs_header{"NBEAM"};
    Dada::logMsg(1, $dl, "supplementCand: obs_header{'NBEAM'}=".$h{"nbeam"});

    # determine the offset from the the boresight
    $h{"beam_spacing"} = $obs_header{"FB_BEAM_SPACING"};

    my $centre_fanbeam = ($h{"nbeam"} / 2) + 1;
    my $delta_beam = $centre_fanbeam - $h{"beam"};
    # offset from HA=0 to MD=0
    my $md_offset = 0.1976;
    $h{"delta_md_deg"} = ($h{"beam_spacing"} * $delta_beam) - $md_offset;
  }
  else
  {
    Dada::logMsg(1, $dl, "supplementCand: obs_header_file [".$obs_header_file."] did not exist");
  }

  # extract some additional information from obs.info
  my $obs_info_file = $cfg{"SERVER_RESULTS_DIR"}."/".$h{"utc_start"}."/obs.info";
  my %obs_info = Dada::readCFGFileIntoHash ($obs_info_file, 0);

  $h{"PID"} = $obs_info{"PID"};
  $h{"SOURCE"} = $obs_info{"SOURCE"};
  $h{"DELAY_TRACKING"} = $obs_info{"DELAY_TRACKING"};

  # determine the current LST
  $h{"lst"} = "Unknown";
  my ($utc, $junk) = split(/\./, $h{"utc"},2);
  $cmd = "mopsr_getlst ".$utc;
  Dada::logMsg(2, $dl, "supplementCand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "supplementCand: ".$result." ".$response);
  if ($result eq "ok")
  {
    my ($junk1, $junk2);
    Dada::logMsg(2, $dl, "supplementCand: ".$response);
    ($junk1, $h{"lst"}, $junk2) = split(/ /, $response);
    Dada::logMsg(3, $dl, "supplementCand: lst=".$h{"lst"});
  }

  $h{"ra"} = $obs_info{"RA"};
  $h{"dec"} = $obs_info{"DEC"};

  # convert the RA, DEC & LST to degrees
  my ($ra_deg, $dec_deg, $lst_deg);
  ($result, $ra_deg) = Dada::convertHHMMSSToDegrees($h{"ra"});
  ($result, $dec_deg) = Dada::convertDDMMSSToDegrees($h{"dec"});
  ($result, $lst_deg) = Dada::convertHHMMSSToDegrees($h{"lst"});

  Dada::logMsg(1, $dl, "supplementCand: RA ".$h{"ra"}." -> ".$ra_deg);
  Dada::logMsg(1, $dl, "supplementCand: DEC ".$h{"dec"}." -> ".$dec_deg);
  Dada::logMsg(1, $dl, "supplementCand: LST  ".$h{"lst"}." -> ".$lst_deg);

  # determine what the boresight was actually looking at
  my $boresight_ra_deg = $lst_deg;
  if ($obs_info{"DELAY_TRACKING"} eq "true")
  {
    $boresight_ra_deg = $ra_deg;
  }
  $h{"boresight_ra"} = Dada::convertDegreesToRA ($boresight_ra_deg);

  # now we can convert from delta_degrees (MD) to delta_degrees (RA)
  my $deg_to_radians = Math::Trig::pi / 180.0;
  $h{"delta_ra_deg"} = $h{"delta_md_deg"} / cos($dec_deg * $deg_to_radians);

  # add the offset (in RA)
  my $cand_ra_deg = $boresight_ra_deg + $h{"delta_ra_deg"};
  Dada::logMsg(1, $dl, "supplementCand: cand_ra=".$cand_ra_deg." == ".$boresight_ra_deg." + ".$h{"delta_ra_deg"});

  # now convert ra_deg back to HH:MM:SS
  $h{"cand_ra"} = Dada::convertDegreesToRA($cand_ra_deg);

  # convert (ra,dec) [degrees] to (gl,gb)
  $h{"gl"} = "Unknown";
  $h{"gb"} = "Unknown";
  $h{"galactic_dm"} = "Unknown";

  $cmd = "eq2gal.py ".$cand_ra_deg." ".$dec_deg;
  Dada::logMsg(3, $dl, "supplementCand: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "supplementCand: ".$result." ".$response);
  if ($result eq "ok")
  {
    ($h{"gl"}, $h{"gb"}) = split(/ /, $response, 2);

    # determine the DM for this GL and GB
    $cmd = "cd \$HOME/opt/NE2001/runtime; ./NE2001 ".$h{"gl"}." ".$h{"gb"}." 100 -1 | grep ModelDM | awk '{print \$1}'";
    Dada::logMsg(2, $dl, "detectFRBs: ".$cmd);
    ($result, $h{"galactic_dm"}) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "detectFRBs: ".$result." ".$h{"galactic_dm"});
  }
  else
  {
    Dada::logMsg(1, $dl, "supplementCand:couldn't calculate Gl, Gb from RA, DEC: ".$response);
  }
  return %h;
}


sub generateEmail ($$$\%)
{
  my ($to_email, $cc_email, $dumped, $cand_ref) = @_;
  my %h = %$cand_ref;
  my ($cmd, $result, $response);
  my ($bp_tag, $subject, $name);

  # generate HTML part of email
  my $html = "<body>";

  # print the candidate update first
  if ($h{"extra_galactic"} == 1)
  {
    $html .= "<h3>FRB Candidate</h3>\n";
    $subject = $h{"frb_prefix"};
    $name = $h{"frb_name"};
  }
  else
  {
    $html .= "<h3>Transient Candidate</h3>\n";
    $subject = $h{"cand_prefix"};
    $name = $h{"cand_name"};
    $to_email = "wfarah\@swin.edu.au";
    $cc_email = "cflynn\@swin.edu.au,bateman.tim\@gmail.com,stefanoslowski\@swin.edu.au,vivekvenkris\@gmail.com,vivekgupta\@swin.edu.au";
  }

  # try and determine the RA/DEC of the boresight beam
  my $msg = MIME::Lite->new(
   From    => 'UTMOST Transient Detector <noreply@utmost.usyd.edu.au>',
    To      => $to_email,
    Cc      => $cc_email,
    Subject => $subject,
    Type    => 'multipart/mixed',
    Data    => "Here's the PNG file you wanted"
  );

  $html .= "<table cellpadding=2 cellspacing=2>\n";
  $html .= "<tr><th style='text-align: left;'>SNR</th><td>".sprintf("%5.2f",$h{"snr"})."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>DM</th><td>".sprintf("%5.2f",$h{"dm"})."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Width</th><td>".sprintf("%5.2f",$h{"width"}*1000)." ms</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Probability</th><td>".sprintf("%5.3f",$h{"probability"})."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Name</th><td>".$name."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>UTC</th><td>".$h{"utc"}."</td></tr>\n";
  $html .= "</table>\n";

  $html .= "<hr/>\n";

  $html .= "<h3>Observation</h3>\n";
  $html .= "<table cellpadding=2 cellspacing=2>\n";
  $html .= "<tr><th style='text-align: left;'>UTC START</th><td>".$h{"utc_start"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Beam</th><td>".$h{"beam"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>MD Offset</th><td>".$h{"delta_md_deg"}." degrees</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>RA Offset</th><td>".$h{"delta_ra_deg"}." degrees</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Total Beams</th><td>".$h{"nbeam"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>PID</th><td>".$h{"PID"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Voltage Dump</th><td>".$dumped."</td></tr>\n";
  if ($dumped eq "true")
  {
    $html .= "<tr><th style='text-align: left;'>Dump Start</th><td>".$h{"start_utc"}."</td></tr>\n";
    $html .= "<tr><th style='text-align: left;'>Dump End</th><td>".$h{"end_utc"}."</td></tr>\n";
  }
  $html .= "</table>\n";

  $html .= "<hr/>\n";

  $html .= "<h3>Boresight Properties</h3>\n";
  $html .= "<table cellpadding=2 cellspacing=2>\n";
  $html .= "<tr><th style='text-align: left;'>SOURCE</th><td>".$h{"SOURCE"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>LST</th><td>".$h{"lst"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>DELAY TRACKING</th><td>".$h{"DELAY_TRACKING"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>BORESIGHT RA</th><td>".$h{"boresight_ra"}."</td></tr>\n";
  $html .= "</table>\n";

  $html .= "<hr/>\n";

  $html .= "<h3>Candidate Position Properties</h3>\n";

  $html .= "<table cellpadding=2 cellspacing=2>\n";
  $html .= "<tr><th style='text-align: left;'>RA</th><td>".$h{"cand_ra"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>DEC</th><td>".$h{"dec"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Gl</th><td>".$h{"gl"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>Gb</th><td>".$h{"gb"}."</td></tr>\n";
  $html .= "<tr><th style='text-align: left;'>NE2001 DM</th><td>".$h{"galactic_dm"}."</td></tr>\n";

  $html .= "</table>\n";

  $html .= "<hr/>\n";

  my $centre_beam = ($bp_ct{"NBEAM"} / 2) + 1;

  # generate a freqplus plot for the incoherrent beam and the 2 surrounding beams
  my @beams = (1, $centre_beam, ($h{"beam"}-1), ($h{"beam"}), ($h{"beam"}+1));
  my @beam_names = ();
  my ($beam, $beam_name, $beam_idx, $host, $i);
  my ($fil_file, $plot_cmd, $local_img);
  foreach $beam (@beams)
  {
    $beam_name = sprintf("BEAM_%03d", $beam);
    push @beam_names, $beam_name;
  }

  $html .= "</body>\n";
  $html .= "</html>\n";

  ### Add the html message part:
  $msg->attach (
    Type     => 'text/html',
    Data     => $html,
  );

  my $j;
  for ($j=0; $j<=$#beams; $j++)
  {
    $beam = $beams[$j];
    $beam_name = $beam_names[$j];
    Dada::logMsg(1, $dl, "generateEmail: plotting beam idx=".$beam." name=".$beam_name);

    # find the host that contains the beam, the CT uses 0 indexing, so -1
    $beam_idx = $beam - 1;
    $bp_tag = "";
    $host = "";
    for ($i=0; $i<$bp_ct{"NRECV"}; $i++)
    {
      if (($host eq "") && ($bp_ct{"BEAM_FIRST_RECV_".$i} <= $beam_idx) && ($beam_idx <= $bp_ct{"BEAM_LAST_RECV_".$i}))
      {
        $host = $bp_ct{"RECV_".$i};
        $bp_tag = sprintf ("BP%02d", $i);
      }
    }

    if ($host ne "")
    {
      $fil_file = $cfg{"CLIENT_RECORDING_DIR"}."/".$bp_tag."/".$h{"utc_start"}."/FB/".$beam_name."/".$h{"utc_start"}.".fil";
  
      $plot_cmd = "trans_freqplus_plot.py -beam ".$beam_name ." ".$fil_file." ".$h{"sample"}." ".$h{"dm"}." ".$h{"filter"}." ".$h{"snr"};
      $local_img = $h{"frb_prefix"}."_".$beam_name."_".$h{"sample"}.".png";

      # create a freq_plus file
      $cmd = "ssh mpsr@".$host." '".$plot_cmd."' > /tmp/".$local_img;

      Dada::logMsg(2, $dl, "generateEmail: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "generateEmail: ".$result." ".$response);

      # get the first 5 chars / bytes of the file 
      $cmd = "head -c 5 /tmp/".$local_img;
      Dada::logMsg(2, $dl, "generateEmail: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "generateEmail: ".$result." ".$response);

      # if we did manage to find the filterbank file
      if (($result eq "ok") && ($response ne "ERROR"))
      {
        $msg->attach(
          Type        => 'image/png',
          Id          => $h{"frb_prefix"}."_".$beam_name.".png",
          Path        => '/tmp/'.$local_img,
          Disposition => 'attachment'
        );
      }
    }
  }

  $local_img = $h{"frb_prefix"}."_localisation_".$h{"sample"}.".png";
  $cmd = "/home/observer/chris/fb2sky/frb_pos_jt.py -s ".$h{"utc_start"}." -e ".$h{"utc"}." -f ".$h{"beam"}." -n /tmp/".$local_img." -N ".$bp_ct{"NBEAM"};
  if ($h{"DELAY_TRACKING"} ne "true")
  {
    $cmd .= " -t";
  }
  Dada::logMsg(1, $dl, "generateEmail: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "generateEmail: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "frb_pos_jt.py failed: ".$response);
  }
  else
  {
    sleep(1);

    # get the first 5 chars / bytes of the file 
    $cmd = "head -c 5 /tmp/".$local_img;
    Dada::logMsg(2, $dl, "generateEmail: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "generateEmail: ".$result." ".$response);

    # if the plotting did work in time 
    if (($result eq "ok") && ($response ne "ERROR"))
    {
      $msg->attach(
        Type        => 'image/png',
        Id          => $h{"frb_prefix"}."_localisation.png",
        Path        => '/tmp/'.$local_img,
        Disposition => 'attachment'
      );
    }
  }

  $msg->send;

  # sleep allow the email to be sent with attachments
  my $email_wait = 30;
  Dada::logMsg(1, $dl, "generateEmail: waiting ".$email_wait." for postfix to send email + attachements");
  sleep($email_wait);

  $cmd = "rm -f /tmp/".$h{"frb_prefix"}."_*_".$h{"sample"}.".png";
  Dada::logMsg(1, $dl, "generateEmail: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "generateEmail: ".$result." ".$response);

  return ("ok", "");
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

  my $cmd = "^coincidencer";
  Dada::logMsg(2, $dl ,"controlThread: killProcess(".$cmd.", dada)");
  my ($result, $response) = Dada::killProcess($cmd, "dada");
  Dada::logMsg(2, $dl ,"controlThread: killProcess() ".$result." ".$response);

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

# 
# Check if furby coincides with FRB trigger
# 
sub furby_check($) {
  }


END { }

1;  # return value from file
