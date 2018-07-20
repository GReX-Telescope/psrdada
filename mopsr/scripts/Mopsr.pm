package Mopsr;

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Math::BigInt;
use Math::BigFloat;
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);
use Dada;

require Exporter;
require AutoLoader;

@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &clientCommand
  &getObsDestinations
  &makePlotsFromArchives
  &getConfig
  &getCornerturnConfig
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant DEBUG_LEVEL  => 0;



sub logMessage($$) 
{
  my ($level, $msg) = @_;
  if ($level <= 2)
  {
     print "[".Dada::getCurrentDadaTime(0)."] ".$msg."\n";
  }
}

sub clientCommand($$)
{
  my ($command, $machine) = @_;

  my %cfg = Mopsr::getConfig();
  my $result = "fail";
  my $response = "Failure Message";

  my $handle = Dada::connectToMachine($machine, $cfg{"CLIENT_MASTER_PORT"}, 0);
  # ensure our file handle is valid
  if (!$handle) {
    return ("fail","Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"});
  }

  ($result, $response) = Dada::sendTelnetCommand($handle,$command);

  $handle->close();

  return ($result, $response);

}

# Return the destinations that an obs with the specified PID should be sent to
sub getObsDestinations($$) {
  
  my ($obs_pid, $dests) = @_;
  
  my $want_swin = 0;
  my $want_parkes = 0;
  
  if ($dests =~ m/swin/) {
    $want_swin = 1;
  }
  if ($dests =~ m/parkes/) {
    $want_parkes = 1;
  }

  return ($want_swin, $want_parkes);

}

sub getConfig(;$) 
{
  (my $sub_type) = @_;
  if ($sub_type eq "")
  {
    $sub_type = "aq";
  }

  my $config_file = $DADA_ROOT."/share/mopsr.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);

  my $ct_config_file = $DADA_ROOT."/share/mopsr_cornerturn.cfg";
  my %ct_config = Dada::readCFGFileIntoHash($ct_config_file, 0);

  my %combined = (%config, %ct_config);

  my $sub_config_file = $DADA_ROOT."/share/mopsr_".$sub_type.".cfg";
  my %sub_config = Dada::readCFGFileIntoHash($sub_config_file, 0);
  %combined = (%combined, %sub_config);

  return %combined;
}


sub getCornerturnConfig($)
{
  (my $type) = @_;

  my $config_file = $DADA_ROOT."/share/mopsr_".$type."_cornerturn.cfg";
  my %config = ();
  if (-f $config_file)
  {
    %config = Dada::readCFGFileIntoHash($config_file, 0);
  }
  else
  {
    print "ERROR: cornerturn config file [".$config_file."] did not exist\n";
  }

  return %config;
}

###############################################################################
#
# Create plots for use in the web interface
#
sub makePlotsFromArchives($$$$$$$\%) 
{
  my ($dir, $total_f_res, $total_t_res, $res, $ten_sec_archive, $source, $dl, $cfg_ref) = @_;
  my %cfg = %$cfg_ref;

  my $web_style_txt = $cfg{"SCRIPTS_DIR"}."/web_style.txt";
  my $args = "-g ".$res." ";
  my $pm_args = "-g ".$res." -m ".$source." ";
  my ($cmd, $result, $response);
  my ($bscrunch, $bscrunch_t);
  my $sdir = $dir."/".$source;

  my $nchan = (int($cfg{"PWC_END_CHAN"}) - int($cfg{"PWC_START_CHAN"})) + 1;
  if ($nchan == 20)
  {
    $nchan = 4;
  }
  if ($nchan == 40)
  {
    $nchan = 8;
  }
  if ($nchan == 320)
  {
    $nchan = 8;
  }

  # If we are plotting hi-res - include
  if ($res ne "1024x768") 
  {
    $args .= " -s ".$web_style_txt." -c below:l=unset";
    $bscrunch = " -j 'B 128'";
    $bscrunch_t = " -j 'B 128'";
    $pm_args .= " -p";
  } else {
    $bscrunch = "";
    $bscrunch_t = "";
  }

  my $bin = Dada::getCurrentBinaryVersion()."/psrplot ".$args;
  my $timestamp = Dada::getCurrentDadaTime(0);

  my $ti = $timestamp.".".$source.".ti.".$res.".png";
  my $fr = $timestamp.".".$source.".fr.".$res.".png";
  my $fl = $timestamp.".".$source.".fl.".$res.".png";
  my $bp = $timestamp.".".$source.".bp.".$res.".png";
  my $pm = $timestamp.".".$source.".pm.".$res.".png";
  my $ta = $timestamp.".".$source.".ta.".$res.".png";
  my $tc = $timestamp.".".$source.".tc.".$res.".png";
  my $l9 = $timestamp.".".$source.".l9.".$res.".png";
  my $st = $timestamp.".".$source.".st.".$res.".png";

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
  if ($ten_sec_archive ne "")
  {
    $cmd = $bin." -pb -x -D ".$dir."/bp_tmp/png ".$ten_sec_archive;
    Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
  }

  # TOAS
  my $timing_repo_topdir = "/home/observer/Timing/";
  my $template_pattern = $timing_repo_topdir."ephemerides/".$source."/*.std";
  my @template = glob($template_pattern);
  my $ephem_pattern = $timing_repo_topdir."ephemerides/".$source."/good.par";
  my @ephem = glob($ephem_pattern);
  if (@template and @ephem) {
    if ( ! ( -f $sdir."/previous.tim" ) ) {
      $cmd = "pat -j FT -s ".$template[0]." -A FDM -f tempo2 /home/observer/Timing/profiles/".$source."/*FT > ".$sdir."/previous.tim";
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::myShellStdout($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
    }

    $cmd = "cp ".$sdir."/previous.tim ".$sdir."/temp_all.tim";
    Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
    ($result, $response) = Dada::myShellStdout($cmd);
    Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
    if ($result eq "ok") {
      # add the current ToA, mark as last, filter 0 uncertainty
      $cmd = "pat -j FT -s ".$template[0]." -A FDM -f tempo2 ".$total_t_res." | grep -v ^FORMAT | sed 's/\$/-last yes/' >> ".$sdir."/temp_all.tim";
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::myShellStdout($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      $cmd= "awk '{if (NF==2 || \$4>0) {print \$0} else print \"C \"\$0}' ".$sdir."/temp_all.tim > ".$sdir."/temp.tim";
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::myShellStdout($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      if ($result eq "ok") {
        $cmd = "tempo2 -gr plk -set FINISH 99999 -setup ".$ENV{"TEMPO2"}."/plugin_data/plk_setup_image_molo.dat -f ".$ephem[0]." ".$sdir."/temp.tim -nofit -xplot 10 -showchisq -grdev ".$dir."/".$ta."/png";
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

        $cmd = "remove-outliers2 -c ".$sdir."/outlier_sweep1 -s 0.3 -m smooth -p ".$ephem[0]." -t ".$sdir."/temp.tim > ".$sdir."/temp.clean_smooth.tim";
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
        # the above command occasionally deletes everything. In that case, just copy the original file
        $cmd ="grep -v -e ^C -e ^FORMAT ".$sdir."/temp.clean_smooth.tim | wc -l";
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
        if ($response lt 2) {
          $cmd ="cp ".$sdir."/temp.tim ".$sdir."/temp.clean_smooth.tim";
          Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
        }

        $cmd = "remove-outliers2 -c ".$sdir."/outlier_sweep2 -m mad -p ".$ephem[0]." -t ".$sdir."/temp.clean_smooth.tim > ".$sdir."/temp.clean.tim";
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

        # ensure last point survives
        $cmd = "cat ".$sdir."/temp.clean.tim | grep -v last > ".$sdir."/temp.clean.tim2 ; cat ".$sdir."/temp.clean.tim | grep last | sed 's/^C//' >> ".$sdir."/temp.clean.tim2; mv ".$sdir."/temp.clean.tim2 ".$sdir."/temp.clean.tim";
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

        if ($result eq "ok") {
          # check if anything survived the cleaning:
          $cmd ="grep -v -e ^C -e ^FORMAT ".$sdir."/temp.clean.tim | wc -l";
          Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

          if ($response gt 0) {
            $cmd = "tempo2 -gr plk -set FINISH 99999 -setup ".$ENV{"TEMPO2"}."/plugin_data/plk_setup_image_molo.dat -f ".$ephem[0]." ".$sdir."/temp.clean.tim -nofit -xplot 10 -showchisq -grdev ".$dir."/".$tc."/png";
            Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
          } else {
            $cmd = "cp ".$dir."/".$ta." ".$dir."/".$tc;
            Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
            ($result, $response) = Dada::mySystem($cmd);
            Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
          }
        }
      }
    }
  } else {
    # Don't have a template and/or ephemeris:
    # Images generated with:
    # convert -size 1024x768 -gravity center -background white -fill "#FF00B8" label:"No template" no_template_1024x768.png
    # convert -size 120x90 -gravity center -background white -fill "#FF00B8" label:"No template" no_template_120x90.png
    # convert -size 1024x768 -gravity center -background white -fill "#FF00B8" label:"No ephemeris" no_ephemeris_1024x768.png
    # convert -size 120x90 -gravity center -background white -fill "#FF00B8" label:"No ephemeris" no_ephemeris_120x90.png
    # convert -size 1024x768 -gravity center -background white -fill "#FF00B8" label:"No template\nNo ephemeris" no_template_ephemeris_1024x768.png
    # convert -size 120x90 -gravity center -background white -fill "#FF00B8" label:"No template\nNo ephemeris" no_template_ephemeris_120x90.png
    if (not @template and not @ephem) {
      $cmd = "cp ".$dir."/../no_template_ephemeris_".$res.".png ".$dir."/".$tc;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      $cmd = "cp ".$dir."/../no_template_ephemeris_".$res.".png ".$dir."/".$ta;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
    } elsif (not @template) {
      $cmd = "cp ".$dir."/../no_template_".$res.".png ".$dir."/".$tc;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      $cmd = "cp ".$dir."/../no_template_".$res.".png ".$dir."/".$ta;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      $cmd = "cp ".$dir."/../no_template_".$res.".png ".$dir."/".$st;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
    } elsif (not @ephem) {
      $cmd = "cp ".$dir."/../no_ephemeris_".$res.".png ".$dir."/".$tc;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      $cmd = "cp ".$dir."/../no_ephemeris_".$res.".png ".$dir."/".$ta;
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
    }
  }

  # POWER MONITOR
  if (-f $sdir."/power_monitor.log")
  {
    $cmd = "mopsr_pmplot -c ".$nchan." ".$pm_args." -D ".$dir."/pm_tmp/png ".$sdir."/power_monitor.log";
    Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
  }

  # plot the last 9 image
  $cmd = "find ".$dir." -name '*.".$source.".l9.".$res.".png' | wc -l";
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
  if (($result eq "ok") && ($response eq "0"))
  {
    my $ft_dir = "/home/observer/Timing/profiles/".$source;
    if ( -d $ft_dir )
    {
      $cmd = "find ".$ft_dir." -name '*.FT' | sort  | tail -n 9";
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

      if ($result eq "ok" && $response ne "")
      {
        $response =~ s/\n/ /g;
        $cmd = "psrplot -jFT -pD -N3,3 -jC ".$args." -D ".$dir."/".$l9."/png ".$response;
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      }
    }
  }

  # plot the standard 
  $cmd = "find ".$dir." -name '*.".$source.".st.".$res.".png' | wc -l";
  Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
  if (($result eq "ok") && ($response eq "0"))
  {
    my $par_dir = "/home/observer/Timing/ephemerides/".$source;
    if ( -d $par_dir )
    {
      $cmd = "find ".$par_dir." -name '*.std' | tail -n 1";
      Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);

      if ($result eq "ok" && $response ne "")
      {
        $response =~ s/\n/ /;
        $cmd = "psrplot -p flux -jC ".$args." -D ".$dir."/".$st."/png ".$response;
        Dada::logMsg(2, $dl, "makePlotsFromArchives: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, $dl, "makePlotsFromArchives: ".$result." ".$response);
      }
    }
  }

  # wait for each file to "appear"
  my $waitMax = 5;
  while ($waitMax) {
    if ( (-f $dir."/pvfl_tmp") &&
         (-f $dir."/pvt_tmp") &&
         (-f $dir."/pvfr_tmp") &&
         (-f $dir."/bp_tmp") &&
         ( (! -f $sdir."/power_monitor.log") || (-f $dir."/pm_tmp") ) )
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
  if ((-f $sdir."/power_monitor.log") && (-f $dir."/pm_tmp" ))
  {
    system("mv -f ".$dir."/pm_tmp ".$dir."/".$pm);
  }
  Dada::logMsg(2, $dl, "makePlotsFromArchives: plots renamed");
}
__END__
