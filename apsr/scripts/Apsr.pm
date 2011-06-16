package Apsr;

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Dada;

require Exporter;
require AutoLoader;

@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &getConfig
  &getApsrConfig
  &getApsrConfigFile
  &makePlotsFromArchives
  &processHeader
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant MAX_VALUE        => 1;
use constant AVG_VALUES       => 2;


sub getApsrConfig() {
  my $config_file = getApsrCFGFile();
  my %config = Dada::readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub getApsrCFGFile() {
  return $DADA_ROOT."/share/apsr.cfg";
}


sub makePlotsFromArchives($$$$$) {

  my ($dir, $source, $total_f_res, $total_t_res, $res) = @_;

  my %local_cfg = Apsr::getApsrConfig();
  my $web_style_txt = $local_cfg{"SCRIPTS_DIR"}."/web_style.txt";
  my $psrplot_args = "-g ".$res." -jpC";
  my $cmd = "";

  # If we are plotting hi-res - include
  if ($res eq "1024x768") {
    # No changes
  } else {
    $psrplot_args .= " -s ".$web_style_txt." -c below:l=unset";
  }

  my $bin = Dada::getCurrentBinaryVersion()."/psrplot ".$psrplot_args;
  my $timestamp = Dada::getCurrentDadaTime(0);

  my $pvt  = "phase_vs_time_".$source."_".$timestamp."_".$res.".png";
  my $pvfr = "phase_vs_freq_".$source."_".$timestamp."_".$res.".png";
  my $pvfl = "phase_vs_flux_".$source."_".$timestamp."_".$res.".png";

  # Combine the archives from the machine into the archive to be processed
  # PHASE vs TIME
  $cmd = $bin." -p time -jFD -D ".$dir."/pvt_tmp/png ".$total_t_res;
  `$cmd`;

  # PHASE vs FREQ
  $cmd = $bin." -p freq -jTD -D ".$dir."/pvfr_tmp/png ".$total_f_res;
  `$cmd`;

  # PHASE vs TOTAL INTENSITY
  $cmd = $bin." -p flux -jTF -D ".$dir."/pvfl_tmp/png ".$total_f_res;
  `$cmd`;


  # Get plots to delete in the destination directory
  # $cmd = "find ".$plotDir." -name '*_".$resolution.IMAGE_TYPE."' -printf '\%h/\%f '";
  # my $curr_plots = `$cmd`;

  # wait for each file to "appear"
  my $waitMax = 5;  
  while ($waitMax) {
    if ( (-f $dir."/pvfl_tmp") && 
         (-f $dir."/pvt_tmp") && 
         (-f $dir."/pvfr_tmp") ) 
    {
      $waitMax = 0;
    } else {
      $waitMax--;
      sleep(1);
    }
  }

  # rename the plot files to their correct names
  system("mv -f ".$dir."/pvt_tmp ".$dir."/".$pvt);
  system("mv -f ".$dir."/pvfr_tmp ".$dir."/".$pvfr);
  system("mv -f ".$dir."/pvfl_tmp ".$dir."/".$pvfl);

}


#
# Removes files that match pattern with age > specified age
#
sub removeFiles($$$;$) {

  (my $dir, my $pattern, my $age, my $loglvl=0) = @_;

  Dada::logMsg(2, $loglvl, "removeFiles(".$dir.", ".$pattern.", ".$age.")");

  my $cmd = "";
  my $result = "";
  my @array = ();

  # find files that match the pattern in the specified dir.
  $cmd  = "find ".$dir." -name '".$pattern."' -printf \"%T@ %f\\n\" | sort -n -r";
  Dada::logMsg(2, $loglvl, "removeFiles: ".$cmd);

  $result = `$cmd`;
  @array = split(/\n/,$result);

  my $time = 0;
  my $file = "";
  my $line = "";
  my $i = 0;

  # foreach file, check the age, always leave the oldest alone :)
  for ($i=1; $i<=$#array; $i++) {

    $line = $array[$i];
    ($time, $file) = split(/ /,$line,2);

    if (($time+$age) < time) {
      $file = $dir."/".$file;
      Dada::logMsg(2, $loglvl, "removeFiles: unlink ".$file);
      unlink($file);
    }
  }

  Dada::logMsg(2, $loglvl, "removeFiles: exiting");
}


sub getConfig() {
  my $config_file = $DADA_ROOT."/share/apsr.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);
  return %config;
}

#
# Determine the processing command line given a raw header for APSR
#
sub processHeader($$) {

  my ($raw_header, $config_dir) = @_;
  
  my $result = "ok";
  my $response = "";
  my $cmd = "";
  my %h = ();
  
  %h = Dada::headerToHash($raw_header);
  
  if (($result eq "ok") && (length($h{"UTC_START"}) < 5)) {
    $result = "fail";
    $response .= "Error: UTC_START was malformed or non existent ";
  }
  
  if (($result eq "ok") && (length($h{"OBS_OFFSET"}) < 1)) {
    $result = "fail";
    $response .= "Error: OBS_OFFSET was malformed or non existent";
  }
  
  if (($result eq "ok") && (length($h{"PROC_FILE"}) < 1)) {
    $result = "fail";
    $response .=  "Error: PROC_FILE was malformed or non existent";
  }
  
  if (($result eq "ok") && (length($h{"SOURCE"}) < 1)) {
    $result = "fail"; 
    $response .=  "Error: SOURCE was malformed or non existent";
  }

  if (($result eq "ok") && (length($h{"FREQ"}) < 1)) {
    $result = "fail";
    $response .=  "Error: FREQ was malformed or non existent";
  }
  
  my $source = $h{"SOURCE"};
  my $proc_cmd = "";
  my $proc_args = "";
  
  # Multi pulsar mode special case
  if ($h{"PROC_FILE"} eq "dspsr.multi") {
    $source =~ s/^[JB]//;
    $source =~ s/[a-zA-Z]*$//;

    # find the source in multi.txt
    $cmd = "grep ^".$source." ".$config_dir."/multi.txt";
    my $multi_string = `$cmd`;

    if ($? != 0) {
      $result = "fail";
      $response = "Error: ".$source." did not exist in multi.txt";

    } else {

      chomp $multi_string;
      my @multis = split(/ +/,$multi_string);

      # If we have a DM specified
      if ($multis[1] ne "CAT") {
        $proc_args .= " -D ".$multis[1];
      }

      $proc_args .= " -N ".$config_dir."/".$multis[2];

      if (! -f $config_dir."/".$multis[2]) {
        $result = "fail";
        $response = "Error: Multi-source file: ".$config_dir.
                    "/".$multis[2]." did not exist";

      } else {
        $cmd = "head -1 ".$config_dir."/".$multis[2];
        $source = `$cmd`;
        chomp $source;
      }
    }

  # If we are writing the data to disk, dont worry about the DM
  } elsif ($h{"PROC_FILE"} =~ m/scratch/) {

    $result = "ok";
    $response = "";

  } else {

    if ($h{"MODE"} eq "PSR") {

      # test if the source is in the catalogue
      my $dm = getDM($source);

      if ($dm eq "unknown") {
        $result = "fail";
        $response = "Error: ".$source." was not in psrcat's catalogue";
      }

    }
  }

  # Add the dada header file to the proc_cmd
  my $proc_cmd_file = $config_dir."/".$h{"PROC_FILE"};
  my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
  $proc_cmd = $proc_cmd_hash{"PROC_CMD"};

  # Select command line arguements special case
  if ($proc_cmd =~ m/SELECT/) {

    my $dspsr_cmd = "dspsr_command_line.pl ".$source." ".$h{"BW"}.
                    " ".$h{"FREQ"}." ".$h{"MODE"};

    my $dspsr_options = `$dspsr_cmd`;

    if ($? != 0) {
      chomp $dspsr_options;
      $result = "fail";
      $response = "Error: dspsr_command_line.pl failed: ".$dspsr_options;

    } else {
      chomp $dspsr_options;
      $proc_cmd =~ s/SELECT/$dspsr_options/;

    }
  }

  $proc_cmd .= $proc_args;

  if ($source =~ m/CalDelay/) {
    if ($proc_cmd =~ m/-2c100/) {
      # TODO put error here
    } else {
      $proc_cmd .= " -2c100";
    }
  }

  if ($result eq "ok") {
    $response = $proc_cmd;
  }

  return ($result, $response)

}



__END__
