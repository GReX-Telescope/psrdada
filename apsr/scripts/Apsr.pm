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
  &getApsrConfig
  &getApsrConfigFile
  &makePlotsFromArchives
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant MAX_VALUE        => 1;
use constant AVG_VALUES       => 2;


sub getApsrConfig() {
  my $config_file = getApsrCFGFile();
  my %config = Dada->readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub getApsrCFGFile() {
  return $DADA_ROOT."/share/apsr.cfg";
}

sub makePlotsFromArchives($$$$$$) {

  my ($module, $plotDir, $dir, $total_f_res, $total_t_res, $resolution) = @_;

  use constant PHASE_VS_TIME_FILE  => "phase_vs_time";
  use constant PHASE_VS_FREQ_FILE  => "phase_vs_freq";
  use constant PHASE_VS_FLUX_FILE  => "phase_vs_flux";
  use constant IMAGE_TYPE          => ".png";

  my %local_cfg = Apsr->getApsrConfig();
  my $web_style_txt = $local_cfg{"SCRIPTS_DIR"}."/web_style.txt";

  my $psrplot_args = "-g ".$resolution." -jp";

  # If we are plotting hi-res - include
  if ($resolution eq "1024x768") {
    # No changes
  } else {
    $psrplot_args .= " -s ".$web_style_txt." -c below:l=unset";
  }

  my $bindir = Dada->getCurrentBinaryVersion();
  my $timestamp = Dada->getCurrentDadaTime(0);
  my $phase_vs_time = $dir."/".PHASE_VS_TIME_FILE."_".$timestamp."_".$resolution.IMAGE_TYPE;
  my $phase_vs_freq = $dir."/".PHASE_VS_FREQ_FILE."_".$timestamp."_".$resolution.IMAGE_TYPE;
  my $phase_vs_flux = $dir."/".PHASE_VS_FLUX_FILE."_".$timestamp."_".$resolution.IMAGE_TYPE;

  my $cmd = "rm -f ".$dir."/*_".$resolution.IMAGE_TYPE;
  `$cmd`;

  # Combine the archives from the machine into the archive to be processed
  # PHASE vs TIME
  $cmd = $bindir."/psrplot ".$psrplot_args." -p time -jF -D ".$phase_vs_time."/png ".$total_t_res;
  # print $cmd."\n";
  `$cmd`;

  # PHASE vs FREQ
  $cmd = $bindir."/psrplot ".$psrplot_args." -p freq -jT -D ".$phase_vs_freq."/png ".$total_f_res;
  # print $cmd."\n";
  `$cmd`;

  # PHASE vs TOTAL INTENSITY
  $cmd = $bindir."/psrplot ".$psrplot_args." -p flux -jTF -D ".$phase_vs_flux."/png ".$total_f_res;
  # print $cmd."\n";
  `$cmd`;

  # If we want to make these plots the "current ones"
  if ($plotDir ne "") {

    $cmd = "rm -f ".$plotDir."/*_".$resolution.IMAGE_TYPE;
    `$cmd`;

    $cmd = "cp ".$phase_vs_time." ".$phase_vs_freq." ".$phase_vs_flux." ".$plotDir."/";
    `$cmd`;
  }

}



__END__
