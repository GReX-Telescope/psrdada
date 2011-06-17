package Caspsr;

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
  &processHeader
);

$VERSION = '1.00';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

##########################################################################
#
# Read the caspsr config file into a hash
#
sub getConfig() {
  my $config_file = $DADA_ROOT."/share/caspsr.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);
  return %config;
}

##########################################################################
#
# Determine the processing command line given a raw header for CASPSR
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

  if (($result eq "ok") && (length($h{"FREQ"}) < 1)) {
    $result = "fail";
    $response .= "Error: FREQ was malformed or non existent";
  }

  if (($result eq "ok") && (length($h{"PID"}) < 1)) {
    $result = "fail";
    $response .= "Error: PID was malformed or non existent";
  }

  if (($result eq "ok") && (length($h{"PROC_FILE"}) < 1)) {
    $result = "fail";
    $response .=  "Error: PROC_FILE was malformed or non existent";
  }

  if (($result eq "ok") && (length($h{"SOURCE"}) < 1)) {
    $result = "fail"; 
    $response .=  "Error: SOURCE was malformed or non existent";
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
      my $dm = Dada::getDM($source);
      if (($dm =~ m/NA/) || ($dm =~ m/unknown/)) {
        $result = "fail";
        $response = "SOURCE ".$source." was not in catalogue";
      }
    }
  }
  
  # Add the dada header file to the proc_cmd
  my $localhost = Dada::getHostMachineName();
  my $proc_cmd_file = $config_dir."/".$h{"PROC_FILE"};

  # Check if a custom processing file for this host exists
  if ( -f $proc_cmd_file."_".$localhost) {
    $proc_cmd_file .= "_".$localhost;
  }

  my %proc_cmd_hash = Dada::readCFGFile($proc_cmd_file);
  $proc_cmd = $proc_cmd_hash{"PROC_CMD"};
  
  $proc_cmd .= $proc_args;
  if ($source =~ m/CalDelay/) {
    if ($proc_cmd =~ m/-2c100/) {
      # TODO put error here
    } else {
      $proc_cmd .= " -2c100";
    }
  }

  # if a CAL or LEVCAL adjust -F512:D to -F512:1024
  if ($h{"MODE"} ne "PSR") {
    $proc_cmd =~ s/-F512:D/-F512:1024/;
    $proc_cmd =~ s/-F256:D/-F256:1024/;
  }

  if ($result eq "ok") {
    $response = $proc_cmd;
  }
  
  return ($result, $response)

}

