#!/usr/bin/env perl 

#
# Author:   Andrew Jameson
# Created:  6 Dec, 2007
# Modified: 9 Jan, 2008
#
# This daemons runs continuously produces feedback plots of the
# current observation


require "Dada.pm";        # DADA Module for configuration options
use strict;               # strict mode (like -Wall)
use File::Basename;


#
# Constants
#
use constant DEBUG_LEVEL         => 0;
use constant PROCESSED_FILE_NAME => "processed.txt";
use constant IMAGE_TYPE          => ".png";
use constant TOTAL_F_RES         => "total_f_res.ar";
use constant TOTAL_T_RES         => "total_t_res.ar";


#
# Global Variable Declarations
#
our %cfg = Dada->getDadaConfig();
our $ext = "ar";


#
# Signal Handlers
#
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;
                                                                                                                    

#
# Local Variable Declarations
#

my $bindir              = Dada->getCurrentBinaryVersion();
my $cmd;
my $num_results = 0;
my $fres = "";
my $tres = "";

# Autoflush output
$| = 1;

# Sanity check for this script
if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
  print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
  print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
  exit(1);
}


# Get command line
if ($#ARGV!=0) {
    usage();
    exit 1;
}

(my $utc_start) = @ARGV;

my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;
my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;


if (! (-d $results_dir)) {
  debugMessage(0, "Error: Results directory \"".$results_dir."\" did not exist");
  exit 1;
}


if (! (-d $archive_dir)) {
  debugMessage(0, "Error: Archive directory \"".$archive_dir."\" did not exist");
  exit 1;
}


#
# Main
#

my $dir      = $archive_dir;
my $plot_dir = $results_dir;


# TODO check that directories are correctly sorted by UTC_START time
debugMessage(2,"Looking obs.start files in ".$results_dir);

# The number of results expected should be the number of obs.start files
$num_results = countObsStart($results_dir);


if ($num_results > 0) {

  # Get rid of the current processed.txt file from this directory
  deleteProcessed($archive_dir);

  debugMessage(1,"Finalising observation: ".$archive_dir);
  ($fres, $tres) = processAllArchives($archive_dir);

  debugMessage(1,"Making final low res archives for ".$archive_dir);
  Dada->makePlotsFromArchives($results_dir, "/tmp", $fres, $tres, "240x180");
  debugMessage(1,"Making final hi res archives for ".$archive_dir);
  Dada->makePlotsFromArchives($results_dir, "/tmp", $fres, $tres, "1024x768");

  # debugMessage(1,"Deleting source lowres archives for ".$obs_results_dir);
  # my $cmd = "rm -f ".$obs_results_dir."/*/*.lowres";
  # system($cmd);

}

exit(0);

###############################################################################
#
# Functions
#


#
# For the given utc_start ($dir), and archive (file) add the archive to the 
# summed archive for the observation
#
sub processArchive($$) {

  (my $dir, my $file) = @_;

  debugMessage(2, "processArchive(".$dir.", ".$file.")");

  my $bindir =      Dada->getCurrentBinaryVersion();
  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};

  # The combined results for this observation (dir == utc_start) 
  my $total_f_res = $dir."/".TOTAL_F_RES;
  my $total_t_res = $dir."/".TOTAL_T_RES;

  # If not all the archives are present, then we must create empty
  # archives in place of the missing ones
  $cmd = "find ".$dir."/* -type d";
  debugMessage(2, $cmd);
  my $find_result = `$cmd`;
  #print $find_result;
  my @sub_dirs = split(/\n/, $find_result);

  # Find out how many archives we actaully hae
  $cmd = "find ".$dir."/*/ -type f -name ".$file;
  debugMessage(2, $cmd);
  $find_result = `$cmd`;
  my @archives = split(/\n/, $find_result);

  debugMessage(2, "Found ".($#archives+1)." / ".($#sub_dirs+1)." archives");

  my $output = "";
  my $real_archives = "";
  my $subdir = "";

  if ($#archives < $#sub_dirs) {

    foreach $subdir (@sub_dirs) {

     debugMessage(2, "subdir = ".$subdir.", file = ".$file); 
     # If the archive does not exist in the frequency dir
      if (!(-f ($subdir."/".$file))) { 

        debugMessage(1, "archive ".$subdir."/".$file." was not present");

        my ($filename, $directories, $suffix) = fileparse($subdir);
        my $band_frequency = $filename;
        my $input_file = $archives[0];
        my $output_file = $subdir."/".$file;
        my $tmp_file = $input_file;
        $tmp_file =~ s/\.$ext$/\.tmp/;

        $cmd = $bindir."/pam -o ".$band_frequency." -e tmp ".$input_file;
        debugMessage(2, $cmd);
        $output = `$cmd`;
        debugMessage(2, $output);

        $cmd = "mv -f ".$tmp_file." ".$output_file;
        $output = `$cmd`;

        $cmd = $bindir."/paz -w 0 -m ".$output_file;
        debugMessage(2, $cmd);
        $output = `$cmd`;
        debugMessage(2, $output);

        debugMessage(2, "Deleting tmp file ".$tmp_file);
        unlink($tmp_file);

      } else {
        my ($filename, $directories, $suffix) = fileparse($subdir);
        $real_archives .= " ".$filename;
      }
    }

  } else {
    foreach $subdir (@sub_dirs) {
      my ($filename, $directories, $suffix) = fileparse($subdir);
      $real_archives .= " ".$filename;
    } 
  }

  # The frequency summed archive for this time period
  my $current_archive = $dir."/".$file;

  # combine all thr frequency channels
  $cmd = $bindir."/psradd -R -f ".$current_archive." ".$dir."/*/".$file;
  $output = `$cmd`;
  debugMessage(2, $output);

  # If this is the first result for this observation
  if (!(-f $total_f_res)) {

    $cmd = "cp ".$current_archive." ".$total_f_res;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);
                                                                                                                 
    # Fscrunc the archive
    $cmd = $bindir."/pam -F -m ".$current_archive;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);
                                                                                                                 
    # Tres operations
    $cmd = "cp ".$current_archive." ".$total_t_res;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);

  } else {

    my $temp_ar = $dir."/temp.ar";
                                                                                                                 
    # Fres Operations
    $cmd = $bindir."/psradd -s -f ".$temp_ar." ".$total_f_res." ".$current_archive;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);
    unlink($total_f_res);
    rename($temp_ar,$total_f_res);
                                                                                                                 
    # Fscrunc the archive
    $cmd = $bindir."/pam -F -m ".$current_archive;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);
                                                                                                                 
    # Tres Operations
    $cmd = $bindir."/psradd -f ".$temp_ar." ".$total_t_res." ".$current_archive;
    debugMessage(2, $cmd);
    $output = `$cmd`;
    debugMessage(2, $output);
    unlink($total_t_res);
    rename($temp_ar,$total_t_res);

  }

  # clean up the current archive
  unlink($current_archive);
  debugMessage(2, "unlinking $current_archive");

  # Record this archive as processed and what sub bands were legit
  recordProcessed($dir, $file, $real_archives);

  return ($current_archive, $total_f_res, $total_t_res);

}

#
# Counts the numbers of *.ext archives in total received
#
sub countArchives($$) {

  my ($dir, $skip_existing_archives) = @_;

  my @processed = getProcessedFiles($dir);
  my $i = 0;

  my $cmd = "find ".$dir."/*/ -name \"*.".$ext."\" -printf \"%P\n\"";
  debugMessage(3, $cmd);
  my $find_result = `$cmd`;

  my %archives = ();

  my @files = split(/\n/,$find_result);
  my $file = "";
  foreach $file (@files) {
  
    my $has_been_processed = 0;
    # check that this file has not already been "processed";
    for ($i=0;$i<=$#processed;$i++) {
      if ($file eq $processed[$i]) {
        $has_been_processed = 1;
      }
    }


    # If we haven't processed this OR we want to get all archives
    if (($has_been_processed == 0) || ($skip_existing_archives == 0)) {
 
      if (!(exists $archives{$file})) {
        $archives{$file} = 1;
      } else {
        $archives{$file} += 1;
      }

    }
  }
  return %archives;

}


sub getProcessedFiles($) {

  (my $dir) = @_;
  my $i = 0;
  my @lines = ();
  my @arr = ();
  my @archives = ();
  my $fname = $dir."/".PROCESSED_FILE_NAME;

  open FH, "<".$fname or return @lines;
  @lines = <FH>;
  close FH;

  for ($i=0;$i<=$#lines;$i++) {
    @arr = split(/ /,$lines[$i]); 
    chomp($arr[0]);
    @archives[$i] = $arr[0];
    # print "Processed file $i = ".$archives[$i]."\n";
  }

  return @archives;

}


sub countObsStart($) {

  my ($dir) = @_;

  my $cmd = "find ".$dir." -name \"obs.start\" | wc -l";
  my $find_result = `$cmd`;
  chomp($find_result);
  return $find_result;

}

sub deleteProcessed($) {

  my ($dir) = @_;

  my $file = $dir."/".PROCESSED_FILE_NAME;

  unlink($file);

}

sub recordProcessed($$$) {

  my ($dir, $archive, $subbands) = @_;

  my $FH = "";
  my $record = $dir."/".PROCESSED_FILE_NAME;
 
  debugMessage(2, "Recording ".$archive." as processed");
  open FH, ">>$record" or return ("fail","Could not append record file: $record");
  print FH $archive.$subbands."\n";
  close FH;
 
  return ("ok","");
}

sub debugMessage($$) {
  (my $level, my $message) = @_;
  if ($level <= DEBUG_LEVEL) {
    print $message."\n";
  }
}

sub processAllArchives($) {

  (my $dir) = @_;

  debugMessage(1, "processAllArchives(".$dir.")");

   # Delete the existing fres and tres files
   unlink($dir."/".TOTAL_T_RES);
   unlink($dir."/".TOTAL_F_RES);
  
   # Get ALL archives in the observation dir
   my %unprocessed = countArchives($dir,0);

   # Sort the files into time order.
   my @keys = sort (keys %unprocessed);
 
   my $i=0;
   my $current_archive = "";
    
   for ($i=0; $i<=$#keys; $i++) {

     debugMessage(1, "Finalising archive ".$dir."/*/".$keys[$i]);
     # process the archive and summ it into the results archive for observation
     ($current_archive, $fres, $tres) = processArchive($dir, $keys[$i]);

   }

   return ($fres, $tres);

}

#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR basename($0)." : Received SIG".$sigName."\n";
  print STDERR basename($0)." : Exiting: ".Dada->getCurrentDadaTime(0)."\n";
  exit(1);

}

sub usage() {
  print "Usage: ".basename($0)." utc_start\n";
  print "  utc_start  utc_start of the observation\n";
}

                                                                                
