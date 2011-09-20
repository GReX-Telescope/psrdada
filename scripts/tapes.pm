package Dada::tapes;

##############################################################################
#
# Dada::tapes
#
# Generic tape functinos
#

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use threads;
use threads::shared;
use Dada;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&loadTapeGeneral &unloadCurrentTape &tapeFSF &tapeIsLoaded &tapeIsReady &tapeGetID &tapeRewind &getTapeStatus);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $dev $robot $quit_daemon);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $dev;
our $robot;
our $quit_daemon : shared;

#
# non-exported package globals go here
#

#
# initialize package globals
#
$dl = 1;
$dev = "/dev/nst0";
$robot = 0;
$quit_daemon = 0;

#
# initialize other variables
#


###################################################################3
#
# Package methods
#
# tapeFSF   seeks forward the specified number of files
#


#
# seek forward the specified number of files
#
sub tapeFSF($) {

  (my $nfiles) = @_;

  Dada::logMsg(2, $dl, "tapeFSF(".$nfiles.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." fsf ".$nfiles;
  Dada::logMsg(3, $dl, "tapeFSF: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tapeFSF: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeFSF: ".$cmd." failed: ".$response);
    return ("fail", "FSF failed: ".$response);
  }

  Dada::logMsg(2, $dl, "tapeFSF() ok");
  return ("ok", "");

}

sub tapeBSFM($) {

  (my $nfiles) = @_;

  Dada::logMsg(2, $dl, "tapeBSFM(".$nfiles.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." bsfm ".$nfiles;
  Dada::logMsg(3, $dl, "tapeBSFM: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tapeBSFM: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeBSFM: ".$cmd." failed: ".$response);
    return ("fail", "BSFM failed: ".$response);
  }

  Dada::logMsg(2, $dl, "tapeBSFM: ok");
  return ("ok", "");

}

#
# Check which status bits are "on" for the tape drive and return
# the value of all the ones that can be enabled
#
sub tapeStatusBits() {

  Dada::logMsg(2, $dl, "tapeStatusBits()");

  my $cmd = "";
  my $result = "";
  my $response = "";
  my @flags = ();
  my $i = 0;
  my $f = "";

  $cmd = "mt -f ".$dev." status | grep -A1 'General status bits on' | tail -n 1";
  Dada::logMsg(3, $dl, "tapeStatusBits: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tapeStatusBits: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeStatusBits: could not read status bits");
    return ("fail", "could not read status bits");
  }
  
  @flags = split(/ /,$response);

  my $eof = 0;
  my $eod = 0;
  my $wrp = 0;
  my $onl = 0;
  my $dro = 0;
  my $ire = 0;
  my $cln = 0;
  
  for ($i=0; $i<=$#flags; $i++) {
    $f = $flags[$i];
    if ($f eq "EOF") { $eof = 1; }
    if ($f eq "EOD") { $eod = 1; }
    if ($f eq "WR_PROT") { $wrp = 1; }
    if ($f eq "ONLINE") { $onl = 1; }
    if ($f eq "DR_OPEN") { $dro = 1; } 
    if ($f eq "IM_REP_EN") { $ire = 1; }
    if ($f eq "CLN") { $cln = 1; }
  }

  Dada::logMsg(2, $dl, "tapeStatusBits() ".$eof.", ".$eod.", ".$wrp.", ".$onl.", ".$dro.", ".$ire.", ".$cln);
  return ($eof, $eod, $wrp, $onl, $dro, $ire, $cln);

}


#
# checks to see if the tape drive thinks it has a tape in it
#
sub tapeIsLoaded() {

  Dada::logMsg(2, $dl, "tapeIsLoaded()");

  my ($eof, $eod, $wrp, $onl, $dro, $ire, $cln) = tapeStatusBits();

  Dada::logMsg(2, $dl, "tapeIsLoaded() ".$onl);
  return ("ok", $onl);

}

#
# Check to see if the drive is ready for write operations
#
sub tapeIsReady() {

  Dada::logMsg(2, $dl, "tapeIsReady()");

  my ($eof, $eod, $wrp, $onl, $dro, $ire, $cln) = tapeStatusBits();

  if ($wrp) {
    return ("fail", "tape is write protected");
  } elsif ($dro) {
    return ("fail", "tape drive door is open - tape not loaded");
  } elsif ($cln) {
    return ("fail", "tape drive requires cleaning");
  } elsif ($onl) {
    return ("ok", "tape is online and ready");
  } else {
    return ("fail", "not sure EOF=".$eof." EOD=".$eod." WR_PROT=".$wrp." ONLINE=".$onl." DR_OPEN=".$dro." IM_REP_EN=".$ire." CLN=".$cln);
  }
  
}


#
# rewind the tape
#
sub tapeRewind() {

  Dada::logMsg(2, $dl, "tapeRewind()");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." rewind";
  Dada::logMsg(2, $dl, "tapeRewind: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "tapeRewind: ".$result." ".$response);

  return ($result, $response);

}


#
# Initialise the tape, writing the ID to the first file on the
# tape
#

sub tapeInit($) {

  (my $id) = @_;

  Dada::logMsg(2, $dl, "tapeInit(".$id.")");

  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl, "tapeInit: tapeWriteID(".$id.")");
  ($result, $response) = tapeWriteID($id);
  Dada::logMsg(2, $dl, "tapeInit: tapeWriteID: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeInit: tapeWriteID() failed: ".$response);
    return ("fail", "could not write tape ID: ". $response);
  }
                                              
  Dada::logMsg(2, $dl, "tapeInit: tapeGetID()");
  ($result, $response) = tapeGetID();
  Dada::logMsg(2, $dl, "tapeInit: tapeGetID: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeInit: tapeGetID() failed: ".$response);
    return ("fail", "could not get tape ID from tape");
  }

  if ($id ne $response) {
    Dada::logMsg(0, $dl, "tapeInit: newly written ID did not match specified");
    return ("fail", "could not write tape ID to tape");
  }

  return ("ok", $id);

}

 
#
# Rewind, and read the first file from the tape
#
sub tapeGetID() {

  Dada::logMsg(2, $dl, "tapeGetID()");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." rewind";
  Dada::logMsg(3, $dl, "tapeGetID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tapeGetID: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeGetID: ".$cmd." failed: ".$response);
    return ("fail", "mt rewind command failed: ".$response);;
  }

  $cmd = "tar -tf ".$dev;
  Dada::logMsg(3, $dl, "tapeGetID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "tapeGetID: ".$result." ".$response);

  if ($result ne "ok") {

    # if there is no ID on the tape this command will fail, 
    # but we can test the output message
    if ($response =~ m/tar: At beginning of tape, quitting now/) {

      Dada::logMsg(0, $dl, "tapeGetID: No ID on Tape");
      return ("ok", "");

    } else {

      Dada::logMsg(0, $dl, "tapeGetID: ".$cmd." failed: ".$response);
      return ("fail", "tar list command failed: ".$response);

    }
  }

  Dada::logMsg(3, $dl, "tapeGetID: ID = ".$response);
  my $tape_label = $response;

  Dada::logMsg(3, $dl, "tapeGetID: tapeFSF(1)");
  ($result, $response) = tapeFSF(1);
  Dada::logMsg(3, $dl, "tapeGetID: tapeFSF: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeGetID: tapeFSF failed: ".$response);
    return ("fail", "tapeFSF() failed to move forward 1 file");
  }

  Dada::logMsg(3, $dl, "tapeGetID: getTapeStatus()");
  ($result, my $filenum, my $blocknum) = getTapeStatus();
  Dada::logMsg(3, $dl, "tapeGetID: getTapeStatus() ".$result." ".$filenum." ".$blocknum);
  if ($result ne "ok") { 
    Dada::logMsg(0, $dl, "tapeGetID: getTapeStatus() failed.");
    return ("fail", "getTapeStatus() failed");
  }

  # if we are not at 0 block of file 1...
  while ($filenum ne 1 || $blocknum ne 0){

    Dada::logMsg(1, $dl, "tapeGetID: Tape out of position (f=$filenum, b=$blocknum), rewinding and skipping to start of data");
    $cmd = "mt -f ".$dev." rewind; mt -f ".$dev." fsf 1";
    Dada::logMsg(3, $dl, "tapeGetID: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "tapeGetID: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "tapeGetID: tape re-wind/skip failed: ".$response);
      return ("fail", "tape re-wind/skip failed: ".$response);
    }
    Dada::logMsg(3, $dl, "tapeGetID: getTapeStatus()");
    ($result, $filenum, $blocknum) = getTapeStatus();
    Dada::logMsg(3, $dl, "tapeGetID: getTapeStatus() ".$result." ".$filenum." ".$blocknum);
    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "tapeGetID: getTapeStatus() failed.");
      return ("fail", "getTapeStatus() failed");
    }
  }
  # The tape MUST now be in the right place to start
  Dada::logMsg(2, $dl, "tapeGetID: ID = ".$tape_label);

  return ("ok", $tape_label);
}

#
# Rewind, and write the first file from the tape
#
sub tapeWriteID($) {

  (my $tape_id) = @_;

  Dada::logMsg(2, $dl, "tapeWriteID()");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." rewind";
  Dada::logMsg(2, $dl, "tapeWriteID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "tapeWriteID: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeWriteID: ".$cmd." failed: ".$response);
    return ("fail", "mt rewind failed: ".$response);
  }

  # create an emprty file in the CWD to use
  $cmd = "touch ".$tape_id;
  Dada::logMsg(2, $dl, "tapeWriteID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "tapeWriteID: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeWriteID: ".$cmd." failed: ".$response);
    return ("fail", "could not create tmp file in cwd: ".$response);
  }

  # write the empty file to tape
  $cmd = "tar -cf ".$dev." ".$tape_id;
  Dada::logMsg(2, $dl, "tapeWriteID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "tapeWriteID: ".$result." ".$response);

  unlink($tape_id);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tapeWriteID: ".$cmd." failed: ".$response);
    return ("fail", "could not write ID to tape: ".$response);
  } 

  # write the EOF marker
  $cmd = "mt -f ".$dev." weof";
  Dada::logMsg(2, $dl, "tapeWriteID: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "tapeWriteID: ".$result." ".$response);

  # Initialisze the tapes DB record also
  #Dada::logMsg(2, $dl, "tapeWriteID: updatesTapesDB(".$tape_id.", ".TAPE_SIZE.", 0, ".TAPE_SIZE.", 1, 0)");
  #($result, $response) = updateTapesDB($tape_id, TAPE_SIZE, 0, TAPE_SIZE, 1, 0);
  #Dada::logMsg(2, $dl, "tapeWriteID: updatesTapesDB(): ".$result.", ".$response);

  return ("ok", $response);

}

sub getTapeStatus() {

  my $cmd = "";
  my $filenum = 0;
  my $blocknum = 0;

  Dada::logMsg(2, $dl, "getTapeStatus()");

  # Parkes robot has a different print out than the swinburne one
  if ($robot eq 0) {
    $cmd="mt -f ".$dev." status | grep 'file number' | awk '{print \$4}'";
  } else {
    $cmd="mt -f ".$dev." status | grep 'File number' | awk -F, '{print \$1}' | awk -F= '{print \$2}'";
  }
  Dada::logMsg(3, $dl, "getTapeStatus: cmd= $cmd");

  my ($result,$response) = Dada::mySystem($cmd);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "getTapeStatus: Failed $response");
    $filenum = -1;
    $blocknum = -1;

  } else {
    $filenum = $response;
    if ($robot eq 0) {
      $cmd="mt -f ".$dev." status | grep 'block number' | awk '{print \$4}'";
    } else {
      $cmd="mt -f ".$dev." status | grep 'block number' | awk -F, '{print \$2}' | awk -F= '{print \$2}'";
    }

    my ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "getTapeStatus: Failed $response");
      $filenum = -1;
      $blocknum = -1;
    } else {
      $blocknum = $response;
    }
  }

  Dada::logMsg(2, $dl, "getTapeStatus() ".$result." ".$filenum." ".$blocknum);
  return ($result,$filenum,$blocknum);
}



###############################################################################
#
# General Tape Functions
#


sub loadTapeGeneral($) {
  
  (my $tape) = @_;
  Dada::logMsg(2, $dl, "Dada::tapes::loadTapeGeneral(".$tape.")");
  my $result = "";
  my $response = "";
  if ($robot) {
    ($result, $response) = loadTapeRobot($tape);
  } else {
    Dada::logMsg(3, $dl, "Dada::tapes::loadTapeGeneral: loadTapeManual(".$tape.")");
    ($result, $response) = loadTapeManual($tape);
    Dada::logMsg(3, $dl, "Dada::tapes::loadTapeGeneral: loadTapeManual() ".$result." ".$response);
  }

  Dada::logMsg(2, $dl, "Dada::tapes::loadTapeGeneral() ".$result." ".$response);
  return ($result, $response);


}  

sub unloadCurrentTape() {

  my $result = "";
  my $response = "";
  if ($robot) {
    ($result, $response) = unloadCurrentTapeRobot();
  } else {
    ($result, $response) = unloadCurrentTapeManual();
  }
  return ($result, $response);
}



#
# General Tape Functions
#

###############################################################################

###############################################################################
#
# Robot Tape Functions
#

sub getCurrentTapeSlotRobot() {

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mtx status | grep 'Data Transfer Element 1' | awk '{print \$7}'";
  Dada::logMsg(3, $dl, "getCurrentTapeSlotRobot: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getCurrentTapeSlotRobot: ".$result." ".$response);

  if ($result ne "ok")  {
    Dada::logMsg(0, $dl, "getCurrentTapeSlotRobot: ".$cmd." failed: ".$response);
    return ("fail", "could not determine current tape in robot");
  }

  Dada::logMsg(2, $dl, "getCurrentTapeSlotRobot: ID = ".$response);
  return ("ok", $response);

}


#
# get the current tape in the robot
#
sub getCurrentTapeRobot() {

  Dada::logMsg(2, $dl, "getCurrentTapeRobot()");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mtx status | grep 'Data Transfer Element 1' | awk '{print \$10}'";
  Dada::logMsg(3, $dl, "getCurrentTapeRobot: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "getCurrentTapeRobot: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "getCurrentTapeRobot: ".$cmd." failed: ".$response);
    return ("fail", "could not determine current tape in robot");
  }

  Dada::logMsg(2, $dl, "getCurrentTapeRobot: ID = ".$response);
  return ("ok", $response);
   
} 


# 
# Return array of current robot status
#   
sub getStatusRobot() {

  Dada::logMsg(2, $dl, "getStatusRobot()");
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $line = "";
  my @lines = ();
  my %results = (); 
  my @tokens = ();
  my $slotid = "";
  my $tapeid = "";
  my $state = "";
  my $junk = "";
 
  $cmd = "mtx status";
  Dada::logMsg(3, $dl, "getStatusRobot: ".$cmd);

  ($result, $response) = Dada::mySystem($cmd);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "getStatusRobot: ".$cmd." failed: ".$response);
    return "fail";
  }

  # parse the response
  $line = "";
  @lines = split(/\n/,$response);

  %results = ();

  foreach $line (@lines) {

    @tokens = ();
    @tokens = split(/ +/,$line);

    if ($line =~ m/^Data Transfer Element 1/) {

      Dada::logMsg(3, $dl, "Transfer: $line");
      if ($tokens[3] eq "1:Full") {
        $results{"transfer"} = $tokens[9];
      } else {
        $results{"transfer"} = "Empty";
      }

    } elsif ($line =~ m/Storage Element/) {

      # special case for IMPORT/EXPORT elements
      if (($#tokens >= 4) && ($tokens[4] =~ m/IMPORT/)) {

        Dada::logMsg(3, $dl, "Storage: $line");
        $slotid = $tokens[3];
        ($junk, $state) = split(/:/,$tokens[4]);

        if ($state eq "Empty") {
          $results{$slotid} = "Empty";
        } else {
          ($junk, $tapeid) = split(/=/,$tokens[5]);
          $results{$slotid} = $tapeid;
        }

      } else { 

        Dada::logMsg(3, $dl, "Storage: $line");
        ($slotid, $state) = split(/:/,$tokens[3]);

        if ($state eq "Empty") {
          $results{$slotid} = "Empty";
        } elsif ($#tokens == 2) {
          $results{$slotid} = "Empty"; 
        } else {
          ($junk, $tapeid) = split(/=/,$tokens[4]);
          $results{$slotid} = $tapeid;
        }
      }
    } else {
      # ignore
    }
  }

  return %results;
}




#
# load the specified $tape 
#
sub loadTapeRobot($) {

  (my $tape) = @_;
  Dada::logMsg(2, $dl, "loadTapeRobot(".$tape.")");

  my $result = "";
  my $response = "";
  my %status = getStatusRobot();
  my @keys = keys (%status);
  my $slot = "none";

  # find the tape
  my $i=0;
  for ($i=0; $i<=$#keys; $i++) {
    if ($tape eq $status{$keys[$i]}) {
      $slot = $keys[$i];
    }
  }

  if ($slot eq "none") {

    Dada::logMsg(0, $dl, "loadTapeRobot: tape ".$tape." did not exist in robot");
    return ("fail", "tape not in robot") ;

  } elsif ($slot eq "transfer") {

    Dada::logMsg(3, $dl, "loadTapeRobot: tape ".$tape." was already in transfer slot");
    return ("ok","");

  } else {

    Dada::logMsg(3, $dl, "loadTapeRobot: tape ".$tape." in slot ".$slot);

    # if a tape was actually loaded
    if ($status{"transfer"} ne "Empty") {

      # unload the current tape
      Dada::logMsg(3, $dl, "loadTapeRobot: unloadCurrentTape()");
      ($result, $response) = unloadCurrentTape();
      if ($result ne "ok") {
        Dada::logMsg(0, $dl, "loadTapeRobot: unloadCurrentTape failed: ".$response);
        return ("fail", "Could not unload current tape: ".$response);
      }
    }

    # load the tape in the specified slot
    Dada::logMsg(3, $dl, "loadTapeRobot: robotLoadTapeFromSlot(".$slot.")");
    ($result, $response) = loadTapeFromSlot($slot);
    Dada::logMsg(3, $dl, "loadTapeRobot: robotLoadTapeFromSlot: ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "loadTapeRobot: robotLoadTapeFromSlot failed: ".$response);
      return ("fail", "Could not load tape in robot slot ".$slot);
    }

    Dada::logMsg(2, $dl, "loadTapeRobot() ok ".$tape);
    return ("ok", $tape);
  }

}

#
# load the tape in the specified slot
#
sub loadTapeFromSlot($) {

  (my $slot) = @_;
  
  Dada::logMsg(2, $dl, "loadTapeFromSlot(".$slot.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mtx load ".$slot." 1";
  Dada::logMsg(3, $dl, "loadTapeFromSlot: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "loadTapeFromSlot: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "loadTapeFromSlot: ".$cmd." failed: ".$response);
    return ("fail", $response);
  }

  Dada::logMsg(2, $dl, "loadTapeFromSlot() ok");
  return ("ok", "");

}

sub unloadCurrentTapeRobot() {

  Dada::logMsg(2, $dl, "unloadCurrentTapeRobot()");

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $current_tape_slot = "";

  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: getCurrentTapeSlotRobot()");
  ($result, $response) = getCurrentTapeSlotRobot();
  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: getCurrentTapeSlotRobot() ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "unloadCurrentTapeRobot: ".$cmd." getCurrentTapeSlotRobot() failed: ".$response);
    return ("fail", "could not determine current tapes slot");
  }
  $current_tape_slot = $response;

  $cmd = "mt -f ".$dev." eject";
  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: ".$result." ".$response);
  
  if ($result ne "ok") { 
    Dada::logMsg(0, $dl, "unloadCurrentTapeRobot: ".$cmd." failed: ".$response);
    return ("fail", "eject command failed");
  }

  $cmd = "mtx unload ".$current_tape_slot." 1";
  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "unloadCurrentTapeRobot: ".$result." ".$response);
  
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "unloadCurrentTapeRobot: ".$cmd." failed: ".$response);
    return ("fail", "mtx unload command failed");
  }

  Dada::logMsg(2, $dl, "unloadCurrentTapeRobot() ok");
  return ("ok", "");
  
}



#
# Robot Tape Functions
#
###############################################################################


###############################################################################
#
# Manual Tape Functions
#

#
# Unloads the tape currently in the robot
#
sub unloadCurrentTapeManual() {

  Dada::logMsg(2, $dl, "unloadCurrentTapeManual()");
  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "mt -f ".$dev." eject";
  Dada::logMsg(3, $dl, "unloadCurrentTapManuale: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "unloadCurrentTapeManual: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "unloadCurrentTapeManual: ".$cmd." failed: ".$response);
    return ("fail", "eject command failed");
  }

  Dada::logMsg(2, $dl, "unloadCurrentTapeManual() ok");
  return ("ok", "");

}

sub loadTapeManual($) {

  (my $tape) = @_;

  Dada::logMsg(2, $dl, "loadTapeManual(".$tape.")");

  Dada::logMsg(1, $dl, "loadTapeManual: asking for tape ".$tape);

  my $cmd = "mt -f ".$dev." offline";
  my $result = "";
  my $response = "";
  my $string = "";

  Dada::logMsg(3, $dl, "loadTapeManual: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "loadTapeManual: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "loadTapeManual: tape offline failed: ".$response);
  }

  my $n_tries = 10;
  my $inserted_tape = "none";

  while (($inserted_tape ne $tape) && ($n_tries >= 0) && (!$quit_daemon)) {

    Dada::logMsg(3, $dl, "loadTapeManual: tapeIsLoaded() [qd=".$quit_daemon."]");
    ($result, $response) = tapeIsLoaded();
    Dada::logMsg(3, $dl, "loadTapeManual: tapeIsLoaded ".$result." ".$response);

    if ($result ne "ok") { 
      Dada::logMsg(0, $dl, "loadTapeManual: tapeIsLoaded failed: ".$response);
      return ("fail", "could not determine if tape is loaded in drive");
    }

    # If a tape was not loaded
    if ($response ne 1) {
      $inserted_tape = "none";
      Dada::logMsg(1, $dl, "loadTapeManual: sleeping 10 seconds for tape online");
      sleep(10);

    } else {

      Dada::logMsg(3, $dl, "loadTapeManual: tapeGetID()");
      ($result, $response) = tapeGetID();
      Dada::logMsg(3, $dl, "loadTapeManual: tapeGetID() ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsg(0, $dl, "loadTapeManual: tapeGetID() failed: ".$response);
        $inserted_tape = "none";
      } else {
        $inserted_tape = $response;
      }
    }
  }

  Dada::logMsg(2, $dl, "loadTapeManual: qd=".$quit_daemon);

  if ($inserted_tape eq $tape) {
    return ("ok", $inserted_tape);
  } else {
    return ("fail", "failed to insert tape: ".$tape);
  }

}



#
# Manual Tape Functions
#
###############################################################################



END { }

1;  # return value from file

