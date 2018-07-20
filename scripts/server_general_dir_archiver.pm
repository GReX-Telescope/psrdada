package Dada::server_general_dir_archiver;

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Time::Local;
use Dada;
use Dada::tapes;

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&main);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = qw($dl $daemon_name $type $drive_id $robot %cfg);

}

our @EXPORT_OK;

#
# exported package globals
#
our $dl;
our $daemon_name;
our $robot;
our $drive_id;
our $type;
our $pid;
our $required_host;
our %cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $warn;
our $error;
our $dev;
our $current_tape;
our $db_dir;
our $db_user;
our $db_host;
our $tapes_db;
our $files_db;

#
# initialize package globals
#
$dl = 1;
$daemon_name = 0;
$robot = 0;
$drive_id = 0;
$pid = "";
$type = "";

#
# initialize other variables
#
$quit_daemon = 0;
$warn = "";
$error = "";
$dev = 0;
$current_tape = "";
$db_dir = "";
$db_user = "";
$db_host = "";
$tapes_db = "";
$files_db = "";

###############################################################################
#
# Package Methods
#

sub main() 
{
  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  if (-f $warn) {
    unlink $warn;
  }
  if ( -f $error) {
   unlink $error;
  }

  # location of DB files
  ($db_user, $db_host, $db_dir) = split(/:/, $cfg{uc($type)."_DB_DIR"});

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $db_dir."/".$daemon_name.".log";
  my $cmd         = "";
  my $result      = "";
  my $response    = "";

	Dada::logMsg(1, $dl, "main: quit_file=".$quit_file);

  $tapes_db = "tapes.".$pid.".db";
  $files_db = "files.".$pid.".db";

  $dev = $cfg{uc($type)."_S4_DEVICE"};

  # Initialise tapes module variables
  $Dada::tapes::dl = $dl;
  $Dada::tapes::dev = $dev;
  $Dada::tapes::robot = $robot;
  $Dada::tapes::drive_id = $drive_id;

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  #
  # Local Varaibles
  #
  my $control_thread = 0;
  my $completed_thread = 0;
  my $i = 0;
  my $j = 0;
  my $expected_tape = "";

  # check the number of dirs is only 1
  my ($user, $host, $path) = split(/:/,$cfg{uc($type)."_DIR_0"},3);

  $j = 0;

  Dada::daemonize($log_file, $pid_file);
  Dada::logMsg(0, $dl, "STARTING SCRIPT");

	Dada::logMsg(1, $dl, "main: quit_file=".$quit_file);
  Dada::logMsg(2, $dl, "main: setDada::tapes dl=".$Dada::tapes::dl." dev=".$Dada::tapes::dev." robot=".$Dada::tapes::robot." drive_id=".$Dada::tapes::drive_id);

  # Start the daemon control thread
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # ensure compression is ON
  $cmd = "mt -f ".$dev." compression 1";
  Dada::logMsg(1, $dl, "main: enabling hardware compression on tape");
  ($result, $response) = Dada::mySystem($cmd);

  # Force a re-read of the current tape. This rewinds the tape
  Dada::logMsg(1, $dl, "main: checking current tape");

  ($result, $response) = getCurrentTape();
  Dada::logMsg(3, $dl, "main: getCurrentTape(): ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "main: getCurrentTape() failed: ".$response);
    ($result, $response) = newTape();
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "main: getNewTape() failed: ".$response);
      exit_script(1);
    }
  }
  $current_tape = $response;
  Dada::logMsg(1, $dl, "main: current tape = ".$current_tape);

  # get the expected tape to be loaded
  Dada::logMsg(2, $dl, "main: getExpectedTape()");
  ($result, $response) = getExpectedTape();
  Dada::logMsg(2, $dl, "main: getExpectedTape() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "main: getExpectedTape() failed: ".$response);
    exit_script(1);
  }

  $expected_tape = $response;
  Dada::logMsg(1, $dl, "main: expected tape = ".$expected_tape);

  # If we need to change the current tape
  if ($current_tape ne $expected_tape) {

    Dada::logMsg(2, $dl, "main: loadTape(".$expected_tape.")");
    ($result, $response) = loadTape($expected_tape);
    Dada::logMsg(2, $dl, "main: loadTape() ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "getCurrentTape: loadTape() failed: ".$response);
      exit_script(1);
    }
    $current_tape = $expected_tape;
  }


  # Seek to the End of Data on this tape
  Dada::logMsg(2, $dl, "main: seekToFile(".$current_tape.", -1)");
  ($result, $response) = seekToFile($current_tape, -1);
  Dada::logMsg(3, $dl, "main: seekToFile() ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "main: seekToFile failed: ".$response);
    exit_script(1);
  }

  my $prev_obs = "";
  my $obs = "";
  my $beam = "";
  my $bytes = "";
  my $dir = "";

  $i = 0;
  $j = 0;
  my $waiting = 0;
  my $give_up = 0;
  my $counter = 0;
  my $try_to_archive = 0;

  ($result, $response) = Dada::tapes::tapeIsReady();
  if ($result ne "ok") 
  {
    Dada::logMsg(0, $dl, "main: tape drive was not ready: ".$response);
    $quit_daemon = 1;
  }

  while (!$quit_daemon) 
  {
    Dada::logMsg(2, $dl, "main: getDirToArchive(".$path.")");
    ($result, $response) = getDirToArchive ($path);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "main: getDirToArchive failed: ".$response);
      $quit_daemon = 1;
      next;
    }

    my $dir_to_archive = $response;

    if ($dir_to_archive ne "")
    {
      # get the dir size in bytes
      $cmd = "du -sLb ".$path."/archive/".$pid."/".$dir_to_archive." | awk '{print \$1}'";
      Dada::logMsg(2, $dl, "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $dl, "main: ".$result." ".$response);
      if ($result ne "ok")
      {
        Dada::logMsg(0, $dl, "main: could not determine file size: ".$response);
        $quit_daemon = 1;
        next;
      }
      else
      {
        $bytes = $response;
      }

      $try_to_archive  = 1;
      while (!$quit_daemon && $try_to_archive)
      {
        # check that the current tape is the expected one
        my $expected_tape = getExpectedTape();

        if ($expected_tape ne $current_tape) 
        {
          Dada::logMsg(2, $dl, "main: expected tape mismatch [".$expected_tape."] != [".$current_tape."]");
          # try to get the tape loaded (robot or manual)
          ($result, $response) = loadTape($expected_tape);

          # if the tape could not be loaded for whatever reason, fail
          if ($result ne "ok")
          {
            Dada::logMsg(0, $dl, "could not load tape ".$expected_tape);
            $quit_daemon = 1;
            $try_to_archive = 0;
          }
          
          $current_tape = $expected_tape;

          Dada::logMsg(2, $dl, "main: seekToFile(".$current_tape.", -1)");
          ($result, $response) = seekToFile($current_tape, -1);
          Dada::logMsg(3, $dl, "main: seekToFile() ".$result." ".$response);
          if ($result ne "ok") {
            Dada::logMsg(0, $dl, "main: seekToFile failed: ".$response);
            exit_script(1);
          }
        }

        if ($try_to_archive)
        {
          my ($utc, $dir) = split(/\//, $dir_to_archive);
          Dada::logMsg(2, $dl, "main: tarDir(".$path.", ".$utc.", ".$dir.", ".$bytes.")");
          ($result, $response) = tarDir ($path, $utc, $dir, $bytes);
          Dada::logMsg(2, $dl, "main: tarDir() ".$result." ".$response);
          if ($result eq "ok") 
          {
            Dada::logMsg(2, $dl, "main: moveCompletedDir(".$path.", ".$utc.", ".$dir.")");
            ($result, $response) = moveCompletedDir ($path, $utc, $dir);
            if ($result ne "ok")
            {
              Dada::logMsg(2, $dl, "main: moveCompletedDir failed: ".$response); 
              $quit_daemon = 1;
            }
            else
            {
              $try_to_archive = 0;
            }
          }
          # see what the problem was
          else
          {
            if ($response eq "beam already archived")
            {
              # dont need to do anything
              $try_to_archive = 0;
            }
            # the only reason to keep trying is if the tape was full
            elsif ($response eq "not enough space on tape")
            {
              # try again after a tape reload
            }
            else
            {
              $try_to_archive = 0;
              $give_up = 1;
              Dada::logMsg(0, $dl, "main: tarDir() failed: ".$response);
            }
          }
        }
      } 

      ($result, $response) = Dada::tapes::tapeIsReady();
      if ($result ne "ok") {
        Dada::logMsg(0, $dl, "main: tape drive was not ready: ".$response);
        $give_up = 1;
      }

      if ($give_up) {
        exit_script(1);
      }
  
    }
    else
    {
      Dada::logMsg(2, $dl, "main: nothing to archive, waiting 60s");
      my $to_wait = 60;
      while (($to_wait > 0) && (!$quit_daemon))
      {
        $to_wait--;
        sleep (1);
      }
    }
  } # main loop

  # ensure tape is rewound
  Dada::logMsg(1, $dl, "Rewinding tape before exiting");
  ($result, $response) = Dada::tapes::tapeRewind();

  # rejoin threads
  $control_thread->join();


  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;
}


###############################################################################
#
# Functions
#


#
# find the oldest beam to tar that exists on all the specified disks
#
sub getDirToArchive ($)
{
  Dada::logMsg(3, $dl, "getDirToArchive()");
  my ($path) = @_;

  my ($cmd, $result, $response);
  
  # format will be DIR / archive / UTC_START /DIR 

  $cmd = "cd ".$path."/archive/".$pid."; find -L . -ignore_readdir_race -mindepth 2 -maxdepth 2 -type d | sort | head -n 1";

  ($result, $response) = Dada::mySystem($cmd);

  if ($result ne "ok") 
  {
		Dada::logMsg(0, $dl, "getDirToArchive: find failed: ".$response);
    return ("fail", "find failed");
  }
  elsif ($response eq "")
  {
    return ("ok", "");
  }
  else
  {
    my ($junk, $utc, $dir) = split(/\//, $response);
    Dada::logMsg(2, $dl, "getDirToArchive: returning ".$utc."/".$dir);
    return ("ok", $utc."/".$dir);
  }
}

#
# tars the beam to the tape drive
#
sub tarDir($$$$)
{
  my ($basedir, $utc, $dir, $est_size_bytes) = @_;

  Dada::logMsg(2, $dl, "tarDir: (".$basedir.", ".$utc.", ".$dir.", ".$est_size_bytes.")");

  my $path = $basedir."/archive/".$pid;
  Dada::logMsg(1, $dl, "Archiving  ".$utc."/".$dir." from ".$path);

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $tape = $current_tape;

  my $est_size_gbytes = ($est_size_bytes / (1000 * 1000 * 1000));

  # check if this beam will fit on the tape
  ($result, $response) = getTapeInfo($tape);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "tarDir: getTapeInfo() failed: ".$response);
    return ("fail", "could not determine tape information from database");
  }

  my ($id, $size, $used, $free, $nfiles, $full) = split(/:/,$response);

  Dada::logMsg(2, $dl, "tarDir: ".$free." GB left on tape");
  Dada::logMsg(2, $dl, "tarDir: size of this beam is estimated at ".$est_size_gbytes." GB");

  # if we estimate that there is not enough spce on the tape
  if ($free < $est_size_gbytes)
  {
    Dada::logMsg(0, $dl, "tarDir: tape ".$tape." full. (".$free." < ".$est_size_gbytes.")");

    # Mark the current tape as full and load a new tape;
    Dada::logMsg(2, $dl, "tarDir: updateTapesDB(".$id.", ".$size.", ".$used.", ".$free.", ".$nfiles);
    ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, 1);
    Dada::logMsg(2, $dl, "tarDir: updateTapesDB() ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "tarDir: updateTapesDB() failed: ".$response);
      return ("fail", "Could not mark tape full");
    }

    return ("fail", "not enough space on tape");
  }

  my $filenum = -1;
  my $blocknum = -1;
  ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "TarBeam: getTapeStatus() failed.");
    return ("fail", "Could not get tape status");
  }

  # try to position the tape on the correct file
  my $ntries = 3;
  while (($filenum ne $nfiles) || ($blocknum ne 0))
  {
    # we are not at 0 block of the next file!
    Dada::logMsg(0, $dl, "tarDir: WARNING: Tape out of position! ".
                 "(file==".$filenum."!=".$nfiles.", block==".$blocknum."!=0) Attempt to get to right place.");

    $cmd = "mt -f ".$dev." rewind; mt -f ".$dev." fsf $nfiles";
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "TarBeam: tape re-wind/skip failed: ".$response);
      return ("fail", "tape re-wind/skip failed: ".$response);
    }

    ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "TarBeam: getTapeStatus() failed.");
      return ("fail", "getTapeStatus() failed");
    }

    if ($ntries < 1) {
      return ("fail", "TarBeam: Could not get to correct place on tape!");
    }

    $ntries--;
  }

  my $tar_beam_result = "";
  my $tar_beam_response = "";
  my $bytes_written = 0;
  my $gbytes_written = 0;

  my $tries = 20;
  while ($tries > 0) 
  {
    # For historical reasons, HRE(robot==0) tapes are written with a slight 
    # different command to HRA (robot==1) tapes
		if ($robot eq 0) {
			$cmd = "cd ".$path."; tar -h -b 128 -c ".$utc."/".$dir." | dd of=".$dev." bs=64K";
		} else {
			$cmd = "cd ".$path."; tar -h -b 128 -c ".$utc."/".$dir." | dd of=".$dev." bs=64k";
		}
				
    Dada::logMsg(2, $dl, "tarDir: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);

    # fatal errors, give up straight away
    if ($response =~ m/Input\/output error/) 
    {
      Dada::logMsg(0, $dl, "tarDir: fatal tape error: ".$response);
      $tar_beam_result = "fail";
      $tar_beam_response = "input output error";
      $tries = -1;
    }

    # the tape is unexpectedly full 
    elsif ($response =~ m/No space left on device/)
    {
      Dada::logMsg(0, $dl, "tarDir: tape unexpectedly ".$tape." full");
      $tar_beam_result = "fail";
      $tar_beam_response = "not enough space on tape";
      $tries = -1;

      # Mark the current tape as full and load a new tape;
      Dada::logMsg(2, $dl, "tarDir: updateTapesDB(".$id.", ".$size.", ".$used.", ".$free.", ".$nfiles);
      ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, 1);
      Dada::logMsg(2, $dl, "tarDir: updateTapesDB() ".$result." ".$response);
      if ($result ne "ok") 
      {
        Dada::logMsg(0, $dl, "tarDir: updateTapesDB() failed: ".$response);
        return ("fail", "Could not mark tape full");
      }
    }
    # non fatal errors
    elsif (($result ne "ok") || ($response =~ m/refused/) || ($response =~ m/^0\+0 records in/)) 
    {
      Dada::logMsg(2, $dl, "tarDir: ".$result." ".$response);

      $tries--;
      $result = "fail";
      $response = "failed attempt at writing archive";
      if ($tries <= 10)
      {
        Dada::logMsg(1, $dl, "tarDir: failed to write archive to tape, attempt ".(20-$tries)." of 20");
      } 
      sleep(1);
    } 
    # check that was written
    else
    {
      # Just the last line of the DD command is relvant:
      my @temp = split(/\n/, $response);
      Dada::logMsg(1, $dl, "Archived   ".$temp[2]);
      my @vals = split(/ /, $temp[2]);
      $bytes_written = int($vals[0]);
      $gbytes_written = $bytes_written / (1000*1000*1000);

      # if we didn't write anything, its due to the nc client connecting 
      # before the nc server was ready to provide the data, simply try to reconnect
      if ($gbytes_written == 0) 
      {
        $tries--;
        sleep(2);
      } 

      # if we did write something, but it didn't match bail!
      elsif ( ($est_size_gbytes - $gbytes_written) > 0.01)
      {
        $result = "fail";
        $response = "not enough data received by nc: ".$est_size_gbytes.
                    " - ".$gbytes_written." = ".($est_size_gbytes - $gbytes_written);
        Dada::logMsg(0, $dl, "tarDir: ".$result." ".$response);
        $tar_beam_result = "fail";
        $tar_beam_response = "not enough data received";
        $tries = -1;
      } 
      else
      {
        $tar_beam_result = "ok";
        Dada::logMsg(2, $dl, "tarDir: est_size ".sprintf("%7.4f GB", $est_size_gbytes).
                      ", size = ".sprintf("%7.4f GB", $gbytes_written));
        $tries = 0;
      }
    }
  }

  if ($tar_beam_result ne "ok")
  {
    Dada::logMsg(0, $dl, "tarDir: failed to write archive to tape: ".$tar_beam_response);
    return ("fail", $tar_beam_response);
  }
  else
  {
    $tar_beam_result = "fail";
  }
  
  # Now check that the File number has been incremented one. Sometimes the
  # file number is not incremented, which usually means an error...
  ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarDir: getTapeStatus() failed.");
    return ("fail", "getTapeStatus() failed");
  }
  
  if (($blocknum ne 0) || ($filenum ne ($nfiles+1))) {
    Dada::logMsg(0, $dl, "tarDir: positioning error after write: filenum=".$filenum.", blocknum=".$blocknum);
    return ("fail", "write failed to complete EOF correctly");
  }

  # else we wrote files to the TAPE in 1 archive and need to update the database files
  $used += $gbytes_written;
  $free -= $gbytes_written;
  $nfiles += 1;

  # If less than 100 MB left, mark tape as full
  if ($free < 0.1) {
    $full = 1;
  }

  Dada::logMsg(3, $dl, "tarDir: updatesTapesDB($id, $size, $used, $free, $nfiles, $full)");
  ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, $full);
  Dada::logMsg(3, $dl, "tarDir: updatesTapesDB(): ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarDir: updateTapesDB() failed: ".$response);
    return("fail", "error ocurred when updating tapes DB: ".$response);
  }

  Dada::logMsg(3, $dl, "tarDir: updatesFilesDB(".$utc."/".$dir.", ".$id.", ".$gbytes_written.", ".($nfiles-1).")");
  ($result, $response) = updateFilesDB($utc."/".$dir, $id, $gbytes_written, ($nfiles-1));
  Dada::logMsg(3, $dl, "tarDir: updatesFilesDB(): ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarDir: updateFilesDB() failed: ".$response);
    return("fail", "error ocurred when updating filesDB: ".$response);
  }

  return ("ok",""); 
}



#
# move a completed file from the archive to the on_tape
#
sub moveCompletedDir($$$) 
{
  my ($basedir, $utc, $dir) = @_;
  Dada::logMsg(3, $dl, "moveCompletedDir(".$basedir.", ".$utc.", ".$dir.")");

  my $from = $basedir."/archive/".$pid."/".$utc;
  my $to   = $basedir."/on_tape/".$pid."/".$utc;

  my $result = "";
  my $response = "";
  my $cmd = "";

  # ensure the destination directory is created
  if (! -d $to )
  {
    $cmd = "mkdir -p ".$to;
    Dada::logMsg(3, $dl, "moveCompletedDir: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "moveCompletedDir: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "moveCompletedDir: failed to create ".$to.": ".$response);
      return ("fail", "failed to create ".$to);
    }
  }

  # move the file link to the dest dir
  $cmd = "mv ".$from."/".$dir." ".$to."/";
  Dada::logMsg(3, $dl, "moveCompletedDir: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "moveCompletedDir: ".$result." ".$response);

  if ($result ne "ok") 
  {
    Dada::logMsg(0, $dl, "moveCompletedDir: failed to move ".$dir." from ".$from." to ".$to.": ".$response);
    return ("fail", "failed to move dir");
  } 
  else 
  {
    # if there are no other files in the directory, then we can delete it
    $cmd = "find -L ".$from."/ -mindepth 1 -type f | wc -l";
    Dada::logMsg(2, $dl, "moveCompletedDir: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "moveCompletedDir: ".$result." ".$response);

    if (($result eq "ok") && ($response eq "0"))
    {
      # delete the remote directory
      $cmd = "rmdir ".$from;
      Dada::logMsg(2, $dl, "moveCompletedDir: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(3, $dl, "moveCompletedDir: ".$result." ".$response);

      if ($result ne "ok") 
      {
        Dada::logMsg(0, $dl, "moveCompletedDir: could not delete ".$from.": ".$response);
      }
    }
  }

  return ($result, $response);
}


#
# Rewinds the current tape and reads the first "index" file
#
sub getCurrentTape() {
 
  Dada::logMsg(2, $dl, "getCurrentTape()"); 

  my $result = "";
  my $response = "";
  my $tape_id = "";

  # First we need to check whether a tape exists in the robot
  Dada::logMsg(3, $dl, "getCurrentTape: tapeIsLoaded()");
  ($result, $response) = Dada::tapes::tapeIsLoaded();
  Dada::logMsg(3, $dl, "getCurrentTape: tapeIsLoaded ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "getCurrentTape: tapeIsLoaded failed: ".$response);
    return ("fail", "could not determine if tape is loaded in drive");
  }

  # if no tape is loaded
  if ($response eq 0) {
    $tape_id = "none";

  } else {

    Dada::logMsg(3, $dl, "getCurrentTape: tapeGetID()");
    ($result, $response) = Dada::tapes::tapeGetID();
    Dada::logMsg(3, $dl, "getCurrentTape: tapeGetID() ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "getCurrentTape: tapeGetID failed: ".$response);
      return ("fail", "bad binary label on current tape");
    }

    # we either have a good ID or no ID
    $tape_id = $response;
  }

  # check for empty tape with no binary label
  if ($tape_id eq "") {
    Dada::logMsg(0, $dl, "getCurrentTape: no binary label existed on tape");
    return ("fail", "no binary label existed on tape");
  }

  Dada::logMsg(2, $dl, "getCurrentTape: current tape=".$tape_id);
  return ("ok", $tape_id);

}


#
# Read the local tape database and determine what the current
# tape should be
#
sub getExpectedTape() {

  my $fname = $db_dir."/".$tapes_db;
  my $expected_tape = "none";

  Dada::logMsg(3, $dl, "getExpectedTape()");

  open FH, "<".$fname or return ("fail", "Could not read tapes db ".$fname); 
  my @lines = <FH>;
  close FH;

  my $line = "";
  # parse the file
  foreach $line (@lines) {

    chomp $line;

    if ($line =~ /^#/) {
      # ignore comments
    } else {

      Dada::logMsg(3, $dl, "getExpectedTape: testing ".$line);

      if ($expected_tape eq "none") {
        my ($id, $size, $used, $free, $nfiles, $full) = split(/ +/,$line);
     
        if (int($full) == 1) {
          Dada::logMsg(3, $dl, "getExpectedTape: skipping tape ".$id.", marked full");
        } elsif ($free < 0.1) {
          Dada::logMsg(3, $dl, "getExpectedTape: skipping tape ".$id." only ".$free." GB left");
        } else {
          $expected_tape = $id;
        }
      }
    }
  }

  if ($expected_tape ne "none") {
    Dada::logMsg(2, $dl, "getExpectedTape: ".$expected_tape);
    return ("ok", $expected_tape);
  } else {
    Dada::logMsg(0, $dl, "getExpectedTape() could not find acceptable tape");
    return ("fail", "could not find acceptable tape");
  }

}

#
# Determine what the next tape should be from tapes.db
# and try to get it loaded
#
sub newTape() {

  Dada::logMsg(2, $dl, "newTape()");

  my $result = "";
  my $response = "";

  # Determine what the "next" tape should be
  Dada::logMsg(3, $dl, "newTape: getExpectedTape()");
  ($result, $response) = getExpectedTape();
  Dada::logMsg(3, $dl, "newTape: getExpectedTape() ".$result." ".$response);

  if ($result ne "ok") {
    return ("fail", "getExpectedTape failed: ".$response);
  }
  
  my $new_tape = $response;

  # Now get the tape loaded
  Dada::logMsg(3, $dl, "newTape: loadTape(".$new_tape.")");
  ($result, $response) = loadTape($new_tape);
  Dada::logMsg(3, $dl, "newTape: loadTape(): ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "newTape: loadTape failed: ".$response);
    return ("fail", "loadTape failed: ".$response);
  }

  Dada::logMsg(2, $dl, "newTape() ".$new_tape);
  return ("ok", $new_tape);

}

################################################################################
##
## DATABASE FUNCTIONS
##


#
# Get a hash of the Tapes DB
#
sub readTapesDB() 
{
  Dada::logMsg(2, $dl, "readTapesDB()");

  my $fname = $db_dir."/".$tapes_db;

  open FH, "<".$fname or return ("fail", "Could not read tapes db ".$fname);
  my @lines = <FH>;
  close FH;

  my $id = "";
  my $size = "";
  my $used = "";
  my $free = "";
  my $nfiles = "";
  my $full = "";
  my $line;

  my @db = ();
  my $i = 0;

  foreach $line (@lines) {

    Dada::logMsg(3, $dl, "readTapesDB: processing line: ".$line);
    ($id, $size, $used, $free, $nfiles, $full) = split(/ +/,$line);

    $db[$i] = {
      id => $id,
      size => $size,
      used => $used,
      free => $free,
      nfiles => int($nfiles),
      full => $full,
    };
    $i++

  }

  return @db;

}


#
# Update the tapes database with the specified information
#
sub updateTapesDB($$$$$$)
{
  my ($id, $size, $used, $free, $nfiles, $full) = @_;

  Dada::logMsg(3, $dl, "updateTapesDB($id, $size, $used, $free, $nfiles, $full)");

  my $fname = $db_dir."/".$tapes_db;
  my $expected_tape = "none";

  open FH, "<".$fname or return ("fail", "Could not read tapes db ".$fname);
  my @lines = <FH>;
  close FH;

  my $newline = $id."  ";

  $newline .= floatPad($size, 3, 2)."  ";
  $newline .= floatPad($used, 3, 2)."  ";
  $newline .= floatPad($free, 3, 2)."  ";
  $newline .= sprintf("%06d",$nfiles)."  ";
  $newline .= $full;

  #my $newline = $id."  ".sprintf("%05.2f",$size)."  ".sprintf("%05.2f",$used).
  #              "  ".sprintf("%05.2f",$free)."  ".$nfiles."       ".$full."\n";

  open FH, ">".$fname or return ("fail", "Could not write to tapes db ".$fname);

  # parse the file
  my $line = "";
  foreach $line (@lines) {

    if ($line =~ /^$id/) {
      Dada::logMsg(1, $dl, "DB update  ".$newline);
      print FH $newline."\n";
    } else {
      print FH $line;
    }
  
  }

  close FH;

  Dada::logMsg(3, $dl, "updateTapesDB() ok");

  return ("ok", "");

}

#
# update the Files DB
#
sub updateFilesDB($$$$) {

  my ($archive, $tape, $fsf, $size) = @_;

  Dada::logMsg(3, $dl, "updateFilesDB(".$archive.", ".$tape.", ".$fsf.", ".$size.")");

  my $fname = $db_dir."/".$files_db;

  my $date = Dada::getCurrentDadaTime();

  my $newline = $archive." ".$tape." ".$date." ".$fsf." ".$size;

  open FH, ">>".$fname or return ("fail", "Could not write to tapes db ".$fname);
  Dada::logMsg(2, $dl, "updateFilesDB: ".$newline);
  print FH $newline."\n";
  close FH;

  Dada::logMsg(3, $dl, "updateFilesDB() ok");

  return ("ok", "");
}


sub getTapeInfo($) {

  my ($id) = @_;

  Dada::logMsg(3, $dl, "getTapeInfo(".$id.")");

  my $fname = $db_dir."/".$tapes_db;

  open FH, "<".$fname or return ("fail", "Could not read tapes db ".$fname);
  my @lines = <FH>;
  close FH;

  my $size = -1;
  my $used = -1;
  my $free = -1;
  my $nfiles = 0;
  my $full = 0;

  # parse the file
  my $line = "";
  foreach $line (@lines) {
    
    chomp $line;

    if ($line =~ m/^$id/) {

      Dada::logMsg(3, $dl, "getTapeInfo: processing line: ".$line);
      ($id, $size, $used, $free, $nfiles, $full) = split(/ +/,$line);
      
    } else {

      Dada::logMsg(3, $dl, "getTapeInfo: ignoring line: ".$line);
      #ignore
    }
  }

  $nfiles = int($nfiles);

  if ($size eq -1) {
    return ("fail", "could not determine space from tapes.db");
  } else {

    Dada::logMsg(2, $dl, "getTapeInfo: id=".$id.", size=".$size.", used=".$used.", free=".$free.", nfiles=".$nfiles.", full=".$full);
    return ("ok", $id.":".$size.":".$used.":".$free.":".$nfiles.":".$full);
  }

}


#
# Checks the beam directory to see if it has been marked as archived
# and also checks the files.db to check if it has been recorded as
# archived. Returns an error on mismatch.
#
sub checkIfArchived($$$$$) {

  my ($user, $host, $dir, $obs, $beam) = @_;
  Dada::logMsg(3, $dl, "checkIfArchived(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  my $archived_db = 0;    # If the obs/beam is recorded in $files_db
  my $archived_disk = 0;  # If the obs/beam has been marked with shent.to.tape file 

  # Check the files.db to see if the beam is recorded there
  $cmd = "grep '".$obs."/".$beam."' ".$db_dir."/".$files_db;
  Dada::logMsg(3, $dl, "checkIfArchived: ".$cmd);
  my $grep_result = `$cmd`;
    
  # If the grep command failed, probably due to the beam not existing in the file
  if ($? != 0) {

    Dada::logMsg(3, $dl, "checkIfArchived: ".$obs."/".$beam." did not exist in ".$db_dir."/".$files_db);
    $archived_db = 0;

  } else {

    Dada::logMsg(3, $dl, "checkIfArchived: ".$obs."/".$beam." existed in ".$db_dir."/".$files_db);
    $archived_db = 1;

    # check there is only 1 entry in files.db
    my @lines = split(/\n/, $grep_result);
    if ($#lines != 0) {
      Dada::logMsg(0, $dl, "checkIfArchived: more than 1 entry for ".$obs."/".$beam." in ".$db_dir."/".$files_db);
      return("fail", $obs."/".$beam." had more than 1 entry in FILES database");
    } 
  }

  # Check the directory for a on.tape.type file
  $cmd = "ls -1 ".$dir."/".$obs."/".$beam."/on.tape.".$type;
  Dada::logMsg(3, $dl, "checkIfArchived: mysSytem(".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "checkIfArchived: localSshCommand() ".$result." ".$response);

  if ($result eq "ok") {
    $archived_disk = 1;
    Dada::logMsg(3, $dl, "checkIfArchived: ".$dir."/".$obs."/".$beam."/on.tape.".$type." existed");
  } else {
    Dada::logMsg(3, $dl, "checkIfArchived: ".$dir."/".$obs."/".$beam."/on.tape.".$type." did not exist");
    $archived_disk = 0;
  }

  if (($archived_disk == 0) && ($archived_db == 0)) {
    Dada::logMsg(2, $dl, "checkIfArchived: ".$obs."/".$beam." not archived");
    return ("ok", "not archived");
  } elsif (($archived_disk == 1) && ($archived_db == 1)) {
    Dada::logMsg(2, $dl, "checkIfArchived: ".$obs."/".$beam." archived");
    return ("ok", "archived");
  } else {
    Dada::logMsg(0, $dl, "checkIfArchived() FILES database does not match flagged files on disk");
    return ("fail", "FILES database does not match flagged files on disk");
  }
}


#
# Polls for the "quitdaemons" file in the control dir
#
sub controlThread($$) {

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl, "controlThread: thread starting");

  # poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    Dada::logMsg(3, $dl, "controlThread: Polling for ".$quit_file);
    sleep(1);
  }

  # signal threads to exit
  $quit_daemon = 1;
  $Dada::tapes::quit_daemon = 1;
  Dada::logMsg(1, $dl, "controlThread: quit signal detected");

  Dada::logMsg(2, $dl, "controlThread: Unlinking PID file ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(2, $dl, "controlThread: exiting");

}


#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $0." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  my $dir;

  sleep(3);
  print STDERR $0." : Exiting: ".Dada::getCurrentDadaTime(0)."\n";

}

sub usage() {
  print STDERR "Usage:\n";
  print STDERR $0." [swin|parkes]\n";
}

sub exit_script($) {

  my $result = "";
  my $response = "";
  
  Dada::logMsg(1, $dl, "exit_script: tapes::tapeRewind()"); 
  ($result, $response) = Dada::tapes::tapeRewind();
  Dada::logMsg(1, $dl, "exit_script: tapes::tapeRewind() ".$result." ".$response);

  my $val = shift;
  print STDERR "exit_script(".$val.")\n";
  $quit_daemon = 1;
  sleep(3);
  exit($val);

}


sub floatPad($$$) {

  my ($val, $n, $m) = @_;

  my $str = "";

  if (($val >= 10.00000) && ($val < 100.00000)) {
    $str = " ".sprintf("%".($n-1).".".$m."f", $val);
  } elsif ($val < 10.0000) {
    $str = "  ".sprintf("%".($n-2).".".$m."f", $val);
  } else {
    $str = sprintf("%".$n.".".$m."f", $val)
  }

  return $str;
}


#
# Estimate the archive size based on file size and number of files
#
sub tarSizeEst($$) 
{
  my ($nfiles, $files_size) = @_;

  # 512 bytes for header and up to 512 bytes padding for data
  my $tar_overhead_files = (1024 * $nfiles);

  # all archives are a multiple of 10240 bytes, add for max limit
  my $tar_overhead_archive = 10240;           

  # upper limit on archive size in bytes
  my $size_est = $files_size + $tar_overhead_files + $tar_overhead_archive;

  # Add 1 MB for good measure
  $size_est += (1024*1024);

  return $size_est;
}


#
# load the specified tape. unloads/ejects any existing tape
#
sub loadTape($) {

  (my $tape) = @_;

  Dada::logMsg(2, $dl, "loadTape(".$tape.")");
  
  my $result = "";
  my $response = "";
  my $string = "";

  if ($robot) {
    $string = "Changing to ".$tape;
  } else {
    $string = "Insert Tape:::".$tape;
    Dada::logMsg(1, $dl, $string);
  }

  Dada::logMsg(3, $dl, "loadTape: Dada::tapes::loadTapeGeneral(".$tape.")");
  ($result, $response) = Dada::tapes::loadTapeGeneral($tape);
  Dada::logMsg(3, $dl, "loadTape: Dada::tapes::loadTapeGeneral() ".$result." ".$response);

  if (($result eq "ok") && ($response eq $tape)) {
    $string = "Current Tape: ".$tape;
  } else {
    $string = "Failed to Load: ".$tape;
  }

  Dada::logMsg(1, $dl, "loadTape: sleeping 5 seconds for tape to get online");
  sleep (5);

  if ($string =~ m/^Current Tape/) {
    Dada::logMsg(2, $dl, "loadTape() ok ".$tape);

    # ensure compression is on
    my $cmd = "mt -f ".$dev." compression 1";
    ($result, $response) = Dada::mySystem($cmd);

    return ("ok", $tape);
  } else {
    Dada::logMsg(0, $dl, "loadTape() failed to load ".$tape);
    return ("fail", "failed to load ".$tape);
  }
}


sub markTapeFull($) {

  my ($tape_id) = @_;

  Dada::logMsg(2, $dl, "markTapeFull(".$tape_id.")");

  my $result = "";
  my $response = "";

  if ($tape_id eq "none") {
    Dada::logMsg(0, $dl, "markTapeFull: error tape_id [".$tape_id."] set to none");
    return ("fail", "current tape set to none");
  }

  ($result, $response) = getTapeInfo($tape_id);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "markTapeFull: getTapeInfo() failed: ".$response);
    return ("fail", "could not determine tape information from database for tape_id=".$tape_id);
  }
 
  # get the info frm the tapes.db 
  my  ($id, $size, $used, $free, $nfiles, $full) = split(/:/,$response);

  Dada::logMsg(3, $dl, "markTapeFull: updateTapesDB(".$id.", ".$size.", ".$used.", ".$free.", ".$nfiles.", 1");
  ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, 1);
  Dada::logMsg(3, $dl, "markTapeFull: updateTapesDB() ".$result." ".$response);

  Dada::logMsg(2, $dl, "markTapeFull() ".$result." ".$response);

  return ($result, $response);

}


#
# Seek to the File number, if none specified, seek to the EOD
#
sub seekToFile($$)  {
  
  my ($tape_id, $file_req) = @_;

  Dada::logMsg(3, $dl, "seekToFile(".$tape_id.", ".$file_req.")");

  my $result = "";
  my $response = "";
  my $filenum = 0;
  my $blocknum = 0;
  my $file_no_seek = 0;

  # Get the tape information from the tape database
  Dada::logMsg(3, $dl, "seekToFile: getTapeInfo(".$tape_id.")");
  ($result, $response) = getTapeInfo($tape_id);
  Dada::logMsg(3, $dl, "seekToFile: getTapeInfo() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "seekToFile: getTapeInfo() failed: ".$response);
    return ("fail", "getTapeInfo() failed: ".$response);
  }

  my ($id, $size, $used, $free, $nfiles, $full) = split(/:/,$response);

  # find out which file we are currently on

  # after a load this doesn't always appear to update immediately
  sleep(1);

  my $ntries = 5;
  while ($ntries > 0) {
    Dada::logMsg(3, $dl, "seekToFile: tapes::getTapeStatus()");
    ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
    Dada::logMsg(3, $dl, "seekToFile: tapes::getTapeStatus() ".$result." ".$filenum." ".$blocknum);
    if (($result ne "ok") || ($filenum eq -1) || ($blocknum eq -1)) {
      Dada::logMsg(1, $dl, "seekToFile: waiting 5 secs for mt [".$result." ".$filenum." ".$blocknum."]");
      sleep(5);
      $ntries--;
    } else {
      $ntries = -1;
    }
  } 

  if (($result ne "ok") || ($ntries == 0)) {
    Dada::logMsg(0, $dl, "seekToFile: tapes::getTapeStatus() failed: ".$result." ".$filenum." ".$blocknum);
    return ("fail", "tapes::getTapeStatus failed");
  }

  # If no file number was requested, seek to the end of the tape
  if ($file_req == -1) {
    $file_no_seek = $nfiles - $filenum;
  } else {
    $file_no_seek = $file_req - $filenum; 
  }

  Dada::logMsg(2, $dl, "seekToFile: file_no_seek=".$file_no_seek);

  if ($file_no_seek == 0) {
    Dada::logMsg(2, $dl, "seekToFile: no seeking required");

  } elsif ($file_no_seek > 0) {
    Dada::logMsg(3, $dl, "seekToFile: tapeFSF(".($nfiles - $filenum).")");
    ($result, $response) = Dada::tapes::tapeFSF($nfiles - $filenum);
    Dada::logMsg(3, $dl, "seekToFile: tapeFSF() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "seekToFile: tapeFSF failed: ".$response);
      return ("fail", "tapeFSF failed: ".$response);
    }

  } else {

    $file_no_seek *= -1;
    Dada::logMsg(3, $dl, "seekToFile: tapeBSFM(". $file_no_seek.")");
    ($result, $response) = Dada::tapes::tapeBSFM($file_no_seek);
    Dada::logMsg(3, $dl, "seekToFile: tapeBSFM() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "seekToFile: tapeBSFM failed: ".$response);
      return ("fail", "tapeBSFM failed: ".$response);
    }

  }

  # now check it all worked...
  Dada::logMsg(3, $dl, "seekToFile: tapes::getTapeStatus()");
  ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
  Dada::logMsg(3, $dl, "seekToFile: tapes::getTapeStatus() ".$result." ".$filenum." ".$blocknum);
    if (($result ne "ok") || ($filenum == -1) || ($blocknum == -1)) {
    Dada::logMsg(0, $dl, "seekToFile: tapes::getTapeStatus() failed: ".$result." ".$filenum." ".$blocknum);
    return ("fail", "tapes::getTapeStatus failed");
  } 
  
  if (($file_req == -1) && ($filenum == $nfiles)) {
    Dada::logMsg(3, $dl, "seekToFile() ok");
    return ("ok", "");
  } elsif ($file_req == $filenum) {
    Dada::logMsg(3, $dl, "seekToFile() ok");
    return ("ok", "");
  } else {
    Dada::logMsg(0, $dl, "seekToFile() seeking failed");
    return ("fail", "seeking failed");
  } 

  return ("ok", "");
}

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
  if (index($required_host, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$required_host.
                    ", not ".Dada::getHostMachineName());
  }

  if (($type ne "swin") && ($type ne "mopsr")) {
    return ("fail", "Error: package global type [".$type."] was not swin or mopsr");
  }

  if (! -f ($db_dir."/".$tapes_db)) {
    return ("fail", "tapes db file [".$db_dir."/".$tapes_db."] did not exist");
  }

  if (! -f ($db_dir."/".$files_db)) {
    return ("fail", "files db file [".$db_dir."/".$files_db."] did not exist");
  }

  # check all the destinations exist and are accessible
  my $i = 0;
  my $user = "";
  my $host = "";
  my $path = "";
  my $fullpath = "";
  my $result = "";
  my $response = "";

  for ($i=0; $i<$cfg{"NUM_".uc($type)."_DIRS"}; $i++) {
    if (!defined ($cfg{uc($type)."_DIR_".$i})) {
      return ("fail", "config file error for ".uc($type)."_DIR_".$i);
    }

    ($user, $host, $path) = split(/:/,$cfg{uc($type)."_DIR_".$i},3);
    $fullpath = $path."/archive/";
    ($result, $response) = Dada::mySystem("ls ".$fullpath);
    if ($result ne "ok") {
      return ("fail", "archival dir [".$fullpath."] was not accessable: ".$response);
    }
  } 

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}


END { }

1;  # return value from file

