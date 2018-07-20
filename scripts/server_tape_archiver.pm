package Dada::server_tape_archiver;

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
  @EXPORT_OK   = qw($dl $daemon_name $type $drive_id $robot $pid $ctrl_dir $ctrl_prefix %cfg);

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
our $ctrl_dir;
our $ctrl_prefix;
our $required_host;
our $ssh_prefix;
our $ssh_suffix;
our $local_fs;
our %cfg;

#
# non-exported package globals go here
#
our $quit_daemon : shared;
our $warn;
our $error;
our $dev;
our $ssh_opts;
our $current_tape;
our $db_dir;
our $db_user;
our $db_host;
our $use_bk;
our $bkid;
our $tapes_db;
our $files_db;
our $bookkeepr;

#
# initialize package globals
#
$dl = 2;
$daemon_name = 0;
$robot = 0;
$drive_id = 0;
$pid = "";
$ctrl_dir = "";
$ctrl_prefix = "";
$type = "";
$ssh_prefix = "";
$ssh_suffix = "";
$local_fs = 0;

#
# initialize other variables
#
$quit_daemon = 0;
$warn = "";
$error = "";
$dev = 0;
$ssh_opts = "-o BatchMode=yes";
$current_tape = "";
$db_dir = "";
$db_user = "";
$db_host = "";
$use_bk = 0;
$bkid = "";
$tapes_db = "";
$files_db = "";
$bookkeepr = "";

use constant TAPE_SIZE    => "750.00";


###############################################################################
#
# Package Methods
#

sub main() {

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
  $bookkeepr = $cfg{uc($type)."_BOOKKEEPR"};

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
  my @hosts = ();
  my @users = ();
  my @paths = ();
  my $user = "";
  my $host = "";
  my $path = "";
  my $path_pid = "";
  my $expected_tape = "";

  # setup the disks
  for ($i=0; $i<$cfg{"NUM_".uc($type)."_DIRS"}; $i++) {
    ($user, $host, $path) = split(/:/,$cfg{uc($type)."_DIR_".$i},3);
    $hosts[$i] = $host;
    $users[$i] = $user;
    $paths[$i] = $path;
  }

  $j = 0;

  Dada::daemonize($log_file, $pid_file);
  Dada::logMsg(0, $dl, "STARTING SCRIPT");
  setStatus("Scripting starting");

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
    setStatus("ERROR: no tapes available, label more tapes");
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

  setStatus("Current tape: ".$current_tape);

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

  my $next_beam_thread = 0;
  my $next_obs = "";
  my $next_beam = "";
  my $next_bytes = "";
  my $next_i = "";

  $i = 0;
  $j = 0;
  my $waiting = 0;
  my $disks_tried = 0;
  my $give_up = 0;
  my $counter = 0;
  my $try_to_archive = 0;

  ($result, $response) = Dada::tapes::tapeIsReady();
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "main: tape drive was not ready: ".$response);
    $quit_daemon = 1;
  }

  # For the first iteration get the next beam to tar
  Dada::logMsg(2, $dl, "main: getOldestBeamToTar()");
  ($obs, $beam, $bytes, $i) = getOldestBeamToTar(\@users, \@hosts, \@paths, $obs, $beam);
  Dada::logMsg(2, $dl, "main: getOldestBeamToTar() ".$obs."/".$beam." [bytes=".$bytes.", i=".$i."]");

  while (!$quit_daemon) {

    # if we had previously launched a thread to get the next beam, get the results from
    # that thread
    if ($next_beam_thread)
    {
      Dada::logMsg(2, $dl, "main: getOldestBeamToTar: joining next_beam_thread");
      ($obs, $beam, $bytes, $i) = $next_beam_thread->join();
      $next_beam_thread = 0;
    }

    # Now launch a thread to get the next oldest beam to tar
    Dada::logMsg(2, $dl, "main: NEXT getOldestBeamToTar(ignoring ".$obs." ".$beam.")");
    ($next_beam_thread) = threads->new(\&getOldestBeamToTar, \@users, \@hosts, \@paths, $obs, $beam);
    Dada::logMsg(2, $dl, "main: NEXT getOldestBeamToTar() ".$obs."/".$beam." [i=".$i."]");

    # look for a file sequentially in each of the @dirs
    $host = $hosts[$i];
    $user = $users[$i];
    $path = $paths[$i];
    $path_pid = $path."/archive/".$pid;

    # If we have one, write to tape
    if (($obs ne "none") && ($beam ne "none")) 
    {

      $disks_tried = 0;

      $try_to_archive = 1;

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
            setStatus("ERROR: could not load ".$expected_tape);
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
          Dada::logMsg(2, $dl, "main: tarBeam(".$user.", ".$host.", ".$path_pid.", ".$obs.", ".$beam.", ".$bytes.")");
          ($result, $response) = tarBeam($user, $host, $path_pid, $obs, $beam, $bytes);
          Dada::logMsg(2, $dl, "main: tarBeam() ".$result." ".$response);
          $waiting = 0;

          if ($result eq "ok") 
          {
            $try_to_archive = 0;
            # if the completion thread has been previously launched, join it
            if ($completed_thread ne 0) 
            {
              $completed_thread->join();
  	          $completed_thread = 0;
            }

            # launch the completion thread
            $completed_thread = threads->new(\&completed_thread, $user, $host, $path, $pid, $obs, $beam);

            # should no longer be necessary
            # if there is only 1 dir to check, wait for the completed thread
            #if ($cfg{"NUM_".uc($type)."_DIRS"} == 1)
            #{
            #  $completed_thread->join();
	          #  $completed_thread = 0;
            #}
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
              setStatus("Error: ".$response);
              Dada::logMsg(0, $dl, "main: tarBeam() failed: ".$response);
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
  
    } else {

      $disks_tried++;
      Dada::logMsg(2, $dl, "main: source [".$i."] had no beams ".$users[$i]."@".$hosts[$i].":".$paths[$i]."/archive/".$pid);

      # If we have cycled through all the disks and no files exist 
      if ($disks_tried > ($#hosts)) {

        Dada::logMsg(2, $dl, "main: tried [".$disks_tried." of ".($#hosts+1)."] disks, doing a long sleep");
        if ($disks_tried == ($#hosts + 1))
        {
          Dada::logMsg(1, $dl, "main: waiting for obs to archive");
          setStatus("Waiting for new");
        }
        $counter = 60;
        while (!$quit_daemon && ($counter>0)) {
          sleep 1;
          $counter--;
        }
      }

      # increment to the next disk
      sleep(1);
    }

  } # main loop

  # ensure tape is rewound
  Dada::logMsg(1, $dl, "Rewinding tape before exiting");
  ($result, $response) = Dada::tapes::tapeRewind();

  # rejoin threads
  $control_thread->join();

  if ($completed_thread ne 0) {
    $completed_thread->join();
  }

  if ($next_beam_thread ne 0) {
    $next_beam_thread->join();
  }

  setStatus("Script stopped");

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
sub getOldestBeamToTar(\@\@\@$$) 
{
  Dada::logMsg(3, $dl, "getOldestBeamToTar()");

  (my $uref, my $href, my $dref, my $obs_ignore, my $beam_ignore) = @_;

  my @users = @$uref;
  my @hosts = @$href;
  my @dirs  = @$dref;

  my $user = "";
  my $host = "";
  my $dir = "";
  my $path = "";
  my $obs = "none";
  my $beam = "none";
  my $subdir = "";
  my @subdirs = ();
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $oldest_obs = "none";
  my $oldest_beam = "none";
  my $oldest_bytes = 0;
  my $oldest_index = 0;
  my $obs_time = 0;
  my $oldest_time = 0;
  my $i = 0;
  my @t = ();
	my $line = "";
	my @lines = ();

  for ($i=0; $i<=$#dirs; $i++)
  {
    $user = $users[$i];
    $host = $hosts[$i];
    $dir = $dirs[$i];
    $path = $dirs[$i]."/archive/".$pid."/";

    $cmd = "cd ".$path."; lfs find . -maxdepth 2 -type l -name \"??\" | sort -r | tail -n 3";
    Dada::logMsg(3, $dl, "getOldestBeamToTar: ".$cmd);
    ($result, $response) = Dada::mySystem ($cmd);
    Dada::logMsg(3, $dl, "getOldestBeamToTar: ".$result." ".$response);

    #$cmd = "cd ".$path."; find -L . -ignore_readdir_race -mindepth 3 -maxdepth 3 -type f -name 'xfer.complete' ".
    #       "-printf '\%h\\n' | sort -r | tail -n 3";
    #Dada::logMsg(3, $dl, "getOldestBeamToTar: localSshCommand(".$user.", ".$host.", ".$cmd.")");
    #($result, $response) = localSshCommand($user, $host, $cmd);
    #Dada::logMsg(3, $dl, "getOldestBeamToTar: localSshCommand() ".$result." ".$response);

		# if the ssh command failed
		if ($result ne "ok") {
			Dada::logMsg(0, $dl, "getOldestBeamToTar: ssh cmd [".$user."@".$host."] failed: ".$cmd);
			Dada::logMsg(0, $dl, "getOldestBeamToTar: ssh response ".$response);
			sleep(1);

		# ssh worked
		} else {

			# find worked 
      if ($response ne "") {

			  @lines = split(/\n/, $response);
			  foreach $line ( @lines)
			  { 
          my @arr = split(/\//,$line);
          $obs = $arr[1];
          $beam = $arr[2];

          # check that the values returned were sensible
          Dada::logMsg(3, $dl, "getOldestBeamToTar: testing ".$obs."/".$beam);
          if (($obs =~ m/(\d\d\d\d)\-(\d\d)\-(\d\d)\-(\d\d):(\d\d):(\d\d)/) && ($beam =~ m/(\d\d)/)) {
            if (!(($obs eq $obs_ignore) && ($beam eq $beam_ignore)))
            {
            	Dada::logMsg(3, $dl, "getOldestBeamToTar: found not to be ignored ".$obs."/".$beam);
              if ($oldest_obs eq "none")
              {

                # check if this is already archived
                Dada::logMsg(3, $dl, "getOldestBeamToTar: checkIfArchived(".$user.", ".$host.", ".$path.", ".$obs.", ".$beam.")");
                ($result, $response) = checkIfArchived($user, $host, $path, $obs, $beam);
                Dada::logMsg(3, $dl, "getOldestBeamToTar: checkIfArchived() ".$result." ".$response);
                if (($result eq "ok") && ($response eq "not archived"))
                {
                  Dada::logMsg(2, $dl, "getOldestBeamToTar: found ".$user."@".$host.":".$path."/".$obs."/".$beam." ".$i);
                  $oldest_obs = $obs;
                  $oldest_beam = $beam;
                  $oldest_index = $i;
                }
                else
                {
                  Dada::logMsg(1, $dl, "getOldestBeamToTar: skipping ".$path."/".$obs."/".$beam." ".$i." as it fails archive test");
                }
              }
              # see if this is older than the obs
              else
              {
                @t = split(/-|:/, $obs);
                $obs_time = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]) + int($beam);
  
                @t = split(/-|:/, $oldest_obs);
                $oldest_time = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]) + int($oldest_beam);
  
                if ($obs_time < $oldest_time)
                {
                  # check if this is already archived
                  Dada::logMsg(3, $dl, "getOldestBeamToTar: checkIfArchived(".$user.", ".$host.", ".$path.", ".$obs.", ".$beam.")");
                  ($result, $response) = checkIfArchived($user, $host, $path, $obs, $beam);
                  Dada::logMsg(3, $dl, "getOldestBeamToTar: checkIfArchived() ".$result." ".$response);
                  if (($result eq "ok") && ($response eq "not archived"))
                  {
                    Dada::logMsg(2, $dl, "getOldestBeamToTar: found ".$user."@".$host.":".$path."/".$obs."/".$beam." ".$i);
                    $oldest_obs = $obs;
                    $oldest_beam = $beam;
                    $oldest_index = $i;
                  }
                  else
                  {
                    Dada::logMsg(1, $dl, "getOldestBeamToTar: skipping ".$path."/".$obs."/".$beam." ".$i." as it fails archive test");
                  }
                }
              }
            }
          } else {
            $obs = "none";
            $beam = "none";
            Dada::logMsg(0, $dl, "WARNING: getOldestBeamToTar: bad response from remote find: ".$response);
          }
        }
      # find failed
      } else {
        Dada::logMsg(2, $dl, "getOldestBeamToTar: could not find observations");
      } 
    } # if ssh worked
  } # foreach dir
  
  if ($oldest_obs ne "none")
  {
    Dada::logMsg(2, $dl, "getOldestBeamToTar: found ".$oldest_obs." ".$oldest_beam." [".$oldest_index."]");
    $user = $users[$oldest_index];
    $host = $hosts[$oldest_index];
    $dir  = $dirs[$oldest_index];
    $path = $dir."/archive/".$pid."/";

    ($result, $response) = getBeamSize($user, $host, $path."/".$oldest_obs."/".$oldest_beam);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "getOldestBeamToTar: getBeamSize failed for ".$user."@".$host.":".$path."/".$oldest_obs."/".$oldest_beam);
      $oldest_bytes = 0;
    }
    $oldest_bytes = $response;
  } 

  Dada::logMsg(2, $dl, "getOldestBeamToTar: returning ".$oldest_obs." ".$oldest_beam." ".$oldest_bytes." ".$oldest_index);
  return ($oldest_obs, $oldest_beam, $oldest_bytes, $oldest_index);
}


#
# tars the beam to the tape drive
#
sub tarBeam($$$$$$)
{
  my ($user, $host, $dir, $obs, $beam, $est_size_bytes) = @_;

  Dada::logMsg(2, $dl, "tarBeam: (".$user." , ".$host.", ".$dir.", ".$obs.", ".$beam.", ".$est_size_bytes.")");
  Dada::logMsg(1, $dl, "Archiving  ".$obs."/".$beam." from ".$host.":".$dir);

  my $cmd = "";
  my $result = "";
  my $response = "";

  # Check if this beam has already been archived
  Dada::logMsg(2, $dl, "tarBeam: checkIfArchived(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.")");
  ($result, $response) = checkIfArchived($user, $host, $dir, $obs, $beam);
  Dada::logMsg(2, $dl, "tarBeam: checkIfArchived() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarBeam: checkIfArchived failed: ".$response);
    return ("fail", "checkIfArchived failed: ".$response);
  }

  # If this beam has been archived, skip it 
  if ($response eq "archived") {
    Dada::logMsg(1, $dl, "Skipping archival of ".$obs."/".$beam.", already archived");
    return("ok", "beam already archived");
  }

  setStatus("Archiving ".$obs." ".$beam);
  my $tape = $current_tape;

  my $est_size_gbytes = ($est_size_bytes / (1000 * 1000 * 1000));

  # check if this beam will fit on the tape
  ($result, $response) = getTapeInfo($tape);
  if ($result ne "ok")
  {
    Dada::logMsg(0, $dl, "tarBeam: getTapeInfo() failed: ".$response);
    return ("fail", "could not determine tape information from database");
  }

  my ($id, $size, $used, $free, $nfiles, $full) = split(/:/,$response);

  Dada::logMsg(2, $dl, "tarBeam: ".$free." GB left on tape");
  Dada::logMsg(2, $dl, "tarBeam: size of this beam is estimated at ".$est_size_gbytes." GB");

  # if we estimate that there is not enough spce on the tape
  if ($free < $est_size_gbytes)
  {
    Dada::logMsg(0, $dl, "tarBeam: tape ".$tape." full. (".$free." < ".$est_size_gbytes.")");

    # Mark the current tape as full and load a new tape;
    Dada::logMsg(2, $dl, "tarBeam: updateTapesDB(".$id.", ".$size.", ".$used.", ".$free.", ".$nfiles);
    ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, 1);
    Dada::logMsg(2, $dl, "tarBeam: updateTapesDB() ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "tarBeam: updateTapesDB() failed: ".$response);
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
    Dada::logMsg(0, $dl, "tarBeam: WARNING: Tape out of position! ".
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

  # try to ensure that no nc is running on the port we want to use
  my $nc_ready = 0;
  my $tries = 10;
  my $port = 25128;
	my $remote_nc_thread = 0;

	if (!$local_fs)
	{
		while ( (!$nc_ready)  && ($tries > 0) && (!$quit_daemon) ) 
		{
			# This tries to connect to any type of service on specified port
			Dada::logMsg(2, $dl, "tarBeam: testing nc on ".$host.":".$port);
			if ($robot eq 0) {
				$cmd = "nc -z ".$host." ".$port." > /dev/null";
			} else {
				$cmd = "nc -zd ".$host." ".$port." > /dev/null";
			}
			($result, $response) = Dada::mySystem($cmd);

			# This is an error condition!
			if ($result eq "ok") 
			{
				Dada::logMsg(0, $dl, "tarBeam: something running on the NC port ".$port);
				if ($tries < 5) {
					Dada::logMsg(0, $dl, "tarBeam: trying to increment port number");
					$port += 1;
				} else {
					Dada::logMsg(0, $dl, "tarBeam: trying again, now that we have tested once");
				}
				$tries--;
				sleep(1);

			# the command failed, meaning there is nothing on that port
			} else {
				Dada::logMsg(2, $dl, "tarBeam: nc will be available on ".$host.":".$port);
				$nc_ready = 1;
			}
		}

		Dada::logMsg(2, $dl, "tarBeam: nc_thread(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.", ".$port.")");
		$remote_nc_thread = threads->new(\&nc_thread, $user, $host, $dir, $obs, $beam, $port);

		# Allow some time for this thread to start-up, ssh and launch tar + nc
		sleep(1);
	}

  my $localhost = Dada::getHostMachineName();

  my $tar_beam_result = "";
  my $tar_beam_response = "";
  my $bytes_written = 0;
  my $gbytes_written = 0;

  $tries=20;
  while ($tries > 0) 
  {
    # For historical reasons, HRE(robot==0) tapes are written with a slight 
    # different command to HRA (robot==1) tapes
		if ($local_fs)
		{
			if ($robot eq 0) {
				$cmd = "cd ".$dir."; tar -h -b 128 -c ".$obs."/".$beam." | dd of=".$dev." bs=64K";
			} else {
				$cmd = "cd ".$dir."; tar -h -b 128 -c ".$obs."/".$beam." | dd of=".$dev." bs=64k";
			}
		} else {
			if ($robot eq 0) {
				$cmd = "nc -w 10 ".$host." ".$port." | dd of=".$dev." bs=64K";
			} else {
				$cmd = "nc -d ".$host." ".$port." | dd of=".$dev." bs=64k";
			}
		}
				
    Dada::logMsg(2, $dl, "tarBeam: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);

    # fatal errors, give up straight away
    if ($response =~ m/Input\/output error/) 
    {
      Dada::logMsg(0, $dl, "tarBeam: fatal tape error: ".$response);
      $tar_beam_result = "fail";
      $tar_beam_response = "input output error";
      $tries = -1;
    }

    # the tape is unexpectedly full 
    elsif ($response =~ m/No space left on device/)
    {
      Dada::logMsg(0, $dl, "tarBeam: tape unexpectedly ".$tape." full");
      $tar_beam_result = "fail";
      $tar_beam_response = "not enough space on tape";
      $tries = -1;

      # Mark the current tape as full and load a new tape;
      Dada::logMsg(2, $dl, "tarBeam: updateTapesDB(".$id.", ".$size.", ".$used.", ".$free.", ".$nfiles);
      ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, 1);
      Dada::logMsg(2, $dl, "tarBeam: updateTapesDB() ".$result." ".$response);
      if ($result ne "ok") 
      {
        Dada::logMsg(0, $dl, "tarBeam: updateTapesDB() failed: ".$response);
        return ("fail", "Could not mark tape full");
      }
    }
    # non fatal errors
    elsif (($result ne "ok") || ($response =~ m/refused/) || ($response =~ m/^0\+0 records in/)) 
    {
      Dada::logMsg(2, $dl, "tarBeam: ".$result." ".$response);

      $tries--;
      $result = "fail";
      $response = "failed attempt at writing archive";
      if ($tries <= 10)
      {
        Dada::logMsg(1, $dl, "tarBeam: failed to write archive to tape, attempt ".(20-$tries)." of 20");
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
				if (!$local_fs)
				{
        	Dada::logMsg(0, $dl, "tarBeam: nc server not ready, sleeping 2 seconds");
				}
        $tries--;
        sleep(2);
      } 

      # if we did write something, but it didn't match bail!
      elsif ( ($est_size_gbytes - $gbytes_written) > 0.01)
      {
        $result = "fail";
        $response = "not enough data received by nc: ".$est_size_gbytes.
                    " - ".$gbytes_written." = ".($est_size_gbytes - $gbytes_written);
        Dada::logMsg(0, $dl, "tarBeam: ".$result." ".$response);
        $tar_beam_result = "fail";
        $tar_beam_response = "not enough data received";
        $tries = -1;
      } 
      else
      {
        $tar_beam_result = "ok";
        Dada::logMsg(2, $dl, "tarBeam: est_size ".sprintf("%7.4f GB", $est_size_gbytes).
                      ", size = ".sprintf("%7.4f GB", $gbytes_written));
        $tries = 0;
      }
    }
  }

  if ($tar_beam_result ne "ok")
  {
    Dada::logMsg(0, $dl, "tarBeam: failed to write archive to tape: ".$tar_beam_response);

		if (!$local_fs)
		{
	    Dada::logMsg(0, $dl, "tarBeam: attempting to clear the current nc server command");
  	  if ($robot eq 0) {
    	  $cmd = "nc -z ".$host." ".$port." > /dev/null";
	    } else {
  	    $cmd = "nc -zd ".$host." ".$port." > /dev/null";
    	}
    	Dada::logMsg(0, $dl, "tarBeam: ".$cmd);
    	($result, $response) = Dada::mySystem($cmd);
    	Dada::logMsg(0, $dl, "tarBeam: ".$result." ".$response);

    	$remote_nc_thread->detach();
		}
    return ("fail", $tar_beam_response);
  }
  else
  {
    $tar_beam_result = "fail";
  }
  
	if (!$local_fs)
	{
  	Dada::logMsg(3, $dl, "tarBeam: joining nc_thread()");
  	$result = $remote_nc_thread->join();
  	Dada::logMsg(2, $dl, "tarBeam: nc_thread() ".$result);

  	if ($result ne "ok") {
    	Dada::logMsg(0, $dl, "tarBeam: remote tar/nc thread failed");
    	return ("fail","Archiving failed");
  	}
	}

  # Now check that the File number has been incremented one. Sometimes the
  # file number is not incremented, which usually means an error...
  ($result, $filenum, $blocknum) = Dada::tapes::getTapeStatus();
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarBeam: getTapeStatus() failed.");
    return ("fail", "getTapeStatus() failed");
  }
  
  if (($blocknum ne 0) || ($filenum ne ($nfiles+1))) {
    Dada::logMsg(0, $dl, "tarBeam: positioning error after write: filenum=".$filenum.", blocknum=".$blocknum);
    return ("fail", "write failed to complete EOF correctly");
  }

  # remove the xfer.complete in the beam dir if it exists
  if ($type eq "parkes")
  {
    $cmd = "ls -1 ".$dir."/".$obs."/".$beam."/xfer.complete";
    Dada::logMsg(3, $dl, "tarBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $response) = localSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "tarBeam: localSshCommand() ".$result." ".$response);

    if ($result eq "ok") {
      $cmd = "rm -f ".$dir."/".$obs."/".$beam."/xfer.complete";
      Dada::logMsg(3, $dl, "tarBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
      ($result, $response) = localSshCommand($user, $host, $cmd);
      Dada::logMsg(3, $dl, "tarBeam: localSshCommand() ".$result." ".$response);
      if ($result ne "ok") {
        Dada::logMsg(2, $dl, "tarBeam: couldnt unlink ".$obs."/".$beam."/xfer.complete");
      }
    }
  }

  # else we wrote files to the TAPE in 1 archive and need to update the database files
  $used += $gbytes_written;
  $free -= $gbytes_written;
  $nfiles += 1;

  # If less than 100 MB left, mark tape as full
  if ($free < 0.1) {
    $full = 1;
  }

  # Log to the bookkeeper if defined
  if ($use_bk) {
    Dada::logMsg(3, $dl, "tarBeam: updateBookKeepr(".$obs."/".$beam."/".$obs."/.psrxml");
    ($result, $response) = updateBookKeepr($obs."/".$beam."/".$obs."/.psrxml",$bookkeepr, $id, ($nfiles-1));
    Dada::logMsg(3, $dl, "tarBeam: updateBookKeepr(): ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "tarBeam: updateBookKeepr() failed: ".$response);
      return("fail", "error ocurred when updating BookKeepr: ".$response);
    }
  }

  Dada::logMsg(3, $dl, "tarBeam: updatesTapesDB($id, $size, $used, $free, $nfiles, $full)");
  ($result, $response) = updateTapesDB($id, $size, $used, $free, $nfiles, $full);
  Dada::logMsg(3, $dl, "tarBeam: updatesTapesDB(): ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarBeam: updateTapesDB() failed: ".$response);
    return("fail", "error ocurred when updating tapes DB: ".$response);
  }

  Dada::logMsg(3, $dl, "tarBeam: updatesFilesDB(".$obs."/".$beam.", ".$id.", ".$gbytes_written.", ".($nfiles-1).")");
  ($result, $response) = updateFilesDB($obs."/".$beam, $id, $gbytes_written, ($nfiles-1));
  Dada::logMsg(3, $dl, "tarBeam: updatesFilesDB(): ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "tarBeam: updateFilesDB() failed: ".$response);
    return("fail", "error ocurred when updating filesDB: ".$response);
  }

  Dada::logMsg(3, $dl, "tarBeam: markSentToTape(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.")");
  ($result, $response) = markSentToTape($user, $host, $dir, $obs, $beam);
  Dada::logMsg(3, $dl, "tarBeam: markSentToTape(): ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(2, $dl, "tarBeam: markSentToTape failed: ".$response);
  }

  return ("ok",""); 

}


#
# get the size of a beam
#
sub getBeamSize($$$)
{
  my ($user, $host, $beam_dir) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";
  my $pipe = "";

  # Find the combined file size in bytes
  $cmd = "du -sLb ".$beam_dir;
  $pipe = "awk '{print \$1}'";
  Dada::logMsg(3, $dl, "getBeamSize: remoteSshCommand(".$user.", ".$host.", ".$cmd.", \"\", ".$pipe.")");
  ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd, "", $pipe);
  Dada::logMsg(3, $dl, "getBeamSize: remoteSshCommand() ".$result." ".$response);

  if ($result ne "ok") 
  {
    Dada::logMsg(0, $dl, "getBeamSize: ssh to ".$user."@".$host." failed: ".$response);
    return ("fail", "ssh failed");
  }
  elsif ($rval != 0) 
  {
    Dada::logMsg(0, $dl, "getBeamSize: ".$user."@".$host.":".$cmd." failed: ".$response);
    return ("fail", "du comand failed");
  }
  else
  {
    my $junk = "";
    my $size_bytes = "";
    ($size_bytes, $junk) = split(/ /,$response,2);

    # get the upper limit on the archive size
    Dada::logMsg(3, $dl, "getBeamSize: tarSizeEst(4, ".$size_bytes.")");
    my $size_est_bytes = tarSizeEst(4, $size_bytes);
    Dada::logMsg(3, $dl, "getBeamSize: tarSizeEst() ".$size_est_bytes." bytes");

    return ("ok", $size_est_bytes);
  }
}



#
# move a completed obs on the remote dir to the on_tape directory
#
sub moveCompletedBeam($$$$$$$) {

  my ($user, $host, $dir, $pid, $obs, $beam, $dest) = @_;
  Dada::logMsg(3, $dl, "moveCompletedBeam(".$user.", ".$host.", ".$dir.", ".$pid.", ".$obs.", ".$beam.", ".$dest.")");

  my $result = "";
  my $response = "";
  my $cmd = "";

  # ensure the remote directory is created
  $cmd = "mkdir -p ".$dir."/".$dest."/".$pid."/".$obs;
  Dada::logMsg(2, $dl, "moveCompletedBeam: ".$user."@".$host.":".$cmd);
  Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
  ($result, $response) = localSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand() ".$result." ".$response);

   # move the beam
  $cmd = "mv ".$dir."/archive/".$pid."/".$obs."/".$beam." ".$dir."/".$dest."/".$pid."/".$obs."/";
  Dada::logMsg(2, $dl, "moveCompletedBeam: ".$user."@".$host.":".$cmd);
  Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
  ($result, $response) = localSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "moveCompletedBeam: failed to move ".$obs."/".$beam." to ".$dest." dir: ".$response);
  } else {

    # if there are no other beams in the observation directory, then we can delete it
    # old, then we can remove it also
    $cmd = "find -L ".$dir."/archive/".$pid."/".$obs."/ -mindepth 1 -type d | wc -l";
    Dada::logMsg(2, $dl, "moveCompletedBeam: ".$user."@".$host.":".$cmd);
    Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $response) = localSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand() ".$result." ".$response);

    if (($result eq "ok") && ($response eq "0")) {
      # delete the remote directory
      $cmd = "rmdir ".$dir."/archive/".$pid."/".$obs;
      Dada::logMsg(2, $dl, "moveCompletedBeam: ".$user."@".$host.":".$cmd);
      Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
      ($result, $response) = localSshCommand($user, $host, $cmd);
      Dada::logMsg(3, $dl, "moveCompletedBeam: localSshCommand() ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsg(0, $dl, "moveCompletedBeam: could not delete ".$user."@".$host.":".$dir."/archive/".$pid."/".$obs.": ".$response);
      }
    }
  }

  return ($result, $response);

}


# Delete the beam from the specified location
#
sub deleteCompletedBeam($$$$$$) {

  my ($user, $host, $dir, $pid, $obs, $beam) = @_;
  Dada::logMsg(2, $dl, "deleteCompletedBeam(".$user.", ".$host.", ".$dir.", ".$pid." ".$obs.", ".$beam.")");

  my $result = "";
  my $response = "";
  my $cmd = "";

  # ensure the remote directory exists
  $cmd = "ls -1d ".$dir."/archive/".$pid."/".$obs."/".$beam;
  Dada::logMsg(2, $dl, "deleteCompletedBeam: ".$user."@".$host.":".$cmd);
  Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
  ($result, $response) = localSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "deleteCompletedBeam: ".$user."@".$host.":".$dir."/archive/".$pid."/".$obs."/".$beam." did not exist");
    
  } else {

    # delete the remote beam directory 
    $cmd = "rm -rf ".$dir."/archive/".$pid."/".$obs."/".$beam;
    Dada::logMsg(2, $dl, "deleteCompletedBeam: ".$user."@".$host.":".$cmd);
    Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $response) = localSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand() ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "deleteCompletedBeam: could not delete ".$user."@".$host.":".$dir.
                   "/archive/".$pid."/".$obs."/".$beam.": ".$response);
    }

    # if there are no other beams in the observation directory, then we can delete it
    $cmd = "find -L ".$dir."/archive/".$pid."/".$obs."/ -mindepth 1 -maxdepth 1 -type d | wc -l";
    Dada::logMsg(2, $dl, "deleteCompletedBeam: ".$user."@".$host.":".$cmd);
    Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $response) = localSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand() ".$result." ".$response);

    if (($result eq "ok") && ($response eq "0")) {

      # delete the remote directory
      $cmd = "rmdir ".$dir."/archive/".$pid."/".$obs;
      Dada::logMsg(2, $dl, "deleteCompletedBeam: ".$user."@".$host.":".$cmd);
      Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand(".$user.", ".$host.", ".$cmd.")");
      ($result, $response) = localSshCommand($user, $host, $cmd);
      Dada::logMsg(3, $dl, "deleteCompletedBeam: localSshCommand() ".$result." ".$response);

      if ($result ne "ok") {
        Dada::logMsg(0, $dl, "deleteCompletedBeam: could not delete ".$user."@".$host.":".$dir."/archive/".$pid."/".$obs.": ".$response);
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
  my $newbkid = "";;
  $bkid = "";

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
        my ($id, $size, $used, $free, $nfiles, $full, $newbkid) = split(/ +/,$line);
     
        if (int($full) == 1) {
          Dada::logMsg(3, $dl, "getExpectedTape: skipping tape ".$id.", marked full");
        } elsif ($free < 0.1) {
          Dada::logMsg(3, $dl, "getExpectedTape: skipping tape ".$id." only ".$free." GB left");
        } else {
          $expected_tape = $id;
          $bkid = $newbkid;
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
# Not sure what this does!
#
sub updateBookKeepr($$$$){
  my ($psrxmlfile,$bookkeepr,$id,$number) = @_;
  my $cmd="";
  my $result="";
  my $response="";
  my $psrxmlid="";
  if ($bkid=="0"){
    $cmd="book_create_tape ".$bookkeepr." ".$id." ".uc($type)." | & sed -e 's:.*<id>\\([^>]*\\)</id>.*:\\1:p' -e 'd'";
    ($result,$response) = Dada::mySystem($cmd);
    if($result ne "ok"){
      return ($result,$response);
    }
    $bkid=$response
  }
  $psrxmlid=`sed -e 's:.*<id>\\([^>]*\\)</id>.*:\\1:p' -e 'd' $psrxmlfile`;
  if($psrxmlid==""){
    return ("fail","<id> not set in the psrxml file");
  }
  $cmd="book_write_to_tape $bookkeepr $psrxmlid $bkid $number";
  ($result,$response) = Dada::mySystem($cmd);
  if($result ne "ok"){
          return ($result,$response);
  }

}

#
# Get a hash of the Tapes DB
#
sub readTapesDB() {


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
  # $newline .= "    $bkid";

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
  Dada::logMsg(3, $dl, "checkIfArchived: localSshCommand(".$user.", ".$host.", ".$cmd);
  ($result, $response) = localSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "checkIfArchived: localSshCommand() ".$result." ".$response);

  if ($result eq "ok") {
    $archived_disk = 1;
    Dada::logMsg(3, $dl, "checkIfArchived: ".$user."@".$host.":".$dir."/".$obs."/".$beam."/on.tape.".$type." existed");
  } else {
    Dada::logMsg(3, $dl, "checkIfArchived: ".$user."@".$host.":".$dir."/".$obs."/".$beam."/on.tape.".$type." did not exist");
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
sub tarSizeEst($$) {

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
# sets status on web interface
#

sub setStatus($) {

  (my $string) = @_;
  Dada::logMsg(2, $dl, "setStatus(".$string.")");

  my $file = $ctrl_prefix.".".$type.".state";
  my $result = "";
  my $response = "";
  my $remote_cmd = "";
  my $cmd = "";

  # Delete the existing state file
  $remote_cmd = "rm -f ".$ctrl_dir."/".$file;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;

  Dada::logMsg(3, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "setStatus: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "setStatus: could not delete the existing state file ".$file.": ".$response);
    return ("fail", "could not remove state file: ".$file);
  }

  # Write the new file
  $remote_cmd = "echo '".$string."' > ".$ctrl_dir."/".$file;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;

  Dada::logMsg(3, $dl, "setStatus: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "setStatus: ".$result." ".$response);

  return ("ok", "");

}

#
# Launches NC on the remote end
#
sub nc_thread($$$$$$) {

  my ($user, $host, $dir, $obs, $beam, $port) = @_;

  # Firstly, try to ensure that no NC is running on the port we want to use
  Dada::logMsg(3, $dl, "nc_thread(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.", ".$port.")"); 

  my $result = "";
  my $response = "";

  my $cmd = "ssh ".$ssh_opts." -l ".$user." ".$host." \"ls ".$dir." > /dev/null\"";
  Dada::logMsg(3, $dl, "nc_thread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "nc_thread: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "nc_thread: could not automount the /nfs/cluster/shrek??? raid disk");
  }

  $cmd = "ssh ".$ssh_opts." -l ".$user." ".$host." \"cd ".$dir."; ".
         "tar -h -b 128 -c ".$obs."/".$beam." | nc -l ".$port."\"";

  Dada::logMsg(2, $dl, "nc_thread: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "nc_thread: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "nc_thread: ".$cmd." failed: ".$response);
    return ("fail");
  }

  Dada::logMsg(3, $dl, "nc_thread() ok");
  return ("ok");
}

#
# Mark the obsevation/beam as on.tape.type, and also remotely in /nfs/archives
#
sub markSentToTape($$$$$) 
{
  my ($user, $host, $dir, $obs, $beam) = @_;

  Dada::logMsg(2, $dl, "markSentToTape(".$user.", ".$host.", ".$dir.", ".$obs.", ".$beam.")");

  my $cmd = "";
  my $remote_cmd = "";
  my $result = "";
  my $response = "";
  my $to_touch = "";

  if ($beam ne "") {
    $to_touch = $obs."/".$beam."/on.tape.".$type;
  } else {
    $to_touch = $obs."/on.tape.".$type;
  }

  $cmd = "touch ".$dir."/".$to_touch;
  Dada::logMsg(3, $dl, "markSentToTape:" .$cmd);
  ($result, $response) = localSshCommand($user, $host, $cmd);
  Dada::logMsg(3, $dl, "markSentToTape:" .$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "markSentToTape: could not touch ".$to_touch);
  }

  my $remote_dir = "";

  # determine if in results dir or old results dir
  $remote_cmd = "ls -1d ".$cfg{"SERVER_RESULTS_DIR"}."/".$obs;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;
  Dada::logMsg(3, $dl, "markSentToTape: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "markSentToTape: ".$result." ".$response);
  if (($result eq "ok") && ($response ne ""))
  {
    $remote_dir = $cfg{"SERVER_RESULTS_DIR"};
  }
  else
  {
    $remote_cmd = "ls -1d ".$cfg{"SERVER_OLD_RESULTS_DIR"}."/".$obs;
    $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;
    Dada::logMsg(3, $dl, "markSentToTape: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, "markSentToTape: ".$result." ".$response);
    if (($result eq "ok") && ($response ne ""))
    {
      $remote_dir = $cfg{"SERVER_OLD_RESULTS_DIR"};
    }
    else
    {
      Dada::logMsg(1, $dl, "markSentToTape: could determine remote directory");
      return ("fail", "could not determine remote dir");
    }
  }

  # if the beam is set, touch the beams on.tape.type
  $remote_cmd = "touch ".$remote_dir."/".$to_touch;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;

  Dada::logMsg(3, $dl, "markSentToTape: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "markSentToTape: ".$result." ".$response);
  if ($result ne "ok") {
    Dada::logMsg(1, $dl, "markSentToTape: could not touch remote on.tape.".$type);
  }

  Dada::logMsg(2, $dl, "markSentToTape() ".$result." ".$response);  
  return ($result, $response);
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
    Dada::logMsg(3, $dl, "loadTape: manualClearResponse()");
    ($result, $response) = manualClearResponse();
    Dada::logMsg(3, $dl, "loadTape: manualClearResponse() ".$result." ".$response);
    if ($result ne "ok") {
      Dada::logMsg(0, $dl, "loadTape: manualClearResponse() failed: ".$response);
      return ("fail", "could not clear response file in web interface");
    }
    $string = "Insert Tape:::".$tape;
  }

  Dada::logMsg(3, $dl, "loadTape: setStatus(".$string.")");
  ($result, $response) = setStatus($string);
  Dada::logMsg(3, $dl, "loadTape: setStatus() ".$result." ".$response);

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

  Dada::logMsg(3, $dl, "loadTape: setStatus(".$string.")");
  ($result, $response) = setStatus($string);
  Dada::logMsg(3, $dl, "loadTape: setStatus() ".$result." ".$response);

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


#
#
#
sub localSshCommand($$$) {

  my ($user, $host, $command) = @_;
  Dada::logMsg(3, $dl, "localSshCommand(".$user.", ".$host.", ".$command);

  my $cmd = "";
  my $result = "";
  my $response = "";

	if ($local_fs) {
  	$cmd = $command;
	} else {
  	$cmd = "ssh -x ".$ssh_opts." -l ".$user." ".$host." \"".$command."\"";
	}

  Dada::logMsg(3, $dl, "localSshCommand:" .$cmd);
  ($result,$response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "localSshCommand:" .$result." ".$response);

  Dada::logMsg(3, $dl, "localSshCommand() ".$result." ".$response);
  return ($result, $response);

}

#
# deletes the beam directory off of a host
#
sub completed_thread($$$$$$) 
{
  my ($user, $host, $dir, $pid, $obs, $beam) = @_;

  Dada::logMsg(3, $dl, "completed_thread(".$user.", ".$host.", ".$dir.", ".$pid.", ".$obs.", ".$beam.")");

  my $result = "";
  my $response = "";

  my $dest = "on_tape";
  Dada::logMsg(2, $dl, "completed_thread: moveCompletedBeam(".$user.", ".$host.", ".$dir.", ".$pid." ".$obs.", ".$beam.", ".$dest.")");
  ($result, $response) = moveCompletedBeam($user, $host, $dir, $pid, $obs, $beam, $dest);
  Dada::logMsg(3, $dl, "completed_thread: moveCompletedBeam() ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "completed_thread: moveCompletedBeam() failed: ".$response);
    setStatus("Error: could not move beam: ".$obs."/".$beam);
  }
  Dada::logMsg(3, $dl, "completed_thread()");
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

  if (! (($type eq "swin") || ($type eq "parkes"))) {
    return ("fail", "Error: package global type [".$type."] was not swin or parkes");
  }

  if (! -f ($db_dir."/".$tapes_db)) {
    return ("fail", "tapes db file [".$db_dir."/".$tapes_db."] did not exist");
  }

  if (! -f ($db_dir."/".$files_db)) {
    return ("fail", "files db file [".$db_dir."/".$files_db."] did not exist");
  }

  if ($ctrl_dir eq "") {
    return ("fail", "control dir not defined");
  }

  if ($ctrl_prefix eq "") {
    return ("fail", "control prefix not defined");
  }

  # check all the destinations exist and are accessible
  my $i = 0;
  my $user = "";
  my $host = "";
  my $path = "";
  my $fullpath = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  for ($i=0; $i<$cfg{"NUM_".uc($type)."_DIRS"}; $i++) {
    if (!defined ($cfg{uc($type)."_DIR_".$i})) {
      return ("fail", "config file error for ".uc($type)."_DIR_".$i);
    }

    ($user, $host, $path) = split(/:/,$cfg{uc($type)."_DIR_".$i},3);
    $fullpath = $path."/archive/".$pid;

    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, "ls", $fullpath);
    if (($result ne "ok") || ($rval ne 0)) {
      return ("fail", "remote dir was not accessable: ".$user."@".$host.":".$fullpath.": ".$result.":".$rval.":".$response);
    }
  } 

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  return ("ok", "");
}


#
# get the .response file from the web interface
#
sub manualGetResponse() {

  Dada::logMsg(2, $dl, "manualGetResponse()");

  my $file = $ctrl_prefix.".".$type.".response";
  my $remote_cmd = "";
  my $cmd = "";
  my $result = "";
  my $response = "";

  # Delete the existing state file
  $remote_cmd = "cat ".$ctrl_dir."/".$file;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;

  # Wait for a response to appear from the user
  Dada::logMsg(3, $dl, "manualGetResponse: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl,"manualGetResponse: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "manualGetResponse: could not read file: ".$file);
    return ("fail", "could not read file: ".$file);
  }

  Dada::logMsg(2, $dl, "manualGetResponse() ".$result." " .$response);
  return ($result, $response);

}

#
# delete the .response file from the web interface
#
sub manualClearResponse() {

  Dada::logMsg(2, $dl, "manualClearResponse()");

  my $result = "";
  my $response = "";
  my $file = $ctrl_prefix.".".$type.".response";
  my $remote_cmd = "";
  my $cmd = "";

  # Delete the existing state file
  $remote_cmd = "rm -f ".$ctrl_dir."/".$file;
  $cmd = $ssh_prefix.$remote_cmd.$ssh_suffix;

  # Wait for a response to appear from the user
  Dada::logMsg(3, $dl, "manualClearResponse: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, "manualClearResponse: ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsg(0, $dl, "manualClearResponse: could not delete the existing state file ".$file.": ".$response);
    return ("fail", "could not remove state file: ".$file);
  }

  Dada::logMsg(2, $dl, "manualClearResponse() ".$result." " .$response);
  return ("ok", "");
}

#
# creates a control file on the remote host specified, if the number of dirs is 1, then this is ignored
#
sub touchFile($$$)
{
  (my $user, my $host, my $file) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  if ($cfg{"NUM_".uc($type)."_DIRS"} == 1)
  {
    Dada::logMsg(2, $dl, "touchFile: NUM_".uc($type)."_DIRS == 1, ignoring");
    return ("ok", "ignored as only 1 DIR configured");
  }
  else
  {
    $cmd = "touch ".$file;
    Dada::logMsg(3, $dl, "touchFile: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "touchFile: remoteSshCommand() ".$result." ".$rval." ".$response);

    if ($result ne "ok") 
    {
      Dada::logMsgWarn($warn, "touchFile: ssh failed ".$response);
      return ("fail", "ssh to ".$user."@".$host." failed");
    } 
    elsif ($rval != 0) 
    {
      Dada::logMsgWarn($warn, "touchFile: could not touch ".$user."@".$host.":".$file.": ".$response);
      return ("fail", "could not touch remote file");
    } 
    else 
    {
      return ("ok", "");
    }
  }
}

sub clearFile($$$)
{
  (my $user, my $host, my $file) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = "";
  my $response = "";

  if ($cfg{"NUM_".uc($type)."_DIRS"} == 1)
  {
    Dada::logMsg(2, $dl, "clearFile: NUM_".uc($type)."_DIRS == 1, ignoring");
    return ("ok", "ignored as only 1 DIR configured");
  }
  else
  {
    $cmd = "rm -f ".$file;
    Dada::logMsg(3, $dl, "clearFile: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "clearFile: remoteSshCommand() ".$result." ".$rval." ".$response);

    if ($result ne "ok") 
    {
      Dada::logMsg(0, $dl, "clearFile: ssh to ".$user."@".$host." failed: ".$response);
      return ("fail", "ssh to ".$user."@".$host." failed");
    } 
    elsif ($rval != 0)
    {
      Dada::logMsg(0, $dl, "clearFile: could not unlink ".$user."@".$host.":".$file.": ".$response);
      return ("fail", "could not unlink remote file");
    } 
    else
    {
      return ("ok", "");
    }
  }
}

sub checkFile($$$)
{
  (my $user, my $host, my $file) = @_;

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  if ($cfg{"NUM_".uc($type)."_DIRS"} == 1)
  {
    Dada::logMsg(2, $dl, "checkFile: NUM_".uc($type)."_DIRS == 1, ignoring");
    return ("ok", "ignored as only 1 DIR configured");
  }
  else
  {
    $cmd = "ls -1 ".$file;
    Dada::logMsg(3, $dl, "checkFile: remoteSshCommand(".$user.", ".$host.", ".$cmd.")");
    ($result, $rval, $response) = Dada::remoteSshCommand($user, $host, $cmd);
    Dada::logMsg(3, $dl, "checkFile: remoteSshCommand() ".$result." ".$rval." ".$response);

    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "checkFile: ssh to ".$user."@".$host." failed: ".$response);
      return ("fail", "ssh to ".$user."@".$host." failed");
    }
    elsif ($rval != 0)
    {
      Dada::logMsg(2, $dl, "checkFile: ".$user."@".$host.":".$file." did not exist");
      return ("ok", "file did not exist");
    }
    else
    {
      Dada::logMsg(2, $dl, "checkFile: ".$user."@".$host.":".$file." did exist");
      return ("ok", "file did exist");
    }
  }
}


END { }

1;  # return value from file

