package Dada;

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use strict;
use vars qw($DADA_ROOT $VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);

require Exporter;
require AutoLoader;
@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &sendTelnetCommand
  &connectToMachine
  &setBinaryDir
  &getAPSRBinaryDir
  &getDADABinaryDir
  &getCurrentBinaryVersion
  &getDefaultBinaryVersion
  &setBinaryDir
  &getAvailableBinaryVersions
  &addToTime
  &getCurrentDadaTime
  &printDadaTime 
  &printTime
  &getPWCCState
  &waitForState
  &getLine
  &getLineSelect
  &readCFGFile
  &readCFGFileIntoHash
  &getDADA_ROOT
  &getDiskInfo
  &getRawDisk
  &getDBInfo
  &getAllDBInfo
  &getUnprocessedFiles
  &getServerResultsNFS
  &getServerArchiveNFS
  &constructRsyncURL
  &headerFormat
  &mySystem
  &killProcess
  &getAPSRConfigVariable
  &nexusLogOpen
  &nexusLogClose
  &nexusLogMessage
  &getHostMachineName
  &daemonize
  &commThread
);

$VERSION = '0.01';

use constant DEBUG_LEVEL         => 0;
use constant IMAGE_TYPE          => ".png";


my $DADA_ROOT = $ENV{'DADA_ROOT'};
if ($DADA_ROOT eq "") {
  print "DADA_ROOT environment variable not set\n";
  exit 5;
}

#
# Dada Config
#

my $DEFAULT_BINARY_DIR = $DADA_ROOT."/src";
my $DEFAULT_APSR_BINARY_DIR = $DADA_ROOT."/apsr/src";

sub getDADA_ROOT() {
  return $DADA_ROOT;
}

sub getDadaConfig() {
  my $config_file = getDadaCFGFile();
  my %config = readCFGFileIntoHash("Dada", $config_file, 0);
  return %config;
}


#
# Returns the name of the file that controls the daemons
#
sub getDaemonControlFile($$) {
  my ($module, $control_dir) = @_;
  return $control_dir."/quitdaemons";
}


sub getHostMachineName() {
 
  my $host = hostname;
  my $machine;
  my $domain;
  ($machine,$domain) = split(/\./,$host,2);
  return $machine;

}

sub getDadaCFGFile() {
  return $DADA_ROOT."/share/dada.cfg";
}

#
# Reads a configuration file in the typical DADA format and strips
# out comments and newlines. Returns as an associative array/hash
#
sub readCFGFile($$) {

  (my $module, my $fname) = @_;
  my %return_array;

  if (!(-f $fname)) {
    print "configuration file \"$fname\" did not exist\n";
    return -1;
  } else {
    open FH,"<$fname" or return -1;
    my @lines = <FH>;
    close FH;

    my @arr;
    my $line;

    foreach $line (@lines) {

      # get rid of newlines
      chomp $line;

      $line =~ s/#.*//;

      # skip blank lines
      if (length($line) > 0) {
        # strip comments
        @arr = split(/ +/,$line,2);
        if ((length(@arr[0]) > 0) && (length(@arr[1]) > 0)) {
          $return_array{$arr[0]} = $arr[1]; 
        }
      }
    }
  }
  return %return_array;
}

sub readRawTextFile($$) {

  (my $module, my $fname) = @_;
  my @return_array;

  if (!(-f $fname)) {
    print "text file $fname did not exist\n";
    return -1;
  } else {
    open FH,"<$fname" or return -1;
    my @lines = <FH>;
    close FH;

    my @arr;
    my $line;
                                                                                                       
    foreach $line (@lines) {
      chomp $line;
      push(@return_array,$line);
    }
    return @return_array;
  }
}


sub setBinaryDir($$) {
  my ($module, $dir) = @_;
  $DEFAULT_BINARY_DIR = $dir;
}

sub getAPSRBinaryDir() {
  return $DEFAULT_APSR_BINARY_DIR;
}

sub getDADABinaryDir() {
  return $DEFAULT_BINARY_DIR;
}


sub connectToMachine($$$;$) {
  
  (my $module, my $machine, my $port, my $ntries=10) = @_;

  my $tries = 0;
  my $handle = 0;

  # Connect a tcp sock with hostname/ip set
  $handle = new IO::Socket::INET (
    PeerAddr => $machine,
    PeerPort => $port,
    Proto => 'tcp',
  );

  # IF we couldn't do it /cry, sleep and try for 10 times...
  while ((!$handle) && ($tries < $ntries)) {

    if (DEBUG_LEVEL >= 1) {
      print "Attempting to connect to: ".$machine.":".$port."\n";
    }

    $handle = new IO::Socket::INET (
      PeerAddr => $machine,
      PeerPort => $port,
      Proto => 'tcp',
    );

    $tries++;
    sleep 1;
  }

  if ($handle) {
    # dont buffer IO
    $handle->autoflush(1);

    if (DEBUG_LEVEL >= 1) {
      print "Connected to ".$machine." on port ".$port."\n";
    }
    return $handle;

  } else {
    if (DEBUG_LEVEL >= 1) {
      print "Error: Could not connect to ".$machine." on port ".$port." : $!\n";
    }
    return 0;
  }


}

sub getLine($$) {

  (my $module, my $handle) = @_;

  my $line = <$handle>;
  $/ = "\n";
  chomp $line;
  $/ = "\r";
  chomp $line;
  $/ = "\n";

  return $line;

}

sub getLineSelect($$$) {

  (my $module, my $handle, my $timeout) = @_;

  my $read_set = new IO::Select($handle);  # create handle set for reading

  my ($readable_handles) = IO::Select->select($read_set, undef, undef, $timeout);

  my $line = "null";
  my $rh = 0;

  foreach $rh (@$readable_handles) {
    $line = Dada->getLine($rh);
  }

  return $line;

}


sub sendTelnetCommand($$$) {

  (my $module, my $handle, my $command) = @_;
  my @lines;
  my $response = "";
  my $result = "fail";
  my $endofmessage = "false";

  my $line;

  print $handle $command."\r\n";
  if (DEBUG_LEVEL >= 1) {
    print "Sending command: \"".$command."\"\n";
  }

  while ($endofmessage eq "false") {
    $line = <$handle>;

    # remove a leading "> " if it exists
    $line =~ s/^> //;

    $/ = "\n";
    chomp $line;
    $/ = "\r";
    chomp $line;
    $/ = "\n";
    if (($line eq "ok") || ($line eq "> ok")) {
      $endofmessage = "true";
      $result = "ok";
    } elsif (($line eq "fail") || ($line eq "> fail")) {
      $endofmessage = "true";
      $result = "fail";
    } else {
      if ($response eq "") {
        $response = $line;
      } else {
        $response = $response."\n".$line;
      }
    }
  }

  $/ = "\n";

  if (DEBUG_LEVEL >= 1) {
    print "Result:          \"".$result."\"\n";
    print "Response:        \"".$response."\"\n";
  }

  return ($result, $response);

}


sub getPWCCState($$) {

  (my $module, my $handle) = @_;
  my $result = "fail";
  my $response = "";

  ($result, $response) = sendTelnetCommand("Dada",$handle,"state");

  if ($result eq "ok") {
    #Parse the $response;
    my @array = split('\n',$response);
    my $line;
    my $temp;
    my @temp_array;

    my $pwcc;
    my @pwcs;
    foreach $line (@array) {
      if (index($line,"> ") == 0) {
        $line = substr($line,2);
      }

      # if the pwcc state
      if (index($line,"overall: ") == 0) {
        $pwcc = substr($line,9);
      } 

      # if a PWC
      if (index($line,"PWC_") == 0) {
        $temp = substr($line,4);
        @temp_array = split(": ",$temp);
        @pwcs[@temp_array[0]] = @temp_array[1];
      }
    }

    return ($pwcc,@pwcs);
  } else {
    return 0;
  }

}


sub addToTime($$$) {

  (my $module, my $time, my $toadd) = @_;

  my @t = split(/-|:/,$time);

  my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], $t[1], $t[0]);

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime ($unixtime + $toadd);

  $year += 1900;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);
                                                                                                               
  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

  
  #@t[5] += $toadd;
  #if (@t[5] >= 60) { @t[4]++; @t[5] -= 60; }
  #if (@t[4] >= 60) { @t[3]++; @t[4] -= 60; }
  #if (@t[3] >=24) {
  #  print "Stop working at midnight!!. Couldn't be bothered ";
  #  print "accounting for this case... :p\n";
  #  exit(0);
  #}
#
#  my $year = @t[0];
#  my $mon = sprintf("%02d", @t[1]);
#  my $mday = sprintf("%02d", @t[2]);
#  my $hour = sprintf("%02d", @t[3]);
#  my $min = sprintf("%02d", @t[4]);
#  my $sec = sprintf("%02d", @t[5]);

#  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

}

sub getCurrentDadaTime($$) {

  (my $module, my $secsToAdd) = @_;

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime (time+$secsToAdd);
  $year += 1900;
  $mon++;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);

  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

}

sub printDadaTime($$) {

  my ($module, $time) = @_;

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime ($time);
  $year += 1900;
  $mon++;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);

  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

}

sub printTime($$$) {

  my ($module, $time, $type) = @_;
                                                                                                                 
  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime ($time);

  if ($type eq "utc") {
    ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = gmtime ($time);
  }

  $year += 1900;
  $mon++;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);
                                                                                                                 
  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

}

sub printDadaLocalTime($$) {

  my ($module, $time) = @_;
  return printTime("Dada", $time, "local");

}

sub printDadaUTCTime($$) {

  my ($module, $time) = @_;
  return printTime("Dada", $time, "utc");

}


sub waitForState($$$$) {
                                                                                
  (my $module, my $stateString, my $handle, my $Twait) = @_;
                                                                                
  my $pwcc;
  my $pwc;
  my @pwcs;
  my $myready = "no";
  my $counter = $Twait;
  my $i=0;

  while (($myready eq "no") && ($counter > 0)) {

    if ($counter == $Twait) {
      ; 
    } elsif ($counter == ($Twait-1)) {
      if (DEBUG_LEVEL >= 1) { print STDERR "Waiting for $stateString."; }
    } else {
      if (DEBUG_LEVEL >= 1) { print STDERR "."; }
    }
                                                                                
    $myready = "yes";
                                                                                
    ($pwcc, @pwcs) = getPWCCState("Dada",$handle);
                                                                                
    if ($pwcc ne $stateString) {
      if (DEBUG_LEVEL >= 1){
        print "Waiting for PWC Controller to transition to ".$stateString."\n";
      }
      $myready = "no";
    }
                                                                                
    for ($i=0; $i<=$#pwcs;$i++) {
      $pwc = @pwcs[$i];
      if ($pwc ne $stateString) {
        if (DEBUG_LEVEL >= 1) {
          print "Waiting for PWC_".$i." to transition to ".$stateString."\n";
        }
        $myready = "no";
      }
    }

    sleep 1;
    $counter--;
  }
  if (($counter+1) != $Twait) {
    if (DEBUG_LEVEL >= 1) { print STDERR "\n"; }
  }

  if ($myready eq "yes") {
    return 0;
  } else {
    return -1;
  }
                                                                                
}

sub runSanityChecks() {

  my $dir = $DADA_ROOT."/bin/stable";
  if (!-d $dir) {
    print STDERR "Stable binary directory \"".$dir."\" did not exist\n";
    return -1; 
  }

  $dir = $DADA_ROOT."/bin/current";
  if (!-d $dir) {
    print STDERR "Currentbinary directory \"".$dir."\" did not exist\n";
    return -1;
  }

  # check that all the required binary files exist in current

  # check that the rawdata directory is writable

  # check that the milli archive directory is writeable

  # check that the the macro archive directory is writeable

}


# gets the current binary version as defined by the what
# the # $DADA_ROOT/bin/current symbolic link points to
# did not exist, report an error, and leave as is...
sub getCurrentBinaryVersion() {

  # my $bin_version_symlink = $DADA_ROOT."/bin/current";
  # if (!-d $bin_version_symlink) {
  #   return ("fail","The current binary version symbolic link \"".$bin_version_symlink."\"did not exist");
  # }
  # my $bin_version = $DADA_ROOT."/bin/".readlink($bin_version_symlink);
  # if (!-d ($bin_version)) {
  #   return ("fail","The linked current binary version \"".$bin_version."\" did not exist");
  # } 
  my $bin_version = $DADA_ROOT."/bin";
  return ("ok",$bin_version);
}

sub getDefaultBinaryVersion() {
  return $DADA_ROOT."/bin/stable";
}

sub setBinaryDir($$) {

  (my $module, my $dir) = @_;
  
  my $result;
  my $response;
  ($result, $response) = getAvailableBinaryVersions();
  if ($result ne "ok") {  
    return ("fail","The specified directory was not a valid binary directory");
  }

  my $version;
  my @versions = split(/\n/,$response);
  my $legal = "false";

  foreach $version (@versions) {
    if ($dir eq $version) {
      $legal = "true";
    }
  }
  if ($legal eq "true") {
    chdir($DADA_ROOT."/bin");
    unlink("./current");
    if ($? != 0) {
      return ("fail", "Could not unlink the \"current\" binary directory");
    }
    symlink($dir,"current");
    if ($? != 0) {
      return ("fail", "Could not link specified directory to the \"current\" binary directory");
    }
  } else {
    return ("fail","The specified directory was not a valid binary directory");
  }
  return ("ok","");

}


sub getAvailableBinaryVersions() {

  my @required_bins = qw(apsr_udpdb dada_db dada_dbdisk dspsr);
  my @valid_binary_dirs = qw();
  my $binary_dir = $DADA_ROOT."/bin";

  if (!(-d $binary_dir)) {
    return ("fail", "Binary directory did not exist: ".$binary_dir);
  } else {

    opendir(DIR,$binary_dir);
    my @dirs = grep { !/^\./ } readdir(DIR);
    closedir DIR;

    my $file;
    my @files;
    my $dir;
    my $dirString = "";

    # For each prospective binary dir, check that it contains the required
    # binary files...
    foreach $dir (@dirs) {

      my $fulldir = $binary_dir."/".$dir;

      if (-d $fulldir) {

        my $valid_binary_dir = "true";
        opendir(DIR,$fulldir);
        @files = grep { !/^\./ } readdir(DIR);
        closedir DIR;

        my $binary;

        foreach $binary (@required_bins) {
          if (!(in_array($binary, @files))) {
            $valid_binary_dir = "false";
          }
        }

        if ($valid_binary_dir eq "true") {
          push(@valid_binary_dirs, $dir);
        }
      }
    }
    if ($#valid_binary_dirs < 1) {
      return ("fail", "No valid binary dirs existed");
    } else {
      foreach $dir (@valid_binary_dirs) {
        $dirString = $dirString.$dir."\n";
      }
    }
    chomp $dirString;
    return ("ok", $dirString);
  }
}

sub in_array() {
  my $val = shift(@_);
  my $elem;
                                                                                                                          
  foreach $elem(@_) {
    if($val eq $elem) {
      return 1;
    }
  }
  return 0;
}

sub getDiskInfo($$) {
  
  (my $module, my $dir) = @_;

  my $dfresult = `df $dir -h 2>&1`;
  if ($? != 0) {
    chomp($dfresult);
    return ("fail",$dfresult);
  } else {
    my $list = `echo "$dfresult" | awk '{print \$2,\$3,\$4,\$5}'`;
    my @values = split(/\n/,$list);
    return ("ok",@values[1]);
  }
}

sub getRawDisk($$) {

  (my $module, my $dir) = @_;

  my $dfresult = `df $dir -B 1048576 2>&1`;
  if ($? != 0) {
    chomp($dfresult);
    return ("fail",$dfresult);
  } else {
    my $list = `echo "$dfresult" | awk '{print \$2,\$3,\$4}'`;
    my @values = split(/\n/,$list);
    return ("ok",@values[1]);
  }

}

sub getUnprocessedFiles($$) {

  (my $module, my $dir) = @_;

  my $duresult = `du -sB 1048576 $dir | awk '{print \$1}' `;
  chomp($duresult);
  if ($? != 0) {
    return ("fail", $duresult);
  } else {
    return ("ok", $duresult);
  }

}

sub getDBInfo($$) {

  my ($module, $key) = @_;

  my $bindir = getCurrentBinaryVersion();
  my $cmd = $bindir."/dada_dbmetric -k ".$key;
  print $cmd."\n";
  my $result = `$cmd 2>&1`;
  chomp $result;
  if ($? != 0) {
    return ("fail","Could not connect to data block");
  } else {
    return ("ok",$result);
  }

}

sub getAllDBInfo($$) {

  my ($module, $key_string) = @_;

  my @keys = split(/ /,$key_string);
  my $key;

  my $bindir = getCurrentBinaryVersion();
  my $cmd;
  my $db_info;
  my $result = "ok";
  my $response = "";

  foreach $key (@keys) {
    $cmd = $bindir."/dada_dbmetric -k ".$key." 2>&1 | awk -F, '{print \$1, \$2}'";
    $db_info = `$cmd`; 
    chomp $db_info;
    if ($? != 0) {
      $result = "fail";
      $response .= "Could not connect to DB ".$key." ";
    }
    $response .= $db_info." ";

  }

  return ($result, $response);  

}



sub getXferInfo() {

  my $bindir = getCurrentBinaryVersion();
  my $cmd = $bindir."/dada_dbxferinfo";
  my $result = `$cmd 2>&1`;
  chomp $result;
  if ($? != 0) {
    return ("fail","Could not connect to data block");
  } else {
    return ("ok",$result);
  }

}

sub getLoadInfo() {

  my $one;
  my $five;
  my $fifteen;
  ($one,$five,$fifteen) = (`uptime` =~ /(\d+\.\d+)/g);
  return ("ok", $one.",".$five.",".$fifteen);

}

sub sendErrorToServer($$$$){
  (my $module, my $host, my $script, my $message) = @_;
}


#
# Logs a message to the nexus machine on the standard logging port.
# The type and level dictate where the log message will be recorded
#
sub nexusLogOpen($$$) {

  my ($module, $host, $port) = @_;

  my $handle = connectToMachine("Dada", $host, $port); 
  # So output goes their straight away
  if (!$handle) {
    print STDERR "Error: $0 could not connect to ".$host.":".$port."\n";
    return 0;
  } else {
    return $handle;
  }

}

sub nexusLogClose($$) { 

  (my $module, my $handle) = @_;
  if ($handle) {
    $handle->close();
  }

}

sub nexusLogMessage($$$$$$$) {

  (my $module, my $handle, my $timestamp, my $type, my $level, my $source, my $message) = @_;
  if ($handle) {
    print $handle $timestamp."|".$type."|".$level."|".$source."|".$message."\r\n";
  }
  return $?;

}

sub getServerArchiveNFS() {
  return "/nfs/archives";
}

sub getServerResultsNFS() {
  return "/nfs/results";
}

sub constructRsyncURL($$$$) {

  my ($module, $user, $host, $dir) = @_;
  return $user."@".$host.":".$dir;

}


sub headerFormat($$$) {

  (my $module, my $key, my $string) = @_;

  my $pad = 20 - length($key);
  my $header_string = $key;
  my $i=0;
  for ($i=0;$i<$pad;$i++) {
    $header_string .= " ";
  }
  $header_string .= $string;

  return $header_string;
}

sub mySystem($$$) {

  (my $module, my $cmd, my $background=0) = @_;

  my $rVal = 0;
  my $result = "ok";
  my $response = "";
  my $realcmd = $cmd." 2>&1";

  if ($background) { $realcmd .= " &"; }

#  if (DEBUG_LEVEL >= 1) {
    #print "$realcmd\n";
#  }
  $response = `$realcmd`;
  $rVal = $?;
  $/ = "\n";
  chomp $response;

  #print $response;

  # If the command failed
  if ($rVal != 0) {
    $result = "fail"
  }

  return ($result,$response);

}


sub killProcess($$) {

  (my $module, my $pname) = @_;

  my $cmd = "ps axu | grep \"".$pname."\" | grep -v grep";
  my $pids_to_kill = `$cmd`;
  # If the process was running

  if ($? == 0) {

    $cmd = $cmd." | awk '{print \$2}'";
    $pids_to_kill = `$cmd`;
    my @arr = split(/\n/,$pids_to_kill);
    my $i=0;

    for ($i=0;$i<=$#arr;$i++) {
      my $pid = @arr[$i];
      $cmd = "kill -TERM ".$pid;
      system($cmd);
    }

    if ($? == 0) {
      return ("ok", "process killed");
    } else {

      # Now try with a KILL signal
      for ($i=0;$i<=$#arr;$i++) {
        my $pid = @arr[$i];
        $cmd = "kill -KILL".$pid;
        system($cmd);
      }
      if ($? == 0) {
        return ("ok", "required -KILL signal");
      } else {
        return ("fail", "process not killed");
      }
    }

  } else {
    return ("ok", "process did not exist");
  }
}

#
# Reads a configuration file in the typical DADA format and strips
# out comments and newlines. Returns as an associative array/hash
#
sub readCFGFileIntoHash($$$) {

  (my $module, my $fname, my $raw) = @_;

  my %values = ();

  if (!(-f $fname)) {
    print "configuration file \"$fname\" did not exist\n";
    return -1;
  } else {
    open FH,"<$fname" or return -1;
    my @lines = <FH>;
    close FH;

    my @arr;
    my $line;

    foreach $line (@lines) {

      # get rid of newlines
      chomp $line;

      $line =~ s/#(.)*//;

      # skip blank lines
      if (length($line) > 0) {
        # strip comments

        if ($raw == 1) {
          @arr = split(/ /,$line,2);
        } else {

          $line =~ s/#(.)*$//;
          @arr = split(/ +/,$line,2);
          $arr[1] =~ s/^\s*//;
          $arr[1] =~ s/\s+$//;
        }
        if ((length(@arr[0]) > 0) && (length(@arr[1]) > 0)) {

          $values{$arr[0]} = $arr[1];

        }
      }
    }
  }
  return %values;
}

sub daemonize($$$) {
               
  my ($module, $logfile, $pidfile) = @_;
                                                                                
  my $pid = fork;
                                                                                
  if ($pid) {
    # Add the childs PID to the control directory
    open FH,">$pidfile";
    print FH $pid."\n";
    close FH;
    exit;
  }
                                                                                
  POSIX::setsid() || die "Cannot detach from controlling process\n";
                                                                                
  chdir '/';
  umask 0;
                                                                                
  open(STDIN, "+>/dev/null") or die "Could not redirect STDOUT to /dev/null";
                                                                                
  if (my $stdout_file = $logfile) {
     open(STDOUT, ">>".$stdout_file) or die "Could not redirect STDOUT to $stdout_file : $!";
  }
  $| = 1;
  open(STDERR, ">&STDOUT") or die "Failed to re-open STDERR to STDOUT";
                                                                                
}


__END__
