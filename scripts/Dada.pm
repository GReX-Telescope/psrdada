package Dada;

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Handle;
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use strict;
use vars qw($DADA_ROOT $VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);

BEGIN {

  require Exporter;
  our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);

  require AutoLoader;

  $VERSION = '1.00';

  @ISA         = qw(Exporter AutoLoader);
  @EXPORT      = qw(&sendTelnetCommand &connectToMachine &getDADABinaryDir &getCurrentBinaryVersion &getDefaultBinaryVersion &setBinaryDir &getAvailableBinaryVersions &addToTime &getUnixTimeUTC &getCurrentDadaTime &printDadaTime &printTime &getPWCCState &printPWCCState &waitForState &getLine &getLines &parseCFGLines &readCFGFile &readCFGFileIntoHash &getDADA_ROOT &getDiskInfo &getRawDisk &getDBInfo &getAllDBInfo &getLoad &getUnprocessedFiles &getServerResultsNFS &getServerArchiveNFS &constructRsyncURL &headerFormat &mySystem &killProcess &getAPSRConfigVariable &nexusLogOpen &nexusLogClose &nexusLogMessage &getHostMachineName &daemonize &commThread &logMsg &logMsgWarn &remoteSshCommand &headerToHash &daemonBaseName &getProjectGroups &processHeader &getDM &getPeriod &checkScriptIsUnique &getObsDestinations &removeFiles);
  %EXPORT_TAGS = ( );
  @EXPORT_OK   = ( );

}

use constant DEBUG_LEVEL         => 0;
use constant IMAGE_TYPE          => ".png";
use constant PWCC_FATAL_ERROR    => -4;
use constant PWCC_HARD_ERROR     => -3;
use constant PWCC_SOFT_ERROR     => -2;


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
  my %config = readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub array_unique
{
  my @list = @_;
  my %finalList;
  foreach(@list)
  {
    $finalList{$_} = 1; # delete double values
  }
  return (keys(%finalList));
}

#
# Returns the name of the file that controls the daemons
#
sub getDaemonControlFile($) {
  my ($control_dir) = @_;
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
# Parse the lines in the array reference and place
# them in an hash of key:value pairs
#
sub parseCFGLines(\@) {
  
  (my $lines_ref) = @_;

  my @lines = @$lines_ref;

  my @arr;
  my $line;
  my %hash = ();

  foreach $line (@lines) {

    # get rid of newlines
    chomp $line;

    $line =~ s/#.*//;
    $line =~ s/ +$//;

    # skip blank lines
    if (length($line) > 0) {
      # strip comments
      @arr = split(/ +/,$line,2);
      if ((length(@arr[0]) > 0) && (length(@arr[1]) > 0)) {
        $hash{$arr[0]} = $arr[1];
      }
    }
  }
  return %hash;
}

#
# Reads a configuration file in the typical DADA format and strips
# out comments and newlines. Returns as an associative array/hash
#
sub readCFGFile($) {

  (my $fname) = @_;
  my %return_array;
  my @lines = ();

  if (!(-f $fname)) {
    print "configuration file \"$fname\" did not exist\n";
    return -1;
  } else {
    open FH,"<$fname" or return -1;
    @lines = <FH>;
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

sub readRawTextFile($) {

  (my $fname) = @_;
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


sub getDADABinaryDir() {
  return $DEFAULT_BINARY_DIR;
}


###############################################################################
# 
# Open a TCP socket to the specified host, port and debug level for messages
#
sub openSocket($$$;$) 
{

  (my $host, my $port, my $dl, my $attempts=10) = @_;

  my $sock = 0;
  my $tries = 0;

  # try to open the socket
  while ((!$sock) && ($tries < $attempts)) 
  {

    $sock = new IO::Socket::INET (
      PeerAddr => $host,
      PeerPort => $port,
      Proto => 'tcp',
    );

    if (!$sock)
    {
      $tries++;
      sleep 1;
    }
  }

  if ($sock) {
    
    # dont buffer IO
    $sock->autoflush(1);

    Dada::logMsg(1, $dl, "Connected to ".$host.":".$port);
    return $sock;
  
  } else {
    Dada::logMsg(0, $dl, "ERROR : Could not connect to ".$host.":".$port." ".$!);
    return 0;
  }
}


###############################################################################
# 
# Send a 'telnet' style command and return the response
#
sub telnetCmd($$$) {
    
  my ($sock, $command, $dl) = @_;
  my @lines;
  my $response = "";
  my $result = "fail";
  my $eof = 0;
    
  my $line;
    
  print $sock $command."\r\n";
  Dada::logMsg(1, $dl, "Dada::telnetCmd: sent ".$command);

  while (!$eof)
  {
  
    # wait no more than 5 seconds for a reponse  
    $line = <$sock>;
    
    # remove a leading "> " if it exists
    $line =~ s/^> //;
  
    $/ = "\n";
    chomp $line;
    $/ = "\r";
    chomp $line;
    $/ = "\n";

    if (($line eq "ok") || ($line eq "> ok")) {
      $eof = 1;
      $result = "ok";
    } elsif (($line eq "fail") || ($line eq "> fail")) {
      $eof = 1;
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

  Dada::logMsg(1, $dl, "Dada::telnetCmd: recv ".$result." ".$response);

  return ($result, $response);

} 


sub connectToMachine($$;$) {
  
  (my $machine, my $port, my $ntries=10) = @_;

  my $tries = 0;
  my $handle = 0;

  # Connect a tcp sock with hostname/ip set
  $handle = new IO::Socket::INET (
    PeerAddr => $machine,
    PeerPort => $port,
    Proto => 'tcp',
    Timeout => 1,
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
      Timeout => 1,
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

sub getLine($) {

  (my $handle) = @_;

  my $line = <$handle>;
  $/ = "\n";
  chomp $line;
  $/ = "\r";
  chomp $line;
  $/ = "\n";

  return $line;

}

sub getLines($) {

  (my $handle) = @_;

  my @lines = $handle->getlines();
  my $i=0;
  for ($i=0; $i<=$#lines; $i++) {
    $/ = "\n";
    chomp $lines[$i];
    $/ = "\r";
    chomp $lines[$i];
    $/ = "\n";
  }
  return @lines;
}

sub sendTelnetCommand($$;$) {

  (my $handle, my $command, my $timeout=1) = @_;

  my $output = "";
  my @lines = ();
  my $res = "";
  my $result = "fail";
  my $response = "";
  my $eod = 0;

  my $line;

  print $handle $command."\r\n";
  if (DEBUG_LEVEL >= 1) {
    print "Sending command: \"".$command."\"\n";
  }

  # my $ofh = select $handle;
  # $| = 1;
  # select $ofh;

  while ($handle && !$eod) {

    ($res, $output) = getLinesSelect($handle, 10);

    if ($res eq "ok") {

      @lines = split(/\n/, $output);

      foreach $line ( @lines ) {
        # remove a leading "> " if it exists
        $line =~ s/^> //;

        # remove a trailing \r\n if it exists
        $line =~ s/\r\n$//;
        $line =~ s/\n$//;

        if (($line eq "ok") || ($line eq "fail")) {
          $eod = 1;
          $result = $line;
        } else {
          if ($response eq "") {
            $response = $line;
          } else {
            if ($line ne "") {
              $response = $response."\n".$line;
            } 
          }
        }
      }
    } else {
      $eod = 1;
      $result = "fail";
      $response = $line;
    }
  }

  if (DEBUG_LEVEL >= 1) {
    print "Result:          \"".$result."\"\n";
    print "Response:        \"".$response."\"\n";
  }

  return ($result, $response);

}


sub getLinesSelect($$) {

  (my $handle, my $timeout) = @_;

  my $blocking_state = $handle->blocking;
  my $irs_state =  $/;
  my $read_set = 0;
  my $rh = 0;
  my $line = "";
  my $result = "fail";
  my $response = "timed out";
  my $poll_handle = 1;

  # set socket to non blocking
  $handle->blocking(0);

  $/ = "\r\n";

  # add handle to a read set
  $read_set = new IO::Select($handle);

  #print "select on $handle for $timeout\n";
  my ($readable_handles) = IO::Select->select($read_set, undef, undef, $timeout);
  #print "select on $handle returns\n";

  foreach $rh (@$readable_handles) {
    if ($rh == $handle) {

      while ($poll_handle) {

        $line = $rh->getline;
        #print "1: line = '$line'\n";

        # if there was nothing at the socket
        if ((! defined $line) || ($line eq "")) {

          $poll_handle = 0;

          if ($response eq "timed out") {
            $result = "fail";
          } else {
            $result = "ok";
          }

        } else {

          my $needs_newline = 0;
          # remove any trailing carriage return + newline characters
          if ($line =~ m/\r\n$/) {
            $needs_newline = 1;
          }
          $line =~ s/\r\n$//;
          #print "2: line = '$line'\n";

          $result = "ok";
          if ($response eq "timed out") {
            $response = $line;
          } else {
            if ($needs_newline) {
              $response .= "\n".$line;
            } else {
              $response .= $line;
            }
          }
        }
      }
    }
  }
  
  #print "[b] line received = ".$response."\n";
  
  # remove read handle from set
  $read_set->remove($handle);
  
  # restore the blocking state of the handle
  $handle->blocking($blocking_state);
  
  # restore the input record seperator
  $/ = $irs_state;
  
  return ($result, $response);

}


sub getPWCCState($) {

  (my $handle) = @_;

  my $result = "fail";
  my $response = "";

  ($result, $response) = sendTelnetCommand($handle, "state");

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

#
# print the nexus state as a simple string
#
sub printPWCCState($) 
{

  (my $handle) = @_;

  my $result = "";
  my $response = "";
  
  ($result, $response) = sendTelnetCommand($handle, "state");
  
  if ($result eq "ok") 
  {
    my @lines = split(/\n/, $response);
    my $line = "";

    my $pwcc_state = "";
    my @pwcs_states = ();

    my $source = "";
    my $state = "";
  
    my $i = 0;

    foreach $line (@lines) 
    {
      if (($line =~ m/^overall:/) || ($line =~ m/^PWC_/))
      {
        ($source, $state) = split(/: /, $line, 2);

        if ($source eq "overall")
        {
          $pwcc_state = $state
        }
        else
        {
          $i = substr($line, 4);
          @pwcs_states[$i] = $state;
        }
      }
      else
      {
        # ignore
      }
    }

    $response = $pwcc_state;
    for ($i=0; $i<$#pwcs_states; $i++)
    {
      $response .= " ".$pwcs_states[$i]; 
    }
  }

  return ($result, $response);
}

###############################################################################

sub addToTime($$) {

  (my $time, my $toadd) = @_;

  my @t = split(/-|:/,$time);

  my $unixtime = timelocal($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime ($unixtime + $toadd);

  $year += 1900;
  $mon++;
  $mon = sprintf("%02d", $mon);
  $mday = sprintf("%02d", $mday);
  $hour = sprintf("%02d", $hour);
  $min = sprintf("%02d", $min);
  $sec = sprintf("%02d", $sec);
                                                                                                               
  return $year."-".$mon."-".$mday."-".$hour.":".$min.":".$sec;

}

###############################################################################

sub getUnixTimeUTC($) {

  (my $time) = @_;

  my @t = split(/-|:/,$time);

  my $unixtime = timegm($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

  return $unixtime;
}

###############################################################################

sub getCurrentDadaTime(;$) {

  (my $secsToAdd=0) = @_;

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

###############################################################################

sub printDadaTime($) {

  my ($time) = @_;

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

###############################################################################

sub printTime($$) {

  my ($time, $type) = @_;
                                                                                                                 
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

sub printDadaLocalTime($) {

  my ($time) = @_;
  return printTime($time, "local");

}

sub printDadaUTCTime($) {

  my ($time) = @_;
  return printTime($time, "utc");

}


#
# wait for the nexus to transition to the required state
# if any of the pwcs states is in error, stop waiting and 
# return the appropriate error value
#
sub waitForState($$$) 
{
                                                                                
  (my $required_state, my $handle, my $wait_secs) = @_;
                                                                                
  my $pwcc;
  my $pwc;
  my @pwcs;
  my $ready = -1;
  my $counter = 0;
  my $i=0;
  my $result = "";
  my $response = "";

  while (($ready == -1) && ($counter < $wait_secs)) 
  {
    if ($counter == $wait_secs) {
      if (DEBUG_LEVEL >= 1) { 
        print "waitForState: waited ".$wait_secs.", timed out\n"; 
      }
    } else {
      if (DEBUG_LEVEL >= 1) { 
        print "waitForState: Waiting for $required_state\n"; 
      }
    }
    
    # presume that we the state change has worked                                                                            
    $ready = 0;

    # parse the nexus' state
    ($pwcc, @pwcs) = getPWCCState($handle);

    # check pwcc state
    if ($pwcc eq "fatal_error") {
      $ready = PWCC_FATAL_ERROR;
    } elsif ($pwcc eq "hard_error") {
      $ready = PWCC_HARD_ERROR;
    } elsif ($pwcc eq "soft_error") {
      $ready = PWCC_SOFT_ERROR;
    } elsif ($pwcc ne $required_state) {
      $ready = -1;
    } else {
      # pwcc is in required state
    }
  
    $response = "PWCC=".$pwcc." ";
                                                                                
    for ($i=0; $i<=$#pwcs;$i++) 
    {
      $pwc = @pwcs[$i];
      $response .= " PWC".$i."=".$pwc;

      if ($pwc eq "fatal_error") {
        $ready = PWCC_FATAL_ERROR;
      } elsif ($pwc eq "hard_error") {
        $ready = PWCC_HARD_ERROR;
      } elsif ($pwc eq "soft_error") {
        $ready = PWCC_SOFT_ERROR;
      } elsif ($pwc ne $required_state) {
        $ready = -1;
      } else {
        # pwc is in required state
      }
    }

    if ($ready == -1) 
    {
      sleep 1;
      $counter++;
    }
  }

  return $ready;
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

sub setBinaryDir($) {

  (my $dir) = @_;
  
  my $result;
  my $response;
  ($result, $response) = Dada::getAvailableBinaryVersions();
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

sub getDiskInfo($) {
  
  (my $dir) = @_;

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

sub getRawDisk($) {

  (my $dir) = @_;

  my $cmd = "df ".$dir." -B 1048576 -P | tail -n 1 | awk '{print \$2,\$3,\$4}'";
  my $dfresult = `$cmd`;
  chomp($dfresult);
  if ($? != 0) {
    return ("fail",$dfresult);
  } else {
    return ("ok", $dfresult);
  }

}

sub getUnprocessedFiles($) {

  (my $dir) = @_;

  my $duresult = `du -sB 1048576 $dir | awk '{print \$1}'`;
  chomp($duresult);
  if ($? != 0) {
    return ("fail", $duresult);
  } else {
    return ("ok", $duresult);
  }

}

sub getDBInfo($) {

  my ($key) = @_;

  my $bindir = getCurrentBinaryVersion();
  my $cmd = $bindir."/dada_dbmetric -k ".$key;
  my $result = `$cmd 2>&1`;
  chomp $result;
  if ($? != 0) {
    return ("fail","Could not connect to data block");
  } else {
    return ("ok",$result);
  }

}

sub getAllDBInfo($) {

  my ($key_string) = @_;

  my @keys = split(/ /,$key_string);
  my $key;

  my $bindir = getCurrentBinaryVersion();
  my $cmd;
  my $metric_out = "";
  my $result = "ok";
  my $response = "";
  my @parts = ();

  foreach $key (@keys) {

    $cmd = $bindir."/dada_dbmetric -k ".$key." 2>&1";
    $metric_out = `$cmd`;

    # If the command failed
    if ($? != 0) {
      $result = "fail";
      $response .= "0 0 ";
    } else {

     @parts = split(/,/,$metric_out);
     $response .= $parts[0]." ".$parts[1]." ";
    } 
  }

  $response =~ s/\s+$//;

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

sub getLoad($) {

  my ($type) = @_;
  my $one;
  my $five;
  my $fifteen;
  my $response;

  ($one,$five,$fifteen) = (`uptime` =~ /(\d+\.\d+)/g);

  if ($type eq "one") {
    $response = $one;
  } elsif ($type eq "five") {
    $response = $five;
  } elsif ($type eq "fifteen") {
    $response = $fifteen;
  } else {
    $response = $one.",".$five.",".$fifteen;
  }

  return $response;

}

#
# Return the chassis tempreature
#
sub getTempInfo() {

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $temp_str = "NA";
  my @lines = ();

  if ( -f "/usr/bin/omreport" ) {

    $cmd = "/usr/bin/omreport chassis temps | grep \"^Reading\" | awk '{print \$(NF-1)}'";
    ($result, $response) = Dada::mySystem($cmd);
    if ($result eq "ok") {
      $temp_str = $response;
    }

  } elsif ( -f "/opt/dell/srvadmin/bin/omreport" ) {

    $cmd = "/opt/dell/srvadmin/bin/omreport chassis temps | grep \"^Reading\" | tail -n 1 | awk '{print \$(NF-1)}'";
    ($result, $response) = Dada::mySystem($cmd);
    if ($result eq "ok") {
      $temp_str = $response;
    }

  } elsif ( -f "/usr/bin/ipmitool") {
    
    $cmd = "/usr/bin/ipmitool sensor get 'System Temp' | grep \"Sensor Reading\" | awk -F: '{print \$2}' | awk '{print \$1}'";
    ($result, $response) = Dada::mySystem($cmd);
    if ($result eq "ok") {
      $temp_str = $response.".0";
    }

  } else {
    $result = "ok";
    $response = "NA";
  }

  # also query the GPU temperatures
  #if ( -f "/usr/bin/nvidia-smi") {
  #  $cmd = "/usr/bin/nvidia-smi -q -a | grep Temp | awk -F: '{print \$2}' | awk '{print \$1}'";
  #  ($result, $response) = Dada::mySystem($cmd);
  #  if ($result eq "ok") {
  #    @lines = split(/\n/, $response);
  #    my $i = 0;
  #    for ($i=0; $i<=$#lines; $i++) {
  #      $temp_str .= " ".$lines[$i];
  #    }
  #  }
  #}

  return ($result, $temp_str);
}

sub sendErrorToServer($$$){
  (my $host, my $script, my $message) = @_;
}


#
# Logs a message to the nexus machine on the standard logging port.
# The type and level dictate where the log message will be recorded
#
sub nexusLogOpen($$) {

  my ($host, $port) = @_;

  my $handle = connectToMachine($host, $port, 1); 
  # So output goes their straight away
  if (!$handle) {
    print STDERR "Error: $0 could not connect to ".$host.":".$port."\n";
    return 0;
  } else {
    return $handle;
  }

}

sub nexusLogClose($) { 

  (my $handle) = @_;
  if ($handle) {
    $handle->close();
  }

}

sub nexusLogMessage($$$$$$) {

  (my $handle, my $timestamp, my $type, my $level, my $source, my $message) = @_;
  if ($handle) {
    print $handle $timestamp."|".$type."|".$level."|".$source."|".$message."\r\n";
  }
  $handle->flush;
  return $?;

}

###############################################################################
# 
# print a timestamped message if debug levels are correct to STDOUT
#
sub log($$$)
{
  my ($lvl, $dlvl, $message) = @_;
  if ($lvl <= $dlvl) 
  {
    print STDOUT "[".Dada::getCurrentDadaTime()."] ".$message."\n";
  }
}


###############################################################################
#
# Prints the message to STDOUT if the level is < dblevel
#
sub logMsg($$$) {

  my ($lvl, $dlvl, $message) = @_;

  # fix for lines that contain ` characters
  $message =~ s/`/'/;

  if ($lvl <= $dlvl) {
    my $time = Dada::getCurrentDadaTime();
    print STDOUT "[".$time."] ".$message."\n";
  }
}

#
# Prints, the message to STDOUT and cat to fname
#
sub logMsgWarn($$) {

  my ($fname, $message) = @_;

  my $type = "";

  if ($fname =~ m/warn$/) {
    $type = "WARN: ";
  }
  if ($fname =~ m/error$/) {
    $type = "ERROR: ";
  } 

  my $time = Dada::getCurrentDadaTime();

  # Print the message to the log
  print STDOUT "[".$time."] ".$type.$message."\n";

  # Additionally, log to the warn/error file
  system('echo "'.$message.'" >> '.$fname);

}


sub getServerArchiveNFS() {
  return "/nfs/archives";
}

sub getServerResultsNFS() {
  return "/nfs/results";
}

sub constructRsyncURL($$$) {

  my ($user, $host, $dir) = @_;
  return $user."@".$host.":".$dir;

}


sub headerFormat($$) {

  (my $key, my $string) = @_;

  my $pad = 20 - length($key);
  my $header_string = $key;
  my $i=0;
  for ($i=0;$i<$pad;$i++) {
    $header_string .= " ";
  }
  $header_string .= $string;

  return $header_string;
}

sub mySystem($;$) {

  (my $cmd, my $background=0) = @_;

  my $rVal = 0;
  my $result = "ok";
  my $response = "";
  my $realcmd = $cmd." 2>&1";

  if ($background) { $realcmd .= " &"; }

  $response = `$realcmd`;
  $rVal = $?;
  $/ = "\n";
  #if ($response =~ /\n$/) {
    chomp $response;
  #}

  # If the command failed
  if ($rVal != 0) {
    $result = "fail"
  }

  return ($result,$response);

}


sub killProcess($;$) {

  (my $regex, my $user="") = @_;

  my $pgrep_cmd = "";
  my $args = "";
  my $cmd = "";
  my $pids = "";
  my $result = "";
  my $response = "";
  my $fnl = 1;

  $args = "-f '".$regex."'";
  if ($user ne "")
  {
    $args = "-u ".$user." ".$args;
  }

  $pgrep_cmd = "pgrep ".$args;

  Dada::logMsg(2, $fnl, "killProcess: ".$pgrep_cmd);
  ($result, $response) = Dada::mySystem($pgrep_cmd);
  Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);

  # We have one or more processes running
  if (($result eq "ok") && ($response ne "")) {

    # send the process the INT signal
    $cmd = "pkill -INT ".$args;
    Dada::logMsg(2, $fnl, "killProcess: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);

    # give the process(es) a chance to exit
    sleep(1);

    # check they are gone
    Dada::logMsg(2, $fnl, "killProcess: ".$pgrep_cmd);
    ($result, $response) = Dada::mySystem($pgrep_cmd);
    Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);

    if (($result eq "ok") && ($response ne "")) {
    
      # send the process the TERM signal
      $cmd = "pkill -TERM ".$args;
      Dada::logMsg(2, $fnl, "killProcess: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);

      # give the process(es) a chance to exit
      sleep(1);
    
      # check they are gone
      Dada::logMsg(2, $fnl, "killProcess: ".$pgrep_cmd);
      ($result, $response) = Dada::mySystem($pgrep_cmd);
      Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);

      if (($result eq "ok") && ($response ne "")) {
       
        # send the process the KILL signal
        $cmd = "pkill -KILL ".$args;
        Dada::logMsg(2, $fnl, "killProcess: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, $fnl, "killProcess: ".$result." ".$response);
      
        if (($result eq "ok") && ($response ne ""))  {
          return ("fail", "process not killed");
        } else {
          return ("ok", "process killed with SIGKILL");
        }
      } else {
        return ("ok", "process killed with SIGTERM");
      }
    } else {
      return ("ok", "process killed with SIGINT");
    }
  } else {
    return ("ok", "process did not exist");
  }
}
        
#
# Reads a configuration file in the typical DADA format and strips
# out comments and newlines. Returns as an associative array/hash
#
sub readCFGFileIntoHash($$) {

  (my $fname, my $raw) = @_;

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

#
# Redirects stdout and stderr to the logfile and closes stdin. Perfect for
# all your daemonizing needs
#
sub daemonize($$) {
               
  my ($logfile, $pidfile) = @_;
                                                                                
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
  open(STDERR, ">&STDOUT") or die "Failed to re-open STDERR to STDOUT";

  # make both STDOUT and STDERR "hot" to disable perl output buffering
  my $ofh = select STDOUT;
  $| = 1;
  select STDERR;
  $| = 1;
  select $ofh;
                                                                                
}


#
# check for more than 1 version of the specified process name
#
sub preventDuplicateDaemon($) {

  my ($process_name) = @_;

  my $cmd = "ps auxwww | grep '".$process_name."' | grep perl | grep -v grep | wc";
  my $result = `$cmd`;
  if ($? != 0) {
    die "Something reall wrong here, daemon itself is not running??\n";
  } else {
    chomp $result;
    if ($result == 1) {
      # ok, let it run
    } else {
      $cmd = "ps auxwww | grep '".$process_name."' | grep perl | grep -v grep";
      $result = `$cmd`;
      print STDERR "Currently running:\n$result";
      die "Cannot launch multiple ".$process_name." daemons\n";

    }
  } 
}

sub checkScriptIsUnique($) {

  my ($process_name) = @_;

  my $result = "";
  my $response = "";
  my $cmd = "";

  $cmd = "ps auxwww | grep '".$process_name."' | grep perl | grep -v grep | wc";
  $result = `$cmd`;
  if ($? != 0) {
    $result = "fail";
    $response = "no process running";
  } else {

    chomp $result;
    if ($result == 1) {
      # ok, let it run
      $result = "ok";
      $response = "";

    } else {
      $cmd = "ps auxwww | grep '".$process_name."' | grep perl | grep -v grep | awk '{print \$2}'";
      my $ps_running = `$cmd`;
      chomp $ps_running;
      $ps_running =~ s/\n/ /g;
      $result = "fail";
      $response = $process_name.": multiple daemons running with PIDs ".$ps_running;
    }
  }
  return ($result, $response);

}

#
# ssh to the host as the user, and run the remote_cmd. Optionally pipe the
# output to the localpipe command
#
sub remoteSshCommand($$$;$$) {

  (my $user, my $host, my $remote_cmd, my $dir="", my $pipe="") = @_;

  my $result = "";
  my $response = "";
  my $opts = "-x -o BatchMode=yes";
  my $cmd = "";
  my $rval = "";
  
  # since the command will be ssh'd, escape the special characters
  $remote_cmd =~ s/\$/\\\$/g;
  $remote_cmd =~ s/\"/\\\"/g;

  # If a dir is specified, then chdir to that dir
  if ($dir ne "") {
    $remote_cmd = "cd ".$dir."; ".$remote_cmd;
  }

  if ($pipe eq "") {
    $cmd = "ssh ".$opts." -l ".$user." ".$host." \"".$remote_cmd."\" 2>&1";
  } else {
    #$pipe =~ s/\$/\\\$/g;
    #$pipe =~ s/\"/\\\"/g;
    $cmd = "ssh ".$opts." -l ".$user." ".$host." \"".$remote_cmd."\" 2>&1 | ".$pipe;
  }

  #print "remoteSshCommand: ".$cmd."\n";
  $response = `$cmd`;
  #print "remoteSshCommand: ".$response."\n";

  # Perl return values are *256, so divide it. An ssh command 
  # fails with a return value of 255
  $rval = $?/256;

  if ($rval == 0) {
    $result = "ok";

  } elsif ($rval == 255) {
    $result = "fail";

  } else {
    $result = "ok"
  }
  chomp $response;

  return ($result, $rval, $response);
}

#
# Takes a DADA header in a string and returns
# a hash of the header with key -> value pairs
#
sub headerToHash($) {

  my ($raw_hdr) = @_;

  my $line = "";
  my @lines = ();
  my $key = "";
  my $val = "";
  my %hash = ();

  @lines = split(/\n/,$raw_hdr);
  foreach $line (@lines) {
    ($key,$val) = split(/ +/,$line,2);
    if ((length($key) > 1) && (length($val) > 0)) {
      # Remove trailing whitespace
      $val =~ s/\s*$//g;
      $hash{$key} = $val;
    }
  }
  return %hash;
}

sub daemonBaseName($) {

  my ($name) = @_;

  $name =~ s/\.pl$//;
  $name =~ s/^.*client_//;
  $name =~ s/^.*server_//;
  $name =~ s/^.*raid_//;

  return ($name);
}

#
# Return the P### groups the specified user belongs to
#
sub getProjectGroups($) {

  (my $user) = @_;

  my $cmd = "groups ".$user;
  my $result = `$cmd`;
  chomp $result;
  my @parts = split(/ /, $result);   
  my $i = 0;
  my @results = ();

  for ($i=0; $i<=$#parts; $i++) {
    if ($parts[$i] =~ m/^P/) {
      push(@results,$parts[$i]);
    }
  }
  return @results;
}


#
# Determine the processing command line given a raw header for APSR/CASPSR
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

#
# Get the DM from either psrcat or tempo's tzpar dir
#
sub getDM($) {

  my ($source) = @_;

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $dm = "unknown";
  my $par_file = "";

  # check if this source is a CAL
  if (($source =~ m/^HYDRA_/) || ($source =~ /_R$/) || ($source =~ /CalDelay/))
  {
    $dm = "N/A CAL";
    return $dm;
  }

  # test if the source is in the catalogue
  $cmd = "psrcat -all -x -c DM ".$source;
  ($result, $response) = mySystem($cmd);

  # If we had a problem getting the DM from the catalogue
  if (($result ne "ok") || ($response =~ m/not in catalogue/) || ($response =~ m/Unknown parameter/)) {

    # strip leading J or B
    $source =~ s/^[JB]//;

    $par_file = "/home/dada/runtime/tempo/tzpar/".$source.".par";

    if ( -f $par_file ) {
      $cmd = "grep ^DM ".$par_file." | grep -v DMEPOCH | awk '{print \$2}'";
      ($result, $response) = mySystem($cmd);
      if ($result eq "ok") {
        $dm = sprintf("%5.4f",$response);
      }
    } 

  # we found the DM ok
  } else {
    my $junk = "";
    ($dm, $junk) = split(/ +/, $response, 2);
    $dm = sprintf("%5.4f",$response);
  }

  return $dm;
}

sub getPeriod($) {

  my ($source) = @_;

  my $cmd = "";
  my $str = "";
  my $period = "unknown";
  my $par_file = "";
  my $result = "";
  my $response = "";

  # check if this source is a CAL
  if (($source =~ m/^HYDRA_/) || ($source =~ /_R$/)|| ($source =~ /CalDelay/))
  {
    $period = "N/A CAL";
    return $period;
  }

  # test if the source is in the catalogue
  $cmd = "psrcat -all -x -c 'P0' ".$source." | awk '{print \$1}'";
  ($result, $response) = Dada::mySystem($cmd);
  
  chomp $response;
  if (($result eq "ok") && (!($response =~ /WARNING:/))) {
    $period = sprintf("%10.9f",$response);
    $period *= 1000;
    $period = sprintf("%5.4f",$period);

  } else {

    # strip leading J or B
    $source =~ s/^[JB]//;

    $par_file = "/home/dada/runtime/tempo/tzpar/".$source.".par";

    if ( -f $par_file ) {

      # try to get the period
      $cmd = "grep ^F0 ".$par_file." | awk '{print \$2}'";
      ($result, $response) = Dada::mySystem($cmd);
      if (($result eq "ok") && ($response ne "")) {
        chomp $response;
        $period = sprintf("%10.9f",$response);
        if ($period != 0) { 
          $period = ( 1 / $period);
          $period *= 1000;
          $period = sprintf("%5.4f",$period);
        }

      # if F0 didn't exist, try for P
      } else {
        $cmd = "grep '^P ' ".$par_file." | awk '{print \$2}'";
        ($result, $response) = Dada::mySystem($cmd);
        if ($result eq "ok") {
          chomp $response;
          $period = sprintf("%10.9f",$response);
          if ($period != 0) {
            $period *= 1000;
            $period = sprintf("%5.4f",$period);
          }
        }
      }
    }
  }
  return $period;
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

sub array_unique
{
  my @list = @_;
  my %finalList;
  foreach(@list)
  {
    $finalList{$_} = 1; # delete double values
  }
  return (keys(%finalList));
}

sub inArray {
  my ($arr,$search_for) = @_;
  my $value;
  foreach $value (@$arr) {
    return 1 if $value eq $search_for;
  }
  return 0;
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

END { }

1;  # return value from file
