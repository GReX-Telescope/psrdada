package Dada;

use IO::Socket;     # Standard perl socket library
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);

require Exporter;
require AutoLoader;
@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &getDFBMachines
  &getPWCMachines
  &getPWCMachinesUDP
  &sendTelnetCommand
  &connectToMachine
  &setBinaryDir
  &getAPSRBinaryDir
  &getDADABinaryDir
  &prepareDFB
  &stopDFB
  &startDFB
  &send_command
  &addToTime
  &getCurrentDadaTime
  &getPWCCState
  &waitForState
  CONFIG_FILE
  NEXUS_MACHINE
  NEXUS_CONTROL_PORT
  CLIENT_CONTROL_PORT
  SERVER_CONTROL_PORT
  DFB_CONTROL_PORT
);

$VERSION = '0.01';

use constant CONFIG_FILE         => '/home/ssi/ajameson/apsr/psrdada/apsr/config/1machine.spec';
use constant NEXUS_MACHINE       => 'shrek110.ssi.swin.edu.au';
use constant NEXUS_CONTROL_PORT  => 12345;
use constant CLIENT_CONTROL_PORT => 57001;
use constant SERVER_CONTROL_PORT => 57002;
use constant DFB_CONTROL_PORT    => 57003;
use constant DEBUG_LEVEL         => 0;

my $DEFAULT_BINARY_DIR = "/home/ssi/ajameson/apsr/psrdada/src";
my $DEFAULT_APSR_BINARY_DIR = "/home/ssi/ajameson/apsr/psrdada/apsr/src";

sub getDFBMachines() {
  my @DFB_MACHINES = qw(
    shrek109.ssi.swin.edu.au 
  );
  return @DFB_MACHINES;
}

sub getPWCMachines() {
  my @PWC_MACHINES = qw(
    shrek117.ssi.swin.edu.au
  );
  return @PWC_MACHINES;
}


sub getPWCMachinesUDP() {
 my @PWC_MACHINES_UDP = qw(
    shrek117.ssi.swin.edu.au
  );
  return @PWC_MACHINES_UDP;
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


sub connectToMachine($$$) {
  
  (my $module, my $machine, my $port) = @_;

  my $tries = 0;
  my $handle = 0;

  # Connect a tcp sock with hostname/ip set
  $handle = new IO::Socket::INET (
    PeerAddr => $machine,
    PeerPort => $port,
    Proto => 'tcp',
  );

  # IF we couldn't do it /cry, sleep and try for 10 times...
  while ((!$handle) && ($tries < 10)) {

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
    if (DEBUG_LEVEL >= 1) {
      print "Connected to ".$machine." on port ".$port."\n";
    }
  } else {
    print "Error: Could not connect to ".$machine." on port ".$port." : $!\n";
  }

  return $handle;
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

sub prepareDFB($$$) {

  (my $module, my $data_rate, my $packet_length) = @_; 

  my @DFB_MACHINES = getDFBMachines();
  my @PWC_MACHINES = getPWCMachinesUDP();
  my $bindir = getAPSRBinaryDir();
  my $response = "";
  my $combinedresponse = "";
  my $result = "";
  my $combinedresult = "ok";

  my $i=0;
  my $rString = "ok";
  for($i=0; $i<=$#DFB_MACHINES; $i++) {
    #print $bindir."/apsr_test_triwave -d -r ".$data_rate." -s ".$packet_length." -n 14400 ".@PWC_MACHINES[$i]."\n";
    ($result, $response) = send_command("Dada",$bindir."/apsr_test_triwave -d -r ".$data_rate." -s ".$packet_length." -n 14400 ".@PWC_MACHINES[$i], @DFB_MACHINES[$i]);
    if ($result ne "ok") {
      print "Response from ".@DFB_MACHINES[$i]." = ".$response."\n";
      $combinedresult = $result;
      $combinedresponse = $response;
    }
  }

  return ($combinedresult,$combinedresponse);

}

sub stopDFB($) {

  (my $module) = @_;
  my @DFB_MACHINES = getDFBMachines();
  my $result = "fail";
  my $response = "";

  ($result, $response) = send_command("Dada", "killall apsr_test_triwave", @DFB_MACHINES);

  return $result;

}


# This function contacts all the DFB machines, and "starts" them. This
# is done in a multithreaded way and the "start" times of the machines
# should all match!! Dont know what to do if they dont :p

sub startDFB($) {
  (my $module) = @_;

  my @machines = getDFBMachines();
  
  my @threads;
  my $machine;
  my $i=0;

  my @returned_data;

  for($i=0;$i<=$#machines;$i++) {
    $machine = @machines[$i];
    @threads[$i] = threads->new(\&dfbstart_thread, $machine);
    if ($? != 0) {
      @returned_data[$i] = "dnf";
    }
  }

  for($i=0;$i<=$#machines;$i++) {
    @returned_data[$i] = @threads[$i]->join;
  }

  my $rString = @returned_data[0];
  chomp $rString;
  return $rString;

}

sub dfbstart_thread($) {

  (my $machine) = @_;

  my $handle = connectToMachine("Dada", $machine,Dada->DFB_CONTROL_PORT);
  # ensure our file handle is valid
  if (!$handle) { return 1; }

  # So output goes their straight away
  $handle->autoflush();

  my $response = sendTelnetCommand("Dada",$handle,"start");

  $handle->close();

  return $response;

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



sub send_command($$$) {

  (my $module, my $command, my @machines) = @_;

  my $machine;
  my $HOSTNAME = $ENV{'HOSTNAME'};
  my $handle = 0;
  my $response = "";
  my $result = "ok";
  my $errorMachine = "none";
  my $errorResponse = "";

  foreach $machine (@machines) {

    $handle = connectToMachine("Dada", $machine,Dada->CLIENT_CONTROL_PORT);

    # ensure our file handle is valid
    if (!$handle) { return 1; }

    # ensure input is flushed straight away
    $handle->autoflush();

    if (DEBUG_LEVEL >= 1) {
      print "Sending \"".$command."\" to machine ".$machine."\n";
    }

    # send command and get response
    ($result, $response) = sendTelnetCommand("Dada",$handle,$command);

    if ($result ne "ok") {
      $errorMachine = $machine;
      $errorResponse = $response;
    }

    if (DEBUG_LEVEL >= 2) {
      print "Closing Socket\n";
    }
    $handle->close;
  }

  if ($errorMachine eq "none") {
    return "ok";
  } else {
    return "$errorResponse";
  }
}

sub addToTime($$$) {

  (my $module, my $time, my $toadd) = @_;

  my @t = split(/-|:/,$time);

  @t[5] += $toadd;
  if (@t[5] >= 60) { @t[4]++; @t[5] -= 60; }
  if (@t[4] >= 60) { @t[3]++; @t[4] -= 60; }
  if (@t[3] >=24) {
    print "Stop working at midnight!!. Couldn't be bothered ";
    print "accounting for this case... :p\n";
    exit(0);
  }

  return @t[0]."-".@t[1]."-".@t[2]."-".@t[3].":".@t[4].":".@t[5];

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
      print STDERR "Waiting for $stateString.";
    } else {
      print STDERR ".";
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
    print STDERR "\n";
  }

  if ($myready eq "yes") {
    return 0;
  } else {
    return -1;
  }
                                                                                
}



__END__
