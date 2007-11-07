#!/usr/bin/env perl

use IO::Socket;     # Standard perl socket library
use Net::hostent;
use Dada;           # DADA Module for configuration options
use threads;
use Switch;

use strict;         # strict mode (like -Wall)

use constant DEBUG_LEVEL => 1;

my @DFB_MACHINES = Dada->getDFBMachines();
my @PWC_MACHINES = Dada->getPWCMachines();
my $val;

print "DFB Machines:\n";
foreach $val (@DFB_MACHINES) {
  print "  ".$val."\n";
}

print "PWC Machines:\n";
foreach $val (@PWC_MACHINES) {
  print "  ".$val."\n";
}

print "NEXUS Machine:\n";
print "  ".Dada->NEXUS_MACHINE.":".Dada->NEXUS_CONTROL_PORT."\n";

# Main Loop
my $quit = 0;
my $command;
my @array;
my $result;
my $response;
my $arg;
my $rVal = 0;   # 0 is good, -1 is not
my $rString = "";

while (!$quit) {

  $rVal = 0;
  $rString = "ok";

  print "\n";
  $response = "";
  $response = lc promptUser("Enter a command");

  @array = split(' ',$response);
  $command = $array[0];

  if (DEBUG_LEVEL >= 1) {
    print "command = $response\n" 
  }

  $result = "fail";
  $response = "";

  switch ($command) {

    # print help commands 
    case "help" {
      print "\nTCS Commands:\n";
      print "    help             prints these commands\n";
      print "    start specfile   starts recording using the specfile specification file\n";
      print "    stop             stops recording immediately\n";
      print "\nNon TCS Commands:\n";
      print "    reboot           Reboots client machines [disabled]\n";
      print "    destroydb        Destroys data blocks on clients\n";
      print "    initdb           Initialises data blocks on clients [1000 x 4MB segmenets]\n";
      print "    initsystem       Set kernel buffers, init data blocks and starts dada daemons\n";
      print "    startdfb         Starts dfb simulator on DFB machines [testing only]\n";
      print "    stopdfb          Stops dfb simulator on DFB machines [testing only]\n";
    }

    # start everything, requries arg to be speicifcation file
    case "start" {
      $arg = $array[1];
      $rString = rec_start($arg, @DFB_MACHINES);
    }

    # stop recording, return to idle state
    case "stop" {
      $rString = rec_stop();
    }

    # quit the tcs simulator
    case "quit" {
      $quit = 1;
      $rString = "ok";
    }

    # reboot machines
    case "reboot" {
      ($result, $response) = Dada->send_command("/sbin/reboot", @PWC_MACHINES);
    }

    case "destroydb" {
      ($result, $response) = Dada->send_command("sudo /usr1/local/bin/dada_db -d", @PWC_MACHINES);
    }

    case "initdb" {
      ($result, $response) = Dada->send_command("sudo /usr1/local/bin/dada_db -b 4194304 -n 500 -l", @PWC_MACHINES);
    } 

    case "initsystem" {

      my $bindir = Dada->getDADABinaryDir();
      ($result, $response) = Dada->send_command("sudo /sbin/sysctl -w net.core.wmem_max=67108864", @PWC_MACHINES);
      if ($result eq "ok") { ($result, $response) = Dada->send_command("sudo /sbin/sysctl -w net.core.rmem_max=67108864", @PWC_MACHINES); }
      if ($result eq "ok") { ($result, $response) = Dada->send_command("sudo /usr1/local/bin/dada_db -b 4194304 -n 500 -l", @PWC_MACHINES); }
      if ($result eq "ok") { ($result, $response) = Dada->send_command($bindir."/apsr_udpdb -d", @PWC_MACHINES); }
      if ($result eq "ok") { ($result, $response) = Dada->send_command($bindir."/apsr_dbtriwave -d", @PWC_MACHINES); }
    } 

    case "stopdaemons" {
      ($result, $response) = Dada->send_command("killall apsr_udpdb", @PWC_MACHINES);
      ($result, $response) = Dada->send_command("killall apsr_dbtriwave", @PWC_MACHINES);
      ($result, $response) = Dada->send_command("killall apsr_dbtriripple", @PWC_MACHINES);
    }

    case "startdaemons" {
      my $bindir = Dada->getAPSRBinaryDir();
      ($result, $response) = Dada->send_command($bindir."/apsr_udpdb -d", @PWC_MACHINES);
      if ($result eq "ok") { 
        ($result, $response) = Dada->send_command($bindir."/apsr_dbtriwave -d", @PWC_MACHINES); }
    }

    case "preparedfb" {
      ($result, $response) = Dada->prepareDFB(64,1458);
    }

    case "startdfb" {
      ($result, $response) = Dada->startDFB();
    }

    case "stopdfb" {
      ($result, $response) = Dada->stopDFB();
    }

    else {
      $result = "fail";
      $response = "Unrecognized command $command\n";
    }

  }
  print $result.": ".$response."\n";
}

exit 0;

#
# Functions
#

sub rec_start($$) {

  (my $file, my @DFB_MACHINES) = @_;

  my $rVal = 0;
  my $cmd;

  print "\nStarting recording with specification file $file\n";
  my $result;
  my $response;

  my $handle = Dada->connectToMachine(Dada->NEXUS_MACHINE,Dada->NEXUS_CONTROL_PORT);

  if (!$handle) {
    return "Error connecting to Nexus:\n";
  } else {

  # So output goes their straight away
  $handle->autoflush(1);

  # Ignore the "welcome" message
  $result = <$handle>;

  $cmd = "config ".$file;
  ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
  
  if ($result ne "ok") { $rVal = -1; }

  if ($rVal == 0) {

    sleep 1;

    $cmd = "clock";
    ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);

    if ($result ne "ok") { $rVal = -1; }

    if ($rVal == 0) {

      sleep 2;

      # Now start the DFB simualtor...
      print STDERR "Starting the DFB\n";
      my $time = Dada->dfbstart(@DFB_MACHINES);

      sleep 1;

      # and start the pwc_command
      $cmd = "set_utc_start ".$time;
      ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);
                                                                                                
      if ($result ne "ok") { $rVal = -1; }

      if ($rVal == 0) {

        sleep 1;
 
        # and start the pwc_command
        $cmd = "rec_start ".$time;
        ($result,$response) = Dada->sendTelnetCommand($handle,$cmd);

        if ($result ne "ok") { $rVal = -1; }

      }
    }
  }
  $handle->close();

  if ($rVal == 0) {
    return "ok"
  } else {
    return $response;
  }

  return $rVal;
  }
}

sub rec_stop() {

  print "\nStopping recording immediately\n";
  my $result;
  my $response;

  my $handle = Dada->connectToMachine(Dada->NEXUS_MACHINE,Dada->NEXUS_CONTROL_PORT);

  if (!$handle) {
    return "Error connecting to Nexus:\n"
  } else {

    # So output goes their straight away
    $handle->autoflush(1);

    # Ignore the "welcome" message
    $result = <$handle>;

    ($result, $response) = Dada->sendTelnetCommand($handle,"stop");

    if ($result eq "ok") {
      return "ok"
    } else {
      return $response;
    }  
  }

}

#sub send_command($$) {
#
#  (my $command, my @machines) = @_;
#
#  my $machine;
#  my $HOSTNAME = $ENV{'HOSTNAME'};
#  my $handle = 0;
#  my $response = "";
#  my $result = "ok";
#  my $errorMachine = "none";
#  my $errorResponse = "";
#
#  foreach $machine (@machines) {
#
#    $handle = Dada->connectToMachine($machine,Dada->CLIENT_CONTROL_PORT);
#
#    # ensure our file handle is valid
#    if (!$handle) { return 1; }
#
#    # ensure input is flushed straight away
#    $handle->autoflush();
#
#    if (DEBUG_LEVEL >= 1) {
#      print "Sending \"".$command."\" to machine ".$machine."\n";
#    }
#
#   # send command and get response
#   ($result, $response) = Dada->sendTelnetCommand($handle,$command);
#
#   if ($result ne "ok") {
#     $errorMachine = $machine;
#     $errorResponse = $response;
#   }
#
#   if (DEBUG_LEVEL >= 2) {
#     print "Closing Socket\n";
#   } 
#   $handle->close;
# }
#
# if ($errorMachine eq "none") {
#   return "ok";
# } else {
#    return "$errorResponse";
#  }
#
#}


#----------------------------(  promptUser  )-----------------------------#
#                                                                         #
#  FUNCTION:  promptUser                                                  #
#                                                                         #
#  PURPOSE: Prompt the user for some type of input, and return the        #
#   input back to the calling program.                                    #
#                                                                         #
#  ARGS:  $promptString - what you want to prompt the user with           #
#   $defaultValue - (optional) a default value for the prompt             #
#                                                                         #
#-------------------------------------------------------------------------#

sub promptUser {


   #my $promptString;
   #my $defaultValue;

   #-------------------------------------------------------------------#
   #  two possible input arguments - $promptString, and $defaultValue  #
   #  make the input arguments local variables.                        #
   #-------------------------------------------------------------------#

   my ($promptString,$defaultValue) = @_;

   $/="\n";
   #-------------------------------------------------------------------#
   #  if there is a default value, use the first print statement; if   #
   #  no default is provided, print the second string.                 #
   #-------------------------------------------------------------------#

   if ($defaultValue) {
      print $promptString, "[", $defaultValue, "]: ";
   } else {
      print $promptString, ": ";
   }

   $| = 1;               # force a flush after our print
   $_ = <STDIN>;         # get the input from STDIN (presumably the keyboard)

   #------------------------------------------------------------------#
   # remove the newline character from the end of the input the user  #
   # gave us.                                                         #
   #------------------------------------------------------------------#

   chomp;

   #-----------------------------------------------------------------#
   #  if we had a $default value, and the user gave us input, then   #
   #  return the input; if we had a default, and they gave us no     #
   #  no input, return the $defaultValue.                            #
   #                                                                 # 
   #  if we did not have a default value, then just return whatever  #
   #  the user gave us.  if they just hit the <enter> key,           #
   #  the calling routine will have to deal with that.               #
   #-----------------------------------------------------------------#

   if ("$defaultValue") {
      return $_ ? $_ : $defaultValue;    # return $_ if it has a value
   } else {
      return $_;
   }
}


# This function contacts all the DFB machines, and "starts" them. This
# is done in a multithreaded way and the "start" times of the machines
# should all match!! Dont know what to do if they dont :p
#sub dfbstart($) {

#  (my @machines) = @_;

#  my @threads;
#  my $machine;
#  my $i=0;

#  my @returned_data;

#  for($i=0;$i<=$#machines;$i++) {
#    $machine = @machines[$i];
#    @threads[$i] = threads->new(\&dfbstart_thread, $machine);
#    if ($? != 0) {
#      @returned_data[$i] = "dnf";    
#    }
#  }

#  for($i=0;$i<=$#machines;$i++) {
#    @returned_data[$i] = @threads[$i]->join; 
#  }

#  my $rString = @returned_data[0];

#  chomp $rString;

#  return $rString;


#}

#sub dfbstart_thread($) {

#  (my $machine) = @_;

#  my $handle = Dada->connectToMachine($machine,Dada->DFB_CONTROL_PORT);
  # ensure our file handle is valid
#  if (!$handle) { return 1; }
  
  # So output goes their straight away
#  $handle->autoflush();

#  my $response = Dada->sendTelnetCommand($handle,"start");

#  $handle->close();

#  return $response;

#}
