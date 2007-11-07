#!/usr/bin/env perl

use strict;         # strict mode (like -Wall)
use IO::Socket;     # Standard perl socket library
use Net::hostent; 
use Dada;           # DADA Module for configuration options:51

my $val;

use constant RUN_SYS_COMMANDS => "true";
use constant DEBUG_LEVEL => 2;              # 0 == no debug
                                            # 1 == verbose
# local host name
my $HOSTNAME = $ENV{'HOSTNAME'};
my $BINDIR = "/home/ssi/ajameson/apsr/psrdada/src";

my $server = new IO::Socket::INET (
    LocalHost => $HOSTNAME,
    LocalPort => Dada->CLIENT_CONTROL_PORT,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
);
die "Could not create socket: $!\n" unless $server;

my $quit = 0;
my $handler;
my $line = "";
my $rVal;
my $rString;


while (!$quit) {

  $handler = "by jingos!\n";
  if (DEBUG_LEVEL >= 2) {
    print "Waiting for connection\n"
  }

  $handler = $server->accept() or die "accept $!\n";
  $handler->autoflush(1);

  my $peeraddr = $handler->peeraddr;
  my $hostinfo = gethostbyaddr($peeraddr);
  if (DEBUG_LEVEL >= 1) {
    printf "Accepting connection from %s\n", $hostinfo->name || $handler->peerhost;
  }

  if (DEBUG_LEVEL >= 2) {
    print "Waiting for command string\n";
  }
  my $command = <$handler>;
  $/="\r\n";
  chomp($command);

  my $cmd;

  # set return value and string to 0;
  $rVal = 0;
  $rString = "ok";

  if (DEBUG_LEVEL >= 2) {
    print "Command = $command\n";
  }

  ($rVal, $rString) = mysystem($command);
  returnStringToServer($rVal,$rString,$handler);
  
  if (DEBUG_LEVEL >= 2) {
    print "Closing Connection\n"; 
  }
  $handler->close;

}

exit 0;

sub mysystem($) {

  (my $cmd) = @_;

  my $rVal = 0;
  my $rString = "";

  if (DEBUG_LEVEL >= 1) {
    print "Running: ".$cmd."\n";
  }

  if (RUN_SYS_COMMANDS eq "true") {
    $rString = `$cmd 2>&1`;
    $rVal = $?;
    $/ = "\n";
    chomp $rString;
  }

  # If the command failed
  if ($rVal != 0) {
    print "Command $cmd failed: $rString\n";
  }

  return ($rVal, $rString);

}

sub returnStringToServer($$$) {

  (my $rval, my $string,my $handler) = @_;

  my $reply;
  if ($rval == 0) {
    $reply = "ok";
  } else {
    $reply = $string;
  } 
  if (DEBUG_LEVEL >= 1) {
    print "Sending \"".$reply."\" to machine ".Dada->NEXUS_MACHINE."\n";
  }
  print $handler $reply."\r\n";

  if ($reply ne "ok") {
    print $handler "fail\r\n";
  }


}


