#!/usr/bin/env perl

###############################################################################
#
# Converts RA and DEC in radians to degrees
# 
use lib $ENV{"DADA_ROOT"}."/bin";

use Dada;
use strict;

use constant DL => 1;
sub usage();


if ($#ARGV != 1)
{
  print STDERR "Expected 2 arguments\n";
  usage();
  exit 1;
}

my $raj  = $ARGV[0];
my $decj = $ARGV[1];

my ($result, $response);

# convert the RA
Dada::logMsg(2, DL, "main: IN  raj = ".$raj);
($result, $response) = Dada::convertRadiansToRA($raj);
Dada::logMsg(2, DL, "main: ".$result." ".$response);
if ($result ne "ok")
{
  Dada::logMsg(0, DL, "main: convertRadiansToRA failed: ".$response);
  exit 1;
}
my $raj_degrees = $response;

#convert the DEC
Dada::logMsg(2, DL, "main: IN  decj = ".$decj);
($result, $response) = Dada::convertRadiansToDEC($decj);
Dada::logMsg(2, DL, "main: ".$result." ".$response);
if ($result ne "ok")
{
  Dada::logMsg(0, DL, "main: convertRadiansToDEC failed: ".$response);
  exit 1;
}
my $decj_degrees = $response;

Dada::logMsg(0, DL, $raj_degrees." ".$decj_degrees);

##############################################################################
#
# Usage
#
sub usage()
{
  my $script_name = Dada::daemonBaseName($0);
  print STDOUT "Usage: ".$script_name." raj dec\n";
}
