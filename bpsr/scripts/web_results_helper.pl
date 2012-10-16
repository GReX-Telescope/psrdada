#!/usr/bin/env perl

#
# Return a list of observations for each PID with total number and size for finished, transferred and deleted
#
#

use lib $ENV{"DADA_ROOT"}."/bin";

use Bpsr;
use strict;
use warnings;

my $dl = 1;
my %cfg = Bpsr::getConfig();      # dada.cfg in a hash

my $cmd = "";
my $result = "";
my $response = "";
my @finished = ();
my @transferred = ();
my @deleted = ();
my @lines = ();
my $line = "";
my $found = 0;
my $i = 0;
my $finished_pid = "";
my $transferred_pid = "";
my $finished_du = "";
my $transferred_du = "";

my %pids = ();
my %pid_counts = ();
my %pid_sizes = ();
my $obs = "";
my $pid = "";

if (! -d $cfg{"CLIENT_ARCHIVE_DIR"})
{
  print "FINISHED\n";
  print "TRANSFERRED\n";
  exit;
}

chdir $cfg{"CLIENT_ARCHIVE_DIR"};
{
  my $ndirs = `ls -1d * | wc -l`;
  if ($ndirs > 0)
  {
    # get a PID listing for each project on local disk
    $cmd = "grep ^PID */*/obs.start | awk -F/ '{print \$1\" \"\$2\" \"\$3}' | awk '{print \$1\"/\"\$2\" \"\$4}'";
    Dada::logMsg(2, $dl, $cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, $dl, $result." ".$response);
    @lines = split(/\n/, $response);
    for ($i=0; $i<=$#lines; $i++)
    {
      ($obs, $pid) = split(/ /, $lines[$i]);
      $pids{$obs} = $pid;
    }
  }

  # get a list of all beams.deleted
  $cmd = "find . -mindepth 3 -maxdepth 3 -type f -name 'beam.deleted' | awk -F/ '{print \$2\"/\"\$3}' | sort";
  Dada::logMsg(2, $dl, $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, $result." ".$response);
  if ($result ne "ok")
  {
    exit 1;
  }
  @deleted = split(/\n/, $response);

  # get a list of all beams.transferred
  $cmd = "find . -mindepth 3 -maxdepth 3 -type f -name 'beam.transferred' | awk -F/ '{print \$2\"/\"\$3}' | sort";
  Dada::logMsg(2, $dl, $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, $result." ".$response);
  if ($result ne "ok")
  {
    exit 1;
  }
  @lines = split(/\n/, $response);
  foreach $line ( @lines )
  {
    $found = 0;
    for ($i=0; $i<=$#deleted; $i++)
    {
      if ($line eq $deleted[$i])
      {
        $found= 1;
      }
    }
    if (!$found)
    { 
      push (@transferred, $line);
      $transferred_pid .= $line."/obs.start ";
      $transferred_du .= $line." ";
    }
  }

  # get a list of all beams.finished
  $cmd = "find . -mindepth 3 -maxdepth 3 -type f -name 'beam.finished' | awk -F/ '{print \$2\"/\"\$3}' | sort";
  Dada::logMsg(2, $dl, $cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, $dl, $result." ".$response);
  if ($result ne "ok")
  {
    exit 1;
  }
  @lines = split(/\n/, $response);
  foreach $line ( @lines )
  {
    $found = 0;
    for ($i=0; $i<=$#deleted; $i++)
    {
      if ($line eq $deleted[$i])
      {
        Dada::logMsg(3, $dl, $line." is marked as deleted");
        $found = 1;
      }
    }
    for ($i=0; $i<=$#transferred; $i++)
    {
      if ($line eq $transferred[$i]) 
      {
        Dada::logMsg(3, $dl, $line." is marked as transferred");
        $found = 1;
      }
    }
    if (!$found)
    {
      Dada::logMsg(3, $dl, $line." is marked as finished");
      push (@finished, $line);
      $finished_pid .= $line."/obs.start ";
      $finished_du .= $line." ";
    }
  }

  Dada::logMsg(2, $dl, "nfinished=".($#finished+1)." transferred=".($#transferred+1)." deleted=".($#deleted));

  my $finished_line = "FINISHED";
  my $transferred_line = "TRANSFERRED";
  #my $deleted_line = "DELETED";
  my @keys = ();

  %pid_counts = ();
  %pid_sizes = ();

  for ($i=0; $i<=$#finished; $i++)
  {
    $obs = $finished[$i];
    $pid = $pids{$obs};
    if (!exists($pid_counts{$pid}))
    {
      $pid_counts{$pid} = 0;
      $pid_sizes{$pid} = 0;
    }
    $pid_counts{$pid}++;
    $pid_sizes{$pid} += `du -b -c $obs | tail -n 1 | awk '{print \$1}'`;
  } 
  @keys = keys %pid_counts;
  for ($i=0; $i<=$#keys; $i++)
  {
    $pid_sizes{$keys[$i]} = sprintf("%0.2f", $pid_sizes{$keys[$i]} / 1073741824);
    $finished_line .= " ".$keys[$i].":".$pid_counts{$keys[$i]}.":".$pid_sizes{$keys[$i]};
    Dada::logMsg(2, $dl, "PID=".$keys[$i]." COUNT=".$pid_counts{$keys[$i]}." SIZES=".$pid_sizes{$keys[$i]});
  } 

  %pid_counts = ();
  %pid_sizes = ();

  for ($i=0; $i<=$#transferred; $i++)
  {
    $obs = $transferred[$i];
    $pid = $pids{$obs};
    if (!exists($pid_counts{$pid}))
    {
      $pid_counts{$pid} = 0;
      $pid_sizes{$pid} = 0;
    }
    $pid_counts{$pid}++;
    $pid_sizes{$pid} += `du -b -c $obs | tail -n 1 | awk '{print \$1}'`;
  } 
  @keys = keys %pid_counts;
  for ($i=0; $i<=$#keys; $i++)
  {
    $pid_sizes{$keys[$i]} = sprintf("%0.2f", $pid_sizes{$keys[$i]} / 1073741824);
    $transferred_line .= " ".$keys[$i].":".$pid_counts{$keys[$i]}.":".$pid_sizes{$keys[$i]};
    Dada::logMsg(2, $dl, "PID=".$keys[$i]." COUNT=".$pid_counts{$keys[$i]}." SIZES=".$pid_sizes{$keys[$i]});
  } 


  #%pid_counts = ();
  #%pid_sizes = ();

  #for ($i=0; $i<=$#deleted; $i++)
  #{
  #  $obs = $deleted[$i];
  #  $pid = $pids{$obs};
  #  if (!exists($pid_counts{$pid}))
  #  {
  #    $pid_counts{$pid} = 0;
  #    $pid_sizes{$pid} = 0;
  #  }
  #  $pid_counts{$pid}++;
  #  $pid_sizes{$pid} += `du -b -c $obs | tail -n 1 | awk '{print \$1}'`;
  #}
  #@keys = keys %pid_counts;
  #for ($i=0; $i<=$#keys; $i++)
  #{
  #  $pid_sizes{$keys[$i]} = sprintf("%0.2f", $pid_sizes{$keys[$i]} / 1048576);
  #  $deleted_line .= " ".$keys[$i].":".$pid_counts{$keys[$i]}.":".$pid_sizes{$keys[$i]};
  #  Dada::logMsg(2, $dl, "PID=".$keys[$i]." COUNT=".$pid_counts{$keys[$i]}." SIZES=".$pid_sizes{$keys[$i]});
  #}

  print $finished_line."\n";
  print $transferred_line."\n";
  #print $deleted_line."\n";

}

exit 0;

