#!/usr/bin/env perl

#
# Reconfigures the DADA machine for a different instrument
#

#use IO::Socket;     # Standard perl socket library
#use Net::hostent;
use Dada;           # DADA Module for configuration options
use threads;
use threads::shared;
use strict;         # strict mode (like -Wall)
use Getopt::Std;


#
# Constants
#

use constant  DEBUG_LEVEL         => 1;

#
# Global Variables
#
our %cfg : shared = Dada->getDadaConfig();      # dada.cfg in a hash
our @serverDaemons = split(/ /,$cfg{"SERVER_DAEMONS"});

#
# Local Variables
#

my $result = "";
my $response = "";
my @clients = ();
my @helpers = ();
my @clients_n_helpers = ();
my $i=0;

my %opts;
getopts('he:', \%opts);

my $new_instrument     = $cfg{"INSTRUMENT"};     # use current as default
my $current_instrument = $cfg{"INSTRUMENT"};     # use current as default

if ($opts{h}) {
  usage();
  exit(0);
}

if ($opts{e}) {
  $new_instrument = $opts{e};
}

if ($new_instrument eq $current_instrument) {
  debugMessage(0, "Restarting instrument: ".$new_instrument);
} else {
  debugMessage(0, "Changing instrument: ".$current_instrument." => ".$new_instrument);
}


# Setup directories should they not exist
my $current_control_dir = "/tmp/".$current_instrument."/control";
my $new_control_dir = "/tmp/".$new_instrument."/control";

if (! -d $current_control_dir) {
  system("mkdir -p ".$current_control_dir);
}
if (! -d $new_control_dir) {
  system("mkdir -p ".$new_control_dir);
}

# Prepare new symlinks for relevant files
my $dada_info = $cfg{"CONFIG_DIR"}."/dada.info";
my $inst_info = $cfg{"CONFIG_DIR"}."/".$new_instrument.".info";

my $dada_viewer = $cfg{"CONFIG_DIR"}."/dada.viewer";
my $inst_viewer = $cfg{"CONFIG_DIR"}."/".$new_instrument.".viewer";

my $dada_config = $cfg{"CONFIG_DIR"}."/dada.cfg";
my $inst_config = $cfg{"CONFIG_DIR"}."/".$new_instrument.".cfg";

my $old_results_dir = $cfg{"WEB_DIR"}."/results";


# Check that all the required files exist
if (! -f $inst_info) {
  print STDERR "The instruments .info file \"".$inst_info."\" did not exist\n";
  exit 1;
}

if (! -f $inst_viewer) {
  print STDERR "The instruments .viewer file \"".$inst_viewer."\" did not exist\n";
  exit 1;
}

if (! -f $inst_config) {
  print STDERR "The instruments .cfg file \"".$inst_config."\" did not exist\n";
  exit 1;
}

# Generate hosts lists
for ($i=0; $i < $cfg{"NUM_PWC"}; $i++) {
  push(@clients,           $cfg{"PWC_".$i});
  push(@clients_n_helpers, $cfg{"PWC_".$i});
}
for ($i=0; $i < $cfg{"NUM_HELP"}; $i++) {
  push(@helpers, $cfg{"HELP_".$i});
  push(@clients_n_helpers, $cfg{"HELP_".$i});
}

# Determine the current configuration


# Stop PWC's
debugMessage(0, "Stopping PWCs");
if (!(issueTelnetCommand("stop_pwcs",\@clients))) {
  debugMessage(0, "stop_pwcs failed");
}

# Stop client scripts
debugMessage(0, "Stopping client scripts");
if (!(issueTelnetCommand("stop_daemons",\@clients_n_helpers))) {
  debugMessage(0,"stop_daemons failed");
}

# Destroy DB's
debugMessage(0, "Destroying Data blocks");
if (!(issueTelnetCommand("destroy_db",\@clients_n_helpers))) {
  debugMessage(0,"destroy_db failed");
}

# Stop server scripts
debugMessage(0, "Stopping server scripts");
($result, $response) = stopDaemons();
if ($result ne "ok") {
  debugMessage(0, "Could not stop server daemons: ".$response);
}

# Stop client mastser script
debugMessage(0, "Stopping client master script");
if (!(issueTelnetCommand("stop_master_script",\@clients_n_helpers))) {
  debugMessage(0,"stop_master_script failed");
}

if ($cfg{"USE_DFB_SIMULATOR"} eq 1) {
  my @arr = ();
  push(@arr,$cfg{"DFB_SIM_HOST"});
  issueTelnetCommand("stop_master_script",\@arr);
}

# Remove the webservers "results" symlink dir
if (-d ($old_results_dir)) {
  unlink ($old_results_dir) || die "Could not unlink old results dir ".$old_results_dir."\n";
}

# Re-link dada.cfg
if (-f $dada_config) {
  unlink ($dada_config) || die "Could not unlink $dada_config\n";
}
debugMessage(0, "Linking ".$inst_config." to ".$dada_config);
symlink ($inst_config, $dada_config) || die "Could not symlink $inst_config to $dada_config\n";

# Re-link dada.viewer
if (-f $dada_viewer) {
  unlink ($dada_viewer) || die "Could not unlink $dada_viewer\n";
}
debugMessage(0, "Linking ".$inst_viewer." to ".$dada_viewer);
symlink ($inst_viewer, $dada_viewer) || die "Could not symlink $inst_viewer to $dada_viewer\n";

# Re-link dada.info
if (-f $dada_info) {
  unlink ($dada_info) || die "Could not unlink $dada_info\n";
}
debugMessage(0, "Linking ".$inst_info." to ".$dada_info);
symlink ($inst_info, $dada_info) || die "Could not symlink $inst_info to $dada_info\n";

# Get a new config potentially...
%cfg = Dada->getDadaConfig();
@serverDaemons = split(/ /,$cfg{"SERVER_DAEMONS"});
@clients = ();
@clients_n_helpers = ();
@helpers = ();


# Generate hosts lists
for ($i=0; $i < $cfg{"NUM_PWC"}; $i++) {
  push(@clients,           $cfg{"PWC_".$i});
  push(@clients_n_helpers, $cfg{"PWC_".$i});
}
for ($i=0; $i < $cfg{"NUM_HELP"}; $i++) {
  push(@helpers, $cfg{"HELP_".$i});
  push(@clients_n_helpers, $cfg{"HELP_".$i});
}

# Re-link the new webserver "results" dir
symlink ($cfg{"SERVER_RESULTS_DIR"}, $cfg{"WEB_DIR"}."/results") || die "Could not symlink ".$cfg{"SERVER_RESULTS_DIR"}." to ".$cfg{"WEB_DIR"}."/results\n";

# Start client master script
debugMessage(0, "Starting client master script");
if (!(issueTelnetCommand("start_master_script",\@clients_n_helpers))) {
  debugMessage(0,"start_master_script failed");
}

if ($cfg{"USE_DFB_SIMULATOR"} eq 1) {
  my @arr = ();
  push(@arr,$cfg{"DFB_SIM_HOST"});
  issueTelnetCommand("start_master_script",\@arr);
}

sleep(5);

# initalize DB's
debugMessage(1, "Initializing Data blocks");
if (!(issueTelnetCommand("init_db",\@clients_n_helpers))) {
  debugMessage(0, "init_db failed");
}

sleep(5);

# Start PWC's
debugMessage(1, "Starting PWCs");
if (!(issueTelnetCommand("start_pwcs",\@clients))) {
  debugMessage(0,"start_pwcs failed");
}

# Start server scripts
debugMessage(0, "Starting server scripts");
($result, $response) = startDaemons();
if ($result ne "ok") {
  debugMessage(0, "Could not start server daemons: ".$response);
}

# Start client scripts
debugMessage(0, "Starting client scripts");
if (!(issueTelnetCommand("start_daemons",\@clients))) {
  debugMessage(0,"start_daemons failed");
}

# Start client helper scripts
debugMessage(0, "Starting client helper scripts");
if (!(issueTelnetCommand("start_helper_daemons",\@helpers))) {
  debugMessage(0,"start_helper_daemons failed");
}

# Rewrite the webservers instrument_i.php file
open FH,">".$cfg{"WEB_DIR"}."/instrument_i.php";
print FH "<?PHP\n";
print FH "define(INSTRUMENT,\"".$new_instrument."\");\n";
print FH "?>\n";
close FH;

# Signal webpage to update
system("touch ".$cfg{"SERVER_CONTROL_DIR"}."/change_instrument");

exit 0;


sub sshCmdThread($) {

  (my $command) = @_;

  my $result = "fail";
  my $response = "Failure Message";

  $response = `$command`;
  if ($? == 0) {
    $result = "ok";
  }
  return ($result, $response);
  
}

sub commThread($$) {

  (my $command, my $machine) = @_;

  my $result = "fail";
  my $response = "Failure Message";
 
  my $handle = Dada->connectToMachine($machine, $cfg{"CLIENT_MASTER_PORT"}, 2);
  # ensure our file handle is valid
  if (!$handle) { 
    debugMessage(0, "Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"});
    return ("fail","Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"}); 
  }

  ($result, $response) = Dada->sendTelnetCommand($handle,$command);
  # debugMessage(0, $command." => ".$result.":".$response);

  $handle->close();

  return ($result, $response);

}

sub usage() {
  print "Usage: ".$0." [options]\n";
  print "   -i instrument    change the instrument \n";
  print "   -h               print help text\n";
}

sub debugMessage($$) {
  my ($level, $message) = @_;

  if (DEBUG_LEVEL >= $level) {

    # print this message to the console
    print "[".Dada->getCurrentDadaTime(0)."] ".$message."\n";
  }
}

sub issueTelnetCommand($\@){

  my ($command, $nodesRef) = @_;

  my @nodes = @$nodesRef;
  my @threads = ();
  my @results = ();
  my @responses = ();
  my $failure = "false";
  my $i=0;

  if ($command eq "start_master_script") {
    for ($i=0; $i<=$#nodes; $i++) {
      my $string = "ssh -x ".$nodes[$i]." \"cd ".$cfg{"SCRIPTS_DIR"}."; ./client_master_control.pl\"";
      @threads[$i] = threads->new(\&sshCmdThread, $string);
    } 
  } else {
    for ($i=0; $i<=$#nodes; $i++) {
      @threads[$i] = threads->new(\&commThread, $command, $nodes[$i]);
    } 
  }  
  for($i=0;$i<=$#nodes;$i++) {
    debugMessage(2, "Waiting for ".$nodes[$i]);
    (@results[$i],@responses[$i]) = $threads[$i]->join;
  }

  for($i=0;$i<=$#nodes;$i++) {
    if (($results[$i] eq "fail") || ($results[$i] eq "dnf")) {
      debugMessage(2, $nodes[$i]." failed \"".$result.":".$responses[$i]."\"");
      $failure = "true";
    }
  }

  if ($failure eq "true") {
    return 0;
  } else {
    return 1;
  }
}


sub stopDaemons() {

  my $allStopped = "false";

  my $daemon_control_file = Dada->getDaemonControlFile($cfg{"SERVER_CONTROL_DIR"});

  my $threshold = 20; # seconds
  my $daemon = "";
  my $allStopped = "false";
  my $result = "";
  my $response = "";

  `touch $daemon_control_file`;

  while (($allStopped eq "false") && ($threshold > 0)) {

    $allStopped = "true";
    foreach $daemon (@serverDaemons) {
      my $cmd = "ps auxwww | grep \"perl ./server_".$daemon.".pl\" | grep -v grep";
      `$cmd`;

      if ($? == 0) {
        debugMessage(1, "daemon ".$daemon." is still running");
        $allStopped = "false";
        if ($threshold < 10) {
          ($result, $response) = Dada->killProcess("server_".$daemon.".pl");
        }
      } else {
        debugMessage(2, "daemon ".$daemon." has been stopped");
      }
    }

    $threshold--;
    sleep(1);
  }

  my $message = "";
  if (unlink($daemon_control_file) != 1) {
    $message = "Could not unlink the daemon control file \"".$daemon_control_file."\"";
    debugMessage(0, "Error: ".$message);
    return ("fail", $message);
  }

  # If we had to resort to a "kill", send an warning message back
  if (($threshold > 0) && ($threshold < 10)) {
    $message = "Daemons did not exit cleanly within ".$threshold." seconds, a KILL signal was used and they exited";
    debugMessage(0, "Error: ".$message);
    return ("fail", $message);
  }
  if ($threshold <= 0) {
    $message = "Daemons did not exit cleanly after ".$threshold." seconds, a KILL signal was used and they exited";
    debugMessage(0, "Error: ".$message);
    return ("fail", $message);
  }

  return ("ok", "all stopped");

}

sub startDaemons() {

  my $daemon;
  my $response = "";
  my $result = "ok";
  my $cmd;
  my $string;

  chdir $cfg{"SCRIPTS_DIR"};

  foreach $daemon (@serverDaemons) {
    $cmd = "server_".$daemon.".pl 2>&1";

    $string = `$cmd`;

    if ($? != 0) {
      $result = "fail";
      $response .= $daemon." failed to start: ".$string.". ";
      debugMessage(0, "Failed to start daemon ".$daemon.": ".$string);
    }
  }
  return ($result, $response);
}
