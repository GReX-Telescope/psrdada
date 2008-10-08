<?PHP
include("../../functions_i.php");
include("../../definitions_i.php");
?>

<html> 

<?

$title = "DADA | BPSR | TCS Simulator";
include("../../header_i.php");

if (!IN_CONTROL) { ?>
<h3><font color=red>Test system disabled as your host is not in control of the instrument</font></h3>
</body>
</html>
<?
  exit(0);
} ?>


<body>
<table width=90% border=0 cellpadding=0 cellspacing=0 class="datatable">
 <tr>
  <th width="50%">TCS Simulator</th>
  <th width="50%">TCS Interface</th>
 </tr>

<?

$sys_config = getConfigFile(SYS_CONFIG);

$ibob_config_file = "";
$spec_file = "";
$duration = 30;

/* Client master control sockets */
$cmc_sockets = array(); 

if (isset($_GET["cfg"])) {
  $ibob_config_file = $_GET["cfg"];
}
if (isset($_GET["spec"])) {
  $spec_file = $sys_config["CONFIG_DIR"]."/".$_GET["spec"];
}
if (isset($_GET["duration"])) {
  $duration = $_GET["duration"];
} else {
  printTR("Error: duration not specified in _GET parameters","");
  printTF();
  printFooter();
  exit(-1);
}


if (!(file_exists($sys_config["CONFIG_DIR"]."/".$ibob_config_file))) {
  printTR("Error: ibob Configuration File \"".$_GET["cfg"]."\" did not exist","");
  printTF();
  printFooter();
  exit(-1);
} else {
  
}

if (!(file_exists($spec_file))) {
  printTR("Error: Specification File \"".$_GET["spec"]."\" did not exist","");
  printTF();
  printFooter();
  exit(-1);
}

$specification = getRawTextFile($spec_file);
$spec = getConfigFile($spec_file);

/* Open a connection to the TCS interface script */
$host = $sys_config["TCS_INTERFACE_HOST"];
$port = $sys_config["TCS_INTERFACE_PORT"];

$tcs_interface_socket = 0;

list ($tcs_interface_socket,$message) = openSocket($host,$port,2);
if (!($tcs_interface_socket)) {
  printTR("Error: opening socket to TCS interface script \"".$message."\"","");
  printTF();
  printFooter();
  exit(-1);
} 

# Send the ibob Configuration File to the tcs_interface socket
$cmd = "CONFIG ".$ibob_config_file."\r\n";
socketWrite($tcs_interface_socket,$cmd);
$result = rtrim(socketRead($tcs_interface_socket));
printTR($cmd,$result);
if ($result != "ok") {
  exit(-1);
}


# Send each header parameter to TCS interface
for ($i=0;$i<count($specification);$i++) {
  $cmd = $specification[$i]."\r\n";
  socketWrite($tcs_interface_socket,$cmd);
  $result = rtrim(socketRead($tcs_interface_socket));
  printTR($cmd,$result);
  if ($result != "ok") {
    exit(-1);
  }
}

# Issue START command to server_tcs_interface 
$cmd = "START\r\n";
socketWrite($tcs_interface_socket,$cmd);
$result = rtrim(socketRead($tcs_interface_socket));
if ($result != "ok") {
  printTR("START command failed on nexus ", $result);
  printTR(rtrim(socketRead($tcs_interface_socket)),"");
  exit(-1);
} else {
  printTR("Send START to nexus", "ok");
}

printTR("Sleeping for 10 seconds","");
sleep(10);


# Now run for the duration of the observation
$have_set_duration = 0;
for ($i=0;$i<$duration;$i++) {

  sleep(1);
  # Every 15 seconds, set the time limit of the script back to 30 seconds
  if ($i % 15 == 0) {
    set_time_limit(30);

    if (!$have_set_duration) {
      $cmd = "DURATION ".$duration."\r\n";
      socketWrite($tcs_interface_socket,$cmd);
      $result = rtrim(socketRead($tcs_interface_socket));

      if ($result != "ok") {
        printTR($cmd, $result);
        printTR("STOPPING","");
        exit(-1);
      }
      printTR($cmd, $result);
      $have_set_duration = 1;
    }
  }
  if ($i % 60 == 0) {
    printTR("Recording: ".(($duration - $i)/60)." minutes remaining","");
  }
}


# 10 extra seconds to ensure things have stopped!
sleep(10);

# Issue the STOP command 
$cmd = "STOP\r\n";
socketWrite($tcs_interface_socket,$cmd);
$result = rtrim(socketRead($tcs_interface_socket));
if ($result != "ok") {
  printTR("\"$cmd\" failed",$result);
  printTR("",rtrim(socketRead($tcs_interface_socket)));
  exit(-1);
} else {
  printTR("Sent \"".$cmd."\" to nexus","ok");
}

printTF();
printFooter();
flush();
sleep(2);
socket_close($tcs_interface_socket);
exit(0);

function printFooter() {

  echo "</body>\n";
  echo "</html>\n";
}

function printTR($tcs_simulator,$tcs_interface) {
  echo " <tr bgcolor=\"white\">\n";
  echo "  <td >".$tcs_simulator."</td>\n";
  echo "  <td align=\"left\">".$tcs_interface."</td>\n";
  echo " </tr>\n";
  echo '<script type="text/javascript">self.scrollBy(0,100);</script>';
  flush();
}

function printTF() {
  echo "</table>\n";
}

function killDFBSimulators($config, $dfbs) {

  $host = $dfbs["DFB_0"];

  list ($sock,$message) = openSocket($host,57001,2);
  if ($sock) {

    $cmd = "kill_process apsr_test_triwave";
    printTR("Writing \"".$cmd."\" to host ".$host, "");

    socketWrite($sock,$cmd."\r\n");

    $result = rtrim(socketRead($sock));

    printTR("Killing DFB simluator on ".$host,$result);

    socket_close($sock);
  } else {
    printTR("Could not open socket to $host",$message);
  }

}
?>
