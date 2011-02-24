<?PHP

include("../definitions_i.php");
include("../functions_i.php");
include("../".$instrument.".lib.php");
$inst = new $instrument();
$cfg = $inst->config;

?>
<html>
<?
$inst->print_head("APSR TCS Simulator");
?>
<body>
<table width=90% border=0 cellpadding=0 cellspacing=0 class="datatable">
 <tr>
  <th width="50%">TCS Simulator</th>
  <th width="50%">TCS Interface</th>
 </tr>

<?

/* Open a connection to the TCS interface script */
$host = $cfg["TCS_INTERFACE_HOST"];
$port = $cfg["TCS_INTERFACE_PORT"];
$sock = 0;

list ($sock,$message) = openSocket($host,$port,2);
if (!($sock)) {
  printTR("Error: opening socket to TCS interface [".$host.":".$port."]: ".$message, "");
  printTF();
  printFooter();
  exit(-1);
}

# If we have a STOP command try and stop the tcs interface
if ($_GET["COMMAND"] == "STOP") {
  $cmd = "STOP\r\n";
  socketWrite($sock,$cmd);
  $result = rtrim(socketRead($sock));
  printTR($cmd,$result);
  printTF();
  printFooter();
  exit(0);
}

# otherwise its a START command

$keys = array_keys($_GET);
for ($i=0; $i<count($keys); $i++) {
  $k = $keys[$i];

  if (($k != "COMMAND") && ($k != "LENGTH") && ($k != "SOURCE_LIST") && ($k != "MODE_LIST")) {

    $cmd = $k." ".$_GET[$k]."\r\n";
    socketWrite($sock,$cmd);
    $result = rtrim(socketRead($sock));
    if ($result != "ok") {
      stopCommand($sock);
      exit(-1);
    }
    printTR($cmd,$result);
  }
}

# Issue START command to server_tcs_interface 
$cmd = "START\r\n";
socketWrite($sock,$cmd);
$result = rtrim(socketRead($sock));
if ($result != "ok") {
  printTR("START command failed on nexus ", $result.": ".rtrim(socketRead($sock)));
  stopCommand($sock);
  exit(-1);
} else {
  printTR("Sent START to nexus", "ok");
}

# wait for the next modulo 10 second time
$time_unix = time();
if ($time_unix % 10 != 0) {
  $sleep_time = 10-($time_unix % 10);
  printTR("Waiting ".$sleep_time." secs for next 10 second boundary", "ok");
  sleep($sleep_time);
}

# now wait 5 seconds before issuing the BAT/SET_UTC_START command
printTR("Waiting 5 seconds to issue SET_UTC_START", "ok");
sleep(5);

# generate a UTC_START time in the future
$time_unix = time();

# ensure the UTC_START is modulo 10
if ($time_unix % 10 != 0) {
  $time_unix += (10 - ($time_unix % 10));
}

$utc_start = gmdate(DADA_TIME_FORMAT,$time_unix);

# Issue SET_UTC_START command
$cmd = "SET_UTC_START ".$utc_start."\r\n";
socketWrite($sock,$cmd);
$result = rtrim(socketRead($sock));

if ($result != "ok") {
  $result .= "<BR>\n".rtrim(socketRead($sock));
  printTR("SET_UTC_START [".$cmd."] failed: ", $result);
  printTR(rtrim(socketRead($sock)),"");
  stopCommand($sock);
  exit(-1);
} else {
  printTR("Sent \"".$cmd."\" to nexus", "ok");
}

printTF();
printFooter();

flush();
sleep(2);
socket_close($sock);
exit(0);


###############################################################################
#
# functions 
#

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


# Issue the STOP command
function stopCommand($socket) {

  $cmd = "STOP\r\n";
  socketWrite($socket,$cmd);

  $result = rtrim(socketRead($socket));
  if ($result != "ok") {
    printTR("\"$cmd\" failed",$result.": ".rtrim(socketRead($socket)));
  } else {
    printTR("Sent \"".$cmd."\" to nexus","ok");
  }
  return $result;
}

?>
