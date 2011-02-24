<?PHP 

include ("bpsr.lib.php");

$inst = new bpsr();

$host = $inst->config["SERVER_HOST"];
$port = $inst->config["SERVER_WEB_MONITOR_PORT"];

list ($socket, $result) = openSocket($host, $port);

if ($result == "ok") {

  $bytes_written = socketWrite($socket, "curr_obs\r\n");
  $string = socketRead($socket);
  socket_close($socket);

} else {
  $string = "Could not connect to $host:$port<BR>\n";
}

echo $string;
