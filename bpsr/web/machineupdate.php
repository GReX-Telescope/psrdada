<?PHP 

include ("bpsr.lib.php");
$inst = new bpsr();

$host = $inst->config["SERVER_HOST"];
$port = $inst->config["SERVER_WEB_MONITOR_PORT"];

list ($socket, $result) = openSocket($host, $port);

if ($result == "ok") {

  $bytes_written = socketWrite($socket, "node_info\r\n");
  $read = socketRead($socket);
  socket_close($socket);
  $string = str_replace(";;;;;;","\n",$read);
  $string = rtrim($string);

} else {
  $string = "Could not connect to $host:$port<BR>\n";
}

echo $string;
