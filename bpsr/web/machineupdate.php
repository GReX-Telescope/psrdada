<?PHP 

include("bpsr.lib.php");
$inst = new bpsr();

$host = $inst->config["SERVER_HOST"];
$port = $inst->config["SERVER_WEB_MONITOR_PORT"];

list ($socket, $result) = openSocket($host, $port);

if ($result == "ok") {

  $bytes_written = socketWrite($socket, "node_info\r\n");
  list ($result, $response) = socketRead($socket);
  socket_close($socket);
  $string = str_replace(";;;;;;","\n",$response);
  $string = rtrim($string);

} else {
  $string = "Could not connect to $host:$port<BR>\n";
}

echo $string;
