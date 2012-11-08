<?PHP 
include("definitions_i.php");
include("functions_i.php");

$config = getConfigFile(SYS_CONFIG,TRUE);
$host = $config["SERVER_HOST"];
$port = $config["SERVER_WEB_MONITOR_PORT"];

if ($config["USE_DFB_SIMULATOR"] == 1) {
  $max_gain = 100;
} else {
  $max_gain = 65535;
}

list ($socket, $result) = openSocket($host, $port);
if ($result == "ok") 
{
  $bytes_written = socketWrite($socket, "gain_info\r\n");
  list ($result, $string) = socketRead($socket);
  socket_close($socket);
}
else
{
  $string = "Could not connect to $host:$port<BR>\n";
}
echo $string." ".$max_gain;
