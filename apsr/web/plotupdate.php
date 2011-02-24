<?PHP 
include("definitions_i.php");
include("functions_i.php");
$config = getConfigFile(SYS_CONFIG);

$host = $config["SERVER_HOST"];
$port = $config["SERVER_WEB_MONITOR_PORT"];
$url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];
                                                                                                
list ($socket, $result) = openSocket($host, $port);
                                                                                                
if ($result == "ok") {
                                                                                                
  $bytes_written = socketWrite($socket, "img_info\r\n");
  $string = socketRead($socket);
  socket_close($socket);
                                                                                                
  # Add the require URL links to the image
  $lines = split(";;;", $string);
  $string = "";

  for ($i=0; $i<count($lines)-1; $i++) {
    $p = split(":::", $lines[$i]);
    if (($p[0] == "utc_start") || ($p[0] == "npsrs") || ( substr($p[0],0,3) == "psr")) {
      $string .=  $p[0].":::".$p[1]."\n";
    } else {
      $string .= $p[0].":::".$url."/apsr/results/".$p[1]."\n";;
    }
  }

} else {
  $string = "Could not connect to $host:$port<BR>\n";
}
                                                                                                  
echo $string;
