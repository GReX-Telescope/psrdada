<?PHP
include("definitions_i.php");
include("functions_i.php");

clearstatcache();

if (!(isset($_GET["port"])) || (!(isset($_GET["host"])))) {
  $string = "Malformed HTTP GET parameters";
} else {

  $host = $_GET["host"];
  $port = $_GET["port"];

  list ($socket, $result) = openSocket($host, $port);

  if ($result == "ok") {
    $bytes_written = socketWrite($socket, "state\r\n");
    $string = "State: ".socketRead($socket);
  } else {
    $string = "TCS INTERFACE STOPPED<BR>\n";
  }
}
echo $string;
flush();

