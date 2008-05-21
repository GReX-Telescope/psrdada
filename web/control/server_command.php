<?PHP
include("../functions_i.php");
include("../definitions_i.php");
?>
<html>
<?
include("../header_i.php");

$cmd         = $_GET["cmd"];
$config      = getConfigFile(SYS_CONFIG);
$host        = $config["SERVER_HOST"];
$control_dir = $config["SERVER_CONTROL_DIR"];

chdir($config["SCRIPTS_DIR"]);

$return_val = 0;

$server_daemons = split(" ",$config["SERVER_DAEMONS"]);
for ($i=0; $i<count($server_daemons);$i++) {
  $server_names[$i] = str_replace("_", " ", $server_daemons[$i]);
}

?>
  <script type="text/javascript">
    function finish(){
      parent.control.location.href=parent.control.location.href;
    }
  </script>
<body>
<center>
<?

/* Start the server side daemons */
if ($cmd == "start_daemons") {

?>

<table class="datatable">
  <tr><th colspan=3>Starting Server Daemons</th></tr>
  <tr><th>Daemon</th><th>Result</th><th>Messages</th></tr>
<?
  for ($i=0; $i<count($server_daemons); $i++) {
    $return_val += startDaemon($server_names[$i], $server_daemons[$i]);
    flush();
  }

/*
  $return_val += startDaemon("PWC Monitor", "pwc_monitor");
  $return_val += startDaemon("SYS Monitor", "sys_monitor");
  $return_val += startDaemon("SRC Monitor", "src_monitor");
  $return_val += startDaemon("Results Manager", "results_manager");
  $return_val += startDaemon("Aux Manager", "aux_manager");
  $return_val += startDaemon("Gain Controller", "gain_controller");
  $return_val += startDaemon("TCS Interface", "tcs_interface");
*/
?>

</table>
<?

} else if ($cmd == "stop_daemons") {
?>

<table class="datatable">
  <tr><th colspan=3>Stopping Server Daemons</th></tr>
  <tr><th>Daemon</th><th>Result</th><th>Messages</th></tr>

<?

  system("touch ".$control_dir."/quitdaemons");

  for ($i=0; $i<count($server_daemons); $i++) {
    $return_val += waitForDaemon($server_names[$i], $server_daemons[$i], $control_dir);
    flush();
  }
/*
  $return_val += waitForDaemon("PWC Monitor", "pwc_monitor", $control_dir);
  flush();
  $return_val += waitForDaemon("SYS Monitor", "sys_monitor", $control_dir);
  flush();
  $return_val += waitForDaemon("SRC Monitor", "src_monitor", $control_dir);
  flush();
  $return_val += waitForDaemon("Results Manager", "results_manager", $control_dir);
  flush();
  $return_val += waitForDaemon("Aux Manager", "aux_manager", $control_dir);
  flush();
  $return_val += waitForDaemon("Gain Controller", "gain_controller", $control_dir);
  flush();
  $return_val += waitForDaemon("TCS Interface", "tcs_interface", $control_dir);
  flush();
 */

  unlink($control_dir."/quitdaemons");

?>
  </table>
<?

} else if ($cmd == "reset_pwcc") {

  echo "opening socket to ".$config["PWCC_HOST"].":".$config["PWCC_PORT"]."<BR>\n";

  flush();

  list($socket, $result) = openSocket($config["PWCC_HOST"], $config["PWCC_PORT"], 10);

  if ($result == "ok") {

    echo "Socket open<BR>\n";
    flush();

    echo "Read: ".socketRead($socket)."<BR>\n";;
    flush();

    socketWrite($socket, "reset\r\n");
    flush();

    echo "Read: ".socketRead($socket)."<BR>\n";;
    flush();

  } else {

    echo "Could not open socket to ".$config["PWCC_HOST"].":".$config["PWCC_PORT"]."<BR>\n";

  }

} else if ($cmd == "restart_all") {

?>
<table class="datatable" width=60%>
  <tr><th colspan=1>Restarting Instrument: <?echo $config["INSTRUMENT"]?></th></tr>

<?
  flush();

  $script_name = "dada_reconfigure.pl -e apsr";
  echo "  <tr style=\"background: white;\">\n";
  echo "    <td align=\"left\">\n";
  $script = "source /home/apsr/.bashrc; ".$script_name." 2>&1";
  $string = exec($script, $output, $return_var);
  for ($i=0; $i<count($output); $i++) {
    echo $output[$i]."<BR>";
  }
  echo "    </td>\n";
  echo "  </tr>\n";
  echo "</table>\n";

} else {

  $result = "fail";
  $response = "Unrecognized command";

}
flush();
sleep(1);

if (!$return_val) {
?>
<script type="text/javascript">finish()</script>
<? } ?>
</center>
</body>
</html>

<?

function startDaemon($title, $name) {

  $script_name = "./server_".$name.".pl";

  echo "  <tr style=\"background: white;\">\n";
  echo "    <td>".$title."</td>\n";
  $script = "source /home/apsr/.bashrc; ".$script_name." 2>&1";
  $string = exec($script, $output, $return_var);
  echo "    <td>";
  echo ($return_var == 0) ? "OK" : "FAIL";
  echo "</td>\n";
  echo "    <td>";
  for ($i=0;$i<count($output);$i++) {
    echo $output[$i]."<BR>\n";
  }
  echo "</td>\n";
  echo "  </tr>\n";

  return $return_var;

}

function waitForDaemon($title, $name, $dir) {

  $cmd = "ps auxwww | grep \"perl ./".$name."\" | grep -v grep";

  $pid_file = $dir."/".$name.".pid";

  $daemon_running = 1;
  $nwait = 5;

  while (($daemon_running) && ($nwait > 0)) {

    $last_line = system($cmd, $ret_val);

    if (((file_exists($pid_file)) || ($ret_val == 0)) && ($nwait < 5)) {
      echo "<tr style=\"background: white;\"><td>".$title."</td><td>FAIL</td><td>still running...</td>";
    } else {
      echo "<tr style=\"background: white;\"><td>".$title."</td><td>OK</td><td>Daemon Exited</td>";
      $daemon_running = 0; 
    }
    sleep(1);
    $nwait--;
  }

  if ($daemon_running) {
    return 1;
  } else {
    return 0;
  }

}
