<?PHP
$title = "APSR | Controls";

include("../definitions_i.php");
include("../functions_i.php");
?>
<html>
<?
include("../header_i.php");
$text = "APSR Controls";
include("../banner.php");

$config = getConfigFile(SYS_CONFIG);
$server_daemon_status = getServerStatus($config);

?>
<body>
<SCRIPT LANGUAGE="JavaScript">
function popUp(URL) {

  parent.output.document.location = URL

  //id = "ControlWindow"
  //eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=1,location=0,statusbar=1,menubar=0,resizable=1,width=600,height=300');");
}
// End -->
</script>

<table border=0 cellpadding=20>
<tr><td>

  <table border=0 class="datatable">
  <tr><th colspan=2>Server Daemons</th></tr>
  <tr> <td>TCS Interface</td><td><? echo statusLight($server_daemon_status["apsr_tcs_interface"]) ?></td> </tr>
  <tr> <td>PWC Monitor</td><td><? echo statusLight($server_daemon_status["pwc_monitor"]) ?></td> </tr>
  <tr> <td>SYS Monitor</td><td><? echo statusLight($server_daemon_status["sys_monitor"]) ?></td> </tr>
  <tr> <td>SRC Monitor</td><td><? echo statusLight($server_daemon_status["src_monitor"]) ?></td> </tr>
  <tr> <td>Results Manager</td><td><? echo statusLight($server_daemon_status["results_manager"]) ?></td> </tr>
  <tr> <td>Gain Controller</td><td><? echo statusLight($server_daemon_status["gain_controller"]) ?></td> </tr>
  <tr> <td>Aux Manager</td><td><? echo statusLight($server_daemon_status["aux_manager"]) ?></td> </tr>
  <tr>
    <td align="center" colspan=2> 
      <div class="btns">
        <a href="javascript:popUp('server_command.php?cmd=start_daemons')"  class="btn" > <span>Start</span> </a>
        <a href="javascript:popUp('server_command.php?cmd=stop_daemons')"  class="btn" > <span>Stop</span> </a>
      </div>
<!--
       <a href="javascript:popUp('server_command.php?cmd=start_daemons')" class="button" ><span></span><i>Start</i><b></b></a>
       <a href="javascript:popUp('server_command.php?cmd=stop_daemons')" class="button" ><span></span><i>Stop</i><b></b></a>
-->
    </td>
  </tr>
  </table>

</td><td>

<table border=0 cellspacing=2  class="datatable">
<tr><th colspan=<?echo ($config{"NUM_PWC"}+1)?>>Client Daemons</th><th colspan="<?echo $config["NUM_HELP"]?>">Helpers</th></tr>

<?
printClientStatus($config);
?>

<tr><td></td><td colspan=<?echo ($config{"NUM_PWC"}+1)?> align=center>
  <div class="btns">
    <a href="javascript:popUp('client_command.php?cmd=start_daemons&autoclose=1')"  class="btn" > <span>Start</span> </a>
    <a href="javascript:popUp('client_command.php?cmd=stop_daemons&autoclose=1')"  class="btn" > <span>Stop</span> </a>
    <a href="javascript:popUp('client_command.php?cmd=start_master_script&autoclose=1')"  class="btn" > <span>Start Master</span> </a>
    <a href="javascript:popUp('client_command.php?cmd=stop_master_script&autoclose=1')"  class="btn" > <span>Stop Master</span> </a>
  </div>

</td></tr>

</table>

</td></tr>
</table>

<table border=0 cellpadding=10>

<tr><td>

<table border=0 cellpadding=0 class="datatable">
<tr><th colspan=4>Admins Only</th></tr>
</table>
                                                                                                                                    
<table border=0 cellpadding=2>
  <tr><td>
    <div class="btns">
      <a href="javascript:popUp('client_command.php?cmd=start_pwcs')"  class="btn" > <span>Start PWCs</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=stop_pwcs')"  class="btn" > <span>Stop PWCs</span> </a>
    </div>

    <div class="btns">
      <a href="javascript:popUp('client_command.php?cmd=stop_pwc&arg=dspsr')"  class="btn" > <span>Kill Dspsr</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=stop_dfbs')"  class="btn" > <span>Stop DFB Sim</span> </a>
    </div>
    <div class="btns">
      <a href="javascript:popUp('server_command.php?cmd=reset_pwcc')"  class="btn" > <span>Reset PWCs</span> </a>
      <a href="javascript:popUp('server_command.php?cmd=restart_all')"  class="btn" > <span>Restart Everything</span> </a>
      
  </td></tr>
</table>
                                                                                                                                    
</td><td>

<table border=0 cellpadding=0 class="datatable">
<tr><th colspan=4>Client Controls</th></tr>
</table>  

<table border=0 cellpadding=2>
  <tr><td style="vertical-align: middle; text-align: right">Data Block</td><td>
    <div class="btns">
      <a href="javascript:popUp('client_command.php?cmd=reset_db')"  class="btn" > <span>Reset</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=init_db')"  class="btn" > <span>Init</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=destroy_db')"  class="btn" > <span>Destroy</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=get_db_info')"  class="btn" > <span>Info</span> </a>
    </div>
  </td></tr>

  <tr><td style="vertical-align: middle; text-align: right">Disk</td><td>
    <div class="btns">
      <a href="javascript:popUp('client_command.php?cmd=get_disk_info')"  class="btn" > <span>Info</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=clean_scratch')"  class="btn" > <span>Rm Scratch</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=clean_archives')"  class="btn" > <span>Rm Archives</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=clean_logs')"  class="btn" > <span>Rm Logs</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=clean_rawdata')"  class="btn" > <span>Rm Rawdata</span> </a>
    </div>
  </td></tr>

  <tr><td style="vertical-align: middle; text-align: right">Other</td><td>

    <div class="btns">
      <a href="javascript:popUp('client_command.php?cmd=get_load_info')"  class="btn" > <span>Load Info</span> </a>
      <a href="javascript:popUp('client_status.php')"  class="btn" > <span>Client Status</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=get_bin_dir')"  class="btn" > <span>Get Bin Dir</span> </a>
      <a href="javascript:popUp('client_command.php?cmd=daemon_info')"  class="btn" > <span>Daemon Info</span> </a>
    </div>

  </td></tr>
</table>  


</td></tr>
</table>

</body>
</html>

<?

function getServerStatus($config) {

  $daemons = array("pwc_monitor","sys_monitor","src_monitor","apsr_tcs_interface","results_manager","aux_manager","gain_controller");
  $control_dir = $config["SERVER_CONTROL_DIR"];
  $results = array();

  for ($i=0; $i<count($daemons); $i++) {
    if (file_exists($control_dir."/".$daemons[$i].".pid")) {
      $results[$daemons[$i]] = 1;
    } else {
      $results[$daemons[$i]] = 0;
    }

    $perl_daemon = "server_".$daemons[$i].".pl";

    $cmd = "ps aux | grep ".$perl_daemon." | grep -v grep > /dev/null";
    $lastline = system($cmd, $retval);
    if ($retval == 0) {
      $results[$daemons[$i]]++;
    }

  }
  return $results;
}

function printClientStatus($config) {

  $machines = "";
  $helpers = "";

  $mc = array();
  $om = array();
  $am = array();
  $pm = array();
  $bp = array();
  $m  = array();

  echo "<tr><td align=right></td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {

    $machines .= $config["PWC_".$i]." ";
    echo "<td align=center>".sprintf("%2d",$i)."</td>";

    $mc[$config["PWC_".$i]] = 0;
    $om[$config["PWC_".$i]] = 0;
    $am[$config["PWC_".$i]] = 0;
    $pm[$config["PWC_".$i]] = 0;
    $bp[$config["PWC_".$i]] = 0;
    $m[$config["PWC_".$i]] = 0;

  }

  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    $machines .= $config["HELP_".$i]." ";
    echo "<td align=center>".sprintf("%2d",$i)."</td>";
    $mc[$config["HELP_".$i]] = 0;
    $om[$config["HELP_".$i]] = 0;
    $am[$config["HELP_".$i]] = 0;
    $pm[$config["HELP_".$i]] = 0;
    $bp[$config["HELP_".$i]] = 0;
     $m[$config["HELP_".$i]] = 0;
  }

  echo "</tr>";

  chdir($config["SCRIPTS_DIR"]);
  $script = "source /home/apsr/.bashrc; ./client_command.pl \"daemon_info\" ".$machines;
  $string = exec($script, $output, $return_var);

  if ($return_var == 0) {

    for($i=0; $i<count($output); $i++) {

      $array = split(":",$output[$i],3);
      $machine = $array[0]; 
      $result = $array[1];
      $string = $array[2];

      if ( (strpos($string, "Could not connect to machine")) !== FALSE) {
	# We could not contact the master control script
      } else {
         $mc[$machine] = 2;
        $daemon_results = split(",",$string);
        for ($j=0; $j<count($daemon_results); $j++) {
          $arr = split(" ",$daemon_results[$j]);
          if ($arr[0] == "observation_manager")  { $om[$machine] = $arr[1]; }
          if ($arr[0] == "archive_manager")      { $am[$machine] = $arr[1]; }
          if ($arr[0] == "processing_manager")   { $pm[$machine] = $arr[1]; }
          if ($arr[0] == "background_processor") { $bp[$machine] = $arr[1]; }
          if ($arr[0] == "auxiliary_manager")    { $ap[$machine] = $arr[1]; }
          if ($arr[0] == "monitor")              { $m[$machine] = $arr[1]; }
        }
      }
    }
  }

  echo "<tr><td align=right>Master control</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td width=17 bgcolor=white>".statusLight($mc[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td width=17 bgcolor=white>".statusLight($mc[$config["HELP_".$i]])."</td>";
  }

  echo "</tr>\n";

  echo "<tr><td align=right>Obs. manager</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($om[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td width=17 bgcolor=white></td>";
  }
  echo "</tr>\n";

  echo "<tr><td align=right>Proc. manager</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($pm[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td bgcolor=white>".statusLight($pm[$config["HELP_".$i]])."</td>";
  }
  echo "</tr>\n";

  echo "<tr><td align=right>Archive mananger</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($am[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td bgcolor=white>".statusLight($am[$config["HELP_".$i]])."</td>";
  }
  echo "</tr>\n";

  echo "<tr><td align=right>Background Processor</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($bp[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td width=17 bgcolor=white></td>";
  }
  echo "</tr>\n";

  echo "<tr><td align=right>Aux. Manager</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($ap[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td width=17 bgcolor=white></td>";
  }

  echo "</tr>\n";
  
  echo "<tr><td align=right>Monitor</td>";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    echo "<td bgcolor=white>".statusLight($m[$config["PWC_".$i]])."</td>";
  }
  for ($i=0; $i<$config["NUM_HELP"]; $i++) {
    echo "<td width=17 bgcolor=white></td>";
  }
  echo "</tr>\n";

}


function statusLight($value) {

  if ($value == 2) {
    return "<img src=\"/images/green_light.png\" width=\"15\" height=\"15\" border=\"none\">";
  } else if ($value == 1) {
    return "<img src=\"/images/yellow_light.png\" width=\"15\" height=\"15\" border=\"none\">";
  } else {
    return "<img src=\"/images/red_light.png\" width=\"15\" height=\"15\" border=\"none\">";
  }
}
