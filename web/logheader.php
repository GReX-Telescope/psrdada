<?PHP?>
<html>
<?
include("definitions_i.php");
include("functions_i.php");
include("header_i.php");

$pwc_config = getConfigFile(SYS_CONFIG);
$num_pwc = $pwc_config["NUM_PWC"];

$server_daemons = split(" ",$pwc_config["SERVER_DAEMONS"]);
$server_daemons_to_exclude = array("pwc_monitor", "sys_monitor", "src_monitor");
$server_daemon_names = getServerDaemonNames();

$client_daemons = split(" ",$pwc_config["CLIENT_DAEMONS"]);
if ($pwc_config["INSTRUMENT"] == "apsr") {
  array_push($client_daemons, "gain_controller");
}
$client_daemon_names = getClientDaemonNames();
$client_daemon_types = getClientDaemonTypes();

$server_daemons_count = count($server_daemons);
for ($i=0; $i<$server_daemons_count;$i++) {
  if (in_array($server_daemons[$i], $server_daemons_to_exclude)) {
    array_splice($server_daemons, $i,1);
    $i--;
    $server_daemons_count = count($server_daemons);
  } 
}

?>
<body>
  <script type="text/javascript">

    function autoScroll() {
      var scroll
      scroll = document.getElementById("auto_scroll").checked
      parent.logwindow.auto_scroll = scroll
    }

    function changeLogWindow() {

      var i
      var length    // Length of the logs
      var daemon    // The daemon/log
      var machine   // Machine in question
      var scroll
      var filter

      i = document.getElementById("active_machine").selectedIndex
      machine = document.getElementById("active_machine").options[i].value

      scroll = document.getElementById("auto_scroll").checked
      filter = document.getElementById("filter").value

      if (machine == "nexus") {
<?
        for ($i=0; $i<count($server_daemons); $i++) {
          echo "Select_Value_Enable(document.getElementById(\"daemon\"), \"".$server_daemons[$i]."\", false);\n";
        }
?>
      } else {
<?
        for ($i=0; $i<count($server_daemons); $i++) {
          echo "Select_Value_Enable(document.getElementById(\"daemon\"), \"".$server_daemons[$i]."\", true);\n";
        }
?>
      }

      i = document.getElementById("daemon").selectedIndex
      daemon  = document.getElementById("daemon").options[i].value

      i = document.getElementById("loglevel").selectedIndex
      level  = document.getElementById("loglevel").options[i].value

      i = document.getElementById("loglength").selectedIndex
      length = document.getElementById("loglength").options[i].value

      var newurl= "logwindow.php?machine="+machine+"&daemon="+daemon+"&loglength="+length+"&loglevel="+level+"&autoscroll="+scroll+"&filter="+filter;
      parent.logwindow.document.location = newurl
  
    }

    function Select_Value_Enable(SelectObject, Value, State) {
      for (index = 0; index < SelectObject.length; index++) {
        if (SelectObject[index].value == Value) {
          SelectObject[index].disabled = State;
          //alert("Setting " + SelectObject[index].value + ".disabled = "+State);
        }
      }
    }

  </script>

  <table border=0 width="100%" height="100%" cellpadding=3 cellspacing=0>
    <tr valign="center">
      <td align="center" colspan=3>
      <!-- Select Node -->
       Node: <select id="active_machine" class="smalltext" onchange="changeLogWindow()">
               <option value="nexus">nexus</option>
<?
for ($i=0;$i<$num_pwc;$i++) {
  echo "              <option value=".$pwc_config["PWC_".$i].">".$pwc_config["PWC_".$i]."</option>";
}
?>
             </select>&nbsp;&nbsp;&nbsp;&nbsp;

      <!-- Select log type -->
      <span id="daemon_span" style="visibility: visible;">Log: <select id="daemon" class="smalltext" onchange="changeLogWindow()">
        <option value="pwc">PWC</option>
        <option value="src">SRC</option>
        <option value="processor">&nbsp;&nbsp;&nbsp;&nbsp;Processor</option>
<?
        for ($i=0; $i<count($client_daemons); $i++) {
          if ($client_daemon_types[$client_daemons[$i]] == "src") 
            echo "        <option value=\"".$client_daemons[$i]."\">&nbsp;&nbsp;&nbsp;&nbsp;".$client_daemon_names[$client_daemons[$i]]."</option>\n";
        }
?>
        <option value="sys">SYS</option>
<?
        for ($i=0; $i<count($client_daemons); $i++) {
          if ($client_daemon_types[$client_daemons[$i]] == "sys")
            echo "        <option value=\"".$client_daemons[$i]."\">&nbsp;&nbsp;&nbsp;&nbsp;".$client_daemon_names[$client_daemons[$i]]."</option>\n";
        }

?>
        <option value="srv" disabled=true visible=true>SERVER ONLY</option>
<?
        for ($i=0; $i<count($server_daemons); $i++) {
          echo "        <option value=\"".$server_daemons[$i]."\">&nbsp;&nbsp;&nbsp;&nbsp;".$server_daemon_names[$server_daemons[$i]]."</option>\n";
        }
?>

      </select></span>&nbsp;&nbsp;&nbsp;&nbsp;

      <!-- Select log level -->
      <span id="log_level_span" style="visibility: visible;">Level: <select id="loglevel" class="smalltext" onchange="changeLogWindow()">
        <option value="all">All</option>
        <option value="error">Errors</option>
        <option value="warn">Warnings</option>
      </select></span>&nbsp;&nbsp;&nbsp;&nbsp;

      <!-- Select log period -->
      <span id="log_length_span" style="visibility: visible;">Period: <select id="loglength" class="smalltext" onchange="changeLogWindow()">
        <option value=1>1 hour</option>
        <option value=3>3 hours</option>
        <option value=6 selected>6 hours</option>
        <option value=12>12 hours</option>
        <option value=24>24 hours</option>
        <option value=168>7 days</option>
        <option value=334>14 days</option>
        <option value=all>Ever</option>
      </select></span>
    </td>
      
  </tr>

  <tr>
    <td class="smalltext" valign="middle"><input type="checkbox" id="auto_scroll" onchange="autoScroll()">Auto scroll?</input></td>
    <td class="smalltext" valign="middle">Filter: <input type="text" id="filter" onchange="changeLogWindow()"></input></td>
    <td align="center">
        <table cellpadding=0 cellspacing=0><tr><td>
        <div class="btns" align="center">
          <a href="javascript:changeLogWindow()"  class="btn" > <span>Reload Logs</span> </a>
          <a href="machine_status.php?machine=nexus" target="logwindow" class="btn" > <span>System Status</span> </a>
        </div>
        </center>
        </td></tr></table>
      </td>
    </tr>
  </table>
</body>
</html>

