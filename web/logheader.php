<?PHP?>
<html>
<?
include("definitions_i.php");
include("functions_i.php");
include("header_i.php");

$pwc_config = getConfigFile(SYS_CONFIG);
$num_pwc = $pwc_config["NUM_PWC"];


?>
<body>
  <script type="text/javascript">


    function changeLogWindow() {

      var i
      var length    // Length of the logs
      var type      // Type of log messages (all, warnings, errors)
      var machine   // Machine in question

      i = document.getElementById("active_machine").selectedIndex
      machine = document.getElementById("active_machine").options[i].value

      if (machine == "nexus") {
        Select_Value_Enable(document.getElementById("logtype"), "srv_apsr_tcs_interface", false);
        //Select_Value_Enable(document.getElementById("logtype"), "srv_dada_pwc", false);
        Select_Value_Enable(document.getElementById("logtype"), "srv_results", false);
        Select_Value_Enable(document.getElementById("logtype"), "srv_gain", false);
        Select_Value_Enable(document.getElementById("logtype"), "srv_aux", false);
      } else {
        Select_Value_Enable(document.getElementById("logtype"), "srv_apsr_tcs_interface", true);
        //Select_Value_Enable(document.getElementById("logtype"), "srv_dada_pwc", true);
        Select_Value_Enable(document.getElementById("logtype"), "srv_results", true);
        Select_Value_Enable(document.getElementById("logtype"), "srv_gain", true);
        Select_Value_Enable(document.getElementById("logtype"), "srv_aux", true);
      }

      i = document.getElementById("logtype").selectedIndex
      type  = document.getElementById("logtype").options[i].value

      i = document.getElementById("loglevel").selectedIndex
      level  = document.getElementById("loglevel").options[i].value

      i = document.getElementById("loglength").selectedIndex
      length = document.getElementById("loglength").options[i].value

      var newurl= "logwindow.php?machine="+machine+"&logtype="+type+"&loglength="+length+"&loglevel="+level
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
      <td align="center">
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
      <span id="log_type_span" style="visibility: visible;">Log: <select id="logtype" class="smalltext" onchange="changeLogWindow()">
        <option value="pwc">PWC</option>
        <option value="src">SRC</option>
        <option value="src_proc_mngr">&nbsp;&nbsp;&nbsp;&nbsp;Proc. Mngr</option>
        <option value="src_dspsr">&nbsp;&nbsp;&nbsp;&nbsp;Dspsr</option>
        <option value="sys">SYS</option>
        <option value="sys_obs_mngr">&nbsp;&nbsp;&nbsp;&nbsp;Obs Mngr</option>
        <option value="sys_arch_mngr">&nbsp;&nbsp;&nbsp;&nbsp;Archive Mngr</option>
        <option value="sys_aux_mngr">&nbsp;&nbsp;&nbsp;&nbsp;Aux. Mngr</option>
        <option value="sys_bg_mngr">&nbsp;&nbsp;&nbsp;&nbsp;BG Mngr</option>
        <option value="sys_monitor">&nbsp;&nbsp;&nbsp;&nbsp;Monitor</option>
        <option value="srv_apsr_tcs_interface">TCS Interface</option>
        <!--<option value="srv_dada_pwc">DADA PWC</option>-->
        <option value="srv_results">Server Results</option>
        <option value="srv_gain">Gain Controller</option>
        <option value="srv_aux">Auxiliary Manager</option>
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
      <td align="center" width=100%>
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

