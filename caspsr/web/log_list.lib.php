<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");


class log_list extends caspsr_webpage 
{

  var $server_daemons = array();
  var $client_daemons = array();
  var $server_log_info = array();
  var $pwcs = array();
  var $demuxers = array();
  var $srvs = array();


  function log_list()
  {
    caspsr_webpage::caspsr_webpage();
    array_push($this->css, "/caspsr/buttons.css");

    $inst = new caspsr();

    /* setup server daemons */
    $this->server_log_info = $inst->getServerLogInfo();
    $keys = array_keys($this->server_log_info);
    $this->server_daemons = array();
    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      if ($this->server_log_info[$k]["tag"] == "server") {
        $this->server_daemons[$k] = $this->server_log_info[$k];
      }
    }

    /* setup client daemons */
    $client_log_info = $inst->getClientLogInfo();
    $keys = array_keys($client_log_info);
    $this->client_daemons = array();
    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      if ($client_log_info[$k]["tag"] == "client") {
        $this->client_daemons[$k] = $client_log_info[$k];
      }
    }

    /* generate a list of machines */
    for ($i=0; $i<$inst->config["NUM_PWC"]; $i++) {
      array_push($this->pwcs, $inst->config["PWC_".$i]);
    }

    for ($i=0; $i<$inst->config["NUM_DEMUX"]; $i++) {
      if (!in_array($inst->config["DEMUX_".$i], $this->demuxers)) {
        array_push($this->demuxers, $inst->config["DEMUX_".$i]);
      }
    }

    array_push($this->srvs, "srv0");
 

  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function autoScroll() {
        var scroll
        scroll = document.getElementById("auto_scroll").checked
        log_window.auto_scroll = scroll
      }

      function Select_Value_Enable(SelectObject, Value, State) {
        for (index = 0; index < SelectObject.length; index++) {
          if (SelectObject[index].value == Value) {
            SelectObject[index].disabled = State;
            //alert("Setting " + SelectObject[index].value + ".disabled = "+State);
          }
        }
      }

      function clearLogWindow() {
        log_window.document.location = "ganglia_window.php";
      }

      function changeLogWindow() {

        var i
        var length    // Length of the logs
        var daemon    // The daemon/log
        var machine   // Machine in question
        var scroll
        var filter

        i = document.getElementById("source").selectedIndex
        machine = document.getElementById("source").options[i].value
  
        scroll = document.getElementById("auto_scroll").checked
        filter = document.getElementById("filter").value
    
        if (machine == "nexus") {
<?
          for ($i=0; $i<count($this->server_daemons); $i++) {
            if ($this->server_log_info[$this->server_daemons[$i]]["type"] == "servver") {
              echo "        Select_Value_Enable(document.getElementById(\"daemon\"), \"".$this->server_daemons[$i]."\", false);\n";
            }
          }
?>
        } else {
<?
          for ($i=0; $i<count($this->server_daemons); $i++) {
            if ($this->server_log_info[$this->server_daemons[$i]]["type"] == "servver") {
              echo "        Select_Value_Enable(document.getElementById(\"daemon\"), \"".$this->server_daemons[$i]."\", true);\n";
            }
          }
?>
        }

        i = document.getElementById("daemon").selectedIndex
        daemon  = document.getElementById("daemon").options[i].value

        //i = document.getElementById("loglevel").selectedIndex
        //level  = document.getElementById("loglevel").options[i].value
        level = document.getElementById("loglevel").value;
  
        i = document.getElementById("loglength").selectedIndex
        length = document.getElementById("loglength").options[i].value

        var newurl= "log_viewer.php?machine="+machine+"&daemon="+daemon+"&length="+length+"&level="+level+"&autoscroll="+scroll+"&filter="+filter;
        log_window.document.location = newurl

      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Logs");
?>
    <span>
      Source:
      <select id="source" onchange="changeLogWindow()">
        <option value="srv0">srv0</option>
        <option value="nexus">All GPUs</option>
<?
        for ($i=0;$i<count($this->pwcs);$i++) {
          echo "        <option value=".$this->pwcs[$i].">&nbsp;".$this->pwcs[$i]."</option>\n";
        }
?>
        <option value="demuxers">All Demuxers</option>
<?
        for ($i=0;$i<count($this->demuxers);$i++) {
          echo "        <option value=".$this->demuxers[$i].">&nbsp;".$this->demuxers[$i]."</option>\n";
        }
?>

      </select>
    </span>&nbsp;&nbsp;

    <!-- Select Log -->
    <span id="daemon_span" style="visibility: visible;">
    Log:
      <select id="daemon" onchange="changeLogWindow()">
        <option value="caspsr_pwc_monitor">PWC</option>
        <option value="caspsr_src_monitor">SRC</option>
<?
        foreach ($this->client_daemons as $key => $value) {
          if ($value["logfile"] == "nexus.src.log") {
            echo "        <option value=\"".$key."\">&nbsp;&nbsp;&nbsp;&nbsp;".$value["name"]."</option>\n";
          }
        }
?>
        <option value="caspsr_sys_monitor">SYS</option>
<?
        foreach ($this->client_daemons as $key => $value) {
          if ($value["logfile"] == "nexus.sys.log") {
            echo "        <option value=\"".$key."\">&nbsp;&nbsp;&nbsp;&nbsp;".$value["name"]."</option>\n";
          }
        }

?>
        <option value="srv" disabled=true visible=true>SERVER ONLY</option>
<?
        foreach ($this->server_log_info as $key => $value) {
          if ($value["tag"] == "server") {
            echo "        <option value=\"".$key."\">&nbsp;&nbsp;&nbsp;&nbsp;".$value["name"]."</option>\n";
          }
        }
?>


      </select>
    </span>&nbsp;&nbsp;

    <!-- Select log level -->
    <input type='hidden' id='loglevel' value='all'>

    <!-- Select log period -->
    <span id="log_length_span" style="visibility: visible;">
      Period:
      <select id="loglength" onchange="changeLogWindow()">
        <option value=1>1 hr</option>
        <option value=3>3 hrs</option>
        <option value=6 selected>6 hrs</option>
        <option value=12>12 hrs</option>
        <option value=24>24 hrs</option>
        <option value=168>7 days</option>
        <option value=334>14 days</option>
        <option value=all>Ever</option>
      </select>
    </span>&nbsp;&nbsp;

    <!-- Auto Scroll -->
    <span>
      <input type="checkbox" id="auto_scroll" onchange="autoScroll()">Auto scroll?
    </span>&nbsp;&nbsp;

    <!-- Filter Logs -->
    <span>
      Filter: <input type="text" id="filter" onchange="changeLogWindow()">
    </span>&nbsp;&nbsp;

    <!-- Reload button -->
    <span class="btns">
      <input type="button" onClick='changeLogWindow()' value='Show'>
      <input type="button" onClick='clearLogWindow()' value='Clear'>
    </span>

    <!--
    <table>
    <tr>
      <td align="center">
          <table cellpadding=0 cellspacing=0><tr><td>
          <div class="btns" align="center">
            <a href="javascript:changeLogWindow()"  class="btn" > <span>Reload Logs</span> </a>
            <a href="ganglia_window.php" target="logwindow" class="btn" > <span>Ganglia Monitor</span> </a>
          </div>
          </center>
          </td></tr></table>
        </td>
      </tr>
    </table>
    -->
    <br>
    <br>
    <iframe name="log_window" src="ganglia_window.lib.php" frameborder=0 width=100% height=320px>
    </iframe>

<?
    $this->closeBlockHeader();
  }
}
