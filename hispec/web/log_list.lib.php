<?PHP

include_once("hispec.lib.php");
include_once("hispec_webpage.lib.php");

class log_list extends hispec_webpage 
{

  var $log_files = array();

  function log_list()
  {
    hispec_webpage::hispec_webpage();

    $inst = new hispec();

    $log_info = $inst->serverLogInfo();
    foreach ($log_info as $key => $value) 
    {
      if (strpos($inst->config["SERVER_DAEMONS"], $key) !== FALSE)
        $this->log_files[$value["name"]] = $value["logfile"];
    }

    $log_info = $inst->clientLogInfo();
    foreach ($log_info as $key => $value)
    {
      if (strpos($inst->config["CLIENT_DAEMONS"], $key) !== FALSE)
        $this->log_files[$value["name"]] = $value["logfile"];
    }
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function autoScroll()
      {
        var scroll
        scroll = document.getElementById("auto_scroll").checked
        log_window.auto_scroll = scroll
      }

      function clearLogWindow()
      {
        log_window.document.location = "ganglia_window.lib.php";
      }

      function changeLogWindow()
      {

        var i
        var length    // Length of the logs
        var logfile      // The daemon/log
        var scroll
        var filter

        scroll = document.getElementById("auto_scroll").checked
        filter = document.getElementById("filter").value

        i = document.getElementById("logfile").selectedIndex
        logfile = document.getElementById("logfile").options[i].value

        level = document.getElementById("loglevel").value;
  
        i = document.getElementById("loglength").selectedIndex
        length = document.getElementById("loglength").options[i].value

        var newurl= "log_viewer.php?logfile="+logfile+"&length="+length+"&level="+level+"&autoscroll="+scroll+"&filter="+filter;
        //alert(newurl);
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
    <span id="daemon_span" style="visibility: visible;">
    Log: 
      <select id="logfile" onchange="changeLogWindow()">
<?
    foreach ($this->log_files as $name => $file) 
      echo "        <option value='".$file."'>".$name."</option>\n";
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
      Filter: <input type="text" id="filter" size='16' onchange="changeLogWindow()">
    </span>&nbsp;&nbsp;

    <!-- Reload button -->
    <span class="btns">
      <input type="button" onClick='changeLogWindow()' value='Show'>
      <input type="button" onClick='clearLogWindow()' value='Clear'>
    </span>

    <br>
    <br>
    <iframe name="log_window" src="ganglia_window.lib.php" frameborder=0 width=100% height=450px>
    </iframe>

<?
    $this->closeBlockHeader();
  }
}
handleDirect("log_list");

