<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class perf_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $num_rx_east = 44;

  var $num_rx_west = 44;

  var $mod_colours = array("B", "G", "Y", "R");

  var $pfb_list = array();

  var $rx_per_pfb = 4;

  var $update_secs = 5;

  var $pfb_to_rx = array();

  var $qsos = array ();

  var $psrs = array ();

  function perf_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MPSR Module Performance Monitor";

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();

    array_push ($this->qsos, "/home/vravi/2015-05-27-22:53:16.quasar");
    array_push ($this->qsos, "/home/observer/performance/2015-07-15-01:32:35/2015-07-15-02:22:17.quasar");

    array_push ($this->psrs, "/home/vravi/2015-05-12-06:51:09.pulsar");
    array_push ($this->psrs, "/home/observer/performance/2015-07-15-01:32:35/2015-07-15-01:32:35.pulsar");

    // open the configuration file
    $handle = fopen( $this->inst->config["CONFIG_DIR"]."/mopsr_signal_paths.txt", "r");
    if (!$handle)
      return $config;

    while (!feof($handle))
    {
      $buffer = fgets($handle, 4096);
      list ($mod_id, $pfb_id, $pfb_input) = preg_split('/\s+/', $buffer);
      $this->pfb_to_rx[$pfb_id."_".$pfb_input] = $mod_id;
    }
    fclose($handle);

    for ($i=1; $i<=11; $i++)
    {
      array_push($this->pfb_list, sprintf("EG%02d",$i));
      array_push($this->pfb_list, sprintf("WG%02d",$i));
    }
  }

  function javaScriptCallback()
  {
    return "perf_monitor_request();";
  }

  function printJavaScriptHead()
  {

?>
    <script type='text/javascript'>  

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_perf_monitor_request(xml_request) 
      {
        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var perf_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("perf_monitor_error").innerHTML = "[" + perf_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {
            }

            var bays = xmlObj.getElementsByTagName ("bay");

            var i, j;
            // parse XML for each RX ID
            for (i=0; i<bays.length; i++)
            {
              var bay = bays[i];
              var bay_id = bay.getAttribute("id");

              // boards will have params and plots
              var nodes = bay.childNodes;
              for (j=0; j<nodes.length; j++)
              {
                var node = nodes[j]
                if (node.nodeType == 1)
                {
                  span_col = node.getAttribute("colour")
                  span_id = bay_id + "_" + node.nodeName;
                  document.getElementById(span_id).innerHTML = node.childNodes[0].nodeValue;
                  document.getElementById(span_id).bgColor = span_col; 
                }
              }
            }
          }
        }
      }
                  
      function perf_monitor_request() 
      {
        var qso_id = document.getElementById ("qso").selectedIndex;
        var psr_id = document.getElementById ("psr").selectedIndex;

        var psr_file = document.getElementById ("psr").options[psr_id].value;
        var qso_file = document.getElementById ("qso").options[qso_id].value;

        //var url = "perf_monitor.lib.php?update=true&qso_file=/home/vravi/2015-05-27-22:53:16.quasar&psr_file=/home/vravi/2015-05-12-06:51:09.pulsar";
        var url = "perf_monitor.lib.php?update=true&qso_file=" + qso_file + "&psr_file=" + psr_file;

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_perf_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type="text/css">
    
      table.perf_monitor {
        border-spacing: 4;
      }

      table.perf_monitor td {
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 1px;
        padding-right: 1px;
      }

      table.perf_monitor img {
        margin-left:  2px;
        margin-right: 2px;
      }

      th {
        padding-right: 10px;
        padding-left: 10px;
      }

    </style>

<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Module Performance Monitor&nbsp;&nbsp;&nbsp;<span id='perf_monitor_error'></span>");
    list ($xres, $yres) = split("x", $this->img_size);

?>
    <table>

      <tr><td valign='top'>

    <table border=1 class='perf_monitor'>
      <tr><th class='pad_right'>Arm</th><th colspan=12>East</th><th></th><th colspan=12>West</th></tr>
      <tr><th class='pad_right'>Bay</th>
          <th colspan='3' width='90px'>B</th><th  colspan='3' width='90px'>G</th><th  colspan='3' width='90px'>Y</th><th  colspan='3' width='90px'>R</th>
          <th></th>
          <th colspan='3' width='90px'>B</th><th  colspan='3' width='90px'>G</th><th  colspan='3' width='90px'>Y</th><th  colspan='3' width='90px'>R</th>
      </tr>
      <tr>
        <th></th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
        <th></th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
        <th>P</th><th>Q</th><th>H</th>
      </tr>
<?
    $num_rows= max($this->num_rx_east, $this->num_rx_west);

    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("%02d", $irow + 1);

      echo "<tr>\n";
      echo "<th>".$bay."</th>";

      $rx_id = "E".$bay;
      # east bays
      foreach ($this->mod_colours as $irx)
      {
        $mod_id = $rx_id."-".$irx;
        echo "<td width='30px' id='".$mod_id."_psr_perf'>1.0</td>";
        echo "<td width='30px' id='".$mod_id."_corr_perf'>0.7</td>";
        echo "<td width='30px' id='".$mod_id."_hg_power'>4.3</td>\n";
      }

      echo "<td>&nbsp;</td>\n";

      # west bays
      $rx_id = "W".$bay;
        
      foreach ($this->mod_colours as $irx)
      {
        $mod_id = $rx_id."-".$irx;
        echo "<td width='30px' id='".$mod_id."_psr_perf'>1.0</td>";
        echo "<td width='30px' id='".$mod_id."_corr_perf'>0.7</td>";
        echo "<td width='30px' id='".$mod_id."_hg_power'>4.3</td>\n";
      }

      echo "</tr>\n";
    }
    echo "</table>\n";
?>

  </td>
  <td valign='top'>

  <table border=0>
    <tr><th colspan=2>Legend</th></tr>
    <tr><td>P</td><td>Pulsar Score </td></tr>
    <tr><td>Q</td><td>Quasar / Correlation Score </td></tr>
    <tr><td>H</td><td>Histogram FWHM counts</td></tr>
    <tr><td>HG</td><td bgcolor=red> &lt; 2</td></tr>
    <tr><td>HG</td><td bgcolor=yellow> &lt; 10</td></tr>
    <tr><td>QSO + PSR</td><td bgcolor=red> &lt; 0.05</td></tr>
    <tr><td>PSR + PSR</td><td bgcolor=yellow> &lt; 0.5</td></tr>
  </table>

  <table>

    <tr>
      <th>QSO</th>
      <td>
        <select name="qso" id="qso" onChange="perf_monitor_request()">
<?
          for ($i=0; $i<count($this->qsos); $i++)
          {
            $qso = $this->qsos[$i];
            $qso_name = basename($qso);
            if ((count($this->qsos)-1) == $i)
              echo "<option value='".$qso."' selected>".$qso_name."</option>\n";
            else
              echo "<option value='".$qso."'>".$qso_name."</option>\n";
          }
?>
        </select>
      </td>
    </tr>
    <tr>
      <th>PSR</th>
      <td>
        <select name="psr" id="psr" onChange="perf_monitor_request()">
<?
          for ($i=0; $i<count($this->psrs); $i++)
          {
            $psr = $this->psrs[$i];
            $psr_name = basename($psr);
            if ((count($this->psrs)-1) == $i)
              echo "<option value='".$psr."' selected>".$psr_name."</option>\n";
            else
              echo "<option value='".$psr."'>".$psr_name."</option>\n";
          }
?>
        </select>
      </td>
    </tr>
  </table>

  </td></tr>
  <table>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    # read the stats files for each PFB in the server stats dir
    $histogram_levels = array();
    foreach ($this->pfb_list as $pfb)
    {
      $cmd = "ls -1 ".$this->inst->config["SERVER_UDP_MONITOR_DIR"]."/2???-??-??-??:??:??.".$pfb.".stats | sort -n | tail -n 1";
      $array = array();
      $stats_file = exec ($cmd, $array, $rval);
      $hg_stats = getConfigFile($stats_file); 
      foreach (array_keys($hg_stats) as $i)
      {
        $pfb_output = $pfb."_".$i;
        $rx = $this->pfb_to_rx[$pfb_output];
        $histogram_levels[$rx] = $hg_stats[$i];
      }
    }

    # read the psr file
    $psr_perf = getConfigFile($get["psr_file"]);

    # read the corr file
    $qso_perf = getConfigFile($get["qso_file"]);

    # now prepare the reply

    $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml .= "<perf_monitor_update>";
    $xml .=   "<error></error>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<timestamp>2014-02-01-02:34:41</timestamp>";

    $num_rows= max($this->num_rx_east, $this->num_rx_west);
    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("%02d", $irow + 1);

      $rx_id = "E".$bay;
      foreach ($this->mod_colours as $irx)
      {
        $mod_id = $rx_id."-".$irx;
        $qso = sprintf ("%3.2f",$qso_perf[$mod_id]);
        $psr = sprintf ("%3.2f", $psr_perf[$mod_id]);
        $hgp = $histogram_levels[$mod_id];

        $hg_col = "green";
        if ($hgp < 10)
          $hg_col = "yellow";
        if ($hgp < 2)
          $hg_col = "red";

        $qso_col = "green";
        if (floatval($qso) < 0.5)
          $qso_col = "yellow";
        if (floatval($qso) < 0.05)
          $qso_col = "red";

        $psr_col = "green";
        if (floatval($psr) < 0.5)
          $psr_col = "yellow";
        if (floatval($psr) < 0.05)
          $psr_col = "red";

        $xml .= "<bay id='".$mod_id."'>";
        $xml .= "<corr_perf colour='".$qso_col."'>".$qso."</corr_perf>";
        $xml .= "<psr_perf colour='".$psr_col."'>".$psr."</psr_perf>";
        $xml .= "<hg_power colour='".$hg_col."'>".$hgp."</hg_power>";
        $xml .= "</bay>";
      }

      $rx_id = "W".$bay;
      foreach ($this->mod_colours as $irx)
      {
        $mod_id = $rx_id."-".($irx);
        $qso = sprintf ("%3.2f",$qso_perf[$mod_id]);
        $psr = sprintf ("%3.2f", $psr_perf[$mod_id]);
        $hgp = $histogram_levels[$mod_id];

        $hg_col = "green";
        if ($hgp < 10)
          $hg_col = "yellow";
        if ($hgp < 2)
          $hg_col = "red";

        $qso_col = "green";
        if (floatval($qso) < 0.5)
          $qso_col = "yellow";
        if (floatval($qso) < 0.05)
          $qso_col = "red";

        $psr_col = "green";
        if (floatval($psr) < 0.5)
          $psr_col = "yellow";
        if (floatval($psr) < 0.05)
          $psr_col = "red";

        $xml .= "<bay id='".$mod_id."'>";
        $xml .= "<corr_perf colour='".$qso_col."'>".$qso."</corr_perf>";
        $xml .= "<psr_perf colour='".$psr_col."'>".$psr."</psr_perf>";
        $xml .= "<hg_power colour='".$hg_col."'>".$hgp."</hg_power>";
        $xml .= "</bay>";
      }
    }
    $xml .= "</perf_monitor_update>";

    header('Content-type: text/xml');
    echo $xml;
  }
}

handleDirect("perf_monitor");
