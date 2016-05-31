<?PHP

ini_set('display_errors',1);
error_reporting(E_ALL);

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class machine_summary extends mopsr_webpage 
{
  // mapping of AQs, BF, BPs to HOST
  var $aqs = array();
  var $bfs = array();
  var $bps = array();
  
  // mapping of HOST to type 
  var $machines = array();
  var $mappings = array();

  var $config = array();

  var $verbose = false;

  function machine_summary()
  {
    mopsr_webpage::mopsr_webpage();

    $this->callback_freq = 10000;
    array_push($this->ejs, "/js/prototype.js");
    array_push($this->ejs, "/js/jsProgressBarHandler.js");

    $inst = new mopsr();

    $this->config = $inst->config;
    $aq_cfg = $inst->configFileToHash(AQ_FILE);
    $bf_cfg = $inst->configFileToHash(BF_FILE);
    $bp_cfg = $inst->configFileToHash(BP_FILE);
    $bf_x_cfg = $inst->configFileToHash(CNR_FILE);
    $bp_x_cfg = $inst->configFileToHash(BP_CNR_FILE);

    // generate a list of machines
    for ($i=0; $i<$aq_cfg["NUM_PWC"]; $i++) 
    {
      $host = $aq_cfg["PWC_".$i];
      $host = str_replace("mpsr-", "", $host);
      $this->aqs[$i] = $host;
      if (!array_key_exists($host, $this->machines))
        $this->machines[$host] = array();
      if (!in_array("aq", $this->machines[$host]))
        array_push ($this->machines[$host], "aq");
      $this->mappings["aq".sprintf("%02d",$i)] = $host;
    }

    for ($i=0; $i<$bf_cfg["NUM_BF"]; $i++)
    {
      $host = $bf_cfg["BF_".$i];
      $host = str_replace("mpsr-", "", $host);
      $this->bfs[$i] = $host;
      if (!array_key_exists($host, $this->machines))
        $this->machines[$host] = array();
      if (!in_array("bf", $this->machines[$host]))
        array_push ($this->machines[$host], "bf");
      $this->mappings["bf".sprintf("%02d",$i)] = $host;
    }

    for ($i=0; $i<$bp_cfg["NUM_BP"]; $i++)
    {
      $host = $bp_cfg["BP_".$i];
      $host = str_replace("mpsr-", "", $host);
      $this->bps[$i] = $host;
      if (!array_key_exists($host, $this->machines))
        $this->machines[$host] = array();
      if (!in_array("bp", $this->machines[$host]))
        array_push ($this->machines[$host], "bp");
      $this->mappings["bp".sprintf("%02d",$i)] = $host;
    }

    list ($server_host, $server_domain) = explode(".", $inst->config["SERVER_HOST"], 2);
    $server_host = str_replace("mpsr-", "", $server_host);
    $this->machines[$server_host] = array("server");
    $this->mappings["srv"] = $host;
  }

  function javaScriptCallback()
  {
    return "machine_summary_request();";
  }

  function printJavaScriptHead()
  {

?>
    <style type="text/css">
      table.machine_summary th {
        text-align: left;
      }
      table.machine_summary td { 
        padding-bottom: 1px;
        padding-top: 1px;
        border-style: none none solid none;
        border-color: #cccccc;
        border-width: 0px 0px 1px 0px;
        font-size: 8pt;
      }
      table.machine_summary span { 
        font-size: 8pt;
      }
      td.gap {
        padding-right: 20px;
      }
      td.gap_line {
        padding-right: 20px;
      }

      a.warning {
        color: #995c00;
        underline: none;
      }
      a.error {
        color: #0000ff;
        underline: none;
      }
      a.ok {
        color: #00ff00;
        underline: none;
      }

    </style>

    <script type='text/javascript'>  

      var current_machines = <?echo count($this->machines)?>;

      var host_mappings = {<?
      foreach ($this->mappings as $map => $host)
      {
        echo $map.":'".$host."',";
      }
      echo "junk:'Junk'";
?>}


      function rtrim(str, chars) {
        chars = chars || "\\s";
        return str.replace(new RegExp("[" + chars + "]+$", "g"), "");
      }

      function PadDigits(n, totalDigits) 
      { 
        var n = n.toString(); 
        var pd = ''; 
        var k = 0;
        if (totalDigits > n.length) 
        { 
          for (k=0; k < (totalDigits-n.length); k++) 
          { 
            pd += '0'; 
          } 
        } 
        return pd + n.toString(); 
      }
<?
      $keys = array_keys($this->machines);
      echo "      var machines = new Array('".$keys[0]."'";
      for ($i=1; $i<count($keys); $i++)
        echo ",'".$keys[$i]."'";
      echo ")\n";
?>

      // Set all the imgs, links and messages not in the excluded array to green
      function resetOthers(excluded) 
      {
        var j = 0;
        var log_length = "6";
        try 
        {
          j = document.getElementById("loglength").selectedIndex;
          log_length = document.getElementById("loglength").options[j].value
        } 
        catch (e)
        {
          // silently ignore
        }

        for (i=0; i<machines.length; i++) 
        {
          var machine = machines[i];
          if (excluded.indexOf(machine) == -1) 
          {
            document.getElementById(machine+"_messages").innerHTML = "&nbsp;";
            document.getElementById(machine+"_img").src = "/images/green_light.png";
          }
        }
      }

      function pad(num, size) {
          var s = num+"";
          while (s.length < size) s = "0" + s;
          return s;
      }

      function setHostNotConnected(host)
      {
        var pb;

        document.getElementById(host + "_load_value").innerHTML = "&nbsp;--";
        document.getElementById(host + "_temperature_value").innerHTML = "--";

        pb = eval(host + "_load"); 
        pb.setPercentage(0.0);
        pb = eval(host + "_disk"); 
        pb.setPercentage(0.0);

      }

      function setAllNotConnected() 
      {
        for (i=0; i<machines.length; i++) {
          setHostNotConnected(machines[i]);
        } 
      }

      function handle_machine_summary_request(xml_request) 
      {
        var children;
        var progress_bar;
        var log_length = 6;

        if (xml_request.readyState == 4) 
        {
          var xmlDoc = xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            // check that data to the web monitor is valid
            var connection = xmlObj.getElementsByTagName("connection");

            if (connection[0].childNodes[0].nodeValue == "bad")
            {
              setAllNotConnected();
            }
            else
            {
              // process node_status tags first
              var node_statuses = xmlObj.getElementsByTagName("node_status");

              var pwcs_per_host = new Object();
              var disk_free_per_host = new Object();

              for (i=0; i<node_statuses.length; i++) 
              {
                host = node_statuses[i].getAttribute("host");
                host = host.replace("mpsr-","");

                pwcs_per_host[host] = 0;

                // process each child key
                children = node_statuses[i].childNodes;
                for (j=0; j<children.length; j++)
                {
                  node = children[j];
                  if (node.nodeType == 1)
                  {
                    key = node.nodeName;
                    //alert("["+host+"] processing key="+key);
        
                    if (key == "load")
                    {
                      var load1 = parseFloat(node.childNodes[0].nodeValue);
                      var ncore = parseFloat(node.getAttribute("ncore"));
                      var load_percent = Math.floor((load1/ncore)*100);
                      progress_bar = eval(host + "_load")
                      progress_bar.setPercentage(load_percent);
                      document.getElementById(host+"_load_value").innerHTML = "&nbsp;"+load1;
                    }
                    else if (key == "temperature")
                    {
                      var temp = node.childNodes[0].nodeValue;
                      document.getElementById(host+"_temperature_value").innerHTML = temp;
                    }
                    else if (key == "disk") 
                    {
                      var disk_used = parseFloat(node.getAttribute("used"));
                      var disk_free = parseFloat(node.childNodes[0].nodeValue);
                      var disk_size = parseFloat(node.getAttribute("size"));
                      var disk_free = (disk_size - disk_used);
                      var disk_percent = Math.floor((disk_used / disk_size) * 100);
                      progress_bar = eval(host+"_disk")
                      progress_bar.setPercentage(disk_percent);
                      disk_free_per_host[host] = disk_free;
                    } 
                    else if (key == "datablock")
                    {
                      ;
                    }
                    else 
                    {
                      alert("unrecognized key: "+key);
                    }
                  }
                }
              }
            

              // update the time remaining for each PWC
              for (var host in pwcs_per_host)
              {
                var free_space = parseFloat(disk_free_per_host[host]);
                var free_space_str = "";
                if (free_space < 1024)
                {
                   free_space_str = free_space.toFixed(1) + " MB";
                }
                else if (free_space < 1024 * 1024)
                {
                  free_space = free_space / 1024;
                  free_space_str = free_space.toFixed(1) + " GB";
                }
                else 
                {
                   free_space = free_space / (1024 * 1024);
                  free_space_str = free_space.toFixed(1) + " TB";
                }
                document.getElementById(host+"_disk_value").innerHTML = "&nbsp;" + free_space_str;
              }

              // process daemon_status tags next
              var daemon_statuses = xmlObj.getElementsByTagName("daemon_status");

              var set = new Array();
              resetOthers(set);

              for (i=0; i<daemon_statuses.length; i++) 
              {
                var node = daemon_statuses[i];
  
                if (node.nodeType == 1)
                {
                  var type = node.getAttribute("type");
                  var area = node.getAttribute("area");
                  var pwc = pad(node.getAttribute("pwc"),2);
                  var tag = node.getAttribute("tag");
                  var msg = node.childNodes[0].nodeValue;

                  var host_id = area + pwc;
                  var host = host_mappings[host_id]

                  var log_file = "";
                  if (pwc != "server")
                  {
                    log_file = "mopsr_" + tag + "_monitor";
                  }
                  else
                  {
                    log_file = tag;
                  }

                  img_id = document.getElementById(host + "_img");

                  // add this light to the list of lights not to be reset
                  set.push(host);

                  if (img_id.src.indexOf("grey_light.png") == -1)
                  {
                    if (type == "ok") 
                      img_id.src = "/images/green_light.png";
                    if (type == "warning")
                      img_id.src = "/images/yellow_light.png";
                    if (type == "error")
                      img_id.src = "/images/red_light.png";
                  }

                  log_level = "all";
        
                  log_length = 6;
                  try  
                  {
                    var j = document.getElementById("loglength").selectedIndex;
                    log_length = document.getElementById("loglength").options[j].value
                  }
                  catch(e) 
                  {
                  }

                  var link = "log_viewer.php?host="+host+"&pwc="+pwc+"&level="+log_level+"&length="+log_length+"&daemon="+log_file+"&autoscroll=false";
                  var msg_element = document.getElementById(host+"_messages");

                  if (msg_element.innerHTML == "&nbsp;") {
                    msg_element.innerHTML = "<a class='"+type+"' target='log_window' href='"+link+"'>"+msg+"</a>";
                  } else {
                    msg_element.innerHTML = msg_element.innerHTML + " | <a class='"+type+"' target='log_window' href='"+link+"'>"+msg+"</a>";
                  }
                }
              }
              resetOthers(set);
            }
          }
          else 
          {
            setAllNotConnected();
          }
        }
      }

      function machine_summary_request() 
      {
        var url = "machine_summary.lib.php?update=true";
  
        if (window.XMLHttpRequest)
          ms_http_request = new XMLHttpRequest();
        else
          ms_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        ms_http_request.onreadystatechange = function() {
          handle_machine_summary_request(ms_http_request)
        };
        ms_http_request.open("GET", url, true);
        ms_http_request.send(null);
      }

    </script>
<?
  }

  function printJavaScriptBody() 
  {
?>
    <script type="text/javascript">
      Event.observe(window, 'load',  function() 
      {
<?
        // machines have load and disk bars
        foreach ($this->machines as $m => $type)
        {
          echo "        ".$m."_load = new JS_BRAMUS.jsProgressBar($('".$m."_load_progress_bar'), 0, ";
          echo " { width : 40, showText : false, animate : false, ".
               "boxImage: '/images/jsprogress/percentImage_40.png', ".
               "barImage : Array( '/images/jsprogress/percentImage_back1_40.png', ".
               "'/images/jsprogress/percentImage_back2_40.png', ".
               "'/images/jsprogress/percentImage_back3_40.png', ".
               "'/images/jsprogress/percentImage_back4_40.png') } );\n";


          echo "        ".$m."_disk = new JS_BRAMUS.jsProgressBar($('".$m."_disk_progress_bar'), 0, ";
          echo " { width : 40, showText : false, animate : false, ".
               "boxImage: '/images/jsprogress/percentImage_40.png', ".
               "barImage : Array( '/images/jsprogress/percentImage_back1_40.png', ".
               "'/images/jsprogress/percentImage_back2_40.png', ".
               "'/images/jsprogress/percentImage_back3_40.png', ".
               "'/images/jsprogress/percentImage_back4_40.png') } );\n";

        }
?>
      }, false);
    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $title = "Machine Summary";
    $title .= " <a target='_ms_popup' href='area_summary.lib.php?single=true&area=aq'>AQ</a>";
    $title .= " <a target='_ms_popup' href='area_summary.lib.php?single=true&area=bf'>BF</a>";
    $title .= " <a target='_ms_popup' href='area_summary.lib.php?single=true&area=bp'>BP</a>";

    $this->openBlockHeader($title);

?>
    <table class="machine_summary" width='100%' border=0 cellspacing=0 cellpadding=0>
        
      <tr>
        <th>Host</th>
        <th colspan=2>Load</th>
        <th colspan=2>Disk</th>
        <th>T &deg;C</th>
        <th></th>
        <th>Messages</th>
      </tr>
    <?

    $status_types = array("pwc", "src", "sys");
    foreach ($this->machines as $m => $types)
    {
      echo " <tr id='".$m."_row'>\n";
  
      $status = STATUS_OK;
      $message = "";
      for ($j=0; $j<count($status_types); $j++) {
        $s = $status_types[$j];
      }

      echo "     <td width='30px' class='gap'>".$m."</td>\n";
        
      // load progress bar and value
      echo "     <td width='40px' style='vertical-align: middle;'>\n"; 
      echo "      <span id='".$m."_load_progress_bar'>[ ... ]</span>\n";
      echo "     </td>\n";
        
      echo "     <td width='40px' class='gap'>\n";
      echo "      <span id='".$m."_load_value'></span>\n";
      echo "     </td>\n";

      // disk progress bar and value
      echo "     <td width='40px' style='vertical-align: middle;'>\n";
      echo "      <span id='".$m."_disk_progress_bar'>[ ... ]</span>\n";
      echo "     </td>\n";

      echo "     <td width='50px' class='gap' align='left'>\n";
      echo "      <span id='".$m."_disk_value'></span>\n";
      echo "     </td>\n";

      // temperature value
      echo "     <td width='20px' class='gap'><span id='".$m."_temperature_value'>NA</span></td>\n";

      echo "     <td width='15px' class='".$class."' valign='center'>\n";
      echo "      ".$this->overallStatusLight($status, $m, $m."_a", $m."_img")."\n";
      echo "     </td>\n";

      //  messages
      echo "     <td class='status_text' align='left'>\n";
      echo "      <div id='".$m."_messages' style='padding-top:4px;'></div>\n";
      echo "     </td>\n";
       
      echo " </tr>\n";
    }
?>
    </table>
<?
    if (!$this->verbose)
      echo "<div align=right><a href='/mopsr/machine_summary.lib.php?single=true&verbose=true' target='_ms_popup'>More Information</a></div>\n";

    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = $this->config["SERVER_HOST"];
    $port = $this->config["SERVER_WEB_MONITOR_PORT"];

    $output = "";
    $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<machine_summary>\n";

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok")
    {
      $xml .= "<connection>good</connection>\n";

      $bytes_written = socketWrite($socket, "aq_node_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<aq_node_info>\n";
        $xml .= $response."\n";
        $xml .= "</aq_node_info>\n";
      }
      socket_close($socket);
      $socket = 0;
    }

    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok")
    {
      $bytes_written = socketWrite($socket, "bf_node_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<bf_node_info length='".strlen($response)."'>\n";
        $xml .= $response."\n";
        $xml .= "</bf_node_info>\n";
      }
      socket_close($socket);
      $socket = 0;
    }
  
    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok")
    {
      $bytes_written = socketWrite($socket, "bp_node_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<bp_node_info length='".strlen($response)."'>\n";
        $xml .= $response."\n";
        $xml .= "</bp_node_info>\n";
      }
      socket_close($socket);
      $socket = 0;
    } 
    else
    {
      $xml .= "<connection>bad</connection>\n";
    }

    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok")
    {
      $bytes_written = socketWrite($socket, "srv_node_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<daemon_statuses>\n";
        $xml .= $response."\n";
        $xml .= "</daemon_statuses>\n";
      }
      socket_close($socket);
      $socket = 0;
    }

    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok") 
    {
      $bytes_written = socketWrite($socket, "status_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<daemon_statuses>\n";
        $xml .= $response."\n";
        $xml .= "</daemon_statuses>\n";
      }
      socket_close($socket);
      $socket = 0;
    }

    $xml .= "</machine_summary>\n";

    header('Content-type: text/xml');
    echo $xml;
  }

  function overallStatusLight($status, $machine, $linkid, $imgid) 
  {

    $lights = array(STATUS_OK => "green_light.png", STATUS_WARN => "yellow_light.png", STATUS_ERROR => "red_light.png");

    if ($status == STATUS_OK) {
      $url = "log_viewer.php?machine=".$machine."&level=all";
      $title = "OK: ".$machine;
    } else if ($status == STATUS_WARN) {
      $url = "log_viewer.php?machine=".$machine."&level=warn";
      $title = "Warning: ".$machine;
    } else if ($status == STATUS_ERROR) {
      $url = "log_viewer.php?machine=".$machine."&level=error";
      $title = "Error: ".$machine;
    } else {
      $url = "log_viewer.php?machine=".$machine."&level=all";
      $title = "Message";
    }

    $string = '<img id="'.$imgid.'" border="none" width="15px" height="15px" '.
              'src="/images/'.$lights[$status].'">';

    return $string;
  }

}

handleDirect("machine_summary");

