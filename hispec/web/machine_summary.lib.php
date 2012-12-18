<?PHP

ini_set('display_errors',1);
error_reporting(E_ALL);

include_once("hispec.lib.php");
include_once("hispec_webpage.lib.php");

class machine_summary extends hispec_webpage 
{
  // Datablock IDs
  var $db_ids = array();

  // mapping of PWC to HOST
  var $pwcs = array();
  
  // mapping of HOST to type 
  var $machines = array();
  var $config = array();
  var $roach = array();

  var $verbose = false;

  function machine_summary()
  {
    hispec_webpage::hispec_webpage();

    $this->callback_freq = 10000;
    array_push($this->ejs, "/js/prototype.js");
    array_push($this->ejs, "/js/jsProgressBarHandler.js");

    $inst = new hispec();
    $this->config = $inst->config;
    $this->roach = $inst->roach;

    // generate a list of machines
    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) 
    {
      $host = $this->config["PWC_".$i];
      $this->pwcs[$i] = $host;
      $this->machines[$host] = "pwc";
    }

    if (array_key_exists("verbose", $_GET) && ($_GET["verbose"] == "true"))
      $this->verbose = true;
    if ($this->verbose)
      $this->db_ids = explode(" ", $this->config["DATA_BLOCK_IDS"]);
    else
      $this->db_ids = array($this->config["RECEIVING_DATA_BLOCK"]);

    list ($server_host, $server_domain) = explode(".", $this->config["SERVER_HOST"], 2);
    $server_host = str_replace("-", "_", $server_host);
    $this->machines[$server_host] = "server";
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
    </style>


    <script type='text/javascript'>  

      var current_machines = <?echo count($this->machines)?>;

      function rtrim(str, chars) {
        chars = chars || "\\s";
        return str.replace(new RegExp("[" + chars + "]+$", "g"), "");
      }

      function PadDigits(n, totalDigits) 
      { 

        //alert("PadDigits("+n+","+totalDigits+")");
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

      $keys = array_keys($this->pwcs);
      echo "     var pwcs = new Array('server'";
      for ($i=0; $i<count($keys); $i++)
        echo ",'".$keys[$i]."'";
      echo ")\n";

      echo "     var db_ids = new Array('".$this->db_ids[0]."'";
      for ($i=1; $i<count($this->db_ids); $i++)
        echo ",'".$this->db_ids[$i]."'";
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
        for (i=0; i<pwcs.length; i++) 
        {
          var pwc = pwcs[i];
          if (excluded.indexOf(pwc) == -1) 
          {
            document.getElementById(pwc+"_messages").innerHTML = "&nbsp;";
            document.getElementById(pwc+"_img").src = "/images/green_light.png";
          }
        }
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


      function setPWCNotConnected(pwc_id) 
      {
        var pb1;
        var buffer;
        var i, j;

        document.getElementById(pwc_id+"_messages").innerHTML = "not connected";
        document.getElementById(pwc_id+"_img").src  = "/images/grey_light.png";

        for (i=0; i<machines.length; i++)
        {
          for (j=0; j<db_ids.length; j++)
          {
            //alert(pwc_id+": testing machines["+i+"]="+machines[i]+", db_ids["+j+"]="+db_ids[j]+" length="+db_ids.length);
            try
            {
              // try to evaluate a javascript variable with this name
              pb1 = eval(machines[i] + "_" + pwc_id + "_" + db_ids[j] + "_buffer");

              // if we didn't throw an exception, then the object exists, set percent to 0
              pb1.setPercentage(0.0);

              // and set the value to 0 also
              document.getElementById(machines[i] + "_" + pwc_id + "_" + db_ids[j] + "_buffer_value").innerHTML = "&nbsp;--";
            }
            catch(e)
            {
              //alert(pwc_id+": skipping machines["+i+"]="+machines[i]+", db_ids["+j+"]="+db_ids[j]);
            }
          }
        }
      }

      function setAllNotConnected() 
      {
        var pwc;
        for (i=0; i<pwcs.length; i++) {
          setPWCNotConnected(pwcs[i]);
        } 
        for (i=0; i<machines.length; i++) {
          setHostNotConnected(machines[i]);
        } 
      }

      function handle_machine_summary_request(xml_request) 
      {
        var children;
        var progress_bar;
        var active_pwcs = 0;
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
                host = host.replace("-","_");

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
                      var disk_used = parseFloat(node.childNodes[0].nodeValue);
                      var disk_size = parseFloat(node.getAttribute("size"));
                      var disk_free = (disk_size - disk_used);
                      var disk_percent = Math.floor((disk_used / disk_size) * 100);
                      progress_bar = eval(host+"_disk")
                      progress_bar.setPercentage(disk_percent);
                      disk_free_per_host[host] = disk_free;
                    } 
                    else if (key == "datablock") 
                    {
                      var pwc_id = node.getAttribute("pwc_id");
                      var db_id = node.getAttribute("db_id");
                      var db_size = parseFloat(node.getAttribute("size"));
                      var db_used = parseFloat(node.childNodes[0].nodeValue);
                      var db_percent = Math.floor((db_used / db_size) * 100);
                      var bg_color = (db_used == 0 ? "#FF2e2e" : "");

                      try 
                      {
                        progress_bar = eval(host+"_"+pwc_id+"_"+db_id+"_buffer");
                        progress_bar.setPercentage(db_percent);
                        document.getElementById(host+"_"+pwc_id+"_"+db_id+"_buffer_value").innerHTML = "&nbsp;" + db_used + "/"  + db_size;
                        document.getElementById(host+"_"+pwc_id+"_"+db_id+"_buffer1").bgColor = bg_color;
                        document.getElementById(host+"_"+pwc_id+"_"+db_id+"_buffer2").bgColor = bg_color;
                      
                        // assume that only pwcs have datablocks
                        active_pwcs ++;
                        pwcs_per_host[host]++;
                      }
                      catch (e)
                      {
                        //alert("disabled datablock ["+host+"_"+pwc_id+"_"+db_id+"_buffer]");
                      }
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
                  var pwc = node.getAttribute("pwc");
                  var tag = node.getAttribute("tag");
                  var msg = node.childNodes[0].nodeValue;

                  var log_file = "";
                  if (pwc != "server")
                  {
                    log_file = "hispec_" + tag + "_monitor";
                  }
                  else
                  {
                    log_file = tag;
                  }

                  img_id = document.getElementById(pwc + "_img");

                  // add this light to the list of lights not to be reset
                  set.push(pwc);

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

                  var link = "log_viewer.php?pwc="+pwc+"&level="+log_level+"&length="+log_length+"&daemon="+log_file+"&autoscroll=false";
                  var msg_element = document.getElementById(pwc+"_messages");

                  if (msg_element.innerHTML == "&nbsp;") {
                    msg_element.innerHTML = "<a class='cln' target='log_window' href='"+link+"'>"+msg+"</a>";
                  } else {
                    msg_element.innerHTML = msg_element.innerHTML + " | <a class='cln' target='log_window' href='"+link+"'>"+msg+"</a>";
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

        // pwcs have buffer bar
        foreach ($this->pwcs as $p => $host)
        {
          foreach ($this->db_ids as $db_id)
          {
            $tag = $host."_".$p."_".$db_id;
            echo "        ".$tag."_buffer = new JS_BRAMUS.jsProgressBar($('".$tag."_buffer_progress_bar'), 0, ";
            echo " { width : 40, showText : false, animate : false, ".
                 "boxImage: '/images/jsprogress/percentImage_40.png', ".
                 "barImage : Array( '/images/jsprogress/percentImage_back1_40.png', ".
                 "'/images/jsprogress/percentImage_back2_40.png', ".
                 "'/images/jsprogress/percentImage_back3_40.png', ".
                 "'/images/jsprogress/percentImage_back4_40.png') } );\n";
          }
        }

?>
      }, false);
    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Machine Summary");
?>
    <table class="machine_summary" width='100%' border=0 cellspacing=0 cellpadding=0>
        
      <tr>
        <th>Host</th>
        <th colspan=2>Load</th>
        <th colspan=2>Disk</th>
        <th>T &deg;C</th>
        <th colspan=2>Beam</th>
<? if ($this->verbose) { ?>
        <th colspan=2>Prim DB</th>
        <th colspan=2>Aux  DB</th>
<? } else { ?>
        <th colspan=2>Buffer</th>
<? } ?> 
        <th>Messages</th>
      </tr>
    <?
    $status_types = array("pwc", "src", "sys");
    foreach ($this->machines as $m => $type)
    {
      $m_pwcs = array();
      foreach ($this->pwcs as $p => $host)
      {
        if ($host == $m)
          array_push($m_pwcs, $p);
      } 
      if (count($m_pwcs) == 0)
        array_push($m_pwcs, "server");

      echo " <tr id='".$m."_row'>\n";
  
      $status = STATUS_OK;
      $message = "";
      for ($j=0; $j<count($status_types); $j++) {
        $s = $status_types[$j];
      }

      $rs = count($m_pwcs);
        
      echo "     <td width='30px' class='gap' rowspan='$rs'>".$m."</td>\n";
        
      // load progress bar and value
      echo "     <td width='40px' style='vertical-align: middle;' rowspan='$rs'>\n"; 
      echo "      <span id='".$m."_load_progress_bar'>[ ... ]</span>\n";
      echo "     </td>\n";
        
      echo "     <td width='40px' class='gap' rowspan='$rs'>\n";
      echo "      <span id='".$m."_load_value'></span>\n";
      echo "     </td>\n";

      // disk progress bar and value
      echo "     <td width='40px' style='vertical-align: middle;' rowspan='$rs'>\n";
      echo "      <span id='".$m."_disk_progress_bar'>[ ... ]</span>\n";
      echo "     </td>\n";

      echo "     <td width='50px' class='gap' align='left' rowspan='$rs'>\n";
      echo "      <span id='".$m."_disk_value'></span>\n";
      echo "     </td>\n";

      // temperature value
      echo "     <td width='20px' class='gap' rowspan='$rs'><span id='".$m."_temperature_value'>NA</span></td>\n";

      for ($i=0; $i<count($m_pwcs); $i++)
      {
        // for rowspanning
        if ($i > 0)
          echo "<tr>\n";

        $mp = $m_pwcs[$i];

        if ($this->verbose)
          $class = "gap_line";
        else
          $class = "gap";

        // Status Lights for each PWC
        $linkid = $mp."_a";
        $imgid = $mp."_img";
        echo "     <td width='15px' class='".$class."' valign='center'>\n";
        echo "      ".$this->overallStatusLight($status, $mp, $linkid, $imgid)."\n";
        echo "     </td>\n";
 
        // PWC IDs [BEAM]       
        echo "     <td width='30px' class='".$class."'>";
        if (strpos($mp, "server") === FALSE)
          echo "      <div>".$this->roach["BEAM_".$mp]."</div>\n";
        else
          echo "&nbsp;";
        echo "     </td>";

        // Receiving Data Block
        $db_id = $this->config["RECEIVING_DATA_BLOCK"];

        // datablock buffers progress bar
        echo "     <td width='40px' style='vertical-align: middle;'>\n";
        if (strpos($mp, "server") === FALSE)
        {
          $tag = $m."_".$mp."_".$db_id;
          echo "      <div id='".$tag."_buffer1'>\n";
          echo "       <span id='".$tag."_buffer_progress_bar'>[ ... ]</span>\n";
          echo "      </div>\n";
        }
        else
          echo "&nbsp;";
        echo "     </td>\n";

        // datablock buffers str value
        echo "     <td width='40px' style='vertical-align: middle;'>\n";
        if (strpos($mp, "server") === FALSE)
        {
          $tag = $m."_".$mp."_".$db_id;
          echo "      <div id='".$tag."_buffer2' style='text-align: right;'>\n";
          echo "       <span id='".$tag."_buffer_value'></span>\n";
          echo "      </div>\n";
        }
        else
          echo "&nbsp;";
        echo "     </td>\n";

        // in verbose mode, print other datablocks too
        if ($this->verbose)
        {
          echo "     <td width='40px' style='vertical-align: middle; padding-left: 10px;'>\n";
          if (strpos($mp, "server") === FALSE)
          {
            foreach ($this->db_ids as $db_id)
            {
              if ($db_id != $this->config["RECEIVING_DATA_BLOCK"])
              {
                $tag = $m."_".$mp."_".$db_id;
                echo "      <div id='".$tag."_buffer1' style='padding-top:4px;'>\n";
                echo "       <span id='".$tag."_buffer_progress_bar'>[ ... ]</span>\n";
                echo "      </div>\n";
              }
            }
          }
          else
            echo "&nbsp;";
          echo "     </td>\n";

          echo "     <td width='40px' style='vertical-align: middle;'>\n";
          if (strpos($mp, "server") === FALSE)
          {
            foreach ($this->db_ids as $db_id)
            {
              if ($db_id != $this->config["RECEIVING_DATA_BLOCK"])
              {
                $tag = $m."_".$mp."_".$db_id;
                echo "      <div id='".$tag."_buffer2' style='padding-top:4px;'>\n";
                echo "       <span id='".$tag."_buffer_value'></span>\n";
                echo "      </div>\n";
              }
            }
          }
          else
            echo "&nbsp;";
          echo "     </td>\n";
        }

        //  messages
        echo "     <td class='status_text' align='left'>\n";
        echo "      <div id='".$mp."_messages' style='padding-top:4px;'></div>\n";
        echo "     </td>\n";
        
        echo " </tr>\n";
      }
    }
?>
    </table>
<?
    if (!$this->verbose)
      echo "<div align=right><a href='/hispec/machine_summary.lib.php?single=true&verbose=true' target='_ms_popup'>More Information</a></div>\n";

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

      $bytes_written = socketWrite($socket, "node_info\r\n");
      list($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $xml .= "<node_info>\n";
        $xml .= $response."\n";
        $xml .= "</node_info>\n";
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

