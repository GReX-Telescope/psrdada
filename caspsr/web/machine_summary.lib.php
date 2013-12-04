<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class machine_summary extends caspsr_webpage 
{
  var $machines = array();
  var $pwcs = array();
  var $demuxers = array();
  var $srvs = array();
  var $config = array();
  var $pwc_db_ids = array();
  var $demux_db_ids = array();

  function machine_summary()
  {
    caspsr_webpage::caspsr_webpage();

    $this->callback_freq = 10000;
    array_push($this->ejs, "/js/prototype.js");
    array_push($this->ejs, "/js/jsProgressBarHandler.js");

    $inst = new caspsr();
    $this->config = $inst->config;

    /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) {
      array_push($this->pwcs, $this->config["PWC_".$i]);
      array_push($this->machines, $this->config["PWC_".$i]);
    }
    $this->pwc_db_ids = array($this->config["RECEIVING_DATA_BLOCK"], $this->config["PROCESSING_DATA_BLOCK"]);

     /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_DEMUX"]; $i++) {
      if (!in_array($this->config["DEMUX_".$i], $this->demuxers)) {
        array_push($this->demuxers, $this->config["DEMUX_".$i]);
        array_push($this->machines, $this->config["DEMUX_".$i]);
      }
    }
    $this->demux_db_ids = explode(" ",$this->config["DEMUX_BLOCK_IDS"]);

    $this->server_host = $this->config["SERVER_HOST"];
    array_push($this->machines, "srv0");
    array_push($this->srvs, "srv0");

  }

  function javaScriptCallback()
  {
    return "machine_summary_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function rtrim(str, chars) {
        chars = chars || "\\s";
        return str.replace(new RegExp("[" + chars + "]+$", "g"), "");
      }
<?

      $keys = $this->machines;
      echo "      var machines = new Array('".$keys[0]."'";
      for ($i=1; $i<count($keys); $i++)
        echo ",'".$keys[$i]."'";
      echo ");\n";

      $keys = $this->pwcs;
      echo "     var pwcs = new Array('server'";
      for ($i=0; $i<count($keys); $i++)
        echo ",'".$keys[$i]."'";
      echo ");\n";

      echo "     var pwc_db_ids = new Array('".$this->pwc_db_ids[0]."'";
      for ($i=1; $i<count($this->pwc_db_ids); $i++)
        echo ",'".$this->pwc_db_ids[$i]."'";
      echo ");\n";

      $keys = $this->demuxers;
      echo "     var demuxers = new Array('server'";
      for ($i=0; $i<count($keys); $i++)
        echo ",'".$keys[$i]."'";
      echo ");\n";

      echo "     var demux_db_ids = new Array('".$this->demux_db_ids[0]."'";
      for ($i=1; $i<count($this->demux_db_ids); $i++)
        echo ",'".$this->demux_db_ids[$i]."'";
      echo ");\n";
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
          host = machines[i];
          if (excluded.indexOf(host) == -1)
          {
            document.getElementById(host+"_messages").innerHTML = "&nbsp;";
            document.getElementById(host+"_img").src = "/images/green_light.png";
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

      function setPWCNotConnected (host)
      {
        var pb1;
        var buffer;
        var i, j;

        document.getElementById(host+"_messages").innerHTML = "not connected";
        document.getElementById(host+"_img").src  = "/images/grey_light.png";

        for (i=0; i<machines.length; i++)
        {
          try
          {
            // try to evaluate a javascript variable with this name
            pb1 = eval(machines[i] + "_buffer");

            // if we didn't throw an exception, then the object exists, set percent to 0
            pb1.setPercentage(0.0);

            // and set the value to 0 also
            document.getElementById(machines[i] + "_buffer_value").innerHTML = "&nbsp;--";
          }
          catch(e)
          {
            //alert("skipping machines["+i+"]="+machines[i]);
          }
        }
      }

      function setAllNotConnected() 
      {
        for (i=0; i<pwcs.length; i++) {
          setPWCNotConnected(pwcs[i]);
        } 
        for (i=0; i<demuxerss.length; i++) {
          setPWCNotConnected(demuxers[i]);
        } 
        for (i=0; i<machines.length; i++) {
          setHostNotConnected(machines[i]);
        } 
      } 

      function handle_machine_summary_request(ms_xml_request) 
      {
        var children;
        var progress_bar;
        var active_pwcs = 0;
        var log_length = 6;

        if (ms_xml_request.readyState == 4) 
        {
          var xmlDoc = ms_xml_request.responseXML;
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

                var db_size = 0;
                var db_used = 0;

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
                      document.getElementById(host+"_load_value").innerHTML = "&nbsp;"+load1.toFixed(2);
                    }
                    else if (key == "temperature")
                    {
                      var temp = node.childNodes[0].nodeValue;
                      document.getElementById(host+"_temperature_value").innerHTML = temp;
                    }
                    else if (key == "disk") 
                    {
                      var disk_free = parseFloat(node.childNodes[0].nodeValue);
                      var disk_size = parseFloat(node.getAttribute("size"));
                      var disk_used = parseFloat(node.getAttribute("used"));
                      // note that disk_used does not typically include 5% space reserved for root
                      disk_used = disk_size - disk_free;
                      var disk_percent = Math.floor((disk_used / disk_size) * 100);
                      progress_bar = eval(host+"_disk")
                      progress_bar.setPercentage(disk_percent);
                      disk_free_per_host[host] = disk_free;
                    } 
                    else if (key == "datablock") 
                    {
                      db_size += parseFloat(node.getAttribute("size"));
                      db_used += parseFloat(node.childNodes[0].nodeValue);
                    } 
                    else 
                    {
                      alert("unrecognized key: "+key);
                    }
                  }
                }

                var db_percent = Math.floor((db_used / db_size) * 100);
                var bg_color = (db_size == 0 ? "#FF2e2e" : "");

                try 
                {
                  progress_bar = eval(host+"_buffer");
                  progress_bar.setPercentage(db_percent);
                  document.getElementById(host+"_buffer_value").innerHTML = "&nbsp;" + db_used + "/"  + db_size;
                  document.getElementById(host+"_buffer1").bgColor = bg_color;
                  document.getElementById(host+"_buffer2").bgColor = bg_color;
                      
                  // assume that only pwcs have datablocks
                  active_pwcs ++;
                  pwcs_per_host[host]++;
                }
                catch (e)
                {
                  //alert("disabled datablock ["+host+"_buffer]");
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
                    log_file = "bpsr_" + tag + "_monitor";
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
        for ($i=0; $i<count($this->machines); $i++) {
          $m = $this->machines[$i];

          if (in_array($m, $this->pwcs) || in_array($m, $this->demuxers)) {
            echo "        ".$m."_buffer = new JS_BRAMUS.jsProgressBar($('".$m."_buffer_progress_bar'), 0, ";
            echo " { width : 80, showText : false, animate : false, barImage : Array( '/images/jsprogress/percentImage_back1_80.png', ".
                                                                    "'/images/jsprogress/percentImage_back2_80.png', ".
                                                                    "'/images/jsprogress/percentImage_back3_80.png', ".
                                                                    "'/images/jsprogress/percentImage_back4_80.png') } );\n";

          }

          echo "        ".$m."_disk = new JS_BRAMUS.jsProgressBar($('".$m."_disk_progress_bar'), 0, ";
          echo " { width : 40, showText : false, animate : false, ".
               "boxImage: '/images/jsprogress/percentImage_40.png', ".
               "barImage : Array( '/images/jsprogress/percentImage_back1_40.png', ".
               "'/images/jsprogress/percentImage_back2_40.png', ".
               "'/images/jsprogress/percentImage_back3_40.png', ".
               "'/images/jsprogress/percentImage_back4_40.png') } );\n";

          echo "        ".$m."_load = new JS_BRAMUS.jsProgressBar($('".$m."_load_progress_bar'), 0, ";
          echo " { width : 80, showText : false, animate : false,  barImage : Array( '/images/jsprogress/percentImage_back1_80.png', ".
                                                                   "'/images/jsprogress/percentImage_back2_80.png', ".
                                                                   "'/images/jsprogress/percentImage_back3_80.png', ".
                                                                   "'/images/jsprogress/percentImage_back4_80.png') } );\n";
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
    <table width='100%' border=0 cellspacing=0 cellpadding=0>
      <tr>
        <th style="width: 30px;">Machine</th>
        <th width='10px'>State</th>
        <th colspan=2>Load</th>
        <th colspan=2>Disk</th>
        <th colspan=2>Buffer</th>
        <th width='20px'>T&nbsp;&deg;C</th>
        <th>Messages</th>
      <tr>
    <?
    $status_types = array("pwc", "src", "sys");
    for ($i=0; $i<count($this->machines); $i++)
    {
      $m = $this->machines[$i];

        echo " <tr id='".$m."_row'>\n";
  
        echo "  <td width='30px'>".$m."</td>\n";
          
        $status = STATUS_OK;
        $message = "";
        $linkid = $m."_a";
        $imgid = $m."_img";
        
        echo "     <td>\n";
        echo "      ".$this->overallStatusLight($status ,$m, $linkid, $imgid);
        echo "     </td>\n";
        
        // load progress bar and value
        echo "     <td width='80px' style='vertical-align: middle;'>\n"; 
        echo "      <span id='".$m."_load_progress_bar'>[  Loading ]</span>\n";
        echo "     </td>\n";
        echo "     <td width='40px'>\n";
        echo "      <span id='".$m."_load_value'></span>\n";
        echo "     </td>\n";

        // disk progress bar and value
        echo "     <td width='40px' style='vertical-align: middle;' rowspan='$rs'>\n";
        echo "      <span id='".$m."_disk_progress_bar'>[ ... ]</span>\n";
        echo "     </td>\n";
        echo "     <td width='60px' class='gap' align='left' rowspan='$rs'>\n";
        echo "      <span id='".$m."_disk_value'></span>\n";
        echo "     </td>\n";

        // buffers progress bar and value
        echo "     <td width='80px' style='vertical-align: middle;' id='".$m."_buffer1'>\n";
        if (in_array($m, $this->pwcs) || in_array($m, $this->demuxers)) {
          echo "      <span id='".$m."_buffer_progress_bar'>[  Loading ]</span>\n";
        }
        echo "     </td>\n";
        echo "     <td width='40px' id='".$m."_buffer2'>\n";
        if (!in_array($m, $this->srvs)) {
          echo "      <span id='".$m."_buffer_value'></span>\n";
        }
        echo "     </td>\n";


        echo "  <td width='10px' class='status_text'><span id='".$m."_temperature_value'>NA</span></td>\n";
        
        echo "  <td class='status_text'><span id='".$m."_messages'>&nbsp;</span></td>\n";
        
        echo " </tr>\n";
      }
      ?>
    </table>
<?
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
      list ($result, $response) = socketRead($socket);
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

