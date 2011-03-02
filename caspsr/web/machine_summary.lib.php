<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class machine_summary extends caspsr_webpage 
{

  var $machines = array();
  var $pwcs = array();
  var $demuxers = array();
  var $srvs = array();
  var $config = array();

  function machine_summary()
  {
    caspsr_webpage::caspsr_webpage();
    array_push($this->ejs, "/js/prototype.js");
    array_push($this->ejs, "/js/jsProgressBarHandler.js");

    $inst = new caspsr();
    $this->config = $inst->config;

    /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) {
      array_push($this->pwcs, $this->config["PWC_".$i]);
      array_push($this->machines, $this->config["PWC_".$i]);
    }

     /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_DEMUX"]; $i++) {
      if (!in_array($this->config["DEMUX_".$i], $this->demuxers)) {
        array_push($this->demuxers, $this->config["DEMUX_".$i]);
        array_push($this->machines, $this->config["DEMUX_".$i]);
      }
    }

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

      var machines = new Array(<?
      for ($i=0; $i<count($this->machines)-1; $i++) { echo "\"".$this->machines[$i]."\","; }
        echo "\"".$this->machines[count($this->machines)-1]."\"";
      ?>);


      // Set all the imgs, links and messages not in the excluded array to green
      function resetOthers(excluded) {

        var j = document.getElementById("loglength").selectedIndex;
        var log_length = document.getElementById("loglength").options[j].value

        for (i=0; i<machines.length; i++) {
          host = machines[i];
          if (excluded.indexOf(host) == -1) {
            document.getElementById(host+"_messages").innerHTML = "";
            document.getElementById(host+"_img").src = "/images/green_light.png";
            //document.getElementById(host+"_a").title = "OK";
            //document.getElementById(host+"_a").href = "";
          }
        }
      }

      function setHostNotConnected(host) {

        document.getElementById(host+"_messages").innerHTML = "not connected";
        document.getElementById(host+"_img").src  = "/images/grey_light.png";
        document.getElementById(host+"_load_value").innerHTML = "&nbsp;--";
        document.getElementById(host+"_buffer1").bgColor = "";
        document.getElementById(host+"_buffer2").bgColor = "";
        document.getElementById(host+"_temperature_value").innerHTML = "--";
        progress_bar = eval(host+"_load"); 
        progress_bar.setPercentage(0.0);
        try {
          document.getElementById(host+"_buffer_value").innerHTML = "&nbsp;--";
          progress_bar = eval(host+"_buffer"); 
          progress_bar.setPercentage(0.0);
        } catch(e) {
        }
      }

      function setAllNotConnected() {
        var progress_bar;
        for (i=0; i<machines.length; i++) {
          host = machines[i];
          setHostNotConnected(host);
        } 
      } 

      function handle_machine_summary_request(ms_http_request) 
      {
        if (ms_http_request.readyState == 4) {

          //alert("handle_machine_summary_request "+ms_http_request.responseText);

          var response = String(ms_http_request.responseText)

          if (response.length > 0) {

            response = rtrim(response);

            if (response.indexOf("Could not connect to") == -1) {

              var instrument = "caspsr"
          
              var lines = response.split("\n");
              var values, key, state, message, machine_update_line;

              var set = new Array();
              resetOthers(set);

              for (i=0; i < lines.length; i++) {

                values = lines[i].split(":::");
                key = values[0];
                state = values[1];
                message = values[2];

                // check to see if the key is in machines array
                machine_update_line = false;
                for (j=0; j < machines.length; j++) {
                  if (machines[j] == key) {
                    machine_update_line = true;
                  }
                } 

                // update the machine 
                if (machine_update_line) {

                  var bits = message.split(";;;");

                  // print grey row if stopped
                  if ((state == "stopped") || (bits.length != 5)) {
                    setHostNotConnected(key);
                  } else {

                    var disks, dbs, loads, temp, progress_bar;

                    disks = bits[0].split(" ");
                    dbs   = bits[1].split(" ");
                    loads = bits[2].split(",");
                    temp  = bits[4];

                    disks_space =  (parseInt(disks[0]) - parseInt(disks[1])) / 1024;  
                    load_percent = Math.floor((parseFloat(loads[0])/8)*100);

                    // Special case for no data block
                    if ((parseInt(dbs[1]) == 0) && (parseInt(dbs[0]) == 0)) {
                      db_percent = 0;
                      try {
                        progress_bar = eval(key+"_buffer");
                        document.getElementById(key+"_buffer1").bgColor = "#FF2e2e";
                        document.getElementById(key+"_buffer2").bgColor = "#FF2e2e";
                      } catch (e) {
                      }
                    } else {
                      db_percent = Math.floor(parseInt(dbs[1]) / parseInt(dbs[0])*100);
                      document.getElementById(key+"_buffer1").bgColor = "";
                      document.getElementById(key+"_buffer2").bgColor = "";
                    }

                    // use try/catch since not all machines have a buffer
                    try { 
                      if (db_percent <= 0)
                        db_percent = 0;
                      if (db_percent >= 100)
                        db_percent = 100;
                      if ((db_percent < 0) || (db_percent > 100))
                      {
                        //alert (key+"_buffer.setPercentage("+db_percent+")");
                      }
                      progress_bar = eval(key+"_buffer");
                      progress_bar.setPercentage(db_percent);
                      document.getElementById(key+"_buffer_value").innerHTML = "&nbsp;"+dbs[1]+"&nbsp;of&nbsp;"+dbs[0];
                    } catch (e) {
                      //alert("key="+key+", "+e);
                    }
                    progress_bar = eval(key+"_load")
                    progress_bar.setPercentage(load_percent);
                    document.getElementById(key+"_load_value").innerHTML = "&nbsp;"+loads[0];
                    document.getElementById(key+"_temperature_value").innerHTML = temp;
                  }

                // we have warn/error status message
                } else {

                  var keyparts, logfile, img_id, host

                  keyparts = key.split("_");

                  if ((keyparts[0] == "pwc") || (keyparts[0] == "src") || (keyparts[0] == "sys")) {
                    host = keyparts[1];
                    log_file = instrument+"_"+keyparts[0]+"_monitor";
                  } else {
                    host = "srv0";
                    log_file = key;
                  }

                  img_id = document.getElementById(host+"_img");
                  //link_id = host+"_a";

                  set.push(host);
    
                  /* get light val */ 
                  if (img_id.src.indexOf("grey_light.png") == -1) {
                    if (state == <?echo STATUS_OK?>) 
                      img_id.src = "/images/green_light.png";
                    if (state == <?echo STATUS_WARN?>) 
                      img_id.src = "/images/yellow_light.png";
                    if (state == <?echo STATUS_ERROR?>) 
                    img_id.src = "/images/red_light.png";
                  }

                  log_level="all";

                  var j = document.getElementById("loglength").selectedIndex;
                  var log_length = document.getElementById("loglength").options[j].value

                  var link = "log_viewer.php?machine="+host+"&level="+log_level+"&length="+log_length+"&daemon="+log_file+"&autoscroll=false";
                  var curr_msg = document.getElementById(host+"_messages").innerHTML;
                  if (curr_msg == "") {
                    document.getElementById(host+"_messages").innerHTML = "<a class='cln' target='log_window' href='"+link+"'>"+message+"</a>";
                  } else {
                    document.getElementById(host+"_messages").innerHTML = curr_msg + " | <a class='cln' target='log_window' href='"+link+"'>"+message+"</a>";
                  }
                  //document.getElementById(link_id).title = message;
                  //document.getElementById(link_id).href = "log_viewer.php?machine="+host+"&level="+log_level+
                  //                                        "&length="+log_length+"&daemon="+log_file+"&autoscroll=false";
    
                }
              }
              resetOthers(set);
            } else {
              setAllNotConnected();
            }
          } else {
            var set = new Array();
            resetOthers(set);
          }
        }
      }

      function machine_summary_request() 
      {
        var url = "machine_summary.lib.php?update=true&host=<?echo $this->config["SERVER_HOST"]?>&port=<?echo $this->config["SERVER_WEB_MONITOR_PORT"]?>";
  
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
        <th width='30px'>Machine</th>
          <th width='10px'>State</th>
          <th colspan=2>Load</th>
          <th colspan=2>Buffer</th>
          <th width='30px'>T &deg;C</th>
          <th>Messages</th>
        </tr>
    <?
    $status_types = array("pwc", "src", "sys");
    for ($i=0; $i<count($this->machines); $i++) {
      $m = $this->machines[$i];

        echo " <tr id='".$m."_row'>\n";
  
        echo "  <td width='30px'>".$m."</td>\n";
          
        $status = STATUS_OK;
        $message = "";
        for ($j=0; $j<count($status_types); $j++) {
          $s = $status_types[$j];
          if ($statuses[$m."_".$s."_STATUS"] == STATUS_WARN) {
            if ($status != STATUS_ERROR) {
              $status = STATUS_WARN;
            }
           
          }
          if ($statuses[$m."_".$s."_STATUS"] == STATUS_ERROR) {
            $status = STATUS_ERROR;
          }

          if  ($statuses[$m."_".$s."_STATUS"] != STATUS_OK) 
            $message .= $statuses[$m."_".$s."_MESSAGE"]."\n";
        }
        
        $linkid = $m."_a";
        $imgid = $m."_img";
        
        echo "     <td>\n";
        echo "      ".$this->overallStatusLight($status ,$m, $linkid, $imgid);
        echo "     </td>\n";
        
        /* load progress bar and value */
        echo "     <td width='80px' style='vertical-align: middle;'>\n"; 
        echo "      <span id='".$m."_load_progress_bar'>[  Loading ]</span>\n";
        echo "     </td>\n";
        
        echo "     <td width='40px'>\n";
        echo "      <span id='".$m."_load_value'></span>\n";
        echo "     </td>\n";
       /* buffers progress bar and value */
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


        echo "  <td class='status_text'><span id='".$m."_temperature_value'>NA</span></td>\n";
        
        echo "  <td class='status_text'><span id='".$m."_messages'></span></td>\n";
        
        echo " </tr>\n";
      }
      ?>
    </table>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($host, $port)
  {

    $output = "";

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "node_info\r\n");
      $read = socketRead($socket);
      $string = str_replace(";;;;;;","\n",$read);
      $output .= rtrim($string)."\n";

      $read = socketRead($socket);

      $bytes_written = socketWrite($socket, "status_info\r\n");
      $read = socketRead($socket);
      $string = str_replace(";;;;;;","\n",$read);
      $output .= rtrim($string);

      socket_close($socket);

    } else {
      $output = "Could not connect to $host:$port<BR>\n";
    }

    echo $output;
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

    //$string = '<a target="log_window" id="'.$linkid.'" href="'.$url.'">'.
    //          '<img id="'.$imgid.'" border="none" width="15px" height="15px" '.
    //          'src="/images/'.$lights[$status].'"></a>';

    return $string;
  }

}

handleDirect("machine_summary");

