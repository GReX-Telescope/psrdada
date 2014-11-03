<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class pfb_board extends mopsr_webpage 
{
  var $inst = 0;

  var $pfb_id;

  var $pfb_name;

  var $img_size = "100x75";

  var $modules = array (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  var $plot_types = array("hg",  "sp", "ts");

  var $plot_titles = array("ts" => "TimeSeries", "sp" => "Spectrum", "hg" => "Histogram");

  var $params = array ("pfb_monitor" => array ("pfb_monitor_error"),
                       "board_parameters" => array ("board_health", "clock", "1pps"),
                       "temperatures"     => array ("sensor1", "sensor2"),   
                       "supply_voltages"  => array("input0_volts", "input1_volts", "input2_volts", "input3_volts", "input4_volts", "input5_volts"),
                       "fpga_parameters"  => array("fpga_status", "fpga_error_state", "fpga_temp"),
                       "10gbe_parameters" => array("8bit_window", "udp_channels", "udp_sequence_reset", "udp_dest_mac", "udp_dest_ip", "udp_dest_port")
                      );

  var $param_titles = array( "pfb_monitor" => "PFB Board", "pfb_monitor_error" => "Errors", 
                             "board_parameters" => "Board", "board_health" => "Health", "clock" => "Clock", "1pps" => "1 PPS", 
                             "temperatures" => "Temps", "sensor1" => "Sensor 1", "sensor2" => "Sensor 2", 
                             "supply_voltages" => "Power Supply Voltages",
                             "input0_volts" => "Input 0", "input1_volts" => "Input 1", 
                             "input2_volts" => "Input 2", "input3_volts" => "Input 3",
                             "input4_volts" => "Input 4", "input5_volts" => "Input 5",
                             "fpga_parameters" => "FPGA", "fpga_status" => "Status",
                             "fpga_error_state" => "Error", "fpga_temp" => "Temp",
                             "10gbe_parameters" => "10GbE Params", "8bit_window" => "8-bit Window", "udp_channels" => "UDP Channels", 
                             "udp_sequence_reset" => "Rearm", "udp_dest_mac" => "Dest MAC",  "udp_dest_ip" => "Dest IP", "udp_dest_port" => "Dest Port");

  var $rw_params = array ("8bit_window", "udp_channels", "udp_sequence_reset", "udp_dest_mac", "udp_dest_ip", "udp_dest_port");

  var $update_secs = 5;

  var $connect_to_socket = false;

  function pfb_board()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR PFB Board";
    $this->pfb_id = $_GET["pfb_id"];
    $this->pfb_name = $_GET["pfb_name"];
    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "pfb_board_update_request();";
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

      function reset_others(excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if ((excluded.indexOf(imgs[i].id) == -1) && (imgs[i].id.indexOf("_mgt") == -1))
          {
            imgs[i].src = "/images/blankimage.gif";
          }
        }
      }

      function handle_pfb_board_update_request(xml_request) 
      {

        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var img_prefix  = xmlObj.getElementsByTagName("img_prefix")[0].childNodes[0].nodeValue;

            var error_message;
            var error_element = xmlObj.getElementsByTagName("error")[0];
            try {
              error_message = error_element.childNodes[0].nodeValue;
              if (error_message != "")
              {
                document.getElementById("pfb_monitor_error").innerHTML = error_message;
                if (error_message != "None")
                {
                  return;
                }
              }
            } catch (e) {
              // do nothing
            }


            var i, j;
            var excluded = Array();
            var pfb_boards = xmlObj.getElementsByTagName ("pfb_board");

            // parse XML for each PFB ID
            for (i=0; i<pfb_boards.length; i++)
            {
              var pfb_board = pfb_boards[i];
              var pfb_board_id = pfb_board.getAttribute("id");

              // boards will have params and plots
              var nodes = pfb_board.childNodes;
              for (j=0; j<nodes.length; j++)
              {
                if (nodes[j].nodeName == "params")
                {
                  params = nodes[j].childNodes;
                  for (k=0; k<params.length; k++)
                  {
                    param = params[k];
                    if (param.nodeType == 1)
                    {
                      var key = param.getAttribute('key');
                      var val = param.childNodes[0].nodeValue;
                      var element = document.getElementById(key);
                      try {
                        element.innerHTML = val;
                      } catch (e) {
                        //alert ("key="+key+ " val="+val);
                      }
                    }
                  } 
                }
                else if (nodes[j].nodeName == "modules")
                {
                  var modules = nodes[j].childNodes;
                  for (k=0; k<modules.length; k++)
                  {
                    var module = modules[k];
                    var module_id = module.getAttribute("id");

                    // for each image in this module
                    var children = module.childNodes;
                    for (l=0; l<children.length; l++)
                    {
                      child = children[l];
                      if (child.nodeName == "plot")
                      {
                        if (child.nodeType == 1)
                        {
                          var type = child.getAttribute("type");
                          var img_id = module_id + "_" + type;
                          var imgurl = http_server + "/" + url_prefix + "/" + img_prefix + "/" + child.childNodes[0].nodeValue;
                          excluded.push(img_id);
                          document.getElementById (img_id).src = imgurl;
                          imgurl = http_server + "/" + url_prefix + "/pfb_plot.php?pfb_id=<?echo $this->pfb_id?>&"+
                                  "mod="+module_id+"&type="+type+"&res=800x600";
                          document.getElementById (img_id + "_link").href = "javascript:popImage('"+imgurl+"')";
                        }
                      }
                      else if (child.nodeName == "mgtlock")
                      {
                        if (child.nodeType == 1)
                        {
                          var img_id = module_id + "_mgt";
                          var state = child.childNodes[0].nodeValue;
                          if (state == "true")
                            document.getElementById (img_id).src = "/images/green_light.png"
                          else if (state == "false")
                            document.getElementById (img_id).src = "/images/red_light.png"
                          else
                            document.getElementById (img_id).src = "/images/yellow_light.png"
                        }
                      }
                    }
                  }
                }
              }
            }
            reset_others (excluded);
          }
        }
      }
                  
      function pfb_board_update_request() 
      {
        var url = "pfb_board.lib.php?update=true&pfb_id=<?echo $this->pfb_id?>";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_pfb_board_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

      function pfb_board_action_request (key) 
      {
        var value = document.getElementById(key + "_new").value;
        var url = "pfb_board.lib.php?action=true&pfb_id=<?echo $this->pfb_id?>&key="+key+"&value="+value;
        var xml_request;

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_pfb_board_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type='text/css'>

      .left {
        text-align: left;
      }

    </style>
<?
  }

  function printSideBarHTML()
  {
    $this->openBlockHeader("Board Parameters");
    echo "<table width='300px' border=0>";

    foreach ($this->params as $section => $values)
    {

      echo " <tr>\n";
      echo "  <th colspan=3 align='left'>".$this->param_titles[$section]."</th>\n";
      echo " </tr>\n";
      foreach ($values as $param)
      {
        $id = $param;
        echo " <tr>\n";
        echo "  <td style='padding-right: 5px;'>".$this->param_titles[$param]."</td>\n";
        echo "  <td><div id='".$id."' style='font-size: 10px;'></div></td>\n";
        if (in_array($param, $this->rw_params))
        {
          echo "  <td>";
          echo "<input type='text' id='".$id."_new' name='".$id."_new' size='4'/>";
          echo "<input type='button' onClick='pfb_board_action_request(\"".$param."\")' value='Set'/>";
          echo "</td>\n";
        }
        echo   " </tr>\n";
      }
      echo " <tr><td colspan='3'>&nbsp;</td></tr>\n";
    }
    echo "</table>";
    $this->closeBlockHeader();
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("PFB Board ".$this->pfb_name);
    list ($xres, $yres) = split("x", $this->img_size);

    echo "<table class='pfb_board'>\n";
    echo " <tr>\n";

    echo "  <th>ID</th>\n";
    foreach ($this->plot_types as $plot)
    {
      echo "  <th>".$this->plot_titles[$plot]."</th>\n";
    }

    echo "  <th width='30px'>&nbsp;</th>\n";

    echo "  <th>ID</th>\n";
    foreach ($this->plot_types as $plot)
    {
      echo "  <th>".$this->plot_titles[$plot]."</th>\n";
    }
    echo " </tr>\n";

    $half_modules = count($this->modules)/2;
    for ($i=0; $i<$half_modules; $i++)      
    {
      echo " <tr>\n";

      $module = $this->modules[$i];
      echo "  <td id='".$module."_id'>\n";
      echo "    <table>\n";
      echo "      <tr><th class='left'>ID</th><td>".$module."</td></tr>\n";
      echo "      <tr><th class='left'>MGT</th><td><img id='".$module."_mgt' src='/images/grey_light.png' width='15px' height='15px'/></td></tr>\n";
      echo "    </table>\n";
      echo "  </td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "  <td>";
        echo "    <a id='".$module."_".$plot."_link'>";
        echo "      <img id='".$module."_".$plot."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px'/>";
        echo "   </a>";
        echo "  </td>\n";
      }

      echo "  <td></td>\n";

      $module = $this->modules[$i + $half_modules];
      echo "  <td id='".$module."_id'>\n";
      echo "    <table>\n";
      echo "      <tr><th class='left'>ID</th><td>".$module."</td></tr>\n";
      echo "      <tr><th class='left'>MGT</th><td><img id='".$module."_mgt' src='/images/grey_light.png' width='15px' height='15px'/></td></tr>\n";
      echo "    </table>\n";
      echo "  </td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "  <td>";
        echo "    <a id='".$module."_".$plot."_link'>";
        echo "      <img id='".$module."_".$plot."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px'/>";
        echo "   </a>";
        echo "  </td>\n";
      }

      echo " </tr>\n";
    }
    echo "</table>\n";

    $this->closeBlockHeader();
  }

  function printActionHTML($get)
  { 
    # prepare the XML command to be sent to the PFB Monitor daemon
    $xml_cmd  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_cmd .= "<mpsr_pfbmonitor_command>";
    $xml_cmd .=   "<pfb_boards>";
    $xml_cmd .=     "<pfb_board id='".$this->pfb_id."'/>";
    $xml_cmd .=   "</pfb_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='".$get["key"]."'>".$get["value"]."</param>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<modules>";
    $xml_cmd .=     "<param type='mgtlock'/>";
    $xml_cmd .=     "<plot type='hg'/>";
    $xml_cmd .=     "<plot type='sp'/>";
    $xml_cmd .=   "</modules>";
    $xml_cmd .= "</mpsr_pfbmonitor_command>";

    $host = "localhost";
    $port = $this->inst->config["SERVER_PFB_MONITOR_PORT"];
   
    $xml_reply = "";

    if ($this->connect_to_socket)
    {
      $xml_reply .= "<pfb_board_update>";
      $xml_reply .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>";
      $xml_reply .=   "<url_prefix>mopsr</url_prefix>";
      $xml_reply .=   "<img_prefix>pfb_monitor</img_prefix>";

      list ($socket, $result) = openSocket($host, $port);
      $response = "initial";

      if ($result == "ok")
      {
        $bytes_written = socketWrite ($socket, $xml_cmd."\r\n");
        $max = 100;
        $reply = "";
        while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
        {
          list ($result, $response) = socketRead($socket);
          if ($result == "ok")
          {
            $reply .= $response;
          }
          $max--;
        }
        $reply = str_replace ("<?xml version='1.0' encoding='ISO-8859-1'?>", "", $reply);
        $xml_reply .= $reply;
        $xml_reply .= "<error type='connection'>None</error>";
      }
      else
      {
        $xml_reply .= "<error type='connection'>".$result."</error>";
      }

      $xml_reply .= "</pfb_board_update>";
      header('Content-type: text/xml');
      echo $xml_reply;
    }
    else
    {
      $this->printUpdateHTML($get);
    }
  }

  function printUpdateHTML($get)
  {
    # since this will only be done infrequently and not by many http clients 
    # (I hope!), do a filesystem lookup!

    $xml_cmd  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_cmd .= "<mpsr_pfbmonitor_command>";
    $xml_cmd .=   "<pfb_boards>";
    $xml_cmd .=     "<pfb_board id='".$this->pfb_id."'/>";
    $xml_cmd .=   "</pfb_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='all'/>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<modules>";
    $xml_cmd .=     "<param type='mgtlock'/>";
    $xml_cmd .=     "<plot type='hg'/>";
    $xml_cmd .=     "<plot type='sp'/>";
    $xml_cmd .=   "</modules>";
    $xml_cmd .= "</mpsr_pfbmonitor_command>";

    $host = "localhost";
    $port = $this->inst->config["SERVER_PFB_MONITOR_PORT"];

    $xml_reply = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_reply .= "<pfb_board_update>";
    $xml_reply .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml_reply .=   "<url_prefix>mopsr</url_prefix>";
    $xml_reply .=   "<img_prefix>monitor/pfb</img_prefix>";

    if ($this->connect_to_socket)
    {
      list ($socket, $result) = openSocket($host, $port);
      $response = "initial";

      if ($result == "ok")
      {
        $reply = "";
        $bytes_written = socketWrite ($socket, $xml_cmd."\r\n");
        $max = 100;
        while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
        {
          list ($result, $response) = socketRead($socket);
          if ($result == "ok")
          {
            $reply .= $response;
          }
          $max--;
        }
        $reply = str_replace ("<?xml version='1.0' encoding='ISO-8859-1'?>", "", $reply);
        $reply = str_replace ("/data/mopsr/", "", $reply);
        $xml_reply .= $reply;
        $xml_reply .= "<error type='no error'>None</error>";
      }
      else
      {
        $xml_reply .= "<error type='connection' host='".$host."' port='".$port."'>".$result."</error>";
      }

      $xml_reply .= "</pfb_board_update>";
    }
    else
    {
      $xml_reply .=   "<pfb_board id='".$this->pfb_id."'>";

      # get a listing of the images for this PFB board
      $cmd = "find ".$this->inst->config["SERVER_PFB_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$this->pfb_id.".*.??.*x*.png' -printf '%f\n' | sort -n";
      $images = Array();
      $lastline = exec($cmd, $images, $rval);
      $to_use = Array();
      if (($rval == 0) && (count($images) > 0))
      {
        # use associative array to store only the most recent images of a module + type + resolution
        foreach ($images as $image)
        {
          list ($time, $rid, $module, $type, $res, $ext) = explode(".", $image);
          if (!array_key_exists($module, $to_use))
            $to_use[$module] = Array();
          $to_use[$module][$type.".".$res] = $image;
        }
      }
      else
      {
        $xml_reply .= "<error return_value='".$rval."'>".$lastline."</error>";
      }

      $xml_reply .= "<modules>";
      foreach ($this->modules as $module)
      {
        $xml_reply .= "<module id='".$module."'>"; 
        $xml_reply .=   "<mgtlock>true</mgtlock>";

        if (array_key_exists($module, $to_use))
        {
          foreach (array_keys($to_use[$module]) as $key)
          {
            list ($type, $res) = explode(".", $key);
            list ($xres, $yres) = explode("x", $res);
            $xml_reply .= "<plot type='".$type."' width='".$xres."' height='".$yres."'>".$to_use[$module][$key]."</plot>";
          }
        }
        $xml_reply .= "</module>";
      }
      $xml_reply .= "</modules>";

      $xml_reply .=   "<params>";
      $xml_reply .=     "<param key='pfb_monitor_error'>Not Implemented, test mode</param>";
      $xml_reply .=     "<param key='board_health'>OK</param>";
      $xml_reply .=     "<param key='clock'>400 MHz</param>";
      $xml_reply .=     "<param key='1pps'>detected</param>";
      $xml_reply .=     "<param key='sensor1'>27.0</param>";
      $xml_reply .=     "<param key='sensor2'>28.0</param>";
      $xml_reply .=     "<param key='input0_volts'>1.23</param>";
      $xml_reply .=     "<param key='input1_volts'>4.56</param>";
      $xml_reply .=     "<param key='input2_volts'>7.89</param>";
      $xml_reply .=     "<param key='input3_volts'>1.23</param>";
      $xml_reply .=     "<param key='input4_volts'>4.56</param>";
      $xml_reply .=     "<param key='input5_volts'>7.89</param>";
      $xml_reply .=     "<param key='fpga_status'>programmed</param>";
      $xml_reply .=     "<param key='fpga_error_state'>None</param>";
      $xml_reply .=     "<param key='fpga_temp'>43.21</param>";
      $xml_reply .=     "<param key='8bit_window'>0</param>";
      $xml_reply .=     "<param key='udp_channels'>0-128</param>";
      $xml_reply .=     "<param key='udp_sequence_reset'>0</param>";
      $xml_reply .=     "<param key='udp_dest_mac'>AA:BB:CC:DD:EE:FF</param>";
      $xml_reply .=     "<param key='udp_dest_ip'>000.111.222.333</param>";
      $xml_reply .=     "<param key='udp_dest_port'>12345</param>";
      $xml_reply .=   "</params>";
      $xml_reply .= "</pfb_board>";

      $xml_reply .= "</pfb_board_update>";
    }
    header('Content-type: text/xml');
    echo $xml_reply;
  }
}

handleDirect("pfb_board");

