<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class rx_board extends mopsr_webpage 
{
  var $inst = 0;

  var $rx_id;

  var $plot_id;

  var $img_size = "200x150";

  var $modules = array (0, 1, 2, 3);

  var $plot_types = array("hg",  "sp");

  var $plot_titles = array("ts" => "TimeSeries", "sp" => "Spectrum", "hg" => "Histogram");

  var $params = array ("rx_monitor" => array ("rx_monitor_error"),
                       "Board_parameters" => array ("board_health"),
                       "DAC_parameters"   => array ("dac_temperature", "dac_gain"),
                       "supply_voltages"  => array("input0_volts", "input1_volts", "input2_volts", "input3_volts", "input4_volts", "input5_volts"),
                       "FPGA_parameters"  => array("fpga_status", "fpga_error_state", "fpga_temperature"));

  var $param_titles = array( "rx_monitor" => "RX Monitor", "rx_monitor_error" => "Errors", 
                             "Board_parameters" => "Board", "board_health" => "Health",
                             "DAC_parameters" => "DAC", "dac_temperature" => "Temp", "dac_gain" => "Gain", 
                             "supply_voltages" => "Power Supply Voltages",
                             "input0_volts" => "Input 0", "input1_volts" => "Input 1", 
                             "input2_volts" => "Input 2", "input3_volts" => "Input 3",
                             "input4_volts" => "Input 4", "input5_volts" => "Input 5",
                             "FPGA_parameters" => "FPGA", "fpga_status" => "Status",
                             "fpga_error_state" => "Error", "fpga_temperature" => "Temp");

  var $rw_params = array ("dac_gain");

  var $update_secs = 5;

  var $connect_to_socket = true;

  function rx_board()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR RX Board";
    $this->rx_id = $_GET["rx_id"];
    if (strcmp($this->rx_id, "RX_TEST_BOARD") == 0)
      $this->plot_id = "TEST";
    else
      $this->plot_id = $this->rx_id;
    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "rx_board_update_request();";
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
          if (excluded.indexOf(imgs[i].id) == -1)
          {
            imgs[i].src = "/images/blankimage.gif";
          }
        }
      }

      function handle_rx_board_update_request(xml_request) 
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
                document.getElementById("rx_monitor_error").innerHTML = error_message;
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
            var rx_boards = xmlObj.getElementsByTagName ("rx_board");

            // parse XML for each RX ID
            for (i=0; i<rx_boards.length; i++)
            {
              var rx_board = rx_boards[i];
              var rx_board_id = rx_board.getAttribute("id");

              // boards will have params and plots
              var nodes = rx_board.childNodes;
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
                        alert ("key="+key+ " val="+val);
                      }
                    }
                  } 
                }
                else if (nodes[j].nodeName == "plots")
                {
                  var modules = nodes[j].childNodes;
                  for (k=0; k<modules.length; k++)
                  {
                    var module = modules[k];
                    var module_id = module.getAttribute("id");

                    // for each image in this module
                    var images = module.childNodes;
                    for (l=0; l<images.length; l++)
                    {
                      img = images[l];
                      if (img.nodeType == 1)
                      {
                        var type = img.getAttribute("type");
                        var img_id = module_id + "_" + type;
                        var imgurl = http_server + "/" + url_prefix + "/" + img_prefix + "/" + img.childNodes[0].nodeValue;
                        excluded.push(img_id);
                        document.getElementById (img_id).src = imgurl;
                        imgurl = http_server + "/" + url_prefix + "/rx_plot.php?rx_id=<?echo $this->rx_id?>&"+
                                "mod="+module_id+"&type="+type+"&res=800x600";
                        document.getElementById (img_id + "_link").href = "javascript:popImage('"+imgurl+"')";
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
                  
      function rx_board_update_request() 
      {
        var url = "rx_board.lib.php?update=true&rx_id=<?echo $this->rx_id?>";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_rx_board_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

      function rx_board_action_request (key) 
      {
        var value = document.getElementById(key + "_new").value;
        var url = "rx_board.lib.php?action=true&rx_id=<?echo $this->rx_id?>&key="+key+"&value="+value;
        var xml_request;

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_rx_board_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>
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
        echo "  <td><div id='".$id."'></div></td>\n";
        if (in_array($param, $this->rw_params))
        {
          echo "  <td>";
          echo "<input type='text' id='".$id."_new' name='".$id."_new' size='4'/>";
          echo "<input type='button' onClick='rx_board_action_request(\"".$param."\")' value='Set'/>";
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
    $this->openBlockHeader("RX Board ".$this->rx_id);
    list ($xres, $yres) = split("x", $this->img_size);

    echo "<table>\n";
    echo " <tr>\n";
    echo "  <th>Module</th>\n";
    foreach ($this->plot_types as $plot)
    {
      echo "  <th>".$this->plot_titles[$plot]."</th>\n";
    }
    echo " </tr>\n";

    foreach ( $this->modules as $module )
    {
      echo " <tr>\n";
      echo "  <td id='".$module."_id'>".$module."</td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "  <td>";
        echo "<a id='".$module."_".$plot."_link'>";
        echo "<img id='".$module."_".$plot."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px'/>";
        echo "</a>";
        echo "</td>\n";
      }
      echo " </tr>\n";
    }
    echo "</table>\n";

    $this->closeBlockHeader();
  }

  function printActionHTML($get)
  { 
    # prepare the XML command to be sent to the RX Monitor daemon
    $xml_cmd  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_cmd .= "<mpsr_rxmonitor_command>";
    $xml_cmd .=   "<rx_boards>";
    $xml_cmd .=     "<rx_board id='".$this->plot_id."'/>";
    $xml_cmd .=   "</rx_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='".$get["key"]."'>".$get["value"]."</param>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<plots>";
    $xml_cmd .=     "<plot type='hg'/>";
    $xml_cmd .=     "<plot type='sp'/>";
    $xml_cmd .=   "</plots>";
    $xml_cmd .= "</mpsr_rxmonitor_command>";

    $host = "localhost";
    $port = $this->inst->config["SERVER_RX_MONITOR_PORT"];
   
    $xml_reply = "";

    if ($this->connect_to_socket)
    {
      $xml_reply .= "<rx_board_update>";
      $xml_reply .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>";
      $xml_reply .=   "<url_prefix>mopsr</url_prefix>";
      $xml_reply .=   "<img_prefix>monitor/rx</img_prefix>";

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

      $xml_reply .= "</rx_board_update>";
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
    $xml_cmd .= "<mpsr_rxmonitor_command>";
    $xml_cmd .=   "<rx_boards>";
    $xml_cmd .=     "<rx_board id='".$this->plot_id."'/>";
    $xml_cmd .=   "</rx_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='all'/>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<plots>";
    $xml_cmd .=     "<plot type='hg'/>";
    $xml_cmd .=     "<plot type='sp'/>";
    $xml_cmd .=   "</plots>";
    $xml_cmd .= "</mpsr_rxmonitor_command>";

    $host = "localhost";
    $port = $this->inst->config["SERVER_RX_MONITOR_PORT"];

    $xml_reply = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_reply .= "<rx_board_update>";
    $xml_reply .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml_reply .=   "<url_prefix>mopsr</url_prefix>";
    $xml_reply .=   "<img_prefix>monitor/rx</img_prefix>";

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

      $xml_reply .= "</rx_board_update>";
    }
    else
    {
      $xml_reply .=   "<rx_board id='".$this->rx_id."'>";

      # get a listing of the images for this RX board
      $cmd = "find ".$this->inst->config["SERVER_RX_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$this->rx_id.".*.??.*x*.png' -printf '%f\n' | sort -n";
      $images = Array();
      $lastline = exec($cmd, $images, $rval);
      if (($rval == 0) && (count($images) > 0))
      {
        $to_use = Array();

        # use associative array to store only the most recent images of a module + type + resolution
        foreach ($images as $image)
        {
          list ($time, $rid, $module, $type, $res, $ext) = explode(".", $image);
          if (!array_key_exists($module, $to_use))
            $to_use[$module] = Array();
          $to_use[$module][$type.".".$res] = $image;
        }

        $xml_reply .= "<plots>";
        # now build XML
        foreach (array_keys($to_use) as $module)
        {
          $xml_reply .= "<module id='".$module."'>";
          foreach (array_keys($to_use[$module]) as $key)
          {
            list ($type, $res) = explode(".", $key);
            list ($xres, $yres) = explode("x", $res);
            $xml_reply .= "<plot type='".$type."' width='".$xres."' height='".$yres."'>".$to_use[$module][$key]."</plot>";
          }
          $xml_reply .= "</module>";
        }
        $xml_reply .= "</plots>";
      }
      else
      {
        $xml_reply .= "<error return_value='".$rval."'>".$lastline."</error>";
      }

      $xml_reply .=   "<params>";
      $xml_reply .=     "<param key='board_health'>OK</param>";
      $xml_reply .=     "<param key='dac_temp'>27.0</param>";
      $xml_reply .=     "<param key='lna_gain'>4</param>";
      $xml_reply .=     "<param key='input0_volts'>1.23</param>";
      $xml_reply .=     "<param key='input1_volts'>4.56</param>";
      $xml_reply .=     "<param key='input2_volts'>7.89</param>";
      $xml_reply .=     "<param key='input3_volts'>1.23</param>";
      $xml_reply .=     "<param key='input4_volts'>4.56</param>";
      $xml_reply .=     "<param key='input5_volts'>7.89</param>";
      $xml_reply .=     "<param key='fpga_status'>programmed</param>";
      $xml_reply .=     "<param key='fpga_error_state'>None</param>";
      $xml_reply .=     "<param key='fpga_temp'>43.21</param>";
      $xml_reply .=   "</params>";
      $xml_reply .= "</rx_board>";

      $xml_reply .= "</rx_board_update>";
    }
    header('Content-type: text/xml');
    echo $xml_reply;
  }
}

handleDirect("rx_board");

