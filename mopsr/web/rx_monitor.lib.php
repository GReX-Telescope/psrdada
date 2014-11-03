<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class rx_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "100x75";

  var $num_rx_east = 44;

  var $num_rx_west = 44;

  var $modules_per_rx = 4;

  var $rx_per_pfb = 4;

  var $default_plot = "hg";

  var $plot_types = array("hg", "sp");

  var $plot_titles = array("sp" => "Spectrum", "hg" => "Histogram");

  var $update_secs = 5;

  var $use_socket_connection = true;

  function rx_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR RX Monitor";

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "rx_monitor_request();";
  }

  function printJavaScriptHead()
  {

?>
    <script type='text/javascript'>  

      var active_img = "sp";

      function reset_other_imgs(excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if (excluded.indexOf(imgs[i].id) == -1)
          {
            imgs[i].src = "/images/blankimage.gif";
            imgs[i].height = 0;
          }
        }
      }

      function reset_other_tds(excluded) 
      {
        var tds = document.getElementsByTagName('td');
        var i=0;

        for (i=0; i< tds.length; i++) 
        {
          if ((tds[i].id != "") && (excluded.indexOf(tds[i].id) == -1))
          {
            tds[i].style.backgroundColor = "#999999";
          }
        }
      }


      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_rx_monitor_request(xml_request) 
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
            var rx_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("rx_monitor_error").innerHTML = "[" + rx_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var rx_boards = xmlObj.getElementsByTagName ("rx_board");
            var idx = document.getElementById('plot_type').selectedIndex;
            var active_img = document.getElementById('plot_type').options[idx].value;

            var i, j;
            var excluded_imgs = Array();
            var excluded_tds  = Array();

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
                    if ((param.nodeType == 1) && (param.getAttribute("key") == "board_health"))
                    {
                      var td_id = document.getElementById(rx_board_id);
                      excluded_tds.push(rx_board_id);
                      if (param.getAttribute("state") == "OK")
                        td_id.style.backgroundColor = "#22FF22";
                      if (param.getAttribute("state") == "WARNING")
                        td_id.style.backgroundColor = "#FFFF00";
                      if (param.getAttribute("state") == "ERROR")
                        td_id.style.backgroundColor = "#FF0000";
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
                        if (type == active_img)
                        {
                          var img_id = rx_board_id + "_" + module_id;
                          var imgurl = http_server + "/" + url_prefix + "/" + img_prefix + "/" + img.childNodes[0].nodeValue;
                          if (parseInt(img.getAttribute("width")) < 400)
                          {
                            excluded_imgs.push(img_id);
                            document.getElementById (img_id).src = imgurl;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            reset_other_imgs (excluded_imgs);
            reset_other_tds (excluded_tds);
          }
        }
      }
                  
      function rx_monitor_request() 
      {
        var idx = document.getElementById('plot_type').selectedIndex;
        var plot_type = document.getElementById('plot_type').options[idx].value;

        var url = "rx_monitor.lib.php?update=true&plot_type="+plot_type;

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_rx_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type="text/css">
    
      table.rx_monitor {
        border-spacing: 4;
      }

      table.rx_monitor td {
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 1px;
        padding-right: 1px;
      }

      table.rx_monitor img {
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
    $this->openBlockHeader("Receiver Board Monitor&nbsp;&nbsp;&nbsp;<span id='rx_monitor_error'></span>");
    list ($xres, $yres) = split("x", $this->img_size);

    echo "   <select name='plot_type' id='plot_type' onChange='rx_monitor_request()'>\n";
    foreach ($this->plot_types as $plot_type)
    {
      if ($plot_type == $this->default_plot)
        echo "    <option value='".$plot_type."' selected>".$this->plot_titles[$plot_type]."</option>\n";
      else
        echo "    <option value='".$plot_type."'>".$this->plot_titles[$plot_type]."</option>\n";
    }
    echo "   </select>\n";
?>
    <table border=0 class='rx_monitor'>
      <tr>
<?
        $rx_id = "RX_TEST_BOARD";
        echo "<th>".$rx_id."</th>\n";
        echo "<td id='".$rx_id."' valign='center'>\n";
        for ($irx = 0; $irx < $this->modules_per_rx; $irx++)
        {
          $module_id = $rx_id."_".($irx);
          echo "<a href='rx_board.lib.php?single=true&rx_id=".$rx_id."'>";
          echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$module_id."'/>";
          echo "</a>";
        }
        echo "</td>\n";
?>
      </tr>
    </table>


    <table border=0 class='rx_monitor'>
      <tr><th class='pad_right'>Bay</th><th colspan=1>East</th><th></th><th colspan=1>West</th></tr>
<?
    $num_rows= max($this->num_rx_east, $this->num_rx_west);

    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("%02d", $irow + 1);

      echo "<tr>\n";
      echo "<th>".$bay."</th>";

      $rx_id = "e".$bay;
      echo "<td id='".$rx_id."' valign='center'>\n";
      # east bays
      for ($irx = 0; $irx < $this->modules_per_rx; $irx++)
      {
        $module_id = $rx_id."_".($irx);
        echo "<a href='rx_board.lib.php?single=true&rx_id=".$rx_id."'>";
        echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$module_id."'/>";
        echo "</a>";
    
      }
      echo "</td>\n";

      echo "<td>&nbsp;</td>\n";

      # west bays
      $rx_id = "w".$bay;
        
      echo "<td id='".$rx_id."'>\n";
      for ($irx = 0; $irx < $this->modules_per_rx; $irx++)
      {
        $module_id = $rx_id."_".($irx);
        echo "<a href='rx_board.lib.php?single=true&rx_id=".$rx_id."'>";
        echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$module_id."'/>";
        echo "</a>";
      }
      echo "</td>\n";

      echo "</tr>\n";
    }
    echo "</table>\n";
?>
  </center>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = "localhost";
    $port = $this->inst->config["SERVER_RX_MONITOR_PORT"];
    if (array_key_exists("plot_type", $get))
      $plot_type = $get["plot_type"];
    else
      $plot_type = $this->default_plot;

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    $xml_cmd  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_cmd .= "<mpsr_rxmonitor_command>";
    $xml_cmd .=   "<rx_boards>";
    $xml_cmd .=     "<rx_board id='all'/>";
    $xml_cmd .=   "</rx_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='board_health'/>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<plots>";
    $xml_cmd .=     "<plot type='".$plot_type."'/>";
    $xml_cmd .=   "</plots>";
    $xml_cmd .= "</mpsr_rxmonitor_command>";

    list ($socket, $result) = openSocket($host, $port);

    # now prepare the reply
    $xml  = "<rx_monitor_update>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<url_prefix>mopsr</url_prefix>";
    $xml .=   "<img_prefix>monitor/rx</img_prefix>";
    $xml .=   "<socket_connection host='".$host."' port='".$port."'/>"; 

    if ($this->use_socket_connection)
    {
      $data = "";
      $response = "initial";

      if ($result == "ok") 
      {
        $bytes_written = socketWrite ($socket, $xml_cmd."\r\n");
        $max = 100;
        while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
        {
          list ($result, $response) = socketRead($socket);
          if ($result == "ok") 
          {
            $data .= $response;
          }
          $max--;
        }
        if (($result == "ok") && ($socket))
          socket_close($socket);
        $socket = 0;
        $data = str_replace ("<?xml version='1.0' encoding='ISO-8859-1'?>", "", $data);
        $xml .= $data;
      } 
      else
      {
        $xml .= "<error type='connection'>".$result."</error>";
      }
    }
    else
    {
      $tmp = "<rx_monitor_mpsr_reply>";
      for ($irow=0; $irow < $this->num_rx_east; $irow++)
      {
        $bay = sprintf ("%02d", $irow + 1);
        $rx_id = "e".$bay;

        $plots = $this->getRXMonitorPlots ($rx_id, $plot_type);

        $tmp .= "<rx_board id='".$rx_id."'>";
        $tmp .=   "<params>";
        $tmp .=     "<param key='board_health' state='OK'>No Errors</param>";
        $tmp .=   "</params>";
        $tmp .=   "<plots>";
        for ($imod = 0; $imod < $this->modules_per_rx; $imod++)
        {
          $tmp .=     "<module id='".$imod."'>";
          $mods = $plots[$imod];
          foreach ($mods as $image)
          {
            list ($time, $rid, $module, $type, $res, $ext) = explode(".", $image);
            list ($xres, $yres) = split('x', $res);
            $tmp .=       "<plot type='".$type."' width='".$res."' height='".$res."'>".$image."</plot>";
          }
          $tmp .=     "</module>";
        }
        $tmp .=   "</plots>";
        $tmp .= "</rx_board>";
      }

      for ($irow=0; $irow < $this->num_rx_east; $irow++)
      {
        $bay = sprintf ("%02d", $irow + 1);
        $rx_id = "w".$bay;

        $plots = $this->getRXMonitorPlots ($rx_id, $plot_type);

        $tmp .= "<rx_board id='".$rx_id."'>";
        $tmp .=   "<params>";
        $tmp .=     "<param key='board_health' state='OK'>No Errors</param>";
        $tmp .=   "</params>";
        $tmp .=   "<plots>";
        for ($imod = 0; $imod < $this->modules_per_rx; $imod++)
        {
          $tmp .=     "<module id='".$imod."'>";
          $mods = $plots[$imod];
          foreach ($mods as $image)
          {
            list ($time, $rid, $module, $type, $res, $ext) = explode(".", $image);
            list ($xres, $yres) = split('x', $res);
            $tmp .=       "<plot type='".$type."' width='".$res."' height='".$res."'>rx_monitor/".$image."</plot>";
          }
          $tmp .=     "</module>";
        }
        $tmp .=   "</plots>";
        $tmp .= "</rx_board>";
      }

      $tmp .= "</rx_monitor_mpsr_reply>";
      $xml .= $tmp;
    }

    $xml .= "</rx_monitor_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function getRXMonitorPlots ($rx_id, $plot_type)
  {
    $plots = Array();

    # get a listing of the images for this RX board
    $cmd = "find ".$this->inst->config["SERVER_RX_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$rx_id.".*.".$plot_type.".*x*.png' -printf '%f\n' | sort -n";
    $lines = Array();
    
    $to_use = Array("0" => Array() ,"1" => Array() ,"2" => Array(),"3" => Array());
    $lastline = exec($cmd, $lines, $rval);
    if (($rval == 0) && (count($lines) > 0))
    {
      # use associative array to store only the most recent images of a module + type + resolution
      foreach ($lines as $image)
      {
        list ($time, $rid, $module, $type, $res, $ext) = explode(".", $image);
        $to_use[$module][$type.".".$res] = $image;
      }

    }
    return $to_use;
  }
}

handleDirect("rx_monitor");
