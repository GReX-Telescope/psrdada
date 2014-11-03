<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class pfb_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "64x48";

  var $num_pfb_east = 11;

  var $num_pfb_west = 11;

  var $modules_per_pfb = 16;

  var $default_plot = "hg";

  var $plot_types = array("hg", "ts", "ff");

  var $plot_titles = array("ts" => "TimeSeries", "ff" => "Spectrum", "hg" => "Histogram");

  var $update_secs = 5;

  var $use_socket_connection = false;

  function pfb_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "PFB Monitor";
    $this->sidebar_width = "200";

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "pfb_monitor_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      var active_img = "hg";

      function reset_other_imgs(excluded) 
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

      function reset_other_trs(excluded) 
      {
        var trs = document.getElementsByTagName('tr');
        var i=0;

        for (i=0; i< trs.length; i++) 
        {
          if ((trs[i].id != "") && (excluded.indexOf(trs[i].id) == -1))
          {
            trs[i].style.backgroundColor = "#999999";
          }
        }
      }


      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_pfb_monitor_request(xml_request) 
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
            var pfb_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("pfb_monitor_error").innerHTML = "[" + pfb_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var idx = document.getElementById('plot_type').selectedIndex;
            var plot_type = document.getElementById('plot_type').options[idx];

            var pfb_boards = xmlObj.getElementsByTagName ("pfb_board");

            var i, j;
            var excluded_imgs = Array();
            var excluded_trs  = Array();

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
                    if ((param.nodeType == 1) && (param.getAttribute("key") == "board_health"))
                    {
                      var tr_id = document.getElementById(pfb_board_id);
                      excluded_trs.push(pfb_board_id);
                      if (param.getAttribute("state") == "OK")
                        tr_id.style.backgroundColor = "#22FF22";
                      if (param.getAttribute("state") == "WARNING")
                        tr_id.style.backgroundColor = "#FFFF00";
                      if (param.getAttribute("state") == "ERROR")
                        tr_id.style.backgroundColor = "#FF0000";
                      var board_msg = "";
                      try {
                        board_msg = param.childNodes[0].nodeValue;
                      } catch (e) {
                      }
                      document.getElementById(pfb_board_id+"_msg").innerHTML = board_msg;
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
                          var img_id = pfb_board_id + "_" + module_id;
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
            reset_other_trs (excluded_trs);
          }
        }
      }
                  
      function pfb_monitor_request() 
      {
        var url = "pfb_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_pfb_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type="text/css">
    
      table.pfb_monitor {
        border-collapse: separate;
        border-spacing: 0px 5px;
      }

      table.pfb_monitor td {
        padding-top: 2px;
        padding-bottom: 2px;
      }

      table.pfb_monitor th {
        padding-left: 5px;
        padding-right: 5px;
      }

      table.pfb_monitor img {
        margin-left:  2px;
        margin-right: 2px;
      }

      .name {
        font-size: 15px;
      }

      .msg{
        font-size: 10px;
        font-weight: normal; 
      }

    </style>

<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("PFB System Monitor&nbsp;&nbsp;&nbsp;<span id='pfb_monitor_error'></span>");
    list ($xres, $yres) = split("x", $this->img_size);
?>
    <table border=0 class='pfb_monitor'>
      <tr><th>Board</th><th colspan='<?echo $this->modules_per_pfb?>'>Modules</th></tr>
      <tr><th></th>
<?
      for ($i=1; $i<=$this->modules_per_pfb; $i++)
      {
        echo "<th>".$i."</th>";
      }
      echo "</tr>\n";


    for ($irow=0; $irow < $this->num_pfb_east; $irow++)
    {
      $pfb    = sprintf ("East&nbsp;%02d", ($irow + 1));
      $pfb_id = sprintf ("e%02d", ($irow + 1));

      echo "<tr id='".$pfb_id."'>\n";
      echo "<th>\n";
      echo "  <div id='".$pfb_id."_name' class='name'>".$pfb."</div>\n";
      echo "  <div id='".$pfb_id."_msg' class='msg'></div>\n";
      echo "</th>\n";
      # east bays
      for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
      {
        $module_id = $pfb_id."_".($imod);
        echo "<td>";
        echo "<a href='pfb_board.lib.php?single=true&pfb_id=".$pfb_id."&pfb_name=".$pfb."'>";
        echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$module_id."'/>";
        echo "</a>";
        echo "</td>\n";
      }
      echo "</td>\n";
      echo "</tr>\n";
    }

    # west bays
    for ($irow=0; $irow < $this->num_pfb_west; $irow++)
    {
      $pfb    = sprintf ("West&nbsp;%02d", ($irow + 1));
      $pfb_id = sprintf ("w%02d", ($irow + 1));
        
      echo "<tr id='".$pfb_id."'>\n";
      echo "<th>\n";
      echo "  <div id='".$pfb_id."_name' class='name'>".$pfb."</div>\n";
      echo "  <div id='".$pfb_id."_msg' class='msg'></div>\n";
      echo "</th>\n";

      for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
      {
        $module_id = $pfb_id."_".($imod);
        echo "<td>";
        echo "<a href='pfb_board.lib.php?single=true&pfb_id=".$pfb_id."&pfb_name=".$pfb."'>";
        echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$module_id."'/>";
        echo "</a>";
        echo "</td>\n";
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


  function printSideBarHTML()
  {
    $this->openBlockHeader("Plot Type");

    echo "<table>\n";
    echo " <tr>\n";
    echo "  <td>Plot Type</td>\n";
    echo "  <td>\n";
    echo "   <select name='plot_type' id='plot_type'>\n";
    foreach ($this->plot_types as $plot_type)
    {
      echo "    <option name='".$plot_type."'>".$this->plot_titles[$plot_type]."</option>\n";
    }

    echo "   </select>\n";
    echo "  </td>\n";
    echo " </tr>\n";
    echo "</table>\n";

    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = "localhost";
    $port = $this->inst->config["SERVER_PFB_MONITOR_PORT"];

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    $xml_cmd  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_cmd .= "<mpsr_pfbmonitor_command>";
    $xml_cmd .=   "<pfb_boards>";
    $xml_cmd .=     "<pfb_board id='all'/>";
    $xml_cmd .=   "</pfb_boards>";
    $xml_cmd .=   "<params>";
    $xml_cmd .=     "<param key='board_health'/>";
    $xml_cmd .=   "</params>";
    $xml_cmd .=   "<plots>";
    $xml_cmd .=     "<plot type='hg'/>";
    $xml_cmd .=   "</plots>";
    $xml_cmd .= "</mpsr_pfbmonitor_command>";


    # now prepare the reply
    $xml  = "<pfb_monitor_update>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<url_prefix>mopsr</url_prefix>";
    $xml .=   "<img_prefix>monitor/pfb</img_prefix>";

    if ($this->use_socket_connection)
    {
      list ($socket, $result) = openSocket($host, $port);

      $data = "";
      $response = "initial";

      if ($result == "ok") 
      {
        $xml .=   "<socket_connection host='".$host."' port='".$port."'>ok</socket_connection>"; 
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
      $tmp = "";
      $tmp .= "<error>running in test mode</error>";
      $tmp .= "<pfb_monitor_mpsr_reply>";
      for ($irow=0; $irow < $this->num_pfb_east; $irow++)
      {
        $bay = sprintf ("%02d", $irow + 1);
        $pfb_id = "e".$bay;

        $plots = $this->getPFBMonitorPlots ($pfb_id, "hg");

        $tmp .= "<pfb_board id='".$pfb_id."'>";
        $tmp .=   "<params>";
        $tmp .=     "<param key='board_health' state='OK'></param>";
        $tmp .=   "</params>";
        $tmp .=   "<plots>";
        for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
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
        $tmp .= "</pfb_board>";
      }

      for ($irow=0; $irow < $this->num_pfb_west; $irow++)
      {
        $bay = sprintf ("%02d", $irow + 1);
        $pfb_id = "w".$bay;

        $plots = $this->getPFBMonitorPlots ($pfb_id, "hg");

        $tmp .= "<pfb_board id='".$pfb_id."'>";
        $tmp .=   "<params>";
        $tmp .=     "<param key='board_health' state='ERROR'></param>";
        $tmp .=   "</params>";
        $tmp .=   "<plots>";
        for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
        {
          $tmp .=     "<module id='".$imod."'>";
          $mods = $plots[$imod];
          foreach ($mods as $image)
          {
            list ($time, $pfb_id, $module, $type, $res, $ext) = explode(".", $image);
            list ($xres, $yres) = split('x', $res);
            $tmp .=       "<plot type='".$type."' width='".$res."' height='".$res."'>pfb_monitor/".$image."</plot>";
          }
          $tmp .=     "</module>";
        }
        $tmp .=   "</plots>";
        $tmp .= "</pfb_board>";
      }

      $tmp .= "</pfb_monitor_mpsr_reply>";
      $xml .= $tmp;
    }

    $xml .= "</pfb_monitor_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function getPFBMonitorPlots ($pfb_id, $plot_type)
  {
    $plots = Array();

    # get a listing of the images for this PFB board
    $cmd = "find ".$this->inst->config["SERVER_PFB_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$pfb_id.".*.".$plot_type.".*x*.png' -printf '%f\n' | sort -n";
    $lines = Array();
   
    $to_use = Array();
    for ($i=0; $i<$this->modules_per_pfb; $i++)
    { 
      $to_use[$i] = Array();
    }
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

handleDirect("pfb_monitor");
