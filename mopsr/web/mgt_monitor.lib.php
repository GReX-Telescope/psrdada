<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class mgt_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $pfb_per_arm = 12;

  var $arm_prefixes = array("EG", "WG");

  var $modules_per_pfb = 16;

  var $update_secs = 5;

  var $use_socket_connection = false;

  function mgt_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MGT Lock Monitor";
    $this->sidebar_width = "200";

    if (isset($_GET["update_secs"]))
      $this->update_secs = $_GET["update_secs"];

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "mgt_monitor_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function reset_other_cells(excluded) 
      {
        var cells = document.getElementsByTagName('td');
        var i=0;
        for (i=0; i< cells.length; i++) 
        {
          if ((cells[i].id.length > 4) && (excluded.indexOf(cells[i].id) == -1))
          {
            cells[i].style.backgroundColor = "grey";
          }
        }
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_mgt_monitor_request(xml_request) 
      {
        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var i, j, k
            var excluded_cells = Array();

            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var mgt_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("mgt_monitor_error").innerHTML = "[" + mgt_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var pfbs = xmlObj.getElementsByTagName ("pfb");
            var inputs;
            for (i=0; i<pfbs.length; i++)
            {
              pfb_id = pfbs[i].getAttribute("id");
              inputs = pfbs[i].childNodes;
              for (j=0; j<inputs.length; j++)
              {
                input_id = inputs[j].getAttribute("id");
                locked = inputs[j].getAttribute("locked");
                cell_id = pfb_id + "_" + input_id
                excluded_cells.push(cell_id);

                if (locked == "true")
                  document.getElementById (cell_id).style.backgroundColor = "#00ee00";
                else if (locked == "false")
                  document.getElementById (cell_id).style.backgroundColor = "red";
                else
                  document.getElementById (cell_id).style.backgroundColor = "yellow";
              }
            }
            reset_other_cells (excluded_cells);
          }
        }
      }
                  
      function mgt_monitor_request() 
      {
        var url = "mgt_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_mgt_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

      function mgt_update_secs()
      {
        idx = document.getElementById("update_secs").selectedIndex;
        update_secs = document.getElementById("update_secs").options[idx].value;
        document.location = "mgt_monitor.lib.php?single=true&update_secs="+update_secs;
      }

    </script>

    <style type="text/css">
    
      table.mgt_monitor {
        border-collapse: none;
        border-spacing: 1px;
      }

      table.mgt_monitor td {
        padding: 2px;
      }

      table.mgt_monitor th {
        padding-left: 5px;
        padding-right: 5px;
        width: 14px;
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
    $this->openBlockHeader("MGT Lock Monitor&nbsp;&nbsp;&nbsp;<span id='mgt_monitor_error'></span>");

    echo  "Update Frequency: ";
    echo "   <select name='update_secs' id='update_secs' onChange='mgt_update_secs()'>\n";
    $secs_options = array(5, 10, 30, 60);
    foreach ($secs_options as $opt)
    {
      if ($opt == $this->update_secs)
        echo "    <option value='".$opt."' selected>".$opt."</option>\n";
      else
        echo "    <option value='".$opt."'>".$opt."</option>\n";
    }
    echo "   </select>\n";


?>
    <table border=0 class='mgt_monitor'>
      <tr><th>Board</th><th colspan='<?echo $this->modules_per_pfb?>'>Modules</th></tr>
      <tr><th></th>
<?
    for ($i=0; $i<$this->modules_per_pfb; $i++)
    {
      echo "<th>".$i."</th>";
      if ($i && (($i+1) % 4 == 0))
        echo "<th>&nbsp;</th>\n";

    }
    echo "</tr>\n";

    $this->signal_paths = $this->inst->readSignalPaths();

    for ($iarm=0; $iarm < count($this->arm_prefixes); $iarm++)
    {
      for ($irow=0; $irow < $this->pfb_per_arm; $irow++)
      {
        $pfb_id = sprintf("%s%02d", $this->arm_prefixes[$iarm], ($irow+1));

        echo "<tr id='".$pfb_id."'>\n";
        echo "<th>\n";
        echo "  <div id='".$pfb_id."_name' class='name'>".$pfb_id."</div>\n";
        echo "  <div id='".$pfb_id."_msg' class='msg'></div>\n";
        echo "</th>\n";
        for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
        {
          $module_id = $pfb_id."_".($imod);
          $title = $this->signal_paths[$module_id];

          echo "<td id='".$module_id."' style='background-color: grey;'>&nbsp;</td>";
          if ($imod && (($imod+1) % 4 == 0))
            echo "<td>&nbsp;</td>\n";
        }
        echo "</tr>\n";
      }
    }

    echo "</table>\n";
  ?>
  </center>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = $this->inst->config["SERVER_HOST"];
    $port = $this->inst->config["SERVER_WEB_MONITOR_PORT"];
    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    # now prepare the reply
    $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml .= "<mgt_monitor_update>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<url_prefix>mopsr</url_prefix>";

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok")
    {
      $data = "";
      $response = "initial";

      if ($result == "ok") 
      {
        $xml .=   "<socket_connection host='".$host."' port='".$port."'>ok</socket_connection>"; 

        $xml .= "<images>";
        $bytes_written = socketWrite($socket, "mgt_lock_info\r\n");
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
        $xml .= $data;
        $xml .="</images>";
      }
    }
    else
    {
      $tmp = "";
      $tmp .= "<error>MPSR Web Monitor daemon not running</error>";

      $cmd = "find ".$this->inst->config["SERVER_UDP_MONITOR_DIR"]." -name '2???-??-??-??:??:??.*.*.??.?.*x*.png' -printf '%f\n' | sort -n";
      $images = Array();
      $lastline = exec($cmd, $images, $rval);
      $to_use = Array();
      if (($rval == 0) && (count($images) > 0))
      {
        # use associative array to store only the most recent images of a input + type + resolution
        foreach ($images as $image)
        {
          list ($time, $pfb, $input, $type, $locked, $res, $ext) = explode(".", $image);
          if (!array_key_exists($pfb, $to_use))
            $to_use[$pfb] = Array();
          if (!array_key_exists($input, $to_use[$pfb]))
            $to_use[$pfb][$input] = Array();
          $to_use[$pfb][$input] = ($locked == 'L') ? "true" : "false";
        }

        $tmp .= "<images>";
        $pfbs = array_keys($to_use);
        foreach ($pfbs as $pfb)
        {
          $tmp .= "<pfb id='".$pfb."'>";

          $inputs = array_keys($to_use[$pfb]);
          foreach ($inputs as $input)
          {
            $tmp .= "<input id='".$input."' locked='".$to_use[$pfb][$input]."'/>";
          }
          $tmp .= "</pfb>";
        }
        $tmp .= "</images>";
      }
      $xml .= $tmp;
    }

    $xml .= "</mgt_monitor_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function getUDPMonitorPlots ($pfb_id, $plot_type)
  {
    $plots = Array();

    # get a listing of the images for this PFB board
    $cmd = "find ".$this->inst->config["SERVER_UDP_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$pfb_id.".*.".$plot_type.".?.*x*.png' -printf '%f\n' | sort -n";
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
        list ($time, $rid, $module, $type, $locked, $res, $ext) = explode(".", $image);
        $to_use[$module][$type.".".$res] = $image;
      }

    }
    return $to_use;
  }
}

handleDirect("mgt_monitor");
