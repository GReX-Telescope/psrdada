<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class udp_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "80x60";

  var $pfb_per_arm = 12;

  var $arm_prefixes = array("EG", "WG");

  var $modules_per_pfb = 16;

  var $default_plot = "hg";

  var $plot_types = array("wf", "hg", "ts", "bp");

  var $plot_titles = array("bp" => "BandPass", "ts" => "TimeSeries", "wf" => "Waterfall", "hg" => "Histogram");

  var $update_secs = 30;

  var $use_socket_connection = false;

  var $signal_paths = array();

  function udp_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "UDP Monitor";
    $this->sidebar_width = "200";

    if (isset($_GET["update_secs"]))
      $this->update_secs = $_GET["update_secs"];

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "udp_monitor_request();";
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
            imgs[i].height = 0;
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

      function handle_udp_monitor_request(xml_request) 
      {
        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var i, j, k, img_id, img_url;
            var excluded_imgs = Array();
            var excluded_trs  = Array();

            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var img_prefix  = xmlObj.getElementsByTagName("img_prefix")[0].childNodes[0].nodeValue;
            var udp_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("udp_monitor_error").innerHTML = "[" + udp_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var idx = document.getElementById('plot_type').selectedIndex;
            var plot_type = document.getElementById('plot_type').options[idx].value;

            var pfbs = xmlObj.getElementsByTagName ("pfb");
            var inputs;
            var images;
            for (i=0; i<pfbs.length; i++)
            {
              pfb_id = pfbs[i].getAttribute("id");
              excluded_trs.push(pfb_id);
              inputs = pfbs[i].childNodes;
              for (j=0; j<inputs.length; j++)
              {
                input_id = inputs[j].getAttribute("id");
  
                if (input_id <= 16)
                {
                  images = inputs[j].childNodes;
                  for (k=0; k<images.length; k++)
                  {
                    image = images[k];
                    if (image.nodeType == 1)
                    {
                      image_type = image.getAttribute("type");
                      image_width = image.getAttribute("width");
                      if ((image_type == plot_type) && (image_width < 300))
                      {
                        img_id = pfb_id + "_" + input_id;
                        img_url = http_server + "/" + url_prefix + "/" + img_prefix + "/" + image.childNodes[0].nodeValue;
                        excluded_imgs.push(img_id);
                        document.getElementById (img_id).src = img_url;
                        document.getElementById (img_id).height = 60;
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
                  
      function udp_monitor_request() 
      {
        var url = "udp_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_udp_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

      function udp_update_secs()
      {
        idx = document.getElementById("update_secs").selectedIndex;
        update_secs = document.getElementById("update_secs").options[idx].value;
        document.location = "udp_monitor.lib.php?single=true&update_secs="+update_secs;
      }

    </script>

    <style type="text/css">
    
      table.udp_monitor {
        border-collapse: separate;
        border-spacing: 0px 5px;
      }

      table.udp_monitor td {
        padding-top: 2px;
        padding-bottom: 2px;
      }

      table.udp_monitor th {
        padding-left: 5px;
        padding-right: 5px;
      }

      table.udp_monitor img {
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
    $this->openBlockHeader("UDP Input System Monitor&nbsp;&nbsp;&nbsp;<span id='udp_monitor_error'></span>");
    list ($xres, $yres) = split("x", $this->img_size);

    echo  "Plot Type: ";
    echo "   <select name='plot_type' id='plot_type' onChange='udp_monitor_request()'>\n";
    foreach ($this->plot_types as $plot_type)
    {
      if ($plot_type == "bp")
        echo "    <option value='".$plot_type."' selected>".$this->plot_titles[$plot_type]."</option>\n";
      else
        echo "    <option value='".$plot_type."'>".$this->plot_titles[$plot_type]."</option>\n";
    }
    echo "   </select>\n";

    echo  "&nbsp;&nbsp;&nbsp;Update Frequency: ";
    echo "   <select name='update_secs' id='update_secs' onChange='udp_update_secs()'>\n";
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
    <table border=0 class='udp_monitor'>
      <tr><th>Board</th><th colspan='<?echo $this->modules_per_pfb?>'>Modules</th></tr>
      <tr><th></th>
<?
    $this->signal_paths = $this->inst->readSignalPaths();

    for ($i=0; $i<$this->modules_per_pfb; $i++)
    {
      echo "<th>".$i."</th>";
    }
    echo "</tr>\n";

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
        # east bays
        for ($imod = 0; $imod < $this->modules_per_pfb; $imod++)
        {
          $module_id = $pfb_id."_".($imod);
          $title = $this->signal_paths[$module_id];
          echo "<td>";
          echo "<a href='udp_input.lib.php?single=true&pfb_id=".$pfb_id."&pfb_name=".$pfb_id."'>";
          echo "<img id='".$module_id."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px' title='".$title."'/>";
          echo "</a>";
          echo "</td>\n";
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


/*
  function printSideBarHTML()
  {
    $this->openBlockHeader("Plot Type");

    echo "<table>\n";
    echo " <tr>\n";
    echo "  <td>Plot Type</td>\n";
    echo "  <td>\n";
    echo "   <select name='plot_type' id='plot_type' onChange='udp_monitor_request()'>\n";
    foreach ($this->plot_types as $plot_type)
    {
      echo "    <option value='".$plot_type."'>".$this->plot_titles[$plot_type]."</option>\n";
    }

    echo "   </select>\n";
    echo "  </td>\n";
    echo " </tr>\n";
    echo "</table>\n";

    $this->closeBlockHeader();
  }
*/
  function printUpdateHTML($get)
  {
    $host = $this->inst->config["SERVER_HOST"];
    $port = $this->inst->config["SERVER_WEB_MONITOR_PORT"];
    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    # now prepare the reply
    $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml .= "<udp_monitor_update>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<url_prefix>mopsr</url_prefix>";
    $xml .=   "<img_prefix>monitor/udp</img_prefix>";

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok")
    {
      $data = "";
      $response = "initial";

      if ($result == "ok") 
      {
        $xml .=   "<socket_connection host='".$host."' port='".$port."'>ok</socket_connection>"; 

        $xml .= "<images>";
        $bytes_written = socketWrite($socket, "udp_monitor_info\r\n");
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
          $to_use[$pfb][$input][$type.".".$res] = $image;
        }

        $tmp .= "<images>";
        $pfbs = array_keys($to_use);
        foreach ($pfbs as $pfb)
        {
          $tmp .= "<pfb id='".$pfb."'>";

          $inputs = array_keys($to_use[$pfb]);
          foreach ($inputs as $input)
          {
            $tmp .= "<input id='".$input."'>";

            foreach (array_keys($to_use[$pfb][$input]) as $key)
            {
              list ($type, $res) = explode(".", $key);
              list ($xres, $yres) = explode("x", $res);
              $tmp .= "<plot type='".$type."' width='".$xres."' height='".$yres."'>".$to_use[$pfb][$input][$key]."</plot>";
            }
            $tmp .= "</input>";
          }
          $tmp .= "</pfb>";
        }
        $tmp .= "</images>";
      }
      $xml .= $tmp;
    }

    $xml .= "</udp_monitor_update>";

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

  function readPreferredModules ()
  {
    $file = $this->inst->config["CONFIG_DIR"]."/preferred_modules.txt";
    

  }
}

handleDirect("udp_monitor");
