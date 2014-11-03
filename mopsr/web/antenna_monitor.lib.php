<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class antenna_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "160x120";

  var $modules = array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  var $plot_types = array("bp", "ts", "wf", "hg");
  var $plot_titles = array("bp" => "Bandpass", "ts" => "TimeSeries", "wf" => "Waterfall", "hg" => "Histogram");

  var $update_secs = 5;


  function antenna_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Antenna Monitor";

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "antenna_monitor_request();";
  }

  function printJavaScriptHead()
  {

?>
    <script type='text/javascript'>  

      function reset_others(excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if (excluded.indexOf(imgs[i].id) == -1)
          {
            imgs[i].height = "0";
            imgs[i].src = "";
          }
        }
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_antenna_monitor_request(am_xml_request) 
      {
        if (am_xml_request.readyState == 4)
        {
          var xmlDoc = am_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var i, j, k, img_id, img_url;
            var excluded = Array();

            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var img_prefix  = xmlObj.getElementsByTagName("img_prefix")[0].childNodes[0].nodeValue;
            var udp_monitor_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("udp_monitor_error").innerHTML = "[" + udp_monitor_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var pfbs = xmlObj.getElementsByTagName ("pfb");
            var inputs;
            var images;
            for (i=0; i<pfbs.length; i++)
            {
              pfb_id = pfbs[i].getAttribute("id");
              inputs = pfbs[i].childNodes;
              for (j=0; j<inputs.length; j++)
              {
                input_id = inputs[j].getAttribute("id");
                images = inputs[j].childNodes;
                for (k=0; k<images.length; k++)
                {
                  image = images[k];
                  if (image.nodeType == 1)
                  {
                    image_type = image.getAttribute("type");
                    img_url = http_server + "/" + url_prefix + "/" + img_prefix + "/" + image.childNodes[0].nodeValue;
                    image_width = image.getAttribute("width");
                    img_id = image_type + "_" + input_id;
                    if (image_width < 300)
                    {
                      excluded.push(img_id);
                      document.getElementById (img_id).src = img_url;
                      document.getElementById (img_id).height = parseInt(image.getAttribute("height"));
                    }
                    else
                      document.getElementById (img_id).href = "javascript:popImage('"+img_url+"')";
                  }
                }
              }
            }
            reset_others(excluded);
          }
        }
      }
                  
      function antenna_monitor_request() 
      {
        var url = "antenna_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          am_xml_request = new XMLHttpRequest();
        else
          am_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        am_xml_request.onreadystatechange = function() {
          handle_antenna_monitor_request(am_xml_request)
        };
        am_xml_request.open("GET", url, true);
        am_xml_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>

<?
    list ($xres, $yres) = split("x", $this->img_size);
    echo "<table>\n";
    echo "<tr><td>ID</td>";
    foreach ($this->plot_types as $plot)
    {
      echo "<th>".$this->plot_titles[$plot]."</th>";
    }
    echo "</tr>\n";

    foreach ( $this->modules as $module )
    {
      echo "<tr>\n";
      echo   "<td id='".$module."_id'></td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "<td>\n";
        echo "<a id='".$plot."_".$module."_link'>";
        echo "<img id='".$plot."_".$module."' src='/images/blackimage.gif' width='".$xres."px' height='0px'/>";
        echo "</a>";
        echo "</td>\n";
      }
      echo "</tr>\n";
    }
    echo "</table>\n";
?>
  </center>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $this->inst->config["SERVER_HOST"];
    $port = $this->inst->config["SERVER_WEB_MONITOR_PORT"];
    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    $xml  = "<antenna_update>";
    $xml .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .=   "<url_prefix>mopsr</url_prefix>";
    $xml .=   "<img_prefix>monitor/udp</img_prefix>";
    $xml .=   "<socket_connection host='".$host."' port='".$port."'/>"; 

    $data = "";
    $response = "initial";

    if ($result == "ok") 
    {
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
    } 
    else
    {
      $xml .= "<error>".$result."</error>";
    }

    $xml .= "</antenna_update>";

    header('Content-type: text/xml');
    echo $xml;
  }
}

handleDirect("antenna_monitor");

