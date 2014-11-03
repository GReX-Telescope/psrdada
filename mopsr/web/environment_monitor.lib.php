<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class environment_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $verbose = false;

  var $types = array ("wind", "temperature");

  function environment_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Environment Monitor";
    $this->callback_freq = 2000;
    $this->verbose = isset($_GET["verbose"]);
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "environment_monitor_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1200,height=900');");
      }

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

      function handle_environment_monitor_request(em_xml_request) 
      {
        if (em_xml_request.readyState == 4)
        {
          var xmlDoc = em_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var excluded_imgs = Array();

            var imgs = xmlObj.getElementsByTagName("plot");
            for (i=0; i<imgs.length; i++)
            {
              var img = imgs[i];
              var img_type = img.getAttribute("type");
              var img_url = http_server + url_prefix + img.childNodes[0].nodeValue;
              document.getElementById (img_type + "_img").src = img_url;
              document.getElementById (img_type + "_link").href = img_url;
              excluded_imgs.push(img_type + "_img");
            }
          }
          reset_other_imgs (excluded_imgs);
        }
      }
                  
      function environment_monitor_request() 
      {
        var url = "environment_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          em_xml_request = new XMLHttpRequest();
        else
          em_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        em_xml_request.onreadystatechange = function() {
          handle_environment_monitor_request(em_xml_request)
        };
        em_xml_request.open("GET", url, true);
        em_xml_request.send(null);
      }

    </script>

<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=2>

      <tr>
        <th>Temperature</th>
        <th>Wind</th>
      </tr>
    
      <tr>
        <td href='' id='temperature_link'>
          <img id='temperature_img' src='/images/blankimage.gif' width='640px' height='480px'/>
        </td>
    
        <td href='' id='wind_link'>
          <img id='wind_img' src='/images/blankimage.gif' width='480px' height='640px'/>
        </td>
      </tr>
  
    </table>
  </center>

<?
  }

  function printUpdateHTML($get)
  {
    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    $xml = "<environment_update>";
    $xml .= "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .= "<url_prefix>/mopsr/</url_prefix>";

    $data = "";
    $response = "initial";
  
    $xml .= "<images>";
     
    $cmd = "find ".$this->inst->config["SERVER_ENV_MONITOR_DIR"]." -name '2???????_??:??:??_*.png' -printf '%f\n' | sort -n";
    $images = Array();
    $lastline = exec($cmd, $images, $rval);
    $to_use = Array();
    if (($rval == 0) && (count($images) > 0))
    {
      # use associative array to store only the most recent images of a module + type + resolution
      foreach ($images as $image)
      {
        list ($day, $time, $type, $ext) = preg_split ("/[_.]/", $image);
        $to_use[$type] = $image;
      }

      foreach ($to_use as $type => $image)
      {
        $xml .= "<plot type='".$type."'>monitor/environment/".$image."</plot>";
      }
    }
 
    $xml .="</images>";

    $xml .= "</environment_update>";

    header('Content-type: text/xml');
    echo $xml;
  }
}

handleDirect("environment_monitor");

