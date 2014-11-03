<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class plot_window extends mopsr_webpage 
{

  var $inst = 0;
  var $verbose = false;

  function plot_window()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Plot Window";
    $this->verbose = isset($_GET["verbose"]);
  }

  function javaScriptCallback()
  {
    return "plot_window_request();";
  }

  function printJavaScriptHead()
  {
    $this->inst = new mopsr();

?>
    <script type='text/javascript'>  

      function popPlotWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1400,height=870');");
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_plot_window_request(pw_xml_request) 
      {
        if (pw_xml_request.readyState == 4)
        {
          var xmlDoc = pw_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            var pfbs = xmlObj.getElementsByTagName("pfb");

            var i = 0;
            for (i=0; i<pfbs.length; i++)
            {
              var pfb = pfbs[i];
              var pfb_name = pfb.getAttribute("name");

              var j = 0;
              for (j=0; j<pfb.childNodes.length; j++)
              {
                img = pfb.childNodes[j];
                if (img.nodeType == 1)
                { 
                  var type = img.getAttribute("type");
                  document.getElementById(type).src = http_server + url_prefix + img.childNodes[0].nodeValue;
                }
              }
            }
          }
        }
      }
                  
      function plot_window_request() 
      {
        var host = "<?echo $this->inst->config["SERVER_HOST"];?>";
        var port = "<?echo $this->inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "plot_window.lib.php?update=true&host="+host+"&port="+port;

        document.getElementById('bp').src = "bpplot.lib.php?update=true"

        if (window.XMLHttpRequest)
          pw_xml_request = new XMLHttpRequest();
        else
          pw_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        pw_xml_request.onreadystatechange = function() {
          handle_plot_window_request(pw_xml_request)
        };
        pw_xml_request.open("GET", url, true);
        pw_xml_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=4>
      <tr>
        <td><img id='ts' src=''/></td>
        <td><img id='hg' src=''/></td>
      </tr>
      
      <tr>
        <td><img id='wf' src=''/></td>
        <td><img id='bp' src=''/></td>
      </tr>
    </table>
  </center>

<?
  }

  function printUpdateHTML($get)
  {
    $host = "mpsr1";
    $port = "32013";

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    $data = "";
    $response = "initial";

    $cmd = $get["cmd"];
    $ant = $get["ant"];
    $chan = $get["chan"];

    if ($result == "ok") 
    {
      $to_send = $get["cmd"];
      if (isset($get["ant"]))
        $to_send .= "&ant=".$get["ant"];
      if (isset($get["chan"]))
        $to_send .= "&chan=".$get["chan"];
      if (isset($get["size"]))
        $to_send .= "&size=".$get["size"];
      if (isset($get["log"]))
        $to_send .= "&log=".$get["log"];
      if (isset($get["plain"]))
        $to_send .= "&plain=".$get["plain"];
      if (isset($get["transpose"]))
        $to_send .= "&transpose=".$get["transpose"];
  
      $bytes_written = socketWrite($socket, $to_send."\r\n");

      $img_data = "";
      $data = socket_read($socket, 8192, PHP_BINARY_READ);
      $img_data = $data;
      while ($data)
      {
         $data = socket_read($socket, 8192, PHP_BINARY_READ);
        $img_data .= $data;
      }
      if ($socket)
        socket_close($socket);
      $socket = 0;
      header('Content-Type: image/png');
      header('Content-Disposition: inline; filename="image.png"');
      echo $img_data;
      return;
    } 
  }
}

handleDirect("plot_window");

