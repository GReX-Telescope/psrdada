<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class transient_window extends bpsr_webpage 
{

  function transient_window()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Transient Pipeline";
  }

  function javaScriptCallback()
  {
    return "transient_window_request();";
  }

  function printJavaScriptHead()
  {

    $inst = new bpsr();

?>
    <script type='text/javascript'>  

      var utc_start = "";

      function start_pipeline()
      {
        url = "transient_window.lib.php?action=start";
        tw_action_request(url);
      }

      function stop_pipeline()
      {
        url = "transient_window.lib.php?action=stop";
        tw_action_request(url);
      }
        
      function tw_action_request()
      {
        var tw_action_http_request;
        if (window.XMLHttpRequest)
          tw_action_http_request = new XMLHttpRequest();
        else
          tw_action_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        tw_action_http_request.onreadystatechange = function() 
        {
          handle_tw_action_request(tw_action_http_request);
        }

        tw_action_http_request.open("GET", url, true);
        tw_action_http_request.send(null);
      }

      function handle_tw_action_request(tw_action_http_request)
      {
        if (tw_action_http_request.readyState == 4)
        {
          var response = String(tw_action_http_request.responseText);
          alert(response);
        }
      }

      function handle_transient_window_request(tw_xml_request) 
      {
        if (tw_xml_request.readyState == 4)
        {
          var xmlDoc = tw_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            var cands = xmlObj.getElementsByTagName("transient_candidates");

            var i = 0;
            for (i=0; i<cands.length; i++)
            {
              var cand = cands[i];
              var img_element = document.getElementById("candidate");

              var j = 0;
              for (j=0; j<cand.childNodes.length; j++)
              {
                img = cand.childNodes[j];
                if ((img.nodeType == 1) && (img.getAttribute("type") == "dm_vs_time"))
                { 
                  img_element.src = http_server + url_prefix + img.childNodes[0].nodeValue;
                }
              }
            }
          }
        }
      }
                  
      function transient_window_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "transient_window.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          tw_xml_request = new XMLHttpRequest();
        else
          tw_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        tw_xml_request.onreadystatechange = function() {
          handle_transient_window_request(tw_xml_request)
        };
        tw_xml_request.open("GET", url, true);
        tw_xml_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <img src="/images/blankimage.gif" border=0 width=700 height=240 id="candidate" TITLE="Current Candidate" alt="alt">
  </center>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    $xml = "<plot_update>";
    $xml .= "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .= "<url_prefix>/bpsr/results/</url_prefix>";

    $data = "";
    $response = "initial";

    if ($result == "ok") 
    {
      $xml .= "<images>";
      $bytes_written = socketWrite($socket, "img_info\r\n");
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

    $xml .= "</plot_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function printActionHTML($get)
  {
  }
}

handleDirect("transient_window");

