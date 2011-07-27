<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class archival_summary extends caspsr_webpage 
{

  function archival_summary()
  {
    caspsr_webpage::caspsr_webpage();
  }

  function javaScriptCallback()
  {
    return "archival_summary_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new caspsr();
?>
    <script type='text/javascript'>  

      // handle the response from an archival_summary request
      function handle_archival_summary_request( as_http_request) 
      {
        if ( as_http_request.readyState == 4) {
          var response = String(as_http_request.responseText)

          if (response.indexOf("Could not connect to") == -1) 
          {

            var lines = response.split(";;;");
            var values;

            for (i=0; i<lines.length; i++) 
            {
              values = lines[i].split(":::");
              if (values[0]) 
              {
                if (document.getElementById(values[0])) {
                  document.getElementById(values[0]).innerHTML = values[1];
                }
              }
            }
          }
        }
      }

      // generate an obsevartaion info request
      function archival_summary_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "archival_summary.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          as_http_request = new XMLHttpRequest();
        else
          as_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        as_http_request.onreadystatechange = function() {
          handle_archival_summary_request( as_http_request)
        };
        as_http_request.open("GET", url, true);
        as_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML($divargs="") 
  {
    $this->openBlockHeader("Archival Summary", $divargs);
?>
    <table cellpadding=1 cellspacing=0 border=0>
      <tr>
        <th align="right">Failed</th>
        <td align="left"><span id="failed"></span></td>
      </tr>
      <tr>
        <th align="right">Finished</th>
        <td align="left"><span id="finished"></span></td>
      </tr>
      <tr>
        <th align="right">Transferred</th>
        <td align="left"><span id="transferred"></span></td>
      </tr>
      <tr>
        <th align="right">Deleted</th>
        <td align="left"><span id="deleted"></span></td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "archival_info\r\n");
      $string = socketRead($socket);
      socket_close($socket);

    } else {
      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;

  }
}
handledirect("archival_summary");
