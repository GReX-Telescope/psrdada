<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class observation_summary extends apsr_webpage 
{

  function observation_summary()
  {
    apsr_webpage::apsr_webpage();
  }

  function javaScriptCallback()
  {
    return "observation_summary_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new apsr();
?>
    <style type="text/css">
      table.curr_obs th {
        text-align: left;
        font-size: 8pt;
        padding-top: 0px;
        padding-bottom: 0px;
      }
      table.curr_obs td {
        text-align: left;
        font-size: 8pt;
        padding-left: 10px;
        padding-right: 30px;
        padding-top: 0px;
        padding-bottom: 0px;
      }
      table.curr_obs span {
        font-size: 8pt;
      }
    </style>

    <script type='text/javascript'>  

      // handle the response from an observation_summary request
      function handle_observation_summary_request( os_http_request) 
      {
        if ( os_http_request.readyState == 4) {
          var response = String(os_http_request.responseText)

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
      function observation_summary_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "observation_summary.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          os_http_request = new XMLHttpRequest();
        else
          os_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        os_http_request.onreadystatechange = function() {
          handle_observation_summary_request( os_http_request)
        };
        os_http_request.open("GET", url, true);
        os_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <table class='curr_obs' width='100%'>
      <tr>
        <th>Source</th>
        <td><span id ="SOURCE"></span></td>
        <th>BW</th>
        <td><span id ="BANDWIDTH"></span>&nbsp;MHz</td>
        <th>NBIT</th>
        <td><span id ="NBIT"></span></td>
      </tr>
      <tr>
        <th>RA</th>
        <td><span id ="RA"></span></td>
        <th>DM</th>
        <td><span id="DM"></span></td>
        <th>NPOL</th>
        <td><span id ="NPOL"></span></td>
      </tr>
      <tr>
        <th>DEC</th>
        <td><span id ="DEC"></span></td>
        <th>P0</th>
        <td><span id="P0"></span></td>
        <th>NPWC</th>
        <td><span id="NUM_PWC"></span></td>
      </tr>
      <tr>
        <th>UTC_START</th>
        <td><span id ="UTC_START"></span></td>
        <th>PID</th>
        <td><span id ="PID"></span></td>
        <th>Length</th>
        <td><span id="INTEGRATED"></span></td>
      </tr>
      <tr>
        <th>PROC_FILE</th>
        <td><span id ="PROC_FILE"></span></td>
        <th>CFREQ</th>
        <td><span id ="CFREQ"></span>&nbsp;MHz</td>
        <th>SNR</th>
        <td><span id="SNR"></span></td>
      </tr>
    </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "curr_obs\r\n");
      list ($result, $response) = socketRead($socket);
      socket_close($socket);

    } else {
      $response = "Could not connect to $host:$port<BR>\n";
    }

    echo $response;

  }
}
handledirect("observation_summary");
