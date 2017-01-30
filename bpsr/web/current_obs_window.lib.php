<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class current_obs extends bpsr_webpage 
{

  function current_obs()
  {
    bpsr_webpage::bpsr_webpage();
  }

  function javaScriptCallback()
  {
    return "current_obs_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new bpsr();
?>
    <style type="text/css">
      table.curr_obs th {
        text-align: left;
        font-size: 8pt;
      }
      table.curr_obs td {
        text-align: left;
        padding-left: 10px;
        padding-right: 30px;
        font-size: 8pt;
      }
      table.curr_obs span {
        font-size: 8pt;
      }
    </style>

    <script type='text/javascript'>  

      // handle the response from an current_obs request
      function handle_current_obs_request( co_http_request) 
      {
        if ( co_http_request.readyState == 4) {
          var response = String(co_http_request.responseText)

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
      function current_obs_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "current_obs_window.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          co_http_request = new XMLHttpRequest();
        else
          co_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        co_http_request.onreadystatechange = function() {
          handle_current_obs_request( co_http_request)
        };
        co_http_request.open("GET", url, true);
        co_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <table class='curr_obs'>
      <tr>
        <th>Source</th>
        <td><span id ="SOURCE"></span></td>
        <th>RA</th>
        <td><span id ="RA"></span></td>
        <th>Acc Length</th>
        <td><span id="ACC_LEN"></span></td>
        <th>REF BEAM</th>
        <td><span id="REF_BEAM"></span></td>
      </tr>
      <tr>
        <th>UTC_START</th>
        <td><span id ="UTC_START"></span></td>
        <th>DEC</th>
        <td><span id ="DEC"></span></td>
        <th>BW</th>
        <td><span id ="BANDWIDTH"></span>&nbsp;MHz</td>
        <th>NBEAM</th>
        <td><span id="NBEAM"></span></td>
      </tr>
      <tr>
        <th>CFREQ</th>
        <td><span id ="CFREQ"></span>&nbsp;MHz</td>
        <th>PROC_FILE</th>
        <td><span id ="PROC_FILE"></span></td>
        <th>Project ID</th>
        <td><span id ="PID"></span></td>
        <th>NPOL</th>
        <td><span id ="NPOL"></span></td>
      </tr>
    </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];
    $response = "";

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") 
    {
      $bytes_written = socketWrite($socket, "curr_obs\r\n");
      list ($result, $response) = socketRead($socket);
      if ($result == "ok")
        socket_close($socket);
    } 
    else
    {
      $response = "Could not connect to $host:$port<BR>\n";
    }

    echo $response;
  }
}
handledirect("current_obs");
