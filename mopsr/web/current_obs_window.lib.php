<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class current_obs extends mopsr_webpage 
{
  var $inst;

  function current_obs()
  {
    mopsr_webpage::mopsr_webpage();
    $this->inst = new mopsr();
  }

  function javaScriptCallback()
  {
    return "current_obs_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new mopsr();
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
                  if (values[0] == "UTC_START") {
                    var start = '<a target="_blank" href=result.lib.php?single=true&utc_start=';
                    document.getElementById(values[0]).innerHTML = start + values[1] + ">" + values[1] + "</a>";
                  }
                  else
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
        var url = "current_obs_window.lib.php?update=true";
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
    <table class='curr_obs' width='100%'>
      <tr>
        <th>SOURCE</th>
        <td><span id ="SOURCE"></span></td>
        <th>UTC_START</th>
        <td><span id ="UTC_START"></span></td>
        <th>DELAY_TRACKING</th>
        <td><span id ="DELAY_TRACKING"></span></td>
      </tr>
      <tr>
        <th>RA</th>
        <td><span id ="RA"></span></td>
        <th>PID</th>
        <td><span id ="PID"></span></td>
        <th>RFI_MITIGATION</th>
        <td><span id ="RFI_MITIGATION"></span></td>
      </tr>
      <tr>
        <th>DEC</th>
        <td><span id ="DEC"></span></td>
        <th>OBSERVER</th>
        <td><span id="OBSERVER"></span></td>
        <th>ANTENNA_WEIGHTS</th>
        <td><span id ="ANTENNA_WEIGHTS"></span></td>
      </tr>
    </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $this->inst->config["SERVER_HOST"];
    $port = $this->inst->config["SERVER_WEB_MONITOR_PORT"];
    $timeout = 1;

    $response = "";

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
handleDirect("current_obs");
