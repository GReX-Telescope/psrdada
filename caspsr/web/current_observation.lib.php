<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class current_observation extends caspsr_webpage 
{

  function current_observation()
  {
    caspsr_webpage::caspsr_webpage();
  }

  function javaScriptCallback()
  {
    return "current_obs_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new caspsr();
?>
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
        var url = "current_observation.lib.php?update=true&host="+host+"&port="+port;

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
  function printHTML($divargs="") 
  {
    $this->openBlockHeader("Current Observation", $divargs);
?>
    <table border=0 cellspacing=0 cellpadding=0>
      <tr>
        <td>
          <table cellpadding=1 cellspacing=0 border=0>
            <tr>
              <th align="right">Source</th>
              <td align="left"><span id ="SOURCE"></span></td>
            </tr>
            <tr>
              <th align="right">UTC_START</th>
              <td align="left"><span id ="UTC_START"></span></td>
            </tr>
            <tr>
              <th align="right">RA</th>
              <td align="left"><span id ="RA"></span></td>
            </tr>
            <tr>
              <th align="right">DEC</th>
              <td align="left"><span id ="DEC"></span></td>
            </tr>
          </table>
        </td>
        <td width=20>&nbsp;</td>
        <td>
          <table cellpadding=0 cellspacing=0 border=0>
            <tr>
              <th align="right">CFREQ</th>
              <td align="left"><span id ="CFREQ"></span>&nbsp;MHz</td>
            </tr>
            <tr>
              <th align="right">P0</th>
              <td align="left"><span id ="P0"></span>&nbsp;ms</td>
            </tr>
            <tr>
              <th align="right">DM</th>
              <td align="left"><span id ="DM"></span></td>
            </tr>
            <tr>
              <th align="right">BW</th>
              <td align="left"><span id ="BANDWIDTH"></span>&nbsp;MHz</td>
            </tr>
          </table>
        </td>
        <td width=20>&nbsp;</td>
        <td valign=top>
          <table cellpadding=0 cellspacing=0 border=0>
            <tr>
              <th align="right">Integrated</th>
              <td align="left"><span id ="INTEGRATED"></span></td>
            </tr>
            <tr>
              <th align="right">SNR</th>
              <td align="left"><span id ="SNR"></span></td>
            </tr>
            <tr>
              <th align="right">PROC FILE</th>
              <td align="left"><span id ="PROC_FILE"></span></td>
            </tr>
            <tr>
              <th align="right">Project ID</th>
              <td align="left"><span id ="PID"></span></td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($host, $port)
  {

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "curr_obs\r\n");
      $string = socketRead($socket);
      socket_close($socket);

    } else {
      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;

  }
}
handledirect("current_observation");
