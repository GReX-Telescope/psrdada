<?PHP

include_once("hispec.lib.php");
include_once("hispec_webpage.lib.php");

class state_banner extends hispec_webpage 
{

  var $show_state;
  var $show_buttons;

  function state_banner()
  {
    hispec_webpage::hispec_webpage();

    $this->show_state = (isset($_GET["show_state"])) ? $_GET["show_state"] : "true";
    $this->show_buttons = (isset($_GET["show_buttons"])) ? $_GET["show_buttons"] : "true";
  }

  function javaScriptCallback()
  {
    if ($this->show_state == "true") 
      return "state_banner_request();";
    else
      return "";
  }

  function printJavaScriptHead()
  {
    $inst = new hispec();
?>
    <script type='text/javascript'>  

      function popUp(URL, type) 
      {
        var to = "toolbar=1";
        var sc = "scrollbars=1";
        var l  = "location=1";
        var st = "statusbar=1";
        var mb = "menubar=1";
        var re = "resizeable=1";

        options = to+","+sc+","+l+","+st+","+mb+","+re
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width=1024,height=768');");
      }

      // handle the response from an state_banner request
      function handle_state_banner_request( sb_http_request) 
      {
        if ( sb_http_request.readyState == 4) {
          var response = String(sb_http_request.responseText)

          if (response.indexOf("Could not connect to") == -1) 
          {
            var lines = response.split(":::");
            document.getElementById("hispec_state").innerHTML = "State: "+lines[0];
            document.getElementById("hispec_beams").innerHTML = "Beams: "+lines[1];
          }
        }
      }

      // generate an obsevartaion info request
      function state_banner_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["TCS_STATE_INFO_PORT"];?>";
        var url = "state_banner.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          sb_http_request = new XMLHttpRequest();
        else
          sb_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        sb_http_request.onreadystatechange = function() {
          handle_state_banner_request( sb_http_request)
        };
        sb_http_request.open("GET", url, true);
        sb_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <table cellspacing=0 cellpadding=0 border=0 width="100%">
      <tr>
        <td width="200px" height="60px"><font size="+4">HISPEC</font></td>
        <td height="60px" align=left style='padding-left: 20px'>
<?
    if ($this->show_state == "true") {
?>
          <span id="hispec_state" class="largetext"></span><br>
          <span id="hispec_beams" class="largetext"></span>
<?
    } 
?>
        </td>
        <td width="520px" align=right>
<?
    if ($this->show_buttons == "true") {
?>
          <div class="btns" align="right">
            <a href="javascript:popUp('/ganglia/','gang')" class="btn" ><span>Ganglia</span></a>
            <a href="javascript:popUp('results.lib.php?single=true', 'res')" class="btn"><span>Results</span></a>
            <a href="javascript:popUp('archival.lib.php?single=true', 'arc')" class="btn"><span>Archival</span></a>
            <a href="javascript:popUp('control.lib.php?single=true', 'ctr')" class="btn"><span>Controls</span></a>
            <a href="javascript:popUp('tcs_simulator.lib.php?single=true', 'test')" class="btn"><span>Test</span></a>
            <a href="javascript:popUp('transient_viewer.lib.php?single=true','tran')" class="btn" ><span>Transients</span></a>
            <a href="javascript:popUp('support.html', 'help')" class="btn"><span>Help</span></a>
          </div>
<?  
    }
?>
        </td>
      </tr>
  </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $state = "TCS Interface Stopped";
    $num_beams = "NA";

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {
      $bytes_written = socketWrite($socket, "state\r\n");
      list ($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $state = $response;
        socket_close($socket);
      }
      $socket = 0;
    }

    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") 
    {
      $bytes_written = socketWrite($socket, "num_beams\r\n");
      list ($result, $response) = socketRead($socket);
      if ($result == "ok")
      {
        $num_beams = $response;
        socket_close($socket);
      }
      $socket = 0;
    }

    $string = $state.":::".$num_beams;

    echo $string;

  }
}
handledirect("state_banner");
