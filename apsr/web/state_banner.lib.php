<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class state_banner extends apsr_webpage 
{

  var $show_state;
  var $show_buttons;

  function state_banner()
  {
    apsr_webpage::apsr_webpage();

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
    $inst = new apsr();
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
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width=1034,height=778');");
      }

      // handle the response from an state_banner request
      function handle_state_banner_request( sb_http_request) 
      {
        if ( sb_http_request.readyState == 4) {
          var response = String(sb_http_request.responseText)

          if (response.indexOf("Could not connect to") == -1) 
          {
            var lines = response.split(":::");
            document.getElementById("apsr_state").innerHTML = "State: "+lines[0];
            var state = lines[0];
            var state = state.replace("TCS Interface Stopped", "Offline");
            document.title = "APSR "+ state;

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
        <td width="380px" height="60px"><img src="/apsr/images/apsr_logo.png" width="380px" height="60px"></td>
        <td height="60px" align=left style='padding-left: 20px'>
<?
    if ($this->show_state == "true") {
?>
          <span id="apsr_state" class="largetext"></span><br>
<?
    } 
?>
        </td>
<?
    if ($this->show_buttons == "true") {
?>
        <td width="350px" align=right>
          <div class="btns" align="right">
            <a href="javascript:popUp('/ganglia/','full')" class="btn" ><span>Ganglia</span></a>
            <a href="javascript:popUp('results.lib.php?single=true', 'full')" class="btn"><span>Results</span></a>
            <!--<a href="javascript:popUp('archival.lib.php?single=true', 'full')" class="btn"><span>Archival</span></a>-->
            <a href="javascript:popUp('control.lib.php?single=true', 'minimum')" class="btn"><span>Controls</span></a>
            <a href="javascript:popUp('tcs_simulator.lib.php?single=true', 'full')" class="btn"><span>Test</span></a>
            <a href="javascript:popUp('support.lib.php?single=true', 'full')" class="btn"><span>Help</span></a>
            <!--<a href="javascript:popUp('config_edit.lib.php?single=true&file=site.cfg', 'full')" class="btn"><span>Site Config</span></a>-->
          </div>
        </td>
<?  
    }
?>
      </tr>
  </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $state = "TCS Interface Stopped";
    $num_pwcs = "NA";

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {
      $bytes_written = socketWrite($socket, "state\r\n");
      $state = socketRead($socket);
      socket_close($socket);
    }

    $string = $state.":::".$pwcs;

    echo $string;

  }
}
handledirect("state_banner");
