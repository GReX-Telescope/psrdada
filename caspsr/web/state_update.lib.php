<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class state_update extends caspsr_webpage 
{

  var $machines = array();
  var $pwcs = array();
  var $srvs = array();
  var $config = array();

  function state_update()
  {
    caspsr_webpage::caspsr_webpage();
    array_push($this->css, "/caspsr/buttons.css");

    $inst = new caspsr();
    $this->config = $inst->config;

    /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) {
      array_push($this->pwcs, $this->config["PWC_".$i]);
      array_push($this->machines, $this->config["PWC_".$i]);
    }

    array_push($this->machines, "srv0");
    array_push($this->srvs, "srv0");

  }

  function javaScriptCallback()
  {
    return "state_update_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function popUp(URL, type) {

        var to = "toolbar=1";
        var sc = "scrollbars=1";
        var l  = "location=1";
        var st = "statusbar=1";
        var mb = "menubar=1";
        var re = "resizeable=1";
        var w = 1024;
        var h = 768;

        if (type == "results")
          w = 1300;

        options = to+","+sc+","+l+","+st+","+mb+","+re
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width="+w+",height="+h+"');");
      }

      function handle_state_update_request(ms_http_request) 
      {
        if (su_http_request.readyState == 4) {
          var response = String(su_http_request.responseText)
          document.getElementById("globalstatus").innerHTML = response
        }
      }

      function state_update_request() 
      {
        var su_http_requset;
        var host = "<?echo $this->config["SERVER_HOST"];?>";
        var port = "<?echo $this->config["TCS_STATE_INFO_PORT"];?>";
        var url = "state_update.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          su_http_request = new XMLHttpRequest()
        else
          su_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        su_http_request.onreadystatechange = function() 
        {
          handle_state_update_request(su_http_request)
        }

        su_http_request.open("GET", url, true)
        su_http_request.send(null)

      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlock();
?>
    <table cellspacing=0 cellpadding=0 border=0 width="100%">
      <tr>
        <td style='vertical-align: middle;'>&nbsp;&nbsp;<span id="globalstatus"></span></td>
        <td width=550px align=right>
          <div class="btns" align="right">
            <a href="javascript:popUp('/ganglia/','gang')" class="btn" ><span>Ganglia</span></a>
            <a href="javascript:popUp('results.lib.php', 'results')" class="btn"><span>Results</span></a>
            <a href="javascript:popUp('download.lib.php', 'downloads')" class="btn"><span>Downloads</span></a>
            <a href="javascript:popUp('archival.lib.php', 'archival')" class="btn"><span>Archival</span></a>
            <a href="javascript:popUp('control.lib.php', 'controls')" class="btn"><span>Controls</span></a>
            <a href="javascript:popUp('tcs_simulator.lib.php', 'test')" class="btn"><span>Test</span></a>
            <a href="javascript:popUp('support.lib.php?single=true', 'help')" class="btn"><span>Help</span></a>
          </div>
        </td>
      </tr>
    </table>
<?
    $this->closeBlock();
  }

  function printUpdateHTML($host, $port)
  {
    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok") {
      $bytes_written = socketWrite($socket, "state\r\n");
      $string = socketRead($socket);
      socket_close($socket);
    } else {
      $string = "TCS Interface Stopped\n";
    }

    echo $string;
    flush();

  }
}

handledirect("state_update");

