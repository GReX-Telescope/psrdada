<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class plot_window extends caspsr_webpage 
{

  function plot_window()
  {
    caspsr_webpage::caspsr_webpage();
  }

  function javaScriptCallback()
  {
    return "plot_window_request();";
  }

  function printJavaScriptHead()
  {

    $inst = new caspsr();

?>
    <script type='text/javascript'>  

      var npsrs = 0;
      var utc_start = "";
      var psrs = new Array();

      function popWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1024,height=700');");
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function handle_plot_window_request(pw_http_request) 
      {
        if (pw_http_request.readyState == 4) {
          var response = String(pw_http_request.responseText)
          var lines = response.split("\n");

          var img
          var link
          var pos = 0
          var i = 0
          var j = 0
          var base = 0;
          var psr

          /* the current select HTML obj */
          var label_psrs = document.getElementById("single_psr");
          var psrs_select = document.getElementById("psrs");
          var utc_start_span = document.getElementById("utc_start");
          var selected_psr = psrs_select.selectedIndex;
          var custom_plots = document.getElementById("custom_plot");

          if ((response.indexOf("Could not connect to") == -1) &&
              (lines.length >= 9) ) {
        
            var rebuild_select = 0;
            psrs = new Array();

            /* parse the utc_start */
            values = lines[0].split(":::");
            if (values[1] != utc_start) {
              rebuild_select = 1;
            }
            utc_start = values[1];
            custom_plots.onclick = new Function("popWindow('custom_plot.lib.php?basedir=<?echo $inst->config["SERVER_RESULTS_DIR"]?>&utc_start="+utc_start+"')");

            /* parse the number of pulsars */ 
            values = lines[1].split(":::");
            if (npsrs != values[1]) {
              rebuild_select = 1;
            }
            npsrs = values[1];

            /* parse the data from each PSR, updating the global array */  
            for (i=0; i<npsrs; i++) {
              base = 2 + (i*11);
              values = lines[base].split(":::");
              psrs[i] = values[1]; 
              if ((!(psrs_select.options[i])) || (psrs[i] != psrs_select.options[i].value)) {
                rebuild_select = 1;
              }
            }

            if (rebuild_select == 1) { 

              //alert("rebuilding select");

              /* destroy and rebuild the select element */
              if (npsrs > 1) {
                psrs_select.options.length = 0;
                for (i=0; i < npsrs; i++) {
                  psrs_select.options[i] = new Option(psrs[i],psrs[i]);
                }
                psrs_select.selectedIndex = 0;
                psrs_select.style.display = "inline";
                label_psrs.style.display = "none";

              /* hide the select element, and just display a simple text field */
              } else {
                psrs_select.style.display = "none";
                label_psrs.style.display = "inline";
                if (psrs.length > 1) {
                  label_psrs.innerHTML = psrs[0];
                }
              }
            } else {
              //alert("not rebuilding select");
            }
            /* extract the images for the currently selected pulsar */
            if (npsrs == 1) {
              selected_psr = 0
            } else {
              selected_psr = psrs_select.selectedIndex;
            }

            /* line the lines array for the selected PSR */
            base = (selected_psr*11) + 2;

            for (i=1; i<=12; i++) {

              values = lines[base+i].split(":::");
              parts = values[0].split("_");

              img = document.getElementById(parts[0]);
              link = document.getElementById(parts[0]+"_a");

              if ((parts[1] == "240x180") || (parts[1] == "200x150") || (parts[1] == "200x75")) {
                if (img.src != values[1]) {
                  img.src = values[1]
                }

              // Hi res image for a "click"
              } else {

                if (values[1].indexOf("blankimage") > 0) {
                  link.href="javascript:void(0)";
                  img.border=0
                } else {
                  link.href="javascript:popImage('"+values[1]+"')";
                  img.border=2
                }
              }
            } 
          }
        }
      }

      function plot_window_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "plot_window.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          pw_http_request = new XMLHttpRequest();
        else
          pw_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        pw_http_request.onreadystatechange = function() {
          handle_plot_window_request(pw_http_request)
        };
        pw_http_request.open("GET", url, true);
        pw_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlock();
?>
    <table border=0 width="100%" cellspacing=0 cellpadding=0>
    <tr>
      <td align="center" style='padding-top: 5px;'>
        <a id="pvfl_a" href="about:blank"><img id="pvfl" src="/images/blankimage.gif" alt="No Data Available" width=201px height=151px></a>
      </td>
    </tr>
    <tr height=2px><td></td></tr>
    <tr>
      <td align="center" style='padding-top: 5px;'>
        <a id="pvt_a" href="about:blank"><img id="pvt" src="/images/blankimage.gif"alt="No Data Available" width=201px height=151px></a>
      </td>
    </tr>
    <tr height=2px><td></td></tr>
    <tr>
      <td align="center" style='padding-top: 5px;'>
        <a id="pvfr_a" href="about:blank"><img id="pvfr" src="/images/blankimage.gif" alt="No Data Available" width=201px height=151px></a>
      </td>
    </tr>
    <tr height=2px><td></td></tr>
    <tr>
      <td align="center" style='padding-top: 5px;'>
        <a id="bp_a" href="about:blank"><img id="bp" src="/images/blankimage.gif" alt="No Data Available" width=201px height=151px></a>
      </td>
    </tr>
    <tr height=2px><td></td></tr>
    <tr>
      <td align="center" style='padding-top: 5px; padding-bottom: 5px;'>
        <a id="snrt_a" href="about:blank"><img id="snrt" src="/images/blankimage.gif" alt="No Data Available" width=200px height=75px></a>
      </td>
    </tr>
    <tr height=2px><td></td></tr>
    <tr>
      <td align="center" style='padding-top: 5px; padding-bottom: 5px;'>
        <a id="snrh_a" href="about:blank"><img id="snrh" src="/images/blankimage.gif" alt="No Data Available" width=200px height=75px></a>
      </td>
    </tr>


    </table>
    <center>
    <span id="single_psr"></span>
    <select id="psrs" onchange="request()"></select>&nbsp;&nbsp;&nbsp;
    <span><input id="custom_plot" type=button value="View Custom Plots" onClick="popWindow('custom_plot.lib.php')"></span>
    </center>
<?
    $this->closeBlock();
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    $timeout = 1;

    list ($socket, $result) = openSocket($host, $port, $timeout);

    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "img_info\r\n");
      list ($result, $response) = socketRead($socket);
      socket_close($socket);

      # Add the require URL links to the image
      $lines = split(";;;", $response);
      $string = "";

      for ($i=0; $i<count($lines)-1; $i++) {
        $p = split(":::", $lines[$i]);
        if (($p[0] == "utc_start") || ($p[0] == "npsrs") || ( substr($p[0],0,3) == "psr")) {
          $string .=  $p[0].":::".$p[1]."\n";
        } else {
          $string .= $p[0].":::".$url."/caspsr/results/".$p[1]."\n";;
        }
      }

    } else {

      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;
  }

}

handledirect("plot_window");
