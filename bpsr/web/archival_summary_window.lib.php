<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

class archival_summary extends bpsr_webpage 
{

  function archival_summary()
  {
    bpsr_webpage::bpsr_webpage();
  }

  function javaScriptCallback()
  {
    return "archival_summary_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new bpsr();
?>

    <style type="text/css">
      table.archival_summary {
        font-size: 8pt;
        width: 100%;
      }

      table.archival_summary th {
        font-size: 8pt;
      }

      table.archival_summary td {
        font-size: 8pt;
      }
    
      table.archival_summary span {
        font-size: 8pt;
      }

    </style>

    <script type="text/javascript" src="/js/soundmanager2.js"></script>

    <script type='text/javascript'>  

      soundManager.url = '/sounds/sm2-swf-movies/'; // directory where SM2 .SWFs live
      soundManager.debugMode = false;
      soundManager.waitForWindowLoad = true;

      soundManager.onload = function() {
        soundManager.createSound('changetape','/sounds/please_change_the_tape.mp3');
      }

      // handle the response from an archival_summary request
      function handle_archival_summary_request( as_http_request) 
      {
        if ( as_http_request.readyState == 4) {
          var response = String(as_http_request.responseText)

          if (response.indexOf("Could not connect to") == -1) 
          {

            var lines = response.split(";;;")
            var span
            var td
            var values
            
            for (i=0; i<lines.length; i++) {

              if (lines[i].length > 0) {
                values = lines[i].split(":::");
        
                if ((values[0]) && (document.getElementById(values[0]))) {  
                  span = document.getElementById(values[0])
                  td = document.getElementById(values[0]+"_td")

                  span.innerHTML = values[1]

                  if (values[1] == "Insert Tape") {
                    var html = "<span>Load "+values[2]+"</span>\n";
                    span.innerHTML = html
                    td.style.backgroundColor = "orange"
                    soundManager.play('changetape');
                  } else {

                    if (values[1].substring(0,5) == "Error") {
                      td.style.backgroundColor = "red"
                    } else {
                      td.style.backgroundColor = ""
                    }
                  }
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
        var url = "archival_summary_window.lib.php?update=true&host="+host+"&port="+port;

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
  function printHTML() 
  {
?>
  <table cellpadding=0 cellspacing=0 border=0 width=100%>
    <tr>
      <td align="center" valign="top">
        <table class='archival_summary'>
          <tr> 
            <td colspan=3 id="XFER_PID_td">
              <b>Transfer Manager</b>&nbsp;&nbsp;&nbsp;
              <span id="XFER_PID"></span>
            </td>
          </tr>
          <tr> 
            <td colspan=3 id="XFER_td" align="center"><span id ="XFER"></span></td> 
          </tr>
          <tr>
            <td id="XFER_FINISHED_td" align="center">Local disks: <span id ="XFER_FINISHED"></span></td>
            <td id="XFER_ON_RAID_td" align="center">RAID disk: <span id ="XFER_ON_RAID"></span></td>
            <td></td>
          </tr>
        </table>
      </td>
      <td align="center" valign="top">
        <table class='archival_summary'>
          <tr>
            <td colspan=2 id="SWIN_TAPE_td" align="center">
              <b>Swin Tape</b>&nbsp;&nbsp;&nbsp;
              <span id ="SWIN_TAPE"></span>
            </td>
            <td id="SWIN_PID_td"><span id="SWIN_PID"></span></td> 
          </tr>
          <tr> 
            <td colspan=3 id="SWIN_STATE_td" align="center" width=100%><span id ="SWIN_STATE"></span></td>
          </tr>
          <tr>
            <td id="SWIN_NUM_td" align="center">Queued: <span id ="SWIN_NUM"></span></td>
            <td id="SWIN_PERCENT_td" align="center"><span id ="SWIN_PERCENT"></span>&#37; full</td>
            <td id="SWIN_TIME_LEFT_td" align="center"><span id ="SWIN_TIME_LEFT"></span> m</td>
          </tr>
        </table>
      </td>
      <td align="center" valign="top">
        <table class='archival_summary'>
          <tr>
            <td colspan=2 id="PARKES_TAPE_td" align="center">
              <b>Parkes Tape</b>&nbsp;&nbsp;&nbsp;
              <span id ="PARKES_TAPE"></span>
            </td>
            <td id="PARKES_PID_td"><span id="PARKES_PID"></span></td>
          </tr>
          <tr> 
            <td colspan=3 id="PARKES_STATE_td" align="center" width=100%><span id ="PARKES_STATE"></span></td>
          </tr>
          <tr>
            <td id="PARKES_NUM_td" align="center">Queued: <span id ="PARKES_NUM"></span></td>
            <td id="PARKES_PERCENT_td" align="center"><span id ="PARKES_PERCENT"></span>&#37; full</td>
            <td id="PARKES_TIME_LEFT_td" align="center"><span id ="PARKES_TIME_LEFT"></span> m</td>
          </tr>
        </table>
      </td>
    </tr>
  </table>

<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "tape_info\r\n");
      $string = socketRead($socket);
      socket_close($socket);

    } else {
      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;
  }

}
handledirect("archival_summary");
