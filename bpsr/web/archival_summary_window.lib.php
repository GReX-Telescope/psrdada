<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class archival_summary extends bpsr_webpage 
{

  function archival_summary()
  {
    bpsr_webpage::bpsr_webpage();
    $this->callback_freq = 60000;
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
        vertical-align: top;
      }
    
      table.archival_summary span {
        font-size: 8pt;
      }

    </style>

    <script type='text/javascript'>  

      // handle the response from an archival_summary request
      function handle_archival_summary_request( as_xml_request) 
      {
        if ( as_xml_request.readyState == 4)
        {
          var xmlDoc = as_xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;
  
            var tags = xmlObj.getElementsByTagName("tape_info");

            var i, key, val;
            for (i=0; i<tags[0].childNodes.length; i++)
            {
              tag = tags[0].childNodes[i];
              if (tag.nodeType == 1)
              {
                key = tag.nodeName;
                val = tag.childNodes[0].nodeValue;
                newval = val.replace(/GB/g, "GB<BR/>");
                document.getElementById(key).innerHTML = newval;
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
          as_xml_request = new XMLHttpRequest();
        else
          as_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        as_xml_request.onreadystatechange = function() {
          handle_archival_summary_request( as_xml_request)
        };
        as_xml_request.open("GET", url, true);
        as_xml_request.send(null);
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
        <table class='archival_summary' border=0>
          <tr>
            <td><b>On HIPSR</b></td>
            <td><b>Uploaded</b></td>
            <td><b>To Swin</b></td>
            <td><b>To ATNF</b></td>
          </tr>
          <tr>
            <td align="center"><span id ="xfer_finished"></span></td>
            <td align="center"><span id ="xfer_uploaded"></span></td>
            <td align="center"><span id ="xfer_to_swin"></span></td>
            <td align="center"><span id ="xfer_to_atnf"></span></td>
          </tr>
        </table>
      </td>
      <td align="center" valign="top">
        <table class='archival_summary'>
          <tr>
            <td colspan=2 id="SWIN_TAPE_td" align="center">
              <b>Swin Tape</b>&nbsp;&nbsp;&nbsp;
              <span id ="swin_tape"></span>
            </td>
            <td id="SWIN_PID_td"><span id="swin_pid"></span></td> 
          </tr>
          <tr> 
            <td colspan=3 id="SWIN_STATE_td" align="center" width=100%><span id ="swin_state"></span></td>
          </tr>
          <tr>
            <td id="SWIN_NUM_td" align="center">Queued: <span id ="swin_num"></span></td>
            <td id="SWIN_PERCENT_td" align="center"><span id ="swin_percent"></span>&#37; full</td>
            <td id="SWIN_TIME_LEFT_td" align="center"><span id ="swin_time_left"></span> m</td>
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
    $response = "";
    $data = "";

    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok")
    {
      $bytes_written = socketWrite($socket, "tape_info\r\n");
      $max = 100;
      $response = "initial";
      while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
      {
        list ($result, $response) = socketRead($socket);
        if ($result == "ok")
        {
          $data .= $response;
        }
        $max--;
      }
      if (($result == "ok") && ($socket))
        socket_close($socket);
      $socket = 0;

    } else {
      $response = "Could not connect to $host:$port<BR>\n";
    }

    header('Content-type: text/xml');
    echo "<?xml version='1.0' encoding='ISO-8859-1'?>";
    echo "<archival_summary_update_request>";
    echo $data;
    echo "</archival_summary_update_request>";
  }

}
handledirect("archival_summary");
