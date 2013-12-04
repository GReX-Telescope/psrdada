<?PHP

include_once("caspsr.lib.php");
include_once("caspsr_webpage.lib.php");
include_once("functions_i.php");

class baseband extends caspsr_webpage
{
  var $inst = 0;
  var $cfg = array();

  function baseband()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
    $this->cfg = $this->inst->config;
    $this->title = "CASPSR | Baseband Record Controller";
    $this->callback_freq = 2000; // 10 seconds
  }

  function javaScriptCallback()
  {
    return "baseband_request();";
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      th {
        text-align: center;
      }
      td {
        vertical-align: top;
        text-align: left;
      }
    </style>

    <script type='text/javascript'>
      
<?
      echo "      var hosts = Array(";
      for ($i=0; $i<$this->cfg["NUM_PWC"]; $i++)
      {
        $h = $this->cfg["PWC_".$i];
        if ($i == 0)
          echo "\"".$h."\"";
        else
          echo ",\"".$h."\"";
      }
      echo ");\n";
?>

      function handle_baseband_action_request(http_request)
      {
        if ((http_request.readyState == 4) || (http_request.readyState == 3)) {

          var response = String(http_request.responseText);
          var lines = response.split("\n");

          var area = "";
          var area_container;
          var parts;
          var host;
          var msg;
          var bits;
          var key;
          var rest;
          var i = 0;

          for (i=0; i<lines.length; i++)
          {
            if (lines[i].length > 2)
            {
              parts = lines[i].split(":::");
              host = parts[0];
              msg  = parts[1];
              if (msg.indexOf("Could not connect to") == -1) 
              {
                bits = msg.split(" ");
                key = bits[0];
                rest = bits[1];
                for (j=2; j<bits.length; j++)
                {
                  rest += " "+bits[j]; 
                }

                area = "state_"+host+"_"+key;
                area_container = document.getElementById(area);
                area_container.innerHTML = rest;
              }
            }
          }
        }
      }

      function baseband_action_request (stream, action)
      {
        var backend_state = document.getElementById("img_caspsr_baseband").src;

        if (backend_state.indexOf("green_light.png",0) == -1)
        {
          alert ("Cannot change baseband recorded state if RAID Recording Deamon is inactive");
          return;
        }

        if (stream.indexOf("auxiliary",0) == -1)
        {
          alert ("Can only change the state of the BASEBAND data stream");
          return;
        }

        var url = "baseband.lib.php?action="+action+"&n_hosts="+hosts.length;
        for (i=0; i<hosts.length; i++)
        {
          url += "&host_"+i+"="+hosts[i];
        }

        var ba_http_request;
        if (window.XMLHttpRequest)
          ba_http_request = new XMLHttpRequest()
        else
          ba_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        ba_http_request.onreadystatechange = function() 
        {
          handle_baseband_action_request(ba_http_request)
        }
        ba_http_request.open("GET", url, true)
        ba_http_request.send(null)
      }

      function handle_baseband_request(b_http_request) 
      {
        if (b_http_request.readyState == 4) {

          var response = String(b_http_request.responseText)
          var lines = response.split("\n");

          var i = 0;
          var j = 0;
          var parts;
          var host;
          var msg;
          var bits;
          var td;
          var img;

          for (i=0; i<lines.length; i++)
          {
            if (lines[i].length > 2)
            {
              //alert(lines[i]);
              parts = lines[i].split(":::");
              host = parts[0];
              msg = parts[1];

              // if we were able to connect to dbdecidb
              if (msg.indexOf("Could not connect to") == -1) 
              {
                // 0 == primary/auxiliary, 1 == state
                bits = msg.split(" ");
                td = document.getElementById("state_"+host+"_"+bits[0]);
                img = document.getElementById("img_"+host+"_"+bits[0]);

                state = bits[1];
                command = bits[2];
              
                if ((state != "inactive") && (command != "disable"))
                {
                  td.innerHTML = "Active "+state;
                  img.src = "/images/green_light.png";
                }
                else if ((state != "inactive") && (command == "disable"))
                {
                  td.innerHTML = state+", queued to stop";
                  img.src = "/images/yellow_light.png";
                }
                else if ((state == "inactive") && (command != "disable"))
                {
                  td.innerHTML = "queued to start " + command;
                  img.src = "/images/yellow_light.png";
                }
                else if ((state == "inactive") && (command == "disable"))
                {
                  td.innerHTML = "Inactive";
                  img.src = "/images/red_light.png";
                }
                else
                {
                  td.innerHTML = "Unknown [" + state + ", " + command + "]";
                  img.src = "/images/grey_light.png";
                }
              }
              else
              {
                td = document.getElementById('state_'+host+'_primary');
                td.innerHTML = 'Not Running';
                td = document.getElementById('state_'+host+'_auxiliary');
                td.innerHTML = 'Not Running';
              }
            }
          }
        }
      }

      function baseband_request() 
      {
        var url = "baseband.lib.php?update=true"
  
        if (window.XMLHttpRequest)
          b_http_request = new XMLHttpRequest();
        else
          b_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        b_http_request.onreadystatechange = function() {
          handle_baseband_request(b_http_request)
        };
        b_http_request.open("GET", url, true);
        b_http_request.send(null);
      }

    </script>
<?
  }

  ###########################################################################
  #
  # Generate the table, which will be filled in via a XML operation
  # 
  function printHTML()
  {
    $this->openBlockHeader("CASPSR Baseband Recording");
?>
    <center>

    <p>Remeber to fully turn off BPSR backed on HIPSR!!!</p>

    <table width='500px' border=0>
      <tr><td style='vertical-align: middle;'>
        <b>RAID Recording Daemon</b>
        <span style='padding-left: 10px; padding-right: 10px;'><? echo $this->statusLight("caspsr", "baseband", -1, "");?></span>
        <span id='state_caspsr_baseband'></span>
      </td></tr>
    </table>

    <br/>

    <table width='700px'border=0 cellpadding=5>
      <tr>
        <th>PWC</th>
        <th>DSPSR</th>
        <th>BASEBAND [full or events]</th>
      </tr>
<?
    for ($i=0; $i<$this->cfg["NUM_PWC"]; $i++)
    {
      $h = $this->cfg["PWC_".$i];
      echo "      <tr>\n";
      echo "        <td>".$h."</td>\n";

      echo "        <td>".$this->statusLight($h, "primary", -1, "").
           "&nbsp;&nbsp;<span id='state_".$h."_primary'>N/A</span></td>\n";
      
      echo "        <td>".$this->statusLight($h, "auxiliary", -1, "").
           "&nbsp;&nbsp;<span id='state_".$h."_auxiliary'>N/A</span></td>\n";

      echo  "     </tr>\n";
    }
?>
      <tr>
        <td colspan=3 align=center>
          <input type='button' value='Start Full Baseband' onClick='baseband_action_request("auxiliary", "baseband")'>
          <input type='button' value='Start Baseband Events' onClick='baseband_action_request("auxiliary", "events")'>
          <input type='button' value='Stop Baseband' onClick='baseband_action_request("auxiliary", "disable")'>
        </td>
      </tr>

      <tr>
        <td colspan=3>
          <br/>
          Usage:
          <ol>
            <li>Ensure all <b>Persistent Server Daemons</b> on APSR, BPSR &amp; CASPSR are stopped
            <li>Start CASPSR's <b>Baseband Ctrlr</b> Persistent Server Daemon
            <li>Click the relevant Start button to begin
          </ol>
        </td>
      </tr>

    </table>

    </center>
<?
    $this->closeBlockHeader();
  }

  #############################################################################
  #
  # print update information for the baseband page
  #
  function printUpdateHTML($get)
  {
    # check whether the caspsr_baseband_controller is running
    $running = 0;
    $pid_file = $this->cfg["SERVER_CONTROL_DIR"]."/caspsr_baseband_controller.pid";
    if (file_exists($pid_file))
      $running++;
    $cmd = "pgrep -u dada -f '^perl.*server_caspsr_baseband_controller.pl'"; 
    $output = array();
    $lastline = exec($cmd, $output, $rval);
    if ($rval == 0)
      $running++;
    if ($running == 2)
      $output = "caspsr:::baseband active active\n";
    else
      $output = "caspsr:::baseband inactive inactive\n";

    $port = $this->cfg["CLIENT_DECIDB_PORT"];
    for ($i=0; $i<$this->cfg["NUM_PWC"]; $i++)
    {
      $host = $this->cfg["PWC_".$i];
      list ($socket, $result) = openSocket($host, $port);

      $eod = 0;

      if ($result == "ok") 
      {
        $bytes_written = socketWrite($socket, "state\r\n");
        $max = 30;
        while (!$eod && $max > 0)
        {
          list ($result, $response) = socketRead($socket);
          if ($response == "ERROR: socket closed before read")
            $eod = 1;
          if (strlen($response) > 1)
          {
            if ((strpos($response, "ok") !== FALSE) || (strpos($response, "fail") != FALSE))
              $eod = 1;
            else
              $output .= $host.":::".$response."\n";
          }
          $max--;
        }
        socket_close($socket);
      } else {
        $output .= $host.":::Could not connect to $host:$port\n";
      }
    }
    echo $output;
  }

  #
  # start/stop baseband recording on the specified machines
  #
  function printActionHTML($get)
  {
    $port = $this->cfg["CLIENT_DECIDB_PORT"];
    $action = $get["action"];

    for ($i=0; $i<$get["n_hosts"]; $i++)
    {
      $host = $get["host_".$i];
      list ($socket, $result) = openSocket($host, $port);
      $eod = 0;

      if ($result == "ok")
      {
        $bytes_written = socketWrite($socket, $action."\r\n");
        $max = 30;
        while (!$eod && $max > 0)
        {
          list ($result, $response) = socketRead($socket);
          if ($response == "ERROR: socket closed before read")
            $eod = 1;
          if (strlen($response) > 1)
          {
            if ((strpos($response, "ok") !== FALSE) || (strpos($response, "fail") != FALSE))
              $eod = 1;
            else
              $output .= $host.":::".$response."\n";
          }
          $max--;
        }
        socket_close($socket);
      } else {
        $output .= $host.":::".$action." Could not connect to $host:$port\n";
      }
    }
    echo $output;
  }

  #
  # prints a status light with link, id and initially set to value
  #
  function statusLight($host, $tag, $value, $jsfunc)
  {
    $id = $host."_".$tag;
    $img_id = "img_".$id;
    $link_id = "link_".$id;
    $colour = "grey";
    if ($value == 0) $colour = "red";
    if ($value == 1) $colour = "yellow";
    if ($value == 2) $colour = "green";

    $img = "<img id='".$img_id."' src='/images/".$colour."_light.png' width='15px' height='15px'>";
    $link = "<a href='javascript:".$jsfunc."(\"".$host."\",\"".$tag."\")'>".$img."</a>";

    if ($jsfunc != "")
      return $link;
    else
      return $img;
  }

}

handleDirect("baseband");

