<?PHP

include("caspsr.lib.php");
include("caspsr_webpage.lib.php");
include("functions_i.php");

class baseband extends caspsr_webpage
{
  var $inst = 0;
  var $cfg = array();
  var $dbkeys = array("eada", "fada");

  function baseband()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
    $this->cfg = $this->inst->config;
    $this->title = "CASPSR | Baseband Record Controller";
    $this->callback_freq = 3000; // 5 seconds
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

      function controlDB(key, action)
      {
        var backend_state = document.getElementById("img_caspsr_baseband").src;

        var i = 0;
        var host = "";
        var img = "";
        var src = "";
        var url = "";

        for (i=0; i<hosts.length; i++)
        {
          host = hosts[i];
          img = document.getElementById("img_"+host+"_"+key);
          src = new String(img.src);

          if (src.indexOf("grey_light.png") != -1)
          {
            alert("Ignoring command as "+host+" was disabled");
            action = "ignore";
            return;
          }

          if ((action == "ACTIVATE") && ((src.indexOf("green_light.png") != -1) || (src.indexOf("yellow_light.png") != -1)))
          {
            alert("Ignoring Start as "+host+" was active");
            action = "ignore";
            return;
          }

          if ((action == "DEACTIVATE") && (src.indexOf("red_light.png") != -1))
          {
            alert("Ignoring Stop as "+host+" was inactive");
            action = "ignore";
            return;
          }
        }
        if (action != "ignore")
        {
          // check that the backend state is ok for starting on fada
          if ((action == "ACTIVATE") && (key == "fada") && (backend_state.indexOf("green_light.png",0) == -1))
          {
            alert("Cannot start Recorder if Baseband Recording Daemon is not running");
            return;
          }

          url = "baseband.lib.php?action="+action+"&n_hosts="+hosts.length+"&key="+key;
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
      }

      function toggleDB(host, key)
      {
        var backend_state = document.getElementById("img_caspsr_baseband").src;

        var img = document.getElementById("img_"+host+"_"+key);
        var src = new String(img.src);
        var action = "";
        var url = "";

        if (src.indexOf("green_light.png",0) != -1)
          action = "DEACTIVATE";
        else if (src.indexOf("red_light.png",0) != -1)
          action = "ACTIVATE";
        else if (src.indexOf("yellow_light.png",0) != -1)
          action = "DEACTIVATE";
        else
          action = "ignore";


        if (action != "ignore") 
        {
          // check that the backend state is ok for starting on fada
          if ((action == "ACTIVATE") && (key == "fada") && (backend_state.indexOf("green_light.png",0) == -1))
          {
            alert("Cannot start Recorder if Baseband Recording Daemon is not running");
            return;
          }
          url = "baseband.lib.php?action="+action+"&n_hosts=1&host_0="+host+"&key="+key;
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
                bits = msg.split(" ");
                // 0 == host, 1 == curr_state, 2 == new_state

                td = document.getElementById("state_"+host+"_"+bits[0]);
                img = document.getElementById("img_"+host+"_"+bits[0]);

                if (bits[1] == "active" && bits[2] == "active")
                {
                  td.innerHTML = "Active";
                  img.src = "/images/green_light.png";
                }
                else if (bits[1] == "inactive" && bits[2] == "active")
                {
                  td.innerHTML = "Queued to start";
                  img.src = "/images/yellow_light.png";
                }
                else if (bits[1] == "active" && bits[2] == "inactive")
                {
                  td.innerHTML = "Queued to stop";
                  img.src = "/images/yellow_light.png";
                }
                else if (bits[1] == "inactive" && bits[2] == "inactive")
                {
                  td.innerHTML = "Inactive";
                  img.src = "/images/red_light.png";
                }
                else
                {
                  td.innerHTML = "Unknown ["+bits[1]+", "+bits[2]+"]";
                  img.src = "/images/grey_light.png";
                }
              }
              else
              {
<?
      for ($i=0; $i<count($this->dbkeys); $i++)
      {
        echo "                td = document.getElementById('state_'+host+'_".$this->dbkeys[$i]."');\n";
        echo "                td.innerHTML = 'Not Running';\n";
        echo "                img = document.getElementById('img_'+host+'_".$this->dbkeys[$i]."');\n";
        echo "                img.src = '/images/grey_light.png';\n";
      }
?>
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

    <table width='500px' border=0>
      <tr><td style='vertical-align: middle;'>
        <b>RAID Recording Daemon</b>
        <span style='padding-left: 10px; padding-right: 10px;'><? echo $this->statusLight("caspsr", "baseband", -1, "");?></span>
        <span id='state_caspsr_baseband'></span>
      </td></tr>
    </table>

    <br/>

    <table width='500px'>
      <tr>
        <th>PWC</th>
        <th>Processor [DB eada]</th>
        <th>Recorder [DB fada]</th>
      </tr>
<?
    for ($i=0; $i<$this->cfg["NUM_PWC"]; $i++)
    {
      $h = $this->cfg["PWC_".$i];
      echo "      <tr>\n";
      echo "        <td>".$h."</td>\n"; 
      for ($j=0; $j<count($this->dbkeys); $j++)
      {
        echo "        <td>\n";
        $k = $this->dbkeys[$j];
        //echo "          ".$this->statusLight($h, $k, -1, "toggleDB")."\n";
        echo "          ".$this->statusLight($h, $k, -1, "")."\n";
        echo "          &nbsp;&nbsp;&nbsp;<span id='state_".$h."_".$k."'>N/A</span>\n";
        echo "        </td>\n";
      }
      echo "      </tr>\n"; 
    }
?>
      <tr>
        <td></td>
<?
      for ($j=0; $j<count($this->dbkeys); $j++)
      {
        echo "        <td>\n";
        echo "          <input type='button' value='Start' onClick='controlDB(\"".$this->dbkeys[$j]."\",\"ACTIVATE\")'>\n";
        echo "          <input type='button' value='Stop' onClick='controlDB(\"".$this->dbkeys[$j]."\",\"DEACTIVATE\")'>\n";
        echo "        </td>\n";
      }
?>
      </tr>

      <tr>
        <td colspan=<?echo (1 + count($this->dbkeys))?>>
          <br/>
          Usage:
          <ol>
            <li>Ensure all <b>Persistent Server Daemons</b> on APSR, BPSR &amp; CASPSR are stopped
            <li>Start CASPSR's <b>Baseband Ctrlr</b> Persistent Server Daemon
            <li>Click the <i>Recorder [DB fada]</i>'s Start/Stop buttons to control baseband recording
            <li>Click the <i>Processor [DB eada]</i>'s Start/Stop buttons to control dspsr 
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
        $bytes_written = socketWrite($socket, "STATE\r\n");
        $max = 30;
        while (!$eod && $max > 0)
        {
          $read = socketRead($socket);
          if ($read == "ERROR: socket closed before read")
            $eod = 1;
          if (strlen($read) > 1)
          {
            if ((strpos($read, "ok") !== FALSE) || (strpos($read, "fail") != FALSE))
              $eod = 1;
            else
              $output .= $host.":::".rtrim($read)."\n";
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
    $key = $get["key"];
    $cmd = $action." ".$key;

    for ($i=0; $i<$get["n_hosts"]; $i++)
    {
      $host = $get["host_".$i];
      list ($socket, $result) = openSocket($host, $port);

      $eod = 0;

      if ($result == "ok")
      {
        $bytes_written = socketWrite($socket, $cmd."\r\n");
        $max = 30;
        while (!$eod && $max > 0)
        {
          $read = socketRead($socket);
          if ($read == "ERROR: socket closed before read")
            $eod = 1;
          if (strlen($read) > 1)
          {
            if ((strpos($read, "ok") !== FALSE) || (strpos($read, "fail") != FALSE))
              $eod = 1;
            else
              $output .= $host.":::".$key." ".$read."\n";
          }
          $max--;
        }
        socket_close($socket);
      } else {
        $output .= $host.":::".$key." Could not connect to $host:$port\n";
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

