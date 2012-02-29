<?PHP

include("caspsr.lib.php");
include("caspsr_webpage.lib.php");

class raid_archival extends caspsr_webpage
{
  var $inst = 0;
  var $cfg = array();
  var $nobs = 0;

  var $swin_dirs = array();
  var $parkes_dirs = array();
  var $swin_db = array();
  var $parkes_db = array();
  var $results = array();

  function raid_archival()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
    $this->cfg = $this->inst->config;
    $this->title = "CASPSR | RAID Archival Pipeline";

    $this->callback_freq = 10000; // 10 seconds

  }

  function javaScriptCallback()
  {
    return "network_rates_request();";
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      td {
        vertical-align: top;
        text-align: center;
      }
      div.list {
        font-family: monospace;
        text-align: left;
        width: 120px;
      }
      
    </style>

    <script type='text/javascript'>

      var du_rates_report = 20;
      var du_rates_count  = 20;

      function handle_disk_usage_request(du_http_request) 
      {
        if (du_http_request.readyState == 4) {

          var response = String(du_http_request.responseText)
          var lines = response.split("\n");

          var i = 0;
          var j = 0;
          var parts;
          var div;
          var list;

          for (i=0; i<lines.length; i++)
          {
            parts = lines[i].split(":");
            div = 0;
            if (parts[0] == "FINISHED") 
            {
              div = document.getElementById("finished_list");
            }
            if (parts[0] == "UNPATCHED") 
            {
              div = document.getElementById("unpatched_list");
            }
            if (parts[0] == "PATCHED") 
            {
              div = document.getElementById("patched_list");
            }
            if (parts[0] == "SENT") 
            {
              div = document.getElementById("archived_list");
            }

            if (div != 0)
            {
              list = "";
              for (j=1; j<parts.length; j++)
              {
                list += parts[j] + "<br/>";
              }
              div.innerHTML = list;
            }
          }
        }
      }

      function disk_usage_request() 
      {
        var url = "raid_archival.lib.php?update=true&type=disk_usage"
  
        if (window.XMLHttpRequest)
          du_http_request = new XMLHttpRequest();
        else
          du_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        du_http_request.onreadystatechange = function() {
          handle_disk_usage_request(du_http_request)
        };
        du_http_request.open("GET", url, true);
        du_http_request.send(null);
      }


      function handle_network_rates_request(nr_http_request) 
      {
        if (nr_http_request.readyState == 4) {

          var response = String(nr_http_request.responseText)
          var lines = response.split("\n");

          var i = 0;
          var parts;

          for (i=0; i<lines.length; i++)
          {
            parts = lines[i].split(":");
            if (parts[0] == "SWIN") 
            {
              document.getElementById("swin_rate").innerHTML = parts[2] + " MB/s";
            }
            if (parts[0] == "PRKS") 
            {
              document.getElementById("prks_rate").innerHTML = parts[2] + " MB/s";
            }
            if (parts[0] == "CASPSR") 
            {
              document.getElementById("caspsr_rate").innerHTML = parts[1] + " MB/s";
            }
          }
        }
      }

      function network_rates_request() 
      {
        var url = "raid_archival.lib.php?update=true&type=network_rates"
  
        if (window.XMLHttpRequest)
          nr_http_request = new XMLHttpRequest();
        else
          nr_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        nr_http_request.onreadystatechange = function() {
          handle_network_rates_request(nr_http_request)
        };
        nr_http_request.open("GET", url, true);
        nr_http_request.send(null);

        if (du_rates_count >= du_rates_report)
        {
          disk_usage_request();
          du_rates_count = 0;
        }
        else
        {
          du_rates_count++;
        }
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
    $this->openBlockHeader("CASPSR RAID Archival Pipeline");
?>
    <table width='100%' border='0'>
      <tr>
        <th>From CASPSR</th>
        <th>Waiting for<br/>Swin Transfer</th>
        <th>To Swin</th>
        <th>Waiting for<br/>PSRFITS Conversion</th>
        <th>Converting</th>
        <th>Waiting for<br/>ATNF Archival</th>
        <th>Archiving</th>
        <th>Archived</th>
      </tr>

      <tr>
        <td>
          <div><img src='images/arrow_right.png'></div>
          <div id='caspsr_rate'></div>
        </td>

        <td>
          <center>
            <div id='finished_list' class='list'>
            </div> 
          </center>
        </td>

        <td>
          <div><img src='images/arrow_right.png'></div>
          <div id='swin_rate'></div>
        </td>

        <td>
          <center>
            <div id='unpatched_list' class='list'>
            </div>
          </center>
        </td>

        <td>
          <div><img src='images/arrow_right.png'></div>
        </td>
        
        <td>
          <center>
            <div id='patched_list' class='list'>
            </div>
          </center>
        </td>


        <td>
          <div><img src='images/arrow_right.png'></div>
          <div id='prks_rate'></div>
        </td>

        <td>
          <center>
            <div id='archived_list' class='list'>
            </div>
          </center>
        </td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }

  #############################################################################
  #
  # print update information for the raid_archival page
  #
  function printUpdateHTML($get)
  {
    if ($get["type"] == "network_rates") 
    {

      # get information about the network interfaces
      $cmd = "sar -n DEV 1 1 | grep Average";
      $output = array();
      exec($cmd, $output);
      $results = "";

      foreach ($output as $line) 
      {
        list ($junk, $dev, $rxpck, $txpck, $rxbyt, $txbyt, $rxcmp, $txcmp, $rxmcst) = split(" +", $line);

        $rx_mb =  sprintf("%2.2f", ($rxbyt/1045876));
        $tx_mb =  sprintf("%2.2f", ($txbyt/1045876));

        // swinburne VLAN
        if ($dev == "eth1")
        {
          $results .= "SWIN:".$rx_mb.":".$tx_mb."\n";
        }

        // PKS VLAN
        if ($dev == "eth2")
        {
          $results .= "PRKS:".$rx_mb.":".$tx_mb."\n";
        }

        // CASPSR Network
        if ($dev == "eth0")
        {
          $results .= "CASPSR:".$rx_mb.":".$tx_mb."\n";
        }
      }
      echo $results;
    }

    if ($get["type"] == "disk_usage") 
    {
      $html = "";
      $lines = array();
      $results = array();

      # get a list of finished obseravations
      $cmd = "cd /lfs/raid0/caspsr/; du -sb finished/P* | awk -F/ '{print $1\" \"$(NF)}' | awk '{print $3\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "FINISHED";
      foreach ($results as $pid => $size)
      {
        $html .= ":".$pid."&nbsp;&nbsp;".sprintf("%5.1f", ($size/1073741824))." GB";
      }
      $html .= "\n";

      # get a list of obs ready to be patched
      $lines = array();
      $results = array();
      $cmd = "cd /lfs/raid0/caspsr; du -sb swin/sent/P* | awk -F/ '{print $1\" \"$(NF)}' | awk '{print $3\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "UNPATCHED";
      foreach ($results as $pid => $size)
      {
        $html .= ":".$pid."&nbsp;&nbsp;".sprintf("%5.1f", ($size/1073741824))." GB";
      }
      $html .= "\n";

      # get a list of obs waiting for transfer to ATNF
      $lines = array();
      $results = array();
      $cmd = "cd /lfs/raid0/caspsr; du -sb atnf/send/P* | awk -F/ '{print $1\" \"$(NF)}' | awk '{print $3\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++) 
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "PATCHED";
      foreach ($results as $pid => $size)
      {
        $html .= ":".$pid."&nbsp;&nbsp;".sprintf("%5.2f", ($size/1073741824))." GB";
      }
      $html .= "\n";

      # get a list of obs that have been transferred to the ATNF
      $lines = array();
      $results = array();
      $cmd = "cd /lfs/raid0/caspsr; du -sb atnf/sent/P* archived/P* | awk -F/ '{print $1\" \"$(NF)}' | awk '{print $3\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "SENT";
      foreach ($results as $pid => $size)
      {
        $html .= ":".$pid."&nbsp;&nbsp;".sprintf("%5.2f", ($size/1073741824))." GB";
      }
      $html .= "\n";

      echo $html;
    }
  }
}

handleDirect("raid_archival");

