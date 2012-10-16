<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class raid_archival extends bpsr_webpage
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
    bpsr_webpage::bpsr_webpage();
    $this->inst = new bpsr();
    $this->cfg = $this->inst->config;
    $this->title = "BPSR | RAID Archival Pipeline";

    $this->callback_freq = 4000; // 4 seconds
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
      table.usage {
      }
      table.usage th {
        padding: 0px 5px 0px 0px; 
        text-align: left;
        font-weight: normal;
      }
      table.usage td {
        padding: 0px 0px 0px 5px; 
        text-align: right;
      }

      .list {
        width: 160px;
        font-family: monospace;
        text-align: center;
      }
      
    </style>

    <script type='text/javascript'>

      var du_rates_report = 80;
      var du_rates_count  = 80;

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
            if (parts[0] == "ARCHIVE") 
            {
              div = document.getElementById("archive_list");
            }
            if (parts[0] == "ARCHIVED") 
            {
              div = document.getElementById("archived_list");
            }
            list = "<center><table class='usage'>";
            for (j=1; j<parts.length; j++)
            {
              bits = parts[j].split(" ", 2);
              list += "<tr><th>" + bits[0] + "</th><td> " + bits[1] + "</td></tr>";
            }
            list += "</table></center>";
            div.innerHTML = list;
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
            if (parts[0] == "BPSR") 
            {
              document.getElementById("bpsr_rate").innerHTML = parts[1] + " MB/s";
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
    $this->openBlockHeader("BPSR RAID Archival Pipeline");
?>
    <table width='100%' border='0'>
      <tr>
        <th>From BPSR</th>
        <th>Waiting for<br/>Swin Transfer</th>
        <th>To Swin</th>
        <th>Waiting for<br/>Tape Archival</th>
        <th>Archiving</th>
        <th>On Tape / Archived</th>
      </tr>

      <tr>
        <td width='80px'>
          <div><img src='images/arrow_right.png'></div>
          <div id='bpsr_rate'></div>
        </td>

        <td id='finished_list'>
        </td>

        <td width='80px'>
          <center>
            <div><img src='images/arrow_right.png'></div>
            <div id='swin_rate'></div>
          </center>
        </td>

        <td id='archive_list'>
        </td>

        <td width='80px'>
          <div><img src='images/arrow_right.png'></div>
          <div id='prks_rate'></div>
        </td>

        <td id='archived_list'>
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

        $rx_mb =  sprintf("%5.2f", ($rxbyt/1045876));
        $tx_mb =  sprintf("%5.2f", ($txbyt/1045876));

        // swinburne VLAN
        if ($dev == "eth1")
        {
          $results .= "SWIN:".$rx_mb.":".$tx_mb."\n";
        }

        // PRKS Network
        if ($dev == "eth3") 
        {
          $results .= "PRKS:".$rx_mb.":".$tx_mb."\n";
        }

        // BPSR Network
        if ($dev == "eth3")
        {
          $results .= "BPSR:".$rx_mb.":".$tx_mb."\n";
        }
      }
      echo $results;
    }

    if ($get["type"] == "disk_usage") 
    {
      $lines = array();

      # get a list of finished obseravations
      $cmd = "cd /lfs/raid0/bpsr; du -sb finished/P* swin/send/P* | awk -F/ '{print \$1\" \"\$(NF)}' | awk '{print \$3\" \"\$1}'";
      exec($cmd, $lines);
      $html = "FINISHED";
      $results = array();
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results)) 
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $keys = array_keys($results);
      for ($i=0; $i<count($keys); $i++)
      {
        $html .= ":".$keys[$i]." ".sprintf("%0.2f",($results[$keys[$i]]/1073741824))." GB";
      }
      $html .= "\n";

      # get a list of waiting for parkes archiving
      $lines = array();
      $results = array();
      $cmd = "cd /lfs/raid0/bpsr/parkes/archive; du -sb P* | awk '{print $2\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "ARCHIVE";
      $keys = array_keys($results);
      for ($i=0; $i<count($keys); $i++)
      {
        $html .= ":".$keys[$i]." ".sprintf("%0.2f",($results[$keys[$i]]/1073741824))." GB";
      }
      $html .= "\n";

      # get a list of PIDs that have been archived
      $lines = array();
      $results = array();
      $cmd = "cd /lfs/raid0/bpsr/archived; du -sb P* | awk '{print $2\" \"$1}'";
      exec($cmd, $lines);
      for ($i=0; $i<count($lines); $i++)
      {
        list ($pid, $size) = split(" ", $lines[$i]);
        if (!array_key_exists($pid, $results))
          $results[$pid] = 0;
        $results[$pid] += $size;
      }
      $html .= "ARCHIVED";
      $keys = array_keys($results);
      for ($i=0; $i<count($keys); $i++)
      {
        $html .= ":".$keys[$i]." ".sprintf("%0.2f",($results[$keys[$i]]/1073741824))." GB";
      }
      $html .= "\n";

      echo $html;

    }
  }
}

handleDirect("raid_archival");

