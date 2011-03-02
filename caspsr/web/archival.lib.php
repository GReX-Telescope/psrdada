<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class archival extends caspsr_webpage 
{

  var $inst = "";
  var $nobs = "";

  function archival()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
  }

  function javaScriptCallback()
  {
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      .hidden {
        display: none;
      }
      .shown{
        display: table-row;
      }
    </style>

    <script type='text/javascript'>

      function handle_archival_request(a_http_request) 
      {
        if (a_http_request.readyState == 4) 
        {
          var response = String(a_http_request.responseText)
          var lines = response.split("\n");
          var bits;
          var i = 0;

          var show_deleted = document.getElementById("show_deleted").checked;
          var show_transferred = document.getElementById("show_transferred").checked;

          var num_shown = 0;
          var total_obs = lines.length;

          for (i=0; i<lines.length; i++)
          {
            if (lines[i].length > 3) 
            {
              bits = lines[i].split(" "); 
              document.getElementById("utc_"+i).innerHTML = "<a href='/caspsr/result.lib.php?utc_start="+bits[0]+"'>"+bits[0]+"</a>";
              document.getElementById("pid_"+i).innerHTML = bits[2];
              document.getElementById("source_"+i).innerHTML = bits[3];

              if (bits[1] == "obs.processing")
              {
                document.getElementById("state_"+i).src = "/images/grey_light.png";
                document.getElementById("state_string_"+i).innerHTML = "processing";
                document.getElementById("row_"+i).className = "shown";
              }
              else if (bits[1] == "obs.finished")
              {
                document.getElementById("state_"+i).src = "/images/yellow_light.png";
                document.getElementById("state_string_"+i).innerHTML = "finished";
                document.getElementById("row_"+i).className = "shown";
              }
              else if (bits[1] == "obs.failed")
              {
                document.getElementById("state_"+i).src = "/images/red_light.png";
                document.getElementById("state_string_"+i).innerHTML = "failed";
                document.getElementById("row_"+i).className = "shown";
              }
              else if (bits[1] == "obs.transfer_error")
              {
                document.getElementById("state_"+i).src = "/images/red_light.png";
                document.getElementById("state_string_"+i).innerHTML = "transfer error";
                document.getElementById("row_"+i).className = "shown";
              }
              else if (bits[1] == "obs.transferred")
              {
                document.getElementById("state_"+i).src = "/images/green_light.png";
                document.getElementById("state_string_"+i).innerHTML = "transferred";
                if (show_transferred)
                  document.getElementById("row_"+i).className = "shown";
                else
                  document.getElementById("row_"+i).className = "hidden";
              }
              else if (bits[1] == "obs.deleted")
              {
                document.getElementById("state_"+i).src = "/images/grey_light.png";
                document.getElementById("state_string_"+i).innerHTML = "deleted";
                if (show_deleted)
                  document.getElementById("row_"+i).className = "shown";
                else
                  document.getElementById("row_"+i).className = "hidden";
              }
              else
                document.getElementById("state_"+i).src = "";

              if (document.getElementById("row_"+i).className == "shown")
                num_shown++;
            }


          }

          document.getElementById("summary_text").innerHTML = "Showing " + num_shown + " of " + total_obs;
        }
      }

      function archival_request() 
      {
        var a_http_request;
        var url = "archival.lib.php?update=true";

        document.getElementById("summary_text").innerHTML = "Retrieving Data...";

        if (window.XMLHttpRequest)
          a_http_request = new XMLHttpRequest()
        else
          a_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        a_http_request.onreadystatechange = function() 
        {
          handle_archival_request(a_http_request)
        }

        a_http_request.open("GET", url, true)
        a_http_request.send(null)

      }
    </script>
<?
  }

  function printJavaScriptBody()
  {
?>
<?
  }

  function printSideBarHTML() 
  {
    $this->openBlockHeader("Legend");
?>
    <p>This page shows the list of observations to be archived and
       deleted from CASPSR.</p>
    <table>
      <tr><th colspan=2>Observation States</th></tr>
      <tr><td><img src="/images/grey_light.png"></td><td>Processing</td></tr>
      <tr><td><img src="/images/red_light.png"></td><td>Failed</td></tr>
      <tr><td><img src="/images/yellow_light.png"></td><td>Finished</td></tr>
      <tr><td><img src="/images/green_light.png"></td><td>Transferred</td></tr>
      <tr><td><img src="/images/grey_light.png"></td><td>Deleted</td></tr>
    </table>
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("View Options");
?>
    <table>
      <tr>
        <td><input type="checkbox" name="show_deleted" id="show_deleted" onChange="archival_request()"></td>
        <td>Show deleted?</td>
      </tr>
      <tr>
        <td><input type="checkbox" name="show_transferred" id="show_transferred" onChange="archival_request()" checked></td>
        <td>Show Transferred?</td>
    </table>
<?
    $this->closeBlockHeader();

  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML() 
  {
?>
<html>
<head>
  <title>CASPSR | Archival</title>
<?
    echo "    <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>\n";
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
  
    $this->printJavaScriptHead();
?>
</head>


<body onload='archival_request()'>
<?
  $this->printJavaScriptBody();
?>
  <div class='PageBackgroundSimpleGradient'>
  </div>
  <div class='Main'>
    <div class="contentLayout">
      <div class="sidebar1">
        <div style='text-align: center; vertical-align: middle;'>
          <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
        </div>
<?
        $this->printSideBarHTML();
?>
      </div><!-- sidebar1 -->
      <div class="content">
<?
        $this->printMainHTML();
?>
      </div> <!-- content -->
    </div> <!-- contentLayout -->
  </div> <!-- main -->
</body>
</html>
<?
  }

  ###########################################################################
  #
  # Generate the table, which will be filled in via a XML operation
  # 
  function printMainHTML()
  {

    $cmd = "find ".$this->inst->config["SERVER_RESULTS_DIR"]." -mindepth 1 -maxdepth 1 -type d -name '2*' | wc -l";
    $this->nobs = exec($cmd, $output, $rval);

    $this->openBlockHeader("Observations in Archival Pipeline [<span id='summary_text'>".$this->nobs."</span>]");
?>
    <table width='100%' border=0>
      <tr>
        <th>#</th>
        <th colspan=2>State</th>
        <th>UTC START</th>
        <th>SOURCE</th>
        <th>PID</th>
      </tr>
<?
      for ($i=0; $i<$this->nobs; $i++)
      {
        echo "      <tr id='row_".$i."' class='hidden'>";
          echo "<td>".($i+1)."</td>";
          echo "<td><img src='/images/grey_light.png' id='state_".$i."'></td>";
          echo "<td id='state_string_".$i."'></td>";
          echo "<td id='utc_".$i."'></td>";
          echo "<td id='source_".$i."'></td>";
          echo "<td id='pid_".$i."'></td>";
        echo "</tr>\n";
      }
?>
    </table>
<?
    $this->closeBlockHeader();
  }

  #############################################################################
  #
  # print update information for the archival page
  #
  function printUpdateHTML($get)
  {

    $dir = $this->inst->config["SERVER_RESULTS_DIR"];

    # get a list of the observations in the RESULTS dir
    $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort";
    $observations = array();
    $rval = 0;
    $lastline = exec($cmd, $observations, $rval);

    # get a list of observations and obs.states
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.*' | grep -v obs.info | grep -v obs.start | awk -F/ '{print \$(NF-1)\" \"\$NF}' | sort";
    $tmp_array = array();
    $rval = 0;
    $lastline = exec($cmd, $tmp_array, $rval);

    $states = array();
    $bits = array();
    for ($i=0; $i<count($tmp_array); $i++) 
    {
      $bits = explode(" ",$tmp_array[$i]);
      $states[$bits[0]] = $bits[1];
    }

    # get a list of the PIDS 
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.info' | xargs grep PID | awk -F/ '{print $(NF-1)\" \"\$NF}' | awk '{print \$1\" \"\$3}'";
    $pids = array();
    $tmp_array = array();
    $rval = 0;
    $lastline = exec($cmd, $tmp_array, $rval);
    for ($i=0; $i<count($tmp_array); $i++)
    {
      $bits = explode(" ",$tmp_array[$i]);
      $pids[$bits[0]] = $bits[1];
    }

    # get a list of the SOURCE
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.info' | xargs grep SOURCE | awk -F/ '{print $(NF-1)\" \"\$NF}' | awk '{print \$1\" \"\$3}'";
    $sources = array();
    $tmp_array = array();
    $rval = 0;
    $lastline = exec($cmd, $tmp_array, $rval);
    for ($i=0; $i<count($tmp_array); $i++)
    {
      $bits = explode(" ",$tmp_array[$i]);
      $sources[$bits[0]] = $bits[1];
    }

    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i];
      echo $o." ".$states[$o]." ".$pids[$o]." ".$sources[$o]."\n";
    }
  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }
  }

}
$obj = new archival();
$obj->handleRequest($_GET);
