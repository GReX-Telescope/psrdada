<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class download extends caspsr_webpage 
{

  var $inst = "";
  var $groups = array();

  function download()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
    $this->groups = getProjects("caspsr");
    sort($this->groups);
  }

  function printJavaScriptHead()
  {
?>
    <style>
      input.check {
        margin: 0px;
        padding: 0px;
      }
    </style>
    <script type='text/javascript'>

      function checkAll(field)
      {
        for (i = 0; i < field.length; i++)
          field[i].checked = true ;
      }

      function uncheckAll(field)
      {
        for (i = 0; i < field.length; i++)
          field[i].checked = false ;
      }

      function downloadSelected(field)
      {
        utcs = new Array();
        for (i = 0; i < field.length; i++) 
        {
          if (field[i].checked == true)  
          {
            utcs.push(field[i].id);
          }
        }

        if (utcs.length > 0) 
        {
          url = "download.lib.php?download=multi&utc_starts="+utcs[0];
          for (i = 1; i < utcs.length; i++)
            url += ","+utcs[i];
          document.location = url;
        }
      }

      function handle_display_request(di_http_request) 
      {
        if (di_http_request.readyState == 4)
        {
          var response = String(di_http_request.responseText)
          var lines = response.split("\n");
          var html = "";
          var bits;
          var i = 0;

          if ((lines.length == 1) && (lines[0].indexOf("ERROR") == 0))
          {
            document.getElementById("observation_list").innerHTML = lines[0];
          }
          else
          {

            document.getElementById("observation_list").innerHTML = "";

            html = "<table width='100%' border='0'>\n";
            html += "<tbody>";
            html += "<tr>";
            html += "<th></th>";
            html += "<th>UTC_START</th>";
            html += "<th>SOURCE</th>";
            html += "<th width=60px>Size [MB]</th>";
            html += "<th>State</th>";
            html += "<th>Download Link</th>";
            html += "</tr>\n";

            for (i=0; i<lines.length; i++)
            {
              if (lines[i].length > 3) 
              {
                bits = lines[i].split(" ");

                state = bits[1].substr(4);
                html += "<tr>";
                if ((state == "finished") || (state == "transferred"))
                  html += "<td><input class='check' type='checkbox' name='list' value='"+i+"' id='"+bits[0]+"'>";
                else
                  html += "<td></td>";

                html += "<td><a href='/caspsr/result.lib.php?utc_start="+bits[0]+"'>"+bits[0]+"</td>";
                html += "<td>"+bits[3]+"</td>";
                html += "<td align=right style='padding-right:40px;'>"+bits[4]+"</td>";
                html += "<td>"+state+"</td>";

                if ((state == "finished") || (state == "transferred"))
                  html += "<td><a href='/caspsr/download.lib.php?download=true&utc_start="+bits[0]+"'>Download</a></td>";
                else
                  html += "<td>--</td>";

                html += "</tr>\n";
              }
            }
            html += "</tbody></table>\n";
            document.getElementById("observation_list").innerHTML += html;
          }
        }
      }

      function display_request(pid)
      {
        var pw = document.getElementById("pw").value;
        var di_http_request;
        var url = "download.lib.php?update=true&pid="+pid+"&pw="+pw;

        if (window.XMLHttpRequest)
          di_http_request = new XMLHttpRequest()
        else
          di_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        di_http_request.onreadystatechange = function() 
        {
          handle_display_request(di_http_request)
        }

        di_http_request.open("GET", url, true)
        di_http_request.send(null)
        document.getElementById("observation_list").innerHTML = "Loading downloadble archives...";
      }

      function change_PID()
      {
        var pid_i = document.getElementById("pid").selectedIndex;
        if (pid_i == 0)
          alert("Please select a PID");
        else 
        {
          var pid = document.getElementById("pid").options[pid_i].value;
          display_request(pid);
        }
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
    $this->openBlockHeader("Display");
?>
    <p>Select the PID (and enter password) for the project you
       wish to access the archives of.</p>

    <table>
      <tr>
        <td>PID</td>
        <td>
          <select id="pid" name="pid">
            <option value="">--</option>
<?
            for ($i=0; $i<count($this->groups); $i++)
            {
              echo "            <option value='".$this->groups[$i]."'>".$this->groups[$i]."</option>\n";
            }
?>
          </select>
        </td>
      </tr>
      <tr>
        <td>Password</td>
        <td><input id="pw" type="password" name="password" size=8></td>
      </tr>
      <tr><td colspan=2><input type="button" value="Show" onClick="change_PID()"></td></tr>
    </table>
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("Select Observations");
?>
    Select: 
    <input type="button" name="All" value="All" onClick="checkAll(document.formlist.list)">
    <input type="button" name="None" value="None" onClick="uncheckAll(document.formlist.list)"><br><br>
    <input type="button" name="Download" value="Download Selected" onClick="downloadSelected(document.formlist.list)">
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
  <title>CASPSR | Results</title>
<?
    echo "    <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>\n";

    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
  
    $this->printJavaScriptHead();
?>
</head>


<body> 
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
    $this->openBlockHeader("Downloadable Observations");
?>
    <form name="formlist">
    <div id="observation_list">
      Please select a PID from the left panel
    </div>
    </form>
<?
    $this->closeBlockHeader();
  }

  #############################################################################
  #
  # print update information for the download page
  #
  function printUpdateHTML($get)
  {

    $dir = $this->inst->config["SERVER_RESULTS_DIR"];
    $pid = $get["pid"];
    $pw  = $get["pw"];

    if ($pw != "mb8782")
    {
      echo "ERROR: Incorrect password for PID ".$pid; 
      exit(0);
    }

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
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.info' | xargs grep PID | grep ".$pid." | awk -F/ '{print $(NF-1)\" \"\$NF}' | awk '{print \$1\" \"\$3}'";
    $pids = array();
    $tmp_array = array();
    $rval = 0;
    $lastline = exec($cmd, $tmp_array, $rval);
    for ($i=0; $i<count($tmp_array); $i++)
    {
      $bits = explode(" ",$tmp_array[$i]);
      $pids[$bits[0]] = $bits[1];
    }

    # get a list of the SOURCES
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

    $observations = array_keys($pids);
    sort($observations);

    # get the size of each observation
    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i];
      $cmd = "du -sk ".$this->inst->config["SERVER_ARCHIVE_DIR"]."/".$o." | awk '{printf \"%1.0f\", ($1/1024) }'";
      $array = array();
      $sizes[$o] = exec($cmd, $array, $rval);
    }

    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i];
      if ($states[$o] != "obs.deleted") 
        echo $o." ".$states[$o]." ".$pids[$o]." ".$sources[$o]." ".$sizes[$o]."\n";
    }
  }

  #
  # Download the observation, this operates by tarring the observation
  # and piping it to the webserver
  #
  function printDownloadFile($get)
  {
    $utc_start = $get["utc_start"];

    $a_dir = $this->inst->config["SERVER_ARCHIVE_DIR"];
    $r_dir = $this->inst->config["SERVER_RESULTS_DIR"];

    if (!is_readable($a_dir."/".$utc_start) || !is_readable($r_dir."/".$utc_start))
    {
      echo "ERROR: could not read archives for UTC_START=$utc_start";
      exit(0);
    }

    $filename = "CASPSR_".$utc_start.".tar";

    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    header('Content-type: application/x-tar');
    header('Content-Disposition: attachment; filename="'.$filename.'"');

    $tar_excludes = "--exclude \"*.png\" --exclude obs.info --exclude obs.finished  --exclude obs.transferred";

    $cmd = "tar -ch ".$utc_start." -C ".$r_dir." ".$utc_start." ".$tar_excludes;

    passthru("cd ".$a_dir."; ".$cmd);

  }

  #
  # Download multiple observations, this operates by tarring the observation
  # and piping it to the webserver
  #
  function printDownloadFiles($get)
  {
    $utc_starts = explode(",", $get["utc_starts"]);
    sort($utc_starts);
    
    $a_dir = $this->inst->config["SERVER_ARCHIVE_DIR"];
    $r_dir = $this->inst->config["SERVER_RESULTS_DIR"];

    # verify the specified directories exist and are readably
    $problem = 0;
    for ($i=0; $i<count($utc_starts); $i++)
    {
      $u = $utc_starts[$i];
      if (!is_readable($a_dir."/".$u) || !is_readable($r_dir."/".$u))
      {
        echo "ERROR: could not read archives for UTC_START=$u";
        $problem = 1;
      }
    }

    if ($problem)
      exit(0);
  
    $filename = "CASPSR_".$utc_starts[0]."_to_".$utc_starts[count($utc_starts)-1].".tar";

    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    header('Content-type: application/x-tar');
    header('Content-Disposition: attachment; filename="'.$filename.'"');

    $tar_excludes = "--exclude \"*.png\" --exclude obs.info --exclude obs.finished  --exclude obs.transferred";

    $cmd = "tar -ch";
    for ($i=0; $i<count($utc_starts); $i++)
    {
      $u = $utc_starts[$i];
      $dirs .= " ".$u;
    }
    $cmd .= $dirs." -C ".$r_dir.$dirs;
    $cmd .= " ".$tar_excludes;

    passthru("cd ".$a_dir."; ".$cmd);
  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else if ($_GET["download"] == "true") {
      $this->printDownloadFile($_GET);
    } else if ($_GET["download"] == "multi") {
      $this->printDownloadFiles($_GET);
    } else {
      $this->printHTML($_GET);
    }
  }

}
$obj = new download();
$obj->handleRequest($_GET);
