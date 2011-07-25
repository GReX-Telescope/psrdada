<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

ini_set("memory_limit","128M");

define(OBF, 0);   # obs.finished
define(OBT, 1);   # obs.txt
define(OBP, 2);   # obs.problem
define(OBX, 3);   # obs.transferred
define(OBA, 4);   # obs.archived
define(OBD, 5);   # obs.delted
define(OBI, 6);   # obs.info
define(OBS, 7);   # obs.start
define(SRC, 8);   # souce name
define(SRV, 9);   # flag for survey pointing 
define(STS, 10);   # sent.to.swin
define(STP, 11);   # sent.to.parkes
define(OTS, 12);   # on.tape.swin
define(OTP, 13);   # on.tape.parkes
define(ETS, 14);   # error.to.swin
define(ETP, 15);   # error.to.parkes
define(INA, 16);   # in /nfs/archives/bpsr
define(INR, 17);   # in /nfs/results/bpsr
define(INL, 18);   # in /lfs/data0/bpsr/archives
define(NBM, 19);   # number of beams
define(PID, 20);   # PID of the observation
define(BMB, 21);   # MB per beam

define(STS_COUNT, 22);
define(STP_COUNT, 23);
define(OTS_COUNT, 24);
define(OTP_COUNT, 25);

class archival extends bpsr_webpage
{
  var $inst = 0;
  var $cfg = array();
  var $nobs = 0;

  var $swin_dirs = array();
  var $parkes_dirs = array();
  var $swin_db = array();
  var $parkes_db = array();
  var $results = array();

  function archival()
  {
    bpsr_webpage::bpsr_webpage();
    $this->inst = new bpsr();
    $this->cfg = $this->inst->config;
    $this->title = "BPSR | Archival Pipeline";

    for ($i=0; $i<$this->cfg["NUM_SWIN_DIRS"]; $i++) {
      $arr = split(":",$this->cfg["SWIN_DIR_".$i]);
      $this->swin_dirs[$i] = array();
      $this->swin_dirs[$i]["user"] = $arr[0];
      $this->swin_dirs[$i]["host"] = $arr[1];
      $this->swin_dirs[$i]["disk"] = $arr[2];
    }

    for ($i=0; $i<$this->cfg["NUM_PARKES_DIRS"]; $i++) {
      $arr = split(":",$this->cfg["PARKES_DIR_".$i]);
      $this->parkes_dirs[$i] = array();
      $this->parkes_dirs[$i]["user"] = $arr[0];
      $this->parkes_dirs[$i]["host"] = $arr[1];
      $this->parkes_dirs[$i]["disk"] = $arr[2];
    }

    # Get the bookkeeping db information
    $arr = split(":",$this->cfg["SWIN_DB_DIR"]);
    $this->swin_db["user"] = $arr[0];
    $this->swin_db["host"] = $arr[1];
    $this->swin_db["dir"]  = $arr[2];
    $this->swin_db["file"] = $arr[2]."/files.P???.db";

    $arr = split(":",$this->cfg["PARKES_DB_DIR"]);
    $this->parkes_db["user"] = $arr[0];
    $this->parkes_db["host"] = $arr[1];
    $this->parkes_db["dir"]  = $arr[2];
    $this->parkes_db["file"] = $arr[2]."/files.P???.db";

    # get the list of all observations in the results dir
    $this->results = array();
    $this->nobs = 0;

    $cmd = "find ".$this->cfg["SERVER_RESULTS_DIR"]." -mindepth 1 -maxdepth 1 -type d -name '2*' -printf '%f\n' | sort";
    $last_line = exec ($cmd, $this->results, $rval);
    if ($rval == 0)
      $this->nobs = count($this->results);

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
      .yellow {
        background-color: yellow;
      }
      .gray {
        background-color: #DDDDDD;
      }
      .green {
        background-color: lightgreen;
      }
      .red {
        background-color: #FF3366;
      }
      .clear {
        background-color: transparent;
      }
    
      table.results {
        margin: 0;
        padding: 0;
        border-collapse: collapse;
      }

      .results th {
        padding: 4px 7px;
        border-width: 1px 1px 1px 1px;
        border-style: solid;
        border-color: #C1C1C1;
        font-size: 9pt;
        font-weight: bold;
        white-space: nowrap;
        text-align: center;
      }

      .results td {
        padding: 1px 6px 1px 6px;
        border: 1px solid #C6C6C6;
        font-size: 8pt;
        vertical-align:top;
        white-space: nowrap;
      }

      .legend td {
        padding-right: 10px;
      }

    </style>

    <script type='text/javascript'>

      function handle_archival_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          var xmlObj=xmlDoc.documentElement; 

          var i, j, k, result, key, value, span, this_result, nbeam, tr, utc_start;

          var results = xmlObj.getElementsByTagName("obs");

          var deleted_new_class;
          if (document.getElementById('hide_deleted').checked == true)
            deleted_new_class = "hidden";
          else
            deleted_new_class = "shown";

          // for each result returned in the XML DOC
          for (i=0; i<results.length; i++) {

            result = results[i];
            this_result = new Array();

            for (j=0; j<result.childNodes.length; j++) 
            {

              // if the child node is an element
              if (result.childNodes[j].nodeType == 1) {
                key = result.childNodes[j].nodeName;
                // if there is a text value in the element
                if (result.childNodes[j].childNodes.length == 1) {
                  value = result.childNodes[j].childNodes[0].nodeValue;
                } else {
                  value = "";
                }
                this_result[key] = value;
              }
            }

            utc_start = this_result["utc_start"];
            nbeam = parseInt(this_result["nbeam"]);

            tr = document.getElementById("row_"+utc_start);
            //tr.className = "shown";

            for ( key in this_result) 
            {
              value = this_result[key];
              try 
              {
                td = document.getElementById(key+"_"+utc_start);
                if (key.indexOf("state") != -1) 
                {
                  td.innerHTML = value;

                  document.getElementById("state_arch_"+utc_start).className = "clear";
                  document.getElementById("state_xfer_"+utc_start).className = "clear";
                  document.getElementById("state_fin_"+utc_start).className = "clear";
                  document.getElementById("state_del_"+utc_start).className = "clear";

                  if ((value == "deleted") || (value == "archived"))
                    document.getElementById("state_arch_"+utc_start).className = "green";

                  if (value == "archiving")
                    document.getElementById("state_arch_"+utc_start).className = "yellow";

                  if ((value == "deleted") || (value.indexOf("archiv") != -1) || (value == "transferred"))
                    document.getElementById("state_xfer_"+utc_start).className = "green";

                  if (value == "transferring")
                    document.getElementById("state_xfer_"+utc_start).className = "yellow";

                  if ((value == "deleted") || (value.indexOf("archiv") != -1) || (value.indexOf("transfer") != -1) || (value == "finished"))
                    document.getElementById("state_fin_"+utc_start).className = "green";

                  if (value == "failed")
                    document.getElementById("state_fin_"+utc_start).className = "red";

                  if (value == "deleted")
                  {
                    document.getElementById("state_del_"+utc_start).className = "green";
                    tr.className = deleted_new_class;
                  }
                  else
                    tr.className = "shown";


                }
                else if ((key.indexOf("sent_to") != -1) || (key.indexOf("on_tape") != -1))
                {
                  value = parseInt(value);
                  if ((nbeam == 0) || (value == -1))
                  {
                    td.innerHTML = "--";
                    td.className = "gray";
                  }
                  else if (value < nbeam)
                  {
                    td.innerHTML = value;
                    td.className = "yellow";
                  }
                  else
                  {
                    td.innerHTML = value;
                    td.className = "green";
                  }
                }
                else  
                {
                  td.innerHTML = value;
                }
              } catch (e) {
                // do nothing 
              }
            }
          }
          document.getElementById("summary_text").innerHTML = "Done";
        }
      }

      function archival_request() 
      {
        var xml_request;
        var pid_index = document.getElementById("pid").selectedIndex;
        var pid = document.getElementById("pid").options[pid_index].value;
        var url = "archival.lib.php?update=true&pid="+pid;

        document.getElementById("summary_text").innerHTML = "Retrieving Data...";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest()
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        xml_request.onreadystatechange = function() 
        {
          handle_archival_request(xml_request)
        }

        xml_request.open("GET", url, true)
        xml_request.send(null)
      }

      function get_fresh_data()
      {
        var trs = document.getElementsByTagName('tr');
        for (i=0; i<trs.length; i++)
        {
          if (trs[i].className == "shown")
          {
            trs[i].className = "hidden";
          }
        }
        archival_request();
      }

      function toggle_deleted()
      {
        var deleted_new_class;
        if (document.getElementById('hide_deleted').checked == true)
          deleted_new_class = "hidden";
        else
          deleted_new_class = "shown";

        var trs = document.getElementsByTagName('tr');
        var row, utc, td;

        for (i=0; i<trs.length; i++)
        {
          row = trs[i].id;
          if (row.indexOf("row_") == 0)
          {
            utc = row.substr(4);
            td = document.getElementById("state_"+utc);
            if (td.innerHTML == "deleted") 
            {
              trs[i].className = deleted_new_class;
            }
          }
        }
      }

    </script>
<?
  }

  ###########################################################################
  # 
  # Print the sidebar HTML
  #
  function printSideBarHTML()
  {

    $this->openBlockHeader("Controls");
?>
      <table>
        <tr>
          <td colspan=2>Limit the observations shown:</td>
        </tr>
        <tr>
          <td>PID:</td>
          <td>
            <select name='pid' id='pid'>
              <option value="all" selected>All</option>
<?
    $pids = $this->inst->getPIDS();
    for ($i=0; $i<count($pids); $i++) {
      echo "<option value=".$pids[$i].">".$pids[$i]."</option>\n";
    }
?>
          </select></td>
        </tr>
        <tr>
          <td>Hide Deleted</td>
          <td><input id='hide_deleted' name='hide_deleted' type='checkbox' onChange='toggle_deleted()'></td>
        </tr>
        <tr>
          <td colspan=2>
            <input type="button" value="Get Fresh Data" onClick="get_fresh_data()">
          </td>
        </tr>
      </table>
<?

    $this->closeBlockHeader();

    echo "<br/>\n";

    $this->openBlockHeader("Legend");
?>
      <table width="100%" class="legend">
        <tr><td>SIZE</td><td>Beam Size [GB] on processing node</td></tr>
      </table>


      <table width="100%" class="legend">
        <tr><th colspan=3>Obs State</th></tr>
        <tr><td>F</td><td class="green">Finished</td><td class="red">Failed</td></tr>
        <tr><td>T</td><td class="yellow">Transfering</td><td class="green">Transferred</td></tr>
        <tr><td>A</td><td class="yellow">Archiving</td><td class="green">Archived</td></tr>
        <tr><td>D</td><td class="green">Deleted</td></tr>
      </table>

      <table width="100%" class="legend">
        <tr><th colspan=2>Data</th></tr>
        <tr><td>N</td><td>Beams recorded</td></tr>
        <tr><td>F</td><td>Beams still on disk</td></tr>
      </table>

      <table width="100%" class="legend">
        <tr><th colspan=2>Sent To</th></tr>
        <tr><td>S</td><td>Beams transferred to Swin</td></tr>
        <tr><td>P</td><td>Beams transferred to Parkes</td></tr>
      </table>

      <table width="100%" class="legend">
        <tr><th colspan=2>On Tape</th></tr>
        <tr><td>S</td><td>Beams archived at Swin</td></tr>
        <tr><td>P</td><td>Beams archived at Parkes</td></tr>
      </table>
<?
    $this->closeBlockHeader();
  }

  ###########################################################################
  #
  # Generate the table, which will be filled in via a XML operation
  # 
  function printHTML()
  {

    # get the list of all observations in the results dir
    if ($this->nobs == 0)
    {
      $this->openBlockHeader("ERROR in Archival Pipeline");
      echo "<font color=red>Failed to read results dir ".$this->cfg["SERVER_RESULTS_DIR"]."</font><br>\n";
    }
    else
    {
      $this->openBlockHeader("Observations in Archival Pipeline [<span id='summary_text'>".$this->nobs."</span>]");

?>
    <table width='100%' border=0 class="results">

      <tr>
        <th colspan=5>
        <th colspan=5>Obs State</th>
        <th colspan=2>Data</th>
        <th colspan=2>Sent To</th>
        <th colspan=2>On Tape</th>
      </tr>
      <tr>
        <th>#</th>
        <th>UTC START</th>
        <th>SOURCE</th>
        <th>PID</th>
        <th>SIZE</th>
        <th></th>
        <th width="10px">F</th>
        <th width="10px">T</th>
        <th width="10px">A</th>
        <th width="10px">D</th>
        <th width="15px">N</th>
        <th width="15px">F</th>
        <th width="15px">S</th>
        <th width="15px">P</th>
        <th width="15px">S</th>
        <th width="15px">P</th>
      </tr>
<?
      for ($i=0; $i<$this->nobs; $i++)
      {
        $o = $this->results[$i];
        echo "      <tr id='row_".$o."' class='hidden'>";
          echo "<td>".($i+1)."</td>";
          echo "<td id='utc_start_".$o."'></td>";
          echo "<td id='source_".$o."'></td>";
          echo "<td id='pid_".$o."'></td>";
          echo "<td id='beamsize_".$o."' style='text-align: right;'></td>";
          echo "<td id='state_".$o."'></td>";
          echo "<td id='state_fin_".$o."'></td>";
          echo "<td id='state_xfer_".$o."'></td>";
          echo "<td id='state_arch_".$o."'></td>";
          echo "<td id='state_del_".$o."'></td>";
          echo "<td id='nbeam_".$o."'></td>";
          echo "<td id='on_local_disk_".$o."'></td>";
          echo "<td id='sent_to_swin_".$o."'></td>";
          echo "<td id='sent_to_parkes_".$o."'></td>";
          echo "<td id='on_tape_swin_".$o."'></td>";
          echo "<td id='on_tape_parkes_".$o."'></td>";
        echo "</tr>\n";
      }
?>
    </table>
<?
    }
    $this->closeBlockHeader();
?>
    <script type="text/javascript">get_fresh_data()</script>
<?
  }

  #############################################################################
  #
  # print update information for the archival page
  #
  function printUpdateHTML($get)
  {

    $show_pid = "all";
    if (isset($get["pid"]))
      $show_pid = $get["pid"];

    $pid_dests = $this->inst->getPIDDestinations();

    # get information about observations in the staging areas
    $staged = $this->getStagedObservations();

    # get information about observations archived to tape
    # $archived = $this->getStagingAreaInfo();

    # get information about observations on local disks
    $local = $this->getLocalObservations($show_pid);

    $keys = array_keys($local);

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<archival>\n";

    for ($i=0; $i<count($keys); $i++)
    {
      $o = $keys[$i];
      $pid = $local[$o][PID];

      if (($show_pid != "all") && ($pid != $show_pid))
        continue;

      $source = $local[$o][SRC];
      $nbeam = $local[$o][NBM];
      $beamsize = sprintf("%5.2f", ($local[$o][BMB]/1024));
      $sent_to_swin = $local[$o][STS_COUNT];
      $sent_to_parkes = $local[$o][STP_COUNT];
      $on_tape_swin = $local[$o][OTS_COUNT];
      $on_tape_parkes = $local[$o][OTP_COUNT];
      $on_local_disk = $local[$o][INL];

      $req_swin = false;
      $req_parkes = false;
      if (array_key_exists($pid, $pid_dests))
      {
        $req_swin = (strpos($pid_dests[$pid], "swin") !== FALSE) ? true : false;
        $req_parkes = (strpos($pid_dests[$pid], "parkes") !== FALSE) ? true : false;
      }
    
      # determine obs state
      $obs_state = 0;
      if (!$obs_state && ($local[$o][OBP] == 0))
        $obs_state = "failed";
      if (!$obs_state && ($local[$o][OBD] == 2))
        $obs_state = "deleted";
      if (!$obs_state && ($local[$o][OBA] == 2))
        $obs_state = "archived";
      if (!$obs_state && ($local[$o][OBX] == 2) && (($req_swin && $on_tape_swin && ($on_tape_swin < $nbeam)) || (($req_parkes && $on_tape_parkes && ($on_tape_parkes < $nbeam)))))
        $obs_state = "archiving";
      if (!$obs_state && ($local[$o][OBX] == 2))
        $obs_state = "transferred";
      if (!$obs_state && ($local[$o][OBF] == 2) && (($req_swin && $sent_to_swin && ($sent_to_swin < $nbeam)) || (($req_parkes && $sent_to_parkes && ($sent_to_parkes < $nbeam)))))
        $obs_state = "transferring";
      if (!$obs_state && ($local[$o][OBF] == 2))
        $obs_state = "finished";
      if (!$obs_state)
        $obs_state = "unknown";

      # state of archival for swin/parkes
      $swin_state = "unknown";
      $parkes_state = "unknown";
      $xml .= "<obs>\n";
      $xml .= "<utc_start>".$o."</utc_start>\n";
      $xml .= "<source>".$source."</source>\n";
      $xml .= "<state>".$obs_state."</state>\n";
      $xml .= "<pid>".$pid."</pid>\n";
      $xml .= "<nbeam>".$nbeam."</nbeam>\n";
      $xml .= "<beamsize>".$beamsize."</beamsize>\n";
      //$xml .= "<req_swin>".$req_swin."</req_swin>\n";
      //$xml .= "<req_parkes>".$req_parkes."</req_parkes>\n";
      $xml .= "<sent_to_swin>".$sent_to_swin."</sent_to_swin>\n";
      $xml .= "<sent_to_parkes>".$sent_to_parkes."</sent_to_parkes>\n";
      $xml .= "<on_tape_swin>".$on_tape_swin."</on_tape_swin>\n";
      $xml .= "<on_tape_parkes>".$on_tape_parkes."</on_tape_parkes>\n";
      $xml .= "<on_local_disk>".$on_local_disk."</on_local_disk>\n";
      $xml .= "</obs>\n";
    }
    $xml .= "</archival>\n";
   
    header('Content-type: text/xml');
    echo $xml;
  }

  function getStagedObservations()
  {

    $staged = array();

    $beam_dir_find_suffix = "find -mindepth 2 -maxdepth 2 -type d' | awk '{print substr($1,3)}' | sort";

    # Get the swin staged listing
    for ($i=0; $i<count($this->swin_dirs); $i++) 
    {
      $u = $this->swin_dirs[$i]["user"];
      $d = $this->swin_dirs[$i]["disk"];
      $h = $this->swin_dirs[$i]["host"];

      $cmd = "ssh -l ".$u." ".$h." 'cd ".$d."/P630; find pulsars staging_area -mindepth 2 -maxdepth 2 -type d' | awk -F/ '{print $2\"/\"$3}'";
      $array = array();
      $lastline = exec($cmd, $array, $return_var);
      for ($j=0; (($return_var == 0) && ($j<count($array))); $j++) 
      {
        list($o, $b) = split("/", $array[$j]);

        if (!array_key_exists($o, $staged))
        {
          $staged[$o] = array();
          $staged[$o]["SB"] = $b;
          $staged[$o]["SC"] = 1;
        }
        else 
        {
          $staged[$o]["SB"] .= " ".$b;
          $staged[$o]["SC"]++;
        }
      }
    }

    # Get the parkes staged listing
    for ($i=0; $i<count($this->parkes_dirs); $i++) 
    {
      $u = $this->parkes_dirs[$i]["user"];
      $d = $this->parkes_dirs[$i]["disk"];
      $h = $this->parkes_dirs[$i]["host"];

      $cmd = "ssh -l ".$u." ".$h." 'cd ".$d."/P630; find pulsars staging_area -mindepth 2 -maxdepth 2 -type d' | awk -F/ '{print $2\"/\"$3}'";
      $array = array();
      $lastline = exec($cmd, $array, $return_var);
      for ($j=0; (($return_var == 0) && ($j<count($array))); $j++)
      {
        list($o, $b) = split("/", $array[$j]);

        if (!array_key_exists($o, $staged))
        {
          $staged[$o] = array();
          $staged[$o]["PB"] = $b;
          $staged[$o]["PC"] = 1;
        }
        else
        {
          $staged[$o]["PB"] .= " ".$b;
          $staged[$o]["PC"]++;
        }
      }
    }

    return $staged;
  }

  function getArchivedObservations()
  {
    $archived = array();

    $u = $this->swin_db["user"];
    $h = $this->swin_db["host"];
    $f = $this->swin_db["file"];

    $cmd = "ssh -l ".$u." ".$h." \"cat ".$f." | sort\" | awk '{print $1}'";
    echo $cmd."<BR>\n";

    $array = array();
    $lastline = exec($cmd, $array, $return_var);
    echo "returned ".count($array)." results<BR>\n"; 
    for ($i=0; $i<count($array); $i++) 
    {
      list ($o, $b) = split("/", $array[$i]);
      # if this is a not an old observation (i.e. we have it in results
      if (in_array($o, $this->results))
      {
        if (!array_key_exists($archived))
        {
          $archived[$o] = array();
          $archived[$o]["SB"] = $b;
          $archived[$o]["SC"] = 1;
        }
        else
        {
          $archived[$o]["SB"] .= " ".$b;
          $archived[$o]["SC"]++;
        }
      }
    }

    $u = $this->parkes_db["user"];
    $h = $this->parkes_db["host"];
    $f = $this->parkes_db["file"];

    $cmd = "ssh -l ".$u." ".$h." \"cat ".$f." | sort\" | awk '{print $1}'";
    echo $cmd."<BR>\n";

    $array = array();
    $lastline = exec($cmd, $array, $return_var);
    echo "returned ".count($array)." results<BR>\n"; 
    for ($i=0; $i<count($array); $i++)
    {
      list ($o, $b) = split("/", $array[$i]);
      # if this is a not an old observation (i.e. we have it in results
      if (in_array ($o, $this->results))
      {
        if (!array_key_exists($archived))
        {
          $archived[$o] = array();
          $archived[$o]["PB"] = $b;
          $archived[$o]["PC"] = 1;
        }
        else
        {
          $archived[$o]["PB"] .= " ".$b;
          $archived[$o]["PC"]++;
        }
      }
    }
  
    return $archived;
  }


  # see what archives exist on local disks
  function getLocalObservations($pid) 
  {

    $a_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $r_dir = $this->cfg["SERVER_RESULTS_DIR"];

    $results = array();

    $obs_a = getSubDirs($a_dir);
    $obs_r = getSubDirs($r_dir);

    $obs_all = array_merge($obs_a, $obs_r);
    $obs_all = array_unique($obs_all);
    sort($obs_all);

    # Get all the obs_a subdirs count
    $obs_a_subdir_count = array();
    $cmd = "find ".$a_dir." -mindepth 2 -maxdepth 2 -type l | awk -F/ '{print $5\" \"$6}'";
    $array = array();
    $lastline = exec($cmd, $array, $return_var);
    for ($j=0; $j<count($array); $j++) {
      $arr = split(" ", $array[$j]);
      if (!array_key_exists($arr[0], $obs_a_subdir_count)) $obs_a_subdir_count[$arr[0]] = 0;
      $obs_a_subdir_count[$arr[0]] += 1;
    }

    # Get the remote file listings soas to not tax NFS too much
    $user = "bpsr";
    $remote = array();
    for ($i=0; $i<=14; $i++) {
      $host = sprintf("apsr%02d",$i);
      $temp_array = $this->getRemoteListing($user, $host, "/lfs/data0/bpsr/archives", $remote);
      $remote = $temp_array;
    }

    $local = $this->getObsDotListing("/nfs/archives/bpsr");

    $tot = count($obs_all);

    // For each observation get a list of frequency channels present
    for ($i=0; $i<$tot; $i++) {

      // the current obs
      $o = $obs_all[$i];

      $results[$o] = array();

      $results[$o][NBM] = 0;
      if (array_key_exists($o, $obs_a_subdir_count))
        $results[$o][NBM] = $obs_a_subdir_count[$o];

      // Check that the observation is listed in the local array
      if (array_key_exists($o, $local)) {
        $results[$o][OBF] = $local[$o][OBF];
        $results[$o][OBX] = $local[$o][OBX];
        $results[$o][OBA] = $local[$o][OBA];
        $results[$o][OBD] = $local[$o][OBD];
        $results[$o][OBI] = $local[$o][OBI];
        $results[$o][OBP] = $local[$o][OBP];
        $results[$o][OBT] = $local[$o][OBT];
        $results[$o][SRC] = $local[$o][SRC];
        $results[$o][SRV] = (substr($results[$o][SRC],0,1) == "G") ? 1 : 0;
        $results[$o][BMB] = 0;
      }
   
      $results[$o][STS_COUNT] = 0; 
      $results[$o][STP_COUNT] = 0; 
      $results[$o][OTS_COUNT] = 0; 
      $results[$o][OTP_COUNT] = 0; 
      $results[$o][INL] = 0;
      if (array_key_exists($o, $remote)) {
        $results[$o][STS_COUNT] = $remote[$o][STS_COUNT];
        $results[$o][STP_COUNT] = $remote[$o][STP_COUNT];
        $results[$o][OTS_COUNT] = $remote[$o][OTS_COUNT];
        $results[$o][OTP_COUNT] = $remote[$o][OTP_COUNT];
        $results[$o][INL]       = $remote[$o][INL];
        $results[$o][BMB]       = $remote[$o][BMB];
      }

      $results[$o][INA] = 0;
      if (in_array($o,$obs_a)) {
        $results[$o][INA] = 2;
      }
      $results[$o][INR] = 0;
      if (in_array($o,$obs_r)) {
        $results[$o][INR] = 2;
      }

      # now remove the obs from the remote and local arrays for memory reasons 
      unset($remote[$o]); 
      unset($local[$o]); 
    }

    # get the PID of each observation from obs.info
    $cmd = "find /nfs/archives/bpsr/ -type f -maxdepth 2 -name 'obs.info' -print0 | xargs -0 grep ^PID | awk -F/ '{print $5\" \"$6}' | awk '{print $1\" \"$3}'";
    $lastline = exec($cmd, $array, $return_val);
    for ($i=0; $i<count($array); $i++) {
      $bits = split(" ",$array[$i]);
      $results[$bits[0]][PID] = $bits[1];
    }

    if ($pid != "all") {
      $keys = array_keys($results);
      for ($i=0; $i<count($keys); $i++) {
        if ($results[$keys[$i]][PID] != $pid) 
          unset($results[$keys[$i]]);
      }
    }

    return $results;

  }

  #   
  # SSH to the remote machine and get listing information on each of the files we need
  #   
  function getRemoteListing($user, $host, $dir, $results) 
  {
    
    $array = array();
    $cmd = "ssh -l ".$user." ".$host." \"web_observations_helper.pl\"";
    $lastline = exec($cmd, $array, $return_val);

    for ($i=0; $i<count($array); $i++) 
    {
      $a = split("/", $array[$i]);

      $o = $a[1];
      $b = $a[2];
      $f = $a[3];
      $s = $a[4]; # in bytes
    
      if (! array_key_exists($o, $results)) {
        $results[$o] = array();
        $results[$o][OBS] = 0;
        $results[$o][INL] = 0;
        $results[$o][STS_COUNT] = 0;
        $results[$o][STP_COUNT] = 0;
        $results[$o][OTS_COUNT] = 0;
        $results[$o][OTP_COUNT] = 0;
        $results[$o][BMB] = 0;
      }
      if (! array_key_exists($b, $results[$o])) {
        $results[$o][$b] = array();
        $results[$o][$b][STS] = 0;
        $results[$o][$b][STP] = 0;
        $results[$o][$b][OTS] = 0;
        $results[$o][$b][OTP] = 0;
        $results[$o][$b][ETS] = 0;
        $results[$o][$b][ETP] = 0;
      }
      if ($f == "obs.start") {
        $results[$o][OBS]++;
      } else if ($f == "sent.to.swin") {
        $results[$o][$b][STS] = 1;
        $results[$o][STS_COUNT] += 1;
      } else if ($f == "sent.to.parkes") {
        $results[$o][$b][STP] = 1;
        $results[$o][STP_COUNT] += 1;
      } else if ($f == "on.tape.swin") {
        $results[$o][$b][OTS] = 1;
        $results[$o][OTS_COUNT] += 1;
      } else if ($f == "on.tape.parkes") {
        $results[$o][$b][OTP] = 1;
        $results[$o][OTP_COUNT] += 1;
      } else if ($f == "error.to.swin") {
        $results[$o][$b][ETS] = 1;
      } else if ($f == "error.to.parkes") {
        $results[$o][$b][ETP] = 1;
      } else if (($f == "integrated.ar") || (substr($f, ".fil") !== FALSE)) {
        $results[$o][INL]++;
        $s_mb = $s / (1024*1024);
        if ($s_mb > $results[$o][BMB]) {
          $results[$o][BMB] = $s_mb;
        }
      } else {
        echo "getRemoteListing: ".$a." was unmatched<BR>\n";
      }
    }

    return $results;
  }

  #
  # Gets a listing of all the obs. files in the dir's subdirs
  #
  function getObsDotListing($dir) {

    $cmd = "cd $dir; find -mindepth 2 -maxdepth 2 -type f -name 'obs.*' -o -name 'error.to.*'";
    $array = array();
    $lastline = exec($cmd, $array, $return_val);

    $results = array();

    for ($i=0; $i<count($array); $i++) {

      $a = split("/", $array[$i]); 
      $o = $a[1];
      $f = $a[2];

      if (! array_key_exists($o, $results)) {
        $results[$o] = array();
        $results[$o][OBF] = -1;
        $results[$o][OBT] = -1;
        $results[$o][OBP] = -1;
        $results[$o][OBX] = -1;
        $results[$o][OBA] = -1;
        $results[$o][OBD] = -1;
        $results[$o][OBI] = -1;
        $results[$o][ETS] = 0;
        $results[$o][ETP] = 0;
      }

      if (($f == "obs.finished") || ($f == "obs.finalized")){
        $results[$o][OBF] = 2;
      }
      if ($f == "obs.txt") {
        $results[$o][OBT] = 2;
      }
      if (($f == "obs.problem") || ($f == "obs.failed")) {
        $results[$o][OBP] = 0;
      }
      if ($f == "obs.transferred") {
        $results[$o][OBX] = 2;
      }
      if ($f == "obs.archived") {
        $results[$o][OBA] = 2;
      }
      if ($f == "obs.deleted") {
        $results[$o][OBD] = 2;
      }
      if ($f == "obs.info") {
        $results[$o][OBI] = 2;
      }
      if ($f == "error.to.swin") {
        $results[$o][ETS] = 1;
      }
      if ($f == "error.to.parkes") {
        $results[$o][ETP] = 1;
      }
    }

    $cmd = "cd $dir; find . -maxdepth 2 -type f -name 'obs.info' -print0 | xargs -0 grep SOURCE | awk -F/ '{print $(NF-1)\" \"\$NF}' | awk '{print $1\" \"$3}'";
    $array = array();
    $lastline = exec($cmd, $array, $return_val);

    for ($i=0; $i<count($array); $i++) {
      $a = split(" ", $array[$i]);
      $o = $a[0];
      $s = $a[1];
      $results[$o][SRC] = $s; 
    }

    unset($array);

    return $results;
  }
}

handleDirect("archival");

