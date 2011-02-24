<?PHP

/* 
 *  Faciliates the 'downloading' of archives as tar files via the web browser 
 */

include("definitions_i.php");
include("functions_i.php");
include("apsr_functions_i.php");
ini_set("memory_limit","128M");

define(YES, "Y");
define(NO, "N");
define(MAYBE, "M");
define(OFF, "_");

define(NFS_RESULTS,  "NFSR");
define(NFS_ARCHIVES, "NFSA");
define(OBS_INFO,     "OBSI");
define(ARCHIVE_SIZE, "ARCHIVE_SIZE");
define(OBS_SIZE,     "OBS_SIZE");
define(N_SUBINT,     "N_SUBINT");
define(N_BAND,       "NBAND");
define(N_SOURCES,    "NSRCS");
define(RAND_BAND,    "RNDB");

$cfg = getConfigFile(SYS_CONFIG,TRUE);

$pid = "none";
if (isset($_GET["pid"])) {
  $pid = $_GET["pid"];
} 

$action = "none";
if (isset($_GET["action"])) {
  $action = $_GET["action"];
} 

# determine the PIDS that this instrument has configured from the unix groups
$cmd = "groups ".INSTRUMENT;
$output = array();
$string = exec($cmd, $output, $return_var);
$array = split(" ",$string);
$groups = array();
for ($i=0; $i<count($array); $i++) {
  if (strpos($array[$i], "P") === 0) {
    array_push($groups, $array[$i]);
  }
}

?>
<html>
<head>
  <title>APSR | Download Archvies</title>
  <link rel="STYLESHEET" type="text/css" href="/bpsr/style.css">
  <link rel="shortcut icon" href="/images/favicon.ico"/>
  <style>
    .active {
      font-family: arial,helvetica,sans-serif,times;
      font-size: 8pt;
      height: 10px;
      padding: 1px 6px 1px 6px;
      border: 1px solid #C6C6C6;
    }
    .inactive {
      font-family: arial,helvetica,sans-serif,times;
      font-size: 0pt;
      height: 0px;
      padding: 0px 0px 0px 0px;
      border: 0px;
    }
    table.obstable {
      margin: 0px 0px 0px 0px;
      padding: 0px 0px 0px 0px;
      border-collapse: separate;
      clear: both;
    }
  </style>
  
  <script language="javascript">

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
        url = "download_obs.php?utc_starts="+utcs[0];
        for (i = 1; i < utcs.length; i++)
          url += ","+utcs[i];
        document.location = url;
      }
    }


  </script>

</head>

<?

$text = "APSR Download Archives";
include("banner.php");

echo "<center>\n";

# Get the disk listing information
$num_swin_dirs = $cfg["NUM_SWIN_DIRS"];
$num_parkes_dirs = $cfg["NUM_PARKES_DIRS"];
$results = array();

if ($pid != "none") {

echo "<span id=\"progress\">\n";

###############################################################################

echo "Checking Results<BR>\n";
flush();
$cmd = "find ".$cfg["SERVER_RESULTS_DIR"]." -maxdepth 2 -type f -name obs.info | xargs grep ^PID | grep ".$pid;
$array = array();
$lastline = exec($cmd, $array, $return_var);
$start_pos = strlen($cfg["SERVER_RESULTS_DIR"]) + 1;

for ($i=0; $i<count($array); $i++) {
  $u = substr($array[$i], $start_pos, 19);

  if (!array_key_exists($u, $results)) {
    $results[$u] = array();
    $results[$u][NFS_RESULTS] = NO;
    $results[$u][NFS_ARCHIVES] = NO;
  }
  $results[$u][NFS_RESULTS] = YES;
  $results[$u][OBS_INFO] = YES;
}

###############################################################################

echo "Checking Archives<br>\n";
flush();
$cmd = "find ".$cfg["SERVER_ARCHIVE_DIR"]." -maxdepth 2 -type f -name obs.info | xargs grep ^PID | grep ".$pid;
$array = array();
$lastline = exec($cmd, $array, $return_var);
$start_pos = strlen($cfg["SERVER_ARCHIVE_DIR"]) + 1;
for ($i=0; $i<count($array); $i++) {
  $u = substr($array[$i], $start_pos, 19);
  if (!array_key_exists($u, $results)) {
    $results[$u] = array();
    $results[$u][NFS_RESULTS] = NO;
    $results[$u][NFS_ARCHIVES] = NO;
  }
  $results[$u][NFS_ARCHIVES] = YES;
  $results[$u][OBS_INFO] = YES;
}

###############################################################################


echo "Parsing Observational Parameters<br>\n";
flush();
$str = "";
for ($i=0; $i<100; $i++) {
  $str .= "&nbsp;";
}

echo "0% [<span id='progress_bar'><tt>".$str."</tt></span>] 100%\n";
flush();

$keys = array_keys($results);
sort($keys);
$params = array("SOURCE", "BANDWIDTH", "NBIT", "CFREQ", "PROC_FILE", "ARCHIVE_SIZE", "N_SUBINT");
$n_keys = count($keys);

$disp_str = "";
for ($i=0; $i<$n_keys; $i++) {

  $tmp_str = "";
  $pc = floor(($i / $n_keys)*100);
  for ($j=0; $j<$pc-1; $j++)   $tmp_str .= ".";
  for ($j=$pc; $j<=100; $j++) $tmp_str .= "&nbsp;";
  
  if ($tmp_str != $disp_str) {
    $disp_str = $tmp_str;
    echo "<script type='text/javascript'>document.getElementById(\"progress_bar\").innerHTML = \"<tt>".$disp_str."</tt>\"</script>\n";
    flush();
  }

  $o = $keys[$i];

  if (($results[$o][NFS_ARCHIVES] == YES) && ($results[$o][OBS_INFO] == YES)) {
    $obs_info = $cfg["SERVER_ARCHIVE_DIR"]."/".$o."/obs.info";
  } elseif (($results[$o][NFS_RESULTS] == YES) && ($results[$o][OBS_INFO] == YES)) {
    $obs_info = $cfg["SERVER_RESULTS_DIR"]."/".$o."/obs.info";
  } else {
    $obs_info = "";
  } 
  
  if ($obs_info != "") {
    $cmd = "cat ".$obs_info;
    $array = array();
    $lastline = exec($cmd, $array, $return_val);
    for ($j=0; $j<count($array); $j++) {
      $a = split("[ \t]+",$array[$j],2);
      if (in_array($a[0],$params)) {
        $results[$o][$a[0]] = $a[1];
      } 
    }
  }

  # choose a random band to 'interrogate'
  $array = array();
  $cmd = "find ".$cfg["SERVER_RESULTS_DIR"]."/".$o." -mindepth 1 -maxdepth 1 -type d -printf '%f\n'";
  $lastline = exec($cmd, $array, $return_val);
  $results[$o][N_BAND] = count($array);
  if (count($array) > 0) {
    $rand_i = rand(0, count($array)-1);
    $results[$o][RAND_BAND] = $array[$rand_i];
  } else {
    $results[$o][RAND_BAND] = "";
  }

  # find out the number of sources
  $cmd = "find ".$cfg["SERVER_RESULTS_DIR"]."/".$o."/".$results[$o][RAND_BAND]." -mindepth 1 -maxdepth 1 -type d | wc -l";
  $lastline = exec($cmd, $array, $return_val);
  $results[$o][N_SOURCES] = $lastline;
  if ($results[$o][N_SOURCES] == 0) {
    $results[$o][N_SOURCES] = 1;
  }

}
echo "<br>\n";


###############################################################################

echo "Determining Integration Lengths<br>\n";
$str = "";
for ($i=0; $i<100; $i++) {
  $str .= "&nbsp;";
}

echo "0% [<span id='progress_bar2'><tt>".$str."</tt></span>] 100%\n";
flush();
$disp_str = "";
for ($i=0; $i<$n_keys; $i++) {

  $tmp_str = "";
  $pc = floor(($i / $n_keys)*100);
  for ($j=0; $j<$pc-1; $j++)   $tmp_str .= ".";
  for ($j=$pc; $j<=100; $j++) $tmp_str .= "&nbsp;";
  
  if ($tmp_str != $disp_str) {
    $disp_str = $tmp_str;
    echo "<script type='text/javascript'>document.getElementById(\"progress_bar2\").innerHTML = \"<tt>".$disp_str."</tt>\"</script>\n";
    flush();
  }

  $o = $keys[$i];


  # if obs.info didn't have the archive size in it
  if ((!array_key_exists(ARCHIVE_SIZE, $results[$o])) || ($results[$o][ARCHIVE_SIZE] == "N/A")) {

    $first_ar = $cfg["SERVER_ARCHIVE_DIR"]."/".$o."/".$results[$o][RAND_BAND]."/".$o.".ar";

    if (! file_exists($first_ar)) {
      $cmd = "find -L ".$cfg["SERVER_ARCHIVE_DIR"]."/".$o."/".$results[$o][RAND_BAND]." -name \"*.ar\" | head -n 1";
      $first_ar = exec($cmd, $array, $return_val);
    } 

    $archive_size = "N/A";
    if (file_exists($first_ar)) {
      $cmd = "du -skL ".$first_ar;
      $lastline = exec($cmd, $array, $return_val);
      $archive_size = sprintf("%d", ($lastline * 1024));
    } 

    # If it previously existed and was erroneous
    if ($results[$o][ARCHIVE_SIZE] == "N/A") {
      $cmd = "perl -i -p -e 's/^ARCHIVE_SIZE        N\/A/ARCHIVE_SIZE        ".$archive_size."/' ".$cfg["SERVER_ARCHIVE_DIR"]."/".$o."/obs.info";
    } else {
      $cmd = "echo \"ARCHIVE_SIZE        ".$archive_size."\" >> ".$cfg["SERVER_ARCHIVE_DIR"]."/".$o."/obs.info";
    }
    exec($cmd, $array, $return_val);
    $results[$o][ARCHIVE_SIZE] = $archive_size;
  }

  if (!array_key_exists(N_SUBINT, $results[$o])) {

    # get the integration lengths
    $results[$o][N_SUBINT] = "N/A";

    $tres = $cfg["SERVER_RESULTS_DIR"]."/".$o."/total_t_res.ar";
    if (file_exists($tres)) {
      $results[$o][N_SUBINT] = getNumSubints($tres);
    } else {
      $src = $results[$o]["SOURCE"];
      if (strlen($src) > 5) {
        $src = preg_replace("/^[JB]/","", $src);
        $tres = $cfg["SERVER_RESULTS_DIR"]."/".$o."/".$src."_t.ar";
        if (file_exists($tres)) {
          $results[$o][N_SUBINT] = getNumSubints($tres);
        }
      }
    }

    $cmd = "echo \"N_SUBINT            ".$results[$o][N_SUBINT]."\" >> ".$cfg["SERVER_ARCHIVE_DIR"]."/".$o."/obs.info";
    exec($cmd, $array, $return_val);
  }

  if (($results[$o][ARCHIVE_SIZE] != "N/A") && ($results[$o][N_SUBINT] != "N/A")) {
    $results[$o][OBS_SIZE] = sprintf("%d",($results[$o][N_SOURCES] * $results[$o][N_BAND] * $results[$o][ARCHIVE_SIZE] * $results[$o][N_SUBINT]) / ( 1024 * 1024 ));
  } else {
    $results[$o][OBS_SIZE] = "N/A";
  }

}

echo "<br>\n";

###############################################################################

echo "DONE, formatting page<br>\n";
echo "</span>";

?>

<script type="text/javascript">
  document.getElementById("progress").innerHTML = "";
</script>

<?
}
?>

<table cellpadding=10>
  <tr><td style="vertical-align: top;">
    <form name="selectpid" action="download_archives.php" method="get">
    <table>
      <tr>
        <td colspan=2 style="vertical-align: middle; text-align: left">Show Observations for PID</td>
        <td align=left>
          <select name="pid" id="pid" onChange="form.submit()">
            <option value="none">--</option>
<?
    for ($i=0; $i<count($groups); $i++) {
      echo "      <option value=\"".$groups[$i]."\"".($groups[$i] == $pid ? " selected" : "").">".$groups[$i]."</option>\n";
    }
?>
          </select>
        </td>
      </tr>
    </table>
    </form>

  </td><td>
<form name="formlist">
<table border=0 cellpadding=0 cellspacing=0 class='obstable'>
 <thead>
  <tr>
    <td colspan=13>Select 
      <input type="button" name="All" value="All" onClick="checkAll(document.formlist.list)">
      <input type="button" name="None" value="None" onClick="uncheckAll(document.formlist.list)">
      <input type="button" name="Download" value="Download Selected" onClick="downloadSelected(document.formlist.list)">
    </td>
  </tr>
  <tr>
    <th colspan=13>&nbsp;
    </th>
  </tr>

  <tr>
   <th class="active"></th>
   <th class="active"></th>
   <th class="active">UTC START</th>
   <th class="active">View</th>
   <th class="active">Download</th>
   <th class="active">SOURCE</th>
   <th class="active">BW</th>
   <th class="active">NBIT</th>
   <th class="active">CFREQ</th>
   <th class="active">PROC FILE</th>
   <th class="active">SIZE</th>
   <!--<th class="active">N SUBINT</th>
   <th class="active">N SOURCES</th>
   <th class="active">N BANDS</th>
   <th class="active">ARCHIVE SIZE</th>
   <th class="active">RAND BAND</th>-->
   <th class="active">r</th>
   <th class="active">a</th>
   <th>&nbsp;&nbsp;&nbsp;</th>
  </tr>
 </thead>
 <tbody style="height: 700px; overflow: auto; overflow-x: hidden;">
<?

$keys = array_keys($results);
sort($keys);
$num = 0;
$class = "active";

for ($i=0; $i<count($keys); $i++) {

  $o = $keys[$i];
  $d = $results[$o];

  if (($d["PROC_FILE"] != "dspsr.single") && ($d["PROC_FILE"] != "dspsr.singleF") && ($d["PROC_FILE"] != "apsr.scratch")) {

  echo "  <tr>\n";

  /* COUNT */
  echo "    <td class=\"".$class."\">".$num."</td>\n";

  echo "    <td class='$class'><input type='checkbox' name='list' value='$i' id='$o'></td>\n";

  /* UTC_START */
  echo "    <td class=\"".$class."\">".$o."</td>\n";

  /* Results */
  echo "    <td class=\"".$class."\"><a href='/apsr/result.php?utc_start=".$o."'>view</a></td>\n";
   
  /* Download Link */
  echo "    <td class=\"".$class."\"><a href='/apsr/download_obs.php?utc_starts=".$o."'>download</a></td>\n";

  /* SOURCE */
  echo "    <td class=\"".$class."\">".$d["SOURCE"]."</td>\n";

  /* BW */
  echo "    <td class=\"".$class."\">".sprintf("%d",$d["BANDWIDTH"])."</td>\n";

  /* NBIT */
  echo "    <td class=\"".$class."\">".$d["NBIT"]."</td>\n";

  /* CFREQ */
  echo "    <td class=\"".$class."\">".sprintf("%d",$d["CFREQ"])."</td>\n";

  /* PROC_FILE */
  echo "    <td class=\"".$class."\">".$d["PROC_FILE"]."</td>\n";

  /* ARCHIVE_SIZE */
  echo "    <td class=\"".$class."\" style='text-align: right'>".($d[OBS_SIZE] != "N/A" ? $d[OBS_SIZE]." MB" : "")."</td>\n";

  /* N SUB INTS */
  /*
  echo "    <td class=\"".$class."\" style='text-align: right'>".$d[N_SUBINT]."</td>\n";
  echo "    <td class=\"".$class."\" style='text-align: right'>".$d[N_SOURCES]."</td>\n";
  echo "    <td class=\"".$class."\" style='text-align: right'>".$d[N_BAND]."</td>\n";
  echo "    <td class=\"".$class."\" style='text-align: right'>".$d[ARCHIVE_SIZE]."</td>\n";
  echo "    <td class=\"".$class."\" style='text-align: right'>".$d[RAND_BAND]."</td>\n";
  */
  /* NFS_RESULTS */
  echo "    ".statusTD($d[NFS_RESULTS],$class,"&nbsp;")."\n";

  /* NFS_ARCHIVES */
  echo "    ".statusTD($d[NFS_ARCHIVES],$class,"&nbsp;")."\n";


  echo "    <td></td>\n";
  echo "  </tr>\n";
  $num++;
  }
}

?>
 </tbody>
</table>
</form>
</td></tr></table>

</body>
</html>

<?

function statusTD($on, $class, $text) {

  if ($text === "") {
    $text = "&nbsp;";
  }

  if ($on == YES) {
    return "<td width=10px bgcolor='lightgreen' class='".$class."'>".$text."</td>";
  } else if ($on == MAYBE) {
    return "<td width=10px bgcolor='yellow' class='".$class."'>".$text."</td>";
  } else if ($on == NO) {
    return "<td width=10px bgcolor='red' class='".$class."'>".$text."</td>";
  } else { 
    return "<td width=10px class='".$class."'>".$text."</td>";
  } 

}


function getRemoteListing($user, $host, $dir, $results) {

  # Return a listing in the format: UTC_START BAND [FILES OF SIGNIFICANCE]
  $cmd = "ssh -l ".$user." ".$host." \"web_apsr_local_observations.pl\"";

  $array = array();
  $lastline = exec($cmd, $array, $return_val);

  for ($i=0; $i<count($array); $i++) {

    $a = split("/", $array[$i]);

    $o = $a[1];
    $b = $a[2];

    if (! array_key_exists($o, $results)) {
      $results[$o] = array();
      $results[$o][OBS] = 0;
      $results[$o][INL] = 0;
      $results[$o][STS_COUNT] = 0;
      $results[$o][STP_COUNT] = 0;
    }
    if (! array_key_exists($b, $results[$o])) {
      $results[$o][$b] = array();
      $results[$o][$b]["sts"] = 0;
      $results[$o][$b]["stp"] = 0;
    }
    if ($a[3] == "obs.start") {
      $results[$o][OBS]++;
    } else if ($a[3] == "sent.to.swin") {
      $results[$o][$b][STS] = 1;
      $results[$o][STS_COUNT] += 1;
    } else if ($a[3] == "sent.to.parkes") {
      $results[$o][$b][STP] = 1;
      $results[$o][STP_COUNT] += 1;
    } else if ($a[3] == "error.to.swin") {
      $results[$o][$b][ETS] = 1;
    } else if ($a[3] == "error.to.parkes") {
      $results[$o][$b][ETP] = 1;
    } else if (($a[3] == "integrated.ar") || (substr($a[3], ".fil") !== FALSE)) {
      $results[$o][INL]++;
    } else {
      echo "getRemoteListing: ".$a." was unmatched<BR>\n";
    }
  }

  return $results;



}

?>
