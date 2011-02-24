<?PHP
include("definitions_i.php");
include("functions_i.php");
include("apsr_functions_i.php");

# Get the system configuration (dada.cfg)
$cfg = getConfigFile(SYS_CONFIG);
$results_dir = "/export/old_results/apsr";

$utc_start = "unknown";
if (isset($_GET["utc_start"])) {
  $utc_start = $_GET["utc_start"];
}

echo "<html>\n";
$title = "APSR | ".$utc_start;
include("header_i.php");

?>
<body>

  <script type="text/javascript">

  window.name = "resultwindow";

  function popWindow(URL) {
    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,"+
         "scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width=1024,height=600');");
  }

  </script>

<? 

include("banner.php");

$obs_dir = $results_dir."/".$utc_start;

if ($utc_start == "unknown") {
  echo "<p>Error: UTC_START was not specified in GET parameters</p>";
  echo "</body>\n";
  echo "</html>\n";
  exit ;
}

if (!file_exists($obs_dir)) {
  echo "<p>Error: UTC_START directory did not exist in ".$results_dir."</p>\n";
  echo "</body>\n";
  echo "</html>\n";
  exit ;
}

# get information about the source(s) in this observation
$source_info = getObsSources($obs_dir);
$obs_info    = getConfigFile($obs_dir."/obs.info");
$source      = $obs_info["SOURCE"];
$obs_start   = getObsStartHeader($obs_dir);
$most_recent = getMostRecentResult($obs_dir);
$imgs        = getObsImages($obs_dir);
$obs_state   = getObsState($obs_dir);

/* determine if this is a single pulse observation */
$single_pulse = 0;
if (strpos($obs_start["PROC_FILE"], "singleF") !== FALSE) {
  $single_pulse = 1;
}

/* If less than 5 minutes since the last archive was recieved */
$locked = "";
if ($most_recent < 120) {
  $locked = " disabled";
}

?>

<table border=0>
  <tr>
    <td align="left" valign="top"> 
<?
    printSummary($results_dir, $utc_start, $source, $source_info, $cfg, $most_recent);
  
    echo "<br>\n";
    echo "<center>\n";
    printSourceInfo($source_info);
    echo "</center>\n";

?></td>
    <td align="right" valign="top"> <?printHeader($obs_start);?> </td>
  </tr>
  <tr>
    <td colspan=2 align=center><input id="custom_plot" type=button value="View Custom Plots" onClick="popWindow('custom_plot.php?basedir=/export/old_results/apsr&utc_start=<?echo $utc_start?>')"></td>
  </tr>
  <tr>
    <td colspan=2><? printPlots($results_dir, $utc_start, $imgs); ?></td>
  </tr>
</table>

</body>
</html>

<?


function printSummary($base_dir, $utc_start, $source, $data, $cfg, $most_recent) {

  $results_link = "<a href='/apsr/old_results/".$utc_start."/'>".$base_dir."</a>";

  $obs_dir = $base_dir."/".$utc_start;

?>
  <table class="results" width=100% border=0>
    <tr><th colspan=2 class="results">OBSERVATION SUMMARY</th></tr>
    <tr><td align="right" class="results" width="50%">UTC Start</td><td class="results" width=50%><?echo $utc_start?></td></tr>
    <tr><td align="right" class="results">Local Time Start</td><td class="results"><?echo localTimeFromGmTime($utc_start)?></td></tr>
    <tr><td align="right" class="results">Time Since Last Result</td><td class="results"><?echo makeTimeString($most_recent)?></td></tr>
    <tr><td align="right" class="results">Results Dir</td><td class="results"><? echo $results_link?></td></tr>
  </table>

<?
}


function printHeader($header) {

  $keys = array_keys($header);
  $keys_to_ignore = array("HDR_SIZE","FILE_SIZE","HDR_VERSION","FREQ","RECV_HOST");

?>
<table class="results">
  <tr> <th class="results" colspan=2>DADA HEADER</th></tr>
<?
  if (count($keys) == 0) {
    echo "<tr><td colspan=2><font color=red>obs.start file did not exist</font></td></tr>\n";
  } else {

    for ($i=0; $i<count($keys); $i++) {

      if (!(in_array($keys[$i], $keys_to_ignore))) {

        echo "  <tr>";
        echo "<td align=\"right\" class=\"results\">".$keys[$i]."</td>";
        echo "<td class=\"results\">".$header[$keys[$i]]."</td>";
        echo "</tr>\n";
      }
    }
  }
?>
</table>
<?

}


function printPlots($results_dir, $utc_start, $imgs) {

?>
<table width=800px>
  <tr>
    <th class="results" colspan=4>PLOTS</th>
  </tr>
  <tr>
    <th class="results">PSR</th>
    <th class="results">Phase vs Flux</th>
    <th class="results">Phase vs Time</th>
    <th class="results">Phase vs Freq</th>
  </tr>
<? 
  $psrs = array_keys($imgs);
  for ($i=0; $i<count($psrs); $i++) {
    $p = $psrs[$i];
    echo "  <tr>\n";
    echo "    <td>".$p."</td>\n";
    printPlotRow($results_dir, $utc_start, $imgs[$p]["phase_vs_flux"], $imgs[$p]["phase_vs_flux_hires"]);
    printPlotRow($results_dir, $utc_start, $imgs[$p]["phase_vs_time"], $imgs[$p]["phase_vs_time_hires"]);
    printPlotRow($results_dir, $utc_start, $imgs[$p]["phase_vs_freq"], $imgs[$p]["phase_vs_freq_hires"]);
    echo "  </tr>\n";
  }
?>
</table>
<?

}

function printPlotRow($results_dir, $utc_start, $image, $image_hires) {

  $have_hires = 0;
  $hires_path = $results_dir."/".$utc_start."/".$image_hires;
  if ((strlen($image_hires) > 1) && (file_exists($hires_path))) {
    $have_hires = 1;
  } 

  echo "    <td class=\"results\" align=\"center\">\n";

  if ($have_hires) {
    echo "      <a href=\"/apsr/old_results/".$utc_start."/".$image_hires."\">";
  }

  echo "      <img width=241px height=181px src=\"/apsr/old_results/".$utc_start."/".$image."\">";

  if ($have_hires) {
    echo "    </a><br>\n";
    echo "    Click for hi-res result\n";
  }

  echo "    </td>\n";

}

function printSourceInfo($data) {

  $vals = array("dm", "p0", "int", "snr", "nsubint");
  $names = array("DM", "P0", "Integrated", "SNR", "nsubint");

  $keys = array_keys($data);
  sort($keys);

  echo "<table class='results' border=0>\n";
  echo "  <tr><th colspan=".(1+count($keys))." class='results'>SOURCE SUMMARY</th></tr>\n";

  echo "  <tr><td align=right class=results>Source</td>";
  for ($j=0; $j<count($keys); $j++) {
    echo "<td align=right class=results>".$keys[$j]."</td>";  
  } 
  echo "</tr>\n";

  for ($i=0; $i<count($vals); $i++) {
    $v = $vals[$i];
  
    echo  "<tr><td align=right class=results>".$names[$i]."</td>";

    for ($j=0; $j<count($keys); $j++) {
      $k = $keys[$j];
      echo "<td align=right class=results>".$data[$k][$v]."</td>";
    }
    echo "</tr>\n";
  }

  echo "</table>\n";
}


/* determine the unix timestamp of the most recent result */
function getMostRecentResult($dir) {

  $current_time = time();
  $difference = 0;

  $cmd = "find ".$dir." -name 'obs.start' -printf '%T@\\n' | sort | tail -n 1";
  $obs_start_time = exec($cmd, $array, $rval);

  if (count($array) == 0) {
    $difference = -1;
  } else {
    $difference = $current_time - $obs_start_time;
  }

  return $difference;
}


/* try to get the header information from the obs.start file */
function getObsStartHeader($dir) {

  $bands = getSubDirs($dir);
  $file = "";

  /* Now get all the .lowres files for each freq channel */
  for ($i=0; (($i<count($bands)) && ($file == "")); $i++) {

    $band = $bands[$i];

    if (file_exists($dir."/".$band."/obs.start")) {
      $file = $dir."/".$band."/obs.start";
    }
  }

  if ($file == "") {
    return array();
  } else {
    return getConfigFile($file, TRUE);
  }

}

/* Determine the 'state' of the observation */
function getObsState($dir) {

  if (file_exists($dir."/obs.failed")) {
    return "failed";
  } else if (file_exists($dir."/obs.finished")) {
    return "finished";
  } else if (file_exists($dir."/obs.processing")) {
    return "processing";
  } else if (file_exists($dir."/obs.transferred")) {
    return "transferred";
  } else if (file_exists($dir."/obs.deleted")) {
    return "deleted";
  } else {
    return "unknown";
  }
}
?>
