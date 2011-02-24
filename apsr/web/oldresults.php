<?PHP
include("definitions_i.php");
include("functions_i.php");
include("apsr_functions_i.php");

define(RESULTS_PER_PAGE,20);
define(PAGE_REFRESH_SECONDS,60);

# Get the system configuration (dada.cfg)
$cfg = getConfigFile(SYS_CONFIG,TRUE);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

$filter_types = array("", "SOURCE", "CFREQ", "BANDWIDTH", "PID", "UTC_START", "PROC_FILE");

$length = RESULTS_PER_PAGE;
if (isset($_GET["length"])) {
  $length = $_GET["length"];
}

$offset = 0;
if (isset($_GET["offset"])) {
  $offset = $_GET["offset"];
} 

$inlineimages = 0;
if (isset($_GET["inlineimages"])) {
  $inlineimages = $_GET["inlineimages"];
}


$hide_CAL = 0;
if (isset($_GET["hidecal"])) {
  $hide_CAL = $_GET["hidecal"];
}

$filter_type = "";
if (isset($_GET["filter_type"])) {
  $filter_type = $_GET["filter_type"];
}

$filter_value = "";
if (isset($_GET["filter_value"])) {
  $filter_value = $_GET["filter_value"];
}


$basedir = "/export/old_results/apsr";
$archive_ext = ".ar";

$cmd = "";
if (($filter_type == "") && ($filter_value == "")) {
  $cmd = "find ".$basedir." -maxdepth 2 -name 'obs.info' | wc -l";
} else {
  if ($filter_type == "UTC_START") {
    $cmd = "find ".$basedir."/*".$filter_value."* -maxdepth 1 -name 'obs.info' | wc -l";
  } else {
    $cmd = "find ".$basedir." -maxdepth 2 -type f -name obs.info | xargs grep ".$filter_type." | grep ".$filter_value." | wc -l";
  }
}
$total_num_results = exec($cmd);


/* If we are on page 2 or more */
$newest_offset = "unset";
$newest_length = "unset";
if ($offset - $length >= $length) {
  $newest_offset = 0;
  $newest_length = $length;
}

/* If we are on page 1 or more */
$newer_offset = "unset";
$newer_length = "unset";
if ($offset > 0) {
  $newer_offset = $offset-$length;
  $newer_length = $length;
}

/* If we have at least 1 page of older results */
$older_offset = "unset";
$older_length = "unset";
if ($offset + $length < $total_num_results) {
  $older_offset = $offset + $length;
  $older_length = MIN($length, $total_num_results - $older_offset);
}

/* If we have at least 2 pages of older results */
$oldest_offset = "unset";
$oldest_length = "unset";
if ($offset + ($length*2) < $total_num_results) {
  $oldest_offset = $length * floor ($total_num_results / $length);
  $oldest_length = $total_num_results - $oldest_offset;
}

$results = getResultsArray($basedir, $archive_ext, $offset, $length, $filter_type, $filter_value);
$keys = array_keys($results);
$num_results = count($keys);

?>

<html>

<?

$title = "APSR | Old Results";
# $refresh = PAGE_REFRESH_SECONDS;
include("header_i.php");

?>

<body>
  <!--  Load tooltip module -->
  <script type="text/javascript" src="/js/wz_tooltip.js"></script>
  <script>
    function changeLength() {
      var newlength = document.getElementById("displayLength");
      var i = newlength.selectedIndex
      var length = newlength.options[i].value;
      var inlineimages = document.getElementById("inlineimages");
      var img = "0";
      if (inlineimages.checked) {
        img = "1";
      }
      var URL = "/apsr/oldresults.php?offset=<?echo $offset?>&length="+length+"&inlineimages="+img

      var filter_type = document.getElementById("filter_type");
      i = filter_type.selectedIndex
      var type = filter_type.options[i].value;
      var filter_value = document.getElementById("filter_value").value;

      if ((filter_value != "") && (type != "")) {
        URL = URL + "&filter_type="+type+"&filter_value="+filter_value;
      }

      document.location = URL;
    }
  </script>

<? 
$text = "Old Results";
include("banner.php");

?>
<div align=right>
<table>
 <tr>

  <td>Filter: 
    <select name="filter_type" id="filter_type">
<?
    for ($i=0; $i<count($filter_types); $i++) {
      $t = $filter_types[$i];
      echoOption($t, $t, FALSE, $filter_type);
    }

?>
    </select>
  </td>
  <td> <input name="filter_value" id="filter_value" value="<?echo $filter_value?>" onChange="changeLength()"></input></td>

  <td>Inline Images <input type=checkbox id="inlineimages" name="inlineimages" onChange="changeLength()"<?  if($inlineimages) echo " checked";?>></input></td>

  <td width=20px>&nbsp;</td>

  <td>
    Num Results: 
    <select name="displayLength" id="displayLength" onChange='changeLength()'>
<?
    echoOption("20", "20", FALSE, $length);
    echoOption("50", "50", FALSE, $length);
    echoOption("100", "100", FALSE, $length);
?>
    </select>
  </td>
  <td width=10>&nbsp;</td>


  <!--<td width=50%><input type="checkbox" value="hide_cal">Hide CAL Observations</input></td>-->
<?
  if ($newest_offset !== "unset") {
    echo "<td><a href=/apsr/oldresults.php?offset=".$newest_offset."&length=".$newest_length."&inlineimages=".$inlineimages.">&#171; Newest</a></td>\n";
    echo "<td width=5>&nbsp;</td>\n";
  }

  if ($newer_offset !== "unset") {
    echo "<td><a href=/apsr/oldresults.php?offset=".$newer_offset."&length=".$newer_length."&inlineimages=".$inlineimages.">&#8249; Newer</a></td>\n";
    echo "<td width=5>&nbsp;</td>\n";
  }
?>

   <td width=10>&nbsp;</td>
   <td><? echo "Showing <b>".$offset."</b> - <b>".($offset+$length)."</b> of <b>".$total_num_results."</b> results";?></td>
   <td width=10>&nbsp;</td>
<?

  if ($older_offset !== "unset") {
    echo "<td><a href=/apsr/oldresults.php?offset=".$older_offset."&length=".$older_length."&inlineimages=".$inlineimages.">Older &#8250;</a></td>\n";
    echo "<td width=5>&nbsp;</td>\n";
  }

  if ($oldest_offset !== "unset") {
    echo "<td><a href=/apsr/oldresults.php?offset=".$oldest_offset."&length=".$oldest_length."&inlineimages=".$inlineimages.">Oldest &#187;</a></td>\n";
  }

?>

 </tr>
</table>
</div>

<br>

<center>
<table class="datatable">
<tr>
<?
if ($inlineimages == 1) {
  echo "  <th>Image</th>\n";
}
?>
  <th>Source</th>
  <th>UTC Start</th>
  <th>CFREQ</th>
  <th>BW</th>
  <th>Int</th>
  <th>NCHAN</th>
  <th>SNR</th>
  <th>PID</th>
  <th>PROC_FILE</th>
  <th class="trunc">Annotation</th>
</tr>

<?

$keys = array_keys($results);

for ($i=0; $i < count($keys); $i++) {

  $k = $keys[$i];

  $data = getObservationImages($basedir."/".$k);

  $freq_keys = array_keys($results[$k]);
  $url = "/apsr/oldresult.php?utc_start=".$keys[$i];
  $mousein = "onmouseover=\"Tip('<img src=\'/apsr/old_results/".$k."/".$data["phase_vs_flux"]."\' width=241 height=181>')\"";
  $mouseout = "onmouseout=\"UnTip()\"";

  $bg_style = "";
  /* If archives have been finalised and its not a brand new obs */
  if ( $results[$keys[$i]]["processing"] === 1) {
    $bg_style = "style=\"background-color: white;\"";
    echo "  <tr class=\"new\">\n";
  } else {
    $bg_style = "style=\"background-color: #cae2ff;\"";
    echo "  <tr>\n";
  }

  /* IMAGE/SOURCE */
  if ($inlineimages == 1) {
    echo "    <td><a href=\"".$url."\" ".$mousein." ".$mouseout.">\n";
    echo "          <img src=/apsr/old_results/".$k."/".$data["phase_vs_flux"]." width=64 height=48>\n";
    echo "        </a></td>\n";
    echo "    <td ".$bg_style."><a href=\"".$url."\">".$results[$k]["SOURCE"]."</a></td>\n";
  } else {
    echo "      <td ".$bg_style."><a href=\"".$url."\" ".$mousein." ".$mouseout.">".$results[$k]["SOURCE"]."</a></td>\n";
  }

  /* UTC_START */
  echo "    <td ".$bg_style.">".$k."</td>\n";

  /* CFREQ */
  echo "    <td ".$bg_style.">".$results[$k]["CFREQ"]."</td>\n";

  /* BW */
  echo "    <td ".$bg_style.">".$results[$k]["BANDWIDTH"]."</td>\n";

  /* INTERGRATION LENGTH */
  echo "    <td ".$bg_style.">".$results[$k]["INT"]."</td>\n";

  /* NCHAN */
  echo "    <td ".$bg_style.">".$results[$k]["NCHAN"]."</td>\n";

  /* SNR */
  echo "    <td ".$bg_style.">".$results[$k]["SNR"]."</td>\n";

  /* Project ID*/
  echo "    <td ".$bg_style.">".$results[$k]["PID"]."</td>\n";

  /* PROC_FILE */
  echo "    <td ".$bg_style.">".$results[$k]["PROC_FILE"]."</td>\n";

  /* ANNOTATION */
  echo "    <td ".$bg_style." class=\"trunc\"><div>".$results[$k]["annotation"]."</div></td>\n";

  echo "  </tr>\n";


}
?>
</table>

<br>

<table>
 <tr><td colspan=3 align=center>Legend</td></tr>
 <tr><td class="smalltext">CFREQ</td><td width=20></td><td class="smalltext">Centre frequency of the observation [MHz]</td></tr>
 <tr><td class="smalltext">BW</td><td width=20></td><td class="smalltext">Total bandwidth [MHz]</td></tr>
 <tr><td class="smalltext">Int</td><td width=20></td><td class="smalltext">Total intergration received [seconds]</td></tr>
 <tr><td class="smalltext">NCHAN</td><td width=20></td><td class="smalltext">Number of hosts
that recieved data</td></tr>
 <tr><td class="smalltext">White</td><td width=20></td><td class="smalltext">Newer results, may still be updated</td></tr>
 <tr><td class="smalltext">Blue</td><td width=20></td><td class="smalltext">Finalised results, no new archives received for 5 minutes</td></tr>
</table> 

</body>
</html>

</center>
<?

function getResultsArray($results_dir, $archive_ext, $offset=0, $length=0, $filter_type, $filter_value) {

  $all_results = array();

  $observations = array();
  $dir = $results_dir;

  if (($filter_type == "") || ($filter_value == "")) {
    $observations = getSubDirs($results_dir, $offset, $length, 1);
  } else {

    # get a complete lists

    if ($filter_type == "UTC_START") {
      $cmd = "find ".$results_dir."/*".$filter_value."* -maxdepth 1 -name 'obs.info' -printf '%h\n' | awk -F/ '{print \$NF}' | sort -r";
    } else {
      $cmd = "find ".$results_dir." -maxdepth 2 -type f -name obs.info | xargs grep ".$filter_type." | grep ".$filter_value." | awk -F/ '{print $(NF-1)}' | sort -r";
    }
    $last = exec($cmd, $all_obs, $rval);
    $observations = array_slice($all_obs, $offset, $length);
  }

  /* For each observation get a list of frequency channels present */   
  for ($i=0; $i<count($observations); $i++) {

    $dir = $results_dir."/".$observations[$i];
    $freq_channels = getSubDirs($dir);

    /* If no archives have been produced */
    if (count($freq_channels) == 0) {
      $all_results[$observations[$i]]["obs_start"] = "unset";
    }
    $all_results[$observations[$i]]["nchan"] = count($freq_channels);
  } 

  for ($i=0; $i<count($observations); $i++) {
    $o = $observations[$i];
    $dir = $results_dir."/".$o;

    /* read the obs.info file into an array */
    if (file_exists($dir."/obs.info")) {
      $arr = getConfigFile($dir."/obs.info");
      $all_results[$o]["SOURCE"] = $arr["SOURCE"];
      $all_results[$o]["CFREQ"] = sprintf("%5.2f",$arr["CFREQ"]);
      $all_results[$o]["BANDWIDTH"] = $arr["BANDWIDTH"];
      $all_results[$o]["NCHAN"] = $arr["NUM_PWC"];
      $all_results[$o]["PID"] = $arr["PID"];
      $all_results[$o]["PROC_FILE"] = $arr["PROC_FILE"];
    }

    $cmd = "find ".$dir." -name \"obs.start\" | tail -n 1";
    $an_obs_start = exec($cmd);
    $all_results[$observations[$i]]["obs_start"] = $an_obs_start;

    # try to find the name of the summed tres/fres archives
    $tres_archive = "";
    $fres_archive = "";

    $ars = array();
    $cmd = "find ".$dir." -maxdepth 1 -name '*.ar'";
    $last = exec($cmd, $ars, $rval);
    for ($j=0; $j<count($ars); $j++) {
      if (strpos($ars[$j],"_t") !== FALSE)
        $tres_archive = $ars[$j];
      if (strpos($ars[$j],"_f") !== FALSE)
        $fres_archive = $ars[$j];
    }

     
    $all_results[$observations[$i]]["INT"] = getIntergrationLength($tres_archive);
    $all_results[$observations[$i]]["SNR"] = getSNR($fres_archive);

    if (file_exists($dir."/obs.txt")) {
      $all_results[$observations[$i]]["annotation"] = file_get_contents($dir."/obs.txt");
    } else {
      $all_results[$observations[$i]]["annotation"] = "";
    }
  
    if (file_exists($dir."/obs.processing")) {
      $all_results[$observations[$i]]["processing"] = 1;
    } else {
      $all_results[$observations[$i]]["processing"] = 0;
    }

  }

  return $all_results;
}

function getObservationImages($obs_dir) {
 
  $data["phase_vs_flux"] = "../../images/blankimage.gif";
  $data["phase_vs_time"] = "../../images/blankimage.gif";
  $data["phase_vs_freq"] = "../../images/blankimage.gif";

  if ($handle = opendir($obs_dir)) {
    while (false !== ($file = readdir($handle))) {
      if ($file != "." && $file != "..") {
        # First handle the images:
        if (preg_match("/^phase_vs_flux.+240x180.png$/",$file)) {
          $data["phase_vs_flux"] = $file;
        }
        if (preg_match("/^phase_vs_time.+240x180.png$/",$file)) {
          $data["phase_vs_time"] = $file;
        }
        if (preg_match("/^phase_vs_freq.+240x180.png$/",$file)) {
          $data["phase_vs_freq"] = $file;
        }
      }
    }
  }
  closedir($handle);
  return $data;
}


?>
