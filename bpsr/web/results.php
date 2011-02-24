<?PHP

include ("bpsr.lib.php");
$inst = new bpsr();


define(RESULTS_PER_PAGE,20);
define(PAGE_REFRESH_SECONDS,20);

# Compact beam-o-vision observations
$beamovision = 1;
if (isset($_GET["beamovision"])) {
  $beamovision = $_GET["beamovision"];
}

# Most recent obs to show
$baseobs = "NONE";
if (isset($_GET["baseobs"])) {
  $baseobs = $_GET["baseobs"];
}

# Number of subsequent observations to show
$length = RESULTS_PER_PAGE;
if (isset($_GET["length"])) {
  $length = $_GET["length"];
}

# Reset the baseobs to the first one if we want to show all
if ($length == "all") {
  $baseobs = "NONE";
}

$basedir = $inst->config["SERVER_RESULTS_DIR"];


# Alternate to the above due to "arg list too long" errors
$cmd = "find ".$basedir." -maxdepth 2 -name 'obs.info' -printf \"%h\\n\" | awk -F/ '{print \$NF}' | sort -r";
$junk = exec($cmd, $array, $rval);

$all_observations = array();
for ($i=0; $i<count($array); $i++) {

  $utc = $array[$i];
  $all_observations[$utc]["SOURCE"] = "unknown";
  if ($baseobs == "NONE") {
    $baseobs = $utc;
  }
}

# try to detect beam-o-vision obserations
if ($beamovision) {
  $list = compactBeamOVision($basedir, $all_observations, $baseobs, $length);
} else {
  $list = $all_observations;
}

$keys = array_keys($list);

$actual_length = $length;
if ($length == "all") {
  $actual_length = count($keys);
}

$base_i = array_search($baseobs, $keys);
$prev_i = $base_i - $actual_length;
if ($prev_i < 0) {
  $prev_i = 0;
}
$next_i = $base_i + $actual_length;
if ($next_i >= count($keys)) {
  $next_i = count($keys)-1;
}

# The "first obs" on a forward/backward navigation
$curr_page_obs = $keys[$base_i];
$prev_page_obs = $keys[$prev_i];
$next_page_obs = $keys[$next_i];

$all_keys = array_keys($all_observations);
$curr_full_i = array_search($curr_page_obs, $all_keys);
$next_full_i = array_search($next_page_obs, $all_keys)-1;
$full_n = count($all_keys);

$start = $base_i;
$end = $base_i + $actual_length;

$display_obs = array();

/* only get additional information for the things we need */
for ($i=$start; $i<$end; $i++) {

  $k = $keys[$i];

  # Get intergration length of obs if not already known
  if (!isset($list[$k]["INT"])) {
    $cmd = "find ".$basedir."/".$k." -name '*.png' -printf '%f\\n' | tail -n 1";
    $array2 = array();
    $img = exec($cmd, $array2, $rval);
    $list[$k]["IMG"] = "bp";
    if ($img != "") {
      if (strpos($img, ".pvf_") !== FALSE) {
         $list[$k]["IMG"] = "pvf";
      } 
      $list[$k]["INT"] = calcIntLength($k, $img);
    } else {
      $list[$k]["INT"] = 0;
    }
  }
  $display_obs[$k] = $list[$k];

}

$types = array("bp", "pvf");
$display_keys = array_keys($display_obs);

# Get all the images in the display keys @ 400x300 resolution, but only beam "1"
$images = $inst->getResults($basedir, $display_keys, $types, "400x300", "1");
$results = getResultsArray($basedir, $display_keys);

?>

<html>
<?
  $inst->print_head("BPSR | Recent Results", 0);
?>

<body>
  <!--  Load tooltip module -->
  <script type="text/javascript" src="/js/wz_tooltip.js"></script>
  <script>
    function changeLength() {
      var newlength = document.getElementById("displayLength");
      var i = newlength.selectedIndex
      var length = newlength.options[i].value;

      var beamovision = document.getElementById("beamovision");
      var beam = "0";
      if (beamovision.checked) {
        beam = "1";
      }
      document.location = "/bpsr/results.php?baseobs=<?echo $curr_page_obs?>&length="+length+"&beamovision="+beam
    }
  </script>
<? 
  $inst->print_banner("BPSR Results");
?>
<div align=right>
<table>

 <tr>
  <td>Beam-O-Vision <input type=checkbox id="beamovision" name="beamovision" onChange="changeLength()"<? if($beamovision) echo " checked";?>></input></td>
  <td width=20px>&nbsp;</td>

  <td>Num Results: 
    <select name="displayLength" id="displayLength" onChange='changeLength()'>
<?  
    echoOption("20", "20", FALSE, $length);
    echoOption("50", "50", FALSE, $length);
    echoOption("100", "100", FALSE, $length); 
    echoOption("500", "500", FALSE, $length);
    echoOption("all", "all", FALSE, $length);
?>
    </select>
  </td>
  <td width=20px>&nbsp;</td>

<?
  echo "<td><a href=/bpsr/results.php?baseobs=".$prev_page_obs."&length=".$length."&beamovision=".$beamovision.">&#8249; Newer</a></td>\n";
  echo "<td width=5>&nbsp;</td>\n";
?>

   <td width=10>&nbsp;</td>
   <td><? echo "Showing <b>".$curr_full_i."</b> - <b>".($next_full_i)."</b> of <b>".$full_n."</b> results";?></td>
   <td width=10>&nbsp;</td>

<?
  echo "<td><a href=/bpsr/results.php?baseobs=".$next_page_obs."&length=".$length."&beamovision=".$beamovision.">Older &#8250;</a></td>\n";
  echo "<td width=5>&nbsp;</td>\n";
?>

 </tr>
</table>
</div>

<br>

<center>
<table class="datatable">
<tr>
  <th>Source</th>
  <th>UTC Start</th>
  <th>Int</th>
  <th>N Beams</th>
  <th class="trunc">Annotation</th>
</tr>

<?

for ($i=0; $i < count($display_keys); $i++) {

  $k = $display_keys[$i];
  $o = $display_obs[$k];

  if ($o["BEAMOVISION"] > 1) {
    $url = "/bpsr/beamovision.lib.php?single=true&utc_start=".$k;
    for ($j=1; $j<=$o["BEAMOVISION"]; $j++) {
      $url .= "&BEAM_".$j."=".$o["BEAM_".$j];
    }
    
    $imagetype = "pvf";
  } else {
    $url = "/bpsr/result.php?utc_start=".$k."&imagetype=".$o["IMG"];
    $imagetype = "pvf";
  }

  $pos = strpos($images[$k][0]["bp_400x300"], "blankimage");

  if ($pos !== FALSE) {
    $img = $images[$k][0]["pvf_400x300"];
  } else {
    $img = $images[$k][0]["bp_400x300"];
  }
  $mousein = "onmouseover=\"Tip('<img src=\'".$img."\' width=400 height=300>')\"";
  $mouseout = "onmouseout=\"UnTip()\"";

  echo "  <tr class=\"new\">\n";

  /* SOURCE */
  echo "    <td>\n";
  if ($o["BEAMOVISION"] > 0) {  
    echo "      <a href=\"".$url."\" ".$mousein." ".$mouseout.">BEAM-O-VISION</a></td>\n";
  } else {
    echo "      <a href=\"".$url."\" ".$mousein." ".$mouseout.">".$o["SOURCE"]."</a></td>\n";
  }

  /* UTC_START */
  echo "    <td>".$k."</td>\n";

  /* INTERGRATION LENGTH */
  echo "    <td>".$o["INT"]."</td>\n";

  /* NUM_BEAMS */
  if ($o["BEAMOVISION"] > 0) {
    echo "    <td>".$o["BEAMOVISION"]."</td>\n";
  } else {
    echo "    <td>".$results[$k]["nbeams"]."</td>\n";
  }

  /* ANNOTATION */
  echo "    <td class=\"trunc\"><div>";
  if ($o["BEAMOVISION"] > 0) {
    echo "SOURCE: ".$o["SOURCE"]." ";
  }
  echo $results[$k]["annotation"]."</div></td>\n";

  echo "  </tr>\n";

}
?>
</table>

</body>
</html>

</center>
<?

function getResultsArray($results_dir, $observations) {

  $all_results = array();

  /* For each observation get a list of frequency channels present */   
  for ($i=0; $i<count($observations); $i++) {

    $o = $observations[$i];
    $dir = $results_dir."/".$o;

    # number of beams
    $all_results[$o]["nbeams"] = exec("ls -1d ".$dir."/??/ | wc -l");

    # obs.start file
    if (file_exists($dir."/01/obs.start")) {
      $all_results[$o]["obs_start"] = $dir."/".$dir."/01/obs.start";  
    } else {
      $cmd = "find ".$dir." -name \"obs.start\" | tail -n 1";
      $all_results[$o]["obs_start"] = exec($cmd);
      if ($all_results[$o]["obs_start"] = "") {
        $all_results[$o]["obs_start"] = "unset";
      }
    }

    # obs.txt file
    if (file_exists($dir."/obs.txt")) {
      $all_results[$o]["annotation"] = file_get_contents($dir."/obs.txt");
    } else {
      $all_results[$o]["annotation"] = "";
    }

  }

  return $all_results;
}


function getRecordingLength($image_name) {

  if ($image_name == "/images/blankimage.gif") {
    return 0;
  } else {

    $array = split("/",$image_name);

    $utc_start = $array[3];
    $image_basename = $array[5];
    $array = split("\.",$image_basename);
    $image_utc = $array[0];

    $offset = 0;
    if (strpos($image_basename, "pvf") !== FALSE) {
      $offset = (11*60*60);
    }

    # add ten as the 10 second image file has a UTC referring to the first byte of the file 
    $length = (unixTimeFromGMTime($image_utc)+(10-$offset)) - unixTimeFromGMTime($utc_start);

    return $length; 

  }

}

function calcIntLength($utc_start, $image) {
  $array = split("\.",$image);
  $image_utc = $array[0];

  $offset = 0;
  if (strpos($image, "pvf") !== FALSE) {
    $offset = (11*60*60);
  }

  # add ten as the 10 second image file has a UTC referring to the first byte of the file 
  $length = (unixTimeFromGMTime($image_utc)+(10-$offset)) - unixTimeFromGMTime($utc_start);

  return $length;
}

function secondsDifference($utc1, $utc2) {

  $diff = unixTimeFromGMTime($utc1) - unixTimeFromGMTime($utc2);
  return $diff;

}


/*
 * compact the full list, by combining beam-o-vision observations
 */
function compactBeamOVision($basedir, $list, $baseobs, $length) {

  $beam_count = 0;
  $outlist = array();

  $surveys = array();
  $folded = array();
  $twobits = array();
  $beamovisions = array();

  $keys = array_keys($list);
  if ($length == "all") {
    $length = count($keys);
  }

  # always be able to have info for going back a step
  $back_search = $length * 13;
  $base_i = array_search($baseobs, $keys);

  $nsearch = $length * 13;
  $search_start = $base_i - $nsearch;
  $search_end = $base_i + $nsearch;

  if ($search_start < 0) {
    $search_start = 0;
  }
  if ($search_end > count($keys)){
    $search_end = count($keys);
  }

  /* base is the index of the currently shown first obs */
  /* length is the number of observations we want to show */
  for ($i=$search_start; ($i < $search_end); $i++) {

    $key = $keys[$i];
    $value = $list[$key];

    $cmd = "grep SOURCE /export/results/bpsr/".$key."/obs.info | awk '{print $2}'";
    $source = exec($cmd, $junkarray, $rval);
    $value["SOURCE"] = rtrim($source);
    //echo $cmd." -> |".$value["SOURCE"]."|<BR>\n";

    /* if the obs has a non G SOURCE */
    if (substr($value["SOURCE"],0,1) == "G") {
      $surveys[$key] = $value;
      $surveys[$key]["BEAMOVISION"] = 0;
      $surveys[$key]["IMG"] = "bp";
      $found++;
      
    /* this is a 2bit known pulsar or a beam-o-vision */
    } else {
      $arr = array();
      $cmd = "find ".$basedir."/".$key."/01 -name '*.png' -printf '%f\\n' | tail -n 1";
      $img = exec($cmd, $arr, $rval);
      $len = 0;
      if ($img != "") {
        $len = calcIntLength($key, $img);
      }
      if (strpos($img, ".pvf_") !== FALSE) {
        $folded[$key] = $value;
        $folded[$key]["INT"] = $len;
        $folded[$key]["IMG"] = "pvf";
      } else {
        $twobits[$key] = $value;
        $twobits[$key]["INT"] = $len;
        $twobits[$key]["IMG"] = "bp";
        $twobits[$key]["BEAMOVISION"] = 0;
      }
    }
  }

  $prev_source = "NONE";
  $prev_key = "NONE";
  $count = 0;
  $b = "NONE";

  # sort the folded, to detect beam-o-visions
  ksort($folded);

  foreach ($folded as $key => $value) {

    if (($count <= 13) && ($value["SOURCE"] == $prev_source) && (secondsDifference($key, $prev_key) < 120)){
      $beamovisions[$b]["BEAM_".$count] = $key;
      $beamovisions[$b]["BEAMOVISION"] = $count;
    } else {

      /* If the current beam-o-vision we are accumulating only has 1 beam in it */
      if ($count == 2) {
        //echo "Removing singular Beam-O-Vision: ".$prev_key."<BR>\n";
        unset($beamovisions[$prev_key]["BEAM_1"]);
        $beamovisions[$prev_key]["BEAMOVISION"] = 0;
        $twobits[$prev_key] = $beamovisions[$prev_key];
        unset($beamovisions[$prev_key]);
      }

      $b = $key;
      $count = 1;
      $beamovisions[$b] = $value;
      $beamovisions[$b]["BEAM_".$count] = $key;
      $beamovisions[$b]["BEAMOVISION"] = $count;
    }
    $count++;
    $prev_key = $key;
    $prev_source=  $value["SOURCE"];
  }

  $outlist = array_merge($surveys, $twobits, $beamovisions);
  krsort($outlist);

  return ($outlist);

}

?>
