<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class asteria_bests extends mopsr_webpage 
{

  var $filter_types = array("", "SOURCE", "FREQ", "BW", "UTC_START", "PROC_FILE");
  var $cfg = array();
  var $length;
  var $offset;
  var $inline_images;
  var $filter_type;
  var $filter_value;
  var $class = "new";
  var $results_dir;
  var $results_link;
  var $archive_dir;

  function asteria()
  {
    mopsr_webpage::mopsr_webpage();
    $inst = new mopsr();
    $this->cfg = $inst->config;

    $this->length = (isset($_GET["length"])) ? $_GET["length"] : 20;
    $this->offset = (isset($_GET["offset"])) ? $_GET["offset"] : 0;
    $this->inline_images = (isset($_GET["inline_images"])) ? $_GET["inline_images"] : "false";
    $this->filter_type = (isset($_GET["filter_type"])) ? $_GET["filter_type"] : "";
    $this->filter_value = (isset($_GET["filter_value"])) ? $_GET["filter_value"] : "";
    $this->results_dir = $this->cfg["SERVER_RESULTS_DIR"];
    $this->archive_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $this->results_link = "/mopsr/results";
    $this->results_title = "Tied Array Beam Pulsar Gallery -- Recent Results";
    $this->class = (isset($_GET["class"])) ? $_GET["class"] : "new";
    if ($this->class == "old")
    {
      $this->results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
      $this->archive_dir = $this->cfg["SERVER_OLD_ARCHIVE_DIR"];
      $this->results_link = "/mopsr/old_results";
      $this->results_title = "Archived Results";
    }
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      .processing {
        background-color: #FFFFFF;
        padding-right: 10px;
      }

      .finished {
        background-color: #cae2ff;
      }

      .transferred {
        background-color: #caffe2;
      }
    </style>

    <script type='text/javascript'>

      // If a page reload is required
      function changeLength() {

        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;

        var show_inline;
        if (document.getElementById("inline_images").checked)
          show_inline = "true";
        else
          show_inline = "false";

        var url = "results.lib.php?single=true&length="+length+"&inline_images="+show_inline;

        i = document.getElementById("filter_type").selectedIndex;
        var filter_type = document.getElementById("filter_type").options[i].value;

        var filter_value = document.getElementById("filter_value").value;

        if ((filter_value != "") && (filter_type != "")) {
          url = url + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }
        document.location = url;
      }

      function toggle_images()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        var img;
        var show_inline

        if (document.getElementById("inline_images").checked) {
          show_inline = true;
          document.getElementById("IMAGE_TR").innerHTML = "IMAGE";
        } else {
          show_inline = false;
          document.getElementById("IMAGE_TR").innerHTML = "";
        }

        for (i=0; i<length; i++) {
          img = document.getElementById("img_"+i);
          if (show_inline) 
          {
            img.style.display = "";
            img.className = "processing";
          }
          else
          {
            img.style.display = "none";
          }
        }
      }
    </script>
<?
  }

  function printJavaScriptBody()
  {
?>
    <!--  Load tooltip module for images as tooltips, hawt -->
    <script type="text/javascript" src="/js/wz_tooltip.js"></script>
<?
  }

  function printHTML() 
  {
?>

    <table cellpadding="10px" width="100%">

      <tr>
        <td width='210px' height='60px'><img src='/mopsr/images/mopsr_logo.png' width='200px' height='60px'></td>
        <td align=left><font size='+2'><?echo $this->results_title?></font></td>
      </tr>

      <tr>
        <td valign="top" width="200px">
<?
    $this->openBlockHeader("Summary");
?>
    <table>
<?php

$pdo = new PDO ('sqlite:/home/dada/linux_64/web/mopsr/asteria.db');

$q = 'SELECT date FROM Updates';

$stmt = $pdo -> query($q);
if (!$stmt) {
  echo "Failed to query:<br>".$q;
  exit(-1);
}

$updated = $stmt ->fetch();

$q = 'SELECT utc FROM UTCs ORDER BY utc LIMIT 1';
$stmt = $pdo -> query($q);
if (!$stmt) {
  echo "Failed to query:<br>".$q;
  exit(-1);
}

$since = $stmt ->fetch();

$q = 'SELECT COUNT(*) FROM Observations';
$stmt = $pdo -> query($q);
if (!$stmt) {
  echo "Failed to query:<br>".$q;
  exit(-1);
}

$count = $stmt ->fetch();

echo "<tr><td>Data since ".substr($since[0], 0, 10)."<td><tr>\n";
echo "<tr><td>".$count[0]." observations</td></tr>\n";
echo "<tr><td>Updated at:<br><span class=best_snr>".$updated[0]."</span></td></tr>\n";
?>
      <tr>
        <td colspan=2><a href="/mopsr/Asteria.php?single=true">Last 100 pulsar</a></td>
      </tr>
      <tr><td><hr/></td><tr>
<?php
  $summary_filename = "/home/dada/linux_64/web/mopsr/latest_summary";
  $summary_contents = file_get_contents($summary_filename);
  $summary_array = explode(PHP_EOL, $summary_contents);
  $counter = 0;
  foreach($summary_array as $line) {
    if ($counter == 0) {
      $updated = $line;
    } elseif ($counter == 1) {
      echo "<tr><td><span class=best_snr>SUMMARY OF LAST 24hrs</span></td></tr>";
      echo "<tr><td><h5>As of:<br>".$updated."</h5></td></tr>";
    } elseif (strpos($line, "FRB Statistics") === 0) {
      echo "<tr><td><hr></td></tr>";
      echo "<tr><td><span class=best_snr>".$line.'</span></td></tr>';
    }else {
      echo "<tr><td>".$line.'</td></tr>';
    }
    $counter += 1;
  }
?>
    </table>
  <?
    $this->closeBlockHeader();

    echo "</td><td>\n";

    $this->openBlockHeader("Best S/N for all pulsars observed ");
?>

<p>
<h3>Please choose SNR range</h3>

<form action="" method="post">
<select name="snr_cut" onchange="this.form.submit()">
<option value="">SNR cut</option>
<option value=">= 10">&#62;= 10</option>
<option value="< 10">&#60; 10</option>
</select>
</form>

</p>


<?php
function rescale_snr_to5min($fSNR, $ftint_m) {
  return $fSNR * sqrt(5./$ftint_m);
}


if ($_REQUEST['snr_cut'] ) {

echo '<p><h2>Displaying data with SNR '.$_REQUEST['snr_cut'].'</h2><br></p>';
  $q = 'SELECT name, dm, period, max_snr_in5min, utc, snr, tint/60. as tint FROM (Pulsars JOIN UTCs JOIN Observations ON Pulsars.id = Observations.psr_id AND UTCs.id = Pulsars.max_snr_obs_id AND Observations.utc_id = UTCs.id) WHERE tint > 1.0 AND max_snr_in5min '.$_REQUEST['snr_cut'].' ORDER BY name ASC';

  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  } else {
    echo "<p><table>\n<tr>\n";
  }

  $results = $stmt->fetchAll(PDO::FETCH_NUM);
  echo "<p><h3>Found ".count($results)." detection in the specified S/N range</p2></h3><br>\n";

  $counter = 0;
  $top_dir = "/data/mopsr/results/";
  $alt_top_dir = "/data/mopsr/old_results/";
  try {
    #while ($row = $stmt -> fetch())
    foreach ($results as $row)
    {
      $counter = $counter + 1;
      $pulsar = $row[0];
      $dm = $row[1];
      $period = $row[2]*1000;
      $max_snr = $row[3];
      $utc = $row[4];
      $snr = $row[5];
      $tint_m = $row[6];

      $class = "&class=new";
      $_result_dir = glob($top_dir.$utc);
      if (empty($_result_dir)) {
        $class = "&class=old";
      }

      $fl_hr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".fl.1024x768.png"));
      if ($fl_hr[0] == "") {
        $fl_hr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".fl.1024x768.png"));
      }
      $fl_lr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".fl.120x90.png"));
      if ($fl_lr[0] == "") {
        $fl_lr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".fl.1024x768.png"));
      }
      if ($fl_lr[0] == "") {
        $fl_lr[0] = $fl_hr[0];
      }

      $fr_hr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".fr.1024x768.png"));
      if ($fr_hr[0] == "") {
        $fr_hr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".fr.1024x768.png"));
      }
      $fr_lr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".fr.120x90.png"));
      if ($fr_lr[0] == "") {
        $fr_lr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".fr.1024x768.png"));
      }
      if ($fr_lr[0] == "") {
        $fr_lr[0] = $fr_hr[0];
      }

      $ti_hr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".ti.1024x768.png"));
      if ($ti_hr[0] == "") {
        $ti_hr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".ti.1024x768.png"));
      }
      $ti_lr = str_replace("/data", "", glob($top_dir.$utc."/20*".$psr.".ti.120x90.png"));
      if ($ti_lr[0] == "") {
        $ti_lr = str_replace("/data", "", glob($alt_top_dir.$utc."/20*".$psr.".ti.1024x768.png"));
      }
      if ($ti_lr[0] == "") {
        $ti_lr[0] = $ti_hr[0];
      }



      echo "<td width=300><a href=/mopsr/results.lib.php?single=true&offset=0&length=20&inline_images=true&filter_type=SOURCE&filter_value=";
      echo $pulsar.">".$pulsar."</a><br>DM : ".round($dm, 2)."<br>period : ".round($period,2 )." ms <br>\n";
      echo '<a href='.$fl_hr[0].'><img src='.$fl_lr[0].' width="120" height="90"></a><br>'."\n";
      echo '<a href="'.$fr_hr[0].'"><img src="'.$fr_lr[0].'" width="120" height="90"></a><br>'."\n";
      echo '<a href="'.$ti_hr[0].'"><img src="'.$ti_lr[0].'" width="120" height="90"></a><br>'."\n";
      echo "SNR = ".round($snr, 2)."<br>t = ".round($tint_m, 2)." minutes<br>SNR(5min) = ".round(rescale_snr_to5min($snr, $tint_m), 2)."\n";
      echo "<br><a href=/mopsr/result.lib.php?single=true".$class."&utc_start=".$utc.">".$utc."</a></td>\n";
      if ($counter %5 == 0) {
        echo "</tr><tr><td>&nbsp;</td></tr>";
      }
    }
  } catch (Exception $e){echo $e->getMessage();}

} else {
  # generate an empty table for nicer formatting of the page:
  echo "<p><table>\n<tr>\n";
  for ($i=0; $i<60; $i++) {
    echo "<td width=300>&nbsp;</td></tr>";
    if ($i %5 == 0) {
      echo "</tr><tr><td>&nbsp;</td></tr>";
    }
  }
}
?>

<?
    $this->closeBlockHeader();

    echo "</td></tr></table>\n";
  }

  function tippedimage($i, $url, $image, $color, $text)
  {
    $mousein = "onmouseover=\"Tip('<img src=\'".$this->results_link."/".$image."\' width=201 height=151>')\"";
    $mouseout = "onmouseout=\"UnTip()\"";
    $link = "<a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout."><font color='".$color."'>".$text."</font></a>&nbsp;&nbsp;";
    return $link;
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/
  function printUpdateHTML($get)
  {
    $results = $this->getResultsArray($this->results_dir,
                                      $this->offset, $this->length, 
                                      $this->filter_type, $this->filter_value);
    $keys = array_keys($results);
    rsort($keys);

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<results>\n";
    foreach ($results as $utc => $array)
    {
      $xml .= "<result>\n";
      $xml .= "<UTC_START>".$utc."</UTC_START>\n";
      foreach ($array as $k => $v)
      {
        if ($k == "SOURCE")
        {
          // ignore
        }
        else if ($k == "SOURCES")
        {
          foreach ($v as $source => $vv)
          {
            $xml .= "<SOURCE name='".$source."'>\n";
            if (is_array($vv))
            {
              foreach ($vv as $kkk => $vvv)
              {
                if ($kkk == "IMAGE")
                {
                  $xml .= $vvv;
                }
              }
            }
            $xml .= "</SOURCE>";
          }
        }
        else
        {
          $xml .= "<".$k.">".htmlspecialchars($v)."</".$k.">\n";
        }
      }
      $xml .= "</result>\n";
    }
    $xml .= "</results>\n";

    header('Content-type: text/xml');
    echo $xml;
  }

  function handleRequest()
  {
    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }

  }

  function getResultsArray($results_dir, $offset=0, $length=0, $filter_type, $filter_value) 
  {
    $all_results = array();

    $observations = array();
    $dir = $results_dir;

    if (($filter_type == "") || ($filter_value == "")) 
    {
      $observations = getSubDirs ($results_dir, $offset, $length, 1);
    } 
    else 
    {
      # get a complete list
      if ($filter_type == "UTC_START") {
        $cmd = "find ".$results_dir."/*".$filter_value."* -maxdepth 1 ".
               "-name 'obs.info' -printf '%h\n' | awk -F/ '{print \$NF}' | sort -r";
      } else {
        $cmd = "find ".$results_dir." -maxdepth 2 -type f -name obs.info ".
               "| xargs grep ".$filter_type." | grep ".$filter_value." ".
               "| awk -F/ '{print $(NF-1)}' | sort -r";
      }
      $last = exec($cmd, $all_obs, $rval);
      $observations = array_slice($all_obs, $offset, $length);
    }

    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i];
      $dir = $results_dir."/".$o;

      // read the obs.info file into an array 
      if (file_exists($dir."/obs.info")) 
      {
        $arr = getConfigFile($dir."/obs.info");
        $all = array();

        $all["STATE"] = "unknown";
        if (file_exists($dir."/obs.processing"))
          $all["STATE"] = "processing";
        else if (file_exists($dir."/obs.finished"))
          $all["STATE"] = "finished";
        else if (file_exists($dir."/obs.transferred"))
          $all["STATE"] = "transferred";
        else if (file_exists($dir."/obs.failed"))
          $all["STATE"] = "failed";
        else
           $all["STATE"] = "unknown";

        $all["SOURCE"] = $arr["SOURCE"];

        $all["SOURCES"] = array();
        if ($arr["FB_ENABLED"] == "true")
        {
          $all["SOURCES"]["FB"] = array();
          $all["SOURCES"]["FB"]["TYPE"] = "FB";
          $all["SOURCES"]["FB"]["IMAGE"] = $this->getFBImage($dir, $o, $arr["FB_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["FB_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "FB_IMG", $all["SOURCES"]["FB"]["IMAGE"]);
        }

        if ($arr["MB_ENABLED"] == "true")
        {
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"]["IMAGE"] = "../../../images/blankimage.gif";
        }

        if ($arr["CORR_ENABLED"] == "true")
        {
          $source = $arr["SOURCE"];
          $all["SOURCES"][$source] = array();
          $all["SOURCES"][$source]["TYPE"] = "CORR";
          $all["SOURCES"][$source]["IMAGE"] = $this->getCorrImage($dir, $o, $arr["CORR_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["CORR_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "CORR_IMG", $all["SOURCES"][$source]["IMAGE"]);
        }

        for ($j=0; $j<4; $j++)
        {
          $tbe_key = "TB".$j."_ENABLED";
          if ((array_key_exists ($tbe_key, $arr)) && ($arr[$tbe_key] == "true"))
          {
            $source = $arr["TB".$j."_SOURCE"];
            $all["SOURCES"][$source] = array();
            $all["SOURCES"][$source]["TYPE"] = "TB";
            $all["SOURCES"][$source]["IMAGE"] = $this->getTBImage($dir, $o, $source, $arr["TB".$j."_IMG"]);
            if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["TB".$j."_IMG"] == ""))
              $this->updateImage ($dir."/obs.info", "TB".$j."_IMG", $all["SOURCES"][$source]["IMAGE"]);
          }
        }

        # use the primary PID
        $all["PID"] = $arr["PID"];

        $all["IMG"] = "NA";
        # find an image of the observation, if not existing
        if (($arr["IMG"] == "NA") || ($arr["IMG"] == ""))
        {
          # preferentially find a pulsar profile plot
          $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
            "-name '*.fl.120x90.png' -printf '%f\n' ".
            "| sort -n | head -n 1";
          $img = exec ($cmd, $output, $rval);

          if (($rval == 0) && ($img != "")) {
            $all["IMG"] = $img;
          } else {
            $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
              "-name '*.*.ad.160x120.png' -printf '%f\n' ".
              "-o -name '*.FB.00.*png' -printf '%f\n' ".
              "| sort -n | head -n 1";
            $img = exec ($cmd, $output, $rval);

            if (($rval == 0) && ($img != "")) {
              $all["IMG"] = $img;
            } else {
              $all["IMG"] = "../../../images/blankimage.gif";
            }
          }
        }
        else
        {
          $all["IMG"] = $arr["IMG"];
        }
      
        # if the integration length does not yet exist
        $int = 0;
        if (($arr["INT"] == "NA") || ($arr["INT"] <= 0))
        {
          if ($arr["CORR_ENABLED"] == "true")
            $int = $this->calcIntLengthCorr($o, $arr["SOURCE"]);
          else if ($arr["FB_ENABLED"] == "true")
            $int = $this->calcIntLengthFB($o, "FB");
          else if ($arr["TB0_ENABLED"] == "true")
            $int = $this->calcIntLengthTB($o, $arr["TB0_SOURCE"]);
          else
            $int = "0";
          $all["INT"] = $int;
        }
        else
          $all["INT"] = $arr["INT"];

        # if the observation is 
        if (($all["STATE"] == "finished") || ($all["STATE"] == "transferred"))
        {
          if (($arr["INT"] == "NA") || ($arr["INT"] <= 0) && ($all["INT"] > 0))
          {
            system("perl -ni -e 'print unless /^INT/' ".$results_dir."/".$o."/obs.info");
            system("echo 'INT              ".$int."' >> ".$results_dir."/".$o."/obs.info");
          }
        }
      }

      if (file_exists($dir."/obs.txt")) {
        $all["ANNOTATION"] = file_get_contents($dir."/obs.txt");
      } else {
        $all["ANNOATATION"] = "";
      }

      $all_results[$o] = $all;
    }

    return $all_results;
  }

  function calcIntLengthCorr($utc_start, $source)
  {
    $cc_file = $this->results_dir."/".$utc_start."/".$source."/cc.sum"; 
    if (file_exists ($cc_file))
    {
      $cmd = "find ".$this->archive_dir."/".$utc_start."/".$source." -name '*.ac' | sort -n | tail -n 1";
      $output = array();
      $ac = exec($cmd, $output, $rval);

      $parts = explode("_", $ac);
      $time_to = $parts[count($parts)-1];

      #$cmd = "grep BYTES_PER_SECOND ".$this->results_dir."/".$utc_start."/".$source."/obs.header | awk '{print $2}'";
      #$output = array();
      #$Bps = exec($cmd, $output, $rval);

      $length = $time_to;
      return sprintf ("%5.0f", $length);
    }
    return 0;
  }

  function calcIntLengthFB ($utc_start, $source)
  {
    $ac_file = $this->results_dir."/".$utc_start."/".$source."/all_candidates.dat";
    if (file_exists($ac_file))
    {
      $cmd = "tail -n 1000 ".$ac_file." | awk '{print $3}' | sort -n | tail -n 1";
      $length = exec($cmd, $output, $rval);
      return sprintf("%5.0f", $length); 
    }
    return 0;
  }

  function calcIntLengthTB($utc_start, $source) 
  {
    $dir = $this->results_dir."/".$utc_start."/".$source." ".
           $this->archive_dir."/".$utc_start."/".$source;
    $length = 0;

    # try to find a TB/*_f.tot file
    $cmd = "find ".$this->results_dir."/".$utc_start."/".$source." -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' | sort -n | tail -n 1";
    $tot = exec($cmd, $output, $rval);
    if ($tot != "")
    {
      $cmd = $this->cfg["SCRIPTS_DIR"]."/psredit -Q -c length ".$tot;
      $output = array();
      $result = exec($cmd, $output, $rval);
      list ($file, $length) = split(" ", $result);
      if ($length != "" && $rval == 0)
        return sprintf ("%5.0f",$length);
    }
    
    # try to find a 2*.ar file
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name '2*.ar' -printf '%f\n' | sort -n | tail -n 1";
    $ar = exec($cmd, $output, $rval);
    if ($ar != "")
    {
      $array = split("\.",$ar);
      $ar_time_str = $array[0];

      # if image is pvf, then it is a local time, convert to unix time
      $ar_time_unix = unixTimeFromGMTime($ar_time_str);
      
      # add ten as the 10 second image file has a UTC referring to the first byte of the file 
      $length = $ar_time_unix - unixTimeFromGMTime($utc_start);
    }

    return $length;
  }

  function getFBImage($dir, $o, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
             " -name '*.FB.00.*png' -printf '%f\n' ".
              "| sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function getCorrImage ($dir, $o, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
             " -name '*.*.ad.160x120.png' -printf '%f\n' ".
              "| sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function getTBImage ($dir, $o, $s, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f".
             " -name '*.".$s.".fl.120x90.png' -printf '%f\n'".
             " | sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function updateImage($file, $key, $value)
  {
    system("perl -ni -e 'print unless /^".$key."/' ".$file);
    system("echo '".$key."            ".$value."' >> ".$file);
  }

}
handledirect("asteria_bests");
