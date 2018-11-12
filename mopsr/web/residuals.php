<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");
include_once("Asteria.lib.php");

class residuals extends mopsr_webpage 
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
  var $old_results_dir;
  var $results_link;
  var $archive_dir;
  var $resid_type;

  function residuals()
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
    $this->old_results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
    $this->archive_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $this->results_link = "/mopsr/results";
    $this->results_title = "Timing programme residuals";
    $this->class = (isset($_GET["class"])) ? $_GET["class"] : "new";
    if ($this->class == "old")
    {
      $this->results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
      $this->archive_dir = $this->cfg["SERVER_OLD_ARCHIVE_DIR"];
      $this->results_link = "/mopsr/old_results";
      $this->results_title = "Archived Results";
    }
    $this->resid_type = (isset($_GET["type"])) ? $_GET["type"] : "tc";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

    function vote(id, psr_id, voter_id, vote) {
      img_id = "img_" + id + "_" + vote;
      console.log("vote: img id:" + img_id);
      console.log("psr_id: " + psr_id);
      console.log("voter_id: " + voter_id);
      console.log("vote: " + vote);
      current_src = document.getElementById(img_id).src;
      console.log("vote: " + psr_id + " " + voter_id + " " + vote);
      var found = current_src.search("Green");
      if (found == -1) {
        var action = "insert";
        console.log("img_"+id + "_" + vote);
        document.getElementById("img_"+id + "_" + vote).src = current_src.replace(/.png/gi, "Green.png");
        var other_vote = vote == 0 ? 1 : 0;
        console.log("img_"+id + "_" + other_vote);
        console.log(document.getElementById("img_"+id + "_" + other_vote).src);
        document.getElementById("img_"+id + "_" + other_vote).src = document.getElementById("img_"+id + "_" + other_vote).src.replace(/Green.png/gi, ".png");
        console.log(document.getElementById("img_"+id + "_" + other_vote).src);
      } else {
        var action = "cancel";
        document.getElementById("img_"+id + "_" + vote).src = current_src.replace(/Green.png/gi, ".png");
      }
      $.ajax({
        type: "POST",
          url: "vote.lib.php",
          data: {action:action, voter_id:voter_id, psr_id:psr_id, vote:vote},
          dataType: "JSON",
          success: function(data) { $("#message").html(data); },
          error: function(err) {
            alert(err);
          }
      });
    }

    function edit_science_case(id, psr_id) {
      var scase_id = "science_case_"+id;
      var scase_el = document.getElementById(scase_id);
      var scase = scase_el.value;
      $.ajax({
        type: "POST",
          url: 'vote.lib.php',
          data: {action: "science", scase: scase, psr_id : psr_id}
      });
    }

    function edit_cadence(id, psr_id) {
      var cadence_id = "cadence_"+id;
      var cadence = document.getElementById(cadence_id).value;
      console.log("Got cadence ");
      console.log(cadence);

      $.ajax({
        type: "POST",
        url: 'vote.lib.php',
        data: {action: "cadence", cadence: cadence, psr_id : psr_id}
      });
    }

    function popWindow(URL) {
      day = new Date();
      id = day.getTime();
      eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,"+
           "scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width=640,height=520');");
    } 

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
<style>
th.tablesorter {
  font: bold 11px "Trebuchet MS", Verdana, Arial, Helvetica,
  sans-serif;
  color: #0F4D0D;
  border-right: 1px solid #C1DAD7;
  border-bottom: 1px solid #C1DAD7;
  border-top: 1px solid #C1DAD7;
  letter-spacing: 2px;
  text-transform: uppercase;
  text-align: center;
  padding: 6px 6px 6px 12px;
  background: #99B090;
  margin: 0px auto;
}

th.nobg {
  border-top: 0;
  border-left: 0;
  border-right: 1px solid #C1DAD7;
  background: none;
}
tr.even td {
  border-right: 1px solid #C1DAD7;
  border-bottom: 1px solid #C1DAD7;
  padding: 6px 6px 6px 12px;
  color: #0F4D0D;
  background:white;
  font-size: large;
  }

tr.odd td {
  /* orig?:
 *   background:#f7fbff*/
  /* orig?:
    *   background: #CAE8EA */
    /* dark:
      *   background: #A8C6C8 */
    /* light:*/
  border-right: 1px solid #C1DAD7;
  border-bottom: 1px solid #C1DAD7;
  padding: 6px 6px 6px 12px;
  color: #0F4D0D;
  background: #99B090;
  font-size: large;
}

a:link, a:visited {
  color: #0F4D0D;
}

tr.alarm td {
  border-right: 1px solid #C1DAD7;
  border-bottom: 1px solid #C1DAD7;
  padding: 6px 6px 6px 12px;
  color: #000000;
  background: #90a2b0;
  font-size: large;
}
.alarm a:link, .alarm a:visited {
  color: #000000;
}
</style>
<script src="./js/jquery-3.3.1.min.js"></script>
<script src="./js/jquery.tablesorter.min.js"></script>
<script src="./js/jquery.tablesorter.widgets.js"></script>
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
<?php

    include MYSQL_DB_CONFIG_FILE;
    $pdo = new PDO ('mysql:dbname='.MYSQL_DB.';host='.MYSQL_HOST, MYSQL_USER, MYSQL_PWD);

    echo '<div class="sticky">';
    $this->openBlockHeader("Summary");
    print_summary($pdo, get_class($this));
    $this->closeBlockHeader();
    echo "</div>";

    echo "</td><td>\n";

    $this->openBlockHeader("Timing Programme Pulsars");

function rescale_snr_to5min($fSNR, $ftint_m) {
  return $fSNR * sqrt(5./$ftint_m);
}

$my_votes = array();


  $q = 'SELECT psr_id, vote FROM Cadence_votes WHERE voter_id ='.$_GET['inspector_id'];
  $stmt = $pdo->prepare($q);
  $stmt->execute();
  $results = $stmt->fetchall();
  foreach ($results as $row) {
    $my_votes[$row[0]] = $row[1];
  }


  $observe_clause = "";
  $observe_clause_and = "WHERE";
  if (($_GET['observe']) == 0) {
    $observe_clause = "WHERE observe = 0";
    $observe_clause_and = "WHERE observe = 0 AND";
  } else if (($_GET['observe']) == 1) {
    $observe_clause = "WHERE observe = 1";
    $observe_clause_and = "WHERE observe = 1 AND";
  }

  $q = 'SELECT name FROM Pulsars '.$observe_clause.' ORDER BY name';
  $stmt = $pdo -> query($q);
  if (!$stmt) {
    echo 'Failed to query:<br>'.$q;
    exit(-1);
  }
  $q = 'SELECT name FROM Pulsars WHERE observe=1 ORDER BY name';
  $stmt = $pdo -> query($q);
  if (!$stmt) {
    echo 'Failed to query:<br>'.$q;
    exit(-1);
  }
  $psr500 = $stmt->fetchAll(PDO::FETCH_COLUMN, 0);

  $counter_7 = 0;
  $counter_7_detected = 0;
  $counter_30 = 0;
  $counter_30_detected = 0;
  $counter_superold = 0;
  $counter_superold_detected = 0;
  $counter_never_observed = 0;
  $counter_never_detected = 0;
?>
<?
  $timezone = new DateTimeZone('UTC');
  $date_now = date_create("now", $timezone);

  $utcs = array();
  $utcs_detected = array();
  $days = array();
  $days_detected = array();
  $detections_count_ever = array();
  $observed_count_ever = array();
  $observed_count_last_Ndays = array();
  $observing = array();
  $science_cases = array();
  $psr_ids = array();
  $periods = array();
  $cadences = array();

  $q = 'SELECT Pulsars.name, Pulsars.science_case, Pulsars.observe, Pulsars.id, MAX(UTCs.utc), Pulsars.period, Pulsars.desired_cadence FROM TB_Obs LEFT JOIN UTCs ON (TB_Obs.utc_id = UTCs.id) LEFT JOIN Pulsars ON TB_Obs.psr_id = Pulsars.id GROUP BY Pulsars.name';
  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  } 
  $results = $stmt->fetchAll(PDO::FETCH_NUM);

  try {
    foreach ($results as $row) {
      $date_last = DateTime::createFromFormat("Y-m-d-H:i:s", $row[4], $timezone);
      if (gettype($date_last) == "object") {
        $date_diff = $date_last->diff($date_now);
        $date_diff_int = intval($date_diff->format("%a"));
        if ( $date_diff_int <=7)
          $counter_7++;
        elseif ($date_diff_int <=30)
          $counter_30++;
        else
          $counter_superold++;

        $utcs[$row[0]] = $row[4];
        $days[$row[0]] = $date_diff_int;
      } else {
        $counter_never_observed++;
        $utcs[$row[0]] = "Never observed";
        $days[$row[0]] = 10000;
      }
      $science_cases[$row[0]] = $row[1];
      $observing[$row[0]] = $row[2];
      $psr_ids[$row[0]] = $row[3];
      $periods[$row[0]] = $row[5];
      $cadences[$row[0]] = $row[6];
    }
  } catch (Exception $e){echo $e->getMessage();};
  # print_r($utcs);

  $q = 'SELECT Pulsars.name, max(UTCs.utc) from TB_Obs left join UTCs on (TB_Obs.utc_id = UTCs.id) left join Pulsars on TB_Obs.psr_id = Pulsars.id  '.$observe_clause_and.' TB_Obs.snr > 10 GROUP BY Pulsars.name';
  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  } 
  $utcs_result = $stmt->fetchAll(PDO::FETCH_NUM);

  try {
    foreach($utcs_result as $utc) {
      $date_last_detected = DateTime::createFromFormat("Y-m-d-H:i:s", $utc[1], $timezone);
      if (gettype($date_last_detected) == "object") {
        $date_diff = $date_last_detected->diff($date_now);
        $date_diff_int = intval($date_diff->format("%a"));
        if ( $date_diff_int <=7)
          $counter_7_detected++;
        elseif ($date_diff_int <=30)
          $counter_30_detected++;
        else
          $counter_superold_detected++;

        $utcs_detected[$utc[0]] = $utc[1];
        $days_detected[$utc[0]] = $date_diff_int;
      }
      else {
        $utcs_detected[$utc[0]] = "Never, inspect";
        $days_detected[$utc[0]] = 10000;
        $counter_never_detected++;
      }
    }
  } catch (Exception $e){
    echo $e->getMessage();
  };

  $q = 'SELECT Pulsars.name, count(*) from (TB_Obs JOIN Pulsars ON Pulsars.id=TB_Obs.psr_id) '.$observe_clause_and.' TB_Obs.snr > 10 GROUP BY Pulsars.name;';
  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $results = $stmt->fetchAll(PDO::FETCH_NUM);

  try {
    foreach($results as $row) {
      $detections_count_ever[$row[0]] = $row[1];
    }
  } catch (Exception $e){
    echo $e->getMessage();
  }

  $q = 'SELECT Pulsars.name, count(*) from (TB_Obs JOIN Pulsars ON Pulsars.id=TB_Obs.psr_id) '.$observe_clause.' GROUP BY Pulsars.name;';
  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  } 
  $results = $stmt->fetchAll(PDO::FETCH_NUM);

  try {
    foreach($results as $row) {
      #array_push($observed_count_ever, $results[0][0]);
      $observed_count_ever[$row[0]] = $row[1];
    }
  } catch (Exception $e){
    echo $e->getMessage();
  };

  if ($_GET['days_for_table'] ) {
    $days_for_table = $_GET['days_for_table'];
  } else {
    $days_for_table = 10;
  }

  $q = 'SELECT Pulsars.name, COUNT(*) from TB_Obs LEFT JOIN UTCs ON (TB_Obs.utc_id = UTCs.id) LEFT JOIN Pulsars ON TB_Obs.psr_id = Pulsars.id  '.$observe_clause.' AND TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) < '.$days_for_table.'*24*60 AND TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) > 0 GROUP BY Pulsars.name;';

  $stmt = $pdo -> query($q);

  if (!$stmt)
  {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $results = $stmt -> fetchAll(PDO::FETCH_NUM);

  try {
    foreach($results as $row) {
      $observed_count_last_Ndays[$row[0]] = $row[1];
    }
  } catch (Exception $e) {
    echo $e->getMessage();
  }

  $number_of_psrs = count($psr500);
//  echo '<h3>Number of unique pulsars detected (observed) within last 7 days: '.$counter_7_detected.' ('.$counter_7.')</h3>';
//  echo '<h3>Number of unique pulsars detected (observed) between last 7 and 30 days: '.$counter_30_detected.' ('.$counter_30.')</h3>';
//  echo '<h3>Number of unique pulsars not observed in the last month: '.$counter_superold.'</h3>';
//  echo '<h3>Number of unique pulsars observed but not detected in the last month: '.$counter_superold_detected.'</h3>';
//  $counter_never_detected = $number_of_psrs - $counter_7_detected - $counter_30_detected - $counter_superold_detected;
//  echo '<h3>Number of unique pulsars never detected: '.$counter_never_detected.'</h3>';
//  $counter_never_observed = $number_of_psrs - $counter_7 - $counter_30 - $counter_superold;
//  echo '<h3>Number of unique pulsars never observed: '.$counter_never_observed.'</h3>';
//  echo '<h4>All numbers are relative to '.$number_of_psrs.' from a curated list.</h4>';
//  echo "Note that currently detections (SN>10) are based on S/N without much RFI cleaning<br>";
//  echo '<b>Everything with <span style="background: #90a2b0">blue-ish</span> background below is 10 days since observation or more</b>';
?>

<table id="psrs" class="tablesorter">
<thead>
<tr>
<th class="tablesorter" style="width:18%">Residuals</th>
<th class="tablesorter" style="width:18%">Residuals</th>
<th class="tablesorter" style="width:18%">Residuals</th>
</tr>
</thead>
<tbody>
<?

  $counter = 0;
  foreach ($psr500 as $psr) {
    //print "BAR ".$psr."<br>";
    $voted_on_this = array_key_exists($psr_ids[$psr], $my_votes);
    if (($_GET['voted'] == 0 && $voted_on_this) || ($_GET['voted'] == 1 && !$voted_on_this) || ($_GET['voted'] == 1 && $_GET['voted_filter'] == 1 && $my_votes[$psr_ids[$psr]] == 0)  || ($_GET['voted'] == 1 && $_GET['voted_filter'] == 0 && $my_votes[$psr_ids[$psr]] == 1 )) {
      continue;
    }
    try {
      if ($counter>0 && $counter%3 == 0)
        echo '<tr>';
      # Residuals
      $tc = "";
      $tc_big = "";
      $results_link = "";
      if (file_exists($this->results_dir."/".$utcs[$psr])) {
        $tc = glob($this->results_dir."/".$utcs[$psr]."/2*".$psr.".".$this->resid_type.".120x90.png");
        $tc_big = glob($this->results_dir."/".$utcs[$psr]."/2*".$psr.".".$this->resid_type.".1024x768.png");
        $results_link = "/mopsr/result.lib.php?single=true&class=new&utc_start=".$utcs[$psr];
      } else {
        $tc = glob($this->old_results_dir."/".$utcs[$psr]."/2*".$psr.".".$this->resid_type.".120x90.png");
        $tc_big = glob($this->old_results_dir."/".$utcs[$psr]."/2*".$psr.".".$this->resid_type.".1024x768.png");
        $results_link = "/mopsr/result.lib.php?single=true&class=old&utc_start=".$utcs[$psr];
      }
      $tc_big = str_replace("/data/mopsr/", "", $tc_big);
      $tc_big = $tc_big[count($tc_big) - 1];
      $tc = str_replace("/data/mopsr/", "", $tc);
      $tc = $tc[count($tc)-1];
      # echo '<td style="width:18%"><a href="'.$tc_big.'"><img src="'.$tc.'" alt="Not found" width="100%"></a></td>';
      echo '<td><a href="'.$results_link.'"><img src="'.$tc.'" alt="Not found" width="100%"></a></td>';

      # Vote
      # if (array_key_exists($psr_ids[$psr], $my_votes));
      if ($cadences[$psr] === NULL) {
        $cadence = -1;
      } else {
        $cadence = $cadences[$psr];
      }
      echo '<input type="hidden" name="single" value="true"></form></td>';
      # echo '<input type="hidden" name="single" value="true"><button type="button" ';
      # echo ' onclick="edit_cadence(\''.$counter.'\', \''.$psr_ids[$psr].'\')">Submit</button></form></td>'; echo '</tr>';
    } catch (Exception $e){echo $e->getMessage();};
    $counter = $counter + 1;
    if ($counter>0 && $counter%3 == 0)
      echo '</tr>';
  }
  
    $this->closeBlockHeader();

    echo "</tbody></table>\n";
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
handledirect("residuals");
