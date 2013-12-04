<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class result extends caspsr_webpage 
{

  var $utc_start = "";
  var $obs_results_dir = "";
  var $inst = "";
  var $imgs;
  var $obs_info = array();

  function result($utc_start)
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();
    $this->utc_start = $utc_start;
    $this->obs_results_dir = $this->inst->config["SERVER_OLD_RESULTS_DIR"]."/".$this->utc_start;
    $this->imgs = $this->inst->getObsImages($this->obs_results_dir);
    $this->obs_info = $this->inst->configFileToHash($this->obs_results_dir."/obs.info");
  }

  function javaScriptCallback()
  {
  }

  function printJavaScriptHead()
  {
?>
  <script type="text/javascript">

    window.name = "resultwindow";
  
    function popWindow(URL) {
      day = new Date();
      id = day.getTime();
      eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,"+
           "scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width=640,height=520');");
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

    $this->openBlockHeader("Observation Summary");

    if ($this->utc_start == "") {
      echo "<p>ERROR: No UTC_START specified</p>\n";
      $this->closeBlockHeader();
      return 0;
    }

    /* Summary of the observation */
    $most_recent  = $this->getMostRecentResult();
    $results_link = "<a href='/caspsr/old_results/".$this->utc_start."/'>link</a>";
    $obs_state    = "old";

?>
    <table>
      <tr><td>UTC </td><td><?echo $this->utc_start?></td></tr>
      <tr><td>Local</td><td><? echo localTimeFromGmTime($this->utc_start)?></td></tr>
      <tr><td>State</td><td><? echo ucfirst($obs_state)?></td></tr>
      <tr><td>Age</td><td><? echo makeTimeString($most_recent)?></td></tr>
      <tr><td>Results</td><td><a href='/caspsr/old_results/<? echo $this->utc_start?>/'>Link</a></td></tr>
      <tr><td>Length</td><td><?echo $this->obs_info["INT"]?> s</td></tr>
      <tr><td>SNR</td><td><?echo $this->obs_info["SNR"]?></td></tr>
    </table>
<?

    $this->closeBlockHeader();

/*
    $this->openBlockHeader("Source Summary");

    $vals = array("dm", "p0", "int", "snr");
    $names = array("DM", "P0", "Integrated", "SNR");
    $psrs = array_keys($this->imgs);

    echo "<table>\n";
    for ($i=0; $i<count($psrs); $i++) 
    {
      $psr = $psrs[$i];
      $row = $this->source_info[$psr];
      $keys = array_keys($row);
      echo "        <tr>\n";
      echo "          <td align=right>Source</td>\n";
      echo "          <td align=left>".$psr."</td>\n";
      echo "        </tr>\n"; 

      for ($j=0; $j<count($vals); $j++) 
      {
        $k = $keys[$j];
        $n = $names[$j];
        $v = $row[$vals[$j]];
        echo "        <tr>\n";
        echo "          <td align=right>".$n."</td>\n";
        echo "          <td align=left>".$v."</td>\n";
        echo "        </tr>\n";
      }
    }
    echo "</table>\n";

    $this->closeBlockHeader();
*/

    /* Actions */
/*
    $this->openBlockHeader("Actions");

    $process_url = "/caspsr/process_obs.lib.php?single=true&utc_start=".$this->utc_start;
    $process_url = str_replace("+", "%2B", $process_url);

    $custom_url = "/caspsr/custom_plot.lib.php?basedir=".$this->obs_results_dir."&utc_start=".$this->utc_start;

    $delete = " disabled";
    if (($obs_state == "failed") || ($obs_state == "finished"))
      $delete = "";

  ?>
    <span><input type="button" onclick="popWindow('<?echo $process_url?>&action=annotate')" value="Annotate"></span>
    <span><input type="button" onclick="popWindow('<?echo $process_url?>&action=delete')" value="Delete"<?echo $delete?>></span>
<?
    $this->closeBlockHeader();
*/
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
<?
    echo "  <title>CASPSR | Result ".$this->utc_start."</title>";
    echo "  <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>\n";
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
      </div> <!-- sidebar1 -->
    </div <!-- contentLayot -->
      <div class="content">
<?
        $this->printMainHTML();
?>
     </div> <!-- content -->
    <!--</div> <!-- contentLayout -->
  </div> <!-- main -->
</body>
</html>
<?
  }

  function printMainHTML()
  {

    /* Plots  */
    $this->openBlockHeader("Plots");

    $imgs = $this->inst->getObsImages($this->obs_results_dir);
    echo "    <table>\n";
    $psrs = array_keys($imgs);
    for ($i=0; $i<count($psrs); $i++) {
      $p = $psrs[$i];
      echo "      <tr>\n";
      $this->printPlotCell($imgs[$p]["phase_vs_flux"], $imgs[$p]["phase_vs_flux_hires"]);
      $this->printPlotCell($imgs[$p]["phase_vs_time"], $imgs[$p]["phase_vs_time_hires"]);
      $this->printPlotCell($imgs[$p]["phase_vs_freq"], $imgs[$p]["phase_vs_freq_hires"]);
      $this->printPlotCell($imgs[$p]["bandpass"], $imgs[$p]["bandpass_hires"]);
      echo "      </tr>\n";
    }
    echo "    </table>\n";

    $this->closeBlockHeader();



    /* print the full dada header */
    $this->openBlockHeader("DADA Header");

    $cmd = "find ".$this->obs_results_dir." -mindepth 1 -maxdepth 1 -type f -name '*obs.start' | tail -n 1";
    $array = array();
    $rval = 0;
    $file = exec($cmd, $array, $rval);

    if ($file == "") {
      $header = array();
    } else {
      $header = getConfigFile($file, TRUE);
    }

    ksort($header);

    $keys_to_ignore = array("HDR_SIZE","FILE_SIZE","HDR_VERSION","FREQ","RECV_HOST");
    for ($i=0; $i<count($keys_to_ignore); $i++) {
      if (array_key_exists($keys_to_ignore[$i], $header)) {
        unset($header[$keys_to_ignore[$i]]);
      }
    }

    if (count($header) > 0)
      $keys = array_keys($header);
    else
      $keys = array();


?>
    <table width='100%' 
<?
    if (count($keys) == 0) {
      echo "      <tr><td colspan=2><font color=red>obs.start file did not exist</font></td></tr>\n";
    } else {

      $sub = floor(count($keys) / 3)+1;

      for ($i=0; $i<$sub; $i++) {

        echo "      <tr>";
        $j = ($sub * 0) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".$header[$keys[$j]]."</td>";

        $j = ($sub * 1) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".$header[$keys[$j]]."</td>";

        $j = ($sub * 2) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".$header[$keys[$j]]."</td>";
        echo "      </tr>\n";
      }
    }
  
?>
    </table>
<?
    $this->closeBlockHeader();

  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }

  }

  function getMostRecentResult() 
  {

    $current_time = time();

    $archive_time = 0;
    $cmd = "find ".$this->obs_results_dir." -mindepth 1 -maxdepth 1 -type f -name '*obs.start' -printf '%T@\\n' | sort | tail -n 1";
    $obs_start_time = exec($cmd, $array, $rval);
    if (count($array) == 0) {
      $difference = -1;
    } else {
      $difference = $current_time - $obs_start_time;
    }

    return $difference;
  }

  function printPlotCell($image, $image_hires) 
  {
    $have_hires = 0;
    $have_lores = 0;
    $hires_path = $this->obs_results_dir."/".$image_hires;
    $lores_path = $this->obs_results_dir."/".$image;

    if ((strlen($image_hires) > 1) && (file_exists($hires_path))) {
      $have_hires = 1;
    } 
    if ((strlen($image) > 1) && (file_exists($lores_path))) {
      $have_lores = 1;
    } 

    echo "    <td align='center'>\n"; 

    if ($have_hires) {
      echo "      <a href=\"/caspsr/old_results/".$this->utc_start."/".$image_hires."\">";
    }
 
    if ($have_lores) {     
      echo "      <img src=\"/caspsr/old_results/".$this->utc_start."/".$image."\">";
    }

    if ($have_hires) {
      echo "    </a><br>\n";
      echo "    Click for hi-res result\n";
    }

    echo "    </td>\n";

  }

  function printSourceCell($psr, $data)
  {

    $vals = array("dm", "p0", "int", "snr");
    $names = array("DM", "P0", "Integrated", "SNR");

    echo "    <td style='vertical-align: middle'>\n";
    echo "      <table border=0>\n";
    echo "        <tr>\n";
    echo "          <td align=right>Source</td>\n";
    echo "          <td align=left>".$psr."</td>\n";
    echo "        </tr>\n";

    $row = $data[$psr];
    $keys = array_keys($row);

    for ($i=0; $i<count($vals); $i++) {
      $k = $keys[$i];
      $n = $names[$i];
      $v = $row[$vals[$i]];
      echo "        <tr>\n";
      echo "          <td align=right>".$n."</td>\n";
      echo "          <td align=left>".$v."</td>\n";
      echo "        </tr>\n";
    }
    echo  "      </table>\n";
    echo  "    </td>\n";
  } 



}
$obj = new result($_GET["utc_start"]);
$obj->printHTML();
