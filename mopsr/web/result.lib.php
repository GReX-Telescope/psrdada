<?PHP

function cmp($a, $b) {
  # Sort such that FB is after J*
  if (!(substr($a, 0, 1) == substr($b, 0,1)))
    return -1 * strcmp(substr($a, 0, 1), substr($b, 0, 1));
  # otherwise, sort normally
  return strcmp($a, $b);
}

include ("mopsr.lib.php" );
include ("mopsr_webpage.lib.php");

class result extends mopsr_webpage 
{

  var $utc_start = "";
  var $obs_results_dir = "";
  var $obs_archive_dir = "";
  var $annotation_file = "";
  var $inst = "";
  var $imgs;
  var $mode;
  var $obs_config = "UNKNOWN";
  var $class = "";

  var $corr_types = array();
  var $fb_types = array();
  var $tb_types = array();
  var $im_types = array();

  function result()
  {
    mopsr_webpage::mopsr_webpage();
    $this->inst = new mopsr();
    $this->utc_start = $_GET["utc_start"];
    $this->class = (isset($_GET["class"])) ? $_GET["class"] : "new";
    if ($this->class == "old")
    {
      $this->obs_results_dir = $this->inst->config["SERVER_OLD_RESULTS_DIR"]."/".$this->utc_start;
      $this->obs_archive_dir = $this->inst->config["SERVER_OLD_ARCHIVE_DIR"]."/".$this->utc_start;
      $this->results_link = "old_results";
    }
    else
    {
      $this->obs_results_dir = $this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start;
      $this->obs_archive_dir = $this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start;
      $this->results_link = "results";
    }
    $this->annotation_file = $this->obs_archive_dir."/obs.txt";
    
    # length and snr for the sources in this observation
    $this->source_info = $this->getObsModules($this->obs_results_dir);

    $this->img_types = array ();
    $this->img_types["TB"] = array("fl", "fr", "ti", "bp", "pm", "st", "l9", "ta", "tc");
    $this->img_types["CORR"] = array("sn", "bd", "ad", "po");
    $this->img_types["FB"] = array("*");

    $this->imgs = $this->inst->getObsImages($this->obs_results_dir, $ant);

  }

  function javaScriptCallback()
  {
  }

  function printJavaScriptHead()
  {
?>

   <style type="text/css">
      .module {
        text-align: left;
        background-color: #FFFFFF;
        padding-right: 10px;
      }
  </style>

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
    $most_recent = $this->getMostRecentResult();
    $results_link = "<a href='/mopsr/".$this->results_link."/".$this->utc_start."/'>link</a>";
    $cmd = "find ".$this->obs_archive_dir." -mindepth 2 -maxdepth 2 -type f -name '*.ar' -printf '%f\n' | sort -n | uniq | wc -l";
    $num_archives = exec($cmd);

    $obs_state   = $this->getObsState();

?>
    <table>
      <tr><td>UTC </td><td><?echo $this->utc_start?></td></tr>
      <tr><td>Local</td><td><? echo localTimeFromGmTime($this->utc_start)?></td></tr>
      <tr><td>State</td><td><? echo ucfirst($obs_state)?></td></tr>
      <tr><td>Age</td><td><? echo makeTimeString($most_recent)?></td></tr>
      <tr><td>Archives</td><td><? echo $num_archives?></td></tr>
      <tr><td>Results</td><td><a href='/mopsr/<? echo $this->results_link."/".$this->utc_start?>/'>Link</a></td></tr>
    </table>
<?


    $this->closeBlockHeader();

    $this->openBlockHeader("Source Summary");

    $vals = array("int", "snr");
    $names = array("Length", "SNR", "Plot");

    $sources = array_keys($this->source_info);

    echo "    <table>\n";
    echo "      <tr><th class='module'>Source</th>";
    foreach ($names as $name)
    {
      echo "<th class='module'>".$name."</th>";
    }
    echo "</tr>\n";

    sort ($sources);
    foreach ($sources as $source)
    {
      $row = $this->source_info[$source];
        
      echo "        <tr>\n";
      echo "          <td class='module'><a href=/mopsr/results.lib.php?single=true&offset=0&length=20&inline_images=true&filter_type=SOURCE&filter_value=".urlencode($source).">".$source."</a></td>\n";

      for ($j=0; $j<count($vals); $j++) 
      {
        $n = $names[$j];
        $v = $row[$vals[$j]];
        echo "          <td class='module'>".$v."</td>\n";
      }
      $regen_url = "/mopsr/result.lib.php?script=true&script_name=manual_plot_make.pl&utc_start=".$this->utc_start."&source=".rawurlencode($source);
?>
    <td class='module'><input type="button" onclick="popWindow('<?echo $regen_url?>')" value="Regenerate"<?echo $delete?>></td>;
<?
      echo "      </tr>\n";
    }
    echo "</table>\n";

    $this->closeBlockHeader();

    if (file_exists($this->annotation_file))
    {
      $this->openBlockHeader("Observation Notes");

      $notes = file_get_contents($this->annotation_file);
      echo "<p>".$notes."</p>\n";

      $this->closeBlockHeader();
    }

    /* Actions */
    $this->openBlockHeader("Actions");

    $process_url = "/mopsr/process_obs.lib.php?single=true&utc_start=".$this->utc_start."&class=".$this->class;
    $process_url = str_replace("+", "%2B", $process_url);

    $custom_url = "/mopsr/custom_plot.lib.php?basedir=".$this->obs_results_dir."&utc_start=".$this->utc_start;

    $delete = " disabled";
    if (($obs_state == "failed") || ($obs_state == "finished"))
      $delete = "";

  ?>
    <span><input type="button" onclick="popWindow('<?echo $process_url?>&action=annotate')" value="Annotate"></span>
    <span><input type="button" onclick="popWindow('<?echo $process_url?>&action=delete')" value="Delete"<?echo $delete?>></span>
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
    $this->openBlockHeader("Plots");

    $sources = array_keys($this->source_info);

    usort($sources, "cmp");

    for ($i=0; $i<count($sources); $i++)
    {
      $source = $sources[$i];
      $type = $this->source_info[$source]["type"];

      if ($type == "TB")
      {
        $this->printTBImgs($source);
      }
      else if ($type == "CORR")
      {
        $this->printCorrImgs($source);
      }
      else if ($type == "FB")
      {
        $this->printFBImgs("FB");
      }
    }

    $this->closeBlockHeader();


    /* print the full dada header */
    $this->openBlockHeader("DADA Header");

    $cmd = "find ".$this->obs_results_dir." -mindepth 2 -maxdepth 2 -type f -name 'obs.header' | tail -n 1";
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
    <table width='100%'>
<?
    if (count($keys) == 0) {
      echo "      <tr><td colspan=2><font color=red>obs.header file did not exist</font></td></tr>\n";
    } else {

      $sub = floor(count($keys) / 3)+1;

      for ($i=0; $i<$sub; $i++) {

        echo "      <tr>";
        $j = ($sub * 0) + $i;
        echo "        <td width='16.6%'>".$keys[$j]."</td>";
        echo "        <td width='16.6%' style='padding-right:10px;'>".substr($header[$keys[$j]],0, 32)."</td>";

        $j = ($sub * 1) + $i;
        echo "        <td width='16.6%'>".$keys[$j]."</td>";
        echo "        <td width='16.6%' style='padding-right:10px;'>".substr($header[$keys[$j]],0, 32)."</td>";

        $j = ($sub * 2) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%' style='padding-right:10px;'>".substr($header[$keys[$j]],0,32)."</td>";
        echo "      </tr>\n";
      }
    }
?>
    </table>
<?
    $this->closeBlockHeader();

  }

  function getMostRecentResult() 
  {

    $current_time = time();

    $cmd = "find ".$this->obs_archive_dir." -mindepth 2 -maxdepth 2 -type f -name '*.ar' -printf '%T@\\n' | sort | tail -n 1";
    $archive_time = exec($cmd, $array, $rval);
    $difference = 0;

    /* If we dont have any archives */
    if (count($array) == 0) {

      $archive_time = 0;
      $cmd = "find ".$this->obs_results_dir." -mindepth 1 -maxdepth 1 -type f -name '*obs.start' -printf '%T@\\n' | sort | tail -n 1";
      $obs_start_time = exec($cmd, $array, $rval);
      if (count($array) == 0) {
        $difference = -1;
      } else {
        $difference = $current_time - $obs_start_time;
      }
    } else {
      $difference = $current_time - $archive_time;
    }

    return $difference;
  }

  /* Determine the 'state' of the observation */
  function getObsState() {

    $dir = $this->obs_archive_dir;

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

  function printPlotCell($image, $image_hires, $width, $height) 
  {
    $have_hires = 0;
    $hires_path = $this->obs_results_dir."/".$image_hires;
    if ((strlen($image_hires) > 1) && (file_exists($hires_path))) {
      $have_hires = 1;
    } 

    echo "    <td align='center'>\n"; 

    if ($have_hires) {
      echo "<a href=\"/mopsr/".$this->results_link."/".$this->utc_start."/".$image_hires."\">";
    }
      
    echo "<img src=\"/mopsr/".$this->results_link."/".$this->utc_start."/".$image."\" width='".$width."px' height=".$height."px' alt='Missing image'>";

    if ($have_hires) {
      echo "</a>";
    }
    

    echo "    </td>\n";

  }

  function printTBImgs($source)
  {
    echo "<table id='tied_array_beam_images'>\n";
    echo   "<tr>\n";
    $counter = 0;
    foreach ($this->img_types["TB"] as $t)
    {
      $this->printPlotCell($this->imgs[$source][$t."_120x90"], $this->imgs[$source][$t."_1024x768"], "120", "90");
      $counter++;
      if ($counter % 6 == 0)
        echo "</tr>\n<tr>";
    }
    echo   "</tr>\n";
    echo "</table>\n";
  }

  function printCorrImgs($source)
  {
    echo "<table id='correlation_images'>\n";
    echo   "<tr>\n";
    foreach ($this->img_types["CORR"] as $t)
      $this->printPlotCell($this->imgs[$source][$t."_160x120"], $this->imgs[$source][$t."_1024x768"], "160", "120");
    echo   "</tr>\n";
    echo "</table>\n";
  }

  function printFBImgs($source)
  {
    $types = array();
    $cmd = "find ".$this->obs_results_dir." -name '????-??-??-??:??:??.FB.*.850x680.png' | awk -F. '{print $3}' | sort -rn | uniq";
    $lastline = exec($cmd, $types, $rval);

    echo "<table id='fan_beam_images'>\n";
    foreach ($types as $t)
    {
      echo   "<tr>\n";
      $this->printPlotCell($this->imgs[$source][$t."_850x680"], $this->imgs[$source][$t."_850x680"], "850", "680");
      echo   "</tr>\n";
    }
    echo "<table>\n";
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

  function getObsModules()
  {
    $dir = $this->obs_results_dir;

    $info = getConfigFile ($dir."/obs.info");
    $results = array();

    if ($info["CORR_ENABLED"] == "true")
    {
      $source = $info["SOURCE"];
      if (file_exists($dir."/".$source))
      {
        $results[$source] = array();
        $results[$source]["type"] = "CORR";
        $results[$source]["int"] = $this->calcIntLengthCorr ($source);
      }
    }

    if ($info["FB_ENABLED"] == "true")
    {
      $source = "FB";
      if (file_exists($dir."/".$source))
      {
        $results[$source] = array();
        $results[$source]["type"] = "FB";
        $results[$source]["int"] = $this->calcIntLengthFB ($source);
        $results[$source]["snr"] = "NA";
      }
    }

    for ($i=0; $i<4; $i++)
    {
      if ($info["TB".$i."_ENABLED"] == "true")
      {
        $source = $info["TB".$i."_SOURCE"];
        if (file_exists($dir."/".$source))
        {
          $results[$source] = array();
          $results[$source]["type"] = "TB";

          $tot_file = $dir."/".$source."/".$source."_t.tot";
          if (file_exists($tot_file))
          {
            $results[$source]["int"]     = $this->inst->getIntergrationLength($tot_file);
            $results[$source]["nsubint"] = $this->inst->getNumSubints($tot_file);
            $results[$source]["snr"]     = instrument::getSNR($tot_file);
          }
        }
      } else if ($i == 0 && strpos($info["CONFIG"], "TIED_ARRAY") !== false ) {
        $source = $info["SOURCE"];
        if (file_exists($dir."/TB"))
        {
          $results[$source] = array();
          $results[$source]["type"] = "TB";

          $tot_file = $dir."/".$source."/".$source."_t.tot";
          if (file_exists($tot_file))
          {
            $results[$source]["int"]     = $this->inst->getIntergrationLength($tot_file);
            $results[$source]["nsubint"] = $this->inst->getNumSubints($tot_file);
            $results[$source]["snr"]     = instrument::getSNR($tot_file);
          }
        }
      }
    }
    return $results;
  }

  function calcIntLengthFB ($source)
  {
    $ac_file = $this->obs_results_dir."/".$source."/all_candidates.dat";
    if (file_exists($ac_file))
    {
      $cmd = "tail -n 1000 ".$ac_file." | awk '{print $3}' | sort -n | tail -n 1";
      $length = exec($cmd, $output, $rval);
      return sprintf("%5.0f", $length);
    }
    return 0;
  }

  function calcIntLengthCorr($source)
  {
    $cc_file = $this->obs_results_dir."/".$source."/cc.sum";
    if (file_exists ($cc_file))
    {
      $cmd = "find ".$this->obs_archive_dir."/".$source." -name '*.ac' | sort -n | tail -n 1";
      $output = array();
      $ac = exec($cmd, $output, $rval);

      $parts = explode("_", $ac);
      $bytes_to = $parts[count($parts)-1];

      $cmd = "grep BYTES_PER_SECOND ".$this->obs_results_dir."/".$source."/obs.header | awk '{print $2}'";
      $output = array();
      $Bps = exec($cmd, $output, $rval);

      $length = $bytes_to / $Bps;
      return sprintf ("%5.0f", $length);
    }
    return 0;
  }

  #
  # Run the specified perl script printing the output
  # to the screen
  #
  function printScript($get)
  {
    $script_name = $get["script_name"];
    if (!array_key_exists("source", $get)) {
      print "Error: Forgot to pass source";
      return -1;
    }
    $source= $get["source"];

?>
<html>
<head>
<?  
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
?>
</head>
<body>
<?
    $this->openBlockHeader("Running ".$script_name);
    echo "<p>Script is now running in background, please wait...</p>\n";
    echo "<br>\n";
    echo "<br>\n";
    flush();
    //$script = "source /home/dada/.bashrc; ".$script_name." ".$this->utc_start." ".$source." 2>&1";
    $script = "source /home/dada/.dadarc; ".$script_name." ".$this->utc_start." ".rawurldecode($source)." 2>&1";
    echo "<pre>\n";
    system($script);
    echo "</pre>\n";
    echo "<p>It is now safe to close this window</p>\n";
    $this->closeBlockHeader();
?>  
</body>
</html>

<?
  }


}

handleDirect("result");
