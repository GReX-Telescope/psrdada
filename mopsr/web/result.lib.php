<?PHP

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
    $this->source_info = $this->getObsModules($this->obs_results_dir);
    $cmd = "grep ^MODE ". $this->obs_results_dir."/obs.info | awk '{print $2}'";
    $this->mode = exec($cmd);
  
    $cmd = "grep ^CONFIG ".$this->obs_results_dir."/obs.info | awk '{print $2}'";
    $result = exec($cmd);
    if ($result != "")
      $this->obs_config = $result;
    else
      if (($this->mode == "CORR") || ($this->mode == "CORR_CAL"))
        $this->obs_config = "CORRELATION";
      else
        $this->obs_config = "INDIVIDUAL_MODULES";

    $ant = array();
    if ($this->obs_config == "FAN_BEAM" || $this->obs_config == "TIED_ARRAY_FAN_BEAM")
    {
      $ant = array_merge ($ant, array("FB"));
      $cmd = "find ".$this->obs_results_dir." -name '????-??-??-??:??:??.FB.*.850x680.png' | awk -F. '{print $3}'";
      $lastline = exec($cmd, $this->fb_types, $rval);
    }
    else if ($this->obs_config == "TIED_ARRAY_BEAM" || $this->obs_config == "TIED_ARRAY_FAN_BEAM")
    {
      $this->tb_types = array("fl", "fr", "ti", "bp", "pm");
      $ant = array_merge ($ant, array("TB"));
    }
    else if ($this->obs_config == "CORRELATION")
    {
      $this->corr_types = array("sn", "bd", "ad", "po");
      $ant = array_merge ($ant, array("CH00"));
    }
    else if ($this->obs_config == "INDIVIDUAL_MODULES")
    {
      $this->im_types = array("fl", "fr", "ti", "bp", "pm");
    }

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
    $names = array("Length", "SNR");

    $ants = array_keys($this->source_info);

    echo "    <table>\n";
    echo "      <tr><th class='module'>Ant</th>";
    foreach ($names as $name)
    {
      echo "<th class='module'>".$name."</th>";
    }
    echo "</tr>\n";

    sort ($ants);
    foreach ($ants as $ant)
    {
      $sources = array_keys($this->source_info[$ant]);
      foreach ($sources as $source)
      {
        $row = $this->source_info[$ant][$source];
        
        echo "        <tr>\n";
        echo "          <td class='module'>".$ant."</td>\n";

        for ($j=0; $j<count($vals); $j++) 
        {
          $n = $names[$j];
          $v = $row[$vals[$j]];
          echo "          <td class='module'>".$v."</td>\n";
        }
        echo "      </tr>\n";
      }
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

    // plot correlation images if they exist
    if ($this->obs_config == "CORRELATION")
    {
      echo "    <table id='correlation_images'>\n";
      echo "      <tr>\n";
      foreach ($this->corr_types as $t)
        $this->printPlotCell($this->imgs["CH00"][$t."_160x120"], $this->imgs["CH00"][$t."_1024x768"]);
      echo "      </tr>\n";
      echo "    </table>\n";
    }

    // print tied array beam images if they exist
    if ($this->obs_config == "TIED_ARRAY_BEAM" || $this->obs_config == "TIED_ARRAY_FAN_BEAM")
    {
      echo "    <table id='tied_array_beam_images'>\n";
      echo "      <tr>\n";
      foreach ($this->tb_types as $t)
        $this->printPlotCell($this->imgs["TB"][$t."_120x90"], $this->imgs["TB"][$t."_1024x768"]);
      echo "      </tr>\n";
      echo "    </table>\n";
    }

    // print fan beam images if they exist
    if ($this->obs_config == "FAN_BEAM" || $this->obs_config == "TIED_ARRAY_FAN_BEAM")
    {
      echo "    <table id='fan_beam_images'>\n";
      rsort($this->fb_types);
      foreach ($this->fb_types as $t)
      {
        echo "      <tr>\n";
        $this->printPlotCell($this->imgs["FB"][$t."_850x680"], $this->imgs["FB"][$t."_850x680"]);
        echo "      </tr>\n";
      }
      echo "    </table>\n";
    }

    if ($this->obs_config == "INDIVIDUAL_MODULES")
    {
      $ants = array_keys($this->imgs);
      sort($ants);
      echo "    <table id='individual_modules'>\n";
      foreach ($ants as $a)
      {
        echo "      <tr>\n";
        echo "        <td>".$a."</td>\n";
        foreach ($this->im_types as $t)
          $this->printPlotCell($this->imgs[$a][$t."_120x90"], $this->imgs[$a][$t."_1024x768"]);
        echo "        </td>\n";
        echo "      </tr>\n";
      }
      echo "    </table>\n";
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
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".substr($header[$keys[$j]],0, 32)."</td>";

        $j = ($sub * 1) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".substr($header[$keys[$j]],0, 32)."</td>";

        $j = ($sub * 2) + $i;
        echo "        <td width='16.6%' align='right'>".$keys[$j]."</td>";
        echo "        <td width='16.6%'>".substr($header[$keys[$j]],0,32)."</td>";
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

  function printPlotCell($image, $image_hires) 
  {


    $have_hires = 0;
    $hires_path = $this->obs_results_dir."/".$image_hires;
    if ((strlen($image_hires) > 1) && (file_exists($hires_path))) {
      $have_hires = 1;
    } 

    echo "    <td align='center'>\n"; 

    if ($have_hires) {
      echo "      <a href=\"/mopsr/".$this->results_link."/".$this->utc_start."/".$image_hires."\">";
    }
      
    echo "      <img src=\"/mopsr/".$this->results_link."/".$this->utc_start."/".$image."\">";

    if ($have_hires) {
      echo "    </a><br>\n";
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

  function getObsModules($dir)
  {
    $rval = 0;
    $dirs = array();
    $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type d -printf '%f\n'";
    $line = exec ($cmd, $dirs, $rval);

    $results = array();
    foreach ($dirs as $subdir)
    {
      $tots = array();
      $cmd = "find ".$dir."/".$subdir." -mindepth 1 -maxdepth 1 -type f -name '*_t.tot' -printf '%f\n'";
      $line = exec($cmd, $tots, $rval);

      $results[$subdir] = array();
      foreach ($tots as $tot)
      {
        $arr = split("_", $tot, 3);
        if (count($arr) == 3)
          $s = $arr[0]."_".$arr[1];
        else
          $s = $arr[0];

        if (!array_key_exists($s, $results[$subdir]))
          $results[$subdir][$s] = array();

        if (strpos($tot, "_t") !== FALSE)
        {
          $results[$subdir][$s]["int"]     = $this->inst->getIntergrationLength($dir."/".$subdir."/".$tot);
          $results[$subdir][$s]["nsubint"] = $this->inst->getNumSubints($dir."/".$subdir."/".$tot);
          $results[$subdir][$s]["snr"]     = instrument::getSNR($dir."/".$subdir."/".$tot);
        }
      }
    }

    return $results;
  }

}

handleDirect("result");
