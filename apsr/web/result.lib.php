<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class result extends apsr_webpage 
{

  var $utc_start;
  var $inst;
  var $obs_info;
  var $header;
  var $config;
  var $results_dir;
  var $archive_dir;
  var $results_link;
  var $process_link;
  var $source_info;

  function result()
  {
    apsr_webpage::apsr_webpage();

    $this->utc_start = $_GET["utc_start"];
    $this->type = $_GET["type"];
    $this->state = $_GET["state"] == "old" ? "old" : "normal";
    $this->title = "APSR Result: ".$this->utc_start;
    $this->inst = new apsr();
    if ($this->state == "normal")
    {
      $this->results_dir = $this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start;
      $this->customplot_dir = $this->inst->config["SERVER_RESULTS_DIR"];
      $this->archive_dir = $this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start;
      $this->client_dir = $this->inst->config["CLIENT_ARCHIVE_DIR"]."/".$this->utc_start;
      $this->results_link = "/apsr/results/".$this->utc_start."/";
    }
    else
    {
      $this->results_dir = "/export/old_results/apsr/".$this->utc_start;
      $this->customplot_dir = "/export/old_results/apsr/";
      $this->archive_dir = "";
      $this->client_dir = "";
      $this->results_link = "/apsr/old_results/".$this->utc_start."/";
    }
    
    # source names, DM's, periods and SNRS
    $this->source_info = $this->inst->getObsSources($this->results_dir);

    # images for this observation
    $this->images = $this->inst->getObsImages($this->results_dir);

    # current state of this observation
    $this->obs_state = $this->inst->getObsState($this->results_dir);

    # how old this observation is
    $this->most_recent = $this->inst->getMostRecentResult($this->results_dir);

    # DADA Header
    $this->header = $this->inst->getDADAHeader($this->results_dir);

    $this->process_link = "/apsr/processresult.php?observation=".$this->utc_start."&source=".$this->header["SOURCE"];
    $this->process_link = str_replace("+", "%2B", $this->process_link);

  }

  function printJavaScriptHead()
  {
?>
    <style>
      table.result {
        font-size: 10pt;
      }
      table.result th {
        font-weight: normal;
        text-align: right;
        padding-right: 10px;
      }
      table.result td {
        text-align: left;
      }
    </style>
  
    <script type='text/javascript'>  

      var npsrs = 0;
      var utc_start = "";
      var psrs = new Array();

      function popWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1024,height=700');");
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }
    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {


?>
    <table border=0 cellspacing=10px>
      <tr>
        <td width='380px' height='60px'><img src='/apsr/images/apsr_logo.png' width='380px' height='60px'></td>
        <td align=left><font size='+2'>Observation: <?echo $this->utc_start?></font></td>
      </tr>
    </table>

<?

    echo "<table cellpadding='5px'>\n";
    echo "<tr><td valign='top'>\n";

    $this->openBlockHeader("Observation Summary");
    $data = $this->inst->getArchiveCount($this->results_dir, $this->archive_dir, $this->client_dir);
?>
    <table class='result'>
      <tr><th>UTC START</th><td><?echo $this->utc_start?></td></tr>
      <tr><th>Start Time</th><td><?echo localTimeFromGmTime($this->utc_start)?></td></tr>
      <tr><th>Last Result Received</th><td><?echo makeTimeString($this->most_recent)?></td></tr>
      <tr><th>Obs State</th><td><?echo $this->obs_state?></td></tr>
      <tr><th>Num Bands</th><td><? echo $data["num_bands"]?></td></tr>
      <tr><th>Archives</th><td><? echo $data["num_archives"];?></td></tr>
      <tr><th>Results</th><td><? echo $data["num_processed"]." of ".$data["num_results"];?> </td></tr>
      <tr><th>Results Dir</th><td><? echo "<a href='".$this->results_link."'>link</a>"?></td></tr>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td>\n";
    echo "<td rowspan='2' valign='top'>\n";

    $this->openBlockHeader("DADA Header");
    $keys = array_keys($this->header);
    $keys_to_ignore = array("HDR_SIZE","FILE_SIZE","HDR_VERSION","FREQ","RECV_HOST",
                            "DSB","INSTRUMENT", "TELESCOPE", "BYTES_PER_SECOND", "UTC_START",
                            "CONFIG", "RESOLUTION");
    if ($this->header["MODE"] == "PSR")
      array_push($keys_to_ignore, "CALFREQ");

    echo "<table class='result'>\n";
    if (count($keys) == 0)
      echo "<tr><td colspan=2><font color=red>obs.start file did not exist</font></td></tr>\n";
    else
    {
      for ($i=0; $i<count($keys); $i++)
      {
        if (!(in_array($keys[$i], $keys_to_ignore)))
        {
          echo "  <tr>";
          echo "<th>".$keys[$i]."</th>";
          echo "<td>".$this->header[$keys[$i]]."</td>";
          echo "</tr>\n";
        }
      }
    }
    echo "</table>\n";
    $this->closeBlockHeader();


    echo "</td></tr>\n";
    echo "<tr><td valign='top'>\n";


    $this->openBlockHeader("Actions");
    $archives = " disabled";
    if (($this->obs_state == "finished") && ($data["num_results"] > 0)) {
      $archives = "";
    }

    $delete = " disabled";
    if (($this->obs_state == "failed") || ($this->obs_state == "finished"))
      $delete = "";
?>

    <table class="result">
      <tr>
        <td align="center">
          <input type="button" onclick="popWindow('<?echo $this->process_link?>&action=plot')" value="Create Plots"<?echo $archives?>>&nbsp;&nbsp;&nbsp;
          <input type="button" onclick="popWindow('<?echo $this->process_link?>&action=reprocess_low')" value="Re Process Results"<?echo $archives?>>&nbsp;&nbsp;&nbsp;
          <input type="button" onclick="popWindow('<?echo $this->process_link?>&action=annotate')" value="Annotate Obs.">&nbsp;&nbsp;&nbsp;
        </td>
      </tr>
      <tr>
        <td align=center>
          <input type="button" onclick="popWindow('<?echo $this->process_link?>&action=delete_obs')" value="Delete Observation"<?echo $delete?>>
          <input type="button" onclick="popWindow('/apsr/custom_plot.lib.php?single=true&basedir=<?echo $this->customplot_dir?>&utc_start=<?echo $this->utc_start?>')" value="Custom Plots">
        </td>
      </tr>
    </table>

<?
    $this->closeBlockHeader();

    echo "</td></tr>\n";

    echo "<tr><td colspan='2'>\n";
    $this->openBlockHeader("Plots");
?>
    <table>
      <tr>
        <th>PSR</th>
        <th>Phase vs Flux</th>
        <th>Phase vs Time</th>
        <th>Phase vs Freq</th>
      </tr>
<?
      $psrs = array_keys($this->images);

      for ($i=0; $i<count($psrs); $i++) {
        $p = $psrs[$i];
        echo "  <tr>\n";
        echo "    <td>\n";
        $this->printSourceSummary($p);
        echo "    </td>\n";
        $this->printPlotRow($this->images[$p]["phase_vs_flux"], $this->images[$p]["phase_vs_flux_hires"]);
        $this->printPlotRow($this->images[$p]["phase_vs_time"], $this->images[$p]["phase_vs_time_hires"]);
        $this->printPlotRow($this->images[$p]["phase_vs_freq"], $this->images[$p]["phase_vs_freq_hires"]);
        echo "  </tr>\n";
      }
?>
    </table>
<?
        $this->closeBlockHeader();
        echo "</td></tr>\n";
        echo "</table>\n";

  }

  function printSourceSummary($psr)
  {
    $vals = array("dm", "p0", "int", "snr");
    $names = array("DM", "P0", "Length", "SNR");

    if (!array_key_exists($psr, $this->source_info))
      return;

    $p = $this->source_info[$psr];

    echo "<table class='result'>\n";
    echo "  <tr><th>Source</th><td>".$psr."</td></tr>\n";
    for ($i=0; $i<count($vals); $i++)
    {
      $v = $vals[$i];
      echo  "<tr><th>".$names[$i]."</th><td>".$p[$v]."</td></tr>\n";
    }
    echo "</table>\n";
  }

  function printPlotRow($image, $image_hires) {

    $have_hires = 0;
    $hires_path = $this->results_dir."/".$image_hires;
    if ((strlen($image_hires) > 1) && (file_exists($hires_path))) {
      $have_hires = 1;
    }
    echo "    <td align=\"center\">\n";

    if ($have_hires) {
      echo "      <a href=\"/apsr/results/".$this->utc_start."/".$image_hires."\">";
    }
    echo "      <img width=241px height=181px src=\"/apsr/results/".$this->utc_start."/".$image."\">";

    if ($have_hires) {
      echo "    </a><br>\n";
      echo "    Click for hi-res result\n";
    }
    echo "    </td>\n";
  }

}

handledirect("result");

