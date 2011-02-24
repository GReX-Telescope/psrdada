<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

class beam_viewer extends bpsr_webpage 
{

  var $inst = 0;
  var $utc_start = "";
  var $beam = 0;
  var $nbeams = 0;

  function beam_viewer()
  {
    bpsr_webpage::bpsr_webpage();
    $this->css = array("bpsr.css");
    $this->inst = new bpsr();

    if (isset($_GET["utc_start"]))
      $this->utc_start = $_GET["utc_start"];
    else
    {
      $cmd = "find ".$this->inst->config["SERVER_RESULTS_DIR"]." -maxdepth 1 -type d -name '2*' -printf '%f\\n' | sort | tail -n 1";
      $this->utc_start = exec($cmd);
    }

    if (isset($_GET["beam"]))
      $this->beam = $_GET["beam"];
    else
      $this->beam = "01";

    $this->nbeams = $this->inst->ibobs["NUM_IBOB"];

    $this->title = "BPSR | Beam Viewer | ".$this->utc_start." ".$this->beam;

  }

  function javaScriptCallback()
  {
    return "beam_viewer_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function handle_beam_viewer_request(bv_http_request) 
      {
        if (bv_http_request.readyState == 4) 
        {
          var response = String(bv_http_request.responseText)
          var lines = response.split("\n");

          var currImg
          var beam
          var size
          var type
          var img

          for (i=0; i < lines.length-1; i++) {

            values = lines[i].split(":::");
            beam = values[0];
            size = values[1];
            type = values[2];
            img  = values[3];
     
            if (size == "112x84") {
              // ignore 
            } else {
              obj = document.getElementById(type+"_"+size);

              if ((size == "400x300") && (obj.src != img)) {
                obj.src = img
              }
              if ((size == "1024x768") && (obj.href != img)) {
                obj.href = img;
              }
            }
          }
        }
      }

      /* Gets the data from the URL */
      function beam_viewer_request() 
      {
        var url = "beam_viewer.lib.php?update=true&type=all&utc_start=<?echo $this->utc_start?>&beam=<?echo $this->beam?>"

        if (window.XMLHttpRequest)
          bv_http_request = new XMLHttpRequest()
        else
          bv_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        bv_http_request.onreadystatechange = function() {
          handle_beam_viewer_request(bv_http_request)
        }
        bv_http_request.open("GET", url, true)
        bv_http_request.send(null)
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {

    $obs_info_file = $this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start."/obs.info";
    $obs_start_file = $this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start."/".$this->beam."/obs.start";

    $header = array();

    if (file_exists($obs_start_file))
    {
      $header = getConfigFile($obs_start_file);
    }

    echo "<table cellpadding=10px><tr><td colspan=2>\n";


    $this->openBlockHeader("Select Beams for ".$this->utc_start);
?>
    <table border=0>
      <tr valign=middle>
<?
  for ($i=1;$i<=$this->nbeams; $i++) {
    $i_str = sprintf("%02d", $i);
    if ($i_str != $this->beam) {
      echo "        <td width=40>\n";
      echo "          <div class=\"btns\">\n";
      echo "            <a href=\"/bpsr/beam_viewer.lib.php?single=true&utc_start=".$this->utc_start."&beam=".$i_str."\" class=\"btn\" > <span>".$i_str."</span></a>\n";
      echo "          </div>\n";
      echo "        </td>\n";
    } else {
      echo "        <td width=40 style='text-align: center;'><b>".$i_str."</b></td>";
    }
  }
?>
      </tr>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td></tr>\n";
    echo "<tr><td rowspan=2>\n";

    $this->openBlockHeader("Beam Plots");
?>
    <table>
      <tr>
        <td align=center width="33%">
          Bandpass<br>
          <a id="bp_1024x768" href="/images/blackimage.gif">
            <img id="bp_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

        <td align=center width="33%">
          DM0 Timeseries<br>
          <a id="ts_1024x768" href="/images/blackimage.gif">
            <img id="ts_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

        <td align=center width="33%">
          Phase vs Freq<br>
          <a id="pvf_1024x768" href="/images/blackimage.gif">
            <img id="pvf_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

      </tr>

      <tr>
        <td align=center>
          PD Bandpass (live)<br>
          <a id="pdbp_1024x768" href="/images/blackimage.gif">
            <img id="pdbp_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

        <td align=center>
          Fluctuation Power Spectrum<br>
          <a id="fft_1024x768" href="/images/blackimage.gif">
            <img id="fft_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

        <td align=center>
          Digitizer Statistics<br>
          <a id="dts_1024x768" href="/images/blackimage.gif">
            <img id="dts_400x300" src="/images/blackimage.gif" width=300 height=225>
          </a>
        </td>

      </tr>
    </table>
<?
    $this->closeBlockHeader();
    
    echo "</td><td valign=top height=10px>";

    $this->openBlockHeader("Beam/Obs Summary");
?>
    <table class="result">
      <tr><td width=50%>Source</td><td align=left width=50%><?echo $header["SOURCE"]?></td></tr>
      <tr><td>UTC_START</td><td align=left><?echo $this->utc_start?></td></tr>
      <tr><td>RA</td><td align=left><?echo $header["RA"]?></td></tr>
      <tr><td>DEC</td><td align=left><?echo $header["DEC"]?></td></tr>
      <tr><td>Beam</td><td align=left><?echo $this->beam?> of <?echo $this->nbeams?></td></tr>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td></tr>\n";
    echo "<tr><td valign=top>\n";

    $this->openBlockHeader("Obs State");

?>
          <table class="result">
            <tr><td>State</td><td><?echo $state["FINALIZED"]?></td></tr>
            <tr><td>Transferred to swin</td><td align=left><?echo $state["sent.to.swin"]?></td></tr>
            <tr><td>Transferred to parkes</td><td align=left><?echo $state["sent.to.parkes"]?></td></tr>
            <tr><td>On tape at swin</td><td align=left><?echo $state["on.tape.swin"]?></td></tr>
            <tr><td>On tape at parkes</td><td align=left><?echo $state["on.tape.parkes"]?></td></tr>
          </table>
<?
    $this->closeBlockHeader();
    echo "</td></tr>\n";
    echo "</table>\n";
  }

  function printUpdateHTML($get)
  {

    $utc_start = $get["utc_start"];
    $beam      = $get["beam"];
    $size      = "all";
    $type      = "all";
    $ibob      = 0;
    $inst      = $this->inst;
     
    $results_dir = $inst->config["SERVER_RESULTS_DIR"];

    # find the ibob control IP that corresponds to the specified beam
    $beam_str = sprintf("%02d",$beam);
    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++) {
      if ($inst->ibobs["BEAM_".$i] == $beam_str) {
        $ibob = $inst->ibobs["CONTROL_IP_".$i];
      }
    }

    $actual_obs_results = array();
    $actual_stats_results = array();

    # get the montioring images for this obs/beam
    $obs_results = $inst->getResults($results_dir, $utc_start, $type, $size, $beam_str);
    if (is_array($obs_results))
    {
      $actual_obs_results = array_pop($obs_results);
      if (is_array($actual_obs_results)) 
        $actual_obs_results = array_pop($actual_obs_results);
      else
        $actual_obs_results = array();
    }

    # get the pdbp images for this ibob
    $stats_results = $inst->getStatsResults($results_dir, $ibob);
    if (is_array($stats_results)) 
      $actual_stats_results = array_pop($stats_results);

    $results = array_merge($actual_obs_results, $actual_stats_results);

    $types = array("bp","ts","fft","dts","pdbp","pvf");
    $sizes = array("112x84", "400x300", "1024x768");

    $string = "";
    for ($i=0; $i<count($types); $i++) {
      for ($j=0; $j<count($sizes); $j++) {
        $key = $types[$i]."_".$sizes[$j];
        if (array_key_exists($key, $results))
          $string .= $beam_str.":::".$sizes[$j].":::".$types[$i].":::".$results[$key]."\n";
        else
          $string .= $beam_str.":::".$sizes[$j].":::".$types[$i].":::../../images/blankimage.gif\n";
      }
    }

    echo $string;
  }

}

handledirect("beam_viewer");

