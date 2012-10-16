<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class result extends bpsr_webpage 
{

  var $utc_start;
  var $inst;
  var $obs_info;
  var $header;
  var $config;
  var $results_dir;
  var $results_link;
  var $type;
  var $state;

  function result()
  {
    bpsr_webpage::bpsr_webpage();

    $this->utc_start = $_GET["utc_start"];
    $this->type = isset($_GET["type"]) ? $_GET["type"] : "";
    $this->state = (isset($_GET["state"]) && $_GET["state"] == "old") ? "old" : "normal";
    $this->title = "BPSR Result: ".$this->utc_start;
    $this->inst = new bpsr();
    if ($this->state == "normal")
    {
      $this->results_dir = $this->inst->config["SERVER_RESULTS_DIR"];
      $this->results_link = "/bpsr/results/".$this->utc_start."/";
    }
    else
    {
      $this->results_dir = $this->inst->config["SERVER_OLD_RESULTS_DIR"];
      $this->results_link = "/bpsr/old_results/".$this->utc_start."/";
    }
  }

  /*
  function javaScriptCallback()
  {
    return "result_request();";
  }
  */

  function printJavaScriptHead()
  {

?>
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

      function handle_result_request(r_http_request) 
      {
        if (r_http_request.readyState == 4) {

          var response = String(r_http_request.responseText)
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
          
            currImg = document.getElementById("beam"+beam);
            if (currImg) {
              if (currImg.src != img) {
                currImg.src = img
              }
            }
          }
        }
      }

      function result_request() 
      {
        var url = "result.lib.php?update=true&utc_start=<?echo $this->utc_start?>&state=<?echo $this->state?>";

        var type = "bp";
        if (document.imageform.imagetype[0].checked == true) 
          type = "bp";

        if (document.imageform.imagetype[1].checked == true) 
          type = "ts";

        if (document.imageform.imagetype[2].checked == true) 
          type = "fft";

        if (document.imageform.imagetype[3].checked == true) 
          type = "pvf";

        url += "&type="+type;

        //alert(url);

        if (window.XMLHttpRequest)
          r_http_request = new XMLHttpRequest();
        else
          r_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        r_http_request.onreadystatechange = function() {
          handle_result_request(r_http_request)
        };
        r_http_request.open("GET", url, true);
        r_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {

    $obs_info = $this->inst->configFileToHash($this->results_dir."/".$this->utc_start."/obs.info");
    $data = array_pop($this->inst->getResultsInfo($this->utc_start, $this->results_dir));
    $header = $this->inst->configFileToHash($data["obs_start"]);
    $state = $this->inst->getObservationState($this->utc_start);

    # try to be a little smarter about the type of image shown
    if ($this->type == "")
    {
      $this->type = "bp";
      if (strpos($obs_info["SOURCE"], "J") === 0)
        $this->type = "pvf";
    }

?>
    <table border=0 cellspacing=10px>
      <tr>
        <td width='210px' height='60px'><img src='/bpsr/images/bpsr_logo.png' width='200px' height='60px'></td>
        <td align=left><font size='+2'>Observation: <?echo $this->utc_start?></font></td>
      </tr>
    </table>

    <table border=0 cellspacing=10px>
      <tr><td rowspan=2 valign="top">
<?
    $this->openBlockHeader("Beam View");
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=5>

      <tr>
        <td rowspan=3 valign="top" align='left'>
          <form name="imageform" class="smalltext">
            <input type="radio" name="imagetype" id="imagetype" value="bp" <?echo ($this->type == "bp") ? "checked " : ""?>onClick="result_request()">Bandpass<br>
            <input type="radio" name="imagetype" id="imagetype" value="ts" <?echo ($this->type == "ts") ? "checked " : ""?>onClick="result_request()">Time Series<br>
            <input type="radio" name="imagetype" id="imagetype" value="fft" <?echo ($this->type == "fft") ? "checked " : ""?>onClick="result_request()">Fluct. PS<br>
            <input type="radio" name="imagetype" id="imagetype" value="pvf" <?echo ($this->type == "pvf") ? "checked " : ""?>onClick="result_request()">Phase v Freq<br>
          </form>
        </td>
        
        <?$this->echoBeam(13)?>
        <?$this->echoBlank()?>
        <?$this->echoBeam(12)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(6)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(7)?>
        <?$this->echoBeam(5)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(8)?>
        <?$this->echoBeam(1)?>
        <?$this->echoBeam(11)?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(2)?>
        <?$this->echoBeam(4)?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(3)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(9)?>
        <?$this->echoBeam(10)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBlank()?>
        <?$this->echoBlank()?>
      </tr>
    </table>
  </center>
<?
    $this->closeBlockHeader();

    echo "</td><td valign=top height=10px>\n";

    $this->openBlockHeader("Obs info");
    
    if (is_numeric($state["BEAM_SIZE"]))
    {
      $beam_size = sprintf("%5.2f", ($state["BEAM_SIZE"] / 1024))." GB";
      $obs_size = sprintf("%5.2f", (($state["BEAM_SIZE"] * $obs_info["NUM_PWC"]) / 1024))." GB";
      if ($header["PROC_FILE"] == "SURVEY.MULTIBEAM")
        $obs_length = ($state["BEAM_SIZE"] / 4)." s";
      else
        $obs_length = "Unsure";
    }
    else
    {
      $beam_size = $state["BEAM_SIZE"];
      $obs_size = $state["BEAM_SIZE"];
      $obs_length = $state["BEAM_SIZE"];

    }

?>
    <table class="result">
      <tr><td>UTC_START</td><td><?echo $obs_info["UTC_START"]?></td></tr>
      <tr><td>SOURCE</td><td><?echo $obs_info["SOURCE"]?></td></tr>
      <tr><td>RA</td><td><?echo $obs_info["RA"]?></td></tr>
      <tr><td>DEC</td><td><?echo $obs_info["DEC"]?></td></tr>
      <tr><td>NUM_BEAMS</td><td><?echo $obs_info["NUM_PWC"]?></td></tr>
      <tr><td>PID</td><td><?echo $obs_info["PID"]?></td></tr>
      <tr><td>ARCHIVAL STATE</td><td><?echo $state["ARCHIVAL_STATE"]?></td></tr>
      <tr><td>BEAM SIZE</td><td><?echo $beam_size?></td></tr>
      <tr><td>OBS SIZE</td><td><?echo $obs_size?></td></tr>
      <tr><td>OBS LENGTH</td><td><?echo $obs_length?></td></tr>
      <tr><td>Results Dir</td><td><a href="<?echo $this->results_link?>">Link</a></td></tr>
    </table>

<?
    $this->closeBlockHeader();

    echo "</td></tr><tr><td valign=top>\n";

    $this->openBlockHeader("Header");
?>
    <table class="result">
      <tr><td>BANDWIDTH</td><td><?echo $header["BW"]?></td></tr>
      <tr><td>CFREQ</td><td><?echo $header["CFREQ"]?></td></tr>
      <tr><td>PROC_FILE</td><td><?echo $header["PROC_FILE"]?></td></tr>
      <tr><td>NBIT</td><td><?echo $header["NBIT"]?></td></tr>
      <tr><td>NDECI_BIT</td><td><?echo $header["NDECI_BIT"]?></td></tr>
      <tr><td>NDIM</td><td><?echo $header["NDIM"]?></td></tr>
      <tr><td>NPOL</td><td><?echo $header["NPOL"]?></td></tr>
      <tr><td>TSAMP</td><td><?echo $header["TSAMP"]?></td></tr>
      <tr><td>STATE</td><td><?echo $header["STATE"]?></td></tr>
      <tr><td>BYTES_P/S</td><td><?echo $header["BYTES_PER_SECOND"]?></td></tr>
    </table>
<?
    $this->closeBlockHeader();
?>
    </td></tr>

    <tr><td colspan=2>
<?
    $this->openBlockHeader("Transient Pipeline");
?>
    <img src="/images/blankimage.gif" border=0 id="beamall" TITLE="Transients" alt="Transients">
<?
    $this->closeBlockHeader();
?>
    </td></tr>
    </table>

    <script type="text/javascript">
      result_request();
    </script>
<?
  }

  function printUpdateHTML($get)
  {
    $type = $_GET["type"];
    $size = "112x84";
    $results = array();
    $beams = array();

    # get the beam listings for this observation
    $cmd = "find ".$this->results_dir."/".$this->utc_start." -mindepth 1 -maxdepth 1 ".
           "-type d -name '??' -printf '%f\n'";
    $lastline = exec($cmd, $beams, $rval);
    sort($beams);

    $results = array_pop($this->inst->getResults($this->results_dir, 
                          $this->utc_start, $type, $size, $beams));

    for ($i=0; $i<count($beams); $i++) {
      $beam = $beams[$i];
      if (array_key_exists($i, $results))
        $beam_found = true;
      else {
        $beam_found = false;
        $results[$i] = array();
      }

      if (!$beam_found)
        $results[$i][$type."_".$size] = "/bpsr/images/bpsr_beam_disabled_240x180.png";
      else if ($results[$i][$type."_".$size] == "")
        $results[$i][$type."_".$size] = "/images/blankimage.gif";
      else
        ;

      $url = "";
      echo $beam.":::".$size.":::".$type.":::".$url.$results[$i][$type."_".$size]."\n";
    }
    echo "all:::1024x768:::cand:::".$url.$results["transients"]["cands_1024x768"]."\n";
  }

  function echoBlank() 
  {
    echo "<td ></td>\n";
  }

  function echoBeam($beam_no) 
  {
    $beam_str = sprintf("%02d", $beam_no);

    echo "<td rowspan=2 align=right>";
    echo "<a border=0px href=\"javascript:popWindow('beam_viewer.lib.php?single=true&beamid=".$beam_no."')\">";
    echo "<img src=\"/images/blankimage.gif\" border=0 width=113 height=85 id=\"beam".$beam_str."\" TITLE=\"Beam ".$beam_str."\" alt=\"Beam ".$beam_no."\">\n";
    echo "</a></td>\n";
  }

}

handledirect("result");

