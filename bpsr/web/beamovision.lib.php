<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

class beamovision extends bpsr_webpage 
{

  var $beams;
  var $beam_utcs;
  var $inst;

  function beamovision()
  {
    bpsr_webpage::bpsr_webpage();

    $this->inst = new bpsr();
    $this->title = "BPSR | Beam-O-Vision ".$_GET["utc_start"];
    $this->beams = array();
    $this->beam_utcs = array();

    # parse the BEAM UTC's
    for ($i=1; $i<=13; $i++) {
      $b = sprintf("%02d",$i);
      if (isset($_GET["BEAM_".$i])) {
        array_push($this->beams, $b);
        array_push($this->beam_utcs, $_GET["BEAM_".$i]);
      }
    }
  }

  function printJavaScriptHead()
  {

?>
    <style type="text/css">
      table.beamovision th {
        text-align: left;
        font-size: 8pt;
      }
      table.beamovision td {
        text-align: left;
        padding-left: 10px;
        padding-right: 50px;
        font-size: 8pt;
      }
      table.beamovision span {
        font-size: 8pt;
      }
    </style>


    <script type='text/javascript'>  

      var npsrs = 0;
      var utc_start = "";
      var psrs = new Array();

      function popWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1400,height=700');");
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $num_beams = count($this->beams);
    $basedir = $this->inst->config["SERVER_RESULTS_DIR"];
    $archives_dir = $this->inst->config["SERVER_ARCHIVE_DIR"];
    $types = "pvf";
    $sizes = array("400x300","112x84");
    $beams = "all";

    # Get the pvf images for the specified utc's
    $images = $this->inst->getResults($basedir, $this->beam_utcs, $types, $sizes, $beams);

    # Determine the SNRS for each beam
    for ($i=0; $i<count($this->beams); $i++)
    {
      $archive = $archives_dir."/".$this->beam_utcs[$i]."/".$this->beams[$i]."/integrated.ar";
      if (file_exists($archive)) 
        $snrs[$i] = instrument::getSNR($archive);
      else
        $snrs[$i] = "N/A";
    }

    # Get some information from the header
    if (array_key_exists(0,$this->beams)) {
      $data = $this->inst->getResultsInfo($this->beam_utcs[0], $basedir);
      $header = $this->inst->configFileToHash($data[$this->beam_utcs[0]]["obs_start"]);
      $obs_info_file = $basedir."/".$this->beam_utcs[0]."/obs.info";
      $obs_info = $this->inst->configFileToHash($obs_info_file);
    }

    /*
    for ($i=1; $i<=13; $i++) {
      $b = sprintf("%02d",$i);
      if ($this->beam_utcs[$b] == "") {
        $this->beam_utcs[$b] = "NONE";
      }
    }
    */

?>
    <table border=0 cellspacing=10px>
      <tr><td rowspan=2>
<?

    $this->openBlockHeader("Beam-O-Vision");
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=5>

      <tr>
        <td rowspan=3 valign="top" align='left'>
        </td>
        
        <?$this->echoBeam(13, $images, $this->beam_utcs[12], $snrs[12])?>
        <?$this->echoBlank()?>
        <?$this->echoBeam(12, $images, $this->beam_utcs[11], $snrs[11])?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(6, $images, $this->beam_utcs[5], $snrs[5])?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(7, $images, $this->beam_utcs[6], $snrs[6])?>
        <?$this->echoBeam(5, $images, $this->beam_utcs[4], $snrs[4])?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(8, $images, $this->beam_utcs[7], $snrs[7])?>
        <?$this->echoBeam(1, $images, $this->beam_utcs[0], $snrs[0])?>
        <?$this->echoBeam(11, $images, $this->beam_utcs[10], $snrs[10])?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(2, $images, $this->beam_utcs[1], $snrs[1])?>
        <?$this->echoBeam(4, $images, $this->beam_utcs[3], $snrs[3])?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(3, $images, $this->beam_utcs[2], $snrs[2])?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(9, $images, $this->beam_utcs[8], $snrs[8])?>
        <?$this->echoBeam(10, $images, $this->beam_utcs[9], $snrs[9])?>
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

?>
    <table class="beamovision">
      <tr><td>UTC_START</td><td><?echo $obs_info["UTC_START"]?></td></tr>
      <tr><td>SOURCE</td><td><?echo $obs_info["SOURCE"]?></td></tr>
      <tr><td>RA</td><td><?echo $obs_info["RA"]?></td></tr>
      <tr><td>DEC</td><td><?echo $obs_info["DEC"]?></td></tr>
      <tr><td>NUM_BEAMS</td><td><?echo $obs_info["NUM_PWC"]?></td></tr>
      <tr><td>PID</td><td><?echo $obs_info["PID"]?></td></tr>
    </table>

<?
    $this->closeBlockHeader();

    echo "</td></tr><tr><td valign=top>\n";

    $this->openBlockHeader("Header");
?>
    <table class="beamovision">
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
    <table>
<?
    $this->closeBlockHeader();

    echo "</td></tr></table>\n";
  }

  function echoBlank() 
  {
    echo "<td ></td>\n";
  }

  function echoBeam($beam_no, $images, $utc_start, $snr) 
  {

    $beam_str = sprintf("%02d", $beam_no);

    if (array_key_exists($utc_start, $images)) {
      $img_med = $images[$utc_start][($beam_no-1)]["pvf_400x300"];
      $img_low = $images[$utc_start][($beam_no-1)]["pvf_112x84"];
    } else {
      $img_med = "/images/blankimage.gif";
      $img_low = "/images/blankimage.gif";
    }

    $mousein = "onmouseover=\"Tip('<img src=".$img_med." width=400 height=300>')\"";
    $mouseout = "onmouseout=\"UnTip()\"";

    echo "<td rowspan=2 class=\"multibeam\" height=84>";
    echo "<a class=\"multibeam\" href=\"javascript:popWindow('/bpsr/beam_viewer.lib.php?single=true&utc_start=".$utc_start."&beam=".$beam_str."')\">";

    echo "<img src=\"".$img_low."\" width=112 height=84 id=\"beam".$beam_str."\" border=0 TITLE=\"Beam ".$beam_str."\" alt=\"Beam ".$beam_str."\"".$mousein." ".$mouseout.">\n";

    echo "</a><br>SNR: ".$snr;
    echo "</td>\n";

  }

}

handledirect("beamovision");

