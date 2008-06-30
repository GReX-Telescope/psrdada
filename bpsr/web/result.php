<?PHP

include("../definitions_i.php");
include("../functions_i.php");
include("bpsr_functions_i.php");

$config = getConfigFile(SYS_CONFIG);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

$utc_start = $_GET["utc_start"];
if (! isset($_GET["imagetype"])) {
  $imagetype = "bandpass";
} else {
  $imagetype = $_GET["imagetype"];
}

$data = getResultsInfo($utc_start, $config["SERVER_RESULTS_DIR"]);
$nbeam = $data["nbeams"];
$header = getConfigFile($data["obs_start"], TRUE);


?>
<html>
<head>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>

</head>
<body>
<script type="text/javascript" src="/js/wz_tooltip.js"></script>

<center>
<input id="utc_start" type="hidden" value="">
<table border=1>
<tr>
  <td>
    <table border=1> 
<?
        echo "    <tr><td>UTC_START</td><td>".$header["UTC_START"]."</td></tr>\n";
        echo "    <tr><td>SOURCE</td><td>".$header["SOURCE"]."</td></tr>\n";
        echo "    <tr><td>RA</td><td>".$header["RA"]."</td></tr>\n";
        echo "    <tr><td>DEC</td><td>".$header["DEC"]."</td></tr>\n";
        echo "    <tr><td>FA</td><td>".$header["FA"]."</td></tr>\n";
?>
    </table>
<?printHeader($header);?>
  </td>

    <td>
      Display<br>
      <form name="imageform">
      <input type="radio" name="imagetype" id="imagetype" value="bandpass" checked onClick="request()">Bandpass<br>
      <input type="radio" name="imagetype" id="imagetype" value="timeseries" onClick="request()">Time Series<br>
      <input type="radio" name="imagetype" id="imagetype" value="powerspectrum" onClick="request()">Power Spectrum<br>
      <input type="radio" name="imagetype" id="imagetype" value="digitizer" onClick="request()">Digitizer Statistics<br>
      </form>
    </td>
</tr>
</table>
</center>

<center>
<table border=0 cellspacing=5 cellpadding=5>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(13, $nbeam, $imagetype, $data)?>
    <?echoBlank()?>
    <?echoBeam(12, $nbeam, $imagetype, $data)?>
    <?echoBlank()?> 
  </tr>
  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(6, $nbeam, $imagetype, $data)?>
    <?echoBlank()?>
  </tr>
  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(7, $nbeam, $imagetype, $data)?>
    <?echoBeam(5, $nbeam, $imagetype, $data)?>
    <?echoBlank()?> 
  </tr>

  <tr height=42>
    <?echoBeam(8, $nbeam, $imagetype, $data)?>
    <?echoBeam(1, $nbeam, $imagetype, $data)?>
    <?echoBeam(11, $nbeam, $imagetype, $data)?>
  </tr>

  <tr height=42>
    <?echoBeam(2, $nbeam, $imagetype, $data)?>
    <?echoBeam(4, $nbeam, $imagetype, $data)?>
  </tr>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(3, $nbeam, $imagetype, $data)?>
    <?echoBlank()?>
  </tr>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(9, $nbeam, $imagetype, $data)?>
    <?echoBeam(10, $nbeam, $imagetype, $data)?>
    <?echoBlank()?>
  </tr>
  
  <tr height=42>
    <?echoBlank()?>
    <?echoBlank()?>
    <?echoBlank()?>
  </tr>
</table>
</center>



</body>
</html>

<?

function echoBlank() {

  echo "<td ></td>\n";
}

function echoBeam($beam_no, $num_beams, $imagetype, $data) {

  if ($beam_no <= $num_beams) {

    $mousein = "onmouseover=\"Tip('<img src=".$data[($beam_no-1)]["dir"]."/".$data[($beam_no-1)][$imagetype."_med"]." width=400 height=300>')\"";
    $mouseout = "onmouseout=\"UnTip()\"";

    echo "<td rowspan=2>";
    echo "<a href=\"javascript:popWindow('bpsr/beamwindow.php?beamid=".$beam_no."')\">";

    echo "<img src=\"".$data[($beam_no-1)]["dir"]."/".$data[($beam_no-1)][$imagetype."_low"]."\" width=120 height=90 id=\"beam".$beam_no."\" TITLE=\"Beam ".$beam_no."\" alt=\"Beam ".$beam_no."\" ".$mousein." ".$mouseout.">\n";
    echo "</a></td>\n";
  } else {
    echo "<td rowspan=2></td>\n";
  }

}

function printHeader($header) {

  $keys = array_keys($header);
  $keys_to_ignore = array("HDR_SIZE","FILE_SIZE","HDR_VERSION","FREQ","RECV_HOST","CONFIG",
                          "DSB", "INSTRUMENT", "NBIT", "NDIM", "NPOL", "PROC_FILE", "TELESCOPE",
                          "BEAM");
  $keys_to_ignore = array("BEAM");
  $header_text = "<table cellpadding=0 cellspacing=0><tr><th align=right>Key</th><td width=10>&nbsp;</td><th align=left>Value</th></tr>";

  for ($i=0; $i<count($keys); $i++) {
    if (!(in_array($keys[$i], $keys_to_ignore))) {
      $header_text .= "<tr><td align=right>".$keys[$i]."</td><td></td><td>".$header[$keys[$i]]."</td></tr>";
    }
  }
  $header_text .= "</table>";

  $mousein = "onmouseover=\"Tip('".$header_text."')\"";
  $mouseout = "onmouseout=\"UnTip()\"";
  echo "<div color=blue ".$mousein." ".$mouseout.">DADA HEADER</div>\n";

}

