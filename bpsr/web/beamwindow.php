<?PHP

include("../definitions_i.php");
include("../functions_i.php");

$obsid = $_GET["obsid"];    // UTC Start
$beam_id = sprintf("%02d", $_GET["beamid"]); // Beam number [1-13]

$config = getConfigFile(SYS_CONFIG);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

$text = "";
$base_dir = $config["SERVER_RESULTS_DIR"]."/".$obsid."/".$beam_id;

if (file_exists($base_dir)) {
  $nbeams = exec("find ".$config["SERVER_RESULTS_DIR"]."/".$obsid."/* -type d | wc -l");
                                                                                                                                                
  $img_base = "/results/".$obsid."/".$beam_id."/";
                                                                                                                                                
  $obs_info =  $config["SERVER_RESULTS_DIR"]."/".$obsid."/obs.info";
  $obs_start = $base_dir."/obs.start";
                                                                                                                                                
  $data = getImages($base_dir, $img_base);
  if ( file_exists($obs_start) ) {
    $header = getConfigFile($obs_start);
    $text = "Beam ".$beam_id." for ".$header["SOURCE"];
  } else {
    $header = array();
    $text = "Beam ".$beam_id." for ".$obsid;
  }

} else {
  $text = "No observation found";
}

?>
<html>
<head>
  <title>BPSR | Result <?echo $obsid?></title>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>
</head>
<body>
  <!--  Load tooltip module -->
  <script type="text/javascript" src="/js/wz_tooltip.js"></script>


<? 

include("../banner.php"); 

if (! (file_exists($base_dir))) {

?>
<center>
<table>
  <tr> <td colspan=2 bgcolor=red>Could not find the observation</td></tr>
  <tr> <td>UTC_START</td><td><?echo $obsid?></td> </tr>
  <tr> <td>BEAM_ID</td><td><?echo $beam_id?></td> </tr>
  <tr> <td>BASE_DIR</td><td><?echo $base_dir?></td> </tr>
</table>
</center>
<?

} else {

?>

  <table border=0>
    <tr valign=middle><th>Beam</th>
<?

  for ($i=1; $i <= $nbeams; $i++) {
    if ($i != $_GET["beamid"]) {
      echo "<td width=40><div class=\"btns\">\n";
      echo "<a href=\"/bpsr/beamwindow.php?beamid=".$i."&obsid=".$obsid."\" class=\"btn\" > <span>".sprintf("%02d",$i)."</span></a>";
      echo "</div></td>";
    } else {
      echo "<td width=40 align=center><b>".sprintf("%02d",$i)."</b></td>";
    }
  }
?>
  </tr></table>

<table cellpadding=5>
<tr><td valign=top>

<table class="datatable" style="width:300px">
<tr><th>Param</th><th>Value</th></tr>
<tr><td>Source</td><td align=left><?echo $header["SOURCE"]?></td></tr>
<tr><td>UTC_START</td><td align=left><?echo $obsid?></td></tr>
<tr><td>ACC_LEN</td><td align=left><?echo $header["ACC_LEN"]?></td></tr>
<tr><td>RA</td><td align=left><?echo $header["RA"]?></td></tr>
<tr><td>DEC</td><td align=left><?echo $header["DEC"]?></td></tr>
<tr><td>FA</td><td align=left><?echo $header["FA"]?></td></tr>
<tr><td>Beam</td><td align=left><?echo $beam_id?> of <?echo $nbeams?></td></tr>
</table>

</td><td>

<table cellpadding=8>

<tr>
  <td align=center>
    Bandpass<br>
<?  echo imageWithRollover($data["bandpass_mid"], 240,180,$data["bandpass_mid"], 400, 300); ?>

  </td>
  <td align=center>
    DM0 Timeseries<br>
<?  echo imageWithRollover($data["timeseries_mid"], 240,180,$data["timeseries_mid"], 400, 300); ?>
  </td>
</tr>

<tr>
  <td align=center>
    Fluctuation Power Spectrum<br>
<?  echo imageWithRollover($data["powerspectrum_mid"], 240,180,$data["powerspectrum_mid"], 400, 300); ?>
  </td>
  <td align=center>
    Digitizer Statistics<br>
<?  echo imageWithRollover($data["digitizer_mid"], 240,180,$data["digitizer_mid"], 400, 300); ?>
  </td>
</tr>

</table>

</td></tr></table>

<? 
}
?>

</body>
</html>

<?

function getImages($dir, $img_base) {

  $data = array();

  /* Find the latest files in the plot file directory */
  $types = array("bandpass", "timeseries", "powerspectrum", "digitizer");

  for ($i=0; $i<count($types); $i++) {
    $data[$types[$i]."_low"] = "/images/blankimage.gif";
    $data[$types[$i]."_mid"] = "/images/blankimage.gif";
    $data[$types[$i]."_hi"] = "/images/blankimage.gif";
  }

  /* Get into a relative dir... */
  $cwd = getcwd();
  chdir($dir);

  for ($i=0; $i<count($types); $i++) {
    /* Find the hi res images */
    $cmd = "find . -name \"".$types[$i]."_*_1024x768.png\" -printf \"%P\"";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1)) {
      $data[$types[$i]."_hi"] = $img_base.$find_result;
    }

    /* Find the low res images */
    $cmd = "find . -name \"".$types[$i]."_*_400x300.png\" -printf \"%P\"";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1))  {
      $data[$types[$i]."_mid"] = $img_base.$find_result;
    }

    /* Find the low res images */
    $cmd = "find . -name \"".$types[$i]."_*_112x84.png\" -printf \"%P\"";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1))  {
      $data[$types[$i]."_low"] = $img_base.$find_result;
    }
  }
  chdir($cwd);

  return $data;
}

function imageWithRollover($img_low, $img_low_x, $img_low_y, $img_hi, $img_hi_x, $img_hi_y) {


  $string = "";

  if ($img_hi != "/images/blankimage.gif") {
    $mousein = "onmouseover=\"Tip('<img src=\'".$img_hi."\' width=".$img_hi_x." height=".$img_hi_y.">')\"";
    $mouseout = "onmouseout=\"UnTip()\"";
    $string = "<a href=".$img_hi.">"; 
  }

  $string .= "<img src=\"".$img_low."\" width=".$img_low_x." height=".$img_low_y." ".$mousein." ".$mouseout.">";

  if ($img_hi != "/images/blankimage.gif") {
    $string .= "</a>\n";
  }

  return $string;

}
?>
