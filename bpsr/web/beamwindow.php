<?PHP

include("../definitions_i.php");
include("../functions_i.php");

$obsid = $_GET["obsid"];    // UTC Start
$beam_id = sprintf("%02d", $_GET["beamid"]); // Beam number [1-13]

$config = getConfigFile(SYS_CONFIG);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);



?>
<html>
<head>
   <title>BPSR | Result <?echo $observation?></title>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>
</head>
<body>

<? 

include("../banner.php"); 

$base_dir = $config["SERVER_RESULTS_DIR"]."/".$obsid."/beam".$beam_id;

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

  $img_base = "/results/".$obsid."/beam".$beam_id."/";

  $obs_info =  $config["SERVER_RESULTS_DIR"]."/".$obsid."/obs.info";
  $obs_start = $base_dir."/obs.start";

  chdir ($base_dir);

  $bandpass = exec("ls -1tr bandpass_*_240x180.png");
  $timeseries = exec("ls -1tr dm0timeseries*_240x180.png");
  $powerspectrum = exec("ls -1tr powerspectrum*_240x180.png");
  $digitizer = exec("ls -1tr digitizer*_240x180.png");

  $bandpass = $img_base.$bandpass;
  $timeseries = $img_base.$timeseries;
  $powerspectrum = $img_base.$powerspectrum;
  $digitizer = $img_base.$digitizer;

?>

<table>

<tr><td valign=center>Beam <?echo $beam_id?> Information: </td></tr>

<tr>
  <td>
    Bandpass<br>
    <img src="<?echo $bandpass?>">
  </td>
  <td>
    DM0 Timeseries<br>
    <img src="<?echo $timeseries?>">
  </td>
</tr>

<tr>
  <td>
    Fluctuation Power Spectrum<br>
    <img src="<?echo $powerspectrum?>">
  </td>
  <td>
    Digitizer Statistics<br>
    <img src="<?echo $digitizer?>">
  </td>
</tr>

</table>

<? 
}
?>

</body>
</html>
