<?PHP

# 
# Display a listing of the Pulsars in psrcats and tempos databases/directory
#

include("definitions_i.php");
include("functions_i.php");
include("apsr_functions_i.php");

# Get the system configuration (dada.cfg)
$cfg = getConfigFile(SYS_CONFIG,TRUE);

# Get the listing of psrcat on srv0
$array = array();
$rval = 0;
$cmd = "export PSRCAT_FILE=/home/dada/runtime/psrcat/psrcat.db;/home/dada/linux_64/bin/psrcat -nohead";
$last = exec($cmd, $array, $rval);

$psrcat_db = array();

for ($i=0; $i<count($array); $i++) {
  $parts = split("[ ]+", $array[$i]);
  $src = $parts[1];
  $f0  = $parts[3];
  $p0  = $parts[6];
  $psrcat_db[$src] = array("f0" => $f0, "p0" => $p0);
}


?>
<html>
<?

$title = "APSR | PSR Catalogue";
include("header_i.php");

?>

<body>
<? 
$text = "APSR Catalogue";
include("banner.php");

?>
<center>
<table>
<tr>
  <th colspan=3>PSRCAT DB</th>
</tr>
<tr>
  <th>Source</th>
  <th>Period [ms]</th>
  <th>DM</th>
</tr>

<?

$keys = array_keys($psrcat_db);

for ($i=0; $i < count($keys); $i++) {

  $k = $keys[$i];
  $psr = $psrcat_db[$k];
  if ($psr["f0"] != 0)
    $period = sprintf("%5.4f", (1000.0 / $psr["f0"]));
  else 
    $period = "NA";

  echo "<tr><td>".$k."</td><td>".$period."</td><td>".$psr["p0"]."</td></tr>\n";
}
?>
</table>

</body>
</html>
