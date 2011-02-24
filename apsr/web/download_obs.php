<?PHP

/* 
 *  Download specified archive[s]
 */

include("definitions_i.php");
include("functions_i.php");
#ini_set("memory_limit","128M");

$cfg = getConfigFile(SYS_CONFIG,TRUE);

$utc_starts = array();
if (isset($_GET["utc_starts"])) {
  $utc_starts= explode(",", $_GET["utc_starts"]);
} 

if (count($utc_starts) == 1) {
  $filename = "APSR_".$utc_starts[0].".tar";
} else {
  $filename = "APSR_Multiple_Obs.tar";
}

header('Cache-Control: no-cache, must-revalidate');
header('Pragma: no-cache');
header('Content-type: application/x-tar');
header('Content-Disposition: attachment; filename="'.$filename.'"');

$dir = $cfg["SERVER_ARCHIVE_DIR"];
$cmd = "tar -ch --exclude obs.* --exclude sent.to.* --exclude band.?res";

for ($i=0; $i<count($utc_starts); $i++) 
  $cmd .= " ".$utc_starts[$i];

passthru("cd ".$dir."; ".$cmd);

?>
