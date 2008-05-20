<?PHP 
include("../definitions_i.php");
include("../functions_i.php");
$config = getConfigFile(SYS_CONFIG);
$imgtype = $_GET["type"];

/* Find the latest files in the plot file directory */
for ($i=0; $i<$config["NUM_PWC"]; $i++) {
  if ($imgtype == "bandpass") {
    $imgs[$i] = "../../images/fakebandpass.png";
  } else {
    $imgs[$i] = "../../images/blankimage.gif";
  }
}

/* Determine the most recent result */
$results_dir = $_GET["results_dir"];
$cmd = "ls -trA ".$results_dir." | tail -n 1";
$result = exec($cmd);
$dir = $results_dir."/".$result;

/* now find the 13 files requested */
if ($handle = opendir($dir)) {
  while (false !== ($file = readdir($handle))) {
    if ($file != "." && $file != "..") {

      /* If this is a beam?? subdirectory */
      if ( (is_dir($dir."/".$file)) && (ereg("^beam([0-9][0-9])$", $file)) ) {

        /* Get into a relative dir... */
        chdir($dir);

        $beamid = (int) substr($file,4,2);

        /* Find the filename we want */
        $cmd = "find ".$file." -name \"".$imgtype."_*_1024x768.png\"";
        $find_result = exec($cmd, $array, $return_val);
        if ($return_val == 0) {
          $imgs[($beamid-1)] = $find_result;
        }
      }
    }
  }
  closedir($handle);
} else {
  echo "Could not open plot directory: ".$dir."<BR>\n";
}

$url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."/results/".$result."/";

for ($i=0; $i<$config["NUM_PWC"]; $i++) {
  echo "img".$i.":::".$url.$imgs[$i].";;;";
  $imgs[$i] = "../../images/blankimage.gif";
}
echo $result.";;;";
