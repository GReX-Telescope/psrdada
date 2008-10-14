<?PHP 
include("definitions_i.php");
include("functions_i.php");
$config = getConfigFile(SYS_CONFIG);
$imgtype = $_GET["type"];

/* Find the latest files in the plot file directory */
for ($i=0; $i<$config["NUM_PWC"]; $i++) {
  $lowres_imgs[$i] = "/images/blankimage.gif";
  $midres_imgs[$i] = "/images/blankimage.gif";
  $hires_imgs[$i] = "/images/blankimage.gif";
}

/* Determine the most recent result */
$results_dir = $_GET["results_dir"];
$cmd = "ls -1 ".$results_dir." | tail -n 1";
$result = exec($cmd);
$dir = $results_dir."/".$result;

/* now find the 13 files requested */
if ($handle = opendir($dir)) {
  while (false !== ($file = readdir($handle))) {
    if ($file != "." && $file != "..") {

      /* If this is a beam?? subdirectory */
      if ( (is_dir($dir."/".$file)) && (ereg("^([0-9][0-9])$", $file)) ) {

        /* Get into a relative dir... */
        chdir($dir);
        $beamid = (int) $file;

        /* Find the hi res images */
        $cmd = "find ".$file." -name \"*.".$imgtype."_1024x768.png\"";
        $find_result = exec($cmd, $array, $return_val);
        if (($return_val == 0) && (strlen($find_result) > 1)) {
          $hires_imgs[($beamid-1)] = "/bpsr/results/".$result."/".$find_result;
        }

        /* Find the mid res images */
        $cmd = "find ".$file." -name \"*.".$imgtype."_400x300.png\"";
        $find_result = exec($cmd, $array, $return_val);
        if (($return_val == 0) && (strlen($find_result) > 1)) {
          $midres_imgs[($beamid-1)] = "/bpsr/results/".$result."/".$find_result;
        }

        /* Find the low res images */
        $cmd = "find ".$file." -name \"*.".$imgtype."_112x84.png\"";
        $find_result = exec($cmd, $array, $return_val);
        if (($return_val == 0) && (strlen($find_result) > 1))  {
          $lowres_imgs[($beamid-1)] = "/bpsr/results/".$result."/".$find_result;
        }

      }
    }
  }
  closedir($handle);
} else {
  echo "Could not open plot directory: ".$dir."<BR>\n";
}

$url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

for ($i=0; $i<$config["NUM_PWC"]; $i++) {
  echo "img".$i.":::".$url.$midres_imgs[$i].":::".$url.$lowres_imgs[$i].";;;";
}
echo $result.";;;";
