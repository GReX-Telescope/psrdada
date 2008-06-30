<?PHP

function getResultsInfo($utc_start, $results_dir) {

  $obs_dir = $results_dir."/".$utc_start;

  $data = array();

  $freq_channels = getSubDirs($obs_dir);

  $data["nbeams"] = count($freq_channels);

  $cmd = "find ".$obs_dir." -name \"obs.start\" | tail -n 1";
  $an_obs_start = exec($cmd);
  $data["obs_start"] = $an_obs_start;

  if (file_exists($obs_dir."/obs.txt")) {
    $data["annotation"] = exec("cat ".$obs_dir."/obs.txt");
  } else {
    $data["annotation"] = "";
  }

  $images = array("bandpass", "timeseries", "powerspectrum", "digitizer");

  for ($i=0; $i<count($freq_channels); $i++) {

    # For each channel, check for the existence of images
    $dir = $obs_dir."/".$freq_channels[$i];

    $data[$i]["dir"] = "/results/".$utc_start."/".$freq_channels[$i];

    for ($j=0; $j<count($images); $j++) {
      $data[$i][$images[$j]."_low"] = "../../../images/blankimage.gif";
      $data[$i][$images[$j]."_med"] = "../../../images/blankimage.gif";
      $data[$i][$images[$j]."_hi"] = "../../../images/blankimage.gif";
    }

    $files = array();

    if (is_dir($dir)) {
      if ($dh = opendir($dir)) {
        while (($file = readdir($dh)) !== false) {

          if ($file != "." && $file != "..") {

            for ($j=0; $j<count($images); $j++) {
              if (ereg("^".$images[$j]."_([A-Za-z0-9\_\:-]*)112x84.png$",$file)) {
                $data[$i]["bandpass_low"] = $file;
              }
              if (ereg("^".$images[$j]."_([A-Za-z0-9\_\:-]*)400x300.png$",$file)) {
                $data[$i]["bandpass_med"] = $file;
              }
              if (ereg("^".$images[$j]."_([A-Za-z0-9\_\:-]*)1024x768.png$",$file)) {
                $data[$i]["bandpass_hi"] = $file;
              }
            }
          }
        }
        closedir($dh);
      }
    }

    /* If no archives have been produced */
    if (count($freq_channels) == 0) {
      $all_results[$observations[$i]]["obs_start"] = "unset";
    }
  } 





  return $data;
}

?>
