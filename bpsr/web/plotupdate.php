<?PHP 

include ("bpsr.lib.php");

$inst = new bpsr();
$config = $inst->config;

/* We may want all the current images for just one beam
   or we may want all the latest images of one type for
   all beams */

$types = array("bp","ts","fft","dts","pdbp","pvf");
$sizes = array("112x84", "400x300", "1024x768");
$beams = array();

# get a listing of the currently configured beams
for ($i=0; $i<$config["NUM_PWC"]; $i++) {
  $beams[$i] = $inst->getBeamForPWCHost($config["PWC_".$i]);
}

$obs  = "latest";

$n_results = $config["NUM_PWC"];
$i_result  = 0;
$url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

# Parse the GET arguments
if ((isset($_GET["type"])) && ($_GET["type"] != "all"))
  $types = split(",",$_GET["type"]);

if ((isset($_GET["beam"])) && ($_GET["beam"] != "all")) {
  $beams = split(",",$_GET["beam"]);
}

if ((isset($_GET["size"])) && ($_GET["size"] != "all"))
  $sizes = split(",",$_GET["size"]);

if (isset($_GET["obs"]))
  $obs = $_GET["obs"];

/* Special case for the web interface's front page */
if (($obs == "latest") && 
    (count($sizes) == 1) && ($sizes[0] ==  "112x84") &&
    (count($types) == 1) && ($_GET["beam"] == "all")) {

  $host = $config["SERVER_HOST"];
  $port = $config["SERVER_WEB_MONITOR_PORT"];

  list ($socket, $result) = openSocket($host, $port);

  if ($result == "ok") {

    $type = $types[0];
    $size = $sizes[0];

    $bytes_written = socketWrite($socket, $type."_img_info\r\n");
    $string = socketRead($socket);
    socket_close($socket);

    # Add the require URL links to the image
    $lines = split(";;;", $string);
    $string = "";
    for ($i=0; $i<count($lines)-1; $i++) {

      $parts = split(":::", $lines[$i]);
      $node = -1;
      if ($type == "pdbp") {
        for ($j=0; $j<$inst->ibobs["NUM_IBOB"]; $j++) {
          if ($parts[0] == $inst->ibobs["CONTROL_IP_".$j]) {
            $node = $inst->ibobs["BEAM_".$j];
          } 
        }
      } else {
        $node = $parts[0];
      }
      if ($node != -1) {
        $string .= $node.":::".$size.":::".$type.":::".$url."/bpsr/results/".$parts[1]."\n";;
      } else {
        $string .= $node.":::".$size.":::".$type.":::".$url."/images/blankimage.gif\n";
      }

    }

  } else {
    $string = "Could not connect to $host:$port<BR>\n";
  }

  echo $string;

} else {

  $result = "";
  $results_dir = $config["SERVER_RESULTS_DIR"];

  if ($obs == "latest") {
    $cmd = "ls -I stats -1 ".$results_dir." | tail -n 1";
    $result = exec($cmd);
  } else {
    $result = $obs;
  }

  /* If we want all image types for one beam only (beamwindow.php) */
  if ((count($beams) == 1) && ($_GET["type"] == "all")) {

    # find the ibob control IP that corresponds to the specified beam
    $beam_str = sprintf("%02d",$beams[0]);
    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++) {
      if ($inst->ibobs["BEAM_".$i] == $beam_str) {
        $ibob = $inst->ibobs["CONTROL_IP_".$i];
      }
    }

    $obs_results = $inst->getResults($results_dir, $obs, "all", "all", $beams[0]);

    $actual_obs_results = array_pop(array_pop($obs_results));

    $stats_results = $inst->getStatsResults($results_dir, $ibob);  

    $actual_stats_results = array_pop($stats_results);

    $results = array_merge($actual_obs_results, $actual_stats_results);

    $types = array("bp","ts","fft","dts","pdbp","pvf");
    $sizes = array("112x84", "400x300", "1024x768");
    # $sizes = array("112x84", "400x300");

    for ($i=0; $i<count($types); $i++) {
      for ($j=0; $j<count($sizes); $j++) {
        $key = $types[$i]."_".$sizes[$j];
        echo $beam_str.":::".$sizes[$j].":::".$types[$i].":::".$results[$key]."\n";
      }
    }

  /* We want all beams, but just one type of image (otherstuff.php) */
  } else {

    $results = array();
    $sizes = array("112x84", "400x300", "1024x768");
    # $sizes = array("112x84", "400x300");

    if ((count($types) == 1) && ($types[0] == "pdbp")) {
      $type = $types[0];
      $results = getBPSRStatsResults($results_dir, "all");
      for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++) {
        $ibob = $config["CONTROL_IP_".$i];
        for ($j=0; $j<count($sizes); $j++) {
          $size = $sizes[$j];
          echo $ibob.":::".$size.":::pdbp:::".$url.$results[$ibob][$type."_".$size]."\n";
        }
      }
    } else {

      $results = array_pop($inst->getResults($results_dir, $obs, $types, $sizes, $beams));
      for ($i=0; $i<count($beams); $i++) {
        $beam = $beams[$i];

        if (array_key_exists($i, $results)) 
          $beam_found = true;
        else {
          $beam_found = false;
          $results[$i] = array();
        }

        for ($j=0; $j<count($sizes); $j++) {
          $size = $sizes[$j];
          for ($k=0; $k<count($types); $k++) {
            $type = $types[$k];

            if (!$beam_found)
               $results[$i][$type."_".$size] = "/bpsr/images/bpsr_beam_disabled_240x180.png";
            else if ($results[$i][$type."_".$size] == "") 
              $results[$i][$type."_".$size] = "/images/blankimage.gif";
            else
              ;
            echo $beam.":::".$size.":::".$type.":::".$url.$results[$i][$type."_".$size]."\n";
          }
        }
      }
    }
  }
}


