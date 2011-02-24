<?PHP

function getResultsInfo($observations, $results_dir) {

  if (!is_array($observations)) {
    $observations = array($observations);
  } 

  $data = array();

  foreach ($observations as $i => $o) {

    $data[$o] = array();

    $obs_dir = $results_dir."/".$o;

    $freq_channels = getSubDirs($obs_dir);
    $data[$o]["nbeams"] = count($freq_channels);

    $cmd = "find ".$obs_dir." -name \"obs.start\" | tail -n 1";
    $an_obs_start = exec($cmd);
    $data[$o]["obs_start"] = $an_obs_start;

    if (file_exists($obs_dir."/obs.txt")) {
      $data[$o]["annotation"] = exec("cat ".$obs_dir."/obs.txt");
    } else {
      $data[$o]["annotation"] = "";
    }

    for ($j=0; $j<count($freq_channels); $j++) {
      $data[$o][$j]["dir"] = "/results/".$o."/".$freq_channels[$j];
    }
  } 

  return $data;
}


/*
 * Gets the most recent image/results files from a BPSR observation
 */

function getBPSRResults($results_dir, $utc_start="latest", $type="all", $size="all", $beam="all") {

  $utc_starts = array();
  $types = array();
  $sizes = array();
  $results = array();
  $beams = array();

  if (is_array($utc_start)) {
    $utc_starts = $utc_start;
  } else if ($utc_start == "all") {
    #$cmd = "ls -1 -I stats ".$results_dir;
    #$str = exec($cmd);
    #$utc_starts = split("\n", rtrim($str));
    $cmd = "find ".$results_dir." -maxdepth 1 -type d -name '2*' -printf '%f\\n' | sort";
    $str = exec($cmd, $utc_starts, $rval);
  } else if ($utc_start == "latest") {
    #$cmd = "ls -1 -I stats ".$results_dir." | tail -n 1";
    $cmd = "find ".$results_dir." -maxdepth 1 -type d -name '2*' -printf '%f\\n' | sort | tail -n 1";
    $str = exec($cmd, $utc_starts, $rval);
    $utc_starts = array(exec($cmd));
  } else {
    $utc_starts = array($utc_start);
  }

  if (is_array($type)) {
    $types = $type;
  } else if ($type == "all") {
    $types = array("bp", "ts", "fft", "dts","pvf");
  } else {
    $types = array($type);
  }
                                                                                                             
  if (is_array($size)) {
    $sizes = $size;
  } else if ($size == "all") {
    $sizes = array("1024x768", "400x300", "112x84");
    # $sizes = array("400x300", "112x84");
  } else {
    $sizes = array($size);
  }

  if (is_array($beam)) {
    $beams = $beam;
  } else if ($beam == "all") {
    $config = getConfigFile(SYS_CONFIG);
    for ($i=0; $i<$config["NUM_PWC"]; $i++) {
      array_push($beams, $config["BEAM_".$i]);
    }
  } else {
    $beams = array($beam);
  }

  foreach ($utc_starts as $u) {

    $dir = $results_dir."/".$u;

    /* now find the 13 files requested */
    if ($handle = opendir($dir)) {
      while (false !== ($file = readdir($handle))) {

        if ($file != "." && $file != "..") {

          /* If this is a beam?? subdirectory */
          if ( (is_dir($dir."/".$file)) && (ereg("^([0-9][0-9])$", $file)) ) {

            /* Get into a relative dir... */
            chdir($dir);
            $beamid = (int) $file;

            if (in_array($beamid, $beams)) {

              # Foreach image type
              foreach ($types as $t) {

                foreach ($sizes as $s) {
    
                  $img = "/images/blankimage.gif";
                  /* Find the hi res images */
                  $cmd = "find ".$file." -name \"*.".$t."_".$s.".png\" | sort -n";
                  $find_result = exec($cmd, $array, $return_val);
                  if (($return_val == 0) && (strlen($find_result) > 1)) {
                    $img = "/bpsr/results/".$u."/".$find_result;
                  }
                  $results[$u][($beamid-1)][$t."_".$s] = $img;
                }
              }
            }
          }
        }                                                                                                   
      }
      closedir($handle);
    } else {
      echo "Could not open plot directory: ".$dir."<BR>\n";
    }
  }

  return $results;
}


function getBPSRStatsResults($results_dir, $ibobs){

  $dir = $results_dir."/stats";
  $config = getConfigFile(SYS_CONFIG);

  $results = array();
  if ($ibobs == "all") {
    for ($i=0; $i<$config["NUM_PWC"]; $i++) {
      $results[$config["IBOB_DEST_".$i]] = array();
    }
  } else {
    $results[$ibobs] = array();
  }

  /* now find the 13 files requested */
  if ($handle = opendir($dir)) {

    $files = array();

    # read all the files
    while ($file = readdir($handle)) {
      if ( ($file != ".") && ($file != "..") ) {
        array_push($files, $file);
      }
    }

    closedir($handle);
    rsort($files);

    # Now ensure we have only the most recent files in the array
    foreach ($results as $key => $value) {
      $ibob = $key;
      $have_low = 0;
      $have_mid = 0;
      $have_hi = 0;

      for ($j=0; $j<count($files); $j++) {
        if ((strpos($files[$j], $ibob."_112x84") !== FALSE) && (!$have_low) ){
          $have_low = 1;
          $value["pdbp_112x84"] = "/bpsr/results/stats/".$files[$j];
        }
        if ((strpos($files[$j], $ibob."_400x300") !== FALSE) && (!$have_mid) ){
          $have_mid = 1;
          $value["pdbp_400x300"] = "/bpsr/results/stats/".$files[$j];
        }
        if ((strpos($files[$j], $ibob."_1024x768") != FALSE) && (!$have_hi) ){
          $have_hi = 1;
          $value["pdbp_1024x768"] = "/bpsr/results/stats/".$files[$j];
        }
      }        
      $results[$key] = $value;
    }

  } else {
     echo "Could not open plot directory: ".$dir."<BR>\n";
  }

  return $results;

}


function getServerLogInformation() {

  $arr = array();
  $arr["bpsr_tcs_interface"]          = array("logfile" => "bpsr_tcs_interface.log", "name" => "TCS Interface", "tag" => "server", "shortname" => "TCS");
  $arr["bpsr_results_manager"]        = array("logfile" => "bpsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
  $arr["dada_pwc_command"]            = array("logfile" => "dada_pwc_command.log", "name" => "dada_pwc_command", "tag" => "server", "shortname" => "PWCC");
  $arr["bpsr_multibob_manager"]       = array("logfile" => "bpsr_multibob_manager.log", "name" => "Multibob", "tag" => "server", "shortname" => "Multibob");
  $arr["bpsr_transfer_manager"]       = array("logfile" => "bpsr_transfer_manager.log", "name" => "Transfer Mngr", "tag" => "server", "shortname" =>"Xfer");
  $arr["bpsr_web_monitor"]            = array("logfile" => "bpsr_web_monitor.log", "name" => "Web Monitor", "tag" => "server", "shortname" => "Monitor");
  $arr["bpsr_pwc_monitor"]            = array("logfile" => "nexus.pwc.log", "name" => "PWC", "tag" => "pwc", "shortname" => "PWC");
  $arr["bpsr_sys_monitor"]            = array("logfile" => "nexus.sys.log", "name" => "SYS", "tag" => "sys", "shortname" => "SYS");
  $arr["bpsr_src_monitor"]            = array("logfile" => "nexus.src.log", "name" => "SRC", "tag" => "src", "shortname" => "SRC");
  $arr["bpsr_swin_tape_controller"]   = array("logfile" => "bpsr_swin_tape_controller.log", "name" => "Swin Tape", "tag" => "server", "shortname" => "SwinTape");
  $arr["bpsr_parkes_tape_controller"] = array("logfile" => "bpsr_parkes_tape_controller.log", "name" => "Parkes Tape", "tag" => "server", "shortname" => "ParkesTape");
  return $arr;

}


function getClientLogInformation() {

  $arr = array();
  #$arr["bpsr_master_control"] = array("logfile" => "bpsr_master_control.log", "name" => "master_control";
  $arr["bpsr_observation_manager"] = array("logfile" => "nexus.sys.log", "name" => "Obs Mngr", "tag" => "obs mngr");
  $arr["bpsr_results_monitor"]     = array("logfile" => "nexus.sys.log", "name" => "Results Mon", "tag" => "results mon");
  $arr["processor"]                = array("logfile" => "nexus.src.log", "name" => "Processor", "tag" => "proc");
  $arr["bpsr_disk_cleaner"]        = array("logfile" => "nexus.sys.log", "name" => "Disk Cleaner", "tag" => "cleaner");
  #$arr["bpsr_pwc_monitor"]         = array("logfile" => "nexus.pwc.log", "name" => "PWC", "tag" => "pwc");
  #$arr["bpsr_sys_monitor"]         = array("logfile" => "nexus.sys.log", "name" => "SYS", "tag" => "sys");
  #$arr["bpsr_src_monitor"]         = array("logfile" => "nexus.src.log", "name" => "SRC", "tag" => "src");

  return $arr;

}

function getClientDaemonNames() {

  $arr = array();

  $arr["bpsr_master_control"] = "Master Ctrl";
  $arr["bpsr_observation_manager"] = "Obs. Mngr";
  $arr["bpsr_results_monitor"] = "Results Monitor";
  $arr["processor"] = "Processor";

  return $arr;
}
                                                                                                                                                                              
function getClientDaemonTypes() {
  $arr = array();
                                                                                                                                                                              
  $arr["bpsr_master_control"] = "none";
  $arr["bpsr_observation_manager"] = "sys";
  $arr["bpsr_results_monitor"] = "sys";
  $arr["processor"] = "src";
  return $arr;
}

function getClientDaemonTags() {

  $arr = array();

  $arr["bpsr_master_control"] = "none";
  $arr["bpsr_observation_manager"] = "obs mngr";
  $arr["bpsr_results_monitor"] = "results mon";
  $arr["processor"] = "proc";
  return $arr;
}
                                                                                                                                                                              
function printCustomControlButtons() {

}



?>
