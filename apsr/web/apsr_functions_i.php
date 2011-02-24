<?PHP

function getServerLogInformation() {
                                                                                                                   
  $arr = array();
  $arr["apsr_tcs_interface"] =    array("logfile" => "apsr_tcs_interface.log", "name" => "TCS Interface", "tag" => "server", "shortname" => "TCS");
  $arr["apsr_results_manager"] =  array("logfile" => "apsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
  $arr["dada_pwc_command"] =      array("logfile" => "dada_pwc_command.log", "name" => "dada_pwc_command", "tag" => "server", "shortname" => "PWCC");
  $arr["apsr_gain_manager"] =     array("logfile" => "apsr_gain_manager.log", "name" => "Gain Mngr", "tag" => "server", "shortname" => "Gain");
  $arr["apsr_auxiliary_manager"] =      array("logfile" => "apsr_auxiliary_manager.log", "name" => "Aux Mngr", "tag" => "server", "shortname" => " Aux");
  $arr["apsr_web_monitor"] =      array("logfile" => "apsr_web_monitor.log", "name" => "Monitor", "tag" => "server", "shortname" => "Monitor");
  $arr["apsr_pwc_monitor"] =      array("logfile" => "nexus.pwc.log", "name" => "PWC Mon", "tag" => "pwc", "shortname" => "PWC Mon");
  $arr["apsr_sys_monitor"] =      array("logfile" => "nexus.sys.log", "name" => "SYS Mon", "tag" => "sys", "shortname" => "SYS Mon");
  $arr["apsr_src_monitor"] =      array("logfile" => "nexus.src.log", "name" => "SRC Mon", "tag" => "src", "shortname" => "SRC Mon");
  $arr["apsr_transfer_manager"] = array("logfile" => "apsr_transfer_manager.log", "name" => "Transfer Manager", "tag" => "server", "shortname" => "Xfer");
  return $arr;

}

                                                                                                                   
function getClientLogInformation() {
                                                                                                                   
  $arr = array();
  #$arr["apsr_master_control"] = array("logfile" => "apsr_master_control.log", "name" => "master_control";
  $arr["apsr_observation_manager"]  = array("logfile" => "nexus.sys.log", "name" => "Obs Mngr", "tag" => "obs mngr");
  $arr["apsr_archive_manager"]      = array("logfile" => "nexus.sys.log", "name" => "Archive Mngr", "tag" => "arch mngr");
  $arr["apsr_background_processor"] = array("logfile" => "nexus.sys.log", "name" => "BG Processor", "tag" => "bg mngr");
  $arr["apsr_gain_controller"]      = array("logfile" => "nexus.sys.log", "name" => "Gain Mon", "tag" => "gain mon");
  $arr["apsr_auxiliary_manager"]    = array("logfile" => "nexus.sys.log", "name" => "Aux Mon", "tag" => "aux mngr");
  $arr["apsr_processing_manager"]   = array("logfile" => "nexus.src.log", "name" => "Proc Mngr", "tag" => "proc mngr");
  $arr["apsr_disk_cleaner"]         = array("logfile" => "nexus.src.log", "name" => "Disk Cleaner", "tag" => "cleaner");
  $arr["processor"]                 = array("logfile" => "nexus.src.log", "name" => "Processor", "tag" => "proc");

  return $arr;

}

function printCustomControlButtons() {

}

function getSNR($archive) {

  if (file_exists($archive)) {

    $cmd = "psrstat -j 'zap median' -j FTp -qc snr ".$archive." 2>&1 | grep snr= | awk -F= '{print \$2}'";
    $script = "source /home/apsr/.bashrc; ".$cmd." 2>&1";
    $string = exec($script, $output, $return_var);
    $snr = $output[0];

    if (is_numeric($snr)) {
      $snr = sprintf("%5.1f",$snr);
    } else {
      $snr = 0;
    }


    return $snr;

  } else {

    return "N/A";

  }
}


#
# Find the most recent images for this observation
#
function getObsImages($dir) {

  # determine how many pulsars are present
  $cmd = "find ".$dir." -maxdepth 1 -name '*_t*.ar' -printf '%f\n'";
  $pulsars = array();
  $rval = 0;
  $line = exec($cmd, $pulsars, $rval);
  $results = array();

  for ($i=0; $i<count($pulsars); $i++) {

    $arr = split("_", $pulsars[$i], 2);
    $p = $arr[0];
    $p_regex = str_replace("+","\+",$p);

    if ($p == "total") {
      $cmd = "find ".$dir." -name '*.png' -printf'%f\n'";
      $pvfl = "phase_vs_flux_";
      $pvfr = "phase_vs_freq_";
      $pvtm = "phase_vs_time_";
    } else {
      $cmd = "find ".$dir." -name '*".$p."*.png' -printf '%f\n'";
      $pvfl = "phase_vs_flux_[JB]*".$p_regex;
      $pvfr = "phase_vs_freq_[JB]*".$p_regex;
      $pvtm = "phase_vs_time_[JB]*".$p_regex;
    }

    $array = array();
    $rval = 0;
    $line = exec($cmd, $array, $rval);

    for ($j=0; $j<count($array); $j++) {

      $f = $array[$j];
    
      if (preg_match("/^".$pvfl.".+240x180.png$/", $f))
        $results[$p]["phase_vs_flux"] = $f;
      if (preg_match("/^".$pvtm.".+240x180.png$/",$f))
        $results[$p]["phase_vs_time"] = $f;
      if (preg_match("/^".$pvfr.".+240x180.png$/",$f))
        $results[$p]["phase_vs_freq"] = $f;
      if (preg_match("/^".$pvfl.".+1024x768.png$/",$f))
        $results[$p]["phase_vs_flux_hires"] = $f;
      if (preg_match("/^".$pvtm.".+1024x768.png$/",$f))
        $results[$p]["phase_vs_time_hires"] = $f;
      if (preg_match("/^".$pvfr.".+1024x768.png$/",$f))
        $results[$p]["phase_vs_freq_hires"] = $f;
    }
  }
  return $results;
      
}

#
# Return the source names, DM's, periods and SNRS 
#
function getObsSources($dir) {

   # determine how many pulsars are present
  $cmd = "find ".$dir." -maxdepth 1 -name '*.ar' -printf '%f\n'";
  $pulsars = array();
  $rval = 0;
  $line = exec($cmd, $pulsars, $rval);
  $results = array();

  for ($i=0; $i<count($pulsars); $i++) {

    $arr = split("_", $pulsars[$i], 3);
    if (count($arr) == 3) 
      $p = $arr[0]."_".$arr[1];
    else 
      $p = $arr[0];

    if (strpos($pulsars[$i], "_t") !== FALSE) {
      $results[$p]["int"] = getIntergrationLength($dir."/".$pulsars[$i]);
      $results[$p]["src"] = getArchiveName($dir."/".$pulsars[$i]);
      $results[$p]["dm"] =  getSourceDM($results[$p]["src"]);
      $results[$p]["p0"] =  getSourcePeriodMS($results[$p]["src"]);
      $results[$p]["nsubint"] =  getNumSubints($dir."/".$pulsars[$i]);
    }

    if (strpos($pulsars[$i], "_f") !== FALSE) {
      $results[$p]["snr"] = getSNR($dir."/".$pulsars[$i]);
    }
  }

  return $results;
}


function getArchiveName($archive) {

  $prefix = "source /home/dada/.bashrc;";
  $bin_dir = DADA_ROOT."/bin";

  $cmd = $prefix." ".$bin_dir."/vap -c 'name' ".$archive." | tail -n 1  | awk '{print \$2}'";
  $array = array();
  $source = exec($cmd, $array, $rval);
  if (($rval != 0) || ($source == "name")) {
    $source = "unknown";
  }
  return $source;
}

function getNumSubints($archive) {

  $prefix = "source /home/dada/.bashrc;";
  $bin_dir = DADA_ROOT."/bin";

  $cmd = $prefix." ".$bin_dir."/psredit -Q -c nsubint ".$archive." | awk '{print \$2}'";
  $array = array();
  $nsubint = exec($cmd, $array, $rval);
  if ($rval != 0) {
    $nsubint = "unknown";
  }
  return $nsubint;

}

function getSourcePeriodMS($source) {

  $prefix = "source /home/dada/.bashrc;";
  $bin_dir = DADA_ROOT."/bin";

  # Determine the Period (P0)
  $cmd = $prefix." ".$bin_dir."/psrcat -x -c \"P0\" ".$source." | awk '{print \$1}'";
  $array = array();
  $P0 = exec($cmd, $array, $rval);
  if (($rval != 0) || ($P0 == "WARNING:")) {
    $P0 = "N/A";
  } else {
    $P0 *= 1000;
    $P0 = sprintf("%5.4f",$P0);
  }

  return $P0;
}


function getSourceDM($source) {

  $prefix = "source /home/dada/.bashrc;";
  $bin_dir = DADA_ROOT."/bin";

  $cmd = $prefix." ".$bin_dir."/psrcat -x -c \"DM\" ".$source." | awk '{print \$1}'";
  $array = array();
  $DM = exec($cmd, $array, $rval);
   if (($rval != 0) || ($DM == "WARNING:")) {
    $DM = "N/A";
  }

  return $DM;
}

?>
