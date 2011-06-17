<?PHP

if (!$_APSR_LIB_PHP) { $_APSR_LIB_PHP = 1;


define(INSTRUMENT, "apsr");
define(CFG_FILE, "/home/dada/linux_64/share/apsr.cfg");
define(CSS_FILE, "apsr/apsr.css");

include("site_definitions_i.php");
include("instrument.lib.php");
include("functions_i.php");

class apsr extends instrument
{

  function apsr()
  {
    instrument::instrument(INSTRUMENT, CFG_FILE, URL);

    $this->css_path = "/apsr/apsr.css";
    $this->banner_image = "/apsr/images/apsr_logo_480x60.png";
    $this->banner_image_repeat = "/apsr/images/apsr_logo_1x60.png";

  }

  function print_head($title, $refresh=0) 
  {
      echo "<head>\n";
      $this->print_head_int($title, $refresh);
      echo "</head>\n";
  }

  function print_head_int($title, $refresh) {
    echo "  <title>".$title."</title>\n";
    instrument::print_css();
    if ($refresh > 0)
      echo "<meta http-equiv='Refresh' content='".$refresh."'>\n";
  }

  function serverLogInfo() {

    $arr = array();
    $arr["apsr_tcs_interface"] =      array("logfile" => "apsr_tcs_interface.log", "name" => "TCS Interface", "tag" => "server", "shortname" => "TCS");
    $arr["apsr_results_manager"] =    array("logfile" => "apsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
    $arr["dada_pwc_command"] =          array("logfile" => "dada_pwc_command.log", "name" => "Nexus", "tag" => "server", "shortname" => "PWCC");
    $arr["apsr_web_monitor"] =        array("logfile" => "apsr_web_monitor.log", "name" => "Monitor", "tag" => "server", "shortname" => "Monitor");
    $arr["apsr_pwc_monitor"] =        array("logfile" => "nexus.pwc.log", "name" => "PWC Mon", "tag" => "pwc", "shortname" => "PWC Mon");
    $arr["apsr_sys_monitor"] =        array("logfile" => "nexus.sys.log", "name" => "SYS Mon", "tag" => "sys", "shortname" => "SYS Mon");
    $arr["apsr_src_monitor"] =        array("logfile" => "nexus.src.log", "name" => "SRC Mon", "tag" => "src", "shortname" => "SRC Mon");
    $arr["apsr_gain_manager"] =       array("logfile" => "apsr_gain_manager.log", "name" => "Gain Mngr", "tag" => "server", "shortname" => "Gain");
    $arr["apsr_transfer_manager"] =   array("logfile" => "apsr_transfer_manager.log", "name" => "Transfer Manager", "tag" => "src", "shortname" => "Xfer");
    return $arr;

  }

  function clientLogInfo() {
    $arr = array();
    $arr["apsr_observation_manager"] = array("logfile" => "nexus.sys.log", "name" => "Obs Mngr", "tag" => "obs mngr");
    $arr["apsr_archive_manager"]     = array("logfile" => "nexus.sys.log", "name" => "Archive Mngr", "tag" => "arch mngr");
    $arr["processor"]                = array("logfile" => "nexus.src.log", "name" => "Processor", "tag" => "proc");
    $arr["apsr_gain_controller"]     = array("logfile" => "nexus.sys.log", "name" => "Gain Mon", "tag" => "gain mon");
    $arr["apsr_disk_cleaner"]        = array("logfile" => "nexus.sys.log", "name" => "Disk Cleaner", "tag" => "cleaner");
    return $arr;
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
        $results[$p]["int"] = instrument::getIntergrationLength($dir."/".$pulsars[$i]);
        $results[$p]["src"] = instrument::getArchiveName($dir."/".$pulsars[$i]);
        $results[$p]["dm"] =  instrument::getSourceDM($results[$p]["src"]);
        $results[$p]["p0"] =  instrument::getSourcePeriodMS($results[$p]["src"]);
        $results[$p]["nsubint"] =  instrument::getNumSubints($dir."/".$pulsars[$i]);
      }

      if (strpos($pulsars[$i], "_f") !== FALSE) {
        $results[$p]["snr"] = instrument::getSNR($dir."/".$pulsars[$i]);
      }
    }

    return $results;
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
        $band = "bandpass_";
      } else { 
        $cmd = "find ".$dir." -name '*".$p."*.png' -printf '%f\n'";
        $pvfl = "phase_vs_flux_[JB]*".$p_regex;
        $pvfr = "phase_vs_freq_[JB]*".$p_regex;
        $pvtm = "phase_vs_time_[JB]*".$p_regex;
        $band = "bandpass_[JB]*".$p_regex;
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
        if (preg_match("/^".$band.".+240x180.png$/",$f))
          $results[$p]["bandpass"] = $f;

        if (preg_match("/^".$pvfl.".+1024x768.png$/",$f))
          $results[$p]["phase_vs_flux_hires"] = $f;
        if (preg_match("/^".$pvtm.".+1024x768.png$/",$f))
          $results[$p]["phase_vs_time_hires"] = $f;
        if (preg_match("/^".$pvfr.".+1024x768.png$/",$f))
          $results[$p]["phase_vs_freq_hires"] = $f;
        if (preg_match("/^".$band.".+1024x768.png$/",$f))
          $results[$p]["bandpass_hires"] = $f;
      }
    }
    return $results;
  }

  function getObsState($dir) {

    if (file_exists($dir."/obs.failed")) {
      return "failed";
    } else if (file_exists($dir."/obs.finished")) {
      return "finished";
    } else if (file_exists($dir."/obs.processing")) {
      return "processing";
    } else if (file_exists($dir."/obs.transferred")) {
      return "transferred";
    } else if (file_exists($dir."/obs.deleted")) {
      return "deleted";
    } else {
      return "unknown";
    }
  }

  /* determine the unix timestamp of the most recent result */
  function getMostRecentResult($dir) {

    $current_time = time();

    $cmd = "find ".$dir." -name '*.lowres' -printf '%T@\\n' | sort | tail -n 1";
    $archive_time = exec($cmd, $array, $rval);

    $difference = 0;

    /* If we dont have any lowres archives */
    if (count($array) == 0) {

      $archive_time = 0;
      $cmd = "find ".$dir." -name 'obs.start' -printf '%T@\\n' | sort | tail -n 1";
      $obs_start_time = exec($cmd, $array, $rval);

      if (count($array) == 0) {
        $difference = -1;
      } else {
        $difference = $current_time - $obs_start_time;
      }

    } else {
      $difference = $current_time - $archive_time;
    }

    return $difference;
  }

  function getArchiveCount($results_dir, $archives_dir, $client_dir)
  {
    $results = array("num_results" => 0, "num_archives" => 0, "num_bands" => 0, "num_processed" => 0);
    $hosts = array();
    
    # determine the number of lowres results received
    $cmd = "find ".$results_dir."/ -name '*.lowres' | wc -l";
    $results["num_results"] = exec($cmd);

    # determine the total number of archives received on client machines
    $results["num_archives"] = 0;

    if ($archives_dir)
    {
      # determine the hosts on which the archives reside
      $cmd = "find ".$archives_dir." -mindepth 1 -maxdepth 1 -type l -printf '%l\n' | awk -F/ '{print $3}' | sort";
      $lastline = exec($cmd, $array, $rval);
      foreach ($array as $host) {
        array_push ($hosts, rtrim($host));
      }
      $results["num_bands"] = count($hosts);
    }

    if ($client_dir) 
    {
      $remote_cmd = "find ".$client_dir." -name '*.ar' | wc -l";
      for ($i=0; $i<count($hosts); $i++)
      {
        $cmd = "ssh -x -o Batchmode=yes -l apsr  ".$hosts[$i]." \"".$remote_cmd."\"";
        $host_num = exec($cmd);
        $results["num_archives"] += $host_num;
      }
    }

    # determine the number of archives processed by the results manager
    if (file_exists($results_dir."/processed.txt")) {
      $processed = getConfigFile($results_dir."/processed.txt");
      $keys = array_keys($processed);
      for($i=0; $i<count($keys); $i++) {
        $results["num_processed"] += count(split(" ",$processed[$keys[$i]]));
      }
    }

    return $results;
  }

  # return the full DADA header from an obs.start file in the results dir
  function getDADAHeader($results_dir)
  {
    $header = array();
    $cmd = "find ".$results_dir." -type f -name 'obs.start' | sort | tail -n 1";
    $obs_start_file = exec($cmd);

    if (file_exists($obs_start_file))
    {
      $header = getConfigFile($obs_start_file);
    }
    return $header;
  }


  function getClientStatusMessages($config) {

    $status = instrument::getPWCStatusMessages($config);

    $message_types = array("pwc", "src", "sys");
    $message_classes = array("warn", "error");
    $message_class_values = array("warn" => STATUS_WARN, "error" => STATUS_ERROR);
    $status_dir = $config["STATUS_DIR"];

    for ($i=0; $i<$config["NUM_DISTRIB"]; $i++) {
      
      $host = $config["DISTRIB_".$i];

      for ($j=0; $j<count($message_types); $j++) {

        $message_type = $message_types[$j];

        for ($k=0; $k<count($message_classes); $k++) {

          $message_class = $messsage_classes[$k];

          $fname = $status_dir."/".$host.".".$message_type.".".$message_class;
          if (file_exists($fname)) {
            $status[$host."_".$message_type."_STATUS"]  = $message_class_values[$message_class];
            $status[$host."_".$message_type."_MESSAGE"] = instrument::getSingleStatusMessage($fname);
          } else {
            $status[$host."_".$message_type."_STATUS"] = STATUS_OK;
             $status[$host."_".$message_type."_MESSAGE"] = "";
          }
        }
      }
    }

    return $status;
  }

  #
  # Return an array of valid BPSR project IDs
  #
  function getPIDS()
  {
    $cmd = "groups apsr";
    $output = array();
    $return_var = 0;

    $string = exec($cmd, $output, $return_var);
    $array = split(" ",$string);
    $groups = array();
    for ($i=0; $i<count($array); $i++) {
      if (strpos($array[$i], "P") === 0) {
        array_push($groups, $array[$i]);
      }
    }
    sort($groups);
    return $groups;
  }


} // END OF CLASS DEFINITION

} // _APSR_LIB_PHP
