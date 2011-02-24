<?PHP

if (!$_APSR_LIB_PHP) { $_APSR_LIB_PHP = 1;


define(INSTRUMENT, "apsr");
define(CFG_FILE, "/home/dada/linux_64/share/apsr.cfg");
define(CSS_FILE, "apsr/apsr.css");

include("site_definitions_i.php");
include("instrument.lib.php");

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

  function getServerLogInfo() {

    $arr = array();
    $arr["apsr_tcs_interface"] =      array("logfile" => "apsr_tcs_interface.log", "name" => "TCS Interface", "tag" => "server", "shortname" => "TCS");
    $arr["apsr_results_manager"] =    array("logfile" => "apsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
    $arr["dada_pwc_command"] =          array("logfile" => "dada_pwc_command.log", "name" => "dada_pwc_command", "tag" => "server", "shortname" => "PWCC");
    $arr["apsr_web_monitor"] =        array("logfile" => "apsr_web_monitor.log", "name" => "Monitor", "tag" => "server", "shortname" => "Monitor");
    $arr["apsr_pwc_monitor"] =        array("logfile" => "nexus.pwc.log", "name" => "PWC Mon", "tag" => "pwc", "shortname" => "PWC Mon");
    $arr["apsr_sys_monitor"] =        array("logfile" => "nexus.sys.log", "name" => "SYS Mon", "tag" => "sys", "shortname" => "SYS Mon");
    $arr["apsr_gain_manager"] =       array("logfile" => "apsr_gain_manager.log", "name" => "Gain Mngr", "tag" => "server", "shortname" => "Gain");
    $arr["apsr_transfer_manager"] =   array("logfile" => "apsr_transfer_manager.log", "name" => "Transfer Manager", "tag" => "src", "shortname" => "Xfer");
    $arr["apsr_auxiliary_manager"] =  array("logfile" => "apsr_auxiliary_manager.log", "name" => "Aux Mngr", "tag" => "server", "shortname" => " Aux");
    return $arr;

  }

  function getClientLogInfo() {
    $arr = array();
    $arr["apsr_observation_manager"]  = array("logfile" => "nexus.sys.log", "name" => "Obs Mngr", "tag" => "obs mngr");
    $arr["apsr_processing_manager"]   = array("logfile" => "nexus.sys.log", "name" => "Proc Mngr", "tag" => "proc mngr");
    $arr["apsr_archive_manager"]      = array("logfile" => "nexus.sys.log", "name" => "Archive Mngr", "tag" => "arch mngr");
    $arr["processor"]                   = array("logfile" => "nexus.src.log", "name" => "Processor", "tag" => "proc");
    $arr["apsr_gain_controller"]      = array("logfile" => "nexus.sys.log", "name" => "Gain Mon", "tag" => "gain mon");
    return $arr;
  }

  function getDistribLogInfo() {
    $arr = array();
    $arr["apsr_distrib_manager"]      = array("logfile" => "distrib.sys.log", "name" => "Distrib Mngr", "tag" => "dist mngr");
    $arr["distrib"]                     = array("logfile" => "distrib.src.log", "name" => "Demuxers", "tag" => "ditrib");
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

} // END OF CLASS DEFINITION

} // _APSR_LIB_PHP
