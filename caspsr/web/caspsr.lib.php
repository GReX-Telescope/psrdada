<?PHP

if (!$_CASPSR_LIB_PHP) { $_CASPSR_LIB_PHP = 1;


define(INSTRUMENT, "caspsr");
define(CFG_FILE, "/home/dada/linux_64/share/caspsr.cfg");
define(CSS_FILE, "caspsr/caspsr.css");

include_once("site_definitions_i.php");
include_once("instrument.lib.php");

class caspsr extends instrument
{

  function caspsr()
  {
    instrument::instrument(INSTRUMENT, CFG_FILE, URL);

    $this->css_path = "/caspsr/caspsr.css";
    $this->banner_image = "/caspsr/images/caspsr_logo_480x60.png";
    $this->banner_image_repeat = "/caspsr/images/caspsr_logo_1x60.png";

  }

  function getServerLogInfo() {

    $arr = array();
    $arr["caspsr_tcs_interface"] =      array("logfile" => "caspsr_tcs_interface.log", "name" => "TCS Interface", "tag" => "server", "shortname" => "TCS");
    $arr["caspsr_results_manager"] =    array("logfile" => "caspsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
    $arr["dada_pwc_command"] =          array("logfile" => "dada_pwc_command.log", "name" => "Nexus", "tag" => "server", "shortname" => "PWCC");
    $arr["caspsr_web_monitor"] =        array("logfile" => "caspsr_web_monitor.log", "name" => "Monitor", "tag" => "server", "shortname" => "Monitor");
    $arr["caspsr_pwc_monitor"] =        array("logfile" => "nexus.pwc.log", "name" => "PWC Mon", "tag" => "pwc", "shortname" => "PWC Mon");
    $arr["caspsr_sys_monitor"] =        array("logfile" => "nexus.sys.log", "name" => "SYS Mon", "tag" => "sys", "shortname" => "SYS Mon");
    $arr["caspsr_demux_monitor"] =      array("logfile" => "demux.src.log", "name" => "DEMUX Mon", "tag" => "demux", "shortname" => "Demux Mon");
    $arr["caspsr_transfer_manager"] =   array("logfile" => "caspsr_transfer_manager.log", "name" => "Transfer Mngr", "tag" => "server", "shortname" => "Xfer");
    $arr["caspsr_raid_pipeline"] =      array("logfile" => "caspsr_raid_pipeline.log", "name" => "RAID Pipeline", "tag" => "server", "shortname" => "RAID");
    $arr["caspsr_baseband_controller"] = array("logfile" => "caspsr_baseband_controller.log", "name" => "Baseband Ctrlr", "tag" => "server", "shortname" => "Baseband");
    return $arr;

  }

  function getClientLogInfo() {
    $arr = array();
    $arr["caspsr_archive_manager"]      = array("logfile" => "nexus.sys.log", "name" => "Archive Mngr", "tag" => "arch mngr");
    $arr["caspsr_processing_manager"]   = array("logfile" => "nexus.sys.log", "name" => "Proc Mngr", "tag" => "proc mngr");
    $arr["caspsr_disk_cleaner"]         = array("logfile" => "nexus.sys.log", "name" => "Disk Cleaner", "tag" => "cleaner");
    $arr["processor"]                   = array("logfile" => "nexus.src.log", "name" => "Processor", "tag" => "proc");
    return $arr;
  }

  function getDemuxLogInfo() {
    $arr = array();
    $arr["caspsr_demux_manager"]        = array("logfile" => "demux.sys.log", "name" => "Demux Mngr", "tag" => "demux mngr");
    $arr["demux"]                       = array("logfile" => "demux.src.log", "name" => "Demuxers", "tag" => "demux");
    return $arr;
  }


  #
  # Return the source names, DM's, periods and SNRS 
  #
  function getObsSources($dir) {

     # determine how many pulsars are present
    $cmd = "find ".$dir." -maxdepth 1 -name '*.tot' -printf '%f\n'";
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

    for ($i=0; $i<$config["NUM_DEMUX"]; $i++) {
      
      $host = $config["DEMUX_".$i];

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
    $cmd = "groups caspsr";
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

} // _CASPSR_LIB_PHP
