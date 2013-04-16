<?PHP

include_once("functions_i.php");

define("INSTRUMENT", "mopsr");
define("CFG_FILE", "/home/dada/linux_64/share/mopsr.cfg");
define("PWC_FILE", "/home/dada/linux_64/share/mopsr_pwcs.cfg");
define("CSS_FILE", "/mopsr/mopsr.css");

include_once("site_definitions_i.php");
include_once("instrument.lib.php");

class mopsr extends instrument
{
  function mopsr()
  {
    instrument::instrument(INSTRUMENT, CFG_FILE, URL_FULL);

    $this->css_path = CSS_FILE;
    $this->banner_image = "/mopsr/images/mopsr_logo_480x60.png";
    $this->banner_image_repeat = "/mopsr/images/mopsr_logo_1x60.png";
    $this->fav_icon = "/mopsr/images/mopsr_favicon.ico";
  }

  function serverLogInfo()
  {
    $arr = array();
    $arr["mopsr_results_manager"]        = array("logfile" => "mopsr_results_manager.log", "name" => "Results Mngr", "tag" => "server", "shortname" => "Results");
    $arr["mopsr_web_monitor"]            = array("logfile" => "mopsr_web_monitor.log", "name" => "Web Monitor", "tag" => "server", "shortname" => "Monitor");
    $arr["mopsr_pwc_monitor"]            = array("logfile" => "nexus.pwc.log", "name" => "PWC", "tag" => "pwc", "shortname" => "PWC");
    $arr["mopsr_sys_monitor"]            = array("logfile" => "nexus.sys.log", "name" => "SYS", "tag" => "sys", "shortname" => "SYS");
    $arr["mopsr_src_monitor"]            = array("logfile" => "nexus.src.log", "name" => "SRC", "tag" => "src", "shortname" => "SRC");
    return $arr;
  }

  function clientLogInfo() {

    $arr = array();
    $arr["mopsr_results_monitor"] = array("logfile" => "nexus.sys.log", "name" => "Results Mon", "tag" => "results mon");
    return $arr;

  }

  function getClientStatusMessages($config)
  {
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
