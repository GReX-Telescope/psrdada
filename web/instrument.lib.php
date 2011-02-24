<?PHP 

class instrument
{
  /* instrument name */
  var $name;

  /* name of the config file that defines everything */
  var $config_file;

  /* a hash of the config file */
  var $config;

  /* base URL of the instrument */
  var $url;

  /* path the CSS file */
  var $css_path = "";

  /* banner image for the instrument */
  var $banner_image = "";

  /* repeating banner image for the instrument */
  var $banner_image_repeat = "";

  /* fav icon */
  var $fav_icon = "/images/favicon.ico";

  function instrument($name, $config_file, $url)
  {
    $this->name = $name;
    $this->config_file = $config_file;
    $this->url = $url;
    $this->config = $this->configFileToHash($config_file);
  }

  function configFileToHash($fname="") {

    if ($fname == "") {
      $fname = $this->config_file;
    }
  
    $fptr = @fopen($fname,"r");
    $returnArray = array();

    if (!$fptr) {
      echo "Could not open file: $fname for reading<BR>\n";
    } else {
      while ($line = fgets($fptr, 1024)) {
     
        $comment_pos = strpos($line,"#");
        if ($comment_pos!==FALSE) {
          $line = substr($line, 0, $comment_pos);
        }

        // Remove trailing whitespace
        $line = chop($line);

        // skip blank lines
        if (strlen($line) > 0) {
          $array = split("[ \t]+",$line,2);   // Split into keyword/value
          $returnArray[$array[0]] = $array[1];
        }
      }
    }
    return $returnArray;
  }

  function open_head() 
  {
    echo "<head>\n";
  }

  function close_head()
  {
    echo "</head>\n";
  }

  function print_head_int($title, $refresh)
  {
    echo "  <title>".$title."</title>\n";
    instrument::print_css();
    instrument::print_favico();
    if ($refresh > 0)
      echo "<meta http-equiv='Refresh' content='".$refresh."'>\n";
  }

  function print_head() {
    instrument::open_head();
    instrument::print_head_int($title, $refresh);
    instrument::close_head();
  }

  function print_css() {
    echo "  <link rel='stylesheet' type='text/css' href='".$this->css_path."'>\n";
  }

  function print_favico() {
    echo "  <link rel='shortcut icon' href='".$this->fav_icon."'/>\n";
  }

  function print_banner($banner_text) {

?>   
<table cellspacing=0 cellpadding=0 border=0 width="100%">
  <tr>
    <td width=480px height=60px><img src="<?echo $this->banner_image?>" width=480 height=60></td>
    <td width="100%" height=60px background="<?echo $this->banner_image_repeat?>" class="largetext"><span id="globalstatus" class="largetext"><?echo $banner_text?></span></td>
  </tr>
</table>
<?
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

  function getIntergrationLength($archive) {

    if (file_exists($archive)) {

      $cmd = "vap -c length -n ".$archive." | awk '{print $2}'";
      $script = "source /home/dada/.bashrc; ".$cmd." 2>&1";
      $string = exec($script, $output, $return_var);
      $int_length = $output[0];

      if (is_numeric($int_length)) {
        $int_length = sprintf("%5.1f",$int_length);
      } else {
        $int_length = 0;
      }

      return $int_length;

    } else {
      return "0";
    }
  }

  function getSNR($archive) {

    if (file_exists($archive)) {

      $cmd = "psrstat -j 'zap median' -j FTp -qc snr ".$archive." 2>&1 | grep snr= | awk -F= '{print \$2}'";
      $script = "source /home/dada/.bashrc; ".$cmd." 2>&1";
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

  function getSingleStatusMessage($fname) {

    $result = "";
    if (file_exists($fname)) {
      $cmd = "tail -n 1 $fname";
      $result = rtrim(`$cmd`);
    }
    return $result;
  }

  function getPWCStatusMessages($config) {

    $message_types = array("pwc", "src", "sys");
    $message_classes = array("warn", "error");
    $message_class_values = array("warn" => STATUS_WARN, "error" => STATUS_ERROR);
    $status_dir = $config["STATUS_DIR"];
    $status = array();
      
    for ($i=0; $i<$config["NUM_PWC"]; $i++) {

      $host = $config["PWC_".$i];

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
  # return a list of all the PSRS listed in this user's psrcat catalogue
  #
  function getPsrcatPsrs() {

    $cmd = "psrcat -all -c \"PSRJ RAJ DECJ\" -nohead | awk '{print $2,$4, $7}'";
    $script = "source /home/dada/.bashrc; ".$cmd." 2>&1";
    $string = exec($script, $output, $return_var);

    $psrs = array();

    for ($i=0; $i<count($output); $i++) {
      $bits = split(" ", $output[$i]);
      $psrs[$bits[0]] = array("RAJ" => $bits[1], "DECJ" => $bits[2]);
    }

    return $psrs;
  }

  #
  # return a sorted list of the project groups for this instrument
  #
  function getProjectGroups() {

    $user = get_class($this);
    $output = array();
    $return_var = 0;

    $string = exec("groups $user",$output,$return_var);
    $array = split(" ", $output[0]);

    $groups = array();
    for ($i=0;$i<count($array); $i++) {
      if (strpos($array[$i], "P") !== FALSE) {
        array_push($groups, $array[$i]);
      }
    }
    sort($groups);

    return $groups;
  }

  function getClientStatuses()
  {

    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) {

      $status["PWC_".$i."_STATUS"] = STATUS_OK;
      $status["PWC_".$i."_MESSAGE"] = "";
      $status["SRC_".$i."_STATUS"] = STATUS_OK;
      $status["SRC_".$i."_MESSAGE"] = "";
      $status["SYS_".$i."_STATUS"] = STATUS_OK;
      $status["SYS_".$i."_MESSAGE"] = "";

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".pwc.warn";
      if (file_exists($fname)) {
        $status["PWC_".$i."_STATUS"] = STATUS_WARN;
        $status["PWC_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
      }

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".pwc.error";
      if (file_exists($fname)) {
        $status["PWC_".$i."_STATUS"] = STATUS_ERROR;
        $status["PWC_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
      }

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".src.warn";
      if (file_exists($fname)) {
        $status["SRC_".$i."_STATUS"] = STATUS_WARN;
        $status["SRC_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
      }

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".src.error";
      if (file_exists($fname)) {
        $status["SRC_".$i."_STATUS"] = STATUS_ERROR;
        $status["SRC_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
      }

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".sys.warn";
      if (file_exists($fname)) {
        $status["SYS_".$i."_STATUS"] = STATUS_WARN;
        $status["SYS_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
      }

      $fname = STATUS_FILE_DIR."/".$this->config["PWC_".$i].".sys.error";
      if (file_exists($fname)) {
        $status["SYS_".$i."_STATUS"] = STATUS_ERROR;
        $status["SYS_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
      }
    }

    return $status;

  }

  function headerFormat($key, $value) {

    $pad = 20 - strlen($key);
    $header_string = $key;
    $i=0;
    for ($i=0;$i<$pad;$i++) {
      $header_string .= " ";
    }
    $header_string .= $value;

    return $header_string;

  }
}

  ?>
