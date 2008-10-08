<?PHP

function getConfigFile($fname, $quiet=FALSE) {

  $fptr = @fopen($fname,"r");

  if (!$fptr) {
    if (!$quiet)
    echo "Could not open file: $fname for reading<BR>\n";
  } else {
    while ($line = fgets($fptr, 1024)) {
      // Search for comments and  remove them from the line to be parsed
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

function getRawTextFile($fname) {

  $fptr = @fopen($fname,"r");
  if (!$fptr) {
    return array("Could not open file: $fname for reading");
  } else {
    $returnArray = array();
    while ($line = fgets($fptr, 1024)) {
      // Remove newlines
      $line = chop($line);
      array_push($returnArray,$line);
    }
    return $returnArray;
  }
}


// TODO Implement
function getAllStatuses($pwc_config) {

  $num_pwc = $pwc_config["NUM_PWC"];

  for ($i=0; $i<$num_pwc; $i++) {

    $status["PWC_".$i."_STATUS"] = STATUS_OK;
    $status["PWC_".$i."_MESSAGE"] = "";
    $status["SRC_".$i."_STATUS"] = STATUS_OK;
    $status["SRC_".$i."_MESSAGE"] = "";
    $status["SYS_".$i."_STATUS"] = STATUS_OK;
    $status["SYS_".$i."_MESSAGE"] = "";

    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".pwc.warn";
    if (file_exists($fname)) {
      $status["PWC_".$i."_STATUS"] = STATUS_WARN;
      $status["PWC_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
    }
    
    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".pwc.error";
    if (file_exists($fname)) {
      $status["PWC_".$i."_STATUS"] = STATUS_ERROR;
      $status["PWC_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
    }

    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".src.warn";
    if (file_exists($fname)) {
      $status["SRC_".$i."_STATUS"] = STATUS_WARN;
      $status["SRC_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
    }
                                                                                                                                                           
    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".src.error";
    if (file_exists($fname)) {
      $status["SRC_".$i."_STATUS"] = STATUS_ERROR;
      $status["SRC_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
    }

    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".sys.warn";
    if (file_exists($fname)) {
      $status["SYS_".$i."_STATUS"] = STATUS_WARN;
      $status["SYS_".$i."_MESSAGE"] = getSingleStatusMessage($fname);
    }
                                                                                                                                                           
    $fname = STATUS_FILE_DIR."/".$pwc_config["PWC_".$i].".sys.error";
    if (file_exists($fname)) {
      $status["SYS_".$i."_STATUS"] = STATUS_ERROR;
      $status["SYS_".$i."_MESSAGE"] =  getSingleStatusMessage($fname);
    }
  }

  return $status;

}

function getSingleStatusMessage($fname) {

  $fptr = fopen($fname,"r");
  if (!$fptr) {
    echo "Could not open status file: $fname<BR>\n";
  } else {
    #echo "Opening $fname<BR>\n";
    $j = 0;
    while (!(feof($fptr))) {
      $string = rtrim(fgets($fptr));
      if (strlen($string) > 0) {
        $array[$j] = $string;
        $j++;
      }
    }
    fclose($fptr);
    // Roll back the j counter
    $j--;
  }
  return $array[$j];
}


function openSocket($host, $port, $timeout=2) {

  //create a socket
  $socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
  if (!(socket_set_nonblock($socket))) {
    return array(0, "Unable to set nonblock on socket");
  }
  $time = time();

  while (!@socket_connect($socket, $host, $port)) {
    $err = socket_last_error($socket);
    if ($err == 115 || $err == 114) {
      if ((time() - $time) >= $timeout) {
        socket_close($socket);
        return array(0,"Connection timed out");
      }
      time_nanosleep(0,10000000);
      continue;
   
    }
    return array(0, socket_strerror($err));
  }                                                                                                                                        
  if (!(socket_set_block($socket))){
    return array(0, "Unable to set block on socket");
  } 
  return array ($socket,"ok");
}

function socketRead($socket) {

  $string = socket_read ($socket, 4096, PHP_NORMAL_READ);
  if ($string == FALSE) {
    $string = "Error on socketRead()\n";
  }
  return $string;
}

function socketWrite($socket, $string) {

  $bytes_to_write = strlen($string);
  $bytes_written = socket_write($socket,$string,$bytes_to_write);

  if ($bytes_written === FALSE) {
    echo "Error writing data with socket_write()<BR>\n";
    return -1;
  }
  if ($bytes_written != $bytes_to_write) {
    echo "Error, tried to write".$bytes_to_write.", but only ".$bytes_written." bytes were written<BR>\n";
    return -1;
  } 
  return $bytes_written;
}


/* Returns a list of all the unix groups this user is a member of */
function getProjects($user) {

  $string = exec("groups $user",$output,$return_var);
  $array = split(" ", $output[0]);

  for ($i=0;$i<count($array); $i++) {
    $groups[$i] = $array[$i];
  }
  return $groups;

}  

function getProject(){

  $fname = CURRENT_PROJECT;
  $fp = fopen($fname,"r");
  $counter = 10;
  while ((!flock($fp,LOCK_SH)) && ($counter > 0)) {
    sleep(1);
    $counter--;
  }
  if ($counter == 0) {
    return "none";
  } else {
    $current_project = fgets($fp,128);
    flock($fp,LOCK_UN);
  }
  fclose($fp);
  return $current_project;
}

function changeProject($group) {

  $fname = CURRENT_PROJECT;
  $fp = fopen($fname,"w");
  $counter = 10;
  while ((!flock($fp,LOCK_EX)) && ($counter > 0)) {
    sleep(1);
    $counter--;
  }
  if ($counter == 0) {
    return -1;
  } else {
    fwrite($fp,$group);
    flock($fp,LOCK_UN);
  }
  fclose($fp);
  return 0;

}

function echoList($name, $selected, $list, $indexlist="none", $readonly=FALSE,$onChange="") {

  echo "<SELECT name=\"".$name."\" id=\"".$name."\" $onChange>\n";
  $i = 0;
  while($i < count($list)) {
    if ($indexlist == "none") {
      $index = $i;
    } else {
      $index = $indexlist[$i];
    }
    $text = $list[$i];
    echoOption($index, $text, $readonly,$selected);
    $i++;
  }
  echo "</SELECT>";
}

function echoOption($value,$name,$readonly=FALSE,$selected="notAlwaysNeeded")
{
  echo  "<OPTION VALUE=\"".$value."\"";
  if ("$selected" == "$value") {
    echo " SELECTED";
  }
  if ($_SESSION["readonly"] == "true") {
    echo " DISABLED";
  }
  echo ">".$name."\n";
}

function getFileListing($dir,$pattern="/*/") {

  $arr = array();

  if ($handle = opendir($dir)) {
    while (false !== ($file = readdir($handle))) {
      if (($file != ".") && ($file != "..") && (preg_match($pattern,$file) > 0)) {
      //if (($file != ".") && ($file != "..")) {
        array_push($arr,$file);
      }
    }
    closedir($handle);
  }
  return $arr;
}

function addToDadaTime($time_string, $nseconds) {

  $a = split('[-:]',$time_string);
  $time_unix = mktime($a[3],$a[4],$a[5],$a[1],$a[2],$a[0]);
  $time_unix += $nseconds;
  $new_time_string = date(DADA_TIME_FORMAT,$time_unix);
  return $new_time_string;

}

function localTimeFromGmTime($time_string) {

  $a = split('[-:]',$time_string);
  $time_unix = gmmktime($a[3],$a[4],$a[5],$a[1],$a[2],$a[0]);
  $new_time_string = date(DADA_TIME_FORMAT, $time_unix);
  return $new_time_string;

}

function gmTimeFromLocalTime($time_string) {
                                                                                                                                            
  $a = split('[-:]',$time_string);
  $time_unix = mktime($a[3],$a[4],$a[5],$a[1],$a[2],$a[0]);
  $new_time_string = gmdate(DADA_TIME_FORMAT, $time_unix);
  return $new_time_string;
                                                                                                                                            
}
                                                                                                                                            


function killProcess($pname) {

  $returnVal = 0;
  $cmd = "ps axu | grep ".$pname." | grep -v grep";
  $pid_to_kill = system($cmd,$returnVal);
  echo "pids to kill = ".$pids_to_kill ."<BR>\n";

  if ($returnVal == 0) {

    $cmd = $cmd." | awk '{print \$2}'";
    $pid_to_kill = system($cmd);
    echo "pid to kill = ".$pid_to_kill ."<BR>\n";

    $cmd = "kill -KILL ".$pid_to_kill;
    echo "cmd = $cmd<BR>\n";
    system($cmd);
    return "killed pid $pid_to_kill";

  } else {
    return "process did not exist";
  }

}

function makeTimeString($time_unix) {

  $ndays = gmdate("z",$time_unix);
  if ($ndays) {
    return $ndays." days, ".gmdate("H:i:s",$time_unix);
  } else {
    return gmdate("H:i:s",$time_unix);
  }
}


/* Get sub directories of dir 
     index is the dir index to start from (sorted)
     ndirs is the numner of dirs to return

 */
function getSubDirs($dir, $offset=0, $length=0, $reverse=0) {

  $subdirs = array();

  if (is_dir($dir)) {
    if ($dh = opendir($dir)) {
      while (($file = readdir($dh)) !== false) {
        if (($file != ".") && ($file != "..") && (is_dir($dir."/".$file))) {
          array_push($subdirs, $file);
        }
      }
      closedir($dh);
    }
  }

  if ($reverse) 
    rsort($subdirs);
  else 
    sort($subdirs);

  if (($offset >= 0) && ($length != 0)) {
    return array_slice($subdirs, $offset, $length);
  } else {
    return($subdirs);
  }

}

function getIntergrationLength($archive) {

  if (file_exists($archive)) {

    $cmd = "vap -c length -n ".$archive." | awk '{print $2}'";
    $script = "source /home/apsr/.bashrc; ".$cmd." 2>&1";
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

function getServerDaemonNames() {

  $arr = array();
  $arr["apsr_tcs_interface"] = "TCS Interface";
  $arr["bpsr_tcs_interface"] = "TCS Interface";
  $arr["pwc_monitor"] = "PWC Mon";
  $arr["sys_monitor"] = "SYS Mon";
  $arr["src_monitor"] = "SRC Mon";
  $arr["results_manager"] = "Results Mngr";
  $arr["bpsr_results_manager"] = "Results Mngr";
  $arr["gain_manager"] = "Gain Mngr";
  $arr["aux_manager"] = "Aux Mngr";
  $arr["ibob_simulator"] = "IBOB Sim.";
  $arr["ibob_connection_manager"] = "IBOB conn.";
  return $arr;
}

function getClientDaemonNames() {

  $arr = array();

  $arr["master_control"] = "Master Ctrl";
  $arr["observation_manager"] = "Obs Mngr";
  $arr["bpsr_observation_manager"] = "Obs. Mngr";
  $arr["bpsr_results_monitor"] = "Results Monitor";
  $arr["processing_manager"] = "Proc. Mngr";
  $arr["processor"] = "Processor";
  $arr["archive_manager"] = "Archive Mngr";
  $arr["spectra_manager"] = "Spectra Mngr";
  $arr["background_processor"] = "BG Proc";
  $arr["aux_manager"] = "Aux Mngr";
  $arr["monitor"] = "Monitor";
  $arr["gain_controller"] = "Gain Ctrl";

  return $arr;

}

function getClientDaemonTypes() {
  $arr = array();

  $arr["master_control"] = "none";
  $arr["observation_manager"] = "sys";
  $arr["bpsr_observation_manager"] = "sys";
  $arr["bpsr_results_monitor"] = "sys";
  $arr["processing_manager"] = "src";
  $arr["processor"] = "src";
  $arr["archive_manager"] = "sys";
  $arr["spectra_manager"] = "sys";
  $arr["background_processor"] = "sys";
  $arr["aux_manager"] = "sys";
  $arr["monitor"] = "sys";
  $arr["gain_controller"] = "sys";
  return $arr;
}

function getClientDaemonTags() {

  $arr = array();

  $arr["master_control"] = "none";
  $arr["observation_manager"] = "obs mngr";
  $arr["bpsr_observation_manager"] = "obs mngr";
  $arr["bpsr_results_monitor"] = "results mon";
  $arr["processing_manager"] = "proc mngr";
  $arr["processor"] = "proc";
  $arr["archive_manager"] = "arch mngr";
  $arr["spectra_manager"] = "spectra mngr";
  $arr["background_processor"] = "bg mngr";
  $arr["aux_manager"] = "aux mngr";
  $arr["monitor"] = "monitor";
  $arr["gain_controller"] = "gain ctrl";
  return $arr;
}

function getServerDaemonTypes() {
                                                                               
  $arr = array();
  $arr["apsr_tcs_interface"] = "apsr_tcs_interface";
  $arr["bpsr_tcs_interface"] = "bpsr_tcs_interface";
  $arr["pwc_monitor"] = "nexus.pwc";
  $arr["sys_monitor"] = "nexus.sys";
  $arr["src_monitor"] = "nexus.src";
  $arr["results_manager"] = "results_manager";
  $arr["bpsr_results_manager"] = "bpsr_results_manager";
  $arr["gain_manager"] = "gain_manager";
  $arr["aux_manager"] = "aux_manager";
  $arr["dada_pwc_command"] = "dada_pwc_command";
  $arr["ibob_simulator"] = "ibob_simulator";
  $arr["ibob_connection_manager"] = "ibob_connection_manager";
  return $arr;
}

?>
