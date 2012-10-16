<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("bpsr.lib.php");

$inst = new bpsr();
$config = $inst->config;

/* Don't allow this page to be cached, since it should always be fresh. */
header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");   // Date in the past

/* parse get params */
#$machine     = (isset($_GET["machine"]) && $_GET["machine"] != "") ? $_GET["machine"] : "nexus";
#$daemon      = (isset($_GET["daemon"]) && $_GET["daemon"] != "") ? $_GET["daemon"] : "all";
$logfile     = (isset($_GET["logfile"]) && $_GET["logfile"] != "") ? $_GET["logfile"] : "";
$auto_scroll = (isset($_GET["autoscroll"]) && $_GET["autoscroll"] != "") ? $_GET["autoscroll"] : "false";
$filter      = (isset($_GET["filter"]) && $_GET["filter"] != "") ? $_GET["filter"] : "";
$length      = (isset($_GET["length"]) && $_GET["length"] != "") ? $_GET["length"] : LOG_FILE_SCROLLBACK_HOURS;
$level       = (isset($_GET["level"]) && $_GET["level"] != "") ? $_GET["level"] : "all";

#$log_type    = "pwc";
$log_tag     = "*";

if ($length == "all") 
  $length = PHP_INT_MAX;
else
  $length *= (60 * 60);

// Determine what type and sub type of logs we will display
$logfile = $config["SERVER_LOG_DIR"]."/".$logfile;
#echo "file: ".$logfile."<BR>\n";
#echo "length: ".$length."<BR>\n";
ob_start();

?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>

  <script type="text/javascript">

    var auto_scroll = <?echo $auto_scroll?>;
    function looper() {
      scrollDown();
      setTimeout('looper()',250)
    }

    function scrollDown() {
      if (auto_scroll) {
        self.scrollByLines(1000);
      }
    }

  </script>
  <!--<meta http-equiv="Refresh" content="0">-->
</head>

<body>
<script type="text/javascript">looper()</script>
<pre>
<p>
<?

$fname = $logfile;
clearstatcache();

// Open the file we wish to read - In this case the access log of our server
if (!($fp = @fopen($fname, 'r'))) {
    // We couldn't open the file, error!
    echo "ERROR: The log file [".$logfile."] did not exist or was not readable\n";
    # ob_end_flush();
    exit();
}

register_shutdown_function("close_log_file",$fp);

// We don't want to display the entire file, it could be huge, so
// let's use a quick way to just display some of the file.  Use fseek
// to head to the end of the file, but back up by 500 bytes.
// fseek($fp, -5000, SEEK_END);

// Read one line of data to throw away, as it may be an incomplete
//  line due to our seek.
# fgets($fp);

$max_time_limit = 20; //seconds;

$atEOF = "false";
$haveSomeLogs = "false";

$current_time_unix = time();
$current_time_string = date(DADA_TIME_FORMAT,$current_time_unix);
$time_pattern = '/\[\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\]/';

// We need to find this point in the file
$time_to_find = ($current_time_unix - $length);

// Use a binary search algorithm to find this timestamp in the file
$still_searching = 1;
$niterations = 50;
$min_size = 0;
$max_size = filesize($fname);
$mid_size = ($max_size - $min_size) / 2;
$lasttime = 0;
$timeunix = 0;

while ($still_searching) {

  fseek($fp,$mid_size);
  assert(!feof($fp));

  # since we will most likely wind up in the middle of a line, chuck it away and get a new line
  $line = fgets($fp);

  $line = fgets($fp);
  $cleanline = chop($line);

  // Check the timestamp vs current time:
  if (preg_match($time_pattern,$cleanline,$matches) == 1) {
    $timestr = substr($matches[0],1,19);
    $a = preg_split('/[-:]/',$timestr);
    $timeunix = mktime($a[3],$a[4],$a[5],$a[1],$a[2],$a[0]);
  }
  //echo $current_time_string." <=> ".$timestr."\n";

  # If the time is less that we require
  if ($timeunix && $timeunix < $time_to_find) {
    $min_size = $mid_size;
    $mid_size = $mid_size + (($max_size - $min_size)/2);
  }

  if ($timeunix && $timeunix > $time_to_find) {
    $max_size = $mid_size;
    $mid_size = $min_size + (($mid_size - $min_size)/2); 
  }

  //echo "SIZES: $min_size -> $mid_size -> $max_size\n";
 
  // if we have found the right time 
  if ($timeunix && $timeunix == $time_to_find) {
    $still_searching = 0;
  }

  // if we cant divide and more...
  if ($lasttime == $timeunix) {
    $still_searching = 0;
  }
  $lasttime = $timeunix;

  $niterations--;
  if ($niterations == 0) {
    $still_searching = 0;
  }

}

$lines_to_flush = 10;

// Keep looping forever
while ($max_time_limit > 0) {

  // We can begin to loop through reading all lines and echoing them:
  while (!feof($fp)) {

    $showline = true;

    if ($line = fgets($fp)) {

      $cleanline = chop($line);

      // Check the sub log type 
      if (($log_tag != "*")  && (strpos($cleanline, "] ".$log_tag.": ") === FALSE)) {
        $showline = false;
      }

      if (($filter != "*") && ($filter != "")) {
        if (preg_match("/".$filter."/",$cleanline) == 0) {
          $showline = false;
        }
      }

      // Check the timestamp vs current time
      if (preg_match($time_pattern,$cleanline,$matches) == 1) {
        $timestr = substr($matches[0],1,19);
        $a = preg_split('/[-:]/',$timestr);
        $timeunix = mktime($a[3],$a[4],$a[5],$a[1],$a[2],$a[0]);
      }

      if ($timeunix < ($current_time_unix - $length)) {
        $showline = false;
      }

      if ($showline) {

        if ($atEOF == "false") {
          $lines_to_flush--;
          if ($lines_to_flush <= 0) {
            $lines_to_flush = 10;
            ob_end_flush();
            ob_start();
            //echo "<script type=\"text/javascript\">self.scrollBy(0,500);</script>\n";
          }
        }

        $haveSomeLogs = "true";

        if (($level == "warn") && (strstr($line,"WARN:")))
          echo $cleanline."\n";
        else if (($level == "error") && (strstr($line,"ERROR:")))
          echo $cleanline."\n";
        else if ($level == "all") 
          echo $cleanline."\n";
        else
          ;
      }   

      // Automatically scroll down a bit
      //if (($auto_scroll) && ($atEOF == "true")) { 
      //  echo "<script type=\"text/javascript\">self.scrollLines(1);</script>\n";
      //}
    }
  }

  if ($atEOF == "false") { 
    $atEOF = "true";
    if ($haveSomeLogs == "false") {
      echo "No logs recorded for the last ".($length/3600)." hours<BR>\n";
    }
?>
<script type="text/javascript">self.scrollBy(0,1000000);</script> 
<?
    ob_end_flush();
  }

  // Ok, we hit the end of the file, there are a few odds-n-ends we need:
  // First reseek to the end of the file, this is necessary to reset
  //  the filepointer, else it won't read anything else.
  fseek($fp, 0, SEEK_END);

  // Secondly flush the output buffers so that all text appears:
  flush();

  // Also reset the time limit for PHP so that we never time out:
  set_time_limit(30);
  $max_time_limit--;

  // Now sleep for 1 second, and then we will try to access it again.
  sleep(1);
}

?>
</pre></body>
</html>
<?

function close_log_file($fp) {
  fclose($fp);
}

?>

