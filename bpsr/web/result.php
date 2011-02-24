<?PHP

include ("bpsr.lib.php");
$inst = new bpsr();

$utc_start = $_GET["utc_start"];
$imagetype = "bp";
if (isset($_GET["imagetype"])) {
  $imagetype = $_GET["imagetype"];
}

$tmp = $inst->getResultsInfo($utc_start, $inst->config["SERVER_RESULTS_DIR"]);
$data = $tmp[$utc_start];

$nbeam = $data["nbeams"];
$header = getConfigFile($data["obs_start"], TRUE);

$obs_info_file = $inst->config["SERVER_RESULTS_DIR"]."/".$utc_start."/obs.info";
$obs_info = getConfigFile($obs_info_file);

?>

<html>
<?
  $inst->open_head();
  $inst->print_head_int("BPSR | Observation ".$utc_start, 0);
?>

  <script type="text/javascript">

  var utc_start = "<?echo $utc_start?>"

  /* Creates a pop up window */
  function popWindow(URL,width,height) {

    var width = "1300";
    var height = "820";

    URL = URL + "&obsid=<?echo $utc_start?>"

    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width="+width+",height="+height+"');");
  }


  /* Looping function to try and refresh the images */
  function looper() {
    request()
    setTimeout('looper()',20000)
  }

  /* Parses the HTTP response and makes changes to images
   * as requried */
  function handle_data(http_request) {

    if (http_request.readyState == 4) {
      var response = String(http_request.responseText)
      var lines = response.split("\n");

      var currImg
      var beam
      var size
      var type
      var img

      for (i=0; i < lines.length-1; i++) {

        values = lines[i].split(":::");
        beam = values[0];
        size = values[1];
        type = values[2];
        img  = values[3];

        if (size == "112x84") {
          currImg = document.getElementById("beam"+beam);

          if (currImg == null) {
            alert("beam"+beam+ " gave a currImg Null");
          }
          if (currImg.src != img) {
            currImg.src = img
          }
        }
      }
    }
  }

  function request() {
    if (window.XMLHttpRequest)
      http_request = new XMLHttpRequest()
    else
      http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
    http_request.onreadystatechange = function() {
      handle_data(http_request)
    }

    var type = "bp";

    if (document.imageform.imagetype[0].checked == true) {
      type = "bp";
    }

    if (document.imageform.imagetype[1].checked == true) {
      type = "ts";
    }

    if (document.imageform.imagetype[2].checked == true) {
      type = "fft";
    }

    if (document.imageform.imagetype[3].checked == true) {
      type = "pvf";
    }

    /* This URL will return the names of the 5 current */
    var url = "plotupdate.php?obs=<?echo $utc_start?>&type="+type+"&size=112x84";

    http_request.open("GET", url, true)
    http_request.send(null)
  }

</script>

<?
  $inst->close_head();
?>
<body onload='looper()'>
<script type="text/javascript" src="/js/wz_tooltip.js"></script>
<? 
$inst->print_banner("Observation ".$utc_start);
?>

<center>

<table>
 <tr>
  <td valign="top">

<!-- Beam/Image Table -->
<table border=0 cellspacing=10 cellpadding=0 class="multibeam">
  <tr>
    <td rowspan=3>
      <form name="imageform" class="smalltext">
<?
      echoRadio("imagetype","bp", "Bandpass", $imagetype); 
      echo "<br>\n";
      echoRadio("imagetype","ts", "Time Series", $imagetype); 
      echo "<br>\n";
      echoRadio("imagetype","fft", "Power Spectrum", $imagetype); 
      echo "<br>\n";
      echoRadio("imagetype","pvf", "Phase v Freq", $imagetype);
  ?>
      </form>
    
<?    echoBeam(13, $nbeam)?>
<?    echoBlank()?>
<?    echoBeam(12, $nbeam)?>
<?    echoBlank()?> 
  </tr>
  <tr>
   
<?    echoBeam(6, $nbeam)?>
<?    echoBlank()?>
  </tr>
  <tr>
  
<?    echoBeam(7, $nbeam)?>
<?    echoBeam(5, $nbeam)?>
<?    echoBlank()?> 
  </tr>

  <tr>
<?    echoBeam(8, $nbeam)?>
<?    echoBeam(1, $nbeam)?>
<?    echoBeam(11, $nbeam)?>
  </tr>

  <tr>
<?    echoBeam(2, $nbeam)?>
<?    echoBeam(4, $nbeam)?>
  </tr>

  <tr>
<?    echoBlank()?>
<?    echoBeam(3, $nbeam)?>
<?    echoBlank()?>
  </tr>

  <tr>
<?    echoBlank()?>
<?    echoBeam(9, $nbeam)?>
<?    echoBeam(10, $nbeam)?>
<?    echoBlank()?>
  </tr>
  
  <tr>
<?    echoBlank()?>
<?    echoBlank()?>
<?    echoBlank()?>
  </tr>
</table>

 </td>

 <td>

 <!-- Obs Summary Table -->
    
<table cellspacing=5>
<tr>
 <td valign="top">
  <table class="datatable"> 
   <tr><th colspan=2>Observation Summary</th></tr>
<?
        echo "   <tr><td>UTC_START</td><td>".$obs_info["UTC_START"]."</td></tr>\n";
        echo "   <tr><td>SOURCE</td><td>".$obs_info["SOURCE"]."</td></tr>\n";
        echo "   <tr><td>RA</td><td>".$obs_info["RA"]."</td></tr>\n";
        echo "   <tr><td>DEC</td><td>".$obs_info["DEC"]."</td></tr>\n";
        echo "   <tr><td>NUM BEAMS</td><td>".$obs_info["NUM_PWC"]."</td></tr>\n";
?>
   </table>
 </td>
</tr>
<tr>
 <td valign="top">
<?
  if (is_array($header)) {
    printHeader($header, $brief);
  } else {
    echo "obs.start file did not exist<BR>\n";
  }
?>
 </td>
</tr>
</table>

</td>
</tr>
</table>
</body>
</html>

<?

function echoRadio($id, $value, $title, $selected) {

  echo "<input type=\"radio\" name=\"".$id."\" id=\"".$id."\" value=\"".$value."\" onChange=\"request()\"";

  if ($value == $selected)
    echo " checked";
  
  echo ">".$title;

}

function echoBlank() {

  echo "    <td><img src=\"/images/spacer.gif\" width=113 height=45></td>\n";
}


function echoBeam($beam_no, $n_beams) {

  $beam_str = sprintf("%02d",$beam_no);

  echo "    <td rowspan=2 class='multibeam' height=84'>\n";
  echo "      <a class='multibeam' href=\"javascript:popWindow('/bpsr/beamwindow.php?beamid=".$beam_str."')\">\n";
  echo "        <img src='/images/blankimage.gif' width='112px' height='84px' id='beam".$beam_str."' border=0>\n";
  echo "      </a>\n";
  echo "    </td>\n";

}


function printHeader($header, $brief) {

  $keys = array_keys($header);
  $keys_to_ignore = array("HDR_SIZE","FILE_SIZE","HDR_VERSION","FREQ","RECV_HOST","CONFIG",
                          "DSB", "INSTRUMENT", "NBIT", "NDIM", "NPOL", "PROC_FILE", "TELESCOPE",
                          "BEAM", "BW", "RA", "DEC", "FA", "UTC_START", "NCHAN", "BANDWIDTH",
                          "ACC_LEN");

  if (substr_count($_SERVER["REQUEST_URI"],"brief=") == 0) {

    $brief_url = $_SERVER["REQUEST_URI"]."&brief=1";
    $full_url = $_SERVER["REQUEST_URI"]."&brief=0";

  } else {
    $brief_url = str_replace("brief=0", "brief=1", $_SERVER["REQUEST_URI"]);
    $full_url  = str_replace("brief=1", "brief=0", $_SERVER["REQUEST_URI"]);
  }

  echo "<table cellpadding=0 cellspacing=0 class=\"datatable\">\n";
  echo "<tr>";
  if ($brief) 
    echo "<th colspan=2>Brief Header | <a href=".$full_url.">Full Header</a></th>\n";
  else 
    echo "<th colspan=2><a href=".$brief_url.">Brief Header</a> | Full Header</th>\n";

  for ($i=0; $i<count($keys); $i++) {
    if ( (!($brief)) || (!(in_array($keys[$i], $keys_to_ignore))) ) {
      echo "<tr><td align=right>".$keys[$i]."</td><td>".$header[$keys[$i]]."</td></tr>\n";
    }
  }
  echo "</table>\n";

  #$mousein = "onmouseover=\"Tip('".$header_text."')\"";
  #$mouseout = "onmouseout=\"UnTip()\"";
  #echo "<div color=blue ".$mousein." ".$mouseout.">DADA HEADER</div>\n";

}

