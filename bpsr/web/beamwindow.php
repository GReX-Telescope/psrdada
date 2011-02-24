<?PHP

include ("bpsr.lib.php");
$inst = new bpsr();

$obsid = $_GET["obsid"];    // UTC Start
$beam_id = sprintf("%02d", $_GET["beamid"]); // Beam number [1-13]

$text = "";
$base_dir = $inst->config["SERVER_RESULTS_DIR"]."/".$obsid."/".$beam_id;

# find the IBOB control IP for the specified beam
$ibob = "";
for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++) {
  if ($inst->ibobs["BEAM_".$i] == $beam_id) {
    $ibob = $inst->ibobs["CONTROL_IP_".$i];
  }
}

# collect observation information
if (file_exists($base_dir)) {

  $nbeams = exec("find ".$inst->config["SERVER_RESULTS_DIR"]."/".$obsid."/* -type d | wc -l");
                                                                                                                                                
  $img_base = "/bpsr/results/".$obsid."/".$beam_id."/";
                                                                                                                                                
  $obs_info =  $inst->config["SERVER_RESULTS_DIR"]."/".$obsid."/obs.info";
  $obs_start = $base_dir."/obs.start";

  $obs_results = array_pop(array_pop($inst->getResults($inst->config["SERVER_RESULTS_DIR"], $obsid, "all", "all", $beam_id)));
  $stats_results = array_pop($inst->getStatsResults($inst->config["SERVER_RESULTS_DIR"], $ibob));

  if (file_exists($obs_start)) 
  {
    $header = getConfigFile($obs_start);
    $text = "Beam ".$beam_id." for ".$header["SOURCE"];
  } else {
    $header = array();
    $text = "Beam ".$beam_id." for ".$obsid;
  }

} else {
  $text = "No observation found";
}

?>
<html>
<?
  $inst->open_head();
  $inst->print_head_int("BPSR | Result ".$obsid, 0);
?>
  
  <script type="text/javascript">
  /* Looping function to try and refresh the images */
  function looper() {
    request()
    setTimeout('looper()',5000)
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
          // ignore 
        } else {
          obj = document.getElementById(type+"_"+size);

          if ((size == "400x300") && (obj.src != img)) {
            obj.src = img
          }
          if ((size == "1024x768") && (obj.href != img)) {
            obj.href = img;
          }
        }
      }
    }
  }

  /* Gets the data from the URL */
  function request() {
    if (window.XMLHttpRequest)
      http_request = new XMLHttpRequest()
    else
      http_request = new ActiveXObject("Microsoft.XMLHTTP");

    http_request.onreadystatechange = function() {
      handle_data(http_request)
    }

    /* This URL will return the names of the 5 current */
    var url = "/bpsr/plotupdate.php?type=all&obs=<?echo $obsid?>&beam=<?echo $beam_id?>"

    http_request.open("GET", url, true)
    http_request.send(null)
  }

  </script>

<?
$inst->close_head();
?>

<body onload="looper()">

<? 

$inst->print_banner("");

if (! (file_exists($base_dir))) {

?>
<center>
<table>
  <tr> <td colspan=2 bgcolor=red>Could not find the observation</td></tr>
  <tr> <td>UTC_START</td><td><?echo $obsid?></td> </tr>
  <tr> <td>BEAM_ID</td><td><?echo $beam_id?></td> </tr>
  <tr> <td>BASE_DIR</td><td><?echo $base_dir?></td> </tr>
</table>
</center>
<?

} else {

?>

<center>
<table border=0>
  <tr valign=middle>
    <td>Beam</td>
<?

  for ($i=1; $i <= $nbeams; $i++) {
    if ($i != $_GET["beamid"]) {
      echo "    <td width=40><div class=\"btns\">\n";
      echo "      <a href=\"/bpsr/beamwindow.php?beamid=".$i."&obsid=".$obsid."\" class=\"btn\" > <span>".sprintf("%02d",$i)."</span></a>";
      echo "    </div></td>";
    } else {
      echo "    <td width=40 align=center><b>".sprintf("%02d",$i)."</b></td>";
    }
  }
?>
  </tr>
</table>

<hr>

</center>


<table cellpadding=5>
<tr><td valign=top align=center>

  <table class="datatable" style="width:300px">
    <tr><th colspan=2>Beam / Obs Info</th></tr>
    <tr><td width=50%>Source</td><td align=left width=50%><?echo $header["SOURCE"]?></td></tr>
    <tr><td>UTC_START</td><td align=left><?echo $obsid?></td></tr>
    <tr><td>ACC_LEN</td><td align=left><?echo $header["ACC_LEN"]?></td></tr>
    <tr><td>RA</td><td align=left><?echo $header["RA"]?></td></tr>
    <tr><td>DEC</td><td align=left><?echo $header["DEC"]?></td></tr>
    <tr><td>FA</td><td align=left><?echo $header["FA"]?></td></tr>
    <tr><td>Beam</td><td align=left><?echo $beam_id?> of <?echo $nbeams?></td></tr>
  </table>

  </td>
  <td valign=top>

   <table class="datatable" style="width:300px">
    <tr><th colspan=2>Obs State Information</th></tr>
    <tr><td width=50%>Finalized</td><td align=left width=50%><?echo $state["FINALIZED"]?></td></tr>
    <tr><td>Transferred to swin</td><td align=left><?echo $state["sent.to.swin"]?></td></tr>
    <tr><td>Transferred to parkes</td><td align=left><?echo $state["sent.to.parkes"]?></td></tr>
    <tr><td>On tape at swin</td><td align=left><?echo $state["on.tape.swin"]?></td></tr>
    <tr><td>On tape at parkes</td><td align=left><?echo $state["on.tape.parkes"]?></td></tr>
  </table>

  </td>
  <td>
  </td>
</tr>
<tr>

  <td align=center width="33%">
    Bandpass<br>
    <a id="bp_1024x768" href="/images/blackimage.gif">
      <img id="bp_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

  <td align=center width="33%">
    DM0 Timeseries<br>
    <a id="ts_1024x768" href="/images/blackimage.gif">
      <img id="ts_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

  <td align=center width="33%">
    Phase vs Freq<br>
    <a id="pvf_1024x768" href="/images/blackimage.gif">
      <img id="pvf_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

</tr>

<tr>
  <td align=center>
    PD Bandpass (live)<br>
    <a id="pdbp_1024x768" href="/images/blackimage.gif">
      <img id="pdbp_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

  <td align=center>
    Fluctuation Power Spectrum<br>
    <a id="fft_1024x768" href="/images/blackimage.gif">
      <img id="fft_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

  <td align=center>
    Digitizer Statistics<br>
    <a id="dts_1024x768" href="/images/blackimage.gif">
      <img id="dts_400x300" src="/images/blackimage.gif" width=401 height=301>
    </a>
  </td>

</tr>

</table>

</td></tr></table>

<? 
}
?>

</body>
</html>

<?

function getImages($dir, $img_base) {

  $data = array();

  /* Find the latest files in the plot file directory */
  $types = array("bp", "ts", "fft", "dts", "pvf");

  for ($i=0; $i<count($types); $i++) {
    $data[$types[$i]."_low"] = "/images/blankimage.gif";
    $data[$types[$i]."_mid"] = "/images/blankimage.gif";
    # $data[$types[$i]."_hi"] = "/images/blankimage.gif";
  }

  /* Get into a relative dir... */
  $cwd = getcwd();
  chdir($dir);

  for ($i=0; $i<count($types); $i++) {
    /* Find the hi res images */
    $cmd = "find . -name \"*.".$types[$i]."_1024x768.png\" -printf \"%P\n\" | sort | tail -n 1";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1)) {
      $data[$types[$i]."_hi"] = $img_base.$find_result;
    }

    /* Find the mid res images */
    $cmd = "find . -name \"*.".$types[$i]."_400x300.png\" -printf \"%P\n\" | sort | tail -n 1";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1))  {
      $data[$types[$i]."_mid"] = $img_base.$find_result;
    }

    /* Find the low res images */
    $cmd = "find . -name \"*.".$types[$i]."_112x84.png\" -printf \"%P\n\" | sort | tail -n 1";
    $find_result = exec($cmd, $array, $return_val);
    if (($return_val == 0) && (strlen($find_result) > 1))  {
      $data[$types[$i]."_low"] = $img_base.$find_result;
    }
  }
  chdir($cwd);

  return $data;
}

?>
